"""
LineRangeSearch - find the lines of several line-list files that lie in a
wavelength range.

Searches HITRAN-format ``.par`` files and CSV line lists (input line lists,
saved-line files, the atomic line list, ...) and returns every line whose
wavelength is inside the range, one row per line, with the name of the file it
came from.  Used by the GUI's ``Find Lines in Range`` window, and usable on its
own::

    from iSLAT.Modules.DataProcessing.LineRangeSearch import search_line_lists

    result = search_line_lists(
        ["HITRANdata/data_Hitran_CO.par", "LINELISTS/Atomic_lines.csv"], 4.6, 4.7)
    result.lines     # pandas DataFrame, one row per line, sorted by wavelength
    result.errors    # {path: reason} for the files that could not be searched

Relative paths are looked up in ``DATAFILES`` first and then in the current
directory, like molecule line-list paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "FILE_COLUMN",
    "LINE_COLUMNS",
    "SEARCHABLE_EXTENSIONS",
    "WAVELENGTH_COLUMNS",
    "LineSearchError",
    "LineSearchResult",
    "find_line_list_files",
    "format_wavelength",
    "is_searchable",
    "lines_in_range",
    "molecule_line_list_files",
    "normalize_range",
    "search_line_lists",
    "species_from_par_header",
    "wavelength_column",
]

PathLike = Union[str, os.PathLike]

#: File extensions that can be searched.
SEARCHABLE_EXTENSIONS: Tuple[str, ...] = (".par", ".csv")

#: Result column holding the name of the file each line came from.
FILE_COLUMN = "file"

#: Line columns that lead the results, in the iSLAT CSV line-list order (so
#: exported results read back as an input line list).  ``.par`` results have
#: exactly these; CSV results keep their other columns after them.
LINE_COLUMNS: Tuple[str, ...] = (
    "species", "lev_up", "lev_low", "lam",
    "a_stein", "e_up", "e_low", "g_up", "g_low",
)

#: CSV column names accepted for the wavelength (in microns), in order of
#: preference.  Matched case-insensitively; the result column is ``lam``.
WAVELENGTH_COLUMNS: Tuple[str, ...] = ("lam", "wave", "wavelength")

#: Older saved-line column names, renamed when the new name is not present.
_COLUMN_ALIASES = {"up_lev": "lev_up", "low_lev": "lev_low"}

#: Sub-folders skipped when adding a folder (derived files, not line lists).
_SKIP_DIRS = frozenset({"cache", "__pycache__"})

#: Database names that can start a .par header before the species.
_PAR_DATABASE_NAMES = frozenset({"HITRAN", "HITEMP", "EXOMOL", "GEISA"})


class LineSearchError(Exception):
    """A file that cannot be searched (missing, unreadable, or not a line list)."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class LineSearchResult:
    """Lines found by :func:`search_line_lists`.

    Attributes
    ----------
    lam_min, lam_max : float
        The searched wavelength range in microns (``lam_min < lam_max``).
    lines : pandas.DataFrame
        One row per line in the range, sorted by wavelength: the
        :data:`FILE_COLUMN`, then :data:`LINE_COLUMNS`, then any other
        columns of the CSV files.  Values a file does not have are NaN.
    files_searched : list of str
        The files that were read, in search order (whether or not they had
        lines in the range).
    errors : dict
        ``{path: reason}`` for the files that could not be searched.
    cancelled : bool
        True when the search was stopped before every file was read; the
        other fields then cover the files read so far.
    """

    lam_min: float
    lam_max: float
    lines: pd.DataFrame
    files_searched: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def n_lines(self) -> int:
        """Number of lines found."""
        return len(self.lines)

    def counts_by_file(self) -> Dict[str, int]:
        """``{file name: number of lines found}`` for every searched file."""
        counts = {os.path.basename(path): 0 for path in self.files_searched}
        if self.n_lines:
            for name, count in self.lines[FILE_COLUMN].value_counts(sort=False).items():
                counts[name] = int(count)
        return counts

    def summary(self) -> str:
        """One-line description, e.g. ``Found 12 lines in 4.6 - 4.7 um (2 of 3 files).``"""
        with_lines = sum(1 for n in self.counts_by_file().values() if n)
        n_files = len(self.files_searched)
        text = (
            f"Found {self.n_lines:,} line{'' if self.n_lines == 1 else 's'} in "
            f"{format_wavelength(self.lam_min)} - {format_wavelength(self.lam_max)} um "
            f"({with_lines} of {n_files} file{'' if n_files == 1 else 's'})."
        )
        if self.errors:
            text += f" {len(self.errors)} file{'' if len(self.errors) == 1 else 's'} could not be searched."
        return text


def format_wavelength(value: float) -> str:
    """*value* with up to six decimals and no trailing zeros (``12.5``, ``4.052281``)."""
    text = f"{float(value):.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def normalize_range(lam_min: Any, lam_max: Any) -> Tuple[float, float]:
    """``(low, high)`` from two wavelength bounds in microns, given in either order.

    The bounds may be numbers or numeric strings (as typed in the GUI).

    Raises
    ------
    ValueError
        If a bound is not a finite number, or the bounds are equal.
    """
    bounds = []
    for name, value in (("start", lam_min), ("end", lam_max)):
        try:
            number = float(str(value).strip()) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"The {name} wavelength must be a number (got {value!r}).") from None
        if not np.isfinite(number):
            raise ValueError(f"The {name} wavelength must be a finite number (got {value!r}).")
        bounds.append(number)
    low, high = sorted(bounds)
    if low == high:
        raise ValueError("The start and end wavelengths are the same; enter a range.")
    return low, high


def is_searchable(path: PathLike) -> bool:
    """True when *path* has an extension the search can read (``.par`` / ``.csv``)."""
    return Path(path).suffix.lower() in SEARCHABLE_EXTENSIONS


def wavelength_column(columns: Iterable[Any]) -> Optional[str]:
    """The column of *columns* holding the wavelength (see :data:`WAVELENGTH_COLUMNS`)."""
    names = [str(c).strip() for c in columns]
    for wanted in WAVELENGTH_COLUMNS:
        for name in names:
            if name.lower() == wanted:
                return name
    return None


def _csv_header(path: PathLike) -> List[str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return [h.strip() for h in fh.readline().split(",")]


def find_line_list_files(folder: PathLike, recursive: bool = True) -> List[str]:
    """The searchable line lists in *folder*, sorted by path.

    ``.par`` files are always included; CSV files only when their header has
    a wavelength column, so other CSVs (molecule saves, settings, ...) are
    left out.  ``cache`` folders are skipped.
    """
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            path = os.path.join(dirpath, name)
            suffix = Path(name).suffix.lower()
            if suffix == ".par":
                found.append(path)
            elif suffix == ".csv":
                try:
                    if wavelength_column(_csv_header(path)) is not None:
                        found.append(path)
                except OSError:
                    continue
        if not recursive:
            break
    return sorted(found, key=lambda p: os.path.normcase(p))


def _resolve(path: PathLike) -> str:
    """The existing file *path* names (``DATAFILES``-relative paths allowed)."""
    from iSLAT.Modules.FileHandling.line_list_paths import find_existing_line_list
    existing = find_existing_line_list(path)
    if existing is None:
        raise LineSearchError("File not found.")
    return os.path.abspath(existing)


def molecule_line_list_files(molecules: Any) -> List[str]:
    """Absolute paths of the line-list files used by *molecules*, without repeats.

    *molecules* is a ``MoleculeDict`` (or any mapping / iterable of
    molecules); molecules without a file, or whose file is missing, are
    skipped.
    """
    if molecules is None:
        return []
    items = molecules.values() if hasattr(molecules, "values") else molecules
    paths: List[str] = []
    seen = set()
    for mol in items:
        filepath = getattr(mol, "filepath", None)
        if not filepath:
            continue
        try:
            path = _resolve(filepath)
        except LineSearchError:
            continue
        key = os.path.normcase(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Reading one file
# ---------------------------------------------------------------------------
def species_from_par_header(comment_lines: Sequence[str]) -> str:
    """The species named on a .par file's first header line, or ``""``.

    iSLAT .par files start with e.g. ``# HITRAN CO; id:5; ...`` or
    ``# HITEMP 2019 CO; ...``: the species is the last word before the first
    ``;`` when that word is not the database name or a year.
    """
    if not comment_lines:
        return ""
    words = comment_lines[0].strip().lstrip("#").split(";", 1)[0].split()
    if len(words) < 2:
        return ""
    name = words[-1]
    if name.isdigit() or name.upper() in _PAR_DATABASE_NAMES:
        return ""
    return name


def _par_lines_in_range(path: str, lam_min: float, lam_max: float) -> pd.DataFrame:
    from iSLAT.Modules.FileHandling.molecular_data_reader import read_par_lines_in_range
    comment_lines, lines_data = read_par_lines_in_range(path, lam_min, lam_max)
    frame = pd.DataFrame(lines_data, columns=[c for c in LINE_COLUMNS if c != "species"])
    frame.insert(0, "species", species_from_par_header(comment_lines))
    return frame


def _csv_lines_in_range(path: str, lam_min: float, lam_max: float) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    column = wavelength_column(frame.columns)
    if column is None:
        raise LineSearchError(
            "Not a line list: no wavelength column (expected one of "
            + ", ".join(repr(c) for c in WAVELENGTH_COLUMNS) + ")."
        )
    lam = pd.to_numeric(frame[column], errors="coerce")
    inside = ((lam >= lam_min) & (lam <= lam_max)).to_numpy()
    found = frame.loc[inside].copy()
    found[column] = lam[inside]

    renames = {column: "lam"} if column != "lam" else {}
    for old, new in _COLUMN_ALIASES.items():
        if old in found.columns and new not in found.columns:
            renames[old] = new
    return found.rename(columns=renames).reset_index(drop=True)


def lines_in_range(path: PathLike, lam_min: float, lam_max: float) -> pd.DataFrame:
    """The lines of one file with ``lam_min <= lam <= lam_max``, in file order.

    ``.par`` files give the :data:`LINE_COLUMNS` (the species comes from the
    file header and is blank when the header does not name one); CSV files
    give all of their columns, with the wavelength column named ``lam``.

    Raises
    ------
    LineSearchError
        If the file is missing, is not a ``.par`` / ``.csv`` file, or cannot
        be read as a line list.
    """
    lam_min, lam_max = normalize_range(lam_min, lam_max)
    resolved = _resolve(path)
    suffix = Path(resolved).suffix.lower()
    if suffix not in SEARCHABLE_EXTENSIONS:
        raise LineSearchError(
            f"Unsupported file type {suffix or '(none)'!r}; expected a .par or .csv line list."
        )
    try:
        if suffix == ".par":
            return _par_lines_in_range(resolved, lam_min, lam_max)
        return _csv_lines_in_range(resolved, lam_min, lam_max)
    except LineSearchError:
        raise
    except Exception as exc:
        raise LineSearchError(f"Could not read the file: {exc}") from exc


# ---------------------------------------------------------------------------
# Searching several files
# ---------------------------------------------------------------------------
def _combine(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-file results with the leading columns first, sorted by wavelength."""
    leading = [FILE_COLUMN, *LINE_COLUMNS]
    if not frames:
        return pd.DataFrame(columns=leading)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    order = [c for c in leading if c in combined.columns]
    order += [c for c in combined.columns if c not in order]
    combined = combined[order]
    return combined.sort_values("lam", kind="mergesort").reset_index(drop=True)


def search_line_lists(
    paths: Iterable[PathLike],
    lam_min: Any,
    lam_max: Any,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> LineSearchResult:
    """Find the lines of every file in *paths* with ``lam_min <= lam <= lam_max``.

    Parameters
    ----------
    paths : iterable of path-like
        ``.par`` files and CSV line lists.  A file listed twice is searched
        once.
    lam_min, lam_max : float or str
        Wavelength range in microns, in either order (see
        :func:`normalize_range`).
    progress : callable, optional
        ``progress(index, total, path)`` is called before each file is read.
    should_stop : callable, optional
        Polled before each file; when it returns True the search stops and
        the result is marked ``cancelled``.

    Returns
    -------
    LineSearchResult
        A file that cannot be searched is reported in ``errors`` and does not
        stop the search.

    Raises
    ------
    ValueError
        If the wavelength range is invalid.
    """
    low, high = normalize_range(lam_min, lam_max)

    unique: List[str] = []
    seen = set()
    for path in paths:
        text = os.fspath(path)
        key = os.path.normcase(os.path.abspath(text))
        if key not in seen:
            seen.add(key)
            unique.append(text)

    frames: List[pd.DataFrame] = []
    searched: List[str] = []
    errors: Dict[str, str] = {}
    cancelled = False
    for index, path in enumerate(unique):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        if progress is not None:
            progress(index, len(unique), path)
        try:
            found = lines_in_range(path, low, high)
        except LineSearchError as exc:
            errors[path] = str(exc)
            continue
        searched.append(path)
        if found.empty:
            continue
        if FILE_COLUMN in found.columns:   # keep a CSV's own "file" column
            found = found.rename(columns={FILE_COLUMN: f"{FILE_COLUMN} (column)"})
        found.insert(0, FILE_COLUMN, os.path.basename(path))
        frames.append(found)

    return LineSearchResult(
        lam_min=low,
        lam_max=high,
        lines=_combine(frames),
        files_searched=searched,
        errors=errors,
        cancelled=cancelled,
    )
