"""
line_list_readers — Protocol and concrete readers for multi-format line list ingestion.

Provides a :class:`LineListReader` Protocol and a :class:`ReadResult` dataclass
that decouple :class:`MoleculeLineList` from any specific file format.

Shipped readers
---------------
* :class:`HitranParReader` — HITRAN ``.par`` fixed-width files
* :class:`CsvLineListReader` — iSLAT-convention CSV line lists (e.g. ``MIRI_H2O_v0-0.csv``)
* :class:`SavedLinesReader` — LINESAVES CSV files with fit-result columns

Third-party readers can be registered at runtime via
``MoleculeLineList.register_reader(name, reader_instance)``.
"""

from __future__ import annotations

import csv
import os
import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

from iSLAT import Constants as c

# ---------------------------------------------------------------------------
# ReadResult — uniform output from every reader
# ---------------------------------------------------------------------------

# Reuse the partition type from MoleculeLineList for consistency
_PartitionType = namedtuple("partition", ["t", "q"])

@dataclass(frozen=True, slots=True)
class ReadResult:
    """Uniform container returned by every :class:`LineListReader`.

    Attributes
    ----------
    lines_data : list[dict]
        Each dict has the 10 core keys expected by ``_LINE_DTYPE``
        (``nr``, ``lev_up``, ``lev_low``, ``lam``, ``freq``,
        ``a_stein``, ``e_up``, ``e_low``, ``g_up``, ``g_low``).
        Missing keys are filled with sensible defaults.
    partition : partition namedtuple or None
        ``partition(t=np.ndarray, q=np.ndarray)`` when available.
    metadata : dict
        Format-specific metadata.  Always contains ``"source_format"``
        (str).  HITRAN adds ``"molar_mass"``; CSV adds ``"species"``.
    extra_columns : dict[str, list]
        Any columns beyond the 10-core set, keyed by column name.
        Values are plain Python lists aligned with *lines_data*.
    """

    lines_data: List[Dict[str, Any]]
    partition: Optional[Any] = None  # namedtuple(t, q)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_columns: Dict[str, list] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# LineListReader Protocol
# ---------------------------------------------------------------------------

# The 10 core field names that every reader must produce
CORE_FIELD_NAMES = (
    "nr", "lev_up", "lev_low", "lam", "freq",
    "a_stein", "e_up", "e_low", "g_up", "g_low",
)

@runtime_checkable
class LineListReader(Protocol):
    """Protocol that every line-list reader must satisfy."""

    def read(self, filepath: Union[str, Path]) -> ReadResult:
        """Read a line-list file and return a :class:`ReadResult`."""
        ...

# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

# Column names that distinguish a "saved lines" CSV from a plain line-list CSV
_SAVED_LINES_MARKER_COLUMNS = frozenset({
    "Flux_data", "Err_data", "Fit_SN", "Flux_fit",
    "FWHM_fit", "Centr_fit", "Red-chisq",
})

def detect_format(filepath: Union[str, Path]) -> str:
    """Inspect *filepath* and return a format tag.

    Returns
    -------
    str
        One of ``"hitran"``, ``"csv"``, ``"saved"``.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".par":
        return "hitran"

    if suffix == ".csv":
        # Peek at the header line to distinguish plain CSV vs saved-lines CSV
        try:
            with open(filepath, "r", newline="") as fh:
                header = fh.readline().strip()
        except OSError:
            return "csv"  # best guess

        header_fields = {h.strip() for h in header.split(",")}

        if header_fields & _SAVED_LINES_MARKER_COLUMNS:
            return "saved"
        return "csv"

    # Fallback: try to sniff content
    try:
        with open(filepath, "r") as fh:
            first = fh.readline().strip()
        # .par files start with a comment "# HITRAN ..." or a bare integer
        if first.startswith("#") or first.isdigit():
            return "hitran"
    except OSError:
        pass

    raise ValueError(f"Cannot auto-detect format of {filepath}")

# ---------------------------------------------------------------------------
# HitranParReader
# ---------------------------------------------------------------------------
class HitranParReader:
    """Reader for HITRAN ``.par`` fixed-width line-list files.

    Delegates the heavy parsing to the existing
    :class:`MolecularDataReader` and wraps the result in a
    :class:`ReadResult`.
    """

    def read(self, filepath: Union[str, Path]) -> ReadResult:
        """Read a ``.par`` file.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.par`` file.

        Returns
        -------
        ReadResult
        """
        from .molecular_data_reader import read_molecular_data

        filepath = str(filepath)
        partition_function, lines_data, other_fields = read_molecular_data(
            molecule_name="", filename=filepath,
        )

        # Extract molar mass from other_fields (list of (name, value) tuples)
        molar_mass = None
        if other_fields:
            for name, value in other_fields:
                if name == "Molar_Mass":
                    molar_mass = value
                    break

        from .molecular_data_reader import read_par_header
        header = read_par_header(filepath)
        metadata: Dict[str, Any] = {
            "source_format": "hitran",
            "hitran_id": header["hitran_id"],
            "hitran_iso": header["hitran_iso"],
            "header_names": header["names"],
        }
        if molar_mass is not None:
            metadata["molar_mass"] = molar_mass

        return ReadResult(
            lines_data=lines_data or [],
            partition=partition_function,
            metadata=metadata,
            extra_columns={},
        )

# ---------------------------------------------------------------------------
# CsvLineListReader
# ---------------------------------------------------------------------------

# Mapping from CSV column names → core field names
_CSV_COLUMN_MAP: Dict[str, str] = {
    "species":  "_species",   # not a core field — stored in metadata
    "lev_up":   "lev_up",
    "lev_low":  "lev_low",
    "lam":      "lam",
    "a_stein":  "a_stein",
    "e_up":     "e_up",
    "e_low":    "e_low",
    "g_up":     "g_up",
    "g_low":    "g_low",
    "freq":     "freq",
    "nr":       "nr",
    # Common alternative names
    "wave":     "lam",
    "wavelength": "lam",
    "einstein_a": "a_stein",
}

class CsvLineListReader:
    """Reader for iSLAT-convention CSV line lists.

    Handles files like ``MIRI_H2O_v0-0.csv`` with columns
    ``species, lev_up, lev_low, lam, a_stein, e_up, e_low, g_up, g_low``
    plus optional extras (``xmin``, ``xmax``, ``note``, etc.).

    Missing ``freq`` is computed from ``lam`` via :math:`\\nu = c / \\lambda`.
    Missing ``nr`` is set to the row index.
    """

    def read(self, filepath: Union[str, Path]) -> ReadResult:
        filepath = Path(filepath)

        # Read with csv module for robustness (no pandas dependency)
        with open(filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return ReadResult([], None, {"source_format": "csv"}, {})
            raw_rows = list(reader)

        if not raw_rows:
            return ReadResult([], None, {"source_format": "csv"}, {})

        header_fields = list(reader.fieldnames)  # type: ignore[arg-type]

        # Determine which header columns map to core fields and which are extras
        core_map: Dict[str, str] = {}       # csv_col → core_field
        extra_cols: List[str] = []
        species_col: Optional[str] = None

        for col in header_fields:
            mapped = _CSV_COLUMN_MAP.get(col)
            if mapped == "_species":
                species_col = col
            elif mapped is not None:
                core_map[col] = mapped
            else:
                extra_cols.append(col)

        lines_data: List[Dict[str, Any]] = []
        extra_columns: Dict[str, list] = {col: [] for col in extra_cols}
        species_set: set = set()

        for i, row in enumerate(raw_rows):
            line_dict: Dict[str, Any] = {}

            # Map core fields
            for csv_col, core_name in core_map.items():
                raw_val = row.get(csv_col, "")
                if raw_val is None:
                    raw_val = ""
                raw_val = str(raw_val).strip()

                if core_name in ("lev_up", "lev_low"):
                    line_dict[core_name] = raw_val
                elif core_name == "nr":
                    line_dict[core_name] = _safe_int(raw_val, default=i)
                else:
                    line_dict[core_name] = _safe_float(raw_val, default=0.0)

            # Defaults for missing core fields
            line_dict.setdefault("nr", i)
            line_dict.setdefault("lev_up", "")
            line_dict.setdefault("lev_low", "")
            line_dict.setdefault("lam", 0.0)
            line_dict.setdefault("a_stein", 0.0)
            line_dict.setdefault("e_up", 0.0)
            line_dict.setdefault("e_low", 0.0)
            line_dict.setdefault("g_up", 0)
            line_dict.setdefault("g_low", 0)

            # Compute freq from lam if not provided
            if "freq" not in line_dict or line_dict["freq"] == 0.0:
                lam_val = line_dict["lam"]
                if lam_val > 0:
                    # c in µm/s,  lam in µm  →  freq in Hz
                    line_dict["freq"] = c.SPEED_OF_LIGHT_MICRONS / lam_val
                else:
                    line_dict["freq"] = 0.0

            lines_data.append(line_dict)

            # Collect extras
            for col in extra_cols:
                extra_columns[col].append(row.get(col, ""))

            # Track species
            if species_col:
                sp = str(row.get(species_col, "")).strip()
                if sp:
                    species_set.add(sp)

        metadata: Dict[str, Any] = {"source_format": "csv"}
        if species_set:
            metadata["species"] = sorted(species_set)

        return ReadResult(
            lines_data=lines_data,
            partition=None,
            metadata=metadata,
            extra_columns=extra_columns,
        )

# ---------------------------------------------------------------------------
# SavedLinesReader
# ---------------------------------------------------------------------------

# Mapping for saved-line CSV columns that correspond to core fields
_SAVED_COLUMN_MAP: Dict[str, str] = {
    "species":  "_species",
    "lev_up":   "lev_up",
    "lev_low":  "lev_low",
    "lam":      "lam",
    "a_stein":  "a_stein",
    "e_up":     "e_up",
    "e_low":    "e_low",
    "g_up":     "g_up",
    "g_low":    "g_low",
    "freq":     "freq",
    "nr":       "nr",
    # Legacy names from LineSaveService
    "up_lev":   "lev_up",
    "low_lev":  "lev_low",
    "wavelength": "lam",
}

class SavedLinesReader:
    """Reader for LINESAVES CSV files (fit-result columns preserved).

    Reads files like those in ``DATAFILES/LINESAVES/``.  Core spectral
    fields are mapped to the standard 10-column dtype; all remaining
    columns (``Flux_data``, ``Err_data``, ``FWHM_fit``, etc.) are
    returned in :attr:`ReadResult.extra_columns`.
    """

    def read(self, filepath: Union[str, Path]) -> ReadResult:
        filepath = Path(filepath)

        with open(filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return ReadResult([], None, {"source_format": "saved"}, {})
            raw_rows = list(reader)

        if not raw_rows:
            return ReadResult([], None, {"source_format": "saved"}, {})

        header_fields = list(reader.fieldnames)  # type: ignore[arg-type]

        core_map: Dict[str, str] = {}
        extra_cols: List[str] = []
        species_col: Optional[str] = None

        for col in header_fields:
            mapped = _SAVED_COLUMN_MAP.get(col)
            if mapped == "_species":
                species_col = col
            elif mapped is not None:
                # Avoid duplicate mapping — first match wins
                if mapped not in core_map.values():
                    core_map[col] = mapped
                else:
                    extra_cols.append(col)
            else:
                extra_cols.append(col)

        lines_data: List[Dict[str, Any]] = []
        extra_columns: Dict[str, list] = {col: [] for col in extra_cols}
        species_set: set = set()

        for i, row in enumerate(raw_rows):
            line_dict: Dict[str, Any] = {}

            for csv_col, core_name in core_map.items():
                raw_val = row.get(csv_col, "")
                if raw_val is None:
                    raw_val = ""
                raw_val = str(raw_val).strip()

                if core_name in ("lev_up", "lev_low"):
                    line_dict[core_name] = raw_val
                elif core_name == "nr":
                    line_dict[core_name] = _safe_int(raw_val, default=i)
                else:
                    line_dict[core_name] = _safe_float(raw_val, default=0.0)

            line_dict.setdefault("nr", i)
            line_dict.setdefault("lev_up", "")
            line_dict.setdefault("lev_low", "")
            line_dict.setdefault("lam", 0.0)
            line_dict.setdefault("a_stein", 0.0)
            line_dict.setdefault("e_up", 0.0)
            line_dict.setdefault("e_low", 0.0)
            line_dict.setdefault("g_up", 0)
            line_dict.setdefault("g_low", 0)

            if "freq" not in line_dict or line_dict["freq"] == 0.0:
                lam_val = line_dict["lam"]
                if lam_val > 0:
                    line_dict["freq"] = c.SPEED_OF_LIGHT_MICRONS / lam_val
                else:
                    line_dict["freq"] = 0.0

            lines_data.append(line_dict)

            for col in extra_cols:
                extra_columns[col].append(row.get(col, ""))

            if species_col:
                sp = str(row.get(species_col, "")).strip()
                if sp:
                    species_set.add(sp)

        metadata: Dict[str, Any] = {"source_format": "saved"}
        if species_set:
            metadata["species"] = sorted(species_set)

        return ReadResult(
            lines_data=lines_data,
            partition=None,
            metadata=metadata,
            extra_columns=extra_columns,
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: str, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* on failure."""
    if not value:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    if not value:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

# ---------------------------------------------------------------------------
# Default reader registry — importable by MoleculeLineList
# ---------------------------------------------------------------------------

DEFAULT_READERS: Dict[str, LineListReader] = {
    "hitran": HitranParReader(),
    "csv":    CsvLineListReader(),
    "saved":  SavedLinesReader(),
}