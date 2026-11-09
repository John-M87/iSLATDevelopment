"""
LineIdentity - names one transition so another window can find it.

The population diagram, the line inspection plot and the line list tables each
hold lines in their own form: an info dict, an entry of the plotted arrays, a
table row.  :class:`LineRef` is the description they hand each other ("show
this line there"), and :func:`match_line` finds the row of a list of lines
that is the described line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Wavelengths within this fraction of each other (1e-6, i.e. 0.3 km/s) are
#: the same line.  Tables may round the wavelength they show (10 significant
#: digits in the line search results), so an exact comparison would miss.
LAM_RTOL = 1e-6

#: Half-width (um) of the inspection window around a line that has no fit
#: range of its own, as for a saved line written without one.
DEFAULT_HALF_WINDOW = 0.015


def _text(value: Any) -> Optional[str]:
    """*value* as stripped text, or ``None`` for a blank or missing value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _number(value: Any) -> Optional[float]:
    """*value* as a finite float, or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class LineRef:
    """One transition: its wavelength, plus what else is known about it.

    Attributes
    ----------
    lam : float
        Wavelength in um.
    molecule : str, optional
        Name of the molecule the line belongs to (its ``molecules_dict`` key).
    lev_up, lev_low : str, optional
        Upper and lower level labels.  They tell apart lines whose
        wavelengths agree within :data:`LAM_RTOL`.
    window : (float, float), optional
        Wavelength range to inspect the line over, e.g. a saved line's fit
        range.
    """
    lam: float
    molecule: Optional[str] = None
    lev_up: Optional[str] = None
    lev_low: Optional[str] = None
    window: Optional[Tuple[float, float]] = None

    @classmethod
    def from_info(cls, info: Optional[Mapping[str, Any]]) -> Optional["LineRef"]:
        """From a line inspection / population diagram info dict, or ``None`` without a wavelength.

        Reads ``lam``, ``molecule_name`` and the level labels (``up_lev`` /
        ``low_lev``, or ``lev_up`` / ``lev_low``).
        """
        if not info:
            return None
        lam = _number(info.get("lam"))
        if lam is None:
            return None
        return cls(
            lam=lam,
            molecule=_text(info.get("molecule_name")),
            lev_up=_text(info.get("up_lev", info.get("lev_up"))),
            lev_low=_text(info.get("low_lev", info.get("lev_low"))),
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any],
                 molecule: Optional[str] = None) -> Optional["LineRef"]:
        """From a line-table row, or ``None`` when it has no wavelength.

        Reads ``lam``, ``lev_up`` / ``lev_low``, the molecule from *molecule*
        or else the row's ``species``, and the window from ``xmin`` / ``xmax``
        when they are a range around the line.
        """
        lam = _number(row.get("lam"))
        if lam is None:
            return None
        xmin, xmax = _number(row.get("xmin")), _number(row.get("xmax"))
        window = None
        if xmin is not None and xmax is not None and xmin < xmax and xmin <= lam <= xmax:
            window = (xmin, xmax)
        return cls(
            lam=lam,
            molecule=_text(molecule) or _text(row.get("species")),
            lev_up=_text(row.get("lev_up")),
            lev_low=_text(row.get("lev_low")),
            window=window,
        )

    def describe(self) -> str:
        """Short label such as ``"H2O 17.2345 um"``."""
        where = f"{self.lam:.4f} um"
        return f"{self.molecule} {where}" if self.molecule else where


def _as_floats(values: Sequence[Any]) -> np.ndarray:
    """*values* as floats; text that is not a number becomes NaN."""
    return pd.to_numeric(pd.Series(list(values), dtype=object),
                         errors="coerce").to_numpy(dtype=float)


def match_line(ref: LineRef, lams: Sequence[Any],
               lev_up: Optional[Sequence[Any]] = None,
               lev_low: Optional[Sequence[Any]] = None) -> Optional[int]:
    """Position in *lams* of the line *ref* describes, or ``None``.

    A line whose level labels equal *ref*'s and whose wavelength is within
    :data:`LAM_RTOL` of it, or inside ``ref.window``, wins: a saved line's
    wavelength can be a fitted center rather than the catalog one.  Without
    one, the nearest line within :data:`LAM_RTOL` is taken.  *lams* may hold
    text; *lev_up* / *lev_low* are aligned with *lams*.
    """
    lam = _as_floats(lams)
    if lam.size == 0:
        return None
    # By position: a filtered table keeps its frame's index labels.
    lev_up, lev_low = _positional(lev_up), _positional(lev_low)
    target = float(ref.lam)
    diff = np.abs(lam - target)
    with np.errstate(invalid="ignore"):
        close = diff <= max(abs(target) * LAM_RTOL, 1e-12)
        near = close
        if ref.window is not None:
            lo, hi = ref.window
            near = close | ((lam >= lo) & (lam <= hi))

    def nearest(idx: np.ndarray) -> Optional[int]:
        return int(idx[np.argmin(diff[idx])]) if idx.size else None

    if ref.lev_up is not None or ref.lev_low is not None:
        # Level labels are compared for the nearby lines only: tables can
        # hold hundreds of thousands of rows.
        same = [i for i in np.flatnonzero(near)
                if _level_matches(lev_up, i, ref.lev_up)
                and _level_matches(lev_low, i, ref.lev_low)]
        found = nearest(np.asarray(same, dtype=int))
        if found is not None:
            return found
    return nearest(np.flatnonzero(close))


def _positional(values: Optional[Sequence[Any]]) -> Optional[Sequence[Any]]:
    """*values* indexable by position (a pandas Series is indexed by label)."""
    to_numpy = getattr(values, "to_numpy", None)
    return to_numpy() if callable(to_numpy) else values


def _level_matches(labels: Optional[Sequence[Any]], i: int,
                   wanted: Optional[str]) -> bool:
    """Whether label *i* of *labels* is *wanted* (always, when nothing is wanted)."""
    if wanted is None:
        return True
    if labels is None:
        return False
    try:
        return _text(labels[i]) == wanted
    except (IndexError, KeyError):
        return False
