"""
PartitionFunctionExtender - extend a partition function Q(T) beyond its
tabulated temperature range.

The intensity calculation interpolates Q(T) from the line list's table and
refuses temperatures outside it, so a molecule whose table stops at, say,
3500 K cannot be modelled hotter.  :func:`extend_partition_function` adds
temperature points below and/or above the table on a regular grid and fills
their Q values with an *extension mode*.

Modes are kept in a registry (:data:`EXTENSION_MODES`) so new ones - e.g. a
power-law or polynomial extrapolation - can be added with
:func:`register_extension_mode` without touching the callers or the GUI.
Currently the only mode is ``"constant"``, which holds Q(T) at the value of
the nearest tabulated edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

#: Upper bound on the number of temperature points one extension may add, to
#: protect against a tiny step over a huge range exhausting memory.
MAX_NEW_POINTS = 1_000_000

#: ``fill(t, q, new_t, side) -> q_new`` where *t*/*q* are the sorted original
#: table, *new_t* the (ascending) temperatures to fill and *side* is
#: ``"low"`` or ``"high"``.
FillFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, str], np.ndarray]


@dataclass(frozen=True)
class ExtensionMode:
    """A named way of filling Q(T) outside the tabulated range."""
    key: str
    label: str
    description: str
    fill: FillFunction


EXTENSION_MODES: Dict[str, ExtensionMode] = {}


def register_extension_mode(mode: ExtensionMode) -> None:
    """Add (or replace) an extension mode in :data:`EXTENSION_MODES`."""
    EXTENSION_MODES[mode.key] = mode


def get_extension_mode(key: str) -> ExtensionMode:
    """Return the registered mode for *key* or raise ``ValueError``."""
    try:
        return EXTENSION_MODES[key]
    except KeyError:
        raise ValueError(
            f"Unknown extension mode {key!r}; available: {sorted(EXTENSION_MODES)}"
        ) from None


def _constant_fill(t: np.ndarray, q: np.ndarray, new_t: np.ndarray, side: str) -> np.ndarray:
    edge_q = q[0] if side == "low" else q[-1]
    return np.full(new_t.shape, edge_q, dtype=np.float64)


register_extension_mode(ExtensionMode(
    key="constant",
    label="Constant Q(T)",
    description=("Hold Q(T) at the value of the nearest end of the table: points "
                 "below the table take Q at the lowest tabulated temperature, "
                 "points above it take Q at the highest."),
    fill=_constant_fill,
))


@dataclass
class ExtendedPartition:
    """Result of :func:`extend_partition_function`.

    Attributes
    ----------
    t, q : np.ndarray
        Ascending temperatures [K] and partition function values, original
        and added points together.
    is_extended : np.ndarray
        Boolean mask, ``True`` for the points added by the extension.
    mode : str
        Key of the extension mode used.
    step : float
        Temperature spacing of the added points [K].
    """
    t: np.ndarray
    q: np.ndarray
    is_extended: np.ndarray
    mode: str = "constant"
    step: float = 1.0
    n_added_low: int = 0
    n_added_high: int = 0
    _notes: List[str] = field(default_factory=list, repr=False)

    @property
    def n_added(self) -> int:
        return self.n_added_low + self.n_added_high

    @property
    def notes(self) -> List[str]:
        """Human-readable remarks, e.g. a requested bound inside the table."""
        return list(self._notes)

    def as_frame(self):
        """Return a ``pandas.DataFrame`` with columns ``'temp'`` and ``'q'``."""
        import pandas as pd
        return pd.DataFrame({"temp": self.t, "q": self.q})


def suggest_step(t) -> float:
    """A sensible default step: the table's median spacing (1 K if unknown)."""
    t = np.unique(np.asarray(t, dtype=np.float64))
    if t.size < 2:
        return 1.0
    step = float(np.median(np.diff(t)))
    return step if np.isfinite(step) and step > 0 else 1.0


def _grid_outward(edge: float, target: float, step: float) -> np.ndarray:
    """Temperatures from *edge* (exclusive) to *target* (inclusive), *step* apart.

    Points sit at ``edge ± k*step``; when the range is not a whole number of
    steps the target itself is appended so the requested bound is reached.
    The result is ascending.
    """
    span = abs(target - edge)
    if span <= 0:
        return np.empty(0, dtype=np.float64)
    tol = 1e-9 * max(1.0, span / step)
    n_full = int(np.floor(span / step + tol))
    if n_full > MAX_NEW_POINTS:
        raise ValueError(
            f"Extending to {target:g} K with a {step:g} K step would add more than "
            f"{MAX_NEW_POINTS:,} points; use a larger step."
        )
    direction = 1.0 if target > edge else -1.0
    pts = edge + direction * step * np.arange(1, n_full + 1, dtype=np.float64)
    if pts.size and abs(pts[-1] - target) <= tol * step:
        pts[-1] = target
    elif not pts.size or pts[-1] != target:
        pts = np.append(pts, target)
    return np.sort(pts)


def _as_table(t, q):
    t = np.asarray(t, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if t.size == 0 or t.shape != q.shape:
        raise ValueError("The partition function needs matching, non-empty T and Q arrays.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
        raise ValueError("The partition function contains non-finite values.")
    order = np.argsort(t, kind="stable")
    return t[order], q[order]


def extend_partition_function(
    t,
    q,
    *,
    t_low: Optional[float] = None,
    t_high: Optional[float] = None,
    step: float = 1.0,
    mode: str = "constant",
) -> ExtendedPartition:
    """Extend the table ``(t, q)`` down to *t_low* and/or up to *t_high*.

    Parameters
    ----------
    t, q : array-like
        Tabulated temperatures [K] and Q(T); any order.
    t_low, t_high : float, optional
        New lowest / highest temperature [K].  ``None`` - or a value that is
        already inside the table - leaves that side unchanged (the table is
        never truncated).
    step : float
        Spacing [K] of the added points, counted outward from the table edge.
    mode : str
        Key of a registered :class:`ExtensionMode` (``"constant"``).

    Returns
    -------
    ExtendedPartition

    Raises
    ------
    ValueError
        For invalid input: empty/mismatched tables, a non-positive step or
        lowest temperature, ``t_low >= t_high``, an unknown mode, or more
        than :data:`MAX_NEW_POINTS` points per side.
    """
    t, q = _as_table(t, q)
    ext_mode = get_extension_mode(mode)

    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError("The step must be a number.") from None
    if not np.isfinite(step) or step <= 0:
        raise ValueError("The step must be a positive number of kelvin.")
    for name, value in (("lowest", t_low), ("highest", t_high)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"The {name} temperature must be a finite number.")
    if t_low is not None and t_low <= 0:
        raise ValueError("The lowest temperature must be above 0 K.")
    if t_low is not None and t_high is not None and t_low >= t_high:
        raise ValueError("The lowest temperature must be below the highest temperature.")

    notes: List[str] = []
    low_t = np.empty(0, dtype=np.float64)
    high_t = np.empty(0, dtype=np.float64)
    if t_low is not None:
        if t_low < t[0]:
            low_t = _grid_outward(float(t[0]), float(t_low), step)
        elif t_low > t[0]:
            notes.append(f"Lowest T {t_low:g} K is inside the table; nothing added below.")
    if t_high is not None:
        if t_high > t[-1]:
            high_t = _grid_outward(float(t[-1]), float(t_high), step)
        elif t_high < t[-1]:
            notes.append(f"Highest T {t_high:g} K is inside the table; nothing added above.")

    low_q = np.asarray(ext_mode.fill(t, q, low_t, "low"), dtype=np.float64) if low_t.size else low_t
    high_q = np.asarray(ext_mode.fill(t, q, high_t, "high"), dtype=np.float64) if high_t.size else high_t

    new_t = np.concatenate([low_t, t, high_t])
    new_q = np.concatenate([low_q, q, high_q])
    mask = np.concatenate([
        np.ones(low_t.size, dtype=bool),
        np.zeros(t.size, dtype=bool),
        np.ones(high_t.size, dtype=bool),
    ])
    return ExtendedPartition(
        t=new_t, q=new_q, is_extended=mask, mode=ext_mode.key, step=step,
        n_added_low=int(low_t.size), n_added_high=int(high_t.size), _notes=notes,
    )
