"""
LineFilterExpression - two-level boolean filter expressions for line lists.

Provides a small, serializable data model for combining line-list predicates
with AND/OR logic::

    (6 <= lam <= 7  OR  12 <= lam <= 13  OR  E_up >= 5000)  AND  (A >= 5e-3)

The model is deliberately **exactly two levels deep**: a
:class:`FilterExpression` holds :class:`ConditionGroup` objects, and each group
holds :class:`Condition` objects.  A group combines its own conditions with one
operator; the expression combines the group masks with one operator.  Groups
never nest.

That restriction is the point.  Because every condition belongs to exactly one
group, and each level has a single operator, there is no operator-precedence
rule to learn and no expression whose visual reading can differ from its
evaluation.  Two levels are also fully general over these predicates - every
boolean formula has a disjunctive normal form, so anything expressible with
arbitrary nesting is expressible here.

This module depends only on numpy/pandas.  It never imports
:mod:`LineListMaker`, :mod:`MoleculeLineList`, or tkinter, so the dependency
direction is strictly ``LineListMaker -> LineFilterExpression``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd

__all__ = [
    "SCHEMA_VERSION",
    "C_KM_S",
    "Condition",
    "ConditionError",
    "ConditionGroup",
    "FilterExpression",
    "MaskContext",
    "condition_mask",
    "describe_condition",
    "describe_expression",
    "describe_group",
    "expression_mask",
    "fold_masks",
    "group_mask",
    "nearest_neighbour_separation_kms",
    "ops_for_kind",
    "validate",
    "velocity_separation_kms",
]

SCHEMA_VERSION = 1

#: Speed of light in km/s, for Doppler velocity separations.
C_KM_S = 299792.458


# ---------------------------------------------------------------------------
# Velocity-separation helpers
# ---------------------------------------------------------------------------

def velocity_separation_kms(lam1: Any, lam2: Any) -> np.ndarray:
    """Doppler velocity separation (km/s) between two wavelengths.

    Both arguments are in the same wavelength unit; the result is
    ``c * |λ1 - λ2| / mean(λ1, λ2)``, so the unit cancels.
    """
    a = np.asarray(lam1, dtype=float)
    b = np.asarray(lam2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return C_KM_S * np.abs(a - b) / (0.5 * (a + b))


def nearest_neighbour_separation_kms(lam: Any) -> np.ndarray:
    """Velocity separation (km/s) from each line to its closest neighbour.

    A line with no neighbour at all gets ``+inf`` (it is infinitely isolated,
    so ``Δv ≥ x`` keeps it and ``Δv ≤ x`` does not), and a line whose own
    wavelength is missing or non-positive gets ``NaN`` so it is excluded from
    both polarities of every bound.
    """
    lam = np.asarray(lam, dtype=float)
    out = np.full(lam.shape, np.nan, dtype=float)
    usable = np.isfinite(lam) & (lam > 0)
    vals = lam[usable]
    if vals.size == 0:
        return out
    if vals.size == 1:
        out[usable] = np.inf
        return out
    order = np.argsort(vals, kind="stable")
    ordered = vals[order]
    gaps = velocity_separation_kms(ordered[:-1], ordered[1:])
    to_left = np.concatenate(([np.inf], gaps))
    to_right = np.concatenate((gaps, [np.inf]))
    nearest = np.empty_like(ordered)
    nearest[order] = np.minimum(to_left, to_right)
    out[usable] = nearest
    return out


# ---------------------------------------------------------------------------
# Vibrational-band helpers
# ---------------------------------------------------------------------------

def _vib_part(level_str: str) -> str:
    """Return the vibrational portion of a quantum-state label (before ``'|'``).

    e.g. ``'0_0_1|16_1_15'`` -> ``'0_0_1'``.
    Falls back to the full string when no ``'|'`` is present.
    """
    return level_str.split("|")[0] if "|" in level_str else level_str


def _vib_perms(n: int, n_modes: int = 3) -> set:
    """All underscore-joined vibrational-mode tuples where ``max(qi) == n``.

    Parameters
    ----------
    n : int
        Target vibrational quantum number (the maximum across all modes).
    n_modes : int
        Number of vibrational modes (default 3 for H₂O, CO₂, etc.).
    """
    from itertools import product as _product
    combos: set = set()
    for vals in _product(range(n + 1), repeat=n_modes):
        if max(vals) == n:
            combos.add("_".join(str(v) for v in vals))
    return combos


def _vib_perms_up_to(n: int, n_modes: int = 3) -> set:
    """All underscore-joined vibrational-mode tuples where ``max(qi) <= n``."""
    result: set = set()
    for m in range(n + 1):
        result |= _vib_perms(m, n_modes)
    return result


def _parse_vib_band(spec: str, n_modes: int = 3):
    """Parse a band-spec string into ``(up_set, low_set)``.

    Supported formats
    -----------------
    ``"v1"``   -> upper from exact v=1 states (``max(qi) == 1``),
                  lower from *any* state up to v=1 (``max(qi) <= 1``).
                  Captures all lines *from* the v=1 level, including both
                  the fundamental band (v=1 → v=0) and hot bands (v=1 → v=1).
    ``"v1-0"`` -> upper from exact v=1, lower from exact v=0.
    ``"v2-1"`` -> upper from exact v=2, lower from exact v=1.

    The leading ``'v'`` is optional.  For the two-number ``"vN-M"`` form both
    sides use **exact** level matching.

    Returns
    -------
    (up_set, low_set) : tuple of set of str
        Sets of underscore-joined vibrational labels.

    Raises
    ------
    ValueError
        If the spec cannot be parsed.
    """
    spec = spec.strip().lower().lstrip("v")
    if not spec:
        raise ValueError("Empty band spec.")
    parts = spec.split("-")
    if len(parts) == 1:
        n_up = int(parts[0])
        up_set = _vib_perms(n_up, n_modes)
        low_set = _vib_perms_up_to(n_up, n_modes)  # cumulative: captures all bands from this level
    elif len(parts) == 2:
        n_up = int(parts[0])
        n_low = int(parts[1])
        up_set = _vib_perms(n_up, n_modes)
        low_set = _vib_perms(n_low, n_modes)  # exact lower level
    else:
        raise ValueError(f"Cannot parse band spec: {spec!r}")
    return up_set, low_set


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConditionError(ValueError):
    """A condition cannot be evaluated against the given DataFrame.

    Carries a ``(group_index, condition_index)`` path so a GUI can red-flag the
    exact row that failed.

    This is **raised, never warned**, because under OR there is no safe identity
    for an unevaluable condition: treating it as all-True widens the whole group
    to the entire line list, while all-False silently deletes a disjunct.
    Warn-and-no-op is defensible under AND (``all-True & X == X``); it is
    catastrophic under OR (``all-True | X == everything``).
    """

    def __init__(self, message: str, path: Tuple[int, int] = (-1, -1)) -> None:
        super().__init__(message)
        self.message = message
        self.path = path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One predicate row.  All fields are JSON-native scalars.

    Attributes
    ----------
    kind : str
        ``"range"``, ``"derived"``, ``"velocity_sep"``, ``"quantum_label"``,
        ``"quantum_field"``, ``"vib_band"``, or ``"species"``.
    field : str
        Column name (``range``) or quantum-field name (``quantum_field``);
        empty otherwise.
    op : str
        See :func:`ops_for_kind`.  Authoritative - the active bounds are
        *derived* from it, so ``op="ge"`` never reads ``max_val``.
    value : Any
        String / exact operand: the pattern for ``quantum_label``, the exact
        value for ``quantum_field``, the spec for ``vib_band``, or the name
        list for ``species``.
    min_val, max_val : float, optional
        Numeric operands.  ``between`` uses both, ``ge`` uses *min_val*,
        ``le`` uses *max_val*, ``eq`` uses *min_val* for both bounds.
    unit : str
        How the bounds are read.  ``""`` (the default) means absolute values.
        ``"percent"`` makes each bound a percentage of the property's largest
        value across the frame - so ``I ≥ 5%`` reads "at least a twentieth of
        the brightest line".  ``"percentile"`` makes each bound a rank in
        0-100 within the property's own distribution - so ``I ≥ p90`` reads
        "the strongest tenth of the lines".  Only ``range`` and ``derived``
        support either.
    state : str
        ``"upper"`` reads ``lev_up``; anything else reads ``lev_low``.
    n_modes : int
        Vibrational mode count, ``vib_band`` only.
    regex : bool
        ``quantum_label`` only.  ``False`` (the default) means a **literal**
        substring, so a label fragment like ``"0|"`` matches those characters
        rather than acting as a regex alternation.
    negate : bool
        Invert this condition.  Rows whose value is missing (NaN / sentinel)
        are selected by **neither** polarity - see :func:`_polarity`.
    enabled : bool
        ``False`` means this condition is **absent** from the fold, not True
        and not False.
    """

    kind: str
    field: str = ""
    op: str = ""
    value: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    unit: str = ""
    state: str = "upper"
    n_modes: int = 3
    regex: bool = False
    negate: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        # Resolve the blank sentinel to the kind's natural operator, and reject
        # an operator that does not belong to the kind.  Without this a
        # Condition("vib_band", value="v1") would carry a range operator: the
        # mask builders that ignore `op` would apply it happily, but the
        # recorded log would then fail to parse back and break summary(),
        # `expression`, and replay.  Making the invalid state unrepresentable
        # is cheaper than defending every consumer.
        if self.unit not in ("", "percent", "percentile"):
            raise ConditionError(
                f"Unknown unit {self.unit!r}; expected '', 'percent' or "
                f"'percentile'."
            )
        if self.unit:
            if self.kind not in _RELATIVE_KINDS:
                raise ConditionError(
                    f"A relative bound is not meaningful for a {self.kind!r} "
                    f"condition; it is only defined for "
                    f"{', '.join(sorted(_RELATIVE_KINDS))}."
                )
            # A percentile above 100 has no rank to look up; a percentage
            # above it is merely a bound nothing reaches, which is legal.
            ceiling = 100.0 if self.unit == "percentile" else float("inf")
            for bound in (self.min_val, self.max_val):
                if bound is not None and not 0.0 <= bound <= ceiling:
                    limit = ("between 0 and 100" if ceiling == 100.0
                             else "0 or more")
                    raise ConditionError(
                        f"A {self.unit} bound must be {limit}, not {bound:g}."
                    )
        valid = ops_for_kind(self.kind)
        if not valid:
            return  # unknown kind - condition_mask raises with a better message
        if not self.op:
            object.__setattr__(self, "op", valid[0])
        elif self.op not in valid:
            raise ConditionError(
                f"Operator {self.op!r} is not valid for a {self.kind!r} "
                f"condition.  Expected one of: {', '.join(valid)}."
            )


@dataclass(frozen=True)
class ConditionGroup:
    """A parenthesised run of conditions combined by ONE operator.

    Attributes
    ----------
    conditions : tuple of Condition
        The members of this group.
    join : str
        ``"AND"`` or ``"OR"`` - how the conditions *inside* this group combine.
    label : str
        Optional user-facing name, carried through serialization.
    enabled : bool
        ``False`` means the whole group is absent from the outer fold.
    """

    conditions: Tuple[Condition, ...] = ()
    join: str = "AND"
    label: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        # Normalise here so no consumer can ever compare a raw "All"/"Any"
        # against "AND"/"OR" and silently fold with the wrong operator.
        object.__setattr__(self, "join", _norm_join(self.join))


@dataclass(frozen=True)
class FilterExpression:
    """Groups combined by ONE outer operator.  Exactly two levels, no recursion.

    Attributes
    ----------
    groups : tuple of ConditionGroup
        The groups to combine.
    join : str
        ``"AND"`` or ``"OR"`` - how the *group masks* combine.
    version : int
        Schema version, for forward-compatibility checks on load.
    """

    groups: Tuple[ConditionGroup, ...] = ()
    join: str = "AND"
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Normalise here so no consumer can ever compare a raw "All"/"Any"
        # against "AND"/"OR" and silently fold with the wrong operator.
        object.__setattr__(self, "join", _norm_join(self.join))

    # -- serialization ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-native dict; default-valued keys are omitted."""
        return {
            "version": self.version,
            "join": _norm_join(self.join),
            "groups": [_group_to_dict(g) for g in self.groups],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FilterExpression":
        """Rebuild from :meth:`to_dict` output.

        Tolerant of unknown and missing keys; strict about an unknown
        ``kind``/``op`` and about a ``version`` newer than
        :data:`SCHEMA_VERSION`.
        """
        if not isinstance(d, Mapping):
            raise ConditionError(
                f"Expected a mapping for a filter expression, got "
                f"{type(d).__name__}."
            )
        version = int(d.get("version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise ConditionError(
                f"Filter expression schema version {version} is newer than the "
                f"supported version {SCHEMA_VERSION}; upgrade iSLAT to load it."
            )
        groups = tuple(
            _group_from_dict(g, gi) for gi, g in enumerate(d.get("groups", ()))
        )
        return cls(groups=groups, join=_norm_join(d.get("join", "AND")),
                   version=version)


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

_OPS_BY_KIND: Dict[str, Tuple[str, ...]] = {
    "range":         ("between", "ge", "le", "eq"),
    "derived":       ("between", "ge", "le", "eq"),
    "velocity_sep":  ("between", "ge", "le", "eq"),
    "quantum_field": ("between", "ge", "le", "eq"),
    "quantum_label": ("contains", "eq", "matches"),
    "vib_band":      ("band",),
    "species":       ("in",),
}

_RANGE_OPS = frozenset({"between", "ge", "le", "eq"})

#: Kinds whose bounds may be given relative to the property's own values.
_RELATIVE_KINDS = frozenset({"range", "derived"})


def ops_for_kind(kind: str) -> Tuple[str, ...]:
    """Valid operator keys for *kind*; an empty tuple for an unknown kind."""
    return _OPS_BY_KIND.get(kind, ())


def _norm_join(join: Any) -> str:
    """Normalise a join operator to ``"AND"`` or ``"OR"``.

    Accepts the GUI's ``"All"``/``"Any"`` wording as well as ``"AND"``/``"OR"``.
    """
    text = str(join).strip().upper()
    if text in ("OR", "ANY"):
        return "OR"
    if text in ("AND", "ALL"):
        return "AND"
    raise ConditionError(f"Unknown join operator {join!r}; expected AND or OR.")


def _bounds(c: Condition) -> Tuple[Optional[float], Optional[float]]:
    """Derive the active ``(lo, hi)`` bounds from ``c.op``.

    The operator is authoritative, so a stale ``max_val`` left in the dataclass
    by an earlier edit can never leak into a ``ge`` comparison.
    """
    if c.op == "between":
        return c.min_val, c.max_val
    if c.op == "ge":
        return c.min_val, None
    if c.op == "le":
        return None, c.max_val
    if c.op == "eq":
        return c.min_val, c.min_val
    raise ConditionError(f"Operator {c.op!r} is not a numeric range operator.")


def _scaled_bounds(c: Condition, values: np.ndarray,
                   name: str) -> Tuple[Optional[float], Optional[float]]:
    """Bounds in the units of *values*, resolving a relative bound against them.

    ``"percent"`` scales by the largest value present, so ``≥ 5%`` follows the
    brightest line however many faint ones surround it.  ``"percentile"``
    looks the bound up as a rank in the distribution, so ``≥ p90`` always
    keeps a tenth of the lines however they are spread.  Both are monotonic in
    their operand, so a ``between`` window survives the conversion in order.
    """
    lo, hi = _bounds(c)
    if not c.unit:
        return lo, hi
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ConditionError(
            f"Cannot take a {c.unit} of {name!r}: it has no values in this "
            f"line list."
        )
    if c.unit == "percentile":
        convert = lambda rank: float(np.percentile(finite, rank))  # noqa: E731
    else:
        peak = float(finite.max())
        if peak <= 0:
            raise ConditionError(
                f"Cannot take a percentage of {name!r}: its largest value in "
                f"this line list is not positive."
            )
        convert = lambda pct: pct * peak / 100.0  # noqa: E731
    return (None if lo is None else convert(lo),
            None if hi is None else convert(hi))


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------

@dataclass
class MaskContext:
    """Per-evaluation state.  Not frozen: it holds the quantum-parse cache.

    Attributes
    ----------
    species : str, optional
        The schema lookup key.  This must be the same value the property
        catalogue was built from, so the fields a user can pick are exactly the
        fields that can be evaluated.
    on_error : str
        ``"raise"`` (default), ``"identity"``, or ``"drop"``.  See
        :func:`group_mask`.
    derived_source : callable, optional
        Zero-argument provider of a DataFrame of model quantities (line
        intensity, optical depth, ...) indexed like the frame being filtered.
        Invoked at most once per evaluation, because producing it runs the
        intensity physics; a ``derived`` condition without one raises.

    Notes
    -----
    The parse cache is a correctness prerequisite rather than an optimisation:
    boolean composition makes every condition see the same full baseline frame,
    so N quantum conditions would otherwise trigger N identical ``parse_bulk``
    passes over the same labels.
    """

    species: Optional[str] = None
    on_error: str = "raise"
    derived_source: Optional[Callable[[], Any]] = _dc_field(
        default=None, repr=False, compare=False
    )
    _parse_cache: Dict[str, Dict[str, np.ndarray]] = _dc_field(
        default_factory=dict, repr=False, compare=False
    )
    _sep_cache: Dict[str, np.ndarray] = _dc_field(
        default_factory=dict, repr=False, compare=False
    )
    _derived_frame: Optional[pd.DataFrame] = _dc_field(
        default=None, repr=False, compare=False
    )

    def derived_array(self, df: pd.DataFrame, key: str) -> np.ndarray:
        """Model quantity *key* for exactly the rows of *df*.

        Aligned by index rather than by position, so a subsampled preview
        frame or any other row subset reads the matching values instead of
        the first N of them.
        """
        if self._derived_frame is None:
            if self.derived_source is None:
                raise ConditionError(
                    f"{key!r} is a model quantity, which needs a molecule "
                    f"with computed intensities; none is available here."
                )
            try:
                frame = self.derived_source()
            except ConditionError:
                raise
            except Exception as exc:
                raise ConditionError(
                    f"Could not compute model quantities: {exc}") from exc
            if frame is None:
                raise ConditionError(
                    "No model quantities are available for this line list.")
            self._derived_frame = frame
        frame = self._derived_frame
        if key not in getattr(frame, "columns", ()):
            raise ConditionError(
                f"Model quantity {key!r} is not available for this line list."
            )
        column = frame[key]
        if not df.index.isin(column.index).all():
            raise ConditionError(
                f"Model quantity {key!r} does not line up with this line list."
            )
        return column.reindex(df.index).to_numpy(dtype=float)

    def neighbour_separation(self, df: pd.DataFrame) -> np.ndarray:
        """Nearest-neighbour velocity separation (km/s) for every row of *df*.

        Cached for the same reason the parse cache exists: boolean composition
        shows every condition the same full baseline frame, so N separation
        conditions would otherwise trigger N identical sorts of the same
        wavelength column.
        """
        cached = self._sep_cache.get("lam")
        if cached is not None:
            return cached
        if "lam" not in df.columns:
            raise ConditionError("This line list has no 'lam' column.")
        sep = nearest_neighbour_separation_kms(
            pd.to_numeric(df["lam"], errors="coerce").to_numpy())
        self._sep_cache["lam"] = sep
        return sep

    def parsed_levels(self, df: pd.DataFrame, lev_col: str) -> Dict[str, np.ndarray]:
        """Parse *lev_col* into quantum fields, caching per column."""
        cached = self._parse_cache.get(lev_col)
        if cached is not None:
            return cached
        if lev_col not in df.columns:
            raise ConditionError(f"This line list has no {lev_col!r} column.")
        from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
        import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401  (registers schemas)

        schema = QuantumStateRegistry.get_schema(self.species)
        labels = np.array(df[lev_col].fillna(""), dtype="U64")
        parsed = schema.parse_bulk(labels)
        self._parse_cache[lev_col] = parsed
        return parsed


# ---------------------------------------------------------------------------
# Mask primitives - the single chokepoint
# ---------------------------------------------------------------------------

def _as_mask(obj: Any, index: pd.Index) -> pd.Series:
    """Coerce any builder return into a clean bool Series indexed like *index*.

    Handles, in order: 0-d arrays and numpy scalars (broadcast to full length);
    nullable ``boolean`` dtype carrying ``pd.NA`` (``fillna(False)`` *before*
    ``astype(bool)``, since ``astype(bool)`` on ``pd.NA`` raises); a length
    mismatch; and an index mismatch.

    Every builder's result passes through here, so a future builder cannot
    forget one of these cases.
    """
    if isinstance(obj, pd.Series):
        if not obj.index.equals(index):
            raise ConditionError("Internal: mask index does not match the frame.")
        return obj.fillna(False).astype(bool)
    arr = np.asarray(obj)
    if arr.ndim == 0:
        arr = np.full(len(index), bool(arr))
    if arr.shape[0] != len(index):
        raise ConditionError(
            f"Internal: mask length {arr.shape[0]} != frame length {len(index)}."
        )
    return pd.Series(arr.astype(bool), index=index)


def _polarity(pred: pd.Series, valid: pd.Series, negate: bool) -> pd.Series:
    """Apply negation without resurrecting rows whose value is unknown.

    NaN and sentinel rows are selected by **neither** polarity: "NOT
    E_up >= 5000" must not sweep in lines whose E_up is unknown, because
    unknown is not the same as "below the threshold".
    """
    return (valid & ~pred) if negate else (valid & pred)


def _field_valid(vals: np.ndarray) -> np.ndarray:
    """Rows whose parsed quantum field actually carries a value.

    QuantumStateSchema marks an unparsed field with -999 for integers and NaN
    for floats, and with an empty string for text fields.  Those rows are
    "unknown", not "some other value", so they must be excluded from BOTH
    polarities of every operator - not just the range ones.
    """
    if vals.dtype.kind in "iuf":
        numeric = np.asarray(vals, dtype=float)
        return np.isfinite(numeric) & (numeric != -999)
    return np.asarray([str(v).strip() != "" for v in vals], dtype=bool)


def _safe_eq(arr: np.ndarray, target: Any, name: str) -> np.ndarray:
    """Element-wise equality that cannot silently produce an all-False branch.

    Comparing an integer array against an incomparable string returns an
    all-False array (not a scalar) on current numpy, so a mistyped exact value
    would otherwise become an invisibly dropped disjunct under OR.  Coercing by
    dtype kind and raising on a genuinely incomparable operand makes that
    failure loud.
    """
    try:
        if arr.dtype.kind in "iuf":
            return np.asarray(arr, dtype=float) == float(target)
        return np.asarray(arr, dtype=str) == str(target)
    except (TypeError, ValueError):
        raise ConditionError(
            f"{target!r} is not a valid value for {name!r} "
            f"(field dtype is {arr.dtype})."
        )


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------

def _mask_range(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Numeric range on a DataFrame column."""
    if c.field not in df.columns:
        raise ConditionError(f"This line list has no {c.field!r} column.")
    s = pd.to_numeric(df[c.field], errors="coerce")
    valid = s.notna()
    if _bounds(c) == (None, None):
        raise ConditionError(
            f"Condition on {c.field!r} has no value - fill in a bound."
        )
    lo, hi = _scaled_bounds(c, s.to_numpy(dtype=float), c.field)
    pred = valid.copy()
    if lo is not None:
        pred &= (s >= lo)
    if hi is not None:
        pred &= (s <= hi)
    return _as_mask(_polarity(_as_mask(pred, df.index), valid, c.negate), df.index)


def _mask_derived(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Numeric range on a model quantity (line intensity, optical depth, ...).

    These are not columns of the line list: they come from the same intensity
    calculation the population diagram plots, so a filter on them tracks the
    molecule's current temperature, column density and broadening.
    """
    if not c.field:
        raise ConditionError("Model-quantity condition has no property.")
    vals = ctx.derived_array(df, c.field)
    if _bounds(c) == (None, None):
        raise ConditionError(
            f"Condition on {c.field!r} has no value - fill in a bound."
        )
    lo, hi = _scaled_bounds(c, vals, c.field)
    valid_arr = np.isfinite(vals)
    pred_arr = valid_arr.copy()
    with np.errstate(invalid="ignore"):
        if lo is not None:
            pred_arr &= vals >= lo
        if hi is not None:
            pred_arr &= vals <= hi
    valid = _as_mask(valid_arr, df.index)
    return _as_mask(_polarity(_as_mask(pred_arr, df.index), valid, c.negate),
                    df.index)


def _mask_velocity_sep(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Doppler separation (km/s) from each line to its closest neighbour.

    The neighbour search runs over the whole frame the expression is evaluated
    against, so ``\u0394v \u2265 5`` selects lines that are isolated within the *line
    list*, not merely within whatever other conditions happen to keep.
    """
    lo, hi = _bounds(c)
    if lo is None and hi is None:
        raise ConditionError(
            "Velocity-separation condition has no value - fill in a bound."
        )
    sep = ctx.neighbour_separation(df)
    valid_arr = ~np.isnan(sep)
    pred_arr = valid_arr.copy()
    with np.errstate(invalid="ignore"):
        if lo is not None:
            pred_arr &= sep >= lo
        if hi is not None:
            pred_arr &= sep <= hi
    valid = _as_mask(valid_arr, df.index)
    return _as_mask(_polarity(_as_mask(pred_arr, df.index), valid, c.negate),
                    df.index)


def _mask_quantum_label(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Substring / regex / exact match on the raw ``lev_up`` or ``lev_low`` text."""
    col = "lev_up" if c.state == "upper" else "lev_low"
    if col not in df.columns:
        raise ConditionError(f"This line list has no {col!r} column.")
    s = df[col].astype("string")
    valid = pd.Series(True, index=df.index)  # an empty label is still evaluable
    pat = "" if c.value is None else str(c.value)
    if not pat:
        raise ConditionError(
            f"Condition on {col!r} has no text - type the label fragment to match."
        )
    try:
        if c.op == "contains":
            pred = s.str.contains(pat, na=False, regex=bool(c.regex))
        elif c.op == "matches":
            pred = s.str.contains(pat, na=False, regex=True)
        elif c.op == "eq":
            pred = (s == pat)
        else:
            raise ConditionError(
                f"Operator {c.op!r} is not valid for a level label.")
    except ConditionError:
        raise
    except Exception as exc:
        # A malformed pattern surfaces as a re.error or a pyarrow ArrowInvalid
        # depending on the pandas string backend; both must become the one
        # error type callers handle, or they escape and kill the live preview.
        raise ConditionError(
            f"{pat!r} is not a valid search pattern: {exc}") from exc
    return _as_mask(_polarity(_as_mask(pred, df.index), valid, c.negate), df.index)


def _mask_quantum_field(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Parsed quantum-number field, via the species schema."""
    col = "lev_up" if c.state == "upper" else "lev_low"
    parsed = ctx.parsed_levels(df, col)  # raises when the column is absent
    if c.field not in parsed:
        raise ConditionError(
            f"Quantum field {c.field!r} is not defined by the schema for species "
            f"{ctx.species!r}.  Available: {', '.join(sorted(parsed)) or '(none)'}"
        )
    vals = np.asarray(parsed[c.field])
    if c.op == "eq":
        if c.value is None or str(c.value) == "":
            raise ConditionError(
                f"Condition on quantum field {c.field!r} has no value."
            )
        pred = _as_mask(_safe_eq(vals, c.value, c.field), df.index)
        # Sentinel-aware, exactly like the range branch below: "J != 5" must
        # not sweep in lines whose J could not be parsed at all.
        valid = _as_mask(_field_valid(vals), df.index)
        return _as_mask(_polarity(pred, valid, c.negate), df.index)
    try:
        v = np.asarray(vals, dtype=float)
    except (ValueError, TypeError):
        raise ConditionError(
            f"Quantum field {c.field!r} is not numeric - use '=' instead of a range."
        )
    lo, hi = _bounds(c)
    if lo is None and hi is None:
        raise ConditionError(
            f"Condition on quantum field {c.field!r} has no value - fill in a bound."
        )
    valid_arr = _field_valid(vals)  # QuantumStateSchema missing-value sentinels
    pred_arr = valid_arr.copy()
    if lo is not None:
        pred_arr &= v >= lo
    if hi is not None:
        pred_arr &= v <= hi
    valid = _as_mask(valid_arr, df.index)
    return _as_mask(_polarity(_as_mask(pred_arr, df.index), valid, c.negate), df.index)


def _mask_vib_band(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Vibrational band membership on both the upper and lower level."""
    for col in ("lev_up", "lev_low"):
        if col not in df.columns:
            raise ConditionError(f"This line list has no {col!r} column.")
    spec = str(c.value or "").strip()
    if not spec:
        raise ConditionError(
            "Vibrational-band condition has no band - e.g. 'v1', 'v1-0', 'v2-1'."
        )
    try:
        up_set, low_set = _parse_vib_band(spec, int(c.n_modes))
    except ValueError as exc:
        raise ConditionError(
            f"Cannot parse vibrational band {spec!r}: {exc}"
        ) from exc
    pred = (
        df["lev_up"].map(lambda x: _vib_part(str(x)) in up_set)
        & df["lev_low"].map(lambda x: _vib_part(str(x)) in low_set)
    )
    valid = pd.Series(True, index=df.index)
    return _as_mask(_polarity(_as_mask(pred, df.index), valid, c.negate), df.index)


def _mask_species(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Membership test on the ``species`` column."""
    if "species" not in df.columns:
        raise ConditionError("This line list has no 'species' column.")
    if isinstance(c.value, (list, tuple)):
        names = list(c.value)
    elif c.value is None or str(c.value).strip() == "":
        raise ConditionError("Species condition has no names - type at least one.")
    else:
        names = [n.strip() for n in str(c.value).split(",") if n.strip()]
    if not names:
        raise ConditionError("Species condition has no names - type at least one.")
    pred = df["species"].astype(str).isin([str(n) for n in names])
    valid = pd.Series(True, index=df.index)
    return _as_mask(_polarity(_as_mask(pred, df.index), valid, c.negate), df.index)


_MASK_BUILDERS: Dict[str, Callable[..., pd.Series]] = {
    "range": _mask_range,
    "derived": _mask_derived,
    "velocity_sep": _mask_velocity_sep,
    "quantum_label": _mask_quantum_label,
    "quantum_field": _mask_quantum_field,
    "vib_band": _mask_vib_band,
    "species": _mask_species,
}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def condition_mask(df: pd.DataFrame, c: Condition, ctx: MaskContext) -> pd.Series:
    """Build one condition's boolean mask.

    Raises
    ------
    ConditionError
        When the condition cannot be evaluated against *df*.
    """
    build = _MASK_BUILDERS.get(c.kind)
    if build is None:
        raise ConditionError(f"Unknown condition kind {c.kind!r}.")
    try:
        return _as_mask(build(df, c, ctx), df.index)
    except ConditionError:
        raise
    except Exception as exc:
        # Single chokepoint: every failure reaches callers as ConditionError,
        # so one unforeseen library error cannot escape a preview callback.
        raise ConditionError(
            f"Could not evaluate this condition: {exc}") from exc


def group_mask(df: pd.DataFrame, g: ConditionGroup,
               ctx: MaskContext) -> Optional[pd.Series]:
    """Fold one group's conditions with ``g.join``.

    Returns
    -------
    pd.Series or None
        ``None`` means the group is **vacuous** and must be dropped from the
        outer fold.  A vacuous group is not True and not False - it is absent.
        Under an outer AND, dropping it is equivalent to contributing True;
        under an outer OR it is equivalent to contributing False.  One rule,
        correct under both joins.

        Folding from operator identities instead is the classic bug: with an
        outer OR, an empty group contributing all-True would return the entire
        line list the instant the user clicks "+ Group".  The guarantee here is
        ``OR(a, EMPTY) == a``.
    """
    active = [c for c in g.conditions if c.enabled]
    if not g.enabled or not active:
        return None
    join = _norm_join(g.join)
    out: Optional[pd.Series] = None
    for c in active:
        try:
            m = condition_mask(df, c, ctx)
        except ConditionError:
            if ctx.on_error == "drop":
                continue
            if ctx.on_error == "identity":
                m = pd.Series(True, index=df.index)
            else:
                raise
        out = m if out is None else ((out & m) if join == "AND" else (out | m))
    return out


def fold_masks(masks: Sequence[pd.Series], join: str,
               index: pd.Index) -> pd.Series:
    """Combine already-built masks with *join*; all-True when there are none.

    Shared by :func:`expression_mask` and by the filter window's live preview,
    so the count shown on screen and the rows actually kept can never diverge.
    """
    if not masks:
        return pd.Series(True, index=index, dtype=bool)
    join = _norm_join(join)
    out = masks[0]
    for m in masks[1:]:
        out = (out & m) if join == "AND" else (out | m)
    return out


def expression_mask(df: pd.DataFrame, expr: FilterExpression,
                    ctx: MaskContext) -> pd.Series:
    """Fold the group masks with ``expr.join``.

    Returns an all-True mask when nothing contributes, so an expression with no
    conditions is a no-op rather than an empty result.
    """
    if len(df.index) == 0:
        # Short-circuit: some quantum schemas return no fields at all for an
        # empty label array, which would make every quantum condition raise.
        return pd.Series(True, index=df.index, dtype=bool)
    masks: List[pd.Series] = []
    for gi, g in enumerate(expr.groups):
        try:
            m = group_mask(df, g, ctx)
        except ConditionError as exc:
            if exc.path == (-1, -1):
                raise ConditionError(exc.message, (gi, -1)) from exc
            raise
        if m is not None:
            masks.append(m)
    return fold_masks(masks, expr.join, df.index)


def validate(df: pd.DataFrame, expr: FilterExpression,
             ctx: MaskContext) -> List[Tuple[Tuple[int, int], str]]:
    """Return ``[((group_idx, cond_idx), message), ...]``; empty when all good.

    Every enabled condition is evaluated in isolation so a caller can flag all
    the bad rows at once instead of only the first one.
    """
    problems: List[Tuple[Tuple[int, int], str]] = []
    if len(df.index) == 0:
        return problems
    for gi, g in enumerate(expr.groups):
        if not g.enabled:
            continue
        for ci, c in enumerate(g.conditions):
            if not c.enabled:
                continue
            try:
                condition_mask(df, c, ctx)
            except ConditionError as exc:
                problems.append(((gi, ci), exc.message))
    return problems


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

_CONDITION_DEFAULTS: Dict[str, Any] = {
    "field": "", "value": None, "min_val": None,
    "max_val": None, "unit": "", "state": "upper", "n_modes": 3, "regex": False,
    "negate": False, "enabled": True,
}


def _default_op(kind: str) -> str:
    """The operator a condition of *kind* takes when none is given."""
    valid = ops_for_kind(kind)
    return valid[0] if valid else ""


def _cond_to_dict(c: Condition) -> Dict[str, Any]:
    """Serialize one condition, omitting keys still at their default."""
    out: Dict[str, Any] = {"kind": c.kind}
    if c.op != _default_op(c.kind):
        out["op"] = c.op
    for key, default in _CONDITION_DEFAULTS.items():
        val = getattr(c, key)
        if key == "value" and isinstance(val, tuple):
            val = list(val)
        if val != default:
            out[key] = val
    return out


def _cond_from_dict(d: Mapping[str, Any], path: Tuple[int, int] = (-1, -1)) -> Condition:
    """Rebuild one condition; unknown keys are ignored, missing keys defaulted."""
    if not isinstance(d, Mapping):
        raise ConditionError(
            f"Expected a mapping for a condition, got {type(d).__name__}.", path)
    kind = str(d.get("kind", ""))
    if kind not in _MASK_BUILDERS:
        raise ConditionError(
            f"Unknown condition kind {kind!r}.  Expected one of: "
            f"{', '.join(sorted(_MASK_BUILDERS))}.", path)
    op = str(d.get("op") or _default_op(kind))
    if op not in ops_for_kind(kind):
        raise ConditionError(
            f"Operator {op!r} is not valid for a {kind!r} condition.  "
            f"Expected one of: {', '.join(ops_for_kind(kind))}.", path)
    value = d.get("value", None)
    if isinstance(value, list):
        value = tuple(value)
    return Condition(
        kind=kind,
        field=str(d.get("field", _CONDITION_DEFAULTS["field"])),
        op=op,
        value=value,
        min_val=_opt_float(d.get("min_val")),
        max_val=_opt_float(d.get("max_val")),
        unit=str(d.get("unit", _CONDITION_DEFAULTS["unit"])),
        state=str(d.get("state", _CONDITION_DEFAULTS["state"])),
        n_modes=int(d.get("n_modes", _CONDITION_DEFAULTS["n_modes"])),
        regex=bool(d.get("regex", _CONDITION_DEFAULTS["regex"])),
        negate=bool(d.get("negate", _CONDITION_DEFAULTS["negate"])),
        enabled=bool(d.get("enabled", _CONDITION_DEFAULTS["enabled"])),
    )


def _opt_float(val: Any) -> Optional[float]:
    """Coerce to float, passing ``None`` through."""
    return None if val is None else float(val)


def _group_to_dict(g: ConditionGroup) -> Dict[str, Any]:
    """Serialize one group, omitting keys still at their default."""
    out: Dict[str, Any] = {
        "join": _norm_join(g.join),
        "conditions": [_cond_to_dict(c) for c in g.conditions],
    }
    if g.label:
        out["label"] = g.label
    if not g.enabled:
        out["enabled"] = False
    return out


def _group_from_dict(d: Mapping[str, Any], gi: int = -1) -> ConditionGroup:
    """Rebuild one group from :func:`_group_to_dict` output."""
    if not isinstance(d, Mapping):
        raise ConditionError(
            f"Expected a mapping for a condition group, got {type(d).__name__}.",
            (gi, -1))
    conditions = tuple(
        _cond_from_dict(c, (gi, ci)) for ci, c in enumerate(d.get("conditions", ()))
    )
    return ConditionGroup(
        conditions=conditions,
        join=_norm_join(d.get("join", "AND")),
        label=str(d.get("label", "")),
        enabled=bool(d.get("enabled", True)),
    )


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

_SYM: Dict[str, str] = {
    "lam": "λ",        # lambda
    "e_up": "E_up",
    "e_low": "E_low",
    "a_stein": "A",
    "g_up": "g_up",
    "g_low": "g_low",
    "freq": "ν",       # nu
    "intens": "I",
    "tau": "τ",
}


def _fmt_num(val: Optional[float]) -> str:
    """Render a bound compactly (``6.0`` -> ``6``, ``0.005`` -> ``0.005``)."""
    if val is None:
        return "?"
    return f"{val:g}"


def _fmt_bound(val: Optional[float], c: Condition) -> str:
    """Render a bound so a relative one cannot be read as an absolute value."""
    if val is None or not c.unit:
        return _fmt_num(val)
    return f"p{val:g}" if c.unit == "percentile" else f"{val:g}%"


def describe_condition(c: Condition) -> str:
    """Render one condition the way it will be evaluated."""
    neg = c.negate
    if c.kind in ("range", "quantum_field", "velocity_sep", "derived"):
        if c.kind in ("range", "derived"):
            name = _SYM.get(c.field, c.field)
        elif c.kind == "velocity_sep":
            name = "Δv"
        else:
            name = f"{c.field}({'up' if c.state == 'upper' else 'low'})"
        # Written once at the end rather than after every bound, so a
        # `between` reading stays short enough to take in at a glance.
        tail = " of max" if c.unit == "percent" else ""
        lo, hi = (c.min_val, c.max_val) if c.op == "between" else (None, None)
        if c.op == "between":
            body = f"{_fmt_bound(lo, c)} ≤ {name} ≤ {_fmt_bound(hi, c)}{tail}"
            return f"not ({body})" if neg else body
        if c.op == "ge":
            rel = "<" if neg else "≥"
            return f"{name} {rel} {_fmt_bound(c.min_val, c)}{tail}"
        if c.op == "le":
            rel = ">" if neg else "≤"
            return f"{name} {rel} {_fmt_bound(c.max_val, c)}{tail}"
        if c.op == "eq":
            val = c.value if c.kind == "quantum_field" else c.min_val
            shown = (_fmt_bound(val, c) if isinstance(val, (int, float))
                     else str(val))
            rel = "≠" if neg else "="
            return f"{name} {rel} {shown}{tail}"
    if c.kind == "quantum_label":
        col = "lev_up" if c.state == "upper" else "lev_low"
        pat = "" if c.value is None else str(c.value)
        if c.op == "eq":
            return f"{col} ≠ '{pat}'" if neg else f"{col} = '{pat}'"
        verb = "matches" if c.op == "matches" else "contains"
        return (f"{col} does not {verb} '{pat}'" if neg
                else f"{col} {verb} '{pat}'")
    if c.kind == "vib_band":
        spec = str(c.value or "")
        return f"not band {spec}" if neg else f"band {spec}"
    if c.kind == "species":
        names = (", ".join(str(n) for n in c.value)
                 if isinstance(c.value, (list, tuple)) else str(c.value))
        return f"species not in [{names}]" if neg else f"species in [{names}]"
    return f"{c.kind}({c.field})"


def describe_group(g: ConditionGroup) -> str:
    """Render one group, parenthesised when it holds more than one condition.

    Returns an empty string for a vacuous group, matching the drop rule in
    :func:`group_mask` exactly - so the rendering can never show a term that
    the evaluation ignores.
    """
    active = [c for c in g.conditions if c.enabled]
    if not g.enabled or not active:
        return ""
    sep = "  AND  " if _norm_join(g.join) == "AND" else "  OR  "
    body = sep.join(describe_condition(c) for c in active)
    return f"({body})" if len(active) > 1 else body


def describe_expression(e: FilterExpression) -> str:
    """Render the whole expression with explicit parentheses.

    This is the single renderer shared by the filter window's live reading, by
    :meth:`LineListMaker.summary`, and by the log line written on apply and
    export - so what the user reads is provably what ran.
    """
    parts = [p for p in (describe_group(g) for g in e.groups) if p]
    if not parts:
        return "all lines (no conditions)"
    sep = "  AND  " if _norm_join(e.join) == "AND" else "  OR  "
    return sep.join(parts)
