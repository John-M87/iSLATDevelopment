"""
LineListFilterWindow - interactive line-list filter popup.

Opens a Toplevel window bound to a Molecule's MoleculeLineList.  Conditions are
organised into **groups**: the conditions inside a group combine with one
operator (All = AND, Any = OR), and the groups themselves combine with one
outer operator.  That makes it possible to ask for things a single fixed form
cannot express, e.g.

    (6 <= lam <= 7  OR  12 <= lam <= 13  OR  E_up >= 5000)  AND  (A >= 5e-3)

Two levels are enough for any boolean formula over these predicates (every
formula has a disjunctive normal form), and because each level has exactly one
operator there is no precedence rule to learn - the parentheses you see drawn
around a group are the parentheses that are evaluated.

Evaluation lives in
:mod:`iSLAT.Modules.DataProcessing.LineFilterExpression`; this module only maps
widgets onto that model.  Results can be exported to CSV or HITRAN ``.par``, or
applied back onto the molecule.
"""
from __future__ import annotations

import tkinter as tk
import dataclasses
from dataclasses import dataclass
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    ConditionError,
    ConditionGroup,
    FilterExpression,
    MaskContext,
    describe_expression,
    fold_masks,
    group_mask,
    validate,
)
from iSLAT.Modules.GUI.GUIFunctions import create_scrollable_frame
from iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow import (
    _population_data_length as _population_row_count,
)
from iSLAT.Modules.GUI.Tooltips import CreateToolTip

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule


# ---------------------------------------------------------------------------
# Operator display names
# ---------------------------------------------------------------------------
# Users pick a natural operator ("not between", "≠") rather than a separate NOT
# checkbox, so a negated condition can never be misread and no structurally
# invalid state is representable.

OP_DISPLAY: Dict[str, str] = {
    "between": "between",
    "not_between": "not between",
    "ge": "≥",
    "le": "≤",
    "eq": "=",
    "ne": "≠",
    "contains": "contains",
    "not_contains": "does not contain",
    "matches": "matches regex",
    "band": "is band",
    "not_band": "is not band",
    "in": "is one of",
    "not_in": "is not one of",
}

#: Display label -> (backend op, negate)
_DISPLAY_TO_OP: Dict[str, Tuple[str, bool]] = {
    "between": ("between", False),
    "not between": ("between", True),
    "≥": ("ge", False),
    "≤": ("le", False),
    "=": ("eq", False),
    "≠": ("eq", True),
    "contains": ("contains", False),
    "does not contain": ("contains", True),
    "matches regex": ("matches", False),
    "is band": ("band", False),
    "is not band": ("band", True),
    "is one of": ("in", False),
    "is not one of": ("in", True),
}

#: Which display operators each condition kind offers, in menu order.
_DISPLAY_OPS_BY_KIND: Dict[str, Tuple[str, ...]] = {
    "range":         ("between", "not between", "≥", "≤", "=", "≠"),
    "derived":       ("between", "not between", "≥", "≤", "=", "≠"),
    "velocity_sep":  ("between", "not between", "≥", "≤", "=", "≠"),
    "quantum_field": ("between", "not between", "≥", "≤", "=", "≠"),
    "quantum_label": ("contains", "does not contain", "=", "≠", "matches regex"),
    "vib_band":      ("is band", "is not band"),
    "species":       ("is one of", "is not one of"),
}

#: Display operators that need a second operand box.
_TWO_OPERAND_OPS = frozenset({"between", "not between"})


def display_ops_for_kind(kind: str) -> Tuple[str, ...]:
    """Display operator labels offered for *kind*, in menu order."""
    return _DISPLAY_OPS_BY_KIND.get(kind, ())


# ---------------------------------------------------------------------------
# Property catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropertySpec:
    """One entry in a condition row's property combobox.

    Attributes
    ----------
    key : str
        Stable identifier, e.g. ``"lam"``, ``"qn:upper:J"``, ``"vib_band"``.
    label : str
        Display text, e.g. ``"Wavelength (µm)"``, ``"J (upper)"``.
    kind, field, state
        The corresponding :class:`Condition` fields.
    editor : str
        Operand widget shape: ``"num"``, ``"text"``, ``"band"``, or ``"csv"``.
    tip : str
        Tooltip shown on the property combobox.
    """

    key: str
    label: str
    kind: str
    field: str = ""
    state: str = "upper"
    editor: str = "num"
    tip: str = ""


#: The six numeric properties the fixed form used to offer, in its exact order,
#: with its tooltip wording preserved.
_ONE_SIDED_NOTE = "\nLeave a bound blank to apply a one-sided limit."

_CORE_NUMERIC: Tuple[Tuple[str, str, str], ...] = (
    ("lam",     "Wavelength (µm)",
     "Filter lines by wavelength (µm)." + _ONE_SIDED_NOTE),
    ("e_up",    "E_up (K)",
     "Filter by upper-level energy (in Kelvin)." + _ONE_SIDED_NOTE),
    ("e_low",   "E_low (K)",
     "Filter by lower-level energy (in Kelvin)." + _ONE_SIDED_NOTE),
    ("a_stein", "Einstein A (s⁻¹)",
     "Filter by Einstein A coefficient (spontaneous emission rate, s⁻¹)."
     + _ONE_SIDED_NOTE),
    ("g_up",    "g_up",
     "Filter by upper-level statistical weight (degeneracy)." + _ONE_SIDED_NOTE),
    ("g_low",   "g_low",
     "Filter by lower-level statistical weight (degeneracy)." + _ONE_SIDED_NOTE),
    ("freq",    "Frequency (Hz)",
     "Filter by transition frequency (Hz)." + _ONE_SIDED_NOTE),
)

#: Tooltip for the derived nearest-neighbour velocity-separation property.
_VELOCITY_SEP_TIP = (
    "Filter by how far each line sits from its CLOSEST neighbour in the line "
    "list,\nmeasured as a Doppler velocity separation Δv = c·|Δλ|/λ (km/s).\n"
    "Use a lower bound to keep well-isolated lines (e.g. ≥ 5 km/s), or an "
    "upper\nbound to find blended pairs."
    + _ONE_SIDED_NOTE
)

#: Suffix explaining the percentile operand, appended to every tip that takes one.
_PERCENT_NOTE = (
    "\nAdd '%' to a bound to read it as a percentile of this line list "
    "instead\nof an absolute value - e.g. '90%' keeps the top tenth, '≤ 50%' "
    "the lower half."
)

#: Model quantities, taken from the same intensity calculation the population
#: diagram plots rather than from any column of the line list.
_DERIVED: Tuple[Tuple[str, str, str], ...] = (
    ("intens", "Line intensity",
     "Filter by the modelled line intensity, exactly as plotted by the "
     "population\ndiagram.  It follows the molecule's current temperature, "
     "column density,\nbroadening, radius and distance."
     + _ONE_SIDED_NOTE + _PERCENT_NOTE),
    ("tau", "Optical depth (τ)",
     "Filter by the line-center optical depth τ from the molecule's current "
     "model.\nUse an upper bound to keep optically thin lines (e.g. ≤ 1)."
     + _ONE_SIDED_NOTE + _PERCENT_NOTE),
)

#: Property kinds whose bounds accept a trailing '%' as a percentile rank.
_PERCENTILE_KINDS: frozenset = frozenset({"range", "derived"})

#: Model-quantity keys, in catalogue order.
_DERIVED_KEYS: Tuple[str, ...] = tuple(key for key, _, _ in _DERIVED)

#: The properties offered by the "Quick add" menu, in the legacy order.
QUICK_ADD_KEYS: Tuple[str, ...] = (
    "lam", "e_up", "e_low", "a_stein", "g_up", "g_low", "velocity_sep",
    "intens", "tau",
)

_NON_PROPERTY_COLUMNS = frozenset({
    "species", "lev_up", "lev_low", "nr",
} | {key for key, _, _ in _CORE_NUMERIC})


def build_property_catalog(df: pd.DataFrame,
                           species: Optional[str],
                           derived: Sequence[str] = ()) -> List[PropertySpec]:
    """Build the selectable property list for *df* and *species*.

    Combines the static core properties, any additional numeric columns the
    frame happens to carry (saved-lines fit results, ``xmin``/``xmax``, ...),
    the raw quantum-label matchers, and the parsed quantum-number fields
    declared by the molecule's schema.

    *derived* names the model quantities the caller can supply (``"intens"``,
    ``"tau"``).  They are not columns, so a caller with no molecule behind it
    passes none and they simply do not appear.

    The schema is looked up with the same key the evaluator uses, so every
    field offered here is a field that can actually be evaluated.  When only
    the generic fallback schema is available it declares no fields, so no
    quantum-field entries are emitted and the caller shows an explanatory note
    rather than an empty combobox that reads as a bug.
    """
    columns = set(df.columns) if df is not None else set()
    catalog: List[PropertySpec] = []

    for key, label, tip in _CORE_NUMERIC:
        if key in columns:
            catalog.append(PropertySpec(key=key, label=label, kind="range",
                                        field=key, editor="num",
                                        tip=tip + _PERCENT_NOTE))

    # Model quantities: computed by the intensity physics, not read off a column.
    for key, label, tip in _DERIVED:
        if key in derived:
            catalog.append(PropertySpec(key=key, label=label, kind="derived",
                                        field=key, editor="num", tip=tip))

    # Derived, not a column: the Doppler separation to the nearest neighbour.
    if "lam" in columns:
        catalog.append(PropertySpec(
            key="velocity_sep", label="Velocity sep. (km/s)",
            kind="velocity_sep", editor="num", tip=_VELOCITY_SEP_TIP))

    # Extra numeric columns (e.g. xmin/xmax, saved-line fit results)
    for col in sorted(c for c in columns if c not in _NON_PROPERTY_COLUMNS):
        if pd.api.types.is_numeric_dtype(df[col]):
            catalog.append(PropertySpec(
                key=col, label=col, kind="range", field=col, editor="num",
                tip=f"Filter by the {col!r} column of this line list."
                    + _ONE_SIDED_NOTE + _PERCENT_NOTE))

    # Raw quantum-label matching
    for state, col, label in (("upper", "lev_up", "Quantum lev_up"),
                              ("lower", "lev_low", "Quantum lev_low")):
        if col in columns:
            catalog.append(PropertySpec(
                key=f"label:{state}", label=label, kind="quantum_label",
                state=state, editor="text",
                tip=f"Filter by the {'upper' if state == 'upper' else 'lower'} "
                    f"quantum level label.\n"
                    f"Enter text to match against the level string.\n"
                    f"'contains' keeps lines whose label contains the value;\n"
                    f"'does not contain' keeps the rest."))

    # Parsed quantum-number fields, from the molecule's schema
    for spec in _quantum_field_specs(species):
        if spec.state == "upper" and "lev_up" not in columns:
            continue
        if spec.state == "lower" and "lev_low" not in columns:
            continue
        catalog.append(spec)

    if "lev_up" in columns and "lev_low" in columns:
        catalog.append(PropertySpec(
            key="vib_band", label="Vib. band", kind="vib_band", editor="band",
            tip="Filter lines belonging to a specific vibrational band.\n"
                "Examples: 'v1' (any band from mode 1), 'v1-0', 'v2-1'."))

    if "species" in columns:
        catalog.append(PropertySpec(
            key="species", label="Species", kind="species", editor="csv",
            tip="Keep only the listed species.\n"
                "Separate multiple names with commas."))

    return catalog


def _quantum_field_specs(species: Optional[str]) -> List[PropertySpec]:
    """Property specs for every parsed quantum field the schema declares."""
    specs: List[PropertySpec] = []
    try:
        from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
        import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401
        schema = QuantumStateRegistry.get_schema(species)
        fields = list(schema.global_fields) + list(schema.local_fields)
    except Exception:
        return specs
    for state in ("upper", "lower"):
        for f in fields:
            desc = getattr(f, "description", "") or ""
            specs.append(PropertySpec(
                key=f"qn:{state}:{f.name}",
                label=f"{f.name} ({state})",
                kind="quantum_field", field=f.name, state=state, editor="num",
                tip=(f"Quantum number {f.name!r} parsed from the "
                     f"{state} level label."
                     + (f"\n{desc}" if desc else "")
                     + "\nUse '=' for a non-numeric field.")))
    return specs


def has_quantum_schema(species: Optional[str]) -> bool:
    """True when *species* has a registered schema declaring named fields."""
    return bool(_quantum_field_specs(species))


# ---------------------------------------------------------------------------
# Widget-state -> Condition mapping
# ---------------------------------------------------------------------------

def parse_float_strict(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse a numeric entry strictly.

    Returns ``(value, error)``: a blank string gives ``(None, None)`` meaning
    "no bound", a valid number gives ``(value, None)``, and anything else gives
    ``(None, message)``.

    Unparsable text must not quietly become "no bound": under OR a typo that
    drops a limit can widen a narrow selection into half the line list with no
    visible sign.
    """
    text = (text or "").strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except ValueError:
        return None, f"{text!r} is not a number."


def split_percent(text: str) -> Tuple[str, bool]:
    """Strip a trailing ``%`` from an operand, reporting whether one was there."""
    text = (text or "").strip()
    if text.endswith("%"):
        return text[:-1].strip(), True
    return text, False


def _spec_by_label(catalog: Sequence[PropertySpec],
                   label: str) -> Optional[PropertySpec]:
    """Look a property spec up by its display label."""
    for spec in catalog:
        if spec.label == label:
            return spec
    return None


def row_state_to_condition(state: Mapping[str, Any],
                           catalog: Sequence[PropertySpec],
                           ) -> Tuple[Optional[Condition], List[str]]:
    """Map one condition row's widget values to a :class:`Condition`.

    Returns ``(condition, errors)``.  A blank row - no property chosen, or no
    operand typed - returns ``(None, [])``: blank rows are **excluded** from
    evaluation rather than treated as all-True, so a half-finished row can
    never silently widen the result.
    """
    errors: List[str] = []
    enabled = bool(state.get("enabled", True))
    label = str(state.get("prop", "") or "").strip()
    if not label:
        return None, errors
    if not enabled:
        # An ignored row is ignored completely: a stale typo in a row the user
        # has switched off must not block the whole expression.
        cond, _ignored = row_state_to_condition({**state, "enabled": True},
                                                catalog)
        return (None if cond is None
                else dataclasses.replace(cond, enabled=False)), []
    spec = _spec_by_label(catalog, label)
    if spec is None:
        return None, [f"Unknown property {label!r}."]

    display_op = str(state.get("op", "") or "").strip()
    if display_op not in _DISPLAY_TO_OP:
        return None, [f"Unknown operator {display_op!r} for {label!r}."]
    op, negate = _DISPLAY_TO_OP[display_op]

    v1 = str(state.get("v1", "") or "").strip()
    v2 = str(state.get("v2", "") or "").strip()
    two_operand = display_op in _TWO_OPERAND_OPS

    if not v1 and not (two_operand and v2):
        return None, errors  # nothing typed yet

    if spec.editor == "num" and not (spec.kind == "quantum_field" and op == "eq"):
        lo_txt, hi_txt = (v1, v2) if two_operand else (v1, "")
        lo_txt, lo_pct = split_percent(lo_txt)
        hi_txt, hi_pct = split_percent(hi_txt)
        lo, err_lo = parse_float_strict(lo_txt)
        hi, err_hi = parse_float_strict(hi_txt)
        for err in (err_lo, err_hi):
            if err:
                errors.append(f"{label}: {err}")
        if lo_pct != hi_pct and lo_txt and hi_txt:
            errors.append(f"{label}: use '%' on both bounds or on neither.")
        if (lo_pct or hi_pct) and spec.kind not in _PERCENTILE_KINDS:
            errors.append(f"{label}: a percentile is not meaningful here.")
        elif (lo_pct or hi_pct) and any(
                b is not None and not 0.0 <= b <= 100.0 for b in (lo, hi)):
            errors.append(f"{label}: a percentile must be between 0 and 100.")
        if errors:
            return None, errors
        if op == "le" and not two_operand:
            # A one-sided "≤" is typed into the first box but bounds the top.
            lo, hi = None, lo
        if lo is None and hi is None:
            return None, errors
        return Condition(kind=spec.kind, field=spec.field, op=op,
                         min_val=lo, max_val=hi,
                         unit="percentile" if (lo_pct or hi_pct) else "",
                         state=spec.state,
                         negate=negate, enabled=enabled), errors

    # Everything else is operand-as-text: quantum labels, an exact quantum
    # field value, a vibrational band spec, or a species list.
    value: Any = v1
    n_modes = 3
    if spec.editor == "band":
        try:
            n_modes = int(state.get("modes", 3))
        except (TypeError, ValueError):
            errors.append(f"{label}: mode count must be a whole number.")
            return None, errors
    regex = bool(state.get("regex", False)) if spec.kind == "quantum_label" else False
    if op == "matches":
        regex = True
    return Condition(kind=spec.kind, field=spec.field, op=op, value=value,
                     state=spec.state, n_modes=n_modes, regex=regex,
                     negate=negate, enabled=enabled), errors


def group_row_indices(group_state: Mapping[str, Any],
                      catalog: Sequence[PropertySpec]) -> List[int]:
    """GUI row indices that produced a condition, in condition order.

    Blank rows yield no condition, so a condition's position inside a group is
    NOT its row position.  Built with the same mapper :func:`build_expression`
    uses, so the two can never drift apart and flag the wrong row.
    """
    indices: List[int] = []
    for row_index, row_state in enumerate(group_state.get("rows", ())):
        cond, _errors = row_state_to_condition(row_state, catalog)
        if cond is not None:
            indices.append(row_index)
    return indices


def build_expression(outer_join: str,
                     group_states: Sequence[Mapping[str, Any]],
                     catalog: Sequence[PropertySpec],
                     ) -> Tuple[FilterExpression, List[str]]:
    """Assemble the full expression from GUI state.

    Returns ``(expression, errors)``.  All errors are collected rather than
    stopping at the first, so every bad row can be flagged at once.
    """
    errors: List[str] = []
    groups: List[ConditionGroup] = []
    for gs in group_states:
        conditions: List[Condition] = []
        for rs in gs.get("rows", ()):
            cond, errs = row_state_to_condition(rs, catalog)
            errors.extend(errs)
            if cond is not None:
                conditions.append(cond)
        groups.append(ConditionGroup(
            conditions=tuple(conditions),
            join=str(gs.get("join", "All")),
            enabled=bool(gs.get("enabled", True)),
        ))
    return FilterExpression(groups=tuple(groups), join=str(outer_join)), errors


def _rows_align(frame: pd.DataFrame, data: Mapping[str, Any]) -> bool:
    """True when *data*'s per-line arrays describe exactly *frame*'s rows.

    Compares wavelengths element for element rather than trusting the row
    count, so a filter mask can never be applied to arrays that merely happen
    to be the same length as the frame it was built from.
    """
    if data is None:
        return False
    lam = data.get("wavelength")
    if lam is None or "lam" not in getattr(frame, "columns", ()):
        return True   # nothing to compare against; the count check stands alone
    lam = np.asarray(lam, dtype=float)
    frame_lam = np.asarray(frame["lam"], dtype=float)
    if lam.shape != frame_lam.shape:
        return False
    return bool(np.allclose(lam, frame_lam, rtol=1e-9, atol=0.0,
                            equal_nan=True))


def diagnose_empty_group(g: ConditionGroup, df: pd.DataFrame,
                         ctx: MaskContext) -> Optional[str]:
    """Explain an empty group, but only when the diagnosis is a computed fact.

    Returns a message **only** when folding the group with AND selects nothing
    while folding the same conditions with OR would select something - i.e.
    switching to "Any" is demonstrably the fix.  Never a guess.
    """
    active = [c for c in g.conditions if c.enabled]
    if len(active) < 2 or str(g.join).upper() not in ("AND", "ALL"):
        return None
    try:
        or_mask = group_mask(df, ConditionGroup(tuple(active), join="OR"), ctx)
    except ConditionError:
        return None
    if or_mask is None or not bool(or_mask.any()):
        return None
    fields = {c.field for c in active if c.field}
    subject = f"on {next(iter(fields))}" if len(fields) == 1 else "here"
    return (f'"All" {subject} matches nothing - no line satisfies every '
            f'condition at once.  Use "Any"?')


# ---------------------------------------------------------------------------
# Condition row
# ---------------------------------------------------------------------------

class _ConditionRow(ttk.Frame):
    """One predicate: [x] [property] [operator] [value] [value] (⧉) (×)."""

    def __init__(self, parent, group: "_GroupRow", catalog: Sequence[PropertySpec],
                 on_change, on_delete, on_duplicate):
        super().__init__(parent)
        self._group = group
        self._catalog = list(catalog)
        self._on_change = on_change
        self._on_delete = on_delete
        self._on_duplicate = on_duplicate
        self._labels = [s.label for s in self._catalog]

        self.enabled_var = tk.BooleanVar(value=True)
        self.prop_var = tk.StringVar()
        self.op_var = tk.StringVar()
        self.v1_var = tk.StringVar()
        self.v2_var = tk.StringVar()
        self.modes_var = tk.IntVar(value=3)
        self.regex_var = tk.BooleanVar(value=False)
        self.err_var = tk.StringVar()

        self.columnconfigure(2, weight=1)

        self._chk = ttk.Checkbutton(self, variable=self.enabled_var,
                                    command=self._on_enabled_toggled)
        self._chk.grid(row=0, column=0, padx=(2, 0), pady=1)
        CreateToolTip(self._chk,
                      "Uncheck to temporarily ignore this condition without "
                      "deleting it.")

        self._warn = ttk.Label(self, text="⚠", style="Error.TLabel")  # gridded on error

        self._prop_combo = ttk.Combobox(self, textvariable=self.prop_var,
                                        values=self._labels, state="readonly",
                                        width=18)
        self._prop_combo.grid(row=0, column=2, padx=2, pady=1, sticky="ew")
        self._prop_combo.bind("<<ComboboxSelected>>", self._on_property_changed)
        self._prop_tip = CreateToolTip(self._prop_combo,
                                       "Choose which line property to test.")

        self._op_combo = ttk.Combobox(self, textvariable=self.op_var,
                                      state="readonly", width=9)
        self._op_combo.grid(row=0, column=3, padx=2, pady=1)
        self._op_combo.bind("<<ComboboxSelected>>", self._on_op_changed)
        CreateToolTip(self._op_combo,
                      "between / not between - two bounds, both inclusive.\n"
                      "≥ ≤ = ≠ - a single value.\n"
                      "contains - literal substring.  matches regex - a regular "
                      "expression.")

        self._v1 = ttk.Entry(self, textvariable=self.v1_var, width=10)
        self._v1.grid(row=0, column=4, padx=2, pady=1)
        self._v1_tip = CreateToolTip(self._v1, "Minimum value (inclusive).")

        self._v2 = ttk.Entry(self, textvariable=self.v2_var, width=10)
        self._v2.grid(row=0, column=5, padx=2, pady=1)
        CreateToolTip(self._v2,
                      "Maximum value (inclusive).\n"
                      "Add '%' to read it as a percentile of this line list.")

        self._modes = ttk.Spinbox(self, from_=2, to=6, width=3,
                                  textvariable=self.modes_var,
                                  command=self._changed)
        self._modes.grid(row=0, column=6, padx=2, pady=1)
        CreateToolTip(self._modes,
                      "Number of vibrational modes (3 for H₂O, CO₂, …).")

        self._regex = ttk.Checkbutton(self, text="regex", variable=self.regex_var,
                                      command=self._changed)
        self._regex.grid(row=0, column=7, padx=2, pady=1)
        CreateToolTip(self._regex,
                      "Treat the text as a regular expression.\n"
                      "Unchecked = literal substring (recommended: '0|' then "
                      "matches those characters rather than acting as an "
                      "alternation).")

        self._dup_btn = ttk.Button(self, text="⧉", width=2,
                                   command=lambda: self._on_duplicate(self))
        self._dup_btn.grid(row=0, column=8, padx=2, pady=1)
        CreateToolTip(self._dup_btn,
                      "Duplicate this condition below, ready for a second "
                      "range.\nIf this group is set to 'All', it switches to "
                      "'Any'.")

        self._del_btn = ttk.Button(self, text="×", width=2,
                                   command=lambda: self._on_delete(self))
        self._del_btn.grid(row=0, column=9, padx=(2, 4), pady=1)
        CreateToolTip(self._del_btn, "Delete this condition.")

        self._err_lbl = ttk.Label(self, textvariable=self.err_var,
                                  style="Error.TLabel", wraplength=460,
                                  justify="left")

        # Keep the trace handles: Tcl holds a callback referencing this row
        # until they are removed, so without this a deleted row (and the window
        # and line list it closes over) would live for the life of the app.
        self._traces = [(var, var.trace_add("write", self._changed))
                        for var in (self.v1_var, self.v2_var)]

        self._sync_editor()

    def destroy(self) -> None:
        """Remove the variable traces before tearing the widgets down."""
        for var, handle in getattr(self, "_traces", ()):
            try:
                var.trace_remove("write", handle)
            except tk.TclError:
                pass
        self._traces = []
        super().destroy()

    # -- state ---------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        """Return this row's widget values as a plain dict."""
        return {
            "prop": self.prop_var.get(),
            "op": self.op_var.get(),
            "v1": self.v1_var.get(),
            "v2": self.v2_var.get(),
            "modes": self.modes_var.get(),
            "regex": self.regex_var.get(),
            "enabled": self.enabled_var.get(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Populate this row from a :meth:`to_state` dict."""
        self.prop_var.set(str(state.get("prop", "")))
        self._sync_editor(keep_op=False)
        if state.get("op"):
            self.op_var.set(str(state["op"]))
        self.v1_var.set(str(state.get("v1", "")))
        self.v2_var.set(str(state.get("v2", "")))
        try:
            self.modes_var.set(int(state.get("modes", 3)))
        except (TypeError, ValueError):
            self.modes_var.set(3)
        self.regex_var.set(bool(state.get("regex", False)))
        self.enabled_var.set(bool(state.get("enabled", True)))
        self._sync_editor()

    def set_catalog(self, catalog: Sequence[PropertySpec]) -> None:
        """Swap the selectable properties, keeping the current pick if it survives.

        Called when the underlying line list is replaced and the available
        columns or quantum fields may have changed.
        """
        self._catalog = list(catalog)
        self._labels = [s.label for s in self._catalog]
        current = self.prop_var.get()
        self._prop_combo.configure(values=self._labels)
        if current and current not in self._labels:
            self.prop_var.set("")
        self._sync_editor()

    def spec(self) -> Optional[PropertySpec]:
        """The currently selected property spec, if any."""
        return _spec_by_label(self._catalog, self.prop_var.get())

    def focus_value(self) -> None:
        """Put keyboard focus in the first operand box."""
        try:
            self._v1.focus_set()
        except tk.TclError:
            pass

    # -- errors --------------------------------------------------------
    def set_error(self, message: str) -> None:
        """Flag this row with a warning glyph and an inline message."""
        self.err_var.set(message)
        self._warn.grid(row=0, column=1, padx=(0, 2))
        self._err_lbl.grid(row=1, column=1, columnspan=9, sticky="w", padx=2)

    def clear_error(self) -> None:
        """Remove any error flagging from this row."""
        if self.err_var.get():
            self.err_var.set("")
        self._warn.grid_remove()
        self._err_lbl.grid_remove()

    # -- internals -----------------------------------------------------
    def _changed(self, *_args) -> None:
        self._on_change()

    def _on_enabled_toggled(self) -> None:
        self._sync_editor()
        self._on_change()

    def _on_property_changed(self, *_args) -> None:
        self._sync_editor(keep_op=False)
        self._on_change()

    def _on_op_changed(self, *_args) -> None:
        self._sync_editor()
        self._on_change()

    def _sync_editor(self, keep_op: bool = True) -> None:
        """Show exactly the operand widgets the current property/op needs."""
        spec = self.spec()
        kind = spec.kind if spec else "range"
        ops = display_ops_for_kind(kind)
        self._op_combo.configure(values=ops)
        if not keep_op or self.op_var.get() not in ops:
            self.op_var.set(ops[0] if ops else "")
        if spec is not None:
            self._prop_tip.text = spec.tip or "Choose which line property to test."

        display_op = self.op_var.get()
        two = display_op in _TWO_OPERAND_OPS
        editor = spec.editor if spec else "num"

        if two:
            self._v2.grid()
        else:
            self._v2.grid_remove()
        self._v1_tip.text = ("Minimum value (inclusive)." if two else
                             "Maximum value (inclusive)." if display_op == "≤" else
                             "Value.")
        if kind in _PERCENTILE_KINDS:
            self._v1_tip.text += ("\nAdd '%' to read it as a percentile of "
                                  "this line list.")
        if editor == "band":
            self._modes.grid()
        else:
            self._modes.grid_remove()
        if editor == "text" and display_op in ("contains", "does not contain"):
            self._regex.grid()
        else:
            self._regex.grid_remove()

        # A disabled condition greys out so it reads as "ignored" rather than
        # "empty"; its values are kept so re-enabling restores them.
        entry_state = "normal" if self.enabled_var.get() else "disabled"
        for w in (self._v1, self._v2, self._modes):
            w.configure(state=entry_state)
        self._op_combo.configure(
            state="readonly" if self.enabled_var.get() else "disabled")


# ---------------------------------------------------------------------------
# Group row
# ---------------------------------------------------------------------------

class _GroupRow(ttk.LabelFrame):
    """A parenthesised set of conditions sharing one All/Any operator."""

    def __init__(self, parent, index: int, catalog: Sequence[PropertySpec],
                 on_change, on_delete):
        # labelwidget lets the header host live controls (the Match combobox
        # and the collapse/delete buttons) instead of a static caption.
        self._header = ttk.Frame(parent)
        super().__init__(parent, labelwidget=self._header, padding=(4, 2))
        self._catalog = list(catalog)
        self._on_change = on_change
        self._on_delete = on_delete
        self._rows: List[_ConditionRow] = []
        self._collapsed = False

        self.join_var = tk.StringVar(value="All")
        self.enabled_var = tk.BooleanVar(value=True)
        self.footer_var = tk.StringVar()

        self._index_lbl = ttk.Label(self._header, text=f"Group {index}")
        self._index_lbl.pack(side="left", padx=(2, 6))
        ttk.Label(self._header, text="Match").pack(side="left")
        self._join_combo = ttk.Combobox(self._header, textvariable=self.join_var,
                                        values=["All", "Any"], state="readonly",
                                        width=5)
        self._join_combo.pack(side="left", padx=4)
        self._join_combo.bind("<<ComboboxSelected>>", lambda e: self._on_change())
        CreateToolTip(self._join_combo,
                      "How the conditions INSIDE this group combine.\n\n"
                      "All = every condition must match (AND).\n"
                      "Any = at least one must match (OR) - use this for two "
                      "wavelength ranges.")

        self._collapse_btn = ttk.Button(self._header, text="∧", width=2,
                                        command=self._toggle_collapse)
        self._collapse_btn.pack(side="left", padx=(6, 0))
        CreateToolTip(self._collapse_btn,
                      "Collapse or expand this group. The matching count stays "
                      "visible.")

        self._del_btn = ttk.Button(self._header, text="×", width=2,
                                   command=lambda: self._on_delete(self))
        self._del_btn.pack(side="left", padx=2)
        CreateToolTip(self._del_btn, "Delete this group and all its conditions.")

        self._body = ttk.Frame(self)
        self._body.pack(fill="x")

        self._rows_host = ttk.Frame(self._body)
        self._rows_host.pack(fill="x")

        self._add_btn = ttk.Button(self._body, text="+ Add Condition",
                                   command=self.add_condition)
        self._add_btn.pack(anchor="e", padx=4, pady=(2, 0))
        CreateToolTip(self._add_btn,
                      "Add another condition to this group.\n"
                      "It starts on the same property and operator as the row "
                      "above, so a second range is quick to enter.")

        self._footer = ttk.Label(self, textvariable=self.footer_var,
                                 style="Muted.TLabel", anchor="w",
                                 wraplength=520, justify="left")
        self._footer.pack(fill="x", padx=4)

    # -- structure -----------------------------------------------------
    @property
    def rows(self) -> List[_ConditionRow]:
        """The condition rows in this group, top to bottom."""
        return list(self._rows)

    def set_index(self, index: int) -> None:
        """Renumber this group's header caption."""
        self._index_lbl.configure(text=f"Group {index}")

    def add_condition(self, at: Optional[int] = None,
                      state: Optional[Mapping[str, Any]] = None,
                      ) -> _ConditionRow:
        """Append (or insert) a condition row.

        With no *state*, the new row inherits the property and operator of the
        row above it: adding a condition is overwhelmingly "another one of
        these", and re-picking the same property every time is the slow part.
        """
        row = _ConditionRow(self._rows_host, self, self._catalog,
                            self._on_change, self._delete_row, self._duplicate_row)
        if state is None and self._rows:
            prev = self._rows[-1 if at is None else max(at - 1, 0)]
            row.load_state({"prop": prev.prop_var.get(), "op": prev.op_var.get(),
                            "modes": prev.modes_var.get(),
                            "regex": prev.regex_var.get()})
        elif state is not None:
            row.load_state(state)

        if at is None or at >= len(self._rows):
            self._rows.append(row)
        else:
            self._rows.insert(at, row)
        self._repack_rows()
        self._on_change()
        return row

    def _repack_rows(self) -> None:
        # Drop any row whose widget is already gone, so one stale entry cannot
        # take the whole repack down with a "bad window path name".
        self._rows = [r for r in self._rows if r.winfo_exists()]
        for row in self._rows:
            row.pack_forget()
        for row in self._rows:
            row.pack(fill="x", pady=1)

    def _delete_row(self, row: _ConditionRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.destroy()
        # An empty group is kept visible rather than auto-removed: deleting the
        # group out from under the user mid-edit is surprising.
        self._on_change()

    def _duplicate_row(self, row: _ConditionRow) -> None:
        """Clone *row* directly below it, with the operands cleared."""
        state = dict(row.to_state())
        same_prop = state.get("prop", "")
        state["v1"] = ""
        state["v2"] = ""
        idx = self._rows.index(row) + 1 if row in self._rows else None
        new_row = self.add_condition(at=idx, state=state)
        # Two ranges on one property under "All" can never both match, so the
        # duplicate would silently produce zero lines.  Switch, and say so.
        if same_prop and self.join_var.get() == "All":
            self.join_var.set("Any")
            self.footer_var.set(
                f'Switched to "Any" - two conditions on {same_prop} under '
                f'"All" would match nothing.')
        new_row.focus_value()
        self._on_change()

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._body.pack_forget()
            self._collapse_btn.configure(text="∨")
        else:
            self._body.pack(fill="x", before=self._footer)
            self._collapse_btn.configure(text="∧")

    # -- state ---------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        """Return this group's state, including every condition row."""
        return {
            "join": self.join_var.get(),
            "enabled": self.enabled_var.get(),
            "rows": [r.to_state() for r in self._rows],
        }

    # -- feedback ------------------------------------------------------
    def set_count(self, count: Optional[int], prefix: str = "") -> None:
        """Update the footer with this group's own matching-line count."""
        n = len(self._rows)
        noun = "condition" if n == 1 else "conditions"
        if count is None:
            self.footer_var.set(f"{n} {noun} · (ignored - nothing to match on)")
        else:
            self.footer_var.set(f"{n} {noun} · {prefix}{count} lines")

    def set_hint(self, hint: Optional[str]) -> None:
        """Append a diagnostic hint to the footer, when there is one."""
        if hint:
            self.footer_var.set(f"{self.footer_var.get()} — {hint}")

    def set_error_at(self, cond_index: int, message: str) -> None:
        """Flag one condition row as unevaluable."""
        if 0 <= cond_index < len(self._rows):
            self._rows[cond_index].set_error(message)

    def clear_errors(self) -> None:
        """Clear error flagging from every row in this group."""
        for row in self._rows:
            row.clear_error()


# ---------------------------------------------------------------------------
# LineListFilterWindow
# ---------------------------------------------------------------------------

class LineListFilterWindow(tk.Toplevel):
    """Interactive popup for filtering a molecule's line list.

    Parameters
    ----------
    parent : tk widget
        Parent window (typically the ControlPanel or root).
    mol_obj : Molecule
        The molecule whose ``line_list`` will be filtered.
    data_field : optional
        iSLAT data field for status messages (``insert_text`` interface).
    islat : optional
        iSLAT instance.  Required for "Apply to Molecule" and
        "Duplicate & Apply" GUI refresh.
    theme : dict, optional
        Active GUI theme; used only to pick readable error/muted colours.
    """

    #: Live windows, keyed by (iSLAT instance, molecule name).
    _instances: Dict[Tuple[int, str], "LineListFilterWindow"] = {}

    @classmethod
    def open(cls, parent, mol_obj: "Molecule", data_field=None, islat=None,
             theme=None) -> "LineListFilterWindow":
        """Open the filter window for *mol_obj*, or raise the existing one."""
        key = (id(islat), str(getattr(mol_obj, "name", "")))
        win = cls._instances.get(key)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win.focus_force()
                    return win
            except tk.TclError:
                pass
            cls._instances.pop(key, None)
        win = cls(parent, mol_obj, data_field, islat=islat, theme=theme)
        cls._instances[key] = win
        win._instance_key = key
        return win

    def __init__(self, parent, mol_obj: "Molecule", data_field=None, islat=None,
                 theme=None):
        super().__init__(parent)
        self.mol_obj = mol_obj
        self.data_field = data_field
        self._islat = islat
        self._theme = theme
        self._instance_key: Optional[Tuple[int, str]] = None
        self._preview_job: Optional[str] = None

        mol_name = getattr(mol_obj, "name", "Molecule")
        self.title(f"Filter Line List: {mol_name}")
        self.resizable(True, True)

        # Build the maker from the molecule's line list
        line_list = mol_obj.line_list
        if line_list is None:
            self.destroy()
            raise ValueError(f"Molecule '{mol_name}' has no line list loaded.")
        self._maker = LineListMaker(line_list)
        self._baseline_id = self._baseline_fingerprint()
        self._total_lines = len(self._maker)
        self._derived_cache: Optional[pd.DataFrame] = None
        self._derived_id: Optional[Tuple] = None
        self._catalog = build_property_catalog(self._maker.original_df,
                                               self._maker.species,
                                               derived=_DERIVED_KEYS)

        self._group_rows: List[_GroupRow] = []
        self._separators: List[ttk.Label] = []
        # Companion views of the filtered result, driven by the preview pass.
        self._table_view = None
        self._pop_window = None
        self._last_mask = None
        self._table_job = None

        self._build_ui()
        self._constrain_to_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._add_group()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _constrain_to_screen(self):
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(620, int(screen_w * 0.50))
        win_h = min(760, int(screen_h * 0.85))
        pos_x = max((screen_w - win_w) // 2, 0)
        pos_y = max((screen_h - win_h) // 2, 0)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(560, 520)

    def _configure_styles(self) -> None:
        """Two named styles for error and muted text.

        Popout Toplevels are not walked by the theme manager, so they inherit
        the native ttk theme; only these two colours need picking by hand.
        """
        style = ttk.Style(self)
        is_dark = "dark" in str(self._theme).lower()
        style.configure("Error.TLabel", foreground="#ff6b6b" if is_dark else "red")
        style.configure("Muted.TLabel", foreground="#888888")

    def _build_ui(self):
        self._configure_styles()

        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=0)  # header
        outer.rowconfigure(1, weight=1)  # groups area (the only weighted row)
        outer.rowconfigure(2, weight=0)  # reading
        outer.rowconfigure(3, weight=0)  # status
        outer.rowconfigure(4, weight=0)  # buttons

        # ── Header ───────────────────────────────────────────────────
        head_wrap = ttk.Frame(outer)
        head_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        hdr = ttk.Frame(head_wrap)
        hdr.pack(fill="x")

        ttk.Label(hdr, text="Match").pack(side="left")
        self._outer_join_var = tk.StringVar(value="All")
        outer_combo = ttk.Combobox(hdr, textvariable=self._outer_join_var,
                                   values=["All", "Any"], state="readonly",
                                   width=5)
        outer_combo.pack(side="left", padx=4)
        outer_combo.bind("<<ComboboxSelected>>", self._on_outer_join_changed)
        CreateToolTip(outer_combo,
                      "How the groups below combine.\n\n"
                      "All = every group must match (AND).\n"
                      "Any = at least one group must match (OR).")
        ttk.Label(hdr, text="of the following groups").pack(side="left")

        add_grp_btn = ttk.Button(hdr, text="+ Group", command=self._add_group)
        add_grp_btn.pack(side="right", padx=(4, 0))
        CreateToolTip(add_grp_btn,
                      "Add a new group of conditions.\n"
                      "Use a second group when you want "
                      "'(this OR that) AND something else'.")

        quick_btn = ttk.Menubutton(hdr, text="Quick add ▾")
        quick_menu = tk.Menu(quick_btn, tearoff=0)
        for key in QUICK_ADD_KEYS:
            spec = next((s for s in self._catalog if s.key == key), None)
            if spec is None:
                continue
            quick_menu.add_command(
                label=spec.label,
                command=lambda lb=spec.label: self._quick_add(lb))
        quick_btn.configure(menu=quick_menu)
        quick_btn.pack(side="right", padx=4)
        CreateToolTip(quick_btn,
                      "Insert a ready-made condition for a common property.")

        # An absent schema means no parsed quantum-number properties are on
        # offer.  Say so, rather than leaving a gap that reads as a bug.
        if not has_quantum_schema(self._maker.species):
            ttk.Label(
                head_wrap, style="Muted.TLabel", anchor="w", justify="left",
                wraplength=560,
                text=(f"No quantum-number schema is registered for "
                      f"'{self._maker.species}', so only column properties and "
                      f"raw level-label matching are available."),
            ).pack(fill="x", pady=(2, 0))

        # ── Groups area (scrollable) ─────────────────────────────────
        scroll_host = ttk.Frame(outer)
        scroll_host.grid(row=1, column=0, sticky="nsew")
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)
        self._groups_host = create_scrollable_frame(
            scroll_host, height=300, width=560, vertical=True, row=0, col=0)

        # ── Reading ──────────────────────────────────────────────────
        reading_lf = ttk.LabelFrame(outer, text="Reading", padding=6)
        reading_lf.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._reading_var = tk.StringVar(value="all lines (no conditions)")
        self._reading_lbl = ttk.Label(reading_lf, textvariable=self._reading_var,
                                      wraplength=540, justify="left")
        self._reading_lbl.pack(fill="x")
        CreateToolTip(self._reading_lbl,
                      "The expression exactly as it will be evaluated, with "
                      "explicit parentheses.")
        reading_lf.bind(
            "<Configure>",
            lambda e: self._reading_lbl.configure(wraplength=max(e.width - 24, 200)))

        # ── Status ───────────────────────────────────────────────────
        stat = ttk.Frame(outer)
        stat.grid(row=3, column=0, sticky="ew", pady=(4, 4))
        self._status_var = tk.StringVar()
        status_lbl = ttk.Label(stat, textvariable=self._status_var,
                               relief="sunken", anchor="w")
        status_lbl.pack(side="left", fill="x", expand=True)
        CreateToolTip(status_lbl,
                      "Lines matching the expression, out of the full line "
                      "list.\nPress 'Apply Filters' to commit.")
        self._live_var = tk.BooleanVar(value=True)
        live_chk = ttk.Checkbutton(stat, text="Live preview",
                                   variable=self._live_var,
                                   command=self._schedule_preview)
        live_chk.pack(side="right", padx=(6, 0))
        CreateToolTip(live_chk,
                      "Recount lines as you type. Turn off for very large line "
                      "lists;\nthe reading above still updates.")

        # ── Action buttons ───────────────────────────────────────────
        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=4, column=0, sticky="ew")
        for col in range(4):
            btn_frame.columnconfigure(col, weight=1)

        self._apply_btn = ttk.Button(btn_frame, text="Apply Filters",
                                     command=self._apply_filters)
        self._apply_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        CreateToolTip(self._apply_btn,
                      "Evaluate the expression above and narrow the line list "
                      "to the matching lines.")
        reset_btn = ttk.Button(btn_frame, text="Reset", command=self._reset_all)
        reset_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        CreateToolTip(reset_btn,
                      "Clear all conditions and reset the line list to its "
                      "original state.")
        csv_btn = ttk.Button(btn_frame, text="Export CSV", command=self._export_csv)
        csv_btn.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        CreateToolTip(csv_btn,
                      "Export the currently filtered line list to a CSV file.")
        par_btn = ttk.Button(btn_frame, text="Export PAR", command=self._export_par)
        par_btn.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        CreateToolTip(par_btn,
                      "Export the currently filtered line list in HITRAN .par "
                      "format.")

        ttk.Separator(btn_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(2, 0))

        # ── Companion views of the filtered result ───────────────────
        views_frame = ttk.Frame(btn_frame)
        views_frame.grid(row=2, column=0, columnspan=4, sticky="ew")
        views_frame.columnconfigure(0, weight=1)
        views_frame.columnconfigure(1, weight=1)

        table_btn = ttk.Button(views_frame, text="View Filtered Lines",
                               command=self._open_table_view)
        table_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        CreateToolTip(table_btn,
                      "Open a sortable, searchable table of the lines the "
                      "filter currently selects.\n"
                      "It refreshes as the filter changes.")
        pop_btn = ttk.Button(views_frame, text="Population Diagram",
                             command=self._open_pop_diagram)
        pop_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        CreateToolTip(pop_btn,
                      "Open a population diagram of the lines the filter "
                      "currently selects.\n"
                      "It redraws as the filter changes, and uses the same "
                      "axis, colour and\nshape settings as the main "
                      "population diagram.")

        ttk.Separator(btn_frame, orient="horizontal").grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(2, 0))

        apply_mol_frame = ttk.Frame(btn_frame)
        apply_mol_frame.grid(row=4, column=0, columnspan=4, sticky="ew")
        apply_mol_frame.columnconfigure(0, weight=1)
        apply_mol_frame.columnconfigure(1, weight=1)

        apply_mol_btn = ttk.Button(apply_mol_frame, text="Apply to Molecule",
                                   command=self._apply_to_molecule)
        apply_mol_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        CreateToolTip(apply_mol_btn,
                      "Replace this molecule's line list with the filtered "
                      "result in-place.\n"
                      "The original line list will be lost for this session.")
        dup_btn = ttk.Button(apply_mol_frame, text="Duplicate & Apply",
                             command=self._duplicate_and_apply)
        dup_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        CreateToolTip(dup_btn,
                      "Create a copy of this molecule with the filtered line "
                      "list applied,\nleaving the original molecule unchanged.")

        # ── Keyboard ─────────────────────────────────────────────────
        self.bind("<Control-Return>", lambda e: self._apply_filters())
        self.bind("<Control-g>", lambda e: self._add_group())
        self.bind("<Escape>", lambda e: self._on_close())

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------
    def _on_outer_join_changed(self, *_args) -> None:
        """Redraw the operator labels between groups, then recount."""
        self._repack_groups()
        self._schedule_preview()

    def _add_group(self, *_args) -> "_GroupRow":
        """Append a new group with one blank condition, and focus it."""
        group = _GroupRow(self._groups_host, len(self._group_rows) + 1,
                          self._catalog, self._schedule_preview,
                          self._delete_group)
        self._group_rows.append(group)
        self._repack_groups()
        group.add_condition()
        return group

    def _delete_group(self, group: "_GroupRow") -> None:
        if group in self._group_rows:
            self._group_rows.remove(group)
        group.destroy()
        self._repack_groups()
        self._schedule_preview()

    def _repack_groups(self) -> None:
        """Re-lay the groups with the outer operator drawn between them."""
        for sep in self._separators:
            sep.destroy()
        self._separators = []
        for group in self._group_rows:
            group.pack_forget()
        join_text = "AND" if self._outer_join_var.get() == "All" else "OR"
        for i, group in enumerate(self._group_rows):
            if i:
                sep = ttk.Label(self._groups_host, text=join_text,
                                anchor="center", style="Muted.TLabel")
                sep.pack(fill="x")
                self._separators.append(sep)
            group.set_index(i + 1)
            group.pack(fill="x", pady=(0, 2), padx=2)

    def _quick_add(self, label: str) -> None:
        """Insert a ready-made condition for *label* into the last group."""
        if not self._group_rows:
            self._add_group()
        group = self._group_rows[-1]
        row = group.add_condition(state={"prop": label})
        row.focus_value()

    def _group_states(self) -> List[Dict[str, Any]]:
        return [g.to_state() for g in self._group_rows]

    # ------------------------------------------------------------------
    # Model quantities (line intensity, optical depth)
    # ------------------------------------------------------------------
    def _model_fingerprint(self) -> Tuple:
        """What the model quantities depend on, beyond the line list itself.

        The molecule's parameters can be edited in the control panel while
        this window is open, and τ and the intensity follow them, so a filter
        on either must be re-evaluated rather than served from a stale cache.
        """
        mol = self.mol_obj
        return (self._baseline_id,) + tuple(
            getattr(mol, attr, None)
            for attr in ("temp", "n_mol", "broad", "radius", "distance")
        )

    def _derived_frame(self) -> pd.DataFrame:
        """Model quantities for the filter's rows, recomputed only when stale.

        Indexed like ``LineListMaker.original_df`` so any row subset - the
        subsampled preview included - selects the matching values.
        """
        fingerprint = self._model_fingerprint()
        if self._derived_cache is not None and self._derived_id == fingerprint:
            return self._derived_cache
        data, error = self._full_population_data()
        if data is None:
            raise ConditionError(error or
                                 "No model data is available for this molecule.")
        base = self._maker.original_df
        columns = {key: np.asarray(data[key], dtype=float)
                   for key in _DERIVED_KEYS
                   if data.get(key) is not None}
        frame = pd.DataFrame(columns, index=base.index)
        self._derived_cache = frame
        self._derived_id = fingerprint
        return frame

    def _mask_context(self) -> MaskContext:
        """The evaluation context every pass of this window must share."""
        return MaskContext(species=self._maker.species,
                           derived_source=self._derived_frame)

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------
    def _schedule_preview(self, *_args) -> None:
        """Debounce a preview recomputation."""
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except (tk.TclError, ValueError):
                pass
        self._preview_job = self.after(250, self._recompute_preview)

    def _recompute_preview(self) -> None:
        """Recount matching lines and refresh the reading, without committing.

        The preview never touches ``self._maker``: it evaluates the unfiltered
        snapshot directly, so the number on screen after Apply always matches
        what an export will write.  The baseline is re-read every pass rather
        than cached, so applying to the molecule cannot leave it stale.
        """
        self._preview_job = None
        self._sync_baseline()
        states = self._group_states()
        expr, errors = build_expression(self._outer_join_var.get(),
                                        states, self._catalog)
        self._reading_var.set(describe_expression(expr))  # always, even on error

        # Stale flags first: a row the user has just fixed must stop looking
        # broken, and a row that is broken must actually be marked.
        for group in self._group_rows:
            group.clear_errors()
        if errors:
            self._set_blocked(highlighted=self._flag_parse_errors(states))
            return
        if not self._live_var.get():
            self._status_var.set(
                f"Lines: {len(self._maker)} / {self._total_lines}  "
                f"(live preview off)")
            self._apply_btn.state(["!disabled"])
            return

        base = self._maker.original_df
        ctx = self._mask_context()
        base, prefix = self._preview_frame(base)

        masks: List[pd.Series] = []
        ok = True
        any_flagged = False
        for gi, (grow, g) in enumerate(zip(self._group_rows, expr.groups)):
            try:
                m = group_mask(base, g, ctx)
            except ConditionError as exc:
                flagged = self._flag_group_error(grow, gi, g, base, ctx, exc)
                ok = False
                any_flagged = any_flagged or flagged
                continue
            grow.set_count(None if m is None else int(m.sum()), prefix)
            if m is not None:
                masks.append(m)
                if not int(m.sum()):
                    grow.set_hint(diagnose_empty_group(g, base, ctx))
        if not ok:
            self._set_blocked(highlighted=any_flagged)
            return

        # Fold with the SAME function the committed filter uses, so the count
        # on screen and the rows Apply keeps can never disagree.
        folded = fold_masks(masks, expr.join, base.index)
        total = int(folded.sum())
        self._status_var.set(
            f"Matching: {prefix}{total} / {self._total_lines} lines")
        self._apply_btn.state(["!disabled"])
        # Only a full-resolution mask can drive the companion views; a
        # subsampled preview would plot the wrong subset.
        self._last_mask = None if prefix else folded.to_numpy()
        self._notify_companions()

    def _flag_parse_errors(self, states: Sequence[Mapping[str, Any]]) -> bool:
        """Mark every row whose typed value cannot be parsed.

        Returns True when at least one row was flagged.
        """
        flagged = False
        for grow, gs in zip(self._group_rows, states):
            for row_index, row_state in enumerate(gs.get("rows", ())):
                _cond, row_errors = row_state_to_condition(row_state,
                                                           self._catalog)
                if row_errors:
                    grow.set_error_at(row_index, "  ".join(row_errors))
                    flagged = True
        return flagged

    def _preview_frame(self, base: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Subsample very large frames so the preview stays responsive."""
        if len(base) > 200_000:
            step = max(len(base) // 20_000, 1)
            return base.iloc[::step], "≈"
        return base, ""

    def _flag_group_error(self, grow: "_GroupRow", group_index: int,
                          g: ConditionGroup, base: pd.DataFrame,
                          ctx: MaskContext, exc: ConditionError) -> bool:
        """Mark the condition rows in *grow* that cannot be evaluated.

        Returns True when at least one row was flagged, so the caller never
        tells the user to fix a highlighted row that does not exist.
        """
        grow.clear_errors()
        row_map = group_row_indices(self._group_states()[group_index],
                                    self._catalog)
        problems = validate(base, FilterExpression(groups=(g,)), ctx)
        if not problems:
            problems = [((0, max(exc.path[1], 0)), exc.message)]
        flagged = False
        for (_gi, ci), message in problems:
            # ci indexes CONDITIONS; blank rows produce none, so translate.
            row_index = row_map[ci] if 0 <= ci < len(row_map) else None
            if row_index is not None:
                grow.set_error_at(row_index, message)
                flagged = True
        if not flagged:
            grow.set_hint(exc.message)
        return flagged

    def _set_blocked(self, highlighted: bool = True) -> None:
        """Show the blocked state and disable Apply."""
        hint = ("fix the highlighted rows" if highlighted
                else "the expression cannot be evaluated")
        self._status_var.set(f"— / {self._total_lines} lines   ({hint})")
        self._apply_btn.state(["disabled"])

    # ------------------------------------------------------------------
    # Companion views
    # ------------------------------------------------------------------
    def _open_table_view(self) -> None:
        """Open (or raise) a table of the lines the filter currently selects."""
        if self._table_view is not None:
            try:
                if self._table_view.winfo_exists():
                    self._table_view.lift()
                    self._table_view.focus_force()
                    self._table_view.refresh()
                    return
            except tk.TclError:
                pass
            self._table_view = None

        from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow

        mol_name = getattr(self.mol_obj, "name", "molecule")
        # A callable, not a frame: the view re-invokes it on every refresh, so
        # it tracks the filter instead of showing a snapshot.
        view = LineListViewWindow.from_dataframe(
            self, self._current_view_frame,
            title=f"Filtered Line List: {mol_name}",
            data_field=self.data_field,
            export_basename=f"{mol_name}_filtered",
        )
        view.protocol("WM_DELETE_WINDOW",
                      lambda v=view: self._close_table_view(v))
        self._table_view = view
        view.refresh()

    def _close_table_view(self, view) -> None:
        if self._table_view is view:
            self._table_view = None
        try:
            view.destroy()
        except tk.TclError:
            pass

    def _current_view_frame(self):
        """The rows the filter currently selects, as a DataFrame.

        Uses the live preview mask when one is available so the table matches
        the count on screen, and falls back to the committed result otherwise.
        """
        mask = self._last_mask
        base = self._maker.original_df
        if mask is not None and len(mask) == len(base):
            return base.loc[np.asarray(mask, dtype=bool)]
        return self._maker.df

    def _open_pop_diagram(self) -> None:
        """Open (or raise) a live population diagram of the filtered lines."""
        if self._pop_window is not None:
            try:
                if self._pop_window.winfo_exists():
                    self._pop_window.lift()
                    self._pop_window.focus_force()
                    return
            except tk.TclError:
                pass
            self._pop_window = None

        self._sync_baseline()
        from iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow import (
            FilteredPopulationDiagramWindow,
        )

        win = FilteredPopulationDiagramWindow(
            self, self.mol_obj, islat=self._islat,
            on_close=self._on_pop_diagram_closed,
        )
        self._pop_window = win

        data, error = self._full_population_data()
        win.set_full_data(data)
        if error:
            win.set_message(error)
        self._notify_companions()

    def _on_pop_diagram_closed(self, win) -> None:
        if self._pop_window is win:
            self._pop_window = None

    def _full_population_data(self):
        """Population-diagram arrays for the rows this filter operates on.

        Computed once and then only indexed, so changing the filter never
        re-runs the intensity physics.  Returns ``(data, error_message)``.

        ``full_range=False`` is deliberate.  The filter evaluates against
        ``LineListMaker.original_df``, which is the line list's *active*
        wavelength range; ``full_range=True`` would return every line in the
        file instead.  With a wavelength range set those two differ (e.g. 494
        rows against 28517), and a mask built for one cannot index the other.
        With no wavelength range the two are identical, so this costs nothing.
        """
        mol = self.mol_obj
        line_list = getattr(mol, "line_list", None)
        if line_list is None:
            return None, "This molecule has no line list."
        try:
            intensity = getattr(mol, "intensity", None)
            if intensity is None and hasattr(mol, "calculate_intensity"):
                mol.calculate_intensity()
                intensity = getattr(mol, "intensity", None)
            if intensity is None:
                from iSLAT.Modules.DataTypes.Intensity import Intensity
                intensity = Intensity(line_list)
                intensity.calc_intensity(
                    t_kin=getattr(mol, "temp", None),
                    n_mol=getattr(mol, "n_mol", None),
                    dv=getattr(mol, "broad", None),
                )
            data = intensity.get_population_diagram_data(
                getattr(mol, "radius", 1.0),
                getattr(mol, "distance", 160.0),
                molecule=mol,
                full_range=False,
            )
        except Exception as exc:
            return None, f"Could not compute population-diagram data: {exc}"
        if data is None:
            return None, "No population-diagram data for this molecule."

        # Refuse to serve arrays a filter mask cannot index: plotting the
        # wrong lines is far worse than plotting none.  Matching row COUNTS is
        # not enough - two different wavelength windows can hold the same
        # number of lines - so check the wavelengths line up element for
        # element before trusting the mask against these arrays.
        base = self._maker.original_df
        expected = len(base)
        actual = _population_row_count(data)
        if actual is not None and actual != expected:
            return None, (
                f"Population-diagram data has {actual} rows but the filter "
                f"works on {expected}; cannot align the two.")
        if not _rows_align(base, data):
            return None, ("Population-diagram data does not line up with the "
                          "filtered line list; cannot plot it safely.")
        return data, None

    def _notify_companions(self) -> None:
        """Push the current selection to whichever companion views are open."""
        # The table rebuilds every row through individual Treeview inserts, so
        # refreshing it inline on each preview pass would stall the filter
        # window while the user is still typing.  Coalesce instead.
        if self._table_view is not None:
            if self._table_job is not None:
                try:
                    self.after_cancel(self._table_job)
                except (tk.TclError, ValueError):
                    pass
            self._table_job = self.after(200, self._refresh_table_view)

        if self._pop_window is not None:
            try:
                if self._pop_window.winfo_exists():
                    mask = self._last_mask
                    base_len = self._pop_window.base_length
                    if mask is not None and base_len is not None \
                            and len(mask) != base_len:
                        mask = None   # stale; show everything rather than lie
                    self._pop_window.update_mask(mask)
                else:
                    self._pop_window = None
            except tk.TclError:
                self._pop_window = None

    def _refresh_table_view(self) -> None:
        """Re-fetch the table companion, dropping it if it has gone away."""
        self._table_job = None
        view = self._table_view
        if view is None:
            return
        try:
            if view.winfo_exists():
                view.refresh()
            else:
                self._table_view = None
        except tk.TclError:
            self._table_view = None

    def _close_companions(self) -> None:
        """Close any companion views this window owns."""
        if self._table_job is not None:
            try:
                self.after_cancel(self._table_job)
            except (tk.TclError, ValueError):
                pass
            self._table_job = None
        view, self._table_view = self._table_view, None
        if view is not None:
            try:
                view.destroy()
            except tk.TclError:
                pass
        win, self._pop_window = self._pop_window, None
        if win is not None:
            try:
                win.close()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _apply_filters(self):
        """Evaluate the expression and narrow the line list to the matches."""
        expr, errors = build_expression(self._outer_join_var.get(),
                                        self._group_states(), self._catalog)
        if errors:
            messagebox.showerror("Invalid Filter", "\n".join(errors), parent=self)
            self._refresh_status()
            return

        ctx = self._mask_context()
        problems = validate(self._maker.original_df, expr, ctx)
        if problems:
            states = self._group_states()
            for grow in self._group_rows:
                grow.clear_errors()
            for (gi, ci), message in problems:
                if 0 <= gi < len(self._group_rows):
                    row_map = group_row_indices(states[gi], self._catalog)
                    if 0 <= ci < len(row_map):
                        self._group_rows[gi].set_error_at(row_map[ci], message)
            messagebox.showerror(
                "Invalid Filter",
                "\n".join(message for _path, message in problems),
                parent=self)
            self._refresh_status()
            return

        self._maker.reset()
        try:
            self._maker.filter_expression(expr,
                                          derived_source=self._derived_frame)
        except ConditionError as exc:
            self._maker.reset()
            messagebox.showerror("Invalid Filter", str(exc), parent=self)
            self._refresh_status()
            return

        self._refresh_status()
        if self.data_field is not None:
            self.data_field.insert_text(
                f"Filter applied: {describe_expression(expr)} "
                f"→ {len(self._maker)} lines")

    def _baseline_fingerprint(self):
        """Cheap identity of the line list this window is currently filtering.

        Includes the wavelength range because changing it re-bases the rows in
        place: the same MoleculeLineList object can hold a completely different
        set of lines, sometimes even the same number of them.
        """
        mol = self.mol_obj
        line_list = getattr(mol, "line_list", None)
        if line_list is None:
            return None
        return (
            id(line_list),
            getattr(line_list, "num_lines", None),
            tuple(getattr(line_list, "wavelength_range", None) or ()),
            tuple(getattr(mol, "wavelength_range", None) or ()),
        )

    def _sync_baseline(self) -> bool:
        """Re-base on the molecule's current line list when it has changed.

        The wavelength range can be edited in the control panel while this
        window is open, which silently replaces the rows underneath it.
        Re-basing keeps the conditions the user typed and re-evaluates them
        against the lines the molecule actually has now.
        """
        fingerprint = self._baseline_fingerprint()
        if fingerprint is None or fingerprint == self._baseline_id:
            return False
        line_list = getattr(self.mol_obj, "line_list", None)
        if line_list is None:
            return False
        self._rebind_to_line_list(line_list)
        return True

    def _rebind_to_line_list(self, line_list) -> None:
        """Re-seat this window on *line_list* after it replaced the molecule's.

        The maker must be rebuilt, not merely reset: its unfiltered snapshot is
        the frame every mask indexes and every companion view renders, so
        leaving the pre-apply snapshot in place would show and export lines the
        molecule no longer contains.
        """
        self._maker = LineListMaker(line_list)
        self._baseline_id = self._baseline_fingerprint()
        self._total_lines = len(self._maker)
        self._derived_cache = None
        self._derived_id = None
        self._catalog = build_property_catalog(self._maker.original_df,
                                               self._maker.species,
                                               derived=_DERIVED_KEYS)
        self._last_mask = None
        for group in self._group_rows:
            for row in group.rows:
                row.set_catalog(self._catalog)
        if self._pop_window is not None:
            data, error = self._full_population_data()
            try:
                self._pop_window.set_full_data(data)
                if error:
                    self._pop_window.set_message(error)
            except tk.TclError:
                self._pop_window = None

    def _reset_all(self):
        """Clear every condition and restore the full line list."""
        self._maker.reset()
        for group in list(self._group_rows):
            group.destroy()
        self._group_rows = []
        for sep in self._separators:
            sep.destroy()
        self._separators = []
        self._outer_join_var.set("All")
        self._live_var.set(True)
        self._add_group()
        self._refresh_status()

    def _refresh_status(self):
        """Recompute the preview and the status line from scratch."""
        self._schedule_preview()
        if not self._group_rows:
            self._status_var.set(f"Lines: {len(self._maker)} / {self._total_lines}")

    # ------------------------------------------------------------------
    # Line-list assignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_line_list(mol_obj, new_ll) -> None:
        """Replace *mol_obj*'s line list and invalidate all cached results.

        Setting ``mol_obj.lines`` directly bypasses the molecule's cache
        invalidation logic, so we must mark every dirty flag and clear the
        flux / intensity / spectrum caches explicitly.
        """
        mol_obj.lines = new_ll
        # Nullify already-computed results so they are recalculated
        mol_obj.intensity = None
        mol_obj.spectrum = None
        # Mark all dirty flags
        try:
            mol_obj._dirty_flags['intensity'] = True
            mol_obj._dirty_flags['spectrum'] = True
            mol_obj._dirty_flags['flux'] = True
        except (AttributeError, KeyError):
            pass
        # Clear the flux cache (dict)
        try:
            mol_obj._flux_cache.clear()
        except AttributeError:
            pass
        # Clear intensity / spectrum caches
        try:
            mol_obj._intensity_cache['data'] = None
            mol_obj._intensity_cache['hash'] = None
        except (AttributeError, KeyError):
            pass
        try:
            mol_obj._spectrum_cache['data'] = None
            mol_obj._spectrum_cache['hash'] = None
        except (AttributeError, KeyError):
            pass

    def _apply_to_molecule(self):
        """Replace the molecule's line list with the current filtered result."""
        if len(self._maker) == 0:
            messagebox.showwarning(
                "Empty Result",
                "The filtered line list is empty - not applied.",
                parent=self,
            )
            return
        mol_name = getattr(self.mol_obj, "name", "this molecule")
        confirm = messagebox.askyesno(
            "Apply to Molecule",
            f"Replace '{mol_name}' line list with {len(self._maker)} filtered lines?\n"
            "This overwrites the in-memory line list and cannot be undone.",
            parent=self,
        )
        if not confirm:
            return
        new_ll = self._maker.to_linelist()
        self._assign_line_list(self.mol_obj, new_ll)
        self._rebind_to_line_list(new_ll)
        self._trigger_refresh(full_rebuild=False)
        self._refresh_status()
        if self.data_field is not None:
            self.data_field.insert_text(
                f"Applied filtered line list to '{mol_name}': {len(new_ll)} lines."
            )

    def _duplicate_and_apply(self):
        """Duplicate the molecule and apply the filtered line list to the copy."""
        if len(self._maker) == 0:
            messagebox.showwarning(
                "Empty Result",
                "The filtered line list is empty - not applied.",
                parent=self,
            )
            return
        if self._islat is None:
            messagebox.showerror(
                "No iSLAT Reference",
                "Cannot duplicate: no iSLAT instance was provided to this window.",
                parent=self,
            )
            return
        mol_name = getattr(self.mol_obj, "name", None)
        if not mol_name:
            messagebox.showerror("Error", "Molecule has no name.", parent=self)
            return
        new_name = self._islat.molecules_dict.duplicate(mol_name)
        if new_name is None:
            messagebox.showerror(
                "Duplicate Failed",
                f"Could not duplicate molecule '{mol_name}'.",
                parent=self,
            )
            return
        new_mol = self._islat.molecules_dict.get(new_name)
        if new_mol is not None:
            self._assign_line_list(new_mol, self._maker.to_linelist())
        self._trigger_refresh(full_rebuild=True)
        msg = (
            f"Created '{new_name}' with {len(self._maker)} filtered lines."
        )
        if self.data_field is not None:
            self.data_field.insert_text(msg)
        else:
            messagebox.showinfo("Done", msg, parent=self)

    def _trigger_refresh(self, full_rebuild: bool = False):
        """Attempt to refresh the iSLAT GUI after a line-list change."""
        try:
            if self._islat is None:
                return
            gui = getattr(self._islat, "GUI", None)
            if gui is None:
                return
            plot = getattr(gui, "plot", None)
            if plot is not None:
                plot.update_model_plot()
            cp = getattr(gui, "control_panel", None)
            if cp is not None:
                if full_rebuild:
                    cp.refresh_from_molecules_dict()
                else:
                    cp._update_molecule_parameter_fields()
        except Exception as e:
            print(f"LineListFilterWindow: GUI refresh error - {e}")

    def _export_csv(self):
        mol_name = getattr(self.mol_obj, "name", "molecule")
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Filtered Line List",
            defaultextension=".csv",
            initialfile=f"{mol_name}_filtered.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            out = self._maker.to_csv(Path(path))
            msg = f"Exported {len(self._maker)} lines → {out.name}"
            if self.data_field is not None:
                self.data_field.insert_text(msg)
            else:
                messagebox.showinfo("Export Complete", msg, parent=self)
        except Exception as e:
            messagebox.showerror("Export Error", str(e), parent=self)

    def _export_par(self):
        mol_name = getattr(self.mol_obj, "name", "molecule")
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Filtered Line List as PAR",
            defaultextension=".par",
            initialfile=f"{mol_name}_filtered.par",
            filetypes=[("PAR files", "*.par"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            out = self._maker.to_par(Path(path))
            msg = f"Exported {len(self._maker)} lines → {out.name}"
            if self.data_field is not None:
                self.data_field.insert_text(msg)
            else:
                messagebox.showinfo("Export Complete", msg, parent=self)
        except RuntimeError as e:
            messagebox.showerror(
                "Export Error",
                f"PAR export requires a HITRAN-sourced line list:\n{e}",
                parent=self,
            )
        except Exception as e:
            messagebox.showerror("Export Error", str(e), parent=self)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        """Cancel any pending preview, drop the registry entry, and close."""
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except (tk.TclError, ValueError):
                pass
            self._preview_job = None
        self._close_companions()
        if self._instance_key is not None:
            type(self)._instances.pop(self._instance_key, None)
        self.destroy()
