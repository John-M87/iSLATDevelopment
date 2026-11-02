"""
Unified registry of line properties.

This is the **single list of line properties** shared by the
population-diagram X / Y axes, color-by and marker-shape settings and by the
line-list filter (:mod:`~iSLAT.Modules.GUI.Widgets.LineListFilterWindow`).
Every property is described by a :class:`PlotAxisEntry` and registered with
the global :class:`PlotAxisRegistry` singleton; the entry says how the plot
reads the property (``resolve_array`` / ``resolve_scalar``) and how the
filter evaluates it (``filter_kind`` / ``filter_field`` / ``filter_state``).

Registration happens in three places:

1. **Standard physics properties** — registered at the bottom of
   :mod:`~iSLAT.Modules.DataTypes.Intensity` after the ``Intensity`` class
   definition (e.g. ``eu``, ``rd_yax``, ``a_stein``, …).

2. **Quantum-field properties** — registered automatically inside
   :meth:`~iSLAT.Modules.DataTypes.QuantumStateSchema.QuantumStateRegistry.register`
   whenever a molecule schema is added, producing ``qn_upper:FIELD`` /
   ``qn_lower:FIELD`` keys.

3. **External / plugin entries** — any code can call
   :meth:`PlotAxisRegistry.register` at any time.

Consumers (plot classes, GUI dialogs, the line-list filter) call:

* :meth:`PlotAxisRegistry.get_property_options` — the shared, ordered list.
* :meth:`PlotAxisRegistry.get_axis_options` — properties valid as a plot axis.
* :meth:`PlotAxisRegistry.get_color_options` — properties valid for color-by.
* :meth:`PlotAxisRegistry.resolve_array` — extract a numpy array from a
  component data dict (as produced by ``_compute_component``).
* :meth:`PlotAxisRegistry.resolve_scalar` — extract a float from a single
  ``MoleculeLine`` + ``mol_id``.
* :meth:`PlotAxisRegistry.get_axis_label` — LaTeX / formatted label string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

__all__ = ["PlotAxisEntry", "PlotAxisRegistry"]


# ═══════════════════════════════════════════════════════════════════
#  PlotAxisEntry
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlotAxisEntry:
    """Descriptor for a single plot / info-panel property.

    Parameters
    ----------
    key:
        Internal identifier used in ``set_axes``, ``color_by``, etc.
        (e.g. ``"eu"``, ``"tau"``, ``"qn_upper:J"``).
    display_name:
        Plain-text label for GUI dropdowns / log output.
    label:
        LaTeX / formatted label for plot axes and colorbars.
    kind:
        ``"continuous"`` for numeric (colormap-compatible) properties;
        ``"categorical"`` for string or discrete integer labels.  A
        categorical property on a plot axis is drawn as a category axis.
    available_as_axis:
        Whether this property can be used as a plot axis.  Default ``True``.
    available_as_color:
        Whether this property can be used for color-by.  Default ``True``.
    suggest_log:
        Whether a log color / axis scale is recommended (e.g. for intensity,
        opacity, Einstein-A).  Default ``False``.
    resolve_array:
        ``(cdata: dict) -> Optional[np.ndarray]`` — extracts a per-line
        array from a component data dict.  ``None`` means no bulk resolution
        is possible (e.g. ``component`` identity).
    resolve_scalar:
        ``(line: MoleculeLine, mol_id: Optional[str]) -> Optional[float]``
        — extracts a scalar from a single line.  ``None`` means the property
        cannot be resolved from a bare ``MoleculeLine`` (e.g. ``rd_yax``
        requires ``beam_s``).
    group:
        Optional grouping label for GUI sections (e.g. ``"Physics"``,
        ``"Quantum — upper"``, ``"Quantum — lower"``).
    quantum_field:
        For quantum-number entries, the schema field name this entry reads
        (e.g. ``"J"``).  ``None`` for species-independent properties.  Used
        to scope option lists to the molecules actually on a plot.
    filter_kind:
        How the line-list filter evaluates this property: the
        :class:`~iSLAT.Modules.DataProcessing.LineFilterExpression.Condition`
        kind (``"range"`` on the line-list column *filter_field*,
        ``"derived"`` for a model quantity of the population-diagram data,
        ``"velocity_sep"``, ``"quantum_label"``, ``"quantum_field"``,
        ``"vib_band"`` or ``"species"``).  ``None`` keeps the property out of
        the filter.
    filter_field:
        The condition field: a column name, model-quantity key or quantum
        field name.
    filter_state:
        ``"upper"`` or ``"lower"`` for level-label and quantum-field
        properties.
    """

    key: str
    display_name: str
    label: str
    kind: Literal["continuous", "categorical"]
    available_as_axis: bool = True
    available_as_color: bool = True
    suggest_log: bool = False
    resolve_array: Optional[Callable[[Dict[str, Any]], Optional[np.ndarray]]] = field(
        default=None, repr=False
    )
    resolve_scalar: Optional[Callable[["MoleculeLine", Optional[str]], Optional[float]]] = field(
        default=None, repr=False
    )
    group: str = ""
    quantum_field: Optional[str] = None
    filter_kind: Optional[str] = None
    filter_field: str = ""
    filter_state: str = "upper"


# ═══════════════════════════════════════════════════════════════════
#  PlotAxisRegistry
# ═══════════════════════════════════════════════════════════════════

#: Legacy / alternative keys accepted wherever a property key is.
_KEY_ALIASES: Dict[str, str] = {
    "lam": "wavelength",
    "e_up": "eu",
    "component": "molecule",
}

class PlotAxisRegistry:
    """Global registry of line properties (plot axes, color-by, marker shape, filter).

    All methods are class-methods — no instantiation is required.

    The registry is ordered by insertion time (Python 3.7+ dict ordering),
    which is the order every option list presents.  Later calls to
    :meth:`register` for the same key overwrite the existing entry
    (idempotent for identical registrations).  Every lookup also accepts the
    legacy keys ``"lam"``, ``"e_up"`` and ``"component"``.
    """

    _entries: ClassVar[Dict[str, PlotAxisEntry]] = {}

    # ── Registration ─────────────────────────────────────────────────

    @classmethod
    def register(cls, entry: PlotAxisEntry) -> None:
        """Register *entry* in the global registry.

        If *entry.key* already exists the old entry is replaced.
        """
        cls._entries[entry.key] = entry

    # ── Lookup ───────────────────────────────────────────────────────

    @classmethod
    def canonical_key(cls, key: str) -> str:
        """Return the registered key for *key*, resolving legacy aliases."""
        if key in cls._entries:
            return key
        return _KEY_ALIASES.get(key, key)

    @classmethod
    def get(cls, key: str) -> Optional[PlotAxisEntry]:
        """Return the entry for *key* (or its alias), or ``None`` if not registered."""
        return cls._entries.get(cls.canonical_key(key))

    @classmethod
    def get_axis_label(cls, key: str) -> str:
        """Return the LaTeX axis/colorbar label for *key*.

        Falls back to *key* itself when not registered.
        """
        entry = cls.get(key)
        return entry.label if entry is not None else key

    @classmethod
    def get_display_name(cls, key: str) -> str:
        """Return the plain-text display name for *key*.

        Falls back to *key* itself when not registered.
        """
        entry = cls.get(key)
        return entry.display_name if entry is not None else key

    @classmethod
    def get_kind(cls, key: str) -> str:
        """Return ``"continuous"`` or ``"categorical"`` for *key*.

        Falls back to ``"continuous"`` for unknown keys.
        """
        entry = cls.get(key)
        return entry.kind if entry is not None else "continuous"

    @classmethod
    def suggests_log(cls, key: str) -> bool:
        """Return whether a log scale is suggested for *key*."""
        entry = cls.get(key)
        return entry.suggest_log if entry is not None else False

    # ── Resolution ───────────────────────────────────────────────────

    @classmethod
    def resolve_array(
        cls, key: str, cdata: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract a per-line numpy array from a component data dict.

        Calls ``entry.resolve_array(cdata)`` when registered; falls back to
        ``np.asarray(cdata[key], dtype=float)`` for unknown keys.
        Categorical properties return an object array of labels.

        Returns ``None`` when the property is absent or cannot be resolved.
        """
        key = cls.canonical_key(key)
        entry = cls._entries.get(key)
        if entry is not None and entry.resolve_array is not None:
            return entry.resolve_array(cdata)
        # Generic fallback: direct key lookup
        arr = cdata.get(key)
        if arr is None:
            return None
        try:
            return np.asarray(arr, dtype=float)
        except (ValueError, TypeError):
            return None

    @classmethod
    def resolve_scalar(
        cls,
        key: str,
        line: "MoleculeLine",
        mol_id: Optional[str] = None,
    ) -> Optional[float]:
        """Extract a float value from a single ``MoleculeLine``.

        Returns ``None`` when no scalar resolver is registered for *key* or
        when the resolver itself returns ``None``.
        """
        entry = cls.get(key)
        if entry is not None and entry.resolve_scalar is not None:
            return entry.resolve_scalar(line, mol_id)
        return None

    # ── Option lists ─────────────────────────────────────────────────

    @classmethod
    def filter_for_species(
        cls,
        entries: Iterable[PlotAxisEntry],
        species: Optional[Iterable[Optional[str]]],
    ) -> List[PlotAxisEntry]:
        """Keep only the *entries* that apply to the given *species*.

        Species-independent entries always pass.  A quantum-field entry
        passes only when at least one of the species' registered schemas
        declares that field — the same lookup the line-list filter window
        uses (:meth:`QuantumStateRegistry.get_fields`).  ``species=None``
        disables filtering and returns every entry.
        """
        entries = list(entries)
        if species is None:
            return entries
        from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
        allowed = QuantumStateRegistry.get_field_names(species)
        return [e for e in entries
                if e.quantum_field is None or e.quantum_field in allowed]

    @staticmethod
    def _schema_field_order(species: Iterable[Optional[str]]) -> List[str]:
        """Quantum field names declared by *species*' schemas, in schema order."""
        from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
        order: List[str] = []
        for mid in species:
            for f in QuantumStateRegistry.get_fields(mid):
                if f.name not in order:
                    order.append(f.name)
        return order

    @classmethod
    def get_property_options(
        cls, species: Optional[Iterable[Optional[str]]] = None,
    ) -> List[PlotAxisEntry]:
        """Return the shared, ordered property list.

        The population-diagram X / Y axis, color-by and marker-shape settings
        all offer exactly this list, and the line-list filter offers the
        entries of it that it can evaluate, so every menu shows the same
        properties under the same names in the same order.

        Species-independent properties come first, in registration order,
        followed by the quantum numbers: every upper-state field, then every
        lower-state field.  When *species* is given only the fields those
        molecules' schemas declare are included, in schema order (see
        :meth:`filter_for_species`).
        """
        species = None if species is None else list(species)
        entries = cls.filter_for_species(cls._entries.values(), species)
        plain = [e for e in entries if e.quantum_field is None]
        quantum = [e for e in entries if e.quantum_field is not None]
        if species is not None:
            rank = {name: i for i, name in enumerate(cls._schema_field_order(species))}
            quantum.sort(key=lambda e: rank.get(e.quantum_field, len(rank)))
        # Stable: keeps the field order within each state.
        quantum.sort(key=lambda e: 0 if e.filter_state == "upper" else 1)
        return plain + quantum

    @classmethod
    def get_axis_options(
        cls, species: Optional[Iterable[Optional[str]]] = None,
    ) -> List[PlotAxisEntry]:
        """Return the :meth:`get_property_options` entries usable as a plot axis."""
        return [e for e in cls.get_property_options(species) if e.available_as_axis]

    @classmethod
    def get_color_options(
        cls, species: Optional[Iterable[Optional[str]]] = None,
    ) -> List[PlotAxisEntry]:
        """Return the :meth:`get_property_options` entries usable for color-by."""
        return [e for e in cls.get_property_options(species) if e.available_as_color]

    @classmethod
    def keys(cls) -> List[str]:
        """Return all registered keys in insertion order."""
        return list(cls._entries.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered entries.  Primarily for testing."""
        cls._entries.clear()
