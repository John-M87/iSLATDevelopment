"""
Unified registry for population-diagram / line-inspection plot properties.

Every property that can be used as a **plot axis**, a **color-by value**, or
that should appear in the **line-info panel** is described by a
:class:`PlotAxisEntry` and registered with the global
:class:`PlotAxisRegistry` singleton.

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

Consumers (plot classes, GUI dialogs) call:

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
        ``"categorical"`` for string or discrete integer labels.
    available_as_axis:
        Whether this property can be used as a plot axis.  Default ``True``.
        Categorical entries (``lev_up``, ``lev_low``, ``component``) are
        typically ``False``.
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


# ═══════════════════════════════════════════════════════════════════
#  PlotAxisRegistry
# ═══════════════════════════════════════════════════════════════════

class PlotAxisRegistry:
    """Global registry of plot-axis / color-by / info-panel properties.

    All methods are class-methods — no instantiation is required.

    The registry is ordered by insertion time (Python 3.7+ dict ordering).
    Later calls to :meth:`register` for the same key overwrite the existing
    entry (idempotent for identical registrations).
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
    def get(cls, key: str) -> Optional[PlotAxisEntry]:
        """Return the entry for *key*, or ``None`` if not registered."""
        return cls._entries.get(key)

    @classmethod
    def get_axis_label(cls, key: str) -> str:
        """Return the LaTeX axis/colorbar label for *key*.

        Falls back to *key* itself when not registered.
        """
        entry = cls._entries.get(key)
        return entry.label if entry is not None else key

    @classmethod
    def get_display_name(cls, key: str) -> str:
        """Return the plain-text display name for *key*.

        Falls back to *key* itself when not registered.
        """
        entry = cls._entries.get(key)
        return entry.display_name if entry is not None else key

    @classmethod
    def get_kind(cls, key: str) -> str:
        """Return ``"continuous"`` or ``"categorical"`` for *key*.

        Falls back to ``"continuous"`` for unknown keys.
        """
        entry = cls._entries.get(key)
        return entry.kind if entry is not None else "continuous"

    @classmethod
    def suggests_log(cls, key: str) -> bool:
        """Return whether a log scale is suggested for *key*."""
        entry = cls._entries.get(key)
        return entry.suggest_log if entry is not None else False

    # ── Resolution ───────────────────────────────────────────────────

    @classmethod
    def resolve_array(
        cls, key: str, cdata: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract a per-line numpy array from a component data dict.

        Calls ``entry.resolve_array(cdata)`` when registered; falls back to
        ``np.asarray(cdata[key], dtype=float)`` for unknown keys.

        Returns ``None`` when the property is absent or cannot be resolved.
        """
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
        entry = cls._entries.get(key)
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

    @classmethod
    def get_axis_options(
        cls, species: Optional[Iterable[Optional[str]]] = None,
    ) -> List[PlotAxisEntry]:
        """Return entries with ``available_as_axis=True``, in order.

        When *species* is given, quantum-field entries are limited to the
        fields those molecules' schemas declare (see :meth:`filter_for_species`).
        """
        return cls.filter_for_species(
            (e for e in cls._entries.values() if e.available_as_axis), species)

    @classmethod
    def get_color_options(
        cls, species: Optional[Iterable[Optional[str]]] = None,
    ) -> List[PlotAxisEntry]:
        """Return entries with ``available_as_color=True``, in order.

        When *species* is given, quantum-field entries are limited to the
        fields those molecules' schemas declare (see :meth:`filter_for_species`).
        """
        return cls.filter_for_species(
            (e for e in cls._entries.values() if e.available_as_color), species)

    @classmethod
    def keys(cls) -> List[str]:
        """Return all registered keys in insertion order."""
        return list(cls._entries.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered entries.  Primarily for testing."""
        cls._entries.clear()
