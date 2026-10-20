"""
Quantum state schema infrastructure for iSLAT.

Provides the abstract base class, quantum field definitions, a registry, and a
generic fallback schema for any line catalog that has not registered a specific molecule schema.

The iSLAT ``.par`` label format encodes an upper- or lower-state quantum description as a single compact string::

    "<global_quanta>|<local_quanta>"

where each part is derived from the HITRAN ``global_upper/lower_quanta`` and
``local_upper/lower_quanta`` columns by replacing internal whitespace runs with underscores::

    "_".join(field.strip().split())

Concrete schemas (e.g. in :mod:`~iSLAT.Modules.DataTypes.HITRANQuantumSchemas`) know how many fields to expect in each part and how to map tokens onto named quantum numbers. 
Non-HITRAN catalogs can register their own schemas at any time via :meth:`QuantumStateRegistry.register`.

Public API
----------
- :class:`QuantumField`
- :class:`QuantumStateSchema`
- :class:`GenericDelimitedSchema`
- :class:`QuantumStateRegistry`
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Iterable, List, Literal, Optional, Set, Tuple

import numpy as np
from typing import NamedTuple

__all__ = [
    "QuantumField",
    "QuantumStateSchema",
    "GenericDelimitedSchema",
    "QuantumStateRegistry",
]

# ═══════════════════════════════════════════════════════════════════
#  QuantumField
# ═══════════════════════════════════════════════════════════════════

class QuantumField(NamedTuple):
    """Metadata descriptor for a single quantum number field.

    Parameters
    ----------
    name:
        Short identifier used as the dict/array key in parsed output
        (e.g. ``'v1'``, ``'J'``, ``'Ka'``).
    dtype:
        Expected data type: ``'int'``, ``'float'``, or ``'str'``.
    description:
        Optional human-readable description (shown in documentation /
        population-diagram axis labels).
    """

    name: str
    dtype: Literal['int', 'float', 'str']
    description: str = ""

# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

#: Sentinel returned for missing/unparseable integer fields.
_INT_SENTINEL: int = -999

#: Sentinel returned for missing/unparseable float fields.
_FLOAT_SENTINEL: float = float('nan')

#: Sentinel returned for missing/unparseable string fields.
_STR_SENTINEL: str = ''

def _coerce(value: str, dtype: Literal['int', 'float', 'str']) -> Any:
    """Convert a string token to *dtype*, returning a sentinel on failure.

    Parameters
    ----------
    value:
        Raw token string (may be empty or ``None``).
    dtype:
        One of ``'int'``, ``'float'``, or ``'str'``.

    Returns
    -------
    int | float | str
        Coerced value, or the dtype-specific sentinel on failure.
    """
    if not value:
        if dtype == 'int':
            return _INT_SENTINEL
        if dtype == 'float':
            return _FLOAT_SENTINEL
        return _STR_SENTINEL

    try:
        if dtype == 'int':
            return int(value)
        if dtype == 'float':
            return float(value)
        return str(value)
    except (ValueError, TypeError):
        if dtype == 'int':
            return _INT_SENTINEL
        if dtype == 'float':
            return _FLOAT_SENTINEL
        return str(value)

# ═══════════════════════════════════════════════════════════════════
#  QuantumStateSchema  (abstract base)
# ═══════════════════════════════════════════════════════════════════
class QuantumStateSchema(ABC):
    """Abstract base for quantum state parsing schemas.

    Subclasses must declare :attr:`global_fields` and :attr:`local_fields`
    as class-level tuples of :class:`QuantumField` instances and implement
    :meth:`parse_label`.

    The iSLAT label format is::

        "<global_str>|<local_str>"

    where ``global_str`` encodes the *vibrational* (and occasionally
    electronic) state, and ``local_str`` encodes the *rotational*, fine, and
    hyperfine structure.  Both parts use ``_`` as a field separator.

    Notes
    -----
    HITRAN adjacent fixed-width fields with no whitespace separator can
    produce merged tokens (e.g. ``"03"`` for *v3 = 0, r = 3* in CO₂, or
    ``"-2-2-2"`` for all-missing H₂O vibrational assignments).  Concrete
    subclasses handle these via fixup hooks or custom ``parse_label``
    implementations.
    """

    #: Vibrational / electronic quantum fields (before ``|``).
    global_fields: Tuple[QuantumField, ...] = ()

    #: Rotational / fine-structure quantum fields (after ``|``).
    local_fields: Tuple[QuantumField, ...] = ()

    # ── Abstract interface ───────────────────────────────────────────

    @abstractmethod
    def parse_label(self, label: str) -> Dict[str, Any]:
        """Parse a single state label string into a dict of quantum numbers.

        Parameters
        ----------
        label:
            A label string such as ``"0_2_2|11_2_9"`` (H₂O upper state:
            vibrational v1=0, v2=2, v3=2; rotational J=11, Ka=2, Kc=9).

        Returns
        -------
        dict
            Mapping from field name (``str``) to a typed scalar value
            (``int``, ``float``, or ``str``).  Fields that are absent or
            unparseable receive a dtype-appropriate sentinel:

            - ``int``   → ``-999``
            - ``float`` → ``nan``
            - ``str``   → ``''``
        """

    # ── Concrete helpers ─────────────────────────────────────────────

    def parse_bulk(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Parse an array of label strings into parallel numpy arrays.

        Calls :meth:`parse_label` for every element via
        :func:`numpy.frompyfunc`, producing one contiguous array per field —
        suitable for vectorised downstream operations.

        Parameters
        ----------
        labels:
            1-D array of state label strings (``dtype=str`` or ``object``).

        Returns
        -------
        dict
            Mapping from field name to a 1-D numpy array of values.
            Array dtypes: ``np.int32`` for ``'int'`` fields,
            ``np.float64`` for ``'float'`` fields, ``object`` for ``'str'``
            fields.

        Notes
        -----
        :class:`GenericDelimitedSchema` overrides this method because its
        field set is determined at parse time (unknown in advance).
        """
        all_fields = self.global_fields + self.local_fields
        if len(labels) == 0:
            return {f.name: np.array([], dtype=self._field_numpy_dtype(f))
                    for f in all_fields}

        # Vectorise over labels; each element is a parsed dict.
        _vec = np.frompyfunc(self.parse_label, 1, 1)
        parsed: np.ndarray = _vec(labels)

        return {
            f.name: np.array([d[f.name] for d in parsed],
                              dtype=self._field_numpy_dtype(f))
            for f in all_fields
        }

    def format_label(self, qdict: Dict[str, Any]) -> str:
        """Reconstruct a label string from a dict of quantum numbers.

        Produces the canonical ``"global|local"`` form used by iSLAT.

        Parameters
        ----------
        qdict:
            Mapping from field name to value (as produced by
            :meth:`parse_label`).

        Returns
        -------
        str
            Reconstructed label string.
        """
        g_parts = [str(qdict.get(f.name, '')) for f in self.global_fields]
        l_parts = [str(qdict.get(f.name, '')) for f in self.local_fields]
        return '_'.join(p for p in g_parts if p) + '|' + '_'.join(p for p in l_parts if p)

    # ── Utility ─────────────────────────────────────────────────────

    @staticmethod
    def _field_numpy_dtype(field: QuantumField):
        """Return the numpy dtype for *field*."""
        if field.dtype == 'int':
            return np.int32
        if field.dtype == 'float':
            return np.float64
        return object  # variable-length unicode strings

    @property
    def all_fields(self) -> Tuple[QuantumField, ...]:
        """All fields in declaration order: global then local."""
        return self.global_fields + self.local_fields

    @property
    def field_names(self) -> Tuple[str, ...]:
        """All field names in declaration order."""
        return tuple(f.name for f in self.all_fields)

    def __repr__(self) -> str:
        g = [f.name for f in self.global_fields]
        l = [f.name for f in self.local_fields]
        return f"{self.__class__.__name__}(global={g}, local={l})"

# ═══════════════════════════════════════════════════════════════════
#  _DelimitedSchema  — shared parsing logic for ``_``-delimited labels
# ═══════════════════════════════════════════════════════════════════
class _DelimitedSchema(QuantumStateSchema):
    """Shared ``_``-delimited parsing logic.

    Subclasses declare :attr:`global_fields` and :attr:`local_fields`;
    :meth:`parse_label` and :meth:`format_label` are implemented here.

    Subclasses may override :meth:`_fixup_global_tokens` and/or
    :meth:`_fixup_local_tokens` to resolve known cases where HITRAN
    adjacent fixed-width fields are concatenated without a whitespace
    separator (e.g. CO₂'s *v3* and *r* fields, or H₂O's all-negative
    vibrational sentinel ``"-2-2-2"``).
    """

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        """Post-process ``_``-split global tokens.

        Override in subclasses when adjacent HITRAN fixed-width fields
        produce fewer tokens than :attr:`global_fields`.  The default
        implementation returns *tokens* unchanged.
        """
        return tokens

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        """Post-process ``_``-split local tokens.

        Override in subclasses when adjacent HITRAN fixed-width fields
        produce fewer tokens than :attr:`local_fields`.  The default
        implementation returns *tokens* unchanged.
        """
        return tokens

    def parse_label(self, label: str) -> Dict[str, Any]:
        parts = label.split('|', 1)
        global_str = parts[0] if parts else ''
        local_str = parts[1] if len(parts) > 1 else ''

        g_tokens = self._fixup_global_tokens(
            global_str.split('_') if global_str else [])
        l_tokens = self._fixup_local_tokens(
            local_str.split('_') if local_str else [])

        result: Dict[str, Any] = {}
        for i, field in enumerate(self.global_fields):
            raw = g_tokens[i] if i < len(g_tokens) else ''
            result[field.name] = _coerce(raw, field.dtype)
        for i, field in enumerate(self.local_fields):
            raw = l_tokens[i] if i < len(l_tokens) else ''
            result[field.name] = _coerce(raw, field.dtype)
        return result

    def format_label(self, qdict: Dict[str, Any]) -> str:
        g_parts = [str(qdict.get(f.name, '')) for f in self.global_fields]
        l_parts = [str(qdict.get(f.name, '')) for f in self.local_fields]
        return '_'.join(p for p in g_parts if p != '') + '|' + \
               '_'.join(p for p in l_parts if p != '')

# ═══════════════════════════════════════════════════════════════════
#  GenericDelimitedSchema  — catalog-agnostic fallback
# ═══════════════════════════════════════════════════════════════════
class GenericDelimitedSchema(QuantumStateSchema):
    """Fallback schema for molecules without a registered specific schema.

    Field names are assigned positionally: ``g0``, ``g1``, … for the global
    (pre-``|``) tokens and ``l0``, ``l1``, … for the local (post-``|``)
    tokens.  All values are stored as strings.

    Because token count varies per line, the returned dict's keys are
    determined dynamically at parse time.  This makes :meth:`parse_bulk`
    slightly more expensive than for fixed-field schemas (it performs two
    passes), but it guarantees no data is lost.

    This schema is also suitable as a base class for non-HITRAN catalogs
    that wish to add named aliases on top of positional access.
    """

    global_fields: Tuple[QuantumField, ...] = ()
    local_fields: Tuple[QuantumField, ...] = ()

    def parse_label(self, label: str) -> Dict[str, Any]:
        parts = label.split('|', 1)
        global_str = parts[0] if parts else ''
        local_str = parts[1] if len(parts) > 1 else ''

        g_tokens = global_str.split('_') if global_str else []
        l_tokens = local_str.split('_') if local_str else []

        result: Dict[str, Any] = {}
        for i, tok in enumerate(g_tokens):
            result[f'g{i}'] = tok
        for i, tok in enumerate(l_tokens):
            result[f'l{i}'] = tok
        return result

    def parse_bulk(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        if len(labels) == 0:
            return {}
        # First pass: parse all labels into dicts.
        parsed = [self.parse_label(lbl) for lbl in labels]
        # Second pass: collect all keys.
        all_keys: Dict[str, None] = {}
        for d in parsed:
            for k in d:
                all_keys.setdefault(k, None)

        def _sort_key(k: str) -> Tuple[str, int]:
            return k[0], int(k[1:])

        result: Dict[str, np.ndarray] = {}
        for key in sorted(all_keys.keys(), key=_sort_key):
            result[key] = np.array([d.get(key, '') for d in parsed], dtype=object)
        return result

    def format_label(self, qdict: Dict[str, Any]) -> str:
        g_keys = sorted((k for k in qdict if k.startswith('g')),
                        key=lambda k: int(k[1:]))
        l_keys = sorted((k for k in qdict if k.startswith('l')),
                        key=lambda k: int(k[1:]))
        g_parts = [str(qdict[k]) for k in g_keys]
        l_parts = [str(qdict[k]) for k in l_keys]
        return '_'.join(g_parts) + '|' + '_'.join(l_parts)

# ═══════════════════════════════════════════════════════════════════
#  QuantumStateRegistry
# ═══════════════════════════════════════════════════════════════════
class QuantumStateRegistry:
    """Central registry mapping molecule identifiers to parsing schemas.

    HITRAN schemas are registered automatically when
    :mod:`~iSLAT.Modules.DataTypes.HITRANQuantumSchemas` is first imported.
    Custom schemas for non-HITRAN catalogs can be registered at any time via
    :meth:`register`.

    All methods are classmethods so the registry acts as a global singleton
    — no instantiation is required.

    Examples
    --------
    Register a custom schema::

        from iSLAT.Modules.DataTypes.QuantumStateSchema import (
            QuantumStateRegistry, QuantumField, GenericDelimitedSchema,
        )

        class MySchemaCls(GenericDelimitedSchema):
            global_fields = (QuantumField('v', 'int', 'Vibrational mode'),)
            local_fields  = (QuantumField('J', 'int', 'Rotational J'),)

        QuantumStateRegistry.register("MyMolecule", MySchemaCls())

    Retrieve the schema for a line list::

        schema = QuantumStateRegistry.get_schema("H2O")
        parsed = schema.parse_label("0_2_2|11_2_9")
        # → {'v1': 0, 'v2': 2, 'v3': 2, 'J': 11, 'Ka': 2, 'Kc': 9}
    """

    _registry: ClassVar[Dict[str, QuantumStateSchema]] = {}
    _fallback: ClassVar[QuantumStateSchema] = GenericDelimitedSchema()

    @classmethod
    def register(cls, molecule_id: str, schema: QuantumStateSchema) -> None:
        """Register *schema* for *molecule_id*.

        Parameters
        ----------
        molecule_id:
            Identifier string as stored in
            :attr:`~iSLAT.Modules.DataTypes.MoleculeLineList.MoleculeLineList.molecule_id`
            (e.g. ``"H2O"``, ``"CO"``, ``"HCN"``).
        schema:
            A concrete :class:`QuantumStateSchema` instance.
        """
        cls._registry[molecule_id] = schema

        # Register quantum-field axis entries with PlotAxisRegistry so that
        # GUI dialogs and plot classes discover them automatically.
        try:
            from iSLAT.Modules.DataTypes.PlotAxisRegistry import (
                PlotAxisEntry,
                PlotAxisRegistry,
            )
            import numpy as _np

            all_fields = list(schema.global_fields) + list(schema.local_fields)
            for _f in all_fields:
                _is_numeric = _f.dtype in ("int", "float")
                _kind: str = "continuous" if _is_numeric else "categorical"
                _fname = _f.name
                _desc = _f.description or _fname

                for _state, _lev_key in (("upper", "lev_up"), ("lower", "lev_low")):
                    _key = f"qn_{_state}:{_fname}"
                    _label = f"{_fname} ({_state})"
                    _display = f"{_fname} \u2014 {_state} state (quantum)"

                    # Capture loop variables by default-argument trick
                    def _make_resolve_array(_lk=_lev_key, _fn=_fname, _s=schema):
                        def _resolve(cdata, __lk=_lk, __fn=_fn, __s=_s):
                            lev_arr = cdata.get(__lk)
                            if lev_arr is None:
                                return None
                            try:
                                labels = _np.asarray(lev_arr, dtype="U64")
                                parsed = __s.parse_bulk(labels)
                                field_vals = parsed.get(__fn)
                                if field_vals is None:
                                    return None
                                # Replace integer sentinel -999 with NaN
                                arr = _np.asarray(field_vals, dtype=float)
                                arr = _np.where(arr == -999, _np.nan, arr)
                                return arr
                            except Exception:
                                return None
                        return _resolve

                    def _make_resolve_scalar(_st=_state, _fn=_fname):
                        def _resolve(line, mol_id, __st=_st, __fn=_fn):
                            try:
                                import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401
                                from iSLAT.Modules.DataTypes.QuantumStateSchema import (
                                    QuantumStateRegistry as _QR,
                                )
                                _mid = mol_id or getattr(line, "molecule_id", None)
                                _schema = _QR.get_schema(_mid)
                                qd = line.get_quantum_dict(__st, schema=_schema)
                                val = qd.get(__fn)
                                if val is None or val == -999:
                                    return None
                                return float(val)
                            except Exception:
                                return None
                        return _resolve

                    PlotAxisRegistry.register(
                        PlotAxisEntry(
                            key=_key,
                            display_name=_display,
                            label=_label,
                            kind=_kind,  # type: ignore[arg-type]
                            available_as_axis=_is_numeric,
                            available_as_color=True,
                            suggest_log=False,
                            resolve_array=_make_resolve_array(),
                            resolve_scalar=_make_resolve_scalar(),
                            group=f"Quantum \u2014 {_state}",
                            quantum_field=_fname,
                        )
                    )
        except Exception:
            pass  # Registry registration is best-effort; never break molecule loading

    @classmethod
    def get_schema(cls, molecule_id: Optional[str]) -> QuantumStateSchema:
        """Return the schema for *molecule_id*, or the generic fallback.

        If *molecule_id* is ``None`` or not registered, the singleton
        :class:`GenericDelimitedSchema` instance is returned so callers
        always receive a usable schema object.

        Parameters
        ----------
        molecule_id:
            Molecule identifier string.  May be ``None``.

        Returns
        -------
        QuantumStateSchema
            Registered schema, or :class:`GenericDelimitedSchema`.
        """
        if molecule_id is None:
            return cls._fallback
        return cls._registry.get(molecule_id, cls._fallback)

    @classmethod
    def get_fields(cls, molecule_id: Optional[str]) -> Tuple[QuantumField, ...]:
        """Return the quantum fields declared for *molecule_id*.

        This is the single source of truth for "which parsed quantum numbers
        does this species expose" and is shared by the line-list filter
        window and the population-diagram axis / color-by dialogs.  An
        unregistered or ``None`` id resolves to the generic fallback schema,
        which declares no fields, so the result is empty.
        """
        return tuple(cls.get_schema(molecule_id).all_fields)

    @classmethod
    def get_field_names(cls, molecule_ids: Iterable[Optional[str]]) -> Set[str]:
        """Return the union of quantum field names declared for *molecule_ids*."""
        names: Set[str] = set()
        for mid in molecule_ids:
            names.update(f.name for f in cls.get_fields(mid))
        return names

    @classmethod
    def list_molecules(cls) -> List[str]:
        """Return a sorted list of all registered molecule identifiers."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, molecule_id: str) -> bool:
        """Return ``True`` if *molecule_id* has a registered schema."""
        return molecule_id in cls._registry

    @classmethod
    def unregister(cls, molecule_id: str) -> None:
        """Remove the schema for *molecule_id* (primarily for testing).

        Parameters
        ----------
        molecule_id:
            Molecule identifier to remove.  Silently ignored if not found.
        """
        cls._registry.pop(molecule_id, None)