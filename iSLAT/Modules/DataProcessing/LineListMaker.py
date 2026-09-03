"""
LineListMaker — chainable masking and saving of molecule line-list objects.

Provides a fluent builder API for filtering :class:`MoleculeLineList` data and
exporting to CSV, ``.par``, or pandas DataFrame.  Every filter method
returns ``self`` so calls can be chained::

    (LineListMaker(h2o_lines)
        .filter_wavelength(5.0, 8.0)
        .filter_eup(max_val=4000)
        .filter_astein(min_val=1e-2)
        .to_csv("h2o_filtered.csv"))
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ..DataTypes.MoleculeLineList import MoleculeLineList
from .LineFilterExpression import (  # noqa: F401  (re-exported for back-compat)
    Condition,
    ConditionError,
    ConditionGroup,
    FilterExpression,
    MaskContext,
    describe_expression,
    expression_mask,
    _parse_vib_band,
    _vib_part,
    _vib_perms,
    _vib_perms_up_to,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Core CSV columns in canonical order (matches iSLAT line-list convention)
_CSV_COLUMNS: List[str] = [
    "species", "lev_up", "lev_low", "lam",
    "a_stein", "e_up", "e_low", "g_up", "g_low",
]

# Extended CSV columns (optional)
_CSV_EXTENDED_COLUMNS: List[str] = _CSV_COLUMNS + ["xmin", "xmax"]


def _ensure_dataframe(df_or_linelist: Union[pd.DataFrame, MoleculeLineList],
                       molecule_id: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame from either a DataFrame or a MoleculeLineList."""
    if isinstance(df_or_linelist, pd.DataFrame):
        return df_or_linelist.copy()
    if isinstance(df_or_linelist, MoleculeLineList):
        df = df_or_linelist.get_pandas_table()
        if molecule_id is None:
            molecule_id = getattr(df_or_linelist, "molecule_id", None)
        if molecule_id and "species" not in df.columns:
            df.insert(0, "species", molecule_id)
        return df
    raise TypeError(
        f"Expected pd.DataFrame or MoleculeLineList, got {type(df_or_linelist).__name__}"
    )


# ╭──────────────────────────────────────────────────────────────────╮
# │  LineListMaker                                                   │
# ╰──────────────────────────────────────────────────────────────────╯

class LineListMaker:
    """Chainable builder for filtering and exporting spectral line lists.

    Parameters
    ----------
    source : MoleculeLineList or pd.DataFrame
        The line-list data to work with.  A *copy* is made so the original
        object is never modified.
    species : str, optional
        Override the species label written to exports.  If ``None``, the
        value is derived from ``source.molecule_id`` (for a
        ``MoleculeLineList``) or from an existing ``"species"`` column.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        source: Union[MoleculeLineList, pd.DataFrame],
        species: Optional[str] = None,
    ) -> None:
        # Keep a reference if user passed a MoleculeLineList (for .par export)
        self._linelist: Optional[MoleculeLineList] = (
            source if isinstance(source, MoleculeLineList) else None
        )

        # Resolve species
        if species is None and isinstance(source, MoleculeLineList):
            species = getattr(source, "molecule_id", None)
        self._species: Optional[str] = species

        # Build the working DataFrame (include extras when available)
        if isinstance(source, MoleculeLineList) and source.extra_fields:
            self._df: pd.DataFrame = _ensure_dataframe(source, molecule_id=species)
            # Append extra columns from the linelist
            n_rows = len(self._df)
            for col, values in source.extra_fields.items():
                if col not in self._df.columns and len(values) == n_rows:
                    self._df[col] = values
        else:
            self._df: pd.DataFrame = _ensure_dataframe(source, molecule_id=species)

        # Overwrite species column if explicitly provided
        if species is not None:
            self._df["species"] = species

        # Keep a snapshot of the original unfiltered data for reset()
        self._df_original: pd.DataFrame = self._df.copy()

        # Active filter registry: list of (name, kwargs) tuples
        self._filters: List[Tuple[str, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        molecule_id: Optional[str] = None,
        format: Optional[str] = None,
    ) -> "LineListMaker":
        """Construct a :class:`LineListMaker` from any supported line-list file.

        Uses :meth:`MoleculeLineList.from_file` under the hood, so all
        registered readers (HITRAN ``.par``, iSLAT CSV, saved-lines CSV,
        and custom formats) are supported.

        Parameters
        ----------
        filepath : str or Path
            Path to the line-list file.
        molecule_id : str, optional
            Molecule identifier.  Derived from file metadata when *None*.
        format : str, optional
            ``"hitran"``, ``"csv"``, ``"saved"``, or *None* for auto-detect.

        Returns
        -------
        LineListMaker
        """
        ll = MoleculeLineList.from_file(
            filepath,
            molecule_id=molecule_id,
            format=format,
        )
        return cls(ll, species=ll.molecule_id)

    @classmethod
    def from_saved_lines(
        cls,
        filepath: Union[str, Path],
        molecule_id: Optional[str] = None,
    ) -> "LineListMaker":
        """Convenience constructor for LINESAVES CSV files.

        Parameters
        ----------
        filepath : str or Path
            Path to a saved-lines CSV (e.g. ``DATAFILES/LINESAVES/BALLS.csv``).
        molecule_id : str, optional
            Molecule identifier.

        Returns
        -------
        LineListMaker
            A maker whose DataFrame includes all fit-result columns
            (``Flux_data``, ``Err_data``, ``FWHM_fit``, etc.).
        """
        return cls.from_file(filepath, molecule_id=molecule_id, format="saved")

    # ------------------------------------------------------------------
    # Repr / info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        species = self._species or "unknown"
        return (
            f"<LineListMaker species={species!r} "
            f"lines={len(self._df)} filters={len(self._filters)}>"
        )

    def __len__(self) -> int:
        """Number of lines after filtering."""
        return len(self._df)

    def summary(self) -> str:
        """Human-readable summary of the current state."""
        lines = [repr(self)]
        if not self._df.empty:
            lam = self._df["lam"]
            lines.append(
                f"  λ range : {lam.min():.5f} - {lam.max():.5f} µm  "
                f"({len(self._df)} lines)"
            )
        if self._filters:
            lines.append("  Active filters:")
            for name, kw in self._filters:
                if name == "filter_expression":
                    expr = FilterExpression.from_dict(kw["expr"])
                    lines.append(f"    • {name}: {describe_expression(expr)}")
                    continue
                param_str = ", ".join(f"{k}={v!r}" for k, v in kw.items())
                lines.append(f"    • {name}({param_str})")
        else:
            lines.append("  No filters applied.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Filter helpers (private)
    # ------------------------------------------------------------------

    def _record_filter(self, name: str, **kwargs: Any) -> None:
        """Store a filter entry for introspection / replay."""
        self._filters.append((name, kwargs))

    def _apply_mask(self, mask: pd.Series) -> "LineListMaker":
        """Apply a boolean mask and return *self* for chaining."""
        self._df = self._df.loc[mask].reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Column range filter (generic)
    # ------------------------------------------------------------------

    def filter_range(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Keep rows where *column* falls within [min_val, max_val].

        Parameters
        ----------
        column : str
            Column name (e.g. ``"lam"``, ``"e_up"``, ``"a_stein"``).
        min_val, max_val : float, optional
            Inclusive bounds.  ``None`` means unbounded.
        """
        if column not in self._df.columns:
            warnings.warn(f"Column {column!r} not in DataFrame — filter skipped.")
            return self
        mask = pd.Series(True, index=self._df.index)
        if min_val is not None:
            mask &= self._df[column] >= min_val
        if max_val is not None:
            mask &= self._df[column] <= max_val
        self._record_filter("filter_range", column=column,
                            min_val=min_val, max_val=max_val)
        return self._apply_mask(mask)

    # ------------------------------------------------------------------
    # Convenience filters (all delegate to filter_range or _apply_mask)
    # ------------------------------------------------------------------

    def filter_wavelength(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by wavelength (µm)."""
        self._record_filter("filter_wavelength",
                            min_val=min_val, max_val=max_val)
        # Direct mask (don't double-record via filter_range)
        return self._range_mask("lam", min_val, max_val)

    def filter_eup(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by upper-state energy *E_up* (K)."""
        self._record_filter("filter_eup", min_val=min_val, max_val=max_val)
        return self._range_mask("e_up", min_val, max_val)

    def filter_elow(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by lower-state energy *E_low* (K)."""
        self._record_filter("filter_elow", min_val=min_val, max_val=max_val)
        return self._range_mask("e_low", min_val, max_val)

    def filter_astein(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by Einstein-A coefficient (s⁻¹)."""
        self._record_filter("filter_astein", min_val=min_val, max_val=max_val)
        return self._range_mask("a_stein", min_val, max_val)

    def filter_freq(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by frequency (Hz)."""
        self._record_filter("filter_freq", min_val=min_val, max_val=max_val)
        return self._range_mask("freq", min_val, max_val)

    def filter_gup(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by upper-state degeneracy."""
        self._record_filter("filter_gup", min_val=min_val, max_val=max_val)
        return self._range_mask("g_up", min_val, max_val)

    def filter_glow(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by lower-state degeneracy."""
        self._record_filter("filter_glow", min_val=min_val, max_val=max_val)
        return self._range_mask("g_low", min_val, max_val)

    # ------------------------------------------------------------------
    # Advanced filters
    # ------------------------------------------------------------------

    def filter_quantum(
        self,
        lev_up: Optional[str] = None,
        lev_low: Optional[str] = None,
        contains: bool = False,
    ) -> "LineListMaker":
        """Filter by quantum-state labels.

        Parameters
        ----------
        lev_up, lev_low : str, optional
            Exact match (or substring if *contains* is ``True``).
        contains : bool
            If ``True``, use substring matching instead of exact equality.
        """
        self._record_filter("filter_quantum",
                            lev_up=lev_up, lev_low=lev_low, contains=contains)
        mask = pd.Series(True, index=self._df.index)
        if lev_up is not None:
            if contains:
                mask &= self._df["lev_up"].str.contains(lev_up, na=False)
            else:
                mask &= self._df["lev_up"] == lev_up
        if lev_low is not None:
            if contains:
                mask &= self._df["lev_low"].str.contains(lev_low, na=False)
            else:
                mask &= self._df["lev_low"] == lev_low
        return self._apply_mask(mask)

    def filter_quantum_field(
        self,
        field: str,
        *,
        value: Optional[Any] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        state: str = "upper",
    ) -> "LineListMaker":
        """Filter by a parsed quantum number field.

        Uses the :class:`~iSLAT.Modules.DataTypes.QuantumStateSchema.QuantumStateRegistry`
        to parse the ``lev_up`` or ``lev_low`` label column and keep only
        rows where the specified quantum-number field matches the given
        criterion.

        Parameters
        ----------
        field : str
            Quantum-field name as defined in the molecule's schema
            (e.g. ``"J"``, ``"v"``, ``"Ka"``, ``"v1"``).
        value : optional
            Exact value to match (equality test).  For int/float fields
            sentinels (-999 / NaN) are treated as *missing* and excluded.
        min_val, max_val : float, optional
            Range limits (inclusive) for numeric fields.  Can be combined
            with each other but are ignored when *value* is given.
        state : str
            ``"upper"`` (default) to parse ``lev_up`` labels, or
            ``"lower"`` to parse ``lev_low`` labels.

        Returns
        -------
        LineListMaker
            ``self`` for chaining.
        """
        self._record_filter(
            "filter_quantum_field",
            field=field, value=value,
            min_val=min_val, max_val=max_val, state=state,
        )

        lev_col = "lev_up" if state == "upper" else "lev_low"
        if lev_col not in self._df.columns:
            warnings.warn(f"Column {lev_col!r} not in DataFrame — filter_quantum_field skipped.")
            return self

        # Lazy import so the quantum machinery is only loaded when needed.
        from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
        import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # register all schemas

        schema = QuantumStateRegistry.get_schema(self._species)
        labels = np.array(self._df[lev_col].fillna(""), dtype="U64")
        parsed = schema.parse_bulk(labels)

        if field not in parsed:
            warnings.warn(
                f"Quantum field {field!r} not found in schema for species "
                f"{self._species!r} — filter_quantum_field skipped."
            )
            return self

        field_vals = parsed[field]

        # Build the mask -----------------------------------------------
        if value is not None:
            # Exact match (works for str, int, and float fields)
            mask_arr = field_vals == value
        else:
            try:
                numeric_vals = np.asarray(field_vals, dtype=float)
            except (ValueError, TypeError):
                warnings.warn(
                    f"Quantum field {field!r} cannot be used with range "
                    f"limits — convert to an exact value match instead."
                )
                return self
            # Exclude sentinel values (-999 maps to NaN after int→float)
            valid = np.isfinite(numeric_vals) & (numeric_vals != -999)
            mask_arr = valid.copy()
            if min_val is not None:
                mask_arr &= numeric_vals >= min_val
            if max_val is not None:
                mask_arr &= numeric_vals <= max_val

        mask = pd.Series(mask_arr, index=self._df.index)
        return self._apply_mask(mask)

    def filter_species(self, *species: str) -> "LineListMaker":
        """Keep only rows matching one of the given species names.

        Useful when working with a DataFrame that contains multiple species.
        """
        self._record_filter("filter_species", species=species)
        if "species" not in self._df.columns:
            warnings.warn("No 'species' column — filter_species skipped.")
            return self
        mask = self._df["species"].isin(species)
        return self._apply_mask(mask)

    def filter_custom(
        self,
        func: Callable[[pd.DataFrame], pd.Series],
        label: str = "custom",
    ) -> "LineListMaker":
        """Apply an arbitrary boolean mask function.

        Parameters
        ----------
        func : callable
            Receives the current DataFrame, must return a boolean Series.
        label : str
            A short description stored in the filter log.

        Example
        -------
        >>> maker.filter_custom(lambda df: df["a_stein"] > df["a_stein"].median(),
        ...                     label="above_median_astein")
        """
        self._record_filter("filter_custom", label=label)
        mask = func(self._df)
        return self._apply_mask(mask)

    def filter_vib_band(
        self,
        spec: str,
        n_modes: int = 3,
    ) -> "LineListMaker":
        """Filter to a specific vibrational band.

        Uses exact vibrational-level matching on both upper and lower states
        (the vibrational portion is the part of the quantum label before the
        ``'|'`` delimiter).

        Parameters
        ----------
        spec : str
            Band specification string.  The leading ``'v'`` is optional.

            - ``"v1"``   — upper from exact v=1, lower from any state up to
              v=1 (``max(qi) ≤ 1``).  Captures fundamental *and* hot bands.
            - ``"v1-0"`` — upper from exact v=1, lower from exact v=0.
            - ``"v2-1"`` — upper from exact v=2, lower from exact v=1.

        n_modes : int
            Number of vibrational modes (default 3 for H₂O, CO₂, etc.).

        Raises
        ------
        ValueError
            If *spec* cannot be parsed.
        """
        up_set, low_set = _parse_vib_band(spec, n_modes)
        self._record_filter("filter_vib_band", spec=spec, n_modes=n_modes)
        mask = (
            self._df["lev_up"].apply(lambda x: _vib_part(x) in up_set)
            & self._df["lev_low"].apply(lambda x: _vib_part(x) in low_set)
        )
        return self._apply_mask(mask)

    # ------------------------------------------------------------------
    # Boolean expression filter (AND / OR of many conditions)
    # ------------------------------------------------------------------

    def filter_expression(
        self,
        expr: Union["FilterExpression", Dict[str, Any]],
        *,
        on_error: str = "raise",
        derived_source: Optional[Callable[[], Any]] = None,
    ) -> "LineListMaker":
        """Apply a two-level boolean expression as a SINGLE filter step.

        Groups combine internally by their own AND/OR; the resulting group
        masks then combine by the expression's AND/OR.  Because there are
        exactly two levels and one operator per level, precedence is
        structural - there is no operator-binding rule to remember.

        The expression is evaluated against the *current* DataFrame, so it
        narrows exactly like every other filter and composes as an implicit
        AND with anything applied before it.  It is recorded as **one** entry
        in the flat filter log, so :meth:`pop_filter`, :meth:`remove_filter`
        and :meth:`copy` all work unchanged.

        >>> maker.filter_expression({
        ...     "join": "AND",
        ...     "groups": [{"join": "OR", "conditions": [
        ...         {"kind": "range", "field": "lam",  "op": "between",
        ...          "min_val": 6.0,  "max_val": 7.0},
        ...         {"kind": "range", "field": "lam",  "op": "between",
        ...          "min_val": 12.0, "max_val": 13.0},
        ...         {"kind": "range", "field": "e_up", "op": "ge",
        ...          "min_val": 5000.0}]}]})

        Parameters
        ----------
        expr : FilterExpression or dict
            A dict is parsed via :meth:`FilterExpression.from_dict`.
        on_error : {"raise", "identity", "drop"}
            ``"raise"`` (default) aborts with :class:`ConditionError` and
            records nothing.  ``"identity"`` treats an unevaluable condition as
            all-True; **this is unsafe inside an OR group**, where it widens
            the group to the whole line list.  ``"drop"`` omits it from the
            fold.
        derived_source : callable, optional
            Zero-argument provider of a DataFrame of model quantities (line
            intensity, optical depth) indexed like the current frame.
            Required only by ``derived`` conditions, which are not
            self-contained in the line list.

        Returns
        -------
        LineListMaker
            ``self`` for chaining.

        Raises
        ------
        ConditionError
            When a condition cannot be evaluated and ``on_error="raise"``.
            Nothing is recorded and the DataFrame is left untouched.
        """
        expr_obj = (expr if isinstance(expr, FilterExpression)
                    else FilterExpression.from_dict(expr))
        ctx = MaskContext(species=self._species, on_error=on_error,
                          derived_source=derived_source)
        # Build the mask BEFORE recording: with on_error="raise" as the
        # default, recording first would leave a recorded-but-never-applied
        # entry that re-raises on the next pop_filter / remove_filter.
        mask = expression_mask(self._df, expr_obj, ctx)
        self._record_filter("filter_expression",
                            expr=expr_obj.to_dict(), on_error=on_error)
        return self._apply_mask(mask)

    def filter_any(self, *conditions: "Condition", **kw: Any) -> "LineListMaker":
        """Keep rows matching ANY of *conditions* (a single OR group).

        >>> maker.filter_any(
        ...     Condition("range", field="lam",  op="between", min_val=6.0,  max_val=7.0),
        ...     Condition("range", field="lam",  op="between", min_val=12.0, max_val=13.0),
        ...     Condition("range", field="e_up", op="ge",      min_val=5000.0))
        """
        return self.filter_expression(
            FilterExpression(groups=(ConditionGroup(tuple(conditions), join="OR"),)),
            **kw,
        )

    def filter_all(self, *conditions: "Condition", **kw: Any) -> "LineListMaker":
        """Keep rows matching ALL of *conditions* (a single AND group)."""
        return self.filter_expression(
            FilterExpression(groups=(ConditionGroup(tuple(conditions), join="AND"),)),
            **kw,
        )

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    def sort(
        self,
        by: Union[str, List[str]] = "lam",
        ascending: bool = True,
    ) -> "LineListMaker":
        """Sort lines by one or more columns.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to sort on.  Default ``"lam"``.
        ascending : bool
            Sort direction.
        """
        self._df = self._df.sort_values(by, ascending=ascending).reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Filter inspection / editing
    # ------------------------------------------------------------------

    @property
    def filters(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return a *copy* of the active filter log."""
        return list(self._filters)

    @property
    def expression(self) -> Optional["FilterExpression"]:
        """The most recently applied boolean expression, or ``None``.

        Derived from the filter log rather than stored separately, so
        :meth:`copy` and :meth:`reset` need no extra bookkeeping.
        """
        for name, kw in reversed(self._filters):
            if name == "filter_expression":
                return FilterExpression.from_dict(kw["expr"])
        return None

    @property
    def original_df(self) -> pd.DataFrame:
        """Read-only view of the unfiltered snapshot.  **Do not mutate.**

        Returned *without* copying so a live GUI preview can evaluate against
        it on every keystroke without cloning a large frame each time.
        """
        return self._df_original

    @property
    def species(self) -> Optional[str]:
        """Currently set species label."""
        return self._species

    @species.setter
    def species(self, value: str) -> None:
        self._species = value
        if "species" in self._df.columns:
            self._df["species"] = value
        if "species" in self._df_original.columns:
            self._df_original["species"] = value

    def reset(self) -> "LineListMaker":
        """Remove all filters and restore the original data."""
        self._df = self._df_original.copy()
        self._filters.clear()
        return self

    def pop_filter(self) -> "LineListMaker":
        """Remove the last filter and replay the remaining ones.

        This rebuilds the DataFrame from the original snapshot and re-applies
        every filter except the last one.

        Returns
        -------
        LineListMaker
            ``self`` for chaining.

        Raises
        ------
        IndexError
            If no filters are applied.
        """
        if not self._filters:
            raise IndexError("No filters to pop.")
        return self._remove_and_replay(len(self._filters) - 1)

    def remove_filter(self, index: int) -> "LineListMaker":
        """Remove the filter at *index* and replay the rest.

        Parameters
        ----------
        index : int
            Zero-based index into :attr:`filters`.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        return self._remove_and_replay(index)

    def _remove_and_replay(self, index: int) -> "LineListMaker":
        """Drop the filter at *index* and replay, atomically.

        On any replay failure the maker keeps the filters and data it had
        before the call, so a failed removal never silently unfilters the
        line list.
        """
        saved = list(self._filters)
        df_at_entry = self._df
        del self._filters[index]
        try:
            return self._replay_filters()
        except Exception:
            self._filters = saved
            self._df = df_at_entry
            raise

    # ------------------------------------------------------------------
    # Export — DataFrame
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Return a *copy* of the current (filtered) DataFrame."""
        return self._df.copy()

    def to_dataframe(self, include_species: bool = True) -> pd.DataFrame:
        """Return a copy of the current filtered data as a DataFrame.

        Parameters
        ----------
        include_species : bool
            If ``True`` (default), ensure a ``"species"`` column is present.
        """
        df = self._df.copy()
        if include_species and "species" not in df.columns and self._species:
            df.insert(0, "species", self._species)
        return df

    # ------------------------------------------------------------------
    # Export — CSV
    # ------------------------------------------------------------------

    def to_csv(
        self,
        path: Union[str, Path],
        extended: bool = False,
        extra_columns: Optional[Dict[str, Any]] = None,
        sort_by: str = "lam",
        **csv_kwargs: Any,
    ) -> Path:
        """Write the filtered line list to a CSV file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        extended : bool
            If ``True``, include ``xmin`` and ``xmax`` columns (set to
            ``NaN`` if not already present).
        extra_columns : dict, optional
            Additional constant-value columns to append (e.g.
            ``{"note": ""}``).
        sort_by : str
            Column to sort on before writing.  Set to ``""`` to skip.
        **csv_kwargs
            Forwarded to :meth:`pandas.DataFrame.to_csv`.

        Returns
        -------
        Path
            The resolved output path.
        """
        path = Path(path)
        df = self._prepare_export_df(extended=extended,
                                     extra_columns=extra_columns,
                                     sort_by=sort_by)
        csv_kwargs.setdefault("index", False)
        df.to_csv(path, **csv_kwargs)
        return path

    # ------------------------------------------------------------------
    # Export — .par
    # ------------------------------------------------------------------

    def to_par(
        self,
        path: Union[str, Path],
        header: Optional[pd.DataFrame] = None,
        partition_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Write the filtered line list to a ``.par`` file.

        This delegates to :meth:`MoleculeLineList.write_par_file`.  A
        ``MoleculeLineList`` source is required (it carries the partition
        function needed for the ``.par`` header).

        Parameters
        ----------
        path : str or Path
            Output file path.
        header : pd.DataFrame, optional
            Single-row DataFrame overriding header fields (``molecule_id``,
            ``source``, ``molar_mass``, etc.).
        partition_df : pd.DataFrame, optional
            DataFrame with columns ``'temp'`` and ``'q'`` for the partition function.
            When provided, this is written instead of the line list's own partition function data.

        Returns
        -------
        Path
            The resolved output path.

        Raises
        ------
        RuntimeError
            If the maker was not initialised from a ``MoleculeLineList``.
        """
        if self._linelist is None:
            raise RuntimeError(
                "to_par() requires a MoleculeLineList source "
                "(pass one to the constructor)."
            )
        path = Path(path)
        # Build the lines DataFrame in the format write_par_file expects
        lines_df = self._df.drop(columns=["species"], errors="ignore").copy()
        self._linelist.write_par_file(
            file_path=path, header=header, lines_df=lines_df, partition_df=partition_df
        )
        return path

    # ------------------------------------------------------------------
    # Export — MoleculeLineList
    # ------------------------------------------------------------------

    def to_linelist(self) -> MoleculeLineList:
        """Create a new :class:`MoleculeLineList` from the filtered data.

        Extra columns beyond the 10-core fields (and ``species``) are
        preserved via :attr:`MoleculeLineList.extra_fields`.

        Returns
        -------
        MoleculeLineList
            A fresh instance containing only the filtered lines.
        """
        from ..DataTypes.MoleculeLineList import MoleculeLineList as _MLL
        from iSLAT.Modules.FileHandling.line_list_readers import CORE_FIELD_NAMES

        df = self._df.drop(columns=["species"], errors="ignore")

        # Separate core columns from extras
        core_cols = set(CORE_FIELD_NAMES)
        extra_cols = [c for c in df.columns if c not in core_cols]

        extra_fields: Dict[str, list] = {}
        for col in extra_cols:
            extra_fields[col] = df[col].tolist()

        # Build lines_data from core columns only
        core_df = df[[c for c in df.columns if c in core_cols]]
        lines_data = core_df.to_dict(orient="records")

        # Carry over source_format and partition from the original linelist
        fmt = None
        partition = None
        molar_mass = None
        if self._linelist is not None:
            fmt = getattr(self._linelist, "source_format", None)
            partition = getattr(self._linelist, "partition_function", None)
            molar_mass = getattr(self._linelist, "_molar_mass", None)

        ll = _MLL(
            molecule_id=self._species,
            lines_data=lines_data,
            format=fmt,
            extra_fields=extra_fields if extra_fields else None,
        )
        if partition is not None:
            ll.partition_function = partition
        if molar_mass is not None:
            ll._molar_mass = molar_mass

        return ll

    # ------------------------------------------------------------------
    # Combination / merging
    # ------------------------------------------------------------------

    def append(
        self,
        other: Union["LineListMaker", MoleculeLineList, pd.DataFrame],
        species: Optional[str] = None,
    ) -> "LineListMaker":
        """Append lines from *other* to this maker (in-place).

        Parameters
        ----------
        other : LineListMaker, MoleculeLineList, or pd.DataFrame
            Additional lines to append.
        species : str, optional
            Species label for *other* (only used if *other* lacks one).

        Returns
        -------
        LineListMaker
            ``self`` for chaining.
        """
        if isinstance(other, LineListMaker):
            other_df = other._df
        else:
            other_df = _ensure_dataframe(other, molecule_id=species)

        self._df = pd.concat(
            [self._df, other_df], ignore_index=True
        )
        self._record_filter("append", species=species or "unknown")
        return self

    @classmethod
    def merge(
        cls,
        *makers: Union["LineListMaker", MoleculeLineList, pd.DataFrame],
        species_override: Optional[str] = None,
    ) -> "LineListMaker":
        """Merge multiple sources into a single ``LineListMaker``.

        Parameters
        ----------
        *makers : LineListMaker | MoleculeLineList | pd.DataFrame
            Two or more line-list sources.
        species_override : str, optional
            If given, every row receives this species label.

        Returns
        -------
        LineListMaker
            A new maker containing the concatenated data.
        """
        frames: List[pd.DataFrame] = []
        for src in makers:
            if isinstance(src, LineListMaker):
                frames.append(src._df)
            else:
                frames.append(_ensure_dataframe(src))

        combined = pd.concat(frames, ignore_index=True)
        return cls(combined, species=species_override)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "LineListMaker":
        """Return a deep copy of this maker (filters, data, and all)."""
        new = LineListMaker.__new__(LineListMaker)
        new._linelist = self._linelist
        new._species = self._species
        new._df = self._df.copy()
        new._df_original = self._df_original.copy()
        new._filters = copy.deepcopy(self._filters)
        return new

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _range_mask(
        self,
        column: str,
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> "LineListMaker":
        """Build and apply a range mask without double-recording the filter."""
        if column not in self._df.columns:
            warnings.warn(f"Column {column!r} not in DataFrame — filter skipped.")
            return self
        mask = pd.Series(True, index=self._df.index)
        if min_val is not None:
            mask &= self._df[column] >= min_val
        if max_val is not None:
            mask &= self._df[column] <= max_val
        return self._apply_mask(mask)

    def _replay_filters(self) -> "LineListMaker":
        """Reset to original data and replay all stored filters.

        If any filter fails to replay, the maker is restored to the state it
        had on entry and the error is re-raised.  Leaving an empty filter log
        over the fully unfiltered frame would be far worse than failing: a
        caller that swallowed the error would go on to export or apply the
        entire line list believing it had been filtered.
        """
        saved = list(self._filters)
        df_at_entry = self._df
        self._df = self._df_original.copy()
        self._filters.clear()

        # Map filter names to methods
        _method_map: Dict[str, Callable[..., "LineListMaker"]] = {
            "filter_range": self.filter_range,
            "filter_wavelength": self.filter_wavelength,
            "filter_eup": self.filter_eup,
            "filter_elow": self.filter_elow,
            "filter_astein": self.filter_astein,
            "filter_freq": self.filter_freq,
            "filter_gup": self.filter_gup,
            "filter_glow": self.filter_glow,
            "filter_quantum": self.filter_quantum,
            "filter_quantum_field": self.filter_quantum_field,
            "filter_species": self.filter_species,
            "filter_vib_band": self.filter_vib_band,
            "filter_expression": self.filter_expression,
        }

        try:
            for name, kwargs in saved:
                if name == "filter_species":
                    # filter_species is var-positional but records a 'species'
                    # kwarg, so method(**kwargs) would raise TypeError.
                    self.filter_species(*kwargs.get("species", ()))
                    continue
                method = _method_map.get(name)
                if method is not None:
                    method(**kwargs)
                elif name == "filter_custom":
                    # Custom lambdas are not replayable — warn and skip
                    warnings.warn(
                        f"Cannot replay filter_custom(label={kwargs.get('label', '?')!r}); "
                        "it has been dropped."
                    )
                # Other entries (e.g. "append") are structural, not replayable
        except Exception:
            self._df = df_at_entry
            self._filters = saved
            raise
        return self

    def _prepare_export_df(
        self,
        extended: bool = False,
        extra_columns: Optional[Dict[str, Any]] = None,
        sort_by: str = "lam",
    ) -> pd.DataFrame:
        """Build the DataFrame ready for file export."""
        df = self._df.copy()

        # Ensure species column
        if "species" not in df.columns and self._species:
            df.insert(0, "species", self._species)

        # Choose column set
        target_cols = list(_CSV_EXTENDED_COLUMNS if extended else _CSV_COLUMNS)

        # Add xmin/xmax as NaN if they are missing and extended is requested
        if extended:
            for col in ("xmin", "xmax"):
                if col not in df.columns:
                    df[col] = np.nan

        # Extra user columns
        if extra_columns:
            for col_name, col_val in extra_columns.items():
                df[col_name] = col_val
                if col_name not in target_cols:
                    target_cols.append(col_name)

        # Keep any existing columns not in target_cols at the end
        remaining = [c for c in df.columns if c not in target_cols]
        ordered = [c for c in target_cols if c in df.columns] + remaining
        df = df[ordered]

        # Sort
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)

        return df

