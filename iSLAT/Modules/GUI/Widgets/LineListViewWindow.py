"""
LineListViewWindow - read-only popout table for a molecule's line list.

Opens a Toplevel window displaying the MoleculeLineList as a sortable,
searchable ttk.Treeview table.  Columns are auto-discovered from the
DataFrame returned by ``MoleculeLineList.get_pandas_table()``.

Right-clicking the table adds further columns: any property the line-list
filter window can filter on that is not already a column of the table (line
intensity, optical depth, velocity separation, vibrational band and the parsed
quantum numbers).  :class:`LineListColumnSource` computes them with the same
evaluation context the filter uses, so a column shows exactly the value a
filter condition on that property tests.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

import numpy as np
import pandas as pd

from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    ConditionError,
    MaskContext,
    _field_valid,
    vib_band_label,
)
from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import (
    _DERIVED_KEYS,
    PropertySpec,
    build_property_catalog,
    model_quantity_frame,
    molecule_population_data,
)

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_FMT = "{:.6g}"   # compact scientific/decimal formatting for floats

def _fmt(value) -> str:
    """Format a single cell value for display."""
    try:
        import math
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            return _FLOAT_FMT.format(value)
        if hasattr(value, "item"):          # numpy scalar → Python primitive
            return _fmt(value.item())
    except Exception:
        pass
    return str(value)


def _is_missing(cell: Any) -> bool:
    """True for a displayed cell that carries no value (blank or NaN)."""
    text = str(cell).strip()
    return not text or text.lower() == "nan"


def sort_rows(rows: Sequence[tuple], col_idx: int,
              reverse: bool = False) -> List[tuple]:
    """Sort displayed rows on column *col_idx*.

    Numeric when every present value parses as a number, lexicographic
    otherwise.  Blank and NaN cells always sort last, whatever the direction,
    so a column with unknown values (e.g. an unparsed quantum number) still
    sorts numerically instead of collapsing to string order.
    """
    present = [r for r in rows if not _is_missing(r[col_idx])]
    missing = [r for r in rows if _is_missing(r[col_idx])]
    try:
        present = sorted(present, key=lambda r: float(r[col_idx]),
                         reverse=reverse)
    except (ValueError, TypeError):
        present = sorted(present, key=lambda r: str(r[col_idx]),
                         reverse=reverse)
    return present + missing


# ---------------------------------------------------------------------------
# Extra columns
# ---------------------------------------------------------------------------

#: Filter-catalogue kinds that are computed from a line list rather than read
#: straight off one of its columns.  Every other kind (value ranges, level
#: labels, species) is already a column of the table, so these are the ones
#: the "Add Column" menu offers.
ADDABLE_KINDS: frozenset = frozenset({
    "derived", "velocity_sep", "quantum_field", "vib_band",
})


def infer_species(df: Optional[pd.DataFrame]) -> Optional[str]:
    """The species named by *df*'s ``species`` column, when it names one."""
    if df is None or "species" not in getattr(df, "columns", ()):
        return None
    names = {str(v).strip() for v in df["species"].dropna().unique()}
    names.discard("")
    return names.pop() if len(names) == 1 else None


def _rows_line_up(rows: pd.DataFrame, base: pd.DataFrame) -> bool:
    """True when every row of *rows* is the row of *base* with its index.

    Checks the wavelengths as well as the index labels, so a re-indexed
    subset (whose labels merely happen to exist in *base*) is rejected rather
    than read back against the wrong lines.
    """
    if rows is base:
        return True
    if not rows.index.isin(base.index).all():
        return False
    if "lam" not in rows.columns or "lam" not in base.columns:
        return True
    try:
        theirs = base.loc[rows.index, "lam"]
    except KeyError:
        return False
    ours = pd.to_numeric(rows["lam"], errors="coerce").to_numpy(dtype=float)
    theirs = pd.to_numeric(theirs, errors="coerce").to_numpy(dtype=float)
    return ours.shape == theirs.shape and bool(
        np.allclose(ours, theirs, rtol=1e-9, atol=0.0, equal_nan=True))


class LineListColumnSource:
    """Computes the line-filter properties as extra table columns.

    Parameters
    ----------
    species : str, optional
        Quantum-schema key used to parse the level labels.  When omitted it
        is taken from the table's ``species`` column if that names exactly
        one species.
    derived_source : callable, optional
        ``derived_source(reference)`` returns a DataFrame of model quantities
        (``intens``, ``tau``) indexed like *reference*, or raises
        :class:`ConditionError`.  Without one the model quantities are not
        offered.
    baseline_source : callable, optional
        Zero-argument callable returning the full frame the displayed rows
        were selected from (with their index preserved).  Velocity
        separations and model quantities are evaluated over it, exactly as the
        filter evaluates them, then read back for the displayed rows.
        Defaults to the displayed frame itself.
    """

    def __init__(self, species: Optional[str] = None,
                 derived_source: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 baseline_source: Optional[Callable[[], pd.DataFrame]] = None):
        self._species = species
        self._derived_source = derived_source
        self._baseline_source = baseline_source

    def species_for(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        """The quantum-schema key for *df*."""
        return self._species if self._species is not None else infer_species(df)

    def catalog(self, df: Optional[pd.DataFrame]) -> List[PropertySpec]:
        """Filter properties of *df* that can be added as columns, in menu order."""
        if df is None:
            return []
        derived = _DERIVED_KEYS if self._derived_source is not None else ()
        return [spec for spec in build_property_catalog(
                    df, self.species_for(df), derived=derived)
                if spec.kind in ADDABLE_KINDS]

    def compute(self, df: pd.DataFrame, specs: Sequence[PropertySpec]
                ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Display strings for each of *specs*, one per row of *df*.

        Returns ``(values, errors)``, both keyed by ``spec.key``; a property
        that cannot be evaluated appears in *errors* only.  Unknown values
        (an unparsed quantum number, a line with no vibrational label) are
        shown blank.
        """
        values: Dict[str, List[str]] = {}
        errors: Dict[str, str] = {}
        if df is None or not specs:
            return values, errors
        if len(df.index) == 0:
            return {spec.key: [] for spec in specs}, errors

        species = self.species_for(df)
        row_ctx = MaskContext(species=species)
        resolved: Dict[str, Tuple[pd.DataFrame, MaskContext]] = {}

        def reference() -> Tuple[pd.DataFrame, MaskContext]:
            if "ref" not in resolved:
                ref = self._reference_for(df)
                source = self._derived_source
                ctx = MaskContext(
                    species=species,
                    derived_source=(lambda: source(ref)) if source else None)
                resolved["ref"] = (ref, ctx)
            return resolved["ref"]

        for spec in specs:
            try:
                values[spec.key] = self._column(df, spec, row_ctx, reference)
            except ConditionError as exc:
                errors[spec.key] = exc.message
            except Exception as exc:
                errors[spec.key] = f"Could not compute this column: {exc}"
        return values, errors

    # ------------------------------------------------------------------
    def _reference_for(self, df: pd.DataFrame) -> pd.DataFrame:
        """The frame neighbour separations and model quantities are taken over."""
        if self._baseline_source is None:
            return df
        base = self._baseline_source()
        if base is None:
            raise ConditionError("The full line list is not available.")
        if not _rows_line_up(df, base):
            raise ConditionError(
                "These rows do not line up with the line list they were "
                "selected from; refresh the table.")
        return base

    @staticmethod
    def _column(df: pd.DataFrame, spec: PropertySpec, row_ctx: MaskContext,
                reference: Callable[[], Tuple[pd.DataFrame, MaskContext]]
                ) -> List[str]:
        kind = spec.kind
        if kind in ("derived", "velocity_sep"):
            ref, ctx = reference()
            if kind == "derived":
                full = ctx.derived_array(ref, spec.field)
            else:
                full = ctx.neighbour_separation(ref)
            if ref is not df:
                full = pd.Series(full, index=ref.index).reindex(df.index)
            return [_fmt(float(v)) for v in np.asarray(full, dtype=float)]

        if kind == "quantum_field":
            col = "lev_up" if spec.state == "upper" else "lev_low"
            parsed = row_ctx.parsed_levels(df, col)
            if spec.field not in parsed:
                raise ConditionError(
                    f"Quantum field {spec.field!r} is not defined for species "
                    f"{row_ctx.species!r}.")
            vals = np.asarray(parsed[spec.field])
            valid = _field_valid(vals)
            return [_fmt(v) if ok else ""
                    for v, ok in zip(vals.tolist(), valid)]

        if kind == "vib_band":
            for col in ("lev_up", "lev_low"):
                if col not in df.columns:
                    raise ConditionError(f"This line list has no {col!r} column.")
            return [vib_band_label(up, low)
                    for up, low in zip(df["lev_up"], df["lev_low"])]

        raise ConditionError(f"{spec.label!r} cannot be shown as a column.")


def molecule_column_source(mol_obj: "Molecule") -> LineListColumnSource:
    """Column source for a molecule's own line list.

    Model quantities come from the molecule's current intensity calculation
    and are refused unless they line up with the displayed rows.
    """
    species = getattr(getattr(mol_obj, "line_list", None), "molecule_id", None)

    def derived(reference: pd.DataFrame) -> pd.DataFrame:
        data, error = molecule_population_data(mol_obj, reference)
        if data is None:
            raise ConditionError(
                error or "No model data is available for this molecule.")
        return model_quantity_frame(data, reference.index)

    return LineListColumnSource(species=species, derived_source=derived)

# ---------------------------------------------------------------------------
# LineListViewWindow
# ---------------------------------------------------------------------------
class LineListViewWindow(tk.Toplevel):
    """Read-only popout table showing a molecule's line list DataFrame.

    Parameters
    ----------
    parent : tk.Widget
        Parent window (typically the ControlPanel or root).
    mol_obj : Molecule, optional
        The molecule whose ``line_list`` will be displayed.  May be ``None``
        when *data_provider* is given.
    data_field : optional
        iSLAT data field for status messages (``insert_text`` interface).
    data_provider : callable, optional
        Zero-argument callable returning the DataFrame to display.  When given
        it replaces the molecule lookup entirely.  Storing a *callable* rather
        than a frame is what makes :meth:`refresh` live: it is re-invoked on
        every refresh, so the window always shows the caller's current data.
    title : str, optional
        Window title override.
    export_basename : str, optional
        Default file name (without extension) offered by "Export CSV".
    column_source : LineListColumnSource or callable, optional
        Where the right-click "Add Column" entries are computed from.  A
        zero-argument callable is re-invoked on every refresh, so the source
        always reflects the caller's current state.  Defaults to
        :func:`molecule_column_source` for a molecule, and otherwise to a
        source without model quantities whose species is read from the
        table's ``species`` column.
    """
    def __init__(self, parent, mol_obj: "Molecule" = None, data_field=None, *,
                 data_provider=None, title: str = None,
                 export_basename: str = None, column_source=None):
        super().__init__(parent)
        self.mol_obj = mol_obj
        self.data_field = data_field
        self._data_provider = data_provider
        self._export_basename = export_basename
        self._column_source = column_source

        mol_name = getattr(mol_obj, "name", "Molecule")
        self.title(title or f"Line List: {mol_name}")
        self.resizable(True, True)
        self._constrain_to_screen()

        self._all_rows: list[tuple] = []    # full unfiltered row data
        self._columns: list[str] = []
        self._sort_col: str | None = None
        self._sort_reverse: bool = False

        # Added columns.  _df and _source are the frame and column source of
        # the last load, kept so a column can be added without a refetch.
        self._df: Optional[pd.DataFrame] = None
        self._source: Optional[LineListColumnSource] = None
        self._addable: List[PropertySpec] = []
        self._added: List[PropertySpec] = []
        self._added_titles: Dict[str, str] = {}    # spec.key -> heading
        self._column_errors: Dict[str, str] = {}   # spec.key -> reason
        self._column_menu: Optional[tk.Menu] = None
        self._menu_vars: List[tk.BooleanVar] = []

        self._build_ui()
        self._load_data()

    @classmethod
    def from_dataframe(cls, parent, source, *, title: str = "Line List",
                       data_field=None, export_basename: str = "line_list",
                       column_source=None):
        """Open a viewer on a DataFrame, or on a callable returning one.

        Passing a *callable* is what makes :meth:`refresh` live: it is
        re-invoked on every refresh, so the window tracks the caller's current
        data instead of a snapshot taken when it opened.
        """
        provider = source if callable(source) else (lambda: source)
        return cls(parent, None, data_field, data_provider=provider,
                   title=title, export_basename=export_basename,
                   column_source=column_source)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _constrain_to_screen(self) -> None:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1100, int(sw * 0.80))
        h = min(700, int(sh * 0.75))
        x = max((sw - w) // 2, 0)
        y = max((sh - h) // 2, 0)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(500, 300)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Top bar: search ────────────────────────────────────────────
        top = ttk.Frame(self, padding=(6, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Search:").grid(row=0, column=0, padx=(0, 4))
        self._search_var = tk.StringVar()
        # Keep the handle: Tcl holds a callback referencing this window until
        # the trace is removed, which would otherwise pin the window, its data
        # provider, and everything that provider closes over.
        self._search_trace = self._search_var.trace_add(
            "write", self._on_search_change)
        search_entry = ttk.Entry(top, textvariable=self._search_var)
        search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))

        try:
            from iSLAT.Modules.GUI.Tooltips import CreateToolTip
            CreateToolTip(
                search_entry,
                "Filter rows by typing any text.\n"
                "All columns are searched (case-insensitive).",
            )
        except Exception:
            pass

        clear_btn = ttk.Button(top, text="x", width=3, command=self._clear_search)
        clear_btn.grid(row=0, column=2, padx=(0, 4))

        refresh_btn = ttk.Button(top, text="Refresh", command=self._refresh)
        refresh_btn.grid(row=0, column=3, padx=(0, 8))
        try:
            from iSLAT.Modules.GUI.Tooltips import CreateToolTip
            CreateToolTip(
                refresh_btn,
                "Reload the line list from the molecule.\n"
                "Useful if the line list has been filtered or changed\n"
                "since this window was opened.",
            )
        except Exception:
            pass

        self._export_btn = ttk.Button(top, text="Export CSV…", command=self._export_csv)
        self._export_btn.grid(row=0, column=4)
        try:
            from iSLAT.Modules.GUI.Tooltips import CreateToolTip
            self._export_tooltip = CreateToolTip(
                self._export_btn,
                "Export selected rows to CSV.\n"
                "If no rows are selected, all visible rows are exported.\n"
                "Ctrl/Cmd+click or Shift+click to select multiple rows.\n"
                "Keyboard shortcut: Ctrl+S / Cmd+S.",
            )
        except Exception:
            pass

        # ── Treeview ───────────────────────────────────────────────────
        table_frame = ttk.Frame(self)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 2))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(table_frame, show="headings", selectmode="extended")
        self._tree.grid(row=0, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_selection_change)
        for sequence in self._context_menu_sequences():
            self._tree.bind(sequence, self._show_column_menu)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self._tree.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Loading…")
        status = ttk.Label(self, textvariable=self._status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 4))

        # Ctrl+S / Cmd+S → export
        self.bind("<Control-s>", self._on_export_shortcut)
        self.bind("<Command-s>", self._on_export_shortcut)

    def _on_export_shortcut(self, _event=None) -> str:
        """Export, then stop the key event so the global Save Parameters shortcut doesn't also fire."""
        self._export_csv()
        return "break"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-fetch from the provider (or the molecule) and redraw.

        Safe to call from another window driving this view: it is a no-op once
        this window has been closed.
        """
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self._refresh()

    def _refresh(self) -> None:
        """Reload the data, preserving the current search and sort order."""
        current_search = self._search_var.get()
        # Sort is preserved: an externally driven refresh must not silently
        # undo the column the user clicked while they carry on typing.
        sort_col, sort_reverse = self._sort_col, self._sort_reverse
        self._all_rows = []
        self._columns = []
        self._sort_col = None
        self._sort_reverse = False
        self._tree.delete(*self._tree.get_children())
        self._status_var.set("Refreshing…")
        self.update_idletasks()
        self._load_data()
        # Re-apply any active search BEFORE re-sorting: _on_search_change
        # repopulates from the unsorted model, which would undo the sort if it
        # ran afterwards.
        if current_search:
            self._search_var.set(current_search)
        if sort_col and sort_col in self._columns:
            # _sort_by flips the direction when re-sorting the same column,
            # so pre-invert to land back on the order the user had.
            self._sort_col = sort_col
            self._sort_reverse = not sort_reverse
            self._sort_by(sort_col)

    def destroy(self) -> None:
        """Remove the search trace and drop the data sources before closing.

        A ``data_provider`` (or ``column_source``) is typically a bound method
        of the window that opened this one, so holding it after close would
        keep that window, its LineListMaker and the whole unfiltered frame
        alive for the life of the Tk application.
        """
        handle = getattr(self, "_search_trace", None)
        if handle is not None:
            try:
                self._search_var.trace_remove("write", handle)
            except tk.TclError:
                pass
            self._search_trace = None
        self._destroy_column_menu()
        self._data_provider = None
        self._column_source = None
        self._source = None
        self._df = None
        self.mol_obj = None
        self._all_rows = []
        super().destroy()

    def _fetch_dataframe(self):
        """Return ``(df, error_message)``.  A data provider, when set, wins."""
        if self._data_provider is not None:
            try:
                return self._data_provider(), None
            except Exception as exc:
                return None, f"Error fetching table: {exc}"
        try:
            ll = self.mol_obj.line_list
        except Exception as exc:
            return None, f"Error accessing line list: {exc}"
        if ll is None:
            return None, "No line list loaded for this molecule."
        try:
            return ll.get_pandas_table(), None
        except Exception as exc:
            return None, f"Error building table: {exc}"

    def _load_data(self) -> None:
        """Fetch the DataFrame and populate the Treeview."""
        df, error = self._fetch_dataframe()
        if error:
            self._df = None
            self._addable = []
            self._column_errors = {}
            self._status_var.set(error)
            return

        self._df = df
        self._source = self._resolve_column_source()
        self._addable = self._safe_catalog(df)
        # Keep the columns the user added, rebound to this load's catalogue;
        # one the new data no longer offers (e.g. another species) is dropped.
        offered = {spec.key: spec for spec in self._addable}
        self._added = [offered[s.key] for s in self._added if s.key in offered]
        self._column_errors = {}

        if df is None or df.empty:
            self._columns = []
            self._tree["columns"] = ()
            self._tree.delete(*self._tree.get_children())
            self._status_var.set("Line list is empty.")
            return

        base_columns = list(df.columns)
        self._added_titles = {}
        for spec in self._added:
            self._added_titles[spec.key] = self._unique_title(
                spec.label, base_columns + list(self._added_titles.values()))
        extra = self._compute_added(df, self._added)
        self._columns = base_columns + [self._added_titles[s.key]
                                        for s in self._added]
        self._setup_columns()

        # Convert all rows to formatted strings up-front for fast filtering
        rows = (tuple(_fmt(v) for v in row)
                for row in df.itertuples(index=False, name=None))
        if extra:
            self._all_rows = [row + added
                              for row, added in zip(rows, zip(*extra))]
        else:
            self._all_rows = list(rows)
        self._populate_tree(self._all_rows)

    def _setup_columns(self) -> None:
        """Configure Treeview columns and headings.

        Columns that were already on screen keep their width, so adding or
        removing a column (or a live refresh) does not undo a resize.
        """
        current = set(self._tree["columns"])
        widths = {col: self._tree.column(col, "width")
                  for col in self._columns if col in current}
        self._tree["columns"] = self._columns
        for col in self._columns:
            self._tree.heading(
                col,
                text=col,
                command=lambda c=col: self._sort_by(c),
            )
            # Estimate a reasonable minimum column width
            width = widths.get(col) or max(70, min(160, len(str(col)) * 9 + 20))
            self._tree.column(col, width=width, minwidth=50, stretch=True)

    def _populate_tree(self, rows: list[tuple]) -> None:
        """Clear and refill the Treeview with *rows*."""
        self._tree.delete(*self._tree.get_children())
        for row in rows:
            self._tree.insert("", "end", values=row)
        self._update_status()

    def _update_status(self) -> None:
        """Refresh the status bar to reflect visible and selected counts."""
        n_shown = len(self._tree.get_children())
        n_total = len(self._all_rows)
        n_sel = len(self._tree.selection())
        if n_sel:
            text = (f"{n_sel:,} selected  ·  "
                    f"{n_shown:,} visible  ·  {n_total:,} total")
        elif n_shown == n_total:
            text = (f"{n_total:,} lines  ·  Ctrl/Cmd+click or Shift+click to "
                    "select  ·  Right-click to add columns")
        else:
            text = (f"{n_shown:,} of {n_total:,} lines (filtered)  ·  "
                    "Ctrl/Cmd+click or Shift+click to select")
        self._status_var.set(text + self._column_error_note())

    def _column_error_note(self) -> str:
        """Status-bar suffix naming the added columns that could not be filled."""
        notes = [f"{self._added_titles.get(key, key)} unavailable: {reason}"
                 for key, reason in self._column_errors.items()]
        return ("  ·  " + "; ".join(notes)) if notes else ""

    def _on_selection_change(self, _event=None) -> None:
        """Called whenever the Treeview selection changes."""
        self._update_status()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _on_search_change(self, *_) -> None:
        query = self._search_var.get().lower()
        if not query:
            self._populate_tree(self._all_rows)
            return
        filtered = [
            row for row in self._all_rows
            if any(query in cell.lower() for cell in row)
        ]
        self._populate_tree(filtered)

    def _clear_search(self) -> None:
        self._search_var.set("")

    # ------------------------------------------------------------------
    # Column sorting
    # ------------------------------------------------------------------
    def _sort_by(self, col: str) -> None:
        """Sort Treeview rows by the clicked column header."""
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = False

        col_idx = self._columns.index(col)
        current = [self._tree.item(iid, "values")
                   for iid in self._tree.get_children()]
        sorted_rows = sort_rows(current, col_idx, self._sort_reverse)

        self._tree.delete(*self._tree.get_children())
        for row in sorted_rows:
            self._tree.insert("", "end", values=row)

        # Update heading indicator
        for c in self._columns:
            self._tree.heading(c, text=c)
        arrow = " ▲" if not self._sort_reverse else " ▼"
        self._tree.heading(col, text=col + arrow)

    # ------------------------------------------------------------------
    # Added columns
    # ------------------------------------------------------------------
    def _context_menu_sequences(self) -> Tuple[str, ...]:
        """Right-click event sequences for this platform.

        Ctrl+click is only a right-click on macOS; elsewhere it is the
        Treeview's toggle-selection gesture and must stay that way.
        """
        try:
            aqua = self.tk.call("tk", "windowingsystem") == "aqua"
        except tk.TclError:
            aqua = False
        if aqua:
            return ("<Button-2>", "<Button-3>", "<Control-Button-1>")
        return ("<Button-3>",)

    def _resolve_column_source(self) -> LineListColumnSource:
        """The column source for this load (see ``column_source``)."""
        source = self._column_source
        try:
            if source is None:
                if self._data_provider is None and self.mol_obj is not None:
                    return molecule_column_source(self.mol_obj)
                return LineListColumnSource()
            if isinstance(source, LineListColumnSource):
                return source
            resolved = source()
            if resolved is not None:
                return resolved
        except Exception as exc:
            print(f"LineListViewWindow: column source unavailable - {exc}")
        return LineListColumnSource()

    def _safe_catalog(self, df) -> List[PropertySpec]:
        try:
            return self._source.catalog(df)
        except Exception as exc:
            print(f"LineListViewWindow: cannot list extra columns - {exc}")
            return []

    @staticmethod
    def _unique_title(label: str, taken: Sequence[Any]) -> str:
        """*label*, suffixed if a column of that name is already shown."""
        taken = {str(t) for t in taken}
        title, n = label, 2
        while title in taken:
            title = f"{label} ({n})"
            n += 1
        return title

    def _compute_added(self, df: pd.DataFrame,
                       specs: Sequence[PropertySpec]) -> List[List[str]]:
        """Display values for *specs*, blank (with a status note) on failure."""
        if not specs:
            return []
        try:
            values, errors = self._source.compute(df, specs)
        except Exception as exc:
            values, errors = {}, {spec.key: str(exc) for spec in specs}
        self._column_errors.update(errors)
        blank = [""] * len(df)
        return [values.get(spec.key, blank) for spec in specs]

    def _is_added(self, spec: PropertySpec) -> bool:
        return any(s.key == spec.key for s in self._added)

    def _column_at(self, event) -> Optional[str]:
        """The column id under *event*, or ``None`` outside every column."""
        ref = self._tree.identify_column(event.x)
        if not ref or ref == "#0":
            return None
        try:
            return self._tree.column(ref, "id")
        except tk.TclError:
            return None

    def _show_column_menu(self, event):
        """Post the add/remove-column menu at the pointer."""
        menu = self._build_column_menu(self._column_at(event))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _destroy_column_menu(self) -> None:
        menu, self._column_menu = self._column_menu, None
        self._menu_vars = []
        if menu is not None:
            try:
                menu.destroy()
            except tk.TclError:
                pass

    def _build_column_menu(self, column: Optional[str] = None) -> tk.Menu:
        """Build the right-click menu; *column* is the column clicked on."""
        self._destroy_column_menu()
        menu = tk.Menu(self, tearoff=0)
        self._column_menu = menu

        add_menu = tk.Menu(menu, tearoff=0)
        general = [s for s in self._addable if s.kind != "quantum_field"]
        quantum = {state: [s for s in self._addable
                           if s.kind == "quantum_field" and s.state == state]
                   for state in ("upper", "lower")}
        for spec in general:
            self._add_menu_check(add_menu, spec, spec.label)
        separated = not general
        for state, label in (("upper", "Upper-Level Quantum Numbers"),
                             ("lower", "Lower-Level Quantum Numbers")):
            if not quantum[state]:
                continue
            if not separated:
                add_menu.add_separator()
                separated = True
            sub = tk.Menu(add_menu, tearoff=0)
            for spec in quantum[state]:
                self._add_menu_check(sub, spec, spec.field or spec.label)
            add_menu.add_cascade(label=label, menu=sub)
        if not self._addable:
            add_menu.add_command(label="No additional columns available",
                                 state="disabled")
        menu.add_cascade(label="Add Column", menu=add_menu)

        clicked = next((s for s in self._added
                        if self._added_titles.get(s.key) == column), None)
        if clicked is not None:
            menu.add_command(
                label=f"Remove Column \u201c{self._added_titles[clicked.key]}\u201d",
                command=lambda s=clicked: self._remove_column(s))
        menu.add_command(
            label="Remove All Added Columns",
            command=self._remove_all_columns,
            state="normal" if self._added else "disabled")
        return menu

    def _add_menu_check(self, menu: tk.Menu, spec: PropertySpec,
                        label: str) -> None:
        var = tk.BooleanVar(master=self, value=self._is_added(spec))
        self._menu_vars.append(var)   # a menu holds no reference of its own
        menu.add_checkbutton(label=label, variable=var,
                             command=lambda s=spec: self._toggle_column(s))

    def _toggle_column(self, spec: PropertySpec) -> None:
        if self._is_added(spec):
            self._remove_column(spec)
        else:
            self._add_column(spec)

    def add_column(self, key: str) -> bool:
        """Add the column whose catalogue key is *key*; True when shown."""
        spec = next((s for s in self._addable if s.key == key), None)
        if spec is None:
            return False
        if not self._is_added(spec):
            self._add_column(spec)
        return self._is_added(spec)

    def _add_column(self, spec: PropertySpec) -> None:
        """Compute *spec* for the loaded rows and append it as a column."""
        df = self._df
        if df is None or self._source is None:
            return
        if df.empty:
            self._added.append(spec)
            return
        try:
            values, errors = self._source.compute(df, [spec])
        except Exception as exc:
            values, errors = {}, {spec.key: str(exc)}
        if spec.key in errors or spec.key not in values:
            messagebox.showwarning(
                "Cannot Add Column",
                f"{spec.label}:\n"
                f"{errors.get(spec.key, 'No values were produced.')}",
                parent=self)
            return
        title = self._unique_title(spec.label, self._columns)
        self._added.append(spec)
        self._added_titles[spec.key] = title
        self._columns.append(title)
        self._all_rows = [row + (value,) for row, value
                          in zip(self._all_rows, values[spec.key])]
        self._rebuild_table()
        # New columns land on the far right; scroll once the width is known.
        self.update_idletasks()
        self._tree.xview_moveto(1.0)

    def _remove_column(self, spec: PropertySpec) -> None:
        self._remove_columns({spec.key})

    def _remove_all_columns(self) -> None:
        self._remove_columns({s.key for s in self._added})

    def _remove_columns(self, keys: set) -> None:
        if not keys:
            return
        titles = {self._added_titles[k] for k in keys if k in self._added_titles}
        self._added = [s for s in self._added if s.key not in keys]
        for key in keys:
            self._added_titles.pop(key, None)
            self._column_errors.pop(key, None)
        keep = [i for i, col in enumerate(self._columns) if col not in titles]
        if len(keep) == len(self._columns):
            return   # none of them were on screen (e.g. an empty table)
        self._columns = [self._columns[i] for i in keep]
        self._all_rows = [tuple(row[i] for i in keep) for row in self._all_rows]
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        """Redraw after the columns changed, keeping the search and the sort."""
        sort_col, sort_reverse = self._sort_col, self._sort_reverse
        self._tree.delete(*self._tree.get_children())
        self._setup_columns()
        self._sort_col, self._sort_reverse = None, False
        self._on_search_change()
        if sort_col in self._columns:
            # _sort_by flips the direction when re-sorting the same column.
            self._sort_col = sort_col
            self._sort_reverse = not sort_reverse
            self._sort_by(sort_col)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_csv(self) -> None:
        """Save selected rows (or all visible rows if none selected) to a CSV file."""
        selected = self._tree.selection()
        if selected:
            rows = [self._tree.item(iid, "values") for iid in selected]
        else:
            rows = [self._tree.item(iid, "values") for iid in self._tree.get_children()]
        if not rows:
            return

        mol_name = getattr(self.mol_obj, "name", "molecule")
        basename = self._export_basename or f"{mol_name}_line_list"
        filepath = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=basename,
        )
        if not filepath:
            return

        try:
            import csv
            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(self._columns)
                writer.writerows(rows)
            msg = f"Line list exported to {filepath}"
            if self.data_field is not None:
                self.data_field.insert_text(msg, console_print=True)
            else:
                print(msg)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)