"""
FilteredPopulationDiagramWindow - live population diagram of a filtered line list.

Opens a Toplevel showing a population diagram of whichever subset of a
molecule's lines the :class:`~iSLAT.Modules.GUI.Widgets.LineListFilterWindow`
currently selects, and redraws as the filter changes.

The diagram is rendered by the *real*
:class:`~iSLAT.Modules.Plotting.PopulationDiagramPlot.PopulationDiagramPlot`,
into a Figure this window owns.  That is what makes "the same axis and
color/shape settings as the existing population diagram view" a fact rather
than a promise: the axis labels, limit padding, marker sizes, alpha, colour
norms, legends, and theming all come from the same code, and the right-click
Plot Settings dialog is the same
:class:`~iSLAT.Modules.Plotting.PopulationDiagramContextMixin.PopulationDiagramContextMixin`
the main view uses.  The six pieces of *state* the user can change there are
copied from the main diagram when this window opens.

Performance note: the physics is computed once, for the molecule's whole line
list, and each redraw merely indexes those arrays with the filter's boolean
mask.  Nothing is recalculated per keystroke.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from iSLAT.Modules.Plotting.PopulationDiagramContextMixin import (
    PopulationDiagramContextMixin,
)

# ---------------------------------------------------------------------------
# Pure helpers (no Tk, no matplotlib display)
# ---------------------------------------------------------------------------

#: Keys in a population-diagram data dict that are per-line arrays and must be
#: subset together.  Anything else (e.g. ``beam_s``) is a scalar and is copied.
_PER_LINE_KEYS: tuple = (
    "eu", "rd_yax", "wavelength", "intens", "a_stein", "g_up", "g_low",
    "lev_up", "lev_low", "e_low", "tau", "valid_mask",
    "fwhm_instrumental_kms", "fwhm_convolved_kms", "freq", "velocity_sep_kms",
)

#: The settings a user can change through the population-diagram context menu.
#: Everything else about the diagram's appearance lives in shared *code* and
#: therefore cannot drift between the two windows.
_STYLE_FIELDS: tuple = (
    "_x_prop", "_y_prop", "_x_log", "_y_log", "_x_lim", "_y_lim",
    "_color_mapping", "_shape_mapping", "_marker_size",
)

def subset_population_data(data: Optional[Dict[str, Any]],
                           mask: Optional[Sequence[bool]]) -> Optional[Dict[str, Any]]:
    """Return *data* restricted to the rows selected by *mask*.

    ``data`` is a dict as returned by
    :meth:`~iSLAT.Modules.DataTypes.Intensity.Intensity.get_population_diagram_data`.
    Per-line arrays are indexed; scalars are passed through untouched.

    Passing ``None`` for *mask* returns the data unchanged, so "no filter" and
    "everything selected" render identically.

    Raises
    ------
    ValueError
        If *mask* length does not match the per-line arrays - indexing with a
        stale mask would silently plot the wrong lines.
    """
    if data is None:
        return None
    if mask is None:
        return data

    mask_arr = np.asarray(mask, dtype=bool)
    length = _population_data_length(data)
    if length is not None and mask_arr.shape[0] != length:
        raise ValueError(
            f"Mask length {mask_arr.shape[0]} does not match the "
            f"population-diagram arrays ({length})."
        )

    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _PER_LINE_KEYS and _is_per_line(value, length):
            out[key] = np.asarray(value)[mask_arr]
        else:
            out[key] = value
    out["valid_mask"] = _recompute_valid_mask(out)
    return out

def _recompute_valid_mask(data: Dict[str, Any]) -> Any:
    """Rebuild ``valid_mask`` from the retained lines' own fluxes.

    Intensity defines it as "flux above 1% of the maximum flux", and the
    renderer uses it to pick the points that set the axis limits.  Slicing the
    original mask would keep that threshold pinned to the *unfiltered* list's
    brightest line, so filtering down to a set of faint lines would mark every
    one of them invalid and draw an empty diagram.  Recomputing over the subset
    reproduces exactly what Intensity would have returned for these lines.
    """
    intens = data.get("intens")
    if intens is None or np.ndim(intens) < 1:
        return data.get("valid_mask")
    values = np.asarray(intens, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    positive = np.isfinite(values) & (values > 0)
    if not positive.any():
        return np.zeros(values.shape, dtype=bool)
    threshold = np.nanmax(values[positive]) / 100.0
    return values > threshold

def _population_data_length(data: Dict[str, Any]) -> Optional[int]:
    """Number of lines represented by a population-diagram data dict."""
    for key in _PER_LINE_KEYS:
        value = data.get(key)
        if value is not None and np.ndim(value) >= 1:
            return int(np.shape(value)[0])
    return None

def _is_per_line(value: Any, length: Optional[int]) -> bool:
    """True when *value* is an array of one entry per line."""
    if value is None or np.ndim(value) < 1:
        return False
    return length is None or int(np.shape(value)[0]) == length

def capture_pdp_style(pdp: Any) -> Dict[str, Any]:
    """Snapshot the user-adjustable settings of a PopulationDiagramPlot.

    Read with ``getattr`` defaults throughout: ``_x_lim``/``_y_lim`` only come
    into existence once ``set_axes`` has run, so a plot the user has never
    configured has no such attributes.
    """
    if pdp is None:
        return {}
    return {field: getattr(pdp, field, None) for field in _STYLE_FIELDS}

def apply_pdp_style(pdp: Any, style: Optional[Dict[str, Any]], *,
                    regenerate: bool = False) -> None:
    """Apply a :func:`capture_pdp_style` snapshot to *pdp*.

    Routed through the public setters rather than by assigning the private
    attributes, so any validation or derived state in those setters runs.
    """
    if pdp is None or not style:
        return

    x_prop = style.get("_x_prop") or "eu"
    y_prop = style.get("_y_prop") or "rd_yax"
    pdp.set_axes(
        x_prop, y_prop,
        bool(style.get("_x_log")), bool(style.get("_y_log")),
        style.get("_x_lim"), style.get("_y_lim"),
        regenerate=False,
    )

    color = style.get("_color_mapping")
    if color and color.get("prop"):
        # Forward every keyword color_by accepts: dropping pmin/pmax or
        # log_scale would give the two diagrams different colour norms while
        # both claimed to be coloured "the same way".  cmap is only passed
        # when set, so None cannot override color_by's own default.
        kwargs = {"vmin": color.get("vmin"), "vmax": color.get("vmax"),
                  "pmin": color.get("pmin"), "pmax": color.get("pmax"),
                  "log_scale": bool(color.get("log_scale", False))}
        if color.get("cmap"):
            kwargs["cmap"] = color["cmap"]
        try:
            pdp.color_by(color["prop"], regenerate=False, **kwargs)
        except Exception:
            pdp.clear_color_mapping(regenerate=False)
    else:
        pdp.clear_color_mapping(regenerate=False)

    shape = style.get("_shape_mapping")
    if shape and shape.get("prop"):
        try:
            pdp.shape_by(shape["prop"], n_bins=int(shape.get("n_bins", 5)),
                         regenerate=False)
        except Exception:
            pdp.clear_shape_mapping(regenerate=False)
    else:
        pdp.clear_shape_mapping(regenerate=False)

    size = style.get("_marker_size")
    if size is None:
        pdp.clear_marker_size(regenerate=False)
    else:
        pdp.set_marker_size(float(size), regenerate=False)

    if regenerate:
        pdp.generate_plot()

def find_main_population_plot(islat: Any) -> Any:
    """Locate the live PopulationDiagramPlot driving the main GUI, or None.

    Checks the dedicated Population Diagram view first, then the population
    panel of the three-panel view.  Both are ``None`` until their view has
    been activated at least once, which is why this returns ``None`` freely -
    the caller then simply keeps the renderer's own defaults.
    """
    plot_manager = getattr(getattr(islat, "GUI", None), "plot", None)
    if plot_manager is None:
        return None

    three_panel = getattr(plot_manager, "_three_panel_view", None)
    standalone = getattr(plot_manager, "_population_diagram_view", None)

    def _from_three_panel():
        for owner in (three_panel, getattr(three_panel, "_grid", None),
                      getattr(three_panel, "grid", None)):
            found = getattr(owner, "pop_diagram_panel", None)
            if found is not None:
                return found
        return None

    # Prefer whichever diagram the user is actually looking at; both views can
    # hold a live plot at once, and they carry independent settings.
    active = getattr(plot_manager, "active_view", None)
    if active is not None:
        if active is three_panel:
            pdp = _from_three_panel()
            if pdp is not None:
                return pdp
        elif active is standalone:
            pdp = getattr(standalone, "_plot", None)
            if pdp is not None:
                return pdp

    pdp = getattr(standalone, "_plot", None)
    if pdp is not None:
        return pdp
    return _from_three_panel()

class _FrozenPopulationSource:
    """Duck-typed stand-in for :class:`Intensity` holding precomputed arrays.

    ``PopulationDiagramPlot``'s bare-intensity branch calls exactly one method
    on the object it is handed::

        data = intensity_obj.get_population_diagram_data(radius, distance)

    Nothing inspects the type, so serving a precomputed dict here swaps *what*
    is drawn without touching any of the plot's settings - and without
    recomputing the intensity physics on every filter keystroke.
    """
    __slots__ = ("_data",)
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data = data

    def set_data(self, data: Optional[Dict[str, Any]]) -> None:
        """Replace the served payload."""
        self._data = data

    def get_population_diagram_data(self, radius: float, distance: float,
                                    **_kwargs) -> Optional[Dict[str, Any]]:
        """Return the precomputed data dict (signature matches Intensity)."""
        return self._data

# ---------------------------------------------------------------------------
# FilteredPopulationDiagramWindow
# ---------------------------------------------------------------------------
class FilteredPopulationDiagramWindow(tk.Toplevel, PopulationDiagramContextMixin):
    """Live population diagram of a filtered subset of a molecule's lines.

    Parameters
    ----------
    parent : tk.Widget
        Owning window (the filter window).
    mol_obj : Molecule
        The molecule the lines belong to; supplies colour, radius and distance.
    islat : optional
        iSLAT instance, used to find the main diagram's settings and theme.
    on_close : callable, optional
        Invoked when this window closes, so the owner can deregister it.
    """
    def __init__(self, parent, mol_obj, islat=None, on_close: Callable = None):
        super().__init__(parent)
        self.mol_obj = mol_obj
        self._islat = islat
        self._on_close_cb = on_close

        self._fig = None
        self._ax = None
        self._canvas = None
        self._toolbar = None
        self._pdp = None
        self._right_click_cid = None
        self._source = _FrozenPopulationSource(None)
        self._full_data: Optional[Dict[str, Any]] = None
        self._base_length: Optional[int] = None
        self._pending_mask: Optional[np.ndarray] = None
        self._redraw_job: Optional[str] = None
        self._sticky_message: Optional[str] = None

        mol_name = getattr(mol_obj, "name", "Molecule")
        self.title(self._window_title(mol_name))
        self.resizable(True, True)

        self._status_var = tk.StringVar(value="No data yet.")
        self._build_ui()
        self._constrain_to_screen()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _window_title(self, mol_name: str) -> str:
        return f"Population Diagram (filtered): {mol_name}"

    def _create_population_plot(self, color: Optional[str]) -> Any:
        """Build the :class:`PopulationDiagramPlot` drawn into this window's figure.

        Served from the frozen source so filter changes only re-index arrays.
        Pass BOTH fig= and ax=: without fig= the renderer skips
        apply_theme_to_figure() and the popout would not match the GUI.
        """
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        mol = self.mol_obj
        return PopulationDiagramPlot(
            intensity=self._source,
            name=f"{getattr(mol, 'name', 'Molecule')} (filtered)",
            color=color,
            radius=getattr(mol, "radius", 1.0),
            distance=getattr(mol, "distance", 160.0),
            ax=self._ax,
            fig=self._fig,
            theme=self._theme() or None,
        )

    def _constrain_to_screen(self) -> None:
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(900, int(sw * 0.55))
        h = min(760, int(sh * 0.80))
        self.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 2, 0)}")
        self.minsize(480, 400)

    def _theme(self) -> Dict[str, Any]:
        """The GUI's live theme dict, or an empty dict before the GUI exists."""
        plot_manager = getattr(getattr(self._islat, "GUI", None), "plot", None)
        theme = getattr(plot_manager, "theme", None)
        if theme is None:
            theme = getattr(getattr(self._islat, "GUI", None), "theme", None)
        return theme if isinstance(theme, dict) else {}

    def _build_ui(self) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )
        from iSLAT.Modules.Plotting.BasePlot import BasePlot

        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(7, 6), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)

        mol = self.mol_obj
        try:
            color = BasePlot.get_molecule_color(mol)
        except Exception:
            color = None

        self._pdp = self._create_population_plot(color)
        # Inherit whatever the user has configured on the main diagram, so the
        # two windows agree on axes, colour mapping, shapes and marker size.
        apply_pdp_style(self._pdp, capture_pdp_style(
            find_main_population_plot(self._islat)), regenerate=False)

        self._canvas = FigureCanvasTkAgg(self._fig, master=outer)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self._right_click_cid = self._canvas.mpl_connect(
            "button_press_event", self._on_canvas_button_press)

        toolbar_frame = ttk.Frame(outer)
        toolbar_frame.pack(fill=tk.X)
        try:
            self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
            self._toolbar.update()
        except Exception:
            self._toolbar = None

        bar = ttk.Frame(outer, padding=(6, 2))
        bar.pack(fill=tk.X)
        ttk.Label(bar, textvariable=self._status_var, anchor="w").pack(
            side="left", fill="x", expand=True)
        sync_btn = ttk.Button(bar, text="Sync Settings",
                              command=self._sync_from_main)
        sync_btn.pack(side="right", padx=4)
        try:
            from iSLAT.Modules.GUI.Tooltips import CreateToolTip
            CreateToolTip(sync_btn,
                          "Re-copy the axis, colour and shape settings from the "
                          "main population diagram.")
            CreateToolTip(widget,
                          "Right-click for Plot Settings (axes, color, markers)")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context menu (same dialogs as the main view)
    # ------------------------------------------------------------------
    def _on_canvas_button_press(self, event) -> None:
        if getattr(event, "button", None) != 3:
            return
        widget = self._canvas.get_tk_widget()
        menu = self.build_context_menu(event, widget)
        if menu is None:
            return
        try:
            menu.tk_popup(widget.winfo_pointerx(), widget.winfo_pointery())
        finally:
            menu.grab_release()

    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Right-click menu: the shared Plot Settings dialog.

        Deliberately *not* the mixin's full menu.  That menu also offers
        "Show All Active Molecules" / "Show Active Molecule Only", which call
        ``set_molecules`` / ``set_molecule`` on the plot.  Here that would
        discard the filtered subset this window exists to show, and would
        register MoleculeDict callbacks on a popout - the leak class that was
        fixed for the main view.  The dialog that carries the settings is the
        mixin's own method, so it stays identical to the main diagram.
        """
        if self._pdp is None:
            return None
        menu = tk.Menu(canvas_widget, tearoff=0)
        has_mapping = getattr(self._pdp, "_color_mapping", None) is not None
        log_scale = bool(has_mapping and
                         self._pdp._color_mapping.get("log_scale", False))

        menu.add_command(
            label="Plot Settings…",
            command=lambda: self._open_plot_settings_dialog(
                self._pdp, canvas_widget, self._draw_idle))
        menu.add_command(label="Clear Color Mapping",
                         command=self._clear_color_mapping)
        menu.add_command(
            label=("✓ " if log_scale else "  ") + "Log Scale Colorbar",
            command=self._toggle_log_scale,
            state="normal" if has_mapping else "disabled")
        menu.add_separator()
        menu.add_command(label="Sync Settings from Main Diagram",
                         command=self._sync_from_main)
        menu.add_separator()
        menu.add_command(label="Save Figure\u2026", command=self.save_figure)
        return menu

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    _SAVE_FILETYPES = (
        ("PDF files", "*.pdf"), ("PNG files", "*.png"), ("SVG files", "*.svg"),
        ("All files", "*.*"),
    )

    def _default_figure_name(self) -> str:
        """Suggested file name (without extension) for :meth:`save_figure`."""
        return f"{getattr(self.mol_obj, 'name', 'molecule')}_population_diagram_filtered"

    def save_figure(self, save_path: Optional[str] = None, dpi: int = 300) -> Optional[str]:
        """Save the diagram as shown; the format follows the file extension (PDF, PNG, SVG, ...).

        Opens a file dialog when *save_path* is ``None``.  Returns the saved
        path, or ``None`` if cancelled or the save failed (the reason is shown
        in the status bar).
        """
        import re
        if self._fig is None:
            return None
        if save_path is None:
            from tkinter import filedialog
            stem = re.sub(r"[^\w.-]+", "_", self._default_figure_name())
            save_path = filedialog.asksaveasfilename(
                parent=self, title="Save Figure", defaultextension=".pdf",
                initialfile=f"{stem}.pdf", filetypes=self._SAVE_FILETYPES,
            )
        if not save_path:
            return None
        try:
            self._fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        except Exception as exc:
            self._status_var.set(f"Could not save figure: {exc}")
            return None
        message = f"Figure saved to: {save_path}"
        self._status_var.set(message)
        data_field = getattr(getattr(self._islat, "GUI", None), "data_field", None)
        if data_field is not None:
            try:
                data_field.insert_text(message)
            except Exception:
                pass
        return str(save_path)

    def _clear_color_mapping(self) -> None:
        if self._pdp is not None:
            self._pdp.clear_color_mapping(regenerate=True)
            self._draw_idle()

    def _toggle_log_scale(self) -> None:
        mapping = getattr(self._pdp, "_color_mapping", None)
        if mapping is None:
            return
        mapping["log_scale"] = not mapping.get("log_scale", False)
        self._pdp.generate_plot()
        self._draw_idle()

    def _draw_idle(self) -> None:
        if self._canvas is not None:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    def _sync_from_main(self) -> None:
        """Re-copy the user-adjustable settings from the main diagram."""
        apply_pdp_style(self._pdp, capture_pdp_style(
            find_main_population_plot(self._islat)), regenerate=False)
        self._render()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_full_data(self, data: Optional[Dict[str, Any]]) -> None:
        """Install the population-diagram arrays for the UNFILTERED line list.

        Every later mask is applied to these arrays, so the physics is computed
        once rather than on every filter change.
        """
        self._full_data = data
        self._base_length = (None if data is None
                             else _population_data_length(data))
        if data is not None:
            self._sticky_message = None

    def set_message(self, message: str) -> None:
        """Show an explanatory message that a later redraw will not overwrite.

        Used for conditions the user needs the specific reason for - an
        unsupported line list, or arrays that cannot be aligned with the
        filter - which a generic "no data" would hide.
        """
        self._sticky_message = message or None
        if self._sticky_message:
            self._status_var.set(self._sticky_message)

    @property
    def base_length(self) -> Optional[int]:
        """Row count the masks passed to :meth:`update_mask` must match."""
        return self._base_length

    def update_mask(self, mask: Optional[Sequence[bool]], *,
                    delay_ms: int = 120) -> None:
        """Redraw for the rows selected by *mask*, coalescing rapid calls.

        Safe to call at the filter window's preview cadence: repeated calls
        collapse into a single redraw.
        """
        if not self._alive():
            return
        self._pending_mask = (None if mask is None
                              else np.asarray(mask, dtype=bool))
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except (tk.TclError, ValueError):
                pass
        self._redraw_job = self.after(max(int(delay_ms), 1), self._render)

    def _render(self) -> None:
        """Subset the precomputed arrays and regenerate the diagram."""
        self._redraw_job = None
        if not self._alive():
            return
        if self._full_data is None:
            self._status_var.set(
                self._sticky_message or "No population-diagram data available.")
            return
        try:
            data = subset_population_data(self._full_data, self._pending_mask)
        except ValueError as exc:
            # A stale mask must not silently plot the wrong lines.
            self._status_var.set(str(exc))
            return

        total = self._base_length or 0
        shown = _population_data_length(data) or 0
        self._source.set_data(data if shown else None)
        try:
            self._pdp.generate_plot()
        except Exception as exc:
            self._status_var.set(f"Could not draw: {exc}")
            return
        self._draw_idle()
        self._status_var.set(
            "No lines selected." if not shown
            else f"Showing {shown} / {total} lines.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _alive(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def close(self) -> None:
        """Tear down the diagram, the canvas and the figure.

        ``PopulationDiagramPlot.close()`` is what drops any registered
        MoleculeDict callbacks; skipping it is precisely the leak that was
        fixed for the main view, and a popout is just as capable of causing it.
        """
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except (tk.TclError, ValueError):
                pass
            self._redraw_job = None

        if self._canvas is not None and self._right_click_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._right_click_cid)
            except Exception:
                pass
            self._right_click_cid = None

        if self._pdp is not None:
            try:
                self._pdp.close()
            except Exception:
                pass
            self._pdp = None

        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None

        self._canvas = None
        self._toolbar = None
        self._source.set_data(None)
        self._full_data = None

        if self._on_close_cb is not None:
            callback, self._on_close_cb = self._on_close_cb, None
            try:
                callback(self)
            except Exception:
                pass
        try:
            self.destroy()
        except tk.TclError:
            pass