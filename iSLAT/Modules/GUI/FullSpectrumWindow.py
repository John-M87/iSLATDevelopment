"""Standalone "Display Full Spectrum" window.

The window embeds its own :class:`FullSpectrumView` (the same view used for
the in-GUI Full Spectrum mode) so it offers the same options: number of
panels, wavelength range, data-density X scaling, residual sub-panels,
atomic / saved lines, summed model, observed data, legend and the Y-axis
policy - plus explicit per-panel wavelength ranges and its own wavelength
marks.  Those settings live in a window-local copy of the GUI's toggle
state, so changing them here never affects the main window.  All controls
sit on one compact bar above the plot.
"""
import copy
import re
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView, SKIP_GAPS_TIP
from iSLAT.Modules.Plotting.PlotView import WavelengthMarkerStoreMixin, normalize_y_data_sync
from iSLAT.Modules.Plotting.StackedSpectralPanel import StackedSpectralPanel
from .Tooltips import MenuToolTip
from .YAxisDataSync import (
    Y_DATA_SYNC_MENU_LABEL, Y_DATA_SYNC_MENU_TIP,
    add_y_data_range_controls, add_y_data_sync_menu, y_data_sync_var_value,
)

if TYPE_CHECKING:
    from iSLAT.iSLATClass import iSLAT

#: Toggle-state defaults used when the main GUI's state is unavailable.
DEFAULT_WINDOW_TOGGLE_STATE: Dict[str, Any] = {
    "atomic_lines": False,
    "saved_lines": False,
    "summed": True,
    "legend": True,
    "show_residuals": False,
    "data_density": False,
    "skip_gaps": False,
    "current_selection": None,
    "y_autoscale": True,
    "y_autoscale_source": "data",
    "y_data_sync": None,
    "y_limits": None,
    "observed": True,
    "wavelength_markers": [],
    "saved_line_labels": True,
}


def _apply_layout(
    plot: Any,
    n_panels: Optional[int] = None,
    xlim_range: Optional[Tuple[float, float]] = None,
    ymax_factor: Optional[float] = None,
) -> None:
    """Set panel count / wavelength range / Y margin on a stacked plot and recompute its panels."""
    if n_panels is not None:
        plot.n_panels = max(1, int(n_panels))
    if xlim_range is not None:
        plot._xlim_start, plot._xlim_end = float(xlim_range[0]), float(xlim_range[1])
    if ymax_factor is not None:
        plot.ymax_factor = float(ymax_factor)
    plot._compute_panel_layout()


#: Range keywords replaced by the first / last observed data point.
DATA_START_KEYWORD = "DATASTART"
DATA_END_KEYWORD = "DATAEND"
_DATA_KEYWORDS = rf"(?:{DATA_START_KEYWORD}|{DATA_END_KEYWORD})"


def parse_panel_ranges(
    text: str,
    data_start: Optional[float] = None,
    data_end: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Parse one ``start end`` wavelength pair per line (blank and ``#`` lines are skipped).

    The two numbers may be separated by spaces, a comma, ``-``, an en dash
    or ``to``.  Either bound may be ``DATASTART`` or ``DATAEND``
    (case-insensitive), replaced by *data_start* / *data_end* (the first /
    last observed data point).  Raises ``ValueError`` naming the offending line.
    """
    keyword_values = {DATA_START_KEYWORD: data_start, DATA_END_KEYWORD: data_end}
    ranges: List[Tuple[float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned = re.sub(r"\bto\b|[,;\u2013\u2014]", " ", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?<=[\d.])\s*-\s*(?=[\d.])", " ", cleaned)
        cleaned = re.sub(rf"\b({_DATA_KEYWORDS})\s*-\s*", r"\1 ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*-\s*({_DATA_KEYWORDS})\b", r" \1", cleaned, flags=re.IGNORECASE)
        parts = cleaned.split()
        if len(parts) != 2:
            raise ValueError(f"Line {lineno}: expected 'start end', got '{raw.strip()}'.")
        bounds = []
        for part in parts:
            keyword = part.upper()
            if keyword in keyword_values:
                value = keyword_values[keyword]
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Line {lineno}: {keyword} needs observed data to be loaded.")
                bounds.append(float(value))
                continue
            try:
                bounds.append(float(part))
            except ValueError:
                raise ValueError(f"Line {lineno}: expected 'start end', got '{raw.strip()}'.") from None
        lo, hi = bounds
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f"Line {lineno}: start must be less than end.")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("Enter at least one range.")
    return ranges


def format_panel_ranges(edges: Sequence[float], ends: Sequence[float]) -> str:
    """Inverse of :func:`parse_panel_ranges` for the current panels."""
    return "\n".join(f"{lo:.4f} {hi:.4f}" for lo, hi in zip(edges, ends))


class _WindowPlotController(WavelengthMarkerStoreMixin):
    """Minimal stand-in for the GUI's ``iSLATPlot`` that gives the window its own
    toggle state and its own wavelength marks."""

    def __init__(self, islat: Any, toggle_state: Dict[str, Any], theme: Any) -> None:
        self.islat = islat
        self.toggle_state = toggle_state
        self.theme = theme if isinstance(theme, dict) else {}
        self.control_bus = None
        self.view: Optional[FullSpectrumView] = None

    def refresh_wavelength_markers(self) -> None:
        if self.view is not None:
            self.view.refresh_wavelength_markers()

    @property
    def legend_toggle(self) -> bool:
        return self.toggle_state.get("legend", True)

    @property
    def summed_toggle(self) -> bool:
        return self.toggle_state.get("summed", True)


class WindowFullSpectrumView(FullSpectrumView):
    """:class:`FullSpectrumView` driven by the window's own controls instead of the ControlBus."""

    def __init__(self, plot_manager: Any, initial_layout: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plot_manager)
        self._initial_layout: Dict[str, Any] = dict(initial_layout or {})
        #: Explicit per-panel ranges, re-applied to every plot this view builds.
        self.panel_ranges: Optional[List[Tuple[float, float]]] = None

    def _create_plot(self):
        plot = super()._create_plot()
        layout, self._initial_layout = self._initial_layout, {}
        if layout:
            _apply_layout(plot, **layout)
        if self.panel_ranges:
            plot.set_panel_ranges(self.panel_ranges)
        return plot

    def set_panel_ranges(self, ranges: Optional[Sequence[Tuple[float, float]]]) -> None:
        """Show one panel per ``(start, end)`` (any order, gaps allowed), or ``None`` for an even split.

        Raises ``ValueError`` for an invalid range.
        """
        if self._plot is None:
            return
        validated = StackedSpectralPanel._validate_panel_ranges(ranges) if ranges is not None else None
        self.panel_ranges = validated
        self._plot.set_panel_ranges(validated)
        self._regenerate_layout()

    def apply_layout(
        self,
        n_panels: Optional[int] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: Optional[float] = None,
    ) -> None:
        """Change the panel count, wavelength range and/or Y margin and re-render once.

        Clears any explicit per-panel ranges (back to an even / data-density split).
        """
        if self._plot is None:
            return
        self.panel_ranges = None
        self._plot.panel_ranges = None
        _apply_layout(self._plot, n_panels, xlim_range, ymax_factor)
        self._regenerate_layout()

    def _register_control_fields(self) -> None:
        """The window provides its own controls; nothing goes on the main ControlBus."""

    def _saved_line_labels_visible(self) -> bool:
        """The window's own saved-line label toggle."""
        return bool(self._pm.toggle_state.get("saved_line_labels", True))

    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        menu = tk.Menu(canvas_widget, tearoff=0)
        menu.add_command(label="Save Figure\u2026", command=self.save_figure)
        return menu

    # Save dialog: any file type, the dialog's own start folder (title, file
    # name and message as in the Full Spectrum view).
    SAVE_INITIAL_DIR: Optional[str] = None
    SAVE_ANY_FILE_TYPE: bool = True

    def _can_save_figure(self) -> bool:
        return self.get_figure() is not None

    def _build_export_figure(self):
        """Save the figure exactly as displayed (residuals, data density and all)."""
        return self.get_figure(), None

class FullSpectrumWindow(tk.Toplevel):
    """
    GUI window for displaying and interacting with full spectrum plots.

    Starts from the main GUI's current toggle state, wavelength marks and
    Full Spectrum layout (panel count, wavelength range, Y margin);
    afterwards every option - including the marks - is independent of the
    main window.  *kwargs* may override the initial ``n_panels``,
    ``xlim_range`` and ``ymax_factor``.
    """
    def __init__(self, parent, islat_ref: "iSLAT", **kwargs):
        super().__init__(parent)
        self.islat_ref = islat_ref
        self.kwargs = kwargs

        self.title("Full Spectrum Output")
        self._constrain_to_screen()

        gui_plot = self._gui_plot()
        self.toggle_state: Dict[str, Any] = copy.deepcopy(DEFAULT_WINDOW_TOGGLE_STATE)
        gui_state = getattr(gui_plot, "toggle_state", None)
        if isinstance(gui_state, dict):
            self.toggle_state.update(copy.deepcopy(gui_state))
        from iSLAT.Modules.Plotting.SavedLineLabels import labels_visible
        self.toggle_state["saved_line_labels"] = labels_visible("Full Spectrum")
        self.controller = _WindowPlotController(islat_ref, self.toggle_state, getattr(gui_plot, "theme", None))
        self.view = WindowFullSpectrumView(self.controller, self._initial_layout(gui_plot, kwargs))
        self.controller.view = self.view

        self._marker_dialog: Optional[tk.Toplevel] = None
        self._ranges_dialog: Optional[tk.Toplevel] = None
        self._y_limits_dialog: Optional[tk.Toplevel] = None

        # --- UI layout: one compact bar, the plot, the navigation toolbar ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=(0, 3))

        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.nav_toolbar: Optional[NavigationToolbar2Tk] = None
        self._nav_canvas: Any = None

        self.view.activate(self.canvas_frame)
        self._sync_nav_toolbar()

        self._create_toolbar()
        self._refresh_controls()
        self.bind("<KeyPress-M>", self._on_shift_m)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _gui_plot(self) -> Any:
        gui = getattr(self.islat_ref, "GUI", None)
        return getattr(gui, "plot", None)

    @staticmethod
    def _initial_layout(gui_plot: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Panel count / range / Y margin of the main Full Spectrum view, then *overrides*."""
        layout: Dict[str, Any] = {}
        live = getattr(getattr(gui_plot, "_full_spectrum_view", None), "_plot", None)
        if live is not None:
            layout = {
                "n_panels": getattr(live, "n_panels", None),
                "xlim_range": (getattr(live, "_xlim_start", None), getattr(live, "_xlim_end", None)),
                "ymax_factor": getattr(live, "ymax_factor", None),
            }
            if None in layout["xlim_range"]:
                layout.pop("xlim_range")
        for key in ("n_panels", "xlim_range", "ymax_factor"):
            if overrides.get(key) is not None:
                layout[key] = overrides[key]
        return {k: v for k, v in layout.items() if v is not None}

    def _message(self, text: str) -> None:
        data_field = getattr(getattr(self.islat_ref, "GUI", None), "data_field", None)
        if data_field is not None:
            data_field.insert_text(text)

    def _constrain_to_screen(self):
        """Set initial window geometry so it fits comfortably on screen."""
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Use at most 85% of screen width and 90% of screen height
        win_w = min(int(screen_w * 0.85), 1400)
        win_h = min(int(screen_h * 0.90), screen_h - 80)
        pos_x = max((screen_w - win_w) // 2, 0)
        pos_y = max((screen_h - win_h) // 2, 0)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.minsize(600, 400)

    def _sync_nav_toolbar(self) -> None:
        """(Re)attach the matplotlib navigation toolbar when the view swaps its canvas."""
        canvas = self.view.get_canvas()
        if canvas is self._nav_canvas:
            return
        if self.nav_toolbar is not None:
            self.nav_toolbar.destroy()
            self.nav_toolbar = None
        self._nav_canvas = canvas
        if canvas is not None:
            self.nav_toolbar = NavigationToolbar2Tk(canvas, self.nav_frame, pack_toolbar=False)
            self.nav_toolbar.update()
            self.nav_toolbar.pack(side=tk.LEFT, fill=tk.X)

    @staticmethod
    def _reuse_dialog(dialog: Optional[tk.Toplevel]) -> bool:
        """Raise *dialog* if it is still open; return whether it was."""
        try:
            if dialog is not None and dialog.winfo_exists():
                dialog.deiconify()
                dialog.lift()
                dialog.focus_set()
                return True
        except tk.TclError:
            pass
        return False

    def _small_dialog(self, title: str) -> Tuple[tk.Toplevel, ttk.Frame]:
        win = tk.Toplevel(self)
        win.title(title)
        win.resizable(False, False)
        win.transient(self)
        content = ttk.Frame(win)
        content.pack(fill="both", expand=True, padx=10, pady=10)
        win.bind("<Escape>", lambda _e: win.destroy())
        return win, content

    # ------------------------------------------------------------------
    # Compact control bar
    # ------------------------------------------------------------------
    def _menubutton(self, text: str) -> Tuple[ttk.Menubutton, tk.Menu]:
        button = ttk.Menubutton(self.toolbar_frame, text=text)
        menu = tk.Menu(button, tearoff=0)
        button.configure(menu=menu)
        button.pack(side=tk.LEFT, padx=(0, 3))
        return button, menu

    def _separator(self) -> None:
        ttk.Separator(self.toolbar_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=4)

    def _create_toolbar(self) -> None:
        bar = self.toolbar_frame
        ts = self.toggle_state

        self.file_button, self.file_menu = self._menubutton("File")
        self.file_menu.add_command(label="Save as PDF\u2026", command=self.save_figure)
        self.file_menu.add_command(label="Save as PNG\u2026", command=self.save_figure_png)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Refresh Plot", command=self.refresh_plot)
        self._separator()

        # Layout: panel count, wavelength range, Y margin.
        self.n_panels_var = tk.StringVar(master=self)
        self.start_var = tk.StringVar(master=self)
        self.end_var = tk.StringVar(master=self)
        self.ymax_var = tk.StringVar(master=self)
        self.layout_entries: Dict[str, ttk.Entry] = {}
        for key, text, var, width in (
            ("n_panels", "Panels", self.n_panels_var, 3),
            ("start", "\u03bb", self.start_var, 7),
            ("end", "\u2013", self.end_var, 7),
            ("ymax", "Y pad", self.ymax_var, 4),
        ):
            ttk.Label(bar, text=text).pack(side=tk.LEFT, padx=(2, 1))
            entry = ttk.Entry(bar, textvariable=var, width=width)
            entry.pack(side=tk.LEFT)
            entry.bind("<Return>", lambda _e: self.apply_settings())
            self.layout_entries[key] = entry
        self.apply_btn = ttk.Button(bar, text="Apply", command=self.apply_settings, width=5)
        self.apply_btn.pack(side=tk.LEFT, padx=(3, 3))
        self.ranges_btn = ttk.Button(bar, text="Ranges\u2026", command=self.open_panel_ranges_dialog, width=8)
        self.ranges_btn.pack(side=tk.LEFT, padx=(0, 3))
        self._separator()

        # Display toggles.
        self.option_vars: Dict[str, tk.BooleanVar] = {}
        self.display_button, self.display_menu = self._menubutton("Display")
        toggles = (
            ("show_residuals", "Residuals", self.set_residuals),
            ("data_density", "Data Density X-Scale", self.set_data_density),
            ("skip_gaps", "Skip Data Gaps", self.set_skip_gaps),
            None,
            ("atomic_lines", "Atomic Lines", self.set_atomic_lines),
            ("saved_lines", "Saved Lines", self.set_saved_lines),
            ("saved_line_labels", "Saved Line Labels", self.set_saved_line_labels),
            ("summed", "Summed Spectrum", self.set_summed),
            ("observed", "Show Observed Data", self.set_observed),
            ("legend", "Legend", self.set_legend),
        )
        for item in toggles:
            if item is None:
                self.display_menu.add_separator()
                continue
            key, text, setter = item
            var = tk.BooleanVar(master=self, value=bool(ts.get(key, DEFAULT_WINDOW_TOGGLE_STATE[key])))
            self.display_menu.add_checkbutton(label=text, variable=var,
                                              command=lambda s=setter, v=var: s(v.get()))
            self.option_vars[key] = var
        MenuToolTip(self.display_menu, {"Skip Data Gaps": SKIP_GAPS_TIP})

        # Y axis: auto-scale, fit to model, manual limits.
        self.y_autoscale_var = tk.BooleanVar(master=self, value=bool(ts.get("y_autoscale", True)))
        self.y_model_var = tk.BooleanVar(master=self, value=ts.get("y_autoscale_source") == "model")
        self.y_button, self.y_menu = self._menubutton("Y Axis")
        self.y_menu.add_checkbutton(label="Auto-scale Y Axis", variable=self.y_autoscale_var,
                                    command=lambda: self.set_y_autoscale(self.y_autoscale_var.get()))
        self.y_menu.add_checkbutton(label="Auto-scale Y to Model Flux", variable=self.y_model_var,
                                    command=lambda: self.set_y_autoscale_source(
                                        "model" if self.y_model_var.get() else "data"))
        self.y_data_sync_var = tk.StringVar(master=self, value=y_data_sync_var_value(ts.get("y_data_sync")))
        add_y_data_sync_menu(self.y_menu, self.y_data_sync_var, self.set_y_data_sync)
        MenuToolTip(self.y_menu, {Y_DATA_SYNC_MENU_LABEL: Y_DATA_SYNC_MENU_TIP})
        self.y_menu.add_separator()
        self.y_menu.add_command(label="Set Y Limits\u2026", command=self.open_y_limits_dialog)

        self.marks_btn = ttk.Button(bar, text="Marks\u2026", command=self.open_marker_dialog, width=7)
        self.marks_btn.pack(side=tk.LEFT, padx=(0, 3))

        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(bar, textvariable=self.status_var, foreground="red").pack(side=tk.LEFT, padx=(6, 0))

    def _refresh_controls(self) -> None:
        """Mirror the plot's actual layout and toggles into the controls."""
        plot = self.view._plot
        if plot is not None:
            self.n_panels_var.set(str(plot.n_panels))
            self.start_var.set(f"{plot._xlim_start:.4f}")
            self.end_var.set(f"{plot._xlim_end:.4f}")
            self.ymax_var.set(f"{plot.ymax_factor:g}")
        custom = bool(self.view.panel_ranges)
        self.ranges_btn.configure(text="Ranges*\u2026" if custom else "Ranges\u2026")
        for key, var in self.option_vars.items():
            var.set(bool(self.toggle_state.get(key, DEFAULT_WINDOW_TOGGLE_STATE[key])))
        self.y_autoscale_var.set(bool(self.toggle_state.get("y_autoscale", True)))
        self.y_model_var.set(self.toggle_state.get("y_autoscale_source") == "model")
        self.y_data_sync_var.set(y_data_sync_var_value(self.toggle_state.get("y_data_sync")))

    def _after_change(self) -> None:
        self.status_var.set("")
        self._sync_nav_toolbar()
        self._refresh_controls()

    # ------------------------------------------------------------------
    # Options (all act on the window's own toggle state)
    # ------------------------------------------------------------------
    def set_residuals(self, show: bool) -> None:
        show = bool(show)
        if show != self.toggle_state.get("show_residuals", False):
            self.toggle_state["show_residuals"] = show
            self.view.toggle_residuals(show)
        self._after_change()

    def set_data_density(self, enabled: bool) -> None:
        self.view._set_data_density(bool(enabled))
        self._after_change()

    def set_skip_gaps(self, enabled: bool) -> None:
        """Break the spectrum at data gaps, crop empty panel edges and leave out empty panels."""
        self.view._set_skip_gaps(bool(enabled))
        self._after_change()

    def set_atomic_lines(self, show: bool) -> None:
        self.toggle_state["atomic_lines"] = bool(show)
        self.view.toggle_atomic_lines(bool(show))
        self._after_change()

    def set_saved_lines(self, show: bool) -> None:
        self.toggle_state["saved_lines"] = bool(show)
        self.view.toggle_saved_lines(bool(show))
        self._after_change()

    def set_saved_line_labels(self, show: bool) -> None:
        """Show or hide the saved-line labels in this window only."""
        self.toggle_state["saved_line_labels"] = bool(show)
        self.view._remove_saved_line_artists()
        if self.toggle_state.get("saved_lines", False):
            self.view._add_saved_line_artists()
        self.view.draw()
        self._after_change()

    def set_summed(self, visible: bool) -> None:
        self.toggle_state["summed"] = bool(visible)
        self.view.toggle_summed_spectrum(bool(visible))
        self._after_change()

    def set_observed(self, visible: bool) -> None:
        self.toggle_state["observed"] = bool(visible)
        self.view.toggle_observed_data(bool(visible))
        self._after_change()

    def set_legend(self, visible: bool) -> None:
        self.toggle_state["legend"] = bool(visible)
        self.view.toggle_legend(bool(visible))
        self._after_change()

    def set_y_autoscale(self, enabled: bool) -> None:
        """Turn Y auto-scaling on (re-fit now) or off (freeze the current limits)."""
        enabled = bool(enabled)
        self.toggle_state["y_autoscale"] = enabled
        if not enabled and self.toggle_state.get("y_limits") is None:
            self.toggle_state["y_limits"] = self.view.get_y_limits()
        self.view.apply_y_limits(self.toggle_state, autoscale_now=True)
        self.view.draw()
        self._after_change()

    def set_y_autoscale_source(self, source: str) -> None:
        """Fit the Y auto-scale to the observed ``"data"`` or the drawn ``"model"``.

        Choosing ``"model"`` stops syncing to the data's min / max.
        """
        self.toggle_state["y_autoscale_source"] = "model" if source == "model" else "data"
        if source == "model":
            self.toggle_state["y_data_sync"] = None
        if self.toggle_state.get("y_autoscale", True):
            self.view.apply_y_limits(self.toggle_state, autoscale_now=True)
            self.view.draw()
        self._after_change()

    def set_y_data_sync(self, scope: Optional[str]) -> None:
        """Sync the panels' Y axes to the observed data's exact min / max, or stop syncing.

        *scope* is ``"global"`` (the whole spectrum), ``"view"`` (the data
        in all panels' ranges, one shared range) or ``"panel"`` (each panel
        to its own range); ``None`` / ``"off"`` returns to the normal
        auto-scale rule.  Syncing turns auto-scale on and the model-flux
        source off.
        """
        scope = normalize_y_data_sync(scope)
        self.toggle_state["y_data_sync"] = scope
        if scope is not None:
            self.toggle_state["y_autoscale_source"] = "data"
            self.toggle_state["y_autoscale"] = True
        if self.toggle_state.get("y_autoscale", True):
            self.view.apply_y_limits(self.toggle_state, autoscale_now=True)
            self.view.draw()
        self._after_change()

    def set_y_limits(self, ymin: float, ymax: float) -> None:
        """Pin every panel to ``[ymin, ymax]`` (switches auto-scaling off).

        Raises ``ValueError`` unless both are finite and ``ymin < ymax``.
        """
        lo, hi = float(ymin), float(ymax)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError("Y min must be less than Y max.")
        self.toggle_state["y_limits"] = (lo, hi)
        self.toggle_state["y_autoscale"] = False
        self.view.apply_y_limits(self.toggle_state)
        self.view.draw()
        self._after_change()

    def apply_settings(self) -> None:
        """Apply the Panels / wavelength range / Y margin entries (clears custom panel ranges)."""
        try:
            n_panels = int(float(self.n_panels_var.get()))
            start, end = float(self.start_var.get()), float(self.end_var.get())
            ymax_factor = float(self.ymax_var.get())
        except ValueError:
            self.status_var.set("Invalid layout value.")
            return
        if n_panels < 1:
            self.status_var.set("Panels must be at least 1.")
            return
        if not (np.isfinite(start) and np.isfinite(end)) or start >= end:
            self.status_var.set("Start must be less than end.")
            return
        if not np.isfinite(ymax_factor) or ymax_factor < 0:
            self.status_var.set("Y margin must be 0 or more.")
            return
        self.view.apply_layout(n_panels=n_panels, xlim_range=(start, end), ymax_factor=ymax_factor)
        self._after_change()

    def set_panel_ranges(self, ranges: Optional[Sequence[Tuple[float, float]]]) -> None:
        """One panel per ``(start, end)`` (any order, gaps allowed), or ``None`` for an even split.

        Raises ``ValueError`` for an invalid range.
        """
        self.view.set_panel_ranges(ranges)
        self._after_change()

    def observed_data_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """First and last observed wavelength as plotted (RV-corrected), or ``(None, None)``."""
        plot = self.view._plot
        wave = getattr(plot, "wave_data", None) if plot is not None else None
        if wave is None:
            return None, None
        wave = np.asarray(wave, dtype=float)
        wave = wave[np.isfinite(wave)]
        if wave.size == 0:
            return None, None
        return float(wave.min()), float(wave.max())

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------
    def open_panel_ranges_dialog(self) -> None:
        """Edit explicit per-panel wavelength ranges, one ``start end`` per line."""
        if self._reuse_dialog(self._ranges_dialog):
            return
        win, content = self._small_dialog("Panel Ranges")
        self._ranges_dialog = win
        ttk.Label(
            content,
            text="One panel per line: start end (\u03bcm).\n"
                 "Panels are shown in this order; ranges may overlap or leave gaps.\n"
                 f"{DATA_START_KEYWORD} / {DATA_END_KEYWORD} = first / last observed data point.",
        ).pack(anchor="w")
        text = tk.Text(content, width=34, height=10, undo=True)
        text.pack(fill="both", expand=True, pady=(6, 4))
        plot = self.view._plot
        if plot is not None:
            text.insert("1.0", format_panel_ranges(plot._panel_edges, plot._panel_ends))
        error_var = tk.StringVar(master=win, value="")
        ttk.Label(content, textvariable=error_var, foreground="red").pack(anchor="w")

        def _apply():
            try:
                self.set_panel_ranges(parse_panel_ranges(text.get("1.0", "end"), *self.observed_data_bounds()))
            except ValueError as exc:
                error_var.set(str(exc))
                return
            error_var.set("")

        def _even_split():
            self.set_panel_ranges(None)
            plot_now = self.view._plot
            text.delete("1.0", "end")
            if plot_now is not None:
                text.insert("1.0", format_panel_ranges(plot_now._panel_edges, plot_now._panel_ends))
            error_var.set("")

        buttons = ttk.Frame(content)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Apply", command=_apply).pack(side="left")
        ttk.Button(buttons, text="Even Split", command=_even_split).pack(side="left", padx=(4, 0))
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="right")
        text.focus_set()

    def open_y_limits_dialog(self) -> None:
        """Small Min / Max dialog for pinning the Y axis."""
        if self._reuse_dialog(self._y_limits_dialog):
            return
        win, content = self._small_dialog("Set Y Limits")
        self._y_limits_dialog = win
        lims = self.toggle_state.get("y_limits") or self.view.get_y_limits() or (0.0, 1.0)
        ymin_var = tk.StringVar(master=win, value=f"{lims[0]:.6g}")
        ymax_var = tk.StringVar(master=win, value=f"{lims[1]:.6g}")
        error_var = tk.StringVar(master=win, value="")
        for row, (text, var) in enumerate((("Y min:", ymin_var), ("Y max:", ymax_var))):
            ttk.Label(content, text=text).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(content, textvariable=var, width=12).grid(row=row, column=1, pady=2)
        ttk.Label(content, textvariable=error_var, foreground="red").grid(
            row=3, column=0, columnspan=2, sticky="w")

        def _fill(lo: float, hi: float) -> None:
            ymin_var.set(f"{lo:.6g}")
            ymax_var.set(f"{hi:.6g}")

        def _sync(scope: str) -> None:
            self.set_y_data_sync(scope)
            win.destroy()

        add_y_data_range_controls(
            content, 2,
            current_scope=self.toggle_state.get("y_data_sync"),
            panel_count=self.view.y_axis_count(),
            get_range=self.view.get_data_y_range,
            on_fill=_fill,
            on_sync=_sync,
            set_status=error_var.set,
            padx=0, pady=(6, 2),
        )

        def _apply(_event=None):
            try:
                self.set_y_limits(float(ymin_var.get()), float(ymax_var.get()))
            except ValueError as exc:
                error_var.set(str(exc) if "Y min" in str(exc) else "Invalid Y limits.")
                return
            win.destroy()

        buttons = ttk.Frame(content)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Button(buttons, text="Apply", command=_apply).pack(side="right", padx=(4, 0))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right")
        win.bind("<Return>", _apply)

    def _main_selection(self) -> Optional[Tuple[float, float]]:
        """The main GUI's span selection (dragging on this window's plot sets it)."""
        state = getattr(self._gui_plot(), "toggle_state", None)
        sel = state.get("current_selection") if isinstance(state, dict) else None
        try:
            lo, hi = float(sel[0]), float(sel[1])
        except (TypeError, ValueError, IndexError):
            return None
        return (lo, hi) if lo <= hi else (hi, lo)

    def open_marker_dialog(self) -> None:
        """Mark wavelengths on this window only (Shift+M while the window has focus)."""
        from .WavelengthMarkerDialog import open_wavelength_marker_dialog
        if self._reuse_dialog(self._marker_dialog):
            return
        plot = self.view._plot
        if plot is not None:
            center = (plot._xlim_start + plot._xlim_end) / 2.0
        else:
            center = 0.0

        def _on_close():
            self._marker_dialog = None

        self._marker_dialog = open_wavelength_marker_dialog(
            self, self.controller,
            default_window=("single", center, center),
            get_selection=self._main_selection,
            on_close=_on_close,
        )

    def _on_shift_m(self, event: Any) -> Optional[str]:
        """Open this window's marks dialog unless the user is typing in a text field."""
        try:
            widget_class = event.widget.winfo_class()
        except Exception:
            return None
        if widget_class in ("Entry", "Text", "TEntry", "TCombobox"):
            return None
        self.open_marker_dialog()
        return "break"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def save_figure(self):
        """Save the displayed figure as PDF (opens a file dialog)."""
        self.view.save_figure(file_format="pdf")

    def save_figure_png(self):
        """Save the displayed figure as a 300-dpi PNG (opens a file dialog)."""
        self.view.save_figure(file_format="png", dpi=300)

    def refresh_plot(self):
        """Reload data and models from the GUI, keeping this window's options and marks."""
        try:
            self.view.update_model_plot()
            self._after_change()
            self._message("Full spectrum plot refreshed")
        except Exception as e:
            self._message(f"Error refreshing plot: {str(e)}")

    def on_closing(self):
        for dialog in (self._marker_dialog, self._ranges_dialog, self._y_limits_dialog):
            try:
                if dialog is not None and dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError:
                pass
        try:
            self.view.destroy()
        except Exception:
            pass
        self.destroy()