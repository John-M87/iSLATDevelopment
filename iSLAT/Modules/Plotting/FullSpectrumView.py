"""
FullSpectrumView -- :class:`PlotView` implementation for multi-panel stacked spectrum layouts.

This view **composes** a :class:`StackedSpectralPanel` subclass (by default a :class:`FullSpectrumPlot`) for all rendering, then adds interactive features on top:

* Span selectors on every panel (click-to-inspect)
* Dynamic overlay toggles (atomic / saved lines, summed spectrum)
* Canvas lifecycle management for the Tk GUI

The view is generic: it can drive **any** :class:`StackedSpectralPanel` implementation that exposes the standard stacked-panel interface
(``fig``, ``subplots``, ``_panel_edges``, ``_step``, ``_xlim_start``, ``_xlim_end``, ``wave_data``, ``generate_plot()``, etc.).

The concrete plot class is determined by :meth:`_create_plot`, which subclasses can override to swap in a different stacked-panel implementation.
"""
from __future__ import annotations

import warnings
from os.path import dirname
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.figure import Figure as MplFigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

from .PlotView import PlotView
from .StackedSpectralPanel import StackedSpectralPanel
from .FullSpectrumPlot import FullSpectrumPlot
from .ResidualSpectrumPlot import ResidualSpectrumPlot
from .BasePlot import BasePlot
from .ToggleMixin import ToggleMixin

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
from iSLAT.Modules.FileHandling import absolute_data_files_path
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh

# Suppress constrained_layout warnings triggered by adding/removing overlay
# artists after the layout engine has already run.
warnings.filterwarnings(
    "ignore",
    message=".*constrained_layout.*",
    category=UserWarning,
    module="matplotlib",
)

# Import debug configuration
try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    class _Fallback:
        def verbose(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): print(f"WARNING: {a}")
        def error(self, *a, **k): print(f"ERROR: {a}")
        def trace(self, *a, **k): pass
    debug_config = _Fallback()

class FullSpectrumView(ToggleMixin, PlotView):
    """
    Multi-panel full spectrum view backed by a :class:`StackedSpectralPanel`.

    Rendering is delegated to the composed stacked-panel plot (by default a :class:`FullSpectrumPlot`)
    which owns the figure, axes, and all the standard rendering helpers.
    This view adds:

    - Tk canvas management (pack / unpack)
    - Span selectors for click-to-inspect
    - Interactive overlay toggles (atomic lines, saved lines, summed)
    - PDF export (creates a fresh standalone plot)

    Toggle-state management (atomic lines, saved lines, summed spectrum, legend) is provided by :class:`ToggleMixin`.

    The view is **generic**: override :meth:`_create_plot` and
    :meth:`_create_standalone_plot` to back it with any
    :class:`StackedSpectralPanel` subclass.
    """

    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat
        self._parent_frame: Any = None

        # Canvas -- built lazily
        self._canvas: Optional[FigureCanvasTkAgg] = None

        # The composed plot -- created on first activation.
        # Typed as the abstract base so the view can drive any subclass.
        self._plot: Optional[StackedSpectralPanel] = None

        # Span selectors for interactive inspection
        self.span_selectors: Dict[int, SpanSelector] = {}

        # Line data (for saved-line annotations in interactive mode)
        self.line_data: Optional[pd.DataFrame] = None

        self._initialised: bool = False  # True after first generate
        self._needs_refresh: bool = True  # Set True when data changes; cleared after re-render

    # --- helpers used by _register_control_fields ---
    def _get_n_panels(self):
        if self._plot is not None:
            return self._plot.n_panels
        return 10

    def _set_n_panels(self, value):
        if self._plot is None:
            return
        n = max(1, int(value))
        if n == self._plot.n_panels:
            return
        self._plot.n_panels = n
        self._plot._compute_panel_layout()
        # Full rebuild required because subplot grid changes
        self.span_selectors.clear()
        self._plot.generate_plot()
        self._install_span_selectors()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _get_display_range(self):
        if self._plot is not None:
            return (self._plot._xlim_start, self._plot._xlim_end)
        return (0.0, 0.0)

    def _set_display_range(self, start, end):
        if self._plot is None:
            return
        self._plot._xlim_start = float(start)
        self._plot._xlim_end = float(end)
        self._plot._compute_panel_layout()
        self.span_selectors.clear()
        self._plot.generate_plot()
        self._install_span_selectors()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _get_data_density(self) -> bool:
        return self._pm.toggle_state.get("data_density", False)

    def _set_data_density(self, value: bool) -> None:
        from .SpectralPanel import XScaling
        value = bool(value)
        if value == self._pm.toggle_state.get("data_density", False):
            return
        self._pm.toggle_state["data_density"] = value
        if self._plot is None:
            return
        self._plot.x_scaling = XScaling.DATA_DENSITY if value else XScaling.WAVELENGTH
        self._plot._compute_panel_layout()
        self.span_selectors.clear()
        self._plot.generate_plot()
        self._install_span_selectors()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ==================================================================
    # Convenience accessors (delegate to composed plot)
    # ==================================================================
    @property
    def fig(self) -> Optional["Figure"]:
        return self._plot.fig if self._plot is not None else None

    @property
    def subplots(self) -> Dict[int, "Axes"]:
        """Return a ``{idx: Axes}`` mapping for every panel in the plot.

        For single-axes-per-cell layouts (e.g. :class:`FullSpectrumPlot`) this returns the dict directly.
        For multi-axes cells (e.g. :class:`ResidualSpectrumPlot` where each cell is a tuple)
        the first axes of each cell is returned so that overlay operations have a consistent interface.
        """
        if self._plot is None:
            return {}
        raw = getattr(self._plot, "subplots", {})
        if not raw:
            return {}
        # Peek at the first value to decide shape
        first = next(iter(raw.values()))
        if isinstance(first, tuple):
            # Multi-axes cell -- return the primary (spectrum) axes
            return {k: v[0] for k, v in raw.items()}
        return raw

    def _iter_all_axes(self) -> List["Axes"]:
        """Yield every axes object across all cells (flat)."""
        if self._plot is None:
            return []
        raw = getattr(self._plot, "subplots", {})
        axes: List["Axes"] = []
        for val in raw.values():
            if isinstance(val, tuple):
                axes.extend(val)
            else:
                axes.append(val)
        return axes

    # ------------------------------------------------------------------
    # ToggleMixin hooks
    # ------------------------------------------------------------------
    def _toggle_ready(self) -> bool:
        return self._initialised

    def _iter_toggle_axes(self):
        return self._iter_all_axes()

    def _load_saved_line_data(self):
        return self._load_line_data()

    # ==================================================================
    # Data loading
    # ==================================================================
    def _load_spectrum_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return observed spectrum arrays from in-memory iSLAT data.

        Returns
        -------
        wave : np.ndarray
            RV-corrected (rest-frame) wavelengths for display.
        flux : np.ndarray
            Observed flux array.
        wave_obs : np.ndarray
            Observer-frame wavelengths, used by MoleculeDict methods that apply the stellar RV correction internally.
        """
        wave_obs = np.array(self._islat.wave_data_original, copy=True)
        wave = self._islat.molecules_dict.apply_stellar_rv(wave_obs)
        flux = np.array(self._islat.flux_data, copy=True)
        return wave, flux, wave_obs

    def _load_line_data(self) -> Optional[pd.DataFrame]:
        """Load the saved line list (for annotations)."""
        try:
            return pd.read_csv(self._islat.input_line_list, sep=",")
        except Exception:
            return None

    # ==================================================================
    # Plot creation / refresh (delegates to StackedSpectralPanel)
    # ==================================================================
    def _create_plot(self) -> StackedSpectralPanel:
        """Build a fresh stacked-panel plot from the current iSLAT state.

        When ``show_residuals`` is active in the toggle state a :class:`ResidualSpectrumPlot` (spectrum + residual sub-panels) is returned; otherwise a plain :class:`FullSpectrumPlot`.

        Subclasses can override this to return a different :class:`StackedSpectralPanel` implementation.
        """
        wave, flux, wave_obs = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        # Compute a figsize appropriate for the interactive GUI view.
        # Without an explicit figsize the auto-calculation in the
        # stacked panel produces dimensions suited for PDF export
        # (e.g. 12 x 16 in) which are far too tall for an embedded
        # canvas -- especially on Windows where the oversized figure
        # causes rendering/layout problems.
        # Cap the height to a screen-proportional value, matching the
        # approach used by FullSpectrumWindow._create_plot().
        figsize = self._compute_interactive_figsize(wave)

        show_residuals = self._pm.toggle_state.get("show_residuals", False)
        from .SpectralPanel import XScaling
        x_scaling = (
            XScaling.DATA_DENSITY
            if self._pm.toggle_state.get("data_density", False)
            else XScaling.WAVELENGTH
        )

        if show_residuals:
            model_flux = self._compute_model_flux(wave_obs, wave)
            error_data = getattr(self._islat, "err_data", None)
            plot = ResidualSpectrumPlot(
                wave_data=wave,
                flux_data=flux,
                model_flux=model_flux,
                error_data=error_data,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                figsize=figsize,
                theme=self._pm.theme,
                x_scaling=x_scaling,
            )
        else:
            # The composed plot handles all rendering via BasePlot helpers.
            # We do NOT pass line_list / atomic_lines here -- those are applied
            # dynamically by sync_toggle_state() so they can be toggled.
            plot = FullSpectrumPlot(
                wave_data=wave,
                flux_data=flux,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                figsize=figsize,
                theme=self._pm.theme,
                x_scaling=x_scaling,
            )
        return plot

    # ------------------------------------------------------------------
    # Model-flux helpers
    # ------------------------------------------------------------------
    def _compute_model_flux(
        self,
        wave_obs: np.ndarray,
        wave_rest: np.ndarray,
    ) -> np.ndarray:
        """Return the summed model flux on the rest-frame wavelength grid.

        The model is constructed from all molecules via
        :meth:`MoleculeDict.get_summed_flux_resampled`, which uses
        flux-conserving resampling (spectres) to map the model onto
        *wave_rest*.  *wave_rest* must already be in the rest frame;
        the observer-frame array *wave_obs* is accepted for API
        compatibility but is no longer used inside this method.
        """
        try:
            _, model_flux = self._islat.molecules_dict.get_summed_flux_resampled(
                wave_rest, visible_only=True,
            )
        except Exception as exc:
            debug_config.warning(
                "full_spectrum_view",
                f"Could not compute model flux for residuals: {exc}",
            )
            return np.zeros_like(wave_rest)

        if len(model_flux) == 0:
            return np.zeros_like(wave_rest)

        return model_flux

    def _compute_interactive_figsize(
        self, wave: np.ndarray
    ) -> Tuple[float, float]:
        """Return a ``(width, height)`` suitable for the embedded GUI canvas.

        Uses the Tk screen height (if available) to cap the figure so it fits within the application window.
        Falls back to sensible defaults when no Tk root is reachable.
        """
        # Try to read screen dimensions from the Tk root
        screen_h_px: Optional[int] = None
        try:
            root = getattr(self._islat, "root", None)
            if root is not None:
                screen_h_px = root.winfo_screenheight()
        except Exception:
            pass

        if screen_h_px is None:
            # Fallback: assume a modest screen height
            screen_h_px = 900

        # Convert screen pixels to approximate matplotlib inches at the
        # logical DPI (100). Leave room for toolbars / window chrome.
        dpi = 100
        max_fig_h = min((screen_h_px - 160) / dpi, 14.0)
        # Ensure at least a reasonable minimum height
        max_fig_h = max(max_fig_h, 4.0)

        return (12, max_fig_h)

    def _rebuild_plot(self) -> None:
        """Refresh data and regenerate the composed plot.

        If the panel layout changed, the figure is rebuilt from scratch.
        If only data/molecules changed and the composed plot supports
        fast in-place updates, existing axes are updated via
        ``update_panels_inplace()`` for a significant speed-up (avoids
        ``fig.clf()`` and re-creating all subplot objects).

        When the plot type needs to change (e.g. the ``show_residuals``
        toggle was flipped while the view was inactive), the plot is
        rebuilt from scratch via :meth:`_create_plot`.
        """
        wave, flux, wave_obs = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        if self._plot is None:
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            return

        # Detect a plot-type mismatch (e.g. FSP is active but show_residuals is now True).
        # Force a full rebuild.
        show_residuals = self._pm.toggle_state.get("show_residuals", False)
        type_mismatch = (
            (show_residuals and not isinstance(self._plot, ResidualSpectrumPlot))
            or (not show_residuals and isinstance(self._plot, ResidualSpectrumPlot))
        )
        if type_mismatch:
            # Preserve the panel wavelength ranges so the user sees the same layout after the toggle.
            old_edges = getattr(self._plot, "_panel_edges", None)
            old_ends = getattr(self._plot, "_panel_ends", None)
            old_step = getattr(self._plot, "_step", None)
            old_xlim = (
                getattr(self._plot, "_xlim_start", None),
                getattr(self._plot, "_xlim_end", None),
            )

            self.span_selectors.clear()
            self._plot = self._create_plot()

            # Restore the panel layout from the previous plot so switching between FSP and RSP keeps the same wavelength ranges.
            if old_edges is not None and old_ends is not None:
                self._plot._panel_edges = old_edges
                self._plot._panel_ends = old_ends
                self._plot.n_panels = len(old_edges)
                self._plot._step = old_step
                if old_xlim[0] is not None:
                    self._plot._xlim_start = old_xlim[0]
                if old_xlim[1] is not None:
                    self._plot._xlim_end = old_xlim[1]

            self._plot.generate_plot()
            self._install_span_selectors()
            return

        # If the active plot is an RSP, keep its model_flux in sync.
        if isinstance(self._plot, ResidualSpectrumPlot):
            model_flux = self._compute_model_flux(wave_obs, wave)
            self._plot.model_flux = model_flux
            self._plot._model_flux_adj = model_flux

        # Try the fast-path: update_data + update_panels_inplace
        updater = getattr(self._plot, "update_data", None)
        if updater is not None:
            error_data = getattr(self._islat, "err_data", None)
            layout_changed = updater(
                wave_data=wave,
                flux_data=flux,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                error_data=error_data,
            )
        else:
            layout_changed = True

        if layout_changed:
            # Panel edges changed -- full rebuild
            self.span_selectors.clear()
            self._plot.generate_plot()
            self._install_span_selectors()
        else:
            # Layout unchanged -- fast in-place update of existing axes.
            inplace = getattr(self._plot, "update_panels_inplace", None)
            if inplace is not None:
                inplace()
            else:
                self._plot.generate_plot()

    # ==================================================================
    # Span selector (interactive-only feature)
    # ==================================================================
    def _install_span_selectors(self) -> None:
        """Add span selectors to every primary subplot for click-to-inspect."""
        self.span_selectors.clear()
        for idx, ax in self.subplots.items():
            span = SpanSelector(
                ax,
                lambda xmin, xmax, i=idx: self._on_span_select(xmin, xmax, i),
                direction="horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="lime"),
                interactive=True,
                drag_from_anywhere=True,
            )
            self.span_selectors[idx] = span

    def _on_span_select(self, xmin: float, xmax: float, subplot_index: int) -> None:
        if abs(xmax - xmin) < 0.001:
            return
        main_plot = self._islat.GUI.plot
        if hasattr(main_plot, "is_full_spectrum") and main_plot.is_full_spectrum:
            main_plot.toggle_full_spectrum()
            if hasattr(self._islat, "root"):
                self._islat.root.after(100, lambda: self._apply_selection(xmin, xmax))
            else:
                self._apply_selection(xmin, xmax)
        else:
            self._apply_selection(xmin, xmax)

    def _apply_selection(self, xmin: float, xmax: float) -> None:
        main_plot = self._islat.GUI.plot

        main_plot.current_selection = (xmin, xmax)

        selection_center = (xmin + xmax) / 2
        selection_width = xmax - xmin
        xlim_start = self._plot._xlim_start
        xlim_end = self._plot._xlim_end
        total_range = xlim_end - xlim_start
        min_padding = total_range * 0.025
        view_padding = max(selection_width * 2, min_padding)

        main_plot.ax1.set_xlim(
            selection_center - view_padding,
            selection_center + view_padding,
        )

        if hasattr(main_plot, "interaction_handler"):
            span = getattr(main_plot.interaction_handler, "span_selector", None)
            if span is not None:
                try:
                    span.set_visible(True)
                    span.extents = (xmin, xmax)
                    span.update()
                except Exception as exc:
                    debug_config.warning("full_spectrum_view", f"Could not set span extents: {exc}")

        main_plot.onselect(xmin, xmax)
        main_plot.canvas.draw_idle()

    # ==================================================================
    # Canvas helpers
    # ==================================================================
    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the :class:`FigureCanvasTkAgg`."""
        fig = self.fig
        if self._canvas is not None:
            if fig is not None and self._canvas.figure is fig:
                return
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if fig is not None:
            self._canvas = FigureCanvasTkAgg(fig, master=self._parent_frame)
            self._canvas.mpl_connect(
                "button_press_event", self._on_canvas_button_press
            )

    def _on_canvas_button_press(self, event: Any) -> None:
        """Route right-click events to :meth:`build_context_menu`."""
        if event.button != 3:
            return
        if self._canvas is None:
            return
        try:
            canvas_widget = self._canvas.get_tk_widget()
        except Exception:
            return
        menu = self.build_context_menu(event, canvas_widget)
        if menu is None:
            return
        try:
            x_root = canvas_widget.winfo_rootx() + int(event.x)
            y_root = canvas_widget.winfo_rooty() + int(
                canvas_widget.winfo_height() - event.y
            )
            menu.tk_popup(x_root, y_root)
        except Exception:
            pass
        finally:
            menu.grab_release()

    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Return a ``tk.Menu`` for the right-clicked panel, or ``None``."""
        try:
            import tkinter as tk
        except ImportError:
            return None

        pm = self._pm
        plot = self._plot

        # Identify which panel was clicked
        panel_idx = None
        if plot is not None:
            for idx, ax in self.subplots.items():
                if event.inaxes is ax:
                    panel_idx = idx
                    break

        menu = tk.Menu(canvas_widget, tearoff=0)

        if panel_idx is not None:
            try:
                xmin = float(plot._panel_edges[panel_idx])
                xmax = float(plot._panel_ends[panel_idx])
            except Exception:
                panel_idx = None

        if panel_idx is not None:
            def _open_in_three_panel():
                pm.islat.display_range = (xmin, xmax)
                # Mirror toggle_three_panel so the back-button works.
                pm._pre_threepanel_view_name = "Full Spectrum"
                pm.switch_view("Three Panel")
                try:
                    pm.ax1.set_xlim(xmin, xmax)
                    pm.match_display_range(match_y=True)
                except Exception as exc:
                    debug_config.warning(
                        "full_spectrum_view",
                        f"build_context_menu: range update failed: {exc}",
                    )
                '''# Fire the span-selector so line inspection populates.
                try:
                    ih = getattr(pm, "interaction_handler", None)
                    if ih is not None and ih.span_selector is not None:
                        ih.span_selector.extents = (xmin, xmax)
                        ih.span_selector.set_visible(True)
                    pm.onselect(xmin, xmax)
                except Exception as exc:
                    debug_config.warning(
                        "full_spectrum_view",
                        f"build_context_menu: onselect failed: {exc}",
                    )'''

            menu.add_command(
                label="Open in Three Panel view",
                command=_open_in_three_panel,
            )

        self._append_save_figure_item(menu)
        return menu

    # ==================================================================
    # PlotView lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        self._parent_frame = parent_frame

        if not self._initialised:
            # First activation - build the composed plot
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            self._initialised = True
            self._needs_refresh = False
        elif self._needs_refresh:
            # Data changed while we were inactive - full refresh
            old_fig = self.fig
            self._rebuild_plot()
            self._needs_refresh = False
            if self.fig is not old_fig:
                if self._canvas is not None:
                    self._canvas.get_tk_widget().destroy()
                    self._canvas = None
        # else: simple view toggle - just repack the existing canvas

        self._ensure_canvas()
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Sync theme in case it changed while the other view was active.
        self.apply_theme(self._pm.theme)

        # Reconcile overlays with the controller's toggle dict
        self.sync_toggle_state(self._pm.toggle_state)

        # Register view-specific controls with the ControlBus
        self._register_control_fields()

        if self._canvas is not None:
            self._canvas.draw_idle()

    def _register_control_fields(self) -> None:
        """Register this view's :class:`ControlField` objects on the :class:`ControlBus`.

        Called from :meth:`activate`.  Safe to call even when the bus is not
        yet wired (e.g. in lightweight test environments) - missing surfaces
        are silently ignored.
        """
        bus = getattr(self._pm, 'control_bus', None)
        if bus is None:
            return

        from iSLAT.Modules.GUI.ControlField import EntryField, DisplayRangeField, ToggleField

        # First unregister any stale fields from a previous activation
        bus.unregister_owner(self)

        # --- ControlPanel fields ---
        bus.register_many([
            EntryField(
                "n_panels", "N Panels:", self._get_n_panels, self._set_n_panels,
                datatype="int", width=5, owner=self,
                tip="Number of panels in the\nfull-spectrum stacked layout",
            ),
            DisplayRangeField(
                "_fsv_display_range",
                getter=self._get_display_range,
                setter=self._set_display_range,
                owner=self,
            ),
        ], "control_panel")

        # --- TopBar toggle ---
        pm = self._pm

        def _residuals_getter() -> bool:
            return pm.toggle_state.get("show_residuals", False)

        def _residuals_setter(value: bool) -> None:
            if value != pm.toggle_state.get("show_residuals", False):
                pm.toggle_residuals()

        bus.register(
            ToggleField(
                "show_residuals", "Residuals",
                getter=_residuals_getter,
                setter=_residuals_setter,
                owner=self,
                tip="Toggle residual sub-panels on/off\nin full spectrum mode\nKeybind: R",
            ),
            "top_bar",
        )

        bus.register(
            ToggleField(
                "data_density", "Data Density X-Scale",
                getter=self._get_data_density,
                setter=self._set_data_density,
                owner=self,
                tip=(
                    "Toggle X-axis scaling mode.\n"
                    "OFF: equal wavelength width per panel (default).\n"
                    "ON:  equal data-point count per panel (DATA_DENSITY)."
                ),
            ),
            "top_bar",
        )

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()

        # Unregister all fields this view registered across all surfaces
        bus = getattr(self._pm, 'control_bus', None)
        if bus is not None:
            bus.unregister_owner(self)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self, theme: dict) -> None:
        """Apply *theme* to the composed plot, figure, and canvas widget.

        Propagates the theme to:
        - the composed :class:`StackedSpectralPanel` (``self._plot.theme``)
        - every axes in the figure via :meth:`BasePlot.apply_theme_to_figure`
        - the Tk canvas widget background
        """
        if self._plot is not None:
            self._plot.theme = theme
            self._plot.apply_theme_to_figure()
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().configure(
                    bg=theme.get("background", "#181A1B")
                )
            except Exception:
                pass
            self._canvas.draw_idle()

    # ==================================================================
    # Core rendering
    # ==================================================================
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not self._initialised:
            self._needs_refresh = True
            return

        old_fig = self.fig
        self._rebuild_plot()
        self._needs_refresh = False

        if self.fig is not old_fig:
            if self._canvas is not None:
                self._canvas.get_tk_widget().pack_forget()
                self._canvas.get_tk_widget().destroy()
                self._canvas = None
            self._ensure_canvas()
            if self._canvas is not None and self._parent_frame is not None:
                self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Apply toggle state overlays after regeneration
        self.sync_toggle_state(self._pm.toggle_state)
        self.draw()

    # ------------------------------------------------------------------
    # Selection & line-inspection (no-ops - FSV doesn't own these panels)
    # ------------------------------------------------------------------
    def on_selection(self, xmin: float, xmax: float) -> None:
        """No-op - FullSpectrumView uses its own span-selector flow."""
        pass

    def clear_selection(self) -> None:
        """No-op - FullSpectrumView has no line-inspection panel."""
        pass

    def clear_active_lines(self) -> None:
        """No-op - FullSpectrumView has no active-line artists."""
        pass

    # ------------------------------------------------------------------
    # Molecule lifecycle callbacks
    # ------------------------------------------------------------------
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Mark stale - FSV doesn't have per-molecule panels to update."""
        self._needs_refresh = True

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Re-render immediately if this view is active, otherwise mark stale."""
        if parameter_name == 'is_visible':
            return

        if not self._initialised or self._plot is None:
            # Not yet rendered - just mark stale for the next activate().
            self._needs_refresh = True
            return

        mol_dict = getattr(self._islat, 'molecules_dict', None)
        if mol_dict is None:
            self._needs_refresh = True
            return

        molecule = mol_dict.get(molecule_name)
        if molecule is None or not getattr(molecule, 'is_visible', True):
            # Hidden molecules don't need an immediate repaint; mark stale
            # so they re-render correctly when made visible again.
            self._needs_refresh = True
            return

        # Active view with a visible molecule - update in-place now.
        self._plot.update_panels_inplace()
        # update_panels_inplace always rebuilds the legend; re-hide it when
        # the user has toggled the legend off.
        if not self._pm.legend_toggle:
            self.toggle_legend(False)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Mark stale so the next activate() does a full rebuild."""
        self._needs_refresh = True

    # ------------------------------------------------------------------
    # Artist-manipulation helpers (replace removed PlotRenderer calls)
    # ------------------------------------------------------------------
    def _set_molecule_visibility(
        self, molecule_name: str, is_visible: bool, ax: "Axes"
    ) -> bool:
        """Toggle visibility of every Line2D tagged with *molecule_name* on *ax*.

        Returns ``True`` if at least one artist was found and toggled.
        """
        found = False
        for line in ax.lines:
            if getattr(line, "_molecule_name", None) == molecule_name:
                line.set_visible(is_visible)
                found = True
        return found

    def _remove_molecule_lines(self, molecule_name: str, ax: "Axes") -> None:
        """Remove all Line2D artists tagged with *molecule_name* from *ax*."""
        to_remove = [
            line for line in ax.lines
            if getattr(line, "_molecule_name", None) == molecule_name
        ]
        for line in to_remove:
            line.remove()

    def _render_molecule_spectrum(
        self, molecule: "Molecule", wave_data: "np.ndarray", ax: "Axes"
    ) -> None:
        """Plot *molecule*'s spectrum onto *ax* using the composed plot's styling.

        Falls back to a full ``update_panels_inplace()`` rebuild when the composed plot object is not yet available.
        """
        if self._plot is None:
            return
        # Delegate to FullSpectrumPlot's mol-cache helper so color/style
        # stays consistent with the rest of the composed plot.
        # _build_mol_cache returns a list of (lam, flux, color, label, name) tuples;
        # convert to a dict keyed by name for O(1) lookup.
        mol_cache_list, _labels, _colors = self._plot._build_mol_cache()
        mol_cache = {tup[4]: tup[:4] for tup in mol_cache_list}
        entry = mol_cache.get(getattr(molecule, "name", None))
        if entry is None:
            # Molecule not in cache - trigger a full inplace refresh instead.
            self._plot.update_panels_inplace()
            if self._canvas is not None:
                self._canvas.draw_idle()
            return
        m_lam, m_flux, m_color, m_label = entry
        xlim = ax.get_xlim()
        m_mask = (m_lam >= xlim[0]) & (m_lam <= xlim[1])
        if np.any(m_mask):
            line, = ax.plot(
                m_lam[m_mask], m_flux[m_mask],
                linestyle="--", color=m_color,
                alpha=self._plot._get_theme_value("full_spectrum_model_alpha", 0.8),
                linewidth=self._plot.get_molecule_linewidth(
                    molecule,
                    self._plot._get_theme_value("full_spectrum_model_linewidth", 0.8),
                ),
                label=m_label,
                zorder=self._plot._get_theme_value("zorder_model", 3),
            )
            line._molecule_name = getattr(molecule, "name", None)

    # ------------------------------------------------------------------
    def on_molecule_visibility_changed(
        self,
        molecule_name: str,
        is_visible: bool,
        molecules_dict: "MoleculeDict",
        wave_data: Any,
        active_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
        force_rerender: bool = False,
    ) -> None:
        if not self._initialised or not self.subplots:
            return

        # Use the same RV-corrected wavelengths as the rendered panels
        # for display positioning, and observer-frame wavelengths for
        # model computation (get_summed_flux expects observer frame).
        rv_wave = self._plot.wave_data if self._plot is not None else wave_data
        wave_obs = (
            getattr(self._plot, "wave_data_obs", self._plot.wave_data)
            if self._plot is not None
            else wave_data
        )

        # For molecule artist operations use the primary (spectrum) axes only.
        # For FSP these are the same as all axes; for RSP this avoids placing molecule overlays on residual sub-panels.
        is_rsp = isinstance(self._plot, ResidualSpectrumPlot)
        primary_axes = list(self.subplots.values())

        # 1. Toggle molecule line visibility per subplot.
        #    When force_rerender is True (parameters changed while hidden),
        #    destroy the stale artists and re-create them from fresh data.
        if force_rerender and is_visible:
            molecule = molecules_dict.get(molecule_name)
            for ax in primary_axes:
                self._remove_molecule_lines(molecule_name, ax)
                if molecule is not None:
                    self._render_molecule_spectrum(molecule, rv_wave, ax)
        else:
            # Try fast artist-toggle first.
            toggled = False
            for ax in primary_axes:
                if self._set_molecule_visibility(molecule_name, is_visible, ax):
                    toggled = True

            # If turning ON but no artists exist (e.g. after a full rebuild while the molecule was hidden), create them from scratch.
            if is_visible and not toggled:
                molecule = molecules_dict.get(molecule_name)
                if molecule is not None:
                    for ax in primary_axes:
                        self._render_molecule_spectrum(molecule, rv_wave, ax)

        # 2. Recompute summed spectrum / model and update residuals + chi^2.
        if is_rsp:
            # RSP's update_panels_inplace recomputes the visible-only model
            # flux, redraws the summed fill on spectrum panels, redraws all
            # residual panels, and refreshes the per-panel and global chi^2
            # annotations - all in one call.
            self._plot.update_panels_inplace()
        else:
            try:
                summed_wavelengths, summed_flux = molecules_dict.get_summed_flux(
                    wave_obs, visible_only=True,
                )
            except Exception as exc:
                debug_config.warning("full_spectrum_view", f"Could not compute summed flux: {exc}")
                summed_wavelengths = rv_wave if rv_wave is not None else np.array([])
                summed_flux = np.zeros_like(summed_wavelengths)

            summed_visible = self._pm.summed_toggle and bool(
                molecules_dict.get_visible_molecules(return_objects=True)
            )

            for ax in primary_axes:
                for coll in ax.collections[:]:
                    if hasattr(coll, "_islat_summed"):
                        coll.remove()
                xlim = ax.get_xlim()
                mask = (summed_wavelengths >= xlim[0]) & (summed_wavelengths <= xlim[1])
                if np.any(mask) and np.any(summed_flux[mask] > 0):
                    fill = ax.fill_between(
                        summed_wavelengths[mask], 0, summed_flux[mask],
                        color=self._plot._get_theme_value("summed_spectra_color", "lightgray"),
                        alpha=1.0, label="Sum",
                        zorder=self._plot._get_theme_value("zorder_summed", 1),
                    )
                    fill._islat_summed = True
                    fill.set_visible(summed_visible)

        # 3. Rebuild legend
        self._update_full_spectrum_legend(molecules_dict)
        self.draw()

    # ==================================================================
    # Toggle helpers - overrides and specializations of ToggleMixin
    # ==================================================================
    def _set_legend_visibility(self, visible: bool) -> None:
        """Override ToggleMixin hook to use the composed plot's legend strategy."""
        legend_ax = self.subplots.get(0)
        if legend_ax is not None and self._plot is not None:
            self._plot.legend_strategy.update_visibility(legend_ax, visible)

    def sync_toggle_state(self, toggle_state: dict) -> None:
        """Override to refresh saved-line data from disk before adding artists."""
        if not self._toggle_ready():
            return

        # Atomic lines
        self._remove_atomic_line_artists()
        if toggle_state.get("atomic_lines", False):
            self._add_atomic_line_artists()

        # Saved lines - always refresh from disk
        self._remove_saved_line_artists()
        if toggle_state.get("saved_lines", False):
            self._set_saved_line_data(self._load_saved_line_data())
            self._add_saved_line_artists()

        # Summed spectrum
        self._set_summed_visibility(toggle_state.get("summed", True))

        # Legend
        self._set_legend_visibility(toggle_state.get("legend", True))

        self.draw()

    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Override to use the composed plot's legend strategy."""
        if not self._toggle_ready():
            return
        legend_ax = self.subplots.get(0)
        if legend_ax is None:
            return
        legend = legend_ax.get_legend()
        if legend is not None:
            if visible is not None:
                self._plot.legend_strategy.update_visibility(legend_ax, visible)
            else:
                self._plot.legend_strategy.update_visibility(
                    legend_ax, not legend.get_visible(),
                )
        self.draw()

    # toggle_summed_spectrum, toggle_saved_lines, toggle_atomic_lines are inherited from ToggleMixin - no override needed.

    def toggle_residuals(self, show: bool) -> None:
        """Switch between :class:`FullSpectrumPlot` and :class:`ResidualSpectrumPlot`.

        Triggers a full plot rebuild because the axes layout changes (RSP has paired spectrum + residual sub-panels).
        The canvas is destroyed and re-created so the Tk widget reflects the new figure dimensions.
        """
        if not self._initialised:
            return

        # Preserve panel layout from the current plot so the user
        # sees the same wavelength ranges after the toggle.
        old_edges = getattr(self._plot, "_panel_edges", None)
        old_ends = getattr(self._plot, "_panel_ends", None)
        old_step = getattr(self._plot, "_step", None)
        old_xlim = (
            getattr(self._plot, "_xlim_start", None),
            getattr(self._plot, "_xlim_end", None),
        )

        # Force a full teardown so the new figure replaces the old one
        old_fig = self.fig
        self.span_selectors.clear()
        self._plot = self._create_plot()

        # Restore the panel layout from the previous plot.
        if old_edges is not None and old_ends is not None:
            self._plot._panel_edges = old_edges
            self._plot._panel_ends = old_ends
            self._plot.n_panels = len(old_edges)
            self._plot._step = old_step
            if old_xlim[0] is not None:
                self._plot._xlim_start = old_xlim[0]
            if old_xlim[1] is not None:
                self._plot._xlim_end = old_xlim[1]

        self._plot.generate_plot()
        self._install_span_selectors()

        # If the figure object changed (it will), rebuild the canvas
        if self.fig is not old_fig:
            if self._canvas is not None:
                self._canvas.get_tk_widget().pack_forget()
                self._canvas.get_tk_widget().destroy()
                self._canvas = None
            self._ensure_canvas()
            if self._canvas is not None and self._parent_frame is not None:
                self._canvas.get_tk_widget().pack(
                    fill="both", expand=True, padx=0, pady=0,
                )

        # Re-apply overlays
        self.sync_toggle_state(self._pm.toggle_state)
        self.draw()

    # ==================================================================
    # Overlay artist helpers (interactive-only)
    # ==================================================================
    def _add_saved_line_artists(self) -> None:
        """Add saved-line annotations to every panel in the composed plot.

        Delegates to :meth:`StackedSpectralPanel.plot_saved_lines` which
        iterates over every :class:`SpectralPanel` so each panel places labels relative to its own y-limits.
        """
        if self._plot is None:
            return

        # Reload line data from disk so we always have the latest
        if self.line_data is None:
            self.line_data = self._load_line_data()
        if self.line_data is None:
            return

        # Ensure the 'line' column exists (BasePlot expects it)
        if "line" not in self.line_data.columns:
            self.line_data["line"] = [""] * len(self.line_data)

        self._plot.plot_saved_lines(self.line_data)

    def _remove_saved_line_artists(self) -> None:
        """Remove all ``_islat_saved_line`` artists."""
        if self._plot is not None:
            self._plot.remove_saved_lines()

    def _add_atomic_line_artists(self) -> None:
        """Add atomic-line annotations to every panel in the composed plot.

        Delegates to :meth:`StackedSpectralPanel.plot_atomic_lines` which iterates over
        every :class:`SpectralPanel` so each panel places labels relative to its own y-limits.
        """
        if self._plot is None:
            return

        atomic_data = load_atomic_lines()
        self._plot.plot_atomic_lines(atomic_data)

    def _remove_atomic_line_artists(self) -> None:
        """Remove all ``_islat_atomic_line`` artists."""
        if self._plot is not None:
            self._plot.remove_atomic_lines()

    # ==================================================================
    # File output  (overrides PlotView.save_figure)
    # ==================================================================
    def _create_standalone_plot(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        wave_obs: np.ndarray,
        *,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        figsize: Tuple[float, float] = (12, 16),
        xlim_range: Optional[Tuple[float, float]] = None,
        n_panels: Optional[int] = None,
    ) -> StackedSpectralPanel:
        """Build a fresh standalone :class:`StackedSpectralPanel` for export.

        Subclasses can override this to return a different stacked-panel implementation (e.g. :class:`ResidualSpectrumPlot`) when the user saves the figure from a residual-backed view.

        The returned plot is **not yet rendered**; the caller is responsible for calling :meth:`generate_plot`.
        """
        kw: dict = dict(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
            line_list=line_list,
            atomic_lines=atomic_lines,
            figsize=figsize,
            wave_data_obs=wave_obs,
            theme=self._pm.theme,
            xlim_range=xlim_range,
        )
        if n_panels is not None:
            kw["n_panels"] = n_panels
        return FullSpectrumPlot(**kw)

    def save_figure(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        dpi: int | None = None,
        rasterized: bool = False,
        **kwargs,
    ) -> str | None:
        """
        Export the full spectrum to a file using a **fresh standalone figure** that has the current toggle state baked in.

        This avoids side-effects on the interactive GUI figure.
        """
        from tkinter import filedialog

        if save_path is None:
            try:
                default_name = Path(self._islat.loaded_spectrum_file).stem + f"_full_output.{file_format}"
            except Exception:
                default_name = f"full_output.{file_format}"
            save_path = filedialog.asksaveasfilename(
                title="Save Spectrum Output",
                defaultextension=f".{file_format}",
                initialfile=default_name,
                initialdir=absolute_data_files_path,
                filetypes=[
                    (f"{file_format.upper()} files", f"*.{file_format}"),
                ],
            )
        if not save_path:
            return None

        # Build toggle-state-aware kwargs for the standalone plot
        ts = self._pm.toggle_state

        line_list_df: Optional[pd.DataFrame] = None
        if ts.get("saved_lines", False):
            line_list_df = self.line_data

        atomic_lines_df: Optional[pd.DataFrame] = None
        if ts.get("atomic_lines", False):
            atomic_lines_df = load_atomic_lines()

        # Create standalone figure via the overridable factory method
        wave, flux, wave_obs = self._load_spectrum_data()

        # Use the live plot's wavelength bounds and panel count so the saved
        # figure matches exactly what the user sees in the GUI.
        xlim_range: Optional[Tuple[float, float]] = None
        n_panels: Optional[int] = None
        if self._plot is not None:
            x_start = getattr(self._plot, '_xlim_start', None)
            x_end   = getattr(self._plot, '_xlim_end',   None)
            if x_start is not None and x_end is not None:
                xlim_range = (float(x_start), float(x_end))
            n_panels = getattr(self._plot, 'n_panels', None)
        # Fall back to molecules' global_wavelength_range when the interactive
        # plot is not yet rendered (e.g. called before first activation).
        if xlim_range is None:
            mol_dict = getattr(self._islat, 'molecules_dict', None)
            mol_range = getattr(mol_dict, '_global_wavelength_range', None) if mol_dict else None
            if mol_range is not None:
                xlim_range = (float(mol_range[0]), float(mol_range[1]))

        standalone = self._create_standalone_plot(
            wave, flux, wave_obs,
            line_list=line_list_df,
            atomic_lines=atomic_lines_df,
            figsize=(12, 16),
            xlim_range=xlim_range,
            n_panels=n_panels,
        )
        standalone.generate_plot()

        # Respect summed toggle -- iterate all axes generically
        if not ts.get("summed", True):
            all_axes: list = []
            raw = getattr(standalone, "subplots", {})
            for val in raw.values():
                if isinstance(val, tuple):
                    all_axes.extend(val)
                else:
                    all_axes.append(val)
            for ax in all_axes:
                for coll in ax.collections[:]:
                    if hasattr(coll, "_islat_summed"):
                        coll.set_visible(False)

        if rasterized:
            for ax in standalone.fig.axes:
                ax.set_rasterized(True)

        save_kw: dict = {"bbox_inches": "tight", "format": file_format}
        if dpi is not None:
            save_kw["dpi"] = dpi
        elif rasterized:
            save_kw["dpi"] = 300
        else:
            save_kw["dpi"] = 300  # high-quality default for all exports
        save_kw.update(kwargs)

        standalone.fig.savefig(save_path, **save_kw)
        standalone.close()

        # Notify user via data field
        if hasattr(self._islat, "GUI") and hasattr(self._islat.GUI, "data_field"):
            self._islat.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

        return save_path

    # ==================================================================
    # Canvas / drawing
    # ==================================================================
    def draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas  # type: ignore[return-value]

    def get_figure(self) -> "Figure":
        return self.fig  # type: ignore[return-value]

    # ==================================================================
    # Internal legend helper
    # ==================================================================
    def _get_legend(self) -> Optional["Legend"]:
        if self.subplots and 0 in self.subplots:
            return self.subplots[0].legend_
        return None

    def _update_full_spectrum_legend(self, molecules_dict: "MoleculeDict") -> None:
        visible_mols = molecules_dict.get_visible_molecules(return_objects=True)
        mol_labels = [BasePlot.get_molecule_display_name(mol) for mol in visible_mols]
        mol_colors = [BasePlot.get_molecule_color(mol) for mol in visible_mols]

        legend_ax = self.subplots.get(0)
        if legend_ax is None:
            return

        self._plot.legend_strategy.build_legend(
            legend_ax, self._plot.fig, mol_labels, mol_colors,
        )

        # Respect the legend toggle state from the controller
        if not self._pm.legend_toggle:
            self._plot.legend_strategy.update_visibility(legend_ax, False)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy(self) -> None:
        """Permanently dispose of the full spectrum plot and canvas."""
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._plot is not None:
            self._plot.close()
            self._plot = None
        self._initialised = False
        self._needs_refresh = True

# ======================================================================
# Backward-compatible top-level function
# ======================================================================
def output_full_spectrum(islat_ref: Any, rasterized: bool = False) -> str | None:
    """
    Create a fresh full-spectrum PDF and a companion parameters CSV.

    This is the backward-compatible replacement for the function that previously lived in ``OutputFullSpectrum.py``.
    """
    # Use in-memory data instead of re-reading the CSV from disk.
    # Delegate the stellar RV correction to MoleculeDict so the formula lives in one place.
    wave_obs = np.array(islat_ref.wave_data_original, copy=True)
    wave = islat_ref.molecules_dict.apply_stellar_rv(wave_obs)
    flux = np.array(islat_ref.flux_data, copy=True)

    # Read toggle state if available
    ts: dict = {}
    theme: Optional[dict] = None
    if hasattr(islat_ref, "GUI") and hasattr(islat_ref.GUI, "plot"):
        ts = getattr(islat_ref.GUI.plot, "toggle_state", {})
        theme = getattr(islat_ref.GUI.plot, "theme", None)

    line_list_df: Optional[pd.DataFrame] = None
    if ts.get("saved_lines", False):
        try:
            line_list_df = pd.read_csv(islat_ref.input_line_list, sep=",")
        except Exception:
            pass

    atomic_lines_df: Optional[pd.DataFrame] = None
    if ts.get("atomic_lines", False):
        atomic_lines_df = load_atomic_lines()

    # Create standalone plot - uses BasePlot._ensure_figure (non-pyplot MplFigure)
    standalone_kwargs = dict(
        wave_data=wave,
        flux_data=flux,
        molecules=islat_ref.molecules_dict,
        line_list=line_list_df,
        atomic_lines=atomic_lines_df,
        figsize=(12, 16),
        wave_data_obs=wave_obs,
    )
    if theme is not None:
        standalone_kwargs["theme"] = theme
    # Respect the GUI's wavelength range so the saved figure matches
    # what is shown on screen rather than always spanning all data.
    mol_range = getattr(islat_ref.molecules_dict, '_global_wavelength_range', None)
    if mol_range is not None:
        standalone_kwargs["xlim_range"] = (float(mol_range[0]), float(mol_range[1]))
    # Respect the GUI's panel count.
    try:
        gui_plot = islat_ref.GUI.plot
        live_fsv = getattr(gui_plot, '_full_spectrum_view', None) or getattr(gui_plot, 'current_view', None)
        live_plot = getattr(live_fsv, '_plot', None)
        if live_plot is None:
            # ThreePanelView / direct attribute
            live_plot = getattr(gui_plot, '_plot', None)
        if live_plot is not None:
            n_panels = getattr(live_plot, 'n_panels', None)
            if n_panels is not None:
                standalone_kwargs["n_panels"] = n_panels
    except Exception:
        pass
    standalone = FullSpectrumPlot(**standalone_kwargs)
    standalone.generate_plot()

    # Respect summed toggle
    if not ts.get("summed", True):
        for ax in standalone.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.set_visible(False)

    from tkinter import filedialog
    default_name = Path(islat_ref.loaded_spectrum_file).stem + "_full_output.pdf"
    save_path = filedialog.asksaveasfilename(
        title="Save Spectrum Output",
        defaultextension=".pdf",
        initialfile=default_name,
        initialdir=absolute_data_files_path,
        filetypes=[("PDF files", "*.pdf")],
    )
    if not save_path:
        standalone.close()
        return None

    if rasterized:
        for ax in standalone.fig.axes:
            ax.set_rasterized(True)

    standalone.fig.savefig(
        save_path,
        bbox_inches="tight",
        format="pdf",
        dpi=300 if rasterized else None,
    )
    standalone.close()

    # Write companion parameters CSV
    file_name = str(Path(save_path).with_suffix("")) + "_parameters.csv"
    ifh.write_molecules_to_csv(
        islat_ref.molecules_dict,
        file_path=dirname(save_path),
        file_name=file_name,
    )

    if hasattr(islat_ref, "GUI") and hasattr(islat_ref.GUI, "data_field"):
        islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

    return save_path