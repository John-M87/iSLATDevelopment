import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List, Tuple, Any, TYPE_CHECKING

import iSLAT.Constants as c
from .CachedLayout import CachedConstrainedLayoutEngine
from .HoldableCanvas import HoldableFigureCanvasTkAgg
from .LegendStrategy import StandardLegend
from .MainPlotGrid import MainPlotGrid
from .Theming import apply_canvas_theme, apply_figure_theme
from .PlotView import (
    PlotView, WavelengthMarkerStoreMixin, data_synced_y_limits, data_y_range,
    fit_y_limits_to_data, normalize_y_data_sync, notify_user,
)
from .ThreePanelView import ThreePanelView
from .FullSpectrumView import FullSpectrumView, SAMPLE_STACK_DEFAULTS
from .PopulationDiagramView import PopulationDiagramView
from .LineInspectionView import LineInspectionView
from .FitLinesPlotGridView import FitLinesPlotGridView
from .OpticalDepthView import OpticalDepthView, TAU_OPTION_DEFAULTS
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
from iSLAT.Modules.GUI.ControlSurface import ControlBus
from iSLAT.Modules.GUI.Widgets.iSLATToolbar import iSLATNavigationToolbar, ConfigureSubplotsSurface

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

from .PlotDebug import debug_config
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer

class iSLATPlot(WavelengthMarkerStoreMixin):
    """
    Main plotting class for iSLAT spectroscopy tool.
    
    This class coordinates between specialized modules to provide comprehensive
    plotting functionality including spectrum visualization, line analysis,
    population diagrams, and interactive features.
    
    Architecture:
    - MainPlotGrid / ThreePanelView: Handles matplotlib rendering via borrowed-axes
    - InteractionHandler: Processes mouse/keyboard interactions
    - FittingEngine: Handles line fitting operations
    - LineAnalyzer: Provides line detection and analysis capabilities
    
    Data Sources:
    - Uses MoleculeLine objects for line data access
    - Leverages Spectrum and Intensity classes for computation
    - Integrates with MoleculeDict for parameter management
    """
    def __init__(self, parent_frame, wave_data, flux_data, theme, islat_class_ref):
        self.theme = theme
        self.islat = islat_class_ref

        self.atomic_lines = []
        self.saved_lines = []

        # Single source of truth for every overlay toggle.
        # Views read this dict on activate() to reconcile their visual state.
        self.toggle_state: dict = {
            "atomic_lines": False,
            "saved_lines":  False,
            "summed":       True,
            "legend":       True,
            "show_residuals": False,
            "data_density": False,       # XScaling.DATA_DENSITY when True
            "skip_gaps":    False,       # GapMode.SKIP in the stacked (multi-panel) views when True
            "current_selection": None,   # (xmin, xmax) or None
            "y_autoscale":  True,        # main-plot Y axis follows the data in view
            "y_autoscale_source": "data",  # "data" or "model": what auto-scale fits to
            "y_data_sync":  None,        # None or "global" / "view" / "panel": auto-scale to the data's exact min / max
            "y_limits":     None,        # manual (ymin, ymax) used when y_autoscale is False
            "observed":     True,        # observed-spectrum artists visible
            "wavelength_markers": [],    # user marks: [{"start", "end", "label", "color"}, ...]
            **TAU_OPTION_DEFAULTS,       # Optical Depth view: log axis, profile / stem, stacking
            **SAMPLE_STACK_DEFAULTS,     # Full Spectrum view: stack the sample's spectra (layout, models)
        }

        # Session-only size overrides for full-spectrum exports (inches; None = default).
        # Read / written through FullSpectrumView.get_export_settings / set_export_settings.
        self.full_spectrum_export_settings: dict = {"width": None, "panel_height": None}
        # Session-only font sizes for the same exports (points per kind of text; None = the plot's own).
        # Read / written through FullSpectrumView.get_export_fonts / set_export_fonts.
        self.full_spectrum_export_fonts: dict = {"ticks": None, "labels": None, "legend": None, "text": None}

        #self.fig = plt.Figure(figsize=(15, 8.5))
        # Constrained layout, re-run only when something that sizes the margins changes.
        self.fig = plt.Figure(layout=CachedConstrainedLayoutEngine())
        # Use shared 3-panel layout helper from MainPlotGrid
        self.ax1, self.ax2, self.ax3 = MainPlotGrid.create_three_panel_axes(self.fig)

        self.ax1.set_title("Full Spectrum with Line Inspection")
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax2.set_title("Line inspection plot")
        # Use placeholder title - will be updated when data is initialized
        self.ax3.set_title("Population diagram")

        self.parent_frame = parent_frame

        self.canvas = HoldableFigureCanvasTkAgg(self.fig, master=self.parent_frame)
        
        # Apply theme to matplotlib figure and toolbar
        #self._apply_plot_theming()

        # Initialize the modular classes
        self.interaction_handler = InteractionHandler(self)
        self.fitting_engine = FittingEngine()
        self.line_analyzer = LineAnalyzer()

        # --- View strategy pattern ---
        # The active_view is the current rendering strategy.
        # ThreePanelView delegates to the existing axes via MainPlotGrid.
        # FullSpectrumView provides the self-contained multi-panel full spectrum layout.
        self._three_panel_view: PlotView = ThreePanelView(self)
        self._full_spectrum_view: PlotView = FullSpectrumView(self)
        self._population_diagram_view: PlotView = PopulationDiagramView(self)
        self._line_inspection_view: PlotView = LineInspectionView(self)
        self._fit_lines_grid_view: FitLinesPlotGridView = FitLinesPlotGridView(self)
        self._optical_depth_view: OpticalDepthView = OpticalDepthView(self)
        self.active_view: PlotView = self._three_panel_view
        self.is_full_spectrum: bool = False
        # Name of the view that was active before entering Full Spectrum mode,
        # used to restore the correct view when toggling back.
        self._pre_fullspectrum_view_name: str = "Three Panel"

        # Ordered dict of all switchable views (name → PlotView).
        # The order determines the order shown in the Views dropdown.
        self._views: dict = {
            "Three Panel": self._three_panel_view,
            "Full Spectrum": self._full_spectrum_view,
            "Optical Depth": self._optical_depth_view,
            "Population Diagram": self._population_diagram_view,
            "Line Inspection": self._line_inspection_view,
            "Line Grid": self._fit_lines_grid_view,
        }

        # View-change callbacks - notified when active_view switches
        self._view_change_callbacks: list = []

        # --- ControlBus: central registry for surface-agnostic UI controls ---
        # Surfaces (ControlPanelSurface, TopBarSurface, ViewsMenuSurface) are registered by GUI.create_window()
        # after ControlPanel and TopBar are constructed.
        self.control_bus = ControlBus()
        self._configure_subplots_surface = ConfigureSubplotsSurface()
        self.control_bus.register_surface("configure_subplots", self._configure_subplots_surface)

        # Set up interaction handler callbacks
        self.interaction_handler.set_span_select_callback(self.onselect)
        self.interaction_handler.set_click_callback(self.on_click)
        
        # self.toolbar.pack(side="top", fill="x", padx=0, pady=0)
        # Not drawn here: the canvas is still at matplotlib's default size.
        # The first draw comes once the window is laid out (GUI.start).
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        self.selected_wave = None
        self.fit_result = None

        # Initial data and model computation using new data structures
        self.summed_flux = np.array([])
        
        # Register callbacks for parameter and molecule changes
        self._register_update_callbacks()
    
    # ------------------------------------------------------------------
    # Backward-compatible properties - read / write the toggle_state dict
    # ------------------------------------------------------------------
    @property
    def atomic_toggle(self) -> bool:
        return self.toggle_state["atomic_lines"]

    @atomic_toggle.setter
    def atomic_toggle(self, value: bool) -> None:
        self.toggle_state["atomic_lines"] = value

    @property
    def line_toggle(self) -> bool:
        return self.toggle_state["saved_lines"]

    @line_toggle.setter
    def line_toggle(self, value: bool) -> None:
        self.toggle_state["saved_lines"] = value

    @property
    def summed_toggle(self) -> bool:
        return self.toggle_state["summed"]

    @summed_toggle.setter
    def summed_toggle(self, value: bool) -> None:
        self.toggle_state["summed"] = value

    @property
    def legend_toggle(self) -> bool:
        return self.toggle_state["legend"]

    @legend_toggle.setter
    def legend_toggle(self, value: bool) -> None:
        self.toggle_state["legend"] = value

    @property
    def residual_toggle(self) -> bool:
        return self.toggle_state["show_residuals"]

    @residual_toggle.setter
    def residual_toggle(self, value: bool) -> None:
        self.toggle_state["show_residuals"] = value

    @property
    def y_autoscale_toggle(self) -> bool:
        return self.toggle_state.get("y_autoscale", True)

    @y_autoscale_toggle.setter
    def y_autoscale_toggle(self, value: bool) -> None:
        self.toggle_state["y_autoscale"] = bool(value)

    @property
    def y_autoscale_source(self) -> str:
        return self.toggle_state.get("y_autoscale_source", "data")

    @y_autoscale_source.setter
    def y_autoscale_source(self, value: str) -> None:
        self.toggle_state["y_autoscale_source"] = "model" if str(value).lower() == "model" else "data"

    @property
    def y_data_sync(self) -> Optional[str]:
        """``"global"``, ``"view"`` or ``"panel"`` while auto-scale syncs to the data's min / max, else ``None``."""
        return normalize_y_data_sync(self.toggle_state.get("y_data_sync"))

    @y_data_sync.setter
    def y_data_sync(self, value: Optional[str]) -> None:
        self.toggle_state["y_data_sync"] = normalize_y_data_sync(value)

    @property
    def observed_toggle(self) -> bool:
        return self.toggle_state.get("observed", True)

    @observed_toggle.setter
    def observed_toggle(self, value: bool) -> None:
        self.toggle_state["observed"] = bool(value)

    def create_toolbar(self, frame):
        self.toolbar = iSLATNavigationToolbar(
            self.canvas,
            window=frame,
            configure_subplots_surface=self._configure_subplots_surface,
        )
        self.sync_toolbar_to_active_view()
        return self.toolbar

    def sync_toolbar_to_active_view(self, reset_history: bool = False) -> None:
        """Point the navigation toolbar (Pan / Zoom / Home / …) at the active view's canvas."""
        toolbar = getattr(self, "toolbar", None)
        view = getattr(self, "active_view", None)
        if toolbar is None or view is None or not hasattr(toolbar, "set_canvas"):
            return
        try:
            canvas = view.get_canvas()
        except Exception:
            canvas = None
        toolbar.set_canvas(canvas, reset_history=reset_history)

    def on_view_canvas_changed(self, view) -> None:
        """Called by a view after it replaced its canvas or redrew its content."""
        if view is getattr(self, "active_view", None):
            self.sync_toolbar_to_active_view(reset_history=True)

    def _apply_plot_theming(self):
        """Apply theme colors to matplotlib figure, axes, data artists, and toolbar.

        Delegates figure/axes background and data-artist re-coloring to
        :func:`~iSLAT.Modules.Plotting.Theming.apply_figure_theme` (with the
        three-panel-view's grid legend strategy when the grid exists) and
        adds GUI-specific extras on top: canvas widget background, toolbar,
        legend frame coloring, grid lines, and ``axis_text_label_color`` support.
        """
        try:
            # --- Common figure/axes/artist theming -----------------------
            grid = getattr(self._three_panel_view, '_grid', None)
            if grid is not None:
                grid.theme = self.theme
            legend_strategy = grid.legend_strategy if grid is not None else StandardLegend()
            apply_figure_theme(self.fig, self.theme, legend_strategy)

            # --- GUI-specific extras ------------------------------------
            # Some GUI themes define 'axis_text_label_color' which is
            # more specific than BasePlot's generic 'foreground'.
            fg = self.theme.get("foreground", "#F0F0F0")
            label_color = self.theme.get("axis_text_label_color", fg)

            if label_color != fg:
                # Re-apply the GUI-specific label color on top
                for ax in [self.ax1, self.ax2, self.ax3]:
                    ax.tick_params(colors=label_color, which='both')
                    ax.xaxis.label.set_color(label_color)
                    ax.yaxis.label.set_color(label_color)
                    ax.title.set_color(label_color)
                    for spine in ax.spines.values():
                        spine.set_color(label_color)

            # Grid lines (GUI-only feature)
            for ax in [self.ax1, self.ax2, self.ax3]:
                if self.theme.get(f'ax{ax.get_gid()}_grid', False):
                    ax.grid(True, color=label_color, alpha=0.3, linestyle='-', linewidth=0.5)

            # Canvas widget background
            apply_canvas_theme(self.canvas, self.theme)

            # Legend frame / text coloring
            for ax in [self.ax1, self.ax2, self.ax3]:
                legend = ax.get_legend()
                if legend is not None:
                    frame = legend.get_frame()
                    if legend.get_visible() and frame.get_visible():
                        frame.set_facecolor(self.theme.get("graph_fill_color", "white"))
                        frame.set_edgecolor(label_color)
                    for text in legend.get_texts():
                        if not getattr(text, '_islat_mol_color', False):
                            text.set_color(label_color)

        except Exception as e:
            debug_config.error("main_plot", f"Could not apply plot theming: {e}")
    
    # ------------------------------------------------------------------
    # View registry - lets external callers enumerate and switch views
    # ------------------------------------------------------------------
    @property
    def views(self) -> dict:
        """Return an ordered ``{name: PlotView}`` mapping of all registered views."""
        return self._views

    @property
    def active_view_name(self) -> str:
        """Return the display name of the currently active view."""
        for name, view in self._views.items():
            if view is self.active_view:
                return name
        return "Three Panel"

    def switch_view(self, view_name: str) -> None:
        """Switch to the named view, deactivating the current one.

        Parameters
        ----------
        view_name : str
            One of the keys in :attr:`views`
            (e.g. ``"Three Panel"``, ``"Full Spectrum"``,
            ``"Population Diagram"``).
        """
        if view_name not in self._views:
            debug_config.warning(
                "main_plot", f"switch_view: unknown view '{view_name}'",
            )
            return
        target = self._views[view_name]
        if target is self.active_view:
            return

        old_view = self.active_view
        old_view.deactivate()

        self.active_view = target

        # Keep is_full_spectrum in sync so existing callers still work
        self.is_full_spectrum = (target is self._full_spectrum_view)

        target.activate(self.parent_frame)
        self.sync_toolbar_to_active_view()
        self._notify_view_change(old_view, target)

        debug_config.info("main_plot", f"switch_view: active view → '{view_name}'")

    # ------------------------------------------------------------------
    # Backward-compatibility property for external code that still
    # references full_spectrum_plot directly.
    # ------------------------------------------------------------------
    @property
    def full_spectrum_plot(self):
        """Return the FullSpectrumView (replaces the old FullSpectrumPlot)."""
        return self._full_spectrum_view

    @property
    def line_inspection_plot(self):
        """Access the active :class:`LineInspectionPlot` delegate from the grid."""
        grid = getattr(self._three_panel_view, '_grid', None)
        return getattr(grid, 'inspection_panel', None)

    @property
    def population_diagram_plot(self):
        """Access the active :class:`PopulationDiagramPlot` from the grid."""
        grid = getattr(self._three_panel_view, '_grid', None)
        return getattr(grid, 'pop_diagram_panel', None)

    def _register_update_callbacks(self):
        """Register callbacks to handle parameter and molecule changes"""
        Molecule.add_molecule_parameter_change_callback(self.on_molecule_parameter_changed)
        
        # Register for global parameter changes if molecules_dict exists
        if hasattr(self.islat, 'molecules_dict'):
            self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_changed)

        # Register for active molecule changes
        self.islat.molecules_dict.add_active_molecule_change_callback(self._on_active_molecule_changed)

        # Refresh the Line Grid view whenever the input line list changes
        if hasattr(self.islat, 'add_line_list_change_callback'):
            self.islat.add_line_list_change_callback(self._on_line_list_changed)
    
    def _on_active_molecule_changed(self, old_molecule, new_molecule):
        """Handle active molecule changes"""
        self.on_active_molecule_changed()
    
    def _on_global_parameter_changed(self, parameter_name, old_value, new_value):
        """Handle global parameter changes that affect all molecules.

        Always preserves the current line-inspection selection so that
        changing global parameters (pixel resolution, distance, …) does
        not blank the inspection / population-diagram panels.
        """
        # Update the main spectrum (and mark the inactive view stale)
        self.update_model_plot()

        # If there's an active line inspection selection, refresh it
        # with the new parameters instead of clearing it.
        current_selection = getattr(self, 'current_selection', None)
        if current_selection is not None:
            xmin, xmax = current_selection
            self.active_view.on_selection(xmin, xmax)
        else:
            # No selection - let the active view refresh its panels
            self.active_view.on_active_molecule_changed(
                new_molecule=getattr(self.islat, 'active_molecule', None),
                current_selection=None,
            )

    def match_display_range(self, match_y=False):
        # Sync plot xlim to islat.display_range if set, else update islat.display_range from plot
        if hasattr(self.islat, 'display_range'):
            # If display_range is set elsewhere, update plot xlim
            if self.islat.display_range:
                wmin, wmax = self.islat.display_range
                current_xlim = self.ax1.get_xlim()
                
                # Only update plot xlim if it's actually different (prevent infinite loops)
                if (abs(current_xlim[0] - wmin) > 1e-10 or 
                    abs(current_xlim[1] - wmax) > 1e-10):
                    self.ax1.set_xlim(wmin, wmax)
            else:
                # If not set, initialize from current plot xlim
                self.islat.display_range = tuple(self.ax1.get_xlim())
        else:
            # If islat has no display_range attribute, do nothing
            return

        # Adjust y-limits according to the auto / manual policy.
        self.apply_y_limits(autoscale_now=match_y)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Main-plot Y axis: auto-scale toggle and manual limits
    # ------------------------------------------------------------------
    def _y_policy_view(self):
        """Return the active view when it implements the shared Y policy, else ``None``."""
        view = getattr(self, 'active_view', None)
        if view is not None and hasattr(view, 'apply_y_limits') and hasattr(view, 'get_y_limits'):
            return view
        return None

    def apply_y_limits(self, autoscale_now: bool = True) -> bool:
        """Apply the Y-axis policy (``toggle_state['y_autoscale']`` / ``['y_limits']``).

        Delegates to the active view's :meth:`PlotView.apply_y_limits`, so
        every view that exposes Y axes (three-panel main plot, full-spectrum
        panels, ...) honours the same setting.  Before a view exists the main
        plot's ``ax1`` is handled directly.

        * Auto-scale on and *autoscale_now*: re-fit the axes to the data.
        * Auto-scale on but *autoscale_now* False: leave them alone.
        * Auto-scale off: pin the axes to the stored manual limits.

        Returns ``True`` when the limits were set.
        """
        view = self._y_policy_view()
        if view is not None:
            return bool(view.apply_y_limits(self.toggle_state, autoscale_now=autoscale_now))
        # Fallback: no active view yet - operate on the main-plot axes.
        if not self.y_autoscale_toggle:
            manual = self.toggle_state.get("y_limits")
            if manual is None:
                return False
            self.ax1.set_ylim(*manual)
            return True
        if not autoscale_now:
            return False
        wave = getattr(self.islat, 'wave_data', None)
        flux = getattr(self.islat, 'flux_data', None)
        scope = self.y_data_sync
        if scope is not None:
            auto = data_synced_y_limits([self.ax1], wave, flux, scope)[0]
        else:
            auto = fit_y_limits_to_data(self.ax1, wave, flux)
        if auto is None:
            return False
        self.ax1.set_ylim(*auto)
        return True

    def get_y_limits(self) -> Tuple[float, float]:
        """Return the active view's current ``(ymin, ymax)`` (main-plot axes as fallback)."""
        view = self._y_policy_view()
        if view is not None:
            lims = view.get_y_limits()
            if lims is not None:
                return lims
        lo, hi = self.ax1.get_ylim()
        return (float(lo), float(hi))

    def _draw_active_view(self) -> None:
        """Flush the active view's canvas (falls back to the main canvas)."""
        view = getattr(self, 'active_view', None)
        if view is not None and hasattr(view, 'draw'):
            view.draw()
        else:
            self.canvas.draw_idle()

    def set_y_autoscale(self, enabled: bool) -> None:
        """Turn Y auto-scaling on or off.

        Turning it on re-fits the axes immediately.  Turning it off freezes
        them at their current limits, which become the manual limits.
        """
        enabled = bool(enabled)
        self.y_autoscale_toggle = enabled
        if not enabled and self.toggle_state.get("y_limits") is None:
            self.toggle_state["y_limits"] = self.get_y_limits()
        self.apply_y_limits(autoscale_now=True)
        self._draw_active_view()

    def toggle_y_autoscale(self) -> None:
        """Flip the Y auto-scale toggle."""
        self.set_y_autoscale(not self.y_autoscale_toggle)

    def set_y_autoscale_source(self, source: str) -> None:
        """Choose what Y auto-scaling fits to: ``"data"`` or ``"model"``.

        Re-fits immediately when auto-scaling is on; otherwise the choice
        takes effect the next time auto-scaling is turned on.  Choosing
        ``"model"`` stops syncing to the data's min / max (the two rules
        are exclusive).
        """
        self.y_autoscale_source = source
        if self.y_autoscale_source == "model":
            self.y_data_sync = None
        if self.y_autoscale_toggle:
            self.apply_y_limits(autoscale_now=True)
            self._draw_active_view()

    def set_y_data_sync(self, scope: Optional[str]) -> None:
        """Sync the Y axes to the observed data's exact min / max, or stop syncing.

        *scope* is ``"global"`` (the whole spectrum), ``"view"`` (the data
        in the active view's visible range, one range shared by all of its
        panels) or ``"panel"`` (each panel to the data in its own range;
        the same as ``"view"`` for single-panel views).  ``None`` or
        ``"off"`` returns auto-scaling to its normal rule.

        Turning syncing on also turns Y auto-scaling on and the model-flux
        source off; the axes are re-fitted immediately.
        """
        self.y_data_sync = scope
        if self.y_data_sync is not None:
            self.y_autoscale_source = "data"
            self.y_autoscale_toggle = True
        if self.y_autoscale_toggle:
            self.apply_y_limits(autoscale_now=True)
            self._draw_active_view()

    def get_data_y_range(self, scope: str = "global") -> Optional[Tuple[float, float]]:
        """The observed data's exact ``(min, max)`` in the active view for *scope*.

        ``"global"`` covers the whole spectrum and ``"view"`` the visible
        range.  ``"panel"`` gives a pair only for single-panel views.
        ``None`` when the active view has no Y axis or no data there.
        """
        view = self._y_policy_view()
        if view is not None:
            if hasattr(view, 'get_data_y_range'):
                return view.get_data_y_range(scope)
            return None
        scope = normalize_y_data_sync(scope)
        if scope is None:
            return None
        wave = getattr(self.islat, 'wave_data', None)
        flux = getattr(self.islat, 'flux_data', None)
        if scope == "global":
            return data_y_range(wave, flux)
        return data_y_range(wave, flux, [self.ax1.get_xlim()])

    def y_axis_count(self) -> int:
        """Number of Y axes (panels) the Y settings apply to in the active view."""
        view = self._y_policy_view()
        if view is not None:
            return int(view.y_axis_count()) if hasattr(view, 'y_axis_count') else 1
        return 1

    def set_y_limits(
        self,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
    ) -> None:
        """Pin the Y axes to specific values.

        Either bound may be omitted to keep its current value.  Setting a
        manual limit switches Y auto-scaling off.  Reversed bounds are
        ignored.
        """
        cur_lo, cur_hi = self.get_y_limits()
        lo = float(ymin) if ymin is not None else cur_lo
        hi = float(ymax) if ymax is not None else cur_hi
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            return
        self.toggle_state["y_limits"] = (lo, hi)
        self.y_autoscale_toggle = False
        self.apply_y_limits()
        self._draw_active_view()

    # ------------------------------------------------------------------
    # Observed-data visibility
    # ------------------------------------------------------------------
    def toggle_observed_data(self, visible: Optional[bool] = None) -> None:
        """Show or hide the observed spectrum on the active view.

        Parameters
        ----------
        visible : bool, optional
            If *None* the toggle is flipped; otherwise the explicit state
            is applied.
        """
        if visible is None:
            self.observed_toggle = not self.observed_toggle
        else:
            self.observed_toggle = visible
        self.active_view.toggle_observed_data(self.observed_toggle)

    # ------------------------------------------------------------------
    # User wavelength markers (Shift+M) - API from WavelengthMarkerStoreMixin
    # ------------------------------------------------------------------
    def refresh_wavelength_markers(self) -> None:
        """Redraw markers on the active view; inactive views resync when activated."""
        view = getattr(self, "active_view", None)
        if view is not None and hasattr(view, "refresh_wavelength_markers"):
            view.refresh_wavelength_markers()

    # ================================
    # Loading Indicator for Async Display
    # ================================
    def show_loading_indicator(self, message="Loading..."):
        """
        Show a loading indicator on the plot while calculations are in progress.
        
        Parameters
        ----------
        message : str
            Message to display on the loading indicator
        """
        self._loading_text = self.ax1.text(
            0.5, 0.5, message,
            transform=self.ax1.transAxes,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='gray', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )
        self.canvas.draw_idle()
        # Process pending events to show immediately
        self.canvas.get_tk_widget().update_idletasks()
    
    def hide_loading_indicator(self):
        """Remove the loading indicator from the plot."""
        if hasattr(self, '_loading_text') and self._loading_text is not None:
            try:
                self._loading_text.remove()
            except ValueError:
                pass  # Already removed
            self._loading_text = None
            self.canvas.draw_idle()

    def _set_initial_zoom_range(self):
        """Set the initial zoom range based on display_range or data range"""
        # Use display_range if available
        if hasattr(self.islat, 'display_range') and self.islat.display_range:
            xmin, xmax = self.islat.display_range
            self.ax1.set_xlim(xmin, xmax)
        elif hasattr(self.islat, 'wave_data') and self.islat.wave_data is not None:
            # Fallback to full data range
            self.ax1.set_xlim(self.islat.wave_data.min(), self.islat.wave_data.max())
        
        # Update display to match and set optimal y-limits
        self.match_display_range(match_y=True)
        self.canvas.draw_idle()

    def make_span_selector(self):
        """Creates a SpanSelector for the main plot to select a region for line inspection."""
        self.span = self.interaction_handler.create_span_selector(self.ax1, self.theme["selection_color"])

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        Delegates to the active view for rendering.
        """
        self.update_model_plot()
        current_selection = getattr(self, 'current_selection', None)
        self.active_view.on_active_molecule_changed(
            new_molecule=getattr(self.islat, 'active_molecule', None),
            current_selection=current_selection,
        )

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.

        Delegates to the active view so the correct panel layout is refreshed.
        The inactive view is marked stale so it re-renders on the next activate().
        """
        # Always update the active view
        self.active_view.update_model_plot()
        # Mark every inactive view stale so it re-renders on next activate().
        self._mark_inactive_views_stale()

    def _mark_inactive_views_stale(self) -> None:
        """Set ``_needs_refresh = True`` on every view that is not currently active.

        Views honour this flag in their ``activate()`` implementation and
        re-render lazily, so no work is done until the user actually switches
        to that view.
        """
        for view in self._views.values():
            if view is not self.active_view and hasattr(view, '_needs_refresh'):
                view._needs_refresh = True

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        self.toggle_state["current_selection"] = (xmin, xmax)
        # Clear previous fit result - it belongs to the old selection
        self.fit_result = None
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.selected_wave = self.islat.wave_data[mask]

        # Delegate rendering to the active view
        self.active_view.on_selection(xmin, xmax)

    #: Views that inspect the selected range themselves.  "Show in Spectrum"
    #: switches any other view to the Three Panel view first.
    LINE_SELECTION_VIEWS = ("Three Panel", "Line Inspection")

    def show_line_in_spectrum(self, ref) -> bool:
        """Select the line *ref* (a :class:`LineRef`) for inspection and bring it into view.

        The range selected is *ref*'s window (e.g. a saved line's fit range),
        or else one as wide as the current selection (2 x
        ``DEFAULT_HALF_WINDOW`` without one), centered on the line.  Views
        that do not inspect a selection switch to the Three Panel view, which
        also centers the spectrum on the line (keeping the plotted width,
        widened to twice the range when the range would fill more than half
        of it).  The range is then selected as dragging across it selects
        it, and the line is highlighted.

        Returns whether the line was highlighted.  It is not when it is not
        among the inspected lines (another molecule, or below the line
        threshold); the data field then says so.
        """
        from iSLAT.Modules.DataProcessing.LineIdentity import DEFAULT_HALF_WINDOW

        lam = float(ref.lam)
        window = ref.window
        if window is None:
            half = DEFAULT_HALF_WINDOW
            selection = getattr(self, "current_selection", None)
            if selection:
                width = abs(float(selection[1]) - float(selection[0]))
                if width > 0:
                    half = width / 2.0
            window = (lam - half, lam + half)
        xmin, xmax = sorted(float(v) for v in window)

        if self.active_view_name not in self.LINE_SELECTION_VIEWS:
            self.switch_view("Three Panel")
        if self.active_view is self._three_panel_view:
            self._center_spectrum_on(lam, xmax - xmin)

        handler = self.interaction_handler
        handler.show_span(xmin, xmax)
        handler.set_selection(xmin, xmax)

        select = getattr(self.active_view, "select_line", None)
        found = bool(select(ref)) if callable(select) else False
        if not found:
            notify_user(
                self.islat,
                f"{ref.describe()} is not among the lines inspected in "
                f"{xmin:.4f} - {xmax:.4f} um (its molecule may not be shown, "
                "or the line is below the line threshold).",
            )
        return found

    def _center_spectrum_on(self, lam: float, min_width: float) -> None:
        """Center the main spectrum on *lam*, keeping its width unless that is under 2 x *min_width*."""
        lo, hi = self.ax1.get_xlim()
        width = max(abs(hi - lo), 2.0 * float(min_width))
        new_range = (lam - width / 2.0, lam + width / 2.0)
        try:
            # Moves the plot and the Plot Start / Plot Range fields.
            self.islat.display_range = new_range
        except (AttributeError, ValueError):
            self.ax1.set_xlim(*new_range)
        self.match_display_range(match_y=True)

    # ------------------------------------------------------------------
    # Data-access & interaction helpers
    # ------------------------------------------------------------------
    def get_molecule_line_data(
        self, molecule: 'Molecule', xmin: float, xmax: float,
    ) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Get molecule lines in a wavelength range.

        This is a data-access operation and belongs in the controller
        rather than the renderer.

        Parameters
        ----------
        molecule : Molecule
            Molecule object
        xmin, xmax : float
            Wavelength range

        Returns
        -------
        List[Tuple[MoleculeLine, float, Optional[float]]]
            List of (MoleculeLine, intensity, tau) tuples
        """
        try:
            # Method 1: Use intensity API
            if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                intensity_obj = molecule.intensity
                if hasattr(intensity_obj, 'get_lines_in_range_with_intensity'):
                    return intensity_obj.get_lines_in_range_with_intensity(xmin, xmax)

            # Method 2: Use MoleculeLineList directly
            if hasattr(molecule, 'lines') and molecule.lines is not None:
                lines = molecule.lines
                if hasattr(lines, 'get_lines_in_range'):
                    lines_in_range = lines.get_lines_in_range(xmin, xmax)
                    # Try to get corresponding intensities
                    if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                        intensity_obj = molecule.intensity
                        if hasattr(intensity_obj, 'intensity') and intensity_obj.intensity is not None:
                            intensities = intensity_obj.intensity
                            tau_values = getattr(intensity_obj, 'tau', None)

                            result = []
                            for i, line in enumerate(lines_in_range):
                                intensity = intensities[i] if i < len(intensities) else 0.0
                                tau = tau_values[i] if tau_values is not None and i < len(tau_values) else None
                                result.append((line, intensity, tau))
                            return result
                    else:
                        return [(line, 0.0, None) for line in lines_in_range]
            return []

        except Exception as e:
            print(f"Error getting molecule lines: {e}")
            return []

    def clear_selection(self):
        self.current_selection = None
        self.toggle_state["current_selection"] = None
        # Delegate rendering cleanup to the active view
        self.active_view.clear_selection()
        return

    def toggle_atomic_lines(self, show: Optional[bool] = None) -> None:
        """
        Toggle atomic line annotations on the active view.

        Parameters
        ----------
        show : bool or None
            If *None* the toggle is flipped; otherwise the explicit state
            is forwarded.
        """
        if show is None:
            self.atomic_toggle = not self.atomic_toggle
        else:
            self.atomic_toggle = show
        self.active_view.toggle_atomic_lines(self.atomic_toggle)

    def toggle_saved_lines(self, show: Optional[bool] = None, loaded_lines=None) -> None:
        """
        Toggle saved-line annotations on the active view.

        Parameters
        ----------
        show : bool or None
            If *None* the toggle is flipped; otherwise the explicit state
            is forwarded.
        loaded_lines : DataFrame or None
            Pre-loaded line data.  If *None* the view will load from disk.
        """
        if show is None:
            self.line_toggle = not self.line_toggle
        else:
            self.line_toggle = show
        self.active_view.toggle_saved_lines(self.line_toggle, loaded_lines=loaded_lines)

    def refresh_saved_line_labels(self) -> None:
        """Redraw saved lines after the label setting changed (see :mod:`SavedLineLabels`).

        The active view is redrawn now; inactive views drop their saved-line
        artists so they are rebuilt with the new labels when activated.
        """
        for view in self._views.values():
            if view is self.active_view:
                continue
            remover = getattr(view, "_remove_saved_line_artists", None)
            if callable(remover):
                try:
                    remover()
                except Exception:
                    pass
            elif hasattr(view, "_needs_refresh"):
                view._needs_refresh = True
        if not self.line_toggle:
            return
        loaded_lines = ifh.read_line_saves(file_name=getattr(self.islat, "input_line_list", None))
        if loaded_lines is None or loaded_lines.empty:
            return
        self.active_view.toggle_saved_lines(False)
        self.active_view.toggle_saved_lines(True, loaded_lines=loaded_lines)

    def toggle_legend(self):
        self.legend_toggle = not self.legend_toggle
        self.active_view.toggle_legend(self.legend_toggle)

    def remove_atomic_lines(self):
        """Remove atomic lines - delegates to the active view."""
        self.atomic_toggle = False
        self.active_view.toggle_atomic_lines(False)

    def plot_atomic_lines(self, data_field=None, atomic_lines=None):
        """Plot atomic lines - delegates to the active view."""
        self.atomic_toggle = True
        self.active_view.toggle_atomic_lines(True)

    def on_click(self, event):
        """Handle mouse click events on the plot."""
        self.interaction_handler.handle_click_event(event)
    
    def on_active_molecule_changed(self):
        """Called when the active molecule changes - delegates to the active view."""
        self.active_view.on_active_molecule_changed(
            new_molecule=getattr(self.islat, 'active_molecule', None),
            current_selection=getattr(self, 'current_selection', None),
        )
        self._mark_inactive_views_stale()

    def on_molecule_parameter_changed(self, molecule_name, parameter_name, old_value, new_value):
        """Called when any molecule parameter changes - delegates to the active view."""
        if parameter_name == 'is_visible':
            return
        self.active_view.on_molecule_parameter_changed(
            molecule_name=molecule_name,
            parameter_name=parameter_name,
            current_selection=getattr(self, 'current_selection', None),
        )
        self._mark_inactive_views_stale()

    def on_molecule_deleted(self, molecule_name):
        """Called when a molecule is deleted - delegates to the active view."""
        self.active_view.on_molecule_deleted(molecule_name)
        self._mark_inactive_views_stale()
    
    def on_molecule_visibility_changed(self, molecule_name, is_visible):
        """Called when a molecule's visibility changes - delegates to the active view."""
        if not hasattr(self.islat, 'molecules_dict'):
            return
        self.active_view.on_molecule_visibility_changed(
            molecule_name=molecule_name,
            is_visible=is_visible,
            molecules_dict=self.islat.molecules_dict,
            wave_data=self.islat.wave_data,
            active_molecule=getattr(self.islat, 'active_molecule', None),
            current_selection=getattr(self, 'current_selection', None),
        )
        self._mark_inactive_views_stale()
    
    def save_fig(self, filename, dpi=300):
        """Save the current figure to a file."""
        try:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            debug_config.info("main_plot", f"Figure saved to {filename}")
        except Exception as e:
            debug_config.error("main_plot", f"Error saving figure: {e}")

    def find_single_lines(self, xmin=None, xmax=None):
        """
        Find single lines using LineAnalyzer with data processing.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range. Uses current_selection if not provided.
            
        Returns
        -------
        list
            List of detected lines with properties
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                return []
        
        # Efficient data selection using vectorized operations
        range_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[range_mask]
        range_flux = self.islat.flux_data[range_mask]
        
        if len(range_wave) < 10:
            return []
        
        try:
            detected_lines = self.line_analyzer.find_single_lines(
                range_wave, range_flux
            )
            
            # Create optimized line data structure
            self.single_lines_list = []
            ylim = self.ax1.get_ylim()
            
            for line in detected_lines:
                line_info = {
                    "wavelength": line['wavelength'], 
                    "ylim": ylim,
                    #"strength": line['line_strength'],
                    #"type": line['type']
                }
                self.single_lines_list.append(line_info)
            
            return self.single_lines_list
        except Exception as e:
            debug_config.error("main_plot", f"Error in line detection: {str(e)}")
            return []
    
    def plot_saved_lines(self, loaded_lines=None, data_field=None):
        """Plot saved lines - delegates to the active view."""
        self.line_toggle = True
        self.active_view.toggle_saved_lines(True, loaded_lines=loaded_lines)

    def remove_saved_lines(self):
        """Remove saved lines - delegates to the active view."""
        self.line_toggle = False
        self.active_view.toggle_saved_lines(False)

    def apply_theme(self, theme=None):
        """Apply theme colors to the plot without recalculating spectra.

        This is a *visual-only* refresh: figure / axes backgrounds,
        tick / label / spine colors, canvas widget, toolbar, and the
        full-spectrum view.  No spectrum data is recomputed, so the
        update is near-instant.

        Both views are updated unconditionally so that a theme change
        made while one view is active is immediately reflected when the
        user switches to the other view.  Each view's
        :meth:`PlotView.apply_theme` implementation handles its own
        figure, axes, canvas, and sub-delegates.
        """
        if theme:
            self.theme = theme

        # Propagate to both views - the currently invisible view will
        # also be themed so the next activate() doesn't flash stale
        # colors.
        self._three_panel_view.apply_theme(self.theme)
        self._full_spectrum_view.apply_theme(self.theme)
        self._optical_depth_view.apply_theme(self.theme)

        # Theme the matplotlib toolbar
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            try:
                self.toolbar.configure(
                    bg=self.theme.get("background", "#181A1B")
                )
                for child in self.toolbar.winfo_children():
                    try:
                        child.configure(bg=self.theme.get("background", "#181A1B"))
                    except Exception:
                        pass
            except Exception:
                pass
    
    # ------------------------------------------------------------------
    # View-change callback infrastructure
    # ------------------------------------------------------------------
    def add_view_change_callback(self, cb):
        """Register a callback invoked when the active view changes.

        The callback signature is ``cb(old_view, new_view)`` where both
        arguments are :class:`PlotView` instances (or *None* for the
        initial call).
        """
        if cb not in self._view_change_callbacks:
            self._view_change_callbacks.append(cb)

    def remove_view_change_callback(self, cb):
        """Remove a previously registered view-change callback."""
        try:
            self._view_change_callbacks.remove(cb)
        except ValueError:
            pass

    def _notify_view_change(self, old_view, new_view):
        """Notify all registered listeners that the active view changed."""
        for cb in self._view_change_callbacks:
            try:
                cb(old_view, new_view)
            except Exception as exc:
                debug_config.warning(
                    "main_plot",
                    f"View-change callback {cb} raised: {exc}",
                )

    def load_full_spectrum(self):
        """Activate the full-spectrum view (delegates to switch_view)."""
        self.switch_view("Full Spectrum")

    def toggle_summed_spectrum(self):
        """Toggle visibility of the summed spectral flux."""
        self.summed_toggle = not self.summed_toggle
        self.active_view.toggle_summed_spectrum(self.summed_toggle)

    def toggle_residuals(self) -> None:
        """Toggle residual sub-panels in the full-spectrum view.

        When activated while the full-spectrum view is showing, the view
        switches from :class:`FullSpectrumPlot` to
        :class:`ResidualSpectrumPlot` (and vice-versa) by rebuilding
        the composed plot.

        If the three-panel view is active the flag is stored but no
        visual change occurs until the user enters full-spectrum mode.
        """
        self.residual_toggle = not self.residual_toggle
        debug_config.info(
            "main_plot",
            f"toggle_residuals: show_residuals = {self.residual_toggle}",
        )

        # Only trigger a rebuild when the full-spectrum view is active
        if self.is_full_spectrum:
            self._full_spectrum_view.toggle_residuals(self.residual_toggle)
        elif self.residual_toggle and self.toggle_state.get("sample_stack", False):
            # Residuals and the sample stack are exclusive (the view enforces
            # this itself when it is showing).
            self.toggle_state["sample_stack"] = False

    def on_sample_changed(self) -> None:
        """Called by iSLAT when the sample (spectra, parameter files, overrides) changes.

        Lets the Full Spectrum view refresh its stack of sample spectra.
        """
        view = getattr(self, "_full_spectrum_view", None)
        handler = getattr(view, "on_sample_changed", None)
        if callable(handler):
            handler()

    def _on_line_list_changed(self, new_path) -> None:
        """Callback fired when ``islat.input_line_list`` is assigned a new value.

        Clears any stale fit grids and rebuilds the Line Grid preview from
        the new line list.  If the view is currently active the display is
        updated immediately; otherwise it is marked for refresh on next
        activation.
        """
        view = self._fit_lines_grid_view
        view._plot_grids = []
        view._needs_refresh = True
        if self.active_view_name == "Line Grid" and view._parent_frame is not None:
            view._rebuild_display()
            view._needs_refresh = False

    def toggle_full_spectrum(self):
        """Toggle between the full-spectrum view and the previously active view.

        * If the full-spectrum view is **not** currently active, the current
          view name is remembered and the display switches to Full Spectrum.
        * If the full-spectrum view **is** currently active, the display
          returns to whichever view was active before (falling back to
          ``"Three Panel"`` if that view is no longer available).
        """
        if self.active_view is not self._full_spectrum_view:
            # Remember where we came from, then switch to Full Spectrum.
            self._pre_fullspectrum_view_name = self.active_view_name
            self.switch_view("Full Spectrum")
        else:
            # Return to the previous view (or Three Panel as a safe fallback).
            return_to = self._pre_fullspectrum_view_name
            if return_to not in self._views or return_to == "Full Spectrum":
                return_to = "Three Panel"
            self.switch_view(return_to)

        debug_config.info(
            "main_plot",
            f"toggle_full_spectrum: active view \u2192 '{self.active_view_name}'",
        )

    def toggle_three_panel(self):
        """Toggle between the three-panel view and the previously active view.

        * If the three-panel view is **not** currently active, the current
          view name is remembered and the display switches to Three Panel.
        * If the three-panel view **is** currently active, the display
          returns to whichever view was active before (falling back to
          ``"Full Spectrum"`` if that view is no longer available).
        """
        if self.active_view_name != "Three Panel":
            self._pre_threepanel_view_name = self.active_view_name
            self.switch_view("Three Panel")
        else:
            return_to = getattr(self, '_pre_threepanel_view_name', None)
            if return_to not in self._views or return_to == "Three Panel":
                return_to = "Full Spectrum"
            self.switch_view(return_to)

        debug_config.info(
            "main_plot",
            f"toggle_three_panel: active view \u2192 '{self.active_view_name}'",
        )