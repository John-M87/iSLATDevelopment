"""
LineInspectionView - :class:`SpectrumPanelView` for a standalone full-size line inspection plot.

Shows a :class:`LineInspectionPlot` for the most recent span-selector selection in a dedicated figure that fills the main plotting area.
When no selection has been made yet the view displays a placeholder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import matplotlib.pyplot as plt

from .SpectrumPanelView import SpectrumPanelView
from .LineInspectionPlot import LineInspectionPlot
from .LineInspectionContextMixin import LineInspectionContextMixin

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
    from typing import Dict

try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    class _FallbackDebug:
        def verbose(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): print(f"WARNING: {kw or a}")
        def error(self, *a, **kw): print(f"ERROR: {kw or a}")
        def trace(self, *a, **kw): pass
    debug_config = _FallbackDebug()

class LineInspectionView(SpectrumPanelView, LineInspectionContextMixin):
    """
    Standalone line-inspection view.

    Renders a :class:`LineInspectionPlot` for the currently selected wavelength range (taken from the main controller's ``toggle_state["current_selection"]``).
    
    The view owns its own :class:`~matplotlib.figure.Figure` and :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` so it
    can be swapped in / out of the main window independently of the three-panel and full-spectrum views.

    When no selection has been made a placeholder message is shown.

    Parameters
    ----------
    plot_manager : iSLATPlot
        The main controller (``iSLAT.Modules.Plotting.MainPlot.iSLATPlot``).
    """
    def __init__(self, plot_manager: Any) -> None:
        super().__init__(plot_manager)

        # Inspection-specific state
        self._plot: Optional[LineInspectionPlot] = None
        self._current_selection: Optional[Tuple[float, float]] = None

        # Active line vline entries: [vline, text, None, info_dict]
        self.active_lines: List[Any] = []
        self.selected_line: Optional[Any] = None

        # matplotlib pick event connection id
        self._pick_cid: Optional[int] = None

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _get_active_molecule(self) -> Optional["Molecule"]:
        return getattr(self._islat, "active_molecule", None)

    def _get_molecules(self) -> Optional["MoleculeDict"]:
        return getattr(self._islat, "molecules_dict", None)

    def _build_plot(self) -> None:
        """Create (or recreate) the owned figure and :class:`LineInspectionPlot`."""
        # Close the previous figure to avoid memory leaks
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._plot = None

        # Clear active lines from any previous render
        self.active_lines.clear()
        self.selected_line = None

        self._fig = plt.Figure(figsize=(10, 5), constrained_layout=True)
        ax = self._fig.add_subplot(111)

        sel = self._current_selection
        if sel is None:
            # No selection yet - draw a placeholder
            ax.set_title("Line Inspection")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Flux density (Jy)")
            ax.text(
                0.5, 0.5,
                "Drag to select a wavelength range\nin the main spectrum",
                ha="center", va="center",
                transform=ax.transAxes,
                color="gray", fontsize=12,
            )
            return

        xmin, xmax = sel
        wave = getattr(self._islat, "wave_data", None)
        flux = getattr(self._islat, "flux_data", None)
        if wave is None or flux is None or len(wave) == 0:
            ax.set_title("Line Inspection")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Flux density (Jy)")
            ax.text(
                0.5, 0.5, "No spectral data available",
                ha="center", va="center",
                transform=ax.transAxes,
                color="gray", fontsize=12,
            )
            return

        error = getattr(self._islat, "error_data", None)
        mol = self._get_active_molecule()
        mols = self._get_molecules()

        # Gather line data for the active molecule (same helper used by ThreePanelView)
        line_data: Optional[List] = None
        if mol is not None and hasattr(self._pm, "get_molecule_line_data"):
            try:
                line_data = self._pm.get_molecule_line_data(mol, xmin, xmax)
            except Exception as exc:
                debug_config.warning(
                    "line_inspection_view",
                    f"get_molecule_line_data failed: {exc}",
                )

        try:
            self._plot = LineInspectionPlot(
                wave_data=wave,
                flux_data=flux,
                xmin=xmin,
                xmax=xmax,
                error_data=error,
                molecule=mol,
                molecules=mols,
                line_data=line_data,
                ax=ax,
                render_all_visible=True,
                theme=self._pm.theme,
            )
            self._plot.generate_plot()

            # Render summed spectrum overlay if the toggle is on
            if getattr(self._pm, 'summed_toggle', False):
                self._render_summed_overlay()

            # Render active line vline markers when line data is available
            if line_data and mol is not None:
                import numpy as np
                data_mask = (wave >= xmin) & (wave <= xmax)
                data_region_y = flux[data_mask]
                max_y = (
                    float(np.nanmax(data_region_y))
                    if len(data_region_y) > 0
                    else 1.0
                )
                threshold = getattr(self._pm, '_line_threshold', 0.0)
                mol_color = getattr(mol, 'color', None) or 'green'
                self._plot.render_active_lines(
                    line_data, self.active_lines,
                    max_y=max_y,
                    threshold=threshold,
                    color=mol_color,
                    molecule_name=getattr(mol, 'name', ''),
                    molecule_color=mol_color,
                )
        except Exception as exc:
            debug_config.warning(
                "line_inspection_view",
                f"LineInspectionPlot.generate_plot failed: {exc}",
            )

        # A fresh plot draws the observed data visible and picks its own
        # Y limits; re-impose the controller's settings.
        self._reapply_display_state(getattr(self._pm, "toggle_state", None))

    # ------------------------------------------------------------------
    # Observed-data / Y-policy hooks (see PlotView)
    # ------------------------------------------------------------------
    def _inspection_axes(self):
        """Yield the inspection axes when a plot has been rendered."""
        ax = getattr(self._plot, "ax", None) if self._plot is not None else None
        if ax is not None:
            yield ax

    def _iter_observed_axes(self):
        return self._inspection_axes()

    def _iter_y_axes(self):
        return self._inspection_axes()

    def _autoscale_y_axes(self) -> bool:
        """Re-fit by rebuilding the plot so its own Y layout is restored exactly."""
        if self._plot is None or not self._initialised:
            return False
        if self._y_autoscale_source() == "model":
            return super()._autoscale_y_axes()
        self.update_model_plot()
        return True

    def _render_summed_overlay(self) -> None:
        """Plot the summed model fill on the inspection axes."""
        if self._plot is None or self._fig is None:
            return
        ax = self._plot.ax
        if ax is None:
            return
        mols = self._get_molecules()
        wave = getattr(self._islat, "wave_data", None)
        if mols is None or wave is None or len(wave) == 0:
            return
        try:
            from .BasePlot import BasePlot
            BasePlot._clear_tagged_artists(ax, "_islat_summed", lines=False)
            import numpy as _np
            s_wave, s_flux = mols.get_summed_flux(wave)
            if s_wave is not None and len(s_wave) > 0:
                sel = self._current_selection
                if sel is not None:
                    mask = (s_wave >= sel[0]) & (s_wave <= sel[1])
                    if _np.any(mask):
                        self._plot._plot_summed_spectrum(
                            ax, s_wave[mask], s_flux[mask], deduplicate=False
                        )
        except Exception as exc:
            debug_config.warning("line_inspection_view", f"_render_summed_overlay failed: {exc}")

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill in the inspection plot."""
        if self._plot is None or self._fig is None:
            return
        ax = self._plot.ax
        if ax is None:
            return
        # Check if we already have the fill rendered
        existing = [c for c in ax.collections if hasattr(c, '_islat_summed')]
        if visible and not existing:
            # Need to render it first
            self._render_summed_overlay()
        else:
            for coll in existing:
                coll.set_visible(visible)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def sync_toggle_state(self, toggle_state: dict) -> None:
        """Apply the summed-spectrum toggle state when this view becomes active.

        Because :meth:`_build_plot` already checks ``summed_toggle`` during construction, this is only needed when the toggle is turned on *after*
        the plot has been built (e.g. switching into this view while the toggle is already set).
        """
        summed = toggle_state.get("summed", False)
        if summed and self._initialised:
            self.toggle_summed_spectrum(True)
        # Observed-data visibility and manual Y limits
        self._reapply_display_state(toggle_state)
        self.draw()

    # ==================================================================
    # PlotView - display range sync
    # ==================================================================
    def _get_display_range(self) -> Tuple[float, float]:
        """Return the current selection bounds for the Plot Start/Range controls."""
        if self._current_selection is not None:
            return self._current_selection
        return (0.0, 0.0)

    def _set_display_range(self, start: float, end: float) -> None:
        """Move the inspection window to [start, end] via a new on_selection call."""
        self.on_selection(float(start), float(end))

    def _register_control_fields(self) -> None:
        """Register a :class:`DisplayRangeField` that routes the Plot Start / Plot Range controls in the control panel to this view's inspection window."""
        bus = getattr(self._pm, 'control_bus', None)
        if bus is None:
            return
        from iSLAT.Modules.GUI.ControlField import DisplayRangeField
        bus.unregister_owner(self)
        bus.register(
            DisplayRangeField(
                "_liv_display_range",
                getter=self._get_display_range,
                setter=self._set_display_range,
                owner=self,
            ),
            "control_panel",
        )

    # ==================================================================
    # Pick event - active line interaction
    # ==================================================================
    def _ensure_canvas(self) -> None:
        """Build the canvas and keep the pick handler attached to it.

        Every selection change creates a fresh figure, so the base class
        replaces the canvas.  The old pick connection dies with the old
        canvas, so forget its id and connect again on the new one;
        otherwise clicks on line markers stop working after the first
        rebuild.
        """
        old_canvas = self._canvas
        super()._ensure_canvas()
        if self._canvas is not old_canvas:
            self._pick_cid = None
            self._connect_pick_event()

    def _connect_pick_event(self) -> None:
        """Connect a pick-event handler to the current canvas (idempotent)."""
        if self._canvas is None or self._pick_cid is not None:
            return
        self._pick_cid = self._canvas.mpl_connect('pick_event', self._on_pick_line)

    def _disconnect_pick_event(self) -> None:
        """Disconnect the pick-event handler."""
        if self._canvas is not None and self._pick_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._pick_cid)
            except Exception:
                pass
        self._pick_cid = None

    def _on_pick_line(self, event: Any) -> None:
        """Highlight the clicked active-line vline and display its info."""
        picked_artist = event.artist
        picked_value = None

        # Reset all vlines to their default color
        for vline, text_obj, _sc, value in self.active_lines:
            mol_color = (
                value.get('molecule_color', 'green') if value else 'green'
            )
            if vline is not None:
                vline.set_color(mol_color)
            if text_obj is not None:
                text_obj.set_color(mol_color)
            if picked_artist is vline or picked_artist is text_obj:
                picked_value = value

        # Highlight the picked vline
        if picked_value is not None:
            for vline, text_obj, _sc, value in self.active_lines:
                if value is picked_value:
                    if vline is not None:
                        vline.set_color('orange')
                    if text_obj is not None:
                        text_obj.set_color('orange')

            self.selected_line = picked_value
            self._display_line_info(picked_value)

        if self._canvas is not None:
            self._canvas.draw_idle()

    def _display_line_info(self, value: "Dict[str, Any]") -> None:
        """Push line info for *value* to the GUI data field."""
        islat = self._islat
        from .LineInspectionPlot import LineInspectionPlot as _LIP
        from iSLAT.Modules.DataProcessing.spectral_utils import flux_integral

        data_flux = None
        model_flux = None
        opacity_in_range = None
        sel = self._current_selection
        if sel is not None:
            xmin, xmax = sel
            err_data = getattr(islat, 'err_data', None)
            line_flux, _ = flux_integral(
                lam=islat.wave_data, flux=islat.flux_data,
                lam_min=xmin, lam_max=xmax, err=err_data,
            )
            data_flux = line_flux[0] if isinstance(line_flux, (list, tuple)) else line_flux
            active_mol = getattr(islat, 'active_molecule', None)
            if active_mol is not None:
                mol_wave, mol_flux_arr = active_mol.get_flux(return_wavelengths=True)
                model_flux, _ = flux_integral(
                    lam=mol_wave, flux=mol_flux_arr,
                    lam_min=xmin, lam_max=xmax, err=None,
                )
                _, mol_opacity_arr = active_mol.get_tau(return_wavelengths=True)
                opacity_in_range, _ = flux_integral(
                    lam=mol_wave, flux=mol_opacity_arr,
                    lam_min=xmin, lam_max=xmax, err=None,
                )

        class _L:
            pass
        _l = _L()
        _l.lam      = value.get('lam')
        _l.e_up     = value.get('e_up',     value.get('e'))
        _l.e_low    = value.get('e_low')
        _l.a_stein  = value.get('a_stein',  value.get('a'))
        _l.g_up     = value.get('g_up',     value.get('g'))
        _l.g_low    = value.get('g_low')
        _l.lev_up   = value.get('up_lev')
        _l.lev_low  = value.get('low_lev')

        info = _LIP.get_line_info(
            _l,
            intensity=value.get('intensity', value.get('inten', 0)),
            tau=value.get('tau'),
            data_flux_in_range=data_flux,
            model_flux_in_range=model_flux,
            opacity_in_range=opacity_in_range,
            molecule=getattr(islat, 'active_molecule', None),
        )
        info_str = _LIP.format_line_info(info)

        if (hasattr(islat, 'GUI') and hasattr(islat.GUI, 'data_field') and
                islat.GUI.data_field is not None):
            try:
                if hasattr(islat.GUI.data_field, 'text') and islat.GUI.data_field.text.winfo_exists():
                    islat.GUI.data_field.insert_text(info_str, clear_after=True)
            except Exception as exc:
                debug_config.warning("line_inspection_view", f"data_field update failed: {exc}")

    # ==================================================================
    # PlotView - lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        """Pack the line-inspection canvas into *parent_frame*."""
        self._parent_frame = parent_frame

        # Sync selection from the controller's shared state
        ctrl_sel = self._pm.toggle_state.get("current_selection")
        if ctrl_sel != self._current_selection:
            self._current_selection = ctrl_sel
            self._needs_refresh = True

        if not self._initialised or self._needs_refresh:
            self._build_plot()
            self._initialised = True
            self._needs_refresh = False

        self._ensure_canvas()
        self.apply_theme(self._pm.theme)
        self.sync_toggle_state(self._pm.toggle_state)
        self._register_control_fields()
        self._connect_pick_event()

        if self._canvas is not None:
            self._canvas.get_tk_widget().pack(
                fill="both", expand=True, padx=0, pady=0
            )
            self._canvas.draw_idle()

    # ==================================================================
    # PlotView - interaction context
    # ==================================================================
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Return the line-inspection right-click menu."""
        menu = self._build_line_inspection_menu(canvas_widget)
        if menu is not None:
            self._append_save_figure_item(menu)
        return menu

    def deactivate(self) -> None:
        """Unregister controls, disconnect pick event, and unpack the canvas."""
        self._disconnect_pick_event()
        bus = getattr(self._pm, 'control_bus', None)
        if bus is not None:
            bus.unregister_owner(self)
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().pack_forget()
            except Exception:
                pass

    def clear_active_lines(self) -> None:
        """Clear vline markers from the plot."""
        if self._plot is not None:
            self._plot.clear_active_lines(self.active_lines)
        else:
            self.active_lines.clear()
        self.selected_line = None

    # ==================================================================
    # PlotView - core rendering
    # ==================================================================
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Rebuild the inspection plot from current iSLAT data."""
        self._needs_refresh = True
        if self._initialised:
            self._build_plot()
            self._needs_refresh = False
            self._ensure_canvas()
            self.apply_theme(self._pm.theme)
            if self._canvas is not None:
                if self._parent_frame is not None:
                    self._canvas.get_tk_widget().pack(
                        fill="both", expand=True, padx=0, pady=0
                    )
                self._canvas.draw_idle()

    # ==================================================================
    # PlotView - selection (override base no-ops)
    # ==================================================================
    def on_selection(self, xmin: float, xmax: float) -> None:
        """Update the displayed wavelength range when the user makes a new selection."""
        self._current_selection = (xmin, xmax)
        self._needs_refresh = True
        if self._initialised:
            self._build_plot()
            self._needs_refresh = False
            self._ensure_canvas()
            self.apply_theme(self._pm.theme)
            if self._canvas is not None:
                if self._parent_frame is not None:
                    self._canvas.get_tk_widget().pack(
                        fill="both", expand=True, padx=0, pady=0
                    )
                self._canvas.draw_idle()

    def clear_selection(self) -> None:
        """Clear the selection and show the placeholder."""
        self._current_selection = None
        self._needs_refresh = True
        if self._initialised:
            self._build_plot()
            self._needs_refresh = False
            self._ensure_canvas()
            if self._canvas is not None:
                if self._parent_frame is not None:
                    self._canvas.get_tk_widget().pack(
                        fill="both", expand=True, padx=0, pady=0
                    )
                self._canvas.draw_idle()

    # ==================================================================
    # PlotView - molecule lifecycle (override to preserve selection)
    # ==================================================================
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Rebuild for the newly selected active molecule."""
        if current_selection is not None:
            self._current_selection = current_selection
        self.update_model_plot()

    def get_selected_line(self) -> Optional[Any]:
        """No persistent selected-line state in this view."""
        return None