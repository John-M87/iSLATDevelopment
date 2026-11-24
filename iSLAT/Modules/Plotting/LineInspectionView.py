"""
LineInspectionView - :class:`SpectrumPanelView` for a standalone full-size line inspection plot.

Shows a :class:`LineInspectionPlot` for the most recent span-selector selection in a dedicated figure that fills the main plotting area.
When no selection has been made yet the view displays a placeholder.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import matplotlib.pyplot as plt

from .ArtistTags import SUMMED_TAG, iter_tagged, remove_tagged
from .SpectrumPanelView import SpectrumPanelView
from .LineInspectionPlot import LineInspectionPlot
from .LineInspectionContextMixin import LineInspectionContextMixin
from .PlotDebug import debug_config
from .PlotView import notify_user

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
    from iSLAT.Modules.DataProcessing.LineIdentity import LineRef
    from typing import Dict

#: Control-panel key of the line-marker threshold entry.
LINE_THRESHOLD_KEY = "line_threshold"

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

        # Marker threshold for this view only (None: the shared user setting), and the one last drawn.
        self._line_threshold: Optional[float] = None
        self._drawn_threshold: Optional[float] = None

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _get_active_molecule(self) -> Optional["Molecule"]:
        return getattr(self._islat, "active_molecule", None)

    def _get_molecules(self) -> Optional["MoleculeDict"]:
        return getattr(self._islat, "molecules_dict", None)

    def _build_plot(self) -> None:
        """Draw the plot for the current selection into the owned figure.

        The figure is created once and then cleared and redrawn, so the canvas
        showing it is kept.  A new figure would need a new canvas widget, which
        must wait for Tk to lay it out before its first draw.  Clearing the
        figure and adding a new axes gives the same plot as a new figure:
        ``Axes.clear()`` alone would keep state such as autoscaling switched
        off by manual Y limits.
        """
        self._plot = None

        # Clear active lines from any previous render
        self.active_lines.clear()
        self.selected_line = None

        if self._fig is None:
            self._fig = plt.Figure(figsize=(10, 5), constrained_layout=True)
        else:
            self._fig.clear()
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

        error = getattr(self._islat, "err_data", None)
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
            # No line_data: the markers come from render_active_lines below.
            self._plot = LineInspectionPlot(
                wave_data=wave,
                flux_data=flux,
                xmin=xmin,
                xmax=xmax,
                error_data=error,
                molecule=mol,
                molecules=mols,
                ax=ax,
                render_all_visible=True,
                theme=self._pm.theme,
            )
            self._plot.generate_plot()

            # Render summed spectrum overlay if the toggle is on
            if getattr(self._pm, 'summed_toggle', False):
                self._render_summed_overlay()

            # Render active line vline markers when line data is available
            self._drawn_threshold = self._get_line_threshold()
            if line_data and mol is not None:
                mol_color = self._molecule_line_color(mol)
                self._plot.render_active_lines(
                    line_data, self.active_lines,
                    max_y=self._marker_max_y(xmin, xmax),
                    threshold=self._drawn_threshold,
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
            remove_tagged(ax, SUMMED_TAG, lines=False)
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
        existing = list(iter_tagged(ax, SUMMED_TAG, lines=False))
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
        """Route Plot Start / Plot Range to the inspection window and add the *Line Threshold* entry."""
        bus = getattr(self._pm, 'control_bus', None)
        if bus is None:
            return
        from iSLAT.Modules.GUI.ControlField import DisplayRangeField, EntryField
        bus.unregister_owner(self)
        bus.register_many([
            DisplayRangeField(
                "_liv_display_range",
                getter=self._get_display_range,
                setter=self._set_display_range,
                owner=self,
            ),
            EntryField(
                LINE_THRESHOLD_KEY, "Line Threshold:",
                getter=self._get_line_threshold,
                setter=self._set_line_threshold,
                datatype="float", width=5, owner=self,
                tip="Mark only lines at least this fraction (0-1) of the strongest line in range.\n"
                    "Applies to this view only.",
            ),
        ], "control_panel")

    def _get_line_threshold(self) -> float:
        """This view's own threshold once set, else the shared user setting."""
        if self._line_threshold is not None:
            return self._line_threshold
        return super()._get_line_threshold()

    def _set_line_threshold(self, value: float) -> None:
        """Set this view's threshold (not the user setting) and redraw; values outside 0-1 are refused."""
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            notify_user(self._islat, "The line threshold must be a number from 0 to 1.")
            return
        changed = value != self._get_line_threshold()
        self._line_threshold = value
        if changed:
            self._rebuild_and_show()
            self.draw()

    # ==================================================================
    # Pick event - active line interaction
    # ==================================================================
    def _on_canvas_created(self, canvas: Any) -> None:
        """Keep the pick handler attached to every new canvas.

        The view redraws into the same figure and canvas, but the canvas is
        built again if the figure is ever replaced.  The old pick connection
        dies with the old canvas, so forget its id and connect again on the
        new one; otherwise clicks on line markers would stop working.
        """
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
        picked_value = next(
            (value for vline, text_obj, _sc, value in self.active_lines
             if picked_artist is vline or picked_artist is text_obj),
            None,
        )
        self._recolor_active_lines(picked_value)

        if picked_value is not None:
            self.selected_line = picked_value
            self._display_line_info(picked_value)

        if self._canvas is not None:
            self._canvas.draw_idle()

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
        # The shared threshold this view follows may have changed while it was hidden.
        if self._plot is not None and self._drawn_threshold != self._get_line_threshold():
            self._needs_refresh = True

        if self._initialised and self._shown_frame_is_current():
            # Unchanged while hidden: re-pack the canvas (it keeps its last frame) without redrawing.
            self._register_control_fields()
            self._connect_pick_event()
            self._show_canvas()
            self._repaint_shown_frame()
            return

        rebuilt_on = None
        if not self._initialised or self._needs_refresh:
            rebuilt_on = self._canvas
            self._build_plot()
            self._initialised = True
            self._needs_refresh = False

        self._ensure_canvas()
        if rebuilt_on is not None and self._canvas is rebuilt_on:
            # New content on the kept canvas: its zoom history no longer applies.
            self._notify_canvas_changed()
        self.apply_theme(self._pm.theme)
        self.sync_toggle_state(self._pm.toggle_state)
        self._register_control_fields()
        self._connect_pick_event()
        self._show_canvas()
        self.draw()

    def _rebuild_and_show(self) -> None:
        """Redraw the plot (once the view has been shown) on its canvas, themed.

        Before the first activation the view is only marked stale; the
        next :meth:`activate` builds it.  The canvas is kept, so the main
        toolbar is told the content changed: zoom history from the previous
        selection no longer applies, and the new axes need its pan / zoom mode.
        """
        self._needs_refresh = True
        if not self._initialised:
            return
        canvas = self._canvas
        self._build_plot()
        self._needs_refresh = False
        self._show_canvas()
        self.apply_theme(self._pm.theme)
        if canvas is not None and self._canvas is canvas:
            # A new canvas has already told the toolbar (_ensure_canvas).
            self._notify_canvas_changed()

    # ==================================================================
    # PlotView - interaction context
    # ==================================================================
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Return the line-inspection right-click menu."""
        menu = self._build_line_inspection_menu(canvas_widget, event)
        if menu is not None:
            self._append_save_figure_item(menu)
        return menu

    def deactivate(self) -> None:
        """Disconnect the pick event, then unregister controls and unpack the canvas."""
        self._remember_shown_frame()
        self._disconnect_pick_event()
        super().deactivate()

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
        self._rebuild_and_show()

    # ==================================================================
    # PlotView - selection (override base no-ops)
    # ==================================================================
    def on_selection(self, xmin: float, xmax: float) -> None:
        """Update the displayed wavelength range when the user makes a new selection."""
        self._current_selection = (xmin, xmax)
        self._rebuild_and_show()

    def clear_selection(self) -> None:
        """Clear the selection and show the placeholder."""
        self._current_selection = None
        self._rebuild_and_show()

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

    def select_line(self, ref: "LineRef") -> bool:
        """Select the line marker *ref* describes, as clicking it does.

        Returns ``False`` when *ref* is not one of the marked lines.
        """
        from types import SimpleNamespace

        entry = self._inspected_entry(ref, with_marker=True)
        if entry is None:
            return False
        self._on_pick_line(SimpleNamespace(artist=entry[0]))
        return self.selected_line is entry[3]
