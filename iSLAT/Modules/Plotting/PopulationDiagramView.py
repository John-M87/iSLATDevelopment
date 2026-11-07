"""
PopulationDiagramView - :class:`PlotView` for a standalone full-size population (Boltzmann) diagram.

Shows the :class:`PopulationDiagramPlot` for the currently active molecule in a dedicated figure that fills the main plotting area.
All spectrum-specific toggles (summed spectrum, saved lines, atomic lines) are no-ops for this view.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .PlotView import PlotView
from .PopulationDiagramPlot import PopulationDiagramPlot
from .PopulationDiagramContextMixin import PopulationDiagramContextMixin

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

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

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    FigureCanvasTkAgg = None

class PopulationDiagramView(PlotView, PopulationDiagramContextMixin):
    """
    Standalone population-diagram view.

    Renders a :class:`PopulationDiagramPlot` for the currently active molecule (or all visible molecules when no single molecule is selected).
    The view owns its own :class:`~matplotlib.figure.Figure` and :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` so
    it can be swapped in / out of the main window independently of the three-panel and full-spectrum views.

    Parameters
    ----------
    plot_manager : iSLATPlot
        The main controller (``iSLAT.Modules.Plotting.MainPlot.iSLATPlot``).
    """
    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat

        self._parent_frame: Any = None
        self._canvas: Optional["FigureCanvasTkAgg"] = None
        self._plot: Optional[PopulationDiagramPlot] = None
        self._fig: Optional["Figure"] = None

        self._initialised: bool = False
        self._needs_refresh: bool = True

        # Pick-event cid so we can disconnect on deactivate
        self._pick_cid: Optional[int] = None
        # Right-click event cid on our own canvas
        self._right_click_cid: Optional[int] = None

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _get_active_molecule(self) -> Optional["Molecule"]:
        return getattr(self._islat, "active_molecule", None)

    def _get_molecules(self) -> Optional["MoleculeDict"]:
        return getattr(self._islat, "molecules_dict", None)

    def _discard_plot(self) -> None:
        """Release the current plot so its MoleculeDict callbacks are dropped."""
        plot = self._plot
        self._plot = None
        if plot is None:
            return
        try:
            plot.close()
        except Exception:
            pass

    def _build_plot(self) -> None:
        """Create a new :class:`PopulationDiagramPlot` and render it."""
        mol = self._get_active_molecule()
        mols = self._get_molecules()

        self._discard_plot()

        # Destroy old figure if present
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

        self._fig = plt.Figure(figsize=(9, 7), constrained_layout=True)
        ax = self._fig.add_subplot(111)

        if mol is not None:
            self._plot = PopulationDiagramPlot(
                molecule=mol,
                ax=ax,
                theme=self._pm.theme,
            )
        elif mols is not None and len(mols) > 0:
            self._plot = PopulationDiagramPlot(
                molecules=mols,
                ax=ax,
                theme=self._pm.theme,
            )
        else:
            # No data yet - draw an empty placeholder
            self._plot = None
            ax.set_title("Population Diagram")
            ax.set_xlabel("Upper energy level  $E_u$  (K)")
            ax.set_ylabel(r"$\ln \left(\frac{4\pi F}{h\nu\,A_u\,g_u}\right)$")
            ax.text(
                0.5, 0.5, "No molecule selected",
                ha="center", va="center",
                transform=ax.transAxes,
                color="gray", fontsize=12,
            )
            return

        try:
            self._plot.generate_plot()
        except Exception as exc:
            debug_config.warning(
                "population_diagram_view",
                f"generate_plot failed: {exc}",
            )

    def _refresh_plot(self) -> None:
        """Re-render onto the *existing* figure/axes without destroying the canvas.

        Called by :meth:`update_model_plot` and
        :meth:`on_molecule_parameter_changed` when the view is already
        active.  Clears only the axes content so the Tk canvas widget
        stays packed and visible.
        """
        if self._fig is None:
            # No figure yet - fall back to a full build.
            self._build_plot()
            return

        mol = self._get_active_molecule()
        mols = self._get_molecules()

        self._discard_plot()

        for ax in self._fig.axes:
            ax.clear()

        ax = self._fig.axes[0] if self._fig.axes else self._fig.add_subplot(111)

        if mol is not None:
            self._plot = PopulationDiagramPlot(
                molecule=mol, ax=ax, theme=self._pm.theme,
            )
        elif mols is not None and len(mols) > 0:
            self._plot = PopulationDiagramPlot(
                molecules=mols, ax=ax, theme=self._pm.theme,
            )
        else:
            self._plot = None
            ax.set_title("Population Diagram")
            ax.set_xlabel("Upper energy level  $E_u$  (K)")
            ax.set_ylabel(r"$\ln \left(\frac{4\pi F}{h\nu\,A_u\,g_u}\right)$")
            ax.text(
                0.5, 0.5, "No molecule selected",
                ha="center", va="center",
                transform=ax.transAxes,
                color="gray", fontsize=12,
            )
            return

        try:
            self._plot.generate_plot()
        except Exception as exc:
            debug_config.warning(
                "population_diagram_view",
                f"_refresh_plot generate_plot failed: {exc}",
            )

    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the :class:`FigureCanvasTkAgg`."""
        if self._canvas is not None:
            if self._fig is not None and self._canvas.figure is self._fig:
                return
            # Disconnect old event before destroying
            if self._right_click_cid is not None:
                try:
                    self._canvas.mpl_disconnect(self._right_click_cid)
                except Exception:
                    pass
                self._right_click_cid = None
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._fig is not None and FigureCanvasTkAgg is not None and self._parent_frame is not None:
            self._canvas = FigureCanvasTkAgg(self._fig, master=self._parent_frame)
            self._right_click_cid = self._canvas.mpl_connect(
                'button_press_event', self._on_canvas_button_press
            )
            self._notify_canvas_changed()

    def _apply_theme_to_fig(self) -> None:
        """Apply the controller's current theme to the owned figure."""
        theme = self._pm.theme
        bg = theme.get("background", "#181A1B")
        fg = theme.get("foreground", "#F0F0F0")

        if self._fig is None:
            return

        self._fig.patch.set_facecolor(bg)
        for ax in self._fig.axes:
            ax.set_facecolor(theme.get("axes_background", bg))
            ax.tick_params(colors=fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_edgecolor(fg)

        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().configure(bg=bg)
            except Exception:
                pass

    # ==================================================================
    # PlotView - lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        """Pack the population-diagram canvas into *parent_frame*."""
        self._parent_frame = parent_frame

        if not self._initialised or self._needs_refresh:
            self._build_plot()
            self._initialised = True
            self._needs_refresh = False

        self._ensure_canvas()

        self.apply_theme(self._pm.theme)

        # Register view-specific controls with the ControlBus
        self._register_control_fields()

        if self._canvas is not None:
            self._canvas.get_tk_widget().pack(
                fill="both", expand=True, padx=0, pady=0
            )
            self._canvas.draw_idle()

    def _on_canvas_button_press(self, event) -> None:
        """Handle button-press events on the view's own canvas."""
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
            y_root = canvas_widget.winfo_rooty() + int(canvas_widget.winfo_height() - event.y)
            menu.tk_popup(x_root, y_root)
        except Exception:
            pass
        finally:
            menu.grab_release()

    def deactivate(self) -> None:
        """Unpack the canvas from its parent frame."""
        # Unregister all ControlBus fields this view registered
        bus = getattr(self._pm, 'control_bus', None)
        if bus is not None:
            bus.unregister_owner(self)

        # Stop live all-molecules re-rendering while the view is hidden; the
        # plot is rebuilt on the next activate().
        if self._plot is not None and getattr(self._plot, "_all_molecules_mode", False):
            self._plot._exit_all_molecules_mode()
            self._needs_refresh = True

        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().pack_forget()
            except Exception:
                pass

    # ==================================================================
    # PlotView - interaction context
    # ==================================================================
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Delegate to the shared population-diagram context menu builder."""
        def draw_idle() -> None:
            # Menu actions regenerate the plot, so earlier zoom history no longer applies.
            self._notify_canvas_changed()
            if self._canvas is not None:
                self._canvas.draw_idle()

        menu = self._build_population_diagram_menu(self._plot, canvas_widget, draw_idle)
        if menu is not None:
            self._append_save_figure_item(menu)
        return menu

    # ==================================================================
    # PlotView - theme
    # ==================================================================
    def apply_theme(self, theme: dict) -> None:
        self._pm.theme = theme
        self._apply_theme_to_fig()
        if self._canvas is not None:
            self._canvas.draw_idle()

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
        """Re-render the population diagram from current iSLAT data."""
        self._needs_refresh = True
        if self._initialised:
            # Re-render in-place so the packed canvas widget stays intact.
            self._refresh_plot()
            self._needs_refresh = False
            self._apply_theme_to_fig()
            self._notify_canvas_changed()
            if self._canvas is not None:
                self._canvas.draw_idle()

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
        """Rebuild the diagram when a molecule's visibility changes."""
        self.update_model_plot()

    # ==================================================================
    # PlotView - selection / line inspection (no-op for this view)
    # ==================================================================
    def on_selection(self, xmin: float, xmax: float) -> None:
        """No-op - this view does not show a spectrum."""
        pass

    def clear_selection(self) -> None:
        """No-op."""
        pass

    def clear_active_lines(self) -> None:
        """No-op - no active-line markers in this view."""
        pass

    # ==================================================================
    # PlotView - molecule lifecycle
    # ==================================================================
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Re-render for the newly selected active molecule."""
        self.update_model_plot()

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Re-render when any molecule parameter changes."""
        if parameter_name == "is_visible":
            return
        if not self._initialised or self._fig is None:
            self._needs_refresh = True
            return
        # Refresh in-place - do NOT destroy and recreate the canvas.
        self._refresh_plot()
        self._apply_theme_to_fig()
        self._notify_canvas_changed()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Re-render after a molecule is deleted."""
        self.update_model_plot()

    # ==================================================================
    # PlotView - toggle helpers (most are no-ops for this view)
    # ==================================================================
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """No visible toggles apply to the population diagram view."""
        pass

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """No-op - no spectrum in this view."""
        pass

    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Toggle the legend on the population diagram axes."""
        if self._fig is None:
            return
        for ax in self._fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                if visible is None:
                    leg.set_visible(not leg.get_visible())
                else:
                    leg.set_visible(visible)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """No-op - no saved lines in the population diagram."""
        pass

    def toggle_atomic_lines(self, show: bool) -> None:
        """No-op - no atomic lines in the population diagram."""
        pass

    # ==================================================================
    # PlotView - canvas / drawing
    # ==================================================================
    def draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas

    def get_figure(self) -> "Figure":
        return self._fig