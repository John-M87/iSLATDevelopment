"""
PopulationDiagramView - :class:`SpectrumPanelView` for a standalone full-size population (Boltzmann) diagram.

Shows the :class:`PopulationDiagramPlot` for the currently active molecule in a dedicated figure that fills the main plotting area.
The canvas lifecycle, theme, legend toggle and molecule callbacks come from :class:`SpectrumPanelView`;
the spectrum-specific toggles (summed spectrum, saved lines, atomic lines) and selection are :class:`PlotView` no-ops.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib.pyplot as plt

from .SpectrumPanelView import SpectrumPanelView
from .PopulationDiagramPlot import PopulationDiagramPlot
from .PopulationDiagramContextMixin import PopulationDiagramContextMixin
from .PlotDebug import debug_config

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class PopulationDiagramView(SpectrumPanelView, PopulationDiagramContextMixin):
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
        super().__init__(plot_manager)
        self._plot: Optional[PopulationDiagramPlot] = None

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

        self._show_canvas()
        self.draw()

    def deactivate(self) -> None:
        """Stop live all-molecules re-rendering, then unregister controls and unpack the canvas."""
        # Stop live all-molecules re-rendering while the view is hidden; the
        # plot is rebuilt on the next activate().
        if self._plot is not None and getattr(self._plot, "_all_molecules_mode", False):
            self._plot._exit_all_molecules_mode()
            self._needs_refresh = True

        super().deactivate()

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

    # ==================================================================
    # PlotView - molecule lifecycle
    # ==================================================================
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
