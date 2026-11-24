"""
PopulationDiagramView - :class:`SpectrumPanelView` for a standalone full-size population (Boltzmann) diagram.

Shows the :class:`PopulationDiagramPlot` for the currently active molecule in a dedicated figure that fills the main plotting area.
The canvas lifecycle, theme, legend toggle and molecule callbacks come from :class:`SpectrumPanelView`;
the spectrum-specific toggles (summed spectrum, saved lines, atomic lines) and selection are :class:`PlotView` no-ops.
"""
from __future__ import annotations

from typing import Any, Optional

from .SpectrumPanelView import SpectrumPanelView
from .PopulationDiagramPlot import PopulationDiagramPlot
from .PopulationDiagramContextMixin import PopulationDiagramContextMixin
from .PlotDebug import debug_config

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

    def _discard_plot(self) -> None:
        """Release the current plot so its MoleculeDict callbacks are dropped."""
        plot, self._plot = self._plot, None
        if plot is not None:
            try:
                plot.close()
            except Exception:
                pass

    def _build_plot(self) -> None:
        """Draw the diagram of the active molecule (else of all molecules) into the owned figure."""
        self._discard_plot()
        ax = self._fresh_axes((9, 7))
        mol, mols = self._get_active_molecule(), self._get_molecules()
        if mol is None and not mols:
            self._draw_placeholder(ax, "Population Diagram", "Upper energy level  $E_u$  (K)",
                                   r"$\ln \left(\frac{4\pi F}{h\nu\,A_u\,g_u}\right)$", "No molecule selected")
            return
        source = {"molecule": mol} if mol is not None else {"molecules": mols}
        self._plot = PopulationDiagramPlot(**source, ax=ax, theme=self._pm.theme)
        try:
            self._plot.generate_plot()
        except Exception as exc:
            debug_config.warning("population_diagram_view", f"generate_plot failed: {exc}")

    def deactivate(self) -> None:
        """Stop live all-molecules re-rendering (the plot is rebuilt on the next activation), then hide the view."""
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
