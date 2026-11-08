"""
SpectrumPanelView - intermediate abstract base class for standalone single-panel views that own their own figure and canvas.

Provides the common infrastructure shared by :class:`LineInspectionView` and :class:`PopulationDiagramView`:

* The private :class:`~matplotlib.figure.Figure` (``_fig``) shown in a
  canvas whose lifecycle comes from :class:`OwnedCanvasMixin`
  (``_ensure_canvas``, the right-click menu, ``deactivate``, ``draw``)
* Concrete ``apply_theme``, ``toggle_legend``, ``get_figure`` and
  molecule-lifecycle callbacks (each rebuilds via ``update_model_plot``),
  so subclasses only need to supply ``activate``, ``_build_plot``, and
  ``update_model_plot``.

Selection and the remaining overlay toggles use the no-op defaults from :class:`PlotView`.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .OwnedCanvasMixin import OwnedCanvasMixin
from .PlotView import PlotView
from .Theming import apply_canvas_theme, apply_figure_theme

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class SpectrumPanelView(OwnedCanvasMixin, PlotView):
    """
    Intermediate abstract base for standalone spectrum-panel views.

    Concrete subclasses must implement :meth:`activate`,
    :meth:`_build_plot`, and :meth:`update_model_plot`.  All other
    :class:`PlotView` abstract methods are given sensible defaults here
    (the canvas ones by :class:`OwnedCanvasMixin`) so subclasses only need
    to override the behaviour that is genuinely different.

    Instance attributes (set by ``super().__init__(plot_manager)`` in the
    subclass ``__init__``):

    * ``_pm``, ``_islat``, ``_parent_frame``, ``_canvas``, ``_initialised``,
      ``_needs_refresh`` - see :class:`PlotView`
    * ``_fig`` - :class:`~matplotlib.figure.Figure` (or ``None``)
    * ``_right_click_cid`` - matplotlib connection id (or ``None``), see
      :class:`OwnedCanvasMixin`
    """
    def __init__(self, plot_manager: Any) -> None:
        super().__init__(plot_manager)
        self._fig: Optional["Figure"] = None

        # Connection id for the right-click handler on this view's canvas
        self._right_click_cid: Optional[int] = None

    # ==================================================================
    # Abstract - subclasses must implement
    # ==================================================================
    @abstractmethod
    def activate(self, parent_frame: Any) -> None:  # pragma: no cover
        """Pack the view's canvas into *parent_frame* and ensure content is fresh."""
        ...

    @abstractmethod
    def _build_plot(self) -> None:  # pragma: no cover
        """(Re)create ``self._fig`` and populate it with fresh content."""
        ...

    @abstractmethod
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:  # pragma: no cover
        """Full re-render triggered by global data or parameter changes."""
        ...

    # ==================================================================
    # Theme
    # ==================================================================
    def _apply_theme_to_fig(self) -> None:
        """Apply ``self._pm.theme`` to the owned figure and canvas widget."""
        if self._fig is None:
            return
        apply_figure_theme(self._fig, self._pm.theme)
        apply_canvas_theme(self._canvas, self._pm.theme)

    def apply_theme(self, theme: dict) -> None:
        self._pm.theme = theme
        self._apply_theme_to_fig()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ==================================================================
    # PlotView - figure (the canvas methods come from OwnedCanvasMixin)
    # ==================================================================
    def get_figure(self) -> "Figure":
        return self._fig

    def is_ready(self) -> bool:
        return self._initialised

    # ==================================================================
    # PlotView - toggle helpers (the other toggles keep PlotView's no-ops)
    # ==================================================================
    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Toggle the legend on all axes of the owned figure."""
        if self._fig is None:
            return
        for ax in self._fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                new_vis = (not leg.get_visible()) if visible is None else visible
                leg.set_visible(new_vis)
        self.draw()

    # ==================================================================
    # PlotView - molecule lifecycle (sensible defaults)
    # ==================================================================
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
        """Rebuild the plot when a molecule's visibility changes."""
        self.update_model_plot()

    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Rebuild for the newly selected active molecule."""
        self.update_model_plot()

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Rebuild when a molecule parameter changes (skip visibility-only events)."""
        if parameter_name == "is_visible":
            return
        self.update_model_plot()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Rebuild after a molecule is removed."""
        self.update_model_plot()