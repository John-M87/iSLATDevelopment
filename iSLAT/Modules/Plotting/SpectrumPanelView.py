"""
SpectrumPanelView - intermediate abstract base class for standalone single-panel views that own their own figure and canvas.

Provides the common infrastructure shared by views such as :class:`LineInspectionView`:

* Private :class:`~matplotlib.figure.Figure` /
  :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` lifecycle
  (``_ensure_canvas``, ``_apply_theme_to_fig``, ``_on_canvas_button_press``)
* Concrete implementations for all ``PlotView`` toggle and
  molecule-lifecycle abstract methods so subclasses only need to supply
  ``activate``, ``_build_plot``, and ``update_model_plot``.
* No-op defaults for selection / active-line helpers (override as needed).
"""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .PlotView import PlotView

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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
except ImportError:
    _FigureCanvasTkAgg = None

class SpectrumPanelView(PlotView):
    """
    Intermediate abstract base for standalone spectrum-panel views.

    Concrete subclasses must implement :meth:`activate`,
    :meth:`_build_plot`, and :meth:`update_model_plot`.  All other
    :class:`PlotView` abstract methods are given sensible defaults here so
    subclasses only need to override the behaviour that is genuinely
    different.

    Expected instance attributes (set in subclass ``__init__`` via
    ``super().__init__(plot_manager)``):

    * ``_pm`` - the :class:`iSLATPlot` controller
    * ``_islat`` - the top-level iSLAT object
    * ``_parent_frame`` - Tk frame the canvas is packed into
    * ``_canvas`` - :class:`FigureCanvasTkAgg` (or ``None``)
    * ``_fig`` - :class:`~matplotlib.figure.Figure` (or ``None``)
    * ``_initialised`` - ``bool``
    * ``_needs_refresh`` - ``bool``
    * ``_right_click_cid`` - matplotlib connection id (or ``None``)
    """
    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat

        self._parent_frame: Any = None
        self._canvas: Optional["FigureCanvasTkAgg"] = None
        self._fig: Optional["Figure"] = None

        self._initialised: bool = False
        self._needs_refresh: bool = True

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
    # Common canvas helpers
    # ==================================================================
    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the :class:`FigureCanvasTkAgg`."""
        if self._canvas is not None:
            if self._fig is not None and self._canvas.figure is self._fig:
                return
            # Disconnect the right-click handler before destroying the old canvas
            if self._right_click_cid is not None:
                try:
                    self._canvas.mpl_disconnect(self._right_click_cid)
                except Exception:
                    pass
                self._right_click_cid = None
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

        if (
            self._fig is not None
            and _FigureCanvasTkAgg is not None
            and self._parent_frame is not None
        ):
            self._canvas = _FigureCanvasTkAgg(self._fig, master=self._parent_frame)
            self._right_click_cid = self._canvas.mpl_connect(
                "button_press_event", self._on_canvas_button_press
            )
            self._notify_canvas_changed()

    def _apply_theme_to_fig(self) -> None:
        """Apply ``self._pm.theme`` to the owned figure and canvas widget."""
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

    # ==================================================================
    # PlotView - lifecycle (concrete defaults)
    # ==================================================================
    def deactivate(self) -> None:
        """Unregister ControlBus fields and unpack the canvas."""
        bus = getattr(self._pm, "control_bus", None)
        if bus is not None:
            bus.unregister_owner(self)
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().pack_forget()
            except Exception:
                pass

    # ==================================================================
    # PlotView - theme (concrete default)
    # ==================================================================
    def apply_theme(self, theme: dict) -> None:
        self._pm.theme = theme
        self._apply_theme_to_fig()
        if self._canvas is not None:
            self._canvas.draw_idle()

    # ==================================================================
    # PlotView - canvas / drawing (concrete)
    # ==================================================================
    def draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas

    def get_figure(self) -> "Figure":
        return self._fig

    def is_ready(self) -> bool:
        return self._initialised

    # ==================================================================
    # PlotView - toggle helpers (no-ops / simple defaults)
    # ==================================================================
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """No visible toggles apply to spectrum-panel views by default."""
        pass

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """No-op - override if the view shows a summed model."""
        pass

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """No-op - override if the view shows saved-line annotations."""
        pass

    def toggle_atomic_lines(self, show: bool) -> None:
        """No-op - override if the view shows atomic-line annotations."""
        pass

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

    # ==================================================================
    # PlotView - selection / active-line helpers (no-op defaults)
    # ==================================================================
    def on_selection(self, xmin: float, xmax: float) -> None:
        """No-op - override in views that respond to span-selector drags."""
        pass

    def clear_selection(self) -> None:
        """No-op - override in views that track a selection."""
        pass

    def clear_active_lines(self) -> None:
        """No-op - override in views that draw active-line markers."""
        pass