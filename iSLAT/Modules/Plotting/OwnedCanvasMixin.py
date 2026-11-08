"""
OwnedCanvasMixin - the Tk canvas of a view that owns its figure.

:class:`SpectrumPanelView` (the single-panel views) and
:class:`FullSpectrumView` (and its subclasses) each show a figure of their
own in a :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`.
This mixin gives them one implementation of the canvas lifecycle:

* :meth:`_ensure_canvas` - (re)build the canvas when the figure changes
* :meth:`_show_canvas` / :meth:`_show_canvas_if_replaced` - and pack it
  into the parent frame
* :meth:`_on_canvas_button_press` - the right-click context menu
* :meth:`deactivate`, :meth:`draw` and :meth:`get_canvas`

The host is a :class:`PlotView`: it provides ``get_figure()`` (the figure
to show), ``build_context_menu()``, ``_notify_canvas_changed()`` and the
``_pm`` / ``_canvas`` / ``_parent_frame`` attributes.  List the mixin
before :class:`PlotView` in the bases so its methods are used.

Usage::

    class MyView(OwnedCanvasMixin, PlotView):
        def get_figure(self):
            return self._fig
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from .HoldableCanvas import HoldableFigureCanvasTkAgg as _FigureCanvasTkAgg
except ImportError:  # pragma: no cover - Tk is not available
    _FigureCanvasTkAgg = None

#: Pack options of a view's canvas widget inside its parent frame.
CANVAS_PACK_OPTIONS: Dict[str, Any] = {"fill": "both", "expand": True, "padx": 0, "pady": 0}


def popup_menu(menu: Any, canvas_widget: Any, event: Any) -> bool:
    """Pop *menu* up where the matplotlib mouse *event* happened on *canvas_widget*.

    Matplotlib measures y up from the bottom of the canvas, Tk down from the
    top.  Returns ``False`` when the menu could not be shown.
    """
    try:
        x_root = canvas_widget.winfo_rootx() + int(event.x)
        y_root = canvas_widget.winfo_rooty() + int(canvas_widget.winfo_height() - event.y)
        menu.tk_popup(x_root, y_root)
        return True
    except Exception:
        return False
    finally:
        try:
            menu.grab_release()
        except Exception:
            pass


class OwnedCanvasMixin:
    """Canvas lifecycle for :class:`PlotView` subclasses that own their figure."""

    #: matplotlib connection id of the right-click handler on the current canvas.
    _right_click_cid: Optional[int] = None

    # ------------------------------------------------------------------
    # Canvas lifecycle
    # ------------------------------------------------------------------
    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the canvas for the figure returned by ``get_figure()``.

        A canvas that still shows that figure is kept.  Otherwise the old
        canvas is destroyed and - when there is a figure and a parent frame -
        a new one is created with the right-click handler connected.  The
        plot manager is told about the new canvas (so the toolbar can follow
        it) and :meth:`_on_canvas_created` runs.
        """
        fig = self.get_figure()
        if self._canvas is not None:
            if fig is not None and self._canvas.figure is fig:
                return
            self._destroy_canvas()
        if fig is None or _FigureCanvasTkAgg is None or self._parent_frame is None:
            return
        self._canvas = _FigureCanvasTkAgg(fig, master=self._parent_frame)
        # A new widget starts at the figure's own size and Tk resizes it once
        # it is packed: draw only when that has happened, not at both sizes.
        if hasattr(self._canvas, "hold_draws_until_laid_out"):
            self._canvas.hold_draws_until_laid_out()
        self._right_click_cid = self._canvas.mpl_connect(
            "button_press_event", self._on_canvas_button_press,
        )
        self._notify_canvas_changed()
        self._on_canvas_created(self._canvas)

    def _on_canvas_created(self, canvas: Any) -> None:
        """Hook called with each newly built canvas, e.g. to connect more event handlers.  Default: nothing."""

    def _destroy_canvas(self) -> None:
        """Disconnect the right-click handler and destroy the canvas widget."""
        canvas, self._canvas = self._canvas, None
        if canvas is None:
            return
        if self._right_click_cid is not None:
            try:
                canvas.mpl_disconnect(self._right_click_cid)
            except Exception:
                pass
            self._right_click_cid = None
        canvas.get_tk_widget().destroy()

    def _show_canvas(self) -> None:
        """Ensure the canvas exists for the current figure and pack it into the parent frame."""
        self._ensure_canvas()
        if self._canvas is not None and self._parent_frame is not None:
            self._canvas.get_tk_widget().pack(**CANVAS_PACK_OPTIONS)

    def _show_canvas_if_replaced(self, old_fig: Any) -> None:
        """Rebuild and repack the canvas when the figure is no longer *old_fig* (e.g. after a plot rebuild)."""
        if self.get_figure() is not old_fig:
            self._show_canvas()

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------
    def _on_canvas_button_press(self, event: Any) -> None:
        """Route right-clicks to :meth:`build_context_menu` and pop the menu up at the mouse."""
        if event.button != 3 or self._canvas is None:
            return
        try:
            canvas_widget = self._canvas.get_tk_widget()
        except Exception:
            return
        menu = self.build_context_menu(event, canvas_widget)
        if menu is not None:
            popup_menu(menu, canvas_widget, event)

    # ------------------------------------------------------------------
    # PlotView methods
    # ------------------------------------------------------------------
    def deactivate(self) -> None:
        """Unregister this view's ControlBus fields and unpack the canvas (it is kept for the next activation)."""
        bus = getattr(self._pm, "control_bus", None)
        if bus is not None:
            bus.unregister_owner(self)
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().pack_forget()
            except Exception:
                pass

    def draw(self) -> None:
        """Redraw the canvas when it is idle."""
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> Any:
        """The canvas showing this view (``None`` until it has been built)."""
        return self._canvas
