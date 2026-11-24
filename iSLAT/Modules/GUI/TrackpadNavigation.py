# -*- coding: utf-8 -*-
"""Trackpad navigation for matplotlib canvases embedded in Tk.

:func:`attach_trackpad_navigation` gives a ``FigureCanvasTkAgg``:

- **Two-finger pan.**  On Tk 9 under macOS the trackpad sends
  ``<TouchpadScroll>`` events, separate from a mouse wheel's
  ``<MouseWheel>``, so the trackpad pans and the wheel keeps its old
  behaviour.  Where the two cannot be told apart (Tk 8.6, or Tk on Windows
  and Linux) wheel/trackpad scrolling pans instead: vertical scrolling pans
  vertically, Shift+scroll (or a sideways swipe) horizontally.
- **Pinch zoom** (macOS).  Tk has no pinch event, so magnify gestures are
  read from Cocoa through PyObjC (the optional ``macos`` extra).  Without
  PyObjC pinch is simply unavailable.
- **Ctrl+scroll zoom** on every platform, around the pointer.

Pan and zoom are applied with matplotlib's own ``Axes.start_pan`` /
``drag_pan`` / ``end_pan``, so log axes, shared axes and twin axes behave as
they do with the toolbar's pan tool.  A burst of gesture events counts as one
navigation step: the toolbar's Back button returns to the view before it.

The Cocoa callback runs while Tk is dispatching events, and calling into Tk
from there can corrupt ``_tkinter``'s thread-state bookkeeping.  It therefore
only appends to a Python list; a Tk timer (:data:`PINCH_POLL_MS`) applies the
queued zoom from the normal event loop.

:func:`bind_touchpad_scroll` and :func:`scroll_canvas_by_pixels` let plain
scrollable Tk canvases (scrolling frames) follow ``<TouchpadScroll>`` too,
which Tk 9 sends instead of ``<MouseWheel>`` for the trackpad.
"""
from __future__ import annotations

import math
import sys
import tkinter as tk
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = [
    "TrackpadNavigator",
    "attach_trackpad_navigation",
    "bind_touchpad_scroll",
    "decode_touchpad_delta",
    "install_pinch_monitor",
    "scroll_canvas_by_pixels",
    "touchpad_scroll_supported",
    "touchpad_distinct_from_wheel",
]

#: Zoom factor per pixel of Ctrl + two-finger scroll (``<TouchpadScroll>``).
TOUCHPAD_ZOOM_PER_PIXEL = 0.01
#: Zoom factor per wheel step (one notch = 1.0) for Ctrl + mouse wheel.
WHEEL_ZOOM_PER_STEP = 1.15
#: Pan distance (pixels) per wheel step where the wheel pans (no separate touchpad events).
WHEEL_PAN_PIXELS_PER_STEP = 40.0
#: Pan distance (pixels) per unit of raw ``<MouseWheel>`` delta on macOS Tk 8.6,
#: which reports trackpad scrolling in coarse "lines".
MAC_TK86_PAN_PIXELS_PER_DELTA = 10.0
#: Quiet time (ms) after which a burst of gesture events counts as finished.
GESTURE_END_MS = 250
#: How often (ms) queued pinch events are applied.
PINCH_POLL_MS = 20

_SHIFT_MASK = 1 << 0
_CONTROL_MASK = 1 << 2

#: Attribute set on a canvas's Tk widget pointing back at its navigator.
_NAV_ATTR = "_islat_trackpad_nav"


# ----------------------------------------------------------------------
# Tk helpers
# ----------------------------------------------------------------------
def touchpad_scroll_supported(widget: tk.Misc) -> bool:
    """Return whether this Tk knows the ``<TouchpadScroll>`` event (Tk 9+)."""
    try:
        return int(str(widget.tk.getvar("tk_version")).split(".")[0]) >= 9
    except (tk.TclError, ValueError):
        return False


def touchpad_distinct_from_wheel(widget: tk.Misc) -> bool:
    """Return whether trackpad scrolling arrives separately from mouse-wheel scrolling.

    True for Tk 9 on macOS, where the trackpad sends ``<TouchpadScroll>``.
    Elsewhere both arrive as ``<MouseWheel>`` and cannot be told apart.
    """
    return sys.platform == "darwin" and touchpad_scroll_supported(widget)


def decode_touchpad_delta(delta: int) -> Tuple[int, int]:
    """Split a ``<TouchpadScroll>`` ``%D`` value into ``(dx, dy)`` pixels.

    Same as Tk's ``tk::PreciseScrollDeltas``: the high 16 bits hold dx and
    the low 16 bits dy, both signed.  Positive values scroll content right
    and down (Tk's own widgets scroll their view by ``-dx`` / ``-dy``).
    """
    delta = int(delta) & 0xFFFFFFFF
    if delta >= 0x80000000:  # Tk passes a signed 32-bit int; accept unsigned too
        delta -= 0x100000000
    dx = delta >> 16
    low = delta & 0xFFFF
    dy = low if low < 0x8000 else low - 0x10000
    return dx, dy


def scroll_canvas_by_pixels(canvas: tk.Canvas, dx: float = 0.0, dy: float = 0.0) -> None:
    """Scroll a Tk canvas's view by *dx*, *dy* pixels (positive = right / down).

    Tk canvases only scroll by units or pages, so this converts pixels to a
    fraction of the ``scrollregion``.
    """
    try:
        region = canvas.tk.splitlist(canvas.cget("scrollregion"))
    except tk.TclError:
        return
    if len(region) != 4:
        return
    x0, y0, x1, y1 = (float(v) for v in region)
    if dx and x1 > x0:
        canvas.xview_moveto(canvas.xview()[0] + dx / (x1 - x0))
    if dy and y1 > y0:
        canvas.yview_moveto(canvas.yview()[0] + dy / (y1 - y0))


def bind_touchpad_scroll(
    widget: tk.Misc,
    scroll_canvas: tk.Canvas,
    *,
    vertical: bool = True,
    horizontal: bool = False,
    bind_all: bool = False,
) -> Optional[Callable[[Any], None]]:
    """Make two-finger trackpad scrolling over *widget* scroll *scroll_canvas* (Tk 9).

    With only *horizontal* scrolling, vertical finger movement scrolls
    sideways too (like the plain mouse wheel in such frames).  With
    *bind_all* the binding is made with ``bind_all`` (for frames that bind
    on ``<Enter>`` and unbind on ``<Leave>``).  Returns the handler, or
    ``None`` when this Tk has no ``<TouchpadScroll>`` event.
    """
    if not touchpad_scroll_supported(widget):
        return None

    def _on_touchpad(event: Any) -> None:
        dx, dy = decode_touchpad_delta(event.delta)
        view_dx = -dx if horizontal else 0
        view_dy = -dy if vertical else 0
        if horizontal and not vertical and not dx:
            view_dx = -dy
        if view_dx or view_dy:
            scroll_canvas_by_pixels(scroll_canvas, view_dx, view_dy)

    if bind_all:
        widget.bind_all("<TouchpadScroll>", _on_touchpad)
    else:
        widget.bind("<TouchpadScroll>", _on_touchpad, add="+")
    return _on_touchpad


# ----------------------------------------------------------------------
# Pinch (macOS magnify gestures through PyObjC)
# ----------------------------------------------------------------------
#: Magnification deltas queued by the Cocoa monitor, applied by the Tk poller.
_pinch_events: List[float] = []
_pinch_monitor: Any = None
_pinch_handler: Any = None
_pinch_unavailable = False


def install_pinch_monitor() -> bool:
    """Start collecting macOS pinch (magnify) gestures; return whether that is possible.

    Installs one process-wide Cocoa local event monitor.  Needs macOS and
    PyObjC (``pyobjc-framework-Cocoa``); returns ``False`` otherwise.
    """
    global _pinch_monitor, _pinch_handler, _pinch_unavailable
    if _pinch_monitor is not None:
        return True
    if _pinch_unavailable or sys.platform != "darwin":
        return False
    try:
        import AppKit  # type: ignore[import-not-found]

        def _on_magnify(ns_event: Any) -> Any:
            # Runs inside Tk's event dispatch: no Tk calls here, only queue it.
            try:
                _pinch_events.append(float(ns_event.magnification()))
            except Exception:
                pass
            return ns_event

        _pinch_handler = _on_magnify
        _pinch_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            AppKit.NSEventMaskMagnify, _on_magnify,
        )
    except Exception:
        _pinch_monitor = None
    if _pinch_monitor is None:
        _pinch_unavailable = True
        return False
    return True


def _take_pinch_factor() -> float:
    """Return the combined zoom factor of all queued pinch events and clear the queue."""
    factor = 1.0
    while _pinch_events:
        magnification = _pinch_events.pop(0)
        factor *= max(1.0 + magnification, 0.05)
    return factor


def _ensure_pinch_poller(widget: tk.Misc) -> None:
    """Start (once per Tk root) the timer that applies queued pinch gestures."""
    root = widget._root()
    if getattr(root, "_islat_pinch_poller", False):
        return
    root._islat_pinch_poller = True  # type: ignore[attr-defined]

    def _poll() -> None:
        try:
            if _pinch_events:
                _dispatch_pinch(root, _take_pinch_factor())
            root.after(PINCH_POLL_MS, _poll)
        except tk.TclError:  # root destroyed
            root._islat_pinch_poller = False  # type: ignore[attr-defined]

    try:
        root.after(PINCH_POLL_MS, _poll)
    except tk.TclError:
        root._islat_pinch_poller = False  # type: ignore[attr-defined]


def _dispatch_pinch(root: tk.Misc, factor: float) -> bool:
    """Zoom the navigator-enabled canvas under the pointer by *factor*; return whether one did."""
    if factor == 1.0:
        return False
    try:
        px, py = root.winfo_pointerxy()
        widget = root.winfo_containing(px, py)
    except (tk.TclError, KeyError):
        return False
    nav = getattr(widget, _NAV_ATTR, None)
    if nav is None or not nav.zoom_enabled:
        return False
    x = px - widget.winfo_rootx()
    y = py - widget.winfo_rooty()
    return nav.zoom_at_widget_point(x, y, factor)


# ----------------------------------------------------------------------
# Navigator
# ----------------------------------------------------------------------
class TrackpadNavigator:
    """Trackpad pan / zoom for one matplotlib Tk canvas (see the module docstring).

    *pan* enables two-finger panning; leave it off for canvases inside a
    scrolling container so two-finger scrolling still scrolls the page.
    *zoom* enables pinch and Ctrl+scroll zoom.
    """

    def __init__(self, canvas: Any, *, pan: bool = True, zoom: bool = True) -> None:
        self.canvas = canvas
        self.pan_enabled = pan
        self.zoom_enabled = zoom
        self.widget: tk.Misc = canvas.get_tk_widget()
        self._gesture_job: Optional[str] = None
        self._frozen_legends: List[Any] = []
        self._touchpad_separate = touchpad_distinct_from_wheel(self.widget)

        setattr(self.widget, _NAV_ATTR, self)
        if touchpad_scroll_supported(self.widget):
            self.widget.bind("<TouchpadScroll>", self._on_touchpad_scroll, add="+")
        self._scroll_cid = canvas.mpl_connect("scroll_event", self._on_mpl_scroll)
        self.pinch_available = False
        if zoom and install_pinch_monitor():
            _ensure_pinch_poller(self.widget)
            self.pinch_available = True

    # -- event handlers ------------------------------------------------
    def _on_touchpad_scroll(self, event: Any) -> Optional[str]:
        """``<TouchpadScroll>``: pan, or zoom with Ctrl held."""
        dx, dy = decode_touchpad_delta(event.delta)
        if not (dx or dy):
            return None
        x, y = self._mpl_xy(event)
        if event.state & _CONTROL_MASK:
            if not self.zoom_enabled or not dy:
                return None
            done = self.zoom_at(x, y, math.exp(dy * TOUCHPAD_ZOOM_PER_PIXEL))
        else:
            if not self.pan_enabled:
                return None
            # Content follows the fingers: right is +x, Tk's +dy is down (matplotlib's -y).
            done = self.pan_by_pixels(x, y, dx, -dy)
        return "break" if done else None

    def _on_mpl_scroll(self, event: Any) -> None:
        """matplotlib ``scroll_event`` (``<MouseWheel>``): Ctrl zooms; pans where the trackpad uses the wheel event."""
        if event.x is None or event.y is None:
            return
        modifiers = set(getattr(event, "modifiers", ()) or ())
        if "ctrl" in modifiers:
            if self.zoom_enabled and event.step:
                self.zoom_at(event.x, event.y, WHEEL_ZOOM_PER_STEP ** event.step)
            return
        if not self.pan_enabled or self._touchpad_separate:
            return
        pixels = self._wheel_pixels(event)
        if not pixels:
            return
        if "shift" in modifiers:
            # Shift / sideways scroll: positive = scroll left, so content moves right.
            self.pan_by_pixels(event.x, event.y, pixels, 0.0)
        else:
            # Positive = scroll up, so content moves down.
            self.pan_by_pixels(event.x, event.y, 0.0, -pixels)

    @staticmethod
    def _wheel_pixels(event: Any) -> float:
        """Pixels to pan for one ``scroll_event``."""
        gui = getattr(event, "guiEvent", None)
        raw = getattr(gui, "delta", None)
        if sys.platform == "darwin" and raw:
            return float(raw) * MAC_TK86_PAN_PIXELS_PER_DELTA
        return float(event.step) * WHEEL_PAN_PIXELS_PER_STEP

    # -- navigation ----------------------------------------------------
    def pan_by_pixels(self, x: float, y: float, dx: float, dy: float) -> bool:
        """Pan the axes at canvas point (*x*, *y*) so their content moves by (*dx*, *dy*) pixels.

        Coordinates are matplotlib display pixels (y up).  Returns whether any
        axes moved.
        """
        if not (dx or dy):
            return False
        axes = self._axes_at(x, y, "pan")
        if not axes:
            return False
        self._begin_gesture()
        self._drag(axes, x, y, 1, lambda ax: (x + dx, y + dy))
        self._changed()
        return True

    def zoom_at(self, x: float, y: float, factor: float) -> bool:
        """Zoom the axes at canvas point (*x*, *y*) by *factor* around that point.

        *factor* > 1 zooms in (the visible range shrinks by that factor).
        Returns whether any axes changed.
        """
        if not factor or factor <= 0 or factor == 1.0 or not math.isfinite(factor):
            return False
        axes = self._axes_at(x, y, "zoom")
        if not axes:
            return False
        self._begin_gesture()
        shift = math.log10(factor)
        # drag_pan with the right button scales the view by 10**(-drag / size).
        self._drag(axes, x, y, 3,
                   lambda ax: (x + ax.bbox.width * shift, y + ax.bbox.height * shift))
        self._changed()
        return True

    def zoom_at_widget_point(self, wx: float, wy: float, factor: float) -> bool:
        """:meth:`zoom_at` for a point in Tk widget coordinates (y down)."""
        x, y = self._mpl_xy(SimpleNamespace(x=wx, y=wy))
        return self.zoom_at(x, y, factor)

    @staticmethod
    def _drag(axes: Sequence[Any], x: float, y: float, button: int,
              target: Callable[[Any], Tuple[float, float]]) -> None:
        # Start every axes before moving any, so twinned axes (which share
        # limits) all start from the same view instead of compounding.
        targets = [target(ax) for ax in axes]
        for ax in axes:
            ax.start_pan(x, y, button)
        try:
            for ax, (tx, ty) in zip(axes, targets):
                ax.drag_pan(button, None, tx, ty)
        finally:
            for ax in axes:
                ax.end_pan()

    def _axes_at(self, x: float, y: float, method: str) -> List[Any]:
        """Axes under (*x*, *y*) that allow *method* ("pan" / "zoom"), as the toolbar picks them."""
        figure = getattr(self.canvas, "figure", None)
        if figure is None:
            return []
        candidates = []
        for ax in figure.get_axes():
            try:
                if (ax.get_visible() and ax.get_navigate()
                        and getattr(ax, f"can_{method}")()
                        and ax.patch.contains_point((x, y))):
                    candidates.append(ax)
            except Exception:
                continue
        if not candidates:
            return []
        # Topmost zorder group first, stopping at an axes that captures events
        # (a visible background patch), plus twins; as NavigationToolbar2 does.
        groups: dict = {}
        for ax in reversed(candidates):
            groups.setdefault(ax.get_zorder(), []).append(ax)
        chosen: List[Any] = []
        for zorder in sorted(groups, reverse=True):
            captured = False
            for ax in groups[zorder]:
                chosen.append(ax)
                twins = getattr(ax, "_twinned_axes", None)
                if twins is not None:
                    chosen.extend(twins.get_siblings(ax))
                forward = "auto"
                try:
                    forward = ax.get_forward_navigation_events()
                except AttributeError:
                    pass
                if forward == "auto":
                    forward = not ax.patch.get_visible()
                if not forward:
                    captured = True
                    break
            if captured:
                break
        return [ax for ax in dict.fromkeys(chosen)
                if ax.get_navigate() and getattr(ax, f"can_{method}")()]

    def _mpl_xy(self, event: Any) -> Tuple[float, float]:
        """Tk widget coordinates of *event* -> matplotlib display coordinates."""
        to_mpl = getattr(self.canvas, "_event_mpl_coords", None)
        if to_mpl is not None:
            try:
                return to_mpl(event)
            except Exception:
                pass
        height = self.canvas.figure.bbox.height
        return float(event.x), float(height - event.y)

    # -- gesture bookkeeping ------------------------------------------
    def _begin_gesture(self) -> None:
        """Record the view before a gesture starts, and (re)arm the end-of-gesture timer."""
        if self._gesture_job is None:
            self._freeze_best_legends()
            toolbar = getattr(self.canvas, "toolbar", None)
            if toolbar is not None:
                try:
                    if toolbar._nav_stack() is None:
                        toolbar.push_current()  # the toolbar's Home view
                except Exception:
                    pass
        else:
            try:
                self.widget.after_cancel(self._gesture_job)
            except tk.TclError:
                pass
        try:
            self._gesture_job = self.widget.after(GESTURE_END_MS, self._end_gesture)
        except tk.TclError:
            self._gesture_job = None

    def _end_gesture(self) -> None:
        """A gesture finished: add its result to the toolbar's Back/Forward history."""
        self._gesture_job = None
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None:
            try:
                toolbar.push_current()
            except Exception:
                pass
        if self._restore_legends():
            self._changed()

    def _freeze_best_legends(self) -> None:
        """Pin ``loc="best"`` axes legends where they are for the length of a gesture.

        Finding the "best" spot scans every plotted point on each draw, which
        on a full spectrum costs more than the rest of the frame.  The legends
        are put back to "best" by :meth:`_restore_legends` when the gesture ends.
        """
        figure = getattr(self.canvas, "figure", None)
        if figure is None:
            return
        try:
            renderer = self.canvas.get_renderer()
        except Exception:
            return
        for ax in figure.get_axes():
            legend = ax.get_legend()
            if legend is None or getattr(legend, "_loc", None) != 0:  # 0 = "best"
                continue
            try:
                if tuple(legend.get_bbox_to_anchor().bounds) != tuple(ax.bbox.bounds):
                    continue  # anchored elsewhere: a tuple loc would mean something else
                box = legend.get_window_extent(renderer)
                x0, y0 = ax.transAxes.inverted().transform((box.x0, box.y0))
                legend.set_loc((float(x0), float(y0)))
            except Exception:
                continue
            self._frozen_legends.append(legend)

    def _restore_legends(self) -> bool:
        """Return legends pinned by :meth:`_freeze_best_legends` to ``loc="best"``."""
        restored = False
        while self._frozen_legends:
            legend = self._frozen_legends.pop()
            try:
                legend.set_loc("best")
                restored = True
            except Exception:
                pass
        return restored

    def _changed(self) -> None:
        self.canvas.draw_idle()

    def disconnect(self) -> None:
        """Stop handling events for this canvas."""
        try:
            self.canvas.mpl_disconnect(self._scroll_cid)
        except Exception:
            pass
        if getattr(self.widget, _NAV_ATTR, None) is self:
            try:
                delattr(self.widget, _NAV_ATTR)
            except AttributeError:
                pass
        if self._gesture_job is not None:
            try:
                self.widget.after_cancel(self._gesture_job)
            except tk.TclError:
                pass
            self._gesture_job = None


def attach_trackpad_navigation(canvas: Any, *, pan: bool = True, zoom: bool = True) -> Optional[TrackpadNavigator]:
    """Give a matplotlib ``FigureCanvasTkAgg`` trackpad pan / pinch / Ctrl-scroll zoom.

    Returns the :class:`TrackpadNavigator`, the existing one when the canvas
    already has one, or ``None`` when the canvas has no Tk widget.
    """
    try:
        widget = canvas.get_tk_widget()
    except Exception:
        return None
    existing = getattr(widget, _NAV_ATTR, None)
    if existing is not None:
        return existing
    return TrackpadNavigator(canvas, pan=pan, zoom=zoom)
