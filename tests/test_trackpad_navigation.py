# -*- coding: utf-8 -*-
"""Trackpad navigation: two-finger pan, pinch / Ctrl+scroll zoom, touchpad scrolling of frames.

Real gestures cannot be produced in a test (and synthetic ``<TouchpadScroll>``
events only reach shown windows), so the handlers are called directly with
event objects shaped like Tk's and matplotlib's.  Windows stay hidden.
"""
import math
import sys
from types import SimpleNamespace

import pytest

from iSLAT.Modules.GUI import TrackpadNavigation as T

CTRL = 1 << 2


def _packed(dx, dy):
    """A ``<TouchpadScroll>`` ``%D`` value for finger movement (dx, dy)."""
    return ((dx & 0xFFFF) << 16) | (dy & 0xFFFF)


@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.withdraw()
    root.destroy()


@pytest.fixture
def figure_canvas(tk_root):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    fig = Figure(figsize=(6, 3), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=tk_root)
    canvas.get_tk_widget().pack()
    yield canvas


def _center(ax):
    return ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2


def _tk_event(canvas, x, y, delta, state=0):
    """A Tk-style event at matplotlib display point (x, y)."""
    return SimpleNamespace(delta=delta, state=state, x=x, y=canvas.figure.bbox.height - y)


def _linear_axes(canvas):
    ax = canvas.figure.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    canvas.draw()
    return ax


# ----------------------------------------------------------------------
# Delta decoding and frame scrolling
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dx, dy", [(0, 0), (5, -7), (-3, 12), (-200, -150), (1, 0), (0, -1)])
def test_decode_touchpad_delta_round_trips(dx, dy):
    assert T.decode_touchpad_delta(_packed(dx, dy)) == (dx, dy)


def test_decode_matches_tk(tk_root):
    """Same split as Tk's own tk::PreciseScrollDeltas."""
    if not T.touchpad_scroll_supported(tk_root):
        pytest.skip("Tk 8.6 has no tk::PreciseScrollDeltas")
    for dx, dy in [(5, -7), (-3, 12)]:
        packed = _packed(dx, dy)
        # Tk sees %D as a signed 32-bit int.
        signed = packed - (1 << 32) if packed >= (1 << 31) else packed
        tk_dx, tk_dy = tk_root.tk.splitlist(tk_root.tk.call("tk::PreciseScrollDeltas", signed))
        assert (int(tk_dx), int(tk_dy)) == T.decode_touchpad_delta(signed) == (dx, dy)


def test_scroll_canvas_by_pixels(tk_root):
    import tkinter as tk
    canvas = tk.Canvas(tk_root, width=100, height=100, scrollregion=(0, 0, 500, 1000))
    canvas.pack()
    tk_root.update_idletasks()
    x0, y0 = canvas.xview()[0], canvas.yview()[0]
    T.scroll_canvas_by_pixels(canvas, dx=50, dy=100)
    assert canvas.xview()[0] - x0 == pytest.approx(0.1, abs=0.005)
    assert canvas.yview()[0] - y0 == pytest.approx(0.1, abs=0.005)
    for _ in range(10):  # relative steps do not drift
        T.scroll_canvas_by_pixels(canvas, dy=10)
    assert canvas.yview()[0] - y0 == pytest.approx(0.2, abs=0.005)


def test_bind_touchpad_scroll_scrolls_frame(tk_root):
    import tkinter as tk
    canvas = tk.Canvas(tk_root, width=100, height=100, scrollregion=(0, 0, 100, 1000))
    canvas.pack()
    tk_root.update_idletasks()
    handler = T.bind_touchpad_scroll(canvas, canvas)
    if not T.touchpad_scroll_supported(tk_root):
        assert handler is None
        return
    y0 = canvas.yview()[0]
    handler(SimpleNamespace(delta=_packed(0, -100)))  # fingers up: content up, view down
    assert canvas.yview()[0] - y0 == pytest.approx(0.1, abs=0.005)


def test_horizontal_only_frame_scrolls_sideways_for_vertical_fingers(tk_root):
    import tkinter as tk
    canvas = tk.Canvas(tk_root, width=100, height=100, scrollregion=(0, 0, 1000, 100))
    canvas.pack()
    tk_root.update_idletasks()
    handler = T.bind_touchpad_scroll(canvas, canvas, vertical=False, horizontal=True)
    if handler is None:
        pytest.skip("no <TouchpadScroll> in Tk 8.6")
    x0 = canvas.xview()[0]
    handler(SimpleNamespace(delta=_packed(0, -100)))
    assert canvas.xview()[0] - x0 == pytest.approx(0.1, abs=0.005)


def test_scrollable_frame_gets_touchpad_binding(tk_root):
    from iSLAT.Modules.GUI.GUIFunctions import create_scrollable_frame
    frame = create_scrollable_frame(tk_root, vertical=True)
    if not T.touchpad_scroll_supported(tk_root):
        pytest.skip("no <TouchpadScroll> in Tk 8.6")
    assert frame.bind("<TouchpadScroll>")


# ----------------------------------------------------------------------
# Pan and zoom
# ----------------------------------------------------------------------
def test_pan_moves_content_with_fingers(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    assert nav.pan_by_pixels(x, y, ax.bbox.width / 10, 0)
    # Content moved right by a tenth of the width: the view moved left.
    assert ax.get_xlim() == pytest.approx((-1.0, 9.0))
    assert ax.get_ylim() == pytest.approx((0.0, 10.0))


def test_pan_log_axis_moves_in_decades(figure_canvas):
    ax = figure_canvas.figure.add_subplot(111)
    ax.set_yscale("log")
    ax.set_ylim(1, 1000)
    figure_canvas.draw()
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    nav.pan_by_pixels(x, y, 0, ax.bbox.height / 3)  # one decade up
    assert ax.get_ylim() == pytest.approx((0.1, 100.0))


def test_twin_axes_pan_once(figure_canvas):
    ax = _linear_axes(figure_canvas)
    twin = ax.twinx()
    twin.set_ylim(0, 1)
    figure_canvas.draw()
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    nav.pan_by_pixels(x, y, ax.bbox.width / 10, ax.bbox.height / 10)
    assert ax.get_xlim() == pytest.approx((-1.0, 9.0))      # shared x moved once, not twice
    assert ax.get_ylim() == pytest.approx((-1.0, 9.0))
    assert twin.get_ylim() == pytest.approx((-0.1, 0.9))     # the twin's own y moved too


def test_zoom_at_point(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    assert nav.zoom_at(x, y, 2.0)
    assert ax.get_xlim() == pytest.approx((2.5, 7.5))
    assert ax.get_ylim() == pytest.approx((2.5, 7.5))
    nav.zoom_at(x, y, 0.5)
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))


def test_zoom_keeps_point_under_pointer(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    px, py = ax.transData.transform((2.0, 8.0))
    nav.zoom_at(px, py, 4.0)
    assert ax.transData.inverted().transform((px, py)) == pytest.approx((2.0, 8.0))
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(2.5)


def test_outside_axes_does_nothing(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    assert not nav.pan_by_pixels(1, 1, 50, 0)
    assert not nav.zoom_at(1, 1, 2.0)
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))


def test_non_navigable_axes_ignored(figure_canvas):
    ax = _linear_axes(figure_canvas)
    ax.set_navigate(False)
    nav = T.attach_trackpad_navigation(figure_canvas)
    assert not nav.pan_by_pixels(*_center(ax), 50, 0)


# ----------------------------------------------------------------------
# Event handlers
# ----------------------------------------------------------------------
def test_touchpad_scroll_pans_and_stops_propagation(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    step = int(ax.bbox.width / 10)
    result = nav._on_touchpad_scroll(_tk_event(figure_canvas, x, y, _packed(step, 0)))
    assert result == "break"
    assert ax.get_xlim()[0] < 0  # content moved right with the fingers
    # Fingers down (Tk +dy) moves content down: the view goes up.
    nav._on_touchpad_scroll(_tk_event(figure_canvas, x, y, _packed(0, int(ax.bbox.height / 10))))
    assert ax.get_ylim()[0] > 0


def test_ctrl_touchpad_scroll_zooms(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    assert nav._on_touchpad_scroll(_tk_event(figure_canvas, x, y, _packed(0, 30), CTRL)) == "break"
    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    assert width == pytest.approx(10 / math.exp(30 * T.TOUCHPAD_ZOOM_PER_PIXEL))


def test_zoom_only_navigator_lets_scrolling_through(figure_canvas):
    """Canvases in scrolling grids: plain two-finger scroll must reach the page."""
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas, pan=False)
    x, y = _center(ax)
    assert nav._on_touchpad_scroll(_tk_event(figure_canvas, x, y, _packed(20, 20))) is None
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))
    assert nav._on_touchpad_scroll(_tk_event(figure_canvas, x, y, _packed(0, 20), CTRL)) == "break"
    assert ax.get_xlim()[1] - ax.get_xlim()[0] < 10


def _scroll_event(canvas, x, y, step, modifiers=(), raw_delta=None):
    return SimpleNamespace(x=x, y=y, step=step, modifiers=list(modifiers),
                           guiEvent=SimpleNamespace(delta=raw_delta) if raw_delta is not None else None)


def test_ctrl_wheel_zooms(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    nav._on_mpl_scroll(_scroll_event(figure_canvas, *_center(ax), 1, ["ctrl"]))
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(10 / T.WHEEL_ZOOM_PER_STEP)


def test_wheel_pans_only_where_trackpad_uses_the_wheel_event(figure_canvas):
    ax = _linear_axes(figure_canvas)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    nav._touchpad_separate = True   # Tk 9 on macOS: a mouse wheel keeps its old (no-op) behaviour
    nav._on_mpl_scroll(_scroll_event(figure_canvas, x, y, 1))
    assert ax.get_ylim() == pytest.approx((0.0, 10.0))
    nav._touchpad_separate = False  # Tk 8.6 / other platforms: scrolling pans
    nav._on_mpl_scroll(_scroll_event(figure_canvas, x, y, 1, raw_delta=120 if sys.platform != "darwin" else 3))
    assert ax.get_ylim()[0] > 0     # scroll up: content down, view up
    nav._on_mpl_scroll(_scroll_event(figure_canvas, x, y, 1, ["shift"], raw_delta=120 if sys.platform != "darwin" else 3))
    assert ax.get_xlim()[0] < 0     # Shift / sideways: content right, view left


# ----------------------------------------------------------------------
# Gestures: legends, toolbar history
# ----------------------------------------------------------------------
def test_best_legend_pinned_during_gesture_and_restored(figure_canvas, tk_root):
    from tests.tk_events import pump_tk
    ax = _linear_axes(figure_canvas)
    ax.plot([0, 10], [0, 10], label="line")
    ax.legend(loc="best")
    figure_canvas.draw()
    nav = T.attach_trackpad_navigation(figure_canvas)
    nav.pan_by_pixels(*_center(ax), 5, 0)
    assert isinstance(ax.get_legend()._loc, tuple)
    assert pump_tk(tk_root, 2.0, until=lambda: nav._gesture_job is None)
    assert ax.get_legend()._loc == 0  # "best" again


def test_explicit_legend_location_untouched(figure_canvas):
    ax = _linear_axes(figure_canvas)
    ax.plot([0, 10], [0, 10], label="line")
    ax.legend(loc="upper left")
    figure_canvas.draw()
    loc = ax.get_legend()._loc
    nav = T.attach_trackpad_navigation(figure_canvas)
    nav.pan_by_pixels(*_center(ax), 5, 0)
    assert ax.get_legend()._loc == loc


def test_gesture_is_one_toolbar_history_step(figure_canvas, tk_root):
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    from tests.tk_events import pump_tk
    ax = _linear_axes(figure_canvas)
    toolbar = NavigationToolbar2Tk(figure_canvas, tk_root, pack_toolbar=False)
    nav = T.attach_trackpad_navigation(figure_canvas)
    x, y = _center(ax)
    for _ in range(5):
        nav.pan_by_pixels(x, y, 10, 0)
    assert pump_tk(tk_root, 2.0, until=lambda: nav._gesture_job is None)
    assert len(toolbar._nav_stack) == 2  # home + the end of the gesture
    toolbar.back()
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))


# ----------------------------------------------------------------------
# Pinch
# ----------------------------------------------------------------------
def test_pinch_factor_combines_queued_events():
    T._pinch_events[:] = [0.1, 0.1, -0.05]
    assert T._take_pinch_factor() == pytest.approx(1.1 * 1.1 * 0.95)
    assert T._pinch_events == []
    assert T._take_pinch_factor() == 1.0


def test_pinch_zooms_canvas_under_pointer(figure_canvas, tk_root, monkeypatch):
    ax = _linear_axes(figure_canvas)
    T.attach_trackpad_navigation(figure_canvas)
    widget = figure_canvas.get_tk_widget()
    x, y = _center(ax)
    wx, wy = int(x), int(figure_canvas.figure.bbox.height - y)
    monkeypatch.setattr(type(tk_root), "winfo_pointerxy", lambda self: (1000 + wx, 2000 + wy))
    monkeypatch.setattr(type(tk_root), "winfo_containing", lambda self, px, py: widget)
    monkeypatch.setattr(type(widget), "winfo_rootx", lambda self: 1000)
    monkeypatch.setattr(type(widget), "winfo_rooty", lambda self: 2000)
    assert T._dispatch_pinch(tk_root, 2.0)
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(5.0, rel=0.02)


def test_pinch_ignored_away_from_plots(tk_root, monkeypatch):
    monkeypatch.setattr(type(tk_root), "winfo_pointerxy", lambda self: (5, 5))
    monkeypatch.setattr(type(tk_root), "winfo_containing", lambda self, px, py: tk_root)
    assert not T._dispatch_pinch(tk_root, 2.0)


def test_pinch_monitor_available_on_macos_with_pyobjc():
    if sys.platform != "darwin":
        assert not T.install_pinch_monitor()
        return
    pytest.importorskip("AppKit")
    assert T.install_pinch_monitor()
    assert T.install_pinch_monitor()  # idempotent: one monitor per process


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def test_attach_is_idempotent(figure_canvas):
    first = T.attach_trackpad_navigation(figure_canvas)
    assert T.attach_trackpad_navigation(figure_canvas) is first


def test_owned_canvas_views_get_navigation(tk_root):
    """Views built on OwnedCanvasMixin (full spectrum, spectrum panels) get trackpad navigation."""
    from matplotlib.figure import Figure
    from iSLAT.Modules.Plotting.OwnedCanvasMixin import OwnedCanvasMixin

    class _View(OwnedCanvasMixin):
        def __init__(self, parent):
            self._fig = Figure()
            self._fig.add_subplot(111)
            self._parent_frame = parent
            self._canvas = None
            self._right_click_cid = None

        def get_figure(self):
            return self._fig

        def _notify_canvas_changed(self):
            pass

    view = _View(tk_root)
    view._ensure_canvas()
    assert getattr(view._canvas.get_tk_widget(), T._NAV_ATTR, None) is not None
