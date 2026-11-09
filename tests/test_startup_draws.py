# -*- coding: utf-8 -*-
"""Plot canvases are drawn once their widget is laid out, and blit opaque figures without alpha.

* HoldableFigureCanvasTkAgg remembers draw requests (and resize draws)
  while held and makes one draw when released.
* repaint_idle shows the last rendered frame again with a blit, once Tk is
  idle, and draws instead when there is no frame of the figure's size.
* hold_draws_until_laid_out releases the hold once the widget is mapped,
  the figure has its size and that size held for a check (or after a time
  limit).  GUI.start uses it for the main plot, OwnedCanvasMixin for every
  new view canvas (e.g. the first switch to the Full Spectrum view).
* GUI._poll_async_result asks for a draw instead of drawing on top of the
  draw that update_model_plot already asked for.
* Opaque figures are blitted to Tk without alpha.
"""
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure

from tests.tk_events import pump_tk


@pytest.fixture(scope="module")
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvas(tk_root):
    from iSLAT.Modules.Plotting.HoldableCanvas import HoldableFigureCanvasTkAgg
    fig = Figure(figsize=(4, 3), dpi=100)
    fig.add_subplot(111).plot([0, 1], [0, 1])
    c = HoldableFigureCanvasTkAgg(fig, master=tk_root)
    c.get_tk_widget().pack()
    draws = []
    original = fig.draw

    def counting(renderer):
        draws.append(1)
        return original(renderer)

    fig.draw = counting
    c.draws = draws
    yield c
    c.get_tk_widget().destroy()


def _process_idle(tk_root):
    tk_root.update_idletasks()


class TestHoldableCanvas:
    def test_draws_normally_when_not_held(self, canvas, tk_root):
        canvas.draw_idle()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1

    def test_held_requests_make_one_draw_on_release(self, canvas, tk_root):
        canvas.hold_draws()
        for _ in range(3):
            canvas.draw_idle()
        _process_idle(tk_root)
        assert canvas.draws == [] and canvas.holding_draws
        canvas.release_draws()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1 and not canvas.holding_draws

    def test_release_without_requests_does_not_draw(self, canvas, tk_root):
        canvas.hold_draws()
        canvas.release_draws()
        _process_idle(tk_root)
        assert canvas.draws == []

    def test_hold_takes_over_an_already_queued_draw(self, canvas, tk_root):
        canvas.draw_idle()          # queued, not yet run
        canvas.hold_draws()
        _process_idle(tk_root)
        assert canvas.draws == []
        canvas.release_draws()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1

    def test_resize_while_held_sizes_the_figure_without_drawing(self, canvas, tk_root):
        canvas.hold_draws()
        canvas.resize(SimpleNamespace(width=800, height=600))
        _process_idle(tk_root)
        assert tuple(canvas.figure.get_size_inches()) == pytest.approx((8.0, 6.0))
        assert canvas.draws == []
        canvas.release_draws()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1


# ----------------------------------------------------------------------
# Showing the last frame again (a re-packed view canvas)
# ----------------------------------------------------------------------
def _count_blits(canvas):
    blits = []
    original = canvas.blit

    def counting(bbox=None):
        blits.append(bbox)
        return original(bbox)

    canvas.blit = counting
    return blits


class TestRepaint:
    def test_blits_the_kept_frame_without_rendering(self, canvas, tk_root):
        canvas.draw()
        canvas.draws.clear()
        blits = _count_blits(canvas)
        canvas.repaint_idle()
        assert blits == []          # waits for Tk to be idle (after the re-pack)
        _process_idle(tk_root)
        assert canvas.draws == [] and blits == [None]

    def test_requests_before_idle_make_one_blit(self, canvas, tk_root):
        canvas.draw()
        blits = _count_blits(canvas)
        canvas.repaint_idle()
        canvas.repaint_idle()
        _process_idle(tk_root)
        assert len(blits) == 1

    def test_draws_when_nothing_was_rendered_yet(self, canvas, tk_root):
        canvas.repaint_idle()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1

    def test_draws_when_the_figure_was_resized_since_the_last_frame(self, canvas, tk_root):
        canvas.draw()
        canvas.draws.clear()
        canvas.figure.set_size_inches(5, 3)
        canvas.repaint_idle()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1

    def test_leaves_it_to_a_queued_draw(self, canvas, tk_root):
        canvas.draw()
        canvas.draws.clear()
        blits = _count_blits(canvas)
        canvas.draw_idle()
        canvas.repaint_idle()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1 and len(blits) == 1

    def test_skips_when_a_draw_is_asked_for_before_idle(self, canvas, tk_root):
        canvas.draw()
        canvas.draws.clear()
        blits = _count_blits(canvas)
        canvas.repaint_idle()
        canvas.draw_idle()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1 and len(blits) == 1

    def test_while_held_draws_on_release(self, canvas, tk_root):
        canvas.draw()
        canvas.draws.clear()
        canvas.hold_draws()
        canvas.repaint_idle()
        _process_idle(tk_root)
        assert canvas.draws == []
        canvas.release_draws()
        _process_idle(tk_root)
        assert len(canvas.draws) == 1


# ----------------------------------------------------------------------
# When the hold is released
# ----------------------------------------------------------------------
class _FakeWidget:
    def __init__(self, size, mapped=True):
        self.size, self.mapped = size, mapped
        self.toplevel = MagicMock()

    def winfo_exists(self):
        return True

    def winfo_width(self):
        return self.size[0]

    def winfo_height(self):
        return self.size[1]

    def winfo_ismapped(self):
        return self.mapped

    def winfo_toplevel(self):
        return self.toplevel


def _held_canvas(widget_size, mapped=True, figure_px=None):
    """A HoldableFigureCanvasTkAgg stand-in: the real methods on a fake widget."""
    from iSLAT.Modules.Plotting.HoldableCanvas import HoldableFigureCanvasTkAgg
    fw, fh = figure_px if figure_px is not None else widget_size
    widget = _FakeWidget(widget_size, mapped)
    c = HoldableFigureCanvasTkAgg.__new__(HoldableFigureCanvasTkAgg)
    c._holding_draws = True
    c._draw_requested = False
    c.figure = Figure(figsize=(fw / 100.0, fh / 100.0), dpi=100)
    c.get_tk_widget = lambda: widget
    c.release_draws = MagicMock()
    return c, widget


class TestReleaseWhenLaidOut:
    def test_releases_once_size_is_stable_and_applied(self):
        c, widget = _held_canvas((1200, 600))
        c._release_when_laid_out(time.monotonic(), (1200, 600))
        c.release_draws.assert_called_once()
        widget.toplevel.after.assert_not_called()

    @pytest.mark.parametrize("case", [
        dict(last_size=None),                                  # first look at the size
        dict(last_size=(1400, 600)),                           # still resizing
        dict(last_size=(1200, 600), mapped=False),             # not on screen yet
        dict(last_size=(1200, 600), figure_px=(640, 480)),     # figure not resized yet
    ], ids=["first-check", "still-resizing", "not-mapped", "figure-not-resized"])
    def test_waits_until_laid_out(self, case):
        c, widget = _held_canvas((1200, 600), mapped=case.get("mapped", True), figure_px=case.get("figure_px"))
        c._release_when_laid_out(time.monotonic(), case["last_size"])
        c.release_draws.assert_not_called()
        delay, callback, started, size = widget.toplevel.after.call_args[0]
        assert callback == c._release_when_laid_out
        assert size == (1200, 600)

    def test_releases_after_the_time_limit_anyway(self):
        c, _widget = _held_canvas((1200, 600), mapped=False)
        c._release_when_laid_out(time.monotonic() - c.LAYOUT_MAX_WAIT_S - 0.1, None)
        c.release_draws.assert_called_once()

    def test_stops_checking_once_released_elsewhere(self):
        c, widget = _held_canvas((1200, 600), mapped=False)
        c._holding_draws = False
        c._release_when_laid_out(time.monotonic(), None)
        widget.toplevel.after.assert_not_called()

    def test_hold_draws_until_laid_out_releases_on_a_real_widget(self, tk_root):
        # The root is withdrawn, so the widget is never mapped: the time limit releases it.
        from iSLAT.Modules.Plotting.HoldableCanvas import HoldableFigureCanvasTkAgg
        c = HoldableFigureCanvasTkAgg(Figure(figsize=(2, 2)), master=tk_root)
        c.get_tk_widget().pack()
        c.LAYOUT_MAX_WAIT_S = 0.2
        c.hold_draws_until_laid_out()
        assert c.holding_draws
        assert pump_tk(tk_root, 3.0, until=lambda: not c.holding_draws)
        c.get_tk_widget().destroy()


# ----------------------------------------------------------------------
# Who holds
# ----------------------------------------------------------------------
def test_gui_start_holds_the_plot_until_laid_out():
    from iSLAT.Modules.GUI.GUI import GUI
    gui = GUI.__new__(GUI)
    gui.config = {"start_on_top": False}
    gui.master = MagicMock()
    gui.window = gui.master
    gui.plot = SimpleNamespace(canvas=MagicMock())
    gui.create_window = MagicMock()
    gui._bind_appearance_change = MagicMock()
    gui.start(display_spectrum_async=False)
    gui.plot.canvas.hold_draws_until_laid_out.assert_called_once()
    gui.window.mainloop.assert_called_once()


def test_new_view_canvases_are_held_until_laid_out(tk_root):
    from iSLAT.Modules.Plotting.OwnedCanvasMixin import OwnedCanvasMixin

    class _View(OwnedCanvasMixin):
        _canvas = None
        _pm = None

        def __init__(self, fig, parent):
            self._fig, self._parent_frame = fig, parent

        def get_figure(self):
            return self._fig

        def _notify_canvas_changed(self):
            pass

    view = _View(Figure(figsize=(3, 2)), tk_root)
    view._show_canvas()
    try:
        assert view._canvas.holding_draws
    finally:
        view._canvas.get_tk_widget().destroy()


def test_poll_asks_for_a_draw_instead_of_drawing():
    from iSLAT.Modules.GUI.GUI import GUI
    gui = GUI.__new__(GUI)
    gui.master = MagicMock()
    gui.plot = MagicMock()
    gui._async_callback = None
    gui._async_result_queue = queue.Queue()
    gui._async_result_queue.put(("success", None))
    gui._poll_async_result()
    gui.plot.update_model_plot.assert_called_once()
    gui.plot.canvas.draw_idle.assert_called()
    gui.plot.canvas.draw.assert_not_called()


# ----------------------------------------------------------------------
# Blitting
# ----------------------------------------------------------------------
class TestBlitOffsets:
    @staticmethod
    def _offsets_used(canvas, monkeypatch):
        import iSLAT.Modules.Plotting.HoldableCanvas as hc
        used = []
        original = hc._backend_tk.blit
        monkeypatch.setattr(hc._backend_tk, "blit", lambda photo, data, offsets, bbox=None: (
            used.append(offsets), original(photo, data, offsets, bbox=bbox)))
        canvas.draw()
        return used

    def test_opaque_figure_is_blitted_without_alpha(self, canvas, monkeypatch):
        from iSLAT.Modules.Plotting.HoldableCanvas import RGB_OFFSETS
        assert self._offsets_used(canvas, monkeypatch) == [RGB_OFFSETS]

    @pytest.mark.parametrize("make_transparent", [
        lambda fig: fig.patch.set_alpha(0.5),
        lambda fig: fig.patch.set_facecolor((1, 1, 1, 0)),
        lambda fig: fig.patch.set_visible(False),
    ], ids=["patch-alpha", "facecolor-alpha", "no-background"])
    def test_transparent_figure_keeps_alpha(self, canvas, monkeypatch, make_transparent):
        from iSLAT.Modules.Plotting.HoldableCanvas import RGBA_OFFSETS
        make_transparent(canvas.figure)
        assert self._offsets_used(canvas, monkeypatch) == [RGBA_OFFSETS]

    def test_opaque_blit_shows_the_same_pixels(self, canvas, monkeypatch, tk_root):
        import iSLAT.Modules.Plotting.HoldableCanvas as hc
        canvas.draw()
        photo = canvas._tkphoto
        w, h = canvas.get_width_height()
        points = [(x, y) for x in range(0, w, 37) for y in range(0, h, 29)]
        hc._backend_tk.blit(photo, canvas.renderer.buffer_rgba(), hc.RGBA_OFFSETS)
        with_alpha = [photo.get(x, y) for x, y in points]
        hc._backend_tk.blit(photo, canvas.renderer.buffer_rgba(), hc.RGB_OFFSETS)
        without_alpha = [photo.get(x, y) for x, y in points]
        assert with_alpha == without_alpha
