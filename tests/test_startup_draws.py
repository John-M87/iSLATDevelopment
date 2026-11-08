# -*- coding: utf-8 -*-
"""Startup draws the main plot only once the window is laid out.

* HoldableFigureCanvasTkAgg remembers draw requests (and resize draws)
  while held and makes one draw when released.
* GUI._release_plot_draws_when_laid_out releases the hold once the canvas
  is mapped, the figure has its size and that size held for a check (or
  after a time limit).
* GUI._poll_async_result asks for a draw instead of drawing on top of the
  draw that update_model_plot already asked for.
"""
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure


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
    from iSLAT.Modules.Plotting.MainPlot import HoldableFigureCanvasTkAgg
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
# When the hold is released
# ----------------------------------------------------------------------
def _gui_with_canvas(widget_size, mapped=True, figure_px=None):
    from iSLAT.Modules.GUI.GUI import GUI
    gui = GUI.__new__(GUI)
    gui.master = MagicMock()
    w, h = widget_size
    fw, fh = figure_px if figure_px is not None else widget_size
    figure = Figure(figsize=(fw / 100.0, fh / 100.0), dpi=100)
    widget = SimpleNamespace(winfo_width=lambda: w, winfo_height=lambda: h, winfo_ismapped=lambda: mapped)
    gui.plot = SimpleNamespace(canvas=SimpleNamespace(
        get_tk_widget=lambda: widget, figure=figure, release_draws=MagicMock(),
    ))
    return gui


class TestReleaseWhenLaidOut:
    def test_releases_once_size_is_stable_and_applied(self):
        gui = _gui_with_canvas((1200, 600))
        gui._release_plot_draws_when_laid_out(time.monotonic(), (1200, 600))
        gui.plot.canvas.release_draws.assert_called_once()
        gui.master.after.assert_not_called()

    @pytest.mark.parametrize("case", [
        dict(last_size=None),                                  # first look at the size
        dict(last_size=(1400, 600)),                           # still resizing
        dict(last_size=(1200, 600), mapped=False),             # not on screen yet
        dict(last_size=(1200, 600), figure_px=(640, 480)),     # figure not resized yet
    ], ids=["first-check", "still-resizing", "not-mapped", "figure-not-resized"])
    def test_waits_until_laid_out(self, case):
        gui = _gui_with_canvas((1200, 600), mapped=case.get("mapped", True), figure_px=case.get("figure_px"))
        gui._release_plot_draws_when_laid_out(time.monotonic(), case["last_size"])
        gui.plot.canvas.release_draws.assert_not_called()
        delay, callback, started, size = gui.master.after.call_args[0]
        assert callback == gui._release_plot_draws_when_laid_out
        assert size == (1200, 600)

    def test_releases_after_the_time_limit_anyway(self):
        from iSLAT.Modules.GUI.GUI import GUI
        gui = _gui_with_canvas((1200, 600), mapped=False)
        gui._release_plot_draws_when_laid_out(time.monotonic() - GUI._LAYOUT_MAX_WAIT_S - 0.1, None)
        gui.plot.canvas.release_draws.assert_called_once()


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
