# -*- coding: utf-8 -*-
"""HoldableFigureCanvasTkAgg redraws only the axes that changed.

Every test compares the buffer after a partial redraw with a full render of
the same figure: they must be identical pixel for pixel.  Windows stay hidden.
"""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting.CachedLayout import CachedConstrainedLayoutEngine
from iSLAT.Modules.Plotting.HoldableCanvas import HoldableFigureCanvasTkAgg


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


def _three_panel(tk_root, legend=True, twin=False, log=False):
    """Like the main plot: one wide panel on top, two below, constrained layout with caching."""
    fig = Figure(figsize=(8, 6), dpi=100, layout=CachedConstrainedLayoutEngine())
    gs = fig.add_gridspec(2, 2)
    top = fig.add_subplot(gs[0, :])
    left = fig.add_subplot(gs[1, 0])
    right = fig.add_subplot(gs[1, 1])
    x = np.linspace(10, 20, 4000)
    top.plot(x, np.sin(x * 7) + 0.2 * np.cos(x * 50), label="spectrum")
    top.plot(x, np.cos(x * 3), label="model")
    top.set_title("Full Spectrum")
    top.set_ylabel("Flux density (Jy)")
    if legend:
        top.legend(loc="best")
    left.plot(x[:200], np.sin(x[:200] * 9))
    left.set_title("Line inspection")
    right.scatter(np.arange(50), np.exp(np.linspace(0, 5, 50)), s=8)
    right.set_title("Population diagram")
    if log:
        right.set_yscale("log")
    if twin:
        twin_ax = left.twinx()
        twin_ax.plot(x[:200], np.cos(x[:200] * 9), color="r")
    canvas = HoldableFigureCanvasTkAgg(fig, master=tk_root)
    canvas.get_tk_widget().pack()
    canvas.draw()
    return canvas, top, left, right


def _partial_draw(canvas, monkeypatch):
    """canvas.draw(), returning whether it was a partial redraw."""
    used = []
    original = HoldableFigureCanvasTkAgg._draw_changed_axes_only

    def spy(self):
        result = original(self)
        used.append(result)
        return result

    monkeypatch.setattr(HoldableFigureCanvasTkAgg, "_draw_changed_axes_only", spy)
    canvas.draw()
    monkeypatch.setattr(HoldableFigureCanvasTkAgg, "_draw_changed_axes_only", original)
    return used[-1]


def _assert_matches_full_render(canvas):
    partial = np.array(canvas.buffer_rgba())
    FigureCanvasAgg.draw(canvas)  # full render of the same state
    full = np.array(canvas.buffer_rgba())
    assert partial.shape == full.shape
    diff = np.argwhere(np.any(partial != full, axis=-1))
    assert diff.size == 0, f"{len(diff)} pixels differ, e.g. at {diff[:5].tolist()}"


@pytest.mark.parametrize("kwargs", [{}, {"legend": False}, {"twin": True}, {"log": True}])
def test_pan_top_panel_is_partial_and_exact(tk_root, monkeypatch, kwargs):
    canvas, top, left, right = _three_panel(tk_root, **kwargs)
    for shift in (0.3, 0.7, -1.1):
        lo, hi = top.get_xlim()
        top.set_xlim(lo + shift, hi + shift)
        assert _partial_draw(canvas, monkeypatch)
        _assert_matches_full_render(canvas)


def test_data_change_in_one_bottom_panel(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    line = left.get_lines()[0]
    line.set_ydata(line.get_ydata() * 0.5)
    assert _partial_draw(canvas, monkeypatch)
    _assert_matches_full_render(canvas)


def test_zoom_bottom_right_log_panel(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root, log=True)
    right.set_xlim(10, 30)
    right.set_ylim(2, 50)
    canvas.draw()  # may lay out again (tick labels change); must still be exact
    _assert_matches_full_render(canvas)


def test_all_axes_changed_is_a_full_draw(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    for ax in (top, left, right):
        ax.set_xlim(*ax.get_xlim())
    assert not _partial_draw(canvas, monkeypatch)


def test_layout_change_is_a_full_draw(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    left.set_ylabel("A much longer label\nover two lines")
    assert not _partial_draw(canvas, monkeypatch)
    _assert_matches_full_render(canvas)


def test_figure_text_is_a_full_draw(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    canvas.figure.suptitle("Title")
    canvas.draw()
    top.set_xlim(12, 18)
    assert not _partial_draw(canvas, monkeypatch)


def test_resize_is_a_full_draw(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    canvas.figure.set_size_inches(9, 6)
    top.set_xlim(12, 18)
    assert not _partial_draw(canvas, monkeypatch)
    _assert_matches_full_render(canvas)


def test_draw_event_sent_once(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    events = []
    canvas.mpl_connect("draw_event", events.append)
    lo, hi = top.get_xlim()
    top.set_xlim(lo + 0.5, hi + 0.5)
    assert _partial_draw(canvas, monkeypatch)
    assert len(events) == 1
    assert not canvas.figure.stale


def test_first_draw_is_full(tk_root, monkeypatch):
    fig = Figure(figsize=(4, 3), dpi=100)
    fig.add_subplot(121)
    fig.add_subplot(122)
    canvas = HoldableFigureCanvasTkAgg(fig, master=tk_root)
    assert not _partial_draw(canvas, monkeypatch)


def test_unchanged_figure_is_only_blitted(tk_root, monkeypatch):
    canvas, top, left, right = _three_panel(tk_root)
    renders, blits = [], []
    monkeypatch.setattr(canvas.figure, "draw", lambda renderer: renders.append(1))
    monkeypatch.setattr(canvas, "blit", lambda bbox=None: blits.append(bbox))
    canvas.draw()
    assert renders == [] and blits == [None]
