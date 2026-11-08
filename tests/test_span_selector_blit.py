# -*- coding: utf-8 -*-
"""SingleDrawSpanSelector: one figure render per draw while a selection is visible."""

import matplotlib
matplotlib.use("Agg")

import io

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler, SingleDrawSpanSelector


def _make(selector_cls, visible=True):
    """Return (fig, ax, span) with the same arguments InteractionHandler uses."""
    fig = Figure(figsize=(6, 3), dpi=80)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    x = np.linspace(10.0, 20.0, 500)
    ax.plot(x, np.sin(x))
    span = selector_cls(
        ax, lambda lo, hi: None,
        direction="horizontal",
        useblit=True,
        props=dict(alpha=0.3, facecolor="lime"),
        interactive=True,
        drag_from_anywhere=True,
    )
    fig.canvas.draw()
    if visible:
        span.extents = (12.0, 13.0)
        span.set_visible(True)
    return fig, ax, span


def _count_renders(fig):
    """Count full figure renders from now on."""
    calls = []
    original = fig.draw

    def counting(renderer):
        calls.append(1)
        return original(renderer)

    fig.draw = counting
    return calls


class TestRenderCount:
    def test_visible_selection_renders_figure_once(self):
        fig, _ax, _span = _make(SingleDrawSpanSelector)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 1

    def test_stock_selector_renders_twice(self):
        # The cost this class removes; if matplotlib stops doing this the
        # subclass is no longer needed.
        fig, _ax, _span = _make(SpanSelector)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 2

    def test_hidden_selection_renders_once(self):
        fig, _ax, _span = _make(SingleDrawSpanSelector, visible=False)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 1


class TestSameOutputAsStock:
    def _draw_both(self, visible=True):
        fig_a, _, span_a = _make(SpanSelector, visible=visible)
        fig_b, _, span_b = _make(SingleDrawSpanSelector, visible=visible)
        fig_a.canvas.draw()
        fig_b.canvas.draw()
        return fig_a, span_a, fig_b, span_b

    def test_background_matches_stock(self):
        _, span_a, _, span_b = self._draw_both()
        np.testing.assert_array_equal(np.asarray(span_a.background), np.asarray(span_b.background))

    def test_background_excludes_selection(self):
        fig_hidden, _, span_hidden = _make(SingleDrawSpanSelector, visible=False)
        fig_hidden.canvas.draw()
        fig_b, _, span_b = _make(SingleDrawSpanSelector)
        fig_b.canvas.draw()
        np.testing.assert_array_equal(np.asarray(span_hidden.background), np.asarray(span_b.background))

    def test_frame_matches_stock(self):
        fig_a, _, fig_b, _ = self._draw_both()
        np.testing.assert_array_equal(
            np.asarray(fig_a.canvas.buffer_rgba()), np.asarray(fig_b.canvas.buffer_rgba()),
        )

    def test_selection_is_drawn_on_frame(self):
        fig_hidden, _, _ = _make(SingleDrawSpanSelector, visible=False)
        fig_hidden.canvas.draw()
        fig_b, _, _ = _make(SingleDrawSpanSelector)
        fig_b.canvas.draw()
        assert not np.array_equal(
            np.asarray(fig_hidden.canvas.buffer_rgba()), np.asarray(fig_b.canvas.buffer_rgba()),
        )

    def test_saved_figure_matches_stock(self):
        fig_a, _, fig_b, _ = self._draw_both()
        out_a, out_b = io.BytesIO(), io.BytesIO()
        fig_a.savefig(out_a, format="png")
        fig_b.savefig(out_b, format="png")
        assert out_a.getvalue() == out_b.getvalue()


class TestFallsBackToStock:
    def test_without_draw_event(self):
        fig, _ax, span = _make(SingleDrawSpanSelector)
        renders = _count_renders(fig)
        span.update_background(None)
        assert len(renders) == 1  # the stock path re-renders the canvas

    def test_when_figure_has_other_animated_artists(self):
        fig, ax, _span = _make(SingleDrawSpanSelector)
        ax.plot([10.0, 20.0], [0.0, 0.5], animated=True)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 2

    def test_when_selector_artist_is_not_animated(self):
        fig, _ax, span = _make(SingleDrawSpanSelector)
        span.artists[0].set_animated(False)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 2

    def test_without_blitting(self):
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        span = SingleDrawSpanSelector(ax, lambda lo, hi: None, "horizontal", useblit=False, interactive=True)
        span.extents = (0.2, 0.4)
        span.set_visible(True)
        renders = _count_renders(fig)
        fig.canvas.draw()
        assert len(renders) == 1
        assert span.background is None


def test_interaction_handler_uses_single_draw_selector():
    h = InteractionHandler.__new__(InteractionHandler)
    fig = Figure()
    FigureCanvasAgg(fig)
    h.ax1 = fig.add_subplot(111)
    h._setup_span_selector()
    assert isinstance(h.span_selector, SingleDrawSpanSelector)
    assert h.span_selector.useblit
