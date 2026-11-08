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


def _background(span):
    """The selector's saved blit background (matplotlib >= 3.11 keeps it on the canvas)."""
    load = getattr(span, "_load_blit_background", None)
    return load() if load is not None else span.background


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
        np.testing.assert_array_equal(np.asarray(_background(span_a)), np.asarray(_background(span_b)))

    def test_background_excludes_selection(self):
        fig_hidden, _, span_hidden = _make(SingleDrawSpanSelector, visible=False)
        fig_hidden.canvas.draw()
        fig_b, _, span_b = _make(SingleDrawSpanSelector)
        fig_b.canvas.draw()
        np.testing.assert_array_equal(np.asarray(_background(span_hidden)), np.asarray(_background(span_b)))

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
        assert _background(span) is None


def test_interaction_handler_uses_single_draw_selector():
    h = InteractionHandler.__new__(InteractionHandler)
    fig = Figure()
    FigureCanvasAgg(fig)
    h.ax1 = fig.add_subplot(111)
    h._setup_span_selector()
    assert isinstance(h.span_selector, SingleDrawSpanSelector)
    assert h.span_selector.useblit


def _frame(fig):
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _true_frame(fig):
    """What a full render of the figure as it is now shows."""
    fig.canvas.draw()
    return _frame(fig)


class TestNoBlitOfAnotherView:
    """A background is only blitted while the axes show what it was captured from.

    Another view hands a selection to the main plot by moving its range and
    then setting the span: restoring the background from before the move
    would paint the old wavelength range under the new selection.
    """

    def test_extents_after_xlim_change_shows_new_range(self):
        fig, ax, span = _make(SingleDrawSpanSelector)
        ax.set_xlim(14.0, 16.0)
        span.extents = (14.5, 15.0)
        shown = _frame(fig)
        np.testing.assert_array_equal(shown, _true_frame(fig))

    def test_extents_after_ylim_change_shows_new_range(self):
        fig, ax, span = _make(SingleDrawSpanSelector)
        ax.set_ylim(-0.2, 0.2)
        span.extents = (12.5, 13.0)
        shown = _frame(fig)
        np.testing.assert_array_equal(shown, _true_frame(fig))

    def test_extents_after_resize_shows_resized_axes(self):
        fig, _ax, span = _make(SingleDrawSpanSelector)
        fig.set_size_inches(5, 2)
        span.extents = (12.5, 13.0)
        shown = _frame(fig)
        np.testing.assert_array_equal(shown, _true_frame(fig))

    def test_redraw_refreshes_blitted_background(self):
        # A selection was blitted at one range; after the range changes and
        # the plot is redrawn, the next press / drag must blit the new range.
        # (matplotlib 3.11 blits only the background stored on the canvas.)
        fig, ax, span = _make(SingleDrawSpanSelector)
        span.extents = (12.2, 12.8)
        ax.set_xlim(14.0, 16.0)
        fig.canvas.draw()
        renders = _count_renders(fig)
        span.extents = (14.5, 15.0)
        assert renders == []
        np.testing.assert_array_equal(_frame(fig), _true_frame(fig))

    def test_redraw_captures_background_of_new_range(self):
        fig, ax, span = _make(SingleDrawSpanSelector)
        ax.set_xlim(14.0, 16.0)
        span.extents = (14.5, 15.0)
        # Later blits (a drag) restore a background of the new range.
        renders = _count_renders(fig)
        span.extents = (14.6, 15.2)
        assert renders == []
        np.testing.assert_array_equal(_frame(fig), _true_frame(fig))

    def test_unchanged_view_still_blits(self):
        # The guard must not turn every drag step into a full render.
        fig, _ax, span = _make(SingleDrawSpanSelector)
        renders = _count_renders(fig)
        span.extents = (12.2, 13.4)
        span.extents = (12.4, 13.8)
        assert renders == []
        np.testing.assert_array_equal(_frame(fig), _true_frame(fig))


class TestShowSpan:
    def _fresh(self):
        """A selector that has never had a selection, like the main plot before any drag."""
        calls = []
        fig = Figure(figsize=(6, 3), dpi=80)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        x = np.linspace(10.0, 20.0, 500)
        ax.plot(x, np.sin(x))
        span = SingleDrawSpanSelector(
            ax, lambda lo, hi: calls.append((lo, hi)), direction="horizontal",
            useblit=True, interactive=True, drag_from_anywhere=True,
        )
        fig.canvas.draw()
        return fig, ax, span, calls

    def test_shows_span_as_finished_selection(self):
        fig, _ax, span, calls = self._fresh()
        span.show_span(12.0, 13.0)
        assert span.extents == (12.0, 13.0)
        assert span.get_visible() and span.artists[0].get_visible()
        # A dragged selection is "completed": its handles respond, a click clears it.
        assert span._selection_completed
        assert calls == []
        np.testing.assert_array_equal(_frame(fig), _true_frame(fig))

    def test_orders_bounds(self):
        _fig, _ax, span, _calls = self._fresh()
        span.show_span(13.0, 12.0)
        assert span.extents == (12.0, 13.0)

    def test_after_range_change_shows_new_range(self):
        fig, ax, span, _calls = self._fresh()
        ax.set_xlim(14.0, 16.0)
        span.show_span(14.5, 15.0)
        np.testing.assert_array_equal(_frame(fig), _true_frame(fig))

    def test_interaction_handler_show_span(self):
        h = InteractionHandler.__new__(InteractionHandler)
        fig = Figure()
        FigureCanvasAgg(fig)
        h.ax1 = fig.add_subplot(111)
        h._setup_span_selector()
        fig.canvas.draw()
        h.show_span(0.3, 0.6)
        assert h.span_selector.extents == (0.3, 0.6)
        assert h.span_selector.get_visible()

    def test_interaction_handler_show_span_without_selector(self):
        h = InteractionHandler.__new__(InteractionHandler)
        h.span_selector = None
        h.show_span(0.3, 0.6)  # no error
