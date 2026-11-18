# -*- coding: utf-8 -*-
"""Atomic-line labels stay out of the figure layout.

The Three Panel view draws a label for every atomic line in the table, most
of them far outside the shown wavelength range.  Counted by constrained
layout, they made it fail ("axes sizes collapsed to zero") and re-run on
every show / hide; the stacked plots re-measured their layout on each toggle.
"""

import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd

from iSLAT.Modules.Plotting.BasePlot import BasePlot
from iSLAT.Modules.Plotting.CachedLayout import CachedConstrainedLayoutEngine

from tests.plotting.test_cached_layout import _figure, _positions, _runs_for, layout_runs  # noqa: F401  (fixture)
from tests.plotting.test_stacked_layout_engine import _drawn, _fsp, layout_passes, molecules  # noqa: F401  (fixtures)

TAG = "_islat_atomic_line"


def _atomic_table(lo=5.0, hi=28.0, n=54):
    waves = np.linspace(lo, hi, n)
    return pd.DataFrame({
        "wave": waves,
        "species": ["Ne II"] * n,
        "line": [f"{w:.3f}" for w in waves],
    })


def _tagged_texts(ax):
    return [t for t in ax.texts if getattr(t, TAG, False)]


def _draw_quietly(fig):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    return [str(w.message) for w in caught]


class TestThreePanelLayout:
    """Labels for the whole table on the spectrum axes, as ThreePanelView draws them (no x range)."""

    def _drawn_figure(self):
        fig = _figure(CachedConstrainedLayoutEngine())
        fig.canvas.draw()
        return fig, fig.axes[0]

    def test_labels_are_not_in_layout(self):
        fig, ax = self._drawn_figure()
        BasePlot._plot_atomic_lines(ax, _atomic_table(), tag=TAG)
        texts = _tagged_texts(ax)
        assert len(texts) == 54
        assert not any(t.get_in_layout() for t in texts)
        assert not any(t in ax.get_default_bbox_extra_artists() for t in texts)

    def test_showing_labels_keeps_the_layout(self, layout_runs):
        fig, ax = self._drawn_figure()
        layout_runs.clear()  # the first draw lays out
        before = _positions(fig)
        BasePlot._plot_atomic_lines(ax, _atomic_table(), tag=TAG)
        messages = _draw_quietly(fig)
        assert _runs_for(layout_runs, fig) == 0
        assert not any("collapsed" in m for m in messages)
        np.testing.assert_array_equal(_positions(fig), before)

    def test_hiding_labels_keeps_the_layout(self, layout_runs):
        fig, ax = self._drawn_figure()
        BasePlot._plot_atomic_lines(ax, _atomic_table(), tag=TAG)
        fig.canvas.draw()
        layout_runs.clear()
        BasePlot._clear_tagged_artists(ax, TAG, lines=True, collections=False, texts=True)
        fig.canvas.draw()
        assert _runs_for(layout_runs, fig) == 0
        assert _tagged_texts(ax) == []

    def test_labels_are_still_drawn(self):
        fig, ax = self._drawn_figure()
        BasePlot._plot_atomic_lines(ax, _atomic_table(5.5, 9.5, n=5), tag=TAG)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axbox = ax.get_window_extent(renderer)
        for text in _tagged_texts(ax):
            assert text.get_visible()
            box = text.get_window_extent(renderer)
            assert axbox.x0 <= box.x0 <= axbox.x1
            assert box.y1 <= axbox.y1 + 1  # hangs from the top of the axes


class TestStackedLayout:
    def test_toggling_labels_skips_the_stacked_layout(self, molecules, layout_passes):
        plot = _fsp(molecules)
        fig = _drawn(plot)
        try:
            before = _positions(fig)
            layout_passes.clear()
            plot.plot_atomic_lines(_atomic_table(10.0, 20.0, n=40))
            fig.canvas.draw()
            assert layout_passes == []
            np.testing.assert_array_equal(_positions(fig), before)
            plot.remove_atomic_lines()
            fig.canvas.draw()
            assert layout_passes == []
        finally:
            plot.close()
