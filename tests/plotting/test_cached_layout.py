# -*- coding: utf-8 -*-
"""CachedConstrainedLayoutEngine: constrained layout only when its inputs change."""

import matplotlib
matplotlib.use("Agg")

import io
import warnings

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import ConstrainedLayoutEngine

from iSLAT.Modules.Plotting.CachedLayout import (
    CachedConstrainedLayoutEngine,
    x_tick_labels_cut_off,
)


def _figure(engine):
    """A figure laid out like the Three Panel view (see MainPlotGrid.create_three_panel_axes)."""
    fig = Figure(figsize=(14, 6), dpi=100, layout=engine)
    FigureCanvasAgg(fig)
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.5], figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.linspace(5.0, 10.0, 400)
    ax1.plot(x, np.sin(x), label="Data")
    ax1.set_xlim(5.2, 9.8)
    ax1.set_title("Full Spectrum with Line Inspection")
    ax1.set_xlabel("Wavelength (um)")
    ax1.set_ylabel("Flux density (Jy)")
    ax1.legend()
    ax2.plot(x, np.cos(x))
    ax2.set_title("Line inspection plot")
    ax3.scatter(x, x ** 2, c=x)
    ax3.set_title("Population diagram")
    ax3.set_xlabel("E_up (K)")
    ax3.set_ylabel("ln(4 pi F / (h nu A g))")
    return fig


def _pair():
    """(stock, cached) figures with identical content, both drawn once."""
    stock = _figure(ConstrainedLayoutEngine())
    cached = _figure(CachedConstrainedLayoutEngine())
    stock.canvas.draw()
    cached.canvas.draw()
    return stock, cached


def _positions(fig):
    return np.array([ax.get_position().bounds for ax in fig.axes])


def _assert_same_layout(fig, other):
    # The stock engine re-solves on every draw, which moves positions by
    # ~1e-16 of the figure size; anything below 1e-9 is the same layout.
    np.testing.assert_allclose(_positions(fig), _positions(other), rtol=0, atol=1e-9)


@pytest.fixture
def layout_runs(monkeypatch):
    """Figures that the real constrained layout ran on, one entry per run."""
    runs = []
    original = ConstrainedLayoutEngine.execute

    def counting(self, fig):
        runs.append(fig)
        return original(self, fig)

    monkeypatch.setattr(ConstrainedLayoutEngine, "execute", counting)
    return runs


def _runs_for(runs, fig):
    return sum(1 for f in runs if f is fig)


def test_is_still_a_constrained_layout():
    fig = _figure(CachedConstrainedLayoutEngine())
    assert fig.get_constrained_layout()
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)


class TestReusesLayout:
    def test_redraw_without_changes(self, layout_runs):
        fig = _figure(CachedConstrainedLayoutEngine())
        for _ in range(3):
            fig.canvas.draw()
        assert _runs_for(layout_runs, fig) == 1

    def test_pan_keeps_positions(self, layout_runs):
        stock, cached = _pair()
        before = _positions(cached)
        runs = _runs_for(layout_runs, cached)
        for fig in (stock, cached):
            fig.axes[0].set_xlim(5.25, 9.85)
        assert not x_tick_labels_cut_off(cached)
        stock.canvas.draw()
        cached.canvas.draw()
        assert _runs_for(layout_runs, cached) == runs
        np.testing.assert_array_equal(_positions(cached), before)
        _assert_same_layout(cached, stock)

    def test_ylim_change_with_same_tick_labels(self, layout_runs):
        fig = _figure(CachedConstrainedLayoutEngine())
        ax3 = fig.axes[2]
        ax3.set_ylim(19.0, 101.0)
        fig.canvas.draw()
        runs = _runs_for(layout_runs, fig)
        ax3.set_ylim(18.0, 102.0)  # same ticks: 20 ... 100
        fig.canvas.draw()
        assert _runs_for(layout_runs, fig) == runs


def _set_ylim_new_labels(fig):
    fig.axes[2].set_ylim(-12000.0, 12000.0)


def _set_title(fig):
    fig.axes[2].set_title("C$_2$H$_2$ Population diagram")


def _set_two_line_label(fig):
    fig.axes[1].set_xlabel("Wavelength (um)\nsecond line")


def _set_tick_label_size(fig):
    fig.axes[0].tick_params(labelsize=16)


def _resize(fig):
    fig.set_size_inches(10.0, 5.0)


def _change_dpi(fig):
    fig.set_dpi(72)


def _text_above_axes(fig):
    ax1 = fig.axes[0]
    ax1.text(0.5, 1.2, "unclipped label", transform=ax1.transAxes)


def _legend_outside(fig):
    fig.axes[1].legend(["a long legend entry"], loc="upper left", bbox_to_anchor=(1.0, 1.0))


def _add_colorbar(fig):
    ax3 = fig.axes[2]
    fig.colorbar(ax3.collections[0], ax=ax3)


def _change_pads(fig):
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.2)


def _hide_axes(fig):
    fig.axes[1].set_visible(False)


@pytest.mark.parametrize("change", [
    _set_ylim_new_labels, _set_title, _set_two_line_label, _set_tick_label_size,
    _resize, _change_dpi, _text_above_axes, _legend_outside, _add_colorbar,
    _change_pads, _hide_axes,
], ids=lambda f: f.__name__.lstrip("_"))
def test_relayout_matches_stock_engine(change, layout_runs):
    stock, cached = _pair()
    runs = _runs_for(layout_runs, cached)
    change(stock)
    change(cached)
    stock.canvas.draw()
    cached.canvas.draw()
    assert _runs_for(layout_runs, cached) == runs + 1
    _assert_same_layout(cached, stock)
    # and the new layout is kept for the next draw
    cached.canvas.draw()
    assert _runs_for(layout_runs, cached) == runs + 1


class TestEdgeGuard:
    def test_label_cut_off_at_edge_triggers_layout(self, layout_runs):
        stock, cached = _pair()
        runs = _runs_for(layout_runs, cached)
        for fig in (stock, cached):
            fig.axes[0].set_xlim(5.0, 10.0)  # "10" now sits on the right edge
        assert x_tick_labels_cut_off(cached)
        stock.canvas.draw()
        cached.canvas.draw()
        assert _runs_for(layout_runs, cached) == runs + 1
        assert not x_tick_labels_cut_off(cached)
        _assert_same_layout(cached, stock)


class TestSaving:
    def test_saved_figure_matches_stock(self):
        stock, cached = _pair()
        out_stock, out_cached = io.BytesIO(), io.BytesIO()
        stock.savefig(out_stock, format="png", dpi=150)
        cached.savefig(out_cached, format="png", dpi=150)
        assert out_stock.getvalue() == out_cached.getvalue()

    def test_screen_layout_redone_after_save(self, layout_runs):
        stock, cached = _pair()
        for fig in (stock, cached):
            fig.savefig(io.BytesIO(), format="png", dpi=150)
        runs = _runs_for(layout_runs, cached)
        stock.canvas.draw()
        cached.canvas.draw()
        assert _runs_for(layout_runs, cached) == runs + 1
        _assert_same_layout(cached, stock)


class TestCollapsedLayout:
    @staticmethod
    def _collapse_warnings(engine, draws=3):
        fig = _figure(engine)
        # Unclipped text far outside the axes, like line labels outside the
        # visible range: constrained layout gives up ("axes sizes collapsed").
        fig.axes[0].text(500.0, 0.0, "far away label")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(draws):
                fig.canvas.draw()
        return [w for w in caught if "constrained_layout not applied" in str(w.message)]

    def test_cached_engine_tries_once(self):
        one_attempt = self._collapse_warnings(ConstrainedLayoutEngine(), draws=1)
        assert one_attempt  # the setup really makes the layout collapse
        assert len(self._collapse_warnings(ConstrainedLayoutEngine(), draws=3)) == 3 * len(one_attempt)
        assert len(self._collapse_warnings(CachedConstrainedLayoutEngine(), draws=3)) == len(one_attempt)
