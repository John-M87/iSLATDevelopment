# -*- coding: utf-8 -*-
"""StackedPanelLayoutEngine: tight margins and equal cells for the full-spectrum style plots.

The stacked plots used fixed fractional margins (6 % left / right, 7 % top,
6 % bottom), which left ~180 px of empty space at each side of a full-screen
canvas.  The engine measures what sticks out of the cells (tick labels, axis
labels, the legend, texts beside the panels, the supylabel, figure texts)
and fits the margins and the gap between cells to it, keeping every cell -
including cells split into sub-panels - the same height.
"""

import io

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import Bbox

from iSLAT.Modules.Plotting import FullSpectrumPlot, ResidualSpectrumPlot
from iSLAT.Modules.Plotting import StackedLayout
from iSLAT.Modules.Plotting.StackedLayout import StackedPanelLayoutEngine

from tests.plotting import make_wave_flux
from tests.plotting.test_optical_depth_view import _MolDict
from tests.plotting.test_plot_views import _make_test_molecule

FIG_SIZE = (30.26, 13.17)  # the main window's plot area on a large screen, at dpi 100


@pytest.fixture(scope="module")
def molecules():
    mols = _MolDict()
    for i, name in enumerate(("H2O", "CO2")):
        mols[name] = _make_test_molecule(name=name, lam_start=12.0 + 0.3 * i)
    return mols


def _fsp(molecules, **kw):
    wave, flux, err = make_wave_flux(10.0, 20.0, n=4000)
    return FullSpectrumPlot(wave, flux, molecules=molecules, error_data=err, n_panels=6, **kw)


def _rsp(molecules, **kw):
    wave, flux, err = make_wave_flux(10.0, 20.0, n=4000)
    return ResidualSpectrumPlot(wave, flux, flux * 0.9, molecules=molecules, error_data=err, n_panels=6, **kw)


def _drawn(plot):
    plot.generate_plot()
    fig = plot.fig
    FigureCanvasAgg(fig)
    fig.set_size_inches(*FIG_SIZE)
    fig.canvas.draw()
    return fig


def _cells(plot):
    return [cell if isinstance(cell, tuple) else (cell,) for _i, cell in sorted(plot.subplots.items())]


def _tight(fig, axes):
    r = fig.canvas.get_renderer()
    return Bbox.union([ax.get_tightbbox(r) for ax in axes])


def _pad_px(fig):
    return StackedPanelLayoutEngine().get()["pad"] * fig.dpi / 72.0


@pytest.fixture
def layout_passes(monkeypatch):
    """Figures the stacked layout measured, one entry per measuring pass."""
    passes = []
    original = StackedLayout._StackedPanelLayout._layout_once

    def counting(self, fig):
        passes.append(fig)
        return original(self, fig)

    monkeypatch.setattr(StackedLayout._StackedPanelLayout, "_layout_once", counting)
    return passes


@pytest.fixture(params=["full", "residual"])
def drawn(request, molecules):
    plot = _fsp(molecules) if request.param == "full" else _rsp(molecules)
    fig = _drawn(plot)
    yield plot, fig
    plot.close()


class TestLayout:
    def test_uses_the_stacked_engine(self, drawn):
        _plot, fig = drawn
        assert isinstance(fig.get_layout_engine(), StackedPanelLayoutEngine)

    def test_cells_have_equal_heights(self, drawn):
        # Including the first cell (legend above) and the last (axis label below).
        plot, _fig = drawn
        heights = {tuple(round(ax.bbox.height, 1) for ax in cell) for cell in _cells(plot)}
        assert len(heights) == 1

    def test_content_reaches_the_pad_at_each_edge(self, drawn):
        plot, fig = drawn
        r = fig.canvas.get_renderer()
        boxes = [_tight(fig, cell) for cell in _cells(plot)]
        boxes.append(fig._supylabel.get_window_extent(r))
        boxes += [t.get_window_extent(r) for t in fig.texts if t is not fig._supylabel]
        content = Bbox.union(boxes)
        pad = _pad_px(fig)
        width, height = fig.bbox.width, fig.bbox.height
        edges = {"left": content.x0, "right": width - content.x1,
                 "top": height - content.y1, "bottom": content.y0}
        for edge, gap in edges.items():
            assert pad - 1 <= gap <= 2 * pad + 4, (edge, gap)

    def test_wide_canvas_has_narrow_side_margins(self, drawn):
        plot, fig = drawn
        first = _cells(plot)[0][0]
        assert first.bbox.x0 < 0.03 * fig.bbox.width            # was 6 % (182 px here)
        assert fig.bbox.width - first.bbox.x1 < 0.04 * fig.bbox.width

    def test_tick_labels_clear_the_next_cell(self, drawn):
        plot, fig = drawn
        cells = _cells(plot)
        for upper, lower in zip(cells, cells[1:]):
            below_upper = _tight(fig, upper).y0
            top_lower = Bbox.union([ax.bbox for ax in lower]).y1
            assert below_upper > top_lower

    def test_supylabel_left_of_the_tick_labels(self, drawn):
        plot, fig = drawn
        r = fig.canvas.get_renderer()
        sup = fig._supylabel.get_window_extent(r)
        assert sup.x0 >= 0
        assert sup.x1 < min(_tight(fig, cell).x0 for cell in _cells(plot))

    def test_legend_above_the_first_panel(self, drawn):
        plot, fig = drawn
        first = _cells(plot)[0][0]
        leg = first.get_legend()
        assert leg is not None
        box = leg.get_window_extent(fig.canvas.get_renderer())
        assert box.y0 >= first.bbox.y1 - 1 and box.y1 <= fig.bbox.height


class TestResidualPlot:
    def test_side_texts_stay_on_the_figure(self, molecules):
        plot = _rsp(molecules)
        fig = _drawn(plot)
        right = max(_tight(fig, cell).x1 for cell in _cells(plot))
        assert right <= fig.bbox.width
        plot.close()

    def test_total_chi2_box_below_the_axes(self, molecules):
        plot = _rsp(molecules)
        fig = _drawn(plot)
        r = fig.canvas.get_renderer()
        texts = [t for t in fig.texts if t is not fig._supylabel and t.get_text()]
        assert texts
        lowest = _tight(fig, _cells(plot)[-1]).y0
        assert all(t.get_window_extent(r).y1 < lowest for t in texts)
        plot.close()

    def test_inplace_update_keeps_the_supylabel(self, molecules):
        """Regression: the in-place update removed every figure text, the supylabel included.

        The figure kept the detached label as ``_supylabel``, so the next draw
        failed in the layout engine (``'NoneType' object has no attribute 'dpi'``).
        """
        plot = _rsp(molecules)
        fig = _drawn(plot)
        plot.update_panels_inplace()
        fig.canvas.draw()
        assert fig._supylabel in fig.texts
        chi2_boxes = [t for t in fig.texts if t is not fig._supylabel and t.get_text()]
        assert len(chi2_boxes) == 1
        plot.close()

    def test_detached_supylabel_does_not_break_the_layout(self, molecules):
        plot = _rsp(molecules)
        fig = _drawn(plot)
        fig._supylabel.remove()
        fig.set_size_inches(FIG_SIZE[0] / 2, FIG_SIZE[1])  # make the layout measure again
        fig.canvas.draw()
        plot.close()


class TestSpacingAndCaching:
    def test_hspace_is_a_minimum_gap(self, molecules):
        plot = _fsp(molecules, hspace=0.6)
        _drawn(plot)
        cells = _cells(plot)
        gap = cells[0][0].bbox.y0 - cells[1][0].bbox.y1
        assert gap >= 0.6 * cells[0][0].bbox.height - 0.5
        plot.close()

    def test_redraw_reuses_the_layout_and_resize_measures_again(self, molecules, layout_passes):
        plot = _fsp(molecules)
        fig = _drawn(plot)
        layout_passes.clear()
        fig.canvas.draw()
        assert layout_passes == []
        fig.set_size_inches(20, 9)
        fig.canvas.draw()
        assert layout_passes
        heights = {round(cell[0].bbox.height, 1) for cell in _cells(plot)}
        assert len(heights) == 1
        plot.close()

    def test_changed_y_limits_measure_again(self, molecules, layout_passes):
        plot = _fsp(molecules)
        fig = _drawn(plot)
        layout_passes.clear()
        _cells(plot)[2][0].set_ylim(0, 1234.5)  # wider tick labels
        fig.canvas.draw()
        assert layout_passes
        plot.close()

    def test_export_lays_out_at_its_own_size(self, molecules):
        plot = _fsp(molecules)
        fig = _drawn(plot)
        before = [cell[0].bbox.height for cell in _cells(plot)]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=50)
        assert buf.getvalue()
        fig.canvas.draw()  # the screen layout comes back
        after = [cell[0].bbox.height for cell in _cells(plot)]
        np.testing.assert_allclose(after, before, atol=0.5)
        plot.close()
