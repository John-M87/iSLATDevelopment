# -*- coding: utf-8 -*-
"""FigureFonts: setting the text sizes of a rendered figure by kind of text.

The full-spectrum exports call :func:`apply_figure_font_sizes` on the plots
the Full Spectrum view builds (plain, residual sub-panels, sample stack),
so it has to reach every kind of text those plots draw - tick labels, axis
and figure labels, the molecule legend, line labels, wavelength marks and
chi^2 values - and leave the kinds it is not given alone.
"""

import io

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.Plotting import FullSpectrumPlot, ResidualSpectrumPlot
from iSLAT.Modules.Plotting.FigureFonts import (
    FONT_SIZE_KINDS,
    MAX_FONT_SIZE,
    apply_figure_font_sizes,
    default_font_sizes,
    validate_font_size,
    validate_font_sizes,
)
from iSLAT.Modules.Plotting.FullSpectrumView import draw_toggle_state_markers
from iSLAT.Modules.Plotting.LegendStrategy import StandardLegend

from tests.plotting import make_atomic_lines, make_line_list, make_wave_flux
from tests.plotting.test_plot_views import _make_test_molecule

COLORS = {"H2O": "#1f77b4", "CO": "#d62728", "OH": "#2ca02c"}


@pytest.fixture(scope="module")
def molecules():
    md = MoleculeDict()
    for (name, color), start in zip(COLORS.items(), (12.0, 14.5, 11.0)):
        mol = _make_test_molecule(name=name, lam_start=start)
        mol.color = color
        md[mol.name] = mol
    return md


def _render(molecules, cls=FullSpectrumPlot, figsize=(12, 9), **kw):
    wave, flux, err = make_wave_flux(n=600)
    kw.setdefault("n_panels", 3)
    if cls is ResidualSpectrumPlot:
        kw.update(model_flux=np.zeros_like(flux), error_data=err)
    plot = cls(
        wave, flux, molecules=molecules, figsize=figsize,
        line_list=make_line_list([12.5, 15.0]), atomic_lines=make_atomic_lines([13.0, 17.0]), **kw,
    )
    plot.generate_plot()
    draw_toggle_state_markers(plot.fig, {"wavelength_markers": [{"start": 14.0, "end": 14.0, "label": "mark"}]})
    return plot


def _draw(fig):
    """Draw *fig* as a save does (the standalone plots have no drawing canvas of their own)."""
    fig.savefig(io.BytesIO(), format="png", dpi=50)


def _legend(plot):
    (legend,) = [ax.get_legend() for ax in plot.fig.axes if ax.get_legend() is not None]
    return legend


def _sizes(fig):
    """The font sizes in *fig*, by kind of text."""
    sizes = {"ticks": set(), "labels": set(), "legend": set(), "text": set()}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            sizes["ticks"] |= {t.get_fontsize() for t in axis.get_ticklabels()}
            if axis.label.get_text():
                sizes["labels"].add(axis.label.get_fontsize())
        if ax.get_legend() is not None:
            sizes["legend"] |= {t.get_fontsize() for t in ax.get_legend().get_texts()}
        sizes["text"] |= {t.get_fontsize() for t in ax.texts if t.get_text()}
    sizes["labels"].add(fig._supylabel.get_fontsize())
    sup = {id(fig._supxlabel), id(fig._supylabel)}
    sizes["text"] |= {t.get_fontsize() for t in fig.texts if id(t) not in sup and t.get_text()}
    return sizes


# ======================================================================
# Validation
# ======================================================================
class TestValidation:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_blank_is_the_default(self, value):
        assert validate_font_size("Legend", value) is None

    @pytest.mark.parametrize("value, expected", [(9, 9.0), ("11.5", 11.5), (MAX_FONT_SIZE, MAX_FONT_SIZE)])
    def test_numbers(self, value, expected):
        assert validate_font_size("Legend", value) == expected

    @pytest.mark.parametrize("value", ["big", 0, -2, float("nan"), float("inf"), MAX_FONT_SIZE + 1, True])
    def test_invalid_sizes_raise_with_the_name(self, value):
        with pytest.raises(ValueError, match="^Legend font size"):
            validate_font_size("Legend", value)

    def test_every_kind_is_given_and_unknown_kinds_raise(self):
        assert validate_font_sizes(None) == default_font_sizes()
        assert validate_font_sizes({"ticks": "8"}) == {"ticks": 8.0, "labels": None, "legend": None, "text": None}
        with pytest.raises(ValueError, match="Unknown kind"):
            validate_font_sizes({"title": 12})
        with pytest.raises(ValueError, match="^Tick label font size"):
            validate_font_sizes({"ticks": "x"})

    def test_kinds(self):
        assert list(FONT_SIZE_KINDS) == ["ticks", "labels", "legend", "text"]


# ======================================================================
# Applying the sizes
# ======================================================================
class TestApplyFigureFontSizes:
    @pytest.mark.parametrize("cls", [FullSpectrumPlot, ResidualSpectrumPlot])
    def test_every_kind_of_text(self, molecules, cls):
        plot = _render(molecules, cls)
        before = _sizes(plot.fig)
        assert all(before.values())       # the plot draws every kind
        apply_figure_font_sizes(plot.fig, {"ticks": 12, "labels": 14, "legend": 16, "text": 9}, plot.legend_strategy)
        assert _sizes(plot.fig) == {"ticks": {12}, "labels": {14}, "legend": {16}, "text": {9}}
        # The offset texts follow the tick labels.
        assert {ax.yaxis.offsetText.get_fontsize() for ax in plot.fig.axes} == {12}
        plot.close()

    def test_none_leaves_that_kind_alone(self, molecules):
        plot = _render(molecules)
        before = _sizes(plot.fig)
        apply_figure_font_sizes(plot.fig, {"ticks": None, "labels": None, "legend": 15, "text": None},
                                plot.legend_strategy)
        after = _sizes(plot.fig)
        assert after["legend"] == {15}
        assert {k: v for k, v in after.items() if k != "legend"} == {k: v for k, v in before.items() if k != "legend"}
        apply_figure_font_sizes(plot.fig, default_font_sizes(), plot.legend_strategy)
        apply_figure_font_sizes(plot.fig, None)
        assert _sizes(plot.fig) == after
        plot.close()

    def test_ticks_made_later_get_the_size(self, molecules):
        # Tick labels are re-created as the limits change; the size is an axis setting.
        plot = _render(molecules)
        apply_figure_font_sizes(plot.fig, {"ticks": 11}, plot.legend_strategy)
        ax = plot.fig.axes[0]
        ax.set_xlim(12.0, 12.5)
        _draw(plot.fig)
        assert {t.get_fontsize() for t in ax.get_xticklabels()} == {11}
        plot.close()

    def test_molecule_legend_is_rebuilt_for_its_new_size(self, molecules):
        # Narrow figure: at a large size the molecule names no longer fit in one row.
        plot = _render(molecules, figsize=(3, 6))
        legend = _legend(plot)
        labels = [t.get_text() for t in legend.get_texts()]
        assert legend._ncols == len(labels) == 3
        apply_figure_font_sizes(plot.fig, {"legend": 40}, plot.legend_strategy)
        rebuilt = _legend(plot)
        assert rebuilt is not legend and rebuilt._ncols < 3
        assert [t.get_text() for t in rebuilt.get_texts()] == labels
        assert [t.get_color() for t in rebuilt.get_texts()] == [COLORS[name] for name in labels]
        assert all(getattr(t, "_islat_mol_color", False) for t in rebuilt.get_texts())
        plot.close()

    def test_hidden_legend_stays_hidden(self, molecules):
        plot = _render(molecules)
        plot.legend_strategy.update_visibility(_legend(plot).axes, False)
        apply_figure_font_sizes(plot.fig, {"legend": 14}, plot.legend_strategy)
        assert not _legend(plot).get_visible()
        plot.close()

    def test_other_legends_only_get_the_text_size(self):
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([0, 1], label="a")
        StandardLegend().build_legend(ax, fig, [], [])
        legend = ax.get_legend()
        legend.set_title("T")
        fig.legend(["b"])
        apply_figure_font_sizes(fig, {"legend": 13}, StandardLegend())
        assert ax.get_legend() is legend
        assert legend.get_texts()[0].get_fontsize() == 13 and legend.get_title().get_fontsize() == 13
        assert fig.legends[0].get_texts()[0].get_fontsize() == 13

    def test_titles_and_suptitle_are_annotations(self):
        fig = Figure()
        ax = fig.add_subplot()
        ax.set_title("centre")
        ax.set_title("left", loc="left")
        fig.suptitle("top")
        fig.supxlabel("x")
        apply_figure_font_sizes(fig, {"text": 7, "labels": 15})
        assert {ax.title.get_fontsize(), ax._left_title.get_fontsize(), fig._suptitle.get_fontsize()} == {7}
        assert fig._supxlabel.get_fontsize() == 15

    def test_layout_makes_room_for_larger_text(self, molecules):
        def margins(sizes):
            plot = _render(molecules)
            apply_figure_font_sizes(plot.fig, sizes, plot.legend_strategy)
            _draw(plot.fig)
            gs = plot.fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
            result = (gs.left, gs.bottom)
            plot.close()
            return result

        default = margins({})
        large = margins({"ticks": 14, "labels": 18})
        assert large[0] > default[0]        # wider y tick labels and flux label
        assert large[1] > default[1]        # taller x tick labels and wavelength label

    def test_large_tick_labels_do_not_reach_the_next_panel(self, molecules):
        plot = _render(molecules)
        apply_figure_font_sizes(plot.fig, {"ticks": 24}, plot.legend_strategy)
        _draw(plot.fig)
        renderer = plot.fig._get_renderer()
        rows = [plot.subplots[idx] for idx in sorted(plot.subplots)]
        for upper, lower in zip(rows, rows[1:]):
            assert upper.get_tightbbox(renderer).y0 >= lower.get_tightbbox(renderer).y1
        plot.close()

    def test_no_figure(self):
        apply_figure_font_sizes(None, {"ticks": 9})
