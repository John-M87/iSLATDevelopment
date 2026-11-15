# -*- coding: utf-8 -*-
"""color_by(all_lines=True) colors every line, not only the valid (strong) ones.

By default a color-mapped population diagram draws only the lines in
``valid_mask`` (flux above 1 % of the strongest line's) and takes its color
scale from them, while the uncolored diagram draws every line.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from iSLAT.Modules.Plotting import PopulationDiagramPlot

N = 6
VALID = np.array([True, True, True, False, False, False])


def _component():
    return {
        "name": "TestMol", "color": "#3399FF", "molecule_id": None,
        "eu": np.arange(N, dtype=float) * 500.0 + 100.0,
        "rd_yax": -np.arange(N, dtype=float),
        "wavelength": np.linspace(4.6, 5.1, N),
        "intens": np.logspace(0, -5, N),
        "a_stein": np.arange(N, dtype=float) + 1.0,     # 1..6; valid lines 1..3
        "g_up": np.full(N, 3.0),
        "lev_up": np.array([f"{v}|" for v in (1, 1, 2, 2, 3, 4)]),
        "lev_low": np.array([f"{v}|R_{j}" for v, j in zip((0, 0, 1, 1, 2, 3), range(N))]),
        "e_low": np.arange(N, dtype=float) * 400.0,
        "valid_mask": VALID.copy(),
        "beam_s": 1.0,
    }


class _FixedDataPlot(PopulationDiagramPlot):
    """A population diagram whose one component has fixed data."""

    def _resolve_components(self):
        return [{"name": "TestMol"}]

    def _compute_component(self, comp):
        return _component()


@pytest.fixture
def pdp():
    fig, ax = plt.subplots()
    plot = _FixedDataPlot(ax=ax)
    yield plot
    plt.close(fig)


def _points(pdp):
    return sum(len(c.get_offsets()) for c in pdp.ax.collections)


class TestContinuous:
    def test_default_colors_only_valid_lines(self, pdp):
        pdp.color_by("a_stein")
        assert _points(pdp) == VALID.sum()
        assert (pdp._colorbar.norm.vmin, pdp._colorbar.norm.vmax) == (1.0, 3.0)

    def test_all_lines_colors_every_line(self, pdp):
        pdp.color_by("a_stein", all_lines=True)
        assert _points(pdp) == N
        assert (pdp._colorbar.norm.vmin, pdp._colorbar.norm.vmax) == (1.0, 6.0)

    def test_all_lines_matches_the_uncolored_diagram(self, pdp):
        pdp.generate_plot()
        plain = np.asarray(pdp.ax.collections[0].get_offsets())
        pdp.color_by("a_stein", all_lines=True)
        np.testing.assert_array_equal(np.asarray(pdp.ax.collections[0].get_offsets()), plain)

    def test_explicit_limits_still_win(self, pdp):
        pdp.color_by("a_stein", vmin=0.0, vmax=10.0, all_lines=True)
        assert (pdp._colorbar.norm.vmin, pdp._colorbar.norm.vmax) == (0.0, 10.0)

    def test_axis_limits_do_not_change(self, pdp):
        pdp.color_by("a_stein")
        limits = (pdp.ax.get_xlim(), pdp.ax.get_ylim())
        pdp.color_by("a_stein", all_lines=True)
        assert (pdp.ax.get_xlim(), pdp.ax.get_ylim()) == limits

    def test_a_new_color_by_goes_back_to_valid_lines(self, pdp):
        pdp.color_by("a_stein", all_lines=True)
        pdp.color_by("a_stein")
        assert _points(pdp) == VALID.sum()


class TestCategorical:
    def _legend_labels(self, pdp):
        return [t.get_text() for t in pdp.ax.get_legend().get_texts()]

    def test_default_colors_only_valid_lines(self, pdp):
        pdp.color_by("lev_up")
        assert _points(pdp) == VALID.sum()
        assert self._legend_labels(pdp) == ["1|", "2|"]

    def test_all_lines_colors_every_line(self, pdp):
        pdp.color_by("lev_up", all_lines=True)
        assert _points(pdp) == N
        assert self._legend_labels(pdp) == ["1|", "2|", "3|", "4|"]
