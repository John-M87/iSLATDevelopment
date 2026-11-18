# -*- coding: utf-8 -*-
"""PopulationDiagramPlot.size_by: marker size from a line property (e.g. the optical depth tau)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.legend import Legend

from iSLAT.Modules.Plotting import PopulationDiagramPlot

TAU = np.array([1e-3, 1e-2, 1.0, 1e2, 1e3, np.nan])
N = len(TAU)
VALID = np.array([True, True, True, True, False, False])
# Sizes for size_by("tau", log_scale=True, vmin=1e-2, vmax=1e2, min_size=2, max_size=40):
# clipped below, 2 at 1e-2, halfway (21) at 1, 40 at 1e2, clipped above, NaN -> min_size.
EXPECTED_LOG = np.array([2.0, 2.0, 21.0, 40.0, 40.0, 2.0])


def _component():
    return {
        "name": "TestMol", "color": "#3399FF", "molecule_id": None,
        "eu": np.arange(N, dtype=float) * 500.0 + 100.0,
        "rd_yax": -np.arange(N, dtype=float),
        "wavelength": np.linspace(4.6, 5.1, N),
        "intens": np.logspace(0, -5, N),
        "tau": TAU.copy(),
        "a_stein": np.arange(N, dtype=float) + 1.0,
        "g_up": np.full(N, 3.0),
        "lev_up": np.array([f"{v}|" for v in (1, 1, 2, 2, 3, 3)]),
        "lev_low": np.array([f"{v}|R_{j}" for v, j in zip((0, 0, 1, 1, 2, 2), range(N))]),
        "e_low": np.arange(N, dtype=float) * 400.0,
        "valid_mask": VALID.copy(),
        "beam_s": 1.0,
    }


class _FixedDataPlot(PopulationDiagramPlot):
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


def _sizes(pdp):
    """Per-point sizes of the (single) scatter."""
    (sc,) = pdp.ax.collections
    return np.broadcast_to(sc.get_sizes(), (len(sc.get_offsets()),))


def _size_legends(pdp):
    return [a for a in pdp.ax.get_children() if isinstance(a, Legend) and a is not pdp.ax.get_legend()]


def _tau_log(pdp, **kw):
    pdp.size_by("tau", log_scale=True, vmin=1e-2, vmax=1e2, min_size=2, max_size=40, **kw)


class TestSizes:
    def test_without_size_by_every_point_has_one_size(self, pdp):
        pdp.generate_plot()
        assert len(set(_sizes(pdp))) == 1

    def test_log_scale_with_fixed_limits(self, pdp):
        _tau_log(pdp)
        np.testing.assert_allclose(_sizes(pdp), EXPECTED_LOG)

    def test_default_limits_span_the_values(self, pdp):
        pdp.size_by("a_stein", min_size=1, max_size=11)
        np.testing.assert_allclose(_sizes(pdp), [1, 3, 5, 7, 9, 11])

    def test_log_scale_default_limits_ignore_non_positive_values(self, pdp):
        pdp.size_by("tau", log_scale=True, min_size=0, max_size=6)
        np.testing.assert_allclose(_sizes(pdp), [0, 1, 3, 5, 6, 0])

    def test_categorical_property_gives_each_category_a_size(self, pdp):
        pdp.size_by("lev_up", min_size=2, max_size=10)
        np.testing.assert_allclose(_sizes(pdp), [2, 2, 6, 6, 10, 10])

    def test_overrides_set_marker_size_until_cleared(self, pdp):
        pdp.set_marker_size(9, regenerate=False)
        _tau_log(pdp)
        np.testing.assert_allclose(_sizes(pdp), EXPECTED_LOG)
        pdp.clear_size_mapping()
        assert set(_sizes(pdp)) == {9.0}

    def test_property_missing_from_the_data_keeps_the_default_size(self, pdp):
        pdp.size_by("no_such_property")
        assert len(set(_sizes(pdp))) == 1

    @pytest.mark.parametrize("kw", [
        dict(vmin=2.0, vmax=1.0),
        dict(log_scale=True, vmin=0.0, vmax=1.0),
        dict(min_size=-1.0),
    ])
    def test_invalid_settings_raise(self, pdp, kw):
        with pytest.raises(ValueError):
            pdp.size_by("tau", regenerate=False, **kw)


class TestWithOtherMappings:
    def test_color_by_valid_lines_sizes_follow_the_drawn_points(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.color_by("a_stein")
        np.testing.assert_allclose(_sizes(pdp), EXPECTED_LOG[VALID])

    def test_color_by_all_lines(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.color_by("a_stein", all_lines=True)
        np.testing.assert_allclose(_sizes(pdp), EXPECTED_LOG)

    def test_categorical_color_by(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.color_by("lev_up", all_lines=True)
        np.testing.assert_allclose(_sizes(pdp), EXPECTED_LOG)

    def test_shape_groups_get_their_own_points_sizes(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.shape_by("lev_up")
        offsets = np.concatenate([c.get_offsets() for c in pdp.ax.collections])
        sizes = np.concatenate([np.broadcast_to(c.get_sizes(), (len(c.get_offsets()),))
                                for c in pdp.ax.collections])
        by_eu = dict(zip(offsets[:, 0], sizes))
        eu = _component()["eu"]
        np.testing.assert_allclose([by_eu[e] for e in eu], EXPECTED_LOG)


class TestLegend:
    def test_size_legend_shows_example_sizes(self, pdp):
        _tau_log(pdp)
        (legend,) = _size_legends(pdp)
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ["<= 0.01", "0.1", "1", "10", ">= 100"]  # whole decades on a log scale
        assert "tau" in legend.get_title().get_text()
        sizes = [h.get_markersize() ** 2 for h in legend.legend_handles]
        np.testing.assert_allclose(sizes, [2.0, 11.5, 21.0, 30.5, 40.0])

    def test_linear_legend_has_even_steps(self, pdp):
        pdp.size_by("a_stein", min_size=1, max_size=11)
        (legend,) = _size_legends(pdp)
        assert [t.get_text() for t in legend.get_texts()] == ["<= 1", "2.67", "4.33", ">= 6"]

    def test_decades_inside_the_range_are_not_marked_as_ends(self, pdp):
        pdp.size_by("tau", log_scale=True, vmin=0.5, vmax=2e3)
        (legend,) = _size_legends(pdp)
        assert [t.get_text() for t in legend.get_texts()] == ["1", "10", "100", "1e+03"]

    def test_keeps_the_color_legend(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.color_by("lev_up", all_lines=True)
        assert pdp.ax.get_legend() is not None          # the category colors
        assert len(_size_legends(pdp)) == 1

    def test_no_legend_when_turned_off(self, pdp):
        _tau_log(pdp, legend=False)
        assert _size_legends(pdp) == []

    def test_one_legend_after_redraws(self, pdp):
        _tau_log(pdp)
        pdp.generate_plot()
        pdp.generate_plot()
        assert len(_size_legends(pdp)) == 1

    def test_follows_the_theme(self):
        from matplotlib.colors import to_hex, to_rgba
        theme = {"foreground": "#F0F0F0", "graph_fill_color": "#282C34", "scatter_main_color": "#9CA3AF"}
        fig, ax = plt.subplots()
        try:
            plot = _FixedDataPlot(ax=ax, theme=theme)
            _tau_log(plot)
            (legend,) = _size_legends(plot)
            assert legend.get_frame().get_facecolor()[:3] == to_rgba("#282C34")[:3]
            assert legend.get_frame().get_edgecolor()[:3] == to_rgba("#F0F0F0")[:3]
            assert to_hex(legend.get_title().get_color()) == "#f0f0f0"
            assert {to_hex(t.get_color()) for t in legend.get_texts()} == {"#f0f0f0"}
            assert {to_hex(h.get_markerfacecolor()) for h in legend.legend_handles} == {"#9ca3af"}
        finally:
            plt.close(fig)

    def test_moves_aside_for_the_shape_legend(self, pdp):
        _tau_log(pdp, regenerate=False)
        pdp.shape_by("lev_up")
        (legend,) = _size_legends(pdp)
        assert legend._loc == Legend.codes["lower center"]
