# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow.

Only the module-level pure helpers are exercised - array subsetting, the
settings snapshot/restore, and the main-diagram lookup - so no Tk root and no
matplotlib canvas is constructed.
"""

from types import SimpleNamespace

import pytest
import numpy as np

from iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow import (
    _FrozenPopulationSource,
    _STYLE_FIELDS,
    apply_pdp_style,
    capture_pdp_style,
    find_main_population_plot,
    subset_population_data,
)


def _data(n=5):
    """A population-diagram data dict shaped like Intensity's real output."""
    return {
        'eu': np.arange(n, dtype=float) * 100.0,
        'rd_yax': np.arange(n, dtype=float),
        'wavelength': np.linspace(4.0, 5.0, n),
        'intens': np.ones(n),
        'a_stein': np.ones(n) * 0.01,
        'g_up': np.arange(n) * 2 + 1,
        'g_low': np.arange(n) * 2 - 1,
        'lev_up': np.array([f'0|{i}' for i in range(n)]),
        'lev_low': np.array([f'0|{i-1}' for i in range(n)]),
        'e_low': np.arange(n, dtype=float) * 90.0,
        'tau': np.ones(n) * 0.5,
        'valid_mask': np.ones(n, dtype=bool),
        'beam_s': 1.234,          # scalar - must survive untouched
    }


class _FakePDP:
    """Minimal stand-in exposing the settings surface the helpers touch."""

    def __init__(self):
        self._x_prop, self._y_prop = 'eu', 'rd_yax'
        self._x_log = self._y_log = False
        self._x_lim = self._y_lim = None
        self._color_mapping = None
        self._shape_mapping = None
        self._marker_size = None
        self.regenerated = 0

    def set_axes(self, x_prop, y_prop, x_log, y_log, x_lim=None, y_lim=None,
                 *, regenerate=True):
        self._x_prop, self._y_prop = x_prop, y_prop
        self._x_log, self._y_log = x_log, y_log
        self._x_lim, self._y_lim = x_lim, y_lim
        if regenerate:
            self.regenerated += 1

    def color_by(self, prop, *, cmap="viridis", vmin=None, vmax=None,
                 pmin=None, pmax=None, log_scale=False, all_lines=False, regenerate=True):
        # Mirrors PopulationDiagramPlot.color_by exactly; see
        # test_fake_matches_real_color_by_signature.
        self._color_mapping = {'prop': prop, 'cmap': cmap,
                               'vmin': vmin, 'vmax': vmax,
                               'pmin': pmin, 'pmax': pmax,
                               'log_scale': log_scale, 'all_lines': all_lines}

    def clear_color_mapping(self, *, regenerate=True):
        self._color_mapping = None

    def shape_by(self, prop, *, n_bins=5, regenerate=True):
        self._shape_mapping = {'prop': prop, 'n_bins': n_bins}

    def clear_shape_mapping(self, *, regenerate=True):
        self._shape_mapping = None

    def set_marker_size(self, size, *, regenerate=True):
        self._marker_size = float(size)

    def clear_marker_size(self, *, regenerate=True):
        self._marker_size = None

    def generate_plot(self):
        self.regenerated += 1


# ============================================================================
# subset_population_data
# ============================================================================

class TestSubsetPopulationData:
    """Indexing the precomputed arrays is what makes redraws cheap."""

    def test_subsets_every_per_line_array(self):
        mask = np.array([True, False, True, False, True])
        out = subset_population_data(_data(), mask)
        assert len(out['eu']) == 3
        assert list(out['eu']) == [0.0, 200.0, 400.0]
        assert list(out['lev_up']) == ['0|0', '0|2', '0|4']

    def test_scalars_pass_through(self):
        out = subset_population_data(_data(), np.array([True] + [False] * 4))
        assert out['beam_s'] == 1.234

    def test_none_mask_returns_data_unchanged(self):
        data = _data()
        assert subset_population_data(data, None) is data

    def test_none_data_returns_none(self):
        assert subset_population_data(None, np.array([True])) is None

    def test_empty_selection_gives_empty_arrays(self):
        out = subset_population_data(_data(), np.zeros(5, dtype=bool))
        assert len(out['eu']) == 0
        assert len(out['lev_up']) == 0

    def test_full_selection_preserves_everything(self):
        out = subset_population_data(_data(), np.ones(5, dtype=bool))
        assert list(out['eu']) == list(_data()['eu'])

    def test_wrong_length_mask_raises(self):
        """A stale mask must fail loudly, not plot the wrong lines."""
        with pytest.raises(ValueError, match="does not match"):
            subset_population_data(_data(), np.ones(3, dtype=bool))

    def test_arrays_stay_aligned_with_each_other(self):
        mask = np.array([False, True, True, False, False])
        out = subset_population_data(_data(), mask)
        lengths = {len(out[k]) for k in
                   ('eu', 'rd_yax', 'wavelength', 'lev_up', 'tau')}
        assert lengths == {2}

    def test_does_not_mutate_the_input(self):
        data = _data()
        subset_population_data(data, np.array([True, False, False, False, False]))
        assert len(data['eu']) == 5


# ============================================================================
# Style snapshot / restore
# ============================================================================

class TestStyleTransfer:
    """The six user-adjustable settings must copy from the main diagram."""

    def test_capture_covers_every_declared_field(self):
        style = capture_pdp_style(_FakePDP())
        assert set(style) == set(_STYLE_FIELDS)

    def test_capture_of_none_is_empty(self):
        assert capture_pdp_style(None) == {}

    def test_axes_transfer(self):
        src = _FakePDP()
        src.set_axes('e_low', 'intens', True, True, ('exact', 1, 2), None,
                     regenerate=False)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert (dst._x_prop, dst._y_prop) == ('e_low', 'intens')
        assert dst._x_log and dst._y_log
        assert dst._x_lim == ('exact', 1, 2)

    def test_color_mapping_transfer(self):
        src = _FakePDP()
        src.color_by('wavelength', cmap='viridis', vmin=1.0, vmax=2.0)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert dst._color_mapping['prop'] == 'wavelength'
        assert dst._color_mapping['cmap'] == 'viridis'

    def test_color_mapping_carries_percentiles_and_log_scale(self):
        """Dropping these would give the two diagrams different colour norms."""
        src = _FakePDP()
        src.color_by('eu', pmin=5.0, pmax=95.0, log_scale=True)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert dst._color_mapping['pmin'] == 5.0
        assert dst._color_mapping['pmax'] == 95.0
        assert dst._color_mapping['log_scale'] is True

    def test_color_mapping_carries_all_lines(self):
        """The filtered diagram colors the same lines as the main one."""
        src = _FakePDP()
        src.color_by('eu', all_lines=True)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert dst._color_mapping['all_lines'] is True
        apply_pdp_style(dst, {'_color_mapping': {'prop': 'eu'}})
        assert dst._color_mapping['all_lines'] is False

    def test_absent_cmap_does_not_override_the_default(self):
        dst = _FakePDP()
        apply_pdp_style(dst, {'_color_mapping': {'prop': 'eu'}})
        assert dst._color_mapping['cmap'] == 'viridis'

    def test_fake_matches_real_color_by_signature(self):
        """Guards this stub against drift in PopulationDiagramPlot.color_by."""
        import inspect
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import (
            PopulationDiagramPlot)
        real = set(inspect.signature(PopulationDiagramPlot.color_by).parameters)
        fake = set(inspect.signature(_FakePDP.color_by).parameters)
        assert real == fake

    def test_shape_mapping_transfer(self):
        src = _FakePDP()
        src.shape_by('eu', n_bins=7)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert dst._shape_mapping == {'prop': 'eu', 'n_bins': 7}

    def test_marker_size_transfer(self):
        src = _FakePDP()
        src.set_marker_size(9)
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(src))
        assert dst._marker_size == 9.0

    def test_absent_mappings_are_cleared_not_left_stale(self):
        dst = _FakePDP()
        dst.color_by('eu')
        dst.shape_by('eu')
        dst.set_marker_size(4)
        apply_pdp_style(dst, capture_pdp_style(_FakePDP()))
        assert dst._color_mapping is None
        assert dst._shape_mapping is None
        assert dst._marker_size is None

    def test_missing_lim_attributes_tolerated(self):
        """_x_lim/_y_lim only exist after set_axes has run on the source."""
        class Bare:
            _x_prop, _y_prop = 'eu', 'rd_yax'
            _x_log = _y_log = False
            _color_mapping = _shape_mapping = _marker_size = None
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(Bare()))
        assert dst._x_lim is None

    def test_apply_does_not_regenerate_by_default(self):
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(_FakePDP()))
        assert dst.regenerated == 0

    def test_apply_can_regenerate(self):
        dst = _FakePDP()
        apply_pdp_style(dst, capture_pdp_style(_FakePDP()), regenerate=True)
        assert dst.regenerated == 1

    def test_apply_of_empty_style_is_a_noop(self):
        dst = _FakePDP()
        dst.set_marker_size(3)
        apply_pdp_style(dst, {})
        assert dst._marker_size == 3.0

    def test_apply_to_none_is_a_noop(self):
        apply_pdp_style(None, {'_x_prop': 'eu'})   # must not raise


# ============================================================================
# find_main_population_plot
# ============================================================================

class TestFindMainPopulationPlot:
    """Locating the diagram to copy settings from must degrade gracefully."""

    def test_none_islat(self):
        assert find_main_population_plot(None) is None

    def test_islat_without_gui(self):
        assert find_main_population_plot(SimpleNamespace()) is None

    def test_gui_without_plot_manager(self):
        assert find_main_population_plot(
            SimpleNamespace(GUI=SimpleNamespace())) is None

    def test_finds_dedicated_view_plot(self):
        sentinel = object()
        plot_manager = SimpleNamespace(
            _population_diagram_view=SimpleNamespace(_plot=sentinel))
        islat = SimpleNamespace(GUI=SimpleNamespace(plot=plot_manager))
        assert find_main_population_plot(islat) is sentinel

    def test_falls_back_to_three_panel_grid(self):
        sentinel = object()
        plot_manager = SimpleNamespace(
            _population_diagram_view=None,
            _three_panel_view=SimpleNamespace(
                _grid=SimpleNamespace(pop_diagram_panel=sentinel)))
        islat = SimpleNamespace(GUI=SimpleNamespace(plot=plot_manager))
        assert find_main_population_plot(islat) is sentinel

    def test_view_present_but_not_yet_activated(self):
        """_plot is None until the view has been activated at least once."""
        plot_manager = SimpleNamespace(
            _population_diagram_view=SimpleNamespace(_plot=None),
            _three_panel_view=None)
        islat = SimpleNamespace(GUI=SimpleNamespace(plot=plot_manager))
        assert find_main_population_plot(islat) is None


# ============================================================================
# _FrozenPopulationSource
# ============================================================================

class TestFrozenPopulationSource:
    """The duck-typed stand-in the renderer pulls its arrays from."""

    def test_serves_the_data_it_was_given(self):
        data = _data()
        src = _FrozenPopulationSource(data)
        assert src.get_population_diagram_data(1.0, 160.0) is data

    def test_set_data_swaps_the_payload(self):
        src = _FrozenPopulationSource(None)
        data = _data()
        src.set_data(data)
        assert src.get_population_diagram_data(1.0, 160.0) is data

    def test_accepts_the_intensity_keyword_arguments(self):
        """Must tolerate the same call shape Intensity is given."""
        src = _FrozenPopulationSource(_data())
        assert src.get_population_diagram_data(
            1.0, 160.0, molecule=None, full_range=True) is not None


# ============================================================================
# valid_mask is recomputed, not sliced (regression)
# ============================================================================

class TestValidMaskRecomputed:
    """Intensity defines valid_mask as "flux above 1% of the maximum flux".

    Slicing the original mask keeps that threshold pinned to the unfiltered
    list's brightest line, so filtering down to faint lines would mark every
    one invalid and draw nothing.
    """

    def _bright_and_faint(self):
        intens = np.array([1000.0, 900.0, 1.0, 0.9, 0.8, 0.7])
        data = _data(6)
        data['intens'] = intens
        data['valid_mask'] = intens > np.nanmax(intens) / 100.0
        return data

    def test_faint_subset_is_not_empty(self):
        data = self._bright_and_faint()
        assert list(data['valid_mask']) == [True, True, False, False, False, False]
        out = subset_population_data(
            data, np.array([False, False, True, True, True, True]))
        assert bool(out['valid_mask'].any())

    def test_threshold_is_relative_to_the_subset(self):
        out = subset_population_data(
            self._bright_and_faint(),
            np.array([False, False, True, True, True, True]))
        # Subset max is 1.0, so the 1% threshold is 0.01 and all four survive.
        assert list(out['valid_mask']) == [True, True, True, True]

    def test_bright_subset_still_excludes_its_own_faint_lines(self):
        out = subset_population_data(self._bright_and_faint(),
                                     np.ones(6, dtype=bool))
        assert list(out['valid_mask']) == [True, True, False, False, False, False]

    def test_mask_length_tracks_the_subset(self):
        out = subset_population_data(
            self._bright_and_faint(),
            np.array([True, False, True, False, True, False]))
        assert len(out['valid_mask']) == 3

    def test_all_zero_intensity_gives_all_false(self):
        data = _data(4)
        data['intens'] = np.zeros(4)
        out = subset_population_data(data, np.ones(4, dtype=bool))
        assert not out['valid_mask'].any()

    def test_empty_selection_gives_empty_mask(self):
        out = subset_population_data(self._bright_and_faint(),
                                     np.zeros(6, dtype=bool))
        assert len(out['valid_mask']) == 0

    def test_nan_intensities_tolerated(self):
        data = _data(4)
        data['intens'] = np.array([np.nan, 5.0, np.nan, 0.01])
        out = subset_population_data(data, np.ones(4, dtype=bool))
        assert list(out['valid_mask']) == [False, True, False, False]

    def test_matches_intensity_definition(self):
        """Reproduces Intensity's own formula over the retained rows."""
        data = self._bright_and_faint()
        mask = np.array([False, True, True, True, False, False])
        out = subset_population_data(data, mask)
        kept = data['intens'][mask]
        expected = kept > (np.nanmax(kept) / 100.0)
        assert list(out['valid_mask']) == list(expected)
