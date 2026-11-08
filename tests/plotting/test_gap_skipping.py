# -*- coding: utf-8 -*-
"""Skip Data Gaps (GapMode.SKIP): gap detection / markers and the GUI views that use them."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting import FullSpectrumPlot, GapMode, ResidualSpectrumPlot, SpectrumPanel
from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView, toggle_state_display_kwargs

GAP_TAG = "_islat_gap_indicator"

# Data at 10-13 um and 17-20 um: no coverage from 13 to 17 um.
WAVE = np.concatenate([np.linspace(10.0, 13.0, 301), np.linspace(17.0, 20.0, 301)])
FLUX = 1.0 + 0.1 * np.sin(WAVE)


def _gap_artists(ax):
    return [a for a in list(ax.lines) + list(ax.texts) + list(ax.patches) + list(ax.collections)
            if hasattr(a, GAP_TAG)]


def _panel(xmin, xmax, flux=FLUX, wave=WAVE):
    ax = Figure().add_subplot(111)
    panel = SpectrumPanel(wave, flux, xmin, xmax, gap_mode=GapMode.SKIP, ax=ax)
    panel.generate_plot()
    return panel


# ======================================================================
# Gap detection
# ======================================================================
class TestEdgeGaps:
    def test_empty_stretch_at_either_edge(self):
        right = _panel(12.5, 15.0).detect_edge_gaps()
        assert right == [pytest.approx((13.0, 15.0))]
        left = _panel(15.0, 17.5).detect_edge_gaps()
        assert left == [pytest.approx((15.0, 17.0))]

    def test_none_when_data_reaches_the_edges(self):
        assert _panel(10.0, 12.5).detect_edge_gaps() == []
        assert _panel(9.99, 12.5).detect_edge_gaps() == []      # under 5 % of the panel: not a gap

    def test_plain_detect_gaps_is_unchanged(self):
        assert _panel(12.5, 15.0).detect_gaps() == []           # still only jumps / NaN runs

    def test_display_gaps_merge_an_empty_edge_with_nan_flux(self):
        flux = FLUX.copy()
        flux[(WAVE >= 10.0) & (WAVE <= 10.4)] = np.nan          # NaN run right where the data starts
        gaps = _panel(9.0, 12.5, flux=flux).detect_display_gaps()
        assert len(gaps) == 1
        assert gaps[0][0] == pytest.approx(9.0)
        assert gaps[0][1] == pytest.approx(10.41)


# ======================================================================
# Gap markers
# ======================================================================
class TestGapMarkers:
    def test_edge_gap_is_cropped_without_an_internal_marker(self):
        panel = _panel(12.5, 15.0)
        panel.draw_gap_indicators()
        lo, hi = panel.ax.get_xlim()
        assert lo == pytest.approx(12.5) and 13.0 < hi < 13.1  # cropped to the data plus a small margin
        markers = _gap_artists(panel.ax)
        assert len(markers) == 1 and markers[0] in panel.ax.texts  # just the edge label
        assert not [p for p in panel.ax.patches if hasattr(p, GAP_TAG)]

    def test_internal_markers_stay_in_place_when_y_changes(self):
        flux = FLUX.copy()
        flux[(WAVE > 11.0) & (WAVE < 11.5)] = np.nan
        panel = _panel(10.0, 12.5, flux=flux)
        ax = panel.ax
        ax.set_ylim(0.0, 2.0)
        panel.draw_gap_indicators()

        def axes_y(artist):
            if hasattr(artist, "get_xydata"):
                points = artist.get_transform().transform(artist.get_xydata())
            else:
                points = artist.get_transform().transform([artist.get_position()])
            return np.round(ax.transAxes.inverted().transform(points)[:, 1], 6).tolist()

        markers = [a for a in list(ax.lines) + list(ax.texts) if hasattr(a, GAP_TAG)]
        assert len(markers) == 3                                 # two break marks and a label
        before = [axes_y(a) for a in markers]
        ax.set_ylim(-5.0, 0.05)                                  # e.g. manual Y limits in the GUI
        assert [axes_y(a) for a in markers] == before
        assert sorted(before) == [[-0.04, 1.04], [-0.04, 1.04], [0.5]]


# ======================================================================
# Stacked plots
# ======================================================================
class TestStackedPlotsInSkipMode:
    def test_requested_figure_size_is_kept(self):
        plot = FullSpectrumPlot(WAVE, FLUX, n_panels=5, gap_mode="skip", figsize=(12.0, 10.0))
        plot.generate_plot()
        assert sorted(plot.subplots) == [0, 1, 3, 4]            # 14-16 um has no data
        assert plot.fig.get_figheight() == pytest.approx(8.0)    # new figure: 4 of 5 rows
        assert plot._figsize == (12.0, 10.0)                      # request untouched
        plot.generate_plot()
        assert plot._figsize == (12.0, 10.0)                      # and never compounded
        plot.close()

    def test_no_rows_skipped_keeps_full_height(self):
        plot = FullSpectrumPlot(WAVE, FLUX, n_panels=4, gap_mode="skip", figsize=(12.0, 10.0))
        plot.generate_plot()
        assert sorted(plot.subplots) == [0, 1, 2, 3]
        assert plot.fig.get_figheight() == pytest.approx(10.0)
        plot.close()

    def test_boundary_gaps_are_cropped(self):
        plot = FullSpectrumPlot(WAVE, FLUX, n_panels=4, gap_mode="skip")   # 12.5-15 and 15-17.5 are half empty
        plot.generate_plot()
        assert plot.subplots[1].get_xlim()[1] < 13.1
        assert plot.subplots[2].get_xlim()[0] > 16.9
        assert _gap_artists(plot.subplots[1]) and _gap_artists(plot.subplots[2])
        plot.close()

    def test_legend_goes_on_the_first_rendered_panel(self):
        plot = FullSpectrumPlot(WAVE, FLUX, n_panels=4, gap_mode="skip", xlim_range=(4.0, 20.0))
        plot.generate_plot()
        assert 0 not in plot.subplots                             # 4-8 um is empty
        assert plot._legend_axes is plot.subplots[min(plot.subplots)]
        plot.close()

    def test_inplace_update_redraws_gap_markers(self, monkeypatch):
        flux = FLUX.copy()
        flux[(WAVE > 11.0) & (WAVE < 11.5)] = np.nan
        plot = FullSpectrumPlot(WAVE, flux, n_panels=4, gap_mode="skip")
        plot.generate_plot()
        ax0 = plot.subplots[0]
        assert _gap_artists(ax0)
        monkeypatch.setattr(plot, "generate_plot", MagicMock(side_effect=AssertionError("full rebuild")))
        assert plot.update_data(WAVE, FLUX) is False              # same layout, the NaN run is gone
        plot.update_panels_inplace()
        assert not _gap_artists(ax0)
        assert _gap_artists(plot.subplots[1])                     # the cropped edge is redrawn
        assert plot.subplots[1].get_xlim()[1] < 13.1
        plot.close()

    def test_inplace_update_rebuilds_when_the_panels_with_data_change(self):
        plot = FullSpectrumPlot(WAVE, FLUX, n_panels=5, gap_mode="skip")
        plot.generate_plot()
        assert 2 not in plot.subplots
        full = np.linspace(10.0, 20.0, 600)
        plot.update_data(full, 1.0 + 0.1 * np.sin(full))
        plot.update_panels_inplace()
        assert sorted(plot.subplots) == [0, 1, 2, 3, 4]
        plot.close()

    def test_residual_plot_rerenders_instead_of_dropping_markers(self, monkeypatch):
        flux = FLUX.copy()
        flux[(WAVE > 11.0) & (WAVE < 11.5)] = np.nan
        plot = ResidualSpectrumPlot(WAVE, flux, flux * 0.99, n_panels=4, gap_mode="skip")
        plot.generate_plot()
        calls = []
        monkeypatch.setattr(plot, "generate_plot", lambda **_k: calls.append(1))
        plot.update_panels_inplace()
        assert calls == [1]
        plot.close()


# ======================================================================
# Full Spectrum view (GUI)
# ======================================================================
class _Molecules:
    _global_wavelength_range = None

    def apply_stellar_rv(self, wave):
        return np.asarray(wave)

    def get_summed_flux(self, wave, **_kw):
        return np.asarray(wave), np.zeros(len(wave))

    def get_summed_flux_resampled(self, wave, **_kw):
        return np.asarray(wave), np.zeros(len(wave))

    def get_visible_molecules(self, return_objects=False):
        return []


def _view(**toggles):
    islat = SimpleNamespace(wave_data_original=WAVE, wave_data=WAVE, flux_data=FLUX, err_data=None,
                            molecules_dict=_Molecules(), input_line_list="missing.csv", root=None)
    pm = MagicMock()
    pm.islat = islat
    pm.theme = {}
    pm.legend_toggle = True
    pm.control_bus = None
    pm.toggle_state = {"atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
                       "show_residuals": False, "data_density": False, "skip_gaps": False,
                       "observed": True, "y_autoscale": True, "y_limits": None,
                       "wavelength_markers": [], **toggles}
    view = FullSpectrumView(pm)
    view._plot = view._create_plot()
    view._plot.n_panels = 5
    view._plot._compute_panel_layout()
    view._plot.generate_plot()
    view._initialised = True
    view._canvas = MagicMock()
    return view, pm


class TestFullSpectrumView:
    def test_plot_follows_the_setting(self):
        view, _pm = _view()
        assert view._plot.gap_mode is GapMode.CONNECT
        view, _pm = _view(skip_gaps=True)
        assert view._plot.gap_mode is GapMode.SKIP
        assert 2 not in view._plot.subplots

    def test_residual_plot_follows_the_setting(self):
        view, _pm = _view(skip_gaps=True, show_residuals=True)
        assert isinstance(view._plot, ResidualSpectrumPlot)
        assert view._plot.gap_mode is GapMode.SKIP

    def test_toggle_rerenders(self):
        view, pm = _view()
        assert sorted(view._plot.subplots) == [0, 1, 2, 3, 4]
        view._set_skip_gaps(True)
        assert pm.toggle_state["skip_gaps"] is True
        assert view._plot.gap_mode is GapMode.SKIP
        assert sorted(view._plot.subplots) == [0, 1, 3, 4]
        view._set_skip_gaps(False)
        assert view._plot.gap_mode is GapMode.CONNECT
        assert sorted(view._plot.subplots) == [0, 1, 2, 3, 4]

    def test_y_settings_still_apply(self):
        view, pm = _view(skip_gaps=True, y_data_sync="panel")
        view._regenerate_layout()
        ax = view._plot.subplots[1]                               # 12-14 um, data up to 13 um
        x0, x1 = ax.get_xlim()
        sel = (WAVE >= x0) & (WAVE <= x1)
        assert ax.get_ylim() == pytest.approx((FLUX[sel].min(), FLUX[sel].max()))

    def test_exports_follow_the_setting(self):
        assert "gap_mode" not in toggle_state_display_kwargs({})
        kw = toggle_state_display_kwargs({"skip_gaps": True})
        assert kw["gap_mode"] == "skip"
        view, _pm = _view(skip_gaps=True)
        standalone = view._create_standalone_plot(WAVE, FLUX, WAVE, display_options=kw)
        assert standalone.gap_mode is GapMode.SKIP
