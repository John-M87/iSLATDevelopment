# -*- coding: utf-8 -*-
"""Main-plot Y-axis auto-scale / manual limits and observed-data visibility."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _make_controller(ax=None):
    """Build an iSLATPlot without running its Tk-heavy constructor.

    Imported lazily (like the view tests do) so collecting this module does
    not pull in the Tk-backed GUI stack before other plotting tests run.
    """
    from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
    pm = iSLATPlot.__new__(iSLATPlot)
    wave = np.linspace(5.0, 10.0, 101)
    flux = np.where(wave < 7.5, 1.0, 4.0)  # left half 1 Jy, right half 4 Jy
    pm.islat = SimpleNamespace(wave_data=wave, flux_data=flux, display_range=(5.0, 10.0))
    pm.ax1 = ax if ax is not None else plt.figure().add_subplot(111)
    pm.ax1.set_xlim(5.0, 10.0)
    pm.canvas = MagicMock()
    pm.control_bus = None
    # The controller delegates the Y policy to the active view, so use the
    # real three-panel view (it only stores the controller and reads ax1).
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
    pm.active_view = ThreePanelView(pm)
    pm.toggle_state = {
        "atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
        "show_residuals": False, "data_density": False, "current_selection": None,
        "y_autoscale": True, "y_limits": None, "observed": True,
    }
    return pm


class TestYAutoscale:
    def test_autoscale_fits_data_in_xrange(self):
        pm = _make_controller()
        pm.ax1.set_xlim(5.0, 7.0)  # only the 1 Jy half is in view
        assert pm.apply_y_limits(autoscale_now=True) is True
        lo, hi = pm.ax1.get_ylim()
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0 + 1.0 / 8)

    def test_autoscale_not_now_leaves_axis_alone(self):
        pm = _make_controller()
        pm.ax1.set_ylim(-3.0, 3.0)
        assert pm.apply_y_limits(autoscale_now=False) is False
        assert pm.ax1.get_ylim() == (-3.0, 3.0)

    def test_match_display_range_uses_policy(self):
        pm = _make_controller()
        pm.match_display_range(match_y=True)
        assert pm.ax1.get_ylim()[1] == pytest.approx(4.5)
        pm.set_y_limits(0.0, 2.0)
        pm.match_display_range(match_y=True)
        assert pm.ax1.get_ylim() == (0.0, 2.0)


class TestManualYLimits:
    def test_set_y_limits_pins_axis_and_disables_autoscale(self):
        pm = _make_controller()
        pm.set_y_limits(ymin=-1.0, ymax=10.0)
        assert pm.y_autoscale_toggle is False
        assert pm.toggle_state["y_limits"] == (-1.0, 10.0)
        assert pm.ax1.get_ylim() == (-1.0, 10.0)
        pm.canvas.draw_idle.assert_called()

    def test_single_bound_keeps_other(self):
        pm = _make_controller()
        pm.ax1.set_ylim(0.0, 5.0)
        pm.set_y_limits(ymax=7.0)
        assert pm.ax1.get_ylim() == (0.0, 7.0)
        pm.set_y_limits(ymin=1.0)
        assert pm.ax1.get_ylim() == (1.0, 7.0)

    def test_reversed_or_nonfinite_bounds_ignored(self):
        pm = _make_controller()
        pm.ax1.set_ylim(0.0, 5.0)
        pm.set_y_limits(ymin=9.0, ymax=1.0)
        assert pm.ax1.get_ylim() == (0.0, 5.0)
        assert pm.y_autoscale_toggle is True
        pm.set_y_limits(ymax=float("nan"))
        assert pm.ax1.get_ylim() == (0.0, 5.0)

    def test_manual_limits_survive_autoscale_requests(self):
        pm = _make_controller()
        pm.set_y_limits(0.0, 2.0)
        assert pm.apply_y_limits(autoscale_now=True) is True
        assert pm.ax1.get_ylim() == (0.0, 2.0)

    def test_disabling_autoscale_freezes_current_limits(self):
        pm = _make_controller()
        pm.ax1.set_ylim(0.25, 0.75)
        pm.set_y_autoscale(False)
        assert pm.toggle_state["y_limits"] == (0.25, 0.75)
        assert pm.ax1.get_ylim() == (0.25, 0.75)

    def test_reenabling_autoscale_refits(self):
        pm = _make_controller()
        pm.set_y_limits(0.0, 2.0)
        pm.set_y_autoscale(True)
        assert pm.y_autoscale_toggle is True
        assert pm.ax1.get_ylim()[1] == pytest.approx(4.5)

    def test_toggle_flips(self):
        pm = _make_controller()
        pm.toggle_y_autoscale()
        assert pm.y_autoscale_toggle is False
        pm.toggle_y_autoscale()
        assert pm.y_autoscale_toggle is True


class TestObservedToggle:
    def test_controller_flips_and_delegates(self):
        pm = _make_controller()
        pm.active_view = MagicMock()
        pm.toggle_observed_data()
        assert pm.observed_toggle is False
        pm.active_view.toggle_observed_data.assert_called_with(False)
        pm.toggle_observed_data(True)
        assert pm.observed_toggle is True
        pm.active_view.toggle_observed_data.assert_called_with(True)

    @staticmethod
    def _view_with_observed_line():
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        fig = plt.figure()
        ax1, ax2, ax3 = (fig.add_subplot(3, 1, i) for i in (1, 2, 3))
        (obs,) = ax1.plot([1, 2], [1, 2])
        obs._islat_observed = True
        (model,) = ax1.plot([1, 2], [2, 3])
        model._molecule_name = "m"
        pm = MagicMock()
        pm.ax1, pm.ax2, pm.ax3 = ax1, ax2, ax3
        pm.toggle_state = {"observed": True, "y_autoscale": True, "summed": True, "legend": True}
        pm.control_bus = None
        view = ThreePanelView(pm)
        return view, obs, model, pm

    def test_view_hides_only_observed_artists(self):
        view, obs, model, _pm = self._view_with_observed_line()
        view.toggle_observed_data(False)
        assert obs.get_visible() is False
        assert model.get_visible() is True
        view.toggle_observed_data(True)
        assert obs.get_visible() is True

    def test_sync_toggle_state_applies_observed_flag(self):
        view, obs, _model, pm = self._view_with_observed_line()
        pm.toggle_state.update({"atomic_lines": False, "saved_lines": False, "observed": False})
        view.sync_toggle_state(pm.toggle_state)
        assert obs.get_visible() is False

    def test_reapply_after_rerender_restores_observed_and_manual_y(self):
        view, obs, _model, pm = self._view_with_observed_line()
        pm.toggle_state.update({"observed": False, "y_autoscale": False, "y_limits": (0.0, 2.0)})
        view.ax1.set_ylim(-3.0, 3.0)
        view._reapply_display_state(pm.toggle_state)
        assert obs.get_visible() is False
        assert view.ax1.get_ylim() == (0.0, 2.0)

    def test_reapply_skips_y_when_autoscaling(self):
        view, _obs, _model, pm = self._view_with_observed_line()
        view.ax1.set_ylim(-3.0, 3.0)
        view._reapply_display_state(pm.toggle_state)
        assert view.ax1.get_ylim() == (-3.0, 3.0)


class TestControllerFallbackWithoutView:
    def test_policy_applies_to_ax1_when_no_view(self):
        pm = _make_controller()
        pm.active_view = None
        pm.ax1.set_xlim(5.0, 7.0)
        assert pm.apply_y_limits(autoscale_now=True) is True
        assert pm.ax1.get_ylim()[0] == pytest.approx(1.0)
        pm.set_y_limits(0.0, 3.0)
        assert pm.ax1.get_ylim() == (0.0, 3.0)
        assert pm.get_y_limits() == (0.0, 3.0)


class TestFullSpectrumViewInheritsPolicy:
    """The same toggle_state drives the full-spectrum view through the shared base."""

    @staticmethod
    def _make_view():
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        from iSLAT.Modules.Plotting.FullSpectrumPlot import FullSpectrumPlot
        wave = np.linspace(5.0, 11.0, 300)
        flux = np.where(wave < 8.0, 1.0, 4.0)
        pm = MagicMock()
        pm.islat.wave_data_original = wave
        pm.islat.flux_data = flux
        pm.islat.wave_data = wave
        pm.toggle_state = {"atomic_lines": False, "saved_lines": False, "summed": True,
                           "legend": True, "observed": True, "y_autoscale": True, "y_limits": None}
        pm.control_bus = None
        view = FullSpectrumView(pm)
        plot = FullSpectrumPlot(wave_data=wave, flux_data=flux, n_panels=3)
        plot.generate_plot()
        view._plot = plot
        view._initialised = True
        view._canvas = MagicMock()
        return view, plot, pm

    def test_y_axes_are_the_spectrum_panels(self):
        view, plot, _pm = self._make_view()
        assert list(view._iter_y_axes()) == list(plot.subplots.values())
        assert len(list(view._iter_observed_axes())) == 3

    def test_manual_limits_pin_every_panel(self):
        view, plot, _pm = self._make_view()
        assert view.apply_y_limits({"y_autoscale": False, "y_limits": (0.0, 9.0)}) is True
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())
        assert view.get_y_limits() == (0.0, 9.0)

    def test_autoscale_restores_layout_limits(self):
        view, plot, _pm = self._make_view()
        expected = plot._compute_panel_ylims(ylim_fn=plot._spectrum_ylim_fn)
        view.apply_y_limits({"y_autoscale": False, "y_limits": (0.0, 9.0)})
        assert view.apply_y_limits({"y_autoscale": True, "y_limits": (0.0, 9.0)}, autoscale_now=True) is True
        for idx, ax in plot.subplots.items():
            assert ax.get_ylim() == pytest.approx(expected[idx])

    def test_autoscale_not_now_leaves_panels_alone(self):
        view, plot, _pm = self._make_view()
        view.apply_y_limits({"y_autoscale": False, "y_limits": (0.0, 9.0)})
        assert view.apply_y_limits({"y_autoscale": True, "y_limits": None}, autoscale_now=False) is False
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())

    def test_observed_toggle_hides_every_panel(self):
        view, plot, _pm = self._make_view()
        observed = [l for ax in plot.subplots.values() for l in ax.lines if hasattr(l, "_islat_observed")]
        assert len(observed) == 3
        view.toggle_observed_data(False)
        assert all(not l.get_visible() for l in observed)
        view._canvas.draw_idle.assert_called()
        view.toggle_observed_data(True)
        assert all(l.get_visible() for l in observed)

    def test_sync_toggle_state_reapplies_both(self):
        view, plot, pm = self._make_view()
        pm.toggle_state.update({"observed": False, "y_autoscale": False, "y_limits": (0.0, 9.0)})
        view.sync_toggle_state(pm.toggle_state)
        observed = [l for ax in plot.subplots.values() for l in ax.lines if hasattr(l, "_islat_observed")]
        assert all(not l.get_visible() for l in observed)
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())

    def test_controller_delegates_to_full_spectrum_view(self):
        view, plot, pm = self._make_view()
        ctrl = _make_controller()
        ctrl.active_view = view
        ctrl.set_y_limits(0.0, 9.0)
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())
        assert ctrl.get_y_limits() == (0.0, 9.0)
        view._canvas.draw_idle.assert_called()
        ctrl.set_y_autoscale(True)
        expected = plot._compute_panel_ylims(ylim_fn=plot._spectrum_ylim_fn)
        for idx, ax in plot.subplots.items():
            assert ax.get_ylim() == pytest.approx(expected[idx])


class TestViewsWithoutSpectrumAreNoOps:
    def test_population_diagram_view_ignores_policy(self):
        from iSLAT.Modules.Plotting.PopulationDiagramView import PopulationDiagramView
        pm = MagicMock()
        pm.toggle_state = {}
        view = PopulationDiagramView(pm)
        assert view.apply_y_limits({"y_autoscale": False, "y_limits": (0.0, 1.0)}) is False
        assert view.get_y_limits() is None
        view.toggle_observed_data(False)  # must not raise or draw
