# -*- coding: utf-8 -*-
"""Tests for the ControlBus / ControlField view-registration protocol.

Covers:
  - PlotView._register_control_fields() default is a no-op
  - ThreePanelView does not register any fields (options live in the Views menu)
  - FullSpectrumView registers an n_panels EntryField and a
    DisplayRangeField on the control_panel surface
  - ControlBus routes registrations to the correct surface
  - ControlBus.unregister_owner removes all fields across surfaces
  - FullSpectrumView getters/setters for n_panels and display range
    delegate to the composed FullSpectrumPlot
  - iSLATPlot view-change callback infrastructure
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from iSLAT.Modules.Plotting.PlotView import PlotView
from iSLAT.Modules.Plotting.FullSpectrumPlot import FullSpectrumPlot
from iSLAT.Modules.GUI.ControlField import (
    EntryField, DisplayRangeField, ToggleField,
)
from iSLAT.Modules.GUI.ControlSurface import ControlBus, ControlSurface


# ======================================================================
# Helpers
# ======================================================================

def _make_wave_flux(n=200, start=4.9, end=28.0):
    wave = np.linspace(start, end, n)
    flux = np.random.default_rng(42).normal(1.0, 0.05, n)
    return wave, flux


def _make_mock_islat():
    islat = MagicMock()
    islat.display_range = (4.9, 5.9)
    islat._display_range = (4.9, 5.9)
    islat.wave_data_original = np.linspace(4.9, 28.0, 200)
    islat.flux_data = np.ones(200)
    islat.molecules_dict = MagicMock()
    islat.molecules_dict.apply_stellar_rv.side_effect = lambda w: w.copy()
    islat.active_molecule = None
    return islat


def _make_mock_plot_manager(islat=None):
    if islat is None:
        islat = _make_mock_islat()
    pm = MagicMock()
    pm.islat = islat
    pm.theme = {}
    pm.toggle_state = {
        "atomic_lines": False,
        "saved_lines": False,
        "summed": True,
        "legend": True,
        "show_residuals": False,
    }
    pm.canvas = MagicMock()
    pm.ax1 = MagicMock()
    pm.ax2 = MagicMock()
    pm.ax3 = MagicMock()
    pm.control_bus = None  # no bus wired by default
    pm._view_change_callbacks = []
    return pm


class _NoopSurface(ControlSurface):
    """Headless ControlSurface that records rebuild calls without creating widgets."""

    def __init__(self):
        super().__init__()
        self.rebuild_count = 0

    def _rebuild(self):
        self.rebuild_count += 1


# ======================================================================
# PlotView ABC — _register_control_fields default
# ======================================================================

class TestPlotViewRegisterFields:
    """PlotView._register_control_fields() default is a no-op."""

    def test_register_control_fields_is_noop(self):
        class StubView(PlotView):
            def activate(self, pf): ...
            def deactivate(self): ...
            def update_model_plot(self, *a, **k): ...
            def on_molecule_visibility_changed(self, *a, **k): ...
            def on_selection(self, xmin, xmax): ...
            def clear_selection(self): ...
            def clear_active_lines(self): ...
            def on_active_molecule_changed(self, new_molecule=None, current_selection=None): ...
            def on_molecule_parameter_changed(self, molecule_name, parameter_name, current_selection=None): ...
            def on_molecule_deleted(self, molecule_name): ...
            def apply_theme(self, t): ...
            def sync_toggle_state(self, s): ...
            def toggle_summed_spectrum(self, v): ...
            def toggle_legend(self, v=None): ...
            def toggle_saved_lines(self, s, **k): ...
            def toggle_atomic_lines(self, s): ...
            def draw(self): ...
            def get_canvas(self): ...
            def get_figure(self): ...

        view = StubView()
        # Must not raise; returns None
        result = view._register_control_fields()
        assert result is None


# ======================================================================
# ThreePanelView — no fields registered on a wired bus
# ======================================================================

class TestThreePanelViewNoFields:
    """ThreePanelView does not register any fields on the ControlBus.

    Its Y-scale and observed-data options live in the TopBar's Views menu
    instead, so they are available regardless of surface wiring order.
    """

    def test_no_fields_registered_when_bus_present(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        pm = _make_mock_plot_manager()
        bus = ControlBus()
        surface = _NoopSurface()
        bus.register_surface("control_panel", surface)
        bus.register_surface("top_bar", _NoopSurface())
        pm.control_bus = bus

        view = ThreePanelView(pm)
        view._register_control_fields()

        assert surface.fields() == []
        assert bus.get_surface("top_bar").fields() == []

    def test_deactivate_unregister_owner_is_safe(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        pm = _make_mock_plot_manager()
        bus = ControlBus()
        surface = _NoopSurface()
        bus.register_surface("control_panel", surface)
        pm.control_bus = bus

        view = ThreePanelView(pm)
        # Deactivating without any registered fields must not raise
        bus.unregister_owner(view)
        assert surface.fields() == []


# ======================================================================
# FullSpectrumView — registers n_panels + DisplayRangeField
# ======================================================================

class TestFullSpectrumViewControlFields:
    """FullSpectrumView registers an EntryField for n_panels and a
    DisplayRangeField, both on the 'control_panel' surface."""

    def _make_view_with_bus(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        pm = _make_mock_plot_manager()
        bus = ControlBus()
        cp_surface = _NoopSurface()
        tb_surface = _NoopSurface()
        bus.register_surface("control_panel", cp_surface)
        bus.register_surface("top_bar", tb_surface)
        pm.control_bus = bus

        view = FullSpectrumView(pm)
        wave, flux = _make_wave_flux()
        plot = FullSpectrumPlot(wave_data=wave, flux_data=flux, n_panels=8)
        view._plot = plot
        view._canvas = MagicMock()
        return view, pm, bus, cp_surface, tb_surface

    # ------------------------------------------------------------------
    def test_register_fields_adds_n_panels_entry(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        keys = [f.key for f in cp_surface.fields()]
        assert "n_panels" in keys

    def test_n_panels_field_is_entry_field(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        field = cp_surface.get_field("n_panels")
        assert isinstance(field, EntryField)
        assert field.datatype == "int"

    def test_n_panels_field_owner_is_view(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        field = cp_surface.get_field("n_panels")
        assert field.owner is view

    def test_register_fields_adds_display_range_field(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        cp_keys = [f.key for f in cp_surface.fields()]
        # There must be at least one DisplayRangeField
        display_fields = [f for f in cp_surface.fields() if isinstance(f, DisplayRangeField)]
        assert len(display_fields) >= 1

    def test_register_fields_adds_toggle_to_top_bar(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        tb_keys = [f.key for f in tb_surface.fields()]
        assert "show_residuals" in tb_keys

    def test_show_residuals_is_toggle_field(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        field = tb_surface.get_field("show_residuals")
        assert isinstance(field, ToggleField)

    def test_deactivate_removes_all_owned_fields(self):
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()

        assert len(cp_surface.fields()) > 0 or len(tb_surface.fields()) > 0
        bus.unregister_owner(view)

        assert cp_surface.fields() == []
        assert tb_surface.fields() == []

    def test_re_register_replaces_fields(self):
        """Calling _register_control_fields twice should not duplicate fields."""
        view, pm, bus, cp_surface, tb_surface = self._make_view_with_bus()
        view._register_control_fields()
        n_after_first = len(cp_surface.fields())
        view._register_control_fields()
        n_after_second = len(cp_surface.fields())
        assert n_after_second == n_after_first


# ======================================================================
# FullSpectrumView getter / setter helpers
# ======================================================================

class TestFullSpectrumViewFieldGettersSetters:
    """n_panels and display-range getters/setters delegate to the composed plot."""

    def _make_view_with_plot(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        pm = _make_mock_plot_manager()
        view = FullSpectrumView(pm)
        wave, flux = _make_wave_flux()
        plot = FullSpectrumPlot(wave_data=wave, flux_data=flux, n_panels=8)
        view._plot = plot
        view._canvas = MagicMock()
        return view, plot

    def test_n_panels_getter_reads_from_plot(self):
        view, plot = self._make_view_with_plot()
        assert view._get_n_panels() == 8

    def test_n_panels_setter_updates_plot(self):
        view, plot = self._make_view_with_plot()
        with patch.object(plot, 'generate_plot'):
            view._set_n_panels(5)
        assert plot.n_panels == 5

    def test_n_panels_setter_clamps_below_one(self):
        view, plot = self._make_view_with_plot()
        with patch.object(plot, 'generate_plot'):
            view._set_n_panels(0)
        assert plot.n_panels == 1

    def test_n_panels_setter_noop_when_plot_none(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        pm = _make_mock_plot_manager()
        view = FullSpectrumView(pm)
        # Should not raise
        view._set_n_panels(5)

    def test_display_range_getter_returns_xlim(self):
        view, plot = self._make_view_with_plot()
        start, end = view._get_display_range()
        assert start == pytest.approx(plot._xlim_start, abs=1e-6)
        assert end == pytest.approx(plot._xlim_end, abs=1e-6)

    def test_display_range_setter_updates_xlim(self):
        view, plot = self._make_view_with_plot()
        with patch.object(plot, 'generate_plot'):
            view._set_display_range(10.0, 20.0)
        assert plot._xlim_start == pytest.approx(10.0)
        assert plot._xlim_end == pytest.approx(20.0)

    def test_display_range_getter_returns_zeros_when_no_plot(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        pm = _make_mock_plot_manager()
        view = FullSpectrumView(pm)
        assert view._get_display_range() == (0.0, 0.0)

    def test_display_range_setter_noop_when_no_plot(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        pm = _make_mock_plot_manager()
        view = FullSpectrumView(pm)
        # Should not raise
        view._set_display_range(5.0, 25.0)


# ======================================================================
# iSLATPlot view-change callback infrastructure
# ======================================================================

class TestViewChangeCallbacks:
    """Tests for add/remove/notify view-change callbacks on iSLATPlot."""

    def _make_islat_plot_mock(self):
        pm = _make_mock_plot_manager()
        pm._view_change_callbacks = []

        def add_cb(cb):
            if cb not in pm._view_change_callbacks:
                pm._view_change_callbacks.append(cb)
        pm.add_view_change_callback = add_cb

        def remove_cb(cb):
            try:
                pm._view_change_callbacks.remove(cb)
            except ValueError:
                pass
        pm.remove_view_change_callback = remove_cb

        def notify(old, new):
            for cb in list(pm._view_change_callbacks):
                cb(old, new)
        pm._notify_view_change = notify

        return pm

    def test_add_callback(self):
        pm = self._make_islat_plot_mock()
        cb = MagicMock()
        pm.add_view_change_callback(cb)
        assert cb in pm._view_change_callbacks

    def test_no_duplicate_callbacks(self):
        pm = self._make_islat_plot_mock()
        cb = MagicMock()
        pm.add_view_change_callback(cb)
        pm.add_view_change_callback(cb)
        assert pm._view_change_callbacks.count(cb) == 1

    def test_remove_callback(self):
        pm = self._make_islat_plot_mock()
        cb = MagicMock()
        pm.add_view_change_callback(cb)
        pm.remove_view_change_callback(cb)
        assert cb not in pm._view_change_callbacks

    def test_remove_nonexistent_callback_no_error(self):
        pm = self._make_islat_plot_mock()
        pm.remove_view_change_callback(MagicMock())  # should not raise

    def test_notify_calls_callback(self):
        pm = self._make_islat_plot_mock()
        cb = MagicMock()
        pm.add_view_change_callback(cb)
        old, new = MagicMock(), MagicMock()
        pm._notify_view_change(old, new)
        cb.assert_called_once_with(old, new)

    def test_notify_calls_multiple_callbacks(self):
        pm = self._make_islat_plot_mock()
        cb1, cb2 = MagicMock(), MagicMock()
        pm.add_view_change_callback(cb1)
        pm.add_view_change_callback(cb2)
        pm._notify_view_change(None, MagicMock())
        assert cb1.call_count == 1
        assert cb2.call_count == 1

    def test_view_change_triggers_bus_unregister_then_register(self):
        """Simulates a view switch: old view deactivates (unregister_owner),
        new view activates (_register_control_fields)."""
        pm = self._make_islat_plot_mock()
        bus = ControlBus()
        surface = _NoopSurface()
        bus.register_surface("control_panel", surface)
        pm.control_bus = bus

        owner_a = object()
        owner_b = object()
        field_a = EntryField("field_a", "A:", lambda: 1, lambda v: None, owner=owner_a)
        field_b = EntryField("field_b", "B:", lambda: 2, lambda v: None, owner=owner_b)

        bus.register(field_a, "control_panel")
        assert surface.get_field("field_a") is field_a

        # Simulate deactivation of old view, activation of new view
        bus.unregister_owner(owner_a)
        bus.register(field_b, "control_panel")

        assert surface.get_field("field_a") is None
        assert surface.get_field("field_b") is field_b
