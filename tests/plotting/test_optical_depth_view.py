# -*- coding: utf-8 -*-
"""Tests for OpticalDepthView, the "Optical Depth" entry in the Views menu.

The view is rendered headlessly (Agg, no Tk canvas) from real test molecules
so the τ profiles, stems and total-τ line are the ones the GUI would draw.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import MagicMock

from iSLAT.Modules.Plotting import (
    FullSpectrumView,
    OpticalDepthSpectrumPlot,
    OpticalDepthView,
)
from iSLAT.Modules.Plotting.OpticalDepthPanel import TAU_STEM_TAG, TOTAL_TAU_TAG
from iSLAT.Modules.Plotting.OpticalDepthView import TAU_OPTION_DEFAULTS
from iSLAT.Modules.Plotting.SavedLineLabels import LABEL_VIEWS
from iSLAT.Modules.GUI.ControlField import DisplayRangeField, EntryField, ToggleField
from iSLAT.Modules.GUI.ControlSurface import ControlBus, ControlSurface

from tests.plotting import make_line_list
from tests.plotting.test_plot_views import _make_test_molecule


# ======================================================================
# Helpers
# ======================================================================

class _MolDict(dict):
    """The slice of the MoleculeDict API the view and plot use."""

    _global_wavelength_range = None

    def get_visible_molecules(self, return_objects=False):
        visible = [m for m in self.values() if m.is_visible]
        return visible if return_objects else [m.name for m in visible]

    def get_matched_sampling_wavelengths(self, wave):
        return False, wave

    def apply_stellar_rv(self, wave):
        return np.array(wave, copy=True)

    def get_summed_flux(self, wave, visible_only=True, extend_to_global_range=False):
        lam, total = None, None
        for mol in self.get_visible_molecules(return_objects=True):
            m_lam, m_flux = mol.get_flux(return_wavelengths=True)
            lam, total = (m_lam, m_flux.copy()) if total is None else (lam, total + m_flux)
        if total is None:
            return np.array([]), np.array([])
        return lam, total


class _NoopSurface(ControlSurface):
    def _rebuild(self):
        pass


def _make_view(n_molecules=2, **toggles):
    wave = np.linspace(10.0, 20.0, 2000)
    flux = np.full_like(wave, 0.05)
    mols = _MolDict()
    for i, name in enumerate(("H2O", "CO2")[:n_molecules]):
        mols[name] = _make_test_molecule(name=name, lam_start=12.0 + 0.3 * i)

    islat = MagicMock()
    islat.wave_data_original = wave
    islat.flux_data = flux
    islat.err_data = np.full_like(wave, 0.005)
    islat.molecules_dict = mols
    islat.input_line_list = "nonexistent.csv"
    islat.root = None

    pm = MagicMock()
    pm.islat = islat
    pm.theme = {}
    pm.legend_toggle = True
    pm.control_bus = None
    pm.active_view = None
    pm.toggle_state = {
        "atomic_lines": False,
        "saved_lines": False,
        "summed": True,
        "legend": True,
        "data_density": False,
        "y_autoscale": True,
        "y_limits": None,
        "observed": True,
        "wavelength_markers": [],
        **TAU_OPTION_DEFAULTS,
        **toggles,
    }
    return OpticalDepthView(pm), pm, islat, mols


def _render(view):
    """What activate() does, minus the Tk canvas."""
    view._plot = view._create_plot()
    view._plot.generate_plot()
    view._install_span_selectors()
    view._initialised = True
    view._needs_refresh = False
    view.sync_toggle_state(view._pm.toggle_state)
    return view._plot


def _total_tau_lines(view):
    return [ln for ax in view._iter_all_axes() for ln in ax.lines if hasattr(ln, TOTAL_TAU_TAG)]


@pytest.fixture
def rendered():
    view, pm, islat, mols = _make_view()
    _render(view)
    yield view, pm, islat, mols
    view.destroy()


# ======================================================================
# Registration
# ======================================================================

class TestRegistration:
    def test_is_a_stacked_view_named_optical_depth(self):
        assert issubclass(OpticalDepthView, FullSpectrumView)
        assert OpticalDepthView.VIEW_NAME == "Optical Depth"

    def test_saved_line_labels_have_a_per_view_toggle(self):
        assert OpticalDepthView.VIEW_NAME in LABEL_VIEWS


# ======================================================================
# Rendering
# ======================================================================

class TestRendering:
    def test_create_plot_follows_toggle_options(self):
        view, *_ = _make_view(tau_log_scale=True, tau_display_mode="stem", tau_stacked_fill=False)
        plot = view._create_plot()
        assert isinstance(plot, OpticalDepthSpectrumPlot)
        assert (plot.log_scale, plot.display_mode, plot.stacked_fill) == (True, "stem", False)
        plot.close()

    def test_renders_tau_panels(self, rendered):
        view, *_ = rendered
        assert len(view.subplots) == view._plot.n_panels
        assert view.fig._supylabel.get_text() == "Optical Depth (τ)"
        assert any(ax.collections for ax in view.subplots.values())  # stacked τ fills

    def test_total_model_toggle_shows_and_hides_total_tau(self, rendered):
        view, *_ = rendered
        assert _total_tau_lines(view)
        view.toggle_summed_spectrum(False)
        assert not any(ln.get_visible() for ln in _total_tau_lines(view))
        view.toggle_summed_spectrum(True)
        assert all(ln.get_visible() for ln in _total_tau_lines(view))

    def test_summed_off_survives_rerender(self, rendered):
        view, pm, *_ = rendered
        pm.toggle_state["summed"] = False
        view._set_log_scale(True)
        assert _total_tau_lines(view)
        assert not any(ln.get_visible() for ln in _total_tau_lines(view))

    def test_flux_y_policy_leaves_tau_limits_alone(self, rendered):
        view, pm, *_ = rendered
        before = [ax.get_ylim() for ax in view.subplots.values()]
        ts = dict(pm.toggle_state, y_autoscale=False, y_limits=(0.0, 5000.0))
        assert view.apply_y_limits(ts) is False
        assert view.apply_y_limits(pm.toggle_state, autoscale_now=True) is False
        assert view.get_y_limits() is None
        assert [ax.get_ylim() for ax in view.subplots.values()] == before

    def test_saved_lines_overlay(self, tmp_path):
        csv = tmp_path / "lines.csv"
        make_line_list([12.5, 15.5]).to_csv(csv, index=False)  # inside panels, not on an edge
        view, pm, islat, _ = _make_view(saved_lines=True)
        islat.input_line_list = str(csv)
        _render(view)
        tagged = [a for ax in view.subplots.values()
                  for a in list(ax.lines) + list(ax.collections) + list(ax.texts)
                  if hasattr(a, "_islat_saved_line")]
        assert tagged
        view.destroy()

    def test_wavelength_markers_on_every_panel(self, rendered):
        view, pm, *_ = rendered
        pm.toggle_state["wavelength_markers"] = [{"start": 12.0, "end": 12.0, "color": "red"}]
        view.refresh_wavelength_markers()
        for ax in view.subplots.values():
            assert any(hasattr(a, "_islat_wavelength_marker") for a in ax.collections)

    def test_apply_theme_recolors_total_tau(self, rendered):
        view, *_ = rendered
        view.apply_theme({"foreground": "#123456", "background": "#000000"})
        assert {ln.get_color() for ln in _total_tau_lines(view)} == {"#123456"}


# ======================================================================
# τ display options
# ======================================================================

class TestDisplayOptions:
    def test_log_scale_toggle(self, rendered):
        view, pm, *_ = rendered
        view._set_log_scale(True)
        assert pm.toggle_state["tau_log_scale"] is True
        assert all(ax.get_yscale() == "log" for ax in view.subplots.values())

    def test_stem_mode_fits_limits_to_stems(self, rendered):
        view, pm, *_ = rendered
        view._set_stem_mode(True)
        assert pm.toggle_state["tau_display_mode"] == "stem"
        stem_panels = 0
        for cell in view._plot.panels.values():
            panel = cell[0]
            peak = panel.stem_tau_max()
            if peak > 0:
                stem_panels += 1
                assert panel.ax.get_ylim()[1] >= peak
                assert panel.ax.get_ylim()[1] < peak * 2  # not scaled to the convolved profile
        assert stem_panels > 0
        assert any(hasattr(c, TAU_STEM_TAG) for ax in view.subplots.values() for c in ax.collections)

    def test_unstacked_draws_one_line_per_molecule(self, rendered):
        view, pm, *_ = rendered
        view._set_stacked_fill(False)
        assert pm.toggle_state["tau_stacked_fill"] is False
        labels = {ln.get_label() for ax in view.subplots.values() for ln in ax.lines
                  if not hasattr(ln, TOTAL_TAU_TAG)}
        assert {"H2O", "CO2"} <= labels

    def test_unchanged_option_does_not_rerender(self, rendered):
        view, *_ = rendered
        view._regenerate_layout = MagicMock()
        view._set_log_scale(False)
        view._regenerate_layout.assert_not_called()

    def test_options_set_before_first_render_apply_on_create(self):
        view, pm, *_ = _make_view()
        view._set_stem_mode(True)  # no plot yet: just stored
        assert pm.toggle_state["tau_display_mode"] == "stem"
        plot = view._create_plot()
        assert plot.display_mode == "stem"
        plot.close()


# ======================================================================
# Molecule callbacks / data refresh
# ======================================================================

class TestMoleculeCallbacks:
    def test_hiding_a_molecule_rerenders(self, rendered):
        view, pm, islat, mols = rendered
        mols["CO2"].is_visible = False
        view.on_molecule_visibility_changed("CO2", False, mols, islat.wave_data_original)
        assert not _total_tau_lines(view)  # one molecule left: no total line
        assert view._plot._mol_tau_cache and len(view._plot._mol_tau_cache) == 1

    def test_deleting_a_molecule_rerenders(self, rendered):
        view, pm, islat, mols = rendered
        del mols["CO2"]
        view.on_molecule_deleted("CO2")
        assert [entry[4] for entry in view._plot._mol_tau_cache] == ["H2O"]

    def test_parameter_change_rerenders_only_visible_molecules(self, rendered):
        view, pm, islat, mols = rendered
        view._regenerate_layout = MagicMock()
        mols["CO2"].is_visible = False
        view.on_molecule_parameter_changed("CO2", "temp")
        view._regenerate_layout.assert_not_called()
        view.on_molecule_parameter_changed("H2O", "is_visible")
        view._regenerate_layout.assert_not_called()
        view.on_molecule_parameter_changed("H2O", "temp")
        view._regenerate_layout.assert_called_once()

    def test_parameter_change_before_first_render_marks_stale(self):
        view, *_ = _make_view()
        view._needs_refresh = False
        view.on_molecule_parameter_changed("H2O", "temp")
        assert view._needs_refresh is True

    def test_model_update_refreshes_data_and_keeps_layout(self, rendered):
        view, pm, islat, mols = rendered
        plot = view._plot
        view._set_n_panels(4)
        view._set_display_range(11.0, 15.0)
        islat.flux_data = np.full_like(islat.flux_data, 0.2)
        view.update_model_plot()
        assert view._plot is plot
        assert plot.n_panels == 4 and len(view.subplots) == 4
        assert (plot._xlim_start, plot._xlim_end) == (11.0, 15.0)
        assert np.allclose(plot.flux_data, 0.2)


# ======================================================================
# Controls, selection and export
# ======================================================================

class TestControlsAndInteraction:
    def test_registers_layout_entries_and_tau_toggles(self, rendered):
        view, pm, *_ = rendered
        bus = ControlBus()
        cp, tb, vm = _NoopSurface(), _NoopSurface(), _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)
        bus.register_surface("views_menu", vm)
        pm.control_bus = bus

        view._register_control_fields()
        assert isinstance(cp.get_field("n_panels"), EntryField)
        assert any(isinstance(f, DisplayRangeField) for f in cp.fields())
        # The tau toggles go to the Views menu's view-controls section, not the top bar.
        assert tb.fields() == []
        vm_keys = [f.key for f in vm.fields()]
        assert vm_keys == ["tau_log_scale", "tau_stem_mode", "tau_fraction_mode", "tau_stacked_fill", "tau_show_flux",
                           "data_density"]
        assert all(isinstance(f, ToggleField) for f in vm.fields())

        log_field = vm.get_field("tau_log_scale")
        log_field.setter(not log_field.getter())
        assert pm.toggle_state["tau_log_scale"] is True

        view.deactivate()
        assert cp.fields() == [] and vm.fields() == []

    def test_span_select_opens_three_panel(self, rendered):
        view, pm, islat, _ = rendered
        pm.active_view = view
        view._on_span_select(12.0, 12.5, 0)
        pm.switch_view.assert_called_once_with("Three Panel")
        assert pm._pre_threepanel_view_name == "Optical Depth"
        islat.GUI.plot.onselect.assert_called_once_with(12.0, 12.5)

    def test_tiny_span_is_ignored(self, rendered):
        view, pm, *_ = rendered
        pm.active_view = view
        view._on_span_select(12.0, 12.0001, 0)
        pm.switch_view.assert_not_called()

    def test_save_figure_matches_live_layout(self, rendered, tmp_path, monkeypatch):
        view, pm, islat, _ = rendered
        view._set_n_panels(3)
        pm.toggle_state.update(summed=False, legend=False,
                               wavelength_markers=[{"start": 13.0, "end": 14.0, "color": "red"}])
        created = []
        original = view._create_standalone_plot

        def _capture(*args, **kwargs):
            plot = original(*args, **kwargs)
            created.append(plot)
            return plot

        monkeypatch.setattr(view, "_create_standalone_plot", _capture)
        out = tmp_path / "tau.png"
        assert view.save_figure(str(out), file_format="png", dpi=40) == str(out)
        assert out.exists() and out.stat().st_size > 0
        assert created[0].n_panels == 3
        assert created[0].fig is None  # closed after saving
        islat.GUI.data_field.insert_text.assert_called_once()


# ======================================================================
# Flux sub-panels below each τ panel
# ======================================================================

def _artists(ax, tag):
    return [a for a in list(ax.lines) + list(ax.collections) if hasattr(a, tag)]


@pytest.fixture
def with_flux():
    view, pm, islat, mols = _make_view(tau_show_flux=True)
    _render(view)
    yield view, pm, islat, mols
    view.destroy()


class TestFluxPanels:
    def test_toggle_adds_and_removes_flux_sub_panels(self, rendered):
        view, pm, *_ = rendered
        n = view._plot.n_panels
        view._set_show_flux(True)
        assert pm.toggle_state["tau_show_flux"] is True
        flux_axes = view._plot.get_flux_axes()
        assert len(flux_axes) == n and len(list(view._iter_all_axes())) == 2 * n
        # The view's primary axes stay the τ panels (legend, overlays).
        assert view.subplots == view._plot.get_tau_axes()
        for idx, ax in flux_axes.items():
            assert _artists(ax, "_islat_observed")
            assert ax.get_shared_x_axes().joined(ax, view.subplots[idx])
        view._set_show_flux(False)
        assert view._plot.get_flux_axes() == {}
        assert len(list(view._iter_all_axes())) == n

    def test_manual_y_limits_apply_to_flux_panels_only(self, with_flux):
        view, pm, *_ = with_flux
        tau_before = [ax.get_ylim() for ax in view.subplots.values()]
        ts = dict(pm.toggle_state, y_autoscale=False, y_limits=(-0.01, 0.5))
        assert view.apply_y_limits(ts) is True
        assert all(ax.get_ylim() == (-0.01, 0.5) for ax in view._plot.get_flux_axes().values())
        assert [ax.get_ylim() for ax in view.subplots.values()] == tau_before
        assert view.get_y_limits() == (-0.01, 0.5)

    def test_autoscale_restores_per_panel_flux_limits(self, with_flux):
        view, pm, *_ = with_flux
        view.apply_y_limits(dict(pm.toggle_state, y_autoscale=False, y_limits=(0.0, 9.0)))
        assert view.apply_y_limits(pm.toggle_state, autoscale_now=True) is True
        expected = view._plot.compute_flux_ylims()
        for idx, ax in view._plot.get_flux_axes().items():
            assert ax.get_ylim() == pytest.approx(expected[idx])

    def test_skip_gaps_reaches_live_and_exported_plots(self, with_flux):
        from iSLAT.Modules.Plotting import GapMode
        view, pm, islat, _mols = with_flux
        view._set_skip_gaps(True)
        assert pm.toggle_state["skip_gaps"] is True
        assert view._plot.gap_mode is GapMode.SKIP
        assert view._plot.get_flux_axes()                     # still rendered with flux panels
        assert view._create_plot().gap_mode is GapMode.SKIP
        wave = islat.wave_data_original
        assert view._create_standalone_plot(wave, islat.flux_data, wave).gap_mode is GapMode.SKIP
        view._set_skip_gaps(False)
        assert view._plot.gap_mode is GapMode.CONNECT

    def test_autoscale_to_model(self, with_flux):
        view, pm, *_ = with_flux
        pm.toggle_state["y_autoscale_source"] = "model"
        assert view.apply_y_limits(pm.toggle_state, autoscale_now=True) is True

    def test_data_sync_applies_to_flux_panels_only(self, with_flux):
        view, pm, *_ = with_flux
        tau_before = [ax.get_ylim() for ax in view.subplots.values()]
        pm.toggle_state["y_data_sync"] = "panel"
        assert view.apply_y_limits(pm.toggle_state, autoscale_now=True) is True
        # The observed flux is a flat 0.05 Jy, widened by 5 % on each side.
        for ax in view._plot.get_flux_axes().values():
            assert ax.get_ylim() == pytest.approx((0.0475, 0.0525))
        assert [ax.get_ylim() for ax in view.subplots.values()] == tau_before
        assert view.y_axis_count() == len(view._plot.get_flux_axes())

    def test_export_helper_applies_data_sync(self, with_flux):
        view, pm, *_ = with_flux
        flux_axes = list(view._plot.get_flux_axes().values())
        for ax in flux_axes:
            ax.set_ylim(-1.0, 1.0)
        view._apply_flux_display_state(view._plot, dict(pm.toggle_state, y_data_sync="global"))
        assert all(ax.get_ylim() == pytest.approx((0.0475, 0.0525)) for ax in flux_axes)
        view._apply_flux_display_state(
            view._plot, dict(pm.toggle_state, y_data_sync="global", y_autoscale=False, y_limits=(0.0, 2.0)))
        assert all(ax.get_ylim() == (0.0, 2.0) for ax in flux_axes)

    def test_observed_toggle_hides_flux_data(self, with_flux):
        view, *_ = with_flux
        view.toggle_observed_data(False)
        for ax in view._plot.get_flux_axes().values():
            assert not any(a.get_visible() for a in _artists(ax, "_islat_observed"))
        view.toggle_observed_data(True)
        for ax in view._plot.get_flux_axes().values():
            assert all(a.get_visible() for a in _artists(ax, "_islat_observed"))

    def test_total_model_toggles_summed_fill_and_total_tau(self, with_flux):
        view, *_ = with_flux
        fills = [a for ax in view._plot.get_flux_axes().values() for a in _artists(ax, "_islat_summed")]
        assert fills and _total_tau_lines(view)
        view.toggle_summed_spectrum(False)
        assert not any(a.get_visible() for a in fills + _total_tau_lines(view))
        view.toggle_summed_spectrum(True)
        assert all(a.get_visible() for a in fills + _total_tau_lines(view))

    def test_settings_survive_rerender(self, with_flux):
        view, pm, *_ = with_flux
        pm.toggle_state.update(observed=False, y_autoscale=False, y_limits=(0.0, 0.25))
        view._set_log_scale(True)  # full re-render
        for ax in view._plot.get_flux_axes().values():
            assert ax.get_ylim() == (0.0, 0.25)
            assert not any(a.get_visible() for a in _artists(ax, "_islat_observed"))

    def test_span_selectors_and_context_lookup_cover_flux_axes(self, with_flux):
        view, *_ = with_flux
        n = view._plot.n_panels
        assert len(view.span_selectors) == 2 * n
        for idx, ax in view._plot.get_flux_axes().items():
            assert view._panel_index_of(ax) == idx
            assert view._panel_index_of(view.subplots[idx]) == idx
        assert view._panel_index_of(None) is None

    def test_markers_and_saved_lines_reach_flux_panels(self, tmp_path):
        csv = tmp_path / "lines.csv"
        make_line_list([12.5]).to_csv(csv, index=False)
        view, pm, islat, _ = _make_view(tau_show_flux=True, saved_lines=True,
                                        wavelength_markers=[{"start": 13.5, "end": 13.5, "color": "red"}])
        islat.input_line_list = str(csv)
        _render(view)
        flux_axes = list(view._plot.get_flux_axes().values())
        assert all(any(hasattr(c, "_islat_wavelength_marker") for c in ax.collections) for ax in flux_axes)
        assert any(_artists(ax, "_islat_saved_line") for ax in flux_axes)
        view.destroy()

    def test_save_figure_includes_flux_settings(self, with_flux, tmp_path, monkeypatch):
        view, pm, islat, _ = with_flux
        pm.toggle_state.update(observed=False, summed=False, y_autoscale=False, y_limits=(0.0, 0.3))
        created = []
        original = view._create_standalone_plot

        def _capture(*args, **kwargs):
            plot = original(*args, **kwargs)
            created.append(plot)
            plot.close = lambda: None  # keep the figure to inspect; closed below
            return plot

        monkeypatch.setattr(view, "_create_standalone_plot", _capture)
        out = tmp_path / "tau_flux.png"
        assert view.save_figure(str(out), file_format="png", dpi=40) == str(out)
        plot = created[0]
        try:
            assert out.exists() and plot.show_flux
            flux_axes = list(plot.get_flux_axes().values())
            assert len(flux_axes) == plot.n_panels
            for ax in flux_axes:
                assert ax.get_ylim() == (0.0, 0.3)
                assert not any(a.get_visible() for a in _artists(ax, "_islat_observed"))
                assert not any(a.get_visible() for a in _artists(ax, "_islat_summed"))
        finally:
            type(plot).close(plot)


# ======================================================================
# Fractional share of the total τ
# ======================================================================

class TestFractionToggle:
    def test_fraction_mode_renders_shares(self, rendered):
        view, pm, *_ = rendered
        view._set_fraction_mode(True)
        assert pm.toggle_state["tau_display_mode"] == "fraction"
        assert view._plot.display_mode == "fraction"
        assert view.fig._supylabel.get_text() == "Fraction of total \u03c4"
        for ax in view.subplots.values():
            assert ax.get_ylim() == pytest.approx((0.0, 1.05))
        assert not _total_tau_lines(view)
        for cell in view._plot.panels.values():
            lam, shares = cell[0].compute_fractional_shares()
            total = np.nansum([s[0] for s in shares], axis=0)
            defined = ~np.isnan(shares[0][0])
            assert np.allclose(total[defined], 1.0)

    def test_fraction_and_stems_are_exclusive(self, rendered):
        view, pm, *_ = rendered
        view._set_stem_mode(True)
        view._set_fraction_mode(True)
        assert (view._get_fraction_mode(), view._get_stem_mode()) == (True, False)
        view._set_stem_mode(False)  # not active: leaves fraction alone
        assert pm.toggle_state["tau_display_mode"] == "fraction"
        view._set_stem_mode(True)
        assert (view._get_fraction_mode(), view._get_stem_mode()) == (False, True)
        view._set_stem_mode(False)
        assert pm.toggle_state["tau_display_mode"] == "profile"

    def test_views_menu_toggle_flips_fraction(self, rendered):
        view, pm, *_ = rendered
        bus = ControlBus()
        vm = _NoopSurface()
        bus.register_surface("views_menu", vm)
        pm.control_bus = bus
        view._register_control_fields()
        field = vm.get_field("tau_fraction_mode")
        field.setter(not field.getter())
        assert pm.toggle_state["tau_display_mode"] == "fraction"
        field.setter(not field.getter())
        assert pm.toggle_state["tau_display_mode"] == "profile"

    def test_fraction_with_flux_panels_and_log(self, rendered):
        view, pm, *_ = rendered
        pm.toggle_state["tau_log_scale"] = True
        view._set_show_flux(True)
        view._set_fraction_mode(True)
        assert len(view._plot.get_flux_axes()) == view._plot.n_panels
        for ax in view.subplots.values():
            assert ax.get_yscale() == "linear" and ax.get_ylim() == pytest.approx((0.0, 1.05))

    def test_export_uses_fraction_mode(self, rendered, tmp_path, monkeypatch):
        view, pm, *_ = rendered
        view._set_fraction_mode(True)
        created = []
        original = view._create_standalone_plot
        monkeypatch.setattr(view, "_create_standalone_plot",
                            lambda *a, **k: created.append(original(*a, **k)) or created[-1])
        assert view.save_figure(str(tmp_path / "frac.png"), file_format="png", dpi=40)
        assert created[0].display_mode == "fraction"
