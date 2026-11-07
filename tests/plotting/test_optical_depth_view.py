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
        cp, tb = _NoopSurface(), _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)
        pm.control_bus = bus

        view._register_control_fields()
        assert isinstance(cp.get_field("n_panels"), EntryField)
        assert any(isinstance(f, DisplayRangeField) for f in cp.fields())
        tb_keys = [f.key for f in tb.fields()]
        assert tb_keys == ["tau_log_scale", "tau_stem_mode", "tau_stacked_fill", "data_density"]
        assert all(isinstance(f, ToggleField) for f in tb.fields())

        log_field = tb.get_field("tau_log_scale")
        log_field.setter(not log_field.getter())
        assert pm.toggle_state["tau_log_scale"] is True

        view.deactivate()
        assert cp.fields() == [] and tb.fields() == []

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
