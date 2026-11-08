# -*- coding: utf-8 -*-
"""SampleStackPlot and the Full Spectrum view's *Stack Sample Spectra* option."""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.DataProcessing.SampleStack import SampleSpectrum
from iSLAT.Modules.GUI.ControlField import DropdownField, ToggleField
from iSLAT.Modules.GUI.ControlSurface import ControlBus, ControlSurface
from iSLAT.Modules.Plotting import FullSpectrumPlot, ResidualSpectrumPlot, SampleStackPlot
from iSLAT.Modules.Plotting.FullSpectrumView import (
    FullSpectrumView, SAMPLE_STACK_DEFAULTS, SAMPLE_STACK_LAYOUT_LABELS, SAMPLE_STACK_MODEL_LABELS,
)
from iSLAT.Modules.Plotting.OpticalDepthView import OpticalDepthView
from iSLAT.Modules.Plotting.SampleStackPlot import (
    ERRORBAR_TAG, MIN_SLOT_FRACTION, STACK_LABEL_TAG, SampleOffsetPanel,
)

ROOT = Path(__file__).resolve().parents[2]
CO_PAR = ROOT / "iSLAT" / "DATAFILES" / "HITRANdata" / "data_Hitran_CO.par"
needs_co = pytest.mark.skipif(not CO_PAR.exists(), reason="CO HITRAN file not available")

WAVE = np.linspace(4.9, 5.2, 300)
LEVELS = (0.010, 0.004, 0.025)     # flux level of each synthetic spectrum


def _flux(level, i):
    return level + 0.5 * level * np.exp(-0.5 * ((WAVE - 4.95 - 0.08 * i) / 0.002) ** 2)


def _spectra(molecules=None):
    return [
        SampleSpectrum(path=f"S{i}.csv", name=f"S{i}", wave_obs=WAVE, wave=WAVE,
                       flux=_flux(level, i), err=np.full_like(WAVE, 0.0005),
                       molecules=molecules, is_active=(i == 1))
        for i, level in enumerate(LEVELS)
    ]


def _co_row(temp=600.0):
    return {"Molecule Name": "CO", "File Path": str(CO_PAR), "Molecule Label": "CO", "Temp": temp,
            "Rad": 0.5, "N_Mol": 1e17, "Color": "#ff0000", "Vis": True, "Dist": 140.0, "StellarRV": 0.0,
            "FWHM": 130.0, "Broad": 2.5, "RV Shift": 0.0}


@pytest.fixture
def live():
    if not CO_PAR.exists():
        pytest.skip("CO HITRAN file not available")
    md = MoleculeDict(global_wavelength_range=(4.9, 5.2), global_model_pixel_res=0.001,
                      global_distance=140.0)
    md.load_molecules([_co_row()], {}, update_global_parameters=True, strategy="sequential")
    yield md
    Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


def _render(spectra, **kw):
    kw.setdefault("n_panels", 2)
    kw.setdefault("figsize", (10, 6))
    plot = SampleStackPlot(spectra, **kw)
    plot.generate_plot()
    return plot


def _tagged(ax, tag):
    return [a for a in list(ax.lines) + list(ax.collections) + list(ax.texts) if hasattr(a, tag)]


# ======================================================================
# The plot
# ======================================================================
class TestOffsetLayout:
    def test_one_axes_per_panel_with_non_overlapping_slots(self):
        plot = _render(_spectra(), layout="offset", model_mode="data")
        assert sorted(plot.subplots) == [0, 1]
        for idx, ax in plot.subplots.items():
            panel = plot.panels[idx][0]
            assert isinstance(panel, SampleOffsetPanel)
            slots = panel.slots
            # The first spectrum is at the top, each slot above the next one
            for upper, lower in zip(slots, slots[1:]):
                assert upper[0] > lower[1]
            # Weak spectra still get a readable slot
            heights = [hi - lo for lo, hi in slots]
            assert min(heights) >= MIN_SLOT_FRACTION * max(heights) - 1e-12
            assert ax.get_ylim()[0] <= slots[-1][0] and ax.get_ylim()[1] >= slots[0][1]
            assert len(_tagged(ax, "_islat_observed")) >= 3
            labels = [t.get_text() for t in ax.texts if hasattr(t, STACK_LABEL_TAG)]
            assert labels == ["S0", "S1", "S2"]
        assert plot.fig._supylabel.get_text() == "Flux Density (Jy) + offset"
        plot.close()

    def test_each_spectrum_is_drawn_at_its_offset(self):
        spectra = _spectra()
        plot = _render(spectra, layout="offset", model_mode="data", n_panels=1)
        panel = plot.panels[0][0]
        lines = plot.subplots[0].lines
        data_lines = [ln for ln in lines if hasattr(ln, "_islat_observed") and not hasattr(ln, ERRORBAR_TAG)]
        bar_lines = [ln for ln in lines if hasattr(ln, ERRORBAR_TAG)]
        assert len(data_lines) == len(bar_lines) == 3
        for line, bars, spectrum, offset in zip(data_lines, bar_lines, spectra, panel.offsets):
            assert np.allclose(line.get_ydata(), spectrum.flux + offset)
            # One NaN-separated segment per point, from flux - err to flux + err
            y = np.asarray(bars.get_ydata())
            assert np.allclose(y[0::3], spectrum.flux + offset - spectrum.err)
            assert np.allclose(y[1::3], spectrum.flux + offset + spectrum.err)
            assert np.all(np.isnan(y[2::3]))
            assert hasattr(bars, "_islat_observed")      # hidden with the observed data
        plot.close()

    def test_spectrum_at_picks_the_slot(self):
        plot = _render(_spectra(), layout="offset", model_mode="data")
        ax = plot.subplots[0]
        panel = plot.panels[0][0]
        for spectrum, (lo, hi) in zip(plot.spectra, panel.slots):
            assert plot.spectrum_at(ax, (lo + hi) / 2) is spectrum
        assert plot.spectrum_at(ax, None) is None
        plot.close()


class TestRowsLayout:
    def test_one_row_per_spectrum_with_its_own_scale(self):
        spectra = _spectra()
        plot = _render(spectra, layout="rows", model_mode="data")
        for idx, cell in plot.subplots.items():
            assert isinstance(cell, tuple) and len(cell) == 3
            assert [plot.spectrum_for_axes(ax) for ax in cell] == spectra
            for ax, spectrum in zip(cell, spectra):
                lo, hi = ax.get_ylim()
                sel = (spectrum.wave >= ax.get_xlim()[0]) & (spectrum.wave <= ax.get_xlim()[1])
                assert hi == pytest.approx(spectrum.flux[sel].max() * (1 + plot.ymax_factor))
            # Only the bottom row of a panel shows wavelength tick labels
            assert [ax.xaxis.get_tick_params()["labelbottom"] for ax in cell[:-1]] == [False, False]
        assert [s.name for s, _axes in plot.axes_by_spectrum()] == ["S0", "S1", "S2"]
        plot.close()

    def test_row_y_options(self):
        spectra = _spectra()
        plot = _render(spectra, layout="rows", model_mode="data", y_data_sync="panel")
        ax = plot.subplots[1][2]
        sel = (WAVE >= ax.get_xlim()[0]) & (WAVE <= ax.get_xlim()[1])
        assert ax.get_ylim() == pytest.approx((spectra[2].flux[sel].min(), spectra[2].flux[sel].max()), rel=0.06)
        plot.close()
        plot = _render(spectra, layout="rows", model_mode="data", y_limits=(0.0, 1.0))
        assert all(ax.get_ylim() == (0.0, 1.0) for cell in plot.subplots.values() for ax in cell)
        plot.close()

    def test_saved_line_labels_only_on_the_top_row(self):
        lines = pd.DataFrame({"wave": [4.95, 5.12], "species": ["CO", "CO"], "line": ["a", "b"]})
        plot = _render(_spectra(), layout="rows", model_mode="data", line_list=lines)
        for cell in plot.subplots.values():
            top, *rest = cell
            assert any(isinstance(a, matplotlib.text.Text) for a in _tagged(top, "_islat_saved_line"))
            for ax in rest:
                tagged = _tagged(ax, "_islat_saved_line")
                assert tagged and not any(isinstance(a, matplotlib.text.Text) for a in tagged)
        plot.remove_saved_lines()
        assert not any(_tagged(ax, "_islat_saved_line") for cell in plot.subplots.values() for ax in cell)
        plot.close()


@needs_co
class TestModelModes:
    @pytest.mark.parametrize("layout", ["offset", "rows"])
    def test_modes(self, live, layout):
        spectra = _spectra(live)
        counts = {}
        for mode in ("data", "total", "molecules"):
            plot = _render(spectra, layout=layout, model_mode=mode, n_panels=1)
            axes = plot.subplots[0] if layout == "rows" else (plot.subplots[0],)
            counts[mode] = (
                sum(len(_tagged(ax, "_islat_summed")) for ax in axes),
                sum(len([ln for ln in ax.lines if getattr(ln, "_molecule_name", None)]) for ax in axes),
                plot.subplots[0 if layout == "offset" else 0] is not None,
            )
            legend_ax = plot._legend_axes
            assert (legend_ax.get_legend() is not None) == (mode == "molecules")
            plot.close()
        assert counts["data"][:2] == (0, 0)
        assert counts["total"][0] == 3 and counts["total"][1] == 0
        assert counts["molecules"][0] == 3 and counts["molecules"][1] == 3

    def test_shared_parameter_sets_compute_their_models_once(self, live, monkeypatch):
        import sys
        ssp = sys.modules["iSLAT.Modules.Plotting.SampleStackPlot"]   # the module (the package exports the class)
        calls = []
        original = ssp.compute_stack_models
        monkeypatch.setattr(ssp, "compute_stack_models",
                            lambda s, mode: calls.append(s.name) or original(s, mode))
        plot = _render(_spectra(live), layout="offset", model_mode="total", n_panels=1)
        assert len(calls) == 1
        plot.close()
        # Matched sampling resamples the models onto each spectrum's own grid
        calls.clear()
        live.match_spectral_sampling = True
        spectra = _spectra(live)
        for s in spectra:                 # each spectrum has its own wavelength grid
            s.wave_obs = s.wave = WAVE.copy()
        plot = _render(spectra, layout="offset", model_mode="total", n_panels=1)
        assert len(calls) == 3
        plot.close()

    def test_offset_models_move_with_their_spectrum(self, live):
        spectra = _spectra(live)
        plot = _render(spectra, layout="offset", model_mode="molecules", n_panels=1)
        panel = plot.panels[0][0]
        model_lines = [ln for ln in plot.subplots[0].lines if getattr(ln, "_molecule_name", None)]
        lam, flux = model_lines[0].get_data()
        # Same model for every spectrum (shared set), drawn at each offset
        for line, offset in zip(model_lines, panel.offsets):
            assert np.allclose(line.get_ydata() - offset, flux - panel.offsets[0])
        plot.close()


class TestMutators:
    def test_set_spectra_keeps_the_display_range(self):
        plot = _render(_spectra(), layout="offset", model_mode="data")
        plot._xlim_start, plot._xlim_end = 4.95, 5.15
        plot._compute_panel_layout()
        plot.set_spectra(list(reversed(_spectra())))
        assert (plot._xlim_start, plot._xlim_end) == (4.95, 5.15)
        assert plot.primary.name == "S1"
        plot.set_options(layout="rows", model_mode="bogus")
        assert (plot.layout, plot.model_mode) == ("rows", "total")
        assert plot.y_axis_label == "Flux Density (Jy)"
        plot.close()

    def test_needs_a_spectrum(self):
        with pytest.raises(ValueError):
            SampleStackPlot([])


# ======================================================================
# The Full Spectrum view
# ======================================================================
class _NoopSurface(ControlSurface):
    def _rebuild(self):
        pass


def _write_spectrum(path, level, i):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wave", "flux", "err"])
        for x, y in zip(WAVE, _flux(level, i)):
            writer.writerow([x, y, 0.0005])


@pytest.fixture
def view_setup(tmp_path, live):
    paths = []
    for i, level in enumerate(LEVELS):
        p = str(tmp_path / f"S{i}.csv")
        _write_spectrum(p, level, i)
        paths.append(p)
    islat = SimpleNamespace(
        molecules_dict=live, loaded_spectrum_file=paths[0], wave_data_original=WAVE.copy(),
        wave_data=WAVE.copy(), flux_data=_flux(LEVELS[0], 0), err_data=np.full_like(WAVE, 0.0005),
        sample_spectra=list(paths), sample_spectra_params={}, sample_spectra_overrides={},
        user_settings={}, initial_molecule_parameters={}, input_line_list="missing.csv", root=None,
    )
    created = []

    def make(cls=FullSpectrumView, **toggles):
        pm = MagicMock()
        pm.islat = islat
        pm.theme = {}
        pm.legend_toggle = True
        pm.control_bus = None
        pm.toggle_state = {
            "atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
            "show_residuals": False, "data_density": False, "skip_gaps": False,
            "observed": True, "y_autoscale": True, "y_autoscale_source": "data", "y_data_sync": None,
            "y_limits": None, "wavelength_markers": [], **SAMPLE_STACK_DEFAULTS, **toggles,
        }
        view = cls(pm)
        view._ensure_canvas = lambda: None
        view._plot = view._create_plot()
        view._plot.n_panels = 2
        view._plot._compute_panel_layout()
        view._plot.generate_plot()
        view._install_span_selectors()
        view._initialised = True
        pm.active_view = view
        created.append(view)
        return view, pm

    yield make, islat, paths
    for view in created:
        view.destroy()


@needs_co
class TestView:
    def test_stack_option_builds_a_sample_stack(self, view_setup):
        make, islat, _paths = view_setup
        view, pm = make(sample_stack=True)
        assert isinstance(view._plot, SampleStackPlot)
        assert [s.name for s in view._plot.spectra] == ["S0", "S1", "S2"]
        assert view._plot.spectra[0].is_active

    def test_toggle_on_and_off_keeps_the_panel_layout(self, view_setup):
        make, *_ = view_setup
        view, pm = make()
        edges = view._plot._panel_edges.copy()
        view._set_sample_stack(True)
        assert isinstance(view._plot, SampleStackPlot)
        assert np.allclose(view._plot._panel_edges, edges)
        view._set_sample_stack(False)
        assert type(view._plot) is FullSpectrumPlot
        assert view._sample_stack_spectra == []

    def test_stack_and_residuals_are_exclusive(self, view_setup):
        make, *_ = view_setup
        view, pm = make(show_residuals=True)
        view._set_sample_stack(True)
        assert pm.toggle_state["show_residuals"] is False
        assert isinstance(view._plot, SampleStackPlot)
        pm.toggle_state["show_residuals"] = True
        view.toggle_residuals(True)
        assert pm.toggle_state["sample_stack"] is False
        assert isinstance(view._plot, ResidualSpectrumPlot)

    def test_layout_and_model_choices(self, view_setup):
        make, *_ = view_setup
        view, pm = make(sample_stack=True)
        fig = view.fig
        view._set_stack_layout_label(SAMPLE_STACK_LAYOUT_LABELS["rows"])
        assert pm.toggle_state["sample_stack_layout"] == "rows"
        assert view._plot.layout == "rows" and view.fig is fig
        assert all(isinstance(cell, tuple) for cell in view._plot.subplots.values())
        # Span selectors on every row
        assert len(view.span_selectors) == 2 * 3
        view._set_stack_models_label(SAMPLE_STACK_MODEL_LABELS["molecules"])
        assert view._plot.model_mode == "molecules"
        assert view._get_stack_models_label() == "Each Molecule + Total"
        view._set_stack_layout_label("not an option")
        assert pm.toggle_state["sample_stack_layout"] == "rows"

    def test_y_policy_hooks(self, view_setup):
        make, *_ = view_setup
        view, pm = make(sample_stack=True)
        assert list(view._iter_y_axes()) == []                  # offset: the stack keeps its limits
        assert len(list(view._iter_observed_axes())) == 2
        view._set_stack_layout_label(SAMPLE_STACK_LAYOUT_LABELS["rows"])
        assert len(list(view._iter_y_axes())) == 6
        pairs = view._data_synced_y_limits_by_axes("panel")
        assert len(pairs) == 6
        for ax, lims in pairs:
            spectrum = view._plot.spectrum_for_axes(ax)
            sel = (spectrum.wave >= ax.get_xlim()[0]) & (spectrum.wave <= ax.get_xlim()[1])
            assert lims == pytest.approx((spectrum.flux[sel].min(), spectrum.flux[sel].max()), rel=0.06)
        pm.toggle_state.update(y_autoscale=False, y_limits=(0.0, 2.0))
        view.apply_y_limits(pm.toggle_state)
        assert all(ax.get_ylim() == (0.0, 2.0) for ax in view._iter_y_axes())

    def test_sample_changes_refresh_the_stack(self, view_setup):
        make, islat, paths = view_setup
        view, pm = make(sample_stack=True)
        plot = view._plot
        view.on_sample_changed()                                   # nothing changed
        assert view._plot.spectra[1].molecules is islat.molecules_dict
        islat.sample_spectra_overrides = {paths[1]: {"distance": 280.0}}
        view.on_sample_changed()
        assert view._plot is plot
        assert view._plot.spectra[1].parameter_source == "current"
        islat.sample_spectra = paths[:2]
        view.on_sample_changed()
        assert [s.name for s in view._plot.spectra] == ["S0", "S1"]

    def test_parameter_changes_reach_copies(self, view_setup):
        make, islat, paths = view_setup
        islat.sample_spectra_overrides = {paths[1]: {"distance": 280.0}}
        view, pm = make(sample_stack=True)
        islat.molecules_dict["CO"].temp = 800.0
        view.on_molecule_parameter_changed("CO", "temp")
        assert view._plot.spectra[1].molecules["CO"].temp == 800.0

    def test_registers_the_stack_controls(self, view_setup):
        make, *_ = view_setup
        view, pm = make()
        bus = ControlBus()
        menu = _NoopSurface()
        bus.register_surface("views_menu", menu)
        pm.control_bus = bus
        view._register_control_fields()
        keys = [f.key for f in menu.fields()]
        assert keys[-3:] == ["sample_stack", "sample_stack_layout", "sample_stack_models"]
        assert isinstance(menu.get_field("sample_stack"), ToggleField)
        layout = menu.get_field("sample_stack_layout")
        assert isinstance(layout, DropdownField)
        assert layout.options == ["Offset in Each Panel", "One Row per Spectrum"]
        assert layout.getter() == "Offset in Each Panel"
        assert menu.get_field("sample_stack_models").options == [
            "Data Only", "Total Model", "Each Molecule + Total"]

    def test_save_figure_exports_the_stack(self, view_setup, tmp_path, monkeypatch):
        make, *_ = view_setup
        view, pm = make(sample_stack=True, sample_stack_layout="rows")
        created = []
        original = view._create_standalone_plot

        def capture(*args, **kwargs):
            plot = original(*args, **kwargs)
            created.append(plot)
            return plot

        monkeypatch.setattr(view, "_create_standalone_plot", capture)
        out = tmp_path / "stack.png"
        assert view.save_figure(str(out), file_format="png", dpi=40) == str(out)
        assert out.exists() and out.stat().st_size > 0
        assert isinstance(created[0], SampleStackPlot) and created[0].layout == "rows"
        assert len(created[0].spectra) == 3

    def test_context_menu_offers_to_activate_another_spectrum(self, view_setup):
        make, islat, paths = view_setup
        view, pm = make(sample_stack=True, sample_stack_layout="rows")
        islat.switch_to_spectrum = MagicMock()
        ax = view._plot.subplots[0][2]
        assert view._plot.spectrum_at(ax).name == "S2"
        view._activate_sample_spectrum(view._plot.spectrum_at(ax))
        islat.switch_to_spectrum.assert_called_once_with(2)

    def test_hint_when_the_sample_has_nothing_to_stack(self, view_setup):
        make, islat, paths = view_setup
        islat.sample_spectra = [paths[0]]
        islat.GUI = MagicMock()
        view, pm = make()
        view._set_sample_stack(True)
        assert isinstance(view._plot, SampleStackPlot) and len(view._plot.spectra) == 1
        text = " ".join(str(c.args[0]) for c in islat.GUI.data_field.insert_text.call_args_list)
        assert "no other spectra" in text

    def test_optical_depth_view_never_stacks(self, view_setup):
        make, *_ = view_setup
        pm = MagicMock()
        pm.toggle_state = {"sample_stack": True, "show_residuals": False}
        view = OpticalDepthView.__new__(OpticalDepthView)
        view._pm = pm
        assert view._sample_stack_on() is False
