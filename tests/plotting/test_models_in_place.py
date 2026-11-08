# -*- coding: utf-8 -*-
"""Molecule parameter changes update the spectrum panel's models in place.

MainPlotGrid.update_models_in_place gives the model lines and summed fill
of the last full render new data instead of re-rendering the panel, so the
observed spectrum (whose error bars are slow to re-create) is kept.  The
result must look exactly like a full re-render.
"""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.Plotting.ArtistTags import OBSERVED_TAG, SUMMED_TAG, iter_molecule_lines, iter_tagged
from iSLAT.Modules.Plotting.MainPlotGrid import MainPlotGrid

from tests.plotting import make_wave_flux
from tests.plotting.test_plot_views import _make_test_molecule


def _molecules():
    md = MoleculeDict()
    for name, start, color in (("H2O", 12.0, "#1f77b4"), ("CO", 14.5, "#d62728")):
        mol = _make_test_molecule(name=name, lam_start=start)
        mol.color = color
        md[mol.name] = mol
    return md


def _grid():
    """A rendered three-panel grid in borrowed-axes mode, as the GUI uses it."""
    fig = Figure(figsize=(12, 7), dpi=80)
    FigureCanvasAgg(fig)
    ax1, ax2, ax3 = MainPlotGrid.create_three_panel_axes(fig)
    wave, flux, err = make_wave_flux(n=600)
    md = _molecules()
    grid = MainPlotGrid(
        wave, flux, error_data=err, molecules=md, active_molecule=md["H2O"],
        ax_spectrum=ax1, ax_inspection=ax2, ax_popdiagram=ax3,
    )
    grid.generate_plot()
    return grid


def _pixels(grid):
    grid.fig.canvas.draw()
    return np.asarray(grid.fig.canvas.buffer_rgba()).copy()


def _observed(ax):
    return list(iter_tagged(ax, OBSERVED_TAG))


def _legend_texts(ax):
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend is not None else []


_CHANGES = {
    "n_mol": lambda md: setattr(md["H2O"], "n_mol", 3e18),
    "temp": lambda md: setattr(md["H2O"], "temp", 900.0),
    "fwhm": lambda md: setattr(md["CO"], "fwhm", 60.0),
    "rv_shift": lambda md: setattr(md["CO"], "rv_shift", 25.0),
    "model_line_width": lambda md: setattr(md["H2O"], "model_line_width", 3.0),
}


class TestMatchesFullRender:
    @pytest.mark.parametrize("change", list(_CHANGES.values()), ids=list(_CHANGES))
    def test_pixels_and_model_data(self, change):
        in_place, full = _grid(), _grid()
        change(in_place.molecules)
        change(full.molecules)
        assert in_place.update_models_in_place()
        full._render_spectrum_panel()
        np.testing.assert_array_equal(_pixels(in_place), _pixels(full))
        for a, b in zip(iter_molecule_lines(in_place.ax_spectrum), iter_molecule_lines(full.ax_spectrum)):
            np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
            assert (a.get_color(), a.get_linewidth(), a.get_label()) == (b.get_color(), b.get_linewidth(), b.get_label())
        assert _legend_texts(in_place.ax_spectrum) == _legend_texts(full.ax_spectrum)

    def test_models_actually_change(self):
        grid = _grid()
        line = next(iter_molecule_lines(grid.ax_spectrum, "H2O"))
        before = line.get_ydata().copy()
        grid.molecules["H2O"].n_mol = 3e18
        assert grid.update_models_in_place()
        assert not np.allclose(line.get_ydata(), before)

    def test_keeps_observed_and_model_artists(self):
        grid = _grid()
        ax = grid.ax_spectrum
        observed = _observed(ax)
        lines = list(iter_molecule_lines(ax))
        fills = list(iter_tagged(ax, SUMMED_TAG, lines=False))
        assert observed and lines and fills
        grid.molecules["H2O"].n_mol = 3e18
        assert grid.update_models_in_place()
        assert all(a is b for a, b in zip(_observed(ax), observed)) and len(_observed(ax)) == len(observed)
        assert list(iter_molecule_lines(ax)) == lines
        assert list(iter_tagged(ax, SUMMED_TAG, lines=False)) == fills

    def test_legend_keeps_entries_of_hidden_artists(self):
        # A full render rebuilds the legend before the toggles hide the
        # summed fill / observed data, so their entries stay.
        in_place, full = _grid(), _grid()
        for grid in (in_place, full):
            for artist in [*_observed(grid.ax_spectrum), *iter_tagged(grid.ax_spectrum, SUMMED_TAG, lines=False)]:
                artist.set_visible(False)
            grid.molecules["H2O"].n_mol = 3e18
        assert in_place.update_models_in_place()
        full._render_spectrum_panel()
        assert "Sum" in _legend_texts(in_place.ax_spectrum)
        assert "Data" in _legend_texts(in_place.ax_spectrum)
        assert _legend_texts(in_place.ax_spectrum) == _legend_texts(full.ax_spectrum)
        # ... and the hidden artists stay hidden
        assert not any(a.get_visible() for a in _observed(in_place.ax_spectrum))
        assert not any(a.get_visible() for a in iter_tagged(in_place.ax_spectrum, SUMMED_TAG, lines=False))


def _line_data(ax):
    return [line.get_xydata().copy() for line in iter_molecule_lines(ax)]


class TestFallsBackToFullRender:
    """update_models_in_place returns False, changing nothing, when a full render is needed."""

    @staticmethod
    def _assert_declines(grid):
        before = _line_data(grid.ax_spectrum)
        assert grid.update_models_in_place() is False
        after = _line_data(grid.ax_spectrum)
        assert len(before) == len(after) and all(np.array_equal(a, b) for a, b in zip(before, after))

    def test_never_rendered(self):
        wave, flux, err = make_wave_flux(n=100)
        assert MainPlotGrid(wave, flux, error_data=err, molecules=_molecules()).update_models_in_place() is False

    def test_observed_data_changed(self):
        grid = _grid()
        grid.flux_data = grid.flux_data * 2.0
        self._assert_declines(grid)

    def test_error_data_changed(self):
        grid = _grid()
        grid.error_data = None
        self._assert_declines(grid)

    def test_wavelength_range_changed(self):
        grid = _grid()
        grid.spectrum_range = (11.0, 18.0)
        self._assert_declines(grid)

    def test_visible_models_changed(self):
        grid = _grid()
        grid.molecules["CO"].is_visible = False
        self._assert_declines(grid)

    def test_model_lines_replaced_since_render(self):
        grid = _grid()
        grid.handle_molecule_visibility_change("CO", False)
        grid.molecules["CO"].is_visible = True
        grid.handle_molecule_visibility_change("CO", True)
        self._assert_declines(grid)
        grid._render_spectrum_panel()  # a full render makes in-place updates possible again
        grid.molecules["H2O"].n_mol = 3e18
        assert grid.update_models_in_place()

    def test_model_line_redrawn_since_render(self):
        # Same molecules in the same order, but the last line is a new artist
        # (drawn the way handle_molecule_visibility_change draws it).
        grid = _grid()
        last_name = list(iter_molecule_lines(grid.ax_spectrum))[-1]._molecule_name
        grid.remove_molecule_lines(last_name)
        grid.spectrum_panel._plot_molecule_spectrum(grid.ax_spectrum, grid.molecules[last_name], grid.wave_data)
        assert [l._molecule_name for l in iter_molecule_lines(grid.ax_spectrum)][-1] == last_name
        self._assert_declines(grid)

    def test_summed_fill_removed_since_render(self):
        grid = _grid()
        next(iter_tagged(grid.ax_spectrum, SUMMED_TAG, lines=False)).remove()
        self._assert_declines(grid)

    def test_nan_in_observed_data_still_matches(self):
        # NaN gaps in the data must not make the same data look changed.
        fig = Figure(figsize=(8, 5), dpi=60)
        FigureCanvasAgg(fig)
        ax1, ax2, ax3 = MainPlotGrid.create_three_panel_axes(fig)
        wave, flux, err = make_wave_flux(n=300, gap=(15.0, 15.5))
        md = _molecules()
        grid = MainPlotGrid(wave, flux, error_data=err, molecules=md, active_molecule=md["H2O"],
                            ax_spectrum=ax1, ax_inspection=ax2, ax_popdiagram=ax3)
        grid.generate_plot()
        grid.flux_data = flux.copy()
        md["H2O"].n_mol = 3e18
        assert grid.update_models_in_place()


# ----------------------------------------------------------------------
# ThreePanelView: a molecule parameter change keeps the observed spectrum
# ----------------------------------------------------------------------
@pytest.fixture
def three_panel():
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
    fig = Figure(figsize=(12, 7), dpi=80)
    FigureCanvasAgg(fig)
    ax1, ax2, ax3 = MainPlotGrid.create_three_panel_axes(fig)
    wave, flux, err = make_wave_flux(n=600)
    md = _molecules()
    islat = SimpleNamespace(
        wave_data=wave, wave_data_original=wave, flux_data=flux, err_data=err,
        molecules_dict=md, active_molecule=md["H2O"], comparison_molecules=[],
        user_settings={"line_threshold": 0.0}, input_line_list=None,
    )
    pm = MagicMock()
    pm.fig, pm.ax1, pm.ax2, pm.ax3 = fig, ax1, ax2, ax3
    pm.islat = islat
    pm.theme = {"background": "#ffffff", "foreground": "#000000"}
    pm.toggle_state = {
        "atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
        "observed": True, "y_autoscale": True, "y_limits": None, "current_selection": None,
    }
    pm.legend_toggle = True
    view = ThreePanelView(pm)
    view.update_model_plot()
    return view, islat


class TestThreePanelView:
    def test_parameter_change_keeps_observed_spectrum(self, three_panel):
        view, islat = three_panel
        ax = view.ax1
        observed = _observed(ax)
        line = next(iter_molecule_lines(ax, "H2O"))
        before = line.get_ydata().copy()
        islat.molecules_dict["H2O"].n_mol = 3e18
        view.on_molecule_parameter_changed("H2O", "n_mol")
        assert _observed(ax) == observed
        assert next(iter_molecule_lines(ax, "H2O")) is line
        assert not np.allclose(line.get_ydata(), before)

    def test_summed_toggle_respected_after_parameter_change(self, three_panel):
        view, islat = three_panel
        view._pm.toggle_state["summed"] = False
        view.toggle_summed_spectrum(False)
        islat.molecules_dict["H2O"].n_mol = 3e18
        view.on_molecule_parameter_changed("H2O", "n_mol")
        fills = list(iter_tagged(view.ax1, SUMMED_TAG, lines=False))
        assert fills and not any(f.get_visible() for f in fills)

    def test_new_observed_data_is_rendered_in_full(self, three_panel):
        view, islat = three_panel
        observed = _observed(view.ax1)
        islat.flux_data = islat.flux_data * 2.0
        islat.molecules_dict["H2O"].n_mol = 3e18
        view.on_molecule_parameter_changed("H2O", "n_mol")
        assert not any(a is b for a, b in zip(_observed(view.ax1), observed))
