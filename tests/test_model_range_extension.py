# -*- coding: utf-8 -*-
"""Models must be drawn over the full global wavelength range even when
"Match Pix. Sampling" resamples them onto an observed grid that is narrower.

Regression for: lowering *Min. Wave* below the first data point left the
model un-drawn there until the molecule's visibility was toggled (the
visibility path draws the native grid, the normal render the resampled one).
"""

import os

import numpy as np
import pytest

from iSLAT.Modules.DataProcessing.spectral_utils import extend_grid_to_range

_CO_PAR = os.path.join("iSLAT", "DATAFILES", "HITRANdata", "data_Hitran_CO.par")


class TestExtendGridToRange:
    def test_pads_below_and_above_at_median_spacing(self):
        grid = np.linspace(5.0, 6.0, 11)              # spacing 0.1
        ext = extend_grid_to_range(grid, 4.55, 6.25)
        assert ext.min() == pytest.approx(4.5)
        assert ext.max() == pytest.approx(6.3)
        assert np.allclose(np.diff(ext), 0.1)
        assert np.array_equal(ext[(ext >= 5.0) & (ext <= 6.0)], grid)

    def test_noop_when_already_covered_or_within_one_pixel(self):
        grid = np.linspace(5.0, 6.0, 11)
        assert extend_grid_to_range(grid, 5.2, 5.8) is grid
        assert extend_grid_to_range(grid, 4.95, 6.05) is grid

    def test_degenerate_inputs_returned_unchanged(self):
        one = np.array([5.0])
        assert extend_grid_to_range(one, 4.0, 6.0) is one
        flat = np.array([5.0, 5.0, 5.0])
        assert extend_grid_to_range(flat, 4.0, 6.0) is flat

    def test_none_bounds_are_ignored(self):
        grid = np.linspace(5.0, 6.0, 11)
        assert extend_grid_to_range(grid, None, None) is grid
        ext = extend_grid_to_range(grid, None, 6.55)
        assert ext.min() == pytest.approx(5.0) and ext.max() == pytest.approx(6.6)


@pytest.fixture(scope="module")
def co_setup():
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    if not os.path.exists(_CO_PAR):
        pytest.skip("CO HITRAN file not available")
    md = MoleculeDict()
    mol = Molecule(
        name='CO', displaylabel='CO', filepath=_CO_PAR, color='#f00', is_visible=True,
        temp=600.0, radius=0.5, n_mol=1e17, distance=140.0, fwhm=130.0,
        wavelength_range=(4.9, 5.2), model_pixel_res=0.001,
        initial_molecule_parameters={'t_kin': 600.0, 'scale_exponent': 17.0,
                                     'scale_number': 1.0, 'radius_init': 0.5},
    )
    md['CO'] = mol
    obs = np.linspace(4.9, 5.2, 301)          # observed data covers 4.9-5.2 only
    md.global_wavelength_range = (4.6, 5.2)   # user lowers Min. Wave below the data
    md.match_spectral_sampling = True
    return md, mol, obs


class TestMatchedSamplingReachesModelRange:
    def test_plot_molecule_data_extends_below_data(self, co_setup):
        from iSLAT.Modules.Plotting.BasePlot import BasePlot
        md, mol, obs = co_setup
        use, target = md.get_matched_sampling_wavelengths(obs)
        assert use is True
        lam, flux = BasePlot.get_molecule_spectrum_data(
            mol, obs, interpolate_to_input=use, target_wavelengths=target, extend_to_range=True)
        assert len(lam) == len(flux)
        assert lam.min() == pytest.approx(4.6, abs=0.002)
        assert np.any(flux[lam < 4.9] > 0), "model not drawn below the first data point"
        assert np.median(np.diff(lam)) == pytest.approx(np.median(np.diff(obs)), rel=1e-6)

    def test_plot_tau_data_extends_below_data(self, co_setup):
        from iSLAT.Modules.Plotting.BasePlot import BasePlot
        md, mol, obs = co_setup
        use, target = md.get_matched_sampling_wavelengths(obs)
        lam, tau = BasePlot.get_molecule_tau_data(
            mol, obs, interpolate_to_input=use, target_wavelengths=target, extend_to_range=True)
        assert len(lam) == len(tau) and lam.min() == pytest.approx(4.6, abs=0.002)

    def test_default_keeps_exact_observed_grid_for_exports(self, co_setup):
        from iSLAT.Modules.Plotting.BasePlot import BasePlot
        md, mol, obs = co_setup
        use, target = md.get_matched_sampling_wavelengths(obs)
        lam, flux = BasePlot.get_molecule_spectrum_data(
            mol, obs, interpolate_to_input=use, target_wavelengths=target)
        np.testing.assert_allclose(lam, obs)

    def test_summed_flux_extends_only_when_asked(self, co_setup):
        md, _mol, obs = co_setup
        s_lam, s_flux = md.get_summed_flux(obs, extend_to_global_range=True)
        assert s_lam.min() == pytest.approx(4.6, abs=0.002)
        assert np.any(s_flux[s_lam < 4.9] > 0)
        e_lam, _ = md.get_summed_flux(obs)
        np.testing.assert_allclose(e_lam, obs)   # cache keyed on the flag

    def test_native_grid_unaffected_when_matching_is_off(self, co_setup):
        from iSLAT.Modules.Plotting.BasePlot import BasePlot
        md, mol, obs = co_setup
        md.match_spectral_sampling = False
        try:
            lam, _ = BasePlot.get_molecule_spectrum_data(mol, obs, extend_to_range=True)
            assert lam.min() == pytest.approx(4.6, abs=0.002)
        finally:
            md.match_spectral_sampling = True

    def test_stellar_rv_keeps_display_and_target_grids_aligned(self, co_setup):
        from iSLAT.Modules.Plotting.BasePlot import BasePlot
        md, mol, obs = co_setup
        old_rv = md.global_stellar_rv
        md.global_stellar_rv = 30.0
        try:
            use, target = md.get_matched_sampling_wavelengths(obs)
            display = md.apply_stellar_rv(obs)
            lam, flux = BasePlot.get_molecule_spectrum_data(
                mol, display, interpolate_to_input=use, target_wavelengths=target, extend_to_range=True)
            assert len(lam) == len(flux)
            # the returned display grid is the extended target scaled by the same RV factor
            ratio = np.median(display / target)
            assert lam[0] / (target[0] - (target[1] - target[0]) * (len(lam) - len(target))) == pytest.approx(ratio, rel=1e-6)
        finally:
            md.global_stellar_rv = old_rv
