# -*- coding: utf-8 -*-
"""Tests for the integrals behind the line-information printout.

* ``flux_integral`` - the flux density integrated over frequency, in
  erg/s/cm^2, with every flux value paired with its own frequency (an earlier
  version mirrored the profile across the selection, which was only correct
  for lines centred in it).
* ``opacity_integral`` - the model optical-depth profile integrated over
  wavelength, i.e. the summed line-center optical depth of the lines in the range.
* The "Data / Model flux in range" and "Model opacity in range" lines that
  the Three Panel and Line Inspection views print.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.special import ndtr

import iSLAT.Constants as c
from iSLAT.Modules.DataProcessing.spectral_utils import flux_integral, opacity_integral

from tests.plotting.test_plot_views import _make_test_intensity, _make_test_molecule

C_UM = c.SPEED_OF_LIGHT_MICRONS


def _reference_flux_integral(lam_fine, flux_fine):
    """Flux integrated over frequency (erg/s/cm^2) on a very fine grid, using dnu = c / lambda^2 dlambda."""
    return np.trapezoid(flux_fine * C_UM / lam_fine ** 2, x=lam_fine) * 1e-23


def _gauss(lam, centre, sigma, peak=1.0):
    return peak * np.exp(-0.5 * ((lam - centre) / sigma) ** 2)


# ======================================================================
# flux_integral
# ======================================================================

class TestFluxIntegralPairing:
    @pytest.mark.parametrize("centre", [10.05, 10.02, 10.005, 10.09])
    def test_line_anywhere_in_range(self, centre):
        lam = np.linspace(10.0, 10.1, 1001)
        flux = _gauss(lam, centre, 0.002)
        result, _ = flux_integral(lam, flux, None, 10.0, 10.1)
        fine = np.linspace(10.0, 10.1, 200001)
        assert result == pytest.approx(_reference_flux_integral(fine, _gauss(fine, centre, 0.002)), rel=1e-4, abs=0)

    def test_wide_range(self):
        lam = np.linspace(10.0, 20.0, 20001)
        flux = _gauss(lam, 11.0, 0.01)
        result, _ = flux_integral(lam, flux, None, 10.0, 20.0)
        fine = np.linspace(10.0, 20.0, 2000001)
        assert result == pytest.approx(_reference_flux_integral(fine, _gauss(fine, 11.0, 0.01)), rel=1e-4, abs=0)

    def test_constant_flux_density_matches_analytic(self):
        lam = np.linspace(12.0, 18.0, 6001)
        result, _ = flux_integral(lam, np.full_like(lam, 2.0), None, 12.0, 18.0)
        assert result == pytest.approx(2.0 * 1e-23 * (C_UM / 12.0 - C_UM / 18.0), rel=1e-6, abs=0)

    def test_input_order_does_not_matter(self):
        lam = np.linspace(10.0, 10.1, 501)
        flux = _gauss(lam, 10.03, 0.003)
        err = 0.1 * flux + 0.01
        forward = flux_integral(lam, flux, err, 10.0, 10.1)
        backward = flux_integral(lam[::-1], flux[::-1], err[::-1], 10.0, 10.1)
        shuffled = np.random.default_rng(0).permutation(len(lam))
        assert backward == pytest.approx(forward, rel=1e-12, abs=0)
        assert flux_integral(lam[shuffled], flux[shuffled], err[shuffled], 10.0, 10.1) == pytest.approx(forward, rel=1e-12, abs=0)

    def test_errors_are_paired_like_the_flux(self):
        lam = np.linspace(10.0, 10.1, 1001)
        err = _gauss(lam, 10.01, 0.002)  # asymmetric error profile
        _, err_int = flux_integral(lam, np.zeros_like(lam), err, 10.0, 10.1)
        fine = np.linspace(10.0, 10.1, 200001)
        assert err_int == pytest.approx(_reference_flux_integral(fine, _gauss(fine, 10.01, 0.002)), rel=1e-4, abs=0)
        assert err_int > 0


# ======================================================================
# opacity_integral
# ======================================================================

class TestOpacityIntegral:
    @staticmethod
    def _profile(lam, centre=15.0, tau0=0.5, sigma=0.003):
        # A line's tau spread over an area-normalised Gaussian, as Spectrum builds it
        return tau0 * np.exp(-0.5 * ((lam - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def test_recovers_line_center_tau(self):
        lam = np.linspace(14.9, 15.1, 4001)
        assert opacity_integral(lam, self._profile(lam), 14.9, 15.1) == pytest.approx(0.5, rel=1e-6, abs=0)

    def test_partial_range_counts_the_part_inside(self):
        lam = np.linspace(14.9, 15.1, 4001)
        expected = 0.5 * (ndtr((15.1 - 15.0) / 0.003) - ndtr((15.002 - 15.0) / 0.003))
        assert opacity_integral(lam, self._profile(lam), 15.002, 15.1) == pytest.approx(expected, rel=1e-3, abs=0)

    def test_sums_several_lines(self):
        lam = np.linspace(14.0, 16.0, 40001)
        tau = self._profile(lam, 14.5, 0.2) + self._profile(lam, 15.5, 1.3)
        assert opacity_integral(lam, tau, 14.0, 16.0) == pytest.approx(1.5, rel=1e-6, abs=0)

    def test_no_unit_conversion_and_order_independent(self):
        lam = np.linspace(14.9, 15.1, 2001)
        tau = self._profile(lam)
        assert opacity_integral(lam[::-1], tau[::-1], 14.9, 15.1) == pytest.approx(0.5, rel=1e-6, abs=0)

    @pytest.mark.parametrize("lo, hi", [(20.0, 21.0), (15.0, 15.00001)])
    def test_fewer_than_two_points_is_zero(self, lo, hi):
        lam = np.linspace(14.9, 15.1, 201)
        assert opacity_integral(lam, self._profile(lam), lo, hi) == 0.0


# ======================================================================
# Against a real model Spectrum
# ======================================================================

def _spectrum(n_mol=1e18):
    from iSLAT.Modules.DataTypes.Spectrum import Spectrum
    inten = _make_test_intensity(n_lines=5, lam_start=12.0, n_mol=n_mol)
    spec = Spectrum(lam_min=11.0, lam_max=17.5, dlambda=0.0005, R=3000.0, distance=160.0)
    spec.add_intensity(inten, np.pi * 1.0 ** 2)
    return spec, inten


class TestAgainstModelSpectrum:
    def test_opacity_integral_is_summed_line_center_tau(self):
        spec, inten = _spectrum(n_mol=1e20)
        assert opacity_integral(spec._lamgrid, spec.tau_profile, 11.0, 17.5) == pytest.approx(
            float(np.sum(inten.tau)), rel=1e-4, abs=0)
        # A range around one line (12, 13, ... um) picks up just that line's tau
        assert opacity_integral(spec._lamgrid, spec.tau_profile, 13.9, 14.1) == pytest.approx(
            float(inten.tau[2]), rel=1e-4, abs=0)

    def test_flux_integral_matches_integrated_f_lambda(self):
        spec, _ = _spectrum()
        f_lambda_integral = np.trapezoid(spec.flux, x=spec._lamgrid)  # erg/s/cm^2
        result, _ = flux_integral(spec._lamgrid, spec.flux_jy, None, 11.0, 17.5)
        assert result == pytest.approx(f_lambda_integral, rel=1e-4, abs=0)
        m = (spec._lamgrid >= 12.98) & (spec._lamgrid <= 13.1)  # line at 13 um, off-centre
        result, _ = flux_integral(spec._lamgrid, spec.flux_jy, None, 12.98, 13.1)
        assert result == pytest.approx(np.trapezoid(spec.flux[m], x=spec._lamgrid[m]), rel=1e-4, abs=0)


# ======================================================================
# The line-information printout of the views
# ======================================================================

def _islat_for_views():
    mol = _make_test_molecule()
    lam_data, flux_data = mol.get_flux(return_wavelengths=True)
    data_field = MagicMock()
    islat = SimpleNamespace(
        wave_data=lam_data, flux_data=flux_data, err_data=None,
        active_molecule=mol, GUI=SimpleNamespace(data_field=data_field),
    )
    return islat, mol, data_field


_LINE = {"lam": 14.0, "e_up": 2000.0, "a_stein": 0.02, "g_up": 7, "intensity": 1e-3, "tau": 0.1,
         "up_lev": "0|6", "low_lev": "0|5"}


def _expected(mol, lo, hi):
    lam_t, tau = mol.get_tau(return_wavelengths=True)
    lam_f, flux = mol.get_flux(return_wavelengths=True)
    return opacity_integral(lam_t, tau, lo, hi), flux_integral(lam_f, flux, None, lo, hi)[0]


def _printed(data_field):
    return data_field.insert_text.call_args[0][0]


class TestViewPrintout:
    def test_three_panel_view(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        islat, mol, data_field = _islat_for_views()
        view = SimpleNamespace(_islat=islat, _pm=SimpleNamespace(toggle_state={"current_selection": (13.9, 14.1)}))
        ThreePanelView._display_line_info(view, dict(_LINE))
        text = _printed(data_field)
        tau_sum, model_flux = _expected(mol, 13.9, 14.1)
        assert f"Model opacity in range (sum of line-center opacities) = {tau_sum:.4g}" in text
        assert f"Model flux in range (erg/s/cm2) = {model_flux:.3e}" in text
        assert tau_sum == pytest.approx(float(mol.intensity.tau[2]), rel=1e-2, abs=0)

    def test_three_panel_view_without_selection(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        islat, _, data_field = _islat_for_views()
        view = SimpleNamespace(_islat=islat, _pm=SimpleNamespace(toggle_state={"current_selection": None}))
        ThreePanelView._display_line_info(view, dict(_LINE))  # used to raise UnboundLocalError
        assert "Model opacity in range (sum of line-center opacities) = N/A" in _printed(data_field)

    def test_line_inspection_view(self):
        from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView
        islat, mol, data_field = _islat_for_views()
        view = SimpleNamespace(_islat=islat, _current_selection=(13.9, 14.1))
        LineInspectionView._display_line_info(view, dict(_LINE))
        tau_sum, _ = _expected(mol, 13.9, 14.1)
        assert f"Model opacity in range (sum of line-center opacities) = {tau_sum:.4g}" in _printed(data_field)
