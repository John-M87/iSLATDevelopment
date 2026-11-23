# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.DataProcessing.chi2spectrum.

Tests the Chi2Spectrum class and FluxMeasurement namedtuple.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from iSLAT.Modules.DataProcessing.chi2spectrum import Chi2Spectrum, FluxMeasurement


# ============================================================================
# FluxMeasurement namedtuple
# ============================================================================

class TestFluxMeasurement:
    """Tests for the FluxMeasurement namedtuple."""

    def test_fields(self):
        fm = FluxMeasurement(lam_min=4.6, lam_max=4.7, flux=2e-16, flux_error=1e-17)
        assert fm.lam_min == 4.6
        assert fm.lam_max == 4.7
        assert fm.flux == 2e-16
        assert fm.flux_error == 1e-17

    def test_unpacking(self):
        fm = FluxMeasurement(4.6, 4.7, 2e-16, 1e-17)
        lam_min, lam_max, flux, flux_error = fm
        assert lam_min == 4.6
        assert flux == 2e-16

    def test_immutable(self):
        fm = FluxMeasurement(4.6, 4.7, 2e-16, 1e-17)
        with pytest.raises(AttributeError):
            fm.flux = 5e-16


# ============================================================================
# Chi2Spectrum — construction and measurements
# ============================================================================

class TestChi2SpectrumMeasurements:
    """Tests for adding and managing measurements."""

    def test_empty_on_creation(self):
        chi2 = Chi2Spectrum()
        assert chi2.measurements == []
        assert chi2.chi2 == []
        assert chi2.chi2_total == 0

    def test_add_single_measurement(self):
        chi2 = Chi2Spectrum()
        fm = FluxMeasurement(4.6, 4.7, 2e-16, 1e-17)
        chi2.add_measurement(fm)
        assert len(chi2.measurements) == 1
        assert chi2.measurements[0] is fm

    def test_add_multiple_measurements(self):
        chi2 = Chi2Spectrum()
        for i in range(5):
            chi2.add_measurement(FluxMeasurement(
                lam_min=4.0 + i * 0.1,
                lam_max=4.0 + (i + 1) * 0.1,
                flux=1e-16 * (i + 1),
                flux_error=1e-17,
            ))
        assert len(chi2.measurements) == 5

    def test_measurements_preserve_order(self):
        chi2 = Chi2Spectrum()
        fluxes = [1e-16, 3e-16, 2e-16]
        for i, f in enumerate(fluxes):
            chi2.add_measurement(FluxMeasurement(4.0 + i * 0.1, 4.1 + i * 0.1, f, 1e-17))
        assert [m.flux for m in chi2.measurements] == fluxes


# ============================================================================
# Chi2Spectrum — evaluate_spectrum
# ============================================================================

class TestChi2SpectrumEvaluate:
    """Tests for chi-squared evaluation."""

    @staticmethod
    def _make_mock_spectrum(lam, flux_ergscm2, flux_jy=None):
        """Create a mock spectrum with lamgrid, flux, and flux_jy attributes."""
        spec = MagicMock()
        spec.lamgrid = lam
        spec.flux = flux_ergscm2
        spec.flux_jy = flux_jy if flux_jy is not None else flux_ergscm2
        return spec

    def test_zero_chi2_perfect_match(self):
        """If model flux integral matches measurement exactly, chi2 should be ~0."""
        lam = np.linspace(4.0, 5.0, 1000)
        # Constant flux of 1e-16 erg/s/cm2
        flux = np.full_like(lam, 1e-16)

        chi2 = Chi2Spectrum()
        # The integral of constant flux over [4.4, 4.6] = 1e-16 * (4.6 - 4.4) = 2e-17
        # But np.trapezoid integrates flux vs lambda so result = flux_value * delta_lam
        # For this range, integral = 1e-16 * 0.2 = 2e-17

        # Compute expected integral
        mask = (lam > 4.4) & (lam < 4.6)
        expected_flux = np.trapezoid(flux[mask], x=lam[mask])

        chi2.add_measurement(FluxMeasurement(
            lam_min=4.4, lam_max=4.6,
            flux=expected_flux,
            flux_error=1e-18,
        ))
        spec = self._make_mock_spectrum(lam, flux)
        chi2.evaluate_spectrum(spec)

        assert chi2.chi2_total == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_chi2(self):
        """Model that doesn't match data should give positive chi2."""
        lam = np.linspace(4.0, 5.0, 1000)
        flux = np.full_like(lam, 1e-16)

        chi2 = Chi2Spectrum()
        # Intentionally wrong flux measurement
        chi2.add_measurement(FluxMeasurement(
            lam_min=4.4, lam_max=4.6,
            flux=9.99e-10,  # way off
            flux_error=1e-18,
        ))
        spec = self._make_mock_spectrum(lam, flux)
        chi2.evaluate_spectrum(spec)

        assert chi2.chi2_total > 0

    def test_chi2_multiple_measurements(self):
        """chi2_total should be sum of individual chi2 values."""
        lam = np.linspace(4.0, 5.0, 1000)
        flux = np.full_like(lam, 1e-16)

        chi2 = Chi2Spectrum()
        chi2.add_measurement(FluxMeasurement(4.2, 4.3, 5e-16, 1e-17))
        chi2.add_measurement(FluxMeasurement(4.5, 4.7, 5e-16, 1e-17))
        chi2.add_measurement(FluxMeasurement(4.8, 4.9, 5e-16, 1e-17))

        spec = self._make_mock_spectrum(lam, flux)
        chi2.evaluate_spectrum(spec)

        assert len(chi2.chi2) == 3
        individual_sum = sum(c.chi2 for c in chi2.chi2)
        assert chi2.chi2_total == pytest.approx(individual_sum)

    def test_chi2_comparison_fields(self):
        """Each chi2 entry should have lam_min, lam_max, flux, flux_error, flux_model, chi2."""
        lam = np.linspace(4.0, 5.0, 500)
        flux = np.ones_like(lam) * 1e-16

        chi2 = Chi2Spectrum()
        chi2.add_measurement(FluxMeasurement(4.2, 4.4, 1e-16, 1e-17))

        spec = self._make_mock_spectrum(lam, flux)
        chi2.evaluate_spectrum(spec)

        entry = chi2.chi2[0]
        assert hasattr(entry, 'lam_min')
        assert hasattr(entry, 'lam_max')
        assert hasattr(entry, 'flux')
        assert hasattr(entry, 'flux_error')
        assert hasattr(entry, 'flux_model')
        assert hasattr(entry, 'chi2')
        assert entry.lam_min == 4.2
        assert entry.lam_max == 4.4

    def test_evaluate_with_jy_units(self):
        """flux_unit='jy' should use spectrum.flux_jy."""
        lam = np.linspace(4.0, 5.0, 500)
        flux_erg = np.ones_like(lam) * 1e-16
        flux_jy = np.ones_like(lam) * 5.0  # Different from erg

        chi2 = Chi2Spectrum()
        mask = (lam > 4.4) & (lam < 4.6)
        expected_jy_integral = np.trapezoid(flux_jy[mask], x=lam[mask])

        chi2.add_measurement(FluxMeasurement(4.4, 4.6, expected_jy_integral, 1e-2))
        spec = self._make_mock_spectrum(lam, flux_erg, flux_jy)
        chi2.evaluate_spectrum(spec, flux_unit="jy")

        assert chi2.chi2_total == pytest.approx(0.0, abs=1e-5)

    def test_evaluate_invalid_flux_unit_raises(self):
        """Invalid flux_unit should raise ValueError."""
        lam = np.linspace(4.0, 5.0, 100)
        flux = np.ones_like(lam)
        chi2 = Chi2Spectrum()
        chi2.add_measurement(FluxMeasurement(4.2, 4.4, 1e-16, 1e-17))
        spec = self._make_mock_spectrum(lam, flux)

        with pytest.raises(ValueError, match="Flux units not known"):
            chi2.evaluate_spectrum(spec, flux_unit="invalid")

    def test_evaluate_replaces_previous(self):
        """Calling evaluate_spectrum twice should replace previous chi2 data."""
        lam = np.linspace(4.0, 5.0, 500)
        flux = np.ones_like(lam) * 1e-16

        chi2 = Chi2Spectrum()
        chi2.add_measurement(FluxMeasurement(4.3, 4.5, 1e-15, 1e-17))

        spec = self._make_mock_spectrum(lam, flux)
        chi2.evaluate_spectrum(spec)
        first_total = chi2.chi2_total

        # Evaluate again — should produce the same result (not accumulate)
        chi2.evaluate_spectrum(spec)
        assert chi2.chi2_total == pytest.approx(first_total)
        assert len(chi2.chi2) == 1


# ============================================================================
# Chi2Spectrum — get_table
# ============================================================================

class TestChi2SpectrumTable:
    """Tests for the get_table property."""

    def test_get_table_returns_dataframe(self):
        pd = pytest.importorskip("pandas")

        lam = np.linspace(4.0, 5.0, 500)
        flux = np.ones_like(lam) * 1e-16

        chi2 = Chi2Spectrum()
        chi2.add_measurement(FluxMeasurement(4.3, 4.5, 1e-15, 1e-17))
        chi2.add_measurement(FluxMeasurement(4.6, 4.8, 2e-15, 2e-17))

        spec = MagicMock()
        spec.lamgrid = lam
        spec.flux = flux
        chi2.evaluate_spectrum(spec)

        table = chi2.get_table
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert set(table.columns) == {'lam_min', 'lam_max', 'flux', 'flux_error', 'flux_model', 'chi2'}

    def test_get_table_empty_before_evaluate(self):
        pd = pytest.importorskip("pandas")
        chi2 = Chi2Spectrum()
        table = chi2.get_table
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 0


class TestContributingMeasurements:
    """n_contributing counts the measurements that enter chi2 (non-zero error)."""

    def test_zero_error_measurements_are_not_counted(self):
        c2 = Chi2Spectrum()
        c2.add_measurement(FluxMeasurement(4.6, 4.7, 2e-16, 1e-17))
        c2.add_measurement(FluxMeasurement(4.7, 4.8, 2e-16, 0.0))
        c2.add_measurement(FluxMeasurement(4.8, 4.9, 3e-16, 2e-17))
        assert c2.n_contributing == 2

    def test_empty(self):
        assert Chi2Spectrum().n_contributing == 0
