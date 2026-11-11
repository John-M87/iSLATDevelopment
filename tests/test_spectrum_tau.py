# -*- coding: utf-8 -*-
"""Unit tests for Spectrum tau convolution and tau_profile property."""

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Spectrum import Spectrum


class TestSpectrumTauProfile:
    """Tests for _convol_tau, tau_profile, and _convol_tau_per_component."""

    def _make_spectrum(self, lam_min=10.0, lam_max=20.0, dlambda=0.01,
                       R=3000.0, distance=160.0):
        return Spectrum(
            lam_min=lam_min, lam_max=lam_max, dlambda=dlambda,
            R=R, distance=distance,
        )

    def _inject_component(self, spec, lam_array, tau_array, name="mol"):
        """Directly inject a component into the spectrum's internal arrays."""
        lam_arr = np.asarray(lam_array)
        tau_arr = np.asarray(tau_array)
        spec._lam_arrays.append(lam_arr)
        spec._tau_arrays.append(tau_arr)
        # I_arrays must stay in sync even though we don't use flux here
        spec._I_arrays.append(np.zeros_like(tau_arr))
        spec._components.append({
            'name': name, 'fname': '', 't_kin': 500.0,
            'n_mol': 1e15, 'dv': 5.0, 'tau': tau_arr, 'area': 1.0,
        })
        spec._invalidate_flux_cache()

    # ------------------------------------------------------------------
    def test_tau_profile_empty(self):
        spec = self._make_spectrum()
        tp = spec.tau_profile
        assert isinstance(tp, np.ndarray)
        assert len(tp) == len(spec.lamgrid)
        np.testing.assert_array_equal(tp, 0.0)

    def test_tau_profile_shape(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [1.0])
        tp = spec.tau_profile
        assert tp.shape == spec.lamgrid.shape

    def test_tau_profile_nonzero_at_line(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [5.0])
        tp = spec.tau_profile
        # The profile should peak near 15.0 μm
        idx_peak = np.argmax(tp)
        peak_lam = spec.lamgrid[idx_peak]
        assert abs(peak_lam - 15.0) < 0.1

    def test_tau_profile_dimensionless(self):
        """Tau should NOT be scaled by distance."""
        spec1 = self._make_spectrum(distance=160.0)
        spec2 = self._make_spectrum(distance=320.0)
        self._inject_component(spec1, [15.0], [3.0])
        self._inject_component(spec2, [15.0], [3.0])
        np.testing.assert_array_almost_equal(
            spec1.tau_profile, spec2.tau_profile,
        )

    def test_tau_profile_caching(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [1.0])
        tp1 = spec.tau_profile
        tp2 = spec.tau_profile
        assert tp1 is tp2  # Same cached object

    def test_tau_profile_invalidation_on_reset(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [1.0])
        _ = spec.tau_profile
        spec.reset()
        tp = spec.tau_profile
        np.testing.assert_array_equal(tp, 0.0)

    def test_tau_profile_multiple_lines(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [12.0, 18.0], [2.0, 3.0])
        tp = spec.tau_profile
        # Should have two peaks
        mask_12 = (spec.lamgrid >= 11.5) & (spec.lamgrid <= 12.5)
        mask_18 = (spec.lamgrid >= 17.5) & (spec.lamgrid <= 18.5)
        assert np.max(tp[mask_12]) > 0
        assert np.max(tp[mask_18]) > 0

    def test_tau_additivity(self):
        """Tau from two components should add to total."""
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [2.0], name="A")
        self._inject_component(spec, [15.0], [3.0], name="B")
        total = spec.tau_profile

        # Build separate spectra for each
        spec_a = self._make_spectrum()
        self._inject_component(spec_a, [15.0], [2.0], name="A")
        spec_b = self._make_spectrum()
        self._inject_component(spec_b, [15.0], [3.0], name="B")

        np.testing.assert_array_almost_equal(
            total, spec_a.tau_profile + spec_b.tau_profile, decimal=10,
        )

    # ------------------------------------------------------------------
    # Per-component convolution
    # ------------------------------------------------------------------
    def test_per_component_empty(self):
        spec = self._make_spectrum()
        result = spec._convol_tau_per_component()
        assert result == []

    def test_per_component_single(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [2.0], name="H2O")
        result = spec._convol_tau_per_component()
        assert len(result) == 1
        assert result[0]['name'] == "H2O"
        assert result[0]['tau_profile'].shape == spec.lamgrid.shape
        np.testing.assert_array_almost_equal(
            result[0]['tau_profile'], spec.tau_profile, decimal=10,
        )

    def test_per_component_sum_equals_total(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [13.0], [1.5], name="A")
        self._inject_component(spec, [17.0], [2.5], name="B")
        total = spec.tau_profile
        per_comp = spec._convol_tau_per_component()
        summed = sum(c['tau_profile'] for c in per_comp)
        np.testing.assert_array_almost_equal(total, summed, decimal=10)

    def test_per_component_names(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [12.0], [1.0], name="H2O")
        self._inject_component(spec, [16.0], [1.0], name="CO2")
        result = spec._convol_tau_per_component()
        names = [r['name'] for r in result]
        assert names == ["H2O", "CO2"]

    # ------------------------------------------------------------------
    # Peak-normalized profile (dimensionless tau)
    # ------------------------------------------------------------------
    def _on_grid_line(self, R=3000.0, tau0=2.5):
        """A fine-grid spectrum with one line exactly on a grid point; returns (spec, lam0, sigma)."""
        spec = self._make_spectrum(lam_min=14.9, lam_max=15.1, dlambda=1e-5, R=R)
        lam0 = float(spec.lamgrid[np.argmin(np.abs(spec.lamgrid - 15.0))])
        self._inject_component(spec, [lam0], [tau0])
        sigma = lam0 / R / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return spec, lam0, sigma

    def test_peak_profile_peaks_at_line_center_tau(self):
        spec, _lam0, _sigma = self._on_grid_line(tau0=2.5)
        assert spec.tau_peak_profile.max() == pytest.approx(2.5, rel=1e-12)

    def test_area_profile_is_tau_per_micron(self):
        """The area profile is the peak profile / (sqrt(2 pi) sigma) and integrates to tau."""
        spec, _lam0, sigma = self._on_grid_line(tau0=2.5)
        np.testing.assert_allclose(
            spec.tau_profile, spec.tau_peak_profile / (np.sqrt(2.0 * np.pi) * sigma), rtol=1e-12,
        )
        assert np.trapezoid(spec.tau_profile, spec.lamgrid) == pytest.approx(2.5, rel=1e-6)

    def test_peak_height_does_not_depend_on_resolution(self):
        """Only the per-micron profile's height scales with the resolving power."""
        lo, _, _ = self._on_grid_line(R=1500.0)
        hi, _, _ = self._on_grid_line(R=3000.0)
        assert lo.tau_peak_profile.max() == pytest.approx(hi.tau_peak_profile.max(), rel=1e-12)
        assert hi.tau_profile.max() == pytest.approx(2.0 * lo.tau_profile.max(), rel=1e-6)

    def test_get_tau_profile_dispatches_on_normalization(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [1.0])
        assert spec.get_tau_profile("peak") is spec.tau_peak_profile
        assert spec.get_tau_profile("area") is spec.tau_profile
        assert spec.get_tau_profile() is spec.tau_profile
        with pytest.raises(ValueError, match="normalization"):
            spec.get_tau_profile("bogus")
        with pytest.raises(ValueError, match="normalization"):
            spec._convol_tau_per_component("bogus")

    def test_peak_profile_invalidated_with_the_spectrum(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [15.0], [1.0])
        first = spec.tau_peak_profile
        assert spec.tau_peak_profile is first  # cached
        self._inject_component(spec, [15.0], [1.0])  # invalidates the caches
        assert spec.tau_peak_profile.max() == pytest.approx(2.0 * first.max())
        spec.reset()
        np.testing.assert_array_equal(spec.tau_peak_profile, 0.0)

    def test_per_component_peak_sum_equals_total(self):
        spec = self._make_spectrum()
        self._inject_component(spec, [13.0], [1.5], name="A")
        self._inject_component(spec, [13.0, 17.0], [0.5, 2.5], name="B")
        summed = sum(c['tau_profile'] for c in spec._convol_tau_per_component("peak"))
        np.testing.assert_allclose(summed, spec.tau_peak_profile, rtol=1e-12)
