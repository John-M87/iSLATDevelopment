# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.DataProcessing.spectral_utils.

Tests the flux-conserving spectral resampling functions: make_bins and spectres.
"""

import pytest
import numpy as np

from iSLAT.Modules.DataProcessing.spectral_utils import make_bins, spectres


# ============================================================================
# make_bins
# ============================================================================

class TestMakeBins:
    """Tests for the make_bins helper."""

    def test_uniform_grid_edges(self):
        """Uniform spacing should give symmetric half-step edges."""
        wavs = np.array([10.0, 11.0, 12.0, 13.0])
        edges, widths = make_bins(wavs)

        assert len(edges) == len(wavs) + 1
        assert len(widths) == len(wavs)
        np.testing.assert_allclose(edges, [9.5, 10.5, 11.5, 12.5, 13.5])
        np.testing.assert_allclose(widths, [1.0, 1.0, 1.0, 1.0])

    def test_non_uniform_grid(self):
        """Non-uniform spacing: edges should be midpoints of neighbours."""
        wavs = np.array([1.0, 2.0, 4.0])
        edges, widths = make_bins(wavs)

        assert edges[0] == pytest.approx(0.5)          # 1.0 - (2.0-1.0)/2
        assert edges[1] == pytest.approx(1.5)          # (1.0+2.0)/2
        assert edges[2] == pytest.approx(3.0)          # (2.0+4.0)/2
        assert edges[3] == pytest.approx(5.0)          # 4.0 + (4.0-2.0)/2
        np.testing.assert_allclose(widths, [1.0, 1.5, 2.0])

    def test_two_element_grid(self):
        """Minimum valid grid with two wavelengths."""
        wavs = np.array([5.0, 7.0])
        edges, widths = make_bins(wavs)

        assert len(edges) == 3
        assert len(widths) == 2
        assert edges[0] == pytest.approx(4.0)
        assert edges[1] == pytest.approx(6.0)
        assert edges[2] == pytest.approx(8.0)

    def test_bin_widths_sum_to_total_range(self):
        """Sum of all bin widths should equal total bin range (edge-to-edge)."""
        wavs = np.linspace(5.0, 25.0, 100)
        edges, widths = make_bins(wavs)
        np.testing.assert_allclose(widths.sum(), edges[-1] - edges[0], rtol=1e-12)

    def test_edges_monotonically_increasing(self):
        """Bin edges must be strictly increasing for a sorted input grid."""
        wavs = np.linspace(4.5, 28.0, 500)
        edges, _ = make_bins(wavs)
        assert np.all(np.diff(edges) > 0)

    def test_widths_positive(self):
        """All bin widths must be positive."""
        wavs = np.array([1.0, 1.5, 3.0, 7.0, 8.0])
        _, widths = make_bins(wavs)
        assert np.all(widths > 0)

    def test_widths_are_edge_differences(self):
        """Every width is the distance between its two edges."""
        wavs = np.sort(np.random.default_rng(0).uniform(5.0, 25.0, 300))
        edges, widths = make_bins(wavs)
        np.testing.assert_array_equal(widths, np.diff(edges))

    def test_accepts_lists_and_integers(self):
        """Array-like input of any numeric type gives float64 bins."""
        edges, widths = make_bins([1, 2, 4])
        assert edges.dtype == np.float64 and widths.dtype == np.float64
        np.testing.assert_allclose(edges, [0.5, 1.5, 3.0, 5.0])

    def test_input_not_modified(self):
        wavs = np.array([1.0, 2.0, 4.0, 4.5])
        before = wavs.copy()
        make_bins(wavs)
        np.testing.assert_array_equal(wavs, before)


# ============================================================================
# spectres — identity / constant tests
# ============================================================================

class TestSpectresBasic:
    """Basic correctness tests for spectres."""

    def test_identity_resampling(self):
        """Resampling onto the same grid should return identical fluxes."""
        wavs = np.linspace(10, 20, 100)
        flux = np.sin(wavs)
        result = spectres(wavs, wavs, flux)
        np.testing.assert_allclose(result, flux, atol=1e-10)

    def test_constant_spectrum_preserved(self):
        """A flat spectrum should remain flat after resampling."""
        old = np.linspace(10, 20, 200)
        new = np.linspace(11, 19, 50)
        flat = np.full_like(old, 42.0)
        result = spectres(new, old, flat)
        np.testing.assert_allclose(result, 42.0, atol=1e-10)

    def test_linear_spectrum_preserved(self):
        """A linear spectrum should be well preserved."""
        old = np.linspace(10, 20, 500)
        new = np.linspace(11, 19, 80)
        linear = 3.0 * old + 7.0
        result = spectres(new, old, linear)
        expected = 3.0 * new + 7.0
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_coarser_resampling(self):
        """Resampling to a coarser grid should still conserve a constant."""
        old = np.linspace(5, 25, 1000)
        new = np.linspace(6, 24, 20)
        flux = np.full_like(old, 100.0)
        result = spectres(new, old, flux)
        np.testing.assert_allclose(result, 100.0, atol=1e-8)

    def test_finer_resampling(self):
        """Resampling to a finer grid should preserve a constant."""
        old = np.linspace(10, 20, 50)
        new = np.linspace(11, 19, 500)
        flux = np.full_like(old, 7.5)
        result = spectres(new, old, flux)
        np.testing.assert_allclose(result, 7.5, atol=1e-8)


# ============================================================================
# spectres — fill / out-of-range
# ============================================================================

class TestSpectresFill:
    """Tests for out-of-range behavior and fill values."""

    def test_fill_default_zero(self):
        """Out-of-range bins should default to 0.0."""
        old = np.linspace(10, 20, 100)
        new = np.array([5.0, 15.0, 25.0])
        flux = np.ones_like(old)
        result = spectres(new, old, flux)
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_fill_custom_value(self):
        """Custom fill value should appear for out-of-range bins."""
        old = np.linspace(10, 20, 100)
        new = np.array([5.0, 15.0, 25.0])
        flux = np.ones_like(old)
        result = spectres(new, old, flux, fill=-999.0)
        assert result[0] == -999.0
        assert result[2] == -999.0
        np.testing.assert_allclose(result[1], 1.0, atol=1e-10)

    def test_all_out_of_range(self):
        """If all new wavelengths are outside old grid, all filled."""
        old = np.linspace(10, 20, 100)
        new = np.array([1.0, 2.0, 3.0])
        flux = np.ones_like(old)
        result = spectres(new, old, flux, fill=-1.0)
        np.testing.assert_array_equal(result, -1.0)

    def test_fill_nan(self):
        """NaN fill value should produce NaN for out-of-range bins."""
        old = np.linspace(10, 20, 100)
        new = np.array([5.0, 15.0])
        flux = np.ones_like(old)
        result = spectres(new, old, flux, fill=np.nan)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 1.0, atol=1e-10)

    def test_verbose_warning(self):
        """verbose=True should emit RuntimeWarning when fill is used."""
        old = np.linspace(10, 20, 100)
        new = np.array([1.0, 2.0])
        flux = np.ones_like(old)
        with pytest.warns(RuntimeWarning, match="spectres"):
            spectres(new, old, flux, verbose=True)

    def test_partial_overlap_left(self):
        """Only the left portion of new grid is out of range."""
        old = np.linspace(10, 20, 200)
        new = np.array([8.0, 9.0, 12.0, 15.0])
        flux = np.full_like(old, 5.0)
        result = spectres(new, old, flux, fill=0.0)
        assert result[0] == 0.0
        assert result[1] == 0.0
        np.testing.assert_allclose(result[2], 5.0, atol=1e-8)
        np.testing.assert_allclose(result[3], 5.0, atol=1e-8)


# ============================================================================
# spectres — flux conservation
# ============================================================================

class TestSpectresConservation:
    """Tests verifying flux-conservation properties."""

    def test_total_flux_conserved_gaussian(self):
        """Total integrated flux of a Gaussian should be roughly conserved."""
        old = np.linspace(10, 20, 1000)
        new = np.linspace(10.5, 19.5, 200)
        # Gaussian centered at 15 with sigma=1
        flux = np.exp(-0.5 * ((old - 15.0) / 1.0) ** 2)
        result = spectres(new, old, flux)

        # Integrated flux comparison (trapezoidal rule)
        old_integral = np.trapezoid(flux, old)
        # Only compare over the overlapping range
        old_mask = (old >= new[0]) & (old <= new[-1])
        old_trimmed_integral = np.trapezoid(flux[old_mask], old[old_mask])
        new_integral = np.trapezoid(result, new)
        np.testing.assert_allclose(new_integral, old_trimmed_integral, rtol=0.02)

    def test_delta_function_peak_preserved(self):
        """A narrow spike's peak region should still be non-zero after resampling."""
        old = np.linspace(10, 20, 500)
        flux = np.zeros_like(old)
        # Put a spike near 15.0
        spike_idx = np.argmin(np.abs(old - 15.0))
        flux[spike_idx] = 100.0
        new = np.linspace(10, 20, 200)
        result = spectres(new, old, flux)
        # The spike should appear somewhere in the result
        assert np.max(result) > 0

    def test_step_function(self):
        """A step function should produce a smooth transition when resampled."""
        old = np.linspace(10, 20, 1000)
        flux = np.where(old >= 15.0, 10.0, 0.0)
        new = np.linspace(10.5, 19.5, 100)
        result = spectres(new, old, flux)
        # Below 15 should be ~0, above should be ~10
        low_mask = new < 14.5
        high_mask = new > 15.5
        np.testing.assert_allclose(result[low_mask], 0.0, atol=1e-8)
        np.testing.assert_allclose(result[high_mask], 10.0, atol=0.5)


# ============================================================================
# spectres — return type and shape
# ============================================================================

class TestSpectresShape:
    """Shape and dtype tests for spectres output."""

    def test_output_length_matches_new_grid(self):
        """Output should always have the same length as new_wavs."""
        old = np.linspace(10, 20, 300)
        for n in [2, 5, 50, 300, 1000]:  # make_bins requires ≥ 2 points
            new = np.linspace(11, 19, n)
            flux = np.ones_like(old)
            result = spectres(new, old, flux)
            assert result.shape == (n,), f"Mismatch for n={n}"

    def test_output_dtype_float64(self):
        """Output should always be float64."""
        old = np.linspace(10, 20, 50)
        new = np.linspace(11, 19, 20)
        flux = np.ones_like(old, dtype=np.float32)
        result = spectres(new, old, flux)
        assert result.dtype == np.float64

    def test_accepts_lists(self):
        result = spectres([1.2, 1.5, 1.8], [1.0, 1.25, 1.5, 1.75, 2.0], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(result, [1.75, 3.0, 4.25])

    def test_inputs_not_modified(self):
        old = np.linspace(10, 20, 50)
        flux = np.cos(old)
        new = np.linspace(10.2, 19.8, 23)
        before = (old.copy(), flux.copy(), new.copy())
        spectres(new, old, flux)
        for arr, copy in zip((old, flux, new), before):
            np.testing.assert_array_equal(arr, copy)


# ============================================================================
# spectres - partial-bin weighting
# ============================================================================

def _reference_spectres(new_wavs, spec_wavs, spec_fluxes, fill=0.0):
    """Independent per-bin loop of the original SpectRes algorithm (Carnall 2017).

    Bin edges use the same formulas as ``make_bins``, so both sides make the
    same decisions for new edges that land exactly on old edges.
    """
    def bins(wavs):
        edges = np.empty(len(wavs) + 1)
        edges[1:-1] = (wavs[1:] + wavs[:-1]) / 2
        edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2
        edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2]) / 2
        return edges, np.diff(edges)

    old_edges, old_widths = bins(np.asarray(spec_wavs, dtype=float))
    new_edges, _ = bins(np.asarray(new_wavs, dtype=float))
    flux = np.asarray(spec_fluxes, dtype=float)
    out = np.full(len(new_wavs), fill, dtype=float)
    start = stop = 0
    for j in range(len(new_wavs)):
        lo, hi = new_edges[j], new_edges[j + 1]
        if lo < old_edges[0] or hi > old_edges[-1]:
            continue
        while old_edges[start + 1] <= lo:
            start += 1
        while old_edges[stop + 1] < hi:
            stop += 1
        if start == stop:
            out[j] = flux[start]
            continue
        w = old_widths[start:stop + 1].copy()
        w[0] *= (old_edges[start + 1] - lo) / (old_edges[start + 1] - old_edges[start])
        w[-1] *= (hi - old_edges[stop]) / (old_edges[stop + 1] - old_edges[stop])
        out[j] = np.sum(w * flux[start:stop + 1]) / np.sum(w)
    return out


def _sorted_grid(rng, lo, hi, n):
    return np.sort(rng.uniform(lo, hi, n))


_UNIFORM = np.linspace(10.0, 20.0, 301)

# (new_wavs, spec_wavs) pairs; random grids are non-uniform.
_GRIDS = {
    "finer new grid": lambda rng: (_sorted_grid(rng, 11, 19, 500), _sorted_grid(rng, 10, 20, 60)),
    "coarser new grid": lambda rng: (_sorted_grid(rng, 11, 19, 40), _sorted_grid(rng, 10, 20, 800)),
    "similar spacing": lambda rng: (_sorted_grid(rng, 10.5, 19.5, 200), _sorted_grid(rng, 10, 20, 210)),
    "partly outside both ends": lambda rng: (_sorted_grid(rng, 8, 22, 150), _sorted_grid(rng, 10, 20, 150)),
    "new edges on old edges": lambda rng: (_UNIFORM[1:-1:3], _UNIFORM),
    "old edges on new edges": lambda rng: (_UNIFORM, _UNIFORM[1:-1:3]),
    "same grid limits": lambda rng: (_UNIFORM[::2], _UNIFORM[::2]),
}


class TestSpectresWeighting:
    """Partial old bins must count with the fraction of them inside the new bin."""

    def test_hand_computed_weights(self):
        # Old bins 0.25 wide centred on 1.0 ... 2.0; new bins 0.3 wide centred on 1.2, 1.5, 1.8.
        result = spectres(np.array([1.2, 1.5, 1.8]), np.array([1.0, 1.25, 1.5, 1.75, 2.0]),
                          np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        expected = [
            (0.075 * 1 + 0.225 * 2) / 0.3,              # [1.05, 1.35]
            (0.025 * 2 + 0.25 * 3 + 0.025 * 4) / 0.3,   # [1.35, 1.65]
            (0.225 * 4 + 0.075 * 5) / 0.3,              # [1.65, 1.95]
        ]
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("grids", list(_GRIDS.values()), ids=list(_GRIDS))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference_loop(self, grids, seed):
        rng = np.random.default_rng(seed)
        new, old = grids(rng)
        flux = rng.normal(size=old.size) * 10.0 ** rng.integers(-6, 7)
        result = spectres(new, old, flux, fill=np.nan)
        expected = _reference_spectres(new, old, flux, fill=np.nan)
        np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
        scale = np.max(np.abs(flux))
        np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-11 * scale, equal_nan=True)

    def test_flux_conserved_exactly(self):
        """sum(new flux * new width) equals the integral of the old (bin-constant) spectrum."""
        rng = np.random.default_rng(3)
        old = _sorted_grid(rng, 10, 20, 700)
        flux = rng.uniform(0.0, 5.0, old.size)
        new = _sorted_grid(rng, 11, 19, 90)
        result = spectres(new, old, flux)
        old_edges, old_widths = make_bins(old)
        new_edges, new_widths = make_bins(new)
        # The old spectrum's running integral is linear between old edges, so interp is exact.
        running = np.concatenate(([0.0], np.cumsum(old_widths * flux)))
        expected = np.interp(new_edges[-1], old_edges, running) - np.interp(new_edges[0], old_edges, running)
        assert np.sum(result * new_widths) == pytest.approx(expected, rel=1e-12)


class TestSpectresLocality:
    """Each new bin depends only on the old bins it overlaps."""

    def test_nan_only_affects_bins_containing_it(self):
        old = np.linspace(10.0, 20.0, 201)
        flux = np.sin(old) + 2.0
        new = np.linspace(10.5, 19.5, 41)  # edges never coincide with old edges
        clean = spectres(new, old, flux)
        flux[100] = np.nan
        result = spectres(new, old, flux)
        old_edges, _ = make_bins(old)
        new_edges, _ = make_bins(new)
        touches = (new_edges[:-1] < old_edges[101]) & (new_edges[1:] > old_edges[100])
        assert touches.sum() == 1
        assert np.isnan(result[touches]).all()
        np.testing.assert_array_equal(result[~touches], clean[~touches])

    def test_bin_touching_nan_only_at_its_edge_is_unaffected(self):
        # Multiples of 1/8 are exact in binary, so the new edges land exactly
        # on old edges: each new bin covers exactly three whole old bins.
        old = 10.0 + 0.25 * np.arange(64)
        new = old[1:-1:3]                      # centred on old points 1, 4, ..., 61
        flux = np.arange(64, dtype=float)
        flux[[30, 32]] = np.nan                # both inside the new bin centred on old point 31
        result = spectres(new, old, flux)
        centres = np.arange(1, 62, 3)
        # The bins centred on old points 28 and 34 only touch old bins 30 / 32 at an edge.
        expected = np.where(centres == 31, np.nan, centres.astype(float))
        np.testing.assert_allclose(result, expected, rtol=1e-15, equal_nan=True)

    def test_faint_bins_next_to_bright_line_keep_precision(self):
        rng = np.random.default_rng(4)
        old = _sorted_grid(rng, 10, 20, 2000)
        flux = 1e-3 * (1.0 + rng.uniform(size=old.size))
        flux[np.abs(old - 11.0) < 0.05] = 1e6  # bright line before the faint part
        new = _sorted_grid(rng, 12, 19, 300)
        np.testing.assert_allclose(spectres(new, old, flux), _reference_spectres(new, old, flux), rtol=1e-12)

    def test_zero_flux_stays_exactly_zero(self):
        old = np.linspace(10.0, 20.0, 401)
        flux = np.where(np.abs(old - 15.0) < 0.1, 3.0, 0.0)  # one emission line
        new = np.linspace(10.3, 19.7, 53)
        result = spectres(new, old, flux)
        far = np.abs(new - 15.0) > 1.0
        assert np.all(result[far] == 0.0)
        assert result[~far].max() > 0.0
