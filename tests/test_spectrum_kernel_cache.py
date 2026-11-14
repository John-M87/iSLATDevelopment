# -*- coding: utf-8 -*-
"""Spectrum's shared convolution-kernel cache gives bit-identical results.

A GUI parameter change (T, N, dv, radius) builds a new Spectrum, but the
Gaussian kernel of every line depends only on the line wavelengths, the
kernel widths and the grid.  Spectrum therefore keeps built kernels in a
process-wide, content-keyed cache and later spectra only multiply them by
their new line weights.

A kernel is built when it is requested a second time by another model: a
new Spectrum, or the same one after ``reset()`` / new intensities (the
slab-fit loop).  The flux and optical depth of one model count as a single
request, so kernels of a one-off width (an FWHM slider drag) are not kept.
Kernels below ``_CACHE_MIN_ELEMENTS`` are never cached.

Every path - first evaluation, the build, cache hits, non-finite and
negative weights, lines past the grid ends, Keplerian broadening (folded
into the GUI's R(lambda) or into a scalar R), a callable R that writes into
its argument - must equal the original per-group bincount loop
(``Spectrum._convolve_lines_reference``) bit for bit.  Removing molecules or
replacing a line list frees the cache.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Spectrum import Spectrum

spectrum_mod = importlib.import_module("iSLAT.Modules.DataTypes.Spectrum")

GRID = dict(lam_min=4.9, lam_max=27.9, dlambda=0.0013, distance=140.0)


@pytest.fixture(autouse=True)
def fresh_cache():
    """Every test starts and ends with an empty cache at the default budget."""
    Spectrum.clear_convolution_cache()
    Spectrum.set_convolution_cache_limit(spectrum_mod._KERNEL_CACHE_DEFAULT_BYTES)
    yield
    Spectrum.clear_convolution_cache()
    Spectrum.set_convolution_cache_limit(spectrum_mod._KERNEL_CACHE_DEFAULT_BYTES)


def _lines(n=4000, seed=3):
    """Line list spanning (and passing) both grid ends, with duplicated wavelengths."""
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(4.85, 27.95, n))
    lam[n // 2:n // 2 + 5] = lam[n // 2]
    return lam, rng.lognormal(-10, 3, n), rng.lognormal(-2, 2, n)


def _intensity(lam, inten, tau):
    return SimpleNamespace(
        molecule=SimpleNamespace(name="X", fname="", get_wavelengths=lambda: lam),
        intensity=inten, tau=tau, t_kin=500.0, n_mol=1e18, dv=1.0,
    )


def _spectrum(lines=None, scale=1.0, R=2500.0, R_func=None):
    lam, inten, tau = lines if lines is not None else _lines()
    s = Spectrum(R=R, R_func=R_func, **GRID)
    s.add_intensity(_intensity(lam, inten * scale, tau), 1.3)
    return s


def _expected(s):
    """Flux and both tau profiles from the original loop, for the spectrum's lines."""
    lam_all = np.concatenate(s._lam_arrays)
    lam, inv = np.unique(lam_all, return_inverse=True)
    sigma = s._line_sigma(lam.copy())
    intens = np.bincount(inv, weights=np.concatenate(s._I_arrays), minlength=lam.size).astype(np.float64)
    tau = np.bincount(inv, weights=np.concatenate(s._tau_arrays), minlength=lam.size).astype(np.float64)
    flux = s._convolve_lines_reference(lam, intens, sigma, "area")
    flux = flux * (s._AU_PC_RATIO_SQ * (1.0 / (s._distance ** 2)))
    return (flux,
            s._convolve_lines_reference(lam, tau, sigma, "area"),
            s._convolve_lines_reference(lam, tau, sigma, "peak"))


def _assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _assert_matches_reference(s):
    flux, tau_area, tau_peak = _expected(s)
    _assert_bits_equal(s.flux, flux)
    _assert_bits_equal(s.tau_profile, tau_area)
    _assert_bits_equal(s.tau_peak_profile, tau_peak)


@pytest.fixture
def builds(monkeypatch):
    """Count kernel builds."""
    calls = []
    original = Spectrum._build_kernel_blocks

    def counting(self, norm, groups):
        calls.append(1)
        return original(self, norm, groups)

    monkeypatch.setattr(Spectrum, "_build_kernel_blocks", counting)
    return calls


class TestBitIdentical:
    def test_first_evaluation_build_and_hits_match_the_original_loop(self, builds):
        lines = _lines()
        for scale in (1.0, 2.5, 0.3, 7.0):            # miss, build, hit, hit
            _assert_matches_reference(_spectrum(lines, scale))
        assert builds == [1]
        assert Spectrum.convolution_cache_info()["kernels"] == 1

    def test_per_component_profiles_match(self):
        lam, inten, tau = _lines()
        for _ in range(3):
            s = Spectrum(R=2500.0, **GRID)
            s.add_intensity(_intensity(lam, inten, tau), 1.0)
            s.add_intensity(_intensity(lam[::4].copy(), inten[::4], tau[::4] * 3), 0.5)
            for norm in ("area", "peak"):
                for comp, (c_lam, c_tau) in zip(s._convol_tau_per_component(norm),
                                                zip(s._lam_arrays, s._tau_arrays)):
                    u, inv = np.unique(c_lam, return_inverse=True)
                    w = np.bincount(inv, weights=c_tau, minlength=u.size).astype(np.float64)
                    expected = s._convolve_lines_reference(u, w, s._line_sigma(u), norm)
                    _assert_bits_equal(comp["tau_profile"], expected)

    def test_non_finite_and_negative_weights_match(self):
        lam, inten, tau = _lines()
        inten = inten.copy()
        inten[[3, 300, 3000]] = [np.nan, np.inf, -np.inf]
        inten[[4, 5]] = [-2e-5, -0.0]
        for _ in range(3):
            _assert_matches_reference(_spectrum((lam, inten, tau)))

    def test_wavelength_dependent_R_matches(self):
        r_func = lambda lam: 1500.0 + 100.0 * np.asarray(lam)
        lines = _lines()
        for _ in range(3):
            _assert_matches_reference(_spectrum(lines, R_func=r_func))

    @pytest.mark.parametrize("profile", ["constant", "miri_mrs"])
    def test_keplerian_broadening_matches(self, profile, builds):
        """The GUI's R(lambda) for an instrumental profile plus Keplerian broadening."""
        from iSLAT.Modules.DataProcessing.InstrumentalProfiles import PROFILE_REGISTRY
        cls = PROFILE_REGISTRY[profile]
        prof = cls(130.0) if profile == "constant" else cls()
        lines = _lines()
        for kep in (40.0, 40.0, 40.0, 75.0, 75.0):        # miss, build, hit; new width: miss, build
            r_func = Spectrum.build_R_func(prof, kep, 130.0)
            _assert_matches_reference(_spectrum(lines, R_func=r_func))
        assert builds == [1, 1]

    def test_keplerian_on_a_scalar_R_matches(self):
        """Spectrum(keplerian_fwhm=...) folds the broadening into a scalar R."""
        lam, inten, tau = _lines()
        for _ in range(3):
            s = Spectrum(R=2500.0, keplerian_fwhm=40.0, **GRID)
            s.add_intensity(_intensity(lam, inten, tau), 1.3)
            _assert_matches_reference(s)

    def test_R_func_writing_into_its_argument_does_not_corrupt_the_cache(self):
        def mutating(lam):
            lam *= 1.0000001
            return np.full(lam.shape, 2400.0)

        lines = _lines()
        first = [_spectrum(lines, R_func=mutating).flux for _ in range(4)]
        for flux in first[1:]:
            _assert_bits_equal(flux, first[0])

    def test_disabled_cache_still_matches(self, builds):
        Spectrum.set_convolution_cache_limit(0)
        lines = _lines()
        for _ in range(3):
            _assert_matches_reference(_spectrum(lines))
        assert builds == []


class TestWhenKernelsAreKept:
    def test_flux_and_tau_of_one_model_count_as_one_request(self, builds):
        s = _spectrum()
        s.flux, s.tau_profile, s.tau_peak_profile
        assert builds == []
        assert Spectrum.convolution_cache_info()["kernels"] == 0

    def test_one_off_width_is_not_kept(self, builds):
        lines = _lines()
        _spectrum(lines).flux
        for R in (2400.0, 2300.0, 2200.0):                 # FWHM slider: every width once
            s = _spectrum(lines, R=R)
            s.flux, s.tau_profile
        assert builds == []

    def test_same_spectrum_after_reset_builds_on_its_second_evaluation(self, builds):
        lam, inten, tau = _lines()
        s = Spectrum(R=2500.0, **GRID)
        for k in range(3):                                 # slab-fit loop: one Spectrum, reset()
            s.reset()
            s.add_intensity(_intensity(lam, inten * (k + 1), tau), 1.0)
            _assert_matches_reference(s)
        assert builds == [1]

    def test_small_kernels_are_never_cached(self, builds):
        lam, inten, tau = _lines(n=40)
        for _ in range(3):
            _spectrum((lam, inten, tau)).flux
        assert builds == []
        assert Spectrum.convolution_cache_info()["pending"] == 0

    def test_budget_is_respected(self):
        lines = _lines()
        for _ in range(2):
            _spectrum(lines).flux
        size = Spectrum.convolution_cache_info()["kernel_bytes"]
        assert size > 0
        Spectrum.set_convolution_cache_limit(size - 1)
        assert Spectrum.convolution_cache_info()["kernels"] == 0


class TestCacheIsFreed:
    def _fill(self):
        lines = _lines()
        for _ in range(2):
            _spectrum(lines).flux
        assert Spectrum.convolution_cache_info()["kernels"] == 1

    @pytest.fixture
    def md(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict()
        yield md
        Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)

    def test_removing_a_molecule(self, md):
        md["A"] = object()
        self._fill()
        del md["A"]
        assert Spectrum.convolution_cache_info()["kernels"] == 0

    def test_clearing_all_molecules(self, md):
        md["A"] = object()
        self._fill()
        md.clear()
        assert Spectrum.convolution_cache_info()["kernels"] == 0

    def test_replacing_a_line_list(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        mol = Molecule(name="X", displaylabel="X", filepath=None, temp=500.0, radius=1.0,
                       n_mol=1e18, distance=140.0, fwhm=130.0,
                       initial_molecule_parameters={"t_kin": 500.0, "scale_exponent": 18.0,
                                                    "scale_number": 1.0, "radius_init": 1.0})
        self._fill()
        mol.change_line_list("not_read_until_used.par")
        assert Spectrum.convolution_cache_info()["kernels"] == 0
