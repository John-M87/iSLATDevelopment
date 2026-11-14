# -*- coding: utf-8 -*-

"""
This class Spectrum creates a spectrum from intensity instances

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* After a Spectrum instance is created with the wavelength grid and resolution, different intensity components can be
  added. When they are added, only the area-scaled intensities per line are stored. The convolution with the resolution
  of the instrument is done at the end (a la lazy evaluation at the first call to retrieve the flux). This improves the
  performance very much when different components of the same molecule are added.
* The get the spectral convolution of the raw lines acceptably quick, only a range of about 15 sigma around the line
  center is evaluated. This is the same trick as has been used in the Fortran 90 code.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

import itertools
import threading
from collections import OrderedDict

import numpy as np
from typing import Optional, List, Dict, Any, Union

import iSLAT.Constants as c

from ._pandas_import import get_pandas as _get_pandas
from ._mixins import CacheStatsMixin, WavelengthRangeMixin


# ---------------------------------------------------------------------------
# Convolution helpers (module level, shared by all Spectrum instances)
# ---------------------------------------------------------------------------
# Kernel elements evaluated per numpy block in the uncached path; a block of
# this size stays in L2/L3 so the memory-bound passes are cheap.
_CHUNK_ELEMENTS = 65536
# Kernels smaller than this are cheap enough to evaluate uncached every time.
_CACHE_MIN_ELEMENTS = 1 << 15


class _ByteLRU:
    """Thread-safe LRU (key -> value) bounded by an explicit byte count."""

    def __init__(self, max_bytes):
        self.max_bytes = int(max_bytes)
        self._entries = OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()

    def get(self, key, verify):
        """Return the value stored under *key* if ``verify(value)`` is true."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or not verify(entry[0]):
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key, value, nbytes):
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._nbytes -= old[1]
            if nbytes > self.max_bytes:
                return
            self._entries[key] = (value, nbytes)
            self._nbytes += nbytes
            while self._nbytes > self.max_bytes:
                _, (_, nb) = self._entries.popitem(last=False)
                self._nbytes -= nb

    def resize(self, max_bytes):
        with self._lock:
            self.max_bytes = int(max_bytes)
            while self._nbytes > self.max_bytes and self._entries:
                _, (_, nb) = self._entries.popitem(last=False)
                self._nbytes -= nb

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._nbytes = 0

    @property
    def nbytes(self):
        return self._nbytes

    def __len__(self):
        return len(self._entries)


# Gaussian line-spread kernels keyed by (grid, line wavelengths, sigma).  They
# do not depend on T, N, dv, radius or distance, so GUI parameter changes and
# slab-fit iterations reuse them.  8 bytes per kernel element (the 6 default
# molecules on the CITau MIRI range need ~65 MiB in total).
_KERNEL_CACHE_DEFAULT_BYTES = 128 * 2 ** 20
_KERNEL_CACHE = _ByteLRU(_KERNEL_CACHE_DEFAULT_BYTES)
# Kernels requested once so far (identity and requester only); a second
# request builds the kernel.  Kept apart so one-off kernels (FWHM slider,
# fits with a free FWHM) never evict built kernels.
_PENDING_KERNELS = _ByteLRU(16 * 2 ** 20)
# np.unique(..., return_inverse=True) of the concatenated line wavelengths.
_UNIQUE_CACHE = _ByteLRU(32 * 2 ** 20)
# Request tokens: a Spectrum takes a new one whenever its intensities change,
# so the flux and optical depth of the same model count as one request.
_REQUEST_TOKENS = itertools.count()


def _bits_equal(a, b):
    """Bitwise array equality (what comparing ``tobytes()`` does, without copies)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        as_int = np.dtype("i%d" % a.dtype.itemsize)
        return bool(np.array_equal(a.view(as_int), b.view(as_int)))
    return bool(np.array_equal(a, b))


def _fingerprint(a):
    """Cheap lookup key for a 1-D array (always verified with _bits_equal)."""
    step = max(1, a.shape[0] // 61)
    return (a.shape[0], a.dtype.str, hash(a[::step].tobytes()), hash(a[-1:].tobytes()))


def _read_only(a):
    """Mark *a* read-only (it is shared through a cache) and return it."""
    a.flags.writeable = False
    return a


def _kernel_block(lamgrid, n_grid, start, K, lam, neg_inv2s):
    """Gaussian of one chunk of lines (sorted by wavelength) as a (2K+1, n) block.

    Row r holds kernel offset K - r (descending offsets), column c line c:
    exp((lamgrid[start_c + 2K - r] - lambda_c)^2 * neg_inv2s_c).  For a fixed
    grid bin the elements of the block then appear in ascending line order,
    which is the order in which a line-major np.bincount adds them.  Returns
    the (clipped) grid indices, the kernel and the in-grid mask (None when no
    window of the chunk crosses a grid end)."""
    W = 2 * K + 1
    idx = start[np.newaxis, :] + np.arange(W - 1, -1, -1, dtype=np.intp)[:, np.newaxis]
    mask = None
    if start[0] < 0 or start[-1] + (W - 1) > n_grid - 1:
        mask = (idx >= 0) & (idx < n_grid)
        np.clip(idx, 0, n_grid - 1, out=idx)
    kern = lamgrid[idx]
    kern -= lam[np.newaxis, :]
    np.multiply(kern, kern, out=kern)
    kern *= neg_inv2s[np.newaxis, :]
    np.exp(kern, out=kern)
    return idx, kern, mask


class Spectrum(CacheStatsMixin, WavelengthRangeMixin):
    """
    Spectrum class for creating and managing spectral data from intensity instances.
    
    Implements lazy evaluation for convolution and caching for performance optimization.
    """
    
    # Pre-computed class-level constants for performance
    _FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548
    _INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)  # ≈ 0.3989
    _AU_PC_RATIO_SQ = (c.ASTRONOMICAL_UNIT_CM / c.PARSEC_CM) ** 2  # (AU/pc)²
    _FLUX_JY_FACTOR = 1e19 / c.SPEED_OF_LIGHT_CGS  # Conversion factor for Jy
    
    __slots__ = (
        '_lam_min', '_lam_max', '_dlambda', '_R', '_R_func', '_distance',
        '_keplerian_fwhm',
        '_lamgrid', '_flux', '_flux_jy', '_I_arrays', '_lam_arrays', '_tau_arrays',
        '_components', '_flux_valid', '_convolution_cache',
        '_kernel_cache', '_cache_stats', '_n_grid_points', '_unique_cache',
        '_wavelength_range', '_tau_profile', '_tau_profile_valid',
        '_tau_peak_profile', '_request_token',
    )
    
    def __init__(self, lam_min: float = None, lam_max: float = None, 
                 dlambda: float = None, R: float = None, distance: float = None,
                 wavelength_range: Optional[tuple] = None,
                 R_func: Optional[Any] = None,
                 keplerian_fwhm: float = 0.0):
        """Initialize a spectrum class and prepare it to add intensity components

        Parameters
        ----------
        lam_min: float
            Lower border of the spectrum in micron
        lam_max: float
            Upper border of the spectrum in micron
        dlambda: float
            Resolution of the spectrum to calculate in micron
        R: float
            Spectral resolution of the instrument as a scalar
            R = lambda/delta_lambda.  Used when *R_func* is not provided.
        distance: float
            Distance to the disk in pc
        wavelength_range: tuple of (float, float), optional
            Wavelength range ``(lam_min, lam_max)`` in microns.  When set,
            only lines within this range are included in the convolution.
            Defaults to ``(lam_min, lam_max)`` when not provided.
        R_func: callable, optional
            A function ``R_func(wavelength_um)`` that returns the
            wavelength-dependent resolving power *R* for one or more
            wavelengths (in microns).  When provided, *R_func* is used
            instead of the scalar *R* during convolution, giving each
            spectral line a kernel width that matches the local
            instrument resolution (e.g. JWST MIRI MRS).

            The callable must accept a float or 1-D ``numpy.ndarray``
            and return a value of the same shape.
        keplerian_fwhm : float, optional
            Keplerian line-of-sight broadening FWHM in km/s (default 0).
            When non-zero, this is combined in quadrature with the
            instrumental profile specified by *R* or *R_func*:

            .. math::
                \\Delta\\lambda_{\\rm total}(\\lambda) =
                    \\sqrt{\\Delta\\lambda_{\\rm inst}(\\lambda)^2
                         + \\Delta\\lambda_{\\rm kep}(\\lambda)^2}

            For a scalar *R* the result is still a scalar (wavelength-
            independent); for a callable *R_func* the result is a new
            callable that wraps the original.
        """

        # assure valid lambda grid range
        if lam_min >= lam_max:
            raise ValueError('lam_min must be < lam_max')

        # store parameters
        self._lam_min = lam_min
        self._lam_max = lam_max
        self._dlambda = dlambda
        self._R = R
        self._R_func = R_func
        self._distance = distance
        self._keplerian_fwhm = float(keplerian_fwhm)
        self._wavelength_range = wavelength_range if wavelength_range is not None else (lam_min, lam_max)

        # Apply Keplerian broadening on top of the instrumental profile
        if self._keplerian_fwhm > 0.0:
            if self._R_func is not None:
                # Wrap the existing wavelength-dependent R_func in quadrature
                _orig = self._R_func
                _kep = self._keplerian_fwhm
                def _kep_R_func(lam, _rfunc=_orig, _kep_kms=_kep):
                    lam = np.atleast_1d(np.asarray(lam, dtype=float))
                    R_inst = np.asarray(_rfunc(lam), dtype=float)
                    fwhm_inst_um = lam / R_inst
                    fwhm_kep_um  = lam * (_kep_kms / c.SPEED_OF_LIGHT_KMS)
                    fwhm_total   = np.sqrt(fwhm_inst_um ** 2 + fwhm_kep_um ** 2)
                    fwhm_total   = np.where(fwhm_total > 0, fwhm_total, fwhm_inst_um)
                    return lam / fwhm_total
                self._R_func = _kep_R_func
            elif self._R is not None and self._R > 0:
                # Scalar R path: combine constant FWHMs in quadrature → new scalar R
                # fwhm_inst_kms = c / R  (constant for all λ)
                fwhm_inst_kms = c.SPEED_OF_LIGHT_KMS / self._R
                fwhm_total_kms = (fwhm_inst_kms ** 2 + self._keplerian_fwhm ** 2) ** 0.5
                self._R = c.SPEED_OF_LIGHT_KMS / fwhm_total_kms

        # create wavelength grid with pre-calculated size
        n_points = int(1 + (lam_max - lam_min) / dlambda)
        self._lamgrid = np.linspace(lam_min, lam_max, n_points)
        self._n_grid_points = n_points  # Cache grid length

        # flux array (cached result)
        self._flux = None
        self._flux_jy = None

        # Use lists of numpy arrays for efficient accumulation
        # (concatenating arrays at convolution time is faster than extending lists)
        self._I_arrays = []
        self._lam_arrays = []
        self._tau_arrays = []

        # list with the different intensity components building up the spectrum
        self._components = []
        
        # Tau profiles (cached results; see tau_profile / tau_peak_profile)
        self._tau_profile = None
        self._tau_profile_valid = False
        self._tau_peak_profile = None

        # Cache invalidation flag and caching system
        self._flux_valid = False
        self._convolution_cache = {}
        self._kernel_cache = {}
        self._init_cache_stats()
        self._unique_cache = None  # Cached (lam_array, lam, index_wavelength) from np.unique
        # Identifies this spectrum's current intensities in kernel-cache requests.
        self._request_token = next(_REQUEST_TOKENS)

    @staticmethod
    def build_R_func(profile, kep_fwhm_kms: float, fwhm_fallback_kms: float):
        """Build an effective resolving-power callable that combines an instrumental
        profile with Keplerian broadening in quadrature.

        Parameters
        ----------
        profile:
            An instrumental-profile object exposing a ``get_R(wavelength_um)``
            method (e.g. instances from ``InstrumentalProfiles.PROFILE_REGISTRY``).
        kep_fwhm_kms : float
            Keplerian FWHM in km/s.  Pass ``0.0`` to include no Keplerian term.
        fwhm_fallback_kms : float
            Constant FWHM (km/s) used as a fallback when the profile returns
            NaN or non-positive R (e.g. wavelength outside instrument coverage).

        Returns
        -------
        R_func : callable
            A function ``R_func(lam)`` that accepts a float or 1-D
            ``numpy.ndarray`` of wavelengths in microns and returns the
            effective resolving power at each wavelength:

            .. math::
                R_{\\rm eff}(\\lambda) = \\frac{\\lambda}{\\Delta\\lambda_{\\rm total}}

            where

            .. math::
                \\Delta\\lambda_{\\rm total} =
                    \\sqrt{\\Delta\\lambda_{\\rm inst}^2 + \\Delta\\lambda_{\\rm kep}^2}
        rep_fwhm_kms : float
            A representative scalar FWHM (km/s) evaluated at the middle of the
            wavelength arrays, used by the caller to size the convolution grid.
            Returns ``fwhm_fallback_kms`` when the profile is out of range.
        """
        _profile = profile
        _kep = kep_fwhm_kms
        _R_fallback = c.SPEED_OF_LIGHT_KMS / fwhm_fallback_kms

        def R_func(lam):
            """Effective resolving power combining instrumental + Keplerian broadening.

            Wavelengths outside the instrument's coverage (R_inst = NaN or ≤ 0)
            fall back to the constant FWHM so that NaN never reaches _convol_flux.
            """
            lam = np.atleast_1d(np.asarray(lam, dtype=float))
            R_inst = np.asarray(_profile.get_R(lam), dtype=float)

            bad = ~np.isfinite(R_inst) | (R_inst <= 0)
            R_inst = np.where(bad, _R_fallback, R_inst)

            fwhm_inst_um = lam / R_inst
            fwhm_kep_um = lam * (_kep / c.SPEED_OF_LIGHT_KMS)
            fwhm_total_um = np.sqrt(fwhm_inst_um ** 2 + fwhm_kep_um ** 2)
            # Guard against zero total FWHM (shouldn't happen in practice)
            fwhm_total_um = np.where(fwhm_total_um > 0, fwhm_total_um, fwhm_inst_um)
            return lam / fwhm_total_um

        return R_func

    def reset(self):
        """Clear all accumulated intensity components, keeping the grid and kernel caches.

        This is much cheaper than constructing a new Spectrum when only the
        physical parameters (T, N, radius) change between evaluations,
        because the wavelength grid and convolution kernels are preserved.
        """
        self._I_arrays.clear()
        self._lam_arrays.clear()
        self._tau_arrays.clear()
        self._components.clear()
        self._flux = None
        self._flux_jy = None
        self._flux_valid = False
        self._tau_profile = None
        self._tau_profile_valid = False
        self._tau_peak_profile = None
        self._unique_cache = None
        self._request_token = next(_REQUEST_TOKENS)

    def add_intensity(self, intensity, dA: float):
        """Adds an intensity component to the spectrum

        Parameters
        ----------
        intensity: Intensity
            Intensity structure to add to the spectrum
        dA: float
            Area of the component in au**2
        """

        # Invalidate cached flux when adding new intensity
        self._invalidate_flux_cache()

        # 1. get intensity and wavelength of the lines
        I_all = intensity.intensity
        # Get wavelengths from the new MoleculeLineList structure
        lam_all = intensity.molecule.get_wavelengths()
        
        # Check if we have valid data to process
        if I_all is None or lam_all is None:
            print(f"Warning: No intensity or wavelength data available for molecule {intensity.molecule.name}")
            return
            
        if len(lam_all) == 0 or len(I_all) == 0:
            # No lines to add, just return without error
            return

        # 2. select only lines within the selected wavelength range using vectorized operations
        wr_min, wr_max = self._wavelength_range
        # Use the minimum R across the range for border padding to capture
        # all lines whose wings could contribute to the grid.
        if self._R_func is not None:
            R_border = float(np.nanmin(np.asarray(
                self._R_func(np.array([wr_min, wr_max])), dtype=float)))
        else:
            R_border = self._R
        select_border = 100 * wr_max / R_border
        mask = (lam_all >= wr_min - select_border) & (lam_all <= wr_max + select_border)
        
        if not np.any(mask):
            return  # No lines in range

        # 3. scale for area in au**2 using vectorized operations
        selected_wavelengths = lam_all[mask]
        selected_intensities = I_all[mask] * dA

        # 4. append numpy arrays directly (much faster than list.extend with numpy arrays)
        self._I_arrays.append(selected_intensities)
        self._lam_arrays.append(selected_wavelengths)
        self._tau_arrays.append(intensity.tau[mask])

        # 5. append to components
        self._components.append({'name': intensity.molecule.name, 'fname': getattr(intensity.molecule, 'fname', ''),
                                't_kin': intensity.t_kin, 'n_mol': intensity.n_mol, 'dv': intensity.dv, 'tau': intensity.tau,
                                 'area': dA})
    
    def _invalidate_flux_cache(self):
        """Invalidate cached flux values"""
        self._flux = None
        self._flux_jy = None
        self._flux_valid = False
        self._tau_profile = None
        self._tau_profile_valid = False
        self._tau_peak_profile = None
        self._unique_cache = None
        self._request_token = next(_REQUEST_TOKENS)
        self._record_cache_invalidation()

    # Number of sigma bins for per-group kernel sizing.
    # Lines are grouped by similar Gaussian width so narrow lines use a tighter
    # kernel instead of the global maximum. More bins = tighter kernels per
    # group (faster), but more loop iterations. 5 is a good balance.
    _N_SIGMA_BINS = 5

    # ------------------------------------------------------------------
    # Convolution kernel cache control
    # ------------------------------------------------------------------
    @staticmethod
    def clear_convolution_cache() -> None:
        """Release all cached convolution kernels and unique-wavelength tables.

        The caches are keyed by content (line wavelengths, kernel widths and
        grid), so they never return stale results; clearing only frees memory,
        e.g. after a molecule was removed or its line list replaced.
        """
        _KERNEL_CACHE.clear()
        _PENDING_KERNELS.clear()
        _UNIQUE_CACHE.clear()

    @staticmethod
    def set_convolution_cache_limit(max_bytes: int) -> None:
        """Set the byte budget of the shared kernel cache (0 disables caching)."""
        _KERNEL_CACHE.resize(max_bytes)

    @staticmethod
    def convolution_cache_info() -> Dict[str, int]:
        """Entries and bytes held by the shared kernel cache."""
        return {
            "kernels": len(_KERNEL_CACHE),
            "kernel_bytes": _KERNEL_CACHE.nbytes,
            "max_bytes": _KERNEL_CACHE.max_bytes,
            "pending": len(_PENDING_KERNELS),
        }

    # ------------------------------------------------------------------
    # Convolution internals
    # ------------------------------------------------------------------
    def _unique_lines(self, lam_array: np.ndarray, use_instance_cache: bool = True):
        """``np.unique(lam_array, return_inverse=True)``, cached per instance and
        in a small process-wide LRU (a new Spectrum is built on every model
        parameter change, so the per-instance cache alone rarely hits).

        The returned arrays may be shared through the cache, so they are
        read-only."""
        uc = self._unique_cache if use_instance_cache else None
        if uc is not None and isinstance(uc[0], np.ndarray) and _bits_equal(uc[0], lam_array):
            return uc[1], uc[2]
        if lam_array.ndim == 1 and lam_array.shape[0] > 0 and lam_array.dtype == np.float64:
            key = _fingerprint(lam_array)
            hit = _UNIQUE_CACHE.get(key, lambda e: _bits_equal(e[0], lam_array))
            if hit is not None:
                lam, index_wavelength = hit[1], hit[2]
            else:
                lam, index_wavelength = np.unique(lam_array, return_inverse=True)
                _read_only(lam)
                _read_only(index_wavelength)
                _UNIQUE_CACHE.put(key, (_read_only(lam_array.copy()), lam, index_wavelength),
                                  2 * lam_array.nbytes + lam.nbytes + index_wavelength.nbytes)
        else:
            lam, index_wavelength = np.unique(lam_array, return_inverse=True)
        if use_instance_cache:
            self._unique_cache = (lam_array, lam, index_wavelength)
        return lam, index_wavelength

    def _line_sigma(self, lam: np.ndarray) -> np.ndarray:
        """Gaussian sigma (micron) of every unique line.

        Uses the wavelength-dependent R(lambda) when a callable is provided,
        otherwise the fixed scalar R."""
        if self._R_func is not None:
            R_per_line = np.atleast_1d(np.asarray(self._R_func(lam), dtype=float))
            fwhm = lam / R_per_line
        else:
            fwhm = lam / self._R
        return fwhm / self._FWHM_TO_SIGMA

    def _sigma_groups(self, sigma: np.ndarray):
        """Per-group kernel sizing: bin lines by similar sigma.

        Instead of using the global max sigma for every line (which wastes
        computation on narrow lines at short wavelengths), lines are binned in
        logarithmically spaced sigma bins (sigma is proportional to the
        wavelength) and every group uses its own kernel half-width."""
        sigma_min = sigma.min()
        sigma_max = sigma.max()
        if sigma_min == sigma_max or sigma.shape[0] <= 1:
            # All lines have the same width -- single group, no binning overhead
            return [np.arange(sigma.shape[0])], [sigma_max]
        bin_edges = np.geomspace(sigma_min * 0.999, sigma_max * 1.001,
                                 num=self._N_SIGMA_BINS + 1)
        group_ids = np.digitize(sigma, bin_edges) - 1  # 0-based bin index
        group_ids = np.clip(group_ids, 0, self._N_SIGMA_BINS - 1)
        unique_groups = np.unique(group_ids)
        sigma_groups = [np.where(group_ids == g)[0] for g in unique_groups]
        sigma_maxes = [sigma[idx].max() for idx in sigma_groups]
        return sigma_groups, sigma_maxes

    def _grid_position(self, lam: np.ndarray) -> np.ndarray:
        """Grid positions for each unique line wavelength."""
        n_grid = self._n_grid_points
        return (n_grid * (lam - self._lam_min) /
                (self._lam_max - self._lam_min)).astype(np.int32)

    def _line_norm(self, sigma: np.ndarray, weights: np.ndarray, normalization: str) -> np.ndarray:
        """Per-line kernel height: unit-area Gaussians for ``"area"``, unit height for ``"peak"``."""
        if normalization == "area":
            return (self._INV_SQRT_2PI / sigma) * weights
        return weights

    def _convolve_lines_reference(self, lam, weights, sigma, normalization="area"):
        """The original per-group bincount loop, kept for the inputs the fast
        paths do not cover (empty, non-finite or non-positive sigma, non-finite
        wavelengths, a malformed R_func), so results and exceptions stay
        unchanged there.  The tests also use it as the reference."""
        n_grid = self._n_grid_points
        norm = self._line_norm(sigma, weights, normalization)
        inv_2sigma_sq = 0.5 / (sigma ** 2)
        lam_grid_position = self._grid_position(lam)
        out = np.zeros(n_grid, dtype=np.float64)
        sigma_groups, sigma_maxes = self._sigma_groups(sigma)
        lamgrid = self._lamgrid
        for group_idx, group_sigma_max in zip(sigma_groups, sigma_maxes):
            kernel_range_size = int(15.0 * group_sigma_max / self._dlambda)
            kernel_range = np.arange(-kernel_range_size, kernel_range_size + 1,
                                     dtype=np.int32)
            g_pos = lam_grid_position[group_idx]
            g_norm = norm[group_idx]
            g_inv2s = inv_2sigma_sq[group_idx]
            g_lam = lam[group_idx]
            grid_indices = g_pos[:, np.newaxis] + kernel_range[np.newaxis, :]
            valid_mask = (grid_indices >= 0) & (grid_indices < n_grid)
            safe_indices = np.clip(grid_indices, 0, n_grid - 1)
            kernel = lamgrid[safe_indices] - g_lam[:, np.newaxis]
            np.multiply(kernel, kernel, out=kernel)
            kernel *= -g_inv2s[:, np.newaxis]
            np.exp(kernel, out=kernel)
            kernel *= g_norm[:, np.newaxis]
            kernel *= valid_mask
            out += np.bincount(safe_indices.ravel(), weights=kernel.ravel(),
                               minlength=n_grid)
        return out

    def _kernel_groups(self, lam, sigma, inv_2sigma_sq):
        """Per sigma group: (line indices, half-width K in pixels, window start
        per line, lambda, -1/(2 sigma^2)).  Lines inside a group are sorted by
        wavelength, so the window starts are non-decreasing."""
        lam_grid_position = self._grid_position(lam)
        sigma_groups, sigma_maxes = self._sigma_groups(sigma)
        neg_inv2s = -inv_2sigma_sq
        groups = []
        for group_idx, group_sigma_max in zip(sigma_groups, sigma_maxes):
            K = int(15.0 * group_sigma_max / self._dlambda)
            start = lam_grid_position[group_idx].astype(np.intp)
            start -= K
            groups.append((group_idx, K, start, lam[group_idx], neg_inv2s[group_idx]))
        return groups

    def _convolve_lines_direct(self, norm, groups):
        """Uncached evaluation; bitwise identical to the original loop.

        The original code built a (n_lines, 2K+1) kernel per sigma group and
        summed it with one np.bincount, so every grid bin received its
        contributions in ascending line order.  Here chunks of ~64k elements
        are evaluated as offset-major blocks (see _kernel_block), which list
        the contributions of each bin in the same ascending line order, and
        np.add.at (sequential, unbuffered) continues that order from chunk to
        chunk in one per-group accumulator.  Clip + mask are only applied to
        chunks that touch a grid end."""
        n_grid = self._n_grid_points
        lamgrid = self._lamgrid
        out = np.zeros(n_grid, dtype=np.float64)
        for group_idx, K, start, g_lam, g_neg in groups:
            g_norm = norm[group_idx]
            n_g = start.shape[0]
            step = max(1, _CHUNK_ELEMENTS // (2 * K + 1))
            acc = np.zeros(n_grid, dtype=np.float64)
            for r0 in range(0, n_g, step):
                r1 = min(n_g, r0 + step)
                idx, kern, mask = _kernel_block(lamgrid, n_grid, start[r0:r1], K,
                                                g_lam[r0:r1], g_neg[r0:r1])
                kern *= g_norm[np.newaxis, r0:r1]
                if mask is not None:
                    kern *= mask
                np.add.at(acc, idx.ravel(), kern.ravel())
            out += acc
        return out

    def _build_kernel_blocks(self, norm, groups):
        """Evaluate the kernel like _convolve_lines_direct and keep its blocks.

        Returns (profile for these weights, cached blocks, bytes).  The blocks
        hold exp(...) * mask (read-only, as they are shared); the grid indices
        are rebuilt on every use."""
        n_grid = self._n_grid_points
        lamgrid = self._lamgrid
        out = np.zeros(n_grid, dtype=np.float64)
        cached = []
        nbytes = 0
        for group_idx, K, start, g_lam, g_neg in groups:
            g_norm = norm[group_idx]
            n_g = start.shape[0]
            step = max(1, _CHUNK_ELEMENTS // (2 * K + 1))
            acc = np.zeros(n_grid, dtype=np.float64)
            blocks = []
            for r0 in range(0, n_g, step):
                r1 = min(n_g, r0 + step)
                idx, kern, mask = _kernel_block(lamgrid, n_grid, start[r0:r1], K,
                                                g_lam[r0:r1], g_neg[r0:r1])
                if mask is not None:
                    kern *= mask
                blocks.append((r0, r1, _read_only(kern), mask is not None))
                nbytes += kern.nbytes
                np.add.at(acc, idx.ravel(), (kern * g_norm[np.newaxis, r0:r1]).ravel())
            out += acc
            cached.append((_read_only(group_idx), K, _read_only(start), blocks))
            nbytes += group_idx.nbytes + start.nbytes
        return out, cached, nbytes

    def _apply_kernel_blocks(self, norm, cached):
        """Profile for new line weights from cached kernel blocks.

        (exp * mask) * w is bitwise equal to (exp * w) * mask for every w
        (also inf / NaN), and the accumulation order is that of the direct
        path, so the result equals the uncached one bit for bit."""
        n_grid = self._n_grid_points
        out = np.zeros(n_grid, dtype=np.float64)
        for group_idx, K, start, blocks in cached:
            g_norm = norm[group_idx]
            kdesc = np.arange(2 * K, -1, -1, dtype=np.intp)
            acc = np.zeros(n_grid, dtype=np.float64)
            for r0, r1, kern, clipped in blocks:
                idx = start[np.newaxis, r0:r1] + kdesc[:, np.newaxis]
                if clipped:
                    np.clip(idx, 0, n_grid - 1, out=idx)
                np.add.at(acc, idx.ravel(), (kern * g_norm[np.newaxis, r0:r1]).ravel())
            out += acc
        return out

    def _convolve_lines(self, lam: np.ndarray, weights: np.ndarray,
                        normalization: str = "area") -> np.ndarray:
        """Spread per-line weights (I*dA or tau, at the sorted unique line
        wavelengths *lam*) over Gaussians of the instrumental width on the grid.

        *normalization* is ``"area"`` (unit-area Gaussians) or ``"peak"``
        (unit height).  Bitwise identical to the original per-group bincount
        loop.  The kernel exp(-(lambda_j - lambda_i)^2 / (2 sigma_i^2)) depends
        only on the line wavelengths, sigma and the grid: when the same kernel
        is requested again by another spectrum, or by this one after its
        intensities changed, its blocks are kept in the process-wide
        _KERNEL_CACHE, and later calls only multiply by the new weights and
        accumulate.  Flux and optical depth of the same model share one kernel.
        The cache is content keyed, so it needs no explicit invalidation."""
        if self._R_func is not None:
            # lam may be shared with the process-wide unique cache: hand the
            # user callable a private copy (as np.unique did before), so a
            # callable that writes into its argument cannot corrupt the cache.
            lam = lam.copy()
        sigma = self._line_sigma(lam)
        norm = self._line_norm(sigma, weights, normalization)
        inv_2sigma_sq = 0.5 / (sigma ** 2)
        n_grid = self._n_grid_points
        span = self._lam_max - self._lam_min
        well_formed = (
            lam.ndim == 1 and lam.shape[0] > 0 and sigma.shape == lam.shape
            and bool(np.isfinite(lam).all())
            and bool(np.isfinite(sigma).all()) and bool((sigma > 0).all())
            and bool(np.isfinite(inv_2sigma_sq).all())
            and abs(n_grid * (lam[0] - self._lam_min) / span) < 2 ** 30
            and abs(n_grid * (lam[-1] - self._lam_min) / span) < 2 ** 30
        )
        if not well_formed:
            return self._convolve_lines_reference(lam, weights, sigma, normalization)

        lamgrid = self._lamgrid
        if _KERNEL_CACHE.max_bytes <= 0:
            return self._convolve_lines_direct(
                norm, self._kernel_groups(lam, sigma, inv_2sigma_sq))
        key = (n_grid, float(self._lam_min), float(self._lam_max), float(self._dlambda),
               int(self._N_SIGMA_BINS), _fingerprint(lam), _fingerprint(sigma))
        entry = _KERNEL_CACHE.get(key, lambda e: (_bits_equal(e[0], lam)
                                                  and _bits_equal(e[1], sigma)
                                                  and _bits_equal(e[2], lamgrid)))
        if entry is not None:
            return self._apply_kernel_blocks(norm, entry[3])        # cache hit
        groups = self._kernel_groups(lam, sigma, inv_2sigma_sq)
        seen = _PENDING_KERNELS.get(key, lambda e: (_bits_equal(e[0], lam)
                                                    and _bits_equal(e[1], sigma)))
        if seen is not None and seen[2] != self._request_token:
            # Requested again by another model: evaluate it once more and keep it.
            out, blocks, nbytes = self._build_kernel_blocks(norm, groups)
            entry = (seen[0], seen[1], _read_only(lamgrid.copy()), blocks)
            _KERNEL_CACHE.put(key, entry, nbytes + entry[0].nbytes
                              + entry[1].nbytes + entry[2].nbytes)
            return out
        if seen is None:
            n_el = sum(g[2].shape[0] * (2 * g[1] + 1) for g in groups)
            est = 8 * n_el + 5 * lam.nbytes + lamgrid.nbytes      # size if it gets cached
            if n_el >= _CACHE_MIN_ELEMENTS and est <= _KERNEL_CACHE.max_bytes:
                # First request: only remember the kernel identity and requester
                # (cheap).  The same model asking again (flux, then tau) does not
                # count, so kernels of a one-off width (FWHM slider) are not kept.
                record = (_read_only(lam.copy()), _read_only(sigma.copy()), self._request_token)
                _PENDING_KERNELS.put(key, record, record[0].nbytes + record[1].nbytes)
        return self._convolve_lines_direct(norm, groups)

    def _convol_flux(self):
        """Internal procedure to carry out the convolution, should never be called directly

        Returns
        -------
        np.ndarray:
            List with fluxes, convolved to the spectral resolution
        """

        # Check if we have any intensities to process
        if len(self._I_arrays) == 0:
            # Return a zero flux array if no intensities were added
            return np.zeros_like(self._lamgrid)

        # Concatenate all accumulated arrays at once (much faster than list extend)
        I_array = np.concatenate(self._I_arrays)
        lam_array = np.concatenate(self._lam_arrays)

        # 1. Summarize intensities at the (exactly) same wavelength. This improves
        #    performance because only one convolution kernel needs to be evaluated per
        #    unique line wavelength (independent of how many intensity components share it).
        lam, index_wavelength = self._unique_lines(lam_array)

        # np.bincount is ~5-20x faster than np.add.at for scatter-add accumulation
        intens = np.bincount(index_wavelength, weights=I_array,
                             minlength=lam.shape[0]).astype(np.float64)

        # 2. Gaussian instrumental profile, normalised per unit wavelength
        flux = self._convolve_lines(lam, intens, "area")

        # 3. Scale for distance and correct units for the area
        inv_dist_sq = 1.0 / (self._distance ** 2)
        scaled_flux = flux * (self._AU_PC_RATIO_SQ * inv_dist_sq)

        return scaled_flux

    @property
    def flux(self) -> np.ndarray:
        """np.ndarray: Flux density in erg/s/cm^2/micron"""
        if self._flux is None or not self._flux_valid:
            self._record_cache_miss()
            self._flux = self._convol_flux()
            self._flux_valid = True
        else:
            self._record_cache_hit()
        return self._flux

    @property
    def flux_jy(self) -> np.ndarray:
        """np.ndarray: Flux density in Jy/micron"""
        if self._flux_jy is None:
            flux_data = self.flux  # This triggers flux calculation if needed
            # Use pre-computed conversion factor
            self._flux_jy = flux_data * self._FLUX_JY_FACTOR * (self._lamgrid ** 2)
        return self._flux_jy

    @property
    def keplerian_fwhm(self) -> float:
        """float: Keplerian broadening FWHM in km/s passed at construction."""
        return self._keplerian_fwhm

    # ------------------------------------------------------------------
    # Optical-depth profile convolution
    # ------------------------------------------------------------------

    #: Valid ``normalization`` values of the optical-depth profiles (see :meth:`_gaussian_tau_profile`).
    TAU_NORMALIZATIONS = ("peak", "area")

    @classmethod
    def _check_tau_normalization(cls, normalization: str) -> str:
        """Return *normalization*, or raise ``ValueError`` if it is not one of :attr:`TAU_NORMALIZATIONS`."""
        if normalization not in cls.TAU_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {cls.TAU_NORMALIZATIONS}, got {normalization!r}"
            )
        return normalization

    def _gaussian_tau_profile(self, lam: np.ndarray, tau_sum: np.ndarray,
                              normalization: str) -> np.ndarray:
        """Spread line-centre optical depths over Gaussians of the instrumental width on ``_lamgrid``.

        Uses the same kernels as :meth:`_convol_flux` (per-group kernel sizing,
        ~15 sigma, and the same cached kernel for the same lines), without
        distance or area scaling.

        Parameters
        ----------
        lam : np.ndarray
            Unique line-centre wavelengths in micron.
        tau_sum : np.ndarray
            Summed line-centre optical depth at each wavelength of *lam*.
        normalization : {"peak", "area"}
            ``"peak"``: every Gaussian has height 1, so an isolated line peaks
            at its line-centre optical depth (dimensionless).
            ``"area"``: every Gaussian has unit area (height
            1 / (sqrt(2 pi) sigma)), so the profile is optical depth per
            micron (1/um) and integrates over wavelength to the summed
            line-centre optical depth.

        Returns
        -------
        np.ndarray
            The profile on ``_lamgrid``.
        """
        return self._convolve_lines(lam, tau_sum, normalization)

    def _convol_tau(self, normalization: str = "area") -> np.ndarray:
        """Convolve the per-line optical depths of all components into a profile on ``_lamgrid``.

        Parameters
        ----------
        normalization : {"area", "peak"}, default "area"
            ``"area"`` gives optical depth per micron (:attr:`tau_profile`),
            ``"peak"`` the dimensionless optical depth
            (:attr:`tau_peak_profile`); see :meth:`_gaussian_tau_profile`.

        Returns
        -------
        np.ndarray
            The profile on ``_lamgrid``.
        """
        self._check_tau_normalization(normalization)
        if len(self._tau_arrays) == 0:
            return np.zeros_like(self._lamgrid)

        tau_array = np.concatenate(self._tau_arrays)
        lam_array = np.concatenate(self._lam_arrays)

        lam, index_wavelength = self._unique_lines(lam_array)

        tau_sum = np.bincount(index_wavelength, weights=tau_array,
                              minlength=lam.shape[0]).astype(np.float64)

        return self._gaussian_tau_profile(lam, tau_sum, normalization)

    @property
    def tau_profile(self) -> np.ndarray:
        """np.ndarray: Optical depth **per micron** (1/um) on the wavelength grid.

        Each line's line-centre optical depth is spread over a unit-area
        Gaussian of the instrumental width, so the profile integrates over
        wavelength to the summed line-centre optical depth (see
        :func:`~iSLAT.Modules.DataProcessing.spectral_utils.opacity_integral`).
        Its height is not an optical depth: an isolated line peaks at
        tau / (sqrt(2 pi) sigma), which depends on the instrumental
        resolution.  :attr:`tau_peak_profile` is the optical depth itself.
        """
        if self._tau_profile is None or not self._tau_profile_valid:
            self._tau_profile = self._convol_tau("area")
            self._tau_profile_valid = True
        return self._tau_profile

    @property
    def tau_peak_profile(self) -> np.ndarray:
        """np.ndarray: Dimensionless optical depth tau(lambda) on the wavelength grid.

        Each line is a Gaussian of the instrumental width whose height is its
        line-centre optical depth, so an isolated line peaks at its tau
        (up to the grid sampling).  Lines closer together than about the
        instrumental FWHM add up here even where their intrinsic profiles
        (width ``dv``) do not overlap, so blends are an upper limit.
        """
        if self._tau_peak_profile is None:
            self._tau_peak_profile = self._convol_tau("peak")
        return self._tau_peak_profile

    def get_tau_profile(self, normalization: str = "area") -> np.ndarray:
        """Return :attr:`tau_profile` (``"area"``, 1/um) or :attr:`tau_peak_profile` (``"peak"``, dimensionless)."""
        if self._check_tau_normalization(normalization) == "peak":
            return self.tau_peak_profile
        return self.tau_profile

    def _convol_tau_per_component(self, normalization: str = "area") -> List[Dict[str, Any]]:
        """Convolve tau for each component separately.

        Parameters
        ----------
        normalization : {"area", "peak"}, default "area"
            See :meth:`_convol_tau`.

        Returns
        -------
        list of dict
            Each dict has keys ``'name'``, ``'lamgrid'``, ``'tau_profile'``.
        """
        self._check_tau_normalization(normalization)
        results: List[Dict[str, Any]] = []
        if len(self._tau_arrays) == 0:
            return results

        n_grid = self._n_grid_points
        lamgrid = self._lamgrid

        for comp_idx, comp in enumerate(self._components):
            tau_arr = self._tau_arrays[comp_idx]
            lam_arr = self._lam_arrays[comp_idx]

            if len(tau_arr) == 0:
                profile = np.zeros(n_grid, dtype=np.float64)
            else:
                lam, index_wavelength = self._unique_lines(np.asarray(lam_arr),
                                                           use_instance_cache=False)
                tau_sum = np.bincount(index_wavelength, weights=tau_arr,
                                      minlength=lam.shape[0]).astype(np.float64)
                profile = self._gaussian_tau_profile(lam, tau_sum, normalization)

            results.append({
                'name': comp['name'],
                'lamgrid': lamgrid,
                'tau_profile': profile,
            })

        return results

    # wavelength_range property provided by WavelengthRangeMixin

    def _on_wavelength_range_changed(self, old, new):
        """Hook called by WavelengthRangeMixin when wavelength_range changes."""
        self._invalidate_flux_cache()

    @property
    def lamgrid(self) -> np.ndarray:
        """np.ndarray: Wavelength grid in micron"""
        return self._lamgrid

    @property
    def components(self) -> List[Dict[str, Any]]:
        """list of dict: Intensity components added to the spectrum"""
        return self._components

    @property
    def R_func(self):
        """callable or None: Wavelength-dependent resolving-power function."""
        return self._R_func

    @R_func.setter
    def R_func(self, value):
        """Set a new R(λ) function.  Invalidates cached flux."""
        if value is not self._R_func:
            self._R_func = value
            self._invalidate_flux_cache()

    def resample_to(self, target_wavelengths: np.ndarray,
                    unit: str = 'jy',
                    rv_shift: float = 0.0,
                    fill: float = 0.0) -> np.ndarray:
        """Flux-conserving resampling of the spectrum onto a target grid.

        Uses the same ``spectres``-style algorithm employed by
        :func:`iSLAT.Modules.DataTypes.Molecule._spectres` to perform
        flux-conserving resampling that correctly handles uneven pixel
        sampling (e.g. JWST MIRI MRS).

        Parameters
        ----------
        target_wavelengths : np.ndarray
            1-D array of desired output wavelength positions (µm).
        unit : str, optional
            ``'jy'`` (default) for Jy, ``'cgs'`` for erg/s/cm²/µm.
        rv_shift : float, optional
            Radial-velocity shift in km/s to apply to the model grid
            **before** resampling.  Positive values red-shift the model
            (i.e. ``λ_obs = λ_rest x (1 + rv / c)``).
        fill : float, optional
            Value used for target pixels that fall outside the model grid.

        Returns
        -------
        np.ndarray
            Resampled flux array with the same length as
            *target_wavelengths*.
        """
        import importlib as _il
        _su = _il.import_module('iSLAT.Modules.DataProcessing.spectral_utils')
        _spectres = _su.spectres

        source_wave = self._lamgrid
        if rv_shift != 0.0:
            doppler = 1.0 + rv_shift / c.SPEED_OF_LIGHT_KMS
            source_wave = source_wave * doppler

        if unit == 'jy':
            source_flux = self.flux_jy
        else:
            source_flux = self.flux

        return _spectres(target_wavelengths, source_wave, source_flux,
                         fill=fill)

    @property
    def get_table(self):
        """pd.DataFrame: Pandas dataframe"""
        pd = _get_pandas()
        # Pre-compute all data to avoid repeated property access
        flux_data = self.flux
        flux_jy_data = self.flux_jy
        
        return pd.DataFrame({
            'lam': self._lamgrid,
            'flux': flux_data,
            'flux_jy': flux_jy_data
        })

    # get_cache_stats() provided by CacheStatsMixin

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.get_table._repr_html_()