# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
        frequencies : np.ndarray
            Line frequencies in Hz with shape (n_lines,)
        sqrt_ln2_inv : float
            Normalization constant

        Returns read access to the fields is granted through properties

- 01/06/2020: SB, initial version
- 11/01/2025 Johnny McCaskill, redesigned for performance and to enable overlapping line treatment

"""
import numpy as np
import time
from typing import Optional, Union, Literal, TYPE_CHECKING, Any
import iSLAT.Constants as c

# Performance logging
from iSLAT.Modules.Debug.PerformanceLogger import perf_log, log_timing, PerformanceSection

# Lazy imports for performance
_scipy_imported = False

def _get_scipy():
    """Lazy import of scipy components"""
    global _scipy_imported
    if not _scipy_imported:
        global fixed_quad
        from scipy.integrate import fixed_quad
        _scipy_imported = True
    return fixed_quad

from ._pandas_import import get_pandas as _get_pandas
from ._mixins import WavelengthRangeMixin

if TYPE_CHECKING:
    import pandas
    from .MoleculeLineList import MoleculeLineList
else:
    # Lazy import for runtime
    def _get_molecule_line_list():
        from .MoleculeLineList import MoleculeLineList
        return MoleculeLineList

__all__ = ["Intensity"]

class Intensity(WavelengthRangeMixin):
    __slots__ = ('_molecule', '_intensity', '_tau', '_t_kin', '_n_mol', '_dv', '_method', '_cache_valid', '_sorted_idx', '_sorted_freq',
                 '_cached_line_scalar', '_cached_e_delta', '_cached_dv_star',
                 '_cached_freq_ratio', '_wavelength_range', '_full_range_cache')
    
    def __init__(self, molecule_line_list: 'MoleculeLineList',
                 wavelength_range: Optional[tuple] = None) -> None:
        """Initialize an intensity class which calculates the intensities for a given molecule and provided
        physical parameters.

        Parameters
        ----------
        molecule_line_list: MoleculeLineList
            Molecular line list data to calculate the intensity
        wavelength_range: tuple of (float, float), optional
            Wavelength range ``(lam_min, lam_max)`` in microns.  When set,
            only lines within this range are used for intensity calculations.
            The range is propagated to the underlying MoleculeLineList.
        """
        if TYPE_CHECKING:
            self._molecule: "MoleculeLineList" = molecule_line_list
        else:
            # Runtime: accept any compatible object
            self._molecule = molecule_line_list
        
        self._wavelength_range = wavelength_range
        if wavelength_range is not None:
            self._molecule.wavelength_range = wavelength_range
            
        self._intensity: Optional[np.ndarray] = None
        self._tau: Optional[np.ndarray] = None
        self._t_kin: Optional[float] = None
        self._n_mol: Optional[float] = None
        self._dv: Optional[float] = None
        self._method: Optional[str] = None  # calculation method of the cached result
        self._cache_valid: bool = False
        self._sorted_idx: Optional[np.ndarray] = None
        self._sorted_freq: Optional[np.ndarray] = None
        # Pre-computed line constants (molecule-dependent, not parameter-dependent)
        self._cached_line_scalar: Optional[np.ndarray] = None
        self._cached_e_delta: Optional[np.ndarray] = None
        self._cached_dv_star: Optional[np.ndarray] = None  # per-neighbor dv thresholds for overlap
        self._cached_freq_ratio: Optional[np.ndarray] = None  # freq / c, shape (1, n_lines)
        # (raw lines, (t_kin, n_mol, dv, method), intensity, tau) of the last
        # full-range calculation (see _compute_full_range_intensity).
        self._full_range_cache: Optional[tuple] = None

    # Pre-computed constants for blackbody calculation (class-level for efficiency)
    _BB_COEFF_RJ = 2.0 * c.BOLTZMANN_CONSTANT / (c.SPEED_OF_LIGHT_CGS ** 2)
    _BB_COEFF_PLANCK = 2.0 * c.PLANCK_CONSTANT / (c.SPEED_OF_LIGHT_CGS ** 2)
    _BB_X_FACTOR = c.PLANCK_CONSTANT / c.BOLTZMANN_CONSTANT
    
    # Pre-computed constants for intensity calculation (avoid recomputing every call)
    _SQRT_LN2_INV = 1.0 / (2.0 * np.sqrt(np.log(2.0)))  # ~ 0.6005
    _C3_OVER_8PI = (c.SPEED_OF_LIGHT_CGS ** 3) / (8.0 * np.pi)  # c^3/(8π)
    _INV_FGAUSS_1E5 = 1.0 / (1e5 * c.FGAUSS_PREFACTOR)  # 1/(10^5 * FGAUSS_PREFACTOR)
    _TWO_SQRT_LN2 = 2.0 * np.sqrt(np.log(2.0))  # ~ 1.6651, for converting dv to x-units
    
    # tau-regime shortcut threshold
    _TAU_THIN = 0.01     # below this, F(tau_0) ~ tau_0 * Sum_k w_k exp(-x_k^2)  (linear regime)
    
    # Pre-computed sqrt(pi) sum for thin-regime shortcut (set by _initialize_gauss_quad)
    _GAUSS_QUAD_SQRT_PI_SUM = None  # Sum_k w_k * exp(-x_k^2) ~ sqrt(pi)
    
    # Weak-blend early-exit threshold in Gaussian widths
    # At 4*sigma, overlap contribution is exp(-16) ~ 1.1e-7 -- negligible
    _WEAK_BLEND_THRESHOLD = 0.01
    
    @staticmethod
    def _bb(nu: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Blackbody function for temperatures and frequencies.
        Handles both scalar T and array T for batch operations.
        Uses optimized approximations for accuracy and performance.

        Parameters
        ----------
        nu: np.ndarray
            Frequency in Hz to calculate the blackbody values, shape (n_lines,)
        T: np.ndarray or float
            Temperature(s) in K. Can be scalar or array shape (n_conditions,)

        Returns
        -------
        np.ndarray:
            Blackbody intensity in erg/s/cm**2/sr/Hz
            Shape (n_lines,) if T is scalar, (n_conditions, n_lines) if T is array
        """
        cls = Intensity
        T = np.atleast_1d(T)
        
        # Check if we have multiple temperatures (batch mode)
        if T.size == 1:
            # Single temperature - branchless scalar path
            # np.clip prevents overflow in np.expm1 for x > ~709 (Wien regime)
            # and for x < 1e-5 (Rayleigh-Jeans), expm1 is already accurate.
            T_val = T.item()
            inv_T = 1.0 / T_val
            x = cls._BB_X_FACTOR * inv_T * nu
            
            nu_cu = nu * nu * nu
            
            return cls._BB_COEFF_PLANCK * nu_cu / np.expm1(np.clip(x, 1e-300, 709.0))
        else:
            # Multiple temperatures - vectorized batch path
            # Shape: T is (n_cond,), nu is (n_lines,)
            # Result should be (n_cond, n_lines)
            inv_T = 1.0 / T  # (n_cond,)
            
            # Broadcast: (n_cond, 1) * (n_lines,) -> (n_cond, n_lines)
            x = cls._BB_X_FACTOR * inv_T[:, np.newaxis] * nu[np.newaxis, :]
            
            nu_sq = nu * nu  # (n_lines,)
            nu_cu = nu_sq * nu  # (n_lines,)
            
            out = np.empty(x.shape, dtype=np.float64)
            
            m_rj = x < 1.0e-5
            m_wien = x > 20.0
            m_mid = ~(m_rj | m_wien)
            
            # Compute directly into masked positions to avoid full-size temporaries
            # Use np.broadcast_to for zero-copy broadcast views
            T_2d = np.broadcast_to(T[:, np.newaxis], x.shape)
            nu_sq_2d = np.broadcast_to(nu_sq[np.newaxis, :], x.shape)
            nu_cu_2d = np.broadcast_to(nu_cu[np.newaxis, :], x.shape)
            
            # Rayleigh-Jeans regime
            if np.any(m_rj):
                out[m_rj] = cls._BB_COEFF_RJ * T_2d[m_rj] * nu_sq_2d[m_rj]
            
            # Wien regime
            if np.any(m_wien):
                out[m_wien] = cls._BB_COEFF_PLANCK * nu_cu_2d[m_wien] * np.exp(-x[m_wien])
            
            # Full Planck regime
            if np.any(m_mid):
                out[m_mid] = cls._BB_COEFF_PLANCK * nu_cu_2d[m_mid] / np.expm1(x[m_mid])
            
            return out

    # Pre-computed Gaussian quadrature points and weights for maximum efficiency
    _GAUSS_QUAD_X = None
    _GAUSS_QUAD_W = None
    _GAUSS_QUAD_INITIALIZED = False

    # Strategy registry for intensity calculation methods
    _strategy_registry: dict = {}

    @classmethod
    def _ensure_strategy_registry(cls) -> None:
        """Populate the strategy registry lazily (avoids circular imports)."""
        if not cls._strategy_registry:
            from ._intensity_strategies import DEFAULT_STRATEGIES
            cls._strategy_registry.update(DEFAULT_STRATEGIES)

    @classmethod
    def register_strategy(cls, method_name: str, strategy) -> None:
        """Register a new (or replacement) intensity calculation strategy.

        Parameters
        ----------
        method_name : str
            Key used in ``calc_intensity(method=...)``.
        strategy : IntensityStrategy
            Any object with a ``calculate(...)`` method matching the
            :class:`IntensityStrategy` protocol.
        """
        cls._ensure_strategy_registry()
        cls._strategy_registry[method_name] = strategy

    @classmethod
    def available_strategies(cls) -> list:
        """Return the names usable as ``calc_intensity(method=...)``.

        The built-in methods come first, in the order of
        :data:`~._intensity_strategies.STRATEGY_DESCRIPTIONS` (default first),
        followed by any added with :meth:`register_strategy`, sorted.
        """
        cls._ensure_strategy_registry()
        from ._intensity_strategies import STRATEGY_DESCRIPTIONS
        builtin = [name for name in STRATEGY_DESCRIPTIONS if name in cls._strategy_registry]
        extra = sorted(name for name in cls._strategy_registry if name not in STRATEGY_DESCRIPTIONS)
        return builtin + extra

    @staticmethod
    def describe_strategy(method: str) -> tuple:
        """Return ``(display_name, description)`` for *method*.

        Methods without a description (e.g. custom ones added with
        :meth:`register_strategy`) are shown by name with an empty description.
        """
        from ._intensity_strategies import STRATEGY_DESCRIPTIONS
        return STRATEGY_DESCRIPTIONS.get(method, (str(method), ""))

    @classmethod
    def _get_strategy(cls, method: str):
        """Look up a strategy by *method* name, raising on unknown keys."""
        cls._ensure_strategy_registry()
        strategy = cls._strategy_registry.get(method)
        if strategy is None:
            raise ValueError(
                f"Unknown intensity method {method!r}. "
                f"Available: {sorted(cls._strategy_registry)}"
            )
        return strategy
    
    @classmethod
    def _initialize_gauss_quad(cls):
        """Initialize Gaussian quadrature points and weights once for all calculations."""
        if not cls._GAUSS_QUAD_INITIALIZED:
            try:
                from numpy.polynomial.legendre import leggauss
                x_quad, w_quad = leggauss(20)
                # Transform from [-1,1] to [-6,6] and pre-compute exp(-x^2)
                #cls._GAUSS_QUAD_X = np.exp(-(6.0 * x_quad) ** 2)  # Pre-compute exp(-x^2)
                cls._GAUSS_QUAD_X = 6.0 * x_quad  # Store actual quadrature points
                cls._GAUSS_QUAD_EXP = np.exp(-(cls._GAUSS_QUAD_X ** 2))
                cls._GAUSS_QUAD_W = 6.0 * w_quad
                cls._GAUSS_QUAD_SQRT_PI_SUM = float(np.dot(cls._GAUSS_QUAD_W, cls._GAUSS_QUAD_EXP))
                cls._GAUSS_QUAD_INITIALIZED = True
            except ImportError:
                # Fallback to scipy method if numpy.polynomial not available
                fixed_quad = _get_scipy()
                # Use a simple tau value to initialize quadrature
                cls._GAUSS_QUAD_X = np.linspace(-6, 6, 20)  # Store actual x values
                cls._GAUSS_QUAD_EXP = np.exp(-(cls._GAUSS_QUAD_X ** 2))
                cls._GAUSS_QUAD_W = np.ones(20) * (12.0 / 20.0)  # Simple uniform weighting
                cls._GAUSS_QUAD_SQRT_PI_SUM = float(np.dot(cls._GAUSS_QUAD_W, cls._GAUSS_QUAD_EXP))
                cls._GAUSS_QUAD_INITIALIZED = True

    def _fint_multi(self, center_tau: np.ndarray, dv_cond: np.ndarray,
                   bb_vals: np.ndarray, freq_ratio: np.ndarray, sqrt_ln2_inv: float,
                   frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate spectral line intensities with proper treatment of overlapping lines.
        
        This method implements the curve-of-growth integral for molecular line intensities,
        handling both isolated and overlapping lines using correct radiative transfer physics.
        For overlapping lines, optical depths are summed before applying the curve-of-growth
        to avoid unphysical intensity enhancement and maintain conservation.
        
        Algorithm Overview:
        1. Identify isolated vs. overlapping lines using velocity separation criteria  
        2. Process isolated lines in batch using vectorized operations
        3. For overlapping lines: sum optical depths -> apply curve-of-growth -> distribute intensity
        
        Physical Basis:
        - Isolated lines: I = (physical_factors) * integral[1 - exp(-tau)] dv
        - Overlapping lines: I_total = integral[1 - exp(-Sum(tau_i))] dv, then distribute proportionally
        
        Performance:
        - Memory complexity: O(n) instead of O(n^2) for overlap detection
        - Time complexity: O(n log n) for sorting + O(n) for processing  
        - Vectorized operations for isolated lines (typically 85%+ of all lines)
        
        Parameters
        ----------
        center_tau : np.ndarray, shape (n_conditions, n_lines)
            Line center optical depths for all physical conditions and lines
        dv_cond : np.ndarray, shape (n_conditions,)
            Doppler broadening velocities (FWHM) in km/s for each condition
        bb_vals : np.ndarray, shape (n_conditions, n_lines)
            Blackbody source function values at line frequencies  
        freq_ratio : np.ndarray, shape (n_conditions, n_lines)
            Frequency ratios (v/c) for unit conversion
        sqrt_ln2_inv : float
            Normalization constant: 1/(2*sqrt(ln(2))) ~ 0.6005 for Gaussian integration
        frequencies : np.ndarray, shape (n_lines,)
            Line rest frequencies in Hz
            
        Returns
        -------
        np.ndarray, shape (n_conditions, n_lines)
            Line-integrated specific intensities following Banzatti et al. 2012 Eq. A1.
            Units: (1/(2*sqrt(ln(2)))) x 10^5 x dv x (nu/c) x B_nu(T) x F(tau_0), where F is the
            curve-of-growth integral. The 10^5 factor converts km/s to cm/s.
            
        Notes
        -----
        - Uses Gaussian quadrature for numerical integration of curve-of-growth
        - Velocity overlap criterion: lines are grouped when their frequency
          separation corresponds to less than the current dv (FWHM)
        """
        cls = self.__class__
        # Initialize quadrature points and weights
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()
        
        # Setup arrays and extract dimensions
        dv_cond = np.atleast_1d(dv_cond)
        n_cond, n_lines = center_tau.shape
        
        # Quadrature setup for curve-of-growth integration
        exp_neg_x_squared = cls._GAUSS_QUAD_EXP  # (n_quad,) - don't add dimension yet
        weights = cls._GAUSS_QUAD_W
        
        # ===============================================================
        # STEP 1: Identify isolated vs. overlapping lines
        # ===============================================================
        
        # Pre-compute physical factors once (used by all code paths below)
        physical_factors = np.empty((n_cond, n_lines), dtype=np.float64)
        np.multiply(freq_ratio, bb_vals, out=physical_factors)
        physical_factors *= (sqrt_ln2_inv * 1e5)
        physical_factors *= dv_cond[:, np.newaxis]

        # Cache sorted order and per-neighbor dv_star thresholds per instance
        # (frequencies are molecule-dependent and stable across calls).
        # dv_star[i] = the minimum dv (km/s) at which sorted lines i and i+1 overlap:
        #   dv_star_i = c * dnu_i / max(nu_i, nu_{i+1})
        if self._sorted_idx is None:
            self._sorted_idx = np.argsort(frequencies)
            self._sorted_freq = frequencies[self._sorted_idx]
            sf = self._sorted_freq
            if sf.size > 1:
                self._cached_dv_star = (c.SPEED_OF_LIGHT_KMS
                    * np.diff(sf) / np.maximum(sf[:-1], sf[1:]))
            else:
                self._cached_dv_star = np.array([], dtype=np.float64)

        sort_indices = self._sorted_idx
        dv_star = self._cached_dv_star  # (n_sorted - 1,)
        
        # Use the effective dv for grouping. In batch mode with varying dv,
        # use max(dv_cond) to find the *superset* of potentially blended pairs.
        # The exact multi-line quadrature in Step 3 handles per-condition
        # offsets correctly, so over-grouping here is conservative, not wrong.
        # For single-condition calls, this is exact.
        dv_group = np.max(dv_cond)
        
        # Fast all-isolated check: if even the widest dv doesn't blend any pair
        if dv_star.size == 0 or np.all(dv_group < dv_star):
            # tau-regime shortcuts: partition lines into thin / normal / thick
            fint_vals = self._cog_with_tau_shortcuts(center_tau, exp_neg_x_squared, weights)
            
            # Final multiplication in-place
            np.multiply(physical_factors, fint_vals, out=physical_factors)
            return physical_factors
        
        # Vectorized overlap grouping using cumsum-based labeling.
        # This avoids np.split (O(n_groups) Python objects) and instead assigns
        # a contiguous integer group label to each sorted line, then uses
        # np.bincount to identify isolated vs blended groups in O(n) time.
        is_boundary = dv_group < dv_star
        
        # Assign group labels via cumulative sum of boundary flags
        group_labels = np.empty(n_lines, dtype=np.int32)
        group_labels[0] = 0
        group_labels[1:] = np.cumsum(is_boundary)
        
        # Count lines per group using bincount
        group_counts = np.bincount(group_labels)
        
        # Map group counts back to each sorted line position
        line_group_counts = group_counts[group_labels]
        
        # Isolated lines: groups with exactly 1 member
        isolated_sorted_mask = line_group_counts == 1
        blended_sorted_mask = ~isolated_sorted_mask
        
        # ===============================================================
        # STEP 2: Process isolated lines
        # ===============================================================
        
        if np.any(isolated_sorted_mask):
            isolated_indices = sort_indices[isolated_sorted_mask]
            tau_batch = center_tau[:, isolated_indices]
            
            # tau-regime shortcuts: partition lines into thin / normal / thick
            fint_vals = self._cog_with_tau_shortcuts(tau_batch, exp_neg_x_squared, weights)
            
            # Multiply curve-of-growth factors in-place into physical_factors
            physical_factors[:, isolated_indices] *= fint_vals
        
        # ===============================================================
        # STEP 3: Process overlapping lines with separation-aware physics
        # ===============================================================
        # Exact multi-line quadrature for blended groups:
        #   Computes tau_tot(x_k) = Sum_j tau_{0,j} exp(-(x_k - delta_j)^2) at each
        #   quadrature point, applies [1 - exp(-tau_tot)] correctly, then
        #   distributes intensity to lines by their fractional opacity.
        #
        # This fixes the co-centered approximation that ignores line separations,
        # which over-saturates peak optical depth and underestimates emission
        # for blends separated by ~0.5-2 line widths.
        
        if not np.any(blended_sorted_mask):
            return physical_factors
        
        x_quad = cls._GAUSS_QUAD_X  # (n_quad,) actual quadrature points in [-6, 6]
        two_sqrt_ln2 = cls._TWO_SQRT_LN2
        
        # Extract blended line data in sorted order
        blended_sort_positions = np.where(blended_sorted_mask)[0]
        blended_orig_indices = sort_indices[blended_sort_positions]
        blended_group_labels = group_labels[blended_sort_positions]
        
        # Find group boundaries within the blended-only arrays
        unique_blend_labels, blend_group_starts = np.unique(blended_group_labels, return_index=True)
        blend_group_ends = np.append(blend_group_starts[1:], len(blended_orig_indices))
        blend_group_sizes = blend_group_ends - blend_group_starts
        
        # ---------------------------------------------------------------
        # STEP 3a: Batch-vectorized processing for common group sizes
        # ---------------------------------------------------------------
        # Groups of the same size can be processed in a single vectorized operation, avoiding per-group Python overhead.
        
        _BATCH_MAX_SIZE = 32  # Batch groups up to this size; larger ones use per-group loop
        
        processed_mask = np.zeros(len(unique_blend_labels), dtype=bool)
        
        for target_size in range(2, _BATCH_MAX_SIZE + 1):
            size_mask = blend_group_sizes == target_size
            if not np.any(size_mask):
                continue
            
            size_group_idx = np.where(size_mask)[0]
            n_groups_this_size = len(size_group_idx)
            
            # Gather indices for all groups of this size into (n_groups, target_size)
            all_indices = np.empty((n_groups_this_size, target_size), dtype=np.intp)
            for k in range(target_size):
                all_indices[:, k] = blended_orig_indices[blend_group_starts[size_group_idx] + k]
            
            # Process each condition
            for ic in range(n_cond):
                tau_batch_g = center_tau[ic][all_indices]     # (n_groups, target_size)
                freq_batch_g = frequencies[all_indices]       # (n_groups, target_size)
                
                # Skip negligible groups
                tau_max_per_group = tau_batch_g.max(axis=1)
                active = tau_max_per_group > 1e-300
                if not np.any(active):
                    continue
                
                tau_bg = tau_batch_g[active]                   # (n_active, target_size)
                freq_bg = freq_batch_g[active]                 # (n_active, target_size)
                active_indices = all_indices[active]           # (n_active, target_size)
                
                # tau-weighted reference frequency
                tau_sums = tau_bg.sum(axis=1)                  # (n_active,)
                safe_sums = np.where(tau_sums > 1e-300, tau_sums, 1.0)
                freq_refs = ((tau_bg * freq_bg).sum(axis=1) / safe_sums)[:, np.newaxis]
                
                # Velocity offsets in quadrature x-units
                dv_kms = (freq_bg - freq_refs) * (c.SPEED_OF_LIGHT_KMS / freq_refs)
                delta_x = (dv_kms / dv_cond[ic]) * two_sqrt_ln2  # (n_active, target_size)
                
                # Shifted Gaussian profiles in-place: (n_quad, n_active, target_size)
                gp = x_quad[:, np.newaxis, np.newaxis] - delta_x[np.newaxis, :, :]
                np.multiply(gp, gp, out=gp)            # gp = shifted_x^2
                np.negative(gp, out=gp)                 # gp = -shifted_x^2
                np.exp(gp, out=gp)                      # gp = exp(-shifted_x^2)
                
                # tau_tot at each quadrature point: (n_quad, n_active)
                tau_at_quad = np.einsum('qgl,gl->qg', gp, tau_bg)
                
                # Curve-of-growth: 1 - exp(-tau_tot)
                # Keep tau_at_quad intact for the division; compute integrand into a separate array
                cog_mask = tau_at_quad > 1e-300
                integrand = np.empty_like(tau_at_quad)
                np.negative(tau_at_quad, out=integrand)         # -tau
                np.expm1(integrand, out=integrand)              # expm1(-tau)
                np.negative(integrand, out=integrand)           # 1-exp(-tau)
                cog_over_tau = np.where(cog_mask, integrand / np.where(cog_mask, tau_at_quad, 1.0), 0.0)
                
                # Distribute intensity proportionally, reusing gp array in-place
                gp *= (cog_over_tau * weights[:, np.newaxis])[:, :, np.newaxis]
                fint = gp.sum(axis=0)                          # (n_active, target_size)
                fint *= tau_bg
                
                # Scatter results back into physical_factors
                physical_factors[ic][active_indices.ravel()] *= fint.ravel()
            
            processed_mask[size_group_idx] = True
        
        # ---------------------------------------------------------------
        # STEP 3b: Per-group loop for remaining large groups
        # ---------------------------------------------------------------
        remaining_idx = np.where(~processed_mask)[0]
        
        if len(remaining_idx) > 0:
            weak_blend_thresh = cls._WEAK_BLEND_THRESHOLD
            tau_thin_thresh = cls._TAU_THIN
            
            for gi in remaining_idx:
                start = blend_group_starts[gi]
                end = blend_group_ends[gi]
                overlapping_indices = blended_orig_indices[start:end]
                
                # Extract optical depths for this blended group
                tau_group = center_tau[:, overlapping_indices]  # (n_cond, n_overlap)
                
                # Early exit: skip quadrature if all tau_0 values are negligible
                if tau_group.max() < 1e-300:
                    continue
                
                freq_group = frequencies[overlapping_indices]
                freq_group_mean = freq_group.mean()
                
                for ic in range(n_cond):
                    tg = tau_group[ic]
                    
                    if tg.max() < 1e-300:
                        continue
                    
                    # tau_0-weighted reference frequency
                    tau_sum_ic = tg.sum()
                    if tau_sum_ic > 1e-300:
                        freq_ref_ic = np.dot(tg, freq_group) / tau_sum_ic
                    else:
                        freq_ref_ic = freq_group_mean
                    
                    dv_kms_ic = (freq_group - freq_ref_ic) * (c.SPEED_OF_LIGHT_KMS / freq_ref_ic)
                    delta_x_ic = (dv_kms_ic / dv_cond[ic]) * two_sqrt_ln2
                    
                    # Weak-blend early exit
                    if np.max(np.abs(delta_x_ic)) > weak_blend_thresh:
                        sorted_dx = np.sort(delta_x_ic)
                        max_neighbor_sep = np.max(np.diff(sorted_dx)) if len(sorted_dx) > 1 else 0.0
                        if max_neighbor_sep > 2.0 * weak_blend_thresh:
                            tau_exp_ic = tg[:, np.newaxis] * exp_neg_x_squared
                            np.negative(tau_exp_ic, out=tau_exp_ic)
                            np.expm1(tau_exp_ic, out=tau_exp_ic)
                            np.negative(tau_exp_ic, out=tau_exp_ic)
                            fint_ic = tau_exp_ic @ weights
                            physical_factors[ic, overlapping_indices] *= fint_ic
                            continue
                    
                    # Full overlap quadrature (in-place shifted_x -> gp)
                    gp = x_quad[:, np.newaxis] - delta_x_ic[np.newaxis, :]
                    np.multiply(gp, gp, out=gp)
                    np.negative(gp, out=gp)
                    np.exp(gp, out=gp)
                    tau_at_quad_ic = gp @ tg
                    
                    if tau_at_quad_ic.max() < tau_thin_thresh:
                        fint_ic = (gp.T @ weights) * tg
                        physical_factors[ic, overlapping_indices] *= fint_ic
                        continue
                    
                    # Curve-of-growth in-place
                    cog_mask_ic = tau_at_quad_ic > 1e-300
                    integrand_ic = np.negative(tau_at_quad_ic)
                    np.expm1(integrand_ic, out=integrand_ic)
                    np.negative(integrand_ic, out=integrand_ic)  # 1 - exp(-tau)
                    cog_over_tau_ic = np.where(cog_mask_ic,
                                              integrand_ic / np.where(cog_mask_ic, tau_at_quad_ic, 1.0), 0.0)
                    # Reuse gp array for weighted sum
                    gp *= (cog_over_tau_ic * weights)[:, np.newaxis]
                    fint_ic = gp.sum(axis=0)
                    fint_ic *= tg
                    physical_factors[ic, overlapping_indices] *= fint_ic

        return physical_factors

    def _cog_with_tau_shortcuts(self, center_tau: np.ndarray,
                                exp_neg_x_squared: np.ndarray,
                                weights: np.ndarray) -> np.ndarray:
        """Curve-of-growth integration with thin-regime shortcut for isolated lines.
        
        Partitions lines into two regimes:
        - Thin (tau_0 < 0.01): F(tau_0) ~ tau_0 * Sum_k w_k exp(-x_k^2)  (linear, no exp needed)
        - Normal: full Gauss-Hermite quadrature (exact for all tau_0)
        
        Parameters
        ----------
        center_tau : np.ndarray, shape (n_cond, n_lines)
            Line center optical depths
        exp_neg_x_squared : np.ndarray, shape (n_quad,)
            Pre-computed exp(-x_k^2) at quadrature points
        weights : np.ndarray, shape (n_quad,)
            Quadrature weights
            
        Returns
        -------
        np.ndarray, shape (n_cond, n_lines)
            Curve-of-growth integral values F(tau_0)
        """
        cls = self.__class__
        n_cond, n_lines = center_tau.shape
        sqrt_pi_sum = cls._GAUSS_QUAD_SQRT_PI_SUM  # ~ sqrt(pi) from quadrature initialization
        tau_thin = cls._TAU_THIN
        
        # Find max tau across all conditions for each line to classify regime
        tau_max = np.max(center_tau, axis=0) if n_cond > 1 else center_tau[0]
        
        # Fast path: all lines optically thin
        all_thin = tau_max.max() < tau_thin if tau_max.size > 0 else False
        
        if all_thin:
            # All lines optically thin: F(tau_0) ~= tau_0 * sqrt_pi_sum
            return center_tau * sqrt_pi_sum
        
        # Fast path: no lines are thin -- full quadrature for all
        all_normal = tau_max.min() >= tau_thin if tau_max.size > 0 else True
        
        if all_normal:
            tau_exp = center_tau[:, :, np.newaxis] * exp_neg_x_squared
            np.negative(tau_exp, out=tau_exp)
            np.expm1(tau_exp, out=tau_exp)
            np.negative(tau_exp, out=tau_exp)
            return tau_exp @ weights
        
        # Mixed: some thin, some normal
        fint_vals = np.empty_like(center_tau)
        
        thin_mask = tau_max < tau_thin
        normal_mask = ~thin_mask
        
        # Thin regime: F ~ tau_0 * sqrt_pi_sum
        if np.any(thin_mask):
            fint_vals[:, thin_mask] = center_tau[:, thin_mask] * sqrt_pi_sum
        
        # Normal regime: full quadrature (handles all tau_0 values exactly)
        if np.any(normal_mask):
            tau_normal = center_tau[:, normal_mask]
            tau_exp = tau_normal[:, :, np.newaxis] * exp_neg_x_squared
            np.negative(tau_exp, out=tau_exp)
            np.expm1(tau_exp, out=tau_exp)
            np.negative(tau_exp, out=tau_exp)
            fint_vals[:, normal_mask] = tau_exp @ weights
        
        return fint_vals

    def _calc_intensity_core(self, t_kin_vals: np.ndarray, n_mol_vals: np.ndarray, 
                            dv_vals: np.ndarray, method: str = "curve_growth") -> tuple:
        """Core intensity calculation using unified vectorized operations.
        
        This method handles all intensity calculations efficiently using a single
        vectorized code path by converting scalar inputs to arrays.
        
        Parameters
        ----------
        t_kin_vals : np.ndarray
            Temperature values (K) - can be scalar or array
        n_mol_vals : np.ndarray  
            Column density values (cm^-2) - can be scalar or array
        dv_vals : np.ndarray
            Line width values (km/s) - can be scalar or array
        method : str
            Calculation method ("curve_growth" or "radex")
            
        Returns
        -------
        tuple
            (intensities, tau_values) as numpy arrays
        """
        m = self._molecule
        lines = m.lines_as_namedtuple
        partition = m.partition

        # Guard: no lines available in the active wavelength range
        if lines is None or len(lines.freq) == 0:
            empty = np.array([])
            return empty, empty

        # Ensure freq is always a numpy array (guards against stale list fallbacks)
        if not isinstance(lines.freq, np.ndarray):
            from collections import namedtuple as _nt
            lines = lines._replace(
                freq=np.asarray(lines.freq, dtype=np.float64),
                a_stein=np.asarray(lines.a_stein, dtype=np.float64),
                e_up=np.asarray(lines.e_up, dtype=np.float64),
                e_low=np.asarray(lines.e_low, dtype=np.float64),
                g_up=np.asarray(lines.g_up, dtype=np.float64),
                lam=np.asarray(lines.lam, dtype=np.float64),
            )

        # Ensure inputs are numpy arrays with consistent shape
        t_kin_vals = np.asarray(t_kin_vals)
        n_mol_vals = np.asarray(n_mol_vals) 
        dv_vals = np.asarray(dv_vals)
        
        # Convert scalar inputs to 1-element arrays
        was_scalar = t_kin_vals.ndim == 0 and n_mol_vals.ndim == 0 and dv_vals.ndim == 0
        if was_scalar:
            t_kin_vals = np.atleast_1d(t_kin_vals)
            n_mol_vals = np.atleast_1d(n_mol_vals)
            dv_vals = np.atleast_1d(dv_vals)
        
        # Validate temperature bounds
        t_min, t_max = np.min(partition.t), np.max(partition.t)
        if np.any(t_kin_vals < t_min) or np.any(t_kin_vals > t_max):
            raise ValueError(f'Temperature values outside partition function range [{t_min}, {t_max}]')
        
        # Broadcast to common shape
        t_kin_vals, n_mol_vals, dv_vals = np.broadcast_arrays(t_kin_vals, n_mol_vals, dv_vals)
        output_shape = t_kin_vals.shape
        
        # Flatten for efficient processing
        t_kin_flat = t_kin_vals.ravel()
        n_mol_flat = n_mol_vals.ravel()
        dv_flat = dv_vals.ravel()
        
        # Vectorized partition function
        q_sum_vals: np.ndarray = np.interp(t_kin_flat, partition.t, partition.q)
        
        invT = (1.0 / t_kin_flat).astype(np.float64, copy=False)

        # Use pre-computed class constants for speed
        cls = self.__class__
        
        # 1D condition terms - use pre-computed constant
        cond_term = cls._INV_FGAUSS_1E5 / dv_flat                         # (n_cond,)
        q_sum_inv = 1.0 / q_sum_vals                                       # (n_cond,)
        
        # Cache line-level constants (molecule-dependent only, computed once per Intensity instance)
        if self._cached_line_scalar is None:
            freq_cubed = lines.freq ** 3
            line_term = cls._C3_OVER_8PI / freq_cubed                     # (n_lines,)
            self._cached_line_scalar = line_term * lines.g_up * lines.a_stein
            self._cached_e_delta = lines.e_up - lines.e_low
            self._cached_freq_ratio = (lines.freq / c.SPEED_OF_LIGHT_CGS)[np.newaxis, :]
        
        line_scalar = self._cached_line_scalar
        e_delta = self._cached_e_delta

        # --- Fewer exponentials for Boltzmann difference ---
        # Replace einsum with direct outer product broadcasting (faster for 2D)
        n_cond = len(invT)
        
        # E_low = invT[:, np.newaxis] * lines.e_low[np.newaxis, :]
        # Use np.outer for single condition, broadcasting for multiple
        if n_cond == 1:
            # Single condition: use direct multiplication (avoids 2D overhead)
            E_low = invT[0] * lines.e_low
            E_delta_scaled = invT[0] * e_delta
            exp_low = np.exp(-E_low)
            boltz_diff = exp_low * (-np.expm1(-E_delta_scaled))
            cond_scalar = n_mol_flat[0] * cond_term[0] * q_sum_inv[0]
            center_tau = (cond_scalar * boltz_diff * line_scalar)[np.newaxis, :]
        else:
            # Multiple conditions: use broadcasting
            E_low = np.multiply.outer(invT, lines.e_low)                  # (n_cond, n_lines)
            E_delta_scaled = np.multiply.outer(invT, e_delta)             # (n_cond, n_lines)
            exp_low = np.exp(-E_low)
            boltz_diff = exp_low * (-np.expm1(-E_delta_scaled))
            
            # Compute center_tau: cond_scalar[:, None] * boltz_diff * line_scalar[None, :]
            cond_scalar = n_mol_flat * cond_term * q_sum_inv              # (n_cond,)
            center_tau = (cond_scalar[:, np.newaxis] * boltz_diff) * line_scalar
        
        # Blackbody calculation - vectorized for all temperatures at once
        bb_vals = self._bb(lines.freq, t_kin_flat)
        
        # Use cached frequency ratio (computed once per Intensity instance)
        freq_ratio = self._cached_freq_ratio
        
        # Use pre-computed constant
        sqrt_ln2_inv = cls._SQRT_LN2_INV
        
        # Dispatch to the registered strategy for the requested method
        strategy = self._get_strategy(method)
        intensity = strategy.calculate(
            self, center_tau, dv_flat, bb_vals, freq_ratio, sqrt_ln2_inv, lines.freq
        )
        
        if not was_scalar:
            intensity = intensity.reshape(output_shape + (len(lines.freq),))
            center_tau = center_tau.reshape(output_shape + (len(lines.freq),))
        elif was_scalar:
            # Drop only the condition axis; squeeze() would also drop a length-1 line axis
            intensity = intensity.reshape(len(lines.freq))
            center_tau = center_tau.reshape(len(lines.freq))

        return intensity, center_tau

    def calc_intensity(self, t_kin: Optional[float] = None, n_mol: Optional[float] = None, 
                      dv: Optional[float] = None, method: Literal["curve_growth", "radex", "curve_growth_no_overlap"] = "curve_growth") -> None:
        """Calculate the intensity for a given set of physical parameters. This implements Eq. A1 and A2 in Banzatti et al. 2012.

        The calculation method to obtain the intensity from the opacity can be switched between the curve-of-growth
        method used in Banzatti et al. 2012, which considers broadening for high values of tau and the simple expression
        used e.g. in RADEX (van der Tak et al. 2007) which saturates at tau ~ few. For low values (tau < 1), they yield the same values.

        Parameters
        ----------
        t_kin: float, optional
            Kinetic temperature in K
        n_mol: float, optional
            Column density in cm**-2
        dv: float, optional
            Intrinsic (turbulent) line width in km/s
        method: Literal["curve_growth", "radex", "curve_growth_no_overlap"], default "curve_growth"
            Calculation method, either "curve_growth" for Eq. A1 or "radex" for less accurate approximation
        """
        start_time = time.perf_counter()
        
        # Check if we can use cached result (computed with the same method)
        if (self._cache_valid and
            self._t_kin == t_kin and
            self._n_mol == n_mol and
            self._dv == dv and
            self._method == method and
            self._intensity is not None):
            log_timing(f"Intensity.calc_intensity(cache_hit)", time.perf_counter() - start_time, verbose=False)
            return

        # Validate inputs
        if t_kin is None or n_mol is None or dv is None:
            raise ValueError("t_kin, n_mol, and dv must all be provided")

        # Use optimized core calculation
        intensity, tau = self._calc_intensity_core(
            np.asarray(t_kin), np.asarray(n_mol), np.asarray(dv), method
        )
        
        # Update instance variables and cache
        self._t_kin = t_kin
        self._n_mol = n_mol
        self._dv = dv
        self._method = method
        self._tau = tau
        self._intensity = intensity
        self._cache_valid = True
        
        log_timing(f"Intensity.calc_intensity({method})", time.perf_counter() - start_time)

    def calc_intensity_batch(self, t_kin_array: np.ndarray, n_mol_array: np.ndarray, 
                           dv_array: np.ndarray, method: Literal["curve_growth", "radex", "curve_growth_no_overlap"] = "curve_growth") -> np.ndarray:
        """Calculate intensities for multiple parameter combinations using vectorized operations.
        
        This method is optimized for processing many parameter sets simultaneously using the
        unified core calculation engine for maximum efficiency.

        Parameters
        ----------
        t_kin_array: np.ndarray
            Array of kinetic temperatures in K
        n_mol_array: np.ndarray  
            Array of column densities in cm**-2
        dv_array: np.ndarray
            Array of intrinsic line widths in km/s
        method: Literal["curve_growth", "radex", "curve_growth_no_overlap"], default "curve_growth"
            Calculation method

        Returns
        -------
        np.ndarray
            Array with intensities - shape depends on input array broadcasting
        """
        # Use the optimized core calculation
        intensity, _ = self._calc_intensity_core(
            np.asarray(t_kin_array), 
            np.asarray(n_mol_array), 
            np.asarray(dv_array), 
            method
        )
        
        return intensity

    def invalidate_cache(self) -> None:
        """Invalidate the calculation cache, forcing recalculation on next call."""
        self._cache_valid = False
        self._intensity = None
        self._tau = None

    def bulk_parameter_update_vectorized(self, parameter_combinations: list, method: Literal["curve_growth", "radex", "curve_growth_no_overlap"] = "curve_growth") -> np.ndarray:
        """Update multiple parameter combinations and calculate intensities in a vectorized manner.
        
        Parameters
        ----------
        parameter_combinations: list
            List of dictionaries, each containing 't_kin', 'n_mol', and 'dv' keys
        method: Literal["curve_growth", "radex", "curve_growth_no_overlap"], default "curve_growth"
            Calculation method
            
        Returns
        -------
        np.ndarray
            2D array with shape (n_combinations, n_lines) containing intensities
        """
        if not parameter_combinations:
            return np.array([])
        
        # Extract parameter arrays efficiently
        t_kin_vals = np.array([combo['t_kin'] for combo in parameter_combinations])
        n_mol_vals = np.array([combo['n_mol'] for combo in parameter_combinations]) 
        dv_vals = np.array([combo['dv'] for combo in parameter_combinations])
        
        # Use optimized core calculation only
        intensity, _ = self._calc_intensity_core(t_kin_vals, n_mol_vals, dv_vals, method)
        return intensity

    def get_table_in_range(self, lam_min: float, lam_max: float) -> "pandas.DataFrame":
        """Get a table of lines within a specific wavelength window.

        Convenience alias for ``build_table(wavelength_range=(lam_min, lam_max))``.

        Parameters
        ----------
        lam_min : float
            Minimum wavelength in microns.
        lam_max : float
            Maximum wavelength in microns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with line data and computed intensity/tau for
            the requested sub-range.
        """
        return self.build_table(wavelength_range=(lam_min, lam_max))

    def get_lines_in_range_with_intensity(self, lam_min: float, lam_max: float):
        """
        Get MoleculeLine objects in the specified wavelength range with computed intensity and tau values.
        
        Parameters
        ----------
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        list[tuple[MoleculeLine, float, float]]
            List of ``(MoleculeLine, intensity, tau)`` tuples within the range.
        """
        # If no intensity calculated yet, return empty list
        if self._intensity is None or self._tau is None:
            return []
        
        # Get line data arrays once
        lines_array = self.molecule.lines_as_namedtuple
        
        # Use vectorized mask to find indices in range - O(n) instead of O(n*m)
        lam_arr = lines_array.lam
        mask = (lam_arr >= lam_min) & (lam_arr <= lam_max)
        indices_in_range = np.nonzero(mask)[0]
        
        if len(indices_in_range) == 0:
            return []
        
        # Get lines in range from the molecule line list
        lines_in_range = self.molecule.get_lines_in_range(lam_min, lam_max)
        
        # Build result list - lines_in_range and indices_in_range should correspond
        # since both use the same wavelength filtering
        result = []
        intensity_arr = self._intensity
        tau_arr = self._tau
        
        for line, idx in zip(lines_in_range, indices_in_range):
            result.append((line, intensity_arr[idx], tau_arr[idx]))
        
        return result

    @property
    def tau(self) -> Optional[np.ndarray]:
        """np.ndarray: Opacities per line"""
        return self._tau

    @property
    def intensity(self) -> Optional[np.ndarray]:
        """np.ndarray: Calculated intensity per line in erg/s/cm**2/sr/Hz"""
        return self._intensity

    @property
    def molecule(self) -> 'MoleculeLineList':
        """MoleculeLineList: Molecular line list data used for calculation"""
        return self._molecule

    @property
    def t_kin(self) -> Optional[float]:
        """float: Kinetic temperature in K used for calculation"""
        return self._t_kin

    @property
    def n_mol(self) -> Optional[float]:
        """float: Molecular column density in cm**-2 used for calculation"""
        return self._n_mol

    @property
    def dv(self) -> Optional[float]:
        """float: Line width in km/s used for calculation"""
        return self._dv

    # wavelength_range property provided by WavelengthRangeMixin

    def _on_wavelength_range_changed(self, old, new):
        """Hook called by WavelengthRangeMixin when wavelength_range changes."""
        self._molecule.wavelength_range = new
        self.invalidate_cache()
        # Also invalidate pre-computed line constants since the set of lines changed
        self._sorted_idx = None
        self._sorted_freq = None
        self._cached_line_scalar = None
        self._cached_e_delta = None
        self._cached_dv_star = None
        self._cached_freq_ratio = None

    def __repr__(self) -> str:
        return f"Intensity(Mol-Name={self.molecule.name}, t_kin={self.t_kin} n_mol={self.n_mol} dv={self.dv}, " \
               f"tau={self.tau}, intensity={self.intensity})"

    def build_table(
        self,
        *,
        full_range: bool = False,
        wavelength_range: Optional[tuple[float, float]] = None,
    ) -> "pandas.DataFrame":
        """Build a DataFrame of line data with computed intensity and tau.

        By default returns only lines within the active wavelength range
        (same behaviour as the legacy ``get_table`` property).  Use the
        keyword arguments to widen or narrow the output.

        Parameters
        ----------
        full_range : bool, optional
            If ``True``, include all lines in the underlying HITRAN
            file regardless of any wavelength filter on the
            ``MoleculeLineList``.  Intensity and tau are computed for
            every line using the current physical parameters.
            Defaults to ``False`` (active wavelength range only).
        wavelength_range : tuple[float, float], optional
            ``(lam_min, lam_max)`` in microns.  When provided, the
            returned table is filtered to this sub-window.
            * If *full_range* is also ``True``, the sub-window is
              applied to the full line list.
            * If *full_range* is ``False``, the sub-window is applied
              to the active (already-filtered) line list.

        Returns
        -------
        pandas.DataFrame
            Columns: ``lev_up``, ``lev_low``, ``lam``, ``tau``,
            ``intens``, ``a_stein``, ``e_up``, ``e_low``, ``g_up``,
            ``g_low``.

        Raises
        ------
        ImportError
            If *pandas* is not installed.

        Examples
        --------
        >>> # Active range only (default)
        >>> df = intensity.build_table()

        >>> # All lines in the HITRAN file with intensity computed for each
        >>> df_full = intensity.build_table(full_range=True)

        >>> # Narrow sub-window of the active range
        >>> df_sub = intensity.build_table(wavelength_range=(4.9, 5.0))
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")

        df = self._build_table(pd, full_range=full_range)

        # Apply optional wavelength sub-window
        if wavelength_range is not None:
            lam_min, lam_max = wavelength_range
            mask = (df["lam"] >= lam_min) & (df["lam"] <= lam_max)
            df = df.loc[mask].reset_index(drop=True)

        return df

    # ---- internal table builder -------------------------------------

    # Column order used by all table-building paths
    _TABLE_COLUMNS = ('lev_up', 'lev_low', 'lam', 'tau', 'intens',
                      'a_stein', 'e_up', 'e_low', 'g_up', 'g_low')

    def _build_table(self, pd, *, full_range: bool) -> "pandas.DataFrame":
        """Unified table builder for both active-range and full-range modes.

        Parameters
        ----------
        pd : module
            The pandas module (already imported by caller).
        full_range : bool
            If ``True``, use all lines from the HITRAN file and compute intensity and tau for every line, not just those in the active wavelength window.
            This reproduces the behaviour from before wavelength-range filtering was introduced.
            If ``False``, use only the active lines.

        Returns
        -------
        pandas.DataFrame
        """
        self.molecule._ensure_data_loaded()

        if full_range:
            raw = self.molecule._raw_lines_data
        else:
            raw = self.molecule._get_active_raw_data()

        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=list(self._TABLE_COLUMNS))

        n_rows = len(raw)

        # ---- intensity / tau mapping --------------------------------
        if not full_range:
            # Active range: computed arrays align 1-to-1 with *raw*
            tau_col = self._tau if self._tau is not None else np.full(n_rows, np.nan)
            intens_col = self._intensity if self._intensity is not None else np.full(n_rows, np.nan)
        else:
            # Full range: compute intensity for ALL lines so that the
            # population diagram (and similar consumers) see the same
            # data they did before wavelength-range filtering existed.
            intens_col, tau_col = self._compute_full_range_intensity(raw)

        return pd.DataFrame({
            'lev_up': raw['lev_up'],
            'lev_low': raw['lev_low'],
            'lam': raw['lam'],
            'tau': tau_col,
            'intens': intens_col,
            'a_stein': raw['a_stein'],
            'e_up': raw['e_up'],
            'e_low': raw['e_low'],
            'g_up': raw['g_up'].astype(int),
            'g_low': raw['g_low'].astype(int),
        })

    def _compute_full_range_intensity(self, raw) -> tuple:
        """Compute intensity and tau for all lines in the HITRAN file.

        Creates a temporary unfiltered MoleculeLineList and Intensity
        object, then runs the standard calculation pipeline with the
        same physical parameters (t_kin, n_mol, dv) stored on this
        instance. This avoids duplicating any calculation logic.

        Parameters
        ----------
        raw : structured np.ndarray
            The full ``_raw_lines_data`` array from the parent
            MoleculeLineList.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            ``(intensity, tau)`` arrays with length ``len(raw)``.
            Returns arrays filled with NaN if no parameters have
            been set yet.
        """
        n_rows = len(raw)
        if self._t_kin is None or self._n_mol is None or self._dv is None:
            return np.full(n_rows, np.nan), np.full(n_rows, np.nan)

        # The result depends only on the line list and the parameters, and it
        # is asked for on every population-diagram redraw (each selection):
        # reuse the last one when neither changed.  The arrays are read-only
        # because callers get the cached objects.
        params = (self._t_kin, self._n_mol, self._dv, self._method or "curve_growth")
        cached = self._full_range_cache
        if cached is not None and cached[0] is raw and cached[1] == params:
            return cached[2], cached[3]

        # Create a temporary unfiltered MoleculeLineList and Intensity
        # so the full calculation pipeline runs over all lines.
        from .MoleculeLineList import MoleculeLineList as _MLL
        tmp_mll = _MLL(
            molecule_id=self.molecule.molecule_id,
            filename=self.molecule._filename,
            wavelength_range=None,
        )
        tmp_intens = Intensity(tmp_mll, wavelength_range=None)
        try:
            tmp_intens.calc_intensity(
                t_kin=self._t_kin,
                n_mol=self._n_mol,
                dv=self._dv,
                method=self._method or "curve_growth",
            )
            full_i, full_t = tmp_intens._intensity, tmp_intens._tau
        except Exception:
            full_i = full_t = None

        if full_i is not None and len(full_i) == n_rows:
            full_i.setflags(write=False)
            if full_t is not None:
                full_t.setflags(write=False)
            self._full_range_cache = (raw, params, full_i, full_t)
            return (full_i, full_t)

        # The rebuild did not line up with *raw*.  This happens whenever the
        # line list has no source file to rebuild from - a filtered or
        # otherwise synthesized MoleculeLineList - in which case the temporary
        # list comes back empty.  For such a list there is no wider range to
        # widen to, so the arrays already computed on this instance ARE the
        # full-range arrays.  Returning them keeps the row counts consistent;
        # anything else would build a DataFrame from mismatched columns.
        if self._intensity is not None and len(self._intensity) == n_rows:
            tau = (self._tau if self._tau is not None
                   and len(self._tau) == n_rows else np.full(n_rows, np.nan))
            return (self._intensity, tau)
        return np.full(n_rows, np.nan), np.full(n_rows, np.nan)

    # ---- Batch evaluation API ------------------------------------

    def prepare_batch_context(self, line_indices: Optional[np.ndarray] = None) -> dict:
        """Pre-extract all line-level data into a flat dict for fast batch evaluation.

        The returned context dict contains NumPy arrays that are
        independent of the physical parameters (T, N, dv) and can be
        reused across thousands of ``calc_intensity_batch_fast`` calls
        without repeated attribute lookups.

        Parameters
        ----------
        line_indices : np.ndarray, optional
            If provided, only extract data for these line indices
            (into the full molecule line list).  This allows overlap
            grouping and all per-step computation to operate on the
            reduced set of lines, dramatically cutting cost when many
            weak lines have been filtered out.  The returned arrays
            will have shape ``(len(line_indices),)`` and
            ``ctx['line_indices']`` will be set to ``None`` (no further
            slicing needed in ``calc_intensity_batch_fast``).

        Returns
        -------
        dict
            Keys:

            * ``freq`` — line frequencies in Hz, shape ``(n_lines,)``
            * ``e_up``, ``e_low`` — upper/lower energy levels
            * ``e_delta`` — ``e_up - e_low``
            * ``line_scalar`` — ``c^3 / (8 pi nu^3) * g_up * A``
            * ``freq_ratio`` — ``nu / c``, shape ``(1, n_lines)``
            * ``partition_t``, ``partition_q`` — partition function table
            * ``x_quad``, ``w_quad``, ``exp_neg_x2`` — quadrature arrays
            * ``sqrt_pi_sum`` — thin-regime constant
            * ``overlap`` — ``None`` until ``prepare_overlap_structure`` is called
        """
        cls = self.__class__
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()

        m = self._molecule
        lines = m.lines_as_namedtuple
        partition = m.partition

        # If line_indices provided, slice all line-level arrays to the subset
        if line_indices is not None:
            freq = np.ascontiguousarray(lines.freq[line_indices])
            e_up = np.ascontiguousarray(lines.e_up[line_indices])
            e_low_arr = np.ascontiguousarray(lines.e_low[line_indices])
            g_up = lines.g_up[line_indices]
            a_stein = lines.a_stein[line_indices]
        else:
            freq = np.ascontiguousarray(lines.freq)
            e_up = np.ascontiguousarray(lines.e_up)
            e_low_arr = np.ascontiguousarray(lines.e_low)
            g_up = lines.g_up
            a_stein = lines.a_stein

        freq_cubed = freq ** 3
        line_scalar = (cls._C3_OVER_8PI / freq_cubed) * g_up * a_stein
        e_delta = e_up - e_low_arr
        freq_ratio = (freq / c.SPEED_OF_LIGHT_CGS)[np.newaxis, :]

        # Float32 line-level arrays (cast once, reused every batch step)
        e_low_f32 = e_low_arr.astype(np.float32)
        e_delta_f32 = e_delta.astype(np.float32)
        line_scalar_f32 = line_scalar.astype(np.float32)
        freq_ratio_f32 = freq_ratio.astype(np.float32)
        freq_f32 = freq.astype(np.float32)

        # Float32 quadrature arrays
        x_quad_f32 = cls._GAUSS_QUAD_X.astype(np.float32)
        w_quad_f32 = cls._GAUSS_QUAD_W.astype(np.float32)
        exp_neg_x2_f32 = cls._GAUSS_QUAD_EXP.astype(np.float32)
        sqrt_pi_sum_f32 = np.float32(cls._GAUSS_QUAD_SQRT_PI_SUM)
        two_sqrt_ln2_f32 = np.float32(cls._TWO_SQRT_LN2)
        tau_thin_f32 = np.float32(cls._TAU_THIN)

        return {
            'freq': freq,
            'e_up': e_up,
            'e_low': e_low_arr,
            'e_delta': e_delta,
            'line_scalar': line_scalar,
            'freq_ratio': freq_ratio,
            'partition_t': np.ascontiguousarray(partition.t),
            'partition_q': np.ascontiguousarray(partition.q),
            'x_quad': cls._GAUSS_QUAD_X,
            'w_quad': cls._GAUSS_QUAD_W,
            'exp_neg_x2': cls._GAUSS_QUAD_EXP,
            'sqrt_pi_sum': cls._GAUSS_QUAD_SQRT_PI_SUM,
            'overlap': None,
            'n_lines': len(freq),
            # Pre-cast float32 arrays
            'e_low_f32': e_low_f32,
            'e_delta_f32': e_delta_f32,
            'line_scalar_f32': line_scalar_f32,
            'freq_ratio_f32': freq_ratio_f32,
            'freq_f32': freq_f32,
            'x_quad_f32': x_quad_f32,
            'w_quad_f32': w_quad_f32,
            'exp_neg_x2_f32': exp_neg_x2_f32,
            'sqrt_pi_sum_f32': sqrt_pi_sum_f32,
            'two_sqrt_ln2_f32': two_sqrt_ln2_f32,
            'tau_thin_f32': tau_thin_f32,
            # When line_indices are baked in, no further slicing needed
            'line_indices': None,
        }

    def prepare_overlap_structure(self, ctx: dict, dv: float) -> None:
        """Pre-compute overlap grouping at a reference line width.

        This populates ``ctx['overlap']`` with the group structure
        needed by ``calc_intensity_batch_fast``.

        Parameters
        ----------
        ctx : dict
            Context dict returned by ``prepare_batch_context()``.
        dv : float
            Reference Doppler width in km/s used to determine which
            lines overlap.
        """
        freq = ctx['freq']
        n_lines = len(freq)

        sort_idx = np.argsort(freq)
        sorted_freq = freq[sort_idx]

        if n_lines <= 1:
            ctx['overlap'] = {'all_isolated': True}
            return

        dv_star = (c.SPEED_OF_LIGHT_KMS
                   * np.diff(sorted_freq)
                   / np.maximum(sorted_freq[:-1], sorted_freq[1:]))

        if np.all(dv < dv_star):
            ctx['overlap'] = {'all_isolated': True}
            return

        # Cumsum-based group labeling (same logic as _fint_multi)
        is_boundary = dv < dv_star
        group_labels = np.empty(n_lines, dtype=np.int32)
        group_labels[0] = 0
        group_labels[1:] = np.cumsum(is_boundary)

        group_counts = np.bincount(group_labels)
        line_group_counts = group_counts[group_labels]

        isolated_mask = line_group_counts == 1
        blended_mask = ~isolated_mask

        # Map back to original (unsorted) indices
        isolated_orig = sort_idx[isolated_mask]
        blended_sort_positions = np.where(blended_mask)[0]
        blended_orig = sort_idx[blended_sort_positions]
        blended_glabels = group_labels[blended_sort_positions]

        unique_labels, group_starts = np.unique(blended_glabels, return_index=True)
        group_ends = np.append(group_starts[1:], len(blended_orig))
        group_sizes = group_ends - group_starts

        # Build per-group index arrays
        groups = []
        for i in range(len(unique_labels)):
            groups.append(blended_orig[group_starts[i]:group_ends[i]])

        # Pre-compute size-bucketed arrays
        # NOTE: freq_all is kept in float64 — float32 causes catastrophic
        # precision loss in freq_ref / delta_x for large blended groups
        # whose frequencies span ~2e13 Hz (only 7 sig figs in f32).
        freq_f64 = ctx['freq']  # already float64
        size_buckets = {}
        unique_sizes = np.unique(group_sizes)
        for target_size in unique_sizes:
            sz = int(target_size)
            gidx_list = [i for i, gs in enumerate(group_sizes) if gs == sz]
            all_idx = np.empty((len(gidx_list), sz), dtype=np.intp)
            for k, gi in enumerate(gidx_list):
                all_idx[k] = groups[gi]
            freq_bucket = freq_f64[all_idx]  # (n_grps, target_size) float64
            size_buckets[sz] = {
                'all_idx': all_idx,
                'freq_all': freq_bucket,
                'n_grps': len(gidx_list),
            }

        ctx['overlap'] = {
            'all_isolated': False,
            'sort_idx': sort_idx,
            'dv_star': dv_star,
            'isolated_indices': isolated_orig,
            'n_isolated': int(isolated_mask.sum()),
            'n_blended_lines': int(blended_mask.sum()),
            'n_groups': len(groups),
            'groups': groups,
            'group_sizes': group_sizes,
            'size_buckets': size_buckets,
        }

    def calc_intensity_batch_fast(self,
                                  t_kin_array: np.ndarray,
                                  n_mol_array: np.ndarray,
                                  dv_array: np.ndarray,
                                  ctx: dict) -> np.ndarray:
        """Fast batch intensity calculation for batch fitting using pre-computed context.

        Bypasses ``_fint_multi`` entirely, operating
        directly on the pre-computed overlap structure from
        ``prepare_overlap_structure`` to avoid re-sorting, re-grouping,
        and per-condition Python loops.

        All heavy computation is vectorized across walkers (conditions)
        simultaneously using ``float32`` arithmetic to halve memory
        bandwidth and improve cache utilisation.

        Parameters
        ----------
        t_kin_array : np.ndarray, shape ``(n_walkers,)``
            Kinetic temperatures in K.
        n_mol_array : np.ndarray, shape ``(n_walkers,)``
            Column densities in cm⁻².
        dv_array : np.ndarray, shape ``(n_walkers,)``
            Line widths in km/s.
        ctx : dict
            Pre-computed context from ``prepare_batch_context`` (with
            overlap structure populated by ``prepare_overlap_structure``).

        Returns
        -------
        np.ndarray, shape ``(n_walkers, n_lines)``
            Line intensities for every walker and every line.
        """
        cls = self.__class__

        # Unpack pre-computed arrays (float64 originals + cached float32)
        freq = ctx['freq']
        part_t = ctx['partition_t']
        part_q = ctx['partition_q']
        n_lines = ctx['n_lines']

        # Use cached float32 arrays
        e_low_f = ctx['e_low_f32']
        e_delta_f = ctx['e_delta_f32']
        line_scalar_f = ctx['line_scalar_f32']
        freq_ratio_f = ctx['freq_ratio_f32']

        t_kin_array = np.asarray(t_kin_array, dtype=np.float64)
        n_mol_array = np.asarray(n_mol_array, dtype=np.float64)
        dv_array = np.asarray(dv_array, dtype=np.float64)
        n_cond = len(t_kin_array)

        # --- Partition function interpolation ---
        q_vals = np.interp(t_kin_array, part_t, part_q)
        invT = 1.0 / t_kin_array

        # --- Boltzmann + tau computation (float32 for batch speed) ---
        cond_scalar = (n_mol_array * cls._INV_FGAUSS_1E5 / dv_array) / q_vals

        invT_f = invT.astype(np.float32)
        cond_scalar_f = cond_scalar.astype(np.float32)

        if n_cond == 1:
            e_low_scaled = np.exp(-invT_f[0] * e_low_f)
            boltz_diff = e_low_scaled * (-np.expm1(-invT_f[0] * e_delta_f))
            center_tau = (cond_scalar_f[0] * boltz_diff * line_scalar_f)[np.newaxis, :]
        else:
            E_low_2d = np.multiply.outer(invT_f, e_low_f)
            E_delta_2d = np.multiply.outer(invT_f, e_delta_f)
            exp_low_2d = np.exp(-E_low_2d)
            boltz_diff = exp_low_2d * (-np.expm1(-E_delta_2d))
            center_tau = (cond_scalar_f[:, np.newaxis] * boltz_diff) * line_scalar_f

        # --- Blackbody (kept in float64 for numerical safety, cast after) ---
        bb_vals = self._bb(freq, t_kin_array).astype(np.float32)

        # --- Physical factors: (freq_ratio * bb) * sqrt_ln2_inv * 1e5 * dv ---
        physical_factors = np.empty((n_cond, n_lines), dtype=np.float32)
        np.multiply(freq_ratio_f, bb_vals, out=physical_factors)
        physical_factors *= np.float32(cls._SQRT_LN2_INV * 1e5)
        dv_f = dv_array.astype(np.float32)
        physical_factors *= dv_f[:, np.newaxis]

        # --- Quadrature arrays (pre-cast in prepare_batch_context) ---
        exp_neg_x2 = ctx['exp_neg_x2_f32']
        weights = ctx['w_quad_f32']
        sqrt_pi_sum = ctx['sqrt_pi_sum_f32']
        x_quad = ctx['x_quad_f32']
        two_sqrt_ln2 = ctx['two_sqrt_ln2_f32']
        tau_thin = ctx['tau_thin_f32']

        # ===============================================================
        # Use pre-computed overlap structure — NO re-sorting / re-grouping
        # ===============================================================
        overlap = ctx.get('overlap')
        all_isolated = (overlap is None or overlap.get('all_isolated', True))

        if all_isolated:
            # --- Fast path: every line is isolated ---
            fint = self._cog_batch_f32(center_tau, exp_neg_x2, weights,
                                       sqrt_pi_sum, tau_thin)
            np.multiply(physical_factors, fint, out=physical_factors)
            return physical_factors

        # --- Mixed: isolated lines + blended groups ---
        isolated_idx = overlap['isolated_indices']  # original-order indices
        groups = overlap['groups']                  # list of index arrays

        # Process isolated lines (vectorized across all conditions)
        if len(isolated_idx) > 0:
            tau_iso = center_tau[:, isolated_idx]
            fint_iso = self._cog_batch_f32(tau_iso, exp_neg_x2, weights,
                                           sqrt_pi_sum, tau_thin)
            physical_factors[:, isolated_idx] *= fint_iso

        # --- Process blended groups fully vectorized across conditions ---
        # Use pre-computed size buckets
        size_buckets = overlap['size_buckets']

        for target_size, bucket in size_buckets.items():
            n_grps = bucket['n_grps']
            all_idx = bucket['all_idx']      # (n_grps, target_size)
            freq_all = bucket['freq_all']    # (n_grps, target_size)

            # Gather arrays for all groups and ALL conditions at once
            # tau_all: (n_cond, n_grps, target_size)
            tau_all = center_tau[:, all_idx.ravel()].reshape(
                n_cond, n_grps, target_size)

            # Skip groups that are negligible across ALL conditions
            tau_max_per_group = tau_all.max(axis=(0, 2))  # (n_grps,)
            active = tau_max_per_group > 1e-30
            if not np.any(active):
                continue

            tau_a = tau_all[:, active]          # (n_cond, n_active, sz)
            freq_a = freq_all[active]           # (n_active, sz)
            idx_a = all_idx[active]             # (n_active, sz)
            n_active = idx_a.shape[0]

            # tau-weighted reference frequency per condition per group
            # Computed in float64 to avoid catastrophic precision loss
            # when freq ~ 2e13 Hz (only 7 sig figs in f32).
            tau_a_f64 = tau_a.astype(np.float64)
            tau_sums = tau_a_f64.sum(axis=2)  # (n_cond, n_active)
            safe_sums = np.where(tau_sums > 1e-300, tau_sums, 1.0)
            # freq_a is now float64 (stored that way in prepare_overlap_structure)
            freq_refs = (tau_a_f64 * freq_a[np.newaxis, :, :]).sum(axis=2) / safe_sums

            # Velocity offsets in x-units: (n_cond, n_active, sz) — float64
            safe_freq_refs = np.where(freq_refs > 0, freq_refs, 1.0)
            dv_kms = ((freq_a[np.newaxis, :, :] - freq_refs[:, :, np.newaxis])
                      * (c.SPEED_OF_LIGHT_KMS / safe_freq_refs[:, :, np.newaxis]))
            delta_x = dv_kms * (float(two_sqrt_ln2) / dv_array[:, np.newaxis, np.newaxis])

            # Shifted Gaussian profiles: (n_quad, n_cond, n_active, sz)
            # Computed in float64 then cast to f32 for the quadrature dot products.
            shifted = (x_quad.astype(np.float64)[:, np.newaxis, np.newaxis, np.newaxis]
                       - delta_x[np.newaxis, :, :, :])
            np.multiply(shifted, shifted, out=shifted)
            np.negative(shifted, out=shifted)
            np.exp(shifted, out=shifted)
            shifted = shifted.astype(np.float32)  # back to f32 for speed

            # Total tau at each quad point: (n_quad, n_cond, n_active)
            # Reshaped matmul replaces einsum for speed:
            # shifted: (n_quad, n_cond, n_active, sz) → (n_quad*n_cond*n_active, sz)
            # tau_a:   (n_cond, n_active, sz)         → (n_cond*n_active, sz, 1) broadcast
            n_quad = shifted.shape[0]
            # Reshape to (n_quad, n_cond*n_active, sz) @ (n_cond*n_active, sz, 1)
            shifted_2d = shifted.reshape(n_quad, n_cond * n_active, target_size)
            tau_a_flat = tau_a.reshape(n_cond * n_active, target_size)
            # Batched dot via element-wise multiply + sum (avoids 4D einsum overhead)
            tau_at_quad = (shifted_2d * tau_a_flat[np.newaxis, :, :]).sum(axis=2)
            tau_at_quad = tau_at_quad.reshape(n_quad, n_cond, n_active)

            # Curve-of-growth: 1 - exp(-tau_tot)
            cog_mask = tau_at_quad > 1e-30
            integrand = np.empty_like(tau_at_quad)
            np.negative(tau_at_quad, out=integrand)
            np.expm1(integrand, out=integrand)
            np.negative(integrand, out=integrand)
            cog_over_tau = np.where(
                cog_mask,
                integrand / np.where(cog_mask, tau_at_quad, np.float32(1.0)),
                np.float32(0.0))

            # Distribute intensity proportionally
            # gp * (cog/tau * w): (n_quad, n_cond, n_active, sz)
            shifted *= (cog_over_tau * weights[:, np.newaxis, np.newaxis])[:, :, :, np.newaxis]
            fint = shifted.sum(axis=0)  # (n_cond, n_active, sz)
            fint *= tau_a

            # Scatter back into physical_factors
            # idx_a is (n_active, sz) — scatter across all conditions
            idx_flat = idx_a.ravel()  # (n_active * sz,)
            fint_flat = fint.reshape(n_cond, -1)  # (n_cond, n_active * sz)
            physical_factors[:, idx_flat] *= fint_flat

        # --- Optional pre-slicing: return only lines in fitting range ---
        cached_idx = ctx.get('line_indices')
        if cached_idx is not None:
            return physical_factors[:, cached_idx]

        return physical_factors

    @staticmethod
    def _cog_batch_f32(center_tau: np.ndarray,
                       exp_neg_x2: np.ndarray,
                       weights: np.ndarray,
                       sqrt_pi_sum: np.float32,
                       tau_thin: np.float32) -> np.ndarray:
        """Curve-of-growth integration in float32 for isolated lines.

        Mirrors ``_cog_with_tau_shortcuts`` but operates entirely in
        ``float32`` and is a static method (no instance lookups).

        Parameters
        ----------
        center_tau : (n_cond, n_lines) float32
        exp_neg_x2 : (n_quad,) float32
        weights : (n_quad,) float32
        sqrt_pi_sum : float32
        tau_thin : float32

        Returns
        -------
        (n_cond, n_lines) float32
        """
        n_cond = center_tau.shape[0]
        tau_max = np.max(center_tau, axis=0) if n_cond > 1 else center_tau[0]

        tmax_global = tau_max.max() if tau_max.size > 0 else np.float32(0.0)

        # All optically thin
        if tmax_global < tau_thin:
            return center_tau * sqrt_pi_sum

        # All normal (no thin lines)
        tmin_global = tau_max.min() if tau_max.size > 0 else np.float32(1.0)
        if tmin_global >= tau_thin:
            tau_exp = center_tau[:, :, np.newaxis] * exp_neg_x2
            np.negative(tau_exp, out=tau_exp)
            np.expm1(tau_exp, out=tau_exp)
            np.negative(tau_exp, out=tau_exp)
            return tau_exp @ weights

        # Mixed
        fint = np.empty_like(center_tau)
        thin_mask = tau_max < tau_thin
        normal_mask = ~thin_mask

        if np.any(thin_mask):
            fint[:, thin_mask] = center_tau[:, thin_mask] * sqrt_pi_sum

        if np.any(normal_mask):
            tau_n = center_tau[:, normal_mask]
            tau_exp = tau_n[:, :, np.newaxis] * exp_neg_x2
            np.negative(tau_exp, out=tau_exp)
            np.expm1(tau_exp, out=tau_exp)
            np.negative(tau_exp, out=tau_exp)
            fint[:, normal_mask] = tau_exp @ weights

        return fint

    # ---- backward-compatible aliases --------------------------------
    @property
    def get_table(self) -> "pandas.DataFrame":
        """pd.DataFrame: Line data for the active wavelength range.

        .. deprecated::
            Prefer ``build_table()`` which accepts ``full_range`` and
            ``wavelength_range`` keyword arguments.  This property is
            kept for backward compatibility.
        """
        return self.build_table()

    @property
    def table(self) -> "pandas.DataFrame":
        """Alias for ``build_table()`` (active range only)."""
        return self.build_table()

    def _repr_html_(self) -> Optional[str]:
        # noinspection PyProtectedMember
        pd = _get_pandas()
        return self.build_table()._repr_html_() if pd is not None else None

    # ------------------------------------------------------------------
    # Axis label registry (shared with the plotting layer)
    # ------------------------------------------------------------------
    # Population-diagram data
    # ------------------------------------------------------------------
    def get_population_diagram_data(
        self,
        radius: float,
        distance: float,
        *,
        molecule: Optional[Any] = None,
        full_range: bool = True,
    ) -> Optional[dict]:
        """Compute all arrays needed for a Boltzmann / rotation diagram.

        This is the canonical source of population-diagram data.  The
        plotting layer (e.g. ``PopulationDiagramPlot``) should call this
        method and use the returned dict for rendering rather than
        re-implementing the physics itself.

        Parameters
        ----------
        radius : float
            Emitting radius in AU.
        distance : float
            Distance to source in pc.
        molecule : optional
            Molecule object.  When supplied the instrumental and Keplerian
            FWHM breakdown arrays (``fwhm_instrumental_kms`` and
            ``fwhm_convolved_kms``) are included in the result.
        full_range : bool
            If ``True`` (default) include all lines in the underlying
            line list.  If ``False`` restrict to the active wavelength
            range.

        Returns
        -------
        dict or None
            ``None`` when no intensity data is available.  Otherwise a
            dict with keys:

            ``eu``, ``rd_yax``, ``wavelength``, ``intens``, ``a_stein``,
            ``g_up``, ``g_low``, ``lev_up``, ``lev_low``, ``e_low``,
            ``tau``, ``valid_mask``, ``beam_s``,
            ``fwhm_instrumental_kms``, ``fwhm_convolved_kms``, ``freq``
            (Hz) and ``velocity_sep_kms`` (Doppler separation to the
            nearest line among those returned).
        """
        df = self.build_table(full_range=full_range)
        if df is None or df.empty:
            return None

        wavelength = np.asarray(df["lam"])
        intens_mod = np.asarray(df["intens"])
        Astein_mod = np.asarray(df["a_stein"])
        gu         = np.asarray(df["g_up"])
        eu         = np.asarray(df["e_up"])

        lev_up = np.asarray(df["lev_up"]) if "lev_up" in df.columns else None
        lev_low = np.asarray(df["lev_low"]) if "lev_low" in df.columns else None
        e_low   = np.asarray(df["e_low"])  if "e_low"  in df.columns else None
        g_low   = np.asarray(df["g_low"])  if "g_low"  in df.columns else None
        tau     = np.asarray(df["tau"], dtype=float) if "tau" in df.columns else None

        area   = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
        dist   = distance * c.PARSEC_CM
        beam_s = area / dist ** 2

        F         = intens_mod * beam_s
        frequency = c.SPEED_OF_LIGHT_MICRONS / wavelength
        freq_col  = np.asarray(df["freq"], dtype=float) if "freq" in df.columns else frequency

        # Nearest-neighbour Doppler separation over the lines returned here,
        # computed exactly as the line-list filter's velocity-separation test.
        from iSLAT.Modules.DataProcessing.LineFilterExpression import (
            nearest_neighbour_separation_kms,
        )
        velocity_sep_kms = nearest_neighbour_separation_kms(wavelength)

        with np.errstate(divide="ignore", invalid="ignore"):
            rd_yax = np.log(
                4 * np.pi * F
                / (Astein_mod * c.PLANCK_CONSTANT * frequency * gu)
            )

        threshold  = np.nanmax(F) / 100 if np.any(F > 0) else 0
        valid_mask = F > threshold

        # Per-line FWHM (instrumental + Keplerian convolution)
        fwhm_instrumental_kms: Optional[np.ndarray] = None
        fwhm_convolved_kms:    Optional[np.ndarray] = None
        if molecule is not None and len(wavelength) > 0:
            try:
                from iSLAT.Modules.DataProcessing.InstrumentalProfiles import (
                    PROFILE_REGISTRY, ConstantProfile,
                )
                profile_key = (
                    getattr(molecule, "instrumental_profile_key", "constant") or "constant"
                )
                profile_cls = PROFILE_REGISTRY.get(profile_key, ConstantProfile)
                _fwhm_const = getattr(molecule, "fwhm", 160.0)
                profile = (
                    profile_cls(_fwhm_const) if profile_key == "constant"
                    else profile_cls()
                )
                R_arr = np.broadcast_to(
                    np.asarray(profile.get_R(wavelength), dtype=float),
                    np.shape(wavelength),
                )
                R_arr = np.where(
                    (np.isfinite(R_arr)) & (R_arr > 0),
                    R_arr,
                    c.SPEED_OF_LIGHT_KMS / _fwhm_const,
                )
                fwhm_inst_arr = c.SPEED_OF_LIGHT_KMS / R_arr
                _kep_raw = getattr(molecule, "keplerian_fwhm", None)
                fwhm_kep = float(_kep_raw) if (_kep_raw is not None and _kep_raw > 0) else 0.0
                fwhm_instrumental_kms = fwhm_inst_arr
                fwhm_convolved_kms    = np.sqrt(fwhm_inst_arr ** 2 + fwhm_kep ** 2)
            except Exception:
                pass

        return {
            "eu":                    eu,
            "rd_yax":                rd_yax,
            "wavelength":            wavelength,
            "intens":                intens_mod,
            "a_stein":               Astein_mod,
            "g_up":                  gu,
            "g_low":                 g_low,
            "lev_up":                lev_up,
            "lev_low":               lev_low,
            "e_low":                 e_low,
            "tau":                   tau,
            "valid_mask":            valid_mask,
            "beam_s":                beam_s,
            "fwhm_instrumental_kms": fwhm_instrumental_kms,
            "fwhm_convolved_kms":    fwhm_convolved_kms,
            "freq":                  freq_col,
            "velocity_sep_kms":      velocity_sep_kms,
        }

    # ------------------------------------------------------------------
    # Per-line information helpers  (used by GUI data-field display)
    # ------------------------------------------------------------------
    @staticmethod
    def get_line_info(
        line: Any,
        intensity: float,
        tau: Optional[float] = None,
        data_flux_in_range: Optional[float] = None,
        model_flux_in_range: Optional[float] = None,
        opacity_in_range: Optional[float] = None,
        molecule: Optional[Any] = None,
    ) -> dict:
        """Build a structured information dict for a single molecular line.

        This is the canonical source of per-line metadata used by both
        the GUI data-field display and standalone notebooks/scripts.
        The plotting layer delegates to this method rather than
        re-implementing the logic itself.

        Parameters
        ----------
        line : MoleculeLine
            The molecular transition.
        intensity : float
            Computed model intensity for this line.
        tau : float, optional
            Line opacity.
        data_flux_in_range : float, optional
            Observed flux integral in the selection range (erg s⁻¹ cm⁻²).
        model_flux_in_range : float, optional
            Model flux integral in the selection range (erg s⁻¹ cm⁻²).
        opacity_in_range : float, optional
            Integral of the model optical-depth profile over the selection
            range, i.e. the summed line-center optical depth of the lines in
            it (dimensionless; see
            :func:`~iSLAT.Modules.DataProcessing.spectral_utils.opacity_integral`).
        molecule : optional
            Active molecule.  When provided the instrumental, Keplerian,
            and convolved FWHM at the line wavelength are included.

        Returns
        -------
        dict
            Keys: ``lam``, ``e_up``, ``e_low``, ``a_stein``, ``g_up``,
            ``g_low``, ``up_lev``, ``low_lev``, ``intensity``, ``tau``,
            ``data_flux_in_range``, ``model_flux_in_range``,
            ``fwhm_instrumental_kms``, ``fwhm_keplerian_kms``,
            ``fwhm_convolved_kms``, ``formatted_text``.
        """
        lam     = getattr(line, "lam",     None)
        e_up    = getattr(line, "e_up",    None)
        e_low   = getattr(line, "e_low",   None)
        a_stein = getattr(line, "a_stein", None)
        g_up    = getattr(line, "g_up",    None)
        g_low   = getattr(line, "g_low",   None)
        up_lev  = getattr(line, "lev_up",  None) or "N/A"
        low_lev = getattr(line, "lev_low", None) or "N/A"
        tau_val = tau if tau is not None else "N/A"

        fwhm_inst = None
        fwhm_kep  = None
        fwhm_conv = None
        if molecule is not None and lam is not None:
            try:
                from iSLAT.Modules.DataProcessing.InstrumentalProfiles import (
                    PROFILE_REGISTRY, ConstantProfile,
                )
                profile_key = (
                    getattr(molecule, "instrumental_profile_key", "constant") or "constant"
                )
                profile_cls = PROFILE_REGISTRY.get(profile_key, ConstantProfile)
                _fwhm_const = getattr(molecule, "fwhm", 160.0)
                profile     = (
                    profile_cls(_fwhm_const) if profile_key == "constant"
                    else profile_cls()
                )
                R_inst = float(np.atleast_1d(
                    np.asarray(profile.get_R(np.array([lam])), dtype=float)
                )[0])
                if not np.isfinite(R_inst) or R_inst <= 0:
                    R_inst = c.SPEED_OF_LIGHT_KMS / _fwhm_const
                fwhm_inst = c.SPEED_OF_LIGHT_KMS / R_inst
                _kep_raw  = getattr(molecule, "keplerian_fwhm", None)
                fwhm_kep  = float(_kep_raw) if (_kep_raw is not None and _kep_raw > 0) else 0.0
                fwhm_conv = float(np.sqrt(fwhm_inst ** 2 + fwhm_kep ** 2))
            except Exception:
                pass

        wav_s   = f"{lam:.6f}"       if lam     is not None                       else "N/A"
        a_s     = f"{a_stein:.3e}"   if a_stein is not None                       else "N/A"
        e_s     = f"{e_up:.0f}"      if e_up    is not None                       else "N/A"
        tau_s   = f"{tau_val:.3f}"   if isinstance(tau_val, (int, float))         else str(tau_val)
        dflux_s = f"{data_flux_in_range:.3e}"  if data_flux_in_range  is not None else "N/A"
        mflux_s = f"{model_flux_in_range:.3e}" if model_flux_in_range is not None else "N/A"
        moptical_s = f"{opacity_in_range:.4g}" if opacity_in_range is not None else "N/A"

        fwhm_block = ""
        if fwhm_inst is not None:
            profile_label = (
                getattr(molecule, "instrumental_profile_key", "constant") or "constant"
            )
            fwhm_block = (
                f"--- FWHM breakdown ({profile_label}) ---\n"
                f"Instrumental FWHM (km/s) = {fwhm_inst:.2f}\n"
                f"Keplerian FWHM (km/s) = {fwhm_kep:.2f}\n"
                f"Convolved FWHM (km/s) = {fwhm_conv:.2f}\n"
            )

        text = (
            "\n--- Line Information ---\n"
            "Selected line:\n"
            f"Upper level = {up_lev}\n"
            f"Lower level = {low_lev}\n"
            f"Wavelength (μm) = {wav_s}\n"
            f"Einstein-A coeff. (1/s) = {a_s}\n"
            f"Upper level energy (K) = {e_s}\n"
            f"Line center flux (erg/s/cm2) = {intensity:.3g}\n"
            f"Line center opacity = {tau_s}\n"
            #f"Model opacity in range (sum of line-center opacities) = {moptical_s}\n"
            f"Sum of model line-center opacities in range = {moptical_s}\n"
            f"Data flux in range (erg/s/cm2) = {dflux_s}\n"
            f"Model flux in range (erg/s/cm2) = {mflux_s}\n"
            + fwhm_block
        )

        return {
            "lam":                   lam,
            "e_up":                  e_up,
            "e_low":                 e_low if e_low else "N/A",
            "a_stein":               a_stein,
            "g_up":                  g_up,
            "g_low":                 g_low if g_low else "N/A",
            "up_lev":                up_lev,
            "low_lev":               low_lev,
            "intensity":             intensity,
            "tau":                   tau_val,
            "data_flux_in_range":    data_flux_in_range,
            "model_flux_in_range":   model_flux_in_range,
            "fwhm_instrumental_kms": fwhm_inst,
            "fwhm_keplerian_kms":    fwhm_kep,
            "fwhm_convolved_kms":    fwhm_conv,
            "formatted_text":        text,
        }

    @staticmethod
    def get_line_info_dataframe(line_data: list) -> "pandas.DataFrame":
        """Build a :class:`~pandas.DataFrame` with one row per molecular line.

        Parameters
        ----------
        line_data : list of (MoleculeLine, intensity, tau)
            Line tuples as returned by
            ``Intensity.get_lines_in_range_with_intensity()``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``wavelength_um``, ``e_up_K``, ``e_low_K``,
            ``a_stein``, ``g_up``, ``g_low``, ``upper_level``,
            ``lower_level``, ``intensity``, ``tau``.
            Rows are sorted by wavelength.
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("pandas is required for get_line_info_dataframe")

        rows = []
        for line, intens_val, tau in line_data:
            info = Intensity.get_line_info(line, intens_val, tau)
            rows.append({
                "wavelength_um": info["lam"],
                "e_up_K":        info["e_up"],
                "e_low_K":       info["e_low"],
                "a_stein":       info["a_stein"],
                "g_up":          info["g_up"],
                "g_low":         info["g_low"],
                "upper_level":   info["up_lev"],
                "lower_level":   info["low_lev"],
                "intensity":     info["intensity"],
                "tau":           info["tau"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("wavelength_um", ignore_index=True)
        return df

# ═══════════════════════════════════════════════════════════════════
#  Register standard physics properties with PlotAxisRegistry
# ═══════════════════════════════════════════════════════════════════
def _register_standard_entries() -> None:
    """Register all standard Intensity / MoleculeLine properties."""
    from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisEntry, PlotAxisRegistry
    import numpy as _np

    def _arr(key):
        """Return a resolve_array callback for a direct component-dict key."""
        def _resolve(cdata, _k=key):
            arr = cdata.get(_k)
            if arr is None:
                return None
            try:
                return _np.asarray(arr, dtype=float)
            except (ValueError, TypeError):
                return None
        return _resolve

    def _scalar(attr):
        """Return a resolve_scalar callback for a MoleculeLine attribute."""
        def _resolve(line, _mol_id, _a=attr):
            val = getattr(line, _a, None)
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return _resolve

    def _labels(key):
        """Return a resolve_array callback for a categorical component-dict key."""
        def _resolve(cdata, _k=key):
            arr = cdata.get(_k)
            if arr is None:
                return None
            return _np.asarray(arr, dtype=object)
        return _resolve

    def _freq_array(cdata):
        """Line-list frequencies (Hz), or c/λ when the data carries none."""
        arr = cdata.get("freq")
        if arr is not None:
            return _np.asarray(arr, dtype=float)
        wav = cdata.get("wavelength")
        if wav is None:
            return None
        with _np.errstate(divide="ignore", invalid="ignore"):
            return c.SPEED_OF_LIGHT_MICRONS / _np.asarray(wav, dtype=float)

    def _freq_scalar(line, _mol_id):
        val = getattr(line, "freq", None)
        if val:
            return float(val)
        lam = getattr(line, "lam", None)
        return c.SPEED_OF_LIGHT_MICRONS / float(lam) if lam else None

    def _velocity_sep_array(cdata):
        """Nearest-neighbour separation (km/s), as the line-list filter computes it."""
        arr = cdata.get("velocity_sep_kms")
        if arr is None:
            wav = cdata.get("wavelength")
            if wav is None:
                return None
            from iSLAT.Modules.DataProcessing.LineFilterExpression import (
                nearest_neighbour_separation_kms,
            )
            arr = nearest_neighbour_separation_kms(wav)
        arr = _np.asarray(arr, dtype=float)
        # A lone line is infinitely isolated: no finite position on an axis.
        return _np.where(_np.isinf(arr), _np.nan, arr)

    def _vib_band_array(cdata):
        up, low = cdata.get("lev_up"), cdata.get("lev_low")
        if up is None or low is None:
            return None
        from iSLAT.Modules.DataProcessing.LineFilterExpression import vib_band_label
        return _np.asarray([vib_band_label(u, l) for u, l in zip(up, low)], dtype=object)

    def _vib_band_scalar(line, _mol_id):
        from iSLAT.Modules.DataProcessing.LineFilterExpression import vib_band_label
        return vib_band_label(getattr(line, "lev_up", ""), getattr(line, "lev_low", "")) or None

    def _molecule_array(cdata):
        wav = cdata.get("wavelength")
        n = 0 if wav is None else len(wav)
        return _np.full(n, cdata.get("name") or "unknown", dtype=object)

    # The shared property list, in the order every menu shows it.  Each entry
    # also says how the line-list filter evaluates it (filter_kind / field).
    entries = [
        # ---- Line-list columns ------------------------------------------
        PlotAxisEntry(
            key="wavelength",
            display_name="Wavelength (\u00b5m)",
            label=r"$\lambda$ ($\mu$m)",
            kind="continuous",
            resolve_array=_arr("wavelength"),
            resolve_scalar=_scalar("lam"),
            group="Line",
            filter_kind="range",
            filter_field="lam",
        ),
        PlotAxisEntry(
            key="eu",
            display_name="E_up (K)",
            label=r"$E_{u}$ (K)",
            kind="continuous",
            resolve_array=_arr("eu"),
            resolve_scalar=_scalar("e_up"),
            group="Line",
            filter_kind="range",
            filter_field="e_up",
        ),
        PlotAxisEntry(
            key="e_low",
            display_name="E_low (K)",
            label=r"$E_{low}$ (K)",
            kind="continuous",
            resolve_array=_arr("e_low"),
            resolve_scalar=_scalar("e_low"),
            group="Line",
            filter_kind="range",
            filter_field="e_low",
        ),
        PlotAxisEntry(
            key="a_stein",
            display_name="Einstein A (s\u207b\u00b9)",
            label=r"$A_{u}$ (s$^{-1}$)",
            kind="continuous",
            suggest_log=True,
            resolve_array=_arr("a_stein"),
            resolve_scalar=_scalar("a_stein"),
            group="Line",
            filter_kind="range",
            filter_field="a_stein",
        ),
        PlotAxisEntry(
            key="g_up",
            display_name="g_up",
            label=r"$g_{u}$",
            kind="continuous",
            resolve_array=_arr("g_up"),
            resolve_scalar=_scalar("g_up"),
            group="Line",
            filter_kind="range",
            filter_field="g_up",
        ),
        PlotAxisEntry(
            key="g_low",
            display_name="g_low",
            label=r"$g_{low}$",
            kind="continuous",
            resolve_array=_arr("g_low"),
            resolve_scalar=_scalar("g_low"),
            group="Line",
            filter_kind="range",
            filter_field="g_low",
        ),
        PlotAxisEntry(
            key="freq",
            display_name="Frequency (Hz)",
            label=r"$\nu$ (Hz)",
            kind="continuous",
            resolve_array=_freq_array,
            resolve_scalar=_freq_scalar,
            group="Line",
            filter_kind="range",
            filter_field="freq",
        ),
        # ---- Model quantities (follow the molecule's parameters) --------
        PlotAxisEntry(
            key="rd_yax",
            display_name="Pop. diagram Y [ln(4\u03c0F/h\u03bdAg)]",
            label=r"ln(4$\pi$F/(h$\nu$$A_{u}$$g_{u}$))",
            kind="continuous",
            resolve_array=_arr("rd_yax"),
            resolve_scalar=None,  # requires beam_s — not derivable from bare MoleculeLine
            group="Model",
            filter_kind="derived",
            filter_field="rd_yax",
        ),
        PlotAxisEntry(
            key="intens",
            display_name="Line intensity",
            label="Intensity",
            kind="continuous",
            suggest_log=True,
            resolve_array=_arr("intens"),
            resolve_scalar=None,
            group="Model",
            filter_kind="derived",
            filter_field="intens",
        ),
        PlotAxisEntry(
            key="tau",
            display_name="Optical depth (\u03c4)",
            label=r"Opacity ($\tau$)",
            kind="continuous",
            suggest_log=True,
            resolve_array=_arr("tau"),
            resolve_scalar=None,
            group="Model",
            filter_kind="derived",
            filter_field="tau",
        ),
        PlotAxisEntry(
            key="fwhm_instrumental_kms",
            display_name="Instrumental FWHM (km/s)",
            label="Instrumental FWHM (km/s)",
            kind="continuous",
            resolve_array=_arr("fwhm_instrumental_kms"),
            resolve_scalar=None,
            group="Model",
            filter_kind="derived",
            filter_field="fwhm_instrumental_kms",
        ),
        PlotAxisEntry(
            key="fwhm_convolved_kms",
            display_name="Convolved FWHM (km/s)",
            label="Convolved FWHM (km/s)",
            kind="continuous",
            resolve_array=_arr("fwhm_convolved_kms"),
            resolve_scalar=None,
            group="Model",
            filter_kind="derived",
            filter_field="fwhm_convolved_kms",
        ),
        # ---- Derived from the line list ---------------------------------
        PlotAxisEntry(
            key="velocity_sep",
            display_name="Velocity sep. (km/s)",
            label=r"$\Delta v$ to nearest line (km/s)",
            kind="continuous",
            suggest_log=True,
            resolve_array=_velocity_sep_array,
            resolve_scalar=None,  # needs the neighbouring lines
            group="Line",
            filter_kind="velocity_sep",
        ),
        # ---- Labels (categorical) ---------------------------------------
        PlotAxisEntry(
            key="lev_up",
            display_name="Upper level (lev_up)",
            label="Upper level",
            kind="categorical",
            resolve_array=_labels("lev_up"),
            resolve_scalar=lambda line, _mol_id: getattr(line, "lev_up", None),
            group="Labels",
            filter_kind="quantum_label",
            filter_state="upper",
        ),
        PlotAxisEntry(
            key="lev_low",
            display_name="Lower level (lev_low)",
            label="Lower level",
            kind="categorical",
            resolve_array=_labels("lev_low"),
            resolve_scalar=lambda line, _mol_id: getattr(line, "lev_low", None),
            group="Labels",
            filter_kind="quantum_label",
            filter_state="lower",
        ),
        PlotAxisEntry(
            key="vib_band",
            display_name="Vib. band",
            label="Vibrational band",
            kind="categorical",
            resolve_array=_vib_band_array,
            resolve_scalar=_vib_band_scalar,
            group="Labels",
            filter_kind="vib_band",
        ),
        PlotAxisEntry(
            key="molecule",
            display_name="Molecule (species)",
            label="Molecule",
            kind="categorical",
            resolve_array=_molecule_array,
            resolve_scalar=None,
            group="Labels",
            filter_kind="species",
        ),
    ]
    for entry in entries:
        PlotAxisRegistry.register(entry)

    # Convenience alias so consumers can do ``Intensity.AxisRegistry``
    Intensity.AxisRegistry = PlotAxisRegistry  # type: ignore[attr-defined]

_register_standard_entries()