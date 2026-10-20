from typing import Optional, Dict, Any, Literal, Tuple, Union, Callable, overload
import numpy as np
import warnings
import time
import threading

# Performance logging
from iSLAT.Modules.Debug.PerformanceLogger import perf_log, log_timing, PerformanceSection
from iSLAT.Modules.Debug.DebugConfig import debug_config

# ================================
# Spectral Resampling Functions (canonical home: DataProcessing.spectral_utils)
# ================================
# Deferred import to avoid circular dependency:
#   Molecule -> DataProcessing.__init__ -> Slabfit -> Molecule
_spectral_utils_cache = None

def _get_spectral_utils():
    """Lazy accessor for spectral_utils — avoids circular import at module load."""
    global _spectral_utils_cache
    if _spectral_utils_cache is None:
        from iSLAT.Modules.DataProcessing.spectral_utils import make_bins, spectres
        _spectral_utils_cache = (make_bins, spectres)
    return _spectral_utils_cache

# Lazy imports with thread safety
_spectrum_module = None
_intensity_module = None
_import_lock = threading.Lock()

def _get_spectrum_module():
    """Thread-safe lazy import of spectrum module"""
    global _spectrum_module
    if _spectrum_module is None:
        with _import_lock:
            if _spectrum_module is None:  # Double-check pattern
                from .Spectrum import Spectrum
                _spectrum_module = Spectrum
    return _spectrum_module

def _get_intensity_module():
    """Thread-safe lazy import of intensity module"""
    global _intensity_module
    if _intensity_module is None:
        with _import_lock:
            if _intensity_module is None:  # Double-check pattern
                from .Intensity import Intensity
                _intensity_module = Intensity
    return _intensity_module

import iSLAT.Constants as c
from .MoleculeLineList import MoleculeLineList
from ._mixins import CacheStatsMixin, WavelengthRangeMixin, ClassObservableMixin

class Molecule(CacheStatsMixin, WavelengthRangeMixin, ClassObservableMixin):
    """
    Optimized Molecule class with enhanced caching and performance improvements.
    """
    __slots__ = (
        'name', 'filepath', 'displaylabel', 'color', '_is_visible', '_is_bold', '_rv_shift',
        'user_save_data', 'hitran_data', 'initial_molecule_parameters',
        'lines', 'intensity', 'spectrum',
        '_temp', '_radius', '_n_mol', '_distance', '_fwhm', '_broad',
        '_keplerian_fwhm', '_instrumental_profile_key',
        '_temp_val', '_radius_val', '_n_mol_val', '_distance_val', '_fwhm_val', '_broad_val',
        '_keplerian_fwhm_val', '_instrumental_profile_key_val',
        '_lines_filepath', '_profile',
        't_kin', 'scale_exponent', 'scale_number', 'radius_init', 'n_mol_init',
        '_wavelength_range', '_model_pixel_res', '_model_line_width',
        '_intensity_cache', '_spectrum_cache', '_flux_cache',
        '_param_hash_cache', '_dirty_flags', '_cache_stats',
        '_molar_mass', '_thermal_broad',
        '_intensity_calculation_method',
        '_line_data_source', '_line_format',
    )
    
    _class_callbacks: list = []
    _shared_calculation_cache = {}
    _cache_lock = threading.Lock()
    
    INTENSITY_AFFECTING_PARAMS = {'temp', 'n_mol', 'broad', 'rv_shift', 'wavelength_range', 'intensity_calculation_method'}
    # model_pixel_res is spectrum-affecting because Spectrum is built with dlambda=model_pixel_res
    SPECTRUM_AFFECTING_PARAMS = {'radius', 'distance', 'fwhm', 'keplerian_fwhm', 'instrumental_profile_key', 'rv_shift', 'wavelength_range', 'model_pixel_res'}
    FLUX_AFFECTING_PARAMS = INTENSITY_AFFECTING_PARAMS | SPECTRUM_AFFECTING_PARAMS | {'model_pixel_res'}
    # Purely cosmetic parameters: changing them must never invalidate any computed cache.
    DISPLAY_ONLY_PARAMS = {'is_bold'}

    # Parameters transferred by Copy/Paste Parameters and Duplicate.
    COPY_PARAMS: Tuple[str, ...] = (
        "temp", "n_mol", "radius", "fwhm", "broad",
        "rv_shift", "keplerian_fwhm", "instrumental_profile_key",
    )
    
    # Backward-compatible aliases for the ClassObservableMixin API
    @classmethod
    def add_molecule_parameter_change_callback(cls, callback):
        """Add a callback function to be called when individual molecule parameters change."""
        cls.add_class_callback(callback)
    
    @classmethod
    def remove_molecule_parameter_change_callback(cls, callback):
        """Remove a callback function for molecule parameter changes."""
        cls.remove_class_callback(callback)
    
    @classmethod
    def _notify_molecule_parameter_change(cls, molecule_name, parameter_name, old_value, new_value):
        """Notify all callbacks that a molecule parameter has changed."""
        cls._notify_class_callbacks(molecule_name, parameter_name, old_value, new_value)
    
    def _notify_my_parameter_change(self, parameter_name, old_value, new_value):
        if old_value == new_value:
            return
        
        self._invalidate_caches_for_parameter(parameter_name)
        self.__class__._notify_molecule_parameter_change(self.name, parameter_name, old_value, new_value)
    
    def _invalidate_caches_for_parameter(self, parameter_name):
        if parameter_name in self.DISPLAY_ONLY_PARAMS:
            return
        if parameter_name in self.INTENSITY_AFFECTING_PARAMS:
            self._dirty_flags['intensity'] = True
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True
        elif parameter_name in self.SPECTRUM_AFFECTING_PARAMS:
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True
        else:
            self._dirty_flags['flux'] = True
        
        if parameter_name in self.FLUX_AFFECTING_PARAMS:
            self._flux_cache.clear()
    
    def __init__(self, **kwargs):
        self._initialize_caching_system()
        
        if 'hitran_data' in kwargs and kwargs['hitran_data'] is not None:
            #print("Generating new molecule from default parameters.")
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs and kwargs['user_save_data'] is not None:
            #print("Generating new molecule from user saved data.")
            self.user_save_data = kwargs['user_save_data']
            self.hitran_data = None
        else:
            self.user_save_data = None
            self.hitran_data = None

        # Format-neutral line data source (takes precedence over hitran_data
        # when present).  Accepted values: any file path string.
        self._line_data_source = kwargs.get('line_data_source', None)
        self._line_format = kwargs.get('line_format', None)  # "hitran", "csv", "saved", or None=auto

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', {})

        # Load parameters from appropriate source
        if self.user_save_data is not None:
            self._load_from_user_save_data(kwargs)
        else:
            self._load_from_kwargs(kwargs)

        self.lines = None
        #self._lines_filepath = absolute_data_files_path / self.filepath
        
        # Calculate derived parameters
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        
        # Set final parameter values
        self._temp = float(getattr(self, '_temp_val', None) or self.t_kin)
        self._radius = float(getattr(self, '_radius_val', None) or self.radius_init)
        self._n_mol = float(getattr(self, '_n_mol_val', None) or self.n_mol_init)
        self._distance = float(getattr(self, '_distance_val', None) or c.DEFAULT_DISTANCE)
        self._fwhm = float(getattr(self, '_fwhm_val', None) or c.DEFAULT_FWHM)
        self._keplerian_fwhm = float(getattr(self, '_keplerian_fwhm_val', None) or 0.0)
        self._instrumental_profile_key = str(getattr(self, '_instrumental_profile_key_val', None) or 'constant')
        self._broad = float(getattr(self, '_broad_val', 1.0) or c.INTRINSIC_LINE_WIDTH)
        self._rv_shift = float(getattr(self, '_rv_shift', None) if getattr(self, '_rv_shift', None) is not None else c.DEFAULT_MOLECULE_RV)

        self._intensity_calculation_method = str(kwargs.get('intensity_calculation_method', 'curve_growth'))
        self._wavelength_range = kwargs.get('wavelength_range', c.WAVELENGTH_RANGE)
        self._model_pixel_res = kwargs.get('model_pixel_res', c.MODEL_PIXEL_RESOLUTION)
        self._model_line_width = kwargs.get('model_line_width', c.MODEL_LINE_WIDTH)

        self.intensity = None
        self.spectrum = None
        
        self._calculate_initial_parameter_hashes()
    
    def _initialize_caching_system(self):
        self._intensity_cache = {'data': None, 'hash': None}
        self._spectrum_cache = {'data': None, 'hash': None}
        self._flux_cache = {}
        self._param_hash_cache = {}
        self._dirty_flags = {
            'intensity': True,
            'spectrum': True, 
            'flux': True
        }
        self._init_cache_stats()

    def change_line_list(self, filepath: str) -> None:
        """Assign a new line-list file and reset all derived state.

        This is the canonical way to swap the line list on a live
        :class:`Molecule` instance.  It resets every cache and object
        that depends on the old lines so that the next spectrum
        calculation automatically reloads from *filepath*.

        Parameters
        ----------
        filepath : str
            Absolute path to the new line-list file (.par or similar).
        """
        self.filepath = filepath
        self._line_format = None   # let the loader auto-detect format

        # Discard old line data and the Intensity object that references it
        self.lines = None
        self.intensity = None

        # Reset properties derived from the line list
        self._molar_mass = None
        if hasattr(self, '_thermal_broad'):
            self._thermal_broad = None

        # Invalidate all computation caches and reset dirty flags
        self._initialize_caching_system()

        # Recompute parameter hashes from the current (non-line-list) params
        self._calculate_initial_parameter_hashes()
    
    def _calculate_initial_parameter_hashes(self):
        self._param_hash_cache = {
            'intensity': self._compute_intensity_hash(),
            'spectrum': self._compute_spectrum_hash(),
            'full': self._compute_full_parameter_hash()
        }
    
    def _compute_intensity_hash(self):
        #wavelength_tuple = tuple(self._wavelength_range) if self._wavelength_range else ()
        # use INTENSITY_AFFECTING_PARAMS instead of hardcoding the parameters to ensure consistency
        return hash(tuple(getattr(self, param) for param in self.INTENSITY_AFFECTING_PARAMS))
    
    def _compute_spectrum_hash(self):
        wavelength_tuple = tuple(self._wavelength_range) if self._wavelength_range else ()
        return hash((
            self._radius,
            self._distance,
            self._fwhm,
            self._keplerian_fwhm,
            self._instrumental_profile_key,
            self._rv_shift,
            self._model_pixel_res,  # Spectrum grid spacing (dlambda) depends on this
            wavelength_tuple,
            self._compute_intensity_hash()  # Include intensity hash for dependencies
        ))

    def _compute_full_parameter_hash(self):
        return hash((self._compute_spectrum_hash(), self._model_pixel_res))

    def _load_from_user_save_data(self, kwargs: Dict[str, Any]):
        """Load parameters from user save data"""
        usd = self.user_save_data
        self.name = usd.get('Molecule Name', kwargs.get('name', 'Unknown Molecule'))
        self.filepath = usd.get('File Path', kwargs.get('filepath', None))
        self.displaylabel = usd.get('Molecule Label', self.name)
        self._temp_val = usd.get('Temp', kwargs.get('temp', None))
        self._radius_val = usd.get('Rad', kwargs.get('radius', None))
        self._n_mol_val = usd.get('N_Mol', kwargs.get('n_mol', None))
        self.color = usd.get('Color', kwargs.get('color', None))
        self._is_visible = usd.get('Vis', kwargs.get('is_visible', True))
        self._is_bold = self._to_bool(kwargs.get('is_bold', False))  # display-only; not persisted to save files

        # Get instance values from user save data or kwargs
        self._distance_val = usd.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = usd.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._keplerian_fwhm_val = usd.get('KeplerianFWHM', kwargs.get('keplerian_fwhm', 0.0))
        self._instrumental_profile_key_val = usd.get('InstrumentalProfile', kwargs.get('instrumental_profile_key', 'constant'))
        self._broad_val = usd.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self._rv_shift = usd.get('RV Shift', kwargs.get('rv_shift', c.DEFAULT_MOLECULE_RV))

        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)

    def _load_from_kwargs(self, kwargs: Dict[str, Any]):
        """Load parameters from kwargs"""
        self.name = kwargs.get('name', kwargs.get('displaylabel', kwargs.get('filepath', 'Unknown Molecule')))
        # Prefer line_data_source over hitran_data for the filepath
        self.filepath = kwargs.get('filepath',
                                   self._line_data_source or
                                   (self.hitran_data if hasattr(self, 'hitran_data') else None))
        self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
        self._temp_val = kwargs.get('temp', self.initial_molecule_parameters.get('t_kin', 300.0))
        self._radius_val = kwargs.get('radius', self.initial_molecule_parameters.get('radius_init', 1.0))
        self._n_mol_val = kwargs.get('n_mol', self.initial_molecule_parameters.get('n_mol', None))
        self.color = kwargs.get('color', None)
        self._is_visible = kwargs.get('is_visible', True)
        self._is_bold = self._to_bool(kwargs.get('is_bold', False))

        # Get instance values from kwargs or defaults
        self._distance_val = kwargs.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = kwargs.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._keplerian_fwhm_val = kwargs.get('KeplerianFWHM', kwargs.get('keplerian_fwhm', 0.0))
        self._instrumental_profile_key_val = kwargs.get('InstrumentalProfile', kwargs.get('instrumental_profile_key', 'constant'))
        self._broad_val = kwargs.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self._rv_shift = kwargs.get('rv_shift', kwargs.get('RV Shift', c.DEFAULT_MOLECULE_RV))

        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)
        
    def _ensure_lines_loaded(self):
        if self.lines is None:
            start_time = time.perf_counter()
            if self.filepath:
                debug_config.info('molecule_dict', f"Loading lines from filepath: {self.filepath}")
                self.lines = MoleculeLineList(
                    molecule_id=self.name, filename=self.filepath,
                    wavelength_range=self._wavelength_range,
                    format=getattr(self, '_line_format', None),
                )
            else:
                print("Creating empty line list")
                self.lines = MoleculeLineList(
                    molecule_id=self.name,
                    wavelength_range=self._wavelength_range
                )
            log_timing(f"Molecule._ensure_lines_loaded({self.name})", time.perf_counter() - start_time)
        elif self.lines.wavelength_range != self._wavelength_range:
            # Wavelength range changed — update the existing line list
            self.lines.wavelength_range = self._wavelength_range
    
    def _ensure_intensity_calculated(self):
        if self._dirty_flags['intensity'] or self._intensity_cache['data'] is None:
            self._calculate_intensity_with_caching()
    
    def _ensure_spectrum_calculated(self):
        self._ensure_intensity_calculated()
        if self._dirty_flags['spectrum'] or self._spectrum_cache['data'] is None:
            self._calculate_spectrum_with_caching()
    
    def _calculate_intensity_with_caching(self):
        section = PerformanceSection(f"Molecule._calculate_intensity({self.name})")
        section.start()
        
        current_hash = self._compute_intensity_hash()
        
        if (self._intensity_cache['hash'] == current_hash and 
            self._intensity_cache['data'] is not None):
            self._record_cache_hit()
            section.mark("cache_hit")
            section.end()
            return
        
        section.mark("ensure_lines")
        self._ensure_lines_loaded()
        
        section.mark("create_intensity_obj")
        if self.intensity is None:
            Intensity = _get_intensity_module()
            self.intensity = Intensity(self.lines, wavelength_range=self._wavelength_range)
        elif self.intensity.wavelength_range != self._wavelength_range:
            # Wavelength range changed — update the existing Intensity object
            self.intensity.wavelength_range = self._wavelength_range
        
        print(f"Calculating intensity for {self.name}: T={self._temp}K, N_mol={self._n_mol:.2e}, dv={self._broad}")
        
        try:
            method = self.intensity_calculation_method
        except AttributeError:
            method = "curve_growth"

        section.mark("calc_intensity")
        self.intensity.calc_intensity(
            t_kin=self._temp,
            n_mol=self._n_mol,
            dv=self._broad,
            method=method
        )
        section.mark("calc_complete")

        intensity_data = {
            'intensity_array': self.intensity._intensity.copy() if self.intensity._intensity is not None else None,
            'tau_array': self.intensity._tau.copy() if self.intensity._tau is not None else None
        }
        
        self._intensity_cache = {
            'data': intensity_data,
            'hash': current_hash
        }
        
        with self._cache_lock:
            if len(self._shared_calculation_cache) > 100:
                oldest_keys = list(self._shared_calculation_cache.keys())[:10]
                for key in oldest_keys:
                    del self._shared_calculation_cache[key]
        
        self._dirty_flags['intensity'] = False
        self._param_hash_cache['intensity'] = current_hash
        self._record_cache_miss()
        
        section.end()
        section.get_breakdown(print_output=True)
    
    def _calculate_spectrum_with_caching(self):
        current_hash = self._compute_spectrum_hash()
        
        if (self._spectrum_cache['hash'] == current_hash and 
            self._spectrum_cache['data'] is not None):
            self._record_cache_hit()
            return
            
        self._ensure_intensity_calculated()
        
        # Always recreate spectrum when parameters change
        Spectrum = _get_spectrum_module()
        mean_wavelength = (self.wavelength_range[0] + self.wavelength_range[1]) / 2.0

        # ------------------------------------------------------------------
        # Build the instrumental profile object, then delegate the quadrature
        # combination with Keplerian FWHM to Spectrum.build_R_func.
        # ------------------------------------------------------------------
        from iSLAT.Modules.DataProcessing.InstrumentalProfiles import (
            PROFILE_REGISTRY, ConstantProfile
        )
        profile_key = self._instrumental_profile_key or 'constant'
        profile_cls = PROFILE_REGISTRY.get(profile_key, ConstantProfile)
        profile = profile_cls(self._fwhm) if profile_key == 'constant' else profile_cls()
        self._profile = profile  # Store for potential future use

        kep_fwhm_kms = self._keplerian_fwhm
        use_constant_scalar = (profile_key == 'constant' and kep_fwhm_kms == 0.0)

        if use_constant_scalar:
            # Fast path: single scalar R — no per-line function overhead
            delta_lambda = mean_wavelength * (self._fwhm / c.SPEED_OF_LIGHT_KMS)
            spectral_resolution = mean_wavelength / delta_lambda if delta_lambda > 0 else self.model_line_width
            R_func = None
            rep_fwhm_kms = self._fwhm
        else:
            # General path: Spectrum owns the quadrature + fallback logic
            R_func = Spectrum.build_R_func(profile, kep_fwhm_kms, self._fwhm)

            # Representative scalar R at the mean wavelength (used to size the grid)
            R_at_mean = float(np.atleast_1d(np.asarray(R_func(np.array([mean_wavelength]))))[0])
            if not np.isfinite(R_at_mean) or R_at_mean <= 0:
                R_at_mean = mean_wavelength / (mean_wavelength * self._fwhm / c.SPEED_OF_LIGHT_KMS)
            spectral_resolution = R_at_mean
            rep_fwhm_kms = c.SPEED_OF_LIGHT_KMS / R_at_mean

        # Use consistent wavelength range for all molecules to ensure identical grids
        if hasattr(self, '_wavelength_range') and self._wavelength_range is not None:
            global_min, global_max = self._wavelength_range
            spectrum_lam_min = global_min
            spectrum_lam_max = global_max
        else:
            spectrum_lam_min = self.wavelength_range[0]
            spectrum_lam_max = self.wavelength_range[1]

        # Always compute the internal spectrum on a fine grid that properly
        # Nyquist-samples the narrowest expected line width. The user's
        # model_pixel_res is applied later via flux-conserving resampling (_spectres) in get_flux().
        #fine_dlambda = (mean_wavelength / c.SPEED_OF_LIGHT_KMS * rep_fwhm_kms) / c.PIXELS_PER_FWHM

        self.spectrum = Spectrum(
            lam_min=spectrum_lam_min,
            lam_max=spectrum_lam_max,
            dlambda=self.model_pixel_res,#fine_dlambda,
            R=spectral_resolution,
            R_func=R_func,
            distance=self._distance,
            wavelength_range=self._wavelength_range
        )
        
        area = np.pi * (self._radius) ** 2  # Area in AU^2 as expected by add_intensity
        self.spectrum.add_intensity(self.intensity, area)
        
        self._spectrum_cache = {
            'data': self.spectrum,
            'hash': current_hash
        }
        
        self._dirty_flags['spectrum'] = False
        self._param_hash_cache['spectrum'] = current_hash
        self._record_cache_miss()
    
    def calculate_intensity(self) -> None:
        """Trigger intensity calculation (lazy — skipped if cache is valid)."""
        self._ensure_intensity_calculated()
    
    def get_parameter_hash(self, cache_type='full'):
        """Get parameter hash for given cache type"""
        if cache_type in self._param_hash_cache:
            return self._param_hash_cache[cache_type]
        return self._compute_full_parameter_hash()

    def _clear_specific_cache(self, cache_type):
        """Clear specific cache and update stats"""
        if cache_type == 'flux':
            self._flux_cache.clear()
            self._dirty_flags['flux'] = True
        elif cache_type == 'intensity':
            self._intensity_cache = {'data': None, 'hash': None}
            self._dirty_flags['intensity'] = True
            self._dirty_flags['spectrum'] = True  # Spectrum depends on intensity
            self._dirty_flags['flux'] = True      # Flux depends on spectrum
        elif cache_type == 'spectrum':
            self._spectrum_cache = {'data': None, 'hash': None}
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True      # Flux depends on spectrum

    @overload
    def get_flux(
        self,
        wavelength_array: Optional[np.ndarray] = ...,
        *,
        return_wavelengths: Literal[True],
        interpolate_to_input: bool = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def get_flux(
        self,
        wavelength_array: Optional[np.ndarray] = ...,
        *,
        return_wavelengths: Literal[False] = ...,
        interpolate_to_input: bool = ...,
    ) -> np.ndarray: ...

    @overload
    def get_flux(
        self,
        wavelength_array: Optional[np.ndarray] = ...,
        return_wavelengths: bool = ...,
        interpolate_to_input: bool = ...,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...

    def get_flux(
        self,
        wavelength_array: Optional[np.ndarray] = None,
        return_wavelengths: bool = False,
        interpolate_to_input: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get flux data with RV shift properly applied.
        
        The RV shift works by:
        1. Calculate spectrum on the unshifted wavelength grid
        2. If RV shift != 0, interpolate the flux as if it came from a shifted source
           back to the unshifted wavelength grid
        3. This ensures all molecules return flux on the same wavelength grid
        
        Parameters
        ----------
        wavelength_array : np.ndarray, optional
            Input wavelength array for additional interpolation.
        return_wavelengths : bool, default False
            If True, return tuple of (wavelengths, flux).
        interpolate_to_input : bool, default False
            If True and wavelength_array is provided, interpolate to match input grid.
            If False, return spectrum's native grid (with RV shift applied).
            
        Returns
        -------
        np.ndarray
            Flux array when *return_wavelengths* is False.
        tuple of (np.ndarray, np.ndarray)
            ``(wavelengths, flux)`` when *return_wavelengths* is True.
        """
        if wavelength_array is not None:
            try:
                cache_key = hash(wavelength_array.tobytes()) if hasattr(wavelength_array, 'tobytes') else str(wavelength_array)
            except (TypeError, ValueError):
                cache_key = f"fallback_{id(wavelength_array)}"
        else:
            cache_key = "no_wavelength_input"
        
        # Add interpolation flag to cache key
        cache_key = f"{cache_key}_interp_{interpolate_to_input}"
        
        current_param_hash = self._compute_full_parameter_hash()
        cache_entry = self._flux_cache.get(cache_key)
        
        # Check both dirty flags and validate cache entry for cache validity
        if (not self._dirty_flags['flux'] and 
            self._validate_flux_cache_entry(cache_entry, current_param_hash)):
            self._record_cache_hit()
            if return_wavelengths:
                # Return copies to prevent accidental modification of cached data
                return cache_entry['wavelengths'].copy(), cache_entry['flux'].copy()
            return cache_entry['flux'].copy()
        
        # Ensure we have a valid spectrum
        self._ensure_spectrum_calculated()
        
        if self.spectrum is None:
            print(f"Warning: No spectrum available for molecule {self.name}")
            empty_array = np.array([])
            if return_wavelengths:
                return empty_array, empty_array
            return empty_array
        
        # Get the spectrum data - this is the native unshifted grid
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux_jy  # Use Jy units for consistency with observed data
        
        if lam_grid is None or flux_grid is None:
            print(f"Warning: Invalid spectrum data for molecule {self.name}")
            empty_array = np.array([])
            if return_wavelengths:
                return empty_array, empty_array
            return empty_array
        
        # Apply RV shift correction to get flux on the unshifted grid
        # Create the shifted wavelength grid where the flux "came from"
        if self._rv_shift != 0:
            # Only print debug info for significant RV shifts or when debugging is enabled
            if abs(self._rv_shift) > 0.1:  # Only log if RV shift > 0.1 km/s
                print(f"Applying RV shift of {self._rv_shift} km/s for molecule {self.name}")
            
            shifted_source_lam_grid = lam_grid + (lam_grid / c.SPEED_OF_LIGHT_KMS * self._rv_shift)

            # Interpolate the flux from the shifted source back to the unshifted grid
            # This simulates observing a shifted source and correcting it back to rest frame
            # Use flux-conserving spectral resampling for RV correction
            _, _spectres = _get_spectral_utils()
            rv_corrected_flux = _spectres(lam_grid, shifted_source_lam_grid, flux_grid, fill=0.0)
        else:
            rv_corrected_flux = flux_grid

        # Now decide on final output grid
        if interpolate_to_input and wavelength_array is not None:
            # Use flux-conserving spectral resampling to match the input wavelength array
            _, _spectres = _get_spectral_utils()
            interpolated_flux = _spectres(wavelength_array, lam_grid, rv_corrected_flux, fill=0.0)
            
            result_wavelengths = wavelength_array.copy()  # Copy to prevent modification
            result_flux = interpolated_flux
        else:
            # The internal spectrum is always computed on a fine Nyquist-sampled grid. 
            # When the user-requested model_pixel_res differs from the internal grid spacing we resample via _spectres (flux-conserving) so that changing pixel resolution never alters the total flux.
            native_dlambda = lam_grid[1] - lam_grid[0] if len(lam_grid) > 1 else self._model_pixel_res
            needs_resample = abs(self._model_pixel_res - native_dlambda) > 1e-12 * native_dlambda

            if needs_resample and self._model_pixel_res > native_dlambda:
                # Build the coarser output grid at the requested pixel resolution
                output_grid = np.arange(lam_grid[0], lam_grid[-1], self._model_pixel_res)
                if len(output_grid) < 2:
                    output_grid = lam_grid  # Fallback: pixel res too coarse for range
                _, _spectres = _get_spectral_utils()
                result_flux = _spectres(output_grid, lam_grid, rv_corrected_flux, fill=0.0)
                result_wavelengths = output_grid
            else:
                # Fine grid is already at or coarser than requested resolution
                result_wavelengths = lam_grid.copy()
                result_flux = rv_corrected_flux
        
        # Cache the result (store copies to prevent modification)
        self._flux_cache[cache_key] = {
            'wavelengths': result_wavelengths.copy(),
            'flux': result_flux.copy(),
            'param_hash': current_param_hash
        }
        
        flux_cache_limit = 12
        flux_oldest_amount = 4

        # Limit cache size
        if len(self._flux_cache) > flux_cache_limit:
            oldest_keys = list(self._flux_cache.keys())[:flux_oldest_amount]
            for key in oldest_keys:
                del self._flux_cache[key]
        
        # Mark flux as clean since we just calculated it
        self._dirty_flags['flux'] = False
        self._record_cache_miss()
        
        if return_wavelengths:
            return result_wavelengths, result_flux
        return result_flux

    # ------------------------------------------------------------------
    # Optical-depth retrieval (analogous to get_flux)
    # ------------------------------------------------------------------

    def get_tau(
        self,
        wavelength_array: Optional[np.ndarray] = None,
        return_wavelengths: bool = False,
        interpolate_to_input: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get the convolved optical-depth profile with RV shift applied.

        This mirrors :meth:`get_flux` but returns the dimensionless
        optical depth τ(λ) instead of flux.

        Parameters
        ----------
        wavelength_array : np.ndarray, optional
            Input wavelength array for additional interpolation.
        return_wavelengths : bool, default False
            If True, return ``(wavelengths, tau)``.
        interpolate_to_input : bool, default False
            If True and *wavelength_array* is provided, interpolate to
            match the input grid.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Optical-depth array, or ``(wavelengths, tau)`` tuple.
        """
        self._ensure_spectrum_calculated()

        if self.spectrum is None:
            empty = np.array([])
            return (empty, empty) if return_wavelengths else empty

        lam_grid = self.spectrum._lamgrid
        tau_grid = self.spectrum.tau_profile

        if lam_grid is None or tau_grid is None:
            empty = np.array([])
            return (empty, empty) if return_wavelengths else empty

        # Apply RV shift (same correction as get_flux)
        if self._rv_shift != 0:
            shifted_source_lam_grid = lam_grid + (
                lam_grid / c.SPEED_OF_LIGHT_KMS * self._rv_shift
            )
            _, _spectres = _get_spectral_utils()
            rv_corrected_tau = _spectres(
                lam_grid, shifted_source_lam_grid, tau_grid, fill=0.0,
            )
        else:
            rv_corrected_tau = tau_grid

        if interpolate_to_input and wavelength_array is not None:
            _, _spectres = _get_spectral_utils()
            result_tau = _spectres(
                wavelength_array, lam_grid, rv_corrected_tau, fill=0.0,
            )
            result_wavelengths = wavelength_array.copy()
        else:
            native_dlambda = (
                lam_grid[1] - lam_grid[0] if len(lam_grid) > 1
                else self._model_pixel_res
            )
            needs_resample = (
                abs(self._model_pixel_res - native_dlambda)
                > 1e-12 * native_dlambda
            )
            if needs_resample and self._model_pixel_res > native_dlambda:
                output_grid = np.arange(
                    lam_grid[0], lam_grid[-1], self._model_pixel_res,
                )
                if len(output_grid) < 2:
                    output_grid = lam_grid
                _, _spectres = _get_spectral_utils()
                result_tau = _spectres(
                    output_grid, lam_grid, rv_corrected_tau, fill=0.0,
                )
                result_wavelengths = output_grid
            else:
                result_wavelengths = lam_grid.copy()
                result_tau = rv_corrected_tau

        if return_wavelengths:
            return result_wavelengths, result_tau
        return result_tau
    
    # Define properties using a factory function approach
    def _make_property(attr_name, converter=float, special_setter=None):
        """Factory function to create properties"""
        private_attr = f'_{attr_name}'
        
        def getter(self):
            return getattr(self, private_attr)
        
        def setter(self, value):
            if converter:
                value = converter(value)
            
            old_value = getattr(self, private_attr)
            setattr(self, private_attr, value)
            
            if special_setter:
                special_setter(self, value)
            
            self._notify_my_parameter_change(attr_name, old_value, value)
        
        return property(getter, setter)

    # Standard numeric parameters
    temp = _make_property('temp', converter=float, special_setter=lambda self, value: setattr(self, 't_kin', value))
    radius = _make_property('radius', converter=float)
    distance = _make_property('distance', converter=float)
    # fwhm — the *instrumental* FWHM constant (km/s). Used by ConstantProfile and as the grid-sizing hint for non-constant profiles.
    fwhm = _make_property('fwhm', converter=float, special_setter=lambda self, value: setattr(self, 'spectrum', None))
    # Alias so GUI / notebooks can use either name
    instrumental_fwhm = fwhm

    # Keplerian FWHM broadening added in quadrature with the instrumental profile
    keplerian_fwhm = _make_property('keplerian_fwhm', converter=float, special_setter=lambda self, value: setattr(self, 'spectrum', None))

    model_pixel_res = _make_property('model_pixel_res', converter=float)
    instrumental_profile_key = _make_property('instrumental_profile_key', converter=str, special_setter=lambda self, value: setattr(self, 'spectrum', None))
    broad = _make_property('broad', converter=float)
    rv_shift = _make_property('rv_shift', converter=float)
    n_mol = _make_property('n_mol', converter=float)
    model_line_width = _make_property('model_line_width', converter=float)
    
    # wavelength_range property provided by WavelengthRangeMixin

    def _on_wavelength_range_changed(self, old, new):
        """Hook called by WavelengthRangeMixin when wavelength_range changes."""
        # Propagate to existing line list and intensity so they filter
        # to the new range without needing a full reload
        if self.lines is not None:
            self.lines.wavelength_range = new
        if self.intensity is not None:
            self.intensity.wavelength_range = new
        self._notify_my_parameter_change('wavelength_range', old, new)
    
    @property
    def is_visible(self) -> bool:
        if isinstance(self._is_visible, str):
            # Convert string representations to proper boolean
            return self._is_visible.lower() in ('true', '1', 'yes', 'on')
        return bool(self._is_visible)
    
    @is_visible.setter
    def is_visible(self, value):
        old_value = self._is_visible
        if isinstance(value, str):
            self._is_visible = value.lower() in ('true', '1', 'yes', 'on')
        else:
            self._is_visible = bool(value)
        self._notify_my_parameter_change('is_visible', old_value, self._is_visible)

    @staticmethod
    def _to_bool(value) -> bool:
        """Coerce a bool-ish value (including CSV strings like 'True'/'0') to bool."""
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'on')
        return bool(value)

    @property
    def is_bold(self) -> bool:
        """Whether this molecule's model line is drawn with a heavier line width."""
        return self._to_bool(getattr(self, '_is_bold', False))

    @is_bold.setter
    def is_bold(self, value):
        old_value = self.is_bold
        self._is_bold = self._to_bool(value)
        self._notify_my_parameter_change('is_bold', old_value, self._is_bold)

    @property
    def line_list(self) -> Optional[MoleculeLineList]:
        '''Access the MoleculeLineList, loading it if necessary.'''
        self._ensure_lines_loaded()
        return self.lines

    @property
    def molar_mass(self) -> float:
        '''Molar mass in g/mol'''
        if not hasattr(self, '_molar_mass') or self._molar_mass is None:
            self._ensure_lines_loaded()
            if self.lines is not None:
                self._molar_mass = self.lines.molar_mass
        return self._molar_mass

    @property
    def thermal_broad(self) -> float:
        try:
            return np.sqrt((c.BOLTZMANN_CONSTANT_JOULE * self._temp) / ((self.molar_mass / 1000) / c.AVAGADRO_NUMBER)) / 1000
        except Exception as e:
            print("Warning: Unable to compute thermal broadening due to missing or invalid molar mass.\n", f"Error details: {e}")
            return 0.0

    @property
    def intensity_calculation_method(self) -> str:
        try:
            if self._intensity_calculation_method is None:
                self._intensity_calculation_method = "curve_growth"
            return self._intensity_calculation_method
        except Exception as e:
            print("Warning: Unable to get intensity calculation method, defaulting to 'curve_growth'.")
            print(f"Error details: {e}")
            return "curve_growth"

    @intensity_calculation_method.setter
    def intensity_calculation_method(self, value: str):
        old_value = getattr(self, '_intensity_calculation_method', None)
        try:
            value = str(value)
        except Exception as e:
            print("Warning: Unable to set intensity calculation method, defaulting to 'curve_growth'.")
            print(f"Error details: {e}")
            value = "curve_growth"
        self._intensity_calculation_method = value
        self._notify_my_parameter_change('intensity_calculation_method', old_value, self._intensity_calculation_method)

    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], skip_notification: bool = False):
        if not parameter_dict:
            return
            
        # Make a copy to avoid modifying the input dict
        params_to_update = parameter_dict.copy()
        old_values = {}
        affected_params = set()
        
        # Batch cache invalidation flags
        needs_intensity_invalidation = False
        needs_spectrum_invalidation = False
        needs_flux_invalidation = False
        
        # Properties with special converters - default to float for all others
        special_converters = {
            'is_visible': lambda x: x.lower() in ('true', '1', 'yes', 'on') if isinstance(x, str) else bool(x),
            'is_bold': lambda x: x.lower() in ('true', '1', 'yes', 'on') if isinstance(x, str) else bool(x),
            'instrumental_profile_key': lambda x: str(x or 'constant'),
        }
        
        # Properties with special setters
        special_setters = {
            'temp': lambda self, value: setattr(self, 't_kin', value),
            'fwhm': lambda self, value: setattr(self, 'spectrum', None),
            'instrumental_fwhm': lambda self, value: setattr(self, 'spectrum', None),
            'keplerian_fwhm': lambda self, value: setattr(self, 'spectrum', None),
            'instrumental_profile_key': lambda self, value: setattr(self, 'spectrum', None),
        }
        
        # Batch process parameters with type conversion
        for param_name, value in params_to_update.items():
            # Apply type conversion - use special converter if available, otherwise float
            try:
                if param_name in special_converters:
                    value = special_converters[param_name](value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                continue  # Skip invalid values
            
            private_attr = f'_{param_name}'
            
            # Try private attribute first (most common case)
            if hasattr(self, private_attr):
                old_values[param_name] = getattr(self, private_attr)
                setattr(self, private_attr, value)
                affected_params.add(param_name)
                
                # Apply special setters
                if param_name in special_setters:
                    special_setters[param_name](self, value)
                
                # Determine cache invalidation needs
                if param_name in self.INTENSITY_AFFECTING_PARAMS:
                    needs_intensity_invalidation = True
                    needs_spectrum_invalidation = True
                    needs_flux_invalidation = True
                elif param_name in self.SPECTRUM_AFFECTING_PARAMS:
                    needs_spectrum_invalidation = True
                    needs_flux_invalidation = True
                else:
                    needs_flux_invalidation = True
                    
            # Fallback to public attribute
            elif hasattr(self, param_name):
                old_values[param_name] = getattr(self, param_name)
                setattr(self, param_name, value)
                affected_params.add(param_name)
                needs_flux_invalidation = True  # Conservative invalidation for unknown params
        
        # Batch cache invalidation - more efficient than individual calls
        if needs_intensity_invalidation:
            self._dirty_flags['intensity'] = True
        if needs_spectrum_invalidation:
            self._dirty_flags['spectrum'] = True
        if needs_flux_invalidation:
            self._dirty_flags['flux'] = True
            self._flux_cache.clear()
        
        # Batch notifications if required
        if not skip_notification and affected_params:
            params_to_notify = affected_params            
            for param_name in params_to_notify:
                old_value = old_values.get(param_name)
                # Fast attribute access
                new_value = getattr(self, f'_{param_name}', None) or getattr(self, param_name, None)
                if old_value != new_value:
                    self._notify_my_parameter_change(param_name, old_value, new_value)
    
    def copy_parameters(self) -> Dict[str, Any]:
        """Return a snapshot dict of this molecule's ``COPY_PARAMS`` values."""
        return {p: getattr(self, p) for p in self.COPY_PARAMS if hasattr(self, p)}

    def force_recalculate(self):
        self.clear_all_caches()
        self._ensure_intensity_calculated()
        self._ensure_spectrum_calculated()
    
    # get_cache_stats() provided by CacheStatsMixin
    
    def clear_all_caches(self):
        self._intensity_cache = {'data': None, 'hash': None}
        self._spectrum_cache = {'data': None, 'hash': None}
        self._flux_cache.clear()
        self._dirty_flags = {'intensity': True, 'spectrum': True, 'flux': True}
        self._record_cache_invalidation()
    
    def is_cache_valid(self, cache_type='full'):
        if cache_type == 'intensity':
            return not self._dirty_flags['intensity'] and self._intensity_cache['data'] is not None
        elif cache_type == 'spectrum':
            return not self._dirty_flags['spectrum'] and self._spectrum_cache['data'] is not None
        elif cache_type == 'flux':
            return not self._dirty_flags['flux'] and len(self._flux_cache) > 0
        else:
            return (self.is_cache_valid('intensity') and 
                   self.is_cache_valid('spectrum') and 
                   self.is_cache_valid('flux'))
    
    def _validate_flux_cache_entry(self, cache_entry, expected_param_hash):
        """Validate that a flux cache entry is still valid"""
        if cache_entry is None:
            return False
        
        # Check parameter hash
        if cache_entry.get('param_hash') != expected_param_hash:
            return False
        
        # Validate that arrays exist and are not empty
        wavelengths = cache_entry.get('wavelengths')
        flux = cache_entry.get('flux')
        
        if (wavelengths is None or flux is None or 
            len(wavelengths) == 0 or len(flux) == 0):
            return False
        
        # Check that arrays have the same length
        if len(wavelengths) != len(flux):
            return False
        
        return True

    def __str__(self):
        return f"Molecule(name={self.name}, temp={self._temp}, radius={self._radius}, n_mol={self._n_mol})"