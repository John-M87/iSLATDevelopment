"""
Slab fitting module for iSLAT.

Accepts a variety of source objects:

* ``Molecule``         — most common; temperature, column density, and radius
                         are fittable by default; spectral config read from
                         molecule attributes.
* ``Intensity``        — intensity parameters fittable; spectral config from
                         ``**kwargs``.
* ``MoleculeLineList`` — full control; all parameters from defaults/``**kwargs``.
* ``Spectrum``         — chi-squared evaluation only; ``fit()`` raises
                         ``ValueError``.

New API example::

    slab = SlabModel(output_folder, source=mol, input_file="obs.csv")
    result = slab.fit()
    slab.update_source_parameters(result)
    slab.save_results(result)

Fit extra parameters::

    slab = SlabModel(output_folder, source=mol, input_file="obs.csv")
    slab.add_fit_parameter('fwhm', initial_value=130, bounds=(10, 500))
    result = slab.fit()

Construct from a specific type::

    slab = SlabModel.from_molecule(mol, chi2_input_file="obs.csv")
    slab = SlabModel.from_intensity(intensity, chi2_input_file="obs.csv",
                                    output_folder=out_dir)

Legacy API (deprecated but still functional)::

    slab = SlabModel(output_folder, mol_object=mol, input_file="obs.csv")
    fitted = slab.fit_parameters(start_t=300, start_n_mol=5e16, start_r=1.5)
    slab.update_molecule_parameters(fitted)
    slab.save_results(fitted)
"""

from __future__ import annotations

import math
import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from iSLAT.Modules.DataProcessing import Chi2Spectrum
from iSLAT.Modules.DataTypes.Intensity import Intensity
from iSLAT.Modules.DataTypes.Spectrum import Spectrum
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
import iSLAT.Constants as c
from iSLAT.Modules.FileHandling import line_saves_file_path

def minimize(*args: Any, **kwargs: Any) -> Any:
    """``scipy.optimize.minimize``, imported on first use.

    scipy.optimize takes ~0.8 s to import and this module is loaded at GUI
    startup (through FittingEngine), long before any slab fit runs.
    """
    from scipy.optimize import minimize as _minimize
    return _minimize(*args, **kwargs)

# ---------------------------------------------------------------------------
# FitParameter
# ---------------------------------------------------------------------------
@dataclass
class FitParameter:
    """Describes a single fittable parameter.

    Parameters
    ----------
    name : str
        Name matching a source-object attribute (e.g. ``'temp'``, ``'n_mol'``).
    initial_value : float
        Starting value in physical units.
    bounds : tuple[float, float], optional
        ``(lower, upper)`` bounds in physical units.  ``(-inf, inf)`` means
        unconstrained.
    log_scale : bool, optional
        If ``True`` the optimizer works in log₁₀ space for this parameter
        (recommended for column density).
    """
    name: str
    initial_value: float
    bounds: Tuple[float, float] = (-math.inf, math.inf)
    log_scale: bool = False

    @property
    def internal_initial(self) -> float:
        """Initial value in the optimizer's internal space."""
        return math.log10(self.initial_value) if self.log_scale else self.initial_value

    def to_physical(self, internal_value: float) -> float:
        """Convert from internal optimizer space to physical value."""
        return 10.0 ** internal_value if self.log_scale else internal_value

    def internal_bounds(self) -> Tuple[float, float]:
        """Bounds in the optimizer's internal (log or linear) space."""
        lo, hi = self.bounds
        if self.log_scale:
            lo = math.log10(lo) if math.isfinite(lo) and lo > 0 else -math.inf
            hi = math.log10(hi) if math.isfinite(hi) and hi > 0 else math.inf
        return (lo, hi)

# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """Results returned by :meth:`SlabModel.fit`.

    Attributes
    ----------
    parameters : dict[str, float]
        Fitted physical values keyed by ``FitParameter.name``.
    chi2_final : float
        Final chi-squared value.
    iterations : int
        Number of optimizer iterations.
    function_calls : int
        Number of objective-function evaluations.
    convergence_flag : int
        ``0`` = converged, ``1`` = max iterations reached.
    message : str
        Optimizer status message.
    """
    parameters: Dict[str, float]
    chi2_final: float
    iterations: int
    function_calls: int
    convergence_flag: int
    message: str = ""

# ---------------------------------------------------------------------------
# Default fit-parameter set
# ---------------------------------------------------------------------------
_DEFAULT_FIT_PARAMETERS: List[FitParameter] = [
    FitParameter('temp',   500.0, (10.0,  5000.0), log_scale=False),
    FitParameter('n_mol',  1e17,  (1e12,  1e25),   log_scale=True),
    FitParameter('radius', 1.0,   (0.01, 1000.0), log_scale=False),
    FitParameter('broad', 1.0,   (0.01, 20.0), log_scale=False),
    FitParameter('keplerian_fwhm', 100.0,   (20.0, 200.0), log_scale=False),
]

# Parameters that feed into Intensity.calc_intensity
_INTENSITY_PARAMS = frozenset({'temp', 'n_mol', 'broad'})
# Parameters that only affect add_intensity area scaling
_AREA_PARAMS = frozenset({'radius'})
# Parameters that require rebuilding the Spectrum object when they are fit
_SPECTRUM_REBUILD_PARAMS = frozenset({
    'distance', 'fwhm', 'keplerian_fwhm', 'model_line_width',
    'min_wavelength', 'max_wavelength', 'model_pixel_res',
})

# ---------------------------------------------------------------------------
# _SlabPipeline — private implementation detail
# ---------------------------------------------------------------------------
def _copy_attr(obj, d: dict, name: str) -> None:
    """Copy attribute *name* from *obj* into dict *d* if present and not None."""
    v = getattr(obj, name, None)
    if v is not None:
        d[name] = v

class _SlabPipeline:
    """Normalises heterogeneous source inputs into a common Intensity+Spectrum pair.

    This object owns its own ``Intensity`` and ``Spectrum`` instances so that
    the source object's display state is never mutated during fitting.
    """
    def __init__(
        self,
        source: Union[Molecule, Intensity, MoleculeLineList, Spectrum],
    ) -> None:
        self.source = source
        self._intensity: Optional[Intensity] = None
        self._spectrum: Optional[Spectrum] = None

        if isinstance(source, Molecule):
            self.input_type = 'molecule'
            self._line_list: Optional[MoleculeLineList] = source.lines
        elif isinstance(source, Intensity):
            self.input_type = 'intensity'
            self._line_list = source._molecule  # MoleculeLineList via private attr
        elif isinstance(source, MoleculeLineList):
            self.input_type = 'line_list'
            self._line_list = source
        elif isinstance(source, Spectrum):
            self.input_type = 'spectrum'
            self._line_list = None
        else:
            raise TypeError(
                f"Unsupported source type: {type(source).__name__}. "
                "Expected Molecule, Intensity, MoleculeLineList, or Spectrum."
            )

        self.source_defaults: Dict[str, float] = self._extract_source_defaults(source)

    # ------------------------------------------------------------------
    # Source name helper
    # ------------------------------------------------------------------
    @property
    def source_name(self) -> str:
        """Human-readable name for the source, used in output filenames."""
        if self.input_type == 'molecule':
            return getattr(self.source, 'name', 'unknown')
        if self.input_type in ('intensity', 'line_list') and self._line_list is not None:
            mol_id = getattr(self._line_list, 'molecule_id', None)
            return str(mol_id) if mol_id else 'unknown'
        return 'spectrum'

    # ------------------------------------------------------------------
    # Default-value extraction from source
    # ------------------------------------------------------------------
    def _extract_source_defaults(self, source) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            'temp':             500.0,
            'n_mol':            1e17,
            'radius':           1.0,
            'distance':         c.DEFAULT_DISTANCE,
            'fwhm':             c.DEFAULT_FWHM,
            'keplerian_fwhm':   0.0,
            'broad':            c.INTRINSIC_LINE_WIDTH,
            'profile': None,
            'min_wavelength':   c.WAVELENGTH_RANGE[0],
            'max_wavelength':   c.WAVELENGTH_RANGE[1],
            'model_pixel_res':  float(c.MODEL_PIXEL_RESOLUTION),
            'model_line_width': float(c.MODEL_LINE_WIDTH),
        }
        if isinstance(source, Molecule):
            for attr in ('temp', 'n_mol', 'radius', 'distance', 'fwhm', 'keplerian_fwhm',
                         'broad', 'model_pixel_res', 'model_line_width', 'profile'):
                _copy_attr(source, defaults, attr)
            wr = getattr(source, 'wavelength_range', None)
            if wr:
                defaults['min_wavelength'] = wr[0]
                defaults['max_wavelength'] = wr[1]
        elif isinstance(source, Intensity):
            v = getattr(source, 't_kin', None)
            if v is not None:
                defaults['temp'] = v
            for attr in ('n_mol',):
                _copy_attr(source, defaults, attr)
            v = getattr(source, 'dv', None)
            if v is not None:
                defaults['broad'] = v
        return defaults

    # ------------------------------------------------------------------
    # Lazy object builders
    # ------------------------------------------------------------------
    def _build_intensity(self) -> Intensity:
        if self._line_list is None:
            raise ValueError("Cannot build Intensity: no line list available.")
        return Intensity(self._line_list)

    def _build_spectrum(self, config: Dict[str, float]) -> Spectrum:
        return Spectrum(
            lam_min=config['min_wavelength'],
            lam_max=config['max_wavelength'],
            dlambda=config['model_pixel_res'],
            R=config['model_line_width'],
            distance=config['distance'],
            keplerian_fwhm=config.get('keplerian_fwhm', 0.0),
            R_func=config.get('profile', None),
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        param_values: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Spectrum:
        """Apply *param_values* and return the populated ``Spectrum``.

        For ``'spectrum'`` input-type the source is returned unchanged.
        """
        if self.input_type == 'spectrum':
            return self.source  # type: ignore[return-value]

        # --- Intensity (lazy build, reused across iterations) ---
        if self._intensity is None:
            self._intensity = self._build_intensity()

        t_kin = param_values.get('temp',  config['temp'])
        n_mol = param_values.get('n_mol', config['n_mol'])
        dv    = param_values.get('broad', config['broad'])
        self._intensity.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=dv)

        # --- Spectrum: rebuild only when grid/distance params are being fitted ---
        needs_rebuild = self._spectrum is None or bool(
            set(param_values) & _SPECTRUM_REBUILD_PARAMS
        )
        if needs_rebuild:
            merged = dict(config)
            for k in _SPECTRUM_REBUILD_PARAMS:
                if k in param_values:
                    merged[k] = param_values[k]
            self._spectrum = self._build_spectrum(merged)
        else:
            # Clear previous iteration's components, preserve kernel cache
            assert self._spectrum is not None
            self._spectrum.reset()

        assert self._spectrum is not None
        radius = param_values.get('radius', config['radius'])
        self._spectrum.add_intensity(self._intensity, radius ** 2 * np.pi)  # type: ignore[arg-type]
        return self._spectrum

# ---------------------------------------------------------------------------
# SlabModel
# ---------------------------------------------------------------------------
class SlabModel:
    """Slab model fitting class — see module docstring for usage examples."""
    def __init__(
        self,
        output_folder=None,
        source: Union[Molecule, Intensity, MoleculeLineList, Spectrum, None] = None,
        data_field=None,
        fit_parameters: Optional[List[FitParameter]] = None,
        *,
        mol_object=None,   # deprecated alias for source
        **kwargs,
    ):
        """
        Initialize the slab fitting system.

        Parameters
        ----------
        output_folder : str or Path, optional
            Directory for saving results and locating chi-squared input data.
            Falls back to the iSLAT default save folder when ``None``.
        source : Molecule | Intensity | MoleculeLineList | Spectrum, optional
            The spectral source to fit.  Also accepted as the deprecated
            ``mol_object`` keyword argument.
        data_field : object, optional
            GUI data field for status messages (``insert_text`` interface).
        fit_parameters : list[FitParameter], optional
            Parameters to optimise.  Defaults to ``[temp, n_mol, radius]``.
        mol_object : deprecated
            Use *source* instead.
        **kwargs
            Spectral-configuration overrides and chi-squared file options:
            ``distance``, ``fwhm``, ``broad`` / ``intrinsic_line_width``,
            ``min_wavelength``, ``max_wavelength``, ``model_pixel_res``,
            ``model_line_width``, ``input_filename``, ``input_file``,
            ``flux_col_name``, ``error_col_name``.
        """
        self.data_field = data_field

        # Resolve deprecated mol_object alias
        if source is None and mol_object is not None:
            warnings.warn(
                "The 'mol_object' parameter is deprecated. Use 'source' instead.",
                DeprecationWarning, stacklevel=2,
            )
            source = mol_object
        # Kept for back-compat attribute access
        self.mol_object = source

        # --- output folder ---
        if output_folder:
            self.output_folder = str(output_folder)
        else:
            self.output_folder = str(line_saves_file_path)
            if self.data_field:
                self.data_field.insert_text(
                    f"Using default output folder: {line_saves_file_path}",
                    clear_after=False,
                )

        # --- chi-squared input file ---
        self.input_filename = kwargs.get('input_filename', 'fit_data.csv')
        self.input_file = kwargs.get(
            'input_file', os.path.join(self.output_folder, self.input_filename)
        )

        # --- pipeline (None source is allowed for chi-squared-only usage) ---
        self._pipeline: Optional[_SlabPipeline] = None
        if source is not None:
            self._pipeline = _SlabPipeline(source)

        # --- spectral config: source defaults -> kwargs overrides ---
        src_defs = self._pipeline.source_defaults if self._pipeline else {}
        self._config: Dict[str, Any] = {
            'temp':             src_defs.get('temp',             500.0),
            'n_mol':            src_defs.get('n_mol',            1e17),
            'radius':           src_defs.get('radius',           1.0),
            'distance':         kwargs.get('distance',
                                src_defs.get('distance',         c.DEFAULT_DISTANCE)),
            'fwhm':             kwargs.get('fwhm',
                                src_defs.get('fwhm',             c.DEFAULT_FWHM)),
            'keplerian_fwhm':   kwargs.get('keplerian_fwhm',
                                src_defs.get('keplerian_fwhm',   0.0)),
            'broad':            kwargs.get('broad',
                                kwargs.get('intrinsic_line_width',
                                src_defs.get('broad',            c.INTRINSIC_LINE_WIDTH))),
            'profile':          kwargs.get('profile',
                                src_defs.get('profile', None)),
            'min_wavelength':   kwargs.get('min_wavelength',
                                src_defs.get('min_wavelength',   c.WAVELENGTH_RANGE[0])),
            'max_wavelength':   kwargs.get('max_wavelength',
                                src_defs.get('max_wavelength',   c.WAVELENGTH_RANGE[1])),
            'model_pixel_res':  kwargs.get('model_pixel_res',
                                src_defs.get('model_pixel_res',  c.MODEL_PIXEL_RESOLUTION)),
            'model_line_width': kwargs.get('model_line_width',
                                src_defs.get('model_line_width', c.MODEL_LINE_WIDTH)),
            'flux_col_name':    kwargs.get('flux_col_name',      'Flux_islat'),
            'error_col_name':   kwargs.get('error_col_name',     'Err_data'),
        }

        # --- fit parameter list (seeded from config defaults) ---
        if fit_parameters is not None:
            self._fit_params: List[FitParameter] = list(fit_parameters)
        else:
            self._fit_params = self._make_default_fit_params()

        # --- chi-squared evaluator ---
        self.chi2_evaluator = Chi2Spectrum()
        if os.path.exists(self.input_file):
            self.chi2_evaluator.load_file(
                self.input_file,
                flux_col_name=self._config['flux_col_name'],
                error_col_name=self._config['error_col_name'],
            )
        elif self.data_field:
            self.data_field.insert_text(
                f"Warning: Input file '{self.input_file}' not found. "
                "Chi-squared evaluation may fail.",
                clear_after=False,
            )

    # ------------------------------------------------------------------
    # Default fit-parameter construction
    # ------------------------------------------------------------------
    def _make_default_fit_params(self) -> List[FitParameter]:
        """Return the default [temp, n_mol, radius] list seeded from config."""
        #return _DEFAULT_FIT_PARAMETERS
        return [
            FitParameter('temp',   self._config['temp'],   (10.0,  5000.0), log_scale=False),
            FitParameter('n_mol',  self._config['n_mol'],  (1e12,  1e35),   log_scale=True),
            FitParameter('radius', self._config['radius'], (-2.5, 8.0), log_scale=True),
            #FitParameter('broad', self._config['broad'],   (0.01, 20.0), log_scale=False),
            #FitParameter('keplerian_fwhm', self._config['keplerian_fwhm'],   (0.0, 170.0), log_scale=False),
        ]

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_molecule(
        cls,
        mol: Molecule,
        chi2_input_file: str,
        output_folder=None,
        **kwargs,
    ) -> 'SlabModel':
        """Construct from a :class:`~iSLAT.Modules.DataTypes.Molecule.Molecule`."""
        return cls(
            output_folder=output_folder or str(line_saves_file_path),
            source=mol,
            input_file=chi2_input_file,
            **kwargs,
        )

    @classmethod
    def from_intensity(
        cls,
        intensity: Intensity,
        chi2_input_file: str,
        output_folder=None,
        **kwargs,
    ) -> 'SlabModel':
        """Construct from an :class:`~iSLAT.Modules.DataTypes.Intensity` object."""
        return cls(
            output_folder=output_folder or str(line_saves_file_path),
            source=intensity,
            input_file=chi2_input_file,
            **kwargs,
        )

    @classmethod
    def from_line_list(
        cls,
        line_list: MoleculeLineList,
        chi2_input_file: str,
        output_folder=None,
        **kwargs,
    ) -> 'SlabModel':
        """Construct from a :class:`~iSLAT.Modules.DataTypes.MoleculeLineList`."""
        return cls(
            output_folder=output_folder or str(line_saves_file_path),
            source=line_list,
            input_file=chi2_input_file,
            **kwargs,
        )

    @classmethod
    def from_spectrum(
        cls,
        spectrum: Spectrum,
        chi2_input_file: str,
        output_folder=None,
        **kwargs,
    ) -> 'SlabModel':
        """Construct from a pre-built :class:`~iSLAT.Modules.DataTypes.Spectrum`.

        Only chi-squared evaluation is possible; calling :meth:`fit` raises
        :exc:`ValueError`.
        """
        return cls(
            output_folder=output_folder or str(line_saves_file_path),
            source=spectrum,
            input_file=chi2_input_file,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Fluent fit-parameter API
    # ------------------------------------------------------------------
    def add_fit_parameter(
        self,
        name: str,
        initial_value: float,
        bounds: Tuple[float, float] = (-math.inf, math.inf),
        log_scale: bool = False,
    ) -> 'SlabModel':
        """Append a new :class:`FitParameter`.  Returns *self* for chaining."""
        self._fit_params.append(FitParameter(name, initial_value, bounds, log_scale))
        return self

    def set_fit_parameters(self, params: List[FitParameter]) -> 'SlabModel':
        """Replace the current parameter list.  Returns *self* for chaining."""
        self._fit_params = list(params)
        return self

    def reset_fit_parameters(self) -> 'SlabModel':
        """Restore the default ``[temp, n_mol, radius]`` list.  Returns *self*."""
        self._fit_params = self._make_default_fit_params()
        return self

    def remove_fit_parameter(self, name: str) -> 'SlabModel':
        """Remove a parameter by name.  Returns *self* for chaining."""
        self._fit_params = [p for p in self._fit_params if p.name != name]
        return self

    # ------------------------------------------------------------------
    # Core chi-squared evaluation
    # ------------------------------------------------------------------
    def evaluate_model(self, **param_values: Any) -> float:
        """Evaluate chi-squared for the given *physical* parameter values.

        Any parameters not supplied fall back to the spectral config defaults
        (source attributes / constructor kwargs).

        Parameters
        ----------
        **param_values
            Physical values for any subset of the fit parameters, e.g.
            ``evaluate_model(temp=500, n_mol=1e17, radius=1.0)``.

        Returns
        -------
        float
            Total chi-squared.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "No source provided. Set 'source' before calling evaluate_model()."
            )
        spectrum = self._pipeline.evaluate(param_values, self._config)
        self.chi2_evaluator.evaluate_spectrum(spectrum)
        chi2_total = self.chi2_evaluator.chi2_total
        parts = ", ".join(f"{k}={v:.3g}" for k, v in param_values.items())
        print(f"[SlabFit] {parts} → χ²={chi2_total:.3e}")
        return chi2_total

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        initial_overrides: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """Fit the slab model parameters using ``scipy.optimize.minimize``.

        Automatically selects ``Nelder-Mead`` when all bounds are infinite, or
        ``L-BFGS-B`` when any bound is finite.

        Parameters
        ----------
        initial_overrides : dict, optional
            Override initial values for specific fit parameters by name,
            e.g. ``{'temp': 300, 'radius': 2.0}``.

        Returns
        -------
        FitResult
            Fitted parameter values and fit statistics.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "No source provided. Set 'source' before calling fit()."
            )
        if self._pipeline.input_type == 'spectrum':
            raise ValueError(
                "Cannot fit a Spectrum source directly. Provide a Molecule, "
                "Intensity, or MoleculeLineList instead."
            )
        if not self._fit_params:
            raise ValueError(
                "No fit parameters defined. Use add_fit_parameter() first."
            )

        if self.data_field is not None:
            self.data_field.insert_text("Fitting slab model...", clear_after=False)

        overrides = initial_overrides or {}

        # Build initial guess in internal (log or linear) space
        x0 = []
        for p in self._fit_params:
            init_phys = overrides.get(p.name, p.initial_value)
            x0.append(math.log10(init_phys) if p.log_scale else init_phys)

        # Determine whether to use bounded method
        scipy_bounds = [p.internal_bounds() for p in self._fit_params]
        has_finite_bounds = any(
            math.isfinite(lo) or math.isfinite(hi)
            for lo, hi in scipy_bounds
        )
        method = 'L-BFGS-B' if has_finite_bounds else 'Nelder-Mead'
        bounds_arg = scipy_bounds if has_finite_bounds else None

        print(f"Starting slab fit using {method} method with initial parameters:")
        for p in self._fit_params:
            init_phys = overrides.get(p.name, p.initial_value)
            print(f"  {p.name} = {init_phys:.4g}  bounds={p.bounds}  log_scale={p.log_scale}")

        def _objective(x_internal):
            phys = {
                p.name: p.to_physical(xi)
                for p, xi in zip(self._fit_params, x_internal)
            }
            return self.evaluate_model(**phys)

        opt_kwargs: dict = {'maxiter': 10000}
        if method == 'Nelder-Mead':
            opt_kwargs.update({'xatol': 1e-6, 'fatol': 1e-8})
        else:
            opt_kwargs['ftol'] = 1e-8

        opt = minimize(
            _objective, x0,
            method=method,
            bounds=bounds_arg,
            options=opt_kwargs,
        )

        fitted_phys = {
            p.name: p.to_physical(xi)
            for p, xi in zip(self._fit_params, opt.x)
        }

        result = FitResult(
            parameters=fitted_phys,
            chi2_final=float(opt.fun),
            iterations=int(getattr(opt, 'nit', 0)),
            function_calls=int(getattr(opt, 'nfev', 0)),
            convergence_flag=0 if opt.success else 1,
            message=getattr(opt, 'message', ''),
        )

        print("\nSlab fit complete:")
        for name, val in result.parameters.items():
            print(f"  {name} = {val:.4g}")
        print(f"  χ² = {result.chi2_final:.3e}  ({result.function_calls} evals)")

        if self.data_field is not None:
            self.data_field.insert_text(
                f"Fitting complete. χ² = {result.chi2_final:.3e}", clear_after=False
            )
        return result

    # ------------------------------------------------------------------
    # Post-fit: update source parameters
    # ------------------------------------------------------------------
    def update_source_parameters(self, result: FitResult) -> None:
        """Apply fitted parameters back to the source object.

        For ``Molecule`` sources, uses ``bulk_update_parameters`` to invalidate caches in one shot.
        For ``Intensity`` sources, sets the relevant attributes.
        For ``Spectrum`` or ``MoleculeLineList`` sources, emits a warning (no persistent state to update).

        Parameters
        ----------
        result : FitResult
            Returned by :meth:`fit`.
        """
        if self._pipeline is None:
            return
        src = self._pipeline.source
        input_type = self._pipeline.input_type

        if input_type == 'molecule':
            src.bulk_update_parameters(result.parameters)  # type: ignore[union-attr]
            print(f"Updated molecule '{self._pipeline.source_name}' with fitted parameters.")

        elif input_type == 'intensity':
            _INTENSITY_ATTR_MAP = {
                'temp':  '_t_kin',
                'n_mol': '_n_mol',
                'broad': '_dv',
            }
            for param_name, attr in _INTENSITY_ATTR_MAP.items():
                if param_name in result.parameters and hasattr(src, attr):
                    setattr(src, attr, result.parameters[param_name])
            print(f"Updated Intensity '{self._pipeline.source_name}' with fitted parameters.")

        elif input_type == 'spectrum':
            warnings.warn(
                "update_source_parameters() has no effect for Spectrum inputs.",
                UserWarning, stacklevel=2,
            )
        else:  # line_list
            warnings.warn(
                "update_source_parameters() has no effect for MoleculeLineList inputs "
                "(no persistent parameter state).",
                UserWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Deprecated aliases — keep for backward compatibility
    # ------------------------------------------------------------------
    def fit_parameters(
        self,
        start_t=None,
        start_n_mol=None,
        start_r=None,
    ) -> FitResult:
        """Deprecated — use :meth:`fit` instead."""
        warnings.warn(
            "fit_parameters() is deprecated. Use fit() instead.",
            DeprecationWarning, stacklevel=2,
        )
        overrides: Dict[str, Any] = {}
        if start_t is not None:
            overrides['temp'] = start_t
        if start_n_mol is not None:
            overrides['n_mol'] = start_n_mol
        if start_r is not None:
            overrides['radius'] = start_r
        return self.fit(initial_overrides=overrides or None)

    def update_molecule_parameters(self, fitted_params) -> None:
        """Deprecated — use :meth:`update_source_parameters` instead."""
        warnings.warn(
            "update_molecule_parameters() is deprecated. "
            "Use update_source_parameters(result) instead.",
            DeprecationWarning, stacklevel=2,
        )
        if isinstance(fitted_params, FitResult):
            self.update_source_parameters(fitted_params)
            return
        # Legacy dict — map old key names to FitResult
        params: Dict[str, float] = {}
        if 'temperature' in fitted_params:
            params['temp'] = fitted_params['temperature']
        for key in ('n_mol', 'radius'):
            if key in fitted_params:
                params[key] = fitted_params[key]
        legacy = FitResult(
            parameters=params,
            chi2_final=fitted_params.get('chi2_final', float('nan')),
            iterations=fitted_params.get('iterations', 0),
            function_calls=fitted_params.get('function_calls', 0),
            convergence_flag=fitted_params.get('convergence_flag', 0),
        )
        self.update_source_parameters(legacy)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def save_results(
        self,
        result: Union[FitResult, dict],
        filename: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """Save fitting results to a file.

        Accepts either a :class:`FitResult` (new API) or a legacy ``dict``
        (old API).

        Parameters
        ----------
        result : FitResult or dict
            Fit result to save.
        filename : str, optional
            Output filename; auto-generated from source name when ``None``.
        format : str
            ``"json"`` (default) or ``"txt"``.

        Returns
        -------
        str
            Full path to the saved file.
        """
        fmt = format.lower()
        if fmt not in ("json", "txt"):
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json' or 'txt'.")

        source_name = self._pipeline.source_name if self._pipeline else "unknown"
        if filename is None:
            filename = f"slab_fit_results_{source_name}.{fmt}"

        output_path = os.path.join(self.output_folder, str(filename))
        os.makedirs(self.output_folder, exist_ok=True)

        if isinstance(result, FitResult):
            if fmt == "json":
                self._save_fit_result_json(result, output_path, source_name)
            else:
                self._save_fit_result_txt(result, output_path, source_name)
        else:
            # Legacy dict path
            if fmt == "json":
                self._save_legacy_json(result, output_path, source_name)
            else:
                self._save_legacy_txt(result, output_path, source_name)

        print(f"Results saved to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private writers — FitResult
    # ------------------------------------------------------------------
    def _save_fit_result_json(
        self, result: FitResult, output_path: str, source_name: str
    ) -> None:
        payload = {
            "source": source_name,
            "input_file": self.input_file,
            "fitted_parameters": result.parameters,
            "fitting_statistics": {
                "chi2_final":       result.chi2_final,
                "iterations":       result.iterations,
                "function_calls":   result.function_calls,
                "convergence_flag": result.convergence_flag,
                "message":          result.message,
            },
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _save_fit_result_txt(
        self, result: FitResult, output_path: str, source_name: str
    ) -> None:
        with open(output_path, "w") as f:
            f.write("Slab Model Fitting Results\n")
            f.write("==========================\n\n")
            f.write(f"Source: {source_name}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            f.write("Fitted Parameters:\n")
            for name, val in result.parameters.items():
                f.write(f"  {name}: {val:.6g}\n")
            f.write("\nFitting Statistics:\n")
            f.write(f"  Final chi2: {result.chi2_final:.6e}\n")
            f.write(f"  Iterations: {result.iterations}\n")
            f.write(f"  Function calls: {result.function_calls}\n")
            f.write(f"  Convergence flag: {result.convergence_flag}\n")
            if result.message:
                f.write(f"  Message: {result.message}\n")

    # ------------------------------------------------------------------
    # Private writers — legacy dict (backward compat)
    # ------------------------------------------------------------------
    def _save_legacy_json(
        self, fitted_params: dict, output_path: str, source_name: str
    ) -> None:
        payload = {
            "source": source_name,
            "input_file": self.input_file,
            "fitted_parameters": {
                "temperature_K": fitted_params.get("temperature"),
                "log_n_mol":     fitted_params.get("log_n_mol"),
                "n_mol_cm2":     fitted_params.get("n_mol"),
                "radius_au":     fitted_params.get("radius"),
            },
            "fitting_statistics": {
                "chi2_final":       fitted_params.get("chi2_final"),
                "iterations":       fitted_params.get("iterations"),
                "function_calls":   fitted_params.get("function_calls"),
                "convergence_flag": fitted_params.get("convergence_flag"),
            },
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _save_legacy_txt(
        self, fitted_params: dict, output_path: str, source_name: str
    ) -> None:
        with open(output_path, "w") as f:
            f.write("Slab Model Fitting Results\n")
            f.write("==========================\n\n")
            f.write(f"Source: {source_name}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            f.write("Fitted Parameters:\n")
            f.write(f"  Temperature: {fitted_params.get('temperature', 'N/A'):.2f} K\n")
            f.write(f"  Column density: {fitted_params.get('n_mol', 'N/A'):.3e} cm**-2\n")
            f.write(f"  Radius: {fitted_params.get('radius', 'N/A'):.3f} au\n\n")
            f.write("Fitting Statistics:\n")
            f.write(f"  Final chi2: {fitted_params.get('chi2_final', 'N/A'):.6e}\n")
            f.write(f"  Iterations: {fitted_params.get('iterations', 'N/A')}\n")
            f.write(f"  Function calls: {fitted_params.get('function_calls', 'N/A')}\n")
            f.write(f"  Convergence flag: {fitted_params.get('convergence_flag', 'N/A')}\n")

# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
SlabFit = SlabModel