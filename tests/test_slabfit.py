# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.DataProcessing.Slabfit.

Covers:
  - FitParameter dataclass (internal space, bounds conversion)
  - FitResult dataclass
  - _SlabPipeline input-type detection and default extraction
  - SlabModel construction (direct + class-method factories + deprecated alias)
  - Config defaults and kwargs overrides
  - Fluent fit-parameter API (add / set / reset / remove)
  - evaluate_model (mocked chi2)
  - fit() happy path + edge cases
  - update_source_parameters for all input types
  - Deprecated wrappers (fit_parameters, update_molecule_parameters)
  - save_results JSON + TXT for FitResult and legacy dict
  - SlabFit backward-compatibility alias
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

import iSLAT.Constants as c
from iSLAT.Modules.DataProcessing.Slabfit import (
    FitParameter,
    FitResult,
    SlabModel,
    SlabFit,
    _DEFAULT_FIT_PARAMETERS,
    _SlabPipeline,
)
from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
from iSLAT.Modules.DataTypes.Intensity import Intensity
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.Spectrum import Spectrum


# ===========================================================================
# Shared helpers / fixtures
# ===========================================================================

_LINES_DATA = [
    {
        'nr': 1, 'lev_up': '0|10', 'lev_low': '0|9',
        'lam': 12.407, 'freq': 2.416e13, 'a_stein': 1.05e-2,
        'e_up': 4586.4, 'e_low': 3379.0, 'g_up': 21, 'g_low': 19,
    },
    {
        'nr': 2, 'lev_up': '0|8', 'lev_low': '0|7',
        'lam': 14.950, 'freq': 2.005e13, 'a_stein': 2.10e-2,
        'e_up': 3200.0, 'e_low': 2500.0, 'g_up': 17, 'g_low': 15,
    },
    {
        'nr': 3, 'lev_up': '0|6', 'lev_low': '0|5',
        'lam': 17.221, 'freq': 1.741e13, 'a_stein': 5.00e-3,
        'e_up': 2100.0, 'e_low': 1600.0, 'g_up': 13, 'g_low': 11,
    },
]


@pytest.fixture
def mll():
    """Minimal MoleculeLineList used across many tests."""
    return MoleculeLineList(molecule_id='H2O', lines_data=_LINES_DATA)


@pytest.fixture
def real_molecule(mll):
    """A real Molecule object with a pre-loaded line list."""
    mol = Molecule(
        name='H2O',
        displaylabel='H$_2$O',
        filepath=None,
        color='#0000FF',
        is_visible=True,
        temp=500.0,
        radius=1.0,
        n_mol=1e17,
        distance=160.0,
        fwhm=130.0,
        initial_molecule_parameters={
            't_kin': 500.0,
            'scale_exponent': 17.0,
            'scale_number': 1.0,
            'radius_init': 1.0,
        },
    )
    mol.lines = mll  # inject line list directly
    return mol


@pytest.fixture
def real_intensity(mll):
    """A minimal Intensity object with pre-set parameters (no partition data needed)."""
    inten = Intensity(mll)
    # Set private backing attrs directly — avoids needing HITRAN partition data
    inten._t_kin = 500.0
    inten._n_mol = 1e17
    inten._dv = 1.0
    return inten


# Convenience aliases so fixture names stay consistent with test parameter names
@pytest.fixture
def mock_molecule(real_molecule):
    return real_molecule


@pytest.fixture
def mock_intensity(real_intensity):
    return real_intensity


@pytest.fixture
def make_slab(tmp_path, real_molecule):
    """Factory that creates a SlabModel with a mocked chi2 evaluator."""
    def _make(source=None, chi2_value=1.0, **kwargs):
        src = source if source is not None else real_molecule
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=src,
            input_file=str(tmp_path / 'nonexistent.csv'),
            **kwargs,
        )
        # Patch chi2 so no real data file is needed
        slab.chi2_evaluator = MagicMock()
        slab.chi2_evaluator.chi2_total = chi2_value
        return slab
    return _make


# ===========================================================================
# FitParameter
# ===========================================================================

class TestFitParameter:

    def test_linear_internal_initial(self):
        p = FitParameter('temp', 500.0)
        assert p.internal_initial == pytest.approx(500.0)

    def test_log_internal_initial(self):
        p = FitParameter('n_mol', 1e17, log_scale=True)
        assert p.internal_initial == pytest.approx(17.0)

    def test_to_physical_linear(self):
        p = FitParameter('radius', 2.5)
        assert p.to_physical(2.5) == pytest.approx(2.5)

    def test_to_physical_log(self):
        p = FitParameter('n_mol', 1e17, log_scale=True)
        assert p.to_physical(18.0) == pytest.approx(1e18)

    def test_internal_bounds_linear(self):
        p = FitParameter('temp', 500.0, bounds=(10.0, 5000.0))
        lo, hi = p.internal_bounds()
        assert lo == pytest.approx(10.0)
        assert hi == pytest.approx(5000.0)

    def test_internal_bounds_log(self):
        p = FitParameter('n_mol', 1e17, bounds=(1e12, 1e25), log_scale=True)
        lo, hi = p.internal_bounds()
        assert lo == pytest.approx(12.0)
        assert hi == pytest.approx(25.0)

    def test_internal_bounds_infinite(self):
        p = FitParameter('x', 1.0)  # default (-inf, inf)
        lo, hi = p.internal_bounds()
        assert math.isinf(lo) and lo < 0
        assert math.isinf(hi) and hi > 0

    def test_internal_bounds_log_zero_lower(self):
        """Lower bound of 0 in log scale should become -inf."""
        p = FitParameter('n_mol', 1e17, bounds=(0.0, 1e25), log_scale=True)
        lo, _ = p.internal_bounds()
        assert math.isinf(lo)


# ===========================================================================
# FitResult
# ===========================================================================

class TestFitResult:

    def test_construction(self):
        r = FitResult(
            parameters={'temp': 300.0, 'n_mol': 5e16, 'radius': 1.5},
            chi2_final=0.42,
            iterations=100,
            function_calls=200,
            convergence_flag=0,
            message='Optimization terminated successfully.',
        )
        assert r.parameters['temp'] == 300.0
        assert r.chi2_final == pytest.approx(0.42)
        assert r.convergence_flag == 0

    def test_default_message(self):
        r = FitResult(
            parameters={},
            chi2_final=1.0,
            iterations=0,
            function_calls=0,
            convergence_flag=0,
        )
        assert r.message == ''


# ===========================================================================
# _SlabPipeline
# ===========================================================================

class TestSlabPipeline:

    def test_molecule_input_type(self, mock_molecule):
        pipeline = _SlabPipeline(mock_molecule)
        assert pipeline.input_type == 'molecule'
        assert pipeline._line_list is mock_molecule.lines

    def test_intensity_input_type(self, mock_intensity, mll):
        pipeline = _SlabPipeline(mock_intensity)
        assert pipeline.input_type == 'intensity'
        assert pipeline._line_list is mll

    def test_line_list_input_type(self, mll):
        pipeline = _SlabPipeline(mll)
        assert pipeline.input_type == 'line_list'
        assert pipeline._line_list is mll

    def test_spectrum_input_type(self):
        spec = MagicMock(spec=Spectrum)
        pipeline = _SlabPipeline(spec)
        assert pipeline.input_type == 'spectrum'
        assert pipeline._line_list is None

    def test_invalid_source_raises(self):
        with pytest.raises(TypeError, match="Unsupported source type"):
            _SlabPipeline("not-a-valid-source")

    def test_defaults_from_molecule(self, mock_molecule):
        pipeline = _SlabPipeline(mock_molecule)
        defs = pipeline.source_defaults
        assert defs['temp'] == pytest.approx(mock_molecule.temp)
        assert defs['n_mol'] == pytest.approx(mock_molecule.n_mol)
        assert defs['radius'] == pytest.approx(mock_molecule.radius)
        assert defs['distance'] == pytest.approx(mock_molecule.distance)
        assert defs['min_wavelength'] == pytest.approx(mock_molecule.wavelength_range[0])
        assert defs['max_wavelength'] == pytest.approx(mock_molecule.wavelength_range[1])

    def test_defaults_from_intensity(self, mock_intensity):
        pipeline = _SlabPipeline(mock_intensity)
        defs = pipeline.source_defaults
        assert defs['temp'] == pytest.approx(mock_intensity.t_kin)
        assert defs['broad'] == pytest.approx(mock_intensity.dv)

    def test_defaults_from_line_list_are_global(self, mll):
        """Without a rich source, defaults fall back to module-level constants."""
        pipeline = _SlabPipeline(mll)
        defs = pipeline.source_defaults
        assert defs['temp'] == pytest.approx(500.0)
        assert defs['distance'] == pytest.approx(c.DEFAULT_DISTANCE)

    def test_source_name_molecule(self, mock_molecule):
        pipeline = _SlabPipeline(mock_molecule)
        assert pipeline.source_name == 'H2O'

    def test_source_name_spectrum(self):
        spec = MagicMock(spec=Spectrum)
        pipeline = _SlabPipeline(spec)
        assert pipeline.source_name == 'spectrum'


# ===========================================================================
# SlabModel construction
# ===========================================================================

class TestSlabModelInit:

    def test_basic_construction(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'obs.csv'),
        )
        assert slab.output_folder == str(tmp_path)
        assert slab._pipeline is not None
        assert slab._pipeline.input_type == 'molecule'

    def test_no_source_allowed(self, tmp_path):
        """SlabModel may be constructed without a source (chi2-only mode)."""
        slab = SlabModel(output_folder=str(tmp_path))
        assert slab._pipeline is None

    def test_deprecated_mol_object(self, tmp_path, mock_molecule):
        with pytest.warns(DeprecationWarning, match="mol_object"):
            slab = SlabModel(
                output_folder=str(tmp_path),
                mol_object=mock_molecule,
                input_file=str(tmp_path / 'obs.csv'),
            )
        assert slab._pipeline is not None
        assert slab.mol_object is mock_molecule

    def test_from_molecule(self, tmp_path, mock_molecule):
        slab = SlabModel.from_molecule(
            mock_molecule,
            chi2_input_file=str(tmp_path / 'obs.csv'),
            output_folder=str(tmp_path),
        )
        assert slab._pipeline.input_type == 'molecule'

    def test_from_intensity(self, tmp_path, mock_intensity):
        slab = SlabModel.from_intensity(
            mock_intensity,
            chi2_input_file=str(tmp_path / 'obs.csv'),
            output_folder=str(tmp_path),
        )
        assert slab._pipeline.input_type == 'intensity'

    def test_from_line_list(self, tmp_path, mll):
        slab = SlabModel.from_line_list(
            mll,
            chi2_input_file=str(tmp_path / 'obs.csv'),
            output_folder=str(tmp_path),
        )
        assert slab._pipeline.input_type == 'line_list'

    def test_from_spectrum(self, tmp_path):
        spec = MagicMock(spec=Spectrum)
        slab = SlabModel.from_spectrum(
            spec,
            chi2_input_file=str(tmp_path / 'obs.csv'),
            output_folder=str(tmp_path),
        )
        assert slab._pipeline.input_type == 'spectrum'


# ===========================================================================
# Config defaults and kwargs overrides
# ===========================================================================

class TestSlabModelConfig:

    def test_config_picks_up_molecule_attrs(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        assert slab._config['temp'] == pytest.approx(mock_molecule.temp)
        assert slab._config['n_mol'] == pytest.approx(mock_molecule.n_mol)
        assert slab._config['radius'] == pytest.approx(mock_molecule.radius)
        assert slab._config['distance'] == pytest.approx(mock_molecule.distance)

    def test_kwargs_override_molecule_attrs(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
            distance=250.0,
            fwhm=50.0,
        )
        assert slab._config['distance'] == pytest.approx(250.0)
        assert slab._config['fwhm'] == pytest.approx(50.0)
        # Other attrs still come from molecule
        assert slab._config['temp'] == pytest.approx(mock_molecule.temp)

    def test_intrinsic_line_width_alias(self, tmp_path, mock_molecule):
        """'intrinsic_line_width' kwarg should populate 'broad' config key."""
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
            intrinsic_line_width=3.5,
        )
        assert slab._config['broad'] == pytest.approx(3.5)

    def test_column_name_overrides(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
            flux_col_name='FluxCustom',
            error_col_name='ErrCustom',
        )
        assert slab._config['flux_col_name'] == 'FluxCustom'
        assert slab._config['error_col_name'] == 'ErrCustom'

    def test_default_fit_params_seeded_from_config(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        names = [p.name for p in slab._fit_params]
        assert 'temp' in names
        assert 'n_mol' in names
        assert 'radius' in names
        # initial values should match molecule
        temp_param = next(p for p in slab._fit_params if p.name == 'temp')
        assert temp_param.initial_value == pytest.approx(mock_molecule.temp)


# ===========================================================================
# Fit-parameter fluent API
# ===========================================================================

class TestFitParameterAPI:

    def test_add_fit_parameter(self, make_slab):
        slab = make_slab()
        initial_count = len(slab._fit_params)
        result = slab.add_fit_parameter('fwhm', 130.0, bounds=(10.0, 500.0))
        assert result is slab  # fluent
        assert len(slab._fit_params) == initial_count + 1
        assert slab._fit_params[-1].name == 'fwhm'

    def test_set_fit_parameters(self, make_slab):
        slab = make_slab()
        new_params = [FitParameter('temp', 300.0, (10.0, 2000.0))]
        result = slab.set_fit_parameters(new_params)
        assert result is slab
        assert len(slab._fit_params) == 1
        assert slab._fit_params[0].name == 'temp'

    def test_reset_fit_parameters(self, make_slab):
        slab = make_slab()
        slab.set_fit_parameters([FitParameter('fwhm', 130.0)])
        result = slab.reset_fit_parameters()
        assert result is slab
        names = [p.name for p in slab._fit_params]
        assert 'temp' in names
        assert 'n_mol' in names
        assert 'radius' in names

    def test_remove_fit_parameter(self, make_slab):
        slab = make_slab()
        count_before = len(slab._fit_params)
        result = slab.remove_fit_parameter('n_mol')
        assert result is slab
        assert len(slab._fit_params) == count_before - 1
        assert all(p.name != 'n_mol' for p in slab._fit_params)

    def test_remove_nonexistent_is_noop(self, make_slab):
        slab = make_slab()
        count_before = len(slab._fit_params)
        slab.remove_fit_parameter('nonexistent_param')
        assert len(slab._fit_params) == count_before

    def test_chaining(self, make_slab):
        slab = make_slab()
        (
            slab
            .remove_fit_parameter('radius')
            .add_fit_parameter('fwhm', 130.0, (10.0, 500.0))
        )
        names = [p.name for p in slab._fit_params]
        assert 'radius' not in names
        assert 'fwhm' in names


# ===========================================================================
# evaluate_model
# ===========================================================================

class TestEvaluateModel:

    def test_returns_float(self, make_slab):
        slab = make_slab(chi2_value=5.3)
        # Must patch pipeline.evaluate to avoid building a real spectrum
        with patch.object(slab._pipeline, 'evaluate', return_value=MagicMock()):
            chi2 = slab.evaluate_model(temp=500.0, n_mol=1e17, radius=1.0)
        assert chi2 == pytest.approx(5.3)

    def test_no_source_raises(self, tmp_path):
        slab = SlabModel(output_folder=str(tmp_path))
        with pytest.raises(RuntimeError, match="No source provided"):
            slab.evaluate_model(temp=500.0)

    def test_calls_pipeline_evaluate(self, make_slab):
        slab = make_slab()
        fake_spectrum = MagicMock()
        with patch.object(slab._pipeline, 'evaluate', return_value=fake_spectrum) as mock_eval:
            slab.evaluate_model(temp=300.0, n_mol=1e16)
        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args[0][0]
        assert call_kwargs['temp'] == pytest.approx(300.0)
        assert call_kwargs['n_mol'] == pytest.approx(1e16)

    def test_calls_chi2_evaluator(self, make_slab):
        slab = make_slab()
        fake_spectrum = MagicMock()
        with patch.object(slab._pipeline, 'evaluate', return_value=fake_spectrum):
            slab.evaluate_model(temp=500.0)
        slab.chi2_evaluator.evaluate_spectrum.assert_called_once_with(fake_spectrum)


# ===========================================================================
# fit()
# ===========================================================================

class TestFit:

    def _make_fitting_slab(self, tmp_path, mock_molecule):
        """Slab ready to run fit() with a patched objective that returns chi2=0."""
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        # Freeze chi2 to a constant so the optimizer converges fast
        slab.chi2_evaluator = MagicMock()
        slab.chi2_evaluator.chi2_total = 0.0
        return slab

    def test_returns_fit_result(self, tmp_path, mock_molecule):
        slab = self._make_fitting_slab(tmp_path, mock_molecule)
        with patch.object(slab._pipeline, 'evaluate', return_value=MagicMock()):
            result = slab.fit()
        assert isinstance(result, FitResult)

    def test_result_has_all_default_params(self, tmp_path, mock_molecule):
        slab = self._make_fitting_slab(tmp_path, mock_molecule)
        with patch.object(slab._pipeline, 'evaluate', return_value=MagicMock()):
            result = slab.fit()
        assert 'temp' in result.parameters
        assert 'n_mol' in result.parameters
        assert 'radius' in result.parameters

    def test_initial_overrides_respected(self, tmp_path, mock_molecule):
        slab = self._make_fitting_slab(tmp_path, mock_molecule)
        captured = []

        def _mock_eval(param_values, config):
            captured.append(dict(param_values))
            return MagicMock()

        with patch.object(slab._pipeline, 'evaluate', side_effect=_mock_eval):
            slab.fit(initial_overrides={'temp': 999.0})

        # First call should use the overridden temperature
        assert captured[0]['temp'] == pytest.approx(999.0)

    def test_no_source_raises(self, tmp_path):
        slab = SlabModel(output_folder=str(tmp_path))
        with pytest.raises(RuntimeError, match="No source provided"):
            slab.fit()

    def test_spectrum_source_raises(self, tmp_path):
        spec = MagicMock(spec=Spectrum)
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=spec,
            input_file=str(tmp_path / 'x.csv'),
        )
        with pytest.raises(ValueError, match="Cannot fit a Spectrum"):
            slab.fit()

    def test_no_fit_params_raises(self, tmp_path, mock_molecule):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        slab.set_fit_parameters([])
        with pytest.raises(ValueError, match="No fit parameters"):
            slab.fit()

    def test_bounded_params_use_lbfgsb(self, tmp_path, mock_molecule):
        """When any FitParameter has finite bounds, L-BFGS-B must be used."""
        slab = self._make_fitting_slab(tmp_path, mock_molecule)
        # Default params all have finite bounds → expect L-BFGS-B
        calls = []

        original_minimize = __import__(
            'iSLAT.Modules.DataProcessing.Slabfit', fromlist=['minimize']
        )

        with patch('iSLAT.Modules.DataProcessing.Slabfit.minimize') as mock_min:
            mock_result = MagicMock()
            mock_result.fun = 0.0
            mock_result.x = np.array([500.0, 17.0, 1.0])
            mock_result.success = True
            mock_result.message = 'OK'
            mock_result.nit = 1
            mock_result.nfev = 1
            mock_min.return_value = mock_result
            slab.fit()
            _, kwargs = mock_min.call_args
            assert kwargs['method'] == 'L-BFGS-B'

    def test_unbounded_params_use_nelder_mead(self, tmp_path, mock_molecule):
        """Params with all-infinite bounds → Nelder-Mead."""
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
            fit_parameters=[
                FitParameter('temp', 500.0),   # bounds = (-inf, inf)
            ],
        )
        slab.chi2_evaluator = MagicMock()
        slab.chi2_evaluator.chi2_total = 0.0

        with patch('iSLAT.Modules.DataProcessing.Slabfit.minimize') as mock_min:
            mock_result = MagicMock()
            mock_result.fun = 0.0
            mock_result.x = np.array([500.0])
            mock_result.success = True
            mock_result.message = 'OK'
            mock_result.nit = 1
            mock_result.nfev = 1
            mock_min.return_value = mock_result
            with patch.object(slab._pipeline, 'evaluate', return_value=MagicMock()):
                slab.fit()
            _, kwargs = mock_min.call_args
            assert kwargs['method'] == 'Nelder-Mead'
            assert kwargs['bounds'] is None


# ===========================================================================
# update_source_parameters
# ===========================================================================

class TestUpdateSourceParameters:

    def _make_result(self, temp=300.0, n_mol=5e16, radius=1.5):
        return FitResult(
            parameters={'temp': temp, 'n_mol': n_mol, 'radius': radius},
            chi2_final=0.1,
            iterations=10,
            function_calls=20,
            convergence_flag=0,
        )

    def test_molecule_calls_bulk_update(self, make_slab, mock_molecule):
        slab = make_slab(source=mock_molecule)
        result = self._make_result()
        with patch.object(Molecule, 'bulk_update_parameters', autospec=True) as mock_bulk:
            slab.update_source_parameters(result)
        mock_bulk.assert_called_once_with(mock_molecule, result.parameters)

    def test_intensity_sets_attributes(self, tmp_path, mock_intensity):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mock_intensity,
            input_file=str(tmp_path / 'x.csv'),
        )
        result = FitResult(
            parameters={'temp': 350.0, 'n_mol': 2e16, 'broad': 2.0},
            chi2_final=0.5,
            iterations=5,
            function_calls=10,
            convergence_flag=0,
        )
        slab.update_source_parameters(result)
        assert mock_intensity.t_kin == pytest.approx(350.0)
        assert mock_intensity.n_mol == pytest.approx(2e16)
        assert mock_intensity.dv == pytest.approx(2.0)

    def test_spectrum_emits_warning(self, tmp_path):
        spec = MagicMock(spec=Spectrum)
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=spec,
            input_file=str(tmp_path / 'x.csv'),
        )
        result = self._make_result()
        with pytest.warns(UserWarning, match="no effect for Spectrum"):
            slab.update_source_parameters(result)

    def test_line_list_emits_warning(self, tmp_path, mll):
        slab = SlabModel(
            output_folder=str(tmp_path),
            source=mll,
            input_file=str(tmp_path / 'x.csv'),
        )
        result = self._make_result()
        with pytest.warns(UserWarning, match="no effect for MoleculeLineList"):
            slab.update_source_parameters(result)

    def test_no_source_is_noop(self, tmp_path):
        slab = SlabModel(output_folder=str(tmp_path))
        result = self._make_result()
        slab.update_source_parameters(result)  # must not raise


# ===========================================================================
# Deprecated aliases
# ===========================================================================

class TestDeprecatedAPI:

    def test_fit_parameters_deprecated(self, make_slab):
        slab = make_slab()
        with patch.object(slab, 'fit', return_value=FitResult(
            parameters={'temp': 500.0, 'n_mol': 1e17, 'radius': 1.0},
            chi2_final=1.0, iterations=1, function_calls=1, convergence_flag=0,
        )) as mock_fit:
            with pytest.warns(DeprecationWarning, match="fit_parameters"):
                result = slab.fit_parameters(start_t=300.0, start_r=2.0)
        mock_fit.assert_called_once()
        overrides = mock_fit.call_args[1]['initial_overrides']
        assert overrides['temp'] == pytest.approx(300.0)
        assert overrides['radius'] == pytest.approx(2.0)
        assert 'n_mol' not in overrides

    def test_fit_parameters_no_args(self, make_slab):
        slab = make_slab()
        with patch.object(slab, 'fit', return_value=FitResult(
            parameters={}, chi2_final=0.0, iterations=0, function_calls=0, convergence_flag=0,
        )) as mock_fit:
            with pytest.warns(DeprecationWarning):
                slab.fit_parameters()
        # When no overrides, should pass None
        call_kwargs = mock_fit.call_args[1]
        assert call_kwargs['initial_overrides'] is None

    def test_update_molecule_parameters_with_fit_result(self, make_slab, mock_molecule):
        slab = make_slab(source=mock_molecule)
        result = FitResult(
            parameters={'temp': 300.0, 'n_mol': 5e16, 'radius': 1.5},
            chi2_final=0.0, iterations=0, function_calls=0, convergence_flag=0,
        )
        with patch.object(Molecule, 'bulk_update_parameters', autospec=True) as mock_bulk:
            with pytest.warns(DeprecationWarning, match="update_molecule_parameters"):
                slab.update_molecule_parameters(result)
        mock_bulk.assert_called_once_with(mock_molecule, result.parameters)

    def test_update_molecule_parameters_with_legacy_dict(self, make_slab, mock_molecule):
        slab = make_slab(source=mock_molecule)
        legacy = {
            'temperature': 400.0,
            'n_mol': 3e17,
            'radius': 2.0,
            'chi2_final': 0.5,
            'iterations': 10,
            'function_calls': 20,
            'convergence_flag': 0,
        }
        with patch.object(Molecule, 'bulk_update_parameters', autospec=True) as mock_bulk:
            with pytest.warns(DeprecationWarning):
                slab.update_molecule_parameters(legacy)
        call_args = mock_bulk.call_args[0][1]  # [0] is self, [1] is the dict arg
        assert call_args['temp'] == pytest.approx(400.0)  # key remapped
        assert call_args['n_mol'] == pytest.approx(3e17)
        assert call_args['radius'] == pytest.approx(2.0)


# ===========================================================================
# save_results
# ===========================================================================

class TestSaveResults:

    def _make_result(self):
        return FitResult(
            parameters={'temp': 350.0, 'n_mol': 5e16, 'radius': 1.2},
            chi2_final=0.42,
            iterations=75,
            function_calls=150,
            convergence_flag=0,
            message='Optimization terminated successfully.',
        )

    def test_saves_json_fit_result(self, make_slab, tmp_path):
        slab = make_slab()
        result = self._make_result()
        output_path = slab.save_results(result, filename='test_out.json', format='json')
        assert Path(output_path).exists()
        with open(output_path) as f:
            data = json.load(f)
        assert 'fitted_parameters' in data
        assert data['fitted_parameters']['temp'] == pytest.approx(350.0)
        assert data['fitting_statistics']['chi2_final'] == pytest.approx(0.42)
        assert data['fitting_statistics']['convergence_flag'] == 0

    def test_saves_txt_fit_result(self, make_slab, tmp_path):
        slab = make_slab()
        result = self._make_result()
        output_path = slab.save_results(result, filename='test_out.txt', format='txt')
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert 'temp' in content
        assert '350' in content  # temp value formatted with :.6g
        assert '4.2' in content  # chi2 0.42 formatted as 4.200000e-01

    def test_saves_legacy_json_dict(self, make_slab, tmp_path):
        slab = make_slab()
        legacy = {
            'temperature': 500.0, 'n_mol': 1e17, 'radius': 1.0,
            'chi2_final': 1.5, 'iterations': 100,
            'function_calls': 200, 'convergence_flag': 0,
        }
        output_path = slab.save_results(legacy, filename='legacy.json', format='json')
        assert Path(output_path).exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data['fitted_parameters']['temperature_K'] == pytest.approx(500.0)

    def test_saves_legacy_txt_dict(self, make_slab, tmp_path):
        slab = make_slab()
        legacy = {
            'temperature': 500.0, 'n_mol': 1e17, 'radius': 1.0,
            'chi2_final': 1.5, 'iterations': 100,
            'function_calls': 200, 'convergence_flag': 0,
        }
        output_path = slab.save_results(legacy, filename='legacy.txt', format='txt')
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert 'Temperature' in content

    def test_auto_filename_from_source_name(self, make_slab, tmp_path):
        slab = make_slab()
        result = self._make_result()
        output_path = slab.save_results(result, format='json')
        # Filename should contain the source name
        assert 'H2O' in Path(output_path).name

    def test_auto_filename_txt_extension(self, make_slab, tmp_path):
        slab = make_slab()
        result = self._make_result()
        output_path = slab.save_results(result, format='txt')
        assert Path(output_path).suffix == '.txt'

    def test_unsupported_format_raises(self, make_slab):
        slab = make_slab()
        with pytest.raises(ValueError, match="Unsupported format"):
            slab.save_results(self._make_result(), format='xml')

    def test_creates_output_directory(self, tmp_path, mock_molecule):
        nested = tmp_path / 'deep' / 'nested' / 'dir'
        slab = SlabModel(
            output_folder=str(nested),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        slab.chi2_evaluator = MagicMock()
        result = self._make_result()
        output_path = slab.save_results(result, filename='out.json', format='json')
        assert Path(output_path).exists()

    def test_returns_output_path_string(self, make_slab):
        slab = make_slab()
        result = self._make_result()
        output_path = slab.save_results(result, filename='retval.json')
        assert isinstance(output_path, str)
        assert output_path.endswith('retval.json')


# ===========================================================================
# SlabFit alias
# ===========================================================================

class TestSlabFitAlias:

    def test_is_same_class(self):
        assert SlabFit is SlabModel

    def test_can_construct(self, tmp_path, mock_molecule):
        slab = SlabFit(
            output_folder=str(tmp_path),
            source=mock_molecule,
            input_file=str(tmp_path / 'x.csv'),
        )
        assert isinstance(slab, SlabModel)


# ===========================================================================
# Reduced chi-squared
# ===========================================================================

class TestReducedChi2:

    @staticmethod
    def _result(chi2=12.0, n_data=10, n_free=3):
        return FitResult(parameters={}, chi2_final=chi2, iterations=1, function_calls=1,
                         convergence_flag=0, n_data=n_data, n_free_params=n_free)

    def test_value(self):
        r = self._result()
        assert r.degrees_of_freedom == 7
        assert r.reduced_chi2 == pytest.approx(12.0 / 7)

    @pytest.mark.parametrize("n_data, n_free", [(3, 3), (2, 3), (0, 0)])
    def test_undefined_without_degrees_of_freedom(self, n_data, n_free):
        r = self._result(n_data=n_data, n_free=n_free)
        assert math.isnan(r.reduced_chi2)
        assert "reduced χ² undefined" in r.chi2_summary()

    def test_summary(self):
        assert self._result().chi2_summary() == (
            "χ² = 1.200e+01, reduced χ² = 1.71 (ν = 10 - 3 = 7)"
        )

    def test_fit_prints_both_to_the_data_field(self, tmp_path, mock_molecule):
        data_field = MagicMock()
        slab = SlabModel(output_folder=str(tmp_path), source=mock_molecule,
                         input_file=str(tmp_path / 'x.csv'), data_field=data_field)
        slab.chi2_evaluator = MagicMock()
        slab.chi2_evaluator.chi2_total = 12.0
        slab.chi2_evaluator.n_contributing = 10
        with patch.object(slab._pipeline, 'evaluate', return_value=MagicMock()):
            result = slab.fit()
        n_free = len(slab._fit_params)
        assert (result.n_data, result.n_free_params) == (10, n_free)
        final = data_field.insert_text.call_args_list[-1].args[0]
        assert final.startswith("Fitting complete. χ² = 1.200e+01")
        assert f"reduced χ² = {12.0 / (10 - n_free):.3g}" in final
        assert f"(ν = 10 - {n_free} = {10 - n_free})" in final

    def test_saved_results_include_it(self, make_slab):
        slab = make_slab()
        r = self._result()
        data = json.loads(Path(slab.save_results(r, filename='r.json', format='json')).read_text())
        stats = data['fitting_statistics']
        assert stats['reduced_chi2'] == pytest.approx(12.0 / 7)
        assert (stats['n_data'], stats['n_free_params'], stats['degrees_of_freedom']) == (10, 3, 7)
        text = Path(slab.save_results(r, filename='r.txt', format='txt')).read_text()
        assert "Reduced chi2: 1.714286e+00 (dof = 10 - 3 = 7)" in text

    def test_saved_json_undefined_is_null(self, make_slab):
        slab = make_slab()
        r = self._result(n_data=2, n_free=3)
        data = json.loads(Path(slab.save_results(r, filename='u.json', format='json')).read_text())
        assert data['fitting_statistics']['reduced_chi2'] is None
