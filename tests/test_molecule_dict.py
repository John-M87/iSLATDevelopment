# -*- coding: utf-8 -*-
"""Unit tests for MoleculeDict and its helper functions."""

import pytest
import numpy as np

import iSLAT.Constants as c


class TestMoleculeDict:
    """Tests for MoleculeDict."""

    def _make_molecule(self, name='TestH2O', **kwargs):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        defaults = {
            'name': name,
            'displaylabel': name,
            'filepath': None,
            'color': '#FF0000',
            'is_visible': True,
            'temp': 500.0,
            'radius': 1.0,
            'n_mol': 1e18,
            'distance': 160.0,
            'fwhm': 130.0,
            'initial_molecule_parameters': {
                't_kin': 500.0,
                'scale_exponent': 18.0,
                'scale_number': 1.0,
                'radius_init': 1.0,
            },
        }
        defaults.update(kwargs)
        return Molecule(**defaults)

    def _make_dict(self, **kwargs):
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        return MoleculeDict(**kwargs)

    def test_init_default(self):
        md = self._make_dict()
        assert isinstance(md, dict)
        assert len(md) == 0

    def test_init_global_parameters(self):
        md = self._make_dict(global_distance=200.0, global_stellar_rv=5.0)
        assert md._global_dist == 200.0
        assert md._global_stellar_rv == 5.0

    def test_add_molecule(self):
        md = self._make_dict()
        mol = self._make_molecule('H2O')
        md['H2O'] = mol
        assert 'H2O' in md
        assert len(md) == 1
        assert md['H2O'].name == 'H2O'

    def test_remove_molecule(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O')
        del md['H2O']
        assert 'H2O' not in md

    def test_get_molecule(self):
        md = self._make_dict()
        mol = self._make_molecule('OH')
        md['OH'] = mol
        result = md.get('OH')
        assert result is mol
        assert md.get('nonexistent') is None

    def test_get_visible_molecules(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O', is_visible=True)
        md['CO'] = self._make_molecule('CO', is_visible=False)
        md['OH'] = self._make_molecule('OH', is_visible=True)

        visible = md.get_visible_molecules()
        assert 'H2O' in visible
        assert 'OH' in visible
        assert 'CO' not in visible

    def test_get_visible_molecules_as_objects(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O', is_visible=True)
        md['CO'] = self._make_molecule('CO', is_visible=False)

        visible = md.get_visible_molecules(return_objects=True)
        assert len(visible) == 1
        assert visible[0].name == 'H2O'

    def test_bulk_set_visibility(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O', is_visible=True)
        md['CO'] = self._make_molecule('CO', is_visible=True)
        md['OH'] = self._make_molecule('OH', is_visible=True)

        md.bulk_set_visibility(False)
        for mol in md.values():
            assert mol.is_visible is False

        md.bulk_set_visibility(True, molecule_names=['H2O', 'OH'])
        assert md['H2O'].is_visible is True
        assert md['OH'].is_visible is True
        assert md['CO'].is_visible is False

    def test_apply_stellar_rv_zero(self):
        md = self._make_dict(global_stellar_rv=0.0)
        wave = np.linspace(10, 20, 100)
        result = md.apply_stellar_rv(wave)
        np.testing.assert_array_equal(result, wave)

    def test_apply_stellar_rv_nonzero(self):
        md = self._make_dict(global_stellar_rv=10.0)
        wave = np.array([10.0, 15.0, 20.0])
        result = md.apply_stellar_rv(wave)
        # With positive RV, wavelengths should shift
        assert not np.array_equal(result, wave)
        # The formula: wave - (wave / c_kms * rv)
        expected = wave - (wave / c.SPEED_OF_LIGHT_KMS * 10.0)
        np.testing.assert_allclose(result, expected)

    def test_clear(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O')
        md['CO'] = self._make_molecule('CO')
        md.clear()
        assert len(md) == 0

    def test_iter_and_contains(self):
        md = self._make_dict()
        md['H2O'] = self._make_molecule('H2O')
        md['CO'] = self._make_molecule('CO')
        keys = list(md)
        assert 'H2O' in keys
        assert 'CO' in keys
        assert 'H2O' in md
        assert 'OH' not in md

    def test_values_and_items(self):
        md = self._make_dict()
        mol = self._make_molecule('H2O')
        md['H2O'] = mol
        vals = list(md.values())
        assert len(vals) == 1
        assert vals[0] is mol
        items = list(md.items())
        assert items[0] == ('H2O', mol)

    def test_global_parameter_defaults(self):
        md = self._make_dict()
        assert md._global_parms['dist'] == c.DEFAULT_DISTANCE
        assert md._global_parms['stellar_rv'] == c.DEFAULT_STELLAR_RV
        assert md._global_parms['wavelength_range'] == c.WAVELENGTH_RANGE

    def test_matched_spectral_sampling_off_by_default(self):
        md = self._make_dict()
        assert md._match_spectral_sampling is False

    def test_get_matched_sampling_wavelengths_off(self):
        md = self._make_dict()
        wave = np.linspace(10, 20, 50)
        use_interp, target = md.get_matched_sampling_wavelengths(wave)
        assert use_interp is False
        np.testing.assert_array_equal(target, wave)

    def test_get_matched_sampling_wavelengths_on(self):
        md = self._make_dict(global_stellar_rv=10.0)
        md._match_spectral_sampling = True
        wave = np.linspace(10, 20, 50)
        use_interp, target = md.get_matched_sampling_wavelengths(wave)
        assert use_interp is True
        # target should be RV-corrected wavelengths
        assert not np.array_equal(target, wave)

    # ── global_wavelength_range setter ─────────────────────────────

    def test_global_wavelength_range_no_intermediate_class_callbacks(self):
        """Setting global_wavelength_range must NOT fire per-molecule
        class callbacks (which would trigger intermediate plot refreshes
        before all molecules are consistent)."""
        from iSLAT.Modules.DataTypes.Molecule import Molecule

        md = self._make_dict()
        mol_a = self._make_molecule('MolA')
        mol_b = self._make_molecule('MolB')
        md['MolA'] = mol_a
        md['MolB'] = mol_b

        original_range = (4.5, 28.0)
        md._global_wavelength_range = original_range
        mol_a._wavelength_range = original_range
        mol_b._wavelength_range = original_range

        # Track per-molecule class callbacks
        class_cb_calls = []
        def class_cb(mol_name, param_name, old_val, new_val):
            class_cb_calls.append((mol_name, param_name))

        Molecule.add_molecule_parameter_change_callback(class_cb)
        try:
            md.global_wavelength_range = (5.0, 20.0)

            # No per-molecule class callbacks should have fired
            wl_calls = [c for c in class_cb_calls if c[1] == 'wavelength_range']
            assert len(wl_calls) == 0, (
                f"Expected 0 per-molecule wavelength_range callbacks, got {len(wl_calls)}"
            )

            # But every molecule should have the new range and dirty flags
            for mol in md.values():
                assert mol._wavelength_range == (5.0, 20.0)
                assert mol._dirty_flags['intensity'] is True
                assert mol._dirty_flags['spectrum'] is True
                assert mol._dirty_flags['flux'] is True
        finally:
            Molecule.remove_molecule_parameter_change_callback(class_cb)

    def test_global_wavelength_range_fires_global_callback(self):
        """The global callback should fire exactly once after all molecules
        are updated."""
        md = self._make_dict()
        mol_a = self._make_molecule('MolA')
        md['MolA'] = mol_a
        md._global_wavelength_range = (4.5, 28.0)
        mol_a._wavelength_range = (4.5, 28.0)

        global_cb_calls = []
        md.add_global_parameter_change_callback(
            lambda p, o, n: global_cb_calls.append((p, o, n))
        )
        md.global_wavelength_range = (5.0, 20.0)

        assert len(global_cb_calls) == 1
        assert global_cb_calls[0] == ('wavelength_range', (4.5, 28.0), (5.0, 20.0))


class TestMoleculeDictActiveMoleculeSet:
    """Tests for the active molecule set managed by MoleculeDict (Phase 1)."""

    def _make_molecule(self, name, **kwargs):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        defaults = {
            'name': name,
            'displaylabel': name,
            'filepath': None,
            'color': '#FF0000',
            'is_visible': True,
            'temp': 500.0,
            'radius': 1.0,
            'n_mol': 1e18,
            'distance': 160.0,
            'fwhm': 130.0,
            'initial_molecule_parameters': {
                't_kin': 500.0,
                'scale_exponent': 18.0,
                'scale_number': 1.0,
                'radius_init': 1.0,
            },
        }
        defaults.update(kwargs)
        return Molecule(**defaults)

    def _make_dict_with_molecules(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict()
        md['H2O'] = self._make_molecule('H2O')
        md['CO2'] = self._make_molecule('CO2')
        md['CO']  = self._make_molecule('CO')
        return md

    # --- active_molecule property/setter -----------------------------------

    def test_active_molecule_defaults_none(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict()
        assert md.active_molecule is None

    def test_active_molecule_set_by_object(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        assert md.active_molecule is md['H2O']

    def test_active_molecule_set_by_string(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = 'CO2'
        assert md.active_molecule is md['CO2']

    def test_active_molecule_set_none(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        md.active_molecule = None
        assert md.active_molecule is None

    def test_active_molecule_unknown_string_raises(self):
        md = self._make_dict_with_molecules()
        with pytest.raises(ValueError):
            md.active_molecule = 'NonExistent'

    def test_active_molecule_bad_type_raises(self):
        md = self._make_dict_with_molecules()
        with pytest.raises(TypeError):
            md.active_molecule = 42

    # --- active_molecule_change callbacks ----------------------------------

    def test_active_molecule_callback_fires(self):
        md = self._make_dict_with_molecules()
        calls = []
        md.add_active_molecule_change_callback(lambda old, new: calls.append((old, new)))
        md.active_molecule = md['H2O']
        assert len(calls) == 1
        assert calls[0] == (None, md['H2O'])

    def test_active_molecule_callback_receives_old_and_new(self):
        md = self._make_dict_with_molecules()
        calls = []
        md.active_molecule = md['H2O']
        md.add_active_molecule_change_callback(lambda old, new: calls.append((old, new)))
        md.active_molecule = md['CO2']
        assert calls[0] == (md['H2O'], md['CO2'])

    def test_remove_active_molecule_callback(self):
        md = self._make_dict_with_molecules()
        calls = []
        cb = lambda old, new: calls.append((old, new))
        md.add_active_molecule_change_callback(cb)
        md.remove_active_molecule_change_callback(cb)
        md.active_molecule = md['H2O']
        assert calls == []

    # --- comparison_molecules / get_active_set ----------------------------

    def test_comparison_molecules_defaults_empty(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict()
        assert md.comparison_molecules == []

    def test_toggle_comparison_adds_molecule(self):
        md = self._make_dict_with_molecules()
        added = md.toggle_comparison_molecule(md['CO2'])
        assert added is True
        assert md['CO2'] in md.comparison_molecules

    def test_toggle_comparison_removes_molecule(self):
        md = self._make_dict_with_molecules()
        md.toggle_comparison_molecule(md['CO2'])
        removed = md.toggle_comparison_molecule(md['CO2'])
        assert removed is False
        assert md['CO2'] not in md.comparison_molecules

    def test_toggle_comparison_by_string(self):
        md = self._make_dict_with_molecules()
        md.toggle_comparison_molecule('CO')
        assert md['CO'] in md.comparison_molecules

    def test_get_active_set_primary_first(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        md.toggle_comparison_molecule(md['CO2'])
        result = md.get_active_set()
        assert result[0] is md['H2O']
        assert md['CO2'] in result

    def test_get_active_set_no_duplicates(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        md.toggle_comparison_molecule(md['H2O'])  # same molecule → should not duplicate
        result = md.get_active_set()
        assert result.count(md['H2O']) == 1

    def test_get_active_set_empty_when_nothing_active(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict()
        assert md.get_active_set() == []

    def test_comparison_molecules_returns_copy(self):
        md = self._make_dict_with_molecules()
        md.toggle_comparison_molecule(md['CO2'])
        copy = md.comparison_molecules
        copy.clear()
        assert md['CO2'] in md.comparison_molecules  # original unchanged

    # --- promote_to_active_molecule ----------------------------------------

    def test_promote_moves_old_primary_to_comparison(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        md.promote_to_active_molecule(md['CO2'])
        assert md.active_molecule is md['CO2']
        assert md['H2O'] in md.comparison_molecules

    def test_promote_removes_new_primary_from_comparison(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        md.toggle_comparison_molecule(md['CO2'])
        md.promote_to_active_molecule(md['CO2'])
        assert md['CO2'] not in md.comparison_molecules

    def test_promote_already_active_returns_false(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        result = md.promote_to_active_molecule(md['H2O'])
        assert result is False

    def test_promote_by_string(self):
        md = self._make_dict_with_molecules()
        md.active_molecule = md['H2O']
        result = md.promote_to_active_molecule('CO2')
        assert result is True
        assert md.active_molecule is md['CO2']

    # --- clear_comparison_molecules ----------------------------------------

    def test_clear_comparison_molecules(self):
        md = self._make_dict_with_molecules()
        md.toggle_comparison_molecule(md['CO2'])
        md.toggle_comparison_molecule(md['CO'])
        md.clear_comparison_molecules()
        assert md.comparison_molecules == []

    def test_clear_fires_comparison_callback(self):
        md = self._make_dict_with_molecules()
        md.toggle_comparison_molecule(md['CO2'])
        calls = []
        md.add_comparison_molecule_change_callback(lambda lst: calls.append(list(lst)))
        md.clear_comparison_molecules()
        assert len(calls) == 1

    # --- comparison_molecule_change callbacks ------------------------------

    def test_comparison_callback_fires_on_toggle(self):
        md = self._make_dict_with_molecules()
        calls = []
        md.add_comparison_molecule_change_callback(lambda lst: calls.append(list(lst)))
        md.toggle_comparison_molecule(md['CO2'])
        assert len(calls) == 1

    def test_remove_comparison_callback(self):
        md = self._make_dict_with_molecules()
        calls = []
        cb = lambda lst: calls.append(list(lst))
        md.add_comparison_molecule_change_callback(cb)
        md.remove_comparison_molecule_change_callback(cb)
        md.toggle_comparison_molecule(md['CO2'])
        assert calls == []


class TestMoleculeDictHelpers:
    """Tests for MoleculeDict helper functions (_ci_get, _safe_float)."""

    def test_ci_get_exact(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _ci_get
        d = {'Temp': 500, 'Name': 'H2O'}
        assert _ci_get(d, 'Temp') == 500
        assert _ci_get(d, 'Name') == 'H2O'

    def test_ci_get_case_insensitive(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _ci_get
        d = {'Temp': 500, 'Name': 'H2O'}
        assert _ci_get(d, 'temp') == 500
        assert _ci_get(d, 'name') == 'H2O'
        assert _ci_get(d, 'TEMP') == 500

    def test_ci_get_missing(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _ci_get
        d = {'Temp': 500}
        assert _ci_get(d, 'missing') is None

    def test_safe_float_basic(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'Temp': '500.5', 'N_mol': 1e18}
        assert _safe_float(d, 'Temp') == 500.5
        assert _safe_float(d, 'N_mol') == 1e18

    def test_safe_float_default(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'Temp': '500'}
        assert _safe_float(d, 'missing', default=42.0) == 42.0

    def test_safe_float_list_keys(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'temperature': 300.0}
        # First key misses, second hits
        assert _safe_float(d, ['Temp', 'temperature']) == 300.0

    def test_safe_float_invalid_value(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'Temp': 'not_a_number'}
        assert _safe_float(d, 'Temp', default=-1.0) == -1.0

    def test_safe_float_none_value(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'Temp': None}
        assert _safe_float(d, 'Temp', default=0.0) == 0.0

    def test_safe_float_case_insensitive(self):
        from iSLAT.Modules.DataTypes.MoleculeDict import _safe_float
        d = {'Temperature': 400.0}
        assert _safe_float(d, 'temperature', case_insensitive=True) == 400.0


# ======================================================================
# Grid-mismatch / canonical-grid fix tests
# ======================================================================

class _FakeMolecule:
    """Minimal molecule stub for testing MoleculeDict summing logic.

    Avoids the need for real HITRAN data.  ``get_flux`` returns a
    deterministic sine-shaped array on whatever ``wavelength_array`` is
    passed when ``interpolate_to_input=True``, or a fixed-size internal
    grid otherwise (size determined by ``native_n``).
    """
    def __init__(self, name, native_n=100, wavelength_range=(4.9, 5.1),
                 pixel_res=0.001):
        self.name = name
        self.molecule_name = name
        self.is_visible = True
        self._wavelength_range = wavelength_range
        self._model_pixel_res = pixel_res
        self._native_n = native_n
        self._rv_shift = 0.0
        self.rv_shift = 0.0          # public alias used by _get_flux_cache_key
        self._dirty_flags = {'intensity': False, 'spectrum': False, 'flux': False}
        self._flux_cache = {}

    @property
    def wavelength_range(self):
        return self._wavelength_range

    @wavelength_range.setter
    def wavelength_range(self, value):
        self._wavelength_range = value

    def get_parameter_hash(self, cache_type='full'):
        """Deterministic hash used by _compute_molecules_parameter_hash."""
        return hash((self.name, self._model_pixel_res, self._rv_shift,
                     self._wavelength_range))

    def get_flux(self, wavelength_array=None, return_wavelengths=False,
                 interpolate_to_input=False):
        if interpolate_to_input and wavelength_array is not None:
            wave = wavelength_array
        else:
            lam_min, lam_max = self._wavelength_range
            wave = np.linspace(lam_min, lam_max, self._native_n)
        flux = np.abs(np.sin(np.linspace(0, np.pi, len(wave)))) * 1e-15
        if return_wavelengths:
            return wave.copy(), flux.copy()
        return flux.copy()


class TestGetSummedFluxCanonicalGrid:
    """Tests for the canonical-grid fix in get_summed_flux.

    The fix ensures that even when _match_spectral_sampling is False,
    all molecules are resampled onto a single shared grid
    (np.arange(lam_min, lam_max, model_pixel_res)) so that molecules
    with different internal Nyquist grid sizes can be summed without a
    ValueError.
    """

    def _make_dict(self, pixel_res=0.001, native_sizes=(80, 150)):
        """MoleculeDict with two _FakeMolecule stubs of deliberately
        different native grid sizes."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(
            global_wavelength_range=(4.9, 5.1),
            global_model_pixel_res=pixel_res,
        )
        md['Narrow'] = _FakeMolecule('Narrow', native_n=native_sizes[0],
                                     wavelength_range=(4.9, 5.1),
                                     pixel_res=pixel_res)
        md['Wide'] = _FakeMolecule('Wide', native_n=native_sizes[1],
                                   wavelength_range=(4.9, 5.1),
                                   pixel_res=pixel_res)
        md._match_spectral_sampling = False
        return md

    # ------------------------------------------------------------------
    # canonical grid is built correctly
    # ------------------------------------------------------------------

    def test_canonical_grid_shape(self):
        """With the fix, get_summed_flux returns a grid whose length equals
        np.arange(lam_min, lam_max, pixel_res)."""
        md = self._make_dict(pixel_res=0.001)
        wave_obs = np.linspace(4.9, 5.1, 200)
        lam_min, lam_max = md._global_wavelength_range
        expected_len = len(np.arange(lam_min, lam_max, md._global_model_pixel_res))

        wave_out, flux_out = md.get_summed_flux(wave_obs, visible_only=False)

        assert len(wave_out) == expected_len, (
            f"Expected {expected_len} grid points, got {len(wave_out)}"
        )
        assert len(flux_out) == expected_len

    def test_no_grid_mismatch_different_native_sizes(self):
        """Molecules with very different native grid sizes (80 vs 150 pts)
        must produce no 'Grid size mismatch' warning."""
        import io, contextlib
        md = self._make_dict(pixel_res=0.001, native_sizes=(80, 150))
        wave_obs = np.linspace(4.9, 5.1, 200)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wave_out, flux_out = md.get_summed_flux(wave_obs, visible_only=False)

        assert 'Grid size mismatch' not in buf.getvalue()
        assert len(wave_out) > 0
        assert len(flux_out) == len(wave_out)

    def test_no_grid_mismatch_fine_pixel_res(self):
        """Fine pixel_res (smaller than some molecules' native step) must
        not trigger a grid size mismatch."""
        import io, contextlib
        md = self._make_dict(pixel_res=0.0003, native_sizes=(30, 200))
        wave_obs = np.linspace(4.9, 5.1, 300)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wave_out, flux_out = md.get_summed_flux(wave_obs, visible_only=False)

        assert 'Grid size mismatch' not in buf.getvalue()
        assert len(wave_out) > 0

    def test_no_grid_mismatch_coarse_pixel_res(self):
        """Coarse pixel_res must also work without mismatch errors."""
        import io, contextlib
        md = self._make_dict(pixel_res=0.05, native_sizes=(5, 300))
        wave_obs = np.linspace(4.9, 5.1, 50)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wave_out, flux_out = md.get_summed_flux(wave_obs, visible_only=False)

        assert 'Grid size mismatch' not in buf.getvalue()
        assert len(wave_out) > 0

    def test_both_molecules_contribute_to_sum(self):
        """Summed flux must be strictly positive, confirming neither
        molecule was silently dropped."""
        md = self._make_dict(pixel_res=0.001, native_sizes=(80, 150))
        wave_obs = np.linspace(4.9, 5.1, 200)
        _, flux_sum = md.get_summed_flux(wave_obs, visible_only=False)

        assert np.any(flux_sum > 0), "Expected non-zero summed flux"

    def test_sum_exceeds_single_molecule(self):
        """Summed flux should be greater than either molecule alone,
        confirming both contribute."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = self._make_dict(pixel_res=0.001, native_sizes=(80, 150))
        wave_obs = np.linspace(4.9, 5.1, 200)

        _, flux_sum = md.get_summed_flux(wave_obs, visible_only=False)

        canonical = np.arange(4.9, 5.1, md._global_model_pixel_res)
        _, flux_narrow = md['Narrow'].get_flux(
            wavelength_array=canonical, return_wavelengths=True, interpolate_to_input=True)
        _, flux_wide = md['Wide'].get_flux(
            wavelength_array=canonical, return_wavelengths=True, interpolate_to_input=True)

        # Sum must be >= each individual (both are non-negative)
        assert np.all(flux_sum >= flux_narrow - 1e-30)
        assert np.all(flux_sum >= flux_wide - 1e-30)

    def test_output_grid_matches_canonical_arange(self):
        """The output wavelength grid must be exactly np.arange(lam_min,
        lam_max, pixel_res)."""
        md = self._make_dict(pixel_res=0.002)
        wave_obs = np.linspace(4.9, 5.1, 100)
        wave_out, _ = md.get_summed_flux(wave_obs, visible_only=False)

        expected = np.arange(4.9, 5.1, 0.002)
        np.testing.assert_allclose(wave_out, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # changing pixel_res clears summed flux cache
    # ------------------------------------------------------------------

    def test_pixel_res_change_clears_cache(self):
        """After changing global_model_pixel_res the summed flux cache
        must be empty so stale results are never served."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(
            global_wavelength_range=(4.9, 5.1),
            global_model_pixel_res=0.001,
        )
        # Manually insert a fake cache entry
        md._summed_flux_cache['dummy_key'] = (np.array([1.0]), np.array([1.0]), 42)
        assert len(md._summed_flux_cache) == 1

        md.global_model_pixel_res = 0.0007
        assert len(md._summed_flux_cache) == 0, (
            "Cache should have been cleared after pixel_res change"
        )

    def test_pixel_res_change_produces_new_grid_size(self):
        """After a pixel_res change the next call returns a grid with the
        correct new length."""
        md = self._make_dict(pixel_res=0.002)
        wave_obs = np.linspace(4.9, 5.1, 100)
        wave1, _ = md.get_summed_flux(wave_obs, visible_only=False)

        # Update pixel_res on the dict and on each stub
        md._global_model_pixel_res = 0.001
        for mol in md.values():
            mol._model_pixel_res = 0.001

        md._summed_flux_cache.clear()
        wave_obs2 = np.linspace(4.9, 5.1, 200)
        wave2, _ = md.get_summed_flux(wave_obs2, visible_only=False)

        expected_len1 = len(np.arange(4.9, 5.1, 0.002))
        expected_len2 = len(np.arange(4.9, 5.1, 0.001))
        assert len(wave1) == expected_len1
        assert len(wave2) == expected_len2
        assert len(wave2) > len(wave1)  # finer grid → more points

    # ------------------------------------------------------------------
    # global_model_pixel_res setter
    # ------------------------------------------------------------------

    def test_global_pixel_res_setter_clears_cache(self):
        """Setting global_model_pixel_res must clear the summed flux cache."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(global_wavelength_range=(4.9, 5.1),
                          global_model_pixel_res=0.001)
        md._summed_flux_cache['key'] = (np.ones(3), np.ones(3), 0)
        md.global_model_pixel_res = 0.002
        assert len(md._summed_flux_cache) == 0

    def test_global_pixel_res_setter_fires_callback(self):
        """The global parameter-change callback must fire exactly once."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(global_wavelength_range=(4.9, 5.1),
                          global_model_pixel_res=0.001)
        calls = []
        md.add_global_parameter_change_callback(
            lambda p, o, n: calls.append((p, o, n))
        )
        md.global_model_pixel_res = 0.0007

        pixel_calls = [c for c in calls if c[0] == 'model_pixel_res']
        assert len(pixel_calls) == 1
        assert pixel_calls[0][2] == pytest.approx(0.0007)

    # ------------------------------------------------------------------
    # defensive spectres resample (fallback safety net)
    # ------------------------------------------------------------------

    def test_defensive_resample_when_mol_returns_wrong_size(self):
        """If a molecule's get_flux returns a different-sized grid after the
        canonical-grid logic runs, the accumulation must still succeed via
        the defensive spectres resample (no exception, no dropped molecule)."""
        import io, contextlib

        class _BadSizeMolecule(_FakeMolecule):
            """Always returns a hardcoded 77-point grid regardless of input."""
            def get_flux(self, wavelength_array=None, return_wavelengths=False,
                         interpolate_to_input=False):
                bad_wave = np.linspace(4.9, 5.1, 77)
                bad_flux = np.ones(77) * 1e-16
                if return_wavelengths:
                    return bad_wave, bad_flux
                return bad_flux

        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(global_wavelength_range=(4.9, 5.1),
                          global_model_pixel_res=0.001)
        md['Normal'] = _FakeMolecule('Normal', native_n=100,
                                     wavelength_range=(4.9, 5.1))
        md['BadSize'] = _BadSizeMolecule('BadSize', wavelength_range=(4.9, 5.1))
        md._match_spectral_sampling = False

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wave_out, flux_out = md.get_summed_flux(
                np.linspace(4.9, 5.1, 200), visible_only=False)

        # Must not raise, must not print the mismatch warning
        assert 'Grid size mismatch' not in buf.getvalue()
        assert len(flux_out) == len(wave_out)
        assert len(wave_out) > 0

    # ------------------------------------------------------------------
    # matched spectral sampling path is unaffected
    # ------------------------------------------------------------------

    def test_matched_sampling_uses_rv_corrected_grid(self):
        """With _match_spectral_sampling=True the method must delegate to
        get_matched_sampling_wavelengths and use its rest-frame grid."""
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        md = MoleculeDict(global_wavelength_range=(4.9, 5.1),
                          global_model_pixel_res=0.001,
                          global_stellar_rv=0.0)
        md['M1'] = _FakeMolecule('M1', native_n=80, wavelength_range=(4.9, 5.1))
        md['M2'] = _FakeMolecule('M2', native_n=150, wavelength_range=(4.9, 5.1))
        md._match_spectral_sampling = True

        wave_obs = np.linspace(4.9, 5.1, 200)
        wave_out, flux_out = md.get_summed_flux(wave_obs, visible_only=False)

        assert len(wave_out) > 0
        assert len(flux_out) == len(wave_out)
        assert np.all(np.isfinite(flux_out))


# ======================================================================
# load_molecules strategy
# ======================================================================

def _save_row(name):
    """One row of a saved molecule list; the line file is only read on first use."""
    return {"Molecule Name": name, "File Path": "not_read_at_load.par", "Molecule Label": name,
            "Temp": 600.0, "Rad": 0.5, "N_Mol": 1e17, "Color": "#ff0000", "Vis": True,
            "Dist": 140.0, "StellarRV": 0.0, "FWHM": 130.0, "Broad": 2.5, "RV Shift": 0.0}


@pytest.fixture
def empty_md():
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    md = MoleculeDict(global_wavelength_range=(4.9, 5.2))
    yield md
    Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


@pytest.fixture
def parallel_loads(empty_md, monkeypatch):
    """Record calls to the process-pool loader (and load sequentially instead)."""
    calls = []

    def recording(molecules_data, initial_molecule_parameters, max_workers=None):
        calls.append([row["Molecule Name"] for row in molecules_data])
        return empty_md._load_molecules_sequential(molecules_data, initial_molecule_parameters)

    monkeypatch.setattr(empty_md, "_load_molecules_parallel", recording)
    return calls


class TestLoadMoleculesStrategy:
    """Molecule construction is lazy, so "auto" never starts a process pool by molecule count alone.

    A pool cost ~1 s at startup for 12 molecules against ~2 ms sequentially.
    """

    NAMES = [f"M{i:02d}" for i in range(12)]

    def test_auto_loads_many_molecules_sequentially_in_file_order(self, empty_md, parallel_loads):
        results = empty_md.load_molecules([_save_row(n) for n in self.NAMES], {})
        assert parallel_loads == []
        assert results["success"] == len(self.NAMES)
        assert list(empty_md.keys()) == self.NAMES

    def test_molecules_are_built_without_reading_line_files(self, empty_md, parallel_loads):
        empty_md.load_molecules([_save_row(n) for n in self.NAMES], {})
        assert all(empty_md[n].lines is None for n in self.NAMES)

    def test_forced_multiprocessing_still_uses_the_pool_when_worthwhile(
            self, empty_md, parallel_loads, monkeypatch):
        monkeypatch.setattr(empty_md, "_should_use_multiprocessing", lambda data: True)
        empty_md.load_molecules([_save_row(n) for n in self.NAMES], {}, force_multiprocessing=True)
        assert parallel_loads == [self.NAMES]

    def test_explicit_parallel_strategy_is_kept(self, empty_md, parallel_loads):
        empty_md.load_molecules([_save_row(n) for n in self.NAMES[:2]], {}, strategy="parallel")
        assert parallel_loads == [self.NAMES[:2]]
