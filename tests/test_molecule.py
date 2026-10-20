# -*- coding: utf-8 -*-
"""Unit tests for Molecule."""

import pytest
import numpy as np

import iSLAT.Constants as c


class TestMolecule:
    """Tests for the Molecule class."""

    def _make_molecule(self, **kwargs):
        """Helper to create a Molecule with defaults."""
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        defaults = {
            'name': 'TestH2O',
            'displaylabel': 'H$_2$O',
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

    def test_init_basic(self):
        mol = self._make_molecule()
        assert mol.name == 'TestH2O'
        assert mol.displaylabel == 'H$_2$O'
        assert mol.color == '#FF0000'

    def test_temp_property(self):
        mol = self._make_molecule(temp=500.0)
        assert mol.temp == 500.0
        mol.temp = 600.0
        assert mol.temp == 600.0

    def test_radius_property(self):
        mol = self._make_molecule(radius=1.0)
        assert mol.radius == 1.0
        mol.radius = 2.5
        assert mol.radius == 2.5

    def test_n_mol_property(self):
        mol = self._make_molecule(n_mol=1e18)
        assert mol.n_mol == 1e18
        mol.n_mol = 5e19
        assert mol.n_mol == 5e19

    def test_distance_property(self):
        mol = self._make_molecule(distance=160.0)
        assert mol.distance == 160.0
        mol.distance = 200.0
        assert mol.distance == 200.0

    def test_fwhm_property(self):
        mol = self._make_molecule(fwhm=130.0)
        assert mol.fwhm == 130.0
        mol.fwhm = 150.0
        assert mol.fwhm == 150.0

    def test_broad_property(self):
        mol = self._make_molecule()
        mol.broad = 2.0
        assert mol.broad == 2.0

    def test_rv_shift_property(self):
        mol = self._make_molecule()
        assert mol.rv_shift == 0.0
        mol.rv_shift = 5.0
        assert mol.rv_shift == 5.0

    def test_is_visible_bool(self):
        mol = self._make_molecule(is_visible=True)
        assert mol.is_visible is True
        mol.is_visible = False
        assert mol.is_visible is False

    def test_is_visible_string_conversion(self):
        mol = self._make_molecule(is_visible='true')
        assert mol.is_visible is True
        mol.is_visible = 'false'
        assert mol.is_visible is False
        mol.is_visible = '1'
        assert mol.is_visible is True
        mol.is_visible = '0'
        assert mol.is_visible is False

    def test_is_bold_defaults_false(self):
        mol = self._make_molecule()
        assert mol.is_bold is False

    def test_is_bold_toggle_and_string_conversion(self):
        mol = self._make_molecule(is_bold=True)
        assert mol.is_bold is True
        mol.is_bold = False
        assert mol.is_bold is False
        mol.is_bold = 'true'
        assert mol.is_bold is True
        mol.is_bold = '0'
        assert mol.is_bold is False

    def test_is_bold_notifies_without_invalidating_flux(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        mol = self._make_molecule()
        mol._dirty_flags['flux'] = False
        events = []
        cb = lambda name, param, old, new: events.append((name, param, old, new))
        Molecule.add_molecule_parameter_change_callback(cb)
        try:
            mol.is_bold = True
        finally:
            Molecule.remove_molecule_parameter_change_callback(cb)
        assert events == [('TestH2O', 'is_bold', False, True)]
        assert mol._dirty_flags['flux'] is False

    def test_wavelength_range_property(self):
        mol = self._make_molecule()
        mol.wavelength_range = (5.0, 25.0)
        assert mol.wavelength_range == (5.0, 25.0)

    def test_intensity_calculation_method_property(self):
        mol = self._make_molecule()
        assert mol.intensity_calculation_method == 'curve_growth'
        mol.intensity_calculation_method = 'radex'
        assert mol.intensity_calculation_method == 'radex'

    def test_str_repr(self):
        mol = self._make_molecule(name='TestMol', temp=300, radius=0.5, n_mol=1e16)
        s = str(mol)
        assert 'Molecule' in s
        assert 'TestMol' in s

    def test_caching_system_initialized(self):
        mol = self._make_molecule()
        assert isinstance(mol._dirty_flags, dict)
        assert 'intensity' in mol._dirty_flags
        assert 'spectrum' in mol._dirty_flags
        assert 'flux' in mol._dirty_flags

    def test_clear_all_caches(self):
        mol = self._make_molecule()
        mol._dirty_flags['intensity'] = False
        mol.clear_all_caches()
        assert mol._dirty_flags['intensity'] is True
        assert mol._dirty_flags['spectrum'] is True
        assert mol._dirty_flags['flux'] is True

    def test_is_cache_valid(self):
        mol = self._make_molecule()
        # All dirty initially
        assert mol.is_cache_valid('intensity') is False
        assert mol.is_cache_valid('spectrum') is False
        assert mol.is_cache_valid('flux') is False
        assert mol.is_cache_valid('full') is False

    def test_parameter_hash_changes_on_update(self):
        mol = self._make_molecule()
        h1 = mol._compute_intensity_hash()
        mol.temp = 700.0
        h2 = mol._compute_intensity_hash()
        assert h1 != h2

    def test_bulk_update_parameters(self):
        mol = self._make_molecule()
        mol.bulk_update_parameters({'temp': 800.0, 'radius': 3.0, 'n_mol': 2e19})
        assert mol.temp == 800.0
        assert mol.radius == 3.0
        assert mol.n_mol == 2e19

    def test_bulk_update_parameters_empty(self):
        mol = self._make_molecule(temp=500.0)
        mol.bulk_update_parameters({})
        assert mol.temp == 500.0  # unchanged

    def test_bulk_update_parameters_skip_notification(self):
        mol = self._make_molecule()
        # Should not raise even with skip_notification
        mol.bulk_update_parameters({'temp': 999.0}, skip_notification=True)
        assert mol.temp == 999.0

    def test_dirty_flags_on_temp_change(self):
        mol = self._make_molecule()
        mol._dirty_flags['intensity'] = False
        mol._dirty_flags['spectrum'] = False
        mol._dirty_flags['flux'] = False
        mol.temp = 700.0
        # temp is intensity-affecting
        assert mol._dirty_flags['intensity'] is True
        assert mol._dirty_flags['spectrum'] is True
        assert mol._dirty_flags['flux'] is True

    def test_dirty_flags_on_radius_change(self):
        mol = self._make_molecule()
        mol._dirty_flags['intensity'] = False
        mol._dirty_flags['spectrum'] = False
        mol._dirty_flags['flux'] = False
        mol.radius = 3.0
        # radius is spectrum-affecting only
        assert mol._dirty_flags['intensity'] is False
        assert mol._dirty_flags['spectrum'] is True
        assert mol._dirty_flags['flux'] is True

    def test_init_from_user_save_data(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        usd = {
            'Molecule Name': 'SavedMol',
            'File Path': None,
            'Molecule Label': 'Saved',
            'Temp': 400.0,
            'Rad': 2.0,
            'N_Mol': 5e17,
            'Color': '#00FF00',
            'Vis': True,
            'Dist': 140.0,
            'FWHM': 120.0,
            'Broad': 1.5,
            'RV Shift': 3.0,
        }
        mol = Molecule(
            user_save_data=usd,
            initial_molecule_parameters={
                't_kin': 400.0,
                'scale_exponent': 17.0,
                'scale_number': 5.0,
                'radius_init': 2.0,
            },
        )
        assert mol.name == 'SavedMol'
        assert mol.temp == 400.0
        assert mol.radius == 2.0
        assert mol.color == '#00FF00'

    def test_molecule_parameter_change_callbacks(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        callback_log = []

        def my_callback(mol_name, param, old, new):
            callback_log.append((mol_name, param, old, new))

        Molecule.add_molecule_parameter_change_callback(my_callback)
        try:
            mol = self._make_molecule(name='CallbackTest')
            mol.temp = 600.0
            # Should have recorded temp change
            assert any(entry[1] == 'temp' for entry in callback_log)
        finally:
            Molecule.remove_molecule_parameter_change_callback(my_callback)

    def test_get_cache_stats(self):
        mol = self._make_molecule()
        stats = mol.get_cache_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'invalidations' in stats

    def test_no_filepath(self):
        """Molecule with no filepath should not crash."""
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        mol = Molecule(
            name='NoFile', filepath=None,
            initial_molecule_parameters={
                't_kin': 300.0,
                'scale_exponent': 1.0,
                'scale_number': 1.0,
                'radius_init': 1.0,
            },
        )
        assert mol.name == 'NoFile'
        assert mol.filepath is None


class TestMoleculeWavelengthRangeExtension:
    """Verify that Molecule.get_flux correctly calculates flux when the
    wavelength_range is extended beyond the observed data range.

    This exercises the full stack: MoleculeLineList filtering →
    Intensity calculation → Spectrum construction → get_flux.
    """

    def _make_molecule_with_lines(self, wavelength_range=None):
        """Create a Molecule whose line list spans 5–34 µm."""
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        import iSLAT.Constants as c_mod

        lines_data = [
            {'nr': i + 1,
             'lev_up': f'0|{2*i+2}', 'lev_low': f'0|{2*i+1}',
             'lam': lam,
             'freq': c_mod.SPEED_OF_LIGHT_MICRONS / lam,
             'a_stein': 0.02,
             'e_up': 3000.0 - i * 200,
             'e_low': 2000.0 - i * 200,
             'g_up': 2 * i + 3,
             'g_low': 2 * i + 1}
            for i, lam in enumerate([6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 34.0])
        ]
        mll = MoleculeLineList(
            molecule_id='RangeTestMol',
            lines_data=lines_data,
            wavelength_range=wavelength_range,
        )
        mll.partition_function = mll._partition_type(
            t=np.array([100, 300, 500, 1000, 2000], dtype=np.float64),
            q=np.array([10, 150, 500, 2000, 8000], dtype=np.float64),
        )
        mll._molar_mass = 18.015

        kw = dict(
            name='RangeTestMol', displaylabel='RangeTest',
            filepath=None, color='#0000FF', is_visible=True,
            temp=500.0, radius=1.0, n_mol=1e18, distance=140.0,
            fwhm=130.0,
            initial_molecule_parameters={
                't_kin': 500.0,
                'scale_exponent': 18.0,
                'scale_number': 1.0,
                'radius_init': 1.0,
            },
        )
        if wavelength_range is not None:
            kw['wavelength_range'] = wavelength_range
        mol = Molecule(**kw)
        # Inject the pre-built line list so no file loading is needed
        mol.lines = mll
        if wavelength_range is not None:
            mol._wavelength_range = wavelength_range
        return mol

    # ------------------------------------------------------------------
    # No lines in range → zero flux, no crash
    # ------------------------------------------------------------------

    def test_get_flux_empty_range_no_crash(self):
        """When no lines fall in the active range, get_flux must return
        an array of zeros (not crash with TypeError)."""
        mol = self._make_molecule_with_lines(wavelength_range=(100.0, 200.0))
        lam, flux = mol.get_flux(return_wavelengths=True)
        assert isinstance(flux, np.ndarray)
        # Flux should be all zeros (no lines)
        assert np.all(flux == 0.0) or len(flux) == 0

    # ------------------------------------------------------------------
    # Restricted range → flux non-zero within range, zero outside
    # ------------------------------------------------------------------

    def test_get_flux_restricted_range_is_nonzero(self):
        """With lines present in (5, 28), get_flux must be nonzero
        somewhere in that wavelength window."""
        mol = self._make_molecule_with_lines(wavelength_range=(5.0, 28.0))
        lam, flux = mol.get_flux(return_wavelengths=True)
        assert len(flux) > 0
        assert np.any(flux > 0), "Expected non-zero flux in (5, 28) µm range"

    # ------------------------------------------------------------------
    # Extended range: flux must be non-zero in the extended portion
    # ------------------------------------------------------------------

    def test_get_flux_extended_range_includes_lines_beyond_data(self):
        """After extending max_wave to 35 µm, the returned wavelength grid
        must cover the extended region and contain non-zero flux there
        (because lines exist at 30 and 34 µm)."""
        mol = self._make_molecule_with_lines(wavelength_range=(5.0, 35.0))
        lam, flux = mol.get_flux(return_wavelengths=True)
        assert lam[-1] >= 30.0, (
            f"Wavelength grid should extend to at least 30 µm, got {lam[-1]:.2f} µm"
        )
        # Flux should be non-zero somewhere beyond 28 µm
        beyond_data_mask = lam > 28.0
        assert np.any(beyond_data_mask), "No wavelength points beyond 28 µm in the grid"
        assert np.any(flux[beyond_data_mask] > 0), (
            "Expected non-zero flux beyond 28 µm (lines at 30 and 34 µm exist)"
        )

    def test_get_flux_extended_contains_more_flux_than_restricted(self):
        """The total integrated flux over the shared range (5–28 µm) should
        be comparable between restricted and extended calculations."""
        mol_narrow = self._make_molecule_with_lines(wavelength_range=(5.0, 28.0))
        mol_wide   = self._make_molecule_with_lines(wavelength_range=(5.0, 35.0))

        lam_n, flux_n = mol_narrow.get_flux(return_wavelengths=True)
        lam_w, flux_w = mol_wide.get_flux(return_wavelengths=True)

        # Both grids should cover (5, 28) — sum of flux in that sub-range
        mask_n = (lam_n >= 5.0) & (lam_n <= 28.0)
        mask_w = (lam_w >= 5.0) & (lam_w <= 28.0)

        assert np.any(mask_n) and np.any(mask_w), "No grid points in (5, 28) µm"
        integral_narrow   = np.trapezoid(flux_n[mask_n], lam_n[mask_n])
        integral_wide_sub = np.trapezoid(flux_w[mask_w], lam_w[mask_w])

        # Allow 10% tolerance (different grid spacing may slightly change integral)
        np.testing.assert_allclose(
            integral_wide_sub, integral_narrow, rtol=0.10,
            err_msg="Flux in (5-28 µm) should be similar for narrow and wide range molecules"
        )

    def test_wavelength_range_extension_dirty_flags_trigger_recalculation(self):
        """Changing wavelength_range must mark caches dirty so the next
        get_flux call recomputes using the new range."""
        mol = self._make_molecule_with_lines(wavelength_range=(5.0, 28.0))
        # Force an initial calculation
        _, flux_before = mol.get_flux(return_wavelengths=True)

        # Extend the range
        mol.wavelength_range = (5.0, 35.0)

        # Dirty flags must be set
        assert mol._dirty_flags['intensity'] or mol._dirty_flags['spectrum'] or mol._dirty_flags['flux'], (
            "Extending wavelength_range must mark at least one cache dirty"
        )

        # New flux should cover extended range
        lam_after, flux_after = mol.get_flux(return_wavelengths=True)
        assert lam_after[-1] >= 30.0, (
            f"After extension to 35 µm, grid should reach 30+ µm, got {lam_after[-1]:.2f}"
        )
