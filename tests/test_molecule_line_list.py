# -*- coding: utf-8 -*-
"""Unit tests for MoleculeLineList."""

import pytest
import numpy as np


class TestMoleculeLineList:
    """Tests for MoleculeLineList."""

    def test_init_from_data(self, sample_lines_data):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        mll = MoleculeLineList(molecule_id='H2O', lines_data=sample_lines_data)
        assert mll.molecule_id == 'H2O'
        assert mll.num_lines == 5
        assert len(mll) == 5

    def test_init_empty(self):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        mll = MoleculeLineList(molecule_id='empty')
        assert mll.molecule_id == 'empty'
        assert mll.num_lines == 0

    def test_init_empty_lines_data(self):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        mll = MoleculeLineList(molecule_id='empty', lines_data=[])
        assert mll.num_lines == 0

    def test_get_wavelengths(self, molecule_line_list):
        wavelengths = molecule_line_list.get_wavelengths()
        assert isinstance(wavelengths, np.ndarray)
        assert len(wavelengths) == 5
        np.testing.assert_allclose(sorted(wavelengths), [6.5, 12.407, 14.95, 17.221, 22.5])

    def test_get_frequencies(self, molecule_line_list):
        freqs = molecule_line_list.get_frequencies()
        assert isinstance(freqs, np.ndarray)
        assert len(freqs) == 5
        assert all(f > 0 for f in freqs)

    def test_get_einstein_coefficients(self, molecule_line_list):
        a_stein = molecule_line_list.get_einstein_coefficients()
        assert isinstance(a_stein, np.ndarray)
        assert len(a_stein) == 5
        assert all(a > 0 for a in a_stein)

    def test_get_upper_energies(self, molecule_line_list):
        e_up = molecule_line_list.get_upper_energies()
        assert isinstance(e_up, np.ndarray)
        assert len(e_up) == 5

    def test_get_lower_energies(self, molecule_line_list):
        e_low = molecule_line_list.get_lower_energies()
        assert isinstance(e_low, np.ndarray)
        assert len(e_low) == 5

    def test_get_upper_weights(self, molecule_line_list):
        g_up = molecule_line_list.get_upper_weights()
        assert isinstance(g_up, np.ndarray)
        assert len(g_up) == 5

    def test_get_lower_weights(self, molecule_line_list):
        g_low = molecule_line_list.get_lower_weights()
        assert isinstance(g_low, np.ndarray)
        assert len(g_low) == 5

    def test_wavelength_range_filtering(self, molecule_line_list_with_range):
        """Lines outside range (10, 20) should be excluded."""
        mll = molecule_line_list_with_range
        n = mll.num_lines
        # Lines at 12.407, 14.95, 17.221 are in [10, 20]; 6.5 and 22.5 are out
        assert n == 3

        wavelengths = mll.get_wavelengths()
        assert len(wavelengths) == 3
        assert all(10.0 <= w <= 20.0 for w in wavelengths)

    def test_wavelength_range_change_invalidates_cache(self, molecule_line_list):
        mll = molecule_line_list
        _ = mll.get_wavelengths()  # populate cache
        assert mll.num_lines == 5

        mll.wavelength_range = (10.0, 20.0)
        assert mll.num_lines == 3

        mll.wavelength_range = None
        assert mll.num_lines == 5

    def test_get_lines_in_range(self, molecule_line_list):
        lines = molecule_line_list.get_lines_in_range(10.0, 15.0)
        assert len(lines) == 2  # 12.407 and 14.95

    def test_get_lines_in_range_values_and_types(self, molecule_line_list):
        """Lines carry Python int / str / float values, as the per-field conversion gave them."""
        raw = molecule_line_list._get_active_raw_data()
        expected = [
            dict(nr=int(r['nr']), lev_up=r['lev_up'].decode(), lev_low=r['lev_low'].decode(),
                 lam=float(r['lam']), freq=float(r['freq']), a_stein=float(r['a_stein']),
                 e_up=float(r['e_up']), e_low=float(r['e_low']),
                 g_up=int(r['g_up']), g_low=int(r['g_low']))
            for r in raw
        ]
        lines = molecule_line_list.get_lines_in_range(0.0, 100.0)
        assert len(lines) == len(expected)
        for line, want in zip(lines, expected):
            assert line.molecule_id == molecule_line_list.molecule_id
            for field, value in want.items():
                got = getattr(line, field)
                assert got == value and type(got) is type(value), field

    def test_get_lines_in_range_empty(self, molecule_line_list):
        lines = molecule_line_list.get_lines_in_range(30.0, 40.0)
        assert len(lines) == 0

    def test_lines_as_namedtuple(self, molecule_line_list):
        nt = molecule_line_list.lines_as_namedtuple
        assert hasattr(nt, 'lam')
        assert hasattr(nt, 'freq')
        assert hasattr(nt, 'a_stein')
        assert hasattr(nt, 'e_up')
        assert hasattr(nt, 'e_low')
        assert hasattr(nt, 'g_up')
        assert hasattr(nt, 'g_low')
        assert hasattr(nt, 'nr')
        assert hasattr(nt, 'lev_up')
        assert hasattr(nt, 'lev_low')
        assert len(nt.lam) == 5

    def test_name_property(self, molecule_line_list):
        assert molecule_line_list.name == 'H2O'

    def test_fname_property(self, molecule_line_list):
        assert molecule_line_list.fname is None
        molecule_line_list.fname = 'test.par'
        assert molecule_line_list.fname == 'test.par'

    def test_get_ndarray(self, molecule_line_list):
        arr = molecule_line_list.get_ndarray()
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 5  # 5 lines
        assert arr.shape[1] == 10  # 10 fields per line

    def test_get_pandas_table(self, molecule_line_list):
        import pandas as pd
        df = molecule_line_list.get_pandas_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'lam' in df.columns
        assert 'freq' in df.columns

    def test_get_ndarray_of_attribute(self, molecule_line_list):
        lam = molecule_line_list.get_ndarray_of_attribute('lam')
        assert isinstance(lam, np.ndarray)
        assert len(lam) == 5
        np.testing.assert_allclose(sorted(lam), [6.5, 12.407, 14.95, 17.221, 22.5])

    def test_caching_behavior(self, molecule_line_list):
        """Second call should return cached result."""
        w1 = molecule_line_list.get_wavelengths()
        w2 = molecule_line_list.get_wavelengths()
        # Should be the same array object due to caching
        np.testing.assert_array_equal(w1, w2)

    def test_invalidate_caches(self, molecule_line_list):
        """After invalidation, cache should be rebuilt."""
        _ = molecule_line_list.get_wavelengths()
        molecule_line_list._invalidate_caches()
        assert molecule_line_list._wavelengths_cache is None
        # Should rebuild on next access
        w = molecule_line_list.get_wavelengths()
        assert len(w) == 5

    def test_single_line(self):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        mll = MoleculeLineList(molecule_id='single', lines_data=[{
            'nr': 1, 'lev_up': '0|2', 'lev_low': '0|1',
            'lam': 15.0, 'freq': 2e13, 'a_stein': 0.01,
            'e_up': 2000, 'e_low': 1000, 'g_up': 5, 'g_low': 3,
        }])
        assert mll.num_lines == 1
        assert mll.get_wavelengths()[0] == 15.0

    def test_duplicate_wavelengths(self):
        """MoleculeLineList should handle duplicate wavelengths."""
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        lines_data = [
            {'nr': 1, 'lev_up': '0|2', 'lev_low': '0|1',
             'lam': 15.0, 'freq': 2e13, 'a_stein': 0.01,
             'e_up': 2000, 'e_low': 1000, 'g_up': 5, 'g_low': 3},
            {'nr': 2, 'lev_up': '0|4', 'lev_low': '0|3',
             'lam': 15.0, 'freq': 2e13, 'a_stein': 0.02,
             'e_up': 3000, 'e_low': 2000, 'g_up': 9, 'g_low': 7},
        ]
        mll = MoleculeLineList(molecule_id='dup', lines_data=lines_data)
        assert mll.num_lines == 2
        wavelengths = mll.get_wavelengths()
        assert np.all(wavelengths == 15.0)


class TestWavelengthRangeExtension:
    """Tests ensuring that extending the wavelength range beyond the
    original data range correctly includes/excludes lines, and that
    the lines_as_namedtuple always returns numpy arrays (never bare lists).
    """

    def _make_mll(self):
        """Line list with lines spread across 5 – 35 µm."""
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        import iSLAT.Constants as c_mod
        lines_data = [
            # Lines within a typical 'data' range (5–28 µm)
            {'nr': 1, 'lev_up': '0|2', 'lev_low': '0|1',
             'lam': 6.0,  'freq': c_mod.SPEED_OF_LIGHT_MICRONS / 6.0,
             'a_stein': 0.02, 'e_up': 5000.0, 'e_low': 3000.0, 'g_up': 5, 'g_low': 3},
            {'nr': 2, 'lev_up': '0|4', 'lev_low': '0|3',
             'lam': 15.0, 'freq': c_mod.SPEED_OF_LIGHT_MICRONS / 15.0,
             'a_stein': 0.01, 'e_up': 3000.0, 'e_low': 2000.0, 'g_up': 9, 'g_low': 7},
            {'nr': 3, 'lev_up': '0|6', 'lev_low': '0|5',
             'lam': 25.0, 'freq': c_mod.SPEED_OF_LIGHT_MICRONS / 25.0,
             'a_stein': 0.005, 'e_up': 1500.0, 'e_low': 1000.0, 'g_up': 13, 'g_low': 11},
            # Lines BEYOND a typical 'data' max of 28 µm
            {'nr': 4, 'lev_up': '0|8', 'lev_low': '0|7',
             'lam': 30.0, 'freq': c_mod.SPEED_OF_LIGHT_MICRONS / 30.0,
             'a_stein': 0.003, 'e_up': 1200.0, 'e_low': 800.0, 'g_up': 17, 'g_low': 15},
            {'nr': 5, 'lev_up': '0|10', 'lev_low': '0|9',
             'lam': 34.0, 'freq': c_mod.SPEED_OF_LIGHT_MICRONS / 34.0,
             'a_stein': 0.001, 'e_up': 900.0, 'e_low': 600.0, 'g_up': 21, 'g_low': 19},
        ]
        mll = MoleculeLineList(molecule_id='ExtTest', lines_data=lines_data)
        mll.partition_function = mll._partition_type(
            t=np.array([100, 300, 500, 1000, 2000], dtype=np.float64),
            q=np.array([10, 150, 500, 2000, 8000], dtype=np.float64),
        )
        mll._molar_mass = 18.015
        return mll

    # ------------------------------------------------------------------
    # lines_as_namedtuple: always numpy arrays
    # ------------------------------------------------------------------

    def test_empty_range_namedtuple_fields_are_numpy_arrays(self):
        """When the wavelength_range filters out ALL lines,
        lines_as_namedtuple fields must be numpy arrays, not Python lists.
        This prevents 'list ** int' errors downstream."""
        mll = self._make_mll()
        # Range that contains no lines
        mll.wavelength_range = (100.0, 200.0)
        nt = mll.lines_as_namedtuple
        assert isinstance(nt.freq,    np.ndarray), "freq must be ndarray when empty"
        assert isinstance(nt.a_stein, np.ndarray), "a_stein must be ndarray when empty"
        assert isinstance(nt.e_up,    np.ndarray), "e_up must be ndarray when empty"
        assert isinstance(nt.e_low,   np.ndarray), "e_low must be ndarray when empty"
        assert isinstance(nt.g_up,    np.ndarray), "g_up must be ndarray when empty"
        assert isinstance(nt.lam,     np.ndarray), "lam must be ndarray when empty"
        assert len(nt.freq) == 0

    def test_empty_range_namedtuple_supports_pow(self):
        """Empty lines_as_namedtuple.freq ** 3 must not raise TypeError."""
        mll = self._make_mll()
        mll.wavelength_range = (100.0, 200.0)
        nt = mll.lines_as_namedtuple
        # This should not raise 'unsupported operand type for **'
        result = nt.freq ** 3
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    # ------------------------------------------------------------------
    # Range extension: lines beyond old max are included
    # ------------------------------------------------------------------

    def test_restricting_range_excludes_outer_lines(self):
        """Lines at 30 and 34 µm must NOT appear when range = (5, 28)."""
        mll = self._make_mll()
        mll.wavelength_range = (5.0, 28.0)
        wavs = mll.get_wavelengths()
        assert len(wavs) == 3, f"Expected 3 lines in (5, 28), got {len(wavs)}"
        assert all(w <= 28.0 for w in wavs)

    def test_extending_max_includes_lines_beyond_data(self):
        """Extending max_wave to 35 µm must include lines at 30 and 34 µm."""
        mll = self._make_mll()
        mll.wavelength_range = (5.0, 28.0)   # restrict first
        assert mll.num_lines == 3

        mll.wavelength_range = (5.0, 35.0)   # extend past data max
        assert mll.num_lines == 5, (
            f"Expected 5 lines after extension to 35 µm, got {mll.num_lines}"
        )
        wavs = mll.get_wavelengths()
        assert any(w > 28.0 for w in wavs), "No lines found beyond 28 µm after range extension"

    def test_extending_min_includes_lines_below_data(self):
        """Extending min_wave to 4 µm must include lines at 6 µm."""
        mll = self._make_mll()
        mll.wavelength_range = (10.0, 35.0)  # restrict below 10
        assert all(w >= 10.0 for w in mll.get_wavelengths())

        mll.wavelength_range = (4.0, 35.0)   # extend below data min
        wavs = mll.get_wavelengths()
        assert any(w < 10.0 for w in wavs), "No lines found below 10 µm after min extension"

    def test_cache_invalidated_after_range_extension(self):
        """Cache must be rebuilt after wavelength_range changes."""
        mll = self._make_mll()
        mll.wavelength_range = (5.0, 28.0)
        _ = mll.get_wavelengths()  # populate cache

        mll.wavelength_range = (5.0, 35.0)
        assert mll._filtered_raw_data is None, "Filtered cache not cleared after range change"
        wavs = mll.get_wavelengths()
        assert len(wavs) == 5

    def test_namedtuple_fields_are_numpy_after_extension(self):
        """After extending the range, namedtuple fields must still be numpy arrays."""
        mll = self._make_mll()
        mll.wavelength_range = (5.0, 35.0)
        nt = mll.lines_as_namedtuple
        assert isinstance(nt.freq, np.ndarray)
        assert len(nt.freq) == 5
