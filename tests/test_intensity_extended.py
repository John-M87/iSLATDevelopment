# -*- coding: utf-8 -*-
"""Extended unit tests for the Intensity class — coverage gaps.

Covers:
- calc_intensity_batch_fast() with pre-computed context
- prepare_batch_context() and prepare_overlap_structure()
- build_table() — full_range and wavelength_range modes
- get_lines_in_range_with_intensity()
- get_table_in_range()
- bulk_parameter_update_vectorized()
- register_strategy() / _get_strategy()
- curve_growth_no_overlap method
- _repr_html_() notebook display
"""

import pytest
import numpy as np

import iSLAT.Constants as c


class _IntensityTestBase:
    """Shared helpers for Intensity tests."""

    def _make_line_list(self, n_lines=10, lam_start=10.0, lam_step=1.0,
                        molecule_id='TestMol'):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        rng = np.random.default_rng(42)
        lines_data = []
        for i in range(n_lines):
            lam = lam_start + i * lam_step
            freq = c.SPEED_OF_LIGHT_MICRONS / lam
            lines_data.append({
                'nr': i + 1,
                'lev_up': f'0|{2 * i + 2}',
                'lev_low': f'0|{2 * i + 1}',
                'lam': lam,
                'freq': freq,
                'a_stein': rng.uniform(1e-3, 5e-2),
                'e_up': 1000.0 + i * 500,
                'e_low': 500.0 + i * 500,
                'g_up': 2 * i + 3,
                'g_low': 2 * i + 1,
            })
        mll = MoleculeLineList(molecule_id=molecule_id, lines_data=lines_data)
        mll.partition_function = mll._partition_type(
            t=np.array([100, 200, 300, 500, 1000, 2000], dtype=np.float64),
            q=np.array([10, 50, 150, 500, 2000, 8000], dtype=np.float64),
        )
        mll._molar_mass = 18.015
        return mll

    def _make_intensity(self, mll=None, t_kin=500.0, n_mol=1e18, dv=1.0):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        if mll is None:
            mll = self._make_line_list()
        inten = Intensity(mll)
        inten.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=dv)
        return inten


class TestCalcIntensityBatchFast(_IntensityTestBase):
    """Tests for the fast batch API used in fitting."""

    def test_prepare_batch_context_keys(self):
        """prepare_batch_context should return the expected key set."""
        inten = self._make_intensity()
        ctx = inten.prepare_batch_context()
        expected_keys = {'freq', 'e_up', 'e_low', 'e_delta', 'line_scalar',
                         'freq_ratio', 'partition_t', 'partition_q',
                         'x_quad', 'w_quad', 'exp_neg_x2', 'sqrt_pi_sum',
                         'overlap', 'n_lines',
                         'e_low_f32', 'e_delta_f32', 'line_scalar_f32',
                         'freq_ratio_f32', 'freq_f32',
                         'x_quad_f32', 'w_quad_f32', 'exp_neg_x2_f32',
                         'sqrt_pi_sum_f32', 'two_sqrt_ln2_f32', 'tau_thin_f32',
                         'line_indices'}
        assert expected_keys.issubset(ctx.keys())

    def test_prepare_batch_context_n_lines(self):
        mll = self._make_line_list(n_lines=7)
        inten = self._make_intensity(mll)
        ctx = inten.prepare_batch_context()
        assert ctx['n_lines'] == 7
        assert ctx['freq'].shape == (7,)

    def test_prepare_batch_context_line_indices(self):
        """Providing line_indices should slice all arrays."""
        inten = self._make_intensity()
        idx = np.array([0, 2, 5])
        ctx = inten.prepare_batch_context(line_indices=idx)
        assert ctx['n_lines'] == 3
        assert ctx['freq'].shape == (3,)
        assert ctx['line_indices'] is None  # baked in

    def test_prepare_overlap_structure_all_isolated(self):
        """With well-separated lines, all should be isolated."""
        mll = self._make_line_list(n_lines=5, lam_step=2.0)
        inten = self._make_intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)
        assert ctx['overlap']['all_isolated'] is True

    def test_prepare_overlap_structure_single_line(self):
        mll = self._make_line_list(n_lines=1)
        inten = self._make_intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)
        assert ctx['overlap']['all_isolated'] is True

    def test_batch_fast_matches_regular_batch(self):
        """calc_intensity_batch_fast should give similar results to calc_intensity_batch."""
        mll = self._make_line_list(n_lines=6, lam_step=1.5)
        inten = self._make_intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)

        t_arr = np.array([400.0, 600.0, 800.0])
        n_arr = np.array([1e17, 1e18, 1e19])
        dv_arr = np.array([1.0, 1.0, 1.0])

        result_fast = inten.calc_intensity_batch_fast(t_arr, n_arr, dv_arr, ctx)
        result_reg = inten.calc_intensity_batch(t_arr, n_arr, dv_arr)

        # float32 vs float64 means tolerance is looser
        np.testing.assert_allclose(result_fast, result_reg, rtol=1e-2, atol=1e-30)

    def test_batch_fast_shape(self):
        mll = self._make_line_list(n_lines=8)
        inten = self._make_intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)

        t_arr = np.array([300.0, 500.0])
        n_arr = np.array([1e17, 1e18])
        dv_arr = np.array([1.0, 1.0])

        result = inten.calc_intensity_batch_fast(t_arr, n_arr, dv_arr, ctx)
        assert result.shape == (2, 8)

    def test_batch_fast_nonnegative(self):
        inten = self._make_intensity()
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)

        result = inten.calc_intensity_batch_fast(
            np.array([500.0]), np.array([1e18]), np.array([1.0]), ctx
        )
        assert np.all(result >= 0)


class TestBuildTable(_IntensityTestBase):
    """Tests for build_table() — DataFrame generation."""

    def test_build_table_default(self):
        """Default build_table returns active-range lines with intensity."""
        inten = self._make_intensity()
        df = inten.build_table()
        assert 'lam' in df.columns
        assert 'tau' in df.columns
        assert 'intens' in df.columns
        assert 'a_stein' in df.columns
        assert len(df) == 10

    def test_build_table_wavelength_range_filter(self):
        """wavelength_range kwarg should filter the table."""
        inten = self._make_intensity()
        df = inten.build_table(wavelength_range=(12.0, 16.0))
        # Only lines with lam in [12, 16]
        assert all(df['lam'] >= 12.0)
        assert all(df['lam'] <= 16.0)
        assert len(df) < 10

    def test_build_table_no_intensity(self):
        """If calc_intensity not called, tau/intens should be NaN."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        mll = self._make_line_list()
        inten = Intensity(mll)
        df = inten.build_table()
        assert all(np.isnan(df['tau']))
        assert all(np.isnan(df['intens']))


class TestGetLinesInRange(_IntensityTestBase):
    """Tests for get_lines_in_range_with_intensity()."""

    def test_returns_tuples(self):
        inten = self._make_intensity()
        results = inten.get_lines_in_range_with_intensity(12.0, 16.0)
        assert isinstance(results, list)
        for item in results:
            assert len(item) == 3  # (line, intensity, tau)

    def test_empty_before_calc(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        mll = self._make_line_list()
        inten = Intensity(mll)
        results = inten.get_lines_in_range_with_intensity(10.0, 20.0)
        assert results == []

    def test_all_lines_in_full_range(self):
        inten = self._make_intensity()
        results = inten.get_lines_in_range_with_intensity(9.0, 21.0)
        assert len(results) == 10

    def test_filtered_by_range(self):
        inten = self._make_intensity()
        results = inten.get_lines_in_range_with_intensity(12.0, 14.5)
        lams = [r[0].lam for r in results]
        for l in lams:
            assert 12.0 <= l <= 14.5

    def test_intensity_and_tau_positive(self):
        inten = self._make_intensity()
        results = inten.get_lines_in_range_with_intensity(10.0, 20.0)
        for _, i_val, tau_val in results:
            assert i_val >= 0
            assert tau_val >= 0


class TestGetTableInRange(_IntensityTestBase):
    """Tests for get_table_in_range() convenience method."""

    def test_returns_dataframe(self):
        inten = self._make_intensity()
        df = inten.get_table_in_range(12.0, 16.0)
        assert hasattr(df, 'columns')
        assert 'lam' in df.columns

    def test_filters_correctly(self):
        inten = self._make_intensity()
        df = inten.get_table_in_range(13.0, 15.0)
        assert all(df['lam'] >= 13.0)
        assert all(df['lam'] <= 15.0)


class TestBulkParameterUpdate(_IntensityTestBase):
    """Tests for bulk_parameter_update_vectorized()."""

    def test_basic_batch(self):
        inten = self._make_intensity()
        combos = [
            {'t_kin': 400.0, 'n_mol': 1e17, 'dv': 1.0},
            {'t_kin': 600.0, 'n_mol': 1e18, 'dv': 2.0},
            {'t_kin': 800.0, 'n_mol': 1e19, 'dv': 1.5},
        ]
        result = inten.bulk_parameter_update_vectorized(combos)
        assert result.shape == (3, 10)
        assert np.all(result >= 0)

    def test_empty_combos(self):
        inten = self._make_intensity()
        result = inten.bulk_parameter_update_vectorized([])
        assert len(result) == 0

    def test_single_combo_matches_calc_intensity(self):
        inten = self._make_intensity()
        combos = [{'t_kin': 500.0, 'n_mol': 1e18, 'dv': 1.0}]
        result = inten.bulk_parameter_update_vectorized(combos)
        # Compare to regular calc_intensity
        inten.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        np.testing.assert_allclose(result[0], inten.intensity, rtol=1e-10)


class TestStrategyPattern(_IntensityTestBase):
    """Tests for the strategy registry and dispatch."""

    def test_default_strategies_exist(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        Intensity._ensure_strategy_registry()
        assert 'curve_growth' in Intensity._strategy_registry
        assert 'radex' in Intensity._strategy_registry
        assert 'curve_growth_no_overlap' in Intensity._strategy_registry

    def test_get_strategy_unknown_raises(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        with pytest.raises(ValueError, match="Unknown intensity method"):
            Intensity._get_strategy('nonexistent_method')

    def test_register_custom_strategy(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity

        class DummyStrategy:
            def calculate(self, intensity_obj, center_tau, dv_vals,
                         bb_vals, freq_ratio, sqrt_ln2_inv, frequencies):
                return np.zeros_like(center_tau)

        Intensity.register_strategy('dummy_test', DummyStrategy())
        strategy = Intensity._get_strategy('dummy_test')
        assert isinstance(strategy, DummyStrategy)
        # Clean up
        del Intensity._strategy_registry['dummy_test']

    def test_curve_growth_no_overlap_method(self):
        """curve_growth_no_overlap should produce valid results."""
        inten = self._make_intensity()
        inten.invalidate_cache()
        inten.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0,
                            method='curve_growth_no_overlap')
        assert inten.intensity is not None
        assert np.all(inten.intensity >= 0)

    def test_curve_growth_vs_no_overlap_similar_for_isolated(self):
        """With well-separated lines, overlap and no-overlap should agree closely."""
        mll = self._make_line_list(n_lines=5, lam_step=2.0)

        from iSLAT.Modules.DataTypes.Intensity import Intensity
        inten1 = Intensity(mll)
        inten1.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0,
                             method='curve_growth')
        inten2 = Intensity(mll)
        inten2.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0,
                             method='curve_growth_no_overlap')
        np.testing.assert_allclose(inten1.intensity, inten2.intensity, rtol=1e-2)


class TestIntensityReprHtml(_IntensityTestBase):
    """Test the _repr_html_ method if it exists."""

    def test_repr_basic(self):
        """__repr__ should contain molecule name."""
        inten = self._make_intensity()
        r = repr(inten)
        assert 'TestMol' in r
        assert 'Intensity' in r

    def test_repr_with_calculated_values(self):
        """__repr__ should show calculated parameters."""
        inten = self._make_intensity(t_kin=700.0)
        r = repr(inten)
        assert '700' in r


class TestWavelengthRangeExtensionIntensity(_IntensityTestBase):
    """Intensity calculations must work correctly when wavelength_range
    is extended beyond the originally loaded observed data range.

    Specifically:
    - Lines that fall in the extended region must be included.
    - When NO lines exist in the active range the calculation must not
      crash (the 'list ** int' bug) and must return empty arrays.
    - After restriction and re-extension the results must be consistent.
    """

    def _make_wide_line_list(self):
        """Line list with lines from 5 µm to 34 µm, spanning both a
        'typical data range' (5-28 µm) and beyond it."""
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        lines_data = [
            {'nr': i + 1,
             'lev_up': f'0|{2*i+2}', 'lev_low': f'0|{2*i+1}',
             'lam': lam,
             'freq': c.SPEED_OF_LIGHT_MICRONS / lam,
             'a_stein': 0.01,
             'e_up': 3000.0 - i * 200,
             'e_low': 2000.0 - i * 200,
             'g_up': 2 * i + 3,
             'g_low': 2 * i + 1}
            for i, lam in enumerate([6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 34.0])
        ]
        mll = MoleculeLineList(molecule_id='WideTest', lines_data=lines_data)
        mll.partition_function = mll._partition_type(
            t=np.array([100, 300, 500, 1000, 2000], dtype=np.float64),
            q=np.array([10, 150, 500, 2000, 8000], dtype=np.float64),
        )
        mll._molar_mass = 18.015
        return mll

    # ------------------------------------------------------------------
    # Empty-range robustness
    # ------------------------------------------------------------------

    def test_calc_intensity_empty_range_returns_empty_arrays(self):
        """When all lines are outside the wavelength_range,
        calc_intensity must return empty arrays, not crash."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        mll = self._make_wide_line_list()
        mll.wavelength_range = (100.0, 200.0)  # no lines here
        inten = Intensity(mll, wavelength_range=(100.0, 200.0))
        # Must not raise TypeError about 'list ** int'
        inten.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        assert inten.intensity is not None
        assert len(inten.intensity) == 0, "Expected empty intensity array for empty range"
        assert len(inten.tau) == 0, "Expected empty tau array for empty range"

    def test_empty_range_freq_is_numpy_array(self):
        """lines_as_namedtuple.freq must be a numpy array even when empty,
        so that freq ** 3 does not raise TypeError."""
        mll = self._make_wide_line_list()
        mll.wavelength_range = (100.0, 200.0)
        nt = mll.lines_as_namedtuple
        assert isinstance(nt.freq, np.ndarray)
        # Arithmetic on an empty numpy array must not raise
        _ = nt.freq ** 3

    # ------------------------------------------------------------------
    # Extended range includes lines beyond 'data max'
    # ------------------------------------------------------------------

    def test_restricting_range_yields_fewer_lines_in_intensity(self):
        """Restricting to (5, 28) should yield fewer intensity values
        than using the full range."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        mll_full = self._make_wide_line_list()
        inten_full = Intensity(mll_full)
        inten_full.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        n_full = len(inten_full.intensity)

        mll_restricted = self._make_wide_line_list()
        mll_restricted.wavelength_range = (5.0, 28.0)
        inten_restricted = Intensity(mll_restricted, wavelength_range=(5.0, 28.0))
        inten_restricted.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        n_restricted = len(inten_restricted.intensity)

        assert n_restricted < n_full, (
            f"Restricted ({n_restricted}) should be fewer lines than full ({n_full})"
        )

    def test_extending_max_beyond_data_includes_additional_lines(self):
        """After extending max from 28 µm to 35 µm, the intensity array
        must be longer (includes lines at 30 and 34 µm)."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity

        mll_narrow = self._make_wide_line_list()
        mll_narrow.wavelength_range = (5.0, 28.0)
        inten_narrow = Intensity(mll_narrow, wavelength_range=(5.0, 28.0))
        inten_narrow.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        n_narrow = len(inten_narrow.intensity)

        mll_extended = self._make_wide_line_list()
        mll_extended.wavelength_range = (5.0, 35.0)
        inten_extended = Intensity(mll_extended, wavelength_range=(5.0, 35.0))
        inten_extended.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        n_extended = len(inten_extended.intensity)

        assert n_extended > n_narrow, (
            f"Extended range ({n_extended} lines) should include more lines than "
            f"narrow range ({n_narrow} lines)"
        )

    def test_extended_intensity_values_are_positive(self):
        """All intensity values for lines in the extended range must be >= 0."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        mll = self._make_wide_line_list()
        mll.wavelength_range = (5.0, 35.0)
        inten = Intensity(mll, wavelength_range=(5.0, 35.0))
        inten.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
        assert np.all(inten.intensity >= 0)
        assert np.all(inten.tau >= 0)

    def test_range_extension_consistent_with_direct_full_range(self):
        """Intensity calculated on the full range should match intensity
        calculated after extending a restricted range back to full."""
        from iSLAT.Modules.DataTypes.Intensity import Intensity

        mll_full = self._make_wide_line_list()
        inten_full = Intensity(mll_full)
        inten_full.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)

        # Simulate what happens when user loads data (restricts range) then
        # manually extends max_wave beyond the data
        mll_ext = self._make_wide_line_list()
        mll_ext.wavelength_range = (5.0, 28.0)   # simulate data load
        inten_ext = Intensity(mll_ext, wavelength_range=(5.0, 28.0))
        inten_ext.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)

        # Now user extends max_wave to 40 µm (beyond any line)
        mll_ext.wavelength_range = (5.0, 40.0)
        inten_ext.wavelength_range = (5.0, 40.0)
        inten_ext.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)

        # Both should contain the same 7 lines (all lines are within 40 µm)
        assert len(inten_ext.intensity) == len(inten_full.intensity), (
            "After extending to 40 µm (covers all lines), intensity length "
            "should match full range"
        )
        np.testing.assert_allclose(
            inten_ext.intensity, inten_full.intensity, rtol=1e-8,
            err_msg="Intensity values should match between full range and extended range"
        )


class TestFullRangeWithoutSourceFile(_IntensityTestBase):
    """build_table(full_range=True) on a line list with no file to rebuild from.

    _compute_full_range_intensity rebuilds a temporary MoleculeLineList from
    the parent's ``_filename`` to compute intensities over every line.  A list
    built from ``lines_data`` - a filtered subset, or any test fixture - has no
    such file, so the rebuild comes back empty.  Returning those zero-length
    arrays produced a DataFrame built from mismatched columns.
    """

    def _intensity(self, n_lines=8):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        ll = self._make_line_list(n_lines=n_lines)
        assert getattr(ll, '_filename', None) is None
        intensity = Intensity(ll)
        intensity.calc_intensity(t_kin=500.0, n_mol=1e17, dv=2.0)
        return intensity, ll

    def test_build_table_full_range_does_not_raise(self):
        intensity, _ll = self._intensity()
        df = intensity.build_table(full_range=True)
        assert df is not None

    def test_build_table_full_range_row_count_matches_line_list(self):
        intensity, ll = self._intensity(n_lines=8)
        df = intensity.build_table(full_range=True)
        assert len(df) == ll.num_lines

    def test_full_range_intensity_column_is_populated(self):
        """The active arrays ARE the full-range arrays for such a list."""
        intensity, _ll = self._intensity()
        df = intensity.build_table(full_range=True)
        assert np.isfinite(np.asarray(df['intens'], dtype=float)).any()

    def test_full_range_matches_active_range(self):
        intensity, _ll = self._intensity()
        full = intensity.build_table(full_range=True)
        active = intensity.build_table(full_range=False)
        assert len(full) == len(active)
        np.testing.assert_allclose(
            np.asarray(full['intens'], dtype=float),
            np.asarray(active['intens'], dtype=float),
        )

    def test_population_diagram_data_available(self):
        intensity, ll = self._intensity()
        data = intensity.get_population_diagram_data(1.0, 160.0)
        assert data is not None
        assert len(data['eu']) == ll.num_lines

    def test_population_diagram_arrays_are_mutually_aligned(self):
        intensity, ll = self._intensity()
        data = intensity.get_population_diagram_data(1.0, 160.0)
        lengths = {len(data[k]) for k in
                   ('eu', 'rd_yax', 'wavelength', 'a_stein', 'g_up')}
        assert lengths == {ll.num_lines}

    def test_no_parameters_set_gives_nan_not_a_crash(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        ll = self._make_line_list(n_lines=5)
        df = Intensity(ll).build_table(full_range=True)
        assert len(df) == ll.num_lines
        assert np.isnan(np.asarray(df['intens'], dtype=float)).all()


class TestPopulationDataAlignsWithPandasTable(_IntensityTestBase):
    """get_population_diagram_data(full_range=False) must align row-for-row
    with get_pandas_table().

    The filtered population diagram builds a boolean mask over the frame from
    get_pandas_table() and uses it to index the population-diagram arrays.  If
    the two disagree in length or order the diagram plots the wrong lines, so
    this alignment is a contract, not an incidental property.
    """

    def _prepared(self, wavelength_range=None):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        ll = self._make_line_list(n_lines=20, lam_start=10.0, lam_step=1.0)
        if wavelength_range is not None:
            ll.wavelength_range = wavelength_range
        intensity = Intensity(ll)
        intensity.calc_intensity(t_kin=600.0, n_mol=1e17, dv=2.0)
        return ll, intensity

    def test_lengths_match_without_wavelength_range(self):
        ll, intensity = self._prepared()
        df = ll.get_pandas_table()
        data = intensity.get_population_diagram_data(1.0, 160.0, full_range=False)
        assert len(df) == len(data['eu'])

    def test_lengths_match_with_wavelength_range(self):
        ll, intensity = self._prepared(wavelength_range=(12.0, 18.0))
        df = ll.get_pandas_table()
        data = intensity.get_population_diagram_data(1.0, 160.0, full_range=False)
        assert len(df) < 20                      # the range really did narrow it
        assert len(df) == len(data['eu'])

    def test_rows_align_elementwise_with_wavelength_range(self):
        ll, intensity = self._prepared(wavelength_range=(12.0, 18.0))
        df = ll.get_pandas_table()
        data = intensity.get_population_diagram_data(1.0, 160.0, full_range=False)
        np.testing.assert_allclose(
            np.asarray(df['lam'], dtype=float),
            np.asarray(data['wavelength'], dtype=float),
        )
        assert [str(v) for v in df['lev_up']] == [str(v) for v in data['lev_up']]

    def test_full_range_diverges_when_a_range_is_set(self):
        """Documents WHY the filtered diagram asks for full_range=False."""
        ll, intensity = self._prepared(wavelength_range=(12.0, 18.0))
        df = ll.get_pandas_table()
        full = intensity.get_population_diagram_data(1.0, 160.0, full_range=True)
        assert len(full['eu']) != len(df)


class TestFullRangeCache(_IntensityTestBase):
    """The full-range intensities behind the population diagram are reused.

    The population diagram asks for them on every redraw.  Recomputing means
    rebuilding the whole line list from its file and recalculating every
    line (~60 ms for H2O), so the result is kept until the parameters or
    the line list change.  The rebuild from file is replaced here by an
    in-memory copy of the same lines, counted per call.
    """

    N_LINES = 12

    @pytest.fixture
    def rebuilds(self, monkeypatch):
        import iSLAT.Modules.DataTypes.MoleculeLineList as mll_module
        calls = []
        n_lines = self.N_LINES
        real = mll_module.MoleculeLineList

        def molecule_line_list(*args, **kwargs):
            if "lines_data" in kwargs:      # in-memory lists (this test's own) are built normally
                return real(*args, **kwargs)
            calls.append(kwargs)            # the full-range rebuild from file
            return self._make_line_list(n_lines=n_lines)

        monkeypatch.setattr(mll_module, "MoleculeLineList", molecule_line_list)
        return calls

    def _intensity(self):
        from iSLAT.Modules.DataTypes.Intensity import Intensity
        inten = Intensity(self._make_line_list(n_lines=self.N_LINES))
        # Optically thick, so every parameter below changes the intensities.
        inten.calc_intensity(t_kin=500.0, n_mol=1e20, dv=2.0)
        return inten

    def test_repeated_tables_reuse_the_calculation(self, rebuilds):
        inten = self._intensity()
        first = inten.build_table(full_range=True)
        second = inten.build_table(full_range=True)
        assert len(rebuilds) == 1
        import pandas as pd
        pd.testing.assert_frame_equal(first, second)
        assert np.isfinite(np.asarray(first["intens"], dtype=float)).all()

    def test_population_diagram_reuses_the_calculation(self, rebuilds):
        inten = self._intensity()
        a = inten.get_population_diagram_data(1.0, 160.0, full_range=True)
        b = inten.get_population_diagram_data(1.0, 160.0, full_range=True)
        assert len(rebuilds) == 1
        np.testing.assert_array_equal(a["intens"], b["intens"])

    @pytest.mark.parametrize("params", [
        dict(t_kin=900.0, n_mol=1e20, dv=2.0),
        dict(t_kin=500.0, n_mol=3e20, dv=2.0),
        dict(t_kin=500.0, n_mol=1e20, dv=5.0),
        dict(t_kin=500.0, n_mol=1e20, dv=2.0, method="radex"),
    ], ids=["t_kin", "n_mol", "dv", "method"])
    def test_parameter_change_recalculates(self, rebuilds, params):
        inten = self._intensity()
        before = inten.build_table(full_range=True)["intens"].to_numpy()
        inten.calc_intensity(**params)
        after = inten.build_table(full_range=True)["intens"].to_numpy()
        assert len(rebuilds) == 2
        assert not np.allclose(before, after)

    def test_new_line_list_recalculates(self, rebuilds):
        inten = self._intensity()
        inten.build_table(full_range=True)
        inten.molecule._raw_lines_data = inten.molecule._raw_lines_data.copy()
        inten.build_table(full_range=True)
        assert len(rebuilds) == 2

    def test_edits_to_a_returned_table_do_not_reach_the_cache(self, rebuilds):
        inten = self._intensity()
        first = inten.build_table(full_range=True)
        expected = first["intens"].to_numpy().copy()
        first.loc[0, "intens"] = -1.0
        first["intens"] *= 2.0
        np.testing.assert_array_equal(inten.build_table(full_range=True)["intens"].to_numpy(), expected)

    def test_list_without_source_file_is_not_cached(self):
        # The rebuild comes back empty; the active arrays are used and nothing is cached.
        inten = self._intensity()
        inten.build_table(full_range=True)
        assert inten._full_range_cache is None
