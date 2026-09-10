# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.DataProcessing.LineFilterExpression.

Covers the two-level boolean model: condition masks, group folding (the OR
semantics the filter window is built on), vacuous-group dropping, mixed AND/OR
across groups, serialization round-trips, and the human-readable rendering.
"""

import dataclasses
import json

import pytest
import numpy as np
import pandas as pd

from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    ConditionError,
    ConditionGroup,
    FilterExpression,
    MaskContext,
    condition_mask,
    describe_condition,
    describe_expression,
    expression_mask,
    group_mask,
    nearest_neighbour_separation_kms,
    ops_for_kind,
    validate,
    velocity_separation_kms,
    _parse_vib_band,
    _vib_perms,
    _vib_perms_up_to,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df():
    """Five sample lines spanning 6.5 - 22.5 µm (mirrors the maker fixture)."""
    return pd.DataFrame([
        {'nr': 1, 'lev_up': '0|10', 'lev_low': '0|9',
         'lam': 12.407, 'freq': 2.416e13, 'a_stein': 1.05e-2,
         'e_up': 4586.4, 'e_low': 3379.0, 'g_up': 21, 'g_low': 19},
        {'nr': 2, 'lev_up': '0|8', 'lev_low': '0|7',
         'lam': 14.950, 'freq': 2.005e13, 'a_stein': 2.10e-2,
         'e_up': 3200.0, 'e_low': 2500.0, 'g_up': 17, 'g_low': 15},
        {'nr': 3, 'lev_up': '0|6', 'lev_low': '0|5',
         'lam': 17.221, 'freq': 1.741e13, 'a_stein': 5.00e-3,
         'e_up': 2100.0, 'e_low': 1600.0, 'g_up': 13, 'g_low': 11},
        {'nr': 4, 'lev_up': '0|20', 'lev_low': '0|19',
         'lam': 6.500, 'freq': 4.612e13, 'a_stein': 3.00e-2,
         'e_up': 8000.0, 'e_low': 6500.0, 'g_up': 41, 'g_low': 39},
        {'nr': 5, 'lev_up': '0|15', 'lev_low': '0|14',
         'lam': 22.500, 'freq': 1.332e13, 'a_stein': 8.00e-3,
         'e_up': 1200.0, 'e_low': 800.0, 'g_up': 31, 'g_low': 29},
    ])


@pytest.fixture
def ctx():
    return MaskContext(species='H2O')


def lams(df, mask):
    """Wavelengths selected by *mask*, rounded and sorted."""
    return sorted(round(v, 3) for v in df.loc[mask, 'lam'])


def rng(field, op='between', lo=None, hi=None, **kw):
    """Shorthand for a numeric range condition."""
    return Condition('range', field=field, op=op, min_val=lo, max_val=hi, **kw)


# ============================================================================
# Condition masks
# ============================================================================

class TestConditionMasks:
    """Tests for the individual mask builders."""

    def test_range_between_inclusive(self, df, ctx):
        m = condition_mask(df, rng('lam', lo=12.0, hi=15.0), ctx)
        assert lams(df, m) == [12.407, 14.950]

    def test_range_bounds_are_inclusive(self, df, ctx):
        m = condition_mask(df, rng('lam', lo=12.407, hi=12.407), ctx)
        assert lams(df, m) == [12.407]

    def test_range_ge_ignores_max_val(self, df, ctx):
        # A stale max_val must not leak in: the operator is authoritative.
        c = Condition('range', field='e_up', op='ge', min_val=5000.0, max_val=1.0)
        assert lams(df, condition_mask(df, c, ctx)) == [6.5]

    def test_range_le_ignores_min_val(self, df, ctx):
        c = Condition('range', field='e_up', op='le', max_val=2100.0, min_val=1e9)
        assert lams(df, condition_mask(df, c, ctx)) == [17.221, 22.5]

    def test_range_missing_column_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="has no 'nope' column"):
            condition_mask(df, rng('nope', op='ge', lo=1.0), ctx)

    def test_range_with_no_bounds_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="has no value"):
            condition_mask(df, rng('lam'), ctx)

    def test_range_nan_excluded_from_both_polarities(self, ctx):
        d = pd.DataFrame({'lam': [1.0, 2.0, 3.0], 'e_up': [1000.0, np.nan, 5000.0]})
        keep = condition_mask(d, rng('e_up', op='ge', lo=2000.0), ctx)
        drop = condition_mask(d, rng('e_up', op='ge', lo=2000.0, negate=True), ctx)
        assert list(d.loc[keep, 'lam']) == [3.0]
        assert list(d.loc[drop, 'lam']) == [1.0]
        # The NaN row is in neither: "unknown" is not "below the threshold".
        assert not bool((keep & drop).any())
        assert list(d.loc[~(keep | drop), 'lam']) == [2.0]

    def test_range_nullable_dtype_gives_clean_bool_mask(self, ctx):
        d = pd.DataFrame({'lam': [1.0, 2.0], 'e_up': pd.array([1.0, pd.NA],
                                                              dtype='Float64')})
        m = condition_mask(d, rng('e_up', op='ge', lo=0.5), ctx)
        assert m.dtype == bool
        assert list(m) == [True, False]

    def test_quantum_label_literal_not_regex_by_default(self, df, ctx):
        # '0|1' is a literal here; as a regex it would alternate and match more.
        literal = Condition('quantum_label', op='contains', value='0|1',
                            state='upper')
        # Only '0|10' and '0|15' contain those three characters.
        assert len(df.loc[condition_mask(df, literal, ctx)]) == 2

    def test_quantum_label_regex_opt_in(self, df, ctx):
        as_regex = Condition('quantum_label', op='contains', value='0|1',
                             state='upper', regex=True)
        assert len(df.loc[condition_mask(df, as_regex, ctx)]) == 5

    def test_quantum_label_negate_is_exact_complement(self, df, ctx):
        base = Condition('quantum_label', op='contains', value='|1', state='upper')
        neg = Condition('quantum_label', op='contains', value='|1',
                        state='upper', negate=True)
        m1 = condition_mask(df, base, ctx)
        m2 = condition_mask(df, neg, ctx)
        assert list(m1) == [not v for v in m2]

    def test_quantum_label_empty_pattern_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="has no text"):
            condition_mask(df, Condition('quantum_label', op='contains',
                                         value=''), ctx)

    def test_quantum_field_range(self, df, ctx):
        c = Condition('quantum_field', field='J', op='between',
                      min_val=8, max_val=20, state='upper')
        assert lams(df, condition_mask(df, c, ctx)) == [6.5, 12.407, 14.95, 22.5]

    def test_quantum_field_unknown_field_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="not defined by the schema"):
            condition_mask(df, Condition('quantum_field', field='Zz', op='ge',
                                         min_val=1), ctx)

    def test_quantum_field_type_mismatch_raises_not_all_false(self, df, ctx):
        # Must be loud: a silent all-False branch is a dropped disjunct under OR.
        c = Condition('quantum_field', field='J', op='eq', value='not-a-number')
        with pytest.raises(ConditionError, match="not a valid value"):
            condition_mask(df, c, ctx)

    def test_vib_band_mask(self, df, ctx):
        c = Condition('vib_band', op='band', value='v0', n_modes=1)
        assert len(df.loc[condition_mask(df, c, ctx)]) == 5

    def test_vib_band_bad_spec_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="Cannot parse vibrational band"):
            condition_mask(df, Condition('vib_band', op='band', value='wat'), ctx)

    def test_species_mask(self, ctx):
        d = pd.DataFrame({'species': ['H2O', 'CO', 'H2O'], 'lam': [1.0, 2.0, 3.0]})
        c = Condition('species', op='in', value=('H2O',))
        assert list(d.loc[condition_mask(d, c, ctx), 'lam']) == [1.0, 3.0]

    def test_species_missing_column_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="no 'species' column"):
            condition_mask(df, Condition('species', op='in', value=('H2O',)), ctx)

    def test_unknown_kind_raises(self, df, ctx):
        with pytest.raises(ConditionError, match="Unknown condition kind"):
            condition_mask(df, Condition('bogus'), ctx)

    def test_mask_index_matches_non_range_index_frame(self, ctx):
        d = pd.DataFrame({'lam': [1.0, 2.0, 3.0]}, index=['a', 'b', 'c'])
        m = condition_mask(d, rng('lam', op='ge', lo=2.0), ctx)
        assert m.index.equals(d.index)


# ============================================================================
# Velocity separation to the nearest neighbouring line
# ============================================================================

class TestVelocitySeparation:
    """Doppler separation (km/s) from each line to its closest neighbour."""

    @pytest.fixture
    def blend(self):
        """Two lines 0.78 km/s apart, plus two far-separated singles."""
        return pd.DataFrame({'nr': [1, 2, 3, 4],
                             'lam': [6.5, 6.500017, 12.0, 20.0]})

    def test_helper_matches_doppler_formula(self):
        assert velocity_separation_kms(6.5, 6.5) == 0.0
        assert velocity_separation_kms(10.0, 10.1) == pytest.approx(
            299792.458 * 0.1 / 10.05)

    def test_nearest_neighbour_is_symmetric_for_a_pair(self, blend):
        sep = nearest_neighbour_separation_kms(blend['lam'])
        assert sep[0] == pytest.approx(sep[1])
        assert sep[0] == pytest.approx(0.784, abs=1e-3)

    def test_lone_line_is_infinitely_isolated(self):
        assert nearest_neighbour_separation_kms([6.5]) == [np.inf]

    def test_ge_keeps_isolated_lines(self, blend, ctx):
        c = Condition('velocity_sep', op='ge', min_val=5.0)
        assert list(blend.loc[condition_mask(blend, c, ctx), 'nr']) == [3, 4]

    def test_le_keeps_the_blended_pair(self, blend, ctx):
        c = Condition('velocity_sep', op='le', max_val=5.0)
        assert list(blend.loc[condition_mask(blend, c, ctx), 'nr']) == [1, 2]

    def test_between_window_matches_notebook_bounds(self, blend, ctx):
        c = Condition('velocity_sep', op='between', min_val=0.5, max_val=1.0)
        assert list(blend.loc[condition_mask(blend, c, ctx), 'nr']) == [1, 2]

    def test_negate_is_the_exact_complement_when_all_lam_are_known(self, blend, ctx):
        keep = condition_mask(blend, Condition('velocity_sep', op='ge',
                                               min_val=5.0), ctx)
        drop = condition_mask(blend, Condition('velocity_sep', op='ge',
                                               min_val=5.0, negate=True), ctx)
        assert list(keep) == [not v for v in drop]

    def test_missing_wavelength_excluded_from_both_polarities(self, ctx):
        d = pd.DataFrame({'nr': [1, 2, 3], 'lam': [6.5, np.nan, 20.0]})
        c = Condition('velocity_sep', op='ge', min_val=5.0)
        keep = condition_mask(d, c, ctx)
        drop = condition_mask(d, dataclasses.replace(c, negate=True), ctx)
        assert not bool((keep & drop).any())
        assert list(d.loc[~(keep | drop), 'nr']) == [2]

    def test_no_bound_raises(self, blend, ctx):
        with pytest.raises(ConditionError, match="has no value"):
            condition_mask(blend, Condition('velocity_sep'), ctx)

    def test_missing_lam_column_raises(self, ctx):
        d = pd.DataFrame({'e_up': [1.0, 2.0]})
        with pytest.raises(ConditionError, match="no 'lam' column"):
            condition_mask(d, Condition('velocity_sep', op='ge', min_val=1.0), ctx)

    def test_neighbours_come_from_the_whole_frame_not_the_group(self, blend, ctx):
        """Δv is measured against the line list, not against other conditions."""
        g = ConditionGroup((rng('lam', op='le', hi=7.0),
                            Condition('velocity_sep', op='ge', min_val=5.0)),
                           join='AND')
        # Lines 1 and 2 blend with each other, so narrowing to λ ≤ 7 must not
        # make either of them look isolated.
        assert list(blend.loc[group_mask(blend, g, ctx), 'nr']) == []

    def test_describe_uses_delta_v(self):
        assert describe_expression(FilterExpression((ConditionGroup((
            Condition('velocity_sep', op='ge', min_val=5.0),)),))) == 'Δv ≥ 5'

    def test_round_trips_through_serialization(self):
        expr = FilterExpression((ConditionGroup((
            Condition('velocity_sep', op='between', min_val=0.8, max_val=50.0),
        )),))
        assert FilterExpression.from_dict(json.loads(json.dumps(
            expr.to_dict()))) == expr


# ============================================================================
# Model quantities (line intensity, optical depth)
# ============================================================================

class TestDerivedQuantities:
    """Conditions on values the intensity calculation produces, not columns."""

    @pytest.fixture
    def lines(self):
        return pd.DataFrame({'nr': [1, 2, 3, 4],
                             'lam': [6.5, 8.0, 12.0, 20.0]})

    @pytest.fixture
    def model(self, lines):
        return pd.DataFrame({'intens': [1.0, 50.0, 100.0, 0.5],
                             'tau': [0.01, 0.5, 3.0, np.nan]},
                            index=lines.index)

    @pytest.fixture
    def dctx(self, model):
        return lambda: MaskContext(species='H2O', derived_source=lambda: model)

    def test_optical_depth_upper_bound(self, lines, model, dctx):
        c = Condition('derived', field='tau', op='le', max_val=1.0)
        assert list(lines.loc[condition_mask(lines, c, dctx()), 'nr']) == [1, 2]

    def test_intensity_lower_bound(self, lines, dctx):
        c = Condition('derived', field='intens', op='ge', min_val=50.0)
        assert list(lines.loc[condition_mask(lines, c, dctx()), 'nr']) == [2, 3]

    def test_missing_model_value_excluded_from_both_polarities(self, lines, dctx):
        c = Condition('derived', field='tau', op='ge', min_val=0.1)
        keep = condition_mask(lines, c, dctx())
        drop = condition_mask(lines, dataclasses.replace(c, negate=True), dctx())
        assert not bool((keep & drop).any())
        assert list(lines.loc[~(keep | drop), 'nr']) == [4]

    def test_aligned_by_index_not_position(self, lines, dctx):
        """A subsampled preview frame must read ITS rows, not the first N."""
        sub = lines.iloc[::2]
        c = Condition('derived', field='intens', op='ge', min_val=50.0)
        assert list(sub.loc[condition_mask(sub, c, dctx()), 'nr']) == [3]

    def test_source_is_invoked_at_most_once(self, lines, model):
        calls = []

        def source():
            calls.append(1)
            return model

        ctx = MaskContext(species='H2O', derived_source=source)
        g = ConditionGroup((Condition('derived', field='tau', op='ge', min_val=0.0),
                            Condition('derived', field='intens', op='ge', min_val=0.0)))
        group_mask(lines, g, ctx)
        assert len(calls) == 1

    def test_without_a_source_raises(self, lines):
        c = Condition('derived', field='tau', op='ge', min_val=1.0)
        with pytest.raises(ConditionError, match="model quantity"):
            condition_mask(lines, c, MaskContext(species='H2O'))

    def test_unknown_quantity_raises(self, lines, dctx):
        c = Condition('derived', field='nope', op='ge', min_val=1.0)
        with pytest.raises(ConditionError, match="not available"):
            condition_mask(lines, c, dctx())

    def test_misaligned_model_raises_rather_than_mismatching(self, lines, model):
        shifted = model.set_axis([10, 11, 12, 13])
        ctx = MaskContext(species='H2O', derived_source=lambda: shifted)
        c = Condition('derived', field='tau', op='ge', min_val=0.1)
        with pytest.raises(ConditionError, match="does not line up"):
            condition_mask(lines, c, ctx)

    def test_source_failure_becomes_condition_error(self, lines):
        def boom():
            raise RuntimeError("no intensity")

        c = Condition('derived', field='tau', op='ge', min_val=1.0)
        with pytest.raises(ConditionError, match="no intensity"):
            condition_mask(lines, c, MaskContext(species='H2O',
                                                 derived_source=boom))

    def test_describe_uses_physics_symbols(self):
        assert describe_expression(FilterExpression((ConditionGroup((
            Condition('derived', field='tau', op='le', max_val=1.0),)),))) == 'τ ≤ 1'


# ============================================================================
# Percentile bounds
# ============================================================================

class TestPercentileBounds:
    """A ``percentile`` bound is a rank in the property's own distribution."""

    def test_upper_percentile_keeps_the_strongest_lines(self, df, ctx):
        # a_stein = [1.05e-2, 2.10e-2, 5.00e-3, 3.00e-2, 8.00e-3]; p50 = 1.05e-2.
        c = Condition('range', field='a_stein', op='ge', min_val=50.0,
                      unit='percentile')
        assert lams(df, condition_mask(df, c, ctx)) == [6.5, 12.407, 14.95]

    def test_lower_percentile_keeps_the_weakest_lines(self, df, ctx):
        c = Condition('range', field='a_stein', op='le', max_val=40.0,
                      unit='percentile')
        assert lams(df, condition_mask(df, c, ctx)) == [17.221, 22.5]

    def test_percentile_between_is_an_inner_window(self, df, ctx):
        c = Condition('range', field='e_up', op='between', min_val=25.0,
                      max_val=75.0, unit='percentile')
        assert lams(df, condition_mask(df, c, ctx)) == [12.407, 14.95, 17.221]

    def test_percentile_matches_the_equivalent_absolute_bound(self, df, ctx):
        pct = condition_mask(df, Condition('range', field='e_up', op='ge',
                                           min_val=50.0, unit='percentile'), ctx)
        # The median e_up of the five lines is 3200.
        absolute = condition_mask(df, rng('e_up', op='ge', lo=3200.0), ctx)
        assert list(pct) == list(absolute)

    def test_percentile_is_scale_free(self, ctx):
        """The same rank selects the same rows however the values are scaled."""
        small = pd.DataFrame({'lam': [1.0, 2.0, 3.0, 4.0],
                              'e_up': [1.0, 2.0, 3.0, 4.0]})
        large = small.assign(e_up=small['e_up'] * 1e9)
        c = Condition('range', field='e_up', op='ge', min_val=50.0,
                      unit='percentile')
        assert (list(condition_mask(small, c, ctx))
                == list(condition_mask(large, c, ctx)) == [False, False, True, True])

    def test_percentile_on_a_model_quantity(self):
        lines = pd.DataFrame({'nr': [1, 2, 3, 4]})
        model = pd.DataFrame({'intens': [0.5, 1.0, 50.0, 100.0]},
                             index=lines.index)
        ctx = MaskContext(species='H2O', derived_source=lambda: model)
        c = Condition('derived', field='intens', op='ge', min_val=50.0,
                      unit='percentile')
        assert list(lines.loc[condition_mask(lines, c, ctx), 'nr']) == [3, 4]

    def test_percentile_of_an_empty_property_raises(self, ctx):
        d = pd.DataFrame({'e_up': [np.nan, np.nan]})
        c = Condition('range', field='e_up', op='ge', min_val=50.0,
                      unit='percentile')
        with pytest.raises(ConditionError, match="no values"):
            condition_mask(d, c, ctx)

    def test_rank_outside_0_100_rejected(self):
        with pytest.raises(ConditionError, match="between 0 and 100"):
            Condition('range', field='lam', op='ge', min_val=120.0,
                      unit='percentile')

    def test_percentile_rejected_for_kinds_that_do_not_offer_it(self):
        with pytest.raises(ConditionError, match="not meaningful"):
            Condition('velocity_sep', op='ge', min_val=1.0, unit='percentile')

    def test_unknown_unit_rejected(self):
        with pytest.raises(ConditionError, match="Unknown unit"):
            Condition('range', field='lam', op='ge', min_val=1.0, unit='dB')

    def test_describe_marks_the_rank(self):
        c = Condition('range', field='a_stein', op='ge', min_val=90.0,
                      unit='percentile')
        assert describe_expression(
            FilterExpression((ConditionGroup((c,)),))) == 'A ≥ p90'

    def test_round_trips_through_serialization(self):
        expr = FilterExpression((ConditionGroup((
            Condition('derived', field='tau', op='between', min_val=25.0,
                      max_val=75.0, unit='percentile'),
        )),))
        assert FilterExpression.from_dict(json.loads(json.dumps(
            expr.to_dict()))) == expr


# ============================================================================
# Percent-of-maximum bounds
# ============================================================================

class TestPercentOfMaxBounds:
    """A ``percent`` bound is a fraction of the property's largest value."""

    def test_lower_bound_follows_the_strongest_line(self, df, ctx):
        # a_stein peaks at 3e-2, so 50% is 1.5e-2.
        c = Condition('range', field='a_stein', op='ge', min_val=50.0,
                      unit='percent')
        assert lams(df, condition_mask(df, c, ctx)) == [6.5, 14.95]

    def test_between_is_a_window_below_the_peak(self, df, ctx):
        # 20%..50% of 3e-2 is 6e-3..1.5e-2.
        c = Condition('range', field='a_stein', op='between', min_val=20.0,
                      max_val=50.0, unit='percent')
        assert lams(df, condition_mask(df, c, ctx)) == [12.407, 22.5]

    def test_matches_the_equivalent_absolute_bound(self, df, ctx):
        pct = condition_mask(df, Condition('range', field='e_up', op='ge',
                                           min_val=50.0, unit='percent'), ctx)
        # e_up peaks at 8000, so 50% is 4000.
        absolute = condition_mask(df, rng('e_up', op='ge', lo=4000.0), ctx)
        assert list(pct) == list(absolute)

    def test_differs_from_the_same_number_as_a_percentile(self, df, ctx):
        """The two units answer different questions and must not be conflated."""
        of_max = condition_mask(df, Condition('range', field='a_stein', op='ge',
                                              min_val=50.0, unit='percent'), ctx)
        rank = condition_mask(df, Condition('range', field='a_stein', op='ge',
                                            min_val=50.0, unit='percentile'), ctx)
        assert list(of_max) != list(rank)

    def test_survives_a_skewed_distribution(self, ctx):
        """One bright line and many faint ones: % of max keeps only the bright."""
        d = pd.DataFrame({'lam': [1.0, 2.0, 3.0, 4.0],
                          'e_up': [1.0, 1.0, 1.0, 1000.0]})
        c = Condition('range', field='e_up', op='ge', min_val=10.0,
                      unit='percent')
        assert list(d.loc[condition_mask(d, c, ctx), 'lam']) == [4.0]

    def test_on_a_model_quantity(self):
        lines = pd.DataFrame({'nr': [1, 2, 3, 4]})
        model = pd.DataFrame({'intens': [0.5, 1.0, 50.0, 100.0]},
                             index=lines.index)
        ctx = MaskContext(species='H2O', derived_source=lambda: model)
        c = Condition('derived', field='intens', op='ge', min_val=10.0,
                      unit='percent')
        assert list(lines.loc[condition_mask(lines, c, ctx), 'nr']) == [3, 4]

    def test_above_100_is_allowed_and_simply_matches_nothing(self, df, ctx):
        c = Condition('range', field='a_stein', op='ge', min_val=150.0,
                      unit='percent')
        assert lams(df, condition_mask(df, c, ctx)) == []

    def test_negative_percentage_rejected(self):
        with pytest.raises(ConditionError, match="0 or more"):
            Condition('range', field='lam', op='ge', min_val=-5.0,
                      unit='percent')

    def test_without_a_positive_peak_raises(self, ctx):
        d = pd.DataFrame({'e_up': [-1.0, -2.0]})
        c = Condition('range', field='e_up', op='ge', min_val=10.0,
                      unit='percent')
        with pytest.raises(ConditionError, match="not positive"):
            condition_mask(d, c, ctx)

    def test_describe_marks_the_reference(self):
        c = Condition('range', field='a_stein', op='ge', min_val=5.0,
                      unit='percent')
        assert describe_expression(
            FilterExpression((ConditionGroup((c,)),))) == 'A ≥ 5% of max'

    def test_describe_between_names_the_reference_once(self):
        c = Condition('derived', field='tau', op='between', min_val=1.0,
                      max_val=50.0, unit='percent')
        assert describe_condition(c) == '1% ≤ τ ≤ 50% of max'

    def test_round_trips_through_serialization(self):
        expr = FilterExpression((ConditionGroup((
            Condition('derived', field='intens', op='ge', min_val=5.0,
                      unit='percent'),
        )),))
        assert FilterExpression.from_dict(json.loads(json.dumps(
            expr.to_dict()))) == expr


# ============================================================================
# Group fold - OR semantics
# ============================================================================

class TestGroupFold:
    """Tests for combining conditions inside one group."""

    def test_or_two_wavelength_ranges(self, df, ctx):
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('lam', lo=12.0, hi=13.0)), join='OR')
        assert lams(df, group_mask(df, g, ctx)) == [6.5, 12.407]

    def test_and_two_wavelength_ranges_is_empty(self, df, ctx):
        # The mistake the window's "Use Any?" hint exists to catch.
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('lam', lo=12.0, hi=13.0)), join='AND')
        assert lams(df, group_mask(df, g, ctx)) == []

    def test_or_is_a_union_not_a_concat(self, df, ctx):
        # The two ranges overlap on nr 1 and 2; each line must appear once.
        g = ConditionGroup((rng('lam', lo=6.0, hi=15.0),
                            rng('lam', lo=12.0, hi=18.0)), join='OR')
        selected = df.loc[group_mask(df, g, ctx)]
        assert len(selected) == len(set(selected.index))
        assert sorted(selected['nr']) == [1, 2, 3, 4]

    def test_group_of_one_equals_the_condition(self, df, ctx):
        c = rng('lam', op='ge', lo=13.0)
        direct = condition_mask(df, c, ctx)
        for join in ('AND', 'OR'):
            folded = group_mask(df, ConditionGroup((c,), join=join), ctx)
            assert list(folded) == list(direct)

    def test_all_any_wording_accepted(self, df, ctx):
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('lam', lo=12.0, hi=13.0)), join='Any')
        assert lams(df, group_mask(df, g, ctx)) == [6.5, 12.407]

    def test_unknown_join_raises(self):
        with pytest.raises(ConditionError, match="Unknown join operator"):
            ConditionGroup((), join='XOR')


# ============================================================================
# Vacuous groups are dropped, never folded as True or False
# ============================================================================

class TestVacuousDrop:
    """A group with nothing to match on is ABSENT, not True and not False."""

    def test_empty_group_returns_none(self, df, ctx):
        assert group_mask(df, ConditionGroup((), join='OR'), ctx) is None

    def test_disabled_group_returns_none(self, df, ctx):
        g = ConditionGroup((rng('lam', op='ge', lo=1.0),), enabled=False)
        assert group_mask(df, g, ctx) is None

    def test_all_conditions_disabled_returns_none(self, df, ctx):
        g = ConditionGroup((rng('lam', op='ge', lo=1.0, enabled=False),))
        assert group_mask(df, g, ctx) is None

    def test_or_with_empty_group_equals_lone_group(self, df, ctx):
        """OR(a, EMPTY) == a - the highest-risk semantic in the whole model."""
        a = ConditionGroup((rng('lam', lo=6.0, hi=7.0),), join='OR')
        lone = expression_mask(df, FilterExpression((a,), join='OR'), ctx)
        withempty = expression_mask(
            df, FilterExpression((a, ConditionGroup((), join='OR')), join='OR'), ctx)
        assert list(lone) == list(withempty)
        assert lams(df, withempty) == [6.5]

    def test_and_with_empty_group_equals_lone_group(self, df, ctx):
        a = ConditionGroup((rng('lam', lo=6.0, hi=7.0),))
        lone = expression_mask(df, FilterExpression((a,), join='AND'), ctx)
        withempty = expression_mask(
            df, FilterExpression((a, ConditionGroup(())), join='AND'), ctx)
        assert list(lone) == list(withempty)

    def test_expression_with_no_groups_is_all_true(self, df, ctx):
        assert int(expression_mask(df, FilterExpression(), ctx).sum()) == 5

    def test_expression_all_groups_vacuous_is_all_true(self, df, ctx):
        e = FilterExpression((ConditionGroup(()), ConditionGroup(())), join='OR')
        assert int(expression_mask(df, e, ctx).sum()) == 5

    def test_disabled_condition_absent_not_false(self, df, ctx):
        """Disabling one of two OR conditions must equal deleting it."""
        keep = rng('lam', lo=6.0, hi=7.0)
        off = rng('lam', lo=12.0, hi=13.0, enabled=False)
        both = group_mask(df, ConditionGroup((keep, off), join='OR'), ctx)
        only = group_mask(df, ConditionGroup((keep,), join='OR'), ctx)
        assert list(both) == list(only)


# ============================================================================
# Expression fold - mixed AND/OR across groups
# ============================================================================

class TestExpressionFold:
    """Tests for combining group masks with the outer operator."""

    def test_user_motivating_example(self, df, ctx):
        """(6<=lam<=7 OR 12<=lam<=13 OR e_up>=5000) - the literal request."""
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('lam', lo=12.0, hi=13.0),
                            rng('e_up', op='ge', lo=5000.0)), join='OR')
        assert lams(df, expression_mask(df, FilterExpression((g,)), ctx)) == \
            [6.5, 12.407]

    def test_or_group_and_group(self, df, ctx):
        """(A OR B) AND e_up <= 5000 -> only 12.407 (6.5 has e_up = 8000)."""
        a = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('lam', lo=12.0, hi=13.0)), join='OR')
        b = ConditionGroup((rng('e_up', op='le', hi=5000.0),), join='AND')
        e = FilterExpression((a, b), join='AND')
        assert lams(df, expression_mask(df, e, ctx)) == [12.407]

    def test_outer_join_changes_the_result(self, df, ctx):
        a = ConditionGroup((rng('lam', lo=6.0, hi=7.0),))
        b = ConditionGroup((rng('e_up', op='le', hi=2000.0),))
        as_and = expression_mask(df, FilterExpression((a, b), join='AND'), ctx)
        as_or = expression_mask(df, FilterExpression((a, b), join='OR'), ctx)
        assert lams(df, as_and) == []
        assert lams(df, as_or) == [6.5, 22.5]

    def test_and_group_or_group(self, df, ctx):
        a = ConditionGroup((rng('lam', op='ge', lo=14.0),
                            rng('e_up', op='le', hi=3500.0)), join='AND')
        b = ConditionGroup((rng('lam', lo=6.0, hi=7.0),), join='AND')
        e = FilterExpression((a, b), join='OR')
        assert lams(df, expression_mask(df, e, ctx)) == [6.5, 14.95, 17.221, 22.5]

    def test_empty_frame_short_circuits(self, df, ctx):
        empty = df.iloc[0:0]
        g = ConditionGroup((Condition('quantum_field', field='J', op='ge',
                                      min_val=1),), join='AND')
        m = expression_mask(empty, FilterExpression((g,)), ctx)
        assert len(m) == 0 and m.dtype == bool


# ============================================================================
# Serialization
# ============================================================================

class TestSerialization:
    """Round-trip and strictness of to_dict / from_dict."""

    def test_round_trip_simple(self):
        e = FilterExpression((ConditionGroup((Condition('range', field='lam',
                                                        op='between',
                                                        min_val=1.0, max_val=2.0),),
                                             join='OR'),), join='AND')
        assert FilterExpression.from_dict(e.to_dict()) == e

    def test_round_trip_all_kinds(self):
        e = FilterExpression((ConditionGroup((
            Condition('range', field='lam', op='ge', min_val=1.0, negate=True),
            Condition('quantum_label', op='contains', value='0|', regex=True),
            Condition('quantum_field', field='J', op='eq', value=5,
                      state='lower'),
            Condition('vib_band', op='band', value='v1-0', n_modes=4),
            Condition('species', op='in', value=('H2O', 'CO'), enabled=False),
        ), join='OR', label='mixed'),), join='OR')
        assert FilterExpression.from_dict(e.to_dict()) == e

    def test_to_dict_omits_defaults(self):
        c = Condition('range', field='lam', op='between', min_val=1.0, max_val=2.0)
        d = FilterExpression((ConditionGroup((c,)),)).to_dict()
        emitted = d['groups'][0]['conditions'][0]
        assert set(emitted) == {'kind', 'field', 'min_val', 'max_val'}

    def test_json_dumps_loads(self):
        e = FilterExpression((ConditionGroup((Condition('range', field='lam',
                                                        op='ge', min_val=6.0),),
                                             join='OR'),))
        assert FilterExpression.from_dict(json.loads(json.dumps(e.to_dict()))) == e

    def test_from_dict_tolerates_unknown_keys(self):
        e = FilterExpression.from_dict({
            'version': 1, 'join': 'AND', 'future': 'ignored',
            'groups': [{'join': 'OR', 'surprise': 1, 'conditions': [
                {'kind': 'range', 'field': 'lam', 'op': 'ge', 'min_val': 1.0,
                 'extra': 'x'}]}]})
        assert e.groups[0].conditions[0].field == 'lam'

    def test_from_dict_defaults_missing_keys(self):
        e = FilterExpression.from_dict({'groups': [{'conditions': [
            {'kind': 'range', 'field': 'lam', 'op': 'ge', 'min_val': 1.0}]}]})
        c = e.groups[0].conditions[0]
        assert (c.state, c.n_modes, c.regex, c.negate, c.enabled) == \
            ('upper', 3, False, False, True)

    def test_from_dict_unknown_kind_raises(self):
        with pytest.raises(ConditionError, match="Unknown condition kind"):
            FilterExpression.from_dict({'groups': [{'conditions': [
                {'kind': 'nope'}]}]})

    def test_from_dict_unknown_op_raises(self):
        with pytest.raises(ConditionError, match="not valid for a 'range'"):
            FilterExpression.from_dict({'groups': [{'conditions': [
                {'kind': 'range', 'field': 'lam', 'op': 'contains'}]}]})

    def test_from_dict_future_version_raises(self):
        with pytest.raises(ConditionError, match="newer than the supported"):
            FilterExpression.from_dict({'version': 99, 'groups': []})

    def test_join_is_normalised_at_construction(self):
        # Guards the whole model against a raw "All" being compared to "AND".
        assert FilterExpression(join='All').join == 'AND'
        assert FilterExpression(join='Any').join == 'OR'
        assert ConditionGroup(join='Any').join == 'OR'


# ============================================================================
# Describe
# ============================================================================

class TestDescribe:
    """The single renderer shared by the window, summary(), and the log."""

    def test_describe_user_example(self):
        e = FilterExpression((ConditionGroup((
            Condition('range', field='lam', op='between', min_val=6.0, max_val=7.0),
            Condition('range', field='lam', op='between', min_val=12.0, max_val=13.0),
            Condition('range', field='e_up', op='ge', min_val=5000.0),
        ), join='OR'),))
        assert describe_expression(e) == \
            "(6 ≤ λ ≤ 7  OR  12 ≤ λ ≤ 13  OR  E_up ≥ 5000)"

    def test_describe_two_groups_shows_outer_operator(self):
        a = ConditionGroup((Condition('range', field='lam', op='between',
                                      min_val=6.0, max_val=7.0),
                            Condition('range', field='lam', op='between',
                                      min_val=12.0, max_val=13.0)), join='OR')
        b = ConditionGroup((Condition('range', field='e_up', op='le',
                                      max_val=5000.0),))
        assert ")  AND  " in describe_expression(FilterExpression((a, b),
                                                                  join='AND'))

    def test_describe_omits_vacuous_group(self):
        a = ConditionGroup((Condition('range', field='lam', op='ge',
                                      min_val=6.0),))
        text = describe_expression(FilterExpression((a, ConditionGroup(())),
                                                    join='AND'))
        assert text == "λ ≥ 6"
        assert "AND" not in text

    def test_describe_no_conditions(self):
        assert describe_expression(FilterExpression()) == "all lines (no conditions)"

    def test_describe_negation_uses_natural_wording(self):
        c = Condition('range', field='e_up', op='ge', min_val=5000.0, negate=True)
        assert describe_expression(
            FilterExpression((ConditionGroup((c,)),))) == "E_up < 5000"

    def test_describe_omits_disabled_condition(self):
        on = Condition('range', field='lam', op='ge', min_val=6.0)
        off = Condition('range', field='e_up', op='ge', min_val=1.0, enabled=False)
        assert describe_expression(
            FilterExpression((ConditionGroup((on, off), join='OR'),))) == "λ ≥ 6"


# ============================================================================
# validate
# ============================================================================

class TestValidate:
    """Collecting every unevaluable condition instead of only the first."""

    def test_validate_clean_returns_empty(self, df, ctx):
        e = FilterExpression((ConditionGroup((rng('lam', op='ge', lo=1.0),)),))
        assert validate(df, e, ctx) == []

    def test_validate_reports_all_bad_rows_with_paths(self, df, ctx):
        e = FilterExpression((
            ConditionGroup((rng('lam', op='ge', lo=1.0), rng('nope', op='ge', lo=1.0))),
            ConditionGroup((rng('alsonope', op='ge', lo=1.0),)),
        ))
        problems = validate(df, e, ctx)
        assert [p for p, _ in problems] == [(0, 1), (1, 0)]

    def test_validate_skips_disabled(self, df, ctx):
        e = FilterExpression((ConditionGroup(
            (rng('nope', op='ge', lo=1.0, enabled=False),)),))
        assert validate(df, e, ctx) == []


# ============================================================================
# on_error policies
# ============================================================================

class TestOnError:
    """raise / identity / drop."""

    def test_raise_is_the_default(self, df):
        g = ConditionGroup((rng('nope', op='ge', lo=1.0),))
        with pytest.raises(ConditionError):
            group_mask(df, g, MaskContext(species='H2O'))

    def test_identity_widens_an_or_group(self, df):
        """Documents WHY on_error='identity' is unsafe inside an OR group:
        one broken condition selects the entire line list."""
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('nope', op='ge', lo=1.0)), join='OR')
        m = group_mask(df, g, MaskContext(species='H2O', on_error='identity'))
        assert int(m.sum()) == 5

    def test_drop_omits_the_condition(self, df):
        g = ConditionGroup((rng('lam', lo=6.0, hi=7.0),
                            rng('nope', op='ge', lo=1.0)), join='OR')
        m = group_mask(df, g, MaskContext(species='H2O', on_error='drop'))
        assert lams(df, m) == [6.5]


# ============================================================================
# Relocated vibrational helpers
# ============================================================================

class TestVibHelpersRelocated:
    """The helpers moved out of LineListMaker but stay importable from it."""

    def test_importable_from_both_modules(self):
        from iSLAT.Modules.DataProcessing.LineListMaker import (
            _parse_vib_band as from_maker)
        assert from_maker is _parse_vib_band

    def test_vib_perms_unchanged(self):
        assert _vib_perms(1, 1) == {'1'}
        assert _vib_perms_up_to(1, 1) == {'0', '1'}
        assert _parse_vib_band('v1-0', 1) == ({'1'}, {'0'})


# ============================================================================
# Operator registry
# ============================================================================

class TestOpsForKind:
    """The registry the window builds its operator menus from."""

    def test_range_ops(self):
        assert ops_for_kind('range') == ('between', 'ge', 'le', 'eq')

    def test_unknown_kind_is_empty(self):
        assert ops_for_kind('nope') == ()


# ============================================================================
# Operator defaulting and validation
# ============================================================================

class TestOperatorDefaulting:
    """A condition can never carry an operator its kind does not support."""

    def test_default_op_is_per_kind(self):
        assert Condition('range', field='lam', min_val=1.0).op == 'between'
        assert Condition('quantum_label', value='x').op == 'contains'
        assert Condition('vib_band', value='v1').op == 'band'
        assert Condition('species', value=('H2O',)).op == 'in'

    def test_invalid_op_for_kind_raises(self):
        with pytest.raises(ConditionError, match="not valid for a 'vib_band'"):
            Condition('vib_band', op='between', value='v1')

    def test_unknown_kind_is_left_to_condition_mask(self):
        assert Condition('bogus').op == ''

    def test_vib_band_default_round_trips(self):
        """Regression: an omitted op used to serialize unreadably."""
        e = FilterExpression((ConditionGroup(
            (Condition('vib_band', value='v1'),)),))
        assert FilterExpression.from_dict(e.to_dict()) == e

    def test_species_default_round_trips(self):
        e = FilterExpression((ConditionGroup(
            (Condition('species', value=('H2O', 'CO')),)),))
        assert FilterExpression.from_dict(e.to_dict()) == e

    def test_default_op_omitted_from_dict(self):
        d = FilterExpression((ConditionGroup(
            (Condition('vib_band', value='v1'),)),)).to_dict()
        assert 'op' not in d['groups'][0]['conditions'][0]

    def test_non_default_op_kept_in_dict(self):
        d = FilterExpression((ConditionGroup(
            (Condition('range', field='lam', op='ge', min_val=1.0),)),)).to_dict()
        assert d['groups'][0]['conditions'][0]['op'] == 'ge'


# ============================================================================
# Missing-value handling for exact matches (regression)
# ============================================================================

class TestQuantumFieldSentinel:
    """An unparsed quantum field is 'unknown', never 'some other value'."""

    @pytest.fixture
    def qdf(self):
        return pd.DataFrame({
            'lev_up': ['0_0_0|5_1_4', '0_0_0|3_2_1', 'JUNK', '0_0_0|9_4_5'],
            'lev_low': ['0_0_0|4_1_3'] * 4,
            'lam': [6.5, 12.4, 14.9, 22.5],
        })

    def test_eq_excludes_unparsed(self, qdf, ctx):
        c = Condition('quantum_field', field='J', op='eq', value='5')
        assert list(condition_mask(qdf, c, ctx)) == [True, False, False, False]

    def test_negated_eq_does_not_resurrect_unparsed(self, qdf, ctx):
        """'J != 5' must not sweep in lines whose J could not be parsed."""
        c = Condition('quantum_field', field='J', op='eq', value='5',
                      negate=True)
        assert list(condition_mask(qdf, c, ctx)) == [False, True, False, True]

    def test_eq_matches_range_polarity_semantics(self, qdf, ctx):
        """'not (5 <= J <= 5)' and 'J != 5' must agree."""
        as_eq = condition_mask(qdf, Condition(
            'quantum_field', field='J', op='eq', value='5', negate=True), ctx)
        as_range = condition_mask(qdf, Condition(
            'quantum_field', field='J', op='between', min_val=5, max_val=5,
            negate=True), ctx)
        assert list(as_eq) == list(as_range)

    def test_unparsed_row_in_neither_polarity(self, qdf, ctx):
        keep = condition_mask(qdf, Condition(
            'quantum_field', field='J', op='eq', value='5'), ctx)
        drop = condition_mask(qdf, Condition(
            'quantum_field', field='J', op='eq', value='5', negate=True), ctx)
        assert not bool((keep | drop).iloc[2])


# ============================================================================
# Malformed patterns surface as ConditionError (regression)
# ============================================================================

class TestPatternErrors:
    """A bad pattern must not escape as a library exception."""

    def test_invalid_regex_raises_condition_error(self, df, ctx):
        c = Condition('quantum_label', op='matches', value='[unclosed')
        with pytest.raises(ConditionError, match="not a valid search pattern"):
            condition_mask(df, c, ctx)

    def test_invalid_regex_via_contains_flag(self, df, ctx):
        c = Condition('quantum_label', op='contains', value='(unclosed',
                      regex=True)
        with pytest.raises(ConditionError, match="not a valid search pattern"):
            condition_mask(df, c, ctx)

    def test_literal_contains_accepts_regex_metacharacters(self, df, ctx):
        """With regex off, '[' is just a character, not a broken pattern."""
        c = Condition('quantum_label', op='contains', value='[unclosed')
        assert int(condition_mask(df, c, ctx).sum()) == 0
