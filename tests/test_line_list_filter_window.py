# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.GUI.Widgets.LineListFilterWindow.

Only the module-level pure helpers are exercised - the mapping from widget
state to a FilterExpression - so no Tk root is constructed and the tests run
headless.
"""

import pytest
import numpy as np
import pandas as pd

from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    ConditionGroup,
    FilterExpression,
    MaskContext,
)
from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import (
    build_expression,
    build_property_catalog,
    diagnose_empty_group,
    display_ops_for_kind,
    group_row_indices,
    has_quantum_schema,
    parse_float_strict,
    row_state_to_condition,
    split_percent,
    _rows_align,
)


def _frame():
    """A minimal line-list frame with the standard core columns."""
    return pd.DataFrame([
        {'nr': 1, 'lev_up': '0|10', 'lev_low': '0|9', 'lam': 12.407,
         'freq': 2.416e13, 'a_stein': 1.05e-2, 'e_up': 4586.4,
         'e_low': 3379.0, 'g_up': 21, 'g_low': 19},
        {'nr': 4, 'lev_up': '0|20', 'lev_low': '0|19', 'lam': 6.500,
         'freq': 4.612e13, 'a_stein': 3.00e-2, 'e_up': 8000.0,
         'e_low': 6500.0, 'g_up': 41, 'g_low': 39},
    ])


def _catalog():
    return build_property_catalog(_frame(), 'H2O')


def _row(prop, op, v1='', v2='', **kw):
    state = {'prop': prop, 'op': op, 'v1': v1, 'v2': v2,
             'modes': 3, 'regex': False, 'enabled': True}
    state.update(kw)
    return state


# ---------------------------------------------------------------------------
# parse_float_strict
# ---------------------------------------------------------------------------

def test_parse_float_strict_blank_is_no_bound():
    assert parse_float_strict('') == (None, None)
    assert parse_float_strict('   ') == (None, None)


def test_parse_float_strict_valid():
    assert parse_float_strict('6.5') == (6.5, None)


def test_parse_float_strict_accepts_scientific_notation():
    assert parse_float_strict('1e-3') == (1e-3, None)


def test_parse_float_strict_garbage_returns_message():
    value, error = parse_float_strict('abc')
    assert value is None
    assert 'not a number' in error


# ---------------------------------------------------------------------------
# build_property_catalog
# ---------------------------------------------------------------------------

def test_catalog_core_fields_keep_their_labels_and_order():
    labels = [s.label for s in _catalog()]
    assert labels[:6] == ['Wavelength (µm)', 'E_up (K)', 'E_low (K)',
                          'Einstein A (s⁻¹)', 'g_up', 'g_low']


def test_catalog_includes_freq():
    assert 'Frequency (Hz)' in [s.label for s in _catalog()]


def test_catalog_includes_velocity_separation():
    assert 'Velocity sep. (km/s)' in [s.label for s in _catalog()]


def test_catalog_omits_velocity_separation_without_lam():
    df = _frame().drop(columns=['lam'])
    assert 'Velocity sep. (km/s)' not in [
        s.label for s in build_property_catalog(df, 'H2O')]


def test_row_state_velocity_separation_builds_condition():
    cond, errors = row_state_to_condition(
        _row('Velocity sep. (km/s)', '≥', '5'), _catalog())
    assert errors == []
    assert cond == Condition('velocity_sep', op='ge', min_val=5.0)


def test_row_state_velocity_separation_between():
    cond, _ = row_state_to_condition(
        _row('Velocity sep. (km/s)', 'between', '0.8', '50'), _catalog())
    assert cond.kind == 'velocity_sep'
    assert (cond.min_val, cond.max_val) == (0.8, 50.0)


# ---------------------------------------------------------------------------
# Model quantities (line intensity, optical depth)
# ---------------------------------------------------------------------------

def _model_catalog():
    return build_property_catalog(_frame(), 'H2O', derived=('intens', 'tau'))


def test_catalog_omits_model_quantities_by_default():
    labels = [s.label for s in _catalog()]
    assert 'Line intensity' not in labels
    assert 'Optical depth (τ)' not in labels


def test_catalog_includes_requested_model_quantities():
    labels = [s.label for s in _model_catalog()]
    assert 'Line intensity' in labels
    assert 'Optical depth (τ)' in labels


def test_catalog_model_quantities_are_derived_kind():
    spec = next(s for s in _model_catalog() if s.label == 'Optical depth (τ)')
    assert spec.kind == 'derived' and spec.field == 'tau'


def test_row_state_optical_depth_builds_condition():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', '≤', '1'), _model_catalog())
    assert errors == []
    assert cond == Condition('derived', field='tau', op='le', max_val=1.0)


# ---------------------------------------------------------------------------
# Percentile operands
# ---------------------------------------------------------------------------

def test_split_percent():
    assert split_percent('5%') == ('5', True)
    assert split_percent(' 5 % ') == ('5', True)
    assert split_percent('5') == ('5', False)
    assert split_percent('') == ('', False)


def test_row_state_percent_sets_percentile_unit():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', '90%'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percentile' and cond.min_val == 90.0


def test_row_state_percentile_on_a_column_property():
    cond, _ = row_state_to_condition(
        _row('Einstein A (s⁻¹)', '≥', '75%'), _catalog())
    assert cond.unit == 'percentile' and cond.min_val == 75.0


def test_row_state_percentile_on_both_bounds():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', 'between', '25%', '75%'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percentile'
    assert (cond.min_val, cond.max_val) == (25.0, 75.0)


def test_row_state_mixed_percentile_and_absolute_is_an_error():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', 'between', '25%', '50'), _model_catalog())
    assert cond is None
    assert len(errors) == 1 and "'%' on both bounds" in errors[0]


def test_row_state_rank_outside_0_100_is_an_error():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', '120%'), _model_catalog())
    assert cond is None
    assert len(errors) == 1 and 'between 0 and 100' in errors[0]


def test_row_state_percentile_rejected_where_it_is_not_offered():
    cond, errors = row_state_to_condition(
        _row('Velocity sep. (km/s)', '≥', '50%'), _catalog())
    assert cond is None
    assert len(errors) == 1 and 'not meaningful' in errors[0]


def test_row_state_percentile_on_one_sided_le():
    cond, _ = row_state_to_condition(
        _row('Line intensity', '≤', '50%'), _model_catalog())
    assert cond.unit == 'percentile'
    assert cond.min_val is None and cond.max_val == 50.0


def test_catalog_includes_numeric_extras():
    df = _frame()
    df['xmin'] = [1.0, 2.0]
    assert 'xmin' in [s.label for s in build_property_catalog(df, 'H2O')]


def test_catalog_includes_quantum_labels():
    labels = [s.label for s in _catalog()]
    assert 'Quantum lev_up' in labels and 'Quantum lev_low' in labels


def test_catalog_quantum_fields_for_h2o():
    labels = [s.label for s in _catalog()]
    assert 'J (upper)' in labels
    assert 'J (lower)' in labels


def test_catalog_no_quantum_fields_for_unknown_species():
    labels = [s.label for s in build_property_catalog(_frame(), 'NotAMolecule')]
    assert not any('(upper)' in lb for lb in labels)
    assert not has_quantum_schema('NotAMolecule')


def test_catalog_species_only_when_column_present():
    assert 'Species' not in [s.label for s in _catalog()]
    df = _frame()
    df['species'] = 'H2O'
    assert 'Species' in [s.label for s in build_property_catalog(df, 'H2O')]


def test_catalog_vib_band_present():
    assert 'Vib. band' in [s.label for s in _catalog()]


def test_catalog_skips_absent_core_columns():
    df = _frame().drop(columns=['freq'])
    assert 'Frequency (Hz)' not in [s.label for s in build_property_catalog(df, 'H2O')]


# ---------------------------------------------------------------------------
# display_ops_for_kind
# ---------------------------------------------------------------------------

def test_display_ops_offer_negated_forms():
    ops = display_ops_for_kind('range')
    assert 'between' in ops and 'not between' in ops and '≠' in ops


def test_display_ops_for_label_kind():
    assert display_ops_for_kind('quantum_label')[0] == 'contains'


# ---------------------------------------------------------------------------
# row_state_to_condition
# ---------------------------------------------------------------------------

def test_row_state_to_condition_range():
    cond, errors = row_state_to_condition(
        _row('Wavelength (µm)', 'between', '6.0', '7.0'), _catalog())
    assert errors == []
    assert cond == Condition('range', field='lam', op='between',
                             min_val=6.0, max_val=7.0)


def test_row_state_blank_row_returns_none():
    cond, errors = row_state_to_condition(_row('', 'between'), _catalog())
    assert cond is None and errors == []


def test_row_state_property_chosen_but_no_value_returns_none():
    """A half-finished row is excluded, never treated as matching everything."""
    cond, errors = row_state_to_condition(
        _row('Wavelength (µm)', 'between'), _catalog())
    assert cond is None and errors == []


def test_row_state_not_between_sets_negate():
    cond, _ = row_state_to_condition(
        _row('Wavelength (µm)', 'not between', '6.0', '7.0'), _catalog())
    assert cond.op == 'between' and cond.negate is True


def test_row_state_ne_sets_negate():
    cond, _ = row_state_to_condition(
        _row('g_up', '≠', '21'), _catalog())
    assert cond.op == 'eq' and cond.negate is True


def test_row_state_le_maps_single_box_to_max_val():
    cond, _ = row_state_to_condition(_row('E_up (K)', '≤', '5000'), _catalog())
    assert cond.min_val is None and cond.max_val == 5000.0


def test_row_state_ge_maps_single_box_to_min_val():
    cond, _ = row_state_to_condition(_row('E_up (K)', '≥', '5000'), _catalog())
    assert cond.min_val == 5000.0 and cond.max_val is None


def test_row_state_garbage_bound_returns_error():
    cond, errors = row_state_to_condition(
        _row('Wavelength (µm)', 'between', 'abc', '7.0'), _catalog())
    assert cond is None
    assert len(errors) == 1 and 'not a number' in errors[0]


def test_row_state_quantum_label_carries_regex_flag():
    cond, _ = row_state_to_condition(
        _row('Quantum lev_up', 'contains', '0|1', regex=True), _catalog())
    assert cond.kind == 'quantum_label' and cond.regex is True


def test_row_state_matches_regex_forces_regex():
    cond, _ = row_state_to_condition(
        _row('Quantum lev_up', 'matches regex', '0.1'), _catalog())
    assert cond.op == 'matches' and cond.regex is True


def test_row_state_vib_band_carries_modes():
    cond, _ = row_state_to_condition(
        _row('Vib. band', 'is band', 'v1-0', modes=4), _catalog())
    assert cond.kind == 'vib_band' and cond.value == 'v1-0' and cond.n_modes == 4


def test_row_state_disabled_flag_is_carried():
    cond, _ = row_state_to_condition(
        _row('Wavelength (µm)', '≥', '6.0', enabled=False), _catalog())
    assert cond.enabled is False


def test_row_state_unknown_property_reports_error():
    cond, errors = row_state_to_condition(_row('Nope', '≥', '1'), _catalog())
    assert cond is None and 'Unknown property' in errors[0]


# ---------------------------------------------------------------------------
# build_expression
# ---------------------------------------------------------------------------

def test_build_expression_user_example():
    """GUI row states must produce exactly the hand-built expression."""
    groups = [{'join': 'Any', 'enabled': True, 'rows': [
        _row('Wavelength (µm)', 'between', '6.0', '7.0'),
        _row('Wavelength (µm)', 'between', '12.0', '13.0'),
        _row('E_up (K)', '≥', '5000'),
    ]}]
    expr, errors = build_expression('All', groups, _catalog())
    assert errors == []
    assert expr == FilterExpression((ConditionGroup((
        Condition('range', field='lam', op='between', min_val=6.0, max_val=7.0),
        Condition('range', field='lam', op='between', min_val=12.0, max_val=13.0),
        Condition('range', field='e_up', op='ge', min_val=5000.0),
    ), join='OR'),), join='AND')


def test_build_expression_normalises_all_any_wording():
    expr, _ = build_expression('Any', [{'join': 'All', 'rows': []}], _catalog())
    assert expr.join == 'OR'
    assert expr.groups[0].join == 'AND'


def test_build_expression_skips_blank_rows():
    groups = [{'join': 'All', 'rows': [
        _row('Wavelength (µm)', '≥', '6.0'),
        _row('', 'between'),
    ]}]
    expr, errors = build_expression('All', groups, _catalog())
    assert errors == []
    assert len(expr.groups[0].conditions) == 1


def test_build_expression_collects_all_errors():
    groups = [{'join': 'All', 'rows': [
        _row('Wavelength (µm)', '≥', 'bad'),
        _row('E_up (K)', '≥', 'alsobad'),
    ]}]
    _expr, errors = build_expression('All', groups, _catalog())
    assert len(errors) == 2


def test_build_expression_keeps_empty_group():
    """An empty group is preserved so the window can still render it."""
    expr, _ = build_expression('All', [{'join': 'All', 'rows': []}], _catalog())
    assert len(expr.groups) == 1 and expr.groups[0].conditions == ()


# ---------------------------------------------------------------------------
# diagnose_empty_group
# ---------------------------------------------------------------------------

def test_diagnose_fires_on_two_and_ranges():
    df = _frame()
    g = ConditionGroup((
        Condition('range', field='lam', op='between', min_val=6.0, max_val=7.0),
        Condition('range', field='lam', op='between', min_val=12.0, max_val=13.0),
    ), join='AND')
    hint = diagnose_empty_group(g, df, MaskContext(species='H2O'))
    assert hint is not None and 'Any' in hint


def test_diagnose_silent_when_or_would_also_be_empty():
    """No false advice: switching to Any would not help here."""
    df = _frame()
    g = ConditionGroup((
        Condition('range', field='lam', op='between', min_val=100.0, max_val=101.0),
        Condition('range', field='lam', op='between', min_val=200.0, max_val=201.0),
    ), join='AND')
    assert diagnose_empty_group(g, df, MaskContext(species='H2O')) is None


def test_diagnose_silent_for_single_condition():
    df = _frame()
    g = ConditionGroup((
        Condition('range', field='lam', op='ge', min_val=999.0),), join='AND')
    assert diagnose_empty_group(g, df, MaskContext(species='H2O')) is None


def test_diagnose_silent_for_or_group():
    df = _frame()
    g = ConditionGroup((
        Condition('range', field='lam', op='between', min_val=6.0, max_val=7.0),
        Condition('range', field='lam', op='between', min_val=12.0, max_val=13.0),
    ), join='OR')
    assert diagnose_empty_group(g, df, MaskContext(species='H2O')) is None


# ---------------------------------------------------------------------------
# Disabled rows are ignored entirely (regression)
# ---------------------------------------------------------------------------

def test_disabled_row_with_garbage_does_not_block():
    """A stale typo in a switched-off row must not block the expression."""
    groups = [{'join': 'All', 'rows': [
        _row('Wavelength (µm)', '≥', '6.0'),
        _row('E_up (K)', '≥', 'garbage', enabled=False),
    ]}]
    expr, errors = build_expression('All', groups, _catalog())
    assert errors == []
    assert len(expr.groups[0].conditions) == 1


def test_disabled_valid_row_still_carried_as_disabled():
    cond, errors = row_state_to_condition(
        _row('Wavelength (µm)', '≥', '6.0', enabled=False), _catalog())
    assert errors == [] and cond is not None and cond.enabled is False


def test_enabled_row_with_garbage_still_reports():
    _cond, errors = row_state_to_condition(
        _row('E_up (K)', '≥', 'garbage'), _catalog())
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# group_row_indices — condition index -> GUI row index (regression)
# ---------------------------------------------------------------------------

def test_group_row_indices_skips_blank_rows():
    """A blank row above a real one shifts condition indices; the map fixes it."""
    state = {'join': 'All', 'rows': [
        _row('', 'between'),
        _row('Wavelength (µm)', '≥', '6.0'),
    ]}
    assert group_row_indices(state, _catalog()) == [1]


def test_group_row_indices_identity_when_no_blanks():
    state = {'join': 'All', 'rows': [
        _row('Wavelength (µm)', '≥', '6.0'),
        _row('E_up (K)', '≤', '5000'),
    ]}
    assert group_row_indices(state, _catalog()) == [0, 1]


def test_group_row_indices_matches_condition_count():
    state = {'join': 'All', 'rows': [
        _row('', 'between'),
        _row('Wavelength (µm)', '≥', '6.0'),
        _row('E_up (K)', '≥', 'garbage', enabled=False),
        _row('E_low (K)', '≤', '900'),
    ]}
    catalog = _catalog()
    expr, _errors = build_expression('All', [state], catalog)
    assert len(group_row_indices(state, catalog)) == \
        len(expr.groups[0].conditions)


# ---------------------------------------------------------------------------
# _rows_align — the guard against indexing arrays from a different line set
# ---------------------------------------------------------------------------

def test_rows_align_accepts_matching_wavelengths():
    frame = pd.DataFrame({'lam': [1.0, 2.0, 3.0]})
    assert _rows_align(frame, {'wavelength': np.array([1.0, 2.0, 3.0])})


def test_rows_align_rejects_same_count_different_lines():
    """Row counts alone are not enough: two wavelength windows can hold the
    same number of lines while describing entirely different ones."""
    frame = pd.DataFrame({'lam': [4.90, 4.95, 4.99]})
    assert not _rows_align(frame, {'wavelength': np.array([6.65, 6.80, 7.19])})


def test_rows_align_rejects_length_mismatch():
    frame = pd.DataFrame({'lam': [1.0, 2.0, 3.0]})
    assert not _rows_align(frame, {'wavelength': np.array([1.0, 2.0])})


def test_rows_align_rejects_reordered_rows():
    frame = pd.DataFrame({'lam': [1.0, 2.0, 3.0]})
    assert not _rows_align(frame, {'wavelength': np.array([3.0, 2.0, 1.0])})


def test_rows_align_tolerates_absent_wavelengths():
    """Nothing to compare: the row-count check stands on its own."""
    frame = pd.DataFrame({'lam': [1.0, 2.0]})
    assert _rows_align(frame, {'eu': np.array([1.0, 2.0])})


def test_rows_align_rejects_none_data():
    assert not _rows_align(pd.DataFrame({'lam': [1.0]}), None)


def test_rows_align_matches_nan_positions():
    frame = pd.DataFrame({'lam': [1.0, np.nan, 3.0]})
    assert _rows_align(frame, {'wavelength': np.array([1.0, np.nan, 3.0])})
