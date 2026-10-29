# -*- coding: utf-8 -*-
"""
Unit tests for iSLAT.Modules.GUI.Widgets.LineListFilterWindow.

Only the module-level pure helpers are exercised - the mapping from widget
state to a FilterExpression - so no Tk root is constructed and the tests run
headless.
"""

import os

import pytest
import numpy as np
import pandas as pd

from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    ConditionError,
    ConditionGroup,
    FilterExpression,
    MaskContext,
)
from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import (
    FORMAT_AUTO,
    build_expression,
    build_property_catalog,
    diagnose_empty_group,
    display_ops_for_kind,
    group_row_indices,
    has_quantum_schema,
    parse_float_strict,
    quantum_format_choices,
    row_state_to_condition,
    split_unit,
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
# Relative operands: '5%' of the maximum, 'p90' as a percentile
# ---------------------------------------------------------------------------

def test_split_unit_percent_of_max():
    assert split_unit('5%') == ('5', 'percent')
    assert split_unit(' 5 % ') == ('5', 'percent')


def test_split_unit_percentile():
    assert split_unit('p90') == ('90', 'percentile')
    assert split_unit('P90') == ('90', 'percentile')
    assert split_unit('90%ile') == ('90', 'percentile')
    assert split_unit('90pct') == ('90', 'percentile')


def test_split_unit_absolute():
    assert split_unit('5') == ('5', '')
    assert split_unit('') == ('', '')
    assert split_unit('1e-3') == ('1e-3', '')


def test_row_state_percent_of_max_sets_percent_unit():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', '5%'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percent' and cond.min_val == 5.0


def test_row_state_percentile_sets_percentile_unit():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', 'p90'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percentile' and cond.min_val == 90.0


def test_row_state_relative_bound_on_a_column_property():
    cond, _ = row_state_to_condition(
        _row('Einstein A (s⁻¹)', '≥', '10%'), _catalog())
    assert cond.unit == 'percent' and cond.min_val == 10.0


def test_row_state_relative_bound_on_both_bounds():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', 'between', 'p25', 'p75'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percentile'
    assert (cond.min_val, cond.max_val) == (25.0, 75.0)


def test_row_state_mixed_units_is_an_error():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', 'between', 'p25', '50%'), _model_catalog())
    assert cond is None
    assert len(errors) == 1 and 'same unit' in errors[0]


def test_row_state_relative_and_absolute_mixed_is_an_error():
    cond, errors = row_state_to_condition(
        _row('Optical depth (τ)', 'between', '25%', '50'), _model_catalog())
    assert cond is None
    assert len(errors) == 1 and 'same unit' in errors[0]


def test_row_state_rank_outside_0_100_is_an_error():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', 'p120'), _model_catalog())
    assert cond is None
    assert len(errors) == 1 and 'between 0 and 100' in errors[0]


def test_row_state_percent_above_100_is_allowed():
    cond, errors = row_state_to_condition(
        _row('Line intensity', '≥', '150%'), _model_catalog())
    assert errors == []
    assert cond.unit == 'percent' and cond.min_val == 150.0


def test_row_state_relative_bound_rejected_where_it_is_not_offered():
    cond, errors = row_state_to_condition(
        _row('Velocity sep. (km/s)', '≥', '50%'), _catalog())
    assert cond is None
    assert len(errors) == 1 and 'not meaningful' in errors[0]


def test_row_state_relative_bound_on_one_sided_le():
    cond, _ = row_state_to_condition(
        _row('Line intensity', '≤', 'p50'), _model_catalog())
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


# ---------------------------------------------------------------------------
# Line-list format override
# ---------------------------------------------------------------------------

_CO_PAR = os.path.join("iSLAT", "DATAFILES", "HITRANdata", "data_Hitran_CO.par")


def _upper_v_is(v):
    return FilterExpression(groups=(ConditionGroup(conditions=(
        Condition('quantum_field', field='v', op='between',
                  min_val=v, max_val=v, state='upper'),)),))


@pytest.fixture
def renamed_co_maker():
    """A CO line list whose name ('CO_v1-0') matches no registered schema."""
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
    if not os.path.exists(_CO_PAR):
        pytest.skip("CO HITRAN file not available")
    line_list = MoleculeLineList(molecule_id='CO_v1-0',
                                 filename=os.path.abspath(_CO_PAR),
                                 wavelength_range=(4.9, 5.2))
    return LineListMaker(line_list)


def test_format_choices_start_with_auto_then_registered_schemas():
    choices = quantum_format_choices()
    assert choices[0] == FORMAT_AUTO
    assert 'H2O' in choices and 'CO' in choices


def test_maker_schema_species_defaults_to_species(renamed_co_maker):
    assert renamed_co_maker.quantum_format is None
    assert renamed_co_maker.schema_species == 'CO_v1-0'


def test_maker_format_override_parses_without_renaming(renamed_co_maker):
    maker = renamed_co_maker
    with pytest.raises(ConditionError):
        maker.filter_expression(_upper_v_is(2))

    maker.quantum_format = 'CO'
    maker.filter_expression(_upper_v_is(2))
    base = maker.original_df
    expected = (base['lev_up'].str.split('|').str[0].str.strip() == '2').sum()
    assert 0 < len(maker) == expected
    assert maker.species == 'CO_v1-0'
    assert set(maker.df['species']) == {'CO_v1-0'}


def test_maker_format_override_survives_copy_and_clears_with_blank(renamed_co_maker):
    renamed_co_maker.quantum_format = 'CO'
    assert renamed_co_maker.copy().quantum_format == 'CO'
    renamed_co_maker.quantum_format = ''
    assert renamed_co_maker.schema_species == 'CO_v1-0'


@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def renamed_co_molecule():
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    if not os.path.exists(_CO_PAR):
        pytest.skip("CO HITRAN file not available")
    LineListFilterWindow._format_overrides.clear()
    LineListFilterWindow._whole_list_choices.clear()
    yield Molecule(
        name='CO_v1-0', displaylabel='CO_v1-0', filepath=_CO_PAR, color='#f00',
        is_visible=True, temp=600.0, radius=0.5, n_mol=1e17, distance=140.0,
        fwhm=130.0, wavelength_range=(4.9, 5.2), model_pixel_res=0.001,
        initial_molecule_parameters={'t_kin': 600.0, 'scale_exponent': 17.0,
                                     'scale_number': 1.0, 'radius_init': 0.5},
    )
    LineListFilterWindow._format_overrides.clear()
    LineListFilterWindow._whole_list_choices.clear()


def _choose_format(win, choice):
    win._format_var.set(choice)
    win._on_format_changed()


def test_window_format_override_offers_quantum_fields(tk_root, renamed_co_molecule):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win = LineListFilterWindow(tk_root, renamed_co_molecule)
    win.withdraw()
    try:
        keys = {s.key for s in win._catalog}
        assert 'qn:upper:J' not in keys
        assert win._schema_note.winfo_manager() == 'pack'

        _choose_format(win, 'CO')
        assert 'qn:upper:J' in {s.key for s in win._catalog}
        assert win._schema_note.winfo_manager() == ''
        assert win._mask_context().species == 'CO'
        assert win._maker.species == 'CO_v1-0'
        assert all(r.spec() is None or r.spec() in win._catalog
                   for g in win._group_rows for r in g.rows)
    finally:
        win._on_close()


def test_window_remembers_format_per_molecule(tk_root, renamed_co_molecule):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win = LineListFilterWindow(tk_root, renamed_co_molecule)
    win.withdraw()
    _choose_format(win, 'CO')
    win._on_close()

    win = LineListFilterWindow(tk_root, renamed_co_molecule)
    win.withdraw()
    try:
        assert win._format_var.get() == 'CO'
        assert win._maker.schema_species == 'CO'
        _choose_format(win, FORMAT_AUTO)
        assert win._maker.quantum_format is None
        assert not LineListFilterWindow._format_overrides
    finally:
        win._on_close()


def test_window_format_change_reapplies_committed_filter(tk_root, renamed_co_molecule):
    from unittest import mock
    from iSLAT.Modules.GUI.Widgets import LineListFilterWindow as mod
    win = mod.LineListFilterWindow(tk_root, renamed_co_molecule)
    win.withdraw()
    try:
        _choose_format(win, 'CO')
        win._maker.filter_expression(_upper_v_is(2))
        kept = len(win._maker)
        assert 0 < kept < win._total_lines

        # A schema that also declares 'v' keeps the filter applied
        _choose_format(win, 'N2')
        assert len(win._maker) == kept
        assert win._committed_mask is not None and win._committed_mask.sum() == kept

        # Without a schema the filter cannot be evaluated: cleared, with a warning
        with mock.patch.object(mod.messagebox, 'showwarning') as warn:
            _choose_format(win, FORMAT_AUTO)
        warn.assert_called_once()
        assert len(win._maker) == win._total_lines
        assert not win._maker.filters
    finally:
        win._on_close()


# ---------------------------------------------------------------------------
# Whole line list vs. model wavelength range
# ---------------------------------------------------------------------------

def _lam_between(lo, hi):
    return FilterExpression(groups=(ConditionGroup(conditions=(
        Condition('range', field='lam', op='between', min_val=lo, max_val=hi),)),))


def test_without_wavelength_range_returns_every_line(renamed_co_maker):
    limited = renamed_co_maker._linelist
    full = limited.without_wavelength_range()
    assert full is not limited and full.wavelength_range is None
    assert len(full) > len(limited)
    assert full.partition_function is limited.partition_function
    assert limited.wavelength_range == (4.9, 5.2)  # original untouched
    assert full.without_wavelength_range() is full


def test_without_wavelength_range_covers_in_memory_lists(renamed_co_maker):
    """A list with no file behind it (e.g. an applied filter result) widens too."""
    full = renamed_co_maker._linelist.without_wavelength_range()
    from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
    in_memory = LineListMaker(full).to_linelist()
    in_memory.wavelength_range = (4.9, 5.2)
    assert len(in_memory.without_wavelength_range()) == len(full)


def _open_whole(tk_root, mol):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win = LineListFilterWindow(tk_root, mol)
    win.withdraw()
    in_range = win._total_lines
    win._scope_var.set('whole')
    win._on_scope_changed()
    return win, in_range


def test_window_whole_list_filters_lines_outside_model_range(tk_root, renamed_co_molecule):
    win, in_range = _open_whole(tk_root, renamed_co_molecule)
    try:
        base = win._maker.original_df
        assert win._total_lines == len(base) > in_range
        assert base['lam'].min() < 4.9 or base['lam'].max() > 5.2
        assert set(base['species']) == {'CO_v1-0'}

        win._maker.filter_expression(_lam_between(4.5, 5.0))
        assert win._maker.df['lam'].min() < 4.9
    finally:
        win._on_close()


def test_window_whole_list_model_quantities_cover_every_line(tk_root, renamed_co_molecule):
    win, _ = _open_whole(tk_root, renamed_co_molecule)
    try:
        derived = win._derived_frame()
        assert len(derived) == win._total_lines
        assert np.isfinite(derived['intens']).all()
        # In-range lines get the same model value either way
        base = win._maker.original_df
        mol = renamed_co_molecule
        mol.calculate_intensity()
        own = mol.intensity.get_population_diagram_data(
            mol.radius, mol.distance, molecule=mol, full_range=False)
        inside = ((base['lam'] >= 4.9) & (base['lam'] <= 5.2)).to_numpy()
        np.testing.assert_allclose(derived['intens'].to_numpy()[inside],
                                   np.asarray(own['intens'], float), rtol=1e-6)
    finally:
        win._on_close()


def test_window_remembers_whole_list_choice(tk_root, renamed_co_molecule):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win, in_range = _open_whole(tk_root, renamed_co_molecule)
    whole = win._total_lines
    win._on_close()

    win = LineListFilterWindow(tk_root, renamed_co_molecule)
    win.withdraw()
    try:
        assert win._scope_var.get() == 'whole' and win._total_lines == whole
        win._scope_var.set('range')
        win._on_scope_changed()
        assert win._total_lines == in_range
    finally:
        win._on_close()


def test_window_whole_list_apply_keeps_lines_outside_range(tk_root, renamed_co_molecule):
    from unittest import mock
    from iSLAT.Modules.GUI.Widgets import LineListFilterWindow as mod
    win, in_range = _open_whole(tk_root, renamed_co_molecule)
    try:
        win._maker.filter_expression(_lam_between(4.5, 5.0))
        kept = len(win._maker)
        with mock.patch.object(mod.messagebox, 'askyesno', return_value=True), \
             mock.patch.object(win, '_trigger_refresh'):
            win._apply_to_molecule()

        mol_ll = renamed_co_molecule.line_list
        assert mol_ll.wavelength_range == (4.9, 5.2)
        assert len(mol_ll.without_wavelength_range()) == kept
        # The window still sees the whole applied list, and can model it
        assert win._total_lines == kept
        assert not win._sync_baseline()
        assert len(win._derived_frame()) == kept
    finally:
        win._on_close()
