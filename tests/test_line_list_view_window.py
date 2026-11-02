# -*- coding: utf-8 -*-
"""
Tests for the extra columns of iSLAT.Modules.GUI.Widgets.LineListViewWindow.

The column source is exercised headless, and checked against the filter
evaluator so a column always shows the value a filter condition tests.  The
window tests need a Tk display and are skipped without one.
"""

import os

import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    ConditionError,
    MaskContext,
    condition_mask,
    nearest_neighbour_separation_kms,
)
from iSLAT.Modules.GUI.Widgets.LineListViewWindow import (
    ADDABLE_KINDS,
    LineListColumnSource,
    infer_species,
    sort_rows,
    vib_band_label,
)

_CO_PAR = os.path.join("iSLAT", "DATAFILES", "HITRANdata", "data_Hitran_CO.par")


def _h2o_frame():
    """A small H2O line list with real-format level labels."""
    return pd.DataFrame([
        {'species': 'H2O', 'lev_up': '1_0_0|9_2_7', 'lev_low': '0_0_0|8_0_8',
         'lam': 6.50, 'a_stein': 1.0e-2, 'e_up': 4000.0, 'e_low': 3000.0,
         'g_up': 19, 'g_low': 17},
        {'species': 'H2O', 'lev_up': '0_1_0|7_2_5', 'lev_low': '0_1_0|6_1_6',
         'lam': 6.51, 'a_stein': 2.0e-2, 'e_up': 5000.0, 'e_low': 4000.0,
         'g_up': 15, 'g_low': 13},
        {'species': 'H2O', 'lev_up': '0_0_0|10_1_9', 'lev_low': '0_0_0|9_0_9',
         'lam': 12.40, 'a_stein': 3.0e-2, 'e_up': 2000.0, 'e_low': 1500.0,
         'g_up': 21, 'g_low': 19},
        {'species': 'H2O', 'lev_up': 'unparsed', 'lev_low': '',
         'lam': 17.20, 'a_stein': 4.0e-3, 'e_up': 1000.0, 'e_low': 800.0,
         'g_up': 13, 'g_low': 11},
    ])


def _specs(source, df, *keys):
    by_key = {s.key: s for s in source.catalog(df)}
    return [by_key[k] for k in keys]


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def test_catalog_offers_only_properties_that_are_not_already_columns():
    df = _h2o_frame()
    catalog = LineListColumnSource().catalog(df)
    assert catalog, "expected computed properties to be offered"
    assert {s.kind for s in catalog} <= ADDABLE_KINDS
    keys = [s.key for s in catalog]
    assert 'velocity_sep' in keys and 'vib_band' in keys
    assert 'qn_upper:J' in keys and 'qn_lower:Ka' in keys
    # Already columns of the table, so never offered again.
    for key in ('wavelength', 'eu', 'a_stein', 'lev_up', 'molecule'):
        assert key not in keys


def test_catalog_offers_model_quantities_only_with_a_derived_source():
    df = _h2o_frame()
    assert 'intens' not in [s.key for s in LineListColumnSource().catalog(df)]
    with_model = LineListColumnSource(derived_source=lambda ref: None)
    keys = [s.key for s in with_model.catalog(df)]
    assert 'intens' in keys and 'tau' in keys


def test_catalog_without_species_has_no_quantum_fields():
    df = _h2o_frame().drop(columns=['species'])
    keys = [s.key for s in LineListColumnSource().catalog(df)]
    assert not [k for k in keys if k.startswith('qn_')]
    assert 'vib_band' in keys


def test_infer_species():
    assert infer_species(_h2o_frame()) == 'H2O'
    mixed = _h2o_frame()
    mixed.loc[0, 'species'] = 'CO'
    assert infer_species(mixed) is None
    assert infer_species(mixed.drop(columns=['species'])) is None
    assert infer_species(None) is None


def test_explicit_species_overrides_the_species_column():
    df = _h2o_frame()
    src = LineListColumnSource(species='CO')
    assert src.species_for(df) == 'CO'
    assert 'qn_upper:Ka' not in [s.key for s in src.catalog(df)]


# ---------------------------------------------------------------------------
# Values agree with the filter evaluator
# ---------------------------------------------------------------------------

def test_quantum_field_column_matches_filter_semantics():
    df = _h2o_frame()
    src = LineListColumnSource()
    values, errors = src.compute(df, _specs(src, df, 'qn_upper:J'))
    assert errors == {}
    assert values['qn_upper:J'] == ['9', '7', '10', '']   # unparsed -> blank
    ctx = MaskContext(species='H2O')
    for row, shown in enumerate(values['qn_upper:J']):
        if not shown:
            continue
        mask = condition_mask(df, Condition('quantum_field', field='J',
                                            state='upper', op='eq',
                                            value=shown), ctx)
        assert bool(mask.iloc[row]), f"J (upper) = {shown} does not select row {row}"


def test_vib_band_column_matches_band_filter():
    df = _h2o_frame()
    src = LineListColumnSource()
    values, _ = src.compute(df, _specs(src, df, 'vib_band'))
    assert values['vib_band'] == ['v1-0', 'v1-1', 'v0-0', '']
    ctx = MaskContext(species='H2O')
    for row, band in enumerate(values['vib_band']):
        if not band:
            continue
        mask = condition_mask(df, Condition('vib_band', op='band', value=band,
                                            n_modes=3), ctx)
        assert bool(mask.iloc[row])


def test_vib_band_label_formats():
    assert vib_band_label('1_0_0|9_2_7', '0_0_1|8_0_8') == 'v1-1'
    assert vib_band_label('4|', '4|R_2') == 'v4-4'          # single-mode (CO)
    assert vib_band_label('unparsed', '0|1') == ''
    assert vib_band_label(float('nan'), '0|1') == ''
    assert vib_band_label(' 1_0_0|9', '0_0_0|8') == ''      # band filter can't match it


def test_velocity_separation_over_the_displayed_frame():
    df = _h2o_frame()
    src = LineListColumnSource()
    values, _ = src.compute(df, _specs(src, df, 'velocity_sep'))
    expected = nearest_neighbour_separation_kms(df['lam'].to_numpy())
    assert values['velocity_sep'] == ['{:.6g}'.format(v) for v in expected]


def test_velocity_separation_is_measured_over_the_baseline():
    """A filtered subset must show the separation the filter tested."""
    base = _h2o_frame()
    subset = base.iloc[[0, 2]]
    src = LineListColumnSource(baseline_source=lambda: base)
    values, errors = src.compute(subset, _specs(src, subset, 'velocity_sep'))
    assert errors == {}
    full = nearest_neighbour_separation_kms(base['lam'].to_numpy())
    assert values['velocity_sep'] == ['{:.6g}'.format(full[0]),
                                      '{:.6g}'.format(full[2])]


def test_reindexed_subset_is_refused_rather_than_misread():
    base = _h2o_frame()
    subset = base.iloc[[2, 3]].reset_index(drop=True)   # labels 0,1 exist in base
    src = LineListColumnSource(baseline_source=lambda: base)
    values, errors = src.compute(subset, _specs(src, subset, 'velocity_sep',
                                                'qn_upper:J'))
    assert 'velocity_sep' in errors and 'velocity_sep' not in values
    # Row-local properties need no baseline and still work.
    assert values['qn_upper:J'] == ['10', '']


def test_model_quantities_are_read_back_for_the_displayed_rows():
    base = _h2o_frame()
    model = pd.DataFrame({'intens': [1.0, 2.0, 3.0, 4.0],
                          'tau': [0.1, 0.2, np.nan, 0.4]}, index=base.index)
    seen = []

    def derived(reference):
        seen.append(reference)
        return model

    src = LineListColumnSource(derived_source=derived,
                               baseline_source=lambda: base)
    subset = base.iloc[[1, 2]]
    values, errors = src.compute(subset, _specs(src, subset, 'intens', 'tau'))
    assert errors == {}
    assert values['intens'] == ['2', '3']
    assert values['tau'] == ['0.2', 'NaN']
    assert len(seen) == 1 and seen[0] is base   # computed once, over the baseline


def test_model_quantity_failure_is_reported_per_column():
    df = _h2o_frame()

    def derived(_reference):
        raise ConditionError("no model here")

    src = LineListColumnSource(derived_source=derived)
    values, errors = src.compute(df, _specs(src, df, 'intens', 'vib_band'))
    assert errors == {'intens': 'no model here'}
    assert 'vib_band' in values


def test_compute_on_an_empty_frame():
    df = _h2o_frame().iloc[0:0]
    src = LineListColumnSource(species='H2O')   # nothing to infer it from
    values, errors = src.compute(df, _specs(src, df, 'qn_upper:J', 'vib_band'))
    assert values == {'qn_upper:J': [], 'vib_band': []} and errors == {}


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_sort_rows_numeric_with_blanks_last_in_both_directions():
    rows = [('a', '10'), ('b', ''), ('c', '9'), ('d', 'NaN'), ('e', '100')]
    assert [r[0] for r in sort_rows(rows, 1)] == ['c', 'a', 'e', 'b', 'd']
    assert [r[0] for r in sort_rows(rows, 1, reverse=True)] == ['e', 'a', 'c', 'b', 'd']


def test_sort_rows_falls_back_to_text():
    rows = [('x', 'v2-1'), ('y', ''), ('z', 'v1-0')]
    assert [r[0] for r in sort_rows(rows, 1)] == ['z', 'x', 'y']


def test_sort_rows_accepts_non_string_cells():
    rows = [('a', 3), ('b', 1), ('c', '')]
    assert [r[0] for r in sort_rows(rows, 1)] == ['b', 'a', 'c']


# ---------------------------------------------------------------------------
# Filter window: the companion table's rows stay aligned with the baseline
# ---------------------------------------------------------------------------

def _bare_filter_window(maker):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win = LineListFilterWindow.__new__(LineListFilterWindow)
    win._maker = maker
    win._last_mask = None
    win._committed_mask = None
    return win


def test_current_view_frame_keeps_the_baseline_index_after_apply():
    from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
    maker = LineListMaker(_h2o_frame(), species='H2O')
    expr_cond = Condition('range', field='lam', op='ge', min_val=10.0)
    maker.filter_all(expr_cond)
    win = _bare_filter_window(maker)
    win._committed_mask = np.array([False, False, True, True])
    frame = win._current_view_frame()
    assert list(frame.index) == [2, 3]
    assert list(frame['lam']) == list(maker.df['lam'])


def test_current_view_frame_without_filters_is_the_baseline():
    from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
    maker = LineListMaker(_h2o_frame(), species='H2O')
    frame = _bare_filter_window(maker)._current_view_frame()
    assert frame.index.equals(maker.original_df.index)
    assert frame is not maker.original_df


def test_table_column_source_refuses_a_replaced_maker():
    from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
    win = _bare_filter_window(LineListMaker(_h2o_frame(), species='H2O'))
    win._derived_frame = lambda: pytest.fail("must not read the new maker's model")
    source = win._table_column_source()
    win._maker = LineListMaker(_h2o_frame(), species='H2O')
    df = _h2o_frame()
    values, errors = source.compute(df, _specs(source, df, 'tau'))
    assert 'tau' in errors and 'changed' in errors['tau']


# ---------------------------------------------------------------------------
# Window tests (need a display)
# ---------------------------------------------------------------------------

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


def _open(root, df, **kw):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
    holder = {'df': df}
    view = LineListViewWindow.from_dataframe(root, lambda: holder['df'], **kw)
    view.withdraw()
    return view, holder


def _cells(view, title):
    idx = view._columns.index(title)
    return [view._tree.item(iid, 'values')[idx]
            for iid in view._tree.get_children()]


def test_add_and_remove_a_column(tk_root):
    view, _ = _open(tk_root, _h2o_frame())
    base = list(view._columns)
    assert view.add_column('qn_upper:J')
    assert view._columns == base + ['J (upper)']
    assert [str(c) for c in _cells(view, 'J (upper)')] == ['9', '7', '10', '']
    assert all(len(r) == len(view._columns) for r in view._all_rows)

    view._remove_column(view._added[0])
    assert view._columns == base and not view._added
    assert all(len(r) == len(base) for r in view._all_rows)
    view.destroy()


def test_added_columns_survive_a_live_refresh_with_search_and_sort(tk_root):
    view, holder = _open(tk_root, _h2o_frame())
    view.add_column('vib_band')
    view._sort_by('Vib. band')
    holder['df'] = _h2o_frame().iloc[[0, 1, 2]]
    view._search_var.set('v1')
    view.refresh()
    assert 'Vib. band' in view._columns
    assert [str(c) for c in _cells(view, 'Vib. band')] == ['v1-0', 'v1-1']
    assert view._sort_col == 'Vib. band'
    view.destroy()


def test_remove_all_added_columns(tk_root):
    view, _ = _open(tk_root, _h2o_frame())
    base = list(view._columns)
    for key in ('vib_band', 'velocity_sep', 'qn_lower:J'):
        assert view.add_column(key)
    view._remove_all_columns()
    assert view._columns == base and not view._added
    view.destroy()


def test_column_menu_lists_columns_and_offers_removal(tk_root):
    view, _ = _open(tk_root, _h2o_frame())
    view.add_column('vib_band')
    menu = view._build_column_menu('Vib. band')
    labels = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)]
    assert labels[0] == 'Add Column'
    assert 'Remove Column \u201cVib. band\u201d' in labels
    add_menu = menu.nametowidget(menu.entrycget(0, 'menu'))
    add_labels = [add_menu.entrycget(i, 'label')
                  for i in range(add_menu.index('end') + 1)
                  if add_menu.type(i) != 'separator']
    assert 'Velocity sep. (km/s)' in add_labels and 'Vib. band' in add_labels
    assert 'Upper-Level Quantum Numbers' in add_labels
    assert 'Lower-Level Quantum Numbers' in add_labels
    # Clicking outside an added column offers no single-column removal.
    other = view._build_column_menu('lam')
    other_labels = [other.entrycget(i, 'label')
                    for i in range(other.index('end') + 1)]
    assert not [l for l in other_labels if l.startswith('Remove Column')]
    view.destroy()


def test_right_click_is_bound_on_the_table(tk_root):
    view, _ = _open(tk_root, _h2o_frame())
    bound = view._tree.bind()
    assert '<Button-3>' in bound
    if view.tk.call('tk', 'windowingsystem') != 'aqua':
        # Ctrl+click must stay multi-select off macOS.
        assert '<Control-Button-1>' not in bound
    view.destroy()


def test_failed_column_is_not_added(tk_root, monkeypatch):
    from iSLAT.Modules.GUI.Widgets import LineListViewWindow as mod
    warnings = []
    monkeypatch.setattr(mod.messagebox, 'showwarning',
                        lambda *a, **k: warnings.append(a))

    def derived(_reference):
        raise ConditionError('no model')

    view, _ = _open(tk_root, _h2o_frame(),
                    column_source=LineListColumnSource(derived_source=derived))
    base = list(view._columns)
    assert not view.add_column('tau')
    assert view._columns == base and warnings
    view.destroy()


def test_column_that_fails_on_refresh_is_blank_with_a_note(tk_root):
    state = {'fail': False}

    def derived(reference):
        if state['fail']:
            raise ConditionError('model went away')
        return pd.DataFrame({'intens': np.arange(len(reference), dtype=float)},
                            index=reference.index)

    view, _ = _open(tk_root, _h2o_frame(),
                    column_source=lambda: LineListColumnSource(derived_source=derived))
    assert view.add_column('intens')
    state['fail'] = True
    view.refresh()
    assert set(_cells(view, 'Line intensity')) == {''}
    assert 'model went away' in view._status_var.get()
    view.destroy()


def test_export_includes_added_columns(tk_root, monkeypatch, tmp_path):
    from iSLAT.Modules.GUI.Widgets import LineListViewWindow as mod
    out = tmp_path / 'out.csv'
    monkeypatch.setattr(mod.filedialog, 'asksaveasfilename',
                        lambda **k: str(out))
    view, _ = _open(tk_root, _h2o_frame())
    view.add_column('vib_band')
    view._export_csv()
    written = pd.read_csv(out, keep_default_na=False)
    assert written['Vib. band'].tolist() == ['v1-0', 'v1-1', 'v0-0', '']
    view.destroy()


@pytest.fixture
def co_molecule():
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    if not os.path.exists(_CO_PAR):
        pytest.skip("CO HITRAN file not available")
    return Molecule(
        name='CO', displaylabel='CO', filepath=_CO_PAR, color='#f00',
        is_visible=True, temp=600.0, radius=0.5, n_mol=1e17, distance=140.0,
        fwhm=130.0, wavelength_range=(4.9, 5.2), model_pixel_res=0.001,
        initial_molecule_parameters={'t_kin': 600.0, 'scale_exponent': 17.0,
                                     'scale_number': 1.0, 'radius_init': 0.5},
    )


def test_molecule_view_adds_model_quantities(tk_root, co_molecule):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
    view = LineListViewWindow(tk_root, co_molecule)
    view.withdraw()
    assert view.add_column('intens') and view.add_column('qn_lower:J')
    data = co_molecule.intensity.get_population_diagram_data(
        co_molecule.radius, co_molecule.distance, molecule=co_molecule,
        full_range=False)
    expected = ['{:.6g}'.format(v) for v in np.asarray(data['intens'], float)]
    idx = view._columns.index('Line intensity')
    assert [r[idx] for r in view._all_rows] == expected
    view.destroy()


def test_filter_companion_table_matches_filter_values(tk_root, co_molecule):
    from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
    win = LineListFilterWindow(tk_root, co_molecule)
    win.withdraw()
    try:
        base = win._maker.original_df
        win._last_mask = (base['lam'] > base['lam'].median()).to_numpy()
        win._open_table_view()
        table = win._table_view
        table.withdraw()
        assert table.add_column('tau') and table.add_column('velocity_sep')

        shown = base.loc[win._last_mask]
        tau = win._derived_frame()['tau'].loc[shown.index]
        sep = nearest_neighbour_separation_kms(base['lam'].to_numpy())[win._last_mask]
        tau_idx = table._columns.index('Optical depth (τ)')
        sep_idx = table._columns.index('Velocity sep. (km/s)')
        assert [r[tau_idx] for r in table._all_rows] == \
            ['{:.6g}'.format(v) for v in tau]
        assert [r[sep_idx] for r in table._all_rows] == \
            ['{:.6g}'.format(v) for v in sep]
    finally:
        win._on_close()
