# -*- coding: utf-8 -*-
"""Tests for the line list table's links to the main window's spectrum.

Each row of a LineListViewWindow keeps its position in the loaded frame as
its item id, whatever the search or sort, so a line can be found again:
select_line selects and scrolls to the row of a line (clearing a search that
hides it).  Right-clicking a row of a window that knows the iSLAT instance
offers "Show in Spectrum", which hands the row's line (full-precision
wavelength, levels, molecule and fit window) to
islat.GUI.plot.show_line_in_spectrum.  show_line_in_line_list opens a
molecule's table, or raises the one already open, with a line selected.

The tests need a Tk display and are skipped without one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataProcessing.LineIdentity import LineRef


@pytest.fixture(scope="module")
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
def windows():
    """Collects the windows a test opens and destroys them afterwards."""
    opened = []
    yield opened
    for win in opened:
        try:
            win.destroy()
        except Exception:
            pass


def _frame():
    """Saved-line style rows: two lines share a wavelength, one has a fit window."""
    return pd.DataFrame([
        {'species': 'H2O', 'lev_up': 'a1', 'lev_low': 'b1', 'lam': 12.123456789,
         'xmin': 12.1, 'xmax': 12.15},
        {'species': 'H2O', 'lev_up': 'a2', 'lev_low': 'b2', 'lam': 13.5,
         'xmin': np.nan, 'xmax': np.nan},
        {'species': 'H2O', 'lev_up': 'a3', 'lev_low': 'b3', 'lam': 13.5,
         'xmin': np.nan, 'xmax': np.nan},
        {'species': 'CO', 'lev_up': 'c1', 'lev_low': 'd1', 'lam': 4.95,
         'xmin': np.nan, 'xmax': np.nan},
    ])


def _islat(**molecules):
    return SimpleNamespace(
        molecules_dict=dict(molecules),
        GUI=SimpleNamespace(data_field=MagicMock(), plot=MagicMock()),
    )


def _open(root, windows, df=None, **kw):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
    view = LineListViewWindow.from_dataframe(root, df if df is not None else _frame(), **kw)
    view.withdraw()
    windows.append(view)
    return view


def _labels(menu):
    end = menu.index("end")
    return [menu.entrycget(i, "label") for i in range(end + 1)
            if menu.type(i) != "separator"]


def _report_messages(islat):
    return [c.args[0] for c in islat.GUI.data_field.insert_text.call_args_list]


# ---------------------------------------------------------------------------
# Row identity
# ---------------------------------------------------------------------------

def test_rows_keep_their_position_through_sort_and_search(tk_root, windows):
    view = _open(tk_root, windows)
    assert view._tree.get_children() == ('0', '1', '2', '3')
    view._sort_by('lam')
    assert view._tree.get_children() == ('3', '0', '1', '2')
    view._sort_by('lam')   # descending
    assert view._tree.get_children() == ('1', '2', '0', '3')
    view._search_var.set('co')
    assert view._tree.get_children() == ('3',)
    # Rows still show their own values.
    assert view._tree.item('3', 'values')[view._columns.index('species')] == 'CO'


# ---------------------------------------------------------------------------
# select_line
# ---------------------------------------------------------------------------

def test_select_line_selects_and_focuses_the_row(tk_root, windows):
    view = _open(tk_root, windows)
    assert view.select_line(LineRef(12.123456789))
    assert view._tree.selection() == ('0',)
    assert view._tree.focus() == '0'


def test_select_line_tells_same_wavelength_lines_apart_by_levels(tk_root, windows):
    view = _open(tk_root, windows)
    assert view.select_line(LineRef(13.5, lev_up='a3', lev_low='b3'))
    assert view._tree.selection() == ('2',)
    assert view.select_line(LineRef(13.5, lev_up='a2'))
    assert view._tree.selection() == ('1',)


def test_select_line_clears_a_hiding_search_and_keeps_the_sort(tk_root, windows):
    view = _open(tk_root, windows)
    view._sort_by('lam')
    view._search_var.set('co')
    assert view.select_line(LineRef(13.5, lev_up='a3'))
    assert view._search_var.get() == ''
    assert view._tree.get_children() == ('3', '0', '1', '2')
    assert view._tree.selection() == ('2',)


def test_select_line_without_the_line(tk_root, windows):
    view = _open(tk_root, windows)
    view._search_var.set('co')
    assert not view.select_line(LineRef(20.0))
    assert view._search_var.get() == 'co'   # nothing to reveal
    empty = _open(tk_root, windows, df=pd.DataFrame())
    assert not empty.select_line(LineRef(20.0))


# ---------------------------------------------------------------------------
# Show in Spectrum
# ---------------------------------------------------------------------------

def test_row_menu_starts_with_show_in_spectrum(tk_root, windows):
    islat = _islat()
    view = _open(tk_root, windows, islat=islat)
    menu = view._build_column_menu('lam', row='0')
    assert _labels(menu)[:2] == ['Show in Spectrum', 'Add Column']
    menu.invoke(0)
    islat.GUI.plot.show_line_in_spectrum.assert_called_once()
    (ref,), _ = islat.GUI.plot.show_line_in_spectrum.call_args
    # The full-precision wavelength, not the rounded one shown.
    assert ref == LineRef(12.123456789, 'H2O', 'a1', 'b1', window=(12.1, 12.15))


def test_show_in_spectrum_uses_the_window_molecule(tk_root, windows):
    islat = _islat()
    view = _open(tk_root, windows, islat=islat, molecule=SimpleNamespace(name='H2O_hot'))
    assert view.row_line('3') == LineRef(4.95, 'H2O_hot', 'c1', 'd1')
    islat.GUI.plot.show_line_in_spectrum.return_value = True
    assert view.show_in_spectrum('3')


def test_no_show_in_spectrum_without_islat_or_off_the_rows(tk_root, windows):
    plain = _open(tk_root, windows)
    assert 'Show in Spectrum' not in _labels(plain._build_column_menu('lam', row='0'))
    linked = _open(tk_root, windows, islat=_islat())
    assert 'Show in Spectrum' not in _labels(linked._build_column_menu('lam'))
    assert _labels(linked._build_column_menu('lam'))[0] == 'Add Column'


def test_row_without_a_wavelength_disables_show_in_spectrum(tk_root, windows):
    df = pd.DataFrame([{'species': 'H2O', 'lam': ''}, {'species': 'H2O', 'lam': 'x'}])
    view = _open(tk_root, windows, df=df, islat=_islat())
    menu = view._build_column_menu('lam', row='1')
    assert menu.entrycget(0, 'label') == 'Show in Spectrum'
    assert menu.entrycget(0, 'state') == 'disabled'
    assert not view.show_in_spectrum('1')


def test_right_click_selects_the_row_under_the_pointer(tk_root, windows, monkeypatch):
    import tkinter as tk
    monkeypatch.setattr(tk.Menu, 'tk_popup', lambda self, *a, **k: None)
    view = _open(tk_root, windows, islat=_islat())
    monkeypatch.setattr(view._tree, 'identify_row', lambda y: '2')
    view._tree.selection_set('0')
    view._show_column_menu(SimpleNamespace(x=5, y=5, x_root=0, y_root=0))
    assert view._tree.selection() == ('2',)
    assert _labels(view._column_menu)[0] == 'Show in Spectrum'
    # Right-clicking inside a multi-row selection keeps it.
    view._tree.selection_set('1', '2')
    view._show_column_menu(SimpleNamespace(x=5, y=5, x_root=0, y_root=0))
    assert set(view._tree.selection()) == {'1', '2'}


# ---------------------------------------------------------------------------
# show_line_in_line_list
# ---------------------------------------------------------------------------

@pytest.fixture
def molecule():
    from tests.plotting.test_line_inspection_population_diagram import _make_molecule
    return _make_molecule()


def _shown(window):
    df = window._df
    return [float(df.iloc[int(iid)]['lam']) for iid in window._tree.selection()]


def test_opens_the_molecule_table_with_the_line_selected(tk_root, windows, molecule):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import (
        LineListViewWindow, show_line_in_line_list,
    )
    islat = _islat(H2O=molecule)
    window = show_line_in_line_list(LineRef(15.0, 'H2O'), islat=islat, parent=tk_root)
    window.withdraw()
    windows.append(window)
    assert window.mol_obj is molecule and window._islat is islat
    assert _shown(window) == [15.0]
    assert LineListViewWindow.find_for_molecule(molecule) is window

    # A second line raises the same table.
    again = show_line_in_line_list(LineRef(16.5, 'H2O'), islat=islat, parent=tk_root)
    assert again is window
    assert _shown(window) == [16.5]


def test_reuses_a_table_opened_from_the_control_panel(tk_root, windows, molecule):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import (
        LineListViewWindow, show_line_in_line_list,
    )
    opened = LineListViewWindow(tk_root, molecule)
    opened.withdraw()
    windows.append(opened)
    assert show_line_in_line_list(LineRef(12.0), molecule=molecule) is opened
    opened.destroy()
    assert LineListViewWindow.find_for_molecule(molecule) is None


def test_refreshes_a_table_older_than_the_line_list(tk_root, windows, molecule):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import (
        LineListViewWindow, show_line_in_line_list,
    )
    from tests.plotting.test_line_inspection_population_diagram import _make_line_list
    opened = LineListViewWindow(tk_root, molecule)
    opened.withdraw()
    windows.append(opened)
    molecule.lines = _make_line_list(n=5, lam_start=20.0)
    assert show_line_in_line_list(LineRef(21.5), molecule=molecule) is opened
    assert _shown(opened) == [21.5]


def test_reports_what_cannot_be_shown(tk_root, windows, molecule):
    from iSLAT.Modules.GUI.Widgets.LineListViewWindow import show_line_in_line_list
    islat = _islat(H2O=molecule)
    assert show_line_in_line_list(LineRef(15.0, 'CO'), islat=islat) is None
    window = show_line_in_line_list(LineRef(30.0, 'H2O'), islat=islat, parent=tk_root)
    window.withdraw()
    windows.append(window)
    assert window._tree.selection() == ()
    messages = _report_messages(islat)
    assert "Cannot show CO 15.0000 um in a line list: its molecule is not loaded." in messages
    assert "H2O 30.0000 um is not in the H2O line list." in messages
