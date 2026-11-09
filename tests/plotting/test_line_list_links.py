# -*- coding: utf-8 -*-
"""Tests for the links between the line inspection plot and the line list tables.

"Show in Line List" (right-click menu of the line inspection plot) names the
line marker under the pointer, or else the selected (orange) line, and hands
it to show_line_in_line_list.

The other way, iSLATPlot.show_line_in_spectrum (used by the tables' "Show in
Spectrum") selects a range around a line, as dragging across it would,
centers the main spectrum on it and highlights the line through the view's
select_line, which follows the same rules as clicking the line.
"""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from iSLAT.Modules.DataProcessing.LineIdentity import DEFAULT_HALF_WINDOW, LineRef
from tests.plotting.test_line_inspection_population_diagram import (
    _build_three_panel_view,
    _make_molecule,
)

#: Wavelengths of the lines of _make_molecule() (lam_start 12.0, 1.5 apart).
LINE_LAMS = (12.0, 13.5, 15.0, 16.5, 18.0)
#: A selection covering every line.
ALL_LINES = (11.5, 19.0)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

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
def shown_in_line_list(monkeypatch):
    """Records the lines the menus ask to show in a line list, instead of opening tables."""
    from iSLAT.Modules.GUI.Widgets import LineListViewWindow as viewer
    calls = []

    def recording(ref, **kwargs):
        calls.append((ref, kwargs))

    monkeypatch.setattr(viewer, "show_line_in_line_list", recording)
    return calls


def _right_click(ax, x, y):
    """A right-button press at data point (*x*, *y*) of *ax*."""
    px, py = ax.transData.transform((x, y))
    return MouseEvent("button_press_event", ax.figure.canvas, px, py, button=3)


def _line_list_entry(menu):
    """Index and label of the menu's "Show ... in Line List" entry."""
    for i in range(menu.index("end") + 1):
        if menu.type(i) != "command":
            continue
        label = menu.entrycget(i, "label")
        if label.startswith("Show") and label.endswith("in Line List"):
            return i, label
    raise AssertionError("the menu has no Show in Line List entry")


def _label_for(info):
    return f"Show {info['molecule_name']} {info['lam']:.4f} um in Line List"


def _entry_at(view, lam):
    return next(e for e in view.active_lines if e[3]['lam'] == pytest.approx(lam))


def _color(entry):
    return mcolors.to_hex(entry[0].get_colors()[0])


def _point_on_marker(view, entry):
    """A data point on *entry*'s dashed marker inside the visible part of ax2."""
    lo, hi = view.ax2.get_ylim()
    top = min(entry[3]['lineheight'], hi)
    bottom = max(0.0, lo)
    assert top > bottom, "marker not visible"
    return entry[3]['lam'], (top + bottom) / 2.0


@pytest.fixture
def selected_view():
    """(view, mol): a ThreePanelView with every line selected and drawn."""
    mol = _make_molecule()
    view, _pm, fig = _build_three_panel_view(mol)
    view.on_selection(*ALL_LINES)
    fig.canvas.draw()
    yield view, mol
    plt.close(fig)


def _other_than_selected(view):
    """An inspected line other than the selected one, with its marker in view."""
    lo, hi = view.ax2.get_ylim()
    for entry in sorted(view.active_lines, key=lambda e: -e[3]['lineheight']):
        if entry[3] is not view.selected_line and min(entry[3]['lineheight'], hi) > max(0.0, lo):
            return entry
    raise AssertionError("no second line in view")


# ---------------------------------------------------------------------------
# Three Panel view menus
# ---------------------------------------------------------------------------

class TestInspectionMenu:
    def test_inspection_menu_names_the_marker_under_the_pointer(self, tk_root, selected_view):
        view, _mol = selected_view
        entry = _other_than_selected(view)
        event = _right_click(view.ax2, *_point_on_marker(view, entry))
        assert event.inaxes is view.ax2
        _, label = _line_list_entry(view.build_context_menu(event, tk_root))
        assert label == _label_for(entry[3])

    def test_inspection_menu_falls_back_to_the_selected_line(self, tk_root, selected_view):
        view, _mol = selected_view
        lo, hi = view.ax2.get_ylim()
        event = _right_click(view.ax2, ALL_LINES[0] + 0.1, (lo + hi) / 2)
        _, label = _line_list_entry(view.build_context_menu(event, tk_root))
        assert label == _label_for(view.selected_line)

    def test_entry_opens_the_line_in_its_line_list(self, tk_root, selected_view, shown_in_line_list):
        view, _mol = selected_view
        entry = _other_than_selected(view)
        event = _right_click(view.ax2, *_point_on_marker(view, entry))
        menu = view.build_context_menu(event, tk_root)
        index, _ = _line_list_entry(menu)
        menu.invoke(index)
        assert len(shown_in_line_list) == 1
        ref, kwargs = shown_in_line_list[0]
        assert ref == LineRef.from_info(entry[3])
        assert ref.lev_up and ref.lev_low
        assert kwargs['islat'] is view._islat
        assert kwargs['parent'] is tk_root

    def test_entry_is_disabled_without_a_line(self, tk_root):
        mol = _make_molecule()
        view, _pm, fig = _build_three_panel_view(mol)   # nothing selected yet
        menu = view._build_line_inspection_menu(tk_root, None)
        index, label = _line_list_entry(menu)
        assert label == "Show in Line List"
        assert menu.entrycget(index, "state") == "disabled"
        plt.close(fig)


# ---------------------------------------------------------------------------
# select_line
# ---------------------------------------------------------------------------

class TestSelectLine:
    def test_highlights_the_line_like_a_click(self, selected_view):
        view, _mol = selected_view
        strongest = view.selected_line
        entry = _other_than_selected(view)
        shown = []
        view._display_line_info = lambda value, **_kw: shown.append(value)

        assert view.select_line(LineRef.from_info(entry[3]))
        assert view.selected_line is entry[3]
        assert shown == [entry[3]]
        assert _color(entry) == mcolors.to_hex('orange')
        previous = next(e for e in view.active_lines if e[3] is strongest)
        assert _color(previous) != mcolors.to_hex('orange')

    def test_an_uninspected_line_is_not_selected(self, selected_view):
        view, _mol = selected_view
        before = view.selected_line
        assert not view.select_line(LineRef(30.0, 'H2O'))
        assert view.selected_line is before

    def test_a_comparison_molecule_line_switches_the_diagram(self):
        primary = _make_molecule(name='H2O', temp=500.0, lam_start=12.0)
        comparison = _make_molecule(name='CO', temp=900.0, lam_start=12.7)
        comparison.color = '#FF5500'
        view, _pm, fig = _build_three_panel_view(primary, [comparison])
        view.on_selection(*ALL_LINES)
        co_entry = next(e for e in view.active_lines if e[3]['molecule_name'] == 'CO')
        assert view.select_line(LineRef(co_entry[3]['lam'], 'CO'))
        assert view.selected_line['molecule_name'] == 'CO'
        assert view._population_diagram_molecule_name() == 'CO'
        plt.close(fig)

    def test_line_inspection_view(self):
        from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView
        view = LineInspectionView(MagicMock())
        ax = Figure().add_subplot(111)   # no pyplot: no GUI backend window
        for lam in (12.0, 12.0, 13.0):
            info = {'lam': lam, 'molecule_name': 'H2O', 'molecule_color': 'green',
                    'up_lev': f"u{len(view.active_lines)}", 'low_lev': 'l'}
            view.active_lines.append([ax.vlines(lam, 0, 1, color='green'),
                                      ax.text(lam, 1, ''), None, info])
        view._display_line_info = lambda value: None
        assert view.select_line(LineRef(12.0, 'H2O', lev_up='u1'))
        assert view.selected_line is view.active_lines[1][3]
        assert _color(view.active_lines[1]) == mcolors.to_hex('orange')
        assert _color(view.active_lines[0]) == mcolors.to_hex('green')
        assert not view.select_line(LineRef(14.0, 'H2O'))


# ---------------------------------------------------------------------------
# iSLATPlot.show_line_in_spectrum
# ---------------------------------------------------------------------------

class _Handler:
    """Stands in for InteractionHandler: a selection runs onselect, as a drag does."""

    def __init__(self, pm):
        self.pm = pm
        self.spans = []
        self.selections = []

    def show_span(self, xmin, xmax):
        self.spans.append((xmin, xmax))

    def set_selection(self, xmin, xmax):
        self.selections.append((xmin, xmax))
        self.pm.onselect(xmin, xmax)


@pytest.fixture
def controller():
    """(pm, view): a real iSLATPlot (built without Tk) driving a real ThreePanelView."""
    from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView

    mol = _make_molecule()
    _stub_view, stub, fig = _build_three_panel_view(mol)
    pm = iSLATPlot.__new__(iSLATPlot)
    pm.fig, pm.ax1, pm.ax2, pm.ax3 = stub.fig, stub.ax1, stub.ax2, stub.ax3
    pm.canvas = MagicMock()
    pm.theme = stub.theme
    pm.islat = stub.islat
    pm.islat.display_range = (10.0, 22.0)
    pm.ax1.set_xlim(10.0, 22.0)
    pm.toggle_state = {
        "atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
        "show_residuals": False, "data_density": False, "current_selection": None,
        "y_autoscale": True, "y_limits": None, "observed": True, "wavelength_markers": [],
    }
    pm.current_selection = None
    pm.fit_result = None
    pm.interaction_handler = _Handler(pm)
    # Moves ax1 as the real one does; the Y policy is not under test here.
    pm.match_display_range = lambda match_y=False: pm.ax1.set_xlim(*pm.islat.display_range)
    view = ThreePanelView(pm)
    pm._three_panel_view = view
    pm.active_view = view
    pm._views = {"Three Panel": view}
    yield pm, view
    plt.close(fig)


class TestShowLineInSpectrum:
    def test_selects_centers_and_highlights_the_line(self, controller):
        pm, view = controller
        pm.ax1.set_xlim(5.0, 25.0)
        pm.onselect(*ALL_LINES)
        strongest = view.selected_line['lam']
        target = next(lam for lam in LINE_LAMS if lam != strongest)

        assert pm.show_line_in_spectrum(LineRef(target, 'H2O', window=ALL_LINES))
        assert view.selected_line['lam'] == pytest.approx(target)
        assert pm.interaction_handler.spans[-1] == ALL_LINES
        assert pm.current_selection == ALL_LINES
        lo, hi = pm.islat.display_range
        assert (lo + hi) / 2 == pytest.approx(target)
        assert hi - lo == pytest.approx(20.0)   # the plotted width is kept
        assert pm.ax1.get_xlim() == pytest.approx((lo, hi))

    def test_window_is_as_wide_as_the_current_selection(self, controller):
        pm, _view = controller
        pm.current_selection = (12.9, 13.1)
        pm.show_line_in_spectrum(LineRef(15.0, 'H2O'))
        assert pm.interaction_handler.selections[-1] == pytest.approx((14.9, 15.1))

    def test_default_window_without_a_selection(self, controller):
        pm, view = controller
        assert pm.show_line_in_spectrum(LineRef(15.0, 'H2O'))
        assert pm.interaction_handler.selections[-1] == pytest.approx(
            (15.0 - DEFAULT_HALF_WINDOW, 15.0 + DEFAULT_HALF_WINDOW))
        assert view.selected_line['lam'] == pytest.approx(15.0)

    def test_a_narrow_view_is_widened_to_fit_the_window(self, controller):
        pm, _view = controller
        pm.ax1.set_xlim(14.99, 15.01)
        pm.show_line_in_spectrum(LineRef(15.0, 'H2O', window=(14.9, 15.1)))
        lo, hi = pm.islat.display_range
        assert hi - lo == pytest.approx(0.4)
        assert (lo + hi) / 2 == pytest.approx(15.0)

    def test_other_views_switch_to_three_panel(self, controller):
        pm, view = controller
        pm._views["Population Diagram"] = SimpleNamespace()
        pm.active_view = pm._views["Population Diagram"]
        switched = []

        def switching(name):
            switched.append(name)
            pm.active_view = pm._views[name]

        pm.switch_view = switching
        assert pm.show_line_in_spectrum(LineRef(13.5, 'H2O'))
        assert switched == ["Three Panel"]
        assert view.selected_line['lam'] == pytest.approx(13.5)

    def test_reports_a_line_that_is_not_inspected(self, controller):
        pm, _view = controller
        insert_text = pm.islat.GUI.data_field.insert_text
        insert_text.reset_mock()
        assert not pm.show_line_in_spectrum(LineRef(14.0, 'CO'))
        messages = [c.args[0] for c in insert_text.call_args_list]
        assert any("CO 14.0000 um is not among the lines inspected" in m for m in messages)
