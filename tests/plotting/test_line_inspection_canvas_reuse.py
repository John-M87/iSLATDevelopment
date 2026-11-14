# -*- coding: utf-8 -*-
"""The Line Inspection view redraws into the figure and canvas it already has.

A new selection, a parameter change or a rebuild while the view was hidden
clears the view's figure and draws the new plot into it, instead of making a
new figure.  The canvas showing the figure is therefore kept: a new canvas
widget has to wait for Tk to lay it out before its first draw, which made each
selection about 0.4 s slower.

Because the canvas is kept, the view tells the plot manager when it shows new
content (so the toolbar drops zoom history from the previous selection and
sets its pan / zoom mode on the new axes), and its pick handler stays
connected.  The new axes starts from scratch, so nothing set on the previous
selection's axes carries over.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView

# The package re-exports classes under their module names, so go via importlib.
canvas_mod = importlib.import_module("iSLAT.Modules.Plotting.OwnedCanvasMixin")


class _CanvasWithWidget(FigureCanvasAgg):
    """An Agg canvas with the Tk-widget accessor the view packs."""

    def __init__(self, figure, master=None):
        super().__init__(figure)
        self._widget = MagicMock()

    def get_tk_widget(self):
        return self._widget


@pytest.fixture
def made_canvases(monkeypatch):
    """Build view canvases as Agg canvases; return the list each new one is added to."""
    made = []

    def make_canvas(figure, master=None):
        canvas = _CanvasWithWidget(figure, master)
        made.append(canvas)
        return canvas

    monkeypatch.setattr(canvas_mod, "_FigureCanvasTkAgg", make_canvas)
    return made


def _line(lam):
    return SimpleNamespace(lam=lam, e_up=1000.0, e_low=900.0, a_stein=0.5,
                           g_up=3, g_low=1, lev_up="u", lev_low="l")


def _line_data(mol, xmin, xmax):
    """Two marker lines inside the selection, standing in for the controller's lookup."""
    mid = 0.5 * (xmin + xmax)
    return [(_line(mid - 0.01), 1.0, 0.1), (_line(mid + 0.01), 0.8, 0.05)]


@pytest.fixture
def view(made_canvases):
    wave = np.linspace(14.0, 15.0, 400)
    islat = SimpleNamespace(
        wave_data=wave,
        flux_data=1.0 + np.exp(-0.5 * ((wave - 14.5) / 0.01) ** 2),
        error_data=None,
        active_molecule=SimpleNamespace(name="H2O", color="tab:blue"),
        molecules_dict=None,
    )
    pm = SimpleNamespace(
        islat=islat, theme={}, toggle_state={"current_selection": (14.2, 14.4)},
        control_bus=None, summed_toggle=False, fit_result=None,
        on_view_canvas_changed=MagicMock(),
        get_molecule_line_data=_line_data,
    )
    view = LineInspectionView(pm)
    view.activate(MagicMock())
    return view


def _canvas_changes(view):
    return view._pm.on_view_canvas_changed.call_count


class TestCanvasIsKept:
    def test_new_selection_draws_into_the_same_figure_and_canvas(self, view, made_canvases):
        fig, canvas = view._fig, view.get_canvas()
        view.on_selection(14.6, 14.8)
        assert view._fig is fig
        assert view.get_canvas() is canvas
        assert made_canvases == [canvas]

    def test_parameter_change_and_clear_selection_keep_the_canvas(self, view, made_canvases):
        canvas = view.get_canvas()
        view.on_molecule_parameter_changed("H2O", "temp")
        view.clear_selection()
        assert view.get_canvas() is canvas and made_canvases == [canvas]

    def test_figure_shows_only_the_new_selection(self, view):
        view.on_selection(14.6, 14.8)
        assert view._fig.axes == [view._plot.ax]
        assert view._plot.ax.get_xlim() == pytest.approx((14.6, 14.8))
        lams = [entry[3]["lam"] for entry in view.active_lines]
        assert lams == pytest.approx([14.69, 14.71])

    def test_state_of_the_previous_axes_does_not_carry_over(self, view):
        old_ax = view._plot.ax
        old_ax.set_yscale("log")
        old_ax.set_autoscaley_on(False)
        view.on_selection(14.6, 14.8)
        new_ax = view._plot.ax
        assert new_ax is not old_ax
        assert new_ax.get_yscale() == "linear"

    def test_rebuild_while_hidden_reuses_the_canvas_on_activation(self, view, made_canvases):
        canvas = view.get_canvas()
        view.deactivate()
        view._pm.toggle_state["current_selection"] = (14.6, 14.8)
        view.activate(MagicMock())
        assert view.get_canvas() is canvas and made_canvases == [canvas]
        assert view._plot.ax.get_xlim() == pytest.approx((14.6, 14.8))


class TestToolbarIsTold:
    def test_each_redraw_reports_new_content(self, view):
        before = _canvas_changes(view)
        view.on_selection(14.6, 14.8)
        view.on_molecule_parameter_changed("H2O", "temp")
        assert _canvas_changes(view) == before + 2

    def test_rebuild_on_activation_reports_new_content(self, view):
        view.deactivate()
        view._pm.toggle_state["current_selection"] = (14.6, 14.8)
        before = _canvas_changes(view)
        view.activate(MagicMock())
        assert _canvas_changes(view) == before + 1

    def test_unchanged_activation_reports_nothing(self, view):
        view.deactivate()
        before = _canvas_changes(view)
        view.activate(MagicMock())
        assert _canvas_changes(view) == before


class TestPickingAfterRedraw:
    def test_clicking_a_marker_of_the_new_selection_highlights_it(self, view):
        view._display_line_info = lambda value: None
        view.on_selection(14.6, 14.8)
        vline = view.active_lines[1][0]
        view.get_canvas().callbacks.process("pick_event", SimpleNamespace(artist=vline))
        assert view.selected_line is view.active_lines[1][3]
        assert mcolors.to_hex(vline.get_color()[0]) == mcolors.to_hex("orange")
