# -*- coding: utf-8 -*-
"""Line markers of the standalone Line Inspection view: threshold, labels and colors."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pytest

from iSLAT.Modules.GUI.ControlField import EntryField
from iSLAT.Modules.Plotting.LineInspectionContextMixin import DEFAULT_LINE_THRESHOLD
from iSLAT.Modules.Plotting.LineInspectionView import LINE_THRESHOLD_KEY, LineInspectionView

# Two lines in the selection, the second at 80 % of the first.
from tests.plotting.test_line_inspection_canvas_reuse import _CanvasWithWidget, _line_data

canvas_mod = importlib.import_module("iSLAT.Modules.Plotting.OwnedCanvasMixin")


@pytest.fixture(autouse=True)
def agg_canvases(monkeypatch):
    monkeypatch.setattr(canvas_mod, "_FigureCanvasTkAgg", _CanvasWithWidget)


def _make_view(user_settings=None, err_data=None, control_bus=None, theme=None):
    wave = np.linspace(14.0, 15.0, 400)
    islat = SimpleNamespace(
        wave_data=wave, flux_data=1.0 + np.exp(-0.5 * ((wave - 14.3) / 0.01) ** 2), err_data=err_data,
        active_molecule=SimpleNamespace(name="H2O", color="tab:blue"), molecules_dict=None,
        GUI=SimpleNamespace(data_field=MagicMock()),
    )
    if user_settings is not None:
        islat.user_settings = user_settings
    pm = SimpleNamespace(
        islat=islat, theme=theme or {}, toggle_state={"current_selection": (14.2, 14.4)},
        control_bus=control_bus, summed_toggle=False, fit_result=None,
        on_view_canvas_changed=MagicMock(), get_molecule_line_data=_line_data,
    )
    view = LineInspectionView(pm)
    view.activate(MagicMock())
    return view


def _view_with_entry(threshold=0.3):
    bus = MagicMock()
    view = _make_view(user_settings={"line_threshold": threshold}, control_bus=bus)
    fields = [f for call in bus.register_many.call_args_list for f in call.args[0]]
    return view, next(f for f in fields if f.key == LINE_THRESHOLD_KEY)


class TestThreshold:
    def test_markers_and_labels_follow_the_setting(self):
        view = _make_view(user_settings={"line_threshold": 0.9})
        # One marker (80 % < 90 %), and no unthresholded static labels under it.
        assert [t for t in view._plot.ax.texts] == [entry[1] for entry in view.active_lines]
        assert len(view.active_lines) == 1

    def test_default_threshold_without_a_setting(self):
        view = _make_view()
        assert view._get_line_threshold() == DEFAULT_LINE_THRESHOLD
        assert len(view.active_lines) == 2

    def test_entry_sets_the_threshold_for_this_view_only(self):
        view, field = _view_with_entry()
        assert isinstance(field, EntryField) and field.owner is view and field.getter() == 0.3
        field.setter(0.9)
        assert field.getter() == 0.9 and len(view.active_lines) == 1
        assert view._islat.user_settings["line_threshold"] == 0.3
        # A later change of the shared setting does not override it.
        view._islat.user_settings["line_threshold"] = 0.0
        view.on_selection(14.2, 14.4)
        assert len(view.active_lines) == 1

    def test_shared_setting_change_while_hidden_redraws_on_activation(self):
        view = _make_view(user_settings={"line_threshold": 0.3})
        view.deactivate()
        view._islat.user_settings["line_threshold"] = 0.9
        view.activate(MagicMock())
        assert len(view.active_lines) == 1

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_out_of_range_threshold_is_refused(self, bad):
        view, field = _view_with_entry()
        field.setter(bad)
        assert field.getter() == 0.3 and len(view.active_lines) == 2
        view._islat.GUI.data_field.insert_text.assert_called()


def test_error_bars_come_from_err_data():
    err = np.full(400, 0.05)
    np.testing.assert_allclose(_make_view(err_data=err)._plot.error_data, err)


class TestMarkerColors:
    def test_pick_highlights_one_marker_and_resets_the_others(self):
        view = _make_view()
        view._display_line_info = lambda value: None
        first, second = view.active_lines[0][0], view.active_lines[1][0]
        view._on_pick_line(SimpleNamespace(artist=second))
        view._on_pick_line(SimpleNamespace(artist=first))
        assert mcolors.to_hex(first.get_color()[0]) == mcolors.to_hex("orange")
        assert mcolors.to_hex(second.get_color()[0]) == mcolors.to_hex("tab:blue")

    def test_molecule_without_a_color_uses_the_theme_color(self):
        view = _make_view(theme={"active_scatter_line_color": "purple"})
        assert view._molecule_line_color(SimpleNamespace(color=None)) == "purple"
