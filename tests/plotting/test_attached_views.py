# -*- coding: utf-8 -*-
"""iSLATPlot passes model changes to views attached from other windows.

The main window shows one view at a time (``active_view``); views shown in
other windows (e.g. the Display Full Spectrum window) are attached with
``iSLATPlot.attach_view`` and get the same model and molecule events as the
active view, each with its own plot manager's selection.  Display toggles
stay with the main window's active view.
"""

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from iSLAT.Modules.Plotting.PlotView import PlotView


class _RecordingView(PlotView):
    """Records every model event it receives."""

    def __init__(self, pm, stale_after=()):
        super().__init__(pm)
        self.calls = []
        self.stale_after = set(stale_after)
        self._initialised = True
        self._needs_refresh = False

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name in self.stale_after:
            self._needs_refresh = True

    def activate(self, parent_frame):
        pass

    def deactivate(self):
        pass

    def update_model_plot(self, *args, **kwargs):
        self._record("update_model_plot")
        self._needs_refresh = False

    def on_molecule_visibility_changed(self, **kwargs):
        self._record("visibility", **kwargs)

    def on_active_molecule_changed(self, new_molecule=None, current_selection=None):
        self._record("active", new_molecule=new_molecule, current_selection=current_selection)

    def on_molecule_parameter_changed(self, molecule_name, parameter_name, current_selection=None):
        self._record("parameter", molecule_name=molecule_name, parameter_name=parameter_name,
                     current_selection=current_selection)

    def on_molecule_deleted(self, molecule_name):
        self._record("deleted", molecule_name=molecule_name)

    def toggle_atomic_lines(self, show):
        self._record("atomic_lines", show=show)

    def apply_theme(self, theme):
        pass

    def draw(self):
        pass

    def get_canvas(self):
        return None

    def get_figure(self):
        return None

    def names(self):
        return [name for name, _ in self.calls]


def _make_main():
    """An iSLATPlot without its Tk-heavy constructor, with recording views."""
    from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
    pm = iSLATPlot.__new__(iSLATPlot)
    pm.islat = SimpleNamespace(wave_data=np.linspace(5, 10, 11), active_molecule="H2O",
                               molecules_dict=MagicMock())
    pm.current_selection = (6.0, 7.0)
    pm.toggle_state = {"atomic_lines": False}
    pm.active_view = _RecordingView(pm)
    pm._views = {"Main": pm.active_view}
    return pm


def _window_view(selection=None, **kwargs):
    window_pm = SimpleNamespace(current_selection=selection, toggle_state={})
    return _RecordingView(window_pm, **kwargs)


class TestDispatch:
    @pytest.mark.parametrize("trigger, event", [
        (lambda pm: pm.update_model_plot(), "update_model_plot"),
        (lambda pm: pm.on_molecule_parameter_changed("H2O", "temp", 1, 2), "parameter"),
        (lambda pm: pm.on_molecule_deleted("H2O"), "deleted"),
        (lambda pm: pm.on_molecule_visibility_changed("H2O", False), "visibility"),
        (lambda pm: pm.on_active_molecule_changed(), "active"),
    ])
    def test_model_events_reach_active_and_attached_views(self, trigger, event):
        pm = _make_main()
        attached = _window_view()
        pm.attach_view(attached)
        trigger(pm)
        assert pm.active_view.names() == [event]
        assert attached.names() == [event]

    def test_each_view_gets_its_own_selection(self):
        pm = _make_main()
        attached = _window_view(selection=(8.0, 9.0))
        pm.attach_view(attached)
        pm.on_molecule_parameter_changed("H2O", "temp", 1, 2)
        assert pm.active_view.calls[0][1]["current_selection"] == (6.0, 7.0)
        assert attached.calls[0][1]["current_selection"] == (8.0, 9.0)

    def test_display_toggles_stay_with_the_main_window(self):
        pm = _make_main()
        attached = _window_view()
        pm.attach_view(attached)
        pm.toggle_atomic_lines(True)
        assert "atomic_lines" in pm.active_view.names()
        assert attached.names() == []

    def test_is_visible_parameter_is_left_to_the_visibility_event(self):
        pm = _make_main()
        attached = _window_view()
        pm.attach_view(attached)
        pm.on_molecule_parameter_changed("H2O", "is_visible", True, False)
        assert attached.names() == []


class TestStaleAttachedViews:
    def test_stale_attached_view_is_rebuilt_at_once(self):
        """An attached view is on screen: being marked stale must not wait for activate()."""
        pm = _make_main()
        attached = _window_view(stale_after={"deleted"})
        pm.attach_view(attached)
        pm.on_molecule_deleted("H2O")
        assert attached.names() == ["deleted", "update_model_plot"]
        assert attached._needs_refresh is False

    def test_attached_view_is_not_marked_stale_like_inactive_views(self):
        pm = _make_main()
        inactive = _RecordingView(pm)
        pm._views["Other"] = inactive
        attached = _window_view()
        pm.attach_view(attached)
        pm.update_model_plot()
        assert inactive._needs_refresh is True
        assert attached._needs_refresh is False


class TestAttachDetach:
    def test_attach_twice_dispatches_once(self):
        pm = _make_main()
        attached = _window_view()
        pm.attach_view(attached)
        pm.attach_view(attached)
        pm.update_model_plot()
        assert attached.names() == ["update_model_plot"]

    def test_detach_stops_events(self):
        pm = _make_main()
        attached = _window_view()
        pm.attach_view(attached)
        pm.detach_view(attached)
        pm.update_model_plot()
        assert attached.names() == []
        assert pm.attached_views == []

    def test_attached_view_is_held_weakly(self):
        pm = _make_main()
        pm.attach_view(_window_view())
        gc.collect()
        assert pm.attached_views == []
        pm.update_model_plot()  # a dead reference is skipped

    def test_active_view_attached_is_not_called_twice(self):
        pm = _make_main()
        pm.attach_view(pm.active_view)
        pm.update_model_plot()
        assert pm.active_view.names() == ["update_model_plot"]

    def test_failing_attached_view_does_not_break_the_others(self):
        pm = _make_main()
        broken = _window_view()
        broken.update_model_plot = MagicMock(side_effect=RuntimeError("boom"))
        healthy = _window_view()
        pm.attach_view(broken)
        pm.attach_view(healthy)
        pm.update_model_plot()
        assert pm.active_view.names() == ["update_model_plot"]
        assert healthy.names() == ["update_model_plot"]
