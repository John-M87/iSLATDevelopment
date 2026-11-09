# -*- coding: utf-8 -*-
"""Switching back to a view that nothing changed in does not re-sync or redraw it.

Tk keeps a hidden canvas's image and re-packing it at the same size does not
redraw it, so a view whose figure was not modified while hidden
(``Figure.stale``), which is not marked for a refresh and whose inputs
(``PlotView._display_inputs``: toggle state, theme, molecules, data, fit
result, plus per-view extras) are unchanged, only re-packs its canvas and
re-registers its controls.  Anything else takes the full activation path.
"""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import MagicMock
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting import (
    LineInspectionView, PopulationDiagramView, ThreePanelView,
)

from tests.plotting import make_line_list
from tests.plotting.test_full_spectrum_overlays import _make_view as _make_fs_view, _render as _render_fs


class _Recorder:
    """Replaces the expensive activation steps; records which ran."""

    def __init__(self, view, names):
        self.calls = []
        for name in names:
            setattr(view, name, self._make(name))

    def _make(self, name):
        def record(*_args, **_kwargs):
            self.calls.append(name)
        return record


def _count_draws(fig):
    draws = []
    original = fig.draw

    def counting(renderer):
        draws.append(1)
        return original(renderer)

    fig.draw = counting
    return draws


# ======================================================================
# Full Spectrum view (real plot, Agg canvas)
# ======================================================================

@pytest.fixture
def fs(tmp_path):
    csv = tmp_path / "lines.csv"
    make_line_list([12.5, 15.5]).to_csv(csv, index=False)
    view, pm, islat = _make_fs_view(csv)
    _render_fs(view)
    view._canvas = view.fig.canvas
    view.fig.canvas.draw()
    yield view, pm, islat, csv
    view._canvas = None  # an Agg canvas has no Tk widget to destroy
    view.destroy()


def _round_trip(view, change=None):
    """Hide the view, apply *change*, show it again; return (synced, draws)."""
    view.deactivate()
    if change is not None:
        change()
    synced = []
    original_sync = view.sync_toggle_state
    view.sync_toggle_state = lambda state: (synced.append(1), original_sync(state))[1]
    draws = _count_draws(view.fig)
    view.activate(None)
    del view.sync_toggle_state
    return bool(synced), len(draws)


class TestFullSpectrumReactivation:
    def test_unchanged_view_is_not_redrawn(self, fs):
        view, *_ = fs
        assert _round_trip(view) == (False, 0)

    def test_figure_changed_while_hidden(self, fs):
        view, *_ = fs
        synced, draws = _round_trip(view, lambda: view.subplots[0].set_title("changed"))
        assert synced and draws >= 1

    def test_marked_for_refresh(self, fs):
        view, *_ = fs
        synced, _ = _round_trip(view, lambda: setattr(view, "_needs_refresh", True))
        assert synced

    def test_toggle_state_changed(self, fs):
        view, pm, *_ = fs
        synced, _ = _round_trip(view, lambda: pm.toggle_state.update(summed=False))
        assert synced

    def test_selection_change_is_ignored(self, fs):
        # The view hands selections on but does not draw them.
        view, pm, *_ = fs
        assert _round_trip(view, lambda: pm.toggle_state.update(current_selection=(14.0, 14.2))) == (False, 0)

    def test_theme_changed(self, fs):
        view, pm, *_ = fs
        synced, _ = _round_trip(view, lambda: setattr(pm, "theme", {"foreground": "#123456"}))
        assert synced

    def test_wavelength_marker_added(self, fs):
        view, pm, *_ = fs
        marker = {"start": 13.0, "end": 13.0, "label": "", "color": "red"}
        synced, _ = _round_trip(view, lambda: pm.toggle_state["wavelength_markers"].append(marker))
        assert synced

    def test_molecule_hidden(self, fs):
        view, _pm, islat, _ = fs
        mol = next(iter(islat.molecules_dict.values()))
        synced, _ = _round_trip(view, lambda: setattr(mol, "is_visible", False))
        assert synced

    def test_new_data_arrays(self, fs):
        view, _pm, islat, _ = fs
        synced, _ = _round_trip(view, lambda: setattr(islat, "flux_data", np.array(islat.flux_data)))
        assert synced

    def test_line_list_edited(self, fs):
        view, _pm, _islat, csv = fs

        def edit():
            make_line_list([12.5, 15.5, 17.5]).to_csv(csv, index=False)
            st = os.stat(csv)
            os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))

        synced, _ = _round_trip(view, edit)
        assert synced

    def test_remembered_state_is_used_once(self, fs):
        view, *_ = fs
        view.deactivate()
        assert view._shown_frame_is_current()
        assert not view._shown_frame_is_current()

    def test_activation_without_a_previous_hide(self, fs):
        view, *_ = fs
        assert not view._shown_frame_is_current()


# ======================================================================
# Three Panel view (controller stand-in; the expensive steps are recorded)
# ======================================================================

class _CanvasWithWidget(FigureCanvasAgg):
    """An Agg canvas with the Tk-widget accessor ThreePanelView packs."""

    def __init__(self, figure):
        super().__init__(figure)
        self._widget = MagicMock()

    def get_tk_widget(self):
        return self._widget


@pytest.fixture
def tpv():
    fig = Figure()
    canvas = _CanvasWithWidget(fig)
    islat = MagicMock()
    islat.molecules_dict = {}
    islat.active_molecule = None
    islat.comparison_molecules = []
    islat.user_settings = {"line_threshold": 0.3}
    pm = MagicMock()
    pm.fig = fig
    pm.canvas = canvas
    pm.islat = islat
    pm.theme = {"foreground": "white"}
    pm.toggle_state = {"summed": True, "current_selection": (14.0, 14.1)}
    pm.fit_result = None
    pm.control_bus = None
    view = ThreePanelView(pm)
    view._needs_refresh = False
    rec = _Recorder(view, ("apply_theme", "sync_toggle_state", "_restore_line_selection", "_do_update_model_plot"))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    canvas.draw()
    return view, pm, rec


class TestThreePanelReactivation:
    def test_unchanged_view_skips_theme_sync_and_selection(self, tpv):
        view, pm, rec = tpv
        view.deactivate()
        view.activate(None)
        assert rec.calls == []
        pm.canvas.get_tk_widget().pack.assert_called()

    def test_first_activation_takes_the_full_path(self, tpv):
        view, _pm, rec = tpv
        view.activate(None)
        assert rec.calls == ["apply_theme", "sync_toggle_state", "_restore_line_selection"]

    @pytest.mark.parametrize("change", [
        lambda view, pm: pm.toggle_state.update(current_selection=(15.0, 15.1)),
        lambda view, pm: pm.islat.user_settings.update(line_threshold=0.6),
        lambda view, pm: setattr(pm, "fit_result", object()),
        lambda view, pm: setattr(pm.islat, "comparison_molecules", [MagicMock(name="CO")]),
        lambda view, pm: pm.fig.axes[0].set_xlim(0.2, 0.8),
    ], ids=["selection", "line-threshold", "fit-result", "comparison-molecules", "figure-changed"])
    def test_changed_while_hidden_takes_the_full_path(self, tpv, change):
        view, pm, rec = tpv
        view.deactivate()
        change(view, pm)
        view.activate(None)
        assert "sync_toggle_state" in rec.calls

    def test_stale_view_rerenders(self, tpv):
        view, _pm, rec = tpv
        view.deactivate()
        view._needs_refresh = True
        view.activate(None)
        assert "_do_update_model_plot" in rec.calls


# ======================================================================
# Population Diagram and Line Inspection views
# ======================================================================

def _owned_view(cls):
    islat = MagicMock()
    islat.molecules_dict = {}
    islat.active_molecule = None
    islat.comparison_molecules = []
    pm = MagicMock()
    pm.islat = islat
    pm.theme = {"foreground": "white"}
    pm.toggle_state = {"summed": True}
    pm.fit_result = None
    pm.control_bus = None
    view = cls(pm)
    view._fig = Figure()
    view._fig.add_subplot(111).plot([0, 1], [0, 1])
    view._canvas = FigureCanvasAgg(view._fig)
    view._canvas.draw()
    view._initialised = True
    view._needs_refresh = False
    rec = _Recorder(view, ("_build_plot", "apply_theme", "sync_toggle_state", "draw"))
    return view, pm, rec


@pytest.mark.parametrize("cls", [PopulationDiagramView, LineInspectionView])
class TestOwnedSinglePanelViews:
    def test_unchanged_view_is_not_redrawn(self, cls):
        view, _pm, rec = _owned_view(cls)
        view.deactivate()
        draws = _count_draws(view._fig)
        view.activate(None)
        assert rec.calls == [] and draws == []

    def test_changed_view_takes_the_full_path(self, cls):
        view, pm, rec = _owned_view(cls)
        view.deactivate()
        pm.theme = {"foreground": "black"}
        view.activate(None)
        assert "apply_theme" in rec.calls and "draw" in rec.calls
