# -*- coding: utf-8 -*-
"""Shared view plumbing: the owned-canvas mixin, ToggleMixin syncing, the
save-figure template, data-field messages and the shared debug logger."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting.FigureExport import default_export_dpi, export_format, save_figure_file
from iSLAT.Modules.Plotting.PlotView import PlotView, default_save_name, notify_user
from iSLAT.Modules.Plotting.ToggleMixin import ToggleMixin

# The package re-exports classes under their module names, so go via importlib.
canvas_mod = importlib.import_module("iSLAT.Modules.Plotting.OwnedCanvasMixin")
OwnedCanvasMixin = canvas_mod.OwnedCanvasMixin


class _StubView(PlotView):
    """Fills in the PlotView abstract methods the tests do not care about."""

    def activate(self, parent_frame):
        self._parent_frame = parent_frame

    def deactivate(self):
        pass

    def update_model_plot(self, *args, **kwargs):
        pass

    def on_molecule_visibility_changed(self, *args, **kwargs):
        pass

    def on_active_molecule_changed(self, *args, **kwargs):
        pass

    def on_molecule_parameter_changed(self, *args, **kwargs):
        pass

    def on_molecule_deleted(self, *args, **kwargs):
        pass

    def apply_theme(self, theme):
        pass


def _pm(**extra):
    defaults = dict(islat=None, theme={}, toggle_state={}, control_bus=MagicMock(),
                    on_view_canvas_changed=MagicMock())
    defaults.update(extra)
    return SimpleNamespace(**defaults)


# ======================================================================
# OwnedCanvasMixin
# ======================================================================
class _FakeCanvas:
    def __init__(self, figure, master=None):
        self.figure, self.master = figure, master
        self.connected, self.disconnected = [], []
        self.widget = MagicMock()

    def mpl_connect(self, name, callback):
        self.connected.append((name, callback))
        return len(self.connected)

    def mpl_disconnect(self, cid):
        self.disconnected.append(cid)

    def get_tk_widget(self):
        return self.widget

    def draw_idle(self):
        pass


class _CanvasView(OwnedCanvasMixin, _StubView):
    def __init__(self, pm, fig=None):
        super().__init__(pm)
        self.fig = fig
        self.created = []

    def get_figure(self):
        return self.fig

    def _on_canvas_created(self, canvas):
        self.created.append(canvas)


@pytest.fixture
def canvas_view(monkeypatch):
    monkeypatch.setattr(canvas_mod, "_FigureCanvasTkAgg", _FakeCanvas)
    pm = _pm()
    view = _CanvasView(pm, Figure())
    view._parent_frame = object()
    return view, pm


class TestOwnedCanvasMixin:
    def test_one_canvas_per_figure(self, canvas_view):
        view, pm = canvas_view
        view._ensure_canvas()
        first = view.get_canvas()
        assert first.figure is view.fig and first.master is view._parent_frame
        assert [name for name, _cb in first.connected] == ["button_press_event"]
        view._ensure_canvas()
        assert view.get_canvas() is first
        assert view.created == [first]
        pm.on_view_canvas_changed.assert_called_once_with(view)

    def test_new_figure_replaces_the_canvas(self, canvas_view):
        view, pm = canvas_view
        view._ensure_canvas()
        first = view.get_canvas()
        view.fig = Figure()
        view._ensure_canvas()
        second = view.get_canvas()
        assert second is not first and second.figure is view.fig
        assert first.disconnected == [1]           # right-click handler dropped
        first.widget.destroy.assert_called_once()
        assert view.created == [first, second]
        assert pm.on_view_canvas_changed.call_count == 2

    def test_no_canvas_without_parent_frame_or_figure(self, canvas_view):
        view, pm = canvas_view
        view._parent_frame = None
        view._ensure_canvas()
        assert view.get_canvas() is None
        view._parent_frame, view.fig = object(), None
        view._ensure_canvas()
        assert view.get_canvas() is None
        pm.on_view_canvas_changed.assert_not_called()

    def test_show_canvas_if_replaced_only_repacks_new_figures(self, canvas_view):
        view, _pm_ = canvas_view
        view._show_canvas()
        canvas = view.get_canvas()
        canvas.widget.pack.assert_called_once_with(fill="both", expand=True, padx=0, pady=0)
        view._show_canvas_if_replaced(view.fig)
        canvas.widget.pack.assert_called_once()
        old_fig, view.fig = view.fig, Figure()
        view._show_canvas_if_replaced(old_fig)
        assert view.get_canvas() is not canvas
        view.get_canvas().widget.pack.assert_called_once()

    def test_deactivate_unregisters_and_unpacks(self, canvas_view):
        view, pm = canvas_view
        view._show_canvas()
        view.deactivate()
        pm.control_bus.unregister_owner.assert_called_once_with(view)
        view.get_canvas().widget.pack_forget.assert_called_once()

    def test_right_click_pops_the_menu_up_at_the_mouse(self, canvas_view):
        view, _pm_ = canvas_view
        view._ensure_canvas()
        widget = view.get_canvas().widget
        widget.winfo_rootx.return_value = 100
        widget.winfo_rooty.return_value = 200
        widget.winfo_height.return_value = 50
        menu = MagicMock()
        view.build_context_menu = MagicMock(return_value=menu)
        view._on_canvas_button_press(SimpleNamespace(button=1, x=10, y=20))
        view.build_context_menu.assert_not_called()
        event = SimpleNamespace(button=3, x=10, y=20)
        view._on_canvas_button_press(event)
        view.build_context_menu.assert_called_once_with(event, widget)
        menu.tk_popup.assert_called_once_with(110, 230)    # matplotlib y is measured from the bottom
        menu.grab_release.assert_called_once()

    def test_popup_menu_reports_failure_and_releases_the_grab(self):
        menu = MagicMock()
        menu.tk_popup.side_effect = RuntimeError("no display")
        widget = MagicMock()
        widget.winfo_rootx.return_value = widget.winfo_rooty.return_value = 0
        widget.winfo_height.return_value = 10
        assert canvas_mod.popup_menu(menu, widget, SimpleNamespace(x=1, y=2)) is False
        menu.grab_release.assert_called_once()

    @pytest.mark.parametrize("module_name", [
        "FullSpectrumView", "OpticalDepthView", "LineInspectionView", "PopulationDiagramView",
    ])
    def test_views_use_the_shared_canvas_code(self, module_name):
        cls = getattr(importlib.import_module(f"iSLAT.Modules.Plotting.{module_name}"), module_name)
        assert issubclass(cls, OwnedCanvasMixin)
        for method in ("_ensure_canvas", "_destroy_canvas", "_on_canvas_button_press", "draw", "get_canvas"):
            assert getattr(cls, method) is getattr(OwnedCanvasMixin, method), method

    def test_line_inspection_deactivate_drops_the_pick_handler(self, monkeypatch):
        from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView
        monkeypatch.setattr(canvas_mod, "_FigureCanvasTkAgg", _FakeCanvas)
        pm = _pm(islat=SimpleNamespace(active_molecule=None, molecules_dict=None))
        view = LineInspectionView(pm)
        view._fig, view._parent_frame = Figure(), object()
        view._ensure_canvas()
        canvas = view.get_canvas()
        assert [name for name, _cb in canvas.connected] == ["button_press_event", "pick_event"]
        view.deactivate()
        assert canvas.disconnected == [2] and view._pick_cid is None
        pm.control_bus.unregister_owner.assert_called_once_with(view)
        canvas.widget.pack_forget.assert_called_once()


# ======================================================================
# ToggleMixin.sync_toggle_state
# ======================================================================
class _ToggleView(ToggleMixin, _StubView):
    """Records the overlay calls ToggleMixin makes; *drawn* answers _overlay_drawn."""

    def __init__(self, drawn=None):
        super().__init__(_pm())
        self.calls = []
        self.drawn = drawn or {}
        self.line_data = None

    def _overlay_drawn(self, kind):
        return self.drawn.get(kind)   # None (the ToggleMixin default) unless given

    def _toggle_ready(self):
        return True

    def _iter_toggle_axes(self):
        return iter(())

    def _add_atomic_line_artists(self):
        self.calls.append("add atomic")

    def _remove_atomic_line_artists(self):
        self.calls.append("remove atomic")

    def _add_saved_line_artists(self):
        self.calls.append(f"add saved from {self.line_data}")

    def _remove_saved_line_artists(self):
        self.calls.append("remove saved")

    def _load_saved_line_data(self):
        self.calls.append("load saved")
        return "disk"

    def draw(self):
        self.calls.append("draw")

    def get_figure(self):
        return None

    def get_canvas(self):
        return None


class TestToggleMixinSync:
    def test_unknown_state_redraws_and_reloads_saved_lines(self):
        view = _ToggleView()
        view.sync_toggle_state({"atomic_lines": True, "saved_lines": True})
        assert view.calls == ["remove atomic", "add atomic",
                              "remove saved", "load saved", "add saved from disk", "draw"]

    def test_unknown_state_removes_unwanted_overlays(self):
        view = _ToggleView()
        view.sync_toggle_state({"atomic_lines": False, "saved_lines": False})
        assert view.calls == ["remove atomic", "remove saved", "draw"]

    def test_known_state_only_changes_what_differs(self):
        view = _ToggleView(drawn={"atomic_lines": True, "saved_lines": False})
        view.sync_toggle_state({"atomic_lines": True, "saved_lines": True})
        assert view.calls == ["load saved", "add saved from disk", "draw"]
        view = _ToggleView(drawn={"atomic_lines": True, "saved_lines": False})
        view.sync_toggle_state({"atomic_lines": False, "saved_lines": False})
        assert view.calls == ["remove atomic", "draw"]


# ======================================================================
# Three Panel view: re-renders keep the toggle state
# ======================================================================
SAVED_LINES = pd.DataFrame({"lam": [15.0], "xmin": [14.99], "xmax": [15.01], "species": ["H2O"]})


@pytest.fixture
def three_panel(monkeypatch):
    import matplotlib.pyplot as plt
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.Plotting.MainPlotGrid import MainPlotGrid
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
    from tests.plotting import make_wave_flux
    from tests.plotting.test_plot_views import _make_test_molecule

    tpv_mod = importlib.import_module("iSLAT.Modules.Plotting.ThreePanelView")
    monkeypatch.setattr(tpv_mod, "load_atomic_lines",
                        lambda: pd.DataFrame({"wave": [15.0], "species": ["Ne II"], "line": [""]}))
    fig = plt.figure(figsize=(12, 8))
    ax1, ax2, ax3 = MainPlotGrid.create_three_panel_axes(fig)
    wave, flux, err = make_wave_flux(n=200)
    mol = _make_test_molecule()
    md = MoleculeDict()
    md[mol.name] = mol
    md.apply_stellar_rv = lambda w: w
    islat = SimpleNamespace(wave_data=wave, wave_data_original=wave.copy(), flux_data=flux,
                            err_data=err, molecules_dict=md, active_molecule=mol,
                            comparison_molecules=[], input_line_list="lines.csv")
    pm = MagicMock()
    pm.fig, pm.ax1, pm.ax2, pm.ax3 = fig, ax1, ax2, ax3
    pm.theme = {}
    pm.islat = islat
    pm.toggle_state = {"atomic_lines": True, "saved_lines": True, "summed": False, "legend": True}
    view = ThreePanelView(pm)
    view._load_saved_line_data = lambda: SAVED_LINES
    yield view, ax1
    plt.close(fig)


def _overlays(ax):
    return {
        "atomic": sum(1 for a in ax.lines if hasattr(a, "_islat_atomic_line")),
        "saved": sum(1 for a in ax.lines if hasattr(a, "_islat_saved_line")),
        "summed": [c.get_visible() for c in ax.collections if hasattr(c, "_islat_summed")],
    }


class TestThreePanelRerenderKeepsToggles:
    def test_full_update_hides_the_summed_fill_so_it_can_come_back(self, three_panel):
        view, ax1 = three_panel
        view._do_update_model_plot()
        state = _overlays(ax1)
        assert state["atomic"] == 1 and state["saved"] > 0
        assert state["summed"] == [False]
        view.toggle_summed_spectrum(True)
        assert _overlays(ax1)["summed"] == [True]

    def test_parameter_change_keeps_overlays_and_summed_setting(self, three_panel):
        view, ax1 = three_panel
        view._do_update_model_plot()
        view._update_spectrum_panel_only()
        state = _overlays(ax1)
        assert state["atomic"] == 1 and state["saved"] > 0
        assert state["summed"] == [False]


# ======================================================================
# Line Grid view on ToggleMixin
# ======================================================================
@pytest.fixture
def line_grid(monkeypatch):
    from iSLAT.Modules.Plotting.FitLinesPlotGridView import FitLinesPlotGridView
    flv_mod = importlib.import_module("iSLAT.Modules.Plotting.FitLinesPlotGridView")
    atomic = pd.DataFrame({"wave": [15.0], "species": ["Ne II"], "line": [""]})
    monkeypatch.setattr(flv_mod, "load_atomic_lines", lambda: atomic)
    monkeypatch.setattr(flv_mod.ifh, "read_line_saves", lambda file_name=None: SAVED_LINES)
    ax = Figure().add_subplot(111)
    ax.set_ylim(1.0, 2.0)
    (observed,) = ax.plot([14.0, 16.0], [1.2, 1.8])
    observed._islat_observed = True
    panel = MagicMock()
    panel.ax = ax
    grid = MagicMock()
    grid.iter_panels.return_value = [panel]
    pm = _pm(islat=SimpleNamespace(wave_data=None, molecules_dict=None, input_line_list="lines.csv"))
    view = FitLinesPlotGridView(pm)
    view._plot_grids = [grid]
    return view, panel, ax, observed, atomic


class TestLineGridToggles:
    def test_atomic_lines_are_drawn(self, line_grid):
        view, panel, *_rest, atomic = line_grid
        view.toggle_atomic_lines(True)
        panel.plot_atomic_lines.assert_called_once_with(atomic)
        view.toggle_atomic_lines(False)
        panel.remove_atomic_lines.assert_called_once()

    def test_saved_lines_load_from_disk_when_not_given(self, line_grid):
        view, panel, *_rest = line_grid
        view.toggle_saved_lines(True)
        assert panel.plot_saved_lines.call_args.args[0] is SAVED_LINES

    def test_main_plot_y_and_observed_settings_leave_the_panels_alone(self, line_grid):
        view, _panel, ax, observed, _atomic = line_grid
        view.sync_toggle_state({"y_autoscale": False, "y_limits": (0.0, 9.0), "observed": False})
        view.toggle_observed_data(False)
        assert ax.get_ylim() == (1.0, 2.0)
        assert observed.get_visible()


# ======================================================================
# Save template and figure export
# ======================================================================
class _LiveFigureView(_StubView):
    """A view that saves the figure it shows (the PlotView default)."""

    def __init__(self, fig, **pm_extra):
        super().__init__(_pm(**pm_extra))
        self.fig = fig

    def draw(self):
        pass

    def get_canvas(self):
        return None

    def get_figure(self):
        return self.fig


def _figure():
    fig = Figure()
    fig.add_subplot(111).plot([1, 2, 3], [3, 1, 2])
    return fig


class TestSaveTemplate:
    def test_saves_the_live_figure_in_the_extension_format(self, tmp_path):
        view = _LiveFigureView(_figure())
        out = tmp_path / "view.png"
        assert view.save_figure(str(out), file_format="pdf", dpi=20) == str(out)
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_rasterizing_does_not_stick_to_the_live_figure(self, tmp_path):
        fig = _figure()
        view = _LiveFigureView(fig)
        view.save_figure(str(tmp_path / "view.pdf"), rasterized=True)
        assert not fig.axes[0].get_rasterized()

    def test_nothing_to_save_or_cancelled_dialog(self, tmp_path, monkeypatch):
        import tkinter.filedialog
        asked = []
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **k: asked.append(k) or "")
        assert _LiveFigureView(None).save_figure() is None
        assert asked == []                          # no figure: the dialog is not even opened
        assert _LiveFigureView(_figure()).save_figure() is None
        assert asked and asked[0]["title"] == "Save Figure"

    def test_export_figure_hook_message_and_release(self, tmp_path):
        export_fig, released = _figure(), []
        data_field = MagicMock()

        class _ExportView(_LiveFigureView):
            SAVE_MESSAGE = "Saved to {path}"

            def _build_export_figure(self):
                return export_fig, lambda: released.append(True)

        view = _ExportView(_figure(), islat=SimpleNamespace(GUI=SimpleNamespace(data_field=data_field)))
        out = tmp_path / "export.png"
        assert view.save_figure(str(out), file_format="png", dpi=20) == str(out)
        assert released == [True]
        data_field.insert_text.assert_called_once_with(f"Saved to {out}")

    def test_dialog_settings_of_each_view(self, monkeypatch):
        import tkinter.filedialog
        from iSLAT.Modules.FileHandling import absolute_data_files_path
        from iSLAT.Modules.GUI.FullSpectrumWindow import WindowFullSpectrumView
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        from iSLAT.Modules.Plotting.OpticalDepthView import OpticalDepthView
        asked = []
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **k: asked.append(k) or "")
        pm = _pm(islat=SimpleNamespace(loaded_spectrum_file="data/spec.csv"))
        for cls in (FullSpectrumView, OpticalDepthView, WindowFullSpectrumView):
            cls(pm)._ask_save_path("png")
        fsv, odv, window = asked
        assert (fsv["title"], fsv["initialfile"], fsv["initialdir"]) == (
            "Save Spectrum Output", "spec_full_output.png", absolute_data_files_path)
        assert (odv["title"], odv["initialfile"], odv["initialdir"]) == (
            "Save Optical Depth Figure", "spec_optical_depth.png", absolute_data_files_path)
        assert fsv["filetypes"] == [("PNG files", "*.png")]
        assert window["initialfile"] == "spec_full_output.png" and "initialdir" not in window
        assert window["filetypes"] == [("PNG files", "*.png"), ("All files", "*.*")]

    def test_default_save_name_without_a_spectrum(self):
        assert default_save_name(SimpleNamespace(loaded_spectrum_file="a/b.csv"), "x", "pdf") == "b_x.pdf"
        assert default_save_name(SimpleNamespace(), "x", "pdf") == "x.pdf"
        assert default_save_name(None, "x", "png") == "x.png"


class TestFigureExport:
    def test_format_follows_known_extensions(self):
        assert export_format("a.PNG", "pdf") == "png"
        assert export_format("a.v2", "pdf") == "pdf"
        assert export_format("a", None) is None

    def test_default_dpi_comes_from_the_display_settings(self, monkeypatch):
        display = importlib.import_module("iSLAT.Modules.GUI.DisplayConfig")
        monkeypatch.setattr(display, "display_config", SimpleNamespace(savefig_dpi=123))
        assert default_export_dpi() == 123
        fig = MagicMock()
        fig.axes = []
        save_figure_file(fig, "out.pdf")
        assert fig.savefig.call_args.kwargs == {"bbox_inches": "tight", "dpi": 123, "format": "pdf"}

    def test_explicit_format_and_dpi_win(self):
        fig = MagicMock()
        fig.axes = []
        save_figure_file(fig, "out.png", file_format="pdf", dpi=50, format="svg")
        assert fig.savefig.call_args.kwargs == {"bbox_inches": "tight", "dpi": 50, "format": "svg"}


# ======================================================================
# Data-field messages and the shared debug logger
# ======================================================================
class TestNotifyUser:
    def test_writes_to_the_data_field(self):
        data_field = MagicMock()
        islat = SimpleNamespace(GUI=SimpleNamespace(data_field=data_field))
        assert notify_user(islat, "hello", clear_after=False) is True
        data_field.insert_text.assert_called_once_with("hello", clear_after=False)

    def test_without_a_data_field(self):
        assert notify_user(None, "hello") is False
        assert notify_user(SimpleNamespace(GUI=SimpleNamespace(data_field=None)), "hello") is False

    def test_errors_are_swallowed(self):
        data_field = MagicMock()
        data_field.insert_text.side_effect = RuntimeError("widget gone")
        assert notify_user(SimpleNamespace(GUI=SimpleNamespace(data_field=data_field)), "x") is False


@pytest.mark.parametrize("module_name", [
    "FitLinesPlotGridView", "FullSpectrumView", "LineInspectionView", "MainPlot",
    "PopulationDiagramView", "ThreePanelView", "PlotView",
])
def test_plotting_modules_share_one_debug_logger(module_name):
    shared = importlib.import_module("iSLAT.Modules.Plotting.PlotDebug").debug_config
    module = importlib.import_module(f"iSLAT.Modules.Plotting.{module_name}")
    assert module.debug_config is shared
