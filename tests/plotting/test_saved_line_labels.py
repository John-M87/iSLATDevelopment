# -*- coding: utf-8 -*-
"""Saved-line labels: chosen CSV column, prefix / suffix, everywhere saved lines are drawn."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import iSLAT.Modules.Plotting.SavedLineLabels as sll
from iSLAT.Modules.Plotting.BasePlot import BasePlot

TAG = "_islat_saved_line"


@pytest.fixture(autouse=True)
def _reset_label_setting():
    sll.reset_saved_line_label_settings()
    yield
    sll.reset_saved_line_label_settings()


def _lines():
    return pd.DataFrame({
        "species": ["H2O", "OH", "CO"],
        "lam": [5.5, 6.25, 7.0],
        "xmin": [5.49, 6.24, 6.99],
        "xmax": [5.51, 6.26, 7.01],
        "lev_up": ["0_1_0|10_3_8", "v1", np.nan],
        "e_up": [4420.5, 1000.0, np.nan],
    })


def _texts(ax):
    return [t.get_text() for t in ax.texts if hasattr(t, TAG)]


class TestLabelText:
    def test_default_is_species_and_line(self):
        assert sll.saved_line_labels(_lines()) == ["H2O", "OH", "CO"]
        df = _lines().assign(line=["a", None, 3])
        assert sll.saved_line_labels(df) == ["H2O a", "OH", "CO 3"]

    def test_column_override_with_prefix_suffix_and_formatting(self):
        sll.set_saved_line_label_settings("e_up", prefix="Eu=", suffix=" K")
        assert sll.saved_line_labels(_lines()) == ["Eu=4420.5 K", "Eu=1000 K", ""]
        sll.set_saved_line_label_settings("lev_up")
        assert sll.saved_line_labels(_lines()) == ["0_1_0|10_3_8", "v1", ""]

    def test_prefix_suffix_on_default_and_missing_column_falls_back(self):
        sll.set_saved_line_label_settings(None, prefix="[", suffix="]")
        assert sll.saved_line_labels(_lines())[0] == "[H2O]"
        sll.set_saved_line_label_settings("no_such_column", prefix="> ")
        assert sll.saved_line_labels(_lines())[1] == "> OH"

    def test_explicit_settings_do_not_touch_app_setting(self):
        s = sll.SavedLineLabelSettings("lam", "", "um")
        assert sll.saved_line_labels(_lines(), s)[0] == "5.5um"
        assert sll.get_saved_line_label_settings() == sll.SavedLineLabelSettings()
        assert sll.saved_line_labels(None) == []

    def test_user_settings_round_trip(self):
        us = {}
        sll.set_saved_line_label_settings(" note ", "a", "b")
        sll.store_in_user_settings(us)
        assert us == {"saved_line_label_column": "note", "saved_line_label_prefix": "a",
                      "saved_line_label_suffix": "b", "saved_line_label_hidden_views": ["Three Panel"]}
        sll.set_saved_line_label_settings()
        assert sll.settings_from_user_settings(us) == sll.SavedLineLabelSettings("note", "a", "b")
        assert sll.settings_from_user_settings({"saved_line_label_column": "  "}).column is None
        assert sll.settings_from_user_settings(None) == sll.SavedLineLabelSettings()


class TestDrawing:
    def test_full_spectrum_annotations_follow_setting(self):
        ax = Figure().add_subplot(111)
        sll.set_saved_line_label_settings("lev_up", suffix="*")
        BasePlot._plot_line_annotations(ax, _lines(), (5.0, 6.5), 0.0, 1.0, tag=TAG)
        assert _texts(ax) == ["0_1_0|10_3_8*", "v1*"]

    def test_markers_without_labels_by_default(self):
        ax = Figure().add_subplot(111)
        BasePlot._plot_saved_line_markers(ax, _lines(), tag=TAG)
        assert _texts(ax) == []

    def test_marker_labels_at_lam_or_xmin(self):
        ax = Figure().add_subplot(111)
        ax.set_xlim(5, 8)
        ax.set_ylim(0, 1)
        df = pd.concat([_lines(), pd.DataFrame({"species": ["NoLam"], "xmin": [7.5], "xmax": [7.6]})])
        df_no_lam = df[df["species"] == "NoLam"].drop(columns=["lam"])
        BasePlot._plot_saved_line_markers(ax, _lines(), tag=TAG, show_labels=True)
        BasePlot._plot_saved_line_markers(ax, df_no_lam, tag=TAG, show_labels=True)
        labels = [t for t in ax.texts if hasattr(t, TAG)]
        assert [t.get_text() for t in labels] == ["H2O", "OH", "CO", "NoLam"]
        assert [t.xy[0] for t in labels] == pytest.approx([5.5, 6.25, 7.0, 7.5])
        assert ax.get_ylim() == (0.0, 1.0)

    def test_three_panel_labels_and_removal(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        fig = Figure()
        ax1, ax2, ax3 = (fig.add_subplot(3, 1, i) for i in (1, 2, 3))
        pm = MagicMock()
        pm.ax1, pm.ax2, pm.ax3 = ax1, ax2, ax3
        pm.theme = {}
        pm.control_bus = None
        view = ThreePanelView(pm)
        view.line_data = _lines()
        sll.set_saved_line_label_settings("lev_up", prefix="u:")
        sll.set_labels_visible("Three Panel", True)
        view._add_saved_line_artists()
        assert _texts(ax1) == ["u:0_1_0|10_3_8", "u:v1"]  # empty label for NaN is skipped
        view._remove_saved_line_artists()
        assert not [a for a in list(ax1.lines) + list(ax1.texts) if hasattr(a, TAG)]

    def test_full_spectrum_plot_export_uses_setting(self):
        from iSLAT.Modules.Plotting import FullSpectrumPlot
        wave = np.linspace(5.0, 8.0, 300)
        sll.set_saved_line_label_settings("species", prefix="#")
        plot = FullSpectrumPlot(wave, np.ones_like(wave), line_list=_lines(), n_panels=2)
        plot.generate_plot()
        texts = [t.get_text() for ax in plot.subplots.values() for t in ax.texts]
        assert {"#H2O", "#OH", "#CO"} <= set(texts)
        plot.close()


class TestRefresh:
    def _controller(self, monkeypatch, line_toggle=True, lines=None):
        from iSLAT.Modules.Plotting import MainPlot as mp
        pm = mp.iSLATPlot.__new__(mp.iSLATPlot)
        pm.toggle_state = {"saved_lines": line_toggle}
        pm.islat = SimpleNamespace(input_line_list="lines.csv")
        active, other = MagicMock(), MagicMock()
        plain = SimpleNamespace(_needs_refresh=False)
        pm.active_view = active
        pm._views = {"a": active, "b": other, "c": plain}
        monkeypatch.setattr(mp.ifh, "read_line_saves",
                            lambda file_name=None: _lines() if lines is None else lines)
        return pm, active, other, plain

    def test_redraws_active_and_resets_inactive(self, monkeypatch):
        pm, active, other, plain = self._controller(monkeypatch)
        pm.refresh_saved_line_labels()
        other._remove_saved_line_artists.assert_called_once()
        assert plain._needs_refresh is True
        calls = active.toggle_saved_lines.call_args_list
        assert calls[0].args == (False,)
        assert calls[1].args == (True,) and calls[1].kwargs["loaded_lines"] is not None

    def test_hidden_or_empty_saved_lines_are_not_redrawn(self, monkeypatch):
        pm, active, _, _ = self._controller(monkeypatch, line_toggle=False)
        pm.refresh_saved_line_labels()
        active.toggle_saved_lines.assert_not_called()
        pm, active, _, _ = self._controller(monkeypatch, lines=pd.DataFrame())
        pm.refresh_saved_line_labels()
        active.toggle_saved_lines.assert_not_called()


class TestApplyAndStartup:
    def test_apply_sets_saves_and_refreshes(self, monkeypatch):
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        from iSLAT.Modules.GUI.Widgets.SavedLineLabelDialog import (
            DEFAULT_CHOICE, apply_saved_line_label_settings)
        saved = []
        monkeypatch.setattr(ifh, "save_user_settings", lambda us: saved.append(dict(us)))
        plot = MagicMock()
        islat = SimpleNamespace(user_settings={}, GUI=SimpleNamespace(plot=plot))
        apply_saved_line_label_settings(islat, "note", "<", ">")
        assert sll.get_saved_line_label_settings() == sll.SavedLineLabelSettings("note", "<", ">")
        assert saved[-1]["saved_line_label_column"] == "note"
        plot.refresh_saved_line_labels.assert_called_once()
        apply_saved_line_label_settings(islat, DEFAULT_CHOICE)
        assert sll.get_saved_line_label_settings().column is None
        assert saved[-1]["saved_line_label_column"] is None

    def test_loading_user_settings_applies_label(self, monkeypatch):
        import iSLAT.iSLATClass as islat_mod
        monkeypatch.setattr(islat_mod, "load_user_settings",
                            lambda: {"saved_line_label_column": "lev_up", "saved_line_label_suffix": "!"})
        islat = islat_mod.iSLAT.__new__(islat_mod.iSLAT)
        islat._user_settings = None
        assert islat.user_settings["saved_line_label_column"] == "lev_up"
        assert sll.get_saved_line_label_settings() == sll.SavedLineLabelSettings("lev_up", "", "!")


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


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


class TestDialog:
    def test_preview_note_apply_reset(self, tk_root, monkeypatch):
        import iSLAT.Modules.GUI.Widgets.SavedLineLabelDialog as dlg_mod
        monkeypatch.setattr(dlg_mod, "_read_line_list", lambda islat: _lines())
        applied = []
        monkeypatch.setattr(dlg_mod, "apply_saved_line_label_settings",
                            lambda islat, c, p="", s="", h=None: applied.append((c, p, s, h)))
        islat = SimpleNamespace()
        win = dlg_mod.open_saved_line_label_dialog(tk_root, islat)
        assert dlg_mod.open_saved_line_label_dialog(tk_root, islat) is win
        view_boxes = {w.cget("text"): w for w in _walk(win) if w.winfo_class() == "TCheckbutton"}
        assert set(view_boxes) == set(sll.LABEL_VIEWS)
        assert {v for v, b in view_boxes.items() if "selected" in b.state()} == {"Full Spectrum", "Optical Depth", "Line Grid"}
        view_boxes["Line Grid"].invoke()
        combo = next(w for w in _walk(win) if w.winfo_class() == "TCombobox")
        assert list(combo.cget("values"))[0] == dlg_mod.DEFAULT_CHOICE
        assert "lev_up" in combo.cget("values")
        prefix, suffix = [w for w in _walk(win) if w.winfo_class() == "TEntry"]

        def labels():
            return [w.cget("text") for w in _walk(win) if w.winfo_class() == "TLabel"]

        assert any(str(t) == "Preview: 'H2O'" for t in labels())
        combo.set("lev_up")
        prefix.insert(0, "v=")
        assert any(str(t) == "Preview: 'v=0_1_0|10_3_8'" for t in labels())
        combo.set("bogus")
        assert any("not in the current line list" in str(t) for t in labels())
        combo.set("lev_up")
        next(w for w in _walk(win) if w.winfo_class() == "TButton" and w.cget("text") == "Apply").invoke()
        assert applied == [("lev_up", "v=", "", frozenset({"Three Panel", "Line Grid"}))]
        next(w for w in _walk(win) if w.winfo_class() == "TButton" and w.cget("text") == "Reset").invoke()
        assert combo.get() == dlg_mod.DEFAULT_CHOICE and prefix.get() == ""
        assert "selected" in view_boxes["Line Grid"].state()
        assert "selected" not in view_boxes["Three Panel"].state()
        next(w for w in _walk(win) if w.winfo_class() == "TButton" and w.cget("text") == "Close").invoke()
        assert islat._saved_line_label_dialog is None


class TestPerViewToggle:
    def test_three_panel_labels_off_by_default(self):
        assert sll.SavedLineLabelSettings().hidden_views == frozenset({"Three Panel"})
        assert not sll.labels_visible("Three Panel")
        # Missing setting -> default; an explicit (even empty) saved choice wins.
        assert not sll.settings_from_user_settings({}).hidden_views.isdisjoint({"Three Panel"})
        assert sll.settings_from_user_settings({"saved_line_label_hidden_views": []}).hidden_views == frozenset()
        assert sll.labels_visible("Three Panel")

    def test_visibility_api_and_persistence(self):
        assert not sll.labels_visible("Three Panel")
        assert sll.labels_visible("Full Spectrum") and sll.labels_visible("Line Grid")
        sll.set_saved_line_label_settings("note", "a")  # keeps per-view visibility
        assert not sll.labels_visible("Three Panel") and sll.labels_visible("Line Grid")
        us = {}
        sll.store_in_user_settings(us)
        assert us["saved_line_label_hidden_views"] == ["Three Panel"]
        sll.reset_saved_line_label_settings()
        sll.settings_from_user_settings(us)
        assert not sll.labels_visible("Three Panel")
        sll.set_labels_visible("Three Panel", True)
        assert sll.labels_visible("Three Panel")
        assert sll.settings_from_user_settings({"saved_line_label_hidden_views": "bad"}).hidden_views == sll.DEFAULT_HIDDEN_VIEWS

    def test_annotations_without_labels_keep_lines(self):
        ax = Figure().add_subplot(111)
        BasePlot._plot_line_annotations(ax, _lines(), (5.0, 8.0), 0.0, 1.0, tag=TAG, show_labels=False)
        assert _texts(ax) == []
        assert len([c for c in ax.collections if hasattr(c, TAG)]) == 3

    def test_three_panel_hidden(self):
        from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
        fig = Figure()
        ax1, ax2, ax3 = (fig.add_subplot(3, 1, i) for i in (1, 2, 3))
        pm = MagicMock()
        pm.ax1, pm.ax2, pm.ax3 = ax1, ax2, ax3
        pm.theme = {}
        pm.control_bus = None
        view = ThreePanelView(pm)
        view.line_data = _lines()
        view._add_saved_line_artists()  # labels are off in Three Panel by default
        assert _texts(ax1) == []
        assert [l for l in ax1.lines if hasattr(l, TAG)]

    def test_full_spectrum_export_follows_full_spectrum_toggle(self):
        from iSLAT.Modules.Plotting import FullSpectrumPlot
        wave = np.linspace(5.0, 8.0, 300)
        sll.set_labels_visible("Full Spectrum", False)
        plot = FullSpectrumPlot(wave, np.ones_like(wave), line_list=_lines(), n_panels=2)
        plot.generate_plot()
        assert not any(t.get_text() == "H2O" for ax in plot.subplots.values() for t in ax.texts)
        plot.show_saved_line_labels = True  # explicit override wins
        plot.generate_plot()
        assert any(t.get_text() == "H2O" for ax in plot.subplots.values() for t in ax.texts)
        plot.close()

    def test_line_grid_follows_its_toggle(self):
        from iSLAT.Modules.Plotting.FitLinesPlotGridView import FitLinesPlotGridView
        panel = MagicMock()
        grid = MagicMock()
        grid.iter_panels.return_value = [panel]
        view = FitLinesPlotGridView.__new__(FitLinesPlotGridView)
        view._plot_grids = [grid]
        view.draw = lambda: None
        sll.set_labels_visible("Line Grid", False)
        view.toggle_saved_lines(True, loaded_lines=_lines())
        panel.remove_saved_lines.assert_called_once()
        assert panel.plot_saved_lines.call_args.kwargs["show_labels"] is False

    def test_view_toggle_saves_and_redraws(self, monkeypatch):
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        from iSLAT.Modules.GUI.Widgets.SavedLineLabelDialog import set_saved_line_label_view_visible
        saved = []
        monkeypatch.setattr(ifh, "save_user_settings", lambda us: saved.append(dict(us)))
        plot = MagicMock()
        islat = SimpleNamespace(user_settings={}, GUI=SimpleNamespace(plot=plot))
        set_saved_line_label_view_visible(islat, "Full Spectrum", False)
        assert not sll.labels_visible("Full Spectrum")
        assert saved[-1]["saved_line_label_hidden_views"] == ["Full Spectrum", "Three Panel"]
        plot.refresh_saved_line_labels.assert_called_once()


class TestViewsMenuToggle:
    def _topbar(self, tk_root, view_name):
        import importlib
        tb_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")
        import tkinter as tk
        bar = tb_mod.TopBar.__new__(tb_mod.TopBar)
        bar.main_plot = SimpleNamespace(active_view_name=view_name)
        bar.islat = SimpleNamespace()
        bar._views_menu = tk.Menu(tk_root, tearoff=0)
        bar._saved_labels_var = tk.BooleanVar(master=tk_root, value=True)
        bar._views_menu.add_checkbutton(label="Show Saved Line Labels", variable=bar._saved_labels_var)
        bar._views_menu.add_command(label="Saved Line Labels\u2026")
        return bar

    def test_menu_mirrors_and_sets_active_view(self, tk_root, monkeypatch):
        import iSLAT.Modules.GUI.Widgets.SavedLineLabelDialog as dlg_mod
        calls = []
        monkeypatch.setattr(dlg_mod, "set_saved_line_label_view_visible",
                            lambda islat, view, visible: calls.append((view, visible)))
        sll.set_labels_visible("Full Spectrum", False)
        bar = self._topbar(tk_root, "Full Spectrum")
        bar._sync_saved_labels_var()
        assert bar._saved_labels_var.get() is False
        assert bar._views_menu.entrycget(0, "state") == "normal"
        bar.set_saved_line_labels_visible(True)
        assert calls == [("Full Spectrum", True)]

    def test_disabled_for_views_without_saved_lines(self, tk_root, monkeypatch):
        import iSLAT.Modules.GUI.Widgets.SavedLineLabelDialog as dlg_mod
        calls = []
        monkeypatch.setattr(dlg_mod, "set_saved_line_label_view_visible",
                            lambda *a: calls.append(a))
        bar = self._topbar(tk_root, "Population Diagram")
        bar._sync_saved_labels_var()
        assert bar._views_menu.entrycget(0, "state") == "disabled"
        assert bar._views_menu.entrycget(1, "state") == "normal"
        bar.set_saved_line_labels_visible(False)
        assert calls == []
