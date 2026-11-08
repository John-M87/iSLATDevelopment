# -*- coding: utf-8 -*-
"""Sync Y to Data Min/Max: the shared menu / dialog controls and the main window's Views menu and dialog."""

import matplotlib
matplotlib.use("Agg")

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.plotting.test_main_plot_y_scale import _make_controller


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


def _button(widget, text):
    return next(w for w in _walk(widget) if w.winfo_class() == "TButton" and w.cget("text") == text)


class TestSyncMenu:
    def test_radio_entries_drive_command_and_variable(self, tk_root):
        import tkinter as tk
        from iSLAT.Modules.GUI.YAxisDataSync import (
            Y_DATA_SYNC_MENU_LABEL, add_y_data_sync_menu, y_data_sync_var_value,
        )
        menu = tk.Menu(tk_root, tearoff=0)
        var = tk.StringVar(master=tk_root, value=y_data_sync_var_value(None))
        calls = []
        sub = add_y_data_sync_menu(menu, var, calls.append)
        assert menu.type(0) == "cascade" and menu.entrycget(0, "label") == Y_DATA_SYNC_MENU_LABEL
        assert [sub.entrycget(i, "label") for i in range(sub.index("end") + 1)] == [
            "Off", "Global Data", "Active View", "Each Panel"]
        sub.invoke(2)
        assert calls == ["view"] and var.get() == "view"
        sub.invoke(0)
        assert calls == ["view", None] and var.get() == "off"

    def test_var_value(self):
        from iSLAT.Modules.GUI.YAxisDataSync import y_data_sync_var_value
        assert y_data_sync_var_value("panel") == "panel"
        assert y_data_sync_var_value(None) == "off"
        assert y_data_sync_var_value("bogus") == "off"


class TestMainWindow:
    @staticmethod
    def _top_bar_module():
        return importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")

    def test_views_menu_check_mark_follows_plot(self, tk_root):
        import tkinter as tk
        tb_mod = self._top_bar_module()
        bar = tb_mod.TopBar.__new__(tb_mod.TopBar)
        for name in ("_y_autoscale_var", "_y_model_var", "_observed_var"):
            setattr(bar, name, tk.BooleanVar(master=tk_root))
        bar._y_data_sync_var = tk.StringVar(master=tk_root, value="off")
        bar._sync_saved_labels_var = lambda: None
        bar.main_plot = SimpleNamespace(y_autoscale_toggle=True, y_autoscale_source="data",
                                        y_data_sync="panel", observed_toggle=True)
        bar._sync_plot_option_vars()
        assert bar._y_data_sync_var.get() == "panel"
        bar.main_plot.y_data_sync = None
        bar._sync_plot_option_vars()
        assert bar._y_data_sync_var.get() == "off"

    def _open_dialog(self, tk_root, monkeypatch, pm):
        """Run TopBar.open_y_limits_dialog with a plain frame standing in for the bar."""
        import tkinter as tk
        monkeypatch.setattr(tk.Toplevel, "grab_set", lambda self: None)
        host = tk.Frame(tk_root)
        host.main_plot = pm
        self._top_bar_module().TopBar.open_y_limits_dialog(host)
        return host.winfo_children()[-1]

    def test_dialog_fills_and_syncs_on_single_panel_view(self, tk_root, monkeypatch):
        pm = _make_controller()                 # flux 1 Jy below 7.5 um, 4 Jy above
        pm.ax1.set_xlim(5.0, 7.0)
        dlg = self._open_dialog(tk_root, monkeypatch, pm)
        combo = next(w for w in _walk(dlg) if w.winfo_class() == "TCombobox")
        # A single-panel view offers no separate "Each panel" choice.
        assert list(dlg.tk.splitlist(combo.cget("values"))) == ["Global data", "Active view"]
        entries = [w for w in _walk(dlg) if w.winfo_class() == "Entry"]
        _button(dlg, "Fill Values").invoke()
        assert (entries[0].get(), entries[1].get()) == ("1", "4")
        combo.set("Active view")
        combo.event_generate("<<ComboboxSelected>>")
        _button(dlg, "Fill Values").invoke()
        assert (entries[0].get(), entries[1].get()) == ("0.95", "1.05")
        _button(dlg, "Keep Synced").invoke()
        assert not dlg.winfo_exists()
        assert pm.y_data_sync == "view"
        assert pm.ax1.get_ylim() == pytest.approx((0.95, 1.05))

    def test_dialog_reports_missing_data(self, tk_root, monkeypatch):
        pm = MagicMock()
        pm.get_y_limits.return_value = (0.0, 1.0)
        pm.y_data_sync = None
        pm.y_axis_count.return_value = 3
        pm.get_data_y_range.return_value = None
        dlg = self._open_dialog(tk_root, monkeypatch, pm)
        combo = next(w for w in _walk(dlg) if w.winfo_class() == "TCombobox")
        assert list(dlg.tk.splitlist(combo.cget("values"))) == ["Global data", "Active view", "Each panel"]
        _button(dlg, "Fill Values").invoke()
        status = [w for w in _walk(dlg) if w.winfo_class() == "Label" and w.cget("fg") == "red"]
        assert status and status[0].cget("text") == "No observed data in that range."
        pm.set_y_data_sync.assert_not_called()
