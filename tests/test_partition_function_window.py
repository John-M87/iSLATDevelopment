# -*- coding: utf-8 -*-
"""Partition function viewer / extender window (control-panel right-click)."""

import matplotlib
matplotlib.use("Agg")

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.test_partition_function_extender import Q, T, _molecule


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


@pytest.fixture
def W():
    from iSLAT.Modules.GUI.Widgets.PartitionFunctionWindow import PartitionFunctionWindow
    return PartitionFunctionWindow


@pytest.fixture
def open_window(tk_root, W):
    windows = []

    def _open(mol, islat=None, data_field=None):
        win = W.open(tk_root, mol, data_field, islat=islat)
        win.withdraw()
        windows.append(win)
        return win

    yield _open
    for win in windows:
        try:
            if win.winfo_exists():
                win._on_close()
        except Exception:
            pass


@pytest.fixture
def auto_yes(monkeypatch):
    import iSLAT.Modules.GUI.Widgets.PartitionFunctionWindow as mod
    monkeypatch.setattr(mod.messagebox, "askyesno", lambda *a, **k: True)
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(mod.messagebox, name, MagicMock())
    return mod


def _set_range(win, low="", high="", step="100"):
    win._tmin_var.set(low)
    win._tmax_var.set(high)
    win._step_var.set(step)
    win._recompute()


class TestWindow:
    def test_defaults_show_the_current_table(self, open_window):
        win = open_window(_molecule())
        assert float(win._tmin_var.get()) == T[0]
        assert float(win._tmax_var.get()) == T[-1]
        assert win._preview is not None and win._preview.n_added == 0
        assert str(win._apply_btn.cget("state")) == "disabled"
        assert win._mode_var.get() == "Constant Q(T)"

    def test_reopen_raises_existing(self, open_window, W, tk_root):
        mol = _molecule()
        win = open_window(mol)
        assert W.open(tk_root, mol) is win

    def test_preview_and_error_state(self, open_window):
        win = open_window(_molecule())
        _set_range(win, "50", "2300", "100")
        assert win._preview.n_added_low == 1 and win._preview.n_added_high == 3
        assert str(win._apply_btn.cget("state")) == "normal"
        assert "+3 above" in win._status_var.get()

        _set_range(win, "abc", "2300", "100")
        assert win._preview is None
        assert "number" in win._status_var.get()
        assert str(win._apply_btn.cget("state")) == "disabled"

    def test_table_lists_extended_rows(self, open_window):
        win = open_window(_molecule())
        _set_range(win, "", "2200", "100")
        win._notebook.select(1)
        win._refresh_table()
        rows = [win._tree.item(i) for i in win._tree.get_children()]
        assert len(rows) == T.size + 2
        assert rows[-1]["values"][2] == "extended"
        assert rows[0]["values"][2] == "original"

    def test_apply_to_molecule(self, open_window, auto_yes):
        mol = _molecule()
        data_field = MagicMock()
        win = open_window(mol, data_field=data_field)
        _set_range(win, "", "3000", "500")
        win._apply_to_molecule()
        assert mol.partition_function.t[-1] == 3000.0
        assert mol.partition_function.q[-1] == Q[-1]
        # the window now shows the applied table as current
        assert win._preview.n_added == 0
        data_field.insert_text.assert_called()

    def test_duplicate_and_apply_leaves_original(self, open_window, auto_yes):
        src = _molecule()
        dup = _molecule()
        dup.name = "H2O_copy"
        mol_dict = MagicMock()
        mol_dict.duplicate.return_value = "H2O_copy"
        mol_dict.get.return_value = dup
        islat = SimpleNamespace(molecules_dict=mol_dict)
        win = open_window(src, islat=islat, data_field=MagicMock())
        _set_range(win, "", "3000", "500")
        win._duplicate_and_apply()
        mol_dict.duplicate.assert_called_once_with("H2O")
        assert dup.partition_function.t[-1] == 3000.0
        assert src.partition_function.t[-1] == T[-1]

    def test_save_par_and_csv(self, open_window, tmp_path):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        win = open_window(_molecule(), data_field=MagicMock())
        _set_range(win, "", "2500", "250")
        out = win._write_partition(tmp_path / "x.par")
        assert MoleculeLineList.from_file(out).partition.t[-1] == 2500.0
        out = win._write_partition(tmp_path / "x.csv")
        assert out.read_text().startswith("temp,q")

    def test_save_figure(self, open_window, tmp_path):
        win = open_window(_molecule(), data_field=MagicMock())
        path = tmp_path / "pf.png"
        assert win.save_figure(str(path)) == str(path)
        assert path.read_bytes()[:4] == b"\x89PNG"

    def test_log_toggles_redraw(self, open_window):
        win = open_window(_molecule())
        win._log_t_var.set(True)
        win._draw()
        assert win._ax.get_xscale() == "log"
        win._log_q_var.set(False)
        win._draw()
        assert win._ax.get_yscale() == "linear"

    def test_close_unregisters(self, open_window, W):
        win = open_window(_molecule())
        key = win._instance_key
        win._on_close()
        assert key not in W._instances


class TestControlPanelMenu:
    def test_menu_entry_opens_window(self, monkeypatch, W):
        cp_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ControlPanel")
        menus = []

        class _Menu:
            def __init__(self, *a, **k):
                self.items = []
                menus.append(self)

            def add_command(self, label=None, command=None, **k):
                self.items.append((label, command))

            def add_separator(self):
                pass

            def tk_popup(self, *a):
                pass

            def grab_release(self):
                pass

        monkeypatch.setattr(cp_mod.tk, "Menu", _Menu)
        panel = cp_mod.ControlPanel.__new__(cp_mod.ControlPanel)
        mol = _molecule()
        panel.islat = SimpleNamespace(molecules_dict={"H2O": mol})
        panel.data_field = None
        panel._parameter_clipboard = None
        panel._get_selected_molecule_names = lambda: ["H2O"]
        panel._is_molecule_bold = lambda _n: False
        opened = MagicMock()
        monkeypatch.setattr(W, "open", opened)

        panel._show_molecule_context_menu("H2O", SimpleNamespace(x_root=0, y_root=0))
        items = dict(menus[0].items)
        items["View Partition Function"]()
        opened.assert_called_once()
        assert opened.call_args.args[:2] == (panel, mol)

    def test_molecule_without_partition_is_reported(self, monkeypatch, W):
        cp_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ControlPanel")
        opened = MagicMock()
        monkeypatch.setattr(W, "open", opened)
        mol = _molecule()
        mol.lines.partition_function = None
        panel = cp_mod.ControlPanel.__new__(cp_mod.ControlPanel)
        panel.islat = SimpleNamespace(molecules_dict={"H2O": mol})
        panel.data_field = MagicMock()
        panel._partition_function_action("H2O")
        opened.assert_not_called()
        assert "No partition function" in panel.data_field.insert_text.call_args.args[0]

    def test_unknown_molecule_is_ignored(self, monkeypatch, W):
        cp_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ControlPanel")
        opened = MagicMock()
        monkeypatch.setattr(W, "open", opened)
        panel = cp_mod.ControlPanel.__new__(cp_mod.ControlPanel)
        panel.islat = SimpleNamespace(molecules_dict={})
        panel._partition_function_action("missing")
        opened.assert_not_called()
