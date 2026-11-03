# -*- coding: utf-8 -*-
"""Popout population diagram for one molecule (control-panel right-click)."""

import matplotlib
matplotlib.use("Agg")

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_CO_PAR = os.path.join("iSLAT", "DATAFILES", "HITRANdata", "data_Hitran_CO.par")


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


def _co(name="CO"):
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    if not os.path.exists(_CO_PAR):
        pytest.skip("CO HITRAN file not available")
    return Molecule(
        name=name, displaylabel=name, filepath=_CO_PAR, color="#f00",
        is_visible=True, temp=600.0, radius=0.5, n_mol=1e17, distance=140.0,
        fwhm=130.0, wavelength_range=(4.9, 5.2), model_pixel_res=0.001,
        initial_molecule_parameters={"t_kin": 600.0, "scale_exponent": 17.0,
                                     "scale_number": 1.0, "radius_init": 0.5},
    )


class _Dict:
    """MoleculeDict stand-in recording global-parameter callbacks."""

    def __init__(self):
        self.callbacks = []

    def add_global_parameter_change_callback(self, cb):
        self.callbacks.append(cb)

    def remove_global_parameter_change_callback(self, cb):
        self.callbacks.remove(cb)


@pytest.fixture
def open_window(tk_root):
    from iSLAT.Modules.GUI.Widgets.MoleculePopulationDiagramWindow import (
        MoleculePopulationDiagramWindow as W,
    )
    windows = []

    def _open(mol, islat=None):
        win = W.open(tk_root, mol, islat=islat)
        win.withdraw()
        windows.append(win)
        return win

    yield _open
    for win in windows:
        if win._alive():
            win.close()


def _intens(win):
    return np.asarray(win._pdp._component_data[0]["intens"], dtype=float)


class TestWindow:
    def test_draws_every_line_of_the_molecule(self, open_window):
        mol = _co()
        win = open_window(mol)
        assert win.title() == "Population Diagram: CO"
        assert win._pdp.molecule is mol
        n = len(win._pdp._component_data[0]["eu"])
        assert n > 0
        assert win._status_var.get() == f"Showing {n} lines."
        assert win._ax.collections  # scatter drawn

    def test_live_update_on_own_parameter_change(self, open_window):
        mol = _co()
        win = open_window(mol)
        before = _intens(win)
        mol.temp = 1500.0
        assert win._redraw_job is not None
        win._render()
        assert not np.allclose(before, _intens(win), equal_nan=True)

    def test_ignores_other_molecules_and_display_only_params(self, open_window):
        mol, other = _co(), _co("CO_other")
        win = open_window(mol)
        other.temp = 900.0
        assert win._redraw_job is None
        win._on_molecule_parameter_changed("CO", "is_bold", False, True)
        win._on_molecule_parameter_changed("CO", "is_visible", True, False)
        assert win._redraw_job is None

    def test_global_parameter_change_and_cleanup(self, open_window):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        mol, mdict = _co(), _Dict()
        win = open_window(mol, islat=SimpleNamespace(molecules_dict=mdict))
        assert len(mdict.callbacks) == 1
        mdict.callbacks[0]("global_distance", 140.0, 200.0)
        assert win._redraw_job is not None
        win.close()
        assert mdict.callbacks == []
        assert win._mol_cb not in Molecule._class_callbacks
        mol.temp = 800.0  # no callback into the destroyed window

    def test_one_window_per_molecule(self, open_window):
        mol, other = _co(), _co("CO_other")
        win = open_window(mol)
        assert open_window(mol) is win
        assert open_window(other) is not win
        win.close()
        assert open_window(mol) is not win

    def test_shared_plot_settings_menu(self, open_window):
        win = open_window(_co())
        menu = win.build_context_menu(None, win._canvas.get_tk_widget())
        labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)
                  if menu.type(i) == "command"]
        assert "Plot Settings\u2026" in labels
        assert "Sync Settings from Main Diagram" in labels
        win._sync_from_main()  # no main diagram: keeps defaults, still draws
        assert win._pdp._component_data


class TestControlPanelMenu:
    def test_right_click_menu_offers_population_diagram(self, monkeypatch):
        cp_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ControlPanel")
        from iSLAT.Modules.GUI.Widgets.MoleculePopulationDiagramWindow import (
            MoleculePopulationDiagramWindow,
        )
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
        mol = object()
        panel.islat = SimpleNamespace(molecules_dict={"CO": mol})
        panel._parameter_clipboard = None
        panel._get_selected_molecule_names = lambda: ["CO"]
        panel._is_molecule_bold = lambda _n: False
        opened = MagicMock()
        monkeypatch.setattr(MoleculePopulationDiagramWindow, "open", opened)

        panel._show_molecule_context_menu("CO", SimpleNamespace(x_root=0, y_root=0))
        items = dict(menus[0].items)
        assert "Population Diagram" in items
        items["Population Diagram"]()
        opened.assert_called_once_with(panel, mol, islat=panel.islat)

    def test_unknown_molecule_is_ignored(self, monkeypatch):
        cp_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ControlPanel")
        from iSLAT.Modules.GUI.Widgets.MoleculePopulationDiagramWindow import (
            MoleculePopulationDiagramWindow,
        )
        opened = MagicMock()
        monkeypatch.setattr(MoleculePopulationDiagramWindow, "open", opened)
        panel = cp_mod.ControlPanel.__new__(cp_mod.ControlPanel)
        panel.islat = SimpleNamespace(molecules_dict={})
        panel._population_diagram_action("missing")
        opened.assert_not_called()


class TestSaveFigure:
    @staticmethod
    def _labels(win):
        menu = win.build_context_menu(None, win._canvas.get_tk_widget())
        return [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)
                if menu.type(i) == "command"]

    @pytest.mark.parametrize("ext, magic", [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<?xml")])
    def test_save_format_follows_extension(self, open_window, monkeypatch, tmp_path, ext, magic):
        import tkinter.filedialog
        target = tmp_path / f"diagram.{ext}"
        seen = {}

        def _ask(**kw):
            seen.update(kw)
            return str(target)

        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", _ask)
        data_field = MagicMock()
        win = open_window(_co(), islat=SimpleNamespace(GUI=SimpleNamespace(data_field=data_field)))
        assert self._labels(win)[-1] == "Save Figure\u2026"
        assert win.save_figure() == str(target)
        assert target.read_bytes().startswith(magic)
        assert seen["initialfile"] == "CO_population_diagram.pdf"
        assert seen["parent"] is win
        assert "Figure saved to" in win._status_var.get()
        data_field.insert_text.assert_called_once()

    def test_cancel_and_bad_extension(self, open_window, monkeypatch, tmp_path):
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: "")
        win = open_window(_co())
        assert win.save_figure() is None
        assert win.save_figure(str(tmp_path / "diagram.notaformat")) is None
        assert win._status_var.get().startswith("Could not save figure")

    def test_filtered_window_has_save_too(self, tk_root, tmp_path):
        from iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow import (
            FilteredPopulationDiagramWindow,
        )
        win = FilteredPopulationDiagramWindow(tk_root, SimpleNamespace(name="H2O"))
        win.withdraw()
        try:
            assert "Save Figure\u2026" in self._labels(win)
            assert win._default_figure_name() == "H2O_population_diagram_filtered"
            out = tmp_path / "f.png"
            assert win.save_figure(str(out)) == str(out) and out.exists()
        finally:
            win.close()
