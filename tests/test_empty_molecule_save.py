# -*- coding: utf-8 -*-
"""Starting iSLAT with no molecules, e.g. after deleting them all and reopening.

Deleting every molecule saves a molecules file with only the header row.  On
the next start the GUI used to crash while building the control panel because
there was no active molecule.  These tests cover:

* ``MoleculeDict`` releasing the active / comparison molecule when it is
  deleted or the dict is cleared (no references to molecules that are gone),
* reading a header-only save file,
* the real start-up path (``load_spectrum`` + ``init_gui``) with an empty
  save, and adding / deleting molecules afterwards.
"""
import os

import pytest

from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

from tests.plotting.test_plot_views import _make_test_molecule
from tests.tk_events import pump_tk

HEADER = (
    "Molecule Name,File Path,Molecule Label,Temp,Rad,N_Mol,Color,Vis,Dist,StellarRV,"
    "FWHM,KeplerianFWHM,InstrumentalProfile,Broad,RV Shift\n"
)


def _dict(*names):
    md = MoleculeDict()
    for name in names:
        md[name] = _make_test_molecule(name=name)
    return md


def _record_active_changes(md):
    changes = []
    md.add_active_molecule_change_callback(
        lambda old, new: changes.append((getattr(old, "name", None), getattr(new, "name", None))))
    return changes


# ======================================================================
# MoleculeDict keeps no references to removed molecules
# ======================================================================

class TestRemovingMolecules:
    def test_deleting_the_last_molecule_clears_the_active_one(self):
        md = _dict("H2O")
        md.active_molecule = "H2O"
        changes = _record_active_changes(md)
        del md["H2O"]
        assert md.active_molecule is None
        assert changes == [("H2O", None)]

    def test_deleting_another_molecule_keeps_the_active_one(self):
        md = _dict("H2O", "CO")
        md.active_molecule = "H2O"
        changes = _record_active_changes(md)
        del md["CO"]
        assert md.active_molecule is md["H2O"]
        assert changes == []

    def test_replacement_set_before_deleting_is_kept(self):
        # What the control panel does: promote another molecule, then delete.
        md = _dict("H2O", "CO")
        md.active_molecule = "H2O"
        md.active_molecule = "CO"
        del md["H2O"]
        assert md.active_molecule is md["CO"]

    def test_deleted_molecule_leaves_the_comparison_list(self):
        md = _dict("H2O", "CO", "OH")
        md.active_molecule = "H2O"
        md.toggle_comparison_molecule("CO")
        md.toggle_comparison_molecule("OH")
        seen = []
        md.add_comparison_molecule_change_callback(lambda mols: seen.append([m.name for m in mols]))
        del md["CO"]
        assert [m.name for m in md.comparison_molecules] == ["OH"]
        assert seen == [["OH"]]

    def test_clear_resets_active_and_comparison_molecules(self):
        md = _dict("H2O", "CO")
        md.active_molecule = "H2O"
        md.toggle_comparison_molecule("CO")
        changes = _record_active_changes(md)
        md.clear()
        assert md.active_molecule is None and md.comparison_molecules == []
        assert changes == [("H2O", None)]

    def test_reorder_keeps_the_active_molecule(self):
        md = _dict("H2O", "CO")
        md.active_molecule = "CO"
        changes = _record_active_changes(md)
        md.reorder(["CO", "H2O"])
        assert md.active_molecule is md["CO"] and changes == []


def test_header_only_save_file_reads_as_no_molecules(tmp_path):
    from iSLAT.Modules.FileHandling.iSLATFileHandling import read_from_user_csv
    (tmp_path / "molecules_list.csv").write_text(HEADER)
    assert read_from_user_csv(str(tmp_path), "molecules_list.csv") == {}


# ======================================================================
# Real start-up with an empty save
# ======================================================================

@pytest.fixture
def empty_start(tmp_path, monkeypatch):
    """Run iSLAT's start-up (load_spectrum + init_gui) with an empty molecules save.

    Everything is sandboxed: saves go to *tmp_path*, user settings are not
    written, the window stays hidden and ``mainloop`` returns immediately.
    """
    import tkinter as tk
    try:
        probe = tk.Tk()
        probe.destroy()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")

    import iSLAT.iSLATClass as islat_module
    import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.FileHandling import example_data_folder_path

    spectrum = os.path.join(example_data_folder_path, "CITau_MIRI_Banzatti+2023b.csv")
    if not os.path.exists(spectrum):
        pytest.skip("example spectrum not available")

    (tmp_path / "molecules_list.csv").write_text(HEADER)
    original_read = ifh.read_from_user_csv

    def read_sandboxed(file_path=None, file_name="molecules_list.csv", update_save_file_names=False):
        return original_read(str(tmp_path) if file_path is None else file_path, file_name, update_save_file_names)

    monkeypatch.setattr(islat_module, "read_from_user_csv", read_sandboxed)
    monkeypatch.setattr(islat_module, "save_folder_path", str(tmp_path))
    monkeypatch.setattr(ifh, "save_user_settings", lambda *a, **k: None)
    if hasattr(islat_module, "save_user_settings"):
        monkeypatch.setattr(islat_module, "save_user_settings", lambda *a, **k: None)

    original_tk_init = tk.Tk.__init__

    def hidden_tk_init(self, *args, **kwargs):
        original_tk_init(self, *args, **kwargs)
        self.withdraw()

    monkeypatch.setattr(tk.Tk, "__init__", hidden_tk_init)
    monkeypatch.setattr(tk.Tk, "deiconify", lambda self: None)
    monkeypatch.setattr(tk.Misc, "mainloop", lambda self, n=0: None)
    callback_errors = []
    monkeypatch.setattr(tk.Tk, "report_callback_exception",
                        lambda self, exc, val, tb: callback_errors.append(val))
    class_callbacks = list(Molecule._class_callbacks)

    app = islat_module.iSLAT()
    app.user_settings["restore_sample_on_startup"] = False
    assert app.load_spectrum(file_path=spectrum)
    try:
        app.init_gui()  # used to raise AttributeError: 'NoneType' object has no attribute 'name'
    except tk.TclError as exc:
        if "usable init.tcl" in str(exc) or "usable tk.tcl" in str(exc):
            Molecule._class_callbacks[:] = class_callbacks
            pytest.skip(f"Tk could not start: {str(exc).splitlines()[0]}")
        raise

    def pump(seconds=0.5):
        pump_tk(app.GUI.window, seconds)

    pump(1.0)
    try:
        yield app, pump, callback_errors
    finally:
        Molecule._class_callbacks[:] = class_callbacks
        try:
            app.GUI.window.destroy()
        except Exception:
            pass


def _panel_label(app):
    return app.GUI.control_panel.selected_label.cget("text").split("\n")[0]


class TestEmptyStartup:
    def test_gui_opens_without_molecules(self, empty_start):
        app, pump, callback_errors = empty_start
        cp = app.GUI.control_panel
        assert len(app.molecules_dict) == 0 and app.active_molecule is None
        assert _panel_label(app) == "Selected Molecule: None"
        # Global and wavelength controls are built even though there are no molecules
        assert "global_distance" in cp._global_parameter_entries
        assert hasattr(cp, "min_wavelength_var") and hasattr(cp, "max_wavelength_var")
        assert len(app.GUI.plot.ax1.lines) > 0  # the observed spectrum is drawn
        assert callback_errors == []

    def test_views_and_selection_work_without_molecules(self, empty_start):
        app, pump, callback_errors = empty_start
        pm = app.GUI.plot
        pm.onselect(14.0, 14.2)
        for view in list(pm.views) + ["Three Panel"]:
            pm.switch_view(view)
            pump(0.2)
        assert callback_errors == []

    def test_molecules_can_be_added_and_deleted_again(self, empty_start):
        app, pump, callback_errors = empty_start
        cp = app.GUI.control_panel

        app.molecules_dict["H2O"] = _make_test_molecule(name="H2O")
        app._set_initial_active_molecule()
        cp.refresh_from_molecules_dict()
        pump(0.3)
        assert app.active_molecule.name == "H2O"
        assert _panel_label(app) == "Selected Molecule: H2O"  # panel follows the dict after an empty start
        assert "H2O" in cp.mol_frames

        cp._delete_molecule("H2O")
        pump(0.3)
        assert len(app.molecules_dict) == 0 and app.active_molecule is None
        assert _panel_label(app) == "Selected Molecule: None"
        assert not cp.mol_frames
        assert callback_errors == []
