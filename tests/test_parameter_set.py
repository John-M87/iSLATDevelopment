"""ParameterSet: capture, apply in place, and the load paths that use it."""

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.DataTypes.ParameterSet import (APPLIED_PARAMETERS, SAVE_COLUMNS, ParameterSet,
                                                   ParameterSetStore)

ROOT = Path(__file__).resolve().parents[1]
HITRAN = ROOT / "iSLAT" / "DATAFILES" / "HITRANdata"
CO_PAR = HITRAN / "data_HITEMP_2019_CO.par"
OTHER_PAR = HITRAN / "data_Arabhavi_2024_C6H6.par"
WAVE = np.linspace(4.92, 5.18, 400)
COMPARED = APPLIED_PARAMETERS + ("color", "displaylabel", "filepath")

line_list_mod = importlib.import_module("iSLAT.Modules.DataTypes.MoleculeLineList")


def _row(name, temp=600.0, **extra):
    row = {"Molecule Name": name, "File Path": str(CO_PAR), "Molecule Label": name, "Temp": temp,
           "Rad": 0.5, "N_Mol": 1e17, "Color": "#ff0000", "Vis": True, "Dist": 140.0,
           "StellarRV": 0.0, "FWHM": 130.0, "KeplerianFWHM": 0.0, "InstrumentalProfile": "constant",
           "Broad": 2.5, "RV Shift": 0.0}
    row.update(extra)
    return row


def _new_dict(rows):
    """The old load: an empty dict, every molecule built from its row."""
    md = MoleculeDict(global_wavelength_range=(4.9, 5.2), global_model_pixel_res=0.001)
    md.load_molecules(rows, {}, update_global_parameters=True, strategy="sequential")
    return md


def _summed(md):
    md.invalidate_summed_flux_cache()
    return md.get_summed_flux(WAVE)[1]


def _params(md):
    return {name: {p: getattr(mol, p) for p in COMPARED} for name, mol in md.items()}


@pytest.fixture(autouse=True)
def _no_cache_files(monkeypatch):
    if not CO_PAR.exists():
        pytest.skip("CO HITRAN file not available")
    monkeypatch.setattr(line_list_mod.MoleculeLineList, "_save_to_cache", lambda self, path: True)


@pytest.fixture
def dicts():
    made = []

    def make(rows):
        md = _new_dict(rows)
        made.append(md)
        return md

    yield make
    for md in made:
        Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


ROWS = [_row("CO"), _row("COb", 900.0), _row("COc", 300.0, Color="green")]


class TestCapture:
    def test_rows_have_every_save_column(self, dicts):
        md = dicts(ROWS)
        rows = ParameterSet.capture(md, name="A").rows()
        assert [r["Molecule Name"] for r in rows] == ["CO", "COb", "COc"]
        assert all(tuple(r) == SAVE_COLUMNS for r in rows)
        assert rows[1]["Temp"] == 900.0 and rows[2]["Color"] == "green"

    def test_capture_is_a_snapshot(self, dicts):
        md = dicts(ROWS)
        snapshot = ParameterSet.capture(md)
        md["CO"].temp = 1000.0
        assert snapshot.rows()[0]["Temp"] == 600.0


class TestApply:
    def test_matches_a_full_load(self, dicts):
        md = dicts(ROWS)
        _summed(md)
        target = [_row("COc", 300.0, Color="green", Vis=False), _row("CO", 750.0, N_Mol=3e17),
                  _row("COb", 900.0)]
        target[0]["Dist"], target[0]["StellarRV"] = 150.0, 5.0  # globals come from the first row

        result = ParameterSet(target).apply_to(md, {})
        reference = dicts(target)

        assert sorted(result.updated) == ["CO", "COc"] and result.unchanged == ["COb"]
        assert list(md) == list(reference) == ["COc", "CO", "COb"]
        assert _params(md) == _params(reference)
        assert (md.global_distance, md.global_stellar_rv) == (150.0, 5.0)
        np.testing.assert_array_equal(_summed(md), _summed(reference))

    def test_csv_strings_parse_like_a_full_load(self, dicts):
        md = dicts(ROWS)
        target = [{k: str(v) for k, v in r.items()} for r in ROWS]
        target[1]["Vis"], target[1]["Temp"] = "False", "925.5"
        ParameterSet(target).apply_to(md, {})
        assert _params(md) == _params(dicts(target))
        assert md["COb"].is_visible is False and md["COb"].temp == 925.5

    def test_unchanged_set_keeps_objects_and_models(self, dicts):
        md = dicts(ROWS)
        _summed(md)
        before = {name: (mol, mol.intensity) for name, mol in md.items()}
        result = ParameterSet.capture(md).apply_to(md, {})
        assert result.unchanged == ["CO", "COb", "COc"] and not result.changed
        assert {name: (mol, mol.intensity) for name, mol in md.items()} == before
        assert not any(mol._dirty_flags["intensity"] for mol in md.values())

    def test_only_changed_molecules_are_recomputed(self, dicts):
        md = dicts(ROWS)
        _summed(md)
        rows = ParameterSet.capture(md).rows()
        rows[0]["Temp"] = 650.0
        ParameterSet(rows).apply_to(md, {})
        assert md["CO"]._dirty_flags["intensity"]
        assert not md["COb"]._dirty_flags["intensity"] and not md["COc"]._dirty_flags["intensity"]

    def test_adds_removes_and_reorders(self, dicts):
        md = dicts(ROWS)
        kept = md["COb"]
        result = ParameterSet([_row("NEW", 500.0), _row("COb", 900.0)]).apply_to(md, {})
        assert result.added == ["NEW"] and sorted(result.removed) == ["CO", "COc"]
        assert list(md) == ["NEW", "COb"] and md["COb"] is kept

    def test_other_line_file_is_rebuilt(self, dicts):
        if not OTHER_PAR.exists():
            pytest.skip("second line list not available")
        md = dicts(ROWS)
        old = md["COb"]
        result = ParameterSet([ROWS[0], _row("COb", 900.0, **{"File Path": str(OTHER_PAR)}), ROWS[2]]).apply_to(md, {})
        assert result.replaced == ["COb"] and md["COb"] is not old
        assert md["COb"].filepath == str(OTHER_PAR)

    def test_replaced_partition_function_is_rebuilt(self, dicts):
        md = dicts(ROWS)
        pf = md["CO"].partition_function
        md["CO"].set_partition_function(np.asarray(pf.t), np.asarray(pf.q) * 2.0)
        old = md["CO"]
        result = ParameterSet(ROWS).apply_to(md, {})
        assert result.replaced == ["CO"] and md["CO"] is not old

    def test_unbuildable_row_is_dropped_like_a_full_load(self, dicts):
        md = dicts(ROWS)
        result = ParameterSet([ROWS[0], _row("COb", "not a number"), ROWS[2]]).apply_to(md, {})
        assert list(result.failed) == ["COb"] and "COb" not in md

    def test_sends_no_parameter_notifications(self, dicts):
        md = dicts(ROWS)
        listener = MagicMock()
        Molecule.add_molecule_parameter_change_callback(listener)
        try:
            ParameterSet([_row("CO", 999.0), _row("COb", 1.0e3, Vis=False)]).apply_to(md, {})
        finally:
            Molecule.remove_molecule_parameter_change_callback(listener)
        listener.assert_not_called()

    def test_summed_flux_cache_is_dropped_on_change(self, dicts):
        md = dicts(ROWS)
        before = md.get_summed_flux(WAVE)[1].copy()
        ParameterSet([_row("CO", 1200.0), ROWS[1], ROWS[2]]).apply_to(md, {})
        assert not np.array_equal(md.get_summed_flux(WAVE)[1], before)


class TestActiveMolecule:
    def test_kept_when_still_in_the_set(self, dicts):
        md = dicts(ROWS)
        md.active_molecule = md["COb"]
        ParameterSet([_row("COb", 400.0), ROWS[0]]).apply_to(md, {})
        assert md.active_molecule is md["COb"]

    def test_follows_a_rebuilt_molecule(self, dicts):
        md = dicts(ROWS)
        md.active_molecule = md["CO"]
        pf = md["CO"].partition_function
        md["CO"].set_partition_function(np.asarray(pf.t), np.asarray(pf.q) * 2.0)
        ParameterSet(ROWS).apply_to(md, {})
        assert md.active_molecule is md["CO"]

    def test_cleared_when_removed(self, dicts):
        md = dicts(ROWS)
        md.active_molecule = md["COc"]
        ParameterSet(ROWS[:2]).apply_to(md, {})
        assert md.active_molecule is None


class TestCsv:
    def test_write_then_read_round_trips(self, dicts, tmp_path):
        from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv
        md = dicts(ROWS)
        md["COb"].is_visible = False
        path = write_molecules_to_csv(md, file_path=str(tmp_path), file_name="set.csv")
        assert Path(path).read_text().splitlines()[0] == ",".join(SAVE_COLUMNS)

        loaded = ParameterSet.from_csv(path)
        other = dicts([_row("CO", 100.0)])
        loaded.apply_to(other, {})
        assert _params(other) == _params(md)

    def test_raw_rows_keep_their_defaults(self, tmp_path):
        from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv
        path = write_molecules_to_csv({"H2O": {"Temp": 500}, "_skip": {}}, file_path=str(tmp_path),
                                      file_name="raw.csv")
        assert Path(path).read_text().splitlines()[1] == "H2O,,H2O,500,,,,,,,,0.0,constant,,"

    def test_missing_or_empty_file_is_none(self, tmp_path):
        assert ParameterSet.from_csv(str(tmp_path / "none.csv")) is None


class TestIslatLoadPaths:
    @pytest.fixture
    def islat(self, dicts):
        from iSLAT.iSLATClass import iSLAT
        obj = iSLAT.__new__(iSLAT)
        obj._user_settings = {"default_active_molecule": "COb", "intensity_calculation_method": "curve_growth"}
        obj._initial_molecule_parameters = {}
        obj.molecules_dict = dicts(ROWS)
        obj.parameter_sets = ParameterSetStore()
        obj.GUI = None
        return obj

    def test_apply_picks_the_default_active_molecule_when_needed(self, islat):
        islat.molecules_dict.active_molecule = islat.molecules_dict["COc"]
        islat.apply_parameter_set(ParameterSet(ROWS[:2]))
        assert islat.molecules_dict.active_molecule is islat.molecules_dict["COb"]
        assert islat._molecules_loaded

    def test_explicit_file_load_updates_in_place(self, islat, tmp_path):
        from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv
        md = islat.molecules_dict
        objects = dict(md)
        md["CO"].temp = 1111.0
        path = write_molecules_to_csv(md, file_path=str(tmp_path), file_name="set.csv")
        md["CO"].temp = 600.0

        assert islat._load_parameters_from_explicit_file(path)
        assert dict(md) == objects and md["CO"].temp == 1111.0


# ---------------------------------------------------------------------------
# ParameterSetStore
# ---------------------------------------------------------------------------

class TestStore:
    @pytest.fixture
    def setup(self, dicts):
        md = dicts(ROWS)
        store = ParameterSetStore()
        store.ensure_active(md)  # as iSLAT does once the first molecules load
        apply = lambda pset: pset.apply_to(md, {})
        return md, store, apply

    def test_open_makes_an_unmodified_active_set(self, setup):
        md, store, apply = setup
        name = store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        assert name == store.active == "A.csv" and store.path() == "/saves/A.csv"
        assert not store.is_modified(md)
        md["CO"].temp = 700.0
        assert store.is_modified(md)
        md["CO"].temp = 600.0
        assert not store.is_modified(md)  # compared with the saved version, not a flag

    def test_edits_survive_switching(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        md["CO"].temp = 777.0
        store.open(ParameterSet([_row("CO", 400.0)], name="B.csv"), md, apply, path="/saves/B.csv")
        assert list(md) == ["CO"] and md["CO"].temp == 400.0 and not store.is_modified(md)
        assert store.is_modified(name="A.csv")

        # Opening A's file again returns to its in-memory version, edits included
        assert store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv") == "A.csv"
        assert md["CO"].temp == 777.0 and list(md) == ["CO", "COb", "COc"]
        assert store.is_modified(md) and store.names == ["A.csv", "B.csv"]

    def test_revert(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        md["CO"].temp = 777.0
        del md["COc"]
        assert store.revert(md, apply)
        assert md["CO"].temp == 600.0 and "COc" in md and not store.is_modified(md)

    def test_changes_without_callbacks_count(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        md["COb"].color = "purple"           # plain attribute: no change callback
        assert store.is_modified(md)
        store.revert(md, apply)
        md.bulk_update_parameters({"temp": 1234.0})  # notifications skipped
        assert store.is_modified(md)

    def test_saving_moves_the_edits_to_the_saved_set(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        md["CO"].temp = 777.0
        assert store.mark_saved(md, "/saves/spectrum-molsave.csv") == "spectrum-molsave.csv"
        assert store.active == "spectrum-molsave.csv" and not store.is_modified(md)
        assert not store.is_modified(name="A.csv")  # A is back to its own saved version
        store.switch("A.csv", md, apply)
        assert md["CO"].temp == 600.0

    def test_saving_back_to_the_open_file(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="A.csv"), md, apply, path="/saves/A.csv")
        md["CO"].temp = 777.0
        assert store.mark_saved(md, "/saves/./A.csv") == "A.csv" and store.names == ["A.csv"]
        assert not store.is_modified(md)

    def test_pathless_sets_open_fresh_and_names_stay_unique(self, setup):
        md, store, apply = setup
        store.open(ParameterSet(ROWS, name="Defaults"), md, apply)
        store.open(ParameterSet(ROWS, name="Defaults"), md, apply)  # unchanged: replaced
        assert store.names == ["Defaults"]
        md["CO"].temp = 777.0
        store.open(ParameterSet(ROWS, name="Defaults"), md, apply)  # edited: kept
        assert md["CO"].temp == 600.0 and store.names == ["Defaults", "Defaults (2)"]
        store.switch("Defaults", md, apply)
        assert md["CO"].temp == 777.0
        store.open(ParameterSet(ROWS, name="set.csv"), md, apply, path="/a/set.csv")
        assert store.open(ParameterSet(ROWS, name="set.csv"), md, apply, path="/b/set.csv") == "set.csv (2)"

    def test_unknown_set(self, setup):
        md, store, apply = setup
        with pytest.raises(KeyError):
            store.switch("nope", md, apply)
        empty = ParameterSetStore()
        assert not empty.revert(md, apply) and not empty.is_modified(md)


class TestIslatParameterSets:
    @pytest.fixture
    def islat(self, dicts):
        from iSLAT.iSLATClass import iSLAT
        obj = iSLAT.__new__(iSLAT)
        obj._user_settings = {"default_active_molecule": "CO", "intensity_calculation_method": "curve_growth"}
        obj._initial_molecule_parameters = {}
        obj.molecules_dict = dicts(ROWS)
        obj.parameter_sets = ParameterSetStore()
        obj.GUI = None
        return obj

    def test_save_open_and_revert(self, islat, tmp_path):
        a = str(tmp_path / "A-molsave.csv")
        assert islat.save_parameters_to(a) == a
        assert islat.parameter_sets.active == "A-molsave.csv"

        islat.molecules_dict["CO"].temp = 1500.0
        b = str(tmp_path / "B.csv")
        islat.save_parameters_to(b)
        islat.molecules_dict["CO"].temp = 1600.0

        assert islat._load_parameters_from_explicit_file(a)  # B's edit stays in memory
        assert islat.molecules_dict["CO"].temp == 600.0
        islat.open_parameter_set(ParameterSet.from_csv(b), path=b)
        assert islat.molecules_dict["CO"].temp == 1600.0
        assert islat.revert_parameters() and islat.molecules_dict["CO"].temp == 1500.0


class TestTopBarSave:
    @pytest.fixture
    def top_bar(self, monkeypatch):
        mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")
        bar = mod.TopBar.__new__(mod.TopBar)
        bar.islat = MagicMock()
        bar.islat.GUI = None
        bar.islat.parameter_sets = ParameterSetStore()
        ask = MagicMock(return_value="yes")
        monkeypatch.setattr(mod.messagebox, "askquestion", ask)
        return bar, ask

    def test_no_question_when_saving_the_set_being_edited(self, top_bar, tmp_path):
        bar, ask = top_bar
        target = tmp_path / "spec-molsave.csv"
        target.write_text("x")
        bar.islat.spectrum_parameter_file.return_value = str(target)
        bar.islat.parameter_sets._set("spec-molsave.csv", ParameterSet([]), str(target))
        bar.save_parameters()
        ask.assert_not_called()
        bar.islat.save_parameters_to.assert_called_once_with(str(target))

    def test_asks_before_replacing_another_sets_file(self, top_bar, tmp_path):
        bar, ask = top_bar
        target = tmp_path / "spec-molsave.csv"
        target.write_text("x")
        bar.islat.spectrum_parameter_file.return_value = str(target)
        ask.return_value = "no"
        bar.save_parameters()
        ask.assert_called_once()
        bar.islat.save_parameters_to.assert_not_called()


class TestStoreManagement:
    @pytest.fixture
    def setup(self, dicts):
        md = dicts(ROWS)
        store = ParameterSetStore()
        store.ensure_active(md)
        apply = lambda pset: pset.apply_to(md, {})
        events = []
        store.add_listener(lambda: events.append(store.active))
        store.open(ParameterSet(ROWS, name="A"), md, apply, path="/saves/A.csv")
        return md, store, apply, events

    def test_new_set_keeps_the_current_parameters(self, setup):
        md, store, apply, events = setup
        md["CO"].temp = 777.0
        assert store.new(md, "  Trial ") == "Trial" and store.active == "Trial"
        assert not store.is_modified(md) and store.path() is None
        assert store.is_modified(name="A")  # the edit stays with A too
        assert store.new(md, "") == "Set" and store.new(md, "Trial") == "Trial (2)"
        assert events[-1] == "Trial (2)"

    def test_rename_keeps_order_path_and_edits(self, setup):
        md, store, apply, events = setup
        store.new(md, "B")
        store.switch("A", md, apply)
        md["CO"].temp = 777.0
        assert store.rename("A", "Fit 1") == "Fit 1"
        assert store.names == ["Fit 1", "B"] and store.active == "Fit 1"
        assert store.path() == "/saves/A.csv" and store.is_modified(md)
        assert store.rename("Fit 1", "B") == "B (2)" and store.rename("B", " ") == "B"
        with pytest.raises(KeyError):
            store.rename("nope", "x")

    def test_close_shows_a_neighbour(self, setup):
        md, store, apply, events = setup
        store.new(md, "B")
        md["CO"].temp = 888.0
        store.new(md, "C")
        store.switch("B", md, apply)
        store.close("B", md, apply)  # the previous set is shown
        assert store.names == ["A", "C"] and store.active == "A" and md["CO"].temp == 600.0
        store.close("A", md, apply)  # no previous set: the next one
        assert store.active == "C" and md["CO"].temp == 888.0
        store.close("C", md, apply)
        assert store.names == [] and store.active is None
        assert store.find("/saves/A.csv") is None

    def test_cycle_wraps(self, setup):
        md, store, apply, events = setup
        store.new(md, "B")
        store.new(md, "C")
        assert [store.cycle(1, md, apply) for _ in range(3)] == ["A", "B", "C"]
        assert store.cycle(-1, md, apply) == "B"
        assert ParameterSetStore().cycle(1, md, apply) is None

    def test_listeners_hear_every_change(self, setup):
        md, store, apply, events = setup
        before = len(events)
        store.new(md, "B")
        store.switch("A", md, apply)
        store.rename("A", "A2")
        store.revert(md, apply)
        store.close("B", md, apply)
        assert len(events) - before == 5
        listener = lambda: events.append("x")
        store.add_listener(listener)
        store.remove_listener(listener)
        store.new(md, "D")
        assert "x" not in events


class TestParameterSetMenu:
    @pytest.fixture
    def menu(self, dicts, monkeypatch):
        import tkinter as tk
        from iSLAT.iSLATClass import iSLAT
        from iSLAT.Modules.GUI.Widgets.ParameterSetMenu import ParameterSetMenu
        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip("Tk display not available")
        root.withdraw()
        islat = iSLAT.__new__(iSLAT)
        islat._user_settings = {"default_active_molecule": "CO", "intensity_calculation_method": "curve_growth"}
        islat._initial_molecule_parameters = {}
        islat.molecules_dict = dicts(ROWS)
        islat.parameter_sets = ParameterSetStore()
        islat.GUI = None
        widget = ParameterSetMenu(root, {}, islat, 0, 0)
        yield widget, islat, root
        Molecule.remove_molecule_parameter_change_callback(widget._on_parameter_change)
        root.withdraw()
        root.destroy()

    def _labels(self, widget):
        widget._rebuild()
        end = widget.menu.index("end")
        return [widget.menu.entrycget(i, "label") for i in range(end + 1)
                if widget.menu.type(i) != "separator"]

    def test_lists_sets_and_marks_unsaved_changes(self, menu):
        widget, islat, root = menu
        assert widget.button.cget("text") == "Parameter Sets"
        assert self._labels(widget)[0] == "No parameter sets yet"
        islat.parameter_sets.ensure_active(islat.molecules_dict)
        assert widget.button.cget("text") == "Untitled"
        islat.open_parameter_set(ParameterSet(ROWS, name="A"), path="/saves/A.csv")
        islat.parameter_sets.new(islat.molecules_dict, "B")
        assert widget.button.cget("text") == "B"
        islat.molecules_dict["CO"].temp = 777.0   # change callback -> refresh when idle
        root.update()
        assert widget.button.cget("text") == "B \u2022"
        assert self._labels(widget)[:2] == ["A", "B \u2022"]

    def test_switching_from_the_menu(self, menu, monkeypatch):
        widget, islat, root = menu
        islat.parameter_sets.ensure_active(islat.molecules_dict)
        islat.open_parameter_set(ParameterSet(ROWS, name="A"), path="/saves/A.csv")
        islat.molecules_dict["CO"].temp = 777.0
        islat.parameter_sets.new(islat.molecules_dict, "B")
        islat.molecules_dict["CO"].temp = 999.0
        widget._rebuild()
        widget.menu.invoke(0)           # "A"
        assert islat.parameter_sets.active == "A" and islat.molecules_dict["CO"].temp == 777.0
        widget.cycle(1)
        assert islat.parameter_sets.active == "B" and islat.molecules_dict["CO"].temp == 999.0

    def test_new_rename_close(self, menu, monkeypatch):
        widget, islat, root = menu
        mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.ParameterSetMenu")
        islat.parameter_sets.ensure_active(islat.molecules_dict)
        islat.open_parameter_set(ParameterSet(ROWS, name="A"), path="/saves/A.csv")
        monkeypatch.setattr(mod.simpledialog, "askstring", lambda *a, **k: "Trial")
        assert widget.new_set() == "Trial"
        monkeypatch.setattr(mod.simpledialog, "askstring", lambda *a, **k: "Trial 2")
        assert widget.rename_set() == "Trial 2"
        islat.molecules_dict["CO"].temp = 777.0
        ask = MagicMock(return_value="no")
        monkeypatch.setattr(mod.messagebox, "askquestion", ask)
        widget.close_set()                    # unsaved changes: asks, and keeps it
        assert "Trial 2" in islat.parameter_sets and ask.called
        ask.return_value = "yes"
        widget.close_set()
        assert islat.parameter_sets.names == ["A"] and islat.molecules_dict["CO"].temp == 600.0
        monkeypatch.setattr(mod.simpledialog, "askstring", lambda *a, **k: None)
        assert widget.new_set() is None       # cancelled


def test_palette_lists_the_other_sets(dicts):
    from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
    from iSLAT.Modules.GUI.KeyBindings import display_title
    md = dicts(ROWS)
    store = ParameterSetStore()
    store.ensure_active(md)
    apply = lambda pset: pset.apply_to(md, {})
    store.open(ParameterSet(ROWS, name="A"), md, apply, path="/saves/A.csv")
    store.new(md, "B")
    menu = MagicMock(store=store)
    h = InteractionHandler.__new__(InteractionHandler)
    h._parameter_set_menu = lambda: menu
    actions = h._parameter_set_actions()
    assert [display_title(a) for a in actions] == ["Switch to set: A"]
    actions[0].callback()
    menu.switch.assert_called_once_with("A")


class TestUntitled:
    @pytest.fixture
    def setup(self, dicts):
        md = dicts(ROWS)
        store = ParameterSetStore()
        apply = lambda pset: pset.apply_to(md, {})
        return md, store, apply

    def test_startup_parameters_are_untitled_and_dropped_when_untouched(self, setup):
        md, store, apply = setup
        assert store.ensure_active(md) == "Untitled" and store.path() is None
        assert store.ensure_active(md) == "Untitled" and store.names == ["Untitled"]
        store.open(ParameterSet([_row("CO", 400.0)], name="A"), md, apply, path="/saves/A.csv")
        assert store.names == ["A"]

    def test_edited_untitled_is_kept(self, setup):
        md, store, apply = setup
        store.ensure_active(md)
        md["CO"].temp = 777.0
        store.open(ParameterSet([_row("CO", 400.0)], name="A"), md, apply, path="/saves/A.csv")
        assert store.names == ["Untitled", "A"]
        store.switch("Untitled", md, apply)
        assert md["CO"].temp == 777.0

    def test_parameters_in_no_set_are_never_dropped(self, setup):
        md, store, apply = setup  # e.g. after closing the last set
        md["CO"].temp = 777.0
        store.open(ParameterSet([_row("CO", 400.0)], name="A"), md, apply, path="/saves/A.csv")
        store.switch("Untitled", md, apply)
        assert md["CO"].temp == 777.0

    def test_saving_from_untitled_moves_it_to_the_file(self, setup):
        md, store, apply = setup
        store.ensure_active(md)
        md["CO"].temp = 777.0
        store.mark_saved(md, "/saves/fit.csv")
        assert store.names == ["fit.csv"] and not store.is_modified(md)

    def test_renamed_untitled_is_kept(self, setup):
        md, store, apply = setup
        store.ensure_active(md)
        store.rename("Untitled", "Baseline")
        store.open(ParameterSet([_row("CO", 400.0)], name="A"), md, apply, path="/saves/A.csv")
        assert store.names == ["Baseline", "A"]

    def test_closing_the_last_set_keeps_the_shown_parameters(self, setup):
        md, store, apply = setup
        store.ensure_active(md)
        store.close("Untitled", md, apply)
        assert store.active is None and len(md) == 3
        store.open(ParameterSet([_row("CO", 400.0)], name="A"), md, apply, path="/saves/A.csv")
        assert store.names == ["Untitled", "A"]

    def test_first_molecules_loaded_become_untitled(self, dicts):
        from iSLAT.iSLATClass import iSLAT
        islat = iSLAT.__new__(iSLAT)
        islat._user_settings = {"default_active_molecule": "CO", "intensity_calculation_method": "curve_growth"}
        islat._initial_molecule_parameters = {}
        islat.molecules_dict = dicts([])
        islat.parameter_sets = ParameterSetStore()
        assert islat.init_molecules({r["Molecule Name"]: r for r in ROWS})
        assert islat.parameter_sets.names == ["Untitled"] and not islat.parameter_sets.is_modified(islat.molecules_dict)
        islat.init_molecules({"NEW": _row("NEW")})  # adding molecules later is an edit, not a new set
        assert islat.parameter_sets.names == ["Untitled"] and islat.parameter_sets.is_modified(islat.molecules_dict)


def test_menu_label_keeps_the_mark_on_long_names(dicts):
    import tkinter as tk
    from iSLAT.Modules.GUI.Widgets.ParameterSetMenu import MODIFIED_MARK, ParameterSetMenu
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available")
    root.withdraw()
    islat = MagicMock()
    islat.molecules_dict = dicts(ROWS)
    islat.parameter_sets = ParameterSetStore()
    widget = ParameterSetMenu(root, {}, islat, 0, 0)
    try:
        islat.parameter_sets.ensure_active(islat.molecules_dict)
        islat.parameter_sets.rename("Untitled", "CITau_1d_v9.0_csub-molsave.csv")
        islat.molecules_dict["CO"].temp = 777.0
        widget.refresh()
        text = widget.button.cget("text")
        assert text.endswith("…" + MODIFIED_MARK) and len(text) < 40
    finally:
        Molecule.remove_molecule_parameter_change_callback(widget._on_parameter_change)
        root.withdraw()
        root.destroy()
