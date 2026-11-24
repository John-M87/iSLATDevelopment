"""ParameterSet: capture, apply in place, and the load paths that use it."""

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.DataTypes.ParameterSet import APPLIED_PARAMETERS, SAVE_COLUMNS, ParameterSet

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
