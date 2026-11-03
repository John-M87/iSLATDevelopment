# -*- coding: utf-8 -*-
"""Line-list paths in molecule lists: keep sub-folders, repair old saves."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import iSLAT.Modules.FileHandling.line_list_paths as llp


@pytest.fixture
def datafiles(tmp_path, monkeypatch):
    """A throw-away DATAFILES tree: HITRANdata/{top.par, Sub/nested.par, cache/nested.par}."""
    root = tmp_path / "DATAFILES"
    hitran = root / "HITRANdata"
    (hitran / "Sub").mkdir(parents=True)
    (hitran / "cache").mkdir()
    (hitran / "top.par").write_text("x")
    (hitran / "Sub" / "nested.par").write_text("x")
    (hitran / "cache" / "nested.par").write_text("cached copy")
    monkeypatch.setattr(llp, "absolute_data_files_path", str(root))
    monkeypatch.setattr(llp, "hitran_data_folder_path", hitran)
    return root


class TestPortablePath:
    def test_inside_datafiles_keeps_sub_folders(self, datafiles):
        nested = datafiles / "HITRANdata" / "Sub" / "nested.par"
        assert llp.to_portable_line_list_path(nested) == "HITRANdata/Sub/nested.par"
        assert llp.to_portable_line_list_path(str(datafiles / "HITRANdata" / "top.par")) == "HITRANdata/top.par"

    def test_outside_datafiles_is_absolute(self, datafiles, tmp_path):
        other = tmp_path / "elsewhere" / "x.par"
        assert llp.to_portable_line_list_path(other) == str(other.resolve())

    def test_relative_input_is_cwd_relative(self, datafiles, monkeypatch):
        monkeypatch.chdir(datafiles / "HITRANdata" / "Sub")
        assert llp.to_portable_line_list_path("nested.par") == "HITRANdata/Sub/nested.par"


class TestFindAndRepair:
    def test_find_existing(self, datafiles):
        assert llp.find_existing_line_list("HITRANdata/Sub/nested.par")
        assert llp.find_existing_line_list(datafiles / "HITRANdata" / "top.par")
        assert llp.find_existing_line_list("HITRANdata/nested.par") is None

    def test_existing_or_empty_path_is_unchanged(self, datafiles):
        assert llp.repair_line_list_path("HITRANdata/top.par") == "HITRANdata/top.par"
        assert llp.repair_line_list_path(None) is None
        assert llp.repair_line_list_path("") == ""

    def test_missing_sub_folder_is_repaired_ignoring_cache(self, datafiles, capsys):
        assert llp.repair_line_list_path("HITRANdata/nested.par") == "HITRANdata/Sub/nested.par"
        assert "not found; using 'HITRANdata/Sub/nested.par'" in capsys.readouterr().out

    def test_ambiguous_or_unknown_is_unchanged(self, datafiles):
        (datafiles / "HITRANdata" / "Other").mkdir()
        (datafiles / "HITRANdata" / "Other" / "nested.par").write_text("x")
        assert llp.repair_line_list_path("HITRANdata/nested.par") == "HITRANdata/nested.par"
        assert llp.repair_line_list_path("HITRANdata/nope.par") == "HITRANdata/nope.par"


class TestMoleculeUsesRepairedPath:
    def _molecule(self, **kw):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        return Molecule(name="X", wavelength_range=(5, 6), **kw)

    def test_saved_row_with_missing_sub_folder(self, datafiles):
        mol = self._molecule(user_save_data={"Molecule Name": "X", "File Path": "HITRANdata/nested.par"})
        assert mol.filepath == "HITRANdata/Sub/nested.par"

    def test_kwargs_filepath_and_untouched_valid_path(self, datafiles):
        assert self._molecule(filepath="HITRANdata/nested.par").filepath == "HITRANdata/Sub/nested.par"
        assert self._molecule(filepath="HITRANdata/top.par").filepath == "HITRANdata/top.par"


class TestAddMoleculeFromHitran:
    def test_stores_path_with_sub_folders(self, datafiles, monkeypatch):
        import iSLAT.iSLATClass as islat_mod
        monkeypatch.setattr(islat_mod, "to_portable_line_list_path", llp.to_portable_line_list_path)
        monkeypatch.setattr(islat_mod, "read_full_molecule_parameters", lambda: {})
        islat = islat_mod.iSLAT.__new__(islat_mod.iSLAT)
        islat.molecules_dict = MagicMock()
        islat.molecules_dict.load_molecules.return_value = {"success": 1}
        nested = str(datafiles / "HITRANdata" / "Sub" / "nested.par")

        islat.add_molecule_from_hitran(refresh=False, hitran_files=[nested],
                                       molecule_names=["Fe I"], name_popups=False)
        (mol_data,) = islat.molecules_dict.load_molecules.call_args.args[0]
        assert mol_data["file"] == "HITRANdata/Sub/nested.par"
        assert mol_data["hitran_data"] == nested


def test_real_sub_folder_line_lists_resolve():
    """Regression for saves listing a HITRANdata sub-folder's files as HITRANdata/<file>."""
    from iSLAT.Modules.FileHandling import hitran_data_folder_path
    root = Path(hitran_data_folder_path)
    nested = sorted(p for p in root.glob("*/*.par")
                    if p.parent.name not in ("cache", "__pycache__")) if root.is_dir() else []
    unique = [p for p in nested if len(list(root.rglob(p.name))) == 1]
    if not unique:
        pytest.skip("No uniquely named line lists in HITRANdata sub-folders in this checkout")
    target = unique[0]
    stale = f"HITRANdata/{target.name}"
    assert llp.repair_line_list_path(stale) == target.relative_to(root.parent).as_posix()
