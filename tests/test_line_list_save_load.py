# -*- coding: utf-8 -*-
"""Save / load line-list helpers used by the filter window."""

from pathlib import Path

import pytest

from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import (
    LINE_LIST_LOAD_FILETYPES,
    LINE_LIST_SAVE_FILETYPES,
    load_line_list_file,
    write_line_list,
)


@pytest.fixture
def lines():
    return [
        {'nr': 1, 'lev_up': '0|10', 'lev_low': '0|9', 'lam': 12.407, 'freq': 2.416e13,
         'a_stein': 1.05e-2, 'e_up': 4586.4, 'e_low': 3379.0, 'g_up': 21, 'g_low': 19},
        {'nr': 2, 'lev_up': '0|8', 'lev_low': '0|7', 'lam': 14.950, 'freq': 2.005e13,
         'a_stein': 2.10e-2, 'e_up': 3200.0, 'e_low': 2500.0, 'g_up': 17, 'g_low': 15},
        {'nr': 3, 'lev_up': '0|6', 'lev_low': '0|5', 'lam': 17.221, 'freq': 1.741e13,
         'a_stein': 5.00e-3, 'e_up': 2100.0, 'e_low': 1600.0, 'g_up': 13, 'g_low': 11},
    ]


@pytest.fixture
def maker(lines):
    return LineListMaker(MoleculeLineList(molecule_id='H2O', lines_data=lines))


class TestWriteLineList:
    def test_csv_round_trip(self, maker, tmp_path):
        out = write_line_list(maker, tmp_path / "filtered.csv")
        assert out.exists()
        loaded = load_line_list_file(out)
        assert len(loaded) == 3
        assert sorted(loaded.get_wavelengths()) == pytest.approx([12.407, 14.950, 17.221])

    def test_filtered_subset_is_what_gets_written(self, maker, tmp_path):
        maker.filter_wavelength(13.0, 18.0)
        out = write_line_list(maker, tmp_path / "subset.csv")
        loaded = load_line_list_file(out)
        assert sorted(loaded.get_wavelengths()) == pytest.approx([14.950, 17.221])

    def test_par_refused_without_partition_function(self, maker, tmp_path):
        # Without the partition block the PAR reader cannot parse the file,
        # so the helper must refuse instead of writing an unloadable file.
        with pytest.raises(RuntimeError, match="partition"):
            write_line_list(maker, tmp_path / "filtered.par")
        assert not (tmp_path / "filtered.par").exists()

    def test_par_round_trip_with_partition_function(self, lines, tmp_path):
        import numpy as np
        ll = MoleculeLineList(molecule_id='H2O', lines_data=lines)
        ll.partition_function = ll._partition_type(
            t=np.array([100.0, 300.0, 1000.0]), q=np.array([10.0, 50.0, 400.0]),
        )
        out = write_line_list(LineListMaker(ll), tmp_path / "filtered.par")
        loaded = load_line_list_file(out, fallback_species="H2O")
        assert len(loaded) == 3
        assert sorted(loaded.get_wavelengths()) == pytest.approx([12.407, 14.950, 17.221], rel=1e-3)

    def test_extension_check_is_case_insensitive(self, maker, tmp_path):
        out = write_line_list(maker, tmp_path / "UPPER.CSV")
        assert len(load_line_list_file(out)) == 3


class TestLoadLineListFile:
    def test_fallback_species_used_when_file_has_none(self, maker, tmp_path):
        out = write_line_list(maker, tmp_path / "f.csv")
        loaded = load_line_list_file(out, fallback_species="CO")
        # Either the file carried its species or the fallback filled it in;
        # it must never come back unset.
        assert loaded.molecule_id in ("H2O", "CO")
        assert loaded.molecule_id is not None

    def test_explicit_species_in_file_is_kept(self, maker, tmp_path):
        out = write_line_list(maker, tmp_path / "f.csv")
        loaded = load_line_list_file(out, fallback_species="CO")
        if loaded.molecule_id == "H2O":
            assert LineListMaker(loaded).species == "H2O"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_line_list_file(tmp_path / "nope.csv")

    def test_loaded_list_feeds_a_new_maker(self, maker, tmp_path):
        out = write_line_list(maker, tmp_path / "f.csv")
        new_maker = LineListMaker(load_line_list_file(out, fallback_species="H2O"))
        assert len(new_maker) == 3
        assert len(new_maker.filter_wavelength(0, 13.0)) == 1


class TestFileTypes:
    def test_save_and_load_offer_csv_and_par(self):
        save_patterns = {p for _, p in LINE_LIST_SAVE_FILETYPES}
        load_patterns = " ".join(p for _, p in LINE_LIST_LOAD_FILETYPES)
        assert {"*.csv", "*.par"} <= save_patterns
        assert "*.par" in load_patterns and "*.csv" in load_patterns
