"""Level labels are stored as byte strings and come back out as str."""
import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataTypes.MoleculeLineList import (
    MoleculeLineList, _LINE_DTYPE, _decode_labels, _encode_labels,
)


def _lines(labels):
    return [{"nr": i + 1, "lev_up": up, "lev_low": low, "lam": 5.0 + i, "freq": 1e13,
             "a_stein": 0.1, "e_up": 100.0, "e_low": 50.0, "g_up": 3, "g_low": 1}
            for i, (up, low) in enumerate(labels)]


HITRAN = [("0_0_0|10_2_9", "0_0_0|9_3_6"), ("0_0_1|5_1_5", "0_0_1|4_2_2")]


def test_row_is_112_bytes():
    assert _LINE_DTYPE.itemsize == 112
    assert _LINE_DTYPE["lev_up"] == np.dtype("S30")


def test_every_accessor_returns_str_labels():
    ll = MoleculeLineList(molecule_id="H2O", lines_data=_lines(HITRAN))
    ups = [u for u, _ in HITRAN]
    lows = [lo for _, lo in HITRAN]
    assert ll._raw_lines_data.dtype["lev_up"].kind == "S"  # stored as bytes
    assert list(ll.get_upper_levels()) == ups
    assert list(ll.get_lower_levels()) == lows
    assert list(ll.get_ndarray_of_attribute("lev_up")) == ups
    assert list(ll.lines_as_namedtuple.lev_up) == ups
    assert list(ll.get_pandas_table()["lev_low"]) == lows
    assert list(ll._get_pandas_dataframe()["lev_up"]) == ups
    lines = ll.get_lines_in_range(0, 100)
    assert [(l.lev_up, l.lev_low) for l in lines] == HITRAN
    assert all(type(l.lev_up) is str for l in lines)
    ll._ensure_lines_created()
    assert [l.lev_up for l in ll.lines] == ups


def test_long_and_non_ascii_labels_are_kept_whole():
    long_label = "x" * 45
    labels = [(long_label, "é|1"), ("a", "b")]
    ll = MoleculeLineList(molecule_id="X", lines_data=_lines(labels))
    assert list(ll.get_upper_levels()) == [long_label, "a"]
    assert list(ll.get_lower_levels()) == ["é|1", "b"]
    assert ll._raw_lines_data.dtype["lev_up"].itemsize >= 45


def test_encode_decode_round_trip():
    for labels in (["0|1", "", "abc"], ["é", "ü|2"], ["y" * 40]):
        assert list(_decode_labels(_encode_labels(labels))) == labels
    assert _decode_labels(np.array([], dtype="S30")).size == 0
    plain = np.array(["a", "b"])
    assert _decode_labels(plain) is plain


def test_par_file_round_trip(tmp_path, monkeypatch):
    import iSLAT.Modules.FileHandling as fh
    monkeypatch.setattr(fh, "line_cache_folder_path", str(tmp_path / "cache"))
    ll = MoleculeLineList(molecule_id="H2O", lines_data=_lines(HITRAN))
    ll._molar_mass = 18.01  # the reader needs a molar mass in the header
    ll.partition_function = ll._partition_type(np.array([10.0, 20.0]), np.array([1.0, 2.0]))
    out = tmp_path / "out.par"
    ll.write_par_file(out)
    text = out.read_text()
    assert "b'" not in text
    back = MoleculeLineList(molecule_id="H2O", filename=str(out))
    back._source_format = "hitran"
    assert list(back.get_upper_levels()) == [u for u, _ in HITRAN]
