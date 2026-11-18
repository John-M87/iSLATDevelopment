"""Wavelength-ordered line lists: binary-search range queries, the v4 cache and deleting invalid caches."""
import os

import numpy as np
import pytest

from iSLAT.Modules.DataTypes import MoleculeLineList as mll_mod
from iSLAT.Modules.DataTypes.MoleculeLineList import (
    MoleculeLineList,
    _CACHE_VERSION,
    _LAM_ASCENDING,
    _LAM_DESCENDING,
    _LAM_UNSORTED,
    _rows_in_wavelength_range,
    _wavelength_order,
)


def _lines(lams):
    return [{"nr": i + 1, "lev_up": f"u{i}", "lev_low": f"l{i}", "lam": lam,
             "freq": 3e14 / lam, "a_stein": 0.1, "e_up": 100.0 + i, "e_low": 50.0,
             "g_up": 3, "g_low": 1} for i, lam in enumerate(lams)]


def _structured(lams):
    return MoleculeLineList(molecule_id="X", lines_data=_lines(lams))._raw_lines_data


ASC = [5.0, 6.0, 6.0, 7.5, 10.0, 12.0, 12.0, 15.0, 20.0]
RANGES = [(6.0, 12.0), (5.5, 11.0), (0.0, 100.0), (21.0, 30.0), (0.0, 4.0),
          (12.0, 12.0), (13.0, 13.5), (15.0, 5.0)]


class TestWavelengthOrder:
    @pytest.mark.parametrize("lams, expected", [
        (ASC, _LAM_ASCENDING), (ASC[::-1], _LAM_DESCENDING),
        ([5.0, 7.0, 6.0], _LAM_UNSORTED), ([1.0, np.nan, 2.0], _LAM_UNSORTED),
        ([], _LAM_ASCENDING), ([3.0], _LAM_ASCENDING),
    ])
    def test_detection(self, lams, expected):
        assert _wavelength_order(np.array(lams, dtype=float)) == expected


class TestRowsInRange:
    @pytest.mark.parametrize("lams", [ASC, ASC[::-1], [12.0, 5.0, 20.0, 6.0, 15.0, 6.0]])
    @pytest.mark.parametrize("lo, hi", RANGES)
    def test_matches_boolean_mask(self, lams, lo, hi):
        data = _structured(lams)
        expected = data[(data["lam"] >= lo) & (data["lam"] <= hi)]
        got = _rows_in_wavelength_range(data, lo, hi, _wavelength_order(data["lam"]))
        np.testing.assert_array_equal(got, expected)  # same rows, same order

    def test_result_is_a_writable_copy(self):
        data = _structured(ASC)
        data.flags.writeable = False
        got = _rows_in_wavelength_range(data, 6.0, 12.0, _LAM_ASCENDING)
        assert got.flags.writeable and not np.shares_memory(got, data)

    def test_only_reads_the_slice(self):
        """Ordered data is bisected: lam is read ~log2(n) times, not n."""
        data = _structured(np.linspace(1.0, 30.0, 4096))

        class CountingLam:
            reads = 0

            def __len__(self):
                return len(data)

            def __getitem__(self, i):
                CountingLam.reads += 1
                return data["lam"][i]

        class Wrapper:
            def __getitem__(self, key):
                return CountingLam() if key == "lam" else data[key]

        got = _rows_in_wavelength_range(Wrapper(), 13.0, 17.0, _LAM_ASCENDING)
        assert CountingLam.reads <= 2 * 13
        assert got["lam"].min() >= 13.0 and got["lam"].max() <= 17.0


class TestMoleculeLineListRanges:
    @pytest.mark.parametrize("lams", [ASC, ASC[::-1], [12.0, 5.0, 20.0, 6.0, 15.0, 6.0]])
    def test_active_data_and_lines_in_range(self, lams):
        ll = MoleculeLineList(molecule_id="X", lines_data=_lines(lams), wavelength_range=(6.0, 15.0))
        lam = np.array(lams)
        np.testing.assert_array_equal(ll.get_wavelengths(), lam[(lam >= 6.0) & (lam <= 15.0)])
        got = [line.lam for line in ll.get_lines_in_range(7.0, 12.0)]
        assert got == [x for x in lams if 6.0 <= x <= 15.0 and 7.0 <= x <= 12.0]

    def test_in_memory_lists_keep_their_order(self):
        lams = [12.0, 5.0, 20.0]
        ll = MoleculeLineList(molecule_id="X", lines_data=_lines(lams))
        np.testing.assert_array_equal(ll.get_wavelengths(), lams)

    def test_copies_keep_the_order(self):
        ll = MoleculeLineList(molecule_id="X", lines_data=_lines(ASC[::-1]), wavelength_range=(6.0, 12.0))
        ll.get_wavelengths()
        assert ll.copy()._lam_order == _LAM_DESCENDING
        assert ll.without_wavelength_range()._lam_order == _LAM_DESCENDING


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

UNSORTED = [12.0, 5.0, 20.0, 6.0, 15.0]


def _write_csv(path, lams):
    rows = ["lam,a_stein,e_up,e_low,g_up,g_low,lev_up,lev_low,note"]
    rows += [f"{lam},0.1,{100 + i},50,3,1,u{i},l{i},n{i}" for i, lam in enumerate(lams)]
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    import iSLAT.Modules.FileHandling as fh
    root = tmp_path / "cache"
    monkeypatch.setattr(fh, "line_cache_folder_path", str(root))
    return root


def _load(path):
    ll = MoleculeLineList(molecule_id="X", filename=str(path), format="csv")
    ll._ensure_data_loaded()
    return ll


def _cache_dir(cache_root, path):
    return cache_root / f"X_{os.path.basename(path)}"


class TestCache:
    def test_unsorted_source_is_cached_sorted_with_extras_aligned(self, tmp_path, cache_root):
        src = tmp_path / "lines.csv"
        _write_csv(src, UNSORTED)
        first = _load(src)  # cache miss: parse, sort, save
        second = _load(src)  # cache hit
        for ll in (first, second):
            assert ll._lam_order == _LAM_ASCENDING
            np.testing.assert_array_equal(ll.get_wavelengths(), sorted(UNSORTED))
            df = ll.get_pandas_table(include_extras=True)
            # each line keeps its own extra value and labels
            for lam, note, lev in zip(df["lam"], df["note"], df["lev_up"]):
                i = UNSORTED.index(lam)
                assert (note, lev) == (f"n{i}", f"u{i}")
        meta = np.load(_cache_dir(cache_root, src) / "metadata.npy")
        assert (int(meta[0]), int(meta[2])) == (_CACHE_VERSION, _LAM_ASCENDING)

    def test_descending_source_keeps_its_order(self, tmp_path, cache_root):
        src = tmp_path / "desc.csv"
        _write_csv(src, sorted(UNSORTED, reverse=True))
        _load(src)
        ll = _load(src)
        assert ll._lam_order == _LAM_DESCENDING
        np.testing.assert_array_equal(ll.get_wavelengths(), sorted(UNSORTED, reverse=True))

    def test_no_temporary_file_left_behind(self, tmp_path, cache_root):
        src = tmp_path / "lines.csv"
        _write_csv(src, UNSORTED)
        _load(src)
        assert sorted(os.listdir(_cache_dir(cache_root, src))) == [
            "extra_columns.npy", "lines.npy", "metadata.npy",
            "partition_q.npy", "partition_t.npy", "source_format.npy",
        ]

    @pytest.mark.parametrize("metadata", [
        np.array([3, 0.0]),            # v3: no wavelength order
        np.array([_CACHE_VERSION]),    # unreadable: too short
    ])
    def test_invalid_cache_is_deleted_and_rebuilt(self, tmp_path, cache_root, metadata):
        src = tmp_path / "old.csv"
        _write_csv(src, UNSORTED)
        _load(src)
        cache = _cache_dir(cache_root, src)
        np.save(cache / "metadata.npy", metadata)
        (cache / "leftover.npy").write_bytes(b"stale")

        ll = _load(src)
        assert not (cache / "leftover.npy").exists()  # old directory was deleted
        assert ll._lam_order == _LAM_ASCENDING
        np.testing.assert_array_equal(ll.get_wavelengths(), sorted(UNSORTED))
        meta = np.load(cache / "metadata.npy")
        assert (int(meta[0]), int(meta[2])) == (_CACHE_VERSION, _LAM_ASCENDING)

    def test_cache_older_than_source_is_deleted_and_rebuilt(self, tmp_path, cache_root):
        src = tmp_path / "edited.csv"
        _write_csv(src, UNSORTED)
        _load(src)
        cache = _cache_dir(cache_root, src)
        (cache / "leftover.npy").write_bytes(b"stale")
        _write_csv(src, [8.0, 3.0])
        later = os.path.getmtime(cache / "metadata.npy") + 10
        os.utime(src, (later, later))

        ll = _load(src)
        assert not (cache / "leftover.npy").exists()
        np.testing.assert_array_equal(ll.get_wavelengths(), [8.0, 3.0])  # new data, kept descending

    def test_cache_kept_when_source_is_missing(self, tmp_path, cache_root):
        src = tmp_path / "gone.csv"
        _write_csv(src, UNSORTED)
        _load(src)
        cache = _cache_dir(cache_root, src)
        src.unlink()
        with pytest.raises(Exception):
            _load(src)
        assert (cache / "lines.npy").exists()

    def test_memory_mapped_cache_range_query(self, tmp_path, cache_root, monkeypatch):
        """Large caches are memory-mapped; the range result is a normal in-memory copy."""
        src = tmp_path / "big.csv"
        lams = list(np.round(np.linspace(30.0, 2.0, 2000), 6))
        _write_csv(src, lams)
        _load(src)
        monkeypatch.setattr(os.path, "getsize", lambda p: 10 ** 9)  # force mmap on load
        ll = MoleculeLineList(molecule_id="X", filename=str(src), format="csv",
                              wavelength_range=(13.0, 17.0))
        ll._ensure_data_loaded()
        assert isinstance(ll._raw_lines_data, np.memmap)
        active = ll._get_active_raw_data()
        assert not isinstance(active, np.memmap) and active.flags.writeable
        lam = np.array(lams)
        np.testing.assert_array_equal(active["lam"], lam[(lam >= 13.0) & (lam <= 17.0)])
