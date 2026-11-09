# -*- coding: utf-8 -*-
"""Tests for finding the lines of several files in a wavelength range.

Covers the range-filtered .par reader (``read_par_lines_in_range``), the
search module (``LineRangeSearch``: .par and CSV files, combining, errors,
cancelling, folder and molecule file lists), the Find Lines in Range window
helpers and the window itself (Tk window tests are skipped without a display).
"""
import gc
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataProcessing import LineRangeSearch as lrs
from iSLAT.Modules.DataProcessing.LineRangeSearch import (
    FILE_COLUMN,
    LINE_COLUMNS,
    LineSearchError,
    find_line_list_files,
    lines_in_range,
    molecule_line_list_files,
    normalize_range,
    search_line_lists,
    species_from_par_header,
)
from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
from iSLAT.Modules.FileHandling.molecular_data_reader import (
    read_molecular_data,
    read_par_lines_in_range,
)
from iSLAT.Modules.GUI.Widgets import LineSearchWindow as lsw

from tests.tk_events import pump_tk


def _line(nr, lam, lev_up="0|10", lev_low="0|9", a_stein=1e-2):
    return {
        "nr": nr, "lev_up": lev_up, "lev_low": lev_low, "lam": lam,
        "freq": 2.99792458e14 / lam, "a_stein": a_stein,
        "e_up": 1000.0 + nr, "e_low": 900.0 + nr, "g_up": 21, "g_low": 19,
    }


CO_LINES = [_line(0, 4.70012), _line(1, 4.65003), _line(2, 4.60051), _line(3, 4.65999)]
H2O_LINES = [_line(0, 12.40702), _line(1, 4.65500), _line(2, 17.22101)]
PARTITION = pd.DataFrame({"temp": [100.0, 200.0, 300.0], "q": [10.0, 20.0, 30.0]})


def _write_par(path, molecule_id, lines):
    MoleculeLineList(molecule_id=molecule_id, lines_data=lines).write_par_file(
        path, partition_df=PARTITION)
    return str(path)


@pytest.fixture
def co_par(tmp_path):
    return _write_par(tmp_path / "co.par", "CO", CO_LINES)


@pytest.fixture
def h2o_par(tmp_path):
    return _write_par(tmp_path / "h2o.par", "H2O", H2O_LINES)


@pytest.fixture
def line_list_csv(tmp_path):
    path = tmp_path / "list.csv"
    pd.DataFrame({
        "species": ["OH", "H2O", "OH"],
        "lev_up": ["a", "b", "c"], "lev_low": ["d", "e", "f"],
        "lam": [4.6512, 9.0, 4.6601],
        "a_stein": [1.0, 2.0, 3.0], "e_up": [1.0, 2.0, 3.0], "e_low": [1.0, 2.0, 3.0],
        "g_up": [1, 2, 3], "g_low": [1, 2, 3],
        "xmin": [4.650, 8.99, 4.659], "xmax": [4.652, 9.01, 4.661],
        "note": ["blend", "", "weak"],
    }).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def atomic_csv(tmp_path):
    """Atomic-line style CSV: 'wave' column and a byte-order mark."""
    path = tmp_path / "atomic.csv"
    path.write_text("\ufeffwave,species,line\n4.652,HI,(7-5)\n7.459882,HI,(6-5)\n",
                    encoding="utf-8")
    return str(path)


# ======================================================================
# Range-filtered .par reader
# ======================================================================

class TestParRangeReader:
    def test_returns_only_lines_in_range(self, co_par):
        comments, lines = read_par_lines_in_range(co_par, 4.65, 4.66)
        assert sorted(line["lam"] for line in lines) == [4.65003, 4.65999]
        assert comments[0].startswith("# HITRAN CO;")

    def test_bounds_are_inclusive(self, co_par):
        _, lines = read_par_lines_in_range(co_par, 4.65003, 4.65999)
        assert len(lines) == 2

    def test_lines_have_the_full_reader_fields(self, co_par):
        _, in_range = read_par_lines_in_range(co_par, 0.0, 100.0)
        _, full, _ = read_molecular_data("CO", co_par)
        assert in_range == full

    def test_empty_range(self, co_par):
        _, lines = read_par_lines_in_range(co_par, 20.0, 30.0)
        assert lines == []

    def test_full_read_is_unchanged(self, co_par):
        partition, lines, other = read_molecular_data("CO", co_par)
        assert len(lines) == len(CO_LINES)
        assert np.allclose(partition.t, PARTITION["temp"])
        assert [line["lam"] for line in lines] == [line["lam"] for line in CO_LINES]

    def test_unreadable_wavelength_is_skipped(self, co_par):
        with open(co_par) as fh:
            text = fh.read().splitlines()
        target = next(i for i, row in enumerate(text) if "4.65003" in row)
        row = text[target]
        text[target] = row[:66] + "    garbage" + row[77:]
        with open(co_par, "w") as fh:
            fh.write("\n".join(text) + "\n")
        _, lines = read_par_lines_in_range(co_par, 4.60, 4.70)
        assert sorted(line["lam"] for line in lines) == [4.60051, 4.65999]


# ======================================================================
# Inputs
# ======================================================================

class TestNormalizeRange:
    def test_orders_the_bounds(self):
        assert normalize_range(5.0, 4.0) == (4.0, 5.0)

    def test_accepts_strings(self):
        assert normalize_range(" 4.6 ", "4.7") == (4.6, 4.7)

    @pytest.mark.parametrize("bad", ["", "abc", None, float("nan"), float("inf")])
    def test_rejects_non_numbers(self, bad):
        with pytest.raises(ValueError):
            normalize_range(bad, 5.0)

    def test_rejects_an_empty_range(self):
        with pytest.raises(ValueError, match="same"):
            normalize_range(4.0, "4.0")


class TestSpeciesFromHeader:
    @pytest.mark.parametrize("line, species", [
        ("# HITRAN CO; id:5; iso:1;gid:26", "CO"),
        ("# HITEMP 2019 CO; id:5; iso:1;gid:26", "CO"),
        ("# HITRAN 2024 H2O; id:1; iso:1;gid:1", "H2O"),
        ("# HITRAN 12CH379Br; id:40; iso:1;gid:93", "12CH379Br"),
        ("# HITRAN ; id:; iso:;gid:", ""),
        ("# HITRAN 2020; id:1", ""),
        ("# some comment", "comment"),
    ])
    def test_header_variants(self, line, species):
        assert species_from_par_header([line]) == species

    def test_no_header(self):
        assert species_from_par_header([]) == ""


# ======================================================================
# One file
# ======================================================================

class TestLinesInRange:
    def test_par_columns_and_species(self, co_par):
        found = lines_in_range(co_par, 4.65, 4.66)
        assert list(found.columns) == list(LINE_COLUMNS)
        assert set(found["species"]) == {"CO"}
        assert sorted(found["lam"]) == [4.65003, 4.65999]

    def test_csv_keeps_all_columns(self, line_list_csv):
        found = lines_in_range(line_list_csv, 4.65, 4.66)
        assert list(found["species"]) == ["OH"]
        assert found["note"].tolist() == ["blend"]
        assert "xmin" in found.columns and "xmax" in found.columns

    def test_csv_wave_column_and_byte_order_mark(self, atomic_csv):
        found = lines_in_range(atomic_csv, 4.6, 4.7)
        assert list(found.columns) == ["lam", "species", "line"]
        assert found["lam"].tolist() == [4.652]
        assert found["line"].tolist() == ["(7-5)"]

    def test_legacy_level_columns_are_renamed(self, tmp_path):
        path = tmp_path / "old.csv"
        pd.DataFrame({"lam": [4.65], "up_lev": ["u"], "low_lev": ["l"]}).to_csv(path, index=False)
        found = lines_in_range(path, 4.6, 4.7)
        assert found["lev_up"].tolist() == ["u"] and found["lev_low"].tolist() == ["l"]

    def test_non_numeric_wavelengths_never_match(self, tmp_path):
        path = tmp_path / "mixed.csv"
        pd.DataFrame({"lam": ["4.65", "n/a", ""]}).to_csv(path, index=False)
        assert lines_in_range(path, 4.6, 4.7)["lam"].tolist() == [4.65]

    def test_csv_without_wavelength_column(self, tmp_path):
        path = tmp_path / "settings.csv"
        pd.DataFrame({"Molecule Name": ["CO"], "Temp": [500]}).to_csv(path, index=False)
        with pytest.raises(LineSearchError, match="no wavelength column"):
            lines_in_range(path, 4.6, 4.7)

    def test_missing_file(self, tmp_path):
        with pytest.raises(LineSearchError, match="not found"):
            lines_in_range(tmp_path / "missing.par", 4.6, 4.7)

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("lam\n4.65\n")
        with pytest.raises(LineSearchError, match="Unsupported"):
            lines_in_range(path, 4.6, 4.7)

    def test_broken_par_file(self, tmp_path):
        path = tmp_path / "broken.par"
        path.write_text("# HITRAN X;\nnot a number\nfoo\nbar\n")
        with pytest.raises(LineSearchError, match="Could not read"):
            lines_in_range(path, 4.6, 4.7)

    def test_datafiles_relative_path(self):
        found = lines_in_range("LINELISTS/Atomic_lines.csv", 4.05, 4.06)
        assert found["species"].tolist() == ["HI"]


# ======================================================================
# Several files
# ======================================================================

class TestSearchLineLists:
    def test_combines_files_sorted_by_wavelength(self, co_par, h2o_par, line_list_csv):
        result = search_line_lists([co_par, h2o_par, line_list_csv], 4.66, 4.65)
        assert (result.lam_min, result.lam_max) == (4.65, 4.66)
        lines = result.lines
        assert lines["lam"].tolist() == [4.65003, 4.6512, 4.655, 4.65999]
        assert lines[FILE_COLUMN].tolist() == ["co.par", "list.csv", "h2o.par", "co.par"]
        assert lines["species"].tolist() == ["CO", "OH", "H2O", "CO"]
        assert list(lines.columns[:len(LINE_COLUMNS) + 1]) == [FILE_COLUMN, *LINE_COLUMNS]
        # CSV extras follow; files without them have NaN there.
        assert list(lines.columns[len(LINE_COLUMNS) + 1:]) == ["xmin", "xmax", "note"]
        assert lines.loc[lines[FILE_COLUMN] == "co.par", "note"].isna().all()

    def test_counts_and_summary(self, co_par, h2o_par):
        result = search_line_lists([co_par, h2o_par], 4.64, 4.66)
        assert result.counts_by_file() == {"co.par": 2, "h2o.par": 1}
        assert result.summary() == "Found 3 lines in 4.64 - 4.66 um (2 of 2 files)."

    def test_file_listed_twice_is_searched_once(self, co_par):
        result = search_line_lists([co_par, co_par], 4.6, 4.7)
        assert result.files_searched == [co_par]
        assert result.n_lines == 3

    def test_bad_files_are_reported_and_skipped(self, co_par, tmp_path):
        missing = str(tmp_path / "missing.par")
        result = search_line_lists([missing, co_par], 4.6, 4.7)
        assert result.n_lines == 3
        assert list(result.errors) == [missing]
        assert result.summary().endswith("1 file could not be searched.")

    def test_no_lines_found(self, co_par):
        result = search_line_lists([co_par], 20.0, 21.0)
        assert result.n_lines == 0
        assert list(result.lines.columns) == [FILE_COLUMN, *LINE_COLUMNS]
        assert result.counts_by_file() == {"co.par": 0}
        assert result.summary() == "Found 0 lines in 20.0 - 21.0 um (0 of 1 file)."

    def test_progress_and_cancel(self, co_par, h2o_par):
        seen = []
        result = search_line_lists(
            [co_par, h2o_par], 4.6, 4.7,
            progress=lambda i, n, path: seen.append((i, n, path)),
            should_stop=lambda: len(seen) >= 1,
        )
        assert seen == [(0, 2, co_par)]
        assert result.cancelled
        assert result.files_searched == [co_par]

    def test_a_csv_file_column_is_kept(self, tmp_path):
        path = tmp_path / "with_file.csv"
        pd.DataFrame({"lam": [4.65], "file": ["orig"]}).to_csv(path, index=False)
        lines = search_line_lists([path], 4.6, 4.7).lines
        assert lines[FILE_COLUMN].tolist() == ["with_file.csv"]
        assert lines["file (column)"].tolist() == ["orig"]

    def test_invalid_range(self, co_par):
        with pytest.raises(ValueError):
            search_line_lists([co_par], "x", 4.7)


class TestFileLists:
    def test_find_line_list_files(self, tmp_path, co_par, line_list_csv, atomic_csv):
        (tmp_path / "settings.csv").write_text("Molecule Name,Temp\nCO,500\n")
        (tmp_path / "notes.txt").write_text("lam\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        nested = _write_par(sub / "nested.par", "OH", CO_LINES)
        cache = tmp_path / "cache"
        cache.mkdir()
        _write_par(cache / "cached.par", "OH", CO_LINES)

        found = find_line_list_files(tmp_path)
        names = [os.path.basename(p) for p in found]
        assert sorted(names) == ["atomic.csv", "co.par", "list.csv", "nested.par"]
        assert nested in found
        top_only = find_line_list_files(tmp_path, recursive=False)
        assert "nested.par" not in [os.path.basename(p) for p in top_only]

    def test_molecule_line_list_files(self, co_par, h2o_par, tmp_path):
        molecules = {
            "CO": SimpleNamespace(filepath=co_par),
            "CO_hot": SimpleNamespace(filepath=co_par),
            "H2O": SimpleNamespace(filepath=h2o_par),
            "lost": SimpleNamespace(filepath=str(tmp_path / "gone.par")),
            "none": SimpleNamespace(filepath=None),
        }
        assert molecule_line_list_files(molecules) == [
            os.path.abspath(co_par), os.path.abspath(h2o_par)]
        assert molecule_line_list_files(None) == []

    def test_molecule_files_relative_to_datafiles(self):
        paths = molecule_line_list_files([SimpleNamespace(filepath="LINELISTS/Atomic_lines.csv")])
        assert len(paths) == 1 and os.path.isabs(paths[0]) and os.path.isfile(paths[0])


# ======================================================================
# Window helpers (no Tk)
# ======================================================================

class TestWindowHelpers:
    def test_display_frame_keeps_wavelength_digits_and_blanks_missing(self):
        lines = pd.DataFrame({"lam": [4.6500312, 12.40702], "note": [np.nan, "x"],
                              "a_stein": [np.nan, 2.5]})
        shown = lsw.display_frame(lines)
        assert shown["lam"].tolist() == ["4.6500312", "12.40702"]
        assert shown["note"].tolist() == ["", "x"]
        assert shown["a_stein"].tolist() == ["", 2.5]
        assert lines["note"].isna().iloc[0]   # input left unchanged

    def test_add_paths(self, tmp_path):
        a, b = str(tmp_path / "a.par"), str(tmp_path / "b.csv")
        paths, added, skipped = lsw.add_paths([a], [a, b, str(tmp_path / "c.txt")])
        assert paths == [a, b] and (added, skipped) == (1, 1)

    def test_error_report(self):
        assert lsw.error_report({}) == ""
        errors = {f"/x/f{i}.par": "File not found." for i in range(5)}
        text = lsw.error_report(errors)
        assert text.startswith("Could not search: f0.par (File not found.)")
        assert text.endswith("and 2 more.")

    def test_titles(self, co_par):
        result = search_line_lists([co_par], 4.6, 4.7)
        assert lsw.results_title(result) == "Lines in 4.6 - 4.7 um (3)"
        assert lsw.export_basename(result) == "lines_4.6-4.7um"

    def test_ranges_from_the_plot(self):
        ax = SimpleNamespace(get_xlim=lambda: (13.0, 12.0))
        plot = SimpleNamespace(current_selection=(4.66, 4.65), ax1=ax)
        assert lsw.selection_range(plot) == (4.65, 4.66)
        assert lsw.initial_range(plot) == (4.65, 4.66)
        plot.current_selection = None
        assert lsw.selection_range(plot) is None
        assert lsw.initial_range(plot) == (12.0, 13.0)
        assert lsw.initial_range(None) is None

    def test_file_location(self):
        from iSLAT.Modules.FileHandling import data_files_path
        assert lsw.file_location(str(data_files_path / "HITRANdata" / "x.par")) == "DATAFILES/HITRANdata"


def test_top_bar_opens_the_window(monkeypatch):
    import importlib
    tb_mod = importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")
    calls = []
    monkeypatch.setattr(tb_mod.LineSearchWindow, "open",
                        classmethod(lambda cls, *args: calls.append(args)))
    bar = tb_mod.TopBar.__new__(tb_mod.TopBar)
    bar.master, bar.islat, bar.theme, bar.main_plot = "root", "islat", {"t": 1}, "plot"
    bar.find_lines_in_range()
    assert calls == [("root", "islat", {"t": 1}, "plot")]


# ======================================================================
# Window (needs a display)
# ======================================================================

@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    for win in list(lsw.LineSearchWindow._instances.values()):
        win._on_close()
    lsw.LineSearchWindow._session_files.clear()
    root.destroy()


class _Handler:
    def __init__(self):
        self.selection_callbacks = {}

    def add_selection_callback(self, name, callback):
        self.selection_callbacks[name] = callback

    def remove_selection_callback(self, name):
        self.selection_callbacks.pop(name, None)

    def select(self, xmin, xmax):
        for callback in list(self.selection_callbacks.values()):
            callback(xmin, xmax)


def _setup(tk_root, files, selection=(4.66, 4.65)):
    plot = SimpleNamespace(current_selection=selection, interaction_handler=_Handler(),
                           ax1=SimpleNamespace(get_xlim=lambda: (4.0, 5.0)))
    molecules = {f"m{i}": SimpleNamespace(filepath=f) for i, f in enumerate(files)}
    islat = SimpleNamespace(molecules_dict=molecules, GUI=None)
    win = lsw.LineSearchWindow.open(tk_root, islat, {}, plot)
    win._win.withdraw()
    return win, islat, plot


def _search(win):
    """Start a search, first collecting leftover Tk objects on the main thread.

    Otherwise a garbage collection triggered on the worker thread can run
    their finalizers there, which Tk refuses outside its main loop.
    """
    gc.collect()
    return win.start_search()


def _wait(tk_root, win, timeout=10.0):
    assert pump_tk(tk_root, timeout, until=lambda: not win.searching)


class TestWindow:
    def test_starts_with_the_selection_and_molecule_files(self, tk_root, co_par, h2o_par):
        win, _, _ = _setup(tk_root, [co_par, h2o_par])
        assert win.get_range() == (4.65, 4.66)
        assert win.files == [os.path.abspath(co_par), os.path.abspath(h2o_par)]
        assert len(win._tree.get_children()) == 2

    def test_falls_back_to_the_plotted_range(self, tk_root, co_par):
        win, _, _ = _setup(tk_root, [co_par], selection=None)
        assert win.get_range() == (4.0, 5.0)

    def test_follows_plot_selections(self, tk_root, co_par):
        win, _, plot = _setup(tk_root, [co_par])
        plot.interaction_handler.select(12.5, 12.0)
        assert win.get_range() == (12.0, 12.5)
        win._follow_var.set(False)
        plot.interaction_handler.select(7.0, 8.0)
        assert win.get_range() == (12.0, 12.5)

    def test_use_plot_selection(self, tk_root, co_par):
        win, _, plot = _setup(tk_root, [co_par])
        win.set_range(1.0, 2.0)
        assert win.use_plot_selection()
        assert win.get_range() == (4.65, 4.66)
        plot.current_selection = None
        assert not win.use_plot_selection()
        assert "No range is selected" in win._status_var.get()
        assert win.use_plot_view() and win.get_range() == (4.0, 5.0)

    def test_editing_the_file_list(self, tk_root, co_par, h2o_par, line_list_csv):
        win, islat, _ = _setup(tk_root, [co_par])
        assert win.add_files([h2o_par, line_list_csv, co_par]) == 2
        win._tree.selection_set(win._iid(0))
        win.remove_selected()
        assert win.files == [os.path.abspath(h2o_par), os.path.abspath(line_list_csv)]
        # Remembered for the session
        win._on_close()
        again = lsw.LineSearchWindow.open(tk_root, islat, {}, None)
        assert again.files == [os.path.abspath(h2o_par), os.path.abspath(line_list_csv)]
        again.clear_files()
        assert again.files == [] and again._count_var.get() == "0 files"
        assert again.add_molecule_files() == 1

    def test_open_raises_the_existing_window(self, tk_root, co_par):
        win, islat, plot = _setup(tk_root, [co_par])
        win.set_range(1.0, 2.0)
        assert lsw.LineSearchWindow.open(tk_root, islat, {}, plot) is win
        assert win.get_range() == (4.65, 4.66)

    def test_invalid_input_does_not_search(self, tk_root, co_par):
        win, _, _ = _setup(tk_root, [co_par])
        win._min_var.set("abc")
        assert not win.start_search()
        assert "must be a number" in win._status_var.get()
        win.set_range(4.6, 4.7)
        win.clear_files()
        assert not win.start_search()
        assert "Add at least one file" in win._status_var.get()

    def test_search_opens_the_results(self, tk_root, co_par, h2o_par, monkeypatch):
        win, _, _ = _setup(tk_root, [co_par, h2o_par])
        shown = []
        monkeypatch.setattr(win, "_open_results", lambda result: shown.append(result))
        assert _search(win)
        assert win._search_btn.cget("text") == "Cancel"
        _wait(tk_root, win)
        assert win._search_btn.cget("text") == "Search"
        assert len(shown) == 1 and shown[0].n_lines == 3
        assert win._status_var.get() == "Found 3 lines in 4.65 - 4.66 um (2 of 2 files)."

    def test_nothing_found_opens_no_table(self, tk_root, co_par, monkeypatch):
        win, _, _ = _setup(tk_root, [co_par])
        shown = []
        monkeypatch.setattr(win, "_open_results", lambda result: shown.append(result))
        win.set_range(30.0, 31.0)
        _search(win)
        _wait(tk_root, win)
        assert shown == []
        assert win._status_var.get().startswith("Found 0 lines")

    def test_errors_are_reported(self, tk_root, co_par, tmp_path, monkeypatch):
        win, _, _ = _setup(tk_root, [co_par])
        monkeypatch.setattr(win, "_open_results", lambda result: None)
        win.add_files([str(tmp_path / "missing.par")])
        _search(win)
        _wait(tk_root, win)
        assert "Could not search: missing.par (File not found.)" in win._status_var.get()

    def test_large_results_ask_first(self, tk_root, co_par, monkeypatch):
        win, _, _ = _setup(tk_root, [co_par])
        monkeypatch.setattr(lsw, "LARGE_RESULT_ROWS", 1)
        asked, shown = [], []
        monkeypatch.setattr(lsw.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
        monkeypatch.setattr(win, "_open_results", lambda result: shown.append(result))
        _search(win)
        _wait(tk_root, win)
        assert len(asked) == 1 and shown == []

    def test_cancel(self, tk_root, co_par, h2o_par, monkeypatch):
        win, _, _ = _setup(tk_root, [co_par, h2o_par])
        shown = []
        monkeypatch.setattr(win, "_open_results", lambda result: shown.append(result))
        _search(win)
        win.cancel_search()
        _wait(tk_root, win)
        assert shown == []
        assert win._status_var.get() == "Search cancelled."

    def test_results_table(self, tk_root, co_par, line_list_csv):
        win, _, _ = _setup(tk_root, [co_par, line_list_csv])
        result = search_line_lists(win.files, *win.get_range())
        view = win._open_results(result)
        try:
            view.withdraw()
            assert view.title() == "Lines in 4.65 - 4.66 um (3)"
            assert view._columns[:2] == [FILE_COLUMN, "species"]
            rows = [view._tree.item(iid, "values") for iid in view._tree.get_children()]
            lam_index = view._columns.index("lam")
            assert [row[lam_index] for row in rows] == ["4.65003", "4.6512", "4.65999"]
            note_index = view._columns.index("note")
            assert [row[note_index] for row in rows] == ["", "blend", ""]
        finally:
            view.destroy()

    def test_close_unregisters(self, tk_root, co_par):
        win, islat, plot = _setup(tk_root, [co_par])
        assert plot.interaction_handler.selection_callbacks
        win._on_close()
        assert plot.interaction_handler.selection_callbacks == {}
        assert id(islat) not in lsw.LineSearchWindow._instances
