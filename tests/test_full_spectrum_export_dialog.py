# -*- coding: utf-8 -*-
"""Full Spectrum Export Settings dialog: the font sizes and the multi-page PDF section.

Font sizes (points) are set for the tick labels, axis labels, legend and
annotations; blank fields keep the plot's own sizes.  Page ranges are typed
one ``start end`` per line, as in the Display Full Spectrum window's Ranges
dialog (``DATASTART`` / ``DATAEND`` stand for the first / last observed
wavelength), with one number of rows for every page or one per page.  The
entries are remembered for the session; invalid ones are reported in the
dialog without storing or exporting.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from iSLAT.Modules.Plotting.FullSpectrumView import get_export_fonts, get_export_settings, get_page_settings

WAVE = np.linspace(5.0, 11.0, 61)


@pytest.fixture(scope="module")
def tk_root():
    # One root for the module: creating many Tk() instances is flaky on Windows.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _button(widget, text):
    return next(w for w in _walk(widget) if w.winfo_class() == "TButton" and w.cget("text") == text)


def _widgets(dialog, widget_class):
    return [w for w in _walk(dialog) if w.winfo_class() == widget_class]


def _status(dialog):
    return next(w for w in _walk(dialog) if w.winfo_class() == "Label" and w.cget("fg") == "red").cget("text")


def _top_bar_module():
    return importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")


class _Molecules:
    def apply_stellar_rv(self, wave):
        return np.asarray(wave)


@pytest.fixture
def exports(monkeypatch):
    """The multi-page exports the dialog started: ``(islat, pages, rows)`` per call."""
    calls = []

    def recording(islat, pages, rows):
        calls.append((islat, pages, rows))

    monkeypatch.setattr(_top_bar_module(), "output_full_spectrum_pages", recording)
    return calls


@pytest.fixture
def open_dialog(tk_root, monkeypatch):
    """Open the dialog with a plain frame standing in for the top bar; returns ``open() -> (dialog, pm)``."""
    import tkinter as tk
    monkeypatch.setattr(tk.Toplevel, "grab_set", lambda self: None)
    host = tk.Frame(tk_root)
    host.main_plot = SimpleNamespace(
        full_spectrum_export_settings={"width": None, "panel_height": None},
        _full_spectrum_view=SimpleNamespace(_plot=SimpleNamespace(n_panels=7)),
    )
    host.islat = SimpleNamespace(
        wave_data_original=WAVE, molecules_dict=_Molecules(),
        GUI=SimpleNamespace(plot=host.main_plot),
    )

    def _open():
        _top_bar_module().TopBar.open_full_spectrum_export_settings(host)
        return host.winfo_children()[-1], host.main_plot

    yield _open
    host.destroy()


def _enter(dialog, pages, rows):
    text = _widgets(dialog, "Text")[0]
    text.delete("1.0", "end")
    text.insert("1.0", pages)
    rows_entry = _widgets(dialog, "Entry")[-1]
    rows_entry.delete(0, "end")
    rows_entry.insert(0, rows)


class TestMultiPageSection:
    def test_defaults_to_the_data_range_and_the_view_panel_count(self, open_dialog):
        dialog, _pm = open_dialog()
        assert _widgets(dialog, "Text")[0].get("1.0", "end").strip() == "DATASTART DATAEND"
        assert _widgets(dialog, "Entry")[-1].get() == "7"

    def test_export_parses_pages_and_rows(self, open_dialog, exports):
        dialog, pm = open_dialog()
        _enter(dialog, "DATASTART 7\n7 - 9\n9 to DATAEND", "3, 4, 5")
        _button(dialog, "Export Multi-page PDF...").invoke()
        assert not dialog.winfo_exists()
        ((_islat, pages, rows),) = exports
        assert pages == [(5.0, 7.0), (7.0, 9.0), (9.0, 11.0)]
        assert rows == [3, 4, 5]
        assert get_page_settings(pm) == {"pages": "DATASTART 7\n7 - 9\n9 to DATAEND", "rows": "3, 4, 5"}

    @pytest.mark.parametrize("pages, rows", [
        ("5 7\n7 9", "2, 3, 4"),     # a count per page, but not one for every page
        ("7 5", "2"),                # start after end
        ("5 7 9", "2"),              # not a start end pair
        ("5 7", "0"),                # no rows
    ])
    def test_invalid_entries_are_reported_and_nothing_exported(self, open_dialog, exports, pages, rows):
        dialog, _pm = open_dialog()
        _enter(dialog, pages, rows)
        _button(dialog, "Export Multi-page PDF...").invoke()
        assert dialog.winfo_exists()
        assert _status(dialog).startswith("Multi-page PDF:")
        assert exports == []

    def test_entries_are_remembered_for_the_next_opening(self, open_dialog):
        dialog, _pm = open_dialog()
        _enter(dialog, "6 8\n8 10", "4")
        _button(dialog, "Save").invoke()
        dialog, _pm = open_dialog()
        assert _widgets(dialog, "Text")[0].get("1.0", "end").strip() == "6 8\n8 10"
        assert _widgets(dialog, "Entry")[-1].get() == "4"

    def test_reset_restores_the_defaults(self, open_dialog):
        dialog, _pm = open_dialog()
        _enter(dialog, "6 8", "4")
        _button(dialog, "Reset to Defaults").invoke()
        assert _widgets(dialog, "Text")[0].get("1.0", "end").strip() == "DATASTART DATAEND"
        assert _widgets(dialog, "Entry")[-1].get() == "7"

    def test_return_saves_from_the_fields_only(self, open_dialog):
        # Checked on the bindings: generated key events go to whichever widget
        # has the system focus, which a test cannot rely on.  The page ranges
        # keep Tk's own Return (a new line).
        dialog, _pm = open_dialog()
        assert dialog.bind("<Return>") == ""
        assert _widgets(dialog, "Text")[0].bind("<Return>") == ""
        entries = _widgets(dialog, "Entry")
        # Width, panel height, the four font sizes and the rows per page.
        assert len(entries) == 7 and all(entry.bind("<Return>") for entry in entries)


def _font_entries(dialog):
    """The font size fields, in order: tick labels, axis labels, legend, annotations."""
    frame = next(w for w in _walk(dialog)
                 if w.winfo_class() == "TLabelframe" and w.cget("text") == "Font sizes (pt)")
    return [w for w in _walk(frame) if w.winfo_class() == "Entry"]


def _type(entry, value):
    entry.delete(0, "end")
    entry.insert(0, value)


class TestFontSizes:
    def test_blank_by_default(self, open_dialog):
        dialog, _pm = open_dialog()
        entries = _font_entries(dialog)
        assert len(entries) == 4 and all(entry.get() == "" for entry in entries)

    def test_save_stores_and_reopening_shows_them(self, open_dialog):
        dialog, pm = open_dialog()
        for entry, value in zip(_font_entries(dialog), ("9", "11.5", "", "6")):
            _type(entry, value)
        _button(dialog, "Save").invoke()
        assert not dialog.winfo_exists()
        assert get_export_fonts(pm) == {"ticks": 9.0, "labels": 11.5, "legend": None, "text": 6.0}
        dialog, _pm = open_dialog()
        assert [entry.get() for entry in _font_entries(dialog)] == ["9", "11.5", "", "6"]

    @pytest.mark.parametrize("value", ["big", "0", "-3", "100"])
    def test_invalid_size_is_reported_and_nothing_stored(self, open_dialog, value):
        dialog, pm = open_dialog()
        _type(_widgets(dialog, "Entry")[0], "30")          # a valid width ...
        _type(_font_entries(dialog)[1], value)              # ... with an invalid axis label size
        _button(dialog, "Save").invoke()
        assert dialog.winfo_exists()
        assert _status(dialog).startswith("Axis label font size")
        assert get_export_settings(pm)["width"] is None
        assert get_export_fonts(pm) == {"ticks": None, "labels": None, "legend": None, "text": None}

    def test_reset_clears_the_sizes(self, open_dialog):
        dialog, _pm = open_dialog()
        for entry in _font_entries(dialog):
            _type(entry, "12")
        _button(dialog, "Reset to Defaults").invoke()
        assert all(entry.get() == "" for entry in _font_entries(dialog))
