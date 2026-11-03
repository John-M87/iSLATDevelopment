# -*- coding: utf-8 -*-
"""Center View dialog helpers: data-point windows, exact ranges, and routing."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure


def _handler(wave=None):
    from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
    h = InteractionHandler.__new__(InteractionHandler)
    if wave is None:
        wave = np.linspace(10.0, 20.0, 101)  # 0.1 um spacing
    cp = MagicMock()
    cp._format_value.side_effect = lambda v, _k: f"{v:.6g}"
    h.islat = SimpleNamespace(wave_data=wave, display_range=None, GUI=SimpleNamespace(control_panel=cp))
    h.plot_manager = SimpleNamespace(active_view=SimpleNamespace())
    h.ax1 = Figure().add_subplot(111)
    h.ax1.set_xlim(14.0, 15.0)
    return h, cp


class TestDataPointWindow:
    def test_window_centred_on_nearest_point(self):
        h, _ = _handler()
        start, end = h._range_from_center_points(14.02, 5)
        assert start == pytest.approx(13.8)
        assert end == pytest.approx(14.2)
        assert h._count_points_in_range(start, end) == 5

    def test_even_count_spans_n_points(self):
        h, _ = _handler()
        start, end = h._range_from_center_points(14.0, 4)
        assert h._count_points_in_range(start, end) == 4

    def test_clamped_at_spectrum_edges(self):
        h, _ = _handler()
        start, end = h._range_from_center_points(10.0, 5)
        assert start == pytest.approx(10.0)
        assert end == pytest.approx(10.4)
        start, end = h._range_from_center_points(20.0, 5)
        assert start == pytest.approx(19.6)
        assert end == pytest.approx(20.0)

    def test_unsorted_and_nan_wavelengths_are_handled(self):
        wave = np.array([15.0, np.nan, 13.0, 14.0, 12.0, 16.0])
        h, _ = _handler(wave)
        assert h._range_from_center_points(14.0, 3) == (13.0, 15.0)

    def test_degenerate_cases_return_none(self):
        h, _ = _handler()
        assert h._range_from_center_points(14.0, 1) is None
        h, _ = _handler(np.array([]))
        assert h._range_from_center_points(14.0, 3) is None
        assert h._count_points_in_range(0.0, 1.0) == 0

    def test_count_points_in_current_view(self):
        h, _ = _handler()
        assert h._count_points_in_range(14.0, 15.0) == 11


class TestApplyPlotRange:
    def test_routes_exact_range_through_control_panel(self):
        h, cp = _handler()
        h._apply_plot_range(14.0, 15.0)
        cp._format_value.assert_any_call(14.0, "display_range_start")
        cp._format_value.assert_any_call(1.0, "display_range_range")
        cp._update_display_range.assert_called_once()

    def test_reversed_bounds_are_swapped(self):
        h, cp = _handler()
        h._apply_plot_range(15.0, 14.0)
        cp._format_value.assert_any_call(14.0, "display_range_start")
        cp._format_value.assert_any_call(1.0, "display_range_range")

    def test_center_view_wrapper_still_works(self):
        h, cp = _handler()
        h._apply_center_view(14.5, 1.0)
        cp._format_value.assert_any_call(14.0, "display_range_start")
        cp._format_value.assert_any_call(1.0, "display_range_range")

    def test_falls_back_to_islat_display_range_without_control_panel(self):
        h, _ = _handler()
        plot = MagicMock()
        h.islat.GUI = SimpleNamespace(plot=plot)
        h._apply_plot_range(14.0, 15.0)
        assert h.islat.display_range == (14.0, 15.0)
        plot.match_display_range.assert_called_once_with(match_y=True)

    def test_public_and_private_dialog_names_match(self):
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        assert InteractionHandler._open_center_view_dialog is InteractionHandler.open_center_view_dialog


# ---------------------------------------------------------------------------
# "Mark center wavelength" toggle
# ---------------------------------------------------------------------------

class _MarkerStore:
    """Plot-manager stand-in with the real Shift+M marker API."""

    def __init__(self):
        from iSLAT.Modules.Plotting.PlotView import WavelengthMarkerStoreMixin

        class _Store(WavelengthMarkerStoreMixin):
            def __init__(self):
                self.toggle_state = {}
                self.refreshes = 0

            def refresh_wavelength_markers(self):
                self.refreshes += 1

        self.store = _Store()


def _marking_handler():
    h, cp = _handler()
    store = _MarkerStore().store
    store.active_view = SimpleNamespace()
    h.plot_manager = store
    return h, cp, store


class TestMarkCenterWavelength:
    def test_adds_one_mark_and_skips_duplicates(self):
        h, _, store = _marking_handler()
        assert h._mark_center_wavelength(14.5) is True
        assert h._mark_center_wavelength(14.5) is False
        assert [(m["start"], m["end"]) for m in store.wavelength_markers] == [(14.5, 14.5)]
        assert store.refreshes == 1

    def test_ranges_at_same_center_are_not_duplicates(self):
        h, _, store = _marking_handler()
        store.add_wavelength_marker(14.0, 15.0)
        assert h._mark_center_wavelength(14.0) is True

    def test_without_marker_store_is_a_no_op(self):
        h, _ = _handler()
        assert h._mark_center_wavelength(14.5) is False


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


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _open_dialog(tk_root, h):
    before = set(tk_root.winfo_children())
    h._tk_root = tk_root
    h.open_center_view_dialog()
    (win,) = [w for w in tk_root.winfo_children() if w not in before]
    return win


def _widget(win, cls, text):
    return next(w for w in _walk(win) if w.winfo_class() == cls and w.cget("text") == text)


class TestCenterDialogMarkToggle:
    def test_off_by_default_applies_range_only(self, tk_root):
        h, cp, store = _marking_handler()
        win = _open_dialog(tk_root, h)
        assert "selected" not in _widget(win, "TCheckbutton", "Mark center wavelength").state()
        _widget(win, "TButton", "Apply").invoke()
        cp._update_display_range.assert_called_once()
        assert store.wavelength_markers == []

    def test_marks_entered_center_and_is_remembered(self, tk_root):
        h, cp, store = _marking_handler()
        win = _open_dialog(tk_root, h)
        _widget(win, "TCheckbutton", "Mark center wavelength").invoke()
        center_entry = next(w for w in _walk(win) if w.winfo_class() == "TEntry")
        center_entry.delete(0, "end")
        center_entry.insert(0, "14.25")
        _widget(win, "TButton", "Apply").invoke()
        cp._format_value.assert_any_call(14.25 - 0.5, "display_range_start")
        assert [(m["start"], m["end"]) for m in store.wavelength_markers] == [(14.25, 14.25)]

        win = _open_dialog(tk_root, h)
        assert "selected" in _widget(win, "TCheckbutton", "Mark center wavelength").state()
        _widget(win, "TButton", "Cancel").invoke()

    def test_exact_range_marks_midpoint(self, tk_root):
        h, _, store = _marking_handler()
        win = _open_dialog(tk_root, h)
        _widget(win, "TCheckbutton", "Mark center wavelength").invoke()
        _widget(win, "TRadiobutton", "Exact range (\u03bcm):").invoke()
        start, end = [w for w in _walk(win) if w.winfo_class() == "TEntry"][-2:]
        start.delete(0, "end"); start.insert(0, "12")
        end.delete(0, "end"); end.insert(0, "13")
        _widget(win, "TButton", "Apply").invoke()
        assert [(m["start"], m["end"]) for m in store.wavelength_markers] == [(12.5, 12.5)]

    def test_invalid_input_adds_no_mark(self, tk_root):
        h, cp, store = _marking_handler()
        win = _open_dialog(tk_root, h)
        _widget(win, "TCheckbutton", "Mark center wavelength").invoke()
        center_entry = next(w for w in _walk(win) if w.winfo_class() == "TEntry")
        center_entry.delete(0, "end")
        center_entry.insert(0, "abc")
        _widget(win, "TButton", "Apply").invoke()
        assert win.winfo_exists()
        assert store.wavelength_markers == []
        cp._update_display_range.assert_not_called()
        win.destroy()
