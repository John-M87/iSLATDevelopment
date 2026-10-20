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
