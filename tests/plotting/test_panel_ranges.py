# -*- coding: utf-8 -*-
"""Explicit per-panel wavelength ranges in the stacked views (Views menu > Panel Ranges...).

Every FullSpectrumView subclass (the Full Spectrum and Optical Depth views,
and any later one that only supplies its own plot and Views-menu fields)
can show one panel per ``(start, end)`` range, in any order, with gaps.
The ranges stay through plot rebuilds (residual sub-panels, flux panels,
data refreshes, display options) and are used by Save Figure, until Even
Split, a new N Panels or a new Plot Start / Plot Range.  Each change shows
the new panel count and span in the control panel.  The Views-menu entry
opens the shared Panel Ranges dialog, which closes with the view.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from iSLAT.Modules.GUI.ControlField import ActionField, DisplayRangeField, EntryField, ToggleField
from iSLAT.Modules.GUI.ControlSurface import ControlBus, ControlSurface
from iSLAT.Modules.GUI.PanelRanges import format_panel_ranges
from iSLAT.Modules.Plotting.FullSpectrumPlot import FullSpectrumPlot
from iSLAT.Modules.Plotting.FullSpectrumView import PANEL_RANGES_LABEL, FullSpectrumView
from iSLAT.Modules.Plotting.ResidualSpectrumPlot import ResidualSpectrumPlot

from tests.plotting.test_optical_depth_view import _make_view as _make_optical_depth_view
from tests.plotting.test_residual_toggle import _make_view as _make_full_spectrum_view


class _MinimalStackedView(FullSpectrumView):
    """A later stacked view: only its own plot and one Views-menu toggle of its own."""

    VIEW_NAME = "Minimal Stack"
    SUPPORTS_SAMPLE_STACK = False

    def _create_plot(self):
        wave, flux, wave_obs = self._load_spectrum_data()
        return FullSpectrumPlot(
            wave_data=wave, flux_data=flux, molecules=self._islat.molecules_dict,
            wave_data_obs=wave_obs, n_panels=4,
        )

    def _view_menu_fields(self):
        return [ToggleField("minimal_option", "Minimal Option",
                            getter=lambda: False, setter=lambda _value: None, owner=self)]


def _full_spectrum():
    view, pm, *_ = _make_full_spectrum_view()
    return view, pm


def _optical_depth():
    view, pm, *_ = _make_optical_depth_view()
    return view, pm


def _minimal_subclass():
    _view, pm = _full_spectrum()
    return _MinimalStackedView(pm), pm


VIEWS = {"full_spectrum": _full_spectrum, "optical_depth": _optical_depth, "subclass": _minimal_subclass}


def _render(view):
    """The first render, as activate() does it minus the Tk canvas."""
    view._rebuild_plot()
    view._initialised = True
    view._needs_refresh = False
    view.sync_toggle_state(view._pm.toggle_state)
    return view._plot


def _stacked(name):
    view, pm = VIEWS[name]()
    pm.control_bus = None
    pm.full_spectrum_export_settings = {"width": None, "panel_height": None}
    return view, pm


@pytest.fixture(params=sorted(VIEWS))
def stacked_view(request):
    """``(view, pm)`` for each kind of stacked view, not yet rendered."""
    view, pm = _stacked(request.param)
    yield view, pm
    view.destroy()


@pytest.fixture
def rendered(stacked_view):
    _render(stacked_view[0])
    return stacked_view


def _ranges_within(view):
    """Three out-of-order, non-contiguous ranges inside the observed data."""
    lo, hi = view.observed_data_bounds()
    width = hi - lo
    return [(lo + 0.6 * width, lo + 0.7 * width),
            (lo + 0.1 * width, lo + 0.2 * width),
            (lo + 0.3 * width, lo + 0.45 * width)]


def _xlims(view):
    return [ax.get_xlim() for ax in view.subplots.values()]


def _assert_shows(view, ranges):
    assert np.allclose(view.shown_panel_ranges(), ranges)
    assert np.allclose(_xlims(view), ranges)


class _RecordingSurface(ControlSurface):
    """Keeps the fields; records the keys asked to refresh."""

    def __init__(self):
        super().__init__()
        self.refreshed = []

    def _rebuild(self):
        pass

    def refresh(self, key):
        self.refreshed.append(key)
        super().refresh(key)


def _wire_bus(pm):
    bus = ControlBus()
    surfaces = {"control_panel": _RecordingSurface(), "views_menu": _RecordingSurface()}
    for name, surface in surfaces.items():
        bus.register_surface(name, surface)
    pm.control_bus = bus
    return surfaces


# ======================================================================
# The panels follow the ranges
# ======================================================================

class TestShowsTheRanges:
    def test_one_panel_per_range_in_order(self, rendered):
        view, _pm = rendered
        ranges = _ranges_within(view)
        view.set_panel_ranges(ranges)
        assert view.panel_ranges == ranges
        _assert_shows(view, ranges)
        assert view._get_n_panels() == 3
        assert np.allclose(view._get_display_range(), (ranges[1][0], ranges[0][1]))

    def test_set_before_the_first_render(self, stacked_view):
        view, _pm = stacked_view
        _render(view)
        ranges = _ranges_within(view)
        view.destroy()
        view.set_panel_ranges(ranges)       # no plot yet
        _render(view)
        _assert_shows(view, ranges)

    def test_kept_through_rebuilds_and_options(self, rendered):
        view, _pm = rendered
        ranges = _ranges_within(view)
        view.set_panel_ranges(ranges)
        view._regenerate_layout()
        _assert_shows(view, ranges)
        view._rebuild_plot()                # data refresh (new spectrum, parameters)
        _assert_shows(view, ranges)
        view._set_data_density(True)        # only changes an automatic split
        _assert_shows(view, ranges)

    def test_invalid_ranges_change_nothing(self, rendered):
        view, _pm = rendered
        before = view.shown_panel_ranges()
        for bad in ([(6.0, 5.0)], [], [(1.0, float("nan"))]):
            with pytest.raises(ValueError):
                view.set_panel_ranges(bad)
        assert view.panel_ranges is None
        assert view.shown_panel_ranges() == before

    def test_save_figure_uses_the_ranges(self, rendered):
        view, _pm = rendered
        ranges = _ranges_within(view)
        view.set_panel_ranges(ranges)
        plot = view._create_export_plot()
        try:
            assert np.allclose(list(zip(plot._panel_edges, plot._panel_ends)), ranges)
            assert np.allclose([ax.get_xlim() for ax in plot.subplots.values()], ranges)
        finally:
            plot.close()


class TestBackToAnEvenSplit:
    def test_even_split_keeps_the_count_over_the_span(self, rendered):
        view, _pm = rendered
        ranges = _ranges_within(view)
        view.set_panel_ranges(ranges)
        view.set_panel_ranges(None)
        assert view.panel_ranges is None
        start, end = ranges[1][0], ranges[0][1]
        edges = np.linspace(start, end, 4)
        _assert_shows(view, list(zip(edges[:-1], edges[1:])))

    @pytest.mark.parametrize("n_panels", [2, 3])      # 3: the count the ranges already have
    def test_new_panel_count(self, rendered, n_panels):
        view, _pm = rendered
        view.set_panel_ranges(_ranges_within(view))
        view._set_n_panels(n_panels)
        assert view.panel_ranges is None and view._plot.panel_ranges is None
        assert len(view.shown_panel_ranges()) == n_panels

    def test_new_plot_start_and_range(self, rendered):
        view, _pm = rendered
        ranges = _ranges_within(view)
        view.set_panel_ranges(ranges)
        start, end = ranges[1][0], ranges[2][1]
        view._set_display_range(start, end)
        assert view.panel_ranges is None
        shown = view.shown_panel_ranges()
        assert len(shown) == 3
        assert shown[0][0] == pytest.approx(start) and shown[-1][1] == pytest.approx(end)


class TestPlotSwitches:
    def test_full_spectrum_residuals_on_and_off(self):
        view, pm = _stacked("full_spectrum")
        try:
            _render(view)
            ranges = _ranges_within(view)
            view.set_panel_ranges(ranges)
            pm.toggle_state["show_residuals"] = True
            view.toggle_residuals(True)                 # replaces the plot
            assert isinstance(view._plot, ResidualSpectrumPlot)
            _assert_shows(view, ranges)
            pm.toggle_state["show_residuals"] = False
            view._rebuild_plot()                        # plot-kind change on a refresh
            assert not isinstance(view._plot, ResidualSpectrumPlot)
            _assert_shows(view, ranges)
            view._set_data_density(True)                # the new plot keeps them too
            _assert_shows(view, ranges)
        finally:
            view.destroy()

    def test_optical_depth_flux_panels(self):
        view, _pm = _stacked("optical_depth")
        try:
            _render(view)
            ranges = _ranges_within(view)
            view.set_panel_ranges(ranges)
            view._set_show_flux(True)
            _assert_shows(view, ranges)
            assert np.allclose([ax.get_xlim() for ax in view._plot.get_flux_axes().values()], ranges)
        finally:
            view.destroy()


# ======================================================================
# Controls
# ======================================================================

class TestControls:
    def test_panel_ranges_entry_comes_first_in_the_views_menu(self, rendered):
        view, pm = rendered
        surfaces = _wire_bus(pm)
        view._register_control_fields()
        first, *own = surfaces["views_menu"].fields()
        assert isinstance(first, ActionField)
        assert (first.key, first.label) == ("panel_ranges", PANEL_RANGES_LABEL)
        assert first.command == view.open_panel_ranges_dialog
        assert [f.key for f in own] == [f.key for f in view._view_menu_fields()]
        panel = surfaces["control_panel"]
        assert isinstance(panel.get_field("n_panels"), EntryField)
        assert isinstance(panel.get_field(view.DISPLAY_RANGE_KEY), DisplayRangeField)

    def test_changes_show_in_the_control_panel(self, rendered):
        view, pm = rendered
        surfaces = _wire_bus(pm)
        view._register_control_fields()
        panel = surfaces["control_panel"]
        view.set_panel_ranges(_ranges_within(view))
        assert panel.refreshed == ["n_panels", view.DISPLAY_RANGE_KEY]
        assert panel.get_field("n_panels").getter() == 3


# ======================================================================
# The dialog, opened from the Views menu
# ======================================================================

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


def _menu_index(menu, label):
    return next(i for i in range(menu.index("end") + 1)
                if menu.type(i) != "separator" and menu.entrycget(i, "label") == label)


@pytest.fixture
def menu_view(tk_root, stacked_view):
    """A rendered view shown in a frame of *tk_root*, its fields on a real Views menu."""
    import tkinter as tk
    from iSLAT.Modules.GUI.Widgets.TopBar import ViewsMenuSurface
    view, pm = stacked_view
    _render(view)
    view._parent_frame = tk.Frame(tk_root)
    menu = tk.Menu(tk_root, tearoff=0)
    bus = ControlBus()
    bus.register_surface("views_menu", ViewsMenuSurface(menu))
    pm.control_bus = bus
    view._register_control_fields()
    yield view, menu
    view.destroy()
    menu.destroy()
    view._parent_frame.destroy()


class TestDialog:
    def test_menu_entry_opens_the_dialog_with_the_panels_shown(self, menu_view, tk_root):
        view, menu = menu_view
        menu.invoke(_menu_index(menu, PANEL_RANGES_LABEL))
        dialog = view._ranges_dialog
        assert dialog is not None and dialog.winfo_exists()
        assert dialog.master is tk_root                    # the window holding the view
        starts, ends = zip(*view.shown_panel_ranges())
        assert dialog.text.get("1.0", "end").strip() == format_panel_ranges(starts, ends)
        menu.invoke(_menu_index(menu, PANEL_RANGES_LABEL))
        assert view._ranges_dialog is dialog                # raised, not opened twice

    def test_apply_errors_and_even_split(self, menu_view):
        view, menu = menu_view
        menu.invoke(_menu_index(menu, PANEL_RANGES_LABEL))
        dialog = view._ranges_dialog
        ranges = _ranges_within(view)

        dialog.text.delete("1.0", "end")
        dialog.text.insert("1.0", "not a range")
        _button(dialog, "Apply").invoke()
        assert view.panel_ranges is None
        assert dialog.error_var.get().startswith("Line 1:")

        dialog.text.delete("1.0", "end")
        dialog.text.insert("1.0", "\n".join(f"{lo} {hi}" for lo, hi in ranges))
        _button(dialog, "Apply").invoke()
        assert dialog.error_var.get() == ""
        assert np.allclose(view.panel_ranges, ranges)
        _assert_shows(view, ranges)

        _button(dialog, "Even Split").invoke()
        assert view.panel_ranges is None
        starts, ends = zip(*view.shown_panel_ranges())
        assert dialog.text.get("1.0", "end").strip() == format_panel_ranges(starts, ends)

    def test_data_keywords_use_the_observed_range(self, menu_view):
        view, menu = menu_view
        menu.invoke(_menu_index(menu, PANEL_RANGES_LABEL))
        dialog = view._ranges_dialog
        lo, hi = view.observed_data_bounds()
        middle = 0.5 * (lo + hi)
        dialog.text.delete("1.0", "end")
        dialog.text.insert("1.0", f"{middle} DATAEND\nDATASTART {middle}")
        _button(dialog, "Apply").invoke()
        assert np.allclose(view.panel_ranges, [(middle, hi), (lo, middle)])

    def test_closes_with_the_view(self, menu_view):
        view, menu = menu_view
        menu.invoke(_menu_index(menu, PANEL_RANGES_LABEL))
        dialog = view._ranges_dialog
        view.deactivate()
        assert not dialog.winfo_exists()
        assert view._ranges_dialog is None
