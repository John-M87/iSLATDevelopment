# -*- coding: utf-8 -*-
"""Display Full Spectrum window: the same options as the in-GUI Full Spectrum view."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

WAVE = np.linspace(5.0, 11.0, 600)
FLUX = np.where(WAVE < 8.0, 1.0, 4.0)
MODEL = 0.2 * np.exp(-0.5 * ((WAVE - 8.0) / 2.0) ** 2)


class _Molecules:
    """Minimal MoleculeDict stand-in: no visible molecules, a summed model."""

    _global_wavelength_range = None

    def apply_stellar_rv(self, wave):
        return np.asarray(wave)

    def get_summed_flux(self, wave, **_kw):
        return WAVE, MODEL

    def get_summed_flux_resampled(self, wave, **_kw):
        return np.asarray(wave), np.interp(wave, WAVE, MODEL)

    def get_visible_molecules(self, return_objects=False):
        return []


def _gui_state(**overrides):
    ts = {"atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
          "show_residuals": False, "data_density": False, "current_selection": None,
          "y_autoscale": True, "y_autoscale_source": "data", "y_limits": None, "observed": True,
          "wavelength_markers": []}
    ts.update(overrides)
    return ts


def _islat(tmp_path, toggle_state=None, live_layout=True):
    lines = tmp_path / "lines.csv"
    lines.write_text("lam,species,line\n6.5,H2O,a\n9.5,CO2,b\n")
    fsv = None
    if live_layout:
        fsv = SimpleNamespace(_plot=SimpleNamespace(n_panels=4, _xlim_start=6.0, _xlim_end=10.0,
                                                    ymax_factor=0.3))
    gui_plot = SimpleNamespace(toggle_state=toggle_state or _gui_state(), theme={},
                               _full_spectrum_view=fsv)
    return SimpleNamespace(
        wave_data_original=WAVE, flux_data=FLUX, err_data=None, molecules_dict=_Molecules(),
        loaded_spectrum_file=str(tmp_path / "spec.csv"), input_line_list=str(lines),
        GUI=SimpleNamespace(plot=gui_plot, data_field=MagicMock()),
    )


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


@pytest.fixture
def make_window(tk_root, tmp_path):
    from iSLAT.Modules.GUI.FullSpectrumWindow import FullSpectrumWindow
    windows = []

    def _make(islat=None, **kwargs):
        islat = islat or _islat(tmp_path)
        win = FullSpectrumWindow(tk_root, islat, **kwargs)
        win.withdraw()
        windows.append(win)
        return win, islat

    yield _make
    for win in windows:
        if win.winfo_exists():
            win.on_closing()


def _tagged(win, tag):
    return [a for ax in win.view.fig.axes for a in ax.get_children() if hasattr(a, tag)]


def _spectrum_axes(win):
    return list(win.view.subplots.values())


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _button(widget, text):
    return next(w for w in _walk(widget) if w.winfo_class() == "TButton" and w.cget("text") == text)


def _ranges(win):
    plot = win.view._plot
    return [(float(a), float(b)) for a, b in zip(plot._panel_edges, plot._panel_ends)]


class TestInitialState:
    def test_starts_from_gui_layout_and_toggles(self, make_window):
        win, islat = make_window()
        plot = win.view._plot
        assert plot.n_panels == 4
        assert (plot._xlim_start, plot._xlim_end) == (6.0, 10.0)
        assert plot.ymax_factor == 0.3
        assert win.n_panels_var.get() == "4"
        assert win.start_var.get() == "6.0000" and win.end_var.get() == "10.0000"
        assert win.option_vars["summed"].get() is True
        assert win.option_vars["show_residuals"].get() is False

    def test_kwargs_override_layout(self, make_window):
        win, _ = make_window(n_panels=2, xlim_range=(7.0, 9.0))
        assert win.view._plot.n_panels == 2
        assert (win.view._plot._xlim_start, win.view._plot._xlim_end) == (7.0, 9.0)

    def test_without_live_view_uses_full_data_range(self, make_window, tmp_path):
        win, _ = make_window(_islat(tmp_path, live_layout=False))
        assert win.view._plot._xlim_start == pytest.approx(5.0)
        assert win.view._plot._xlim_end == pytest.approx(11.0)

    def test_window_state_is_independent_of_gui(self, make_window):
        win, islat = make_window()
        win.set_atomic_lines(True)
        win.set_residuals(True)
        assert islat.GUI.plot.toggle_state["atomic_lines"] is False
        assert islat.GUI.plot.toggle_state["show_residuals"] is False
        assert win.toggle_state is not islat.GUI.plot.toggle_state

    def test_inherits_gui_overlays(self, make_window, tmp_path):
        islat = _islat(tmp_path, _gui_state(atomic_lines=True, show_residuals=True, data_density=True))
        win, _ = make_window(islat)
        from iSLAT.Modules.Plotting.ResidualSpectrumPlot import ResidualSpectrumPlot
        from iSLAT.Modules.Plotting.SpectralPanel import XScaling
        assert isinstance(win.view._plot, ResidualSpectrumPlot)
        assert win.view._plot.x_scaling is XScaling.DATA_DENSITY
        assert _tagged(win, "_islat_atomic_line")


class TestLayout:
    def test_apply_panels_range_and_margin(self, make_window):
        win, _ = make_window()
        win.n_panels_var.set("6")
        win.start_var.set("5.5")
        win.end_var.set("10.5")
        win.ymax_var.set("0.5")
        win.apply_settings()
        plot = win.view._plot
        assert plot.n_panels == 6 and len(_spectrum_axes(win)) == 6
        assert (plot._panel_edges[0], plot._panel_ends[-1]) == pytest.approx((5.5, 10.5))
        assert plot.ymax_factor == 0.5
        assert win.status_var.get() == ""

    @pytest.mark.parametrize("field, value", [
        ("n_panels_var", "0"), ("n_panels_var", "x"), ("start_var", "12"), ("ymax_var", "-1"),
    ])
    def test_invalid_layout_is_rejected(self, make_window, field, value):
        win, _ = make_window()
        getattr(win, field).set(value)
        win.apply_settings()
        assert win.status_var.get()
        assert win.view._plot.n_panels == 4


class TestDisplayOptions:
    def test_residuals_toggle_swaps_plot_and_toolbar(self, make_window):
        from iSLAT.Modules.Plotting.ResidualSpectrumPlot import ResidualSpectrumPlot
        win, _ = make_window()
        edges = np.array(win.view._plot._panel_edges)
        win.set_residuals(True)
        assert isinstance(win.view._plot, ResidualSpectrumPlot)
        assert np.allclose(win.view._plot._panel_edges, edges)
        assert win.nav_toolbar.canvas is win.view.get_canvas()
        win.set_residuals(False)
        assert not isinstance(win.view._plot, ResidualSpectrumPlot)
        assert win.nav_toolbar.canvas is win.view.get_canvas()

    def test_data_density(self, make_window):
        from iSLAT.Modules.Plotting.SpectralPanel import XScaling
        win, _ = make_window()
        win.set_data_density(True)
        assert win.view._plot.x_scaling is XScaling.DATA_DENSITY
        assert win.toggle_state["data_density"] is True
        win.set_data_density(False)
        assert win.view._plot.x_scaling is XScaling.WAVELENGTH

    def test_atomic_and_saved_lines(self, make_window):
        win, _ = make_window()
        win.set_atomic_lines(True)
        assert _tagged(win, "_islat_atomic_line")
        win.set_atomic_lines(False)
        assert not _tagged(win, "_islat_atomic_line")
        win.set_saved_lines(True)
        assert _tagged(win, "_islat_saved_line")
        win.set_saved_lines(False)
        assert not _tagged(win, "_islat_saved_line")

    def test_summed_and_observed_visibility(self, make_window):
        win, _ = make_window()
        assert _tagged(win, "_islat_summed") and _tagged(win, "_islat_observed")
        win.set_summed(False)
        win.set_observed(False)
        assert not any(a.get_visible() for a in _tagged(win, "_islat_summed"))
        assert not any(a.get_visible() for a in _tagged(win, "_islat_observed"))
        win.set_summed(True)
        win.set_observed(True)
        assert all(a.get_visible() for a in _tagged(win, "_islat_observed"))

    def test_overlays_survive_layout_change(self, make_window):
        win, _ = make_window()
        win.set_atomic_lines(True)
        win.set_observed(False)
        win.n_panels_var.set("3")
        win.apply_settings()
        assert _tagged(win, "_islat_atomic_line")
        assert not any(a.get_visible() for a in _tagged(win, "_islat_observed"))


class TestYAxis:
    def test_manual_limits_then_autoscale(self, make_window):
        win, _ = make_window()
        win.set_y_limits(-1, 9)
        assert all(ax.get_ylim() == (-1.0, 9.0) for ax in _spectrum_axes(win))
        assert win.y_autoscale_var.get() is False
        win.set_y_autoscale(True)
        assert all(ax.get_ylim() != (-1.0, 9.0) for ax in _spectrum_axes(win))

    def test_disabling_autoscale_freezes_limits(self, make_window):
        win, _ = make_window()
        before = [ax.get_ylim() for ax in _spectrum_axes(win)]
        win.set_y_autoscale(False)
        assert win.toggle_state["y_limits"] == before[0]

    def test_fit_to_model(self, make_window):
        win, _ = make_window()
        win.set_y_autoscale_source("model")
        assert all(ax.get_ylim()[1] < 1.0 for ax in _spectrum_axes(win))  # model peaks at 0.2 Jy

    def test_invalid_limits(self, make_window):
        win, _ = make_window()
        with pytest.raises(ValueError):
            win.set_y_limits(5, 1)
        assert win.toggle_state["y_autoscale"] is True

    def test_y_limits_dialog(self, make_window):
        win, _ = make_window()
        win.open_y_limits_dialog()
        dlg = win._y_limits_dialog
        entries = [w for w in _walk(dlg) if w.winfo_class() == "TEntry"]
        entries[0].delete(0, "end"); entries[0].insert(0, "abc")
        _button(dlg, "Apply").invoke()
        assert dlg.winfo_exists()  # stays open on bad input
        entries[0].delete(0, "end"); entries[0].insert(0, "0")
        entries[1].delete(0, "end"); entries[1].insert(0, "7")
        _button(dlg, "Apply").invoke()
        assert not dlg.winfo_exists()
        assert all(ax.get_ylim() == (0.0, 7.0) for ax in _spectrum_axes(win))

    # The window starts with 4 panels over 6-10 um; the data is 1 Jy below 8 um and 4 Jy above.
    def test_sync_to_data_min_max(self, make_window):
        win, _ = make_window()
        win.set_y_limits(-1, 9)
        win.set_y_data_sync("panel")
        assert win.toggle_state["y_autoscale"] is True and win.y_autoscale_var.get() is True
        assert win.y_data_sync_var.get() == "panel"
        lims = [ax.get_ylim() for ax in _spectrum_axes(win)]
        assert lims[0] == pytest.approx((0.95, 1.05)) and lims[-1] == pytest.approx((3.8, 4.2))
        win.set_y_data_sync("view")
        assert all(ax.get_ylim() == (1.0, 4.0) for ax in _spectrum_axes(win))
        win.set_y_autoscale_source("model")                 # exclusive with syncing
        assert win.toggle_state["y_data_sync"] is None
        assert win.y_data_sync_var.get() == "off"

    def test_sync_inherited_from_gui_and_kept_on_relayout(self, make_window, tmp_path):
        win, _ = make_window(_islat(tmp_path, _gui_state(y_data_sync="panel")))
        assert win.y_data_sync_var.get() == "panel"
        assert _spectrum_axes(win)[0].get_ylim() == pytest.approx((0.95, 1.05))
        win.n_panels_var.set("2")
        win.apply_settings()
        lims = [ax.get_ylim() for ax in _spectrum_axes(win)]
        assert lims == [pytest.approx((0.95, 1.05)), pytest.approx((3.8, 4.2))]

    def test_sync_menu(self, make_window):
        win, _ = make_window()
        index = next(i for i in range(win.y_menu.index("end") + 1)
                     if win.y_menu.type(i) == "cascade")
        assert win.y_menu.entrycget(index, "label") == "Sync Y to Data Min/Max"
        sub = win.nametowidget(win.y_menu.entrycget(index, "menu"))
        labels = [sub.entrycget(i, "label") for i in range(sub.index("end") + 1)]
        assert labels == ["Off", "Global Data", "Active View", "Each Panel"]
        sub.invoke(3)
        assert win.toggle_state["y_data_sync"] == "panel"
        sub.invoke(1)
        assert win.toggle_state["y_data_sync"] == "global"
        assert all(ax.get_ylim() == (1.0, 4.0) for ax in _spectrum_axes(win))
        sub.invoke(0)
        assert win.toggle_state["y_data_sync"] is None

    def test_y_limits_dialog_data_min_max(self, make_window):
        win, _ = make_window()
        win.open_y_limits_dialog()
        dlg = win._y_limits_dialog
        combo = next(w for w in _walk(dlg) if w.winfo_class() == "TCombobox")
        assert list(dlg.tk.splitlist(combo.cget("values"))) == ["Global data", "Active view", "Each panel"]
        entries = [w for w in _walk(dlg) if w.winfo_class() == "TEntry"]
        _button(dlg, "Fill Values").invoke()
        assert (entries[0].get(), entries[1].get()) == ("1", "4")
        assert dlg.winfo_exists()                           # filling only fills the entries
        combo.set("Each panel")
        combo.event_generate("<<ComboboxSelected>>")
        assert _button(dlg, "Fill Values").instate(["disabled"])
        _button(dlg, "Keep Synced").invoke()
        assert not dlg.winfo_exists()
        assert win.toggle_state["y_data_sync"] == "panel"
        assert _spectrum_axes(win)[0].get_ylim() == pytest.approx((0.95, 1.05))


class TestMarkersSaveRefresh:
    def test_marks_start_from_gui_then_are_window_only(self, make_window, tmp_path):
        from iSLAT.Modules.Plotting.PlotView import WAVELENGTH_MARKER_TAG
        marker = {"start": 7.0, "end": 7.0, "label": "", "color": "red"}
        islat = _islat(tmp_path, _gui_state(wavelength_markers=[marker]))
        win, _ = make_window(islat)
        assert len(_tagged(win, WAVELENGTH_MARKER_TAG)) == 4  # one per panel
        win.controller.add_wavelength_marker(8.0, 8.5, label="w")
        assert len(win.controller.wavelength_markers) == 2
        assert islat.GUI.plot.toggle_state["wavelength_markers"] == [marker]
        islat.GUI.plot.toggle_state["wavelength_markers"] = []
        win.refresh_plot()  # keeps the window's own marks
        assert len(win.controller.wavelength_markers) == 2
        assert _tagged(win, WAVELENGTH_MARKER_TAG)
        win.controller.clear_wavelength_markers()
        assert not _tagged(win, WAVELENGTH_MARKER_TAG)

    def test_marks_dialog_adds_to_window(self, make_window):
        from iSLAT.Modules.Plotting.PlotView import WAVELENGTH_MARKER_TAG
        win, islat = make_window()
        win.open_marker_dialog()
        dlg = win._marker_dialog
        assert dlg is not None and dlg.winfo_exists()
        win.open_marker_dialog()  # re-uses the open dialog
        assert win._marker_dialog is dlg
        _button(dlg, "Add Mark").invoke()
        assert win.controller.wavelength_markers[0]["start"] == pytest.approx(8.0)  # window centre
        assert _tagged(win, WAVELENGTH_MARKER_TAG)
        assert islat.GUI.plot.toggle_state["wavelength_markers"] == []
        _button(dlg, "Close").invoke()
        assert win._marker_dialog is None

    def test_shift_m_opens_window_marks(self, make_window):
        win, _ = make_window()
        canvas_widget = win.view.get_canvas().get_tk_widget()
        assert win._on_shift_m(SimpleNamespace(widget=canvas_widget)) == "break"
        assert win._marker_dialog is not None
        entry = win.layout_entries["n_panels"]
        win._marker_dialog.destroy()
        win._marker_dialog = None
        assert win._on_shift_m(SimpleNamespace(widget=entry)) is None
        assert win._marker_dialog is None

    def test_save_png_is_what_is_shown(self, make_window, monkeypatch, tmp_path):
        import tkinter.filedialog
        target = tmp_path / "out.png"
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: str(target))
        win, islat = make_window()
        win.set_residuals(True)
        win.save_figure_png()
        assert target.exists() and target.stat().st_size > 0
        islat.GUI.data_field.insert_text.assert_called()

    def test_refresh_keeps_window_options(self, make_window):
        win, _ = make_window()
        win.set_data_density(True)
        win.n_panels_var.set("3")
        win.apply_settings()
        win.refresh_plot()
        assert win.view._plot.n_panels == 3
        assert win.toggle_state["data_density"] is True

    def test_close_disposes_view(self, make_window):
        win, _ = make_window()
        win.on_closing()
        assert win.view._plot is None


class TestCompactBar:
    def test_all_controls_on_one_bar(self, make_window):
        win, _ = make_window()
        frames = [w for w in win.main_frame.winfo_children() if w.winfo_manager() == "pack"]
        assert frames == [win.toolbar_frame, win.nav_frame, win.canvas_frame]
        for child in win.toolbar_frame.winfo_children():
            assert child.winfo_manager() == "pack"
            assert child.pack_info()["side"] == "left"
        texts = {w.cget("text") for w in win.toolbar_frame.winfo_children()
                 if w.winfo_class() in ("TButton", "TMenubutton")}
        assert {"File", "Apply", "Ranges\u2026", "Display", "Y Axis", "Marks\u2026"} <= texts
        file_items = [win.file_menu.entrycget(i, "label") for i in range(win.file_menu.index("end") + 1)
                      if win.file_menu.type(i) == "command"]
        assert file_items == ["Save as PDF\u2026", "Save as PNG\u2026", "Refresh Plot"]

    def test_menus_drive_options(self, make_window):
        win, _ = make_window()
        labels = [win.display_menu.entrycget(i, "label")
                  for i in range(win.display_menu.index("end") + 1)
                  if win.display_menu.type(i) == "checkbutton"]
        assert labels == ["Residuals", "Data Density X-Scale", "Atomic Lines", "Saved Lines",
                          "Saved Line Labels", "Summed Spectrum", "Show Observed Data", "Legend"]
        win.display_menu.invoke(labels.index("Show Observed Data") + 1)  # +1 for the separator
        assert win.toggle_state["observed"] is False
        assert not any(a.get_visible() for a in _tagged(win, "_islat_observed"))
        y_labels = [win.y_menu.entrycget(i, "label") for i in range(win.y_menu.index("end") + 1)
                    if win.y_menu.type(i) != "separator"]
        assert y_labels == ["Auto-scale Y Axis", "Auto-scale Y to Model Flux",
                            "Sync Y to Data Min/Max", "Set Y Limits\u2026"]
        win.y_menu.invoke(1)
        assert win.toggle_state["y_autoscale_source"] == "model"
        win.y_menu.invoke(0)
        assert win.toggle_state["y_autoscale"] is False


class TestPanelRanges:
    RANGES = [(9.0, 9.5), (5.5, 6.0), (7.0, 8.0)]

    def test_parse_and_format(self):
        from iSLAT.Modules.GUI.FullSpectrumWindow import parse_panel_ranges, format_panel_ranges
        text = "5.5 6.0\n\n# comment\n7-8\n9, 9.5\n10 to 11  # trailing\n12\u201313"
        assert parse_panel_ranges(text) == [(5.5, 6.0), (7.0, 8.0), (9.0, 9.5), (10.0, 11.0), (12.0, 13.0)]
        assert format_panel_ranges([1, 3], [2, 4]) == "1.0000 2.0000\n3.0000 4.0000"
        for bad in ("", "5", "5 6 7", "6 5", "a b"):
            with pytest.raises(ValueError):
                parse_panel_ranges(bad)

    def test_custom_non_contiguous_ranges(self, make_window):
        win, _ = make_window()
        win.set_panel_ranges(self.RANGES)
        assert _ranges(win) == self.RANGES
        axes = _spectrum_axes(win)
        assert [ax.get_xlim() for ax in axes] == pytest.approx(self.RANGES)
        assert win.n_panels_var.get() == "3"
        assert win.ranges_btn.cget("text") == "Ranges*\u2026"

    def test_ranges_survive_other_options(self, make_window):
        win, _ = make_window()
        win.set_panel_ranges(self.RANGES)
        win.set_residuals(True)
        assert _ranges(win) == self.RANGES
        win.set_data_density(True)
        assert _ranges(win) == self.RANGES
        win.set_residuals(False)
        win.refresh_plot()
        assert _ranges(win) == self.RANGES

    def test_apply_layout_or_even_split_clears_ranges(self, make_window):
        win, _ = make_window()
        win.set_panel_ranges(self.RANGES)
        win.n_panels_var.set("2")
        win.apply_settings()
        assert win.view.panel_ranges is None
        assert len(_ranges(win)) == 2
        win.set_panel_ranges(self.RANGES)
        win.set_panel_ranges(None)
        assert win.view.panel_ranges is None
        assert _ranges(win)[0][0] == pytest.approx(5.5) and _ranges(win)[-1][1] == pytest.approx(9.5)

    def test_invalid_ranges_rejected(self, make_window):
        win, _ = make_window()
        before = _ranges(win)
        with pytest.raises(ValueError):
            win.set_panel_ranges([(6.0, 5.0)])
        assert _ranges(win) == before

    def test_ranges_dialog(self, make_window):
        win, _ = make_window()
        win.open_panel_ranges_dialog()
        dlg = win._ranges_dialog
        text = next(w for w in _walk(dlg) if w.winfo_class() == "Text")
        assert len(text.get("1.0", "end").split()) == 8  # 4 panels prefilled
        text.delete("1.0", "end")
        text.insert("1.0", "bad line")
        _button(dlg, "Apply").invoke()
        assert win.view.panel_ranges is None
        text.delete("1.0", "end")
        text.insert("1.0", "9 9.5\n5.5 6")
        _button(dlg, "Apply").invoke()
        assert _ranges(win) == [(9.0, 9.5), (5.5, 6.0)]
        _button(dlg, "Even Split").invoke()
        assert win.view.panel_ranges is None

    def test_parse_data_keywords(self):
        from iSLAT.Modules.GUI.FullSpectrumWindow import parse_panel_ranges
        text = "DATASTART 6\n7 dataend\nDataStart-6.5\n8-DATAEND\ndatastart to DATAEND"
        assert parse_panel_ranges(text, data_start=5.0, data_end=11.0) == [
            (5.0, 6.0), (7.0, 11.0), (5.0, 6.5), (8.0, 11.0), (5.0, 11.0)]
        assert parse_panel_ranges("1e-1 2e-1") == [(0.1, 0.2)]  # exponent dashes are not separators
        for bad, kwargs in (("DATASTART 6", {}),                                   # no data loaded
                            ("7 DATAEND", {"data_start": 5.0}),
                            ("DATAEND 10", {"data_start": 5.0, "data_end": 11.0}),  # start >= end
                            ("DATASTARTX 6", {"data_start": 5.0})):
            with pytest.raises(ValueError):
                parse_panel_ranges(bad, **kwargs)

    def test_ranges_dialog_data_keywords(self, make_window):
        win, _ = make_window()
        assert win.observed_data_bounds() == pytest.approx((WAVE[0], WAVE[-1]))
        win.open_panel_ranges_dialog()
        dlg = win._ranges_dialog
        text = next(w for w in _walk(dlg) if w.winfo_class() == "Text")
        text.delete("1.0", "end")
        text.insert("1.0", "DATASTART 6\n9 DATAEND")
        _button(dlg, "Apply").invoke()
        assert _ranges(win) == pytest.approx([(WAVE[0], 6.0), (9.0, WAVE[-1])])


class TestStackedPanelRanges:
    def test_plot_level_api(self):
        from iSLAT.Modules.Plotting import FullSpectrumPlot
        plot = FullSpectrumPlot(WAVE, FLUX, molecules=_Molecules(), panel_ranges=[(8, 9), (5, 6)])
        assert list(zip(plot._panel_edges, plot._panel_ends)) == [(8, 9), (5, 6)]
        assert (plot.n_panels, plot._xlim_start, plot._xlim_end) == (2, 5.0, 9.0)
        plot.generate_plot()
        assert [ax.get_xlim() for ax in plot.subplots.values()] == pytest.approx([(8, 9), (5, 6)])
        plot._compute_panel_layout()  # e.g. update_data / regenerate keep the ranges
        assert list(plot._panel_edges) == [8, 5]
        plot.set_panel_ranges(None)
        assert plot._panel_edges[0] == 5.0 and plot._panel_ends[-1] == 9.0
        with pytest.raises(ValueError):
            plot.set_panel_ranges([])
        plot.close()


class TestWindowSavedLineLabels:
    def test_window_only_label_toggle(self, make_window):
        import iSLAT.Modules.Plotting.SavedLineLabels as sll
        sll.reset_saved_line_label_settings()
        try:
            win, _ = make_window()
            assert win.toggle_state["saved_line_labels"] is True
            win.set_saved_lines(True)
            texts = lambda: [t for ax in win.view.fig.axes for t in ax.texts if hasattr(t, "_islat_saved_line")]
            assert texts()
            win.set_saved_line_labels(False)
            assert not texts()
            assert _tagged(win, "_islat_saved_line")  # the lines stay
            assert sll.labels_visible("Full Spectrum")  # main GUI setting untouched
            win.set_saved_line_labels(True)
            assert texts()
        finally:
            sll.reset_saved_line_label_settings()

    def test_starts_from_main_full_spectrum_setting(self, make_window):
        import iSLAT.Modules.Plotting.SavedLineLabels as sll
        sll.set_labels_visible("Full Spectrum", False)
        try:
            win, _ = make_window()
            assert win.toggle_state["saved_line_labels"] is False
            assert win.option_vars["saved_line_labels"].get() is False
        finally:
            sll.reset_saved_line_label_settings()
