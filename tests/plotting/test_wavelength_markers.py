# -*- coding: utf-8 -*-
"""User wavelength markers (Shift+M): rendering, view hooks, controller API, keybind and dialog."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pytest

from iSLAT.Modules.Plotting.PlotView import (
    WAVELENGTH_MARKER_TAG,
    DEFAULT_WAVELENGTH_MARKER_COLOR,
    make_wavelength_marker,
    draw_wavelength_markers,
    clear_wavelength_markers,
)


def _marker_artists(ax):
    return [a for a in ax.get_children() if hasattr(a, WAVELENGTH_MARKER_TAG)]


SINGLE = {"start": 6.0, "end": 6.0, "label": "", "color": "red"}
RANGE = {"start": 7.0, "end": 7.5, "label": "band", "color": "cyan"}


class TestMakeMarker:
    def test_single_wavelength(self):
        m = make_wavelength_marker(12.5)
        assert m == {"start": 12.5, "end": 12.5, "label": "", "color": DEFAULT_WAVELENGTH_MARKER_COLOR,
                     "label_color": DEFAULT_WAVELENGTH_MARKER_COLOR,
                     "label_side": "center", "linestyle": "dashed"}

    def test_label_color_defaults_to_line_color_and_can_differ(self):
        assert make_wavelength_marker(1.0, color="#123456")["label_color"] == "#123456"
        m = make_wavelength_marker(1.0, color="red", label_color=" #00ff00 ")
        assert (m["color"], m["label_color"]) == ("red", "#00ff00")

    def test_reversed_range_is_swapped_and_label_stripped(self):
        m = make_wavelength_marker(13.0, 12.0, label="  CO2 ", color="blue")
        assert (m["start"], m["end"], m["label"], m["color"]) == (12.0, 13.0, "CO2", "blue")

    def test_label_side_and_linestyle_are_normalised(self):
        m = make_wavelength_marker(1.0, label_side=" Right ", linestyle="DOTTED")
        assert (m["label_side"], m["linestyle"]) == ("right", "dotted")

    @pytest.mark.parametrize("alias, name", [("-", "solid"), ("--", "dashed"), (":", "dotted"),
                                             ("-.", "dashdot"), ("dash-dot", "dashdot")])
    def test_linestyle_aliases(self, alias, name):
        assert make_wavelength_marker(1.0, linestyle=alias)["linestyle"] == name

    @pytest.mark.parametrize("kwargs", [
        {"start": float("nan")},
        {"start": 1.0, "end": float("inf")},
        {"start": 1.0, "color": "not-a-color"},
        {"start": 1.0, "label_color": "not-a-color"},
        {"start": 1.0, "label_side": "top"},
        {"start": 1.0, "linestyle": "wavy"},
    ])
    def test_invalid_input_raises(self, kwargs):
        with pytest.raises(ValueError):
            make_wavelength_marker(**kwargs)


def _labels(ax):
    return [t for t in ax.texts if hasattr(t, WAVELENGTH_MARKER_TAG)]


def _dashes(linestyle, lw):
    """Dash pattern matplotlib uses for *linestyle* at linewidth *lw*."""
    return LineCollection([[(0, 0), (0, 1)]], linestyles=linestyle, linewidths=lw).get_linestyle()


class TestDrawMarkers:
    def test_line_and_band_without_changing_limits(self):
        ax = Figure().add_subplot(111)
        ax.plot([5.0, 10.0], [0.0, 1.0])
        ax.set_xlim(5.0, 10.0)
        ax.set_ylim(0.0, 1.0)
        far = {"start": 50.0, "end": 60.0, "label": "", "color": "red"}
        draw_wavelength_markers(ax, [SINGLE, RANGE, far])
        assert ax.get_xlim() == (5.0, 10.0)
        assert ax.get_ylim() == (0.0, 1.0)
        arts = _marker_artists(ax)
        # one line for the single mark, two edge lines (one collection) per range
        assert sum(isinstance(a, LineCollection) for a in arts) == 3
        assert sum(isinstance(a, PolyCollection) and not isinstance(a, LineCollection) for a in arts) == 2
        assert [t.get_text() for t in _labels(ax)] == ["band"]

    @pytest.mark.parametrize("side, x, ha, dx_sign", [
        ("left", 7.0, "right", -1), ("center", 7.25, "center", 0), ("right", 7.5, "left", 1),
    ])
    def test_label_side_on_range(self, side, x, ha, dx_sign):
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(RANGE, label_side=side)])
        (label,) = _labels(ax)
        assert label.xy == pytest.approx((x, 0.98))
        assert label.get_horizontalalignment() == ha
        assert np.sign(label.xyann[0]) == dx_sign

    @pytest.mark.parametrize("side, ha", [("left", "right"), ("center", "center"), ("right", "left")])
    def test_label_side_on_single_line(self, side, ha):
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(SINGLE, label="x", label_side=side)])
        (label,) = _labels(ax)
        assert label.xy[0] == pytest.approx(6.0)
        assert label.get_horizontalalignment() == ha

    @pytest.mark.parametrize("linestyle", ["solid", "dashed", "dotted", "dashdot"])
    def test_linestyle_applies_to_single_line_and_range_edges(self, linestyle):
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(SINGLE, linestyle=linestyle), dict(RANGE, linestyle=linestyle)])
        single, edges = [a for a in _marker_artists(ax) if isinstance(a, LineCollection)]
        assert single.get_linestyle() == _dashes(linestyle, 1.2)
        assert edges.get_linestyle() == _dashes(linestyle, 1.0)
        assert len(edges.get_segments()) == 2

    def test_legacy_marker_without_new_keys_uses_defaults(self):
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(SINGLE, label="old", linestyle="bogus")])
        (line,) = [a for a in _marker_artists(ax) if isinstance(a, LineCollection)]
        assert line.get_linestyle() == _dashes("dashed", 1.2)
        assert _labels(ax)[0].get_horizontalalignment() == "center"

    def test_label_color_is_independent_of_line_color(self):
        from matplotlib.colors import to_rgba
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(SINGLE, label="x", label_color="blue")])
        (line,) = [a for a in _marker_artists(ax) if isinstance(a, LineCollection)]
        assert tuple(line.get_colors()[0]) == to_rgba("red")
        assert to_rgba(_labels(ax)[0].get_color()) == to_rgba("blue")

    def test_label_color_falls_back_to_line_color(self):
        from matplotlib.colors import to_rgba
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [dict(SINGLE, label="x")])
        assert to_rgba(_labels(ax)[0].get_color()) == to_rgba("red")

    def test_markers_do_not_affect_relim_autoscale(self):
        ax = Figure().add_subplot(111)
        ax.plot([5.0, 10.0], [0.0, 1.0])
        draw_wavelength_markers(ax, [{"start": 100.0, "end": 100.0, "label": "", "color": "red"}])
        ax.relim()
        ax.autoscale_view()
        assert ax.get_xlim()[1] < 20.0

    def test_clear_removes_only_markers(self):
        ax = Figure().add_subplot(111)
        (data,) = ax.plot([1, 2], [1, 2])
        draw_wavelength_markers(ax, [SINGLE, RANGE])
        clear_wavelength_markers(ax)
        assert _marker_artists(ax) == []
        assert data in ax.lines

    def test_malformed_markers_are_skipped(self):
        ax = Figure().add_subplot(111)
        draw_wavelength_markers(ax, [{"start": "x"}, {}, SINGLE])
        assert len(_marker_artists(ax)) == 1


def _three_panel_view(markers):
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView
    fig = Figure()
    ax1, ax2, ax3 = (fig.add_subplot(3, 1, i) for i in (1, 2, 3))
    pm = MagicMock()
    pm.ax1, pm.ax2, pm.ax3 = ax1, ax2, ax3
    pm.toggle_state = {"observed": True, "y_autoscale": True, "summed": True, "legend": True,
                       "atomic_lines": False, "saved_lines": False,
                       "wavelength_markers": list(markers)}
    pm.control_bus = None
    return ThreePanelView(pm), pm


class TestThreePanelView:
    def test_markers_on_spectrum_and_inspection_not_population(self):
        view, pm = _three_panel_view([SINGLE, RANGE])
        view._reapply_display_state(pm.toggle_state)
        assert len(_marker_artists(view.ax1)) == 4  # line + band fill + band edges + label
        assert len(_marker_artists(view.ax2)) == 4
        assert _marker_artists(view.ax3) == []

    def test_reapply_is_idempotent_and_follows_state(self):
        view, pm = _three_panel_view([SINGLE])
        view._reapply_display_state(pm.toggle_state)
        view.sync_toggle_state(pm.toggle_state)
        assert len(_marker_artists(view.ax1)) == 1
        pm.toggle_state["wavelength_markers"] = []
        view.refresh_wavelength_markers()
        assert _marker_artists(view.ax1) == []

    def test_markers_survive_axes_clear(self):
        view, pm = _three_panel_view([SINGLE])
        view._reapply_display_state(pm.toggle_state)
        view.ax2.clear()
        view._apply_wavelength_markers(pm.toggle_state)
        assert len(_marker_artists(view.ax2)) == 1


class TestFullSpectrumView:
    def test_markers_on_every_panel(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView
        from iSLAT.Modules.Plotting.FullSpectrumPlot import FullSpectrumPlot
        wave = np.linspace(5.0, 11.0, 300)
        flux = np.ones_like(wave)
        pm = MagicMock()
        pm.toggle_state = {"atomic_lines": False, "saved_lines": False, "summed": True,
                           "legend": True, "observed": True, "y_autoscale": True,
                           "y_limits": None, "wavelength_markers": [SINGLE]}
        pm.control_bus = None
        view = FullSpectrumView(pm)
        plot = FullSpectrumPlot(wave_data=wave, flux_data=flux, n_panels=3)
        plot.generate_plot()
        view._plot = plot
        view._initialised = True
        view._canvas = MagicMock()
        view.sync_toggle_state(pm.toggle_state)
        axes = list(view._iter_all_axes())
        assert len(axes) == 3
        assert all(len(_marker_artists(ax)) == 1 for ax in axes)


class TestLineInspectionView:
    def test_build_plot_draws_markers(self):
        from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView
        wave = np.linspace(5.0, 11.0, 300)
        pm = MagicMock()
        pm.islat = SimpleNamespace(wave_data=wave, flux_data=np.ones_like(wave), error_data=None,
                                   active_molecule=None, molecules_dict=None)
        pm.theme = {}
        pm.summed_toggle = False
        pm.toggle_state = {"summed": False, "observed": True, "y_autoscale": True,
                           "y_limits": None, "current_selection": (6.0, 8.0),
                           "wavelength_markers": [RANGE]}
        pm.control_bus = None
        view = LineInspectionView(pm)
        view._current_selection = (6.0, 8.0)
        view._build_plot()
        assert len(_marker_artists(view._plot.ax)) == 3  # fill + edges + label


class TestPopulationDiagramView:
    def test_refresh_is_a_no_op(self):
        from iSLAT.Modules.Plotting.PopulationDiagramView import PopulationDiagramView
        pm = MagicMock()
        pm.toggle_state = {"wavelength_markers": [SINGLE]}
        view = PopulationDiagramView(pm)
        assert list(view._iter_marker_axes()) == []
        view.refresh_wavelength_markers()  # must not raise


class TestController:
    @staticmethod
    def _controller():
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        pm = iSLATPlot.__new__(iSLATPlot)
        pm.toggle_state = {}
        pm.active_view = MagicMock()
        return pm

    def test_add_remove_clear(self):
        pm = self._controller()
        pm.add_wavelength_marker(6.0)
        pm.add_wavelength_marker(8.0, 7.0, label="b", color="lime", label_side="left", linestyle="dotted",
                                 label_color="black")
        assert pm.wavelength_markers == [
            {"start": 6.0, "end": 6.0, "label": "", "color": DEFAULT_WAVELENGTH_MARKER_COLOR,
             "label_color": DEFAULT_WAVELENGTH_MARKER_COLOR,
             "label_side": "center", "linestyle": "dashed"},
            {"start": 7.0, "end": 8.0, "label": "b", "color": "lime", "label_color": "black",
             "label_side": "left", "linestyle": "dotted"},
        ]
        assert pm.active_view.refresh_wavelength_markers.call_count == 2
        pm.remove_wavelength_marker(0)
        pm.remove_wavelength_marker(5)  # out of range: ignored
        assert [m["start"] for m in pm.wavelength_markers] == [7.0]
        pm.clear_wavelength_markers()
        assert pm.wavelength_markers == []
        assert pm.active_view.refresh_wavelength_markers.call_count == 4

    def test_invalid_marker_is_rejected(self):
        pm = self._controller()
        with pytest.raises(ValueError):
            pm.add_wavelength_marker(6.0, color="nope")
        assert pm.wavelength_markers == []
        pm.active_view.refresh_wavelength_markers.assert_not_called()

    def test_end_to_end_with_three_panel_view(self):
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        view, view_pm = _three_panel_view([])
        pm = iSLATPlot.__new__(iSLATPlot)
        pm.toggle_state = view_pm.toggle_state
        pm.active_view = view
        pm.add_wavelength_marker(6.0)
        assert len(_marker_artists(view.ax1)) == 1
        pm.clear_wavelength_markers()
        assert _marker_artists(view.ax1) == []


class TestExportHelper:
    def test_draws_on_every_figure_axes(self):
        from iSLAT.Modules.Plotting.FullSpectrumView import draw_toggle_state_markers
        fig = Figure()
        axes = [fig.add_subplot(2, 1, i) for i in (1, 2)]
        draw_toggle_state_markers(fig, {"wavelength_markers": [SINGLE]})
        assert all(len(_marker_artists(ax)) == 1 for ax in axes)
        draw_toggle_state_markers(fig, {})
        draw_toggle_state_markers(None, {"wavelength_markers": [SINGLE]})
        assert all(len(_marker_artists(ax)) == 1 for ax in axes)


def _handler(selection=None):
    from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
    h = InteractionHandler.__new__(InteractionHandler)
    h.islat = SimpleNamespace(display_range=None)
    h.plot_manager = SimpleNamespace(active_view=SimpleNamespace(),
                                     toggle_state={"current_selection": selection})
    h.ax1 = Figure().add_subplot(111)
    h.ax1.set_xlim(14.0, 15.0)
    return h


def _key(keysym, state=0):
    widget = MagicMock()
    widget.winfo_class.return_value = "Canvas"
    return SimpleNamespace(keysym=keysym, state=state, widget=widget)


class TestKeybinding:
    def test_shift_m_opens_dialog_plain_m_toggles_summed(self):
        h = _handler()
        h._open_wavelength_marker_dialog = MagicMock()
        h._toggle_summed_spectrum = MagicMock()
        assert h._on_tk_keypress(_key("M", state=0x1)) == "break"
        h._open_wavelength_marker_dialog.assert_called_once()
        h._toggle_summed_spectrum.assert_not_called()
        assert h._on_tk_keypress(_key("m")) == "break"
        h._toggle_summed_spectrum.assert_called_once()
        h._open_wavelength_marker_dialog.assert_called_once()

    def test_typing_in_entry_is_ignored(self):
        h = _handler()
        h._open_wavelength_marker_dialog = MagicMock()
        event = _key("M", state=0x1)
        event.widget.winfo_class.return_value = "TEntry"
        assert h._on_tk_keypress(event) is None
        h._open_wavelength_marker_dialog.assert_not_called()

    def test_default_window_prefers_selection(self):
        assert _handler(selection=(14.2, 14.4))._default_marker_window() == ("range", 14.2, 14.4)
        assert _handler()._default_marker_window() == ("single", 14.5, 14.5)
        assert _handler(selection=(14.3, 14.3))._default_marker_window()[0] == "single"

    def test_describe_marker(self):
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        describe = InteractionHandler._describe_wavelength_marker
        assert describe(SINGLE) == "6.00000 \u03bcm   (red, dashed)"
        assert describe(dict(RANGE, label_side="right", linestyle="solid")) == (
            "7.00000 \u2013 7.50000 \u03bcm   [band, right]   (cyan, solid)"
        )
        assert describe(dict(RANGE, label_color="cyan")).startswith("7.00000 \u2013 7.50000 \u03bcm   [band, center]")
        assert "[band, center, #00ff00]" in describe(dict(RANGE, label_color="#00ff00"))


# ---------------------------------------------------------------------------
# Dialog smoke test (needs a display)
# ---------------------------------------------------------------------------

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


def _find(widget, cls_name, text=None):
    for child in widget.winfo_children():
        if child.winfo_class() == cls_name and (text is None or child.cget("text") == text):
            return child
        found = _find(child, cls_name, text)
        if found is not None:
            return found
    return None


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


class TestDialog:
    @staticmethod
    def _open(tk_root):
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        h = _handler(selection=(14.2, 14.4))
        pm = iSLATPlot.__new__(iSLATPlot)
        pm.toggle_state = {"current_selection": (14.2, 14.4)}
        pm.active_view = MagicMock()
        h.plot_manager = pm
        h._tk_root = tk_root
        h.open_wavelength_marker_dialog()
        return h, pm, h._marker_dialog

    @staticmethod
    def _swatches(win):
        from iSLAT.Modules.GUI.GUIFunctions import ColorButton
        line, label = [w for w in _walk(win) if isinstance(w, ColorButton)]
        return line, label

    def test_add_and_clear_via_dialog(self, tk_root):
        h, pm, win = self._open(tk_root)
        assert win is not None and win.winfo_exists()

        h.open_wavelength_marker_dialog()  # re-uses the open dialog
        assert h._marker_dialog is win

        _find(win, "TButton", "Add Mark").invoke()
        assert pm.wavelength_markers == [
            {"start": 14.2, "end": 14.4, "label": "", "color": "#ff00ff", "label_color": "#ff00ff",
             "label_side": "center", "linestyle": "dashed"},
        ]
        listbox = _find(win, "Listbox")
        assert listbox.size() == 1

        _find(win, "TRadiobutton", "Right").invoke()
        (style_combo,) = [c for c in _walk(win)
                          if c.winfo_class() == "TCombobox" and "Dotted" in c.cget("values")]
        assert str(style_combo.cget("state")) == "readonly"
        style_combo.set("Dotted")
        _find(win, "TButton", "Add Mark").invoke()
        assert pm.wavelength_markers[-1]["label_side"] == "right"
        assert pm.wavelength_markers[-1]["linestyle"] == "dotted"
        assert listbox.size() == 2

        _find(win, "TButton", "Clear All").invoke()
        assert pm.wavelength_markers == []
        assert listbox.size() == 0

        _find(win, "TButton", "Close").invoke()
        assert h._marker_dialog is None

    def test_color_pickers(self, tk_root, monkeypatch):
        import tkinter.colorchooser as colorchooser
        h, pm, win = self._open(tk_root)
        line_swatch, label_swatch = self._swatches(win)
        assert (line_swatch.color, label_swatch.color) == ("#ff00ff", "#ff00ff")
        same = _find(win, "TCheckbutton", "Same as line")

        calls = []

        def fake_askcolor(color=None, **kw):
            calls.append((kw.get("title"), color, kw.get("parent")))
            return ((0, 0, 0), picks.pop(0))

        monkeypatch.setattr(colorchooser, "askcolor", fake_askcolor)

        # Line color: the label follows while "Same as line" is on.
        picks = ["#00ff00"]
        line_swatch.command()
        assert calls[-1] == ("Pick marker line color", "#ff00ff", win)
        assert (line_swatch.color, label_swatch.color) == ("#00ff00", "#00ff00")
        assert line_swatch.cget("bg") == "#00ff00"

        # Label color: independent, unticks "Same as line".
        picks = ["#123456"]
        label_swatch.command()
        assert calls[-1][:2] == ("Pick marker label color", "#00ff00")
        assert label_swatch.color == "#123456"
        assert "selected" not in same.state()

        # Now changing the line color leaves the label color alone.
        picks = ["#abcdef"]
        line_swatch.command()
        assert (line_swatch.color, label_swatch.color) == ("#abcdef", "#123456")

        _find(win, "TButton", "Add Mark").invoke()
        assert (pm.wavelength_markers[-1]["color"], pm.wavelength_markers[-1]["label_color"]) == (
            "#abcdef", "#123456")

        # Cancelling the picker changes nothing.
        monkeypatch.setattr(colorchooser, "askcolor", lambda **kw: (None, None))
        line_swatch.command()
        assert line_swatch.color == "#abcdef"

        # Re-ticking "Same as line" resets the label color to the line color.
        same.invoke()
        assert label_swatch.color == "#abcdef"
        _find(win, "TButton", "Add Mark").invoke()
        assert pm.wavelength_markers[-1]["label_color"] == "#abcdef"
        _find(win, "TButton", "Close").invoke()
