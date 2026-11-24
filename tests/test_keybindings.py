"""Tests for the declarative keyboard shortcut system (KeyBindings + InteractionHandler dispatch)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from iSLAT.Modules.GUI.KeyBindings import (
    DEFAULT_KEYMAP,
    Action,
    ActionRegistry,
    chord_from_event,
    format_chord,
    normalize_chord,
    shortcut_text,
)

SHIFT, CTRL, CMD, CAPS = 0x1, 0x4, 0x8, 0x2


def _event(keysym, state=0, widget_class="Canvas"):
    widget = MagicMock()
    widget.winfo_class.return_value = widget_class
    return SimpleNamespace(keysym=keysym, state=state, widget=widget)


# ---------------------------------------------------------------------------
# Chord parsing / formatting
# ---------------------------------------------------------------------------

class TestChordFromEvent:
    @pytest.mark.parametrize("keysym,state,expected", [
        ("f", 0, "F"),
        ("F", SHIFT, "Shift+F"),
        ("f", CTRL, "Ctrl+F"),
        ("F", CTRL | SHIFT, "Ctrl+Shift+F"),
        ("F", CAPS, "F"),                       # Caps Lock doesn't add Shift
        ("Up", 0, "Up"),
        ("Left", 0, "Left"),
        ("comma", 0, ","),
        ("less", SHIFT, "Shift+,"),
        ("period", 0, "."),
        ("greater", SHIFT, "Shift+."),
        ("<", SHIFT, "Shift+,"),
        (">", 0, "."),
    ])
    def test_windows_linux(self, keysym, state, expected):
        assert chord_from_event(_event(keysym, state), system="Windows") == expected

    def test_command_is_accelerator_on_mac_only(self):
        assert chord_from_event(_event("s", CMD), system="Darwin") == "Ctrl+S"
        assert chord_from_event(_event("s", CTRL), system="Darwin") == "Ctrl+S"
        # On Windows bit 0x8 is not Ctrl (e.g. Num Lock) and must be ignored.
        assert chord_from_event(_event("s", CMD), system="Windows") == "S"

    @pytest.mark.parametrize("keysym", ["Shift_L", "Control_R", "Alt_L", "Caps_Lock", "??", ""])
    def test_modifier_only_or_unknown_is_none(self, keysym):
        assert chord_from_event(_event(keysym), system="Linux") is None


class TestNormalizeAndFormat:
    def test_normalize(self):
        assert normalize_chord("ctrl+shift+f") == "Ctrl+Shift+F"
        assert normalize_chord("Shift+Ctrl+f") == "Ctrl+Shift+F"
        assert normalize_chord("cmd+s") == "Ctrl+S"
        assert normalize_chord("Shift+<") == "Shift+,"
        assert normalize_chord("up") == "Up"

    @pytest.mark.parametrize("bad", ["", "Hyper+F", "Ctrl+"])
    def test_normalize_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            normalize_chord(bad)

    def test_format_platform_aware(self):
        assert format_chord("Ctrl+Shift+F", system="Windows") == "Ctrl+Shift+F"
        assert format_chord("Ctrl+Shift+F", system="Darwin") == "Cmd+Shift+F"
        assert format_chord(",", system="Linux") == "<"
        assert format_chord("Shift+.", system="Linux") == "Shift+>"

    def test_shortcut_text(self):
        assert shortcut_text("save_parameters", system="Windows") == "Ctrl+S"
        assert shortcut_text("save_parameters", system="Darwin") == "Cmd+S"
        assert shortcut_text("open_wavelength_marker_dialog") == "Shift+M"
        assert shortcut_text("advance_plot_start") == ">"
        with pytest.raises(KeyError):
            shortcut_text("no_such_action")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestActionRegistry:
    def test_dispatch_runs_bound_action(self):
        reg = ActionRegistry()
        cb = MagicMock()
        reg.register(Action("go", cb, ("ctrl+g",)))
        assert reg.chords_for("go") == ("Ctrl+G",)
        assert reg.dispatch("Ctrl+G") is True
        cb.assert_called_once_with()
        assert reg.dispatch("G") is False
        assert reg.dispatch(None) is False

    def test_conflicts_and_duplicates_rejected(self):
        reg = ActionRegistry()
        reg.register(Action("a", MagicMock(), ("X",)))
        with pytest.raises(ValueError):
            reg.register(Action("b", MagicMock(), ("x",)))
        with pytest.raises(ValueError):
            reg.register(Action("a", MagicMock(), ("Y",)))

    def test_unregister_frees_chord(self):
        reg = ActionRegistry()
        reg.register(Action("a", MagicMock(), ("X",)))
        assert reg.unregister("a") is True
        assert reg.unregister("a") is False
        assert reg.action_for_chord("X") is None
        reg.register(Action("b", MagicMock(), ("X",)))

    def test_register_defaults_requires_every_callback(self):
        with pytest.raises(KeyError):
            ActionRegistry().register_defaults({})

    def test_default_keymap_has_no_conflicts(self):
        reg = ActionRegistry()
        reg.register_defaults({a: MagicMock() for a in DEFAULT_KEYMAP})
        assert len(reg.actions()) == len(DEFAULT_KEYMAP)


# ---------------------------------------------------------------------------
# InteractionHandler dispatch (behaviour must match the old if/elif chain)
# ---------------------------------------------------------------------------

def _handler():
    from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
    h = InteractionHandler.__new__(InteractionHandler)
    for method in set(InteractionHandler._ACTION_METHODS.values()) - {
        "_retreat_plot_start_half", "_advance_plot_start_half",
    }:
        setattr(h, method, MagicMock(name=method))
    plot_range_var = MagicMock()
    plot_range_var.get.return_value = "0.4"
    h.islat = SimpleNamespace(GUI=SimpleNamespace(
        control_panel=SimpleNamespace(plot_range_var=plot_range_var)))
    return h


# (keysym, state) -> handler method expected to run (previous behaviour)
_PREVIOUS_BINDINGS = [
    (("f", 0), "_toggle_full_spectrum"),
    (("f", CTRL), "_open_full_spectrum_window"),
    (("F", CTRL | SHIFT), "_output_full_spectrum"),
    (("s", 0), "_toggle_saved_lines"),
    (("s", CTRL), "_save_parameters"),
    (("S", CTRL | SHIFT), "_save_parameters_to_file"),
    (("l", 0), "_toggle_legend"),
    (("l", CTRL), "_load_parameters"),
    (("L", CTRL | SHIFT), "_load_parameters_from_file"),
    (("a", 0), "_toggle_atomic_lines"),
    (("m", 0), "_toggle_summed_spectrum"),
    (("M", SHIFT), "_open_wavelength_marker_dialog"),
    (("v", 0), "_toggle_active_molecule_visibility"),
    (("V", SHIFT), "_toggle_all_molecule_visibility"),
    (("r", 0), "_toggle_residuals"),
    (("t", 0), "_toggle_three_panel"),
    (("c", 0), "_open_center_view_dialog"),
    (("Up", 0), "_select_previous_molecule"),
    (("Down", 0), "_select_next_molecule"),
    (("Left", 0), "_cycle_spectrum_previous"),
    (("Right", 0), "_cycle_spectrum_next"),
    (("comma", 0), "_retreat_plot_start"),
    (("period", 0), "_advance_plot_start"),
    (("k", CTRL), "_open_command_palette"),
    (("o", CTRL), "_load_line_list"),
    (("O", CTRL | SHIFT), "_set_output_file"),
    (("p", 0), "_toggle_match_pixel_sampling"),
    (("n", CTRL), "_add_molecule"),
    (("F", SHIFT), "_filter_active_line_list"),
    (("S", SHIFT), "_save_line"),
    (("g", 0), "_fit_line"),
    (("G", SHIFT), "_single_slab_fit"),
    (("z", 0), "_halve_plot_range"),
    (("Z", SHIFT), "_double_plot_range"),
    (("d", CTRL), "_load_spectrum"),
    (("D", CTRL | SHIFT), "_open_sample_manager"),
    (("C", CTRL | SHIFT), "_copy_view_to_clipboard"),
    (("e", CTRL), "_export_models"),
    (("o", 0), "_toggle_observed_data"),
    (("y", 0), "_toggle_y_autoscale"),
    (("Y", SHIFT), "_open_y_limits_dialog"),
    (("n", 0), "_next_line_list_line"),
    (("N", SHIFT), "_previous_line_list_line"),
]


class TestInteractionHandlerDispatch:
    @pytest.mark.parametrize("key,method", _PREVIOUS_BINDINGS)
    def test_previous_bindings_preserved(self, key, method, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Windows")
        h = _handler()
        assert h._on_tk_keypress(_event(*key)) == "break"
        getattr(h, method).assert_called_once_with()
        for other in {m for _, m in _PREVIOUS_BINDINGS} - {method}:
            getattr(h, other).assert_not_called()

    @pytest.mark.parametrize("keysym,method", [("less", "_retreat_plot_start"),
                                               ("greater", "_advance_plot_start")])
    def test_shift_nav_moves_half_range(self, keysym, method):
        h = _handler()
        assert h._on_tk_keypress(_event(keysym, SHIFT)) == "break"
        getattr(h, method).assert_called_once_with(extra_amount=pytest.approx(0.2))

    def test_cmd_is_accelerator_on_mac(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        h = _handler()
        assert h._on_tk_keypress(_event("s", CMD)) == "break"
        h._save_parameters.assert_called_once_with()

    def test_unbound_keys_propagate(self):
        h = _handler()
        assert h._on_tk_keypress(_event("q")) is None
        assert h._on_tk_keypress(_event("c", CTRL)) is None  # Ctrl+C no longer opens Center View

    def test_text_input_ignored(self):
        h = _handler()
        assert h._on_tk_keypress(_event("f", widget_class="TEntry")) is None
        h._toggle_full_spectrum.assert_not_called()

    def test_destroyed_widget_ignored(self):
        h = _handler()
        assert h._on_tk_keypress(SimpleNamespace(keysym="f", state=0, widget=".dead")) is None

    def test_overrides_after_first_dispatch_are_honoured(self):
        h = _handler()
        h._on_tk_keypress(_event("f"))
        replacement = MagicMock()
        h._toggle_full_spectrum = replacement
        h._on_tk_keypress(_event("f"))
        replacement.assert_called_once_with()

    def test_every_default_action_has_a_method(self):
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        assert set(InteractionHandler._ACTION_METHODS) == set(DEFAULT_KEYMAP)
        for method in InteractionHandler._ACTION_METHODS.values():
            assert callable(getattr(InteractionHandler, method))


class TestToggleGrid:
    def test_toggle_grid_flips_all_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler

        fig = Figure()
        h = InteractionHandler.__new__(InteractionHandler)
        h.ax1, h.ax2 = fig.add_subplot(121), fig.add_subplot(122)
        h.ax3 = None
        h.canvas = MagicMock()

        def grid_on(ax):
            return any(line.get_visible() for line in ax.xaxis.get_gridlines())

        h._toggle_grid()
        assert grid_on(h.ax1) and grid_on(h.ax2)
        h._toggle_grid()
        assert not grid_on(h.ax1) and not grid_on(h.ax2)
        assert h.canvas.draw_idle.call_count == 2


class TestLineListViewExportShortcut:
    def test_export_shortcut_stops_propagation(self):
        from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
        win = LineListViewWindow.__new__(LineListViewWindow)
        win._export_csv = MagicMock()
        assert LineListViewWindow._on_export_shortcut(win, None) == "break"
        win._export_csv.assert_called_once_with()


# ---------------------------------------------------------------------------
# Command palette (Ctrl/Cmd+K)
# ---------------------------------------------------------------------------

def test_cmd_k_opens_palette_even_from_a_text_field():
    h = _handler()
    assert h._on_tk_keypress(_event("k", CTRL, widget_class="TEntry")) == "break"
    h._open_command_palette.assert_called_once_with()
    assert h._on_tk_keypress(_event("s", CTRL, widget_class="TEntry")) is None  # other shortcuts still ignored
    h._save_parameters.assert_not_called()


def test_action_title():
    from iSLAT.Modules.GUI.KeyBindings import action_title
    assert action_title("toggle_full_spectrum") == "Toggle full spectrum"
    assert action_title("open_center_view_dialog") == "Open center view dialog"


def _registry():
    registry = ActionRegistry()
    registry.register_defaults({action_id: MagicMock(name=action_id) for action_id in DEFAULT_KEYMAP})
    return registry


def test_filter_actions_matches_every_word_and_ranks_prefix_matches_first():
    from iSLAT.Modules.GUI.KeyBindings import filter_actions
    actions = _registry().actions()
    assert filter_actions(actions, "") == actions
    found = [a.id for a in filter_actions(actions, "full spec")]
    assert set(found) == {"toggle_full_spectrum", "open_full_spectrum_window", "output_full_spectrum"}
    assert [a.id for a in filter_actions(actions, "output")][0] == "output_full_spectrum"
    assert [a.id for a in filter_actions(actions, "SAVE param")] == ["save_parameters", "save_parameters_to_file"]
    assert filter_actions(actions, "no such command") == []


def test_filter_actions_finds_shortcuts():
    from iSLAT.Modules.GUI.KeyBindings import filter_actions
    actions = _registry().actions()
    assert [a.id for a in filter_actions(actions, "ctrl+shift+s", system="Windows")] == ["save_parameters_to_file"]
    assert [a.id for a in filter_actions(actions, "cmd+shift+s", system="Darwin")] == ["save_parameters_to_file"]


@pytest.fixture
def palette(monkeypatch):
    """A CommandPalette over a hidden root, itself kept hidden."""
    import tkinter as tk
    from iSLAT.Modules.GUI.KeyBindings import CommandPalette
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    monkeypatch.setattr(CommandPalette, "_place", lambda self, parent: self.withdraw())
    registry = _registry()
    pal = CommandPalette(root, registry)
    yield pal, registry, root
    pal.close()
    root.withdraw()
    root.destroy()


def test_palette_lists_every_action_but_itself(palette):
    pal, registry, _root = palette
    titles = [pal.tree.item(i, "text") for i in pal.tree.get_children()]
    assert len(titles) == len(DEFAULT_KEYMAP) - 1
    assert "Open command palette" not in titles
    first = pal.tree.get_children()[0]
    assert pal.tree.item(first, "values")[0] == format_chord(DEFAULT_KEYMAP[pal.selected_action().id][0])


def test_palette_filters_moves_and_runs(palette):
    pal, registry, root = palette
    pal.query.set("full spectrum")
    assert len(pal.tree.get_children()) == 3
    first = pal.selected_action()
    pal._move(1)
    chosen = pal.selected_action()
    assert chosen is not first
    pal._move(-5)  # stays on the list
    assert pal.selected_action() is first
    pal._move(1)
    pal.run_selected()
    assert pal._closed
    root.update_idletasks()  # the action runs once the palette is gone
    chosen.callback.assert_called_once_with()
    first.callback.assert_not_called()


def test_palette_escape_closes_without_running(palette):
    pal, registry, root = palette
    pal.close()
    root.update_idletasks()
    for action in registry.actions():
        action.callback.assert_not_called()


class TestGuiActions:
    """The new shortcuts call the same GUI handlers as their buttons / menu items."""

    def _real_handler(self):
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        h = InteractionHandler.__new__(InteractionHandler)
        gui = SimpleNamespace(file_interaction_pane=MagicMock(), control_panel=MagicMock(), top_bar=MagicMock())
        h.islat = SimpleNamespace(GUI=gui, active_molecule=SimpleNamespace(name="CO2"))
        return h, gui

    def test_each_calls_its_gui_handler(self):
        h, gui = self._real_handler()
        h._load_line_list()
        gui.file_interaction_pane._load_input_line_list.assert_called_once_with()
        h._set_output_file()
        gui.file_interaction_pane._load_output_line_measurements.assert_called_once_with()
        h._toggle_match_pixel_sampling()
        gui.control_panel._toggle_match_spectral_sampling.assert_called_once_with()
        h._add_molecule()
        gui.top_bar.add_molecule.assert_called_once_with()
        h._filter_active_line_list()
        gui.control_panel._filter_line_list_action.assert_called_once_with("CO2")
        h._save_line()
        gui.top_bar.save_line.assert_called_once_with()
        h._fit_line()
        gui.top_bar.fit_selected_line.assert_called_once_with()
        h._single_slab_fit()
        gui.top_bar.single_slab_fit.assert_called_once_with()
        h._halve_plot_range()
        gui.control_panel.scale_plot_range.assert_called_once_with(0.5)
        h._double_plot_range()
        gui.control_panel.scale_plot_range.assert_called_with(2)
        h._load_spectrum()
        gui.file_interaction_pane.load_spectra.assert_called_once_with()
        h._open_sample_manager()
        gui.top_bar.manage_sample.assert_called_once_with()
        h._export_models()
        gui.top_bar.export_models.assert_called_once_with()

    def test_before_the_gui_exists_they_do_nothing(self):
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        h = InteractionHandler.__new__(InteractionHandler)
        h.islat = SimpleNamespace(active_molecule=None)
        for method in ("_load_line_list", "_set_output_file", "_toggle_match_pixel_sampling",
                       "_add_molecule", "_filter_active_line_list", "_save_line", "_fit_line",
                       "_single_slab_fit", "_halve_plot_range", "_double_plot_range",
                       "_load_spectrum", "_open_sample_manager", "_export_models",
                       "_copy_view_to_clipboard"):
            getattr(h, method)()  # no error

    def test_gui_handlers_exist(self):
        from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanel
        from iSLAT.Modules.GUI.Widgets.FileInteractionPane import FileInteractionPane
        from iSLAT.Modules.GUI.Widgets.TopBar import TopBar
        for cls, name in ((FileInteractionPane, "_load_input_line_list"),
                          (FileInteractionPane, "_load_output_line_measurements"),
                          (ControlPanel, "_toggle_match_spectral_sampling"),
                          (ControlPanel, "_filter_line_list_action"),
                          (ControlPanel, "scale_plot_range"),
                          (TopBar, "add_molecule"),
                          (TopBar, "save_line"),
                          (TopBar, "fit_selected_line"),
                          (TopBar, "single_slab_fit"),
                          (TopBar, "manage_sample"),
                          (TopBar, "export_models"),
                          (FileInteractionPane, "load_spectra")):
            assert callable(getattr(cls, name)), f"{cls.__name__}.{name}"


@pytest.mark.parametrize("start, plot_range, new_start, new_range, fmt", [
    (17.0, 0.4, 17.1, 0.2, "{:.6g}"),
    (16.8075, 2.3815, 17.403, 1.19075, "{:.6g}"),
    (16.8075, 2.3815, 17.40324, 1.19, "{:.2f}"),  # a range field that rounds
])
def test_halve_plot_range_keeps_centre(start, plot_range, new_start, new_range, fmt):
    from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanel

    class Var:
        def __init__(self, value): self.value = str(value)
        def get(self): return self.value
        def set(self, value): self.value = value

    applied = []
    panel = SimpleNamespace(
        plot_start_var=Var(start), plot_range_var=Var(plot_range),
        _set_var=lambda var, value: var.set(value),
        _format_value=lambda value, name: (fmt if name == "display_range_range" else "{:.6g}").format(value),
        _update_display_range=lambda: applied.append(True),
        islat=SimpleNamespace(display_range=None),  # no exact range known: use the fields
    )
    ControlPanel.scale_plot_range(panel, 0.5)
    got_start, got_range = float(panel.plot_start_var.get()), float(panel.plot_range_var.get())
    assert (got_start, got_range) == pytest.approx((new_start, new_range), abs=1e-4)
    assert got_start + got_range / 2 == pytest.approx(start + plot_range / 2, abs=1e-4)
    assert applied == [True]


def test_halve_plot_range_uses_the_exact_range_not_the_rounded_fields():
    from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanel

    class Var:
        def __init__(self, value): self.value = str(value)
        def get(self): return self.value
        def set(self, value): self.value = value

    panel = SimpleNamespace(
        plot_start_var=Var("16.8075"), plot_range_var=Var("2.38"),  # field rounded from 2.3815
        _set_var=lambda var, value: var.set(value),
        _format_value=lambda value, name: ("{:.2f}" if name == "display_range_range" else "{:.6g}").format(value),
        _update_display_range=lambda: None,
        islat=SimpleNamespace(display_range=(16.8075, 19.189)),
    )
    ControlPanel.scale_plot_range(panel, 0.5)
    start, plot_range = float(panel.plot_start_var.get()), float(panel.plot_range_var.get())
    assert start + plot_range / 2 == pytest.approx((16.8075 + 19.189) / 2, abs=1e-4)


def test_double_plot_range_keeps_centre():
    from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanel

    class Var:
        def __init__(self, value): self.value = str(value)
        def get(self): return self.value
        def set(self, value): self.value = value

    panel = SimpleNamespace(
        plot_start_var=Var("17.0"), plot_range_var=Var("0.4"),
        _set_var=lambda var, value: var.set(value),
        _format_value=lambda value, name: f"{value:.6g}",
        _update_display_range=lambda: None,
        islat=SimpleNamespace(display_range=(17.0, 17.4)),
    )
    ControlPanel.scale_plot_range(panel, 2)
    assert (float(panel.plot_start_var.get()), float(panel.plot_range_var.get())) == pytest.approx((16.8, 0.8))


def test_match_sampling_button_label_has_no_shortcut_text():
    from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanel
    assert ControlPanel._match_sampling_text(True) == "✓ Match Pix. Sampling"
    assert ControlPanel._match_sampling_text(False) == "  Match Pix. Sampling"


def test_copy_view_copies_the_active_view():
    from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
    h = InteractionHandler.__new__(InteractionHandler)
    view = MagicMock()
    h.plot_manager = SimpleNamespace(active_view=view)
    h._copy_view_to_clipboard()
    view.copy_figure_to_clipboard.assert_called_once_with()
    h.plot_manager = SimpleNamespace(active_view=None)
    h._copy_view_to_clipboard()  # no view yet: no error


def test_load_spectrum_click_and_shortcut_share_one_method():
    """A plain click on Load Spectrum runs load_spectra (the keybind's method); Cmd/Ctrl-click loads parameters."""
    from iSLAT.Modules.GUI.Widgets.FileInteractionPane import FileInteractionPane
    pane = SimpleNamespace(load_spectra=MagicMock(), islat_class=MagicMock())
    FileInteractionPane._handle_load_spectrum_click(pane, SimpleNamespace(state=0))
    pane.load_spectra.assert_called_once_with()
    pane.islat_class.load_spectrum.assert_not_called()
    FileInteractionPane._handle_load_spectrum_click(pane, SimpleNamespace(state=CTRL | CMD))
    pane.islat_class.load_spectrum.assert_called_once_with(load_parameters=True, force_dialog=True)


class TestStepLineList:
    LAMS = [17.30, 17.10, 17.20]  # unsorted on purpose

    def _handler(self, monkeypatch, selection=None, plot_range=(17.0, 17.4)):
        import pandas as pd
        from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
        monkeypatch.setattr("iSLAT.Modules.FileHandling.iSLATFileHandling.read_line_saves",
                            lambda **_: pd.DataFrame({"lam": self.LAMS}))
        h = InteractionHandler.__new__(InteractionHandler)
        h.islat = SimpleNamespace(input_line_list="lines.csv")
        h.selected_range = selection
        h._get_active_plot_range = lambda: plot_range

        def show(ref):  # selects a window that is not centred on the line
            h.selected_range = (ref.lam - 0.01, ref.lam + 0.05)
        h.plot_manager = SimpleNamespace(show_line_in_spectrum=MagicMock(side_effect=show))
        return h

    def _shown(self, h):
        return h.plot_manager.show_line_in_spectrum.call_args.args[0].lam

    def test_steps_from_plot_centre_then_from_the_last_line(self, monkeypatch):
        h = self._handler(monkeypatch, plot_range=(17.0, 17.3))
        h._next_line_list_line()
        assert self._shown(h) == pytest.approx(17.20)
        h._next_line_list_line()
        assert self._shown(h) == pytest.approx(17.30)
        h._previous_line_list_line()
        assert self._shown(h) == pytest.approx(17.20)

    def test_steps_from_a_new_selection(self, monkeypatch):
        h = self._handler(monkeypatch)
        h._next_line_list_line()
        h.selected_range = (17.0, 17.1)  # user selects elsewhere
        h._next_line_list_line()
        assert self._shown(h) == pytest.approx(17.10)

    def test_stops_at_the_ends(self, monkeypatch):
        h = self._handler(monkeypatch, selection=(17.25, 17.35))
        h._next_line_list_line()
        h.plot_manager.show_line_in_spectrum.assert_not_called()


# ---------------------------------------------------------------------------
# Action descriptions and recently used actions (command palette)
# ---------------------------------------------------------------------------

def test_every_action_has_a_description_and_no_stray_ones():
    from iSLAT.Modules.GUI.KeyBindings import ACTION_DESCRIPTIONS
    assert set(ACTION_DESCRIPTIONS) == set(DEFAULT_KEYMAP)
    assert all(text.strip() for text in ACTION_DESCRIPTIONS.values())


def test_register_defaults_fills_descriptions():
    from iSLAT.Modules.GUI.KeyBindings import ACTION_DESCRIPTIONS
    for action in _registry().actions():
        assert action.description == ACTION_DESCRIPTIONS[action.id]


def test_action_tooltip_is_description_then_keybind():
    from iSLAT.Modules.GUI.KeyBindings import action_tooltip
    assert action_tooltip("fit_line", system="Windows") == (
        "Fit a Gaussian to the currently\nselected spectral line.\nKeybind: G")
    assert action_tooltip("save_parameters", system="Darwin").endswith("\nKeybind: Cmd+S")


def test_one_line_collapses_breaks():
    from iSLAT.Modules.GUI.KeyBindings import one_line
    assert one_line("Fit a Gaussian to the currently\nselected  line.") == "Fit a Gaussian to the currently selected line."


def test_filter_actions_falls_back_to_descriptions():
    from iSLAT.Modules.GUI.KeyBindings import filter_actions
    actions = _registry().actions()
    assert [a.id for a in filter_actions(actions, "gaussian")] == ["fit_line"]
    assert "copy_view_to_clipboard" in [a.id for a in filter_actions(actions, "image")]
    # A title match wins: descriptions are not searched then.
    assert [a.id for a in filter_actions(actions, "fit line")] == ["fit_line"]


def test_filter_actions_lists_recent_first():
    from iSLAT.Modules.GUI.KeyBindings import filter_actions
    actions = _registry().actions()
    recent = ["export_models", "toggle_legend"]
    shown = [a.id for a in filter_actions(actions, "", recent=recent)]
    assert shown[:2] == recent
    assert shown[2:] == [a.id for a in actions if a.id not in recent]
    # With a query, prefix matches still lead; recency orders within them.
    shown = [a.id for a in filter_actions(actions, "toggle", recent=["toggle_legend"])]
    assert shown[0] == "toggle_legend"
    assert set(shown) == {a.id for a in actions if a.id.startswith("toggle")}


class TestRecentActions:
    def test_most_recent_first_without_duplicates(self):
        registry = _registry()
        for action_id in ("fit_line", "save_line", "fit_line"):
            registry.note_used(action_id)
        assert registry.recent() == ["fit_line", "save_line"]

    def test_capped(self):
        registry = _registry()
        ids = list(DEFAULT_KEYMAP)[: ActionRegistry.RECENT_LIMIT + 2]
        for action_id in ids:
            registry.note_used(action_id)
        assert registry.recent() == ids[::-1][: ActionRegistry.RECENT_LIMIT]

    def test_unknown_ignored_and_unregister_forgets(self):
        registry = _registry()
        registry.note_used("no_such_action")
        assert registry.recent() == []
        registry.note_used("fit_line")
        registry.unregister("fit_line")
        assert registry.recent() == []

    def test_key_presses_are_not_recorded(self):
        registry = _registry()
        registry.dispatch("G")
        assert registry.recent() == []


def test_palette_shows_the_selected_description(palette):
    from iSLAT.Modules.GUI.KeyBindings import one_line
    pal, _registry_, _root = palette
    pal.query.set("gaussian")
    assert pal.selected_action().id == "fit_line"
    assert pal.description.get() == one_line(pal.selected_action().description)
    pal.query.set("no such command")
    assert pal.description.get() == ""


def test_palette_remembers_what_it_ran(palette, monkeypatch):
    from iSLAT.Modules.GUI.KeyBindings import CommandPalette
    pal, registry, root = palette
    pal.query.set("export models")
    pal.run_selected()
    root.update_idletasks()
    assert registry.recent() == ["export_models"]

    again = CommandPalette(root, registry)
    try:
        assert again.selected_action().id == "export_models"
        assert again.description.get().startswith(CommandPalette.RECENT_NOTE)
        again._move(1)
        assert not again.description.get().startswith(CommandPalette.RECENT_NOTE)
    finally:
        again.close()
