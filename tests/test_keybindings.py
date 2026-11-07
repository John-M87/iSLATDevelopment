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
