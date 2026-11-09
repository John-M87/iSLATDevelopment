# -*- coding: utf-8 -*-
"""ViewsMenuSurface: the active view's own toggles as a section of the Views menu.

Views register their toggles (residuals, data density, the optical depth
options, ...) on the ControlBus "views_menu" surface, which shows them as
check-buttons under a "<view name> Controls" header at the bottom of the
TopBar's Views menu.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import pytest

from iSLAT.Modules.GUI.ControlField import ActionField, DropdownField, EntryField, ToggleField
from iSLAT.Modules.GUI.ControlSurface import ControlBus


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


def _top_bar_module():
    return importlib.import_module("iSLAT.Modules.GUI.Widgets.TopBar")


STATIC = [("command", "Three Panel"), ("separator", None), ("command", "Saved Line Labels")]


@pytest.fixture
def menu(tk_root):
    import tkinter as tk
    m = tk.Menu(tk_root, tearoff=0)
    m.add_command(label="Three Panel")
    m.add_separator()
    m.add_command(label="Saved Line Labels")
    yield m
    m.destroy()


@pytest.fixture
def surface(menu):
    tips = {"Three Panel": "static tip"}
    return _top_bar_module().ViewsMenuSurface(menu, tips), tips


class _View:
    VIEW_NAME = "Optical Depth"


def _toggle(key, label, state, owner, tip=None):
    return ToggleField(key, label, getter=lambda: state[key],
                       setter=lambda v: state.__setitem__(key, v), owner=owner, tip=tip)


def _entries(menu):
    end = menu.index("end")
    if end is None:
        return []
    out = []
    for i in range(end + 1):
        kind = menu.type(i)
        out.append((kind, None if kind == "separator" else menu.entrycget(i, "label")))
    return out


def _checked(menu, index):
    return bool(menu.tk.getboolean(menu.getvar(menu.entrycget(index, "variable"))))


class TestViewsMenuSurface:
    def test_no_fields_adds_nothing(self, menu, surface):
        assert _entries(menu) == STATIC

    def test_toggles_render_as_a_section_after_the_static_entries(self, menu, surface):
        surf, _ = surface
        view = _View()
        state = {"log": True, "stems": False}
        surf.register_many([_toggle("log", "Log tau", state, view), _toggle("stems", "Stems", state, view)])
        n = len(STATIC)
        assert _entries(menu) == STATIC + [
            ("separator", None), ("command", "Optical Depth Controls"),
            ("checkbutton", "Log tau"), ("checkbutton", "Stems"),
        ]
        assert menu.entrycget(n + 1, "state") == "disabled"   # header only
        assert _checked(menu, n + 2) and not _checked(menu, n + 3)

    def test_invoking_an_entry_calls_the_setter(self, menu, surface):
        surf, _ = surface
        state = {"log": False}
        surf.register(_toggle("log", "Log tau", state, _View()))
        idx = menu.index("end")
        menu.invoke(idx)
        assert state["log"] is True and _checked(menu, idx)
        menu.invoke(idx)
        assert state["log"] is False and not _checked(menu, idx)

    def test_entry_shows_the_real_state_when_the_setter_refuses(self, menu, surface):
        surf, _ = surface
        surf.register(ToggleField("locked", "Locked", getter=lambda: False,
                                  setter=lambda v: None, owner=_View()))
        idx = menu.index("end")
        menu.invoke(idx)
        assert not _checked(menu, idx)

    def test_refresh_all_follows_outside_changes(self, menu, surface):
        surf, _ = surface
        state = {"log": False}
        surf.register(_toggle("log", "Log tau", state, _View()))
        state["log"] = True   # e.g. changed with a keybind
        surf.refresh_all()
        assert _checked(menu, menu.index("end"))

    def test_unregister_owner_removes_the_section_and_its_tips(self, menu, surface):
        surf, tips = surface
        view = _View()
        surf.register(_toggle("log", "Log tau", {"log": False}, view, tip="log tip"))
        assert tips["Log tau"] == "log tip"
        surf.unregister_owner(view)
        assert _entries(menu) == STATIC
        assert tips == {"Three Panel": "static tip"}

    def test_switching_views_replaces_the_section(self, menu, surface):
        surf, _ = surface
        state = {"a": False, "b": False}
        first, second = _View(), SimpleNamespace(VIEW_NAME="Full Spectrum")
        surf.register(_toggle("a", "A", state, first))
        surf.unregister_owner(first)
        surf.register(_toggle("b", "B", state, second))
        assert _entries(menu) == STATIC + [
            ("separator", None), ("command", "Full Spectrum Controls"), ("checkbutton", "B"),
        ]

    def test_static_tips_are_kept(self, menu, surface):
        surf, tips = surface
        view = _View()
        surf.register(_toggle("x", "Three Panel", {"x": False}, view, tip="field tip"))
        assert tips["Three Panel"] == "static tip"
        surf.unregister_owner(view)
        assert tips["Three Panel"] == "static tip"

    def test_fields_without_a_menu_form_leave_no_empty_section(self, menu, surface):
        surf, _ = surface
        surf.register(EntryField("n", "N:", getter=lambda: 1, setter=lambda v: None,
                                 datatype="int", owner=_View()))
        assert _entries(menu) == STATIC

    def test_generic_header_without_a_view_name(self, menu, surface):
        surf, _ = surface
        surf.register(_toggle("x", "X", {"x": False}, owner=object()))
        assert ("command", "View Controls") in _entries(menu)


class TestViewsRegisterOnTheMenu:
    def test_full_spectrum_view_toggles(self, menu):
        from iSLAT.Modules.Plotting.FullSpectrumView import PANEL_RANGES_LABEL, FullSpectrumView
        bus = ControlBus()
        bus.register_surface("views_menu", _top_bar_module().ViewsMenuSurface(menu))
        pm = MagicMock()
        pm.toggle_state = {"show_residuals": True, "data_density": False}
        pm.control_bus = bus
        view = FullSpectrumView(pm)

        view._register_control_fields()
        n = len(STATIC)
        assert _entries(menu)[n:] == [
            ("separator", None), ("command", "Full Spectrum Controls"),
            ("command", PANEL_RANGES_LABEL),
            ("checkbutton", "Residuals"), ("checkbutton", "Data Density X-Scale"),
            ("checkbutton", "Stack Sample Spectra"),
            ("cascade", "Stack Layout"), ("cascade", "Stack Models"),
        ]
        assert _checked(menu, n + 3) and not _checked(menu, n + 4) and not _checked(menu, n + 5)

        view.deactivate()
        assert _entries(menu) == STATIC
        assert menu.winfo_children() == []      # the submenus went with their entries


class TestTopBarSync:
    def test_opening_the_menu_refreshes_the_view_toggles(self, menu):
        tb_mod = _top_bar_module()
        bar = tb_mod.TopBar.__new__(tb_mod.TopBar)
        bar.main_plot = SimpleNamespace()
        bar._sync_saved_labels_var = lambda: None
        surf = tb_mod.ViewsMenuSurface(menu)
        bar._views_menu_surface = surf
        state = {"x": False}
        surf.register(_toggle("x", "X", state, _View()))

        state["x"] = True
        bar._sync_plot_option_vars()
        assert _checked(menu, menu.index("end"))


def _dropdown(state, owner):
    return DropdownField(
        "pick", "Pick", getter=lambda: state["v"], setter=lambda v: state.__setitem__("v", v),
        options=["A", "B", "C"], owner=owner, tip="pick tip",
    )


class TestDropdownMenuEntry:
    def test_dropdown_is_a_cascade_of_choices(self, menu, surface):
        surf, tips = surface
        state = {"v": "B"}
        surf.register(_dropdown(state, _View()))
        idx = menu.index("end")
        assert menu.type(idx) == "cascade" and menu.entrycget(idx, "label") == "Pick"
        sub = menu.nametowidget(menu.entrycget(idx, "menu"))
        assert [sub.entrycget(i, "label") for i in range(sub.index("end") + 1)] == ["A", "B", "C"]
        var = sub.entrycget(0, "variable")
        assert menu.getvar(var) == "B"
        sub.invoke(2)
        assert state["v"] == "C"
        state["v"] = "A"          # changed elsewhere
        surf.refresh_all()
        assert menu.getvar(var) == "A"
        assert tips["Pick"] == "pick tip"

    def test_rebuilding_destroys_the_old_submenus(self, menu, surface):
        surf, _ = surface
        view = _View()
        field = _dropdown({"v": "A"}, view)
        surf.register(field)
        assert len(menu.winfo_children()) == 1
        for _ in range(3):
            surf.register(field)
        assert len(menu.winfo_children()) == 1
        surf.unregister_owner(view)
        assert menu.winfo_children() == []


class TestActionMenuEntry:
    def test_action_is_a_command_that_runs_on_invoke(self, menu, surface):
        surf, tips = surface
        opened = []

        def open_dialog():
            opened.append(True)

        surf.register(ActionField("ranges", "Ranges", command=open_dialog, owner=_View(), tip="ranges tip"))
        idx = menu.index("end")
        assert (menu.type(idx), menu.entrycget(idx, "label")) == ("command", "Ranges")
        assert menu.entrycget(idx, "state") == "normal"      # unlike the section header
        menu.invoke(idx)
        assert opened == [True]
        surf.refresh_all()                                   # nothing to re-read
        assert tips["Ranges"] == "ranges tip"
