# -*- coding: utf-8 -*-
"""Dedicated unit tests for ControlField, ControlSurface, and ControlBus.

All tests are fully headless — no Tk display is required.  The tests exercise
the pure-Python registration, routing, and state logic without constructing
any Tk widgets.
"""

import pytest
from unittest.mock import MagicMock, call

from iSLAT.Modules.GUI.ControlField import (
    ActionField,
    ControlField,
    ToggleField,
    EntryField,
    DisplayRangeField,
    DropdownField,
    LabelField,
    RenderContext,
)
from iSLAT.Modules.GUI.ControlSurface import ControlBus, ControlSurface


# ======================================================================
# Headless surface stub
# ======================================================================

class _NoopSurface(ControlSurface):
    """Concrete ControlSurface that does nothing on _rebuild."""

    def __init__(self):
        super().__init__()
        self.rebuild_count = 0

    def _rebuild(self):
        self.rebuild_count += 1


# ======================================================================
# ControlField base — common attribute contract
# ======================================================================

class TestControlFieldBase:
    """ControlField stores key, label, owner, and tip on construction."""

    def _make_entry(self, key="my_key", label="My Label", owner=None, tip=None):
        return EntryField(
            key, label,
            getter=lambda: 42.0,
            setter=lambda v: None,
            owner=owner,
            tip=tip,
        )

    def test_key_stored(self):
        f = self._make_entry(key="alpha")
        assert f.key == "alpha"

    def test_label_stored(self):
        f = self._make_entry(label="Alpha:")
        assert f.label == "Alpha:"

    def test_owner_stored(self):
        sentinel = object()
        f = self._make_entry(owner=sentinel)
        assert f.owner is sentinel

    def test_tip_stored(self):
        f = self._make_entry(tip="A helpful tip")
        assert f.tip == "A helpful tip"

    def test_tip_defaults_to_none(self):
        f = self._make_entry()
        assert f.tip is None

    def test_owner_defaults_to_none(self):
        f = self._make_entry()
        assert f.owner is None


# ======================================================================
# ToggleField
# ======================================================================

class TestToggleField:
    """ToggleField stores getter/setter and is a ControlField subclass."""

    def _make(self, value=True, owner=None):
        state = [value]
        getter = lambda: state[0]
        setter = lambda v: state.__setitem__(0, v)
        return ToggleField("show_x", "Show X", getter=getter, setter=setter, owner=owner), state

    def test_is_control_field(self):
        f, _ = self._make()
        assert isinstance(f, ControlField)

    def test_getter_returns_initial(self):
        f, state = self._make(value=False)
        assert f.getter() is False

    def test_setter_updates_value(self):
        f, state = self._make(value=False)
        f.setter(True)
        assert state[0] is True

    def test_build_widget_returns_empty_list_without_tk(self):
        """build_widget requires a Tk parent; just confirm it's callable."""
        f, _ = self._make()
        assert callable(f.build_widget)

    def test_refresh_is_callable(self):
        f, _ = self._make()
        # refresh with no widget refs should be a no-op
        f.refresh([])


# ======================================================================
# EntryField
# ======================================================================

class TestEntryField:
    """EntryField stores getter/setter, datatype, width."""

    def _make(self, datatype="float", value=3.14, owner=None):
        state = [value]
        getter = lambda: state[0]
        setter = lambda v: state.__setitem__(0, v)
        return EntryField(
            "temp", "Temp:", getter=getter, setter=setter,
            datatype=datatype, width=8, owner=owner,
        ), state

    def test_is_control_field(self):
        f, _ = self._make()
        assert isinstance(f, ControlField)

    def test_datatype_stored(self):
        f, _ = self._make(datatype="int")
        assert f.datatype == "int"

    def test_width_stored(self):
        f, _ = self._make()
        assert f.width == 8

    def test_getter_returns_value(self):
        f, state = self._make(value=2.71)
        assert f.getter() == pytest.approx(2.71)

    def test_setter_updates_state(self):
        f, state = self._make(value=0.0)
        f.setter(9.99)
        assert state[0] == pytest.approx(9.99)

    def test_coerce_float(self):
        f, _ = self._make(datatype="float")
        assert f._coerce("3.5") == pytest.approx(3.5)

    def test_coerce_int(self):
        f, _ = self._make(datatype="int")
        assert f._coerce("7") == 7

    def test_coerce_str(self):
        f, _ = self._make(datatype="str")
        assert f._coerce("hello") == "hello"

    def test_format_float(self):
        f, _ = self._make(datatype="float")
        result = f._format(1.23456789)
        assert "1.23" in result  # 4 significant figures

    def test_format_int(self):
        f, _ = self._make(datatype="int")
        assert f._format(5.9) == "5"

    def test_format_str(self):
        f, _ = self._make(datatype="str")
        assert f._format("abc") == "abc"


# ======================================================================
# DisplayRangeField
# ======================================================================

class TestDisplayRangeField:
    """DisplayRangeField stores getter/setter; build_widget returns empty list."""

    def _make(self, owner=None):
        state = [(5.0, 25.0)]
        getter = lambda: state[0]
        setter = lambda s, e: state.__setitem__(0, (s, e))
        return DisplayRangeField(
            "_range", getter=getter, setter=setter, owner=owner,
        ), state

    def test_is_control_field(self):
        f, _ = self._make()
        assert isinstance(f, ControlField)

    def test_key_stored(self):
        f, _ = self._make()
        assert f.key == "_range"

    def test_label_is_empty(self):
        f, _ = self._make()
        assert f.label == ""

    def test_getter_returns_tuple(self):
        f, _ = self._make()
        assert f.getter() == (5.0, 25.0)

    def test_setter_updates_state(self):
        f, state = self._make()
        f.setter(10.0, 20.0)
        assert state[0] == (10.0, 20.0)

    def test_build_widget_requires_no_parent(self):
        """build_widget should return [] without needing a real Tk parent."""
        f, _ = self._make()
        # Pass None as parent — DisplayRangeField creates no widgets
        result = f.build_widget(None, RenderContext.CONTROL_PANEL)
        assert result == []


# ======================================================================
# DropdownField
# ======================================================================

class TestDropdownField:
    """DropdownField stores options list, getter, setter."""

    def _make(self, options=None, owner=None):
        if options is None:
            options = ["a", "b", "c"]
        state = [options[0]]
        getter = lambda: state[0]
        setter = lambda v: state.__setitem__(0, v)
        return DropdownField(
            "mode", "Mode:", getter=getter, setter=setter,
            options=options, owner=owner,
        ), state

    def test_is_control_field(self):
        f, _ = self._make()
        assert isinstance(f, ControlField)

    def test_options_stored_as_copy(self):
        opts = ["x", "y"]
        f, _ = self._make(options=opts)
        opts.append("z")  # mutating original should not affect stored list
        assert f.options == ["x", "y"]

    def test_getter_returns_initial(self):
        f, state = self._make()
        assert f.getter() == "a"

    def test_setter_updates_state(self):
        f, state = self._make()
        f.setter("b")
        assert state[0] == "b"


# ======================================================================
# ActionField
# ======================================================================

class TestActionField:
    """ActionField runs its command; it has no value to refresh."""

    def test_is_control_field(self):
        f = ActionField("act", "Do It", command=lambda: None)
        assert isinstance(f, ControlField)
        assert (f.key, f.label, f.owner, f.tip) == ("act", "Do It", None, None)

    def test_menu_entry_runs_the_command(self):
        ran = []

        def command():
            ran.append(True)

        menu = MagicMock()
        refs = ActionField("act", "Do It", command=command).build_menu_entry(menu)
        assert refs == []
        (entry,) = menu.add_command.call_args_list
        assert entry.kwargs["label"] == "Do It"
        entry.kwargs["command"]()
        assert ran == [True]

    def test_refresh_does_nothing(self):
        ActionField("act", "Do It", command=lambda: None).refresh([])


# ======================================================================
# ControlPanelSurface: refreshing the Plot Start / Plot Range binding
# ======================================================================

class TestControlPanelSurfaceRefresh:
    """Refreshing the DisplayRangeField re-reads Plot Start / Plot Range through the surface's callback."""

    def _surface(self):
        from iSLAT.Modules.GUI.Widgets.ControlPanel import ControlPanelSurface
        bindings = []
        container = MagicMock()
        container.winfo_children.return_value = []
        return ControlPanelSurface(container, on_display_range_changed=bindings.append), bindings

    def test_refresh_passes_the_binding_again(self):
        surface, bindings = self._surface()
        field = DisplayRangeField("_range", getter=lambda: (5.0, 6.0), setter=lambda s, e: None)
        surface.register(field)
        assert bindings == [field]          # the rebuild
        surface.refresh("_range")
        assert bindings == [field, field]

    def test_other_and_unknown_keys_leave_the_binding_alone(self):
        surface, bindings = self._surface()
        surface.register(DisplayRangeField("_range", getter=lambda: (5.0, 6.0), setter=lambda s, e: None))
        surface.refresh("missing")
        assert len(bindings) == 1


# ======================================================================
# LabelField
# ======================================================================

class TestLabelField:
    """LabelField stores getter; build_widget deferred to Tk tests."""

    def test_is_control_field(self):
        f = LabelField("lbl", "Label:", getter=lambda: "val")
        assert isinstance(f, ControlField)

    def test_getter_returns_value(self):
        f = LabelField("lbl", "Label:", getter=lambda: 99)
        assert f.getter() == 99


# ======================================================================
# RenderContext enum
# ======================================================================

class TestRenderContext:
    def test_values(self):
        assert RenderContext.CONTROL_PANEL.value == "control_panel"
        assert RenderContext.TOP_BAR.value == "top_bar"
        assert RenderContext.DIALOG.value == "dialog"


# ======================================================================
# ControlSurface — registration and field management
# ======================================================================

class TestControlSurface:
    """Tests for the ControlSurface ABC via the _NoopSurface stub."""

    def _field(self, key="f1", owner=None):
        return EntryField(key, key + ":", getter=lambda: 0, setter=lambda v: None, owner=owner)

    # ------------------------------------------------------------------
    def test_register_adds_field(self):
        s = _NoopSurface()
        f = self._field("k1")
        s.register(f)
        assert s.get_field("k1") is f

    def test_register_triggers_rebuild(self):
        s = _NoopSurface()
        s.register(self._field("k1"))
        assert s.rebuild_count == 1

    def test_register_replaces_existing_key(self):
        s = _NoopSurface()
        f1 = self._field("k1")
        f2 = self._field("k1")
        s.register(f1)
        s.register(f2)
        assert s.get_field("k1") is f2

    def test_register_many_adds_all(self):
        s = _NoopSurface()
        fields = [self._field(f"k{i}") for i in range(3)]
        s.register_many(fields)
        for f in fields:
            assert s.get_field(f.key) is f

    def test_register_many_triggers_single_rebuild(self):
        s = _NoopSurface()
        s.register_many([self._field("a"), self._field("b")])
        assert s.rebuild_count == 1

    def test_register_many_empty_list_no_rebuild(self):
        s = _NoopSurface()
        s.register_many([])
        assert s.rebuild_count == 0

    def test_unregister_removes_field(self):
        s = _NoopSurface()
        s.register(self._field("x"))
        removed = s.unregister("x")
        assert removed is True
        assert s.get_field("x") is None

    def test_unregister_missing_returns_false(self):
        s = _NoopSurface()
        assert s.unregister("nonexistent") is False

    def test_unregister_owner_removes_matching(self):
        s = _NoopSurface()
        owner = object()
        s.register(self._field("a", owner=owner))
        s.register(self._field("b", owner=owner))
        s.register(self._field("c"))
        removed = s.unregister_owner(owner)
        assert removed is True
        assert s.get_field("a") is None
        assert s.get_field("b") is None
        assert s.get_field("c") is not None

    def test_unregister_owner_no_match_returns_false(self):
        s = _NoopSurface()
        s.register(self._field("a"))
        assert s.unregister_owner(object()) is False

    def test_unregister_owner_triggers_single_rebuild(self):
        s = _NoopSurface()
        owner = object()
        s.register(self._field("a", owner=owner))
        s.register(self._field("b", owner=owner))
        s.rebuild_count = 0  # reset after setup
        s.unregister_owner(owner)
        assert s.rebuild_count == 1

    def test_get_field_returns_none_for_missing(self):
        s = _NoopSurface()
        assert s.get_field("missing") is None

    def test_fields_returns_snapshot(self):
        s = _NoopSurface()
        f1 = self._field("a")
        f2 = self._field("b")
        s.register(f1)
        s.register(f2)
        snap = s.fields()
        assert snap == [f1, f2]

    def test_fields_is_copy(self):
        """Mutating the returned list should not affect the surface's registry."""
        s = _NoopSurface()
        s.register(self._field("a"))
        snap = s.fields()
        snap.clear()
        assert len(s.fields()) == 1

    def test_fields_preserves_insertion_order(self):
        s = _NoopSurface()
        keys = ["z", "a", "m"]
        for k in keys:
            s.register(self._field(k))
        assert [f.key for f in s.fields()] == keys

    def test_refresh_noop_when_no_widget_refs(self):
        s = _NoopSurface()
        getter = MagicMock(return_value=7.0)
        f = EntryField("k", "K:", getter=getter, setter=lambda v: None)
        s.register(f)
        # No widget refs stored (NoopSurface._rebuild doesn't populate _widget_refs)
        s.refresh("k")  # should not raise
        # getter not called during refresh because no widget refs exist
        getter.assert_not_called()

    def test_refresh_missing_key_no_error(self):
        s = _NoopSurface()
        s.refresh("does_not_exist")  # must not raise


# ======================================================================
# ControlBus — surface management
# ======================================================================

class TestControlBusSurfaceManagement:
    """Tests for registering and retrieving surfaces on a ControlBus."""

    def test_register_surface_and_get(self):
        bus = ControlBus()
        surf = _NoopSurface()
        bus.register_surface("cp", surf)
        assert bus.get_surface("cp") is surf

    def test_get_unknown_surface_returns_none(self):
        bus = ControlBus()
        assert bus.get_surface("unknown") is None

    def test_surface_names(self):
        bus = ControlBus()
        bus.register_surface("cp", _NoopSurface())
        bus.register_surface("tb", _NoopSurface())
        assert set(bus.surface_names()) == {"cp", "tb"}

    def test_register_surface_replaces_existing(self):
        bus = ControlBus()
        s1 = _NoopSurface()
        s2 = _NoopSurface()
        bus.register_surface("cp", s1)
        bus.register_surface("cp", s2)
        assert bus.get_surface("cp") is s2


# ======================================================================
# ControlBus — field routing
# ======================================================================

class TestControlBusFieldRouting:
    """Tests for register/unregister/refresh field routing on ControlBus."""

    def _setup(self):
        bus = ControlBus()
        cp = _NoopSurface()
        tb = _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)
        return bus, cp, tb

    def _field(self, key, owner=None):
        return EntryField(key, key + ":", getter=lambda: 0, setter=lambda v: None, owner=owner)

    # ------------------------------------------------------------------
    def test_register_routes_to_correct_surface(self):
        bus, cp, tb = self._setup()
        f = self._field("my_field")
        bus.register(f, "control_panel")
        assert cp.get_field("my_field") is f
        assert tb.get_field("my_field") is None

    def test_register_unknown_surface_no_error(self):
        bus, cp, tb = self._setup()
        f = self._field("x")
        bus.register(f, "nonexistent")  # must not raise

    def test_register_many_routes_to_surface(self):
        bus, cp, tb = self._setup()
        fields = [self._field(f"f{i}") for i in range(3)]
        bus.register_many(fields, "control_panel")
        for f in fields:
            assert cp.get_field(f.key) is f

    def test_unregister_removes_from_surface(self):
        bus, cp, tb = self._setup()
        f = self._field("k1")
        bus.register(f, "control_panel")
        removed = bus.unregister("k1", "control_panel")
        assert removed is True
        assert cp.get_field("k1") is None

    def test_unregister_missing_key_returns_false(self):
        bus, cp, tb = self._setup()
        assert bus.unregister("nope", "control_panel") is False

    def test_unregister_unknown_surface_returns_false(self):
        bus, cp, tb = self._setup()
        assert bus.unregister("k", "phantom") is False

    def test_unregister_owner_fans_out_to_all_surfaces(self):
        bus, cp, tb = self._setup()
        owner = object()
        f_cp = EntryField("a", "A:", getter=lambda: 0, setter=lambda v: None, owner=owner)
        f_tb = ToggleField("b", "B", getter=lambda: True, setter=lambda v: None, owner=owner)
        bus.register(f_cp, "control_panel")
        bus.register(f_tb, "top_bar")

        bus.unregister_owner(owner)

        assert cp.get_field("a") is None
        assert tb.get_field("b") is None

    def test_unregister_owner_leaves_other_owners(self):
        bus, cp, tb = self._setup()
        owner_a = object()
        owner_b = object()
        f_a = self._field("a", owner=owner_a)
        f_b = self._field("b", owner=owner_b)
        bus.register(f_a, "control_panel")
        bus.register(f_b, "control_panel")

        bus.unregister_owner(owner_a)

        assert cp.get_field("a") is None
        assert cp.get_field("b") is f_b

    def test_unregister_owner_no_match_no_error(self):
        bus, cp, tb = self._setup()
        bus.unregister_owner(object())  # must not raise

    def test_refresh_routes_to_surface(self):
        bus, cp, tb = self._setup()
        f = self._field("k")
        bus.register(f, "control_panel")
        bus.refresh("k", "control_panel")  # must not raise

    def test_refresh_unknown_surface_no_error(self):
        bus, cp, tb = self._setup()
        bus.refresh("k", "nonexistent")  # must not raise


# ======================================================================
# ControlBus — multiple surfaces, isolated rebuilds
# ======================================================================

class TestControlBusIsolatedRebuilds:
    """Registering on one surface should not trigger rebuilds on others."""

    def test_register_only_rebuilds_target_surface(self):
        bus = ControlBus()
        cp = _NoopSurface()
        tb = _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)

        f = EntryField("x", "X:", getter=lambda: 0, setter=lambda v: None)
        bus.register(f, "control_panel")

        assert cp.rebuild_count == 1
        assert tb.rebuild_count == 0

    def test_unregister_owner_only_rebuilds_affected_surfaces(self):
        bus = ControlBus()
        cp = _NoopSurface()
        tb = _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)

        owner = object()
        f_cp = EntryField("a", "A:", getter=lambda: 0, setter=lambda v: None, owner=owner)
        bus.register(f_cp, "control_panel")
        cp.rebuild_count = 0  # reset

        bus.unregister_owner(owner)
        assert cp.rebuild_count == 1
        assert tb.rebuild_count == 0


# ======================================================================
# Integration — view lifecycle simulation
# ======================================================================

class TestViewLifecycleIntegration:
    """Simulate a full view activate/deactivate cycle on the ControlBus."""

    def test_activate_deactivate_cycle(self):
        bus = ControlBus()
        cp = _NoopSurface()
        tb = _NoopSurface()
        bus.register_surface("control_panel", cp)
        bus.register_surface("top_bar", tb)

        view = object()  # stand-in for a PlotView

        # Activate: register fields
        entry = EntryField("n_panels", "Panels:", getter=lambda: 8, setter=lambda v: None, owner=view)
        toggle = ToggleField("show_residuals", "Residuals", getter=lambda: False, setter=lambda v: None, owner=view)
        display = DisplayRangeField("_range", getter=lambda: (5.0, 25.0), setter=lambda s, e: None, owner=view)

        bus.register(entry, "control_panel")
        bus.register(display, "control_panel")
        bus.register(toggle, "top_bar")

        assert cp.get_field("n_panels") is entry
        assert cp.get_field("_range") is display
        assert tb.get_field("show_residuals") is toggle

        # Deactivate: unregister all owned fields
        bus.unregister_owner(view)

        assert cp.fields() == []
        assert tb.fields() == []

    def test_two_views_sequential(self):
        """Switching from view_a to view_b should leave only view_b's fields."""
        bus = ControlBus()
        cp = _NoopSurface()
        bus.register_surface("control_panel", cp)

        view_a = object()
        view_b = object()

        fa = EntryField("param_a", "A:", getter=lambda: 1, setter=lambda v: None, owner=view_a)
        fb = EntryField("param_b", "B:", getter=lambda: 2, setter=lambda v: None, owner=view_b)

        # View A activates
        bus.register(fa, "control_panel")
        assert cp.get_field("param_a") is fa

        # View A deactivates, View B activates
        bus.unregister_owner(view_a)
        bus.register(fb, "control_panel")

        assert cp.get_field("param_a") is None
        assert cp.get_field("param_b") is fb

    def test_same_key_different_owners_replaced(self):
        """When view_b uses the same field key as view_a, it should replace it."""
        bus = ControlBus()
        cp = _NoopSurface()
        bus.register_surface("control_panel", cp)

        view_a = object()
        view_b = object()

        fa = EntryField("n_panels", "Panels:", getter=lambda: 8, setter=lambda v: None, owner=view_a)
        fb = EntryField("n_panels", "Panels:", getter=lambda: 12, setter=lambda v: None, owner=view_b)

        bus.register(fa, "control_panel")
        bus.unregister_owner(view_a)
        bus.register(fb, "control_panel")

        field = cp.get_field("n_panels")
        assert field is fb
        assert field.getter() == 12
