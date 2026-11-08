"""
ControlField — portable, surface-agnostic UI controls for iSLAT.

A :class:`ControlField` describes a single UI control (toggle, entry, …) that
can be rendered on any registered :class:`ControlSurface` (ControlPanel,
TopBar, the Views menu, ConfigureSubplots dialog).  Views register fields on
activation and un-register them on deactivation via a :class:`ControlBus`.

Hierarchy::

    ControlField (ABC)
    ├── ToggleField          bool  → Checkbutton or Button
    ├── EntryField           float/int/str → Label + Entry
    │   └── DisplayRangeField   metadata-only binding for Plot Start/Range
    ├── DropdownField        str → Label + Combobox
    └── LabelField           read-only display Label pair
"""

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Render-context enum
# ---------------------------------------------------------------------------

class RenderContext(Enum):
    """The surface context in which a :class:`ControlField` is being rendered."""
    CONTROL_PANEL = "control_panel"
    TOP_BAR = "top_bar"
    DIALOG = "dialog"

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ControlField(ABC):
    """Abstract base for a single UI control that can render on multiple surfaces.

    Parameters
    ----------
    key:
        Unique string identifier within a surface.
    label:
        Human-readable display label.
    owner:
        The object that registered this field (e.g. a :class:`PlotView`).
        Used by :meth:`ControlSurface.unregister_owner`.
    tip:
        Optional tooltip text.
    """

    def __init__(
        self,
        key: str,
        label: str,
        owner: Any,
        tip: Optional[str] = None,
    ) -> None:
        self.key = key
        self.label = label
        self.owner = owner
        self.tip = tip

    @abstractmethod
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        """Build widget(s) inside *parent* for the given surface *context*.

        Implementations should *not* call ``grid()`` / ``pack()`` on the
        returned widgets — the surface is responsible for layout.  Instead,
        attach any necessary callbacks and return the widget list; the surface
        stores the references for later :meth:`refresh` calls.

        Parameters
        ----------
        parent:
            Tk container widget supplied by the surface.
        context:
            Which surface is requesting the widget.

        Returns
        -------
        List[tk.Widget]
            All created widgets, in the order they should be laid out.
        """
        ...

    def refresh(self, widget_refs: List[tk.Widget]) -> None:
        """Re-read the current value via the getter and update widget state.

        The default implementation is a no-op.  Subclasses with live state
        (e.g. :class:`EntryField`) override this.

        Parameters
        ----------
        widget_refs:
            The same list that was returned by :meth:`build_widget`.
        """

    def build_menu_entry(self, menu: tk.Menu) -> List[Any]:
        """Append this field to *menu* as menu entries.

        Returns the references that :meth:`refresh` needs later (for
        example the entry's :class:`tk.Variable`).  The default adds
        nothing and returns an empty list: fields with no menu form
        (entries, display-range bindings, ...) are skipped by menu surfaces.
        """
        return []

# ---------------------------------------------------------------------------
# Concrete field types
# ---------------------------------------------------------------------------

class ToggleField(ControlField):
    """A boolean toggle.

    * **CONTROL_PANEL** context → :class:`ttk.Checkbutton` bound to a
      :class:`tk.BooleanVar`.
    * **TOP_BAR / DIALOG** context → :class:`ttk.Button` that flips the
      state on each click.
    * In a menu (:meth:`build_menu_entry`) → a check-button entry bound to a
      :class:`tk.BooleanVar`.
    """

    def __init__(
        self,
        key: str,
        label: str,
        getter: Callable[[], bool],
        setter: Callable[[bool], None],
        owner: Any = None,
        tip: Optional[str] = None,
    ) -> None:
        super().__init__(key, label, owner, tip)
        self.getter = getter
        self.setter = setter

    # ------------------------------------------------------------------
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        if context == RenderContext.CONTROL_PANEL:
            var = tk.BooleanVar(value=bool(self.getter()))

            def _on_toggle() -> None:
                self.setter(var.get())

            widget: tk.Widget = ttk.Checkbutton(
                parent,
                text=self.label,
                variable=var,
                command=_on_toggle,
            )
            widget._field_var = var  # type: ignore[attr-defined]
            if self.tip:
                _tooltip(widget, self.tip)
            return [widget]

        else:
            # TOP_BAR / DIALOG — plain push-button that flips the bool
            def _on_click() -> None:
                self.setter(not self.getter())

            widget = ttk.Button(parent, text=self.label, command=_on_click)
            if self.tip:
                _tooltip(widget, self.tip)
            return [widget]

    # ------------------------------------------------------------------
    def build_menu_entry(self, menu: tk.Menu) -> List[Any]:
        var = tk.BooleanVar(master=menu, value=bool(self.getter()))

        def _on_toggle() -> None:
            self.setter(var.get())
            # The setter may refuse or adjust the change; show the real state.
            try:
                var.set(bool(self.getter()))
            except Exception:
                pass

        menu.add_checkbutton(label=self.label, variable=var, command=_on_toggle)
        return [var]

    # ------------------------------------------------------------------
    def refresh(self, widget_refs: List[Any]) -> None:
        if not widget_refs:
            return
        ref = widget_refs[0]
        # Menu entries store the variable itself; widgets carry it as _field_var.
        var: Optional[tk.Variable] = ref if isinstance(ref, tk.Variable) else getattr(ref, "_field_var", None)
        if var is not None:
            try:
                var.set(bool(self.getter()))
            except Exception:
                pass

# ---------------------------------------------------------------------------

class EntryField(ControlField):
    """A labelled entry box for a scalar value (float, int, or str).

    The widget pair is ``[ttk.Label, tk.Entry]`` regardless of context.
    The entry commits changes on ``<Return>``.
    """

    def __init__(
        self,
        key: str,
        label: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        datatype: str = "float",
        width: int = 7,
        owner: Any = None,
        tip: Optional[str] = None,
    ) -> None:
        super().__init__(key, label, owner, tip)
        self.getter = getter
        self.setter = setter
        self.datatype = datatype
        self.width = width

    # ------------------------------------------------------------------
    def _coerce(self, raw: str) -> Any:
        if self.datatype == "int":
            return int(float(raw))
        if self.datatype == "float":
            return float(raw)
        return raw  # str — pass through unchanged

    def _format(self, value: Any) -> str:
        if self.datatype == "int":
            return str(int(value))
        if self.datatype == "float":
            try:
                return f"{float(value):.4g}"
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    # ------------------------------------------------------------------
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        lbl = ttk.Label(parent, text=self.label)
        if self.tip:
            _tooltip(lbl, self.tip)

        var = tk.StringVar(value=self._format(self.getter()))

        def _on_change(*_args: Any) -> None:
            try:
                value = self._coerce(var.get())
                self.setter(value)
                canonical = self.getter()
                var.set(self._format(canonical))
            except (ValueError, TypeError):
                pass

        entry = tk.Entry(parent, textvariable=var, width=self.width, justify="left")
        entry.bind("<Return>", _on_change)
        entry._field_var = var  # type: ignore[attr-defined]
        return [lbl, entry]

    # ------------------------------------------------------------------
    def refresh(self, widget_refs: List[tk.Widget]) -> None:
        for w in widget_refs:
            if isinstance(w, tk.Entry):
                var: Optional[tk.StringVar] = getattr(w, "_field_var", None)
                if var is not None:
                    try:
                        var.set(self._format(self.getter()))
                    except Exception:
                        pass
                return

# ---------------------------------------------------------------------------

class DisplayRangeField(ControlField):
    """Metadata-only marker that supplies a display-range binding.

    When a ``DisplayRangeField`` is registered with a
    :class:`ControlPanelSurface`, the surface routes the existing
    *Plot Start* / *Plot Range* controls through this field's *getter* and
    *setter* instead of ``islat.display_range``.

    :meth:`build_widget` returns an empty list — no visual widget is created.
    """

    def __init__(
        self,
        key: str,
        getter: Callable[[], Tuple[float, float]],
        setter: Callable[[float, float], None],
        owner: Any = None,
    ) -> None:
        super().__init__(key, label="", owner=owner)
        self.getter = getter
        self.setter = setter

    # ------------------------------------------------------------------
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        # Purely metadata — no widget to create.
        return []

# ---------------------------------------------------------------------------

class DropdownField(ControlField):
    """A labelled combobox for selecting one item from a fixed option list."""

    def __init__(
        self,
        key: str,
        label: str,
        getter: Callable[[], str],
        setter: Callable[[str], None],
        options: List[str],
        owner: Any = None,
        tip: Optional[str] = None,
    ) -> None:
        super().__init__(key, label, owner, tip)
        self.getter = getter
        self.setter = setter
        self.options = list(options)

    # ------------------------------------------------------------------
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        lbl = ttk.Label(parent, text=self.label)
        if self.tip:
            _tooltip(lbl, self.tip)

        var = tk.StringVar(value=self.getter())
        combo = ttk.Combobox(
            parent,
            textvariable=var,
            values=self.options,
            state="readonly",
        )

        def _on_select(event: Any = None) -> None:
            self.setter(var.get())

        combo.bind("<<ComboboxSelected>>", _on_select)
        combo._field_var = var  # type: ignore[attr-defined]
        return [lbl, combo]

    # ------------------------------------------------------------------
    def refresh(self, widget_refs: List[tk.Widget]) -> None:
        for w in widget_refs:
            if isinstance(w, ttk.Combobox):
                var: Optional[tk.StringVar] = getattr(w, "_field_var", None)
                if var is not None:
                    try:
                        var.set(self.getter())
                    except Exception:
                        pass
                return

# ---------------------------------------------------------------------------

class LabelField(ControlField):
    """A read-only display that shows the current value from a getter.

    Renders as a ``[ttk.Label(name), ttk.Label(value)]`` pair.
    Call :meth:`refresh` to update the value label.
    """

    def __init__(
        self,
        key: str,
        label: str,
        getter: Callable[[], Any],
        owner: Any = None,
        tip: Optional[str] = None,
    ) -> None:
        super().__init__(key, label, owner, tip)
        self.getter = getter

    # ------------------------------------------------------------------
    def build_widget(
        self,
        parent: tk.Widget,
        context: RenderContext,
    ) -> List[tk.Widget]:
        name_lbl = ttk.Label(parent, text=self.label)
        if self.tip:
            _tooltip(name_lbl, self.tip)

        val_var = tk.StringVar(value=str(self.getter()))
        val_lbl = ttk.Label(parent, textvariable=val_var)
        val_lbl._field_var = val_var  # type: ignore[attr-defined]
        return [name_lbl, val_lbl]

    # ------------------------------------------------------------------
    def refresh(self, widget_refs: List[tk.Widget]) -> None:
        if len(widget_refs) < 2:
            return
        var: Optional[tk.StringVar] = getattr(widget_refs[1], "_field_var", None)
        if var is not None:
            try:
                var.set(str(self.getter()))
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _tooltip(widget: tk.Widget, text: str) -> None:
    """Attach a tooltip to *widget* if the Tooltips module is available."""
    try:
        from iSLAT.Modules.GUI.Tooltips import CreateToolTip
        CreateToolTip(widget, text)
    except Exception:
        pass