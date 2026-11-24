"""Parameter Sets menu: lists the session's parameter sets and switches between them."""

import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Any, Dict, Optional

from iSLAT.Modules.DataTypes.Molecule import Molecule
from ..GUIFunctions import create_menu_btn
from ..KeyBindings import shortcut_text
from ..Tooltips import CreateToolTip

MODIFIED_MARK = " •"   # shown after a set with unsaved changes
NO_SET_LABEL = "Parameter Sets"
_MAX_LABEL_CHARS = 28


class ParameterSetMenu:
    """A menu button showing the active parameter set, with every set listed to switch to."""

    def __init__(self, parent: tk.Widget, theme: Dict[str, Any], islat: Any,
                 row: int, column: int, menu_options: Optional[Dict[str, Any]] = None) -> None:
        self.islat = islat
        self.button = create_menu_btn(parent, theme, NO_SET_LABEL, row, column)
        self.menu = tk.Menu(self.button, tearoff=0, postcommand=self._rebuild, **(menu_options or {}))
        self.button.config(menu=self.menu)
        self._choice = tk.StringVar(master=self.button)
        self._refresh_pending = False
        CreateToolTip(self.button,
                      "The parameter sets opened or made this session.\n"
                      f"Choose one to show it; edits to the others are kept.\n{MODIFIED_MARK.strip()} = unsaved changes\n"
                      f"Previous / next set: {shortcut_text('previous_parameter_set')} / "
                      f"{shortcut_text('next_parameter_set')}")

        self.store.add_listener(self.refresh)
        Molecule.add_molecule_parameter_change_callback(self._on_parameter_change)
        try:
            islat.molecules_dict.add_global_parameter_change_callback(self._on_parameter_change)
        except AttributeError:
            pass
        self.refresh()

    @property
    def store(self):
        return self.islat.parameter_sets

    # -- display ---------------------------------------------------------
    def _label(self, name: str) -> str:
        modified = self.store.is_modified(self.islat.molecules_dict, name)
        return name + (MODIFIED_MARK if modified else "")

    def refresh(self) -> None:
        """Show the active set (and whether it has unsaved changes) on the button."""
        self._refresh_pending = False
        name = self.store.active
        if name is None:
            text = NO_SET_LABEL
        else:
            # Shorten the name, never the unsaved-changes mark
            shown = name if len(name) <= _MAX_LABEL_CHARS else name[:_MAX_LABEL_CHARS - 1] + "…"
            text = shown + (MODIFIED_MARK if self.store.is_modified(self.islat.molecules_dict, name) else "")
        try:
            self.button.config(text=text)
        except tk.TclError:
            pass

    def _on_parameter_change(self, *_args) -> None:
        # Parameter edits arrive in bursts (e.g. typing); refresh once they settle.
        if self._refresh_pending:
            return
        self._refresh_pending = True
        try:
            self.button.after_idle(self.refresh)
        except tk.TclError:
            self._refresh_pending = False

    def _rebuild(self) -> None:
        """Fill the menu as it opens, so the unsaved-change marks are current."""
        menu, store = self.menu, self.store
        menu.delete(0, "end")
        self._choice.set(store.active or "")
        for name in store.names:
            menu.add_radiobutton(label=self._label(name), variable=self._choice, value=name,
                                 command=lambda n=name: self.switch(n))
        if not store.names:
            menu.add_command(label="No parameter sets yet", state="disabled")
        menu.add_separator()
        many = "normal" if len(store.names) > 1 else "disabled"
        menu.add_command(label="Previous Set", accelerator=shortcut_text("previous_parameter_set"),
                         command=lambda: self.cycle(-1), state=many)
        menu.add_command(label="Next Set", accelerator=shortcut_text("next_parameter_set"),
                         command=lambda: self.cycle(1), state=many)
        menu.add_separator()
        has_active = "normal" if store.active else "disabled"
        menu.add_command(label="New Set from Current Parameters…", command=self.new_set)
        menu.add_command(label="Rename Set…", command=self.rename_set, state=has_active)
        menu.add_command(label="Close Set", command=self.close_set, state=has_active)
        self.refresh()

    def _announce(self, message: str) -> None:
        data_field = getattr(getattr(self.islat, "GUI", None), "data_field", None)
        if data_field is not None:
            data_field.insert_text(message, clear_after=True)

    # -- actions ---------------------------------------------------------
    def switch(self, name: str) -> None:
        if name != self.store.active:
            self.islat.switch_parameter_set(name)
            self._announce(f"Showing parameter set: {name}")

    def cycle(self, step: int) -> None:
        if len(self.store.names) > 1:
            name = self.islat.cycle_parameter_set(step)
            self._announce(f"Showing parameter set: {name}")

    def new_set(self) -> Optional[str]:
        name = simpledialog.askstring(
            "New Parameter Set", "Name of a new set holding the current parameters:",
            initialvalue=f"Set {len(self.store.names) + 1}", parent=self.button)
        if not name or not name.strip():
            return None
        name = self.store.new(self.islat.molecules_dict, name)
        self._announce(f"Made parameter set: {name}")
        return name

    def rename_set(self) -> Optional[str]:
        old = self.store.active
        if old is None:
            return None
        new = simpledialog.askstring("Rename Parameter Set", "New name:", initialvalue=old, parent=self.button)
        if not new or not new.strip():
            return None
        return self.store.rename(old, new)

    def close_set(self) -> None:
        name = self.store.active
        if name is None:
            return
        if self.store.is_modified(self.islat.molecules_dict, name) and messagebox.askquestion(
                "Close Parameter Set", f"Close {name} and discard its unsaved changes?",
                parent=self.button) == "no":
            return
        self.islat.close_parameter_set(name)
        shown = self.store.active
        self._announce(f"Closed parameter set: {name}" + (f"; showing {shown}" if shown else ""))
