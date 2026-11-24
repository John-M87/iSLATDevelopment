"""
KeyBindings -- declarative keyboard shortcuts for the iSLAT GUI.

Keyboard shortcuts are described as *chords*: canonical strings of the form
``"[Ctrl+][Shift+]KEY"`` such as ``"F"``, ``"Shift+M"``, ``"Ctrl+Shift+S"``,
``"Up"`` or ``","``.  ``Ctrl`` is the platform's primary accelerator: Control
on Windows / Linux and Command (or Control) on macOS.

:data:`DEFAULT_KEYMAP` is the single source of truth mapping *action IDs* to
their default chords.  The GUI binds callbacks to action IDs through an
:class:`ActionRegistry`, and UI text (menu labels, tooltips) is generated with
:func:`shortcut_text` so it always matches the real bindings.

:data:`ACTION_DESCRIPTIONS` says what each action does; the menu tooltips and
the command palette both show that text.

:class:`CommandPalette` (Ctrl/Cmd+K) lists every action with its shortcut and
the selected action's description: type to filter, Up / Down to choose, Enter
(or a double-click) to run.  Actions recently run from the palette are listed
first.
Everything else here works without a Tk window, so it can be unit-tested headlessly.
"""

from __future__ import annotations

import platform
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ACTION_DESCRIPTIONS",
    "DEFAULT_KEYMAP",
    "TEXT_INPUT_CLASSES",
    "Action",
    "ActionRegistry",
    "CommandPalette",
    "action_description",
    "action_title",
    "display_title",
    "action_tooltip",
    "filter_actions",
    "chord_from_event",
    "format_chord",
    "is_mac",
    "normalize_chord",
    "one_line",
    "shortcut_text",
    "with_shortcut",
]

# Tk ``event.state`` modifier bits.
_SHIFT_MASK = 0x1
_CONTROL_MASK = 0x4
_MAC_COMMAND_MASK = 0x8

_MODIFIER_ORDER = ("Ctrl", "Shift")
_MODIFIER_ALIASES = {
    "ctrl": "Ctrl", "control": "Ctrl", "cmd": "Ctrl", "command": "Ctrl",
    "shift": "Shift",
}

# Tk keysyms (lower-cased) -> canonical key names.  The shifted punctuation
# variants collapse onto their base key so Shift is carried only by the
# modifier (Tk reports Shift+',' as 'less' on most platforms).
_KEYSYM_ALIASES = {
    "comma": ",", "less": ",", "<": ",",
    "period": ".", "greater": ".", ">": ".",
    "bracketleft": "[", "braceleft": "[", "{": "[",
    "bracketright": "]", "braceright": "]", "}": "]",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "escape": "Escape", "return": "Return", "tab": "Tab", "space": "Space",
    "delete": "Delete", "backspace": "BackSpace",
}

_KEY_DISPLAY = {",": "<", ".": ">"}

# Widget classes that receive typed text; global shortcuts must not fire there.
TEXT_INPUT_CLASSES = frozenset({"Entry", "Text", "TEntry", "TCombobox"})

# Action ID -> default chords.  Order of actions is the order they are listed
# in generated help; the first chord of each action is its "primary" one.
DEFAULT_KEYMAP: Dict[str, Tuple[str, ...]] = {
    "toggle_full_spectrum": ("F",),
    "open_full_spectrum_window": ("Ctrl+F",),
    "output_full_spectrum": ("Ctrl+Shift+F",),
    "toggle_three_panel": ("T",),
    "toggle_residuals": ("R",),
    "toggle_summed_spectrum": ("M",),
    "open_wavelength_marker_dialog": ("Shift+M",),
    "toggle_saved_lines": ("S",),
    "toggle_atomic_lines": ("A",),
    "toggle_legend": ("L",),
    "toggle_grid": ("H",),
    "toggle_active_molecule_visibility": ("V",),
    "toggle_all_molecule_visibility": ("Shift+V",),
    "toggle_active_molecule_bold": ("B",),
    "open_center_view_dialog": ("C",),
    "save_parameters": ("Ctrl+S",),
    "save_parameters_to_file": ("Ctrl+Shift+S",),
    "load_parameters": ("Ctrl+L",),
    "load_parameters_from_file": ("Ctrl+Shift+L",),
    "revert_parameters": (),  # no default key: it discards edits
    "previous_parameter_set": ("Ctrl+[",),
    "next_parameter_set": ("Ctrl+]",),
    "new_parameter_set": (),
    "select_previous_molecule": ("Up",),
    "select_next_molecule": ("Down",),
    "cycle_spectrum_previous": ("Left",),
    "cycle_spectrum_next": ("Right",),
    "retreat_plot_start": (",",),
    "retreat_plot_start_half": ("Shift+,",),
    "advance_plot_start": (".",),
    "advance_plot_start_half": ("Shift+.",),
    "load_line_list": ("Ctrl+O",),
    "set_output_file": ("Ctrl+Shift+O",),
    "toggle_match_pixel_sampling": ("P",),
    "add_molecule": ("Ctrl+N",),
    "filter_active_line_list": ("Shift+F",),
    "clear_selection": ("Escape",),
    "save_line": ("Shift+S",),
    "fit_line": ("G",),
    "single_slab_fit": ("Shift+G",),
    "halve_plot_range": ("Z",),
    "double_plot_range": ("Shift+Z",),
    "load_spectrum": ("Ctrl+D",),
    "open_sample_manager": ("Ctrl+Shift+D",),  # not Cmd+Shift+M: macOS's "Open man Page in Terminal"
    "copy_view_to_clipboard": ("Ctrl+Shift+C",),  # not Ctrl+C: keep the platform's text copy
    "export_models": ("Ctrl+E",),
    "toggle_observed_data": ("O",),
    "toggle_y_autoscale": ("Y",),
    "open_y_limits_dialog": ("Shift+Y",),
    "next_line_list_line": ("N",),
    "previous_line_list_line": ("Shift+N",),
    "open_command_palette": ("Ctrl+K",),
}

# Action ID -> what the action does.  This is the text of the menu / button
# tooltips (built with :func:`action_tooltip`) and of the command palette's
# description line.  Line breaks are placed for the tooltips; the palette
# joins the lines.
ACTION_DESCRIPTIONS: Dict[str, str] = {
    "toggle_full_spectrum": "Toggle full spectrum view on/off",
    "open_full_spectrum_window": "Open the full combined model spectrum\nin a separate window.",
    "output_full_spectrum": (
        "Save the full combined model spectrum\nto a file, matching the current overlay,\n"
        "observed-data and Y-axis settings."
    ),
    "toggle_three_panel": (
        "Switch to the 3-panel view (full spectrum,\nline inspection and population diagram),\n"
        "or back to the previous view."
    ),
    "toggle_residuals": "Toggle residual sub-panels on/off\nin full spectrum mode",
    "toggle_summed_spectrum": "Toggle summed model flux on/off\n(gray fill in plot)",
    "open_wavelength_marker_dialog": (
        "Mark a wavelength (line) or a range\n(shaded band) on the spectrum plots,\n"
        "with an optional label (left / center /\nright of the mark), line and label\n"
        "colors (color pickers) and line style.\nMarks are included in full-spectrum output."
    ),
    "toggle_saved_lines": "Show saved lines\nfrom the 'Input Line List'",
    "toggle_atomic_lines": "Show atomic lines\nusing separation threshold\nset in the 'Line Separ.' field",
    "toggle_legend": "Turn legend on/off",
    "toggle_grid": "Show / hide grid lines\non the plots.",
    "toggle_active_molecule_visibility": "Show / hide the active molecule's\nmodel on the plots.",
    "toggle_all_molecule_visibility": (
        "Hide every molecule's model if any\nis visible, otherwise show them all."
    ),
    "toggle_active_molecule_bold": (
        "Draw the active molecule's model\nwith a heavier line, or back to normal."
    ),
    "clear_selection": (
        "Clear the selected wavelength range\nand the line inspection it shows."
    ),
    "open_center_view_dialog": (
        "Reposition the plot window: center + range,\ncenter + number of data points,\n"
        "or an exact start / end.\nOptionally mark the center wavelength."
    ),
    "save_parameters": "Save current molecule parameters\nto the default save file\nfor this spectrum.",
    "save_parameters_to_file": "Save current molecule parameters\nto a user-selected CSV file.",
    "load_parameters": "Load molecule parameters from\nthe default save file\nfor this spectrum.",
    "load_parameters_from_file": (
        "Load molecule parameters from\na user-selected CSV save file,\nreplacing existing molecules."
    ),
    "revert_parameters": "Discard the unsaved changes to the\nactive parameter set (back to its\nlast saved version).",
    "previous_parameter_set": "Show the previous parameter set\n(edits to the current one are kept).",
    "next_parameter_set": "Show the next parameter set\n(edits to the current one are kept).",
    "new_parameter_set": "Make a new parameter set holding\nthe current parameters.",
    "select_previous_molecule": "Make the molecule above the active one\nin the control panel active.",
    "select_next_molecule": "Make the molecule below the active one\nin the control panel active.",
    "cycle_spectrum_previous": "Switch to the previous spectrum\nin the loaded sample.",
    "cycle_spectrum_next": "Switch to the next spectrum\nin the loaded sample.",
    "retreat_plot_start": "Retreat the plot start\nby the current range value",
    "retreat_plot_start_half": "Retreat the plot start\nby half the current range value",
    "advance_plot_start": "Advance the plot start\nby the current range value",
    "advance_plot_start_half": "Advance the plot start\nby half the current range value",
    "load_line_list": "Choose the input line list file.",
    "set_output_file": "Choose the output line measurements\nfile that saved lines are written to.",
    "toggle_match_pixel_sampling": (
        "Toggle matched spectral sampling.\nWhen enabled, model flux is interpolated\n"
        "pixel-by-pixel to match the spectrum's\nwavelength grid"
    ),
    "add_molecule": "Add a molecule from the available\nHITRAN files.",
    "filter_active_line_list": "Open the line list filter\nfor the active molecule.",
    "save_line": "Save the currently selected line\nto the output line measurements file.",
    "fit_line": "Fit a Gaussian to the currently\nselected spectral line.",
    "single_slab_fit": "Run a single-slab LTE model fit\nusing the saved line measurements.",
    "halve_plot_range": "Halve the plot range,\nkeeping the view centered.",
    "double_plot_range": "Double the plot range,\nkeeping the view centered.",
    "load_spectrum": "Choose a spectrum file to load\n(select several files to load a sample).",
    "open_sample_manager": "Open the sample manager to add,\nremove, and switch between spectra.",
    "copy_view_to_clipboard": "Copy the current plot view\nto the clipboard as an image.",
    "export_models": "Export current model spectra\nto CSV files for external use.",
    "toggle_observed_data": "Show / hide the observed spectrum\non the main plot.",
    "toggle_y_autoscale": (
        "ON: the main plot's Y axis follows\nthe data in view.\n"
        "OFF: it stays at the current / set limits."
    ),
    "open_y_limits_dialog": "Pin the main plot's Y axis to\nspecific values (turns auto-scale off).",
    "next_line_list_line": "Select the next line of the input\nline list (toward longer wavelengths).",
    "previous_line_list_line": (
        "Select the previous line of the input\nline list (toward shorter wavelengths)."
    ),
    "open_command_palette": "Search for any command and run it.",
}


def is_mac(system: Optional[str] = None) -> bool:
    """Return ``True`` when *system* (default: the running OS) is macOS."""
    return (system if system is not None else platform.system()) == "Darwin"


def _canonical_key(key: str) -> str:
    alias = _KEYSYM_ALIASES.get(key.lower())
    if alias is not None:
        return alias
    if len(key) == 1:
        return key.upper()
    return key[:1].upper() + key[1:]


def _build_chord(modifiers: Iterable[str], key: str) -> str:
    mods = set(modifiers)
    return "+".join([m for m in _MODIFIER_ORDER if m in mods] + [key])


def normalize_chord(text: str) -> str:
    """Return the canonical form of a chord string.

    >>> normalize_chord("ctrl+shift+f")
    'Ctrl+Shift+F'
    >>> normalize_chord("Shift+<")
    'Shift+,'
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Invalid key chord: {text!r}")
    *parts, key = text.strip().split("+")
    modifiers = []
    for part in parts:
        mod = _MODIFIER_ALIASES.get(part.strip().lower())
        if mod is None:
            raise ValueError(f"Unknown modifier {part!r} in key chord {text!r}")
        modifiers.append(mod)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing key in key chord: {text!r}")
    return _build_chord(modifiers, _canonical_key(key))


def chord_from_event(event: Any, system: Optional[str] = None) -> Optional[str]:
    """Build the canonical chord for a Tk ``<KeyPress>`` *event*.

    Returns ``None`` for events without a usable keysym (e.g. bare modifier
    presses).  Letter case comes from the Shift bit only, so Caps Lock does
    not change the chord.
    """
    keysym = getattr(event, "keysym", None)
    if not keysym or keysym == "??":
        return None
    lowered = keysym.lower()
    if lowered.startswith(("shift_", "control_", "alt_", "meta_", "super_")) or lowered in (
        "caps_lock", "num_lock",
    ):
        return None

    state = int(getattr(event, "state", 0) or 0)
    accel_mask = _CONTROL_MASK | (_MAC_COMMAND_MASK if is_mac(system) else 0)
    modifiers = []
    if state & accel_mask:
        modifiers.append("Ctrl")
    if state & _SHIFT_MASK:
        modifiers.append("Shift")
    return _build_chord(modifiers, _canonical_key(keysym))


def format_chord(chord: str, system: Optional[str] = None) -> str:
    """Return user-facing text for *chord* (``Cmd`` instead of ``Ctrl`` on macOS)."""
    *mods, key = normalize_chord(chord).split("+")
    accel = "Cmd" if is_mac(system) else "Ctrl"
    shown = [accel if m == "Ctrl" else m for m in mods]
    return "+".join(shown + [_KEY_DISPLAY.get(key, key)])


def shortcut_text(action_id: str, system: Optional[str] = None) -> str:
    """Return the display text of *action_id*'s primary default chord."""
    chords = DEFAULT_KEYMAP.get(action_id)
    if not chords:
        raise KeyError(f"No default key binding for action {action_id!r}")
    return format_chord(chords[0], system)


def with_shortcut(label: str, action_id: str, system: Optional[str] = None) -> str:
    """Menu label *label* followed by *action_id*'s keybind: ``"Fit Line (G)"``.

    *label* is returned unchanged when the action has no default keybind.
    """
    if not DEFAULT_KEYMAP.get(action_id):
        return label
    return f"{label} ({shortcut_text(action_id, system)})"


def action_title(action_id: str) -> str:
    """Readable name of an action ID: ``"toggle_full_spectrum"`` -> ``"Toggle full spectrum"``."""
    return action_id.replace("_", " ").strip().capitalize()


def action_description(action_id: str) -> str:
    """What *action_id* does (tooltip line breaks kept), or ``""`` if not described."""
    return ACTION_DESCRIPTIONS.get(action_id, "")


def action_tooltip(action_id: str, system: Optional[str] = None) -> str:
    """Tooltip text for *action_id*: its description, then its keybind."""
    return f"{action_description(action_id)}\nKeybind: {shortcut_text(action_id, system)}"


def one_line(text: str) -> str:
    """*text* with its line breaks and repeated spaces collapsed to single spaces."""
    return " ".join(text.split())


def display_title(action: "Action") -> str:
    """The name the command palette shows for *action*."""
    return action.title or action_title(action.id)


def filter_actions(
    actions: Iterable["Action"],
    query: str,
    system: Optional[str] = None,
    recent: Sequence[str] = (),
) -> List["Action"]:
    """Actions whose title, ID or shortcut contain every word of *query* (any case).

    When nothing matches that way, actions whose description contains every
    word are returned instead (so "gaussian" finds "Fit line").

    Titles that start with the query come first (with an empty query, every
    action does); within each group the action IDs in *recent* (most recent
    first) lead, and otherwise the given order is kept.
    """
    actions = list(actions)
    words = query.lower().split()
    query = query.strip().lower()

    def text(action: "Action") -> str:
        keys = " ".join(format_chord(c, system) for c in action.chords)
        return f"{display_title(action)} {action.id} {keys}".lower()

    matches = [a for a in actions if all(word in text(a) for word in words)]
    if not matches and words:
        matches = [a for a in actions if all(word in one_line(a.description).lower() for word in words)]

    rank = {action_id: i for i, action_id in enumerate(recent)}
    return sorted(matches, key=lambda a: (not display_title(a).lower().startswith(query),
                                          rank.get(a.id, len(rank))))


@dataclass
class Action:
    """A named, keyboard-invokable command."""

    id: str
    callback: Callable[[], Any]
    chords: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    title: str = ""  # shown instead of action_title(id) when set

    def __post_init__(self) -> None:
        self.chords = tuple(normalize_chord(c) for c in self.chords)


class ActionRegistry:
    """Maps chords to :class:`Action` objects and dispatches key presses.

    It also remembers the actions most recently run from the command palette
    (:meth:`note_used` / :meth:`recent`), for as long as the registry lives.
    """

    #: How many recently used actions are remembered.
    RECENT_LIMIT = 5

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}
        self._by_chord: Dict[str, str] = {}
        self._recent: List[str] = []

    def register(self, action: Action) -> None:
        """Add *action*; raises ``ValueError`` on a duplicate ID or chord conflict."""
        if action.id in self._actions:
            raise ValueError(f"Action {action.id!r} is already registered")
        for chord in action.chords:
            owner = self._by_chord.get(chord)
            if owner is not None:
                raise ValueError(
                    f"Key chord {chord!r} for {action.id!r} is already bound to {owner!r}"
                )
        self._actions[action.id] = action
        for chord in action.chords:
            self._by_chord[chord] = action.id

    def register_defaults(
        self,
        callbacks: Dict[str, Callable[[], Any]],
        keymap: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> None:
        """Register one action per entry of *keymap* (default :data:`DEFAULT_KEYMAP`).

        Every action ID in the keymap must have a callback in *callbacks*.
        Descriptions come from :data:`ACTION_DESCRIPTIONS`.
        """
        keymap = DEFAULT_KEYMAP if keymap is None else keymap
        missing = [a for a in keymap if a not in callbacks]
        if missing:
            raise KeyError(f"No callback for action(s): {', '.join(missing)}")
        for action_id, chords in keymap.items():
            self.register(Action(action_id, callbacks[action_id], tuple(chords),
                                 action_description(action_id)))

    def unregister(self, action_id: str) -> bool:
        """Remove *action_id*; returns ``True`` if it was registered."""
        action = self._actions.pop(action_id, None)
        if action is None:
            return False
        for chord in action.chords:
            self._by_chord.pop(chord, None)
        if action_id in self._recent:
            self._recent.remove(action_id)
        return True

    def note_used(self, action_id: str) -> None:
        """Record that *action_id* was just run (it becomes the most recent)."""
        if action_id not in self._actions:
            return
        if action_id in self._recent:
            self._recent.remove(action_id)
        self._recent.insert(0, action_id)
        del self._recent[self.RECENT_LIMIT:]

    def recent(self) -> List[str]:
        """IDs of the recently used actions, most recent first."""
        return list(self._recent)

    def get(self, action_id: str) -> Optional[Action]:
        return self._actions.get(action_id)

    def actions(self) -> List[Action]:
        return list(self._actions.values())

    def chords_for(self, action_id: str) -> Tuple[str, ...]:
        action = self._actions.get(action_id)
        return action.chords if action is not None else ()

    def action_for_chord(self, chord: Optional[str]) -> Optional[Action]:
        if not chord:
            return None
        action_id = self._by_chord.get(chord)
        return self._actions.get(action_id) if action_id is not None else None

    def dispatch(self, chord: Optional[str]) -> bool:
        """Run the action bound to *chord*; returns ``True`` if one was found."""
        action = self.action_for_chord(chord)
        if action is None:
            return False
        action.callback()
        return True


class CommandPalette(tk.Toplevel):
    """Searchable list of the actions in an :class:`ActionRegistry`.

    Below the list, the selected action's description is shown.  Actions run
    from the palette are recorded in the registry and listed first next time.
    """

    #: Actions not listed (opening the palette from itself does nothing useful).
    EXCLUDED = ("open_command_palette",)
    #: Text lines reserved for the description, so the window keeps its size.
    DESCRIPTION_LINES = 4
    RECENT_NOTE = "Recently used."

    def __init__(self, parent: tk.Misc, registry: ActionRegistry,
                 extra_actions: Iterable[Action] = ()) -> None:
        super().__init__(parent)
        self.title("Commands")
        self.transient(parent)
        self.resizable(False, False)
        self._registry = registry
        # Registered actions, then *extra_actions* (made for this opening, e.g. one per parameter set)
        self._actions: List[Action] = [a for a in registry.actions() if a.id not in self.EXCLUDED]
        self._actions += list(extra_actions)
        self._shown: List[Action] = []
        self._closed = False

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill="both", expand=True)
        self.query = tk.StringVar(master=self)
        self.entry = ttk.Entry(frame, textvariable=self.query, width=48)
        self.entry.pack(fill="x")
        self.tree = ttk.Treeview(frame, columns=("keys",), show="tree", height=12,
                                 selectmode="browse", takefocus=False)
        self.tree.column("#0", width=320, stretch=True)
        self.tree.column("keys", width=110, anchor="e", stretch=False)
        self.tree.pack(fill="both", expand=True, pady=(6, 0))

        # Fixed-height description area: a frame that does not shrink to its
        # label, so moving between short and long descriptions never resizes
        # the window.
        self.description = tk.StringVar(master=self)
        linespace = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        description_frame = ttk.Frame(frame, height=self.DESCRIPTION_LINES * linespace + 8)
        description_frame.pack_propagate(False)
        description_frame.pack(fill="x", pady=(6, 0))
        self.description_label = ttk.Label(description_frame, textvariable=self.description,
                                           justify="left", anchor="nw")
        self.description_label.pack(fill="both", expand=True)
        description_frame.bind(
            "<Configure>", lambda e: self.description_label.configure(wraplength=max(e.width - 4, 1)))

        self.query.trace_add("write", lambda *_: self._refresh())
        self.entry.bind("<Down>", lambda _e: self._move(1))
        self.entry.bind("<Up>", lambda _e: self._move(-1))
        self.entry.bind("<Return>", lambda _e: self.run_selected())
        self.tree.bind("<Double-Button-1>", lambda _e: self.run_selected())
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_description())
        # Keep the keyboard in the entry: arrow and letter keys on the list
        # would otherwise also reach iSLAT's global shortcuts.
        self.tree.bind("<ButtonRelease-1>", lambda _e: self.entry.focus_set())
        self.bind("<Escape>", lambda _e: self.close())
        self.bind("<FocusOut>", lambda _e: self.after(100, self._close_if_unfocused))
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._refresh()
        self._place(parent)
        self.entry.focus_set()

    # -- list ---------------------------------------------------------
    def _refresh(self) -> None:
        self._shown = filter_actions(self._actions, self.query.get(), recent=self._registry.recent())
        self.tree.delete(*self.tree.get_children())
        for index, action in enumerate(self._shown):
            keys = format_chord(action.chords[0]) if action.chords else ""
            self.tree.insert("", "end", iid=str(index), text=display_title(action), values=(keys,))
        if self._shown:
            self._select(0)
        else:
            self.description.set("")

    def _select(self, index: int) -> None:
        iid = str(index)
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self._show_description()

    def _show_description(self) -> None:
        """Show the selected action's description, noting if it was recently used."""
        action = self.selected_action()
        if action is None:
            self.description.set("")
            return
        text = one_line(action.description)
        if action.id in self._registry.recent():
            text = f"{self.RECENT_NOTE} {text}".strip()
        self.description.set(text)

    def _move(self, step: int) -> str:
        if self._shown:
            current = self.tree.selection()
            index = int(current[0]) if current else 0
            self._select(max(0, min(len(self._shown) - 1, index + step)))
        return "break"

    def selected_action(self) -> Any:
        current = self.tree.selection()
        return self._shown[int(current[0])] if current else None

    # -- running / closing --------------------------------------------
    def run_selected(self) -> str:
        """Close the palette, then run the selected action (after the palette is gone).

        The action is recorded as recently used, so it is listed first next time.
        """
        action = self.selected_action()
        parent = self.master
        self.close()
        if action is not None:
            self._registry.note_used(action.id)
            parent.after_idle(action.callback)
        return "break"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.destroy()
        except tk.TclError:
            pass

    def _close_if_unfocused(self) -> None:
        if self._closed:
            return
        try:
            focused = self.focus_get()
        except (KeyError, tk.TclError):
            focused = None
        if focused is None or focused.winfo_toplevel() is not self:
            self.close()

    def _place(self, parent: tk.Misc) -> None:
        """Centre horizontally over *parent*, near its top (like editors' command palettes)."""
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_reqwidth()) // 2
            y = parent.winfo_rooty() + 80
        except tk.TclError:
            return
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
