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

This module has no Tk dependency so it can be unit-tested headlessly.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "DEFAULT_KEYMAP",
    "TEXT_INPUT_CLASSES",
    "Action",
    "ActionRegistry",
    "chord_from_event",
    "format_chord",
    "is_mac",
    "normalize_chord",
    "shortcut_text",
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
    "toggle_active_molecule_visibility": ("V",),
    "toggle_all_molecule_visibility": ("Shift+V",),
    "open_center_view_dialog": ("C",),
    "save_parameters": ("Ctrl+S",),
    "save_parameters_to_file": ("Ctrl+Shift+S",),
    "load_parameters": ("Ctrl+L",),
    "load_parameters_from_file": ("Ctrl+Shift+L",),
    "select_previous_molecule": ("Up",),
    "select_next_molecule": ("Down",),
    "cycle_spectrum_previous": ("Left",),
    "cycle_spectrum_next": ("Right",),
    "retreat_plot_start": (",",),
    "retreat_plot_start_half": ("Shift+,",),
    "advance_plot_start": (".",),
    "advance_plot_start_half": ("Shift+.",),
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


@dataclass
class Action:
    """A named, keyboard-invokable command."""

    id: str
    callback: Callable[[], Any]
    chords: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    def __post_init__(self) -> None:
        self.chords = tuple(normalize_chord(c) for c in self.chords)


class ActionRegistry:
    """Maps chords to :class:`Action` objects and dispatches key presses."""

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}
        self._by_chord: Dict[str, str] = {}

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
        """
        keymap = DEFAULT_KEYMAP if keymap is None else keymap
        missing = [a for a in keymap if a not in callbacks]
        if missing:
            raise KeyError(f"No callback for action(s): {', '.join(missing)}")
        for action_id, chords in keymap.items():
            self.register(Action(action_id, callbacks[action_id], tuple(chords)))

    def unregister(self, action_id: str) -> bool:
        """Remove *action_id*; returns ``True`` if it was registered."""
        action = self._actions.pop(action_id, None)
        if action is None:
            return False
        for chord in action.chords:
            self._by_chord.pop(chord, None)
        return True

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
