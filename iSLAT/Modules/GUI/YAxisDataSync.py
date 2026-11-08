"""Shared Tk controls for syncing a plot's Y axis to the observed data's min / max.

Used by the main window (the ``Views`` menu and its *Set Y Limits* dialog)
and by the Full Spectrum window (its ``Y Axis`` menu and dialog), so both
offer the same scopes with the same labels:

* *Global data*: the min / max of the whole spectrum, on every panel.
* *Active view*: the min / max of the data in the visible range, shared by
  all panels of the view.
* *Each panel*: every panel scaled to the data in its own range.

In the Three Panel view the line inspection plot follows the setting too
(while a range is selected): it shares the main panel's range, or with
*Each panel* fits the data in the selected range.
"""
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

from iSLAT.Modules.Plotting.PlotView import normalize_y_data_sync
from .Tooltips import CreateToolTip, MenuToolTip

#: Label of the menu cascade holding the sync choices.
Y_DATA_SYNC_MENU_LABEL = "Sync Y to Data Min/Max"
#: Radio-button value for "not syncing".
Y_DATA_SYNC_OFF = "off"
#: ``(scope, menu label, dialog label)`` for every sync scope.
Y_DATA_SYNC_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    ("global", "Global Data", "Global data"),
    ("view", "Active View", "Active view"),
    ("panel", "Each Panel", "Each panel"),
)
#: Tooltip for the cascade entry in the parent menu.
Y_DATA_SYNC_MENU_TIP = (
    "Auto-scale the Y axis to the exact\n"
    "min / max of the observed data\n"
    "(no extra headroom), over the whole\n"
    "spectrum, the visible range, or\n"
    "each panel's own range.\n"
    "Choosing a range turns auto-scale on."
)
_SUBMENU_TIPS: Dict[str, str] = {
    "Off": "Auto-scale with the normal rule.",
    "Global Data": "Min / max of the whole spectrum,\non every panel and the line\ninspection plot.",
    "Active View": "Min / max of the data in the visible\nrange, shared by all panels and\nthe line inspection plot.",
    "Each Panel": "Each panel, and the line inspection\nplot, scaled to the min / max of\nthe data in its own range.",
}


def y_data_sync_var_value(scope: Any) -> str:
    """The radio-button value for *scope* (``"off"`` when not syncing)."""
    return normalize_y_data_sync(scope) or Y_DATA_SYNC_OFF


def add_y_data_sync_menu(
    menu: tk.Menu,
    variable: tk.StringVar,
    command: Callable[[Optional[str]], None],
    **menu_options: Any,
) -> tk.Menu:
    """Add the *Sync Y to Data Min/Max* cascade to *menu* and return its submenu.

    The submenu has radio buttons *Off*, *Global Data*, *Active View* and
    *Each Panel* bound to *variable* (which holds the scope, or ``"off"``).
    *command* is called with the chosen scope, or ``None`` for *Off*.
    *menu_options* (colors) are passed on to the submenu.
    """
    submenu = tk.Menu(menu, tearoff=0, **menu_options)
    submenu.add_radiobutton(label="Off", value=Y_DATA_SYNC_OFF, variable=variable,
                            command=lambda: command(None))
    for scope, label, _dialog_label in Y_DATA_SYNC_CHOICES:
        submenu.add_radiobutton(label=label, value=scope, variable=variable,
                                command=lambda s=scope: command(s))
    menu.add_cascade(label=Y_DATA_SYNC_MENU_LABEL, menu=submenu)
    MenuToolTip(submenu, _SUBMENU_TIPS)
    return submenu


def add_y_data_range_controls(
    parent: Any,
    row: int,
    *,
    current_scope: Optional[str],
    panel_count: int,
    get_range: Callable[[str], Optional[Tuple[float, float]]],
    on_fill: Callable[[float, float], None],
    on_sync: Callable[[str], None],
    set_status: Callable[[str], None],
    columnspan: int = 2,
    padx: Any = 8,
    pady: Any = 4,
) -> ttk.LabelFrame:
    """Grid a *Data min/max* section into *parent* at *row* and return its frame.

    It holds a range choice (*Global data*, *Active view* and, when the view
    has more than one panel, *Each panel*) and two buttons:

    * *Fill Values* calls *on_fill(min, max)* with ``get_range(scope)``, so
      the dialog's Y min / Y max entries can be filled (and then applied
      or edited).  Disabled for *Each panel*, which has no single range;
      *set_status* reports a range without data.
    * *Keep Synced* calls *on_sync(scope)* to keep the axis synced to the
      data's min / max as the view changes.
    """
    choices = [c for c in Y_DATA_SYNC_CHOICES if c[0] != "panel" or panel_count > 1]
    scope_by_label = {dialog_label: scope for scope, _menu_label, dialog_label in choices}
    scope = normalize_y_data_sync(current_scope) or "global"
    if scope not in scope_by_label.values():
        scope = "view"  # "Each panel" on a single-panel view is the active view
    initial = next(dialog_label for s, _m, dialog_label in choices if s == scope)

    frame = ttk.LabelFrame(parent, text="Data min/max")
    frame.grid(row=row, column=0, columnspan=columnspan, sticky="ew", padx=padx, pady=pady)

    ttk.Label(frame, text="Range:").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=(4, 2))
    scope_var = tk.StringVar(master=frame, value=initial)
    combo = ttk.Combobox(frame, textvariable=scope_var, values=list(scope_by_label),
                         state="readonly", width=14)
    combo.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(4, 2))

    def _selected() -> str:
        return scope_by_label.get(scope_var.get(), "global")

    def _fill() -> None:
        lims = get_range(_selected())
        if lims is None:
            set_status("No observed data in that range.")
            return
        set_status("")
        on_fill(float(lims[0]), float(lims[1]))

    def _sync() -> None:
        on_sync(_selected())

    buttons = ttk.Frame(frame)
    buttons.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
    fill_btn = ttk.Button(buttons, text="Fill Values", command=_fill)
    fill_btn.pack(side="left")
    sync_btn = ttk.Button(buttons, text="Keep Synced", command=_sync)
    sync_btn.pack(side="left", padx=(4, 0))
    CreateToolTip(fill_btn, "Fill Y min / Y max with the data's\nmin / max for this range\n(Apply then pins them).")
    CreateToolTip(sync_btn, "Keep the Y axis synced to the data's\nmin / max for this range as the\nview changes (turns auto-scale on).")

    def _update_fill_state(_event: Any = None) -> None:
        fill_btn.state(["disabled"] if _selected() == "panel" else ["!disabled"])

    combo.bind("<<ComboboxSelected>>", _update_fill_state)
    _update_fill_state()
    return frame
