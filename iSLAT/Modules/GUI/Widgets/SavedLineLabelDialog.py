"""Saved Line Labels dialog: choose how saved lines are labelled on the plots.

Picks the column of the saved-line CSV used as each line's label (or the
default ``species line``), an optional prefix / suffix, and the views in which
labels are shown.  Applying stores the choice in the user settings and
redraws the saved lines.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Iterable, List, Optional

from iSLAT.Modules.Plotting.SavedLineLabels import (
    DEFAULT_HIDDEN_VIEWS,
    LABEL_VIEWS,
    get_saved_line_label_settings,
    saved_line_labels,
    set_labels_visible,
    set_saved_line_label_settings,
    store_in_user_settings,
    SavedLineLabelSettings,
)

#: Combobox entry meaning "no column override".
DEFAULT_CHOICE = "(default: species + line)"


def _read_line_list(islat: Any):
    import pandas as pd
    path = getattr(islat, "input_line_list", None)
    if not path:
        return pd.DataFrame()
    import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
    return ifh.read_line_saves(file_name=path)


def apply_saved_line_label_settings(
    islat: Any,
    column: Optional[str],
    prefix: str = "",
    suffix: str = "",
    hidden_views: Optional[Iterable[str]] = None,
) -> SavedLineLabelSettings:
    """Set the saved-line label, store it in the user settings and redraw the plots.

    *hidden_views* ``None`` keeps the current per-view visibility.
    """
    if column == DEFAULT_CHOICE:
        column = None
    settings = set_saved_line_label_settings(column, prefix, suffix, hidden_views)
    _save_and_redraw(islat, settings)
    return settings


def set_saved_line_label_view_visible(islat: Any, view: str, visible: bool) -> SavedLineLabelSettings:
    """Show or hide saved-line labels in one view, store it and redraw."""
    settings = set_labels_visible(view, visible)
    _save_and_redraw(islat, settings)
    return settings


def _save_and_redraw(islat: Any, settings: SavedLineLabelSettings) -> None:
    user_settings = getattr(islat, "user_settings", None)
    if isinstance(user_settings, dict):
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        store_in_user_settings(user_settings, settings)
        try:
            ifh.save_user_settings(user_settings)
        except Exception as exc:
            print(f"Could not save the saved-line label setting: {exc}")
    plot = getattr(getattr(islat, "GUI", None), "plot", None)
    if plot is not None and hasattr(plot, "refresh_saved_line_labels"):
        plot.refresh_saved_line_labels()


def open_saved_line_label_dialog(parent: Any, islat: Any) -> tk.Toplevel:
    """Open (or raise) the Saved Line Labels dialog and return its window."""
    existing = getattr(islat, "_saved_line_label_dialog", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_set()
            return existing
    except tk.TclError:
        pass

    line_list = _read_line_list(islat)
    columns: List[str] = [str(c) for c in getattr(line_list, "columns", [])]
    current = get_saved_line_label_settings()

    win = tk.Toplevel(parent)
    win.title("Saved Line Labels")
    win.resizable(False, False)
    win.transient(parent)
    islat._saved_line_label_dialog = win

    content = ttk.Frame(win)
    content.pack(fill="both", expand=True, padx=12, pady=12)

    column_var = tk.StringVar(master=win, value=current.column or DEFAULT_CHOICE)
    prefix_var = tk.StringVar(master=win, value=current.prefix)
    suffix_var = tk.StringVar(master=win, value=current.suffix)
    preview_var = tk.StringVar(master=win)
    note_var = tk.StringVar(master=win)
    view_vars = {view: tk.BooleanVar(master=win, value=view not in current.hidden_views)
                 for view in LABEL_VIEWS}

    rows = (
        ("Label column:", ttk.Combobox(content, textvariable=column_var, width=26,
                                       values=[DEFAULT_CHOICE] + columns)),
        ("Prefix:", ttk.Entry(content, textvariable=prefix_var, width=28)),
        ("Suffix:", ttk.Entry(content, textvariable=suffix_var, width=28)),
    )
    for row, (text, widget) in enumerate(rows):
        ttk.Label(content, text=text).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        widget.grid(row=row, column=1, sticky="ew", pady=(0, 6))
    column_box = rows[0][1]

    ttk.Label(content, text="Show labels in:").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
    views_frame = ttk.Frame(content)
    views_frame.grid(row=3, column=1, sticky="w", pady=(0, 6))
    for view, var in view_vars.items():
        ttk.Checkbutton(views_frame, text=view, variable=var).pack(side="left", padx=(0, 6))

    ttk.Label(content, textvariable=preview_var).grid(
        row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
    ttk.Label(content, textvariable=note_var, foreground="gray").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=(0, 6))

    def _pending() -> SavedLineLabelSettings:
        column = column_var.get().strip()
        return SavedLineLabelSettings(
            None if column in ("", DEFAULT_CHOICE) else column,
            prefix_var.get(), suffix_var.get(),
            frozenset(view for view, var in view_vars.items() if not var.get()),
        )

    def _update_preview(*_):
        pending = _pending()
        if line_list is None or len(line_list) == 0:
            preview_var.set("Preview: (no saved lines loaded)")
        else:
            preview_var.set(f"Preview: {saved_line_labels(line_list.head(1), pending)[0]!r}")
        missing = pending.column is not None and columns and pending.column not in columns
        note_var.set("Column not in the current line list; the default label is used."
                     if missing else "")

    for var in (column_var, prefix_var, suffix_var):
        var.trace_add("write", _update_preview)
    _update_preview()

    def _apply(_event=None):
        pending = _pending()
        apply_saved_line_label_settings(islat, pending.column, pending.prefix, pending.suffix,
                                        pending.hidden_views)

    def _reset():
        column_var.set(DEFAULT_CHOICE)
        prefix_var.set("")
        suffix_var.set("")
        for view, var in view_vars.items():
            var.set(view not in DEFAULT_HIDDEN_VIEWS)

    def _close(_event=None):
        islat._saved_line_label_dialog = None
        win.destroy()

    buttons = ttk.Frame(content)
    buttons.grid(row=6, column=0, columnspan=2, sticky="ew")
    ttk.Button(buttons, text="Reset", command=_reset).pack(side="left")
    ttk.Button(buttons, text="Close", command=_close).pack(side="right")
    ttk.Button(buttons, text="Apply", command=_apply).pack(side="right", padx=(0, 4))

    win.bind("<Return>", _apply)
    win.bind("<Escape>", _close)
    win.protocol("WM_DELETE_WINDOW", _close)
    column_box.focus_set()
    return win
