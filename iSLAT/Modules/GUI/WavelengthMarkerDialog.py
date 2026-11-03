"""Mark Wavelengths dialog (Shift+M).

Works with any *marker store*: an object exposing ``wavelength_markers``,
``add_wavelength_marker(...)``, ``remove_wavelength_marker(index)`` and
``clear_wavelength_markers()`` - see
:class:`~iSLAT.Modules.Plotting.PlotView.WavelengthMarkerStoreMixin`.  The
main plot controller and the Display Full Spectrum window each provide one.
"""
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

from matplotlib.colors import to_hex

from iSLAT.Modules.Plotting.PlotView import DEFAULT_WAVELENGTH_MARKER_COLOR
from .GUIFunctions import ColorButton

#: (value, display name) pairs for the dialog's label-position and line-style choices.
LABEL_SIDE_CHOICES = (("left", "Left"), ("center", "Center"), ("right", "Right"))
LINESTYLE_CHOICES = (
    ("dashed", "Dashed"), ("solid", "Solid"), ("dotted", "Dotted"), ("dashdot", "Dash-dot"),
)


def describe_wavelength_marker(marker: Dict[str, Any]) -> str:
    """One-line summary of a marker for the dialog list."""
    start, end = marker["start"], marker["end"]
    where = f"{start:.5f} \u03bcm" if end == start else f"{start:.5f} \u2013 {end:.5f} \u03bcm"
    color = marker.get('color', '')
    label = marker.get("label")
    text = where
    if label:
        details = [label, marker.get('label_side', 'center')]
        label_color = marker.get('label_color')
        if label_color and label_color != color:
            details.append(label_color)
        text = f"{where}   [{', '.join(details)}]"
    return f"{text}   ({color}, {marker.get('linestyle', 'dashed')})"


def ask_marker_color(parent: Any, title: str, initial: str) -> Optional[str]:
    """Open the system color picker (as for molecule colors); return a hex color or ``None``."""
    from tkinter import colorchooser
    return colorchooser.askcolor(title=title, color=initial, parent=parent)[1]


def open_wavelength_marker_dialog(
    parent: Any,
    store: Any,
    default_window: Tuple[str, float, float],
    get_selection: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
    on_close: Optional[Callable[[], None]] = None,
    ask_color: Callable[[Any, str, str], Optional[str]] = ask_marker_color,
) -> tk.Toplevel:
    """Open the non-modal Mark Wavelengths dialog for *store* and return its window.

    Adds a vertical line (single wavelength) or a shaded band (range), with
    an optional label, label position (left / center / right of the marker),
    line color, independent label color (color pickers, as for molecule
    colors) and line style.  Lists existing marks so they can be removed.

    *default_window* is ``(mode, lo, hi)`` with mode ``"single"`` or
    ``"range"``; *get_selection* feeds the "Use Selection" button and
    *on_close* runs after the dialog is closed.
    """
    mode, lo, hi = default_window

    win = tk.Toplevel(parent)
    win.title("Mark Wavelengths")
    win.resizable(False, False)
    win.transient(parent)

    content = ttk.Frame(win)
    content.pack(fill="both", expand=True, padx=12, pady=12)

    mode_var = tk.StringVar(master=win, value=mode)
    wave_var = tk.StringVar(master=win, value=f"{(lo + hi) / 2.0:.5f}")
    start_var = tk.StringVar(master=win, value=f"{lo:.5f}")
    end_var = tk.StringVar(master=win, value=f"{hi:.5f}")
    label_var = tk.StringVar(master=win, value="")
    line_color_var = tk.StringVar(master=win, value=to_hex(DEFAULT_WAVELENGTH_MARKER_COLOR))
    label_color_var = tk.StringVar(master=win, value=line_color_var.get())
    label_same_var = tk.BooleanVar(master=win, value=True)
    label_side_var = tk.StringVar(master=win, value="center")
    linestyle_names = {value: name for value, name in LINESTYLE_CHOICES}
    linestyle_values = {name: value for value, name in LINESTYLE_CHOICES}
    linestyle_var = tk.StringVar(master=win, value=linestyle_names["dashed"])
    error_var = tk.StringVar(master=win, value="")

    row = 0
    ttk.Radiobutton(content, text="Wavelength (\u03bcm):", variable=mode_var,
                    value="single").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
    wave_entry = ttk.Entry(content, textvariable=wave_var, width=12)
    wave_entry.grid(row=row, column=1, sticky="w", pady=(0, 6))

    row += 1
    ttk.Radiobutton(content, text="Range (\u03bcm):", variable=mode_var,
                    value="range").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
    range_frame = ttk.Frame(content)
    range_frame.grid(row=row, column=1, sticky="w", pady=(0, 6))
    start_entry = ttk.Entry(range_frame, textvariable=start_var, width=10)
    start_entry.pack(side="left")
    ttk.Label(range_frame, text=" to ").pack(side="left")
    end_entry = ttk.Entry(range_frame, textvariable=end_var, width=10)
    end_entry.pack(side="left")

    def _use_selection():
        sel = get_selection() if get_selection is not None else None
        if sel is None or sel[1] <= sel[0]:
            error_var.set("No span selection on the plot.")
            return
        start_var.set(f"{sel[0]:.5f}")
        end_var.set(f"{sel[1]:.5f}")
        mode_var.set("range")
        error_var.set("")

    ttk.Button(range_frame, text="Use Selection", command=_use_selection).pack(
        side="left", padx=(6, 0)
    )

    row += 1
    ttk.Label(content, text="Label (optional):").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
    )
    label_entry = ttk.Entry(content, textvariable=label_var, width=24)
    label_entry.grid(row=row, column=1, sticky="ew", pady=(0, 6))

    row += 1
    ttk.Label(content, text="Label position:").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
    )
    side_frame = ttk.Frame(content)
    side_frame.grid(row=row, column=1, sticky="w", pady=(0, 6))
    for value, name in LABEL_SIDE_CHOICES:
        ttk.Radiobutton(side_frame, text=name, variable=label_side_var,
                        value=value).pack(side="left", padx=(0, 6))

    row += 1
    ttk.Label(content, text="Line color:").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
    )
    line_swatch = ColorButton(content, color=line_color_var.get())
    line_swatch.grid(row=row, column=1, sticky="w", pady=(0, 6))

    row += 1
    ttk.Label(content, text="Label color:").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
    )
    label_color_frame = ttk.Frame(content)
    label_color_frame.grid(row=row, column=1, sticky="w", pady=(0, 6))
    label_swatch = ColorButton(label_color_frame, color=label_color_var.get())
    label_swatch.pack(side="left")

    def _set_swatch(swatch, var, color):
        var.set(color)
        swatch.change_color(color)
        swatch.config(bg=swatch.color)

    def _pick_line_color():
        chosen = ask_color(win, "Pick marker line color", line_color_var.get())
        if chosen:
            _set_swatch(line_swatch, line_color_var, chosen)
            if label_same_var.get():
                _set_swatch(label_swatch, label_color_var, chosen)

    def _pick_label_color():
        chosen = ask_color(win, "Pick marker label color", label_color_var.get())
        if chosen:
            label_same_var.set(False)
            _set_swatch(label_swatch, label_color_var, chosen)

    def _on_label_same_toggled():
        if label_same_var.get():
            _set_swatch(label_swatch, label_color_var, line_color_var.get())

    line_swatch.add_command(_pick_line_color)
    label_swatch.add_command(_pick_label_color)
    ttk.Checkbutton(label_color_frame, text="Same as line", variable=label_same_var,
                    command=_on_label_same_toggled).pack(side="left", padx=(8, 0))

    row += 1
    ttk.Label(content, text="Line style:").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
    )
    ttk.Combobox(content, textvariable=linestyle_var, state="readonly", width=12,
                 values=[name for _value, name in LINESTYLE_CHOICES],
                 ).grid(row=row, column=1, sticky="w", pady=(0, 6))

    row += 1
    ttk.Label(content, textvariable=error_var, foreground="red").grid(
        row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )

    def _sync_entry_states(*_):
        single = mode_var.get() == "single"
        wave_entry.configure(state="normal" if single else "disabled")
        for entry in (start_entry, end_entry):
            entry.configure(state="disabled" if single else "normal")

    mode_var.trace_add("write", _sync_entry_states)
    _sync_entry_states()

    row += 1
    ttk.Separator(content, orient="horizontal").grid(
        row=row, column=0, columnspan=2, sticky="ew", pady=(2, 8)
    )

    row += 1
    ttk.Label(content, text="Current marks:").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    list_frame = ttk.Frame(content)
    list_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(2, 6))
    listbox = tk.Listbox(list_frame, height=6, width=48, selectmode="extended",
                         exportselection=False)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _refresh_list():
        listbox.delete(0, "end")
        for marker in store.wavelength_markers:
            listbox.insert("end", describe_wavelength_marker(marker))

    def _add(event=None):
        try:
            if mode_var.get() == "single":
                start = end = float(wave_var.get())
            else:
                start, end = float(start_var.get()), float(end_var.get())
                if start == end:
                    error_var.set("Range start and end must differ.")
                    return
        except ValueError:
            error_var.set("Invalid wavelength.")
            return
        try:
            store.add_wavelength_marker(
                start, end, label=label_var.get(), color=line_color_var.get(),
                label_color=label_color_var.get(),
                label_side=label_side_var.get(),
                linestyle=linestyle_values.get(linestyle_var.get(), linestyle_var.get()),
            )
        except ValueError as exc:
            error_var.set(str(exc))
            return
        error_var.set("")
        label_var.set("")
        _refresh_list()

    def _remove_selected():
        for index in sorted(listbox.curselection(), reverse=True):
            store.remove_wavelength_marker(index)
        _refresh_list()

    def _clear_all():
        store.clear_wavelength_markers()
        _refresh_list()

    def _close(event=None):
        win.destroy()
        if on_close is not None:
            on_close()

    row += 1
    btn_frame = ttk.Frame(content)
    btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
    ttk.Button(btn_frame, text="Add Mark", command=_add).pack(side="left")
    ttk.Button(btn_frame, text="Remove Selected", command=_remove_selected).pack(
        side="left", padx=(4, 0)
    )
    ttk.Button(btn_frame, text="Clear All", command=_clear_all).pack(side="left", padx=(4, 0))
    ttk.Button(btn_frame, text="Close", command=_close).pack(side="right")

    win.bind("<Return>", _add)
    win.bind("<Escape>", _close)
    win.protocol("WM_DELETE_WINDOW", _close)
    _refresh_list()

    win.update_idletasks()
    x = win.winfo_screenwidth() // 2 - win.winfo_width() // 2
    y = win.winfo_screenheight() // 2 - win.winfo_height() // 2
    win.geometry(f"+{x}+{y}")
    focus_entry = wave_entry if mode_var.get() == "single" else start_entry
    focus_entry.focus_set()
    focus_entry.select_range(0, "end")
    return win
