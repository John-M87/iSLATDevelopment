"""Explicit per-panel wavelength ranges for the stacked spectrum views.

The ranges are typed one ``start end`` pair per line (``DATASTART`` /
``DATAEND`` stand for the first / last observed data point).  The
:class:`PanelRangesDialog` edits them for any stacked view: the Views menu's
*Panel Ranges...* entry of the Full Spectrum and Optical Depth views (and of
any other :class:`FullSpectrumView` subclass) and the Display Full Spectrum
window's *Ranges...* button.  The ranges can be typed or dragged out on the
plot (*Select on Plot*).  The multi-page PDF export reads its page ranges
with the same :func:`parse_panel_ranges`.
"""
import re
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

#: Range keywords replaced by the first / last observed data point.
DATA_START_KEYWORD = "DATASTART"
DATA_END_KEYWORD = "DATAEND"
_DATA_KEYWORDS = rf"(?:{DATA_START_KEYWORD}|{DATA_END_KEYWORD})"


def parse_panel_ranges(
    text: str,
    data_start: Optional[float] = None,
    data_end: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Parse one ``start end`` wavelength pair per line (blank and ``#`` lines are skipped).

    The two numbers may be separated by spaces, a comma, ``-``, an en dash
    or ``to``.  Either bound may be ``DATASTART`` or ``DATAEND``
    (case-insensitive), replaced by *data_start* / *data_end* (the first /
    last observed data point).  Raises ``ValueError`` naming the offending line.
    """
    keyword_values = {DATA_START_KEYWORD: data_start, DATA_END_KEYWORD: data_end}
    ranges: List[Tuple[float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned = re.sub(r"\bto\b|[,;\u2013\u2014]", " ", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?<=[\d.])\s*-\s*(?=[\d.])", " ", cleaned)
        cleaned = re.sub(rf"\b({_DATA_KEYWORDS})\s*-\s*", r"\1 ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\s*-\s*({_DATA_KEYWORDS})\b", r" \1", cleaned, flags=re.IGNORECASE)
        parts = cleaned.split()
        if len(parts) != 2:
            raise ValueError(f"Line {lineno}: expected 'start end', got '{raw.strip()}'.")
        bounds = []
        for part in parts:
            keyword = part.upper()
            if keyword in keyword_values:
                value = keyword_values[keyword]
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Line {lineno}: {keyword} needs observed data to be loaded.")
                bounds.append(float(value))
                continue
            try:
                bounds.append(float(part))
            except ValueError:
                raise ValueError(f"Line {lineno}: expected 'start end', got '{raw.strip()}'.") from None
        lo, hi = bounds
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError(f"Line {lineno}: start must be less than end.")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("Enter at least one range.")
    return ranges


def format_panel_ranges(edges: Sequence[float], ends: Sequence[float]) -> str:
    """Inverse of :func:`parse_panel_ranges` for the current panels."""
    return "\n".join(f"{lo:.4f} {hi:.4f}" for lo, hi in zip(edges, ends))


#: Help text above the dialog's text box.
PANEL_RANGES_HELP = (
    "One panel per line: start end (\u03bcm).\n"
    "Panels are shown in this order; ranges may overlap or leave gaps.\n"
    f"{DATA_START_KEYWORD} / {DATA_END_KEYWORD} = first / last observed data point.\n"
    "A new panel count or wavelength range returns to an even split."
)
#: Help text added when the ranges can be dragged out on the plot.
PICK_HELP = "Select on Plot: drag across the panels to pick the ranges."
#: Shown while picking on the plot.
PICKING_HINT = "Drag across the panels: each drag adds a range.\nApply to show them, Stop Selecting to edit them."

#: Labels of the button that starts / stops picking ranges on the plot.
PICK_LABEL = "Select on Plot"
STOP_PICK_LABEL = "Stop Selecting"


def raise_dialog(dialog: Optional[tk.Toplevel]) -> bool:
    """Bring *dialog* to the front if it is still open; return whether it was."""
    try:
        if dialog is not None and dialog.winfo_exists():
            dialog.deiconify()
            dialog.lift()
            dialog.focus_set()
            return True
    except tk.TclError:
        pass
    return False


class PanelRangesDialog(tk.Toplevel):
    """Edit a stacked view's panel ranges, one ``start end`` per line.

    The text starts with the panels currently shown.  *Apply* parses it and
    passes the ranges to *set_ranges*; *Even Split* passes ``None`` and then
    shows the panels that result.  Parse errors and a ``ValueError`` from
    *set_ranges* are shown in the dialog, which then changes nothing.

    With *pick_on_plot*, *Select on Plot* lets the user drag the ranges out
    on the plot instead: the text is emptied and each drag adds its range as
    a line.  *Stop Selecting*, *Apply*, *Even Split* or closing the dialog
    ends it; stopping before any drag puts the previous text back.

    Parameters
    ----------
    parent : tk widget
        The window the dialog belongs to.
    title : str
        Window title.
    get_ranges : callable
        ``() -> [(start, end), ...]``: the panels currently shown.
    set_ranges : callable
        ``(ranges or None) -> None``: show explicit ranges, or ``None`` for an
        even split.
    data_bounds : callable
        ``() -> (start, end)``: the first / last observed wavelength that
        ``DATASTART`` / ``DATAEND`` stand for (``(None, None)`` without data).
    pick_on_plot : callable, optional
        ``(on_pick) -> stop``: report each range dragged on the plot as
        ``on_pick(start, end)`` until ``stop()`` is called.
    """

    def __init__(
        self,
        parent: Any,
        *,
        title: str,
        get_ranges: Callable[[], Sequence[Tuple[float, float]]],
        set_ranges: Callable[[Optional[List[Tuple[float, float]]]], None],
        data_bounds: Callable[[], Tuple[Optional[float], Optional[float]]],
        pick_on_plot: Optional[Callable[[Callable[[float, float], None]], Callable[[], None]]] = None,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        try:
            self.transient(parent.winfo_toplevel())
        except (AttributeError, tk.TclError):
            pass
        self.bind("<Escape>", lambda _e: self.destroy())
        # However the dialog goes (Close, the window manager, with its parent
        # window or the view), picking on the plot ends with it.
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._get_ranges = get_ranges
        self._set_ranges = set_ranges
        self._data_bounds = data_bounds
        self._pick_on_plot = pick_on_plot
        self._stop_picking: Optional[Callable[[], None]] = None
        self._text_before_picking = ""
        self._picked = 0

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=10, pady=10)
        help_text = PANEL_RANGES_HELP + ("\n" + PICK_HELP if pick_on_plot is not None else "")
        ttk.Label(content, text=help_text).pack(anchor="w")
        self.text = tk.Text(content, width=34, height=10, undo=True)
        self.text.pack(fill="both", expand=True, pady=(6, 4))
        self.hint_var = tk.StringVar(master=self, value="")
        ttk.Label(content, textvariable=self.hint_var, foreground="gray").pack(anchor="w")
        self.error_var = tk.StringVar(master=self, value="")
        ttk.Label(content, textvariable=self.error_var, foreground="red").pack(anchor="w")

        buttons = ttk.Frame(content)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Apply", command=self.apply).pack(side="left")
        ttk.Button(buttons, text="Even Split", command=self.even_split).pack(side="left", padx=(4, 0))
        self.pick_button: Optional[ttk.Button] = None
        if pick_on_plot is not None:
            self.pick_button = ttk.Button(buttons, text=PICK_LABEL, command=self.toggle_picking)
            self.pick_button.pack(side="left", padx=(4, 0))
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side="right")

        self._show_current()
        self.text.focus_set()

    def _show_current(self) -> None:
        self.text.delete("1.0", "end")
        ranges = list(self._get_ranges() or [])
        if ranges:
            starts, ends = zip(*ranges)
            self.text.insert("1.0", format_panel_ranges(starts, ends))

    def apply(self) -> bool:
        """Show the typed (or picked) ranges; returns ``False`` (and shows why) when they are invalid."""
        self.stop_picking()
        try:
            ranges = parse_panel_ranges(self.text.get("1.0", "end"), *self._data_bounds())
            self._set_ranges(ranges)
        except ValueError as exc:
            self.error_var.set(str(exc))
            return False
        self.error_var.set("")
        return True

    def even_split(self) -> None:
        """Back to the automatic split, then show the panels it made."""
        self.stop_picking()
        self._set_ranges(None)
        self._show_current()
        self.error_var.set("")

    # --- Select on Plot ---------------------------------------------------
    @property
    def picking(self) -> bool:
        """Whether drags on the plot are being picked up as ranges."""
        return self._stop_picking is not None

    def toggle_picking(self) -> None:
        """*Select on Plot* / *Stop Selecting*."""
        if self.picking:
            self.stop_picking()
        else:
            self.start_picking()

    def start_picking(self) -> None:
        """Empty the text and add each range dragged on the plot to it."""
        if self._pick_on_plot is None or self.picking:
            return
        self._text_before_picking = self.text.get("1.0", "end-1c")
        self._picked = 0
        self.text.delete("1.0", "end")
        self.error_var.set("")
        self.hint_var.set(PICKING_HINT)
        if self.pick_button is not None:
            self.pick_button.configure(text=STOP_PICK_LABEL)
        self._stop_picking = self._pick_on_plot(self._add_picked_range)

    def stop_picking(self) -> None:
        """Stop picking; with nothing picked, the previous text comes back."""
        stop, self._stop_picking = self._stop_picking, None
        if stop is None:
            return
        stop()
        if not self._picked:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", self._text_before_picking)
        self.hint_var.set("")
        if self.pick_button is not None:
            self.pick_button.configure(text=PICK_LABEL)

    def _add_picked_range(self, start: float, end: float) -> None:
        if self.text.get("1.0", "end-1c").strip():
            self.text.insert("end", "\n")
        self.text.insert("end", format_panel_ranges([start], [end]))
        self.text.see("end")
        self._picked += 1
        self.error_var.set("")

    def _on_destroy(self, event: Any) -> None:
        if event.widget is self:
            stop, self._stop_picking = self._stop_picking, None
            if stop is not None:
                stop()
