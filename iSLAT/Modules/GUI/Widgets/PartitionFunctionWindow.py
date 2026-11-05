"""
PartitionFunctionWindow - view and extend a molecule's partition function.

Opened from the molecule control panel's right-click menu.  It plots and
tabulates Q(T) from the molecule's line list and lets the user extend the
table to lower and/or higher temperatures with one of the registered
:mod:`~iSLAT.Modules.DataProcessing.PartitionFunctionExtender` modes (for now
only "Constant Q(T)").  The extension is previewed live; it only changes the
molecule when "Apply to Molecule" or "Duplicate & Apply" is pressed, and can
be saved to disk as a HITRAN ``.par`` line list carrying the new partition
function or as a ``temp,q`` CSV table - mirroring the line list filter window.
"""
from __future__ import annotations

import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from iSLAT.Modules.DataProcessing.PartitionFunctionExtender import (
    EXTENSION_MODES,
    ExtendedPartition,
    extend_partition_function,
    suggest_step,
)
from iSLAT.Modules.FileHandling.partition_function_writer import write_partition_function_csv
from iSLAT.Modules.GUI.Tooltips import CreateToolTip

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule

PARTITION_SAVE_FILETYPES = [
    ("HITRAN line list with this partition function", "*.par"),
    ("Partition function table (CSV)", "*.csv"),
    ("All files", "*.*"),
]
_FIGURE_FILETYPES = (
    ("PDF files", "*.pdf"), ("PNG files", "*.png"), ("SVG files", "*.svg"),
    ("All files", "*.*"),
)
#: Tables longer than this are shown decimated in the table tab.
TABLE_ROW_LIMIT = 20000
_EXTENDED_COLOR = "#d62728"
_HITRAN_HEADER_RE = re.compile(r"id:\s*([^;]*);\s*iso:\s*([^;]*);\s*gid:\s*(\S*)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Tk)
# ---------------------------------------------------------------------------
def parse_temperature(text: str, name: str, allow_blank: bool = True) -> Optional[float]:
    """Parse a temperature entry; blank means "unchanged" (``None``)."""
    text = (text or "").strip()
    if not text:
        if allow_blank:
            return None
        raise ValueError(f"{name} is required.")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name} must be a number (got {text!r}).") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def molecule_partition(mol_obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The molecule's partition function as sorted ``(t, q)`` arrays, or None."""
    try:
        line_list = mol_obj.line_list
        partition = line_list.partition if line_list is not None else None
    except Exception:
        return None
    if partition is None:
        return None
    t = np.asarray(partition.t, dtype=np.float64).ravel()
    q = np.asarray(partition.q, dtype=np.float64).ravel()
    if t.size == 0 or t.shape != q.shape:
        return None
    order = np.argsort(t, kind="stable")
    return t[order], q[order]


def _source_header(line_list) -> Dict[str, Any]:
    """Best-effort HITRAN id / iso / gid from the source ``.par`` header."""
    filename = getattr(line_list, "_filename", None)
    if not filename:
        return {}
    try:
        from iSLAT.Modules.FileHandling import absolute_data_files_path
        path = Path(filename)
        if not path.is_absolute() and not path.exists():
            path = Path(absolute_data_files_path) / path
        with open(path, "r") as fh:
            first = fh.readline()
    except Exception:
        return {}
    match = _HITRAN_HEADER_RE.search(first)
    if not match:
        return {}
    return {"M": match.group(1).strip(), "isotopologue": match.group(2).strip(),
            "G": match.group(3).strip()}


def write_partition_file(mol_obj, partition: ExtendedPartition, path) -> Path:
    """Write *partition* to *path*; the extension picks the format.

    ``.par`` writes the molecule's whole line list (ignoring the model
    wavelength range) with *partition* as its partition function, so it can
    be loaded back with "Change Line List…".  Anything else writes a
    ``temp,q`` CSV table.
    """
    path = Path(path)
    if path.suffix.lower() != ".par":
        write_partition_function_csv(path, partition.t, partition.q)
        return path

    import pandas as pd
    line_list = mol_obj.line_list
    if line_list is None:
        raise RuntimeError("the molecule has no line list to write")
    header = dict(_source_header(line_list))
    header["source"] = (
        f"Partition function extended in iSLAT ({EXTENSION_MODES[partition.mode].label}, "
        f"{partition.t[0]:g}-{partition.t[-1]:g} K, step {partition.step:g} K)"
        if partition.n_added else "Written from iSLAT")
    line_list.write_par_file(path, header=pd.DataFrame([header]),
                             partition_df=partition.as_frame())
    return path


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class PartitionFunctionWindow(tk.Toplevel):
    """View a molecule's partition function and extend its temperature range.

    Use :meth:`open` to get one window per molecule.

    Parameters
    ----------
    parent : tk widget
        Parent window (typically the ControlPanel).
    mol_obj : Molecule
        The molecule whose partition function is shown.
    data_field : optional
        iSLAT data field for status messages (``insert_text`` interface).
    islat : optional
        iSLAT instance; needed for "Duplicate & Apply" and GUI refreshes.
    theme : optional
        Active GUI theme, used to pick readable error colours.
    """

    #: Live windows, keyed by (iSLAT instance, molecule name).
    _instances: Dict[Tuple[int, str], "PartitionFunctionWindow"] = {}

    @classmethod
    def open(cls, parent, mol_obj: "Molecule", data_field=None, islat=None,
             theme=None) -> "PartitionFunctionWindow":
        """Open the window for *mol_obj*, or raise the existing one."""
        key = (id(islat), str(getattr(mol_obj, "name", "")))
        win = cls._instances.get(key)
        if win is not None:
            try:
                if win.winfo_exists() and win.mol_obj is mol_obj:
                    win.deiconify()
                    win.lift()
                    win.focus_force()
                    return win
            except tk.TclError:
                pass
            cls._instances.pop(key, None)
        win = cls(parent, mol_obj, data_field, islat=islat, theme=theme)
        cls._instances[key] = win
        win._instance_key = key
        return win

    def __init__(self, parent, mol_obj: "Molecule", data_field=None, islat=None,
                 theme=None):
        super().__init__(parent)
        self.mol_obj = mol_obj
        self.data_field = data_field
        self._islat = islat
        self._theme_name = theme
        self._instance_key: Optional[Tuple[int, str]] = None
        self._preview: Optional[ExtendedPartition] = None
        self._error: Optional[str] = None
        self._preview_job: Optional[str] = None
        self._table_dirty = True
        self._fig = None
        self._ax = None
        self._canvas = None
        self._right_click_cid = None

        mol_name = getattr(mol_obj, "name", "Molecule")
        self.title(f"Partition Function: {mol_name}")
        self.resizable(True, True)

        base = molecule_partition(mol_obj)
        if base is None:
            self.destroy()
            raise ValueError(f"Molecule '{mol_name}' has no partition function.")

        self._build_ui()
        self._reset_range(schedule=False)
        self._constrain_to_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._recompute()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _constrain_to_screen(self) -> None:
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(760, int(sw * 0.55))
        h = min(780, int(sh * 0.85))
        self.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 2, 0)}")
        self.minsize(560, 560)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        is_dark = "dark" in str(self._theme_name).lower()
        style.configure("Error.TLabel", foreground="#ff6b6b" if is_dark else "red")
        style.configure("Muted.TLabel", foreground="#888888")

    def _plot_theme(self) -> Dict[str, Any]:
        gui = getattr(self._islat, "GUI", None)
        theme = getattr(getattr(gui, "plot", None), "theme", None)
        if theme is None:
            theme = getattr(gui, "theme", None)
        return theme if isinstance(theme, dict) else {}

    def _build_ui(self) -> None:
        self._configure_styles()
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        self._info_var = tk.StringVar()
        ttk.Label(outer, textvariable=self._info_var, anchor="w").grid(
            row=0, column=0, sticky="ew", pady=(0, 4))

        # ── Plot / table ──────────────────────────────────────────────
        self._notebook = ttk.Notebook(outer)
        self._notebook.grid(row=1, column=0, sticky="nsew")
        self._plot_tab = ttk.Frame(self._notebook)
        self._table_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._plot_tab, text="Plot")
        self._notebook.add(self._table_tab, text="Table")
        self._notebook.bind("<<NotebookTabChanged>>", lambda _e: self._refresh_table())
        self._build_plot(self._plot_tab)
        self._build_table(self._table_tab)

        # ── Extension controls ────────────────────────────────────────
        ext = ttk.LabelFrame(outer, text="Extend temperature range", padding=6)
        ext.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        for col in (1, 3, 5):
            ext.columnconfigure(col, weight=1)

        ttk.Label(ext, text="Mode").grid(row=0, column=0, sticky="w")
        self._mode_labels = {m.label: m.key for m in EXTENSION_MODES.values()}
        self._mode_var = tk.StringVar(value=next(iter(self._mode_labels)))
        mode_combo = ttk.Combobox(ext, textvariable=self._mode_var, state="readonly",
                                  values=list(self._mode_labels), width=18)
        mode_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=4, pady=2)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)
        CreateToolTip(mode_combo, "How Q(T) is filled at the added temperatures.")
        reset_btn = ttk.Button(ext, text="Reset", command=self._reset_range)
        reset_btn.grid(row=0, column=5, sticky="e", padx=4)
        CreateToolTip(reset_btn, "Set the range back to the molecule's current table "
                                 "and the step to its typical spacing.")

        self._tmin_var = tk.StringVar()
        self._tmax_var = tk.StringVar()
        self._step_var = tk.StringVar()
        fields = (
            ("Lowest T [K]", self._tmin_var,
             "New lowest temperature. Values at or above the current lowest\n"
             "temperature add nothing below the table."),
            ("Highest T [K]", self._tmax_var,
             "New highest temperature. Values at or below the current highest\n"
             "temperature add nothing above the table."),
            ("Step [K]", self._step_var,
             "Spacing of the added temperatures, counted outward from the end\n"
             "of the table. The requested bound is always included."),
        )
        for i, (label, var, tip) in enumerate(fields):
            ttk.Label(ext, text=label).grid(row=1, column=2 * i, sticky="w", pady=2)
            entry = ttk.Entry(ext, textvariable=var, width=10)
            entry.grid(row=1, column=2 * i + 1, sticky="ew", padx=4, pady=2)
            CreateToolTip(entry, tip)
            var.trace_add("write", self._schedule_preview)

        self._mode_desc_var = tk.StringVar()
        ttk.Label(ext, textvariable=self._mode_desc_var, style="Muted.TLabel",
                  wraplength=520, justify="left").grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(2, 0))
        self._update_mode_description()

        # ── Status ────────────────────────────────────────────────────
        self._status_var = tk.StringVar()
        self._status_lbl = ttk.Label(outer, textvariable=self._status_var,
                                     relief="sunken", anchor="w")
        self._status_lbl.grid(row=3, column=0, sticky="ew", pady=(6, 4))

        # ── Actions ───────────────────────────────────────────────────
        btns = ttk.Frame(outer)
        btns.grid(row=4, column=0, sticky="ew")
        for col in range(3):
            btns.columnconfigure(col, weight=1)
        save_btn = ttk.Button(btns, text="Save Partition Function\u2026",
                              command=self._save_partition_function)
        save_btn.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        CreateToolTip(save_btn,
                      "Save the partition function shown to a file.\n"
                      ".par writes the molecule's full line list with this partition\n"
                      "function (load it later with 'Change Line List…');\n"
                      ".csv writes a temp,q table. The format follows the extension.")
        self._apply_btn = ttk.Button(btns, text="Apply to Molecule",
                                     command=self._apply_to_molecule)
        self._apply_btn.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        CreateToolTip(self._apply_btn,
                      "Replace this molecule's partition function with the extended one\n"
                      "in-place for this session. The line list file is not changed.")
        self._dup_btn = ttk.Button(btns, text="Duplicate & Apply",
                                   command=self._duplicate_and_apply)
        self._dup_btn.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        CreateToolTip(self._dup_btn,
                      "Create a copy of this molecule using the extended partition\n"
                      "function, leaving the original molecule unchanged.")

        self.bind("<Escape>", lambda _e: self._on_close())

    def _build_plot(self, parent) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X)
        self._log_q_var = tk.BooleanVar(value=True)
        self._log_t_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Log Q(T)", variable=self._log_q_var,
                        command=self._draw).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(bar, text="Log T", variable=self._log_t_var,
                        command=self._draw).pack(side="left")

        self._fig = Figure(figsize=(6, 4), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        self._right_click_cid = self._canvas.mpl_connect(
            "button_press_event", self._on_canvas_button_press)
        CreateToolTip(widget, "Right-click to save the figure")
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        try:
            NavigationToolbar2Tk(self._canvas, toolbar_frame).update()
        except Exception:
            pass

    def _build_table(self, parent) -> None:
        cols = ("temp", "q", "source")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings")
        for col, heading, width in (("temp", "T [K]", 110), ("q", "Q(T)", 160),
                                    ("source", "Source", 90)):
            self._tree.heading(col, text=heading)
            self._tree.column(col, width=width, anchor="e" if col != "source" else "w")
        self._tree.tag_configure("extended", foreground=_EXTENDED_COLOR)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._table_note_var = tk.StringVar()
        ttk.Label(parent, textvariable=self._table_note_var, style="Muted.TLabel").pack(
            side="bottom", fill="x")
        scroll.pack(side="right", fill="y")
        self._tree.pack(side="left", fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _mode_key(self) -> str:
        return self._mode_labels.get(self._mode_var.get(), "constant")

    def _update_mode_description(self) -> None:
        mode = EXTENSION_MODES.get(self._mode_key())
        self._mode_desc_var.set(mode.description if mode else "")

    def _on_mode_changed(self, *_args) -> None:
        self._update_mode_description()
        self._schedule_preview()

    def _reset_range(self, schedule: bool = True) -> None:
        base = molecule_partition(self.mol_obj)
        if base is None:
            return
        t, _q = base
        self._tmin_var.set(f"{t[0]:g}")
        self._tmax_var.set(f"{t[-1]:g}")
        self._step_var.set(f"{suggest_step(t):g}")
        if schedule:
            self._schedule_preview()

    def _schedule_preview(self, *_args) -> None:
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except (tk.TclError, ValueError):
                pass
        try:
            self._preview_job = self.after(200, self._recompute)
        except tk.TclError:
            self._preview_job = None

    def _recompute(self) -> None:
        """Re-read the molecule's table and rebuild the extension preview."""
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except (tk.TclError, ValueError):
                pass
            self._preview_job = None
        base = molecule_partition(self.mol_obj)
        self._preview = None
        self._error = None
        if base is None:
            self._error = "The molecule no longer has a partition function."
        else:
            t, q = base
            self._info_var.set(
                f"Current: {t.size} points, T = {t[0]:g} \u2013 {t[-1]:g} K, "
                f"Q = {q.min():.6g} \u2013 {q.max():.6g}")
            try:
                self._preview = extend_partition_function(
                    t, q,
                    t_low=parse_temperature(self._tmin_var.get(), "Lowest T"),
                    t_high=parse_temperature(self._tmax_var.get(), "Highest T"),
                    step=parse_temperature(self._step_var.get(), "Step", allow_blank=False),
                    mode=self._mode_key(),
                )
            except ValueError as exc:
                self._error = str(exc)
        self._update_status()
        self._draw()
        self._table_dirty = True
        self._refresh_table()

    def _update_status(self) -> None:
        if self._error:
            self._status_var.set(self._error)
            self._status_lbl.configure(style="Error.TLabel")
        else:
            p = self._preview
            msg = (f"Preview: {p.t.size} points, T = {p.t[0]:g} \u2013 {p.t[-1]:g} K "
                   f"(+{p.n_added_low} below, +{p.n_added_high} above)."
                   if p.n_added else "No extension: the requested range is within the "
                                     "current table.")
            if p.notes:
                msg += " " + " ".join(p.notes)
            self._status_var.set(msg)
            self._status_lbl.configure(style="TLabel")
        can_apply = self._preview is not None and self._preview.n_added > 0
        state = "normal" if can_apply else "disabled"
        self._apply_btn.configure(state=state)
        self._dup_btn.configure(state=state)

    def _draw(self) -> None:
        if self._ax is None:
            return
        ax = self._ax
        ax.clear()
        theme = self._plot_theme()
        fg = theme.get("foreground", "black")
        bg = theme.get("background", "white")
        base = molecule_partition(self.mol_obj)
        color = getattr(self.mol_obj, "color", None) or "tab:blue"
        if base is not None:
            ax.plot(base[0], base[1], color=color, lw=1.6, label="Current Q(T)")
        p = self._preview
        if p is not None and p.n_added and base is not None:
            # Join each added segment to the table edge so the curve is continuous
            ext_t, ext_q = p.t[p.is_extended], p.q[p.is_extended]
            low = ext_t < base[0][0]
            label = f"Extended ({EXTENSION_MODES[p.mode].label})"
            if low.any():
                ax.plot(np.append(ext_t[low], base[0][0]), np.append(ext_q[low], base[1][0]),
                        ls="--", lw=1.6, color=_EXTENDED_COLOR, label=label)
                label = None
            if (~low).any():
                ax.plot(np.insert(ext_t[~low], 0, base[0][-1]),
                        np.insert(ext_q[~low], 0, base[1][-1]),
                        ls="--", lw=1.6, color=_EXTENDED_COLOR, label=label)
        temp = getattr(self.mol_obj, "temp", None)
        try:
            temp = float(temp)
        except (TypeError, ValueError):
            temp = None
        if temp is not None and np.isfinite(temp) and temp > 0:
            ax.axvline(temp, color=fg, ls=":", lw=1.0, label=f"Model T = {temp:g} K")
        ax.set_xscale("log" if self._log_t_var.get() else "linear")
        ax.set_yscale("log" if self._log_q_var.get() else "linear")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Q(T)")
        ax.set_title(f"Partition function: {getattr(self.mol_obj, 'name', '')}")
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(loc="best", fontsize="small")
            legend.get_frame().set_facecolor(theme.get("graph_fill_color", bg))
            for text in legend.get_texts():
                text.set_color(fg)
        self._fig.set_facecolor(bg)
        ax.set_facecolor(theme.get("graph_fill_color", bg))
        ax.tick_params(colors=fg, which="both")
        for item in (ax.xaxis.label, ax.yaxis.label, ax.title):
            item.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        if self._canvas is not None:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    def _refresh_table(self) -> None:
        """Fill the table, lazily: only when it is visible and out of date."""
        if not self._table_dirty:
            return
        try:
            if self._notebook.index(self._notebook.select()) != 1:
                return
        except tk.TclError:
            return
        self._table_dirty = False
        self._tree.delete(*self._tree.get_children())
        if self._preview is not None:
            t, q, ext = self._preview.t, self._preview.q, self._preview.is_extended
        else:
            base = molecule_partition(self.mol_obj)
            if base is None:
                self._table_note_var.set("")
                return
            t, q = base
            ext = np.zeros(t.size, dtype=bool)
        idx = np.arange(t.size)
        if t.size > TABLE_ROW_LIMIT:
            stride = int(np.ceil(t.size / TABLE_ROW_LIMIT))
            idx = np.unique(np.append(idx[::stride], t.size - 1))
            self._table_note_var.set(
                f"Showing every {stride}th of {t.size} rows (saved files contain all rows).")
        else:
            self._table_note_var.set(f"{t.size} rows")
        for i in idx:
            self._tree.insert("", "end", values=(
                f"{t[i]:g}", f"{q[i]:.6f}", "extended" if ext[i] else "original"),
                tags=("extended",) if ext[i] else ())

    # ------------------------------------------------------------------
    # Figure context menu
    # ------------------------------------------------------------------
    def _on_canvas_button_press(self, event) -> None:
        if getattr(event, "button", None) != 3:
            return
        widget = self._canvas.get_tk_widget()
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Save Figure\u2026", command=self.save_figure)
        try:
            menu.tk_popup(widget.winfo_pointerx(), widget.winfo_pointery())
        finally:
            menu.grab_release()

    def save_figure(self, save_path: Optional[str] = None, dpi: int = 300) -> Optional[str]:
        """Save the plot; the format follows the extension. Returns the path or None."""
        if self._fig is None:
            return None
        if save_path is None:
            stem = re.sub(r"[^\w.-]+", "_", f"{getattr(self.mol_obj, 'name', 'molecule')}"
                                             "_partition_function")
            save_path = filedialog.asksaveasfilename(
                parent=self, title="Save Figure", defaultextension=".pdf",
                initialfile=f"{stem}.pdf", filetypes=_FIGURE_FILETYPES)
        if not save_path:
            return None
        try:
            self._fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save figure:\n{exc}", parent=self)
            return None
        self._report(f"Figure saved to: {save_path}", "Save Complete")
        return str(save_path)

    # ------------------------------------------------------------------
    # Apply / duplicate / save
    # ------------------------------------------------------------------
    def _report(self, msg: str, title: str = "Done") -> None:
        """Show *msg* in the data field when available, else in a dialog."""
        if self.data_field is not None:
            self.data_field.insert_text(msg)
        else:
            messagebox.showinfo(title, msg, parent=self)

    def _current_extension(self) -> Optional[ExtendedPartition]:
        """The previewed extension, recomputed now so pending edits count."""
        if self._preview_job is not None:
            self._recompute()
        if self._preview is None:
            messagebox.showwarning("Invalid Extension", self._error or "Nothing to apply.",
                                   parent=self)
        return self._preview

    def _apply_to_molecule(self) -> None:
        p = self._current_extension()
        if p is None:
            return
        if not p.n_added:
            messagebox.showinfo("Nothing to Apply",
                                "The requested range is within the current table.",
                                parent=self)
            return
        mol_name = getattr(self.mol_obj, "name", "this molecule")
        if not messagebox.askyesno(
                "Apply to Molecule",
                f"Replace the partition function of '{mol_name}' with the extended one "
                f"({p.t.size} points, {p.t[0]:g}\u2013{p.t[-1]:g} K)?\n"
                "This changes the in-memory molecule for this session; the line list "
                "file is not modified.",
                parent=self):
            return
        try:
            self.mol_obj.set_partition_function(p.t, p.q)
        except Exception as exc:
            messagebox.showerror("Apply Failed", str(exc), parent=self)
            return
        self._trigger_refresh(full_rebuild=False)
        self._recompute()
        self._report(f"Applied extended partition function to '{mol_name}': "
                     f"T = {p.t[0]:g}\u2013{p.t[-1]:g} K ({p.t.size} points).")

    def _duplicate_and_apply(self) -> None:
        p = self._current_extension()
        if p is None:
            return
        if not p.n_added:
            messagebox.showinfo("Nothing to Apply",
                                "The requested range is within the current table.",
                                parent=self)
            return
        mol_dict = getattr(self._islat, "molecules_dict", None)
        if mol_dict is None:
            messagebox.showerror(
                "No iSLAT Reference",
                "Cannot duplicate: no iSLAT instance was provided to this window.",
                parent=self)
            return
        mol_name = getattr(self.mol_obj, "name", None)
        new_name = mol_dict.duplicate(mol_name) if mol_name else None
        new_mol = mol_dict.get(new_name) if new_name else None
        if new_mol is None:
            messagebox.showerror("Duplicate Failed",
                                 f"Could not duplicate molecule '{mol_name}'.", parent=self)
            return
        try:
            # Keep the source's current lines (they may differ from its file)
            new_mol.lines = self.mol_obj.line_list.copy()
            new_mol.set_partition_function(p.t, p.q)
        except Exception as exc:
            messagebox.showerror("Apply Failed", str(exc), parent=self)
            return
        self._trigger_refresh(full_rebuild=True)
        self._report(f"Created '{new_name}' with the extended partition function "
                     f"(T = {p.t[0]:g}\u2013{p.t[-1]:g} K).")

    def _trigger_refresh(self, full_rebuild: bool = False) -> None:
        """Refresh the iSLAT GUI after a partition-function change."""
        try:
            gui = getattr(self._islat, "GUI", None)
            if gui is None:
                return
            plot = getattr(gui, "plot", None)
            if plot is not None:
                plot.update_model_plot()
            cp = getattr(gui, "control_panel", None)
            if cp is not None and full_rebuild:
                cp.refresh_from_molecules_dict()
        except Exception as exc:
            print(f"PartitionFunctionWindow: GUI refresh error - {exc}")

    def _save_partition_function(self) -> None:
        p = self._current_extension()
        if p is None:
            return
        stem = re.sub(r"[^\w.-]+", "_", f"{getattr(self.mol_obj, 'name', 'molecule')}"
                                         f"{'_partition_extended' if p.n_added else '_partition'}")
        path = filedialog.asksaveasfilename(
            parent=self, title="Save Partition Function", defaultextension=".par",
            initialfile=f"{stem}.par", filetypes=PARTITION_SAVE_FILETYPES)
        if not path:
            return
        self._write_partition(Path(path))

    def _write_partition(self, path: Path) -> Optional[Path]:
        p = self._preview
        if p is None:
            return None
        try:
            out = write_partition_file(self.mol_obj, p, path)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)
            return None
        self._report(f"Saved partition function ({p.t.size} points) \u2192 {out.name}",
                     "Save Complete")
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except (tk.TclError, ValueError):
                pass
            self._preview_job = None
        if self._canvas is not None and self._right_click_cid is not None:
            try:
                self._canvas.mpl_disconnect(self._right_click_cid)
            except Exception:
                pass
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
        self._fig = self._ax = self._canvas = None
        key = self._instance_key
        if key is not None and type(self)._instances.get(key) is self:
            del type(self)._instances[key]
        try:
            self.destroy()
        except tk.TclError:
            pass
