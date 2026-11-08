"""LineSearchWindow - popout window for finding the lines in a wavelength range.

The user picks a wavelength range (typed in, or taken from the range selected
on the plot) and a set of line-list files (HITRAN ``.par`` files and CSV line
lists), and every line of those files inside the range is listed in a
sortable, searchable table that can be exported to CSV.  The search itself is
:func:`~iSLAT.Modules.DataProcessing.LineRangeSearch.search_line_lists`; it
runs on a worker thread so large files do not freeze the GUI.

The file list is remembered for the session.  It starts with the line lists of
the loaded molecules.

Usage
-----
    from iSLAT.Modules.GUI.Widgets.LineSearchWindow import LineSearchWindow
    LineSearchWindow.open(parent_widget, islat_instance, theme, main_plot)
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from iSLAT.Modules.DataProcessing.LineRangeSearch import (
    LineSearchResult,
    find_line_list_files,
    format_wavelength,
    is_searchable,
    molecule_line_list_files,
    normalize_range,
    search_line_lists,
)

#: Results with more lines than this ask before being shown (a Tk table of
#: that size takes a while to fill).
LARGE_RESULT_ROWS = 20000

_POLL_MS = 50


# ----------------------------------------------------------------------
# Logic (no Tk) - also used by the tests
# ----------------------------------------------------------------------
def selection_range(main_plot: Any) -> Optional[Tuple[float, float]]:
    """The wavelength range selected on the plot, ``(low, high)``, or ``None``."""
    selection = getattr(main_plot, "current_selection", None)
    if not selection or len(selection) < 2:
        return None
    try:
        return normalize_range(selection[0], selection[-1])
    except ValueError:
        return None


def view_range(main_plot: Any) -> Optional[Tuple[float, float]]:
    """The wavelength range shown on the main plot, ``(low, high)``, or ``None``."""
    ax = getattr(main_plot, "ax1", None)
    if ax is None:
        return None
    try:
        return normalize_range(*ax.get_xlim())
    except (TypeError, ValueError):
        return None


def initial_range(main_plot: Any) -> Optional[Tuple[float, float]]:
    """The plot selection when there is one, otherwise the plotted range."""
    return selection_range(main_plot) or view_range(main_plot)


def file_location(path: str) -> str:
    """Folder of *path* for display: relative to ``DATAFILES`` when inside it."""
    from iSLAT.Modules.FileHandling.line_list_paths import to_portable_line_list_path
    portable = to_portable_line_list_path(path)
    if os.path.isabs(portable):
        return os.path.dirname(portable)
    folder = os.path.dirname(portable)
    return f"DATAFILES/{folder}" if folder else "DATAFILES"


def display_frame(lines: pd.DataFrame) -> pd.DataFrame:
    """*lines* prepared for the results table.

    Wavelengths are shown in full (the table otherwise rounds numbers to six
    significant digits), and values a file does not have are blank instead
    of ``NaN``.
    """
    shown = lines.copy()
    if "lam" in shown.columns:
        shown["lam"] = [f"{value:.10g}" for value in shown["lam"].astype(float)]
    for column in shown.columns:
        missing = shown[column].isna()
        if missing.any():
            shown[column] = shown[column].astype(object).where(~missing, "")
    return shown


def results_title(result: LineSearchResult) -> str:
    """Title of the results window."""
    return (
        f"Lines in {format_wavelength(result.lam_min)} - "
        f"{format_wavelength(result.lam_max)} um ({result.n_lines:,})"
    )


def export_basename(result: LineSearchResult) -> str:
    """Default file name (without extension) for exporting *result*."""
    return f"lines_{format_wavelength(result.lam_min)}-{format_wavelength(result.lam_max)}um"


def error_report(errors: Dict[str, str], limit: int = 3) -> str:
    """Short description of the files that could not be searched."""
    if not errors:
        return ""
    items = [f"{os.path.basename(path)} ({reason})" for path, reason in errors.items()]
    text = "Could not search: " + "; ".join(items[:limit])
    if len(items) > limit:
        text += f"; and {len(items) - limit} more"
    return text + "."


def add_paths(current: Sequence[str], new: Sequence[str]) -> Tuple[List[str], int, int]:
    """*current* with the searchable files of *new* appended.

    Returns ``(paths, n_added, n_skipped)``; files already in *current* are
    not added twice, and files that are not ``.par`` / ``.csv`` are skipped.
    """
    paths = list(current)
    seen = {os.path.normcase(os.path.abspath(p)) for p in paths}
    added = skipped = 0
    for path in new:
        if not is_searchable(path):
            skipped += 1
            continue
        path = os.path.abspath(path)
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        added += 1
    return paths, added, skipped


# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------
class LineSearchWindow:
    """Singleton popout window for finding the lines in a wavelength range."""

    # Class-level registries, keyed by id(islat): the open window, and the
    # file list (remembered for the session when the window is closed).
    _instances: Dict[int, "LineSearchWindow"] = {}
    _session_files: Dict[int, List[str]] = {}

    @classmethod
    def open(
        cls,
        parent: tk.Widget,
        islat: Any,
        theme: Optional[Dict[str, Any]] = None,
        main_plot: Any = None,
    ) -> "LineSearchWindow":
        """Open (or raise) the line search window for *islat*.

        A window that is already open takes the range selected on the plot
        (if there is one) when ``Follow plot selection`` is on.
        """
        key = id(islat)
        existing = cls._instances.get(key)
        if existing is not None:
            try:
                if existing._win.winfo_exists():
                    if existing._follow_var.get():
                        existing.use_plot_selection(quiet=True)
                    existing._win.lift()
                    existing._win.focus_force()
                    return existing
            except Exception:
                pass
            del cls._instances[key]

        instance = cls(parent, islat, theme or {}, main_plot)
        cls._instances[key] = instance
        return instance

    def __init__(self, parent: tk.Widget, islat: Any, theme: Dict[str, Any],
                 main_plot: Any = None) -> None:
        self._parent = parent
        self._islat = islat
        self._theme = theme
        self._main_plot = main_plot if main_plot is not None else getattr(
            getattr(islat, "GUI", None), "plot", None)

        key = id(islat)
        if key not in LineSearchWindow._session_files:
            LineSearchWindow._session_files[key] = molecule_line_list_files(
                getattr(islat, "molecules_dict", None))
        self._files: List[str] = list(LineSearchWindow._session_files[key])

        # Search state (the worker thread only touches the queue and the event)
        self._queue: Optional[queue.Queue] = None
        self._cancel: Optional[threading.Event] = None
        self._poll_id: Optional[str] = None

        self._win = tk.Toplevel(parent)
        self._win.title("Find Lines in Range")
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win.minsize(560, 380)

        self._min_var = tk.StringVar(master=self._win)
        self._max_var = tk.StringVar(master=self._win)
        self._follow_var = tk.BooleanVar(master=self._win, value=True)
        self._status_var = tk.StringVar(master=self._win)
        start = initial_range(self._main_plot)
        if start is not None:
            self.set_range(*start)

        self._build_ui()
        self._refresh_file_list()
        self._set_status("Choose a wavelength range and the files to search, then press Search.")

        # Follow new selections made on the plot.
        self._callback_name = f"line_search_window_{id(self)}"
        handler = getattr(self._main_plot, "interaction_handler", None)
        if handler is not None and hasattr(handler, "add_selection_callback"):
            handler.add_selection_callback(self._callback_name, self._on_plot_selection)

        # Centre on the screen.
        self._win.update_idletasks()
        x = self._win.winfo_screenwidth() // 2 - self._win.winfo_width() // 2
        y = self._win.winfo_screenheight() // 2 - self._win.winfo_height() // 2
        self._win.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        from iSLAT.Modules.GUI.Tooltips import CreateToolTip

        content = ttk.Frame(self._win)
        content.pack(fill="both", expand=True, padx=10, pady=10)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        # -- Wavelength range ------------------------------------------
        range_frame = ttk.LabelFrame(content, text="Wavelength range (um)")
        range_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(range_frame, text="From").grid(row=0, column=0, padx=(6, 4), pady=6)
        min_entry = ttk.Entry(range_frame, textvariable=self._min_var, width=12)
        min_entry.grid(row=0, column=1, pady=6)
        ttk.Label(range_frame, text="to").grid(row=0, column=2, padx=4, pady=6)
        max_entry = ttk.Entry(range_frame, textvariable=self._max_var, width=12)
        max_entry.grid(row=0, column=3, pady=6)
        for entry in (min_entry, max_entry):
            entry.bind("<Return>", lambda _e: self.start_search())
        selection_btn = ttk.Button(range_frame, text="Use Plot Selection",
                                   command=self.use_plot_selection)
        selection_btn.grid(row=0, column=4, padx=(10, 4), pady=6)
        view_btn = ttk.Button(range_frame, text="Use Plot View", command=self.use_plot_view)
        view_btn.grid(row=0, column=5, padx=(0, 6), pady=6)
        follow = ttk.Checkbutton(range_frame, text="Follow plot selection",
                                 variable=self._follow_var)
        follow.grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 6))
        CreateToolTip(selection_btn, "Use the range selected (dragged)\non the spectrum plot.")
        CreateToolTip(view_btn, "Use the wavelength range\nshown on the main plot.")
        CreateToolTip(follow, "Update the range whenever a new range\nis selected on the spectrum plot.")

        # -- Files ------------------------------------------------------
        files_frame = ttk.LabelFrame(content, text="Line lists to search (.par and .csv)")
        files_frame.grid(row=1, column=0, sticky="nsew")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(files_frame, columns=("name", "folder"), show="headings",
                                  selectmode="extended", height=8)
        self._tree.heading("name", text="File")
        self._tree.heading("folder", text="Folder")
        self._tree.column("name", width=260, stretch=True)
        self._tree.column("folder", width=220, stretch=True)
        self._tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        self._tree.bind("<Delete>", lambda _e: self.remove_selected())
        self._tree.bind("<BackSpace>", lambda _e: self.remove_selected())
        vsb = ttk.Scrollbar(files_frame, orient="vertical", command=self._tree.yview)
        vsb.grid(row=0, column=1, sticky="ns", pady=6)
        self._tree.configure(yscrollcommand=vsb.set)

        buttons = ttk.Frame(files_frame)
        buttons.grid(row=0, column=2, sticky="n", padx=6, pady=6)
        file_buttons = (
            ("Add Files\u2026", self._browse_files,
             "Add .par files or CSV line lists."),
            ("Add Folder\u2026", self._browse_folder,
             "Add every .par file and CSV line list\nin a folder and its sub-folders."),
            ("Add Molecule Files", self.add_molecule_files,
             "Add the line lists of the\nloaded molecules."),
            ("Remove", self.remove_selected,
             "Remove the selected files from the list\n(Delete key)."),
            ("Clear", self.clear_files, "Remove every file from the list."),
        )
        for row, (text, command, tip) in enumerate(file_buttons):
            button = ttk.Button(buttons, text=text, command=command)
            button.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            CreateToolTip(button, tip)
        self._count_var = tk.StringVar(master=self._win)
        ttk.Label(files_frame, textvariable=self._count_var).grid(
            row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        # -- Status and actions -----------------------------------------
        self._progress = ttk.Progressbar(content, mode="determinate")
        self._progress.grid(row=2, column=0, sticky="ew", pady=(8, 2))
        ttk.Label(content, textvariable=self._status_var, wraplength=540,
                  justify="left").grid(row=3, column=0, sticky="w", pady=(0, 8))

        actions = ttk.Frame(content)
        actions.grid(row=4, column=0, sticky="e")
        self._search_btn = ttk.Button(actions, text="Search", command=self.start_search)
        self._search_btn.pack(side="right", padx=(4, 0))
        ttk.Button(actions, text="Close", command=self._on_close).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    # Range
    # ------------------------------------------------------------------
    def set_range(self, low: float, high: float) -> None:
        """Show ``low`` and ``high`` (microns) in the range fields."""
        self._min_var.set(format_wavelength(low))
        self._max_var.set(format_wavelength(high))

    def get_range(self) -> Tuple[float, float]:
        """The range in the fields, ``(low, high)``; raises ``ValueError`` if invalid."""
        return normalize_range(self._min_var.get(), self._max_var.get())

    def use_plot_selection(self, quiet: bool = False) -> bool:
        """Take the range selected on the plot; True when there was one."""
        selected = selection_range(self._main_plot)
        if selected is None:
            if not quiet:
                self._set_status("No range is selected on the plot: drag across "
                                 "the spectrum to select one.")
            return False
        self.set_range(*selected)
        return True

    def use_plot_view(self) -> bool:
        """Take the range shown on the main plot; True when it is available."""
        shown = view_range(self._main_plot)
        if shown is None:
            self._set_status("The plotted wavelength range is not available.")
            return False
        self.set_range(*shown)
        return True

    def _on_plot_selection(self, xmin: float, xmax: float) -> None:
        if not self._follow_var.get():
            return
        try:
            self.set_range(*normalize_range(xmin, xmax))
        except (ValueError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    @property
    def files(self) -> List[str]:
        """The files that will be searched, in list order."""
        return list(self._files)

    def add_files(self, paths: Sequence[str]) -> int:
        """Add *paths* (``.par`` / ``.csv`` only, no repeats); returns the number added."""
        self._files, added, skipped = add_paths(self._files, paths)
        self._files_changed()
        message = f"Added {added} file{'' if added == 1 else 's'}."
        if skipped:
            message += f" Skipped {skipped} that {'is' if skipped == 1 else 'are'} not .par or .csv."
        self._set_status(message)
        return added

    def add_molecule_files(self) -> int:
        """Add the line lists of the loaded molecules; returns the number added."""
        paths = molecule_line_list_files(getattr(self._islat, "molecules_dict", None))
        if not paths:
            self._set_status("No loaded molecule has a line-list file.")
            return 0
        return self.add_files(paths)

    def remove_selected(self) -> None:
        """Remove the files selected in the list."""
        selected = set(self._tree.selection())
        if not selected:
            return
        self._files = [p for i, p in enumerate(self._files) if self._iid(i) not in selected]
        self._files_changed()

    def clear_files(self) -> None:
        """Remove every file from the list."""
        self._files = []
        self._files_changed()

    @staticmethod
    def _iid(index: int) -> str:
        return f"file{index}"

    def _files_changed(self) -> None:
        LineSearchWindow._session_files[id(self._islat)] = list(self._files)
        self._refresh_file_list()

    def _refresh_file_list(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for index, path in enumerate(self._files):
            self._tree.insert("", "end", iid=self._iid(index),
                              values=(os.path.basename(path), file_location(path)))
        n = len(self._files)
        self._count_var.set(f"{n} file{'' if n == 1 else 's'}")

    def _initial_dir(self) -> str:
        from iSLAT.Modules.FileHandling import hitran_data_folder_path
        if self._files:
            return os.path.dirname(self._files[-1])
        return str(hitran_data_folder_path)

    def _browse_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self._win,
            title="Select line lists to search",
            initialdir=self._initial_dir(),
            filetypes=[
                ("Line lists", "*.par *.csv"),
                ("HITRAN .par files", "*.par"),
                ("CSV line lists", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.add_files(list(paths))

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(
            parent=self._win, title="Select a folder of line lists",
            initialdir=self._initial_dir(),
        )
        if not folder:
            return
        found = find_line_list_files(folder)
        if not found:
            self._set_status(f"No .par files or CSV line lists found in {folder}.")
            return
        self.add_files(found)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @property
    def searching(self) -> bool:
        """True while a search is running."""
        return self._queue is not None

    def start_search(self) -> bool:
        """Start searching the files for the lines in the range; True when started."""
        if self.searching:
            return False
        try:
            low, high = self.get_range()
        except ValueError as exc:
            self._set_status(str(exc))
            return False
        if not self._files:
            self._set_status("Add at least one file to search.")
            return False

        files = list(self._files)
        self._queue = results = queue.Queue()
        self._cancel = cancel = threading.Event()

        def work() -> None:
            try:
                result = search_line_lists(
                    files, low, high,
                    progress=lambda i, n, path: results.put(("progress", (i, n, path))),
                    should_stop=cancel.is_set,
                )
                results.put(("done", result))
            except Exception as exc:  # reported on the GUI thread
                results.put(("error", exc))

        self._progress.configure(maximum=len(files), value=0)
        self._search_btn.configure(text="Cancel", command=self.cancel_search)
        self._set_status(f"Searching {len(files)} file{'' if len(files) == 1 else 's'}\u2026")
        threading.Thread(target=work, name="LineSearch", daemon=True).start()
        self._poll_id = self._win.after(_POLL_MS, self._poll)
        return True

    def cancel_search(self) -> None:
        """Stop the running search after the file being read."""
        if self._cancel is not None:
            self._cancel.set()
            self._set_status("Cancelling\u2026")

    def _poll(self) -> None:
        self._poll_id = None
        results = self._queue
        if results is None:
            return
        while True:
            try:
                kind, payload = results.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                index, total, path = payload
                self._progress.configure(value=index)
                if not self._cancel.is_set():
                    self._set_status(f"Searching {index + 1} of {total}: {os.path.basename(path)}")
            else:
                self._search_finished()
                if kind == "done":
                    self._show_result(payload)
                else:
                    self._set_status(f"Search failed: {payload}")
                return
        self._poll_id = self._win.after(_POLL_MS, self._poll)

    def _search_finished(self) -> None:
        self._queue = None
        self._cancel = None
        self._progress.configure(value=0)
        self._search_btn.configure(text="Search", command=self.start_search)

    def _show_result(self, result: LineSearchResult) -> None:
        if result.cancelled:
            self._set_status("Search cancelled.")
            return
        summary = result.summary()
        errors = error_report(result.errors)
        self._set_status(summary + (f"\n{errors}" if errors else ""))
        self._report(summary + (f" {errors}" if errors else ""))
        if result.n_lines == 0:
            return
        if result.n_lines > LARGE_RESULT_ROWS and not messagebox.askyesno(
            "Many Lines Found",
            f"{result.n_lines:,} lines were found. A table this large can take a "
            "while to display.\n\nShow them anyway? (Choose No to narrow the range.)",
            parent=self._win,
        ):
            return
        self._open_results(result)

    def _open_results(self, result: LineSearchResult) -> Any:
        from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
        shown = display_frame(result.lines)
        # Parented to the main window, so results stay open after this one closes.
        return LineListViewWindow.from_dataframe(
            self._parent, shown,
            title=results_title(result),
            data_field=self._data_field(),
            export_basename=export_basename(result),
        )

    # ------------------------------------------------------------------
    # Messages and closing
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        try:
            self._status_var.set(message)
        except tk.TclError:
            pass

    def _data_field(self) -> Any:
        return getattr(getattr(self._islat, "GUI", None), "data_field", None)

    def _report(self, message: str) -> None:
        data_field = self._data_field()
        if data_field is not None and hasattr(data_field, "insert_text"):
            data_field.insert_text(message)
        else:
            print(message)

    def _on_close(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
        self._queue = None
        if self._poll_id is not None:
            try:
                self._win.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        handler = getattr(self._main_plot, "interaction_handler", None)
        if handler is not None and hasattr(handler, "remove_selection_callback"):
            handler.remove_selection_callback(self._callback_name)
        LineSearchWindow._instances.pop(id(self._islat), None)
        try:
            self._win.destroy()
        except Exception:
            pass
