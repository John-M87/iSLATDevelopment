"""SampleManagerWindow — popout window for managing the spectrum sample.

Provides a scrollable list of all spectra in the current iSLAT sample with:
  * Radio-button selection to switch the active spectrum.
  * Per-row x button to remove a spectrum from the sample.
  * Per-row parameter file assignment: choose a CSV that is loaded when the spectrum is activated; leave blank to keep whatever parameters are current.
  * Per-row global-parameter overrides (e.g. distance, stellar RV): values set here are applied whenever the spectrum is activated and win over parameter-file values. Blank = no override.
  * "Add Spectra…" to open a file dialog and append new files.
  * "Clear All" to remove every entry.

The sample (files, parameter files, overrides) is persisted to
``DATAFILES/CONFIG/SampleConfig.json`` and restored on startup.

Usage
-----
    from iSLAT.Modules.GUI.Widgets.SampleManagerWindow import SampleManagerWindow
    SampleManagerWindow.open(parent_widget, islat_instance, theme)
"""
from __future__ import annotations

import os
import tkinter as tk
import tkinter.ttk as ttk
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from iSLAT.Modules.GUI.Tooltips import CreateToolTip
from iSLAT.Modules.GUI.TrackpadNavigation import bind_touchpad_scroll

if TYPE_CHECKING:
    pass

def load_global_override_fields() -> Dict[str, Dict[str, Any]]:
    """Return the overridable global fields from ControlPanelFields.json.

    Only entries that define a ``property`` (the attribute name on
    ``MoleculeDict``) are returned; documentation keys are skipped.
    """
    try:
        from iSLAT.Modules.FileHandling.iSLATFileHandling import (
            load_control_panel_fields_config,
        )
        fields = load_control_panel_fields_config().get("global_fields", {})
    except Exception as exc:
        print(f"[SampleManagerWindow] Could not load global fields: {exc}")
        return {}
    return {
        key: cfg
        for key, cfg in fields.items()
        if not key.startswith("_") and isinstance(cfg, dict) and cfg.get("property")
    }

def parse_override_value(text: str) -> Tuple[Optional[float], bool]:
    """Parse an override entry string.

    Returns ``(value, valid)``.  A blank string means "no override" and
    returns ``(None, True)``; an unparsable string returns ``(None, False)``.
    """
    text = (text or "").strip()
    if not text:
        return None, True
    try:
        return float(text), True
    except ValueError:
        return None, False

def format_override_value(value: Any, fmt: str = "{:.6g}") -> str:
    """Format an override value for display in an entry widget."""
    try:
        return fmt.format(float(value))
    except Exception:
        return str(value)

class SampleManagerWindow:
    """Singleton popout window that manages the spectrum sample."""
    # Class-level registry: one window per (islat instance id)
    _instances: Dict[int, "SampleManagerWindow"] = {}
    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        parent: tk.Widget,
        islat: Any,
        theme: Optional[Dict[str, Any]] = None,
    ) -> "SampleManagerWindow":
        """Open (or raise) the sample manager for *islat*.

        Parameters
        ----------
        parent:
            Tk parent widget (used as the ``Toplevel`` parent).
        islat:
            The active :class:`iSLAT` instance.
        theme:
            Optional theme dict used for highlight colours.
        """
        key = id(islat)
        existing = cls._instances.get(key)
        if existing is not None:
            try:
                if existing._win.winfo_exists():
                    existing._win.lift()
                    existing._win.focus_force()
                    return existing
            except Exception:
                pass
            del cls._instances[key]

        instance = cls(parent, islat, theme or {})
        cls._instances[key] = instance
        return instance

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: tk.Widget,
        islat: Any,
        theme: Dict[str, Any],
    ) -> None:
        self._islat = islat
        self._theme = theme

        # ------ top-level window --------------------------------------
        self._win = tk.Toplevel(parent)
        self._win.title("Manage Sample")
        self._win.resizable(True, True)
        self._win.minsize(680, 200)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._active_var = tk.IntVar(
            value=int(getattr(islat, "sample_spectra_index", 0))
        )

        self._row_widgets: List[tk.Frame] = []
        self._expanded: Set[str] = set()  # spectrum paths with open override editor
        self._ovr_buttons: Dict[str, ttk.Button] = {}
        self._fields: Dict[str, Dict[str, Any]] = load_global_override_fields()

        self._build_ui()
        self._rebuild_rows()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        win = self._win

        # Header
        hdr = tk.Frame(win)
        hdr.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(hdr, text="Active", width=6, anchor="center").pack(side="left")
        tk.Label(hdr, text="Spectrum file", anchor="w").pack(
            side="left", padx=(4, 0)
        )
        tk.Label(hdr, text="Parameter file", anchor="e").pack(
            side="right", padx=(0, 4)
        )

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Bottom button bar: packed before the rows area (from the bottom) so
        # a short window shrinks the rows area, not the buttons.
        btn_bar = tk.Frame(win)
        btn_bar.pack(side="bottom", fill="x", padx=8, pady=(2, 8))
        ttk.Separator(win, orient="horizontal").pack(
            side="bottom", fill="x", padx=8, pady=(4, 2)
        )

        ttk.Button(
            btn_bar, text="Add Spectra\u2026", command=self._add_spectra
        ).pack(side="left", padx=(0, 4))
        ttk.Button(btn_bar, text="Clear All", command=self._clear_all).pack(
            side="left", padx=4
        )
        ttk.Button(btn_bar, text="Close", command=self._win.destroy).pack(
            side="right"
        )

        # Scrollable rows area
        container = tk.Frame(win)
        container.pack(fill="both", expand=True, padx=8, pady=2)

        self._canvas = tk.Canvas(container, highlightthickness=0)
        vsb = ttk.Scrollbar(
            container, orient="vertical", command=self._canvas.yview
        )
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._rows_frame = tk.Frame(self._canvas)
        self._rows_frame_id = self._canvas.create_window(
            (0, 0), window=self._rows_frame, anchor="nw"
        )

        self._rows_frame.bind("<Configure>", self._on_rows_configure)
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(
                self._rows_frame_id, width=e.width
            ),
        )

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------
    def _rebuild_rows(self) -> None:
        """Destroy and recreate all spectrum rows."""
        for w in self._rows_frame.winfo_children():
            w.destroy()
        self._row_widgets.clear()

        islat = self._islat
        sample: List[str] = list(getattr(islat, "sample_spectra", []))
        cur_idx: int = int(getattr(islat, "sample_spectra_index", 0))
        params: dict = getattr(islat, "sample_spectra_params", {})
        overrides_map: dict = getattr(islat, "sample_spectra_overrides", {})
        self._active_var.set(cur_idx)
        self._ovr_buttons.clear()
        # Drop stale expansion state for removed spectra
        self._expanded &= set(sample)

        is_dark = "dark" in str(self._theme).lower()
        hl_bg = "#2a5a8a" if is_dark else "#cce0f5"

        for idx, path in enumerate(sample):
            name = os.path.basename(path)
            param_path: Optional[str] = params.get(path)

            # outer container for this spectrum entry
            outer = tk.Frame(self._rows_frame)
            outer.pack(fill="x", pady=(2, 0))

            # top row: radio + spectrum name + x
            top_row = tk.Frame(outer)
            top_row.pack(fill="x")

            rb = tk.Radiobutton(
                top_row,
                variable=self._active_var,
                value=idx,
                command=lambda i=idx: self._activate(i),
                width=4,
            )
            rb.pack(side="left")

            lbl = tk.Label(top_row, text=name, anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True, padx=(2, 8))
            lbl.bind("<Button-1>", lambda e, i=idx: self._activate(i))
            CreateToolTip(lbl, path)

            ttk.Button(
                top_row,
                text="\u2715",
                width=2,
                command=lambda i=idx: self._remove(i),
            ).pack(side="right", padx=(0, 2))

            if idx == cur_idx:
                for w in (outer, top_row):
                    w.configure(bg=hl_bg)
                lbl.configure(bg=hl_bg, font=("TkDefaultFont", 9, "bold"))
                rb.configure(bg=hl_bg, activebackground=hl_bg)

            # bottom row: param file indicator + Browse + Clear
            param_row = tk.Frame(outer)
            param_row.pack(fill="x", padx=(28, 2), pady=(0, 2))

            param_lbl_text = (
                os.path.basename(param_path) if param_path else "(use existing parameters)"
            )
            param_lbl_kw: dict = dict(
                text=param_lbl_text,
                anchor="w",
                font=("TkDefaultFont", 8, "italic" if not param_path else "normal"),
            )
            if not param_path:
                param_lbl_kw["foreground"] = "#888888"
            param_lbl = tk.Label(param_row, **param_lbl_kw)
            param_lbl.pack(side="left", fill="x", expand=True)
            if param_path:
                CreateToolTip(param_lbl, param_path)

            ttk.Button(
                param_row,
                text="Browse\u2026",
                width=8,
                command=lambda i=idx, p=path: self._browse_param_file(i, p),
            ).pack(side="right", padx=(2, 0))

            clear_btn = ttk.Button(
                param_row,
                text="Clear",
                width=5,
                command=lambda i=idx, p=path: self._clear_param_file(i, p),
                state="normal" if param_path else "disabled",
            )
            clear_btn.pack(side="right", padx=(2, 0))

            # overrides row: toggle button + summary + optional inline editor
            ovr: Dict[str, Any] = overrides_map.get(path, {}) or {}
            ovr_row = tk.Frame(outer)
            ovr_row.pack(fill="x", padx=(28, 2), pady=(0, 2))

            expanded = path in self._expanded
            arrow = "\u25be" if expanded else "\u25b8"
            btn_text = f"{arrow} Overrides ({len(ovr)})" if ovr else f"{arrow} Overrides"
            ovr_btn = ttk.Button(
                ovr_row,
                text=btn_text,
                width=14,
                command=lambda p=path: self._toggle_overrides(p),
            )
            ovr_btn.pack(side="left")
            CreateToolTip(
                ovr_btn,
                "Per-spectrum global parameter overrides.\n"
                "Applied when this spectrum is activated;\n"
                "they win over parameter-file values.\n"
                "Leave a field blank for no override.",
            )
            self._ovr_buttons[path] = ovr_btn

            if ovr and not expanded:
                summary = ", ".join(
                    f"{self._fields.get(k, {}).get('label', k).rstrip(':')}"
                    f"={format_override_value(v)}"
                    for k, v in ovr.items()
                )
                tk.Label(
                    ovr_row,
                    text=summary,
                    anchor="w",
                    font=("TkDefaultFont", 8, "italic"),
                    foreground="#888888",
                ).pack(side="left", fill="x", expand=True, padx=(6, 0))

            if expanded:
                editor = tk.Frame(outer)
                editor.pack(fill="x", padx=(46, 2), pady=(0, 3))
                self._build_override_editor(editor, path, ovr)

            ttk.Separator(self._rows_frame, orient="horizontal").pack(
                fill="x", padx=4, pady=1
            )

            self._row_widgets.append(outer)

        self._bind_scroll(self._rows_frame)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        # Resize window to fit content (clamped at 600 px)
        n = max(len(sample), 1)
        extra = sum(
            len(self._fields) * 26 + 8 for p in sample if p in self._expanded
        )
        self._win.geometry(f"680x{min(120 + n * 75 + extra, 600)}")

    def _build_override_editor(
        self, parent: tk.Frame, path: str, ovr: Dict[str, Any]
    ) -> None:
        """Create one labelled entry per global field inside *parent*."""
        for key, cfg in self._fields.items():
            line = tk.Frame(parent)
            line.pack(fill="x", pady=1)

            lbl = tk.Label(
                line,
                text=cfg.get("label", key),
                width=12,
                anchor="w",
                font=("TkDefaultFont", 8),
            )
            lbl.pack(side="left")
            if cfg.get("tip"):
                CreateToolTip(lbl, cfg["tip"])

            entry = tk.Entry(line, width=10, font=("TkDefaultFont", 8))
            entry.pack(side="left", padx=(2, 4))
            if key in ovr:
                entry.insert(
                    0, format_override_value(ovr[key], cfg.get("format", "{:.6g}"))
                )

            entry.bind(
                "<Return>",
                lambda e, p=path, k=key, w=entry: self._commit_override(p, k, w),
            )
            entry.bind(
                "<FocusOut>",
                lambda e, p=path, k=key, w=entry: self._commit_override(p, k, w),
            )

            tk.Label(
                line,
                text="(blank = no override)",
                font=("TkDefaultFont", 7, "italic"),
                foreground="#888888",
            ).pack(side="left")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _activate(self, idx: int) -> None:
        """Switch the active spectrum to *idx*."""
        try:
            self._islat.switch_to_spectrum(idx)
            self._active_var.set(idx)
        except Exception as exc:
            print(f"[SampleManagerWindow] switch error: {exc}")
        self._rebuild_rows()
        self._refresh_file_pane()

    def _remove(self, idx: int) -> None:
        """Remove the spectrum at *idx* from the sample."""
        islat = self._islat
        if hasattr(islat, "remove_sample_spectrum"):
            islat.remove_sample_spectrum(idx)
        else:
            # Fallback for older iSLAT instances
            sample: List[str] = list(getattr(islat, "sample_spectra", []))
            if not sample or idx >= len(sample):
                return
            removed = sample.pop(idx)
            getattr(islat, "sample_spectra_params", {}).pop(removed, None)
            getattr(islat, "sample_spectra_overrides", {}).pop(removed, None)
            cur: int = int(getattr(islat, "sample_spectra_index", 0))
            if len(sample) == 0:
                islat.sample_spectra_index = 0
            elif cur >= len(sample):
                islat.sample_spectra_index = len(sample) - 1
            elif idx < cur:
                islat.sample_spectra_index = cur - 1
            islat.sample_spectra = sample
        self._rebuild_rows()
        self._refresh_file_pane()

    def _browse_param_file(self, idx: int, spectrum_path: str) -> None:
        """Open a file dialog and assign a parameter CSV to *spectrum_path*."""
        from iSLAT.Modules.GUI import GUI as GUIModule
        from iSLAT.Modules.FileHandling.iSLATFileHandling import save_folder_path

        chosen = GUIModule.file_selector(
            title="Select Parameter File",
            initialdir=save_folder_path,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            allow_multiple=False,
        )
        if chosen:
            path = chosen if isinstance(chosen, str) else chosen[0]
            params: dict = getattr(self._islat, "sample_spectra_params", {})
            params[spectrum_path] = path
            self._islat.sample_spectra_params = params
            self._save_state()
        self._rebuild_rows()

    def _clear_param_file(self, idx: int, spectrum_path: str) -> None:
        """Remove the parameter file assignment for *spectrum_path*."""
        params: dict = getattr(self._islat, "sample_spectra_params", {})
        params.pop(spectrum_path, None)
        self._islat.sample_spectra_params = params
        self._save_state()
        self._rebuild_rows()

    def _toggle_overrides(self, spectrum_path: str) -> None:
        """Expand or collapse the override editor for *spectrum_path*."""
        if spectrum_path in self._expanded:
            self._expanded.discard(spectrum_path)
        else:
            self._expanded.add(spectrum_path)
        self._rebuild_rows()

    def _commit_override(
        self, spectrum_path: str, field_key: str, entry: tk.Entry
    ) -> None:
        """Parse *entry* and store/clear the override for *field_key*."""
        try:
            text = entry.get()
        except tk.TclError:
            return  # widget already destroyed (e.g. during rebuild)

        value, valid = parse_override_value(text)
        if not valid:
            try:
                entry.configure(background="#ffb3b3")
            except tk.TclError:
                pass
            return
        try:
            entry.configure(background="white")
        except tk.TclError:
            pass

        islat = self._islat
        overrides_map: dict = getattr(islat, "sample_spectra_overrides", {})
        ovr: Dict[str, Any] = overrides_map.get(spectrum_path, {})

        changed = False
        if value is None:
            if field_key in ovr:
                ovr.pop(field_key, None)
                changed = True
        elif ovr.get(field_key) != value:
            ovr[field_key] = value
            changed = True

        if not changed:
            return

        if ovr:
            overrides_map[spectrum_path] = ovr
        else:
            overrides_map.pop(spectrum_path, None)
        islat.sample_spectra_overrides = overrides_map

        # Apply immediately when editing the active spectrum
        if value is not None and self._is_active_path(spectrum_path):
            cfg = self._fields.get(field_key) or {}
            prop = cfg.get("property")
            if prop:
                try:
                    setattr(islat.molecules_dict, prop, value)
                except Exception as exc:
                    print(f"[SampleManagerWindow] apply error: {exc}")

        self._save_state()
        self._update_override_button(spectrum_path, len(ovr))

    def _is_active_path(self, spectrum_path: str) -> bool:
        """Return True when *spectrum_path* is the active sample entry."""
        islat = self._islat
        sample: List[str] = list(getattr(islat, "sample_spectra", []))
        cur: int = int(getattr(islat, "sample_spectra_index", 0))
        return bool(sample) and 0 <= cur < len(sample) and sample[cur] == spectrum_path

    def _update_override_button(self, spectrum_path: str, count: int) -> None:
        """Refresh the override toggle button badge without a full rebuild."""
        btn = self._ovr_buttons.get(spectrum_path)
        if btn is None:
            return
        arrow = "\u25be" if spectrum_path in self._expanded else "\u25b8"
        text = f"{arrow} Overrides ({count})" if count else f"{arrow} Overrides"
        try:
            btn.configure(text=text)
        except tk.TclError:
            pass

    def _save_state(self) -> None:
        """Persist the sample via the iSLAT instance when supported."""
        save = getattr(self._islat, "save_sample_state", None)
        if callable(save):
            try:
                save()
            except Exception as exc:
                print(f"[SampleManagerWindow] save error: {exc}")

    def _add_spectra(self) -> None:
        """Open a file dialog and add selected files to the sample."""
        from iSLAT.Modules.GUI import GUI as GUIModule
        from iSLAT.Modules.FileHandling.iSLATFileHandling import (
            example_data_folder_path,
        )

        paths = GUIModule.file_selector(
            title="Add Spectra to Sample",
            initialdir=example_data_folder_path,
            allow_multiple=True,
        )
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            self._islat.add_sample_spectra(list(paths))
        self._rebuild_rows()
        self._refresh_file_pane()

    def _clear_all(self) -> None:
        """Remove all spectra from the sample."""
        self._islat.clear_sample_spectra()
        self._rebuild_rows()
        self._refresh_file_pane()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refresh_file_pane(self) -> None:
        """Sync the FileInteractionPane label if the GUI is available."""
        try:
            islat = self._islat
            if (
                hasattr(islat, "GUI")
                and islat.GUI
                and hasattr(islat.GUI, "file_interaction_pane")
            ):
                islat.GUI.file_interaction_pane.update_file_label()
        except Exception:
            pass

    def _on_rows_configure(self, event: Any) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.itemconfig(
            self._rows_frame_id, width=self._canvas.winfo_width()
        )

    def _bind_scroll(self, widget: tk.Widget) -> None:
        """Recursively bind mouse-wheel scrolling to *widget* and children."""

        def _on_scroll(event: Any) -> None:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        widget.bind("<MouseWheel>", _on_scroll)
        # Tk 9 sends the trackpad as <TouchpadScroll>, not <MouseWheel>.
        bind_touchpad_scroll(widget, self._canvas)
        for child in widget.winfo_children():
            self._bind_scroll(child)

    def _on_close(self) -> None:
        key = id(self._islat)
        SampleManagerWindow._instances.pop(key, None)
        self._win.destroy()