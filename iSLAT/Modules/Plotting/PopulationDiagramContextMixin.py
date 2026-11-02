"""
PopulationDiagramContextMixin - shared right-click context menu for the population diagram panel.

Both :class:`ThreePanelView` and :class:`PopulationDiagramView` inherit this mixin so that the *Plot Settings* dialog (axes, color and marker settings in one tabbed window), and the menu-building logic, live in exactly one place.

Usage::

    class MyView(PlotView, PopulationDiagramContextMixin):
        def build_context_menu(self, event, canvas_widget):
            pdp = ...  # get the PopulationDiagramPlot instance
            draw_idle = ...  # callable that flushes the canvas
            return self._build_population_diagram_menu(pdp, canvas_widget, draw_idle)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional

class PopulationDiagramContextMixin:
    """Mixin that supplies the population-diagram right-click menu and dialogs.

    Methods intentionally receive ``pdp`` (a :class:`PopulationDiagramPlot`)
    and ``draw_idle`` (a zero-argument callable that flushes the canvas) as explicit parameters so the mixin stays decoupled from any particular view structure.
    """

    @staticmethod
    def _plot_species(pdp: Any) -> Optional[list]:
        """Species ids of the molecules on *pdp*, or ``None`` to offer everything.

        Passed to :meth:`PlotAxisRegistry.get_property_options` so the Plot
        Settings menus only offer quantum numbers that the plotted molecules'
        schemas declare — the same scoping the line-list filter window applies.
        """
        if pdp is None or not hasattr(pdp, "get_molecule_ids"):
            return None
        try:
            return list(pdp.get_molecule_ids())
        except Exception:
            return None
    # ------------------------------------------------------------------
    # Public helper - build the tk.Menu
    # ------------------------------------------------------------------
    def _build_population_diagram_menu(
        self,
        pdp: Any,
        canvas_widget: Any,
        draw_idle: Callable[[], None],
    ) -> Optional[Any]:
        """Return a populated ``tk.Menu`` for the population diagram, or ``None``.

        Parameters
        ----------
        pdp :
            The active :class:`PopulationDiagramPlot` instance, or ``None``
            if no diagram has been rendered yet.
        canvas_widget :
            The Tk widget that acts as the menu's parent.
        draw_idle :
            Zero-argument callable that triggers a canvas redraw
            (typically ``canvas.draw_idle``).
        """
        try:
            import tkinter as tk
        except ImportError:
            return None

        menu = tk.Menu(canvas_widget, tearoff=0)

        # ---- helpers --------------------------------------------------
        def _plot_settings_dialog():
            self._open_plot_settings_dialog(pdp, canvas_widget, draw_idle)

        def _clear_color_mapping():
            if pdp is not None:
                pdp.clear_color_mapping(regenerate=True)
                draw_idle()

        def _toggle_log_scale():
            if pdp is None or getattr(pdp, '_color_mapping', None) is None:
                return
            current = pdp._color_mapping.get("log_scale", False)
            pdp._color_mapping["log_scale"] = not current
            pdp.generate_plot()
            draw_idle()

        def _show_all_active_molecules():
            if pdp is None:
                return
            islat = getattr(self, '_islat', None)
            molecules_dict = getattr(islat, 'molecules_dict', None) if islat else None
            if molecules_dict is None:
                return
            pdp.set_molecules(molecules_dict)
            draw_idle()

        def _show_active_molecule_only():
            if pdp is None:
                return
            islat = getattr(self, '_islat', None)
            active_mol = getattr(islat, 'active_molecule', None) if islat else None
            if active_mol is None:
                return
            if isinstance(active_mol, str):
                molecules_dict = getattr(islat, 'molecules_dict', None)
                if molecules_dict is not None and active_mol in molecules_dict:
                    active_mol = molecules_dict[active_mol]
                else:
                    return
            pdp.set_molecule(active_mol)
            draw_idle()

        # ---- state indicators -----------------------------------------

        _all_mode = pdp is not None and getattr(pdp, '_all_molecules_mode', False)
        _has_mapping = pdp is not None and getattr(pdp, '_color_mapping', None) is not None
        _log_scale = _has_mapping and pdp._color_mapping.get("log_scale", False)

        # ---- menu items -----------------------------------------------

        menu.add_command(label="Plot Settings\u2026", command=_plot_settings_dialog)
        menu.add_command(label="Clear Color Mapping", command=_clear_color_mapping)
        menu.add_command(
            label=("\u2713 " if _log_scale else "  ") + "Log Scale Colorbar",
            command=_toggle_log_scale,
            state="normal" if _has_mapping else "disabled",
        )
        menu.add_separator()
        menu.add_command(
            label=("\u2713 " if _all_mode else "  ") + "Show All Active Molecules",
            command=_show_all_active_molecules,
        )
        menu.add_command(
            label=("\u2713 " if not _all_mode else "  ") + "Show Active Molecule Only",
            command=_show_active_molecule_only,
        )
        menu.add_separator()

        # --- Save active lines as line list ----------------------------
        _active_info = self._get_active_lines_info(pdp)

        def _save_line_list():
            self._save_active_lines_as_line_list(canvas_widget)

        menu.add_command(
            label="Save Active Lines as Line List\u2026",
            command=_save_line_list,
            state="normal" if _active_info else "disabled",
        )

        # ------------------------------------------------------------------
        # View switching
        # ------------------------------------------------------------------
        pm = getattr(self, '_pm', None)
        if pm is not None and hasattr(pm, 'switch_view'):
            menu.add_separator()

            active_name = getattr(pm, 'active_view_name', None)

            if active_name == "Three Panel":
                # Inside the three-panel layout - offer to pop out to standalone views
                def _to_population_diagram():
                    pm.switch_view("Population Diagram")

                def _to_line_inspection():
                    pm.switch_view("Line Inspection")

                menu.add_command(
                    label="Switch to Population Diagram View",
                    command=_to_population_diagram,
                )
                menu.add_command(
                    label="Switch to Line Inspection View",
                    command=_to_line_inspection,
                )
            else:
                # Already in a standalone view - offer to go back or cross-navigate
                def _to_three_panel():
                    pm.switch_view("Three Panel")

                def _to_line_inspection():
                    pm.switch_view("Line Inspection")

                menu.add_command(
                    label="Switch to Three Panel View",
                    command=_to_three_panel,
                )
                menu.add_command(
                    label="Switch to Line Inspection View",
                    command=_to_line_inspection,
                )

        return menu

    # ------------------------------------------------------------------
    # Save active lines as line list
    # ------------------------------------------------------------------
    def _get_active_lines_info(self, pdp: Any) -> list:
        """Return the list of ``info_dict`` entries for currently active lines.

        Tries, in order:
        1. ``self.active_lines`` - populated by :class:`ThreePanelView`.
        2. ``pdp._active_lines_cache`` - populated by the standalone
           :class:`PopulationDiagramView`.

        Returns an empty list when no active lines are available.
        """
        # ThreePanelView path
        own_lines = getattr(self, "active_lines", None)
        if own_lines:
            return [entry[3] for entry in own_lines if entry[3] is not None]

        # Standalone PopulationDiagramView path
        if pdp is not None:
            cache = getattr(pdp, "_active_lines_cache", None)
            if cache:
                cached_entries = cache.get("active_lines", [])
                if cached_entries:
                    return [entry[3] for entry in cached_entries if entry[3] is not None]

        return []

    def _save_active_lines_as_line_list(self, parent_widget: Any) -> None:
        """Open a save-as dialog and write the currently active lines to a CSV line list file compatible with iSLAT's fitting workflow.

        Each row contains the columns produced by
        :meth:`LineSaveService.format_line_for_save`:
        ``species``, ``lam``, ``xmin``, ``xmax``, ``lev_up``,
        ``lev_low``, ``tau``, ``intens``, ``a_stein``,
        ``e_up``, ``g_up``, ``e_low``, ``g_low``.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            import pandas as pd
        except ImportError:
            return

        # Resolve the active pdp so we can read its cache
        pdp = getattr(self, "_plot", None)
        info_list = self._get_active_lines_info(pdp)

        if not info_list:
            messagebox.showinfo(
                "No Active Lines",
                "No highlighted lines are available to save.\n"
                "Select a wavelength range first to activate lines.",
                parent=parent_widget,
            )
            return

        # Build rows - derive xmin/xmax as a ±0.015 µm window if not present
        _DEFAULT_HALF_WIN = 0.015
        rows = []
        for info in info_list:
            lam = info.get("lam")
            if lam is None:
                continue
            lam = float(lam)
            rows.append({
                "species":  info.get("molecule_name", ""),
                "lam":      lam,
                "xmin":     float(info.get("xmin", lam - _DEFAULT_HALF_WIN)),
                "xmax":     float(info.get("xmax", lam + _DEFAULT_HALF_WIN)),
                "lev_up":   info.get("up_lev", ""),
                "lev_low":  info.get("low_lev", ""),
                "tau":      info.get("tau", 0.0),
                "intens":   info.get("intensity", info.get("inten", 0.0)),
                "a_stein":  info.get("a_stein", 0.0),
                "e_up":     info.get("e_up", 0.0),
                "g_up":     info.get("g_up", 1.0),
                "e_low":    info.get("e_low", 0.0),
                "g_low":    info.get("g_low", 1.0),
            })

        if not rows:
            messagebox.showinfo(
                "No Valid Lines",
                "None of the active lines contained wavelength data.",
                parent=parent_widget,
            )
            return

        df = pd.DataFrame(rows)

        filepath = filedialog.asksaveasfilename(
            title="Save Active Lines as Line List",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            initialfile="active_lines.csv",
            parent=parent_widget,
        )
        if not filepath:
            return

        try:
            df.to_csv(filepath, index=False)
            messagebox.showinfo(
                "Saved",
                f"Saved {len(df)} line(s) to:\n{filepath}",
                parent=parent_widget,
            )
        except Exception as exc:
            messagebox.showerror(
                "Save Failed",
                f"Could not write file:\n{exc}",
                parent=parent_widget,
            )

    # ------------------------------------------------------------------
    # Plot Settings dialog (axes, color and markers in one window)
    # ------------------------------------------------------------------

    def _open_plot_settings_dialog(
        self,
        pdp: Any,
        parent_widget: Any,
        draw_idle: Callable[[], None],
        initial_tab: str = "axes",
    ) -> None:
        """Open the *Plot Settings* dialog for *pdp*.

        One window with *Axes*, *Color* and *Markers* tabs and a single
        Apply.  Every property menu in it offers the same list
        (:meth:`PlotAxisRegistry.get_property_options`), which is also the
        list the line-list filter offers.
        """
        try:
            import tkinter as tk
            import tkinter.ttk as ttk
        except ImportError:
            return

        # Ensure quantum schemas for any currently-loaded molecule are
        # registered so their qn_upper/qn_lower entries appear in the list.
        try:
            import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401
        except Exception:
            pass

        from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry as _Reg

        entries = _Reg.get_property_options(species=self._plot_species(pdp))
        labels = [e.display_name for e in entries]
        label_to_key = {e.display_name: e.key for e in entries}
        key_to_label = {e.key: e.display_name for e in entries}
        if not labels:
            return

        current = capture_plot_settings(pdp)

        def _label_for(key: Optional[str], fallback: str) -> str:
            if not key:
                return fallback
            return key_to_label.get(_Reg.canonical_key(key), fallback)

        def _kind(label: str) -> Optional[str]:
            key = label_to_key.get(label)
            return None if key is None else _Reg.get_kind(key)

        def _set_state(widgets, enabled: bool) -> None:
            for w in widgets:
                try:
                    w.configure(state="normal" if enabled else "disabled")
                except Exception:
                    pass

        def _fmt(value: Any) -> str:
            return "" if value is None else f"{value:g}" if isinstance(value, float) else str(value)

        win = tk.Toplevel(parent_widget)
        win.title("Population Diagram \u2014 Plot Settings")
        win.resizable(False, False)
        win.grab_set()

        pad = {"padx": 8, "pady": 4}
        pad_tight = {"padx": 8, "pady": 1}

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        axes_tab = ttk.Frame(notebook, padding=6)
        color_tab = ttk.Frame(notebook, padding=6)
        markers_tab = ttk.Frame(notebook, padding=6)
        notebook.add(axes_tab, text="Axes")
        notebook.add(color_tab, text="Color")
        notebook.add(markers_tab, text="Markers")

        form: Dict[str, Any] = {}

        # ---- Axes tab ------------------------------------------------
        row = 0
        for axis, name in (("x", "X"), ("y", "Y")):
            if axis == "y":
                ttk.Separator(axes_tab, orient="horizontal").grid(
                    row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
                row += 1
            prop_var = tk.StringVar(value=_label_for(current[f"{axis}_prop"], labels[0]))
            ttk.Label(axes_tab, text=f"{name} axis:").grid(row=row, column=0, sticky="w", **pad)
            ttk.Combobox(axes_tab, textvariable=prop_var, values=labels,
                         state="readonly", width=38).grid(
                row=row, column=1, columnspan=3, sticky="ew", **pad)

            log_var = tk.BooleanVar(value=bool(current[f"{axis}_log"]))
            log_chk = ttk.Checkbutton(axes_tab, text=f"Log scale ({name})", variable=log_var)
            log_chk.grid(row=row + 1, column=0, columnspan=4, sticky="w", **pad)

            lim = current[f"{axis}_lim"]
            mode_var = tk.StringVar(value=lim[0] if lim else "auto")
            lo_var = tk.StringVar(value=_fmt(lim[1]) if lim else "")
            hi_var = tk.StringVar(value=_fmt(lim[2]) if lim else "")
            ttk.Label(axes_tab, text=f"{name} limits:").grid(
                row=row + 2, column=0, sticky="w", **pad_tight)
            mode_frame = ttk.Frame(axes_tab)
            mode_frame.grid(row=row + 2, column=1, columnspan=3, sticky="w", **pad_tight)
            for mode, text in (("auto", "Auto"), ("exact", "Exact values"),
                               ("percentile", "Percentile")):
                ttk.Radiobutton(mode_frame, text=text, variable=mode_var,
                                value=mode).pack(side="left", padx=4)

            lim_frame = ttk.Frame(axes_tab)
            lim_frame.grid(row=row + 3, column=0, columnspan=4, sticky="ew", padx=16, pady=1)
            ttk.Label(lim_frame, text="Min:").pack(side="left")
            lo_entry = ttk.Entry(lim_frame, textvariable=lo_var, width=10)
            lo_entry.pack(side="left", padx=4)
            ttk.Label(lim_frame, text="Max:").pack(side="left", padx=(8, 0))
            hi_entry = ttk.Entry(lim_frame, textvariable=hi_var, width=10)
            hi_entry.pack(side="left", padx=4)
            ttk.Label(lim_frame, text="(leave blank = auto for that bound)",
                      foreground="grey").pack(side="left", padx=8)

            def _sync_axis(*_, pv=prop_var, chk=log_chk, mv=mode_var,
                           bounds=(lo_entry, hi_entry)):
                # A category axis is always linear.
                _set_state([chk], _kind(pv.get()) != "categorical")
                _set_state(bounds, mv.get() != "auto")

            prop_var.trace_add("write", _sync_axis)
            mode_var.trace_add("write", _sync_axis)
            _sync_axis()

            form.update({
                f"{axis}_prop": prop_var, f"{axis}_log": log_var,
                f"{axis}_lim_mode": mode_var, f"{axis}_lim_lo": lo_var,
                f"{axis}_lim_hi": hi_var,
            })
            row += 4

        ttk.Label(axes_tab, text="Categorical properties (labels, band, molecule) "
                                 "are drawn as a category axis.",
                  foreground="grey").grid(row=row, column=0, columnspan=4, sticky="w", **pad)

        # ---- Color tab -----------------------------------------------
        color = current.get("color") or {}
        color_var = tk.StringVar(value=_label_for(color.get("prop"), NONE_LABEL))
        ttk.Label(color_tab, text="Color by:").grid(row=0, column=0, sticky="w", **pad)
        color_combo = ttk.Combobox(color_tab, textvariable=color_var,
                                   values=[NONE_LABEL] + labels, state="readonly", width=38)
        color_combo.grid(row=0, column=1, sticky="ew", **pad)

        cmap_var = tk.StringVar(value=color.get("cmap") or "viridis")
        ttk.Label(color_tab, text="Colormap:").grid(row=1, column=0, sticky="w", **pad)
        cmap_combo = ttk.Combobox(color_tab, textvariable=cmap_var, values=CMAP_OPTIONS,
                                  state="normal", width=20)
        cmap_combo.grid(row=1, column=1, sticky="w", **pad)

        vmin_var = tk.StringVar(value=_fmt(color.get("vmin")))
        vmax_var = tk.StringVar(value=_fmt(color.get("vmax")))
        ttk.Label(color_tab, text="vmin (optional):").grid(row=2, column=0, sticky="w", **pad)
        vmin_entry = ttk.Entry(color_tab, textvariable=vmin_var, width=12)
        vmin_entry.grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(color_tab, text="vmax (optional):").grid(row=3, column=0, sticky="w", **pad)
        vmax_entry = ttk.Entry(color_tab, textvariable=vmax_var, width=12)
        vmax_entry.grid(row=3, column=1, sticky="w", **pad)

        ttk.Separator(color_tab, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        ttk.Label(color_tab, text="\u2014 or clip by percentile (overrides vmin/vmax) \u2014",
                  foreground="gray").grid(row=5, column=0, columnspan=2, **pad)

        pmin_var = tk.StringVar(value=_fmt(color.get("pmin")))
        pmax_var = tk.StringVar(value=_fmt(color.get("pmax")))
        ttk.Label(color_tab, text="Min percentile % (optional):").grid(
            row=6, column=0, sticky="w", **pad)
        pmin_spin = ttk.Spinbox(color_tab, from_=0, to=100, increment=1,
                                textvariable=pmin_var, width=8)
        pmin_spin.grid(row=6, column=1, sticky="w", **pad)
        ttk.Label(color_tab, text="Max percentile % (optional):").grid(
            row=7, column=0, sticky="w", **pad)
        pmax_spin = ttk.Spinbox(color_tab, from_=0, to=100, increment=1,
                                textvariable=pmax_var, width=8)
        pmax_spin.grid(row=7, column=1, sticky="w", **pad)

        color_log_var = tk.BooleanVar(value=bool(color.get("log_scale", False)))
        color_log_chk = ttk.Checkbutton(color_tab, text="Log scale colorbar",
                                        variable=color_log_var)
        color_log_chk.grid(row=8, column=0, columnspan=2, sticky="w", **pad)

        def _sync_color(*_):
            kind = _kind(color_var.get())
            _set_state([cmap_combo], kind is not None)
            # Value limits and log scaling only apply to a continuous colormap.
            _set_state([vmin_entry, vmax_entry, pmin_spin, pmax_spin, color_log_chk],
                       kind == "continuous")

        def _on_color_selected(*_):
            key = label_to_key.get(color_var.get())
            if key is not None:
                color_log_var.set(_Reg.suggests_log(key))
            _sync_color()

        color_combo.bind("<<ComboboxSelected>>", _on_color_selected)
        _sync_color()

        form.update({
            "color_prop": color_var, "cmap": cmap_var, "vmin": vmin_var,
            "vmax": vmax_var, "pmin": pmin_var, "pmax": pmax_var,
            "color_log": color_log_var,
        })

        # ---- Markers tab ---------------------------------------------
        shape = current.get("shape") or {}
        shape_var = tk.StringVar(value=_label_for(shape.get("prop"), NONE_LABEL))
        ttk.Label(markers_tab, text="Marker shape by:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(markers_tab, textvariable=shape_var, values=[NONE_LABEL] + labels,
                     state="readonly", width=38).grid(row=0, column=1, sticky="ew", **pad)

        shape_bins_var = tk.StringVar(value=str(int(shape.get("n_bins", 5))))
        ttk.Label(markers_tab, text="Bins (continuous properties):").grid(
            row=1, column=0, sticky="w", **pad)
        bins_spin = ttk.Spinbox(markers_tab, from_=2, to=10, increment=1,
                                textvariable=shape_bins_var, width=5)
        bins_spin.grid(row=1, column=1, sticky="w", **pad)

        def _sync_shape(*_):
            _set_state([bins_spin], _kind(shape_var.get()) == "continuous")

        shape_var.trace_add("write", _sync_shape)
        _sync_shape()

        ttk.Separator(markers_tab, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        size_var = tk.StringVar(value=_fmt(current.get("marker_size")))
        ttk.Label(markers_tab, text="Marker size (pt\u00b2, blank = auto):").grid(
            row=3, column=0, sticky="w", **pad)
        ttk.Entry(markers_tab, textvariable=size_var, width=7).grid(
            row=3, column=1, sticky="w", **pad)

        form.update({"shape_prop": shape_var, "shape_bins": shape_bins_var,
                     "marker_size": size_var})

        # ---- Buttons -------------------------------------------------
        status = ttk.Label(win, text="", foreground="red")
        status.pack(fill="x", padx=12)
        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=(2, 8))

        def _apply(*_):
            values = {name: var.get() for name, var in form.items()}
            try:
                settings = parse_plot_settings(values, label_to_key)
            except ValueError as exc:
                status.configure(text=str(exc))
                return
            if pdp is not None:
                apply_plot_settings(pdp, settings)
                draw_idle()
            win.destroy()

        def _reset():
            if pdp is not None:
                apply_plot_settings(pdp, DEFAULT_PLOT_SETTINGS)
                draw_idle()
            win.destroy()

        ttk.Button(btn_frame, text="Apply", command=_apply).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset Defaults", command=_reset).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side="left", padx=4)

        tabs = {"axes": axes_tab, "color": color_tab, "markers": markers_tab}
        notebook.select(tabs.get(initial_tab, axes_tab))

        win.bind("<Return>", _apply)
        win.bind("<Escape>", lambda _e: win.destroy())


# ======================================================================
# Plot settings: the state the Plot Settings dialog edits
# ======================================================================
#: Label of the "no mapping" choice in the Color / Marker shape menus.
NONE_LABEL = "None"

#: Colormaps offered in the Color tab (any matplotlib name may be typed).
CMAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdYlBu", "Spectral", "tab10", "tab20",
]

#: The diagram's settings when nothing has been changed.
DEFAULT_PLOT_SETTINGS: Dict[str, Any] = {
    "x_prop": "eu", "y_prop": "rd_yax",
    "x_log": False, "y_log": False,
    "x_lim": None, "y_lim": None,
    "color": None, "shape": None, "marker_size": None,
}


def capture_plot_settings(pdp: Any) -> Dict[str, Any]:
    """Return *pdp*'s user-adjustable settings (defaults when *pdp* is ``None``).

    ``color`` / ``shape`` are copies of the plot's color / shape mapping
    dicts, or ``None`` when that mapping is off.
    """
    if pdp is None:
        return dict(DEFAULT_PLOT_SETTINGS)
    color = getattr(pdp, "_color_mapping", None)
    shape = getattr(pdp, "_shape_mapping", None)
    return {
        "x_prop": getattr(pdp, "_x_prop", "eu"),
        "y_prop": getattr(pdp, "_y_prop", "rd_yax"),
        "x_log": bool(getattr(pdp, "_x_log", False)),
        "y_log": bool(getattr(pdp, "_y_log", False)),
        "x_lim": getattr(pdp, "_x_lim", None),
        "y_lim": getattr(pdp, "_y_lim", None),
        "color": dict(color) if color else None,
        "shape": dict(shape) if shape else None,
        "marker_size": getattr(pdp, "_marker_size", None),
    }


def parse_plot_settings(values: Mapping[str, Any],
                        label_to_key: Mapping[str, str]) -> Dict[str, Any]:
    """Turn the Plot Settings form *values* into a settings dict.

    *values* maps the form field names to their widget values (property
    menus hold display labels); *label_to_key* maps those labels to
    property keys.  Raises :class:`ValueError` with a user-facing message
    when a field is invalid.
    """
    def _number(field: str, what: str) -> Optional[float]:
        text = str(values.get(field, "") or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{what} must be a number (got {text!r}).") from None

    def _prop(field: str, what: str, allow_none: bool) -> Optional[str]:
        label = str(values.get(field, "") or "")
        if allow_none and label in ("", NONE_LABEL):
            return None
        if label not in label_to_key:
            raise ValueError(f"Choose a property for {what}.")
        return label_to_key[label]

    settings: Dict[str, Any] = {}
    for axis, name in (("x", "X"), ("y", "Y")):
        settings[f"{axis}_prop"] = _prop(f"{axis}_prop", f"the {name} axis", False)
        settings[f"{axis}_log"] = bool(values.get(f"{axis}_log", False))
        mode = str(values.get(f"{axis}_lim_mode", "auto") or "auto")
        if mode == "auto":
            settings[f"{axis}_lim"] = None
        else:
            lo = _number(f"{axis}_lim_lo", f"{name} min")
            hi = _number(f"{axis}_lim_hi", f"{name} max")
            if mode == "percentile" and any(b is not None and not 0 <= b <= 100
                                            for b in (lo, hi)):
                raise ValueError(f"{name} percentile limits must be between 0 and 100.")
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"{name} min must be less than {name} max.")
            settings[f"{axis}_lim"] = (mode, lo, hi)

    color_prop = _prop("color_prop", "Color by", True)
    if color_prop is None:
        settings["color"] = None
    else:
        pmin = _number("pmin", "Min percentile")
        pmax = _number("pmax", "Max percentile")
        settings["color"] = {
            "prop": color_prop,
            "cmap": str(values.get("cmap", "") or "").strip() or "viridis",
            "vmin": _number("vmin", "vmin"),
            "vmax": _number("vmax", "vmax"),
            "pmin": None if pmin is None else max(0.0, min(100.0, pmin)),
            "pmax": None if pmax is None else max(0.0, min(100.0, pmax)),
            "log_scale": bool(values.get("color_log", False)),
        }

    shape_prop = _prop("shape_prop", "Marker shape", True)
    if shape_prop is None:
        settings["shape"] = None
    else:
        bins = _number("shape_bins", "Bins")
        settings["shape"] = {"prop": shape_prop,
                             "n_bins": int(bins) if bins is not None else 5}

    size = _number("marker_size", "Marker size")
    if size is not None and size <= 0:
        raise ValueError("Marker size must be greater than 0.")
    settings["marker_size"] = size
    return settings


def apply_plot_settings(pdp: Any, settings: Mapping[str, Any], *,
                        regenerate: bool = True) -> None:
    """Apply a settings dict (see :func:`capture_plot_settings`) to *pdp* in one redraw."""
    pdp.set_axes(
        x_prop=settings.get("x_prop") or "eu",
        y_prop=settings.get("y_prop") or "rd_yax",
        x_log=bool(settings.get("x_log", False)),
        y_log=bool(settings.get("y_log", False)),
        x_lim=settings.get("x_lim"),
        y_lim=settings.get("y_lim"),
        regenerate=False,
    )
    color = settings.get("color")
    if color:
        pdp.color_by(
            color["prop"],
            cmap=color.get("cmap") or "viridis",
            vmin=color.get("vmin"), vmax=color.get("vmax"),
            pmin=color.get("pmin"), pmax=color.get("pmax"),
            log_scale=bool(color.get("log_scale", False)),
            regenerate=False,
        )
    else:
        pdp.clear_color_mapping(regenerate=False)
    shape = settings.get("shape")
    if shape:
        pdp.shape_by(shape["prop"], n_bins=int(shape.get("n_bins", 5)), regenerate=False)
    else:
        pdp.clear_shape_mapping(regenerate=False)
    size = settings.get("marker_size")
    if size is None:
        pdp.clear_marker_size(regenerate=False)
    else:
        pdp.set_marker_size(float(size), regenerate=False)
    if regenerate:
        pdp.generate_plot()
