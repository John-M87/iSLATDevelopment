"""
PopulationDiagramContextMixin - shared right-click context menu for the population diagram panel.

Both :class:`ThreePanelView` and :class:`PopulationDiagramView` inherit this mixin so that the *Color By* and *Axis Settings* dialogs, and the menu-building logic, live in exactly one place.

Usage::

    class MyView(PlotView, PopulationDiagramContextMixin):
        def build_context_menu(self, event, canvas_widget):
            pdp = ...  # get the PopulationDiagramPlot instance
            draw_idle = ...  # callable that flushes the canvas
            return self._build_population_diagram_menu(pdp, canvas_widget, draw_idle)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

class PopulationDiagramContextMixin:
    """Mixin that supplies the population-diagram right-click menu and dialogs.

    Methods intentionally receive ``pdp`` (a :class:`PopulationDiagramPlot`)
    and ``draw_idle`` (a zero-argument callable that flushes the canvas) as explicit parameters so the mixin stays decoupled from any particular view structure.
    """

    @staticmethod
    def _plot_species(pdp: Any) -> Optional[list]:
        """Species ids of the molecules on *pdp*, or ``None`` to offer everything.

        Passed to the :class:`PlotAxisRegistry` option lists so the axis,
        color-by and marker-shape menus only offer quantum numbers that the
        plotted molecules' schemas declare — the same scoping the line-list
        filter window applies.
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
        def _color_by_dialog():
            self._open_color_by_dialog(pdp, canvas_widget, draw_idle)

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

        menu.add_command(label="Color By\u2026", command=_color_by_dialog)
        menu.add_command(label="Clear Color Mapping", command=_clear_color_mapping)
        menu.add_command(
            label=("\u2713 " if _log_scale else "  ") + "Log Scale Colorbar",
            command=_toggle_log_scale,
            state="normal" if _has_mapping else "disabled",
        )
        menu.add_separator()
        menu.add_command(
            label="Axis Settings\u2026",
            command=lambda: self._open_axis_settings_dialog(pdp, canvas_widget, draw_idle),
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
    # Color By dialog
    # ------------------------------------------------------------------

    def _open_color_by_dialog(
        self,
        pdp: Any,
        parent_widget: Any,
        draw_idle: Callable[[], None],
    ) -> None:
        """Open the *Color By* dialog for *pdp*."""
        try:
            import tkinter as tk
            import tkinter.ttk as ttk
        except ImportError:
            return

        # Ensure quantum schemas for any currently-loaded molecule are
        # registered so their qn_upper/qn_lower entries appear in the list.
        if pdp is not None:
            try:
                import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401
            except Exception:
                pass

        from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry as _Reg

        color_entries = _Reg.get_color_options(species=self._plot_species(pdp))
        PROP_OPTIONS = [(e.display_name, e.key) for e in color_entries]
        PROP_LABELS  = [p[0] for p in PROP_OPTIONS]
        PROP_VALUES  = [p[1] for p in PROP_OPTIONS]

        CMAP_OPTIONS = [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "coolwarm", "RdYlBu", "Spectral", "tab10", "tab20",
        ]

        current_mapping = getattr(pdp, '_color_mapping', None) if pdp else None
        initial_prop_idx = 0
        initial_cmap = "viridis"
        initial_vmin = initial_vmax = initial_pmin = initial_pmax = ""
        initial_log_scale = False

        if current_mapping:
            prop_val = current_mapping.get("prop", "e_up")
            if prop_val in PROP_VALUES:
                initial_prop_idx = PROP_VALUES.index(prop_val)
            initial_cmap = current_mapping.get("cmap", "viridis")
            v = current_mapping.get("vmin")
            initial_vmin = str(v) if v is not None else ""
            v = current_mapping.get("vmax")
            initial_vmax = str(v) if v is not None else ""
            v = current_mapping.get("pmin")
            initial_pmin = str(v) if v is not None else ""
            v = current_mapping.get("pmax")
            initial_pmax = str(v) if v is not None else ""
            initial_log_scale = bool(current_mapping.get("log_scale", False))
        else:
            # No existing mapping: suggest log scale based on registry metadata.
            initial_prop = PROP_VALUES[initial_prop_idx]
            initial_log_scale = _Reg.suggests_log(initial_prop)

        win = tk.Toplevel(parent_widget)
        win.title("Population Diagram \u2014 Color By")
        win.resizable(False, False)
        win.grab_set()

        pad = {"padx": 8, "pady": 4}

        tk.Label(win, text="Color by property:").grid(row=0, column=0, sticky="w", **pad)
        prop_var = tk.StringVar(value=PROP_LABELS[initial_prop_idx])
        log_scale_var = tk.BooleanVar(value=initial_log_scale)

        def _on_prop_changed(*_):
            lbl = prop_var.get()
            if lbl in PROP_LABELS:
                pval = PROP_VALUES[PROP_LABELS.index(lbl)]
                if _Reg.suggests_log(pval):
                    log_scale_var.set(True)

        prop_combo = ttk.Combobox(win, textvariable=prop_var, values=PROP_LABELS,
                     state="readonly", width=32)
        prop_combo.grid(row=0, column=1, sticky="ew", **pad)
        prop_combo.bind("<<ComboboxSelected>>", _on_prop_changed)

        tk.Label(win, text="Colormap:").grid(row=1, column=0, sticky="w", **pad)
        cmap_var = tk.StringVar(value=initial_cmap)
        ttk.Combobox(win, textvariable=cmap_var, values=CMAP_OPTIONS,
                     state="normal", width=20).grid(row=1, column=1, sticky="ew", **pad)

        tk.Label(win, text="vmin (optional):").grid(row=2, column=0, sticky="w", **pad)
        vmin_var = tk.StringVar(value=initial_vmin)
        tk.Entry(win, textvariable=vmin_var, width=12).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(win, text="vmax (optional):").grid(row=3, column=0, sticky="w", **pad)
        vmax_var = tk.StringVar(value=initial_vmax)
        tk.Entry(win, textvariable=vmax_var, width=12).grid(row=3, column=1, sticky="w", **pad)

        ttk.Separator(win, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        tk.Label(win, text="\u2014 or clip by percentile (overrides vmin/vmax) \u2014",
                 font=("TkDefaultFont", 8), foreground="gray"
                 ).grid(row=5, column=0, columnspan=2, **pad)

        tk.Label(win, text="Min percentile % (optional):").grid(row=6, column=0, sticky="w", **pad)
        pmin_var = tk.StringVar(value=initial_pmin)
        tk.Spinbox(win, from_=0, to=100, increment=1, textvariable=pmin_var,
                   width=8).grid(row=6, column=1, sticky="w", **pad)

        tk.Label(win, text="Max percentile % (optional):").grid(row=7, column=0, sticky="w", **pad)
        pmax_var = tk.StringVar(value=initial_pmax)
        tk.Spinbox(win, from_=0, to=100, increment=1, textvariable=pmax_var,
                   width=8).grid(row=7, column=1, sticky="w", **pad)

        tk.Checkbutton(win, text="Log scale colorbar",
                       variable=log_scale_var).grid(row=8, column=0, columnspan=2,
                                                    sticky="w", **pad)

        btn_frame = tk.Frame(win)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=6)

        def _apply():
            prop_label = prop_var.get()
            if prop_label not in PROP_LABELS:
                return
            prop = PROP_VALUES[PROP_LABELS.index(prop_label)]
            cmap = cmap_var.get().strip() or "viridis"
            try:
                vmin = float(vmin_var.get()) if vmin_var.get().strip() else None
            except ValueError:
                vmin = None
            try:
                vmax = float(vmax_var.get()) if vmax_var.get().strip() else None
            except ValueError:
                vmax = None
            try:
                pmin = float(pmin_var.get()) if pmin_var.get().strip() else None
                if pmin is not None:
                    pmin = max(0.0, min(100.0, pmin))
            except ValueError:
                pmin = None
            try:
                pmax = float(pmax_var.get()) if pmax_var.get().strip() else None
                if pmax is not None:
                    pmax = max(0.0, min(100.0, pmax))
            except ValueError:
                pmax = None

            if pdp is not None:
                pdp.color_by(prop, cmap=cmap, vmin=vmin, vmax=vmax,
                             pmin=pmin, pmax=pmax,
                             log_scale=log_scale_var.get(), regenerate=True)
                draw_idle()
            win.destroy()

        def _cancel():
            win.destroy()

        ttk.Button(btn_frame, text="Apply",  command=_apply).pack(side="left",  padx=4)
        ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side="left", padx=4)

        win.bind("<Return>", lambda _e: _apply())
        win.bind("<Escape>", lambda _e: _cancel())

    # ------------------------------------------------------------------
    # Axis Settings dialog
    # ------------------------------------------------------------------
    def _open_axis_settings_dialog(
        self,
        pdp: Any,
        parent_widget: Any,
        draw_idle: Callable[[], None],
    ) -> None:
        """Open the *Axis Settings* dialog for *pdp*."""
        try:
            import tkinter as tk
            import tkinter.ttk as ttk
        except ImportError:
            return

        # Ensure quantum schemas for any currently-loaded molecule are
        # registered so their qn_upper/qn_lower entries appear in the list.
        if pdp is not None:
            try:
                import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401
            except Exception:
                pass

        from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry as _Reg

        axis_entries = _Reg.get_axis_options(species=self._plot_species(pdp))
        AXIS_OPTIONS = [(e.display_name, e.key) for e in axis_entries]
        AXIS_LABELS = [o[0] for o in AXIS_OPTIONS]
        AXIS_VALUES = [o[1] for o in AXIS_OPTIONS]

        cur_x     = getattr(pdp, '_x_prop', 'eu')     if pdp else 'eu'
        cur_y     = getattr(pdp, '_y_prop', 'rd_yax') if pdp else 'rd_yax'
        cur_x_log = getattr(pdp, '_x_log', False)     if pdp else False
        cur_y_log = getattr(pdp, '_y_log', False)     if pdp else False
        cur_x_lim = getattr(pdp, '_x_lim', None)      if pdp else None
        cur_y_lim = getattr(pdp, '_y_lim', None)      if pdp else None

        init_x_idx = AXIS_VALUES.index(cur_x) if cur_x in AXIS_VALUES else 1
        init_y_idx = AXIS_VALUES.index(cur_y) if cur_y in AXIS_VALUES else 0

        win = tk.Toplevel(parent_widget)
        win.title("Population Diagram \u2014 Axis Settings")
        win.resizable(False, False)
        win.grab_set()

        pad = {"padx": 8, "pady": 4}
        pad_tight = {"padx": 8, "pady": 1}

        # ---- X axis --------------------------------------------------
        tk.Label(win, text="X axis:").grid(row=0, column=0, sticky="w", **pad)
        x_var = tk.StringVar(value=AXIS_LABELS[init_x_idx])
        ttk.Combobox(win, textvariable=x_var, values=AXIS_LABELS,
                     state="readonly", width=38).grid(row=0, column=1, columnspan=3, sticky="ew", **pad)

        x_log_var = tk.BooleanVar(value=cur_x_log)
        tk.Checkbutton(win, text="Log scale (X)",
                       variable=x_log_var).grid(row=1, column=0, columnspan=4,
                                                sticky="w", **pad)

        # X limit mode
        _x_init_mode = cur_x_lim[0] if cur_x_lim else 'auto'
        _x_init_lo   = str(cur_x_lim[1]) if cur_x_lim and cur_x_lim[1] is not None else ''
        _x_init_hi   = str(cur_x_lim[2]) if cur_x_lim and cur_x_lim[2] is not None else ''

        x_mode_var = tk.StringVar(value=_x_init_mode)
        tk.Label(win, text="X limits:").grid(row=2, column=0, sticky="w", **pad_tight)
        x_mode_frame = tk.Frame(win)
        x_mode_frame.grid(row=2, column=1, columnspan=3, sticky="w", **pad_tight)
        for _mode, _label in [('auto', 'Auto'), ('exact', 'Exact values'), ('percentile', 'Percentile')]:
            tk.Radiobutton(x_mode_frame, text=_label, variable=x_mode_var,
                           value=_mode).pack(side='left', padx=4)

        x_lo_var = tk.StringVar(value=_x_init_lo)
        x_hi_var = tk.StringVar(value=_x_init_hi)
        x_lim_frame = tk.Frame(win)
        x_lim_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=16, pady=1)
        tk.Label(x_lim_frame, text="Min:").pack(side='left')
        tk.Entry(x_lim_frame, textvariable=x_lo_var, width=10).pack(side='left', padx=4)
        tk.Label(x_lim_frame, text="Max:").pack(side='left', padx=(8, 0))
        tk.Entry(x_lim_frame, textvariable=x_hi_var, width=10).pack(side='left', padx=4)
        tk.Label(x_lim_frame, text="(leave blank = auto for that bound)",
                 fg='grey').pack(side='left', padx=8)

        def _toggle_x_lim_frame(*_):
            if x_mode_var.get() == 'auto':
                for w in x_lim_frame.winfo_children():
                    w.configure(state='disabled') if hasattr(w, 'configure') else None
            else:
                for w in x_lim_frame.winfo_children():
                    try:
                        w.configure(state='normal')
                    except Exception:
                        pass
        x_mode_var.trace_add('write', _toggle_x_lim_frame)
        _toggle_x_lim_frame()

        ttk.Separator(win, orient="horizontal").grid(
            row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=4)

        # ---- Y axis --------------------------------------------------
        tk.Label(win, text="Y axis:").grid(row=5, column=0, sticky="w", **pad)
        y_var = tk.StringVar(value=AXIS_LABELS[init_y_idx])
        ttk.Combobox(win, textvariable=y_var, values=AXIS_LABELS,
                     state="readonly", width=38).grid(row=5, column=1, columnspan=3, sticky="ew", **pad)

        y_log_var = tk.BooleanVar(value=cur_y_log)
        tk.Checkbutton(win, text="Log scale (Y)",
                       variable=y_log_var).grid(row=6, column=0, columnspan=4,
                                                sticky="w", **pad)

        # Y limit mode
        _y_init_mode = cur_y_lim[0] if cur_y_lim else 'auto'
        _y_init_lo   = str(cur_y_lim[1]) if cur_y_lim and cur_y_lim[1] is not None else ''
        _y_init_hi   = str(cur_y_lim[2]) if cur_y_lim and cur_y_lim[2] is not None else ''

        y_mode_var = tk.StringVar(value=_y_init_mode)
        tk.Label(win, text="Y limits:").grid(row=7, column=0, sticky="w", **pad_tight)
        y_mode_frame = tk.Frame(win)
        y_mode_frame.grid(row=7, column=1, columnspan=3, sticky="w", **pad_tight)
        for _mode, _label in [('auto', 'Auto'), ('exact', 'Exact values'), ('percentile', 'Percentile')]:
            tk.Radiobutton(y_mode_frame, text=_label, variable=y_mode_var,
                           value=_mode).pack(side='left', padx=4)

        y_lo_var = tk.StringVar(value=_y_init_lo)
        y_hi_var = tk.StringVar(value=_y_init_hi)
        y_lim_frame = tk.Frame(win)
        y_lim_frame.grid(row=8, column=0, columnspan=4, sticky="ew", padx=16, pady=1)
        tk.Label(y_lim_frame, text="Min:").pack(side='left')
        tk.Entry(y_lim_frame, textvariable=y_lo_var, width=10).pack(side='left', padx=4)
        tk.Label(y_lim_frame, text="Max:").pack(side='left', padx=(8, 0))
        tk.Entry(y_lim_frame, textvariable=y_hi_var, width=10).pack(side='left', padx=4)
        tk.Label(y_lim_frame, text="(leave blank = auto for that bound)",
                 fg='grey').pack(side='left', padx=8)

        def _toggle_y_lim_frame(*_):
            if y_mode_var.get() == 'auto':
                for w in y_lim_frame.winfo_children():
                    try:
                        w.configure(state='disabled')
                    except Exception:
                        pass
            else:
                for w in y_lim_frame.winfo_children():
                    try:
                        w.configure(state='normal')
                    except Exception:
                        pass
        y_mode_var.trace_add('write', _toggle_y_lim_frame)
        _toggle_y_lim_frame()

        # ---- Marker shape by ----------------------------------------
        ttk.Separator(win, orient="horizontal").grid(
            row=9, column=0, columnspan=4, sticky="ew", padx=8, pady=4)

        cur_shape_mapping = getattr(pdp, '_shape_mapping', None) if pdp else None
        cur_shape_prop = cur_shape_mapping["prop"] if cur_shape_mapping else None
        cur_shape_bins = int(cur_shape_mapping.get("n_bins", 5)) if cur_shape_mapping else 5

        shape_labels = ["None"] + AXIS_LABELS
        shape_values = [None] + AXIS_VALUES
        init_shape_label = "None"
        if cur_shape_prop is not None and cur_shape_prop in AXIS_VALUES:
            init_shape_label = AXIS_LABELS[AXIS_VALUES.index(cur_shape_prop)]

        tk.Label(win, text="Marker shape by:").grid(row=10, column=0, sticky="w", **pad)
        shape_var = tk.StringVar(value=init_shape_label)
        shape_combo = ttk.Combobox(win, textvariable=shape_var, values=shape_labels,
                                   state="readonly", width=38)
        shape_combo.grid(row=10, column=1, columnspan=3, sticky="ew", **pad)

        bins_frame = tk.Frame(win)
        bins_frame.grid(row=11, column=0, columnspan=4, sticky="w", **pad_tight)
        tk.Label(bins_frame, text="Bins (continuous properties):").pack(side="left")
        shape_bins_var = tk.IntVar(value=cur_shape_bins)
        bins_spin = tk.Spinbox(bins_frame, from_=2, to=10, increment=1,
                               textvariable=shape_bins_var, width=5)
        bins_spin.pack(side="left", padx=4)

        def _toggle_bins(*_):
            sel = shape_var.get()
            if sel == "None":
                bins_spin.configure(state="disabled")
            else:
                prop_key = shape_values[shape_labels.index(sel)]
                from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry as _R2
                kind = _R2.get_kind(prop_key) if prop_key else "continuous"
                bins_spin.configure(state="disabled" if kind == "categorical" else "normal")

        shape_var.trace_add('write', _toggle_bins)
        _toggle_bins()

        # ---- Marker size override -----------------------------------
        ttk.Separator(win, orient="horizontal").grid(
            row=12, column=0, columnspan=4, sticky="ew", padx=8, pady=4)

        cur_marker_size = getattr(pdp, '_marker_size', None) if pdp else None
        init_size_str = str(int(cur_marker_size)) if cur_marker_size is not None else ""

        size_frame = tk.Frame(win)
        size_frame.grid(row=13, column=0, columnspan=4, sticky="w", **pad_tight)
        tk.Label(size_frame, text="Marker size (pt\u00b2, blank = auto):").pack(side="left")
        marker_size_var = tk.StringVar(value=init_size_str)
        tk.Entry(size_frame, textvariable=marker_size_var, width=7).pack(side="left", padx=4)

        btn_frame = tk.Frame(win)
        btn_frame.grid(row=14, column=0, columnspan=4, pady=8)

        def _parse_lim(mode_var, lo_var, hi_var):
            """Return None (auto) or a (mode, lo, hi) tuple."""
            mode = mode_var.get()
            if mode == 'auto':
                return None
            def _val(s):
                s = s.strip()
                return float(s) if s else None
            return (mode, _val(lo_var.get()), _val(hi_var.get()))

        def _apply():
            x_label = x_var.get()
            y_label = y_var.get()
            if x_label not in AXIS_LABELS or y_label not in AXIS_LABELS:
                return
            x_prop = AXIS_VALUES[AXIS_LABELS.index(x_label)]
            y_prop = AXIS_VALUES[AXIS_LABELS.index(y_label)]
            if pdp is not None:
                pdp.set_axes(
                    x_prop=x_prop, y_prop=y_prop,
                    x_log=x_log_var.get(), y_log=y_log_var.get(),
                    x_lim=_parse_lim(x_mode_var, x_lo_var, x_hi_var),
                    y_lim=_parse_lim(y_mode_var, y_lo_var, y_hi_var),
                    regenerate=False,
                )
                # Apply shape mapping before triggering regeneration
                shape_sel = shape_var.get()
                if shape_sel == "None":
                    pdp.clear_shape_mapping(regenerate=False)
                else:
                    shape_prop = shape_values[shape_labels.index(shape_sel)]
                    pdp.shape_by(shape_prop, n_bins=shape_bins_var.get(), regenerate=False)
                # Apply marker size override
                _sz_str = marker_size_var.get().strip()
                if _sz_str:
                    try:
                        pdp.set_marker_size(float(_sz_str), regenerate=False)
                    except ValueError:
                        pass
                else:
                    pdp.clear_marker_size(regenerate=False)
                pdp.generate_plot()
                draw_idle()
            win.destroy()

        def _reset():
            if pdp is not None:
                pdp.clear_shape_mapping(regenerate=False)
                pdp.clear_marker_size(regenerate=False)
                pdp.set_axes(regenerate=True)
                draw_idle()
            win.destroy()

        def _cancel():
            win.destroy()

        ttk.Button(btn_frame, text="Apply",          command=_apply).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset Defaults",  command=_reset).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel",          command=_cancel).pack(side="left", padx=4)

        win.bind("<Return>", lambda _e: _apply())
        win.bind("<Escape>", lambda _e: _cancel())