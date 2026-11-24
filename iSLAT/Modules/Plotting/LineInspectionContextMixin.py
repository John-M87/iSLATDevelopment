"""
LineInspectionContextMixin - shared right-click context menu for the
line inspection panel (ax2).

:class:`ThreePanelView` inherits this mixin so that the line-inspection
context menu lives in one place and can be reused by any future view that
embeds an ax2-style line inspection panel.

Usage::

    class MyView(PlotView, LineInspectionContextMixin):
        def build_context_menu(self, event, canvas_widget):
            if event.inaxes == self.ax2:
                return self._build_line_inspection_menu(canvas_widget, event)
            return None
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from iSLAT.Modules.DataProcessing.LineIdentity import LineRef, match_line
from iSLAT.Modules.GUI.KeyBindings import with_shortcut

from .ArtistTags import SUMMED_TAG, iter_tagged
from .ModelOverlays import draw_summed_fill
from .OwnedCanvasMixin import add_view_switch_items

#: Menu label of "Show in Line List" when there is no line to show.
SHOW_IN_LINE_LIST_LABEL = "Show in Line List"

SELECTED_LINE_COLOR = "orange"
DEFAULT_LINE_THRESHOLD = 0.3  # when user_settings has no line_threshold


def show_in_line_list_label(ref: Optional[LineRef]) -> str:
    """Menu label for showing *ref* in its line list (``Show H2O 17.2345 um in Line List``)."""
    if ref is None:
        return SHOW_IN_LINE_LIST_LABEL
    return f"Show {ref.describe()} in Line List"


class LineInspectionContextMixin:
    """Mixin that supplies the line inspection panel right-click menu.

    Requires ``self._islat`` (the iSLAT application) and ``self._pm`` (the plot manager).
    The line markers are read from ``self.active_lines`` (``[vline, text, scatter, info]`` entries)
    and the selected line from ``self.selected_line``.  Also holds the marker
    threshold, colors, highlight and line printout the views share.
    """
    def _get_line_threshold(self) -> float:
        """Marker threshold (0-1, of the strongest line in range) from ``user_settings``."""
        settings = getattr(self._islat, "user_settings", None) or {}
        return settings.get("line_threshold", DEFAULT_LINE_THRESHOLD)

    def _marker_max_y(self, xmin: float, xmax: float, fallback: float = 1.0) -> float:
        """Peak observed flux in ``[xmin, xmax]``, the height the markers scale to."""
        wave = self._islat.wave_data
        region = self._islat.flux_data[(wave >= xmin) & (wave <= xmax)]
        return float(np.nanmax(region)) if len(region) > 0 else fallback

    def _molecule_line_color(self, molecule: Any = None) -> Any:
        """*molecule*'s color, else the theme's active-line color."""
        theme = getattr(getattr(self, "_pm", None), "theme", None) or {}
        return getattr(molecule, "color", None) or theme.get("active_scatter_line_color", "green")

    def _recolor_active_lines(
        self,
        selected: Optional[Dict[str, Any]] = None,
        *,
        match_wavelength: bool = False,
        scatter_zorder: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Color every marker and scatter point by molecule, and the *selected* line orange.

        *selected* is matched by identity, or by ``lam`` with *match_wavelength*
        (after a re-render).  Returns the highlighted info dict, or ``None``.
        """
        import matplotlib.colors as mcolors

        hit = None
        if selected is not None:
            hit = next((v for *_, v in self.active_lines if v is not None and (
                v.get("lam") == selected.get("lam") if match_wavelength else v is selected)), None)

        default = self._molecule_line_color()
        for line, text_obj, _scatter, value in self.active_lines:
            color = SELECTED_LINE_COLOR if hit is not None and value is hit else (
                (value or {}).get("molecule_color") or default)
            for artist in (line, text_obj):
                if artist is not None:
                    artist.set_color(color)

        mol_dict = getattr(self._islat, "molecules_dict", None) or {}
        hit_info = hit or {}
        hit_index = hit_info.get("_scatter_point_index")
        for mol_name, (sc, count) in (getattr(self, "_active_scatter_collections", None) or {}).items():
            if count <= 0:
                continue
            colors = [mcolors.to_rgba(self._molecule_line_color(mol_dict.get(mol_name)))] * count
            if mol_name == hit_info.get("molecule_name") and hit_index is not None and hit_index < count:
                colors[hit_index] = mcolors.to_rgba(SELECTED_LINE_COLOR)
            sc.set_facecolors(colors)
            if scatter_zorder is not None:
                sc.set_zorder(scatter_zorder)
        return hit

    def _line_info_selection(self) -> Optional[Tuple[float, float]]:
        """The range the line printout integrates over (``None``: none)."""
        return self._current_selection

    def _display_line_info(self, value: Dict[str, Any], clear_data_field: bool = True) -> None:
        """Print the line *value*, with flux and opacity integrals over the selection, to the data field."""
        from iSLAT.Modules.DataProcessing.spectral_utils import flux_integral, opacity_integral
        from .LineInspectionPlot import LineInspectionPlot
        from .PlotView import notify_user

        islat = self._islat
        active_mol = getattr(islat, "active_molecule", None)
        data_flux = model_flux = opacity_in_range = None
        selection = self._line_info_selection()
        if selection is not None:
            xmin, xmax = selection
            line_flux, _ = flux_integral(
                lam=islat.wave_data, flux=islat.flux_data,
                lam_min=xmin, lam_max=xmax, err=getattr(islat, "err_data", None),
            )
            data_flux = line_flux[0] if isinstance(line_flux, (list, tuple)) else line_flux
            if active_mol is not None:
                mol_wave, mol_flux = active_mol.get_flux(return_wavelengths=True)
                model_flux, _ = flux_integral(
                    lam=mol_wave, flux=mol_flux, lam_min=xmin, lam_max=xmax, err=None,
                )
                tau_wave, tau_profile = active_mol.get_tau(
                    return_wavelengths=True, normalization="area",  # 1/um: integrates to the summed tau
                )
                opacity_in_range = opacity_integral(
                    lam=tau_wave, tau=tau_profile, lam_min=xmin, lam_max=xmax,
                )

        # Marker info dicts use either set of key names.
        line = SimpleNamespace(
            lam=value.get("lam"),
            e_up=value.get("e_up", value.get("e")),
            e_low=value.get("e_low"),
            a_stein=value.get("a_stein", value.get("a")),
            g_up=value.get("g_up", value.get("g")),
            g_low=value.get("g_low"),
            lev_up=value.get("up_lev"),
            lev_low=value.get("low_lev"),
        )
        info = LineInspectionPlot.get_line_info(
            line,
            intensity=value.get("intensity", value.get("inten", 0)),
            tau=value.get("tau"),
            data_flux_in_range=data_flux,
            model_flux_in_range=model_flux,
            opacity_in_range=opacity_in_range,
            molecule=active_mol,
        )
        notify_user(islat, LineInspectionPlot.format_line_info(info), clear_after=clear_data_field)

    def _inspection_line_at(self, event: Any) -> Optional[LineRef]:
        """The line whose inspection marker (dashed line or label) is under *event*, or ``None``.

        When several markers are within reach, the one nearest the pointer
        in X wins.
        """
        if event is None or getattr(event, "inaxes", None) is None:
            return None
        best = None   # (pixel distance, info)
        for entry in getattr(self, "active_lines", None) or []:
            info = entry[3] if len(entry) > 3 else None
            if not info or info.get("lam") is None:
                continue
            for artist in entry[:2]:
                if artist is None or getattr(artist, "axes", None) is not event.inaxes:
                    continue
                try:
                    hit, _ = artist.contains(event)
                except Exception:
                    hit = False
                if not hit:
                    continue
                try:
                    x_line = event.inaxes.transData.transform((float(info["lam"]), 0.0))[0]
                    distance = abs(x_line - event.x)
                except Exception:
                    distance = float("inf")
                if best is None or distance < best[0]:
                    best = (distance, info)
        return LineRef.from_info(best[1]) if best is not None else None

    def _inspected_entry(self, ref: LineRef, with_marker: bool = False) -> Optional[list]:
        """The ``active_lines`` entry of the line *ref* describes, or ``None``.

        Lines of other molecules can share a wavelength, so the lines of
        *ref*'s molecule are searched first when any are shown.
        *with_marker* skips entries without an inspection marker.
        """
        entries = [e for e in getattr(self, "active_lines", None) or []
                   if len(e) > 3 and e[3] and (e[0] is not None or not with_marker)]
        if ref.molecule is not None:
            own = [e for e in entries if e[3].get("molecule_name") == ref.molecule]
            entries = own or entries
        infos = [e[3] for e in entries]
        pos = match_line(
            ref,
            [i.get("lam") for i in infos],
            [i.get("up_lev", i.get("lev_up")) for i in infos],
            [i.get("low_lev", i.get("lev_low")) for i in infos],
        )
        return None if pos is None else entries[pos]

    def _add_show_in_line_list_item(self, menu: Any, ref: Optional[LineRef],
                                    parent_widget: Any) -> None:
        """Append "Show <line> in Line List" for *ref* to *menu* (disabled when *ref* is ``None``)."""
        if ref is None:
            menu.add_command(label=SHOW_IN_LINE_LIST_LABEL, state="disabled")
            return
        menu.add_command(label=show_in_line_list_label(ref),
                         command=lambda: self._show_in_line_list(ref, parent_widget))

    def _show_in_line_list(self, ref: LineRef, parent_widget: Any) -> Any:
        """Open (or raise) the line list table of *ref*'s molecule with *ref* selected."""
        from iSLAT.Modules.GUI.Widgets.LineListViewWindow import show_line_in_line_list
        parent = parent_widget
        try:
            # Not the canvas itself: a view replaces its canvas when it
            # rebuilds, and that would take the table window with it.
            parent = parent_widget.winfo_toplevel()
        except Exception:
            pass
        return show_line_in_line_list(
            ref, islat=getattr(self, "_islat", None), parent=parent)

    def _toggle_inspection_summed(self, insp_ax: Any) -> None:
        """Toggle the summed-model fill on *insp_ax* only.

        If a fill tagged ``_islat_summed`` is already present it is shown
        or hidden in-place.  If none exists the overlay is computed from
        the current molecules and rendered for the first time.  The canvas
        that holds *insp_ax* is redrawn after the change.

        Parameters
        ----------
        insp_ax :
            The matplotlib :class:`~matplotlib.axes.Axes` that hosts the
            line-inspection panel (``ax2`` in Three Panel, the plot ``ax``
            in standalone Line Inspection).
        """
        if insp_ax is None:
            return

        existing = list(iter_tagged(insp_ax, SUMMED_TAG, lines=False))
        currently_visible = any(c.get_visible() for c in existing)

        if existing:
            # Just flip visibility - no re-render needed.
            for c in existing:
                c.set_visible(not currently_visible)
        else:
            # First-time render: compute summed flux over the visible x-range.
            islat = getattr(self, '_islat', None)
            mols = getattr(islat, 'molecules_dict', None) if islat is not None else None
            wave = getattr(islat, 'wave_data', None) if islat is not None else None
            if mols is None or wave is None:
                return
            try:
                import numpy as np
                xmin, xmax = insp_ax.get_xlim()
                mask = (wave >= xmin) & (wave <= xmax)
                if not np.any(mask):
                    return
                s_wave, s_flux = mols.get_summed_flux(wave[mask])
                if s_wave is None or not np.any(s_flux > 0):
                    return
                # Use the SpectrumPanel helper if available; otherwise fill directly.
                panel = getattr(self, '_plot', None)
                if hasattr(panel, '_plot_summed_spectrum'):
                    panel._plot_summed_spectrum(insp_ax, s_wave, s_flux, deduplicate=True)
                else:
                    draw_summed_fill(insp_ax, s_wave, s_flux, color='lightgray', zorder=1)
            except Exception:
                return

        insp_ax.figure.canvas.draw_idle()

    def _build_line_inspection_menu(self, canvas_widget: Any, event: Any = None) -> Optional[Any]:
        """Return a populated ``tk.Menu`` for the line inspection panel.

        Parameters
        ----------
        canvas_widget :
            The Tk widget used as the menu's parent.
        event :
            The right-click (a matplotlib mouse event).  "Show in Line List"
            acts on the line marker under it, or else on the selected line.
        """
        try:
            import tkinter as tk
        except ImportError:
            return None

        islat = getattr(self, '_islat', None)
        menu = tk.Menu(canvas_widget, tearoff=0)

        def _save_current_line():
            if islat is not None and hasattr(islat, 'GUI') and hasattr(islat.GUI, 'top_bar'):
                islat.GUI.top_bar.save_line(save_type="selected")

        def _fit_current_line():
            if islat is not None and hasattr(islat, 'GUI') and hasattr(islat.GUI, 'top_bar'):
                islat.GUI.top_bar.fit_selected_line(deblend=False)

        def _run_deblender():
            if islat is not None and hasattr(islat, 'GUI') and hasattr(islat.GUI, 'top_bar'):
                islat.GUI.top_bar.fit_selected_line(deblend=True)

        def _save_all_lines_in_range():
            if islat is not None and hasattr(islat, 'GUI') and hasattr(islat.GUI, 'top_bar'):
                islat.GUI.top_bar.save_all_lines_in_range()

        # ------------------------------------------------------------------
        # Spectrum-navigation helpers - only available from ThreePanelView
        # (i.e. when self has ax1 / ax2 / _pm).
        # ------------------------------------------------------------------
        ax1 = getattr(self, 'ax1', None)
        ax2 = getattr(self, 'ax2', None)
        pm  = getattr(self, '_pm', None)

        def _get_ax2_range():
            """Return (xmin, xmax) from ax2, or None if unavailable."""
            if ax2 is None:
                return None
            try:
                x0, x1 = ax2.get_xlim()
                if x0 < x1:
                    return x0, x1
            except Exception:
                pass
            return None

        def _refresh_ax1():
            """Apply new ax1 limits and refresh via the plot manager."""
            if pm is not None and hasattr(pm, 'match_display_range'):
                pm.match_display_range(match_y=True)
            elif pm is not None and hasattr(pm, 'canvas'):
                pm.canvas.draw_idle()

        def _match_spectrum_to_inspection():
            """Set ax1 xlim to the current inspection (ax2) range."""
            r = _get_ax2_range()
            if r is None or ax1 is None:
                return
            xmin, xmax = r
            ax1.set_xlim(xmin, xmax)
            if pm is not None and hasattr(pm, 'islat'):
                pm.islat.display_range = (xmin, xmax)
            _refresh_ax1()

        def _center_spectrum_on_inspection():
            """Center the spectrum view (ax1) on the inspection midpoint,
            keeping the current ax1 x-span unchanged."""
            r = _get_ax2_range()
            if r is None or ax1 is None:
                return
            insp_mid = (r[0] + r[1]) / 2.0
            try:
                cur_lo, cur_hi = ax1.get_xlim()
                half_span = (cur_hi - cur_lo) / 2.0
            except Exception:
                return
            new_lo = insp_mid - half_span
            new_hi = insp_mid + half_span
            ax1.set_xlim(new_lo, new_hi)
            if pm is not None and hasattr(pm, 'islat'):
                pm.islat.display_range = (new_lo, new_hi)
            _refresh_ax1()

        # Resolve the inspection axes - ax2 for ThreePanelView, or the
        # standalone plot ax for LineInspectionView.
        insp_ax = ax2 or getattr(getattr(self, '_plot', None), 'ax', None)

        def _toggle_total_model():
            self._toggle_inspection_summed(insp_ax)
            # Reflect the new state in the menu label (best-effort).
            existing = [] if insp_ax is None else iter_tagged(insp_ax, SUMMED_TAG, lines=False)
            visible = any(c.get_visible() for c in existing)
            try:
                idx = menu.index("Show Total Model" if visible else "Hide Total Model")
                menu.entryconfig(idx, label="Hide Total Model" if visible else "Show Total Model")
            except Exception:
                pass

        menu.add_command(label=with_shortcut("Save Current Line", "save_line"), command=_save_current_line)
        menu.add_command(label=with_shortcut("Fit Current Line", "fit_line"),   command=_fit_current_line)
        menu.add_command(label="Run Deblender",           command=_run_deblender)
        menu.add_separator()
        menu.add_command(label="Save All Lines in Range", command=_save_all_lines_in_range)
        # The line marker right-clicked, or else the selected (orange) line.
        self._add_show_in_line_list_item(
            menu,
            self._inspection_line_at(event)
            or LineRef.from_info(getattr(self, "selected_line", None)),
            canvas_widget,
        )

        if insp_ax is not None:
            summed_on = any(c.get_visible() for c in iter_tagged(insp_ax, SUMMED_TAG, lines=False))
            menu.add_separator()
            menu.add_command(
                label="Hide Total Model" if summed_on else "Show Total Model",
                command=_toggle_total_model,
            )

        if ax1 is not None and ax2 is not None:
            menu.add_separator()
            menu.add_command(
                label="Match Spectrum to Inspection Range",
                command=_match_spectrum_to_inspection,
            )
            menu.add_command(
                label="Center Spectrum on Inspection Location",
                command=_center_spectrum_on_inspection,
            )

        add_view_switch_items(menu, pm, ("Line Inspection", "Three Panel", "Population Diagram"))

        return menu