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

from typing import Any, Optional

from iSLAT.Modules.DataProcessing.LineIdentity import LineRef, match_line

from .ArtistTags import SUMMED_TAG, iter_tagged
from .ModelOverlays import draw_summed_fill

#: Menu label of "Show in Line List" when there is no line to show.
SHOW_IN_LINE_LIST_LABEL = "Show in Line List"


def show_in_line_list_label(ref: Optional[LineRef]) -> str:
    """Menu label for showing *ref* in its line list (``Show H2O 17.2345 um in Line List``)."""
    if ref is None:
        return SHOW_IN_LINE_LIST_LABEL
    return f"Show {ref.describe()} in Line List"


class LineInspectionContextMixin:
    """Mixin that supplies the line inspection panel right-click menu.

    Requires that the host class exposes ``self._islat`` (the main iSLAT application object with a ``GUI.top_bar`` attribute).
    The line markers are read from ``self.active_lines`` (``[vline, text, scatter, info]`` entries)
    and the selected line from ``self.selected_line``.
    """
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
        is redrawn after the change.

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

        # Redraw via the plot manager canvas if available, else fall back.
        pm = getattr(self, '_pm', None)
        canvas = getattr(pm, 'canvas', None) or getattr(self, '_canvas', None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass

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

        menu.add_command(label="Save Current Line",       command=_save_current_line)
        menu.add_command(label="Fit Current Line",        command=_fit_current_line)
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

        # ------------------------------------------------------------------
        # View switching
        # ------------------------------------------------------------------
        if pm is not None and hasattr(pm, 'switch_view'):
            menu.add_separator()

            active_name = getattr(pm, 'active_view_name', None)

            if active_name == "Three Panel":
                # Inside the three-panel layout - offer to pop out to standalone views
                def _to_line_inspection():
                    pm.switch_view("Line Inspection")

                def _to_population_diagram():
                    pm.switch_view("Population Diagram")

                menu.add_command(
                    label="Switch to Line Inspection View",
                    command=_to_line_inspection,
                )
                menu.add_command(
                    label="Switch to Population Diagram View",
                    command=_to_population_diagram,
                )
            else:
                # Already in a standalone view - offer to go back or cross-navigate
                def _to_three_panel():
                    pm.switch_view("Three Panel")

                def _to_population_diagram():
                    pm.switch_view("Population Diagram")

                in_line_inspection = (active_name == "Line Inspection")

                menu.add_command(
                    label="Switch to Three Panel View",
                    command=_to_three_panel,
                )
                menu.add_command(
                    label="Switch to Population Diagram View",
                    command=_to_population_diagram,
                    state="disabled" if in_line_inspection and active_name == "Population Diagram" else "normal",
                )

        return menu