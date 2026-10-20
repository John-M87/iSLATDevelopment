"""
FitLinesPlotGridView - :class:`PlotView` for displaying a :class:`FitLinesPlotGrid` within the main iSLAT plotting area.

Shows a grid of individual line-fit results for the current fitting session. 
When multiple grids are available (e.g. after a batch fit) they are presented in a :class:`ttk.Notebook` 
with one tab per grid, mirroring the layout of :class:`~iSLAT.Modules.GUI.PlotGridWindow.PlotGridWindow`.
Each tab is scrollable to accommodate grids that exceed the frame height.

When no plot grids have been set a placeholder message is shown.

Typical usage::

    # inside TopBar / fitting callback
    plot_grid = FitLinesPlotGrid(fit_data=fit_data, ...)
    plot_grid.generate_plot()
    main_plot.fit_lines_grid_view.set_plot_grids([plot_grid])
    main_plot.switch_view("Line Grid")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .PlotView import PlotView
from .BasePlot import BasePlot
from .FitLinesPlotGrid import FitLinesPlotGrid

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    class _FallbackDebug:
        def verbose(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): print(f"WARNING: {kw or a}")
        def error(self, *a, **kw): print(f"ERROR: {kw or a}")
        def trace(self, *a, **kw): pass
    debug_config = _FallbackDebug()

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    FigureCanvasTkAgg = None  # type: ignore[assignment,misc]
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

# Size constants mirrored from PlotGridWindow
SUBPLOT_WIDTH_INCHES = 1.8
SUBPLOT_HEIGHT_INCHES = 1.8

class FitLinesPlotGridView(PlotView):
    """
    Embedded fit-lines plot grid view.

    Renders one or more :class:`FitLinesPlotGrid` objects inside the main iSLAT plotting area.
    When multiple grids are available they are presented in a :class:`ttk.Notebook` with one tab per grid,
    mirroring the layout of :class:`~iSLAT.Modules.GUI.PlotGridWindow.PlotGridWindow`.
    Each tab is scrollable to accommodate grids that exceed the frame height.

    When no plot grids have been set a placeholder message is displayed.

    Call :meth:`set_plot_grids` to push a new list of
    :class:`FitLinesPlotGrid` objects into the view; the view will
    automatically refresh if it is currently active.

    Parameters
    ----------
    plot_manager : iSLATPlot
        The main controller (``iSLAT.Modules.Plotting.MainPlot.iSLATPlot``).
    """
    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat

        self._parent_frame: Any = None

        # Primary canvas / figure references (point at first tab or placeholder)
        self._canvas: Optional["FigureCanvasTkAgg"] = None
        self._fig: Optional["Figure"] = None

        # Placeholder figure (shown when _plot_grids is empty)
        self._placeholder_fig: Optional["Figure"] = None
        self._placeholder_canvas: Optional["FigureCanvasTkAgg"] = None

        # Notebook and per-tab canvases for multi-grid display
        self._notebook: Optional[Any] = None            # ttk.Notebook
        self._tab_canvases: List["FigureCanvasTkAgg"] = []
        self._container_frame: Optional[Any] = None    # ttk.Frame packed into parent

        # The list of FitLinesPlotGrid objects to display
        self._plot_grids: List[FitLinesPlotGrid] = []

        self._initialised: bool = False
        self._needs_refresh: bool = True
        self._show_model_lines: bool = False

    # ==================================================================
    # Public API
    # ==================================================================
    def set_plot_grids(self, plot_grids: List[FitLinesPlotGrid]) -> None:
        """Set the list of :class:`FitLinesPlotGrid` objects to display.

        Calling this method marks the view as needing a refresh.
        If the view is already active the display is updated immediately.

        Parameters
        ----------
        plot_grids : list of FitLinesPlotGrid
            The new grids to display.  Pass an empty list (or ``None``) to
            show the placeholder.
        """
        self._plot_grids = list(plot_grids) if plot_grids else []
        self._needs_refresh = True
        if self._initialised and self._parent_frame is not None:
            self._rebuild_display()

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _build_line_list_preview(self) -> Optional[FitLinesPlotGrid]:
        """Build a :class:`FitLinesPlotGrid` from the loaded line list with no fit data.

        Returns a grid whose panels show the observed spectrum around every
        line in ``islat.input_line_list`` but with *no* Gaussian overlay -
        identical to the post-fit grid except fit results are absent.

        Returns ``None`` if the required data (line list file, wave/flux arrays) is unavailable.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        line_list_path = getattr(self._islat, "input_line_list", None)
        if not line_list_path:
            return None

        wave = getattr(self._islat, "wave_data", None)
        flux = getattr(self._islat, "flux_data", None)
        if wave is None or flux is None or len(wave) == 0:
            return None

        try:
            df = pd.read_csv(line_list_path)
        except Exception as exc:
            debug_config.warning(
                "fit_lines_grid_view",
                f"Could not read line list for preview: {exc}",
            )
            return None

        if df.empty:
            return None

        # Build fit_csv_dict from the line list rows.  Only the keys consumed
        # by FitLinesPlotGrid._create_panels and _post_render are required:
        # xmin, xmax, species, lam, Fit_det.
        fit_csv_dict: dict = {}
        for idx, row in enumerate(df.itertuples(index=False)):
            xmin = getattr(row, "xmin", None)
            xmax = getattr(row, "xmax", None)
            if xmin is None or xmax is None:
                # Fall back to a +/-0.015 um window around lam if bounds missing
                lam = getattr(row, "lam", None)
                if lam is None:
                    continue
                xmin = float(lam) - 0.015
                xmax = float(lam) + 0.015
            fit_csv_dict[idx] = {
                "xmin": float(xmin),
                "xmax": float(xmax),
                "species": str(getattr(row, "species", "")),
                "lam": float(getattr(row, "lam", 0.0)),
                "Fit_det": False,
            }

        if not fit_csv_dict:
            return None

        n = len(fit_csv_dict)
        # Null fit tuples - _post_render handles None gracefully
        null_fits = [None] * n
        fit_data_tuple_list = (null_fits, null_fits, null_fits)

        spectrum_name = getattr(
            self._islat, "loaded_spectrum_name",
            "Line List Preview",
        )

        try:
            err = getattr(self._islat, "err_data", None)
            grid = FitLinesPlotGrid(
                fit_data=(fit_csv_dict, fit_data_tuple_list),
                wave_data=wave,
                flux_data=flux,
                err_data=err,
                spectrum_name=spectrum_name,
                # set cols dynamically based on the number of lines and the figure SUBPLOT_WIDTH_INCHES constant, which determines how many panels can fit horizontally in a typical display width.
                cols=min(n, max(1, int(plt.gcf().get_size_inches()[0] // SUBPLOT_WIDTH_INCHES)))
            )
            grid.generate_plot()
            return grid
        except Exception as exc:
            debug_config.warning(
                "fit_lines_grid_view",
                f"Failed to build line list preview grid: {exc}",
            )
            return None

    def _build_placeholder(self) -> None:
        """Create a simple placeholder figure when no grid data is available."""
        if self._placeholder_fig is not None:
            plt.close(self._placeholder_fig)

        self._placeholder_fig = plt.Figure(figsize=(8, 5), constrained_layout=True)
        ax = self._placeholder_fig.add_subplot(111)
        ax.set_title("Line Grid", fontsize=14)
        ax.text(
            0.5, 0.5,
            "No line list loaded.\n\nLoad a line list to populate this view.",
            ha="center", va="center",
            transform=ax.transAxes,
            color="gray", fontsize=12,
        )
        ax.axis("off")

    def _apply_theme_to_fig(self, fig: "Figure") -> None:
        """Apply the controller's current theme to *fig*."""
        theme = self._pm.theme
        bg = theme.get("background", "#181A1B")
        fg = theme.get("foreground", "#F0F0F0")

        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(theme.get("axes_background", bg))
            ax.tick_params(colors=fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_edgecolor(fg)

    def _teardown_display(self) -> None:
        """Destroy all Tk widgets owned by this view."""
        for canvas in self._tab_canvases:
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
        self._tab_canvases.clear()

        if self._placeholder_canvas is not None:
            try:
                self._placeholder_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self._placeholder_canvas = None

        # Clear primary canvas reference (widget already destroyed above)
        self._canvas = None
        self._fig = None

        if self._notebook is not None:
            try:
                self._notebook.destroy()
            except Exception:
                pass
            self._notebook = None

        if self._container_frame is not None:
            try:
                self._container_frame.destroy()
            except Exception:
                pass
            self._container_frame = None

    def _rebuild_display(self) -> None:
        """Tear down and recreate all display widgets for the current plot grids."""
        self._teardown_display()

        if self._parent_frame is None or tk is None:
            return

        if not self._plot_grids:
            self._show_placeholder()
            return

        # Outer container frame packed into the parent
        self._container_frame = ttk.Frame(self._parent_frame)
        self._container_frame.pack(fill="both", expand=True)

        if len(self._plot_grids) == 1:
            # Single grid - no notebook overhead, just one scrollable canvas
            self._create_scrollable_tab(self._container_frame, self._plot_grids[0])
        else:
            # Multiple grids - tabbed notebook, one tab per grid
            self._notebook = ttk.Notebook(self._container_frame)
            self._notebook.pack(fill="both", expand=True)
            for plot_grid in self._plot_grids:
                tab_frame = ttk.Frame(self._notebook)
                self._notebook.add(tab_frame, text=plot_grid.spectrum_name)
                self._create_scrollable_tab(tab_frame, plot_grid)

    def _show_placeholder(self) -> None:
        """Display the placeholder figure inside the parent frame.

        If a line list is loaded but no fit has been run, automatically renders a preview grid instead of a blank placeholder.
        """
        # Try to build a line-list preview before falling back to blank
        preview = self._build_line_list_preview()
        if preview is not None:
            # Temporarily set as the only grid and render it
            self._plot_grids = [preview]
            self._rebuild_display()
            return

        self._build_placeholder()
        self._apply_theme_to_fig(self._placeholder_fig)

        if FigureCanvasTkAgg is not None and self._parent_frame is not None:
            self._placeholder_canvas = FigureCanvasTkAgg(
                self._placeholder_fig, master=self._parent_frame
            )
            self._placeholder_canvas.draw()
            self._placeholder_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Expose as primary canvas / figure for get_canvas / get_figure
        self._canvas = self._placeholder_canvas
        self._fig = self._placeholder_fig

    def _create_scrollable_tab(
        self,
        parent: Any,
        plot_grid: FitLinesPlotGrid,
    ) -> None:
        """Build a scrollable matplotlib canvas for *plot_grid* inside *parent*.

        The layout mirrors :meth:`~iSLAT.Modules.GUI.PlotGridWindow.PlotGridWindow._create_tab`.

        Parameters
        ----------
        parent : Tk widget
            The frame (tab or container) to build the scrollable view in.
        plot_grid : FitLinesPlotGrid
            The grid whose figure to embed.
        """
        # --- Scrollable area ------------------------------------------------
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        scroll_canvas = tk.Canvas(
            canvas_frame, highlightthickness=0,
            yscrollcommand=scrollbar.set,
        )
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=scroll_canvas.yview)

        inner_frame = ttk.Frame(scroll_canvas)
        canvas_window = scroll_canvas.create_window(
            (0, 0), window=inner_frame, anchor="nw"
        )

        # --- Configure figure size ------------------------------------------
        # The figure is sized dynamically in :meth:`_fit_grid_to_width` so it
        # always fills the available horizontal space while keeping panels
        # roughly square.  These values are only an initial guess that is
        # corrected by the first ``<Configure>`` event of the scroll canvas.
        plot_grid.fig.set_dpi(150)
        fig_width = SUBPLOT_WIDTH_INCHES * plot_grid.cols
        fig_height = SUBPLOT_WIDTH_INCHES * plot_grid.rows  # same as width -> square cells
        plot_grid.fig.set_size_inches(fig_width, fig_height)
        # Disable any active layout engine before calling subplots_adjust,
        # which is incompatible with constrained_layout / tight_layout.
        # Using None (not the string "none") fully removes the engine.
        try:
            plot_grid.fig.set_layout_engine(None)
        except Exception:
            try:
                plot_grid.fig.set_constrained_layout(False)
            except Exception:
                pass
            try:
                plot_grid.fig.set_tight_layout(False)
            except Exception:
                pass
        plot_grid.fig.subplots_adjust(
            left=0.06, right=0.98,
            top=0.96, bottom=0.05,
            wspace=0.18, hspace=0.32,
        )

        # Compact font sizes for dense subplot grids
        for ax in plot_grid.axs.flat:
            ax.tick_params(axis="both", labelsize=5, pad=1)
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=45)
            for lbl in ax.get_yticklabels():
                lbl.set_ha("right")
                lbl.set_va("center")
            ax.title.set_fontsize(7)
            ax.title.set_position((0.5, 1.0))
            if ax.yaxis.label:
                ax.yaxis.label.set_fontsize(5)
            if ax.xaxis.label:
                ax.xaxis.label.set_fontsize(5)

        self._apply_theme_to_fig(plot_grid.fig)

        # --- Embed matplotlib figure ----------------------------------------
        dpi = plot_grid.fig.get_dpi()
        px_w = int(fig_width * dpi)
        px_h = int(SUBPLOT_WIDTH_INCHES * plot_grid.rows * dpi)
        fig_canvas = FigureCanvasTkAgg(plot_grid.fig, master=inner_frame)
        fig_canvas.draw()
        widget = fig_canvas.get_tk_widget()
        widget.config(width=px_w, height=px_h)
        widget.pack(fill="none", expand=False)
        self._tab_canvases.append(fig_canvas)

        # Attach right-click context menu to this canvas.
        self._connect_panel_context_menu(fig_canvas, plot_grid)

        # First tab's canvas / figure become the primary references
        if self._canvas is None:
            self._canvas = fig_canvas
            self._fig = plot_grid.fig

        # --- Scroll behaviour -----------------------------------------------
        def update_scroll_region(event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        inner_frame.bind("<Configure>", update_scroll_region)

        def configure_inner_frame(event):
            scroll_canvas.itemconfig(canvas_window, width=event.width)
            # Resize the embedded figure so it fills the available width while keeping each panel roughly square.
            # Guard against redundant redraws when the width has not actually changed.
            if getattr(fig_canvas, "_islat_last_w", None) != event.width:
                fig_canvas._islat_last_w = event.width  # type: ignore[attr-defined]
                self._fit_grid_to_width(plot_grid, fig_canvas, event.width)

        scroll_canvas.bind("<Configure>", configure_inner_frame)

        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 60)), "units")

        def bind_mousewheel(event):
            scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            scroll_canvas.unbind_all("<MouseWheel>")

        scroll_canvas.bind("<Enter>", bind_mousewheel)
        scroll_canvas.bind("<Leave>", unbind_mousewheel)

    def _fit_grid_to_width(
        self,
        plot_grid: FitLinesPlotGrid,
        fig_canvas: "FigureCanvasTkAgg",
        avail_px: int,
    ) -> None:
        """Resize *plot_grid*'s figure to fill *avail_px* pixels horizontally.

        Each panel keeps a roughly square aspect ratio: the total figure
        height is set to ``(avail_px / cols) * rows`` pixels so that every
        cell is as tall as it is wide.

        Parameters
        ----------
        plot_grid : FitLinesPlotGrid
            The grid whose figure to resize.
        fig_canvas : FigureCanvasTkAgg
            The Tk-embedded canvas wrapping ``plot_grid.fig``.
        avail_px : int
            The width, in pixels, that is currently available for the figure.
        """
        if fig_canvas is None or avail_px <= 1:
            return
        cols = max(1, int(plot_grid.cols))
        rows = max(1, int(plot_grid.rows))
        dpi = plot_grid.fig.get_dpi() or 100
        cell_px = avail_px / cols
        total_h_px = int(round(cell_px * rows))
        # Update the matplotlib figure size (inches) so it renders at the
        # correct scale, then match the Tk widget's pixel dimensions.
        plot_grid.fig.set_size_inches(avail_px / dpi, total_h_px / dpi)
        widget = fig_canvas.get_tk_widget()
        widget.config(width=int(avail_px), height=total_h_px)
        try:
            fig_canvas.draw_idle()
        except Exception:
            pass

    # ==================================================================
    # Overlay helpers - molecule model lines
    # ==================================================================
    def _apply_model_line_overlays(self) -> None:
        """Plot visible molecule model lines on every panel of every plot grid.

        Mirrors the ``mol_cache`` rendering loop inside :meth:`SpectrumPanel.generate_plot` so that overlays are consistent in style.
        Each plotted artist is tagged with ``._molecule_name`` so it can be found by :meth:`_remove_model_line_overlays`.
        """
        import numpy as np

        mols_dict = getattr(self._islat, "molecules_dict", None)
        if mols_dict is None:
            return
        visible_names = mols_dict.get_visible_molecules(return_objects=False)
        if not visible_names:
            return

        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                ax = getattr(panel, "ax", None)
                if ax is None:
                    continue
                xlim = getattr(panel, "xlim", None)
                if xlim is None:
                    continue
                xmin, xmax = xlim
                for mol_name in visible_names:
                    mol = mols_dict.get(mol_name)
                    if mol is None:
                        continue
                    try:
                        lam, flux = mol.get_flux(return_wavelengths=True)
                    except Exception:
                        continue
                    mask = (lam >= xmin) & (lam <= xmax)
                    if not np.any(mask):
                        continue
                    (line,) = ax.plot(
                        lam[mask],
                        flux[mask],
                        linestyle="--",
                        color=getattr(mol, "color", None),
                        alpha=0.8,
                        linewidth=BasePlot.get_molecule_linewidth(mol, 0.8),
                        label=getattr(mol, "displaylabel", mol_name),
                        zorder=3,
                    )
                    line._molecule_name = mol_name  # type: ignore[attr-defined]

    def _remove_model_line_overlays(self) -> None:
        """Remove all molecule model line artists previously added by
        :meth:`_apply_model_line_overlays` from every panel axes."""
        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                ax = getattr(panel, "ax", None)
                if ax is None:
                    continue
                to_remove = [
                    line for line in ax.lines
                    if hasattr(line, "_molecule_name")
                ]
                for line in to_remove:
                    try:
                        line.remove()
                    except Exception:
                        pass

    def _set_show_model_lines(self, value: bool) -> None:
        """Toggle molecule model line overlays on or off."""
        self._show_model_lines = value
        if value:
            self._apply_model_line_overlays()
        else:
            self._remove_model_line_overlays()
        self.draw()

    def _refresh_model_line_overlays(self) -> None:
        """Re-render model line overlays to reflect the current molecule state.

        Acts as the single update path used by the molecule visibility, parameter, and model-change callbacks.
        When the master ``show_model_lines`` switch is off (or no grids are present) this is a no-op so that toggling the switch remains the only way to display overlays.
        When on, all existing overlays are cleared and re-drawn from the currently *visible* molecules, so visibility toggles and parameter edits are reflected immediately.
        """
        if not (self._show_model_lines and self._plot_grids):
            return
        self._remove_model_line_overlays()
        self._apply_model_line_overlays()
        self.draw()

    # ==================================================================
    # PlotView - lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        """Pack the fit-lines grid view into *parent_frame* and refresh if needed.

        If no plot grids have been set but a line list is loaded, a preview
        grid is built automatically from the line list so the view is never shown empty when data are available.
        """
        self._parent_frame = parent_frame

        # When no explicit grids are present, always re-attempt a preview
        # (the line list or spectrum data may have changed since last activation).
        if not self._plot_grids:
            self._needs_refresh = True

        if not self._initialised or self._needs_refresh:
            self._rebuild_display()
            self._initialised = True
            self._needs_refresh = False

        self.apply_theme(self._pm.theme)
        self._register_control_fields()

    # ------------------------------------------------------------------
    # Grid-shape control-field helpers (n_cols / n_rows)
    # ------------------------------------------------------------------
    def _get_n_cols(self) -> int:
        """Return the current column count of the displayed grid(s)."""
        if self._plot_grids:
            return int(self._plot_grids[0].cols)
        return 10

    def _set_n_cols(self, value: Any) -> None:
        """Set the number of columns for every displayed grid and re-render."""
        self._apply_grid_shape(cols=max(1, int(value)))

    def _get_n_rows(self) -> int:
        """Return the current row count of the displayed grid(s)."""
        if self._plot_grids:
            return int(self._plot_grids[0].rows)
        return 1

    def _set_n_rows(self, value: Any) -> None:
        """Set the number of rows for every displayed grid and re-render."""
        self._apply_grid_shape(rows=max(1, int(value)))

    def _apply_grid_shape(
        self,
        cols: Optional[int] = None,
        rows: Optional[int] = None,
    ) -> None:
        """Apply a new ``cols`` / ``rows`` shape to all grids and rebuild.

        Only the dimension(s) supplied are changed; the other keeps its current value.
        Each affected grid is regenerated and the embedded canvases are recreated so the new layout fills the display.
        """
        if not self._plot_grids:
            return

        changed = False
        for grid in self._plot_grids:
            new_cols = grid.cols if cols is None else cols
            new_rows = grid.rows if rows is None else rows
            if new_cols == grid.cols and new_rows == grid.rows:
                continue
            grid.cols = new_cols
            grid.rows = new_rows
            try:
                grid.generate_plot()
                changed = True
            except Exception as exc:
                debug_config.warning(
                    "fit_lines_grid_view",
                    f"_apply_grid_shape: regenerate failed: {exc}",
                )

        if not changed:
            return

        # Re-apply model-line overlays since generate_plot() cleared the axes.
        if self._show_model_lines:
            self._apply_model_line_overlays()

        self._rebuild_display()

    def _register_control_fields(self) -> None:
        """Register the grid-shape entries and the ``show_model_lines`` toggle.

        Two :class:`EntryField` objects (``n_cols`` / ``n_rows``) are placed on
        the control panel - mirroring the ``n_panels`` field of the full
        spectrum view - and a ``show_model_lines`` :class:`ToggleField` is placed on the top bar.
        """
        bus = getattr(self._pm, "control_bus", None)
        if bus is None:
            return

        from iSLAT.Modules.GUI.ControlField import EntryField, ToggleField

        bus.unregister_owner(self)

        bus.register_many(
            [
                EntryField(
                    "n_cols", "N Cols:",
                    self._get_n_cols, self._set_n_cols,
                    datatype="int", width=5, owner=self,
                    tip="Number of columns in the\nfit-lines plot grid",
                ),
                EntryField(
                    "n_rows", "N Rows:",
                    self._get_n_rows, self._set_n_rows,
                    datatype="int", width=5, owner=self,
                    tip="Number of rows in the\nfit-lines plot grid",
                ),
            ],
            "control_panel",
        )

        bus.register(
            ToggleField(
                "show_model_lines",
                "Model Lines",
                getter=lambda: self._show_model_lines,
                setter=self._set_show_model_lines,
                owner=self,
                tip="Toggle molecule model line overlays\non the fit-lines plot grid",
            ),
            "top_bar",
        )

    def deactivate(self) -> None:
        """Tear down the view's widgets and unregister any ControlBus fields."""
        bus = getattr(self._pm, "control_bus", None)
        if bus is not None:
            bus.unregister_owner(self)

        if self._show_model_lines:
            self._remove_model_line_overlays()
        self._teardown_display()
        # Force a full rebuild on the next activation so the widgets are recreated inside whatever parent frame activate() receives.
        self._initialised = False

    # ==================================================================
    # PlotView - interaction context
    # ==================================================================
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """No custom right-click menu via the main InteractionHandler.

        Context menus for individual panels are handled by per-canvas
        ``button_press_event`` listeners attached in :meth:`_connect_panel_context_menu`.
        """
        return None

    def _connect_panel_context_menu(
        self,
        fig_canvas: "FigureCanvasTkAgg",
        plot_grid: FitLinesPlotGrid,
    ) -> None:
        """Attach a right-click context-menu handler to *fig_canvas*.

        Builds an ``Axes -> fit-index`` lookup from the grid's ``axs`` array and ``fit_csv_dict``. 
        When the user right-clicks an individual panel two actions are offered:

        * **Open in Line Inspection** - switches to the standalone Line
          Inspection view and calls ``on_selection(xmin, xmax)`` so the
          panel's wavelength range is shown immediately.
        * **Open in Three Panel (centered)** - switches to the Three Panel
          view and centers the spectrum overview on the panel's line.
        """
        if tk is None or fig_canvas is None or plot_grid.axs is None:
            return

        # Build ax -> (idx, fit_entry) lookup once.
        ax_map: Dict[Any, Tuple[int, dict]] = {}
        for idx, entry in plot_grid.fit_csv_dict.items():
            if idx >= plot_grid.rows * plot_grid.cols:
                break
            row = idx // plot_grid.cols
            col = idx % plot_grid.cols
            try:
                ax = plot_grid.axs[row, col]
                ax_map[ax] = (idx, entry)
            except (IndexError, KeyError):
                pass

        pm = self._pm

        def _on_button_press(event):
            if event.button != 3 or event.inaxes is None:
                return
            hit = ax_map.get(event.inaxes)
            if hit is None:
                return

            idx, entry = hit
            xmin = float(entry.get("xmin", 0.0))
            xmax = float(entry.get("xmax", 0.0))
            lam  = float(entry.get("lam",  (xmin + xmax) / 2.0))
            species = str(entry.get("species", ""))
            label = f"{species} {lam:.4f} μm" if species else f"{lam:.4f} μm"

            def _open_line_inspection():
                pm.switch_view("Line Inspection")
                try:
                    pm.active_view.on_selection(xmin, xmax)
                except Exception as exc:
                    debug_config.warning(
                        "fit_lines_grid_view",
                        f"_open_line_inspection: on_selection failed: {exc}",
                    )

            def _open_three_panel():
                # center ax1 on the line; keep the current span if possible.
                half_span = (xmax - xmin) / 2.0
                try:
                    cur_lo, cur_hi = pm.ax1.get_xlim()
                    cur_half = (cur_hi - cur_lo) / 2.0
                    if cur_half > 0:
                        half_span = cur_half
                except Exception:
                    pass
                new_lo = lam - half_span
                new_hi = lam + half_span
                pm.islat.display_range = (new_lo, new_hi)
                pm.switch_view("Three Panel")
                try:
                    pm.ax1.set_xlim(new_lo, new_hi)
                    pm.match_display_range(match_y=True)
                except Exception as exc:
                    debug_config.warning(
                        "fit_lines_grid_view",
                        f"_open_three_panel: range update failed: {exc}",
                    )

                # Create a span matching the grid panel's wavelength range so the line inspection panels populate immediately.
                try:
                    ih = getattr(pm, "interaction_handler", None)
                    if ih is not None and ih.span_selector is not None:
                        ih.span_selector.extents = (xmin, xmax)
                        ih.span_selector.set_visible(True)
                    pm.onselect(xmin, xmax)
                except Exception as exc:
                    debug_config.warning(
                        "fit_lines_grid_view",
                        f"_open_three_panel: span creation failed: {exc}",
                    )

            try:
                canvas_widget = fig_canvas.get_tk_widget()
                menu = tk.Menu(canvas_widget, tearoff=0)
                menu.add_command(
                    label=f"Open in Line Inspection  [{label}]",
                    command=_open_line_inspection,
                )
                menu.add_command(
                    label=f"Open in Three Panel  [{label}]",
                    command=_open_three_panel,
                )
                menu.add_separator()
                menu.add_command(label="Save Figure…", command=self.save_figure)
                # Convert matplotlib canvas coords to screen coords.
                x_root = canvas_widget.winfo_rootx() + int(event.x)
                y_root = (
                    canvas_widget.winfo_rooty()
                    + int(canvas_widget.winfo_height() - event.y)
                )
                menu.tk_popup(x_root, y_root)
            except Exception as exc:
                debug_config.warning(
                    "fit_lines_grid_view",
                    f"_connect_panel_context_menu popup failed: {exc}",
                )
            finally:
                try:
                    menu.grab_release()
                except Exception:
                    pass

        fig_canvas.mpl_connect("button_press_event", _on_button_press)

    # ==================================================================
    # PlotView - theme
    # ==================================================================
    def apply_theme(self, theme: dict) -> None:
        """Apply the current theme to all embedded figures and canvases."""
        self._pm.theme = theme

        for plot_grid in self._plot_grids:
            try:
                self._apply_theme_to_fig(plot_grid.fig)
            except Exception as exc:
                debug_config.warning("fit_lines_grid_view", f"apply_theme error: {exc}")

        if not self._plot_grids and self._placeholder_fig is not None:
            try:
                self._apply_theme_to_fig(self._placeholder_fig)
            except Exception:
                pass

        for canvas in self._tab_canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass

        if self._canvas is not None and self._canvas not in self._tab_canvases:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    # ==================================================================
    # PlotView - core rendering
    # ==================================================================
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Refresh molecule model line overlays when the model changes.

        If ``show_model_lines`` is active the overlays are removed and re-drawn so that changes in molecule parameters are reflected.
        """
        self._refresh_model_line_overlays()
        self._refresh_summed_spectrum()

    def on_molecule_visibility_changed(
        self,
        molecule_name: str,
        is_visible: bool,
        molecules_dict: "MoleculeDict",
        wave_data: Any,
        active_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
        force_rerender: bool = False,
    ) -> None:
        """Re-render model line overlays when a molecule's visibility changes.

        The ``show_model_lines`` switch is the master control: while it is on,
        any molecule whose own visibility is enabled is overlaid, and any molecule that becomes hidden is removed.
        While it is off this is a no-op.
        """
        self._refresh_model_line_overlays()
        self._refresh_summed_spectrum()

    # ==================================================================
    # PlotView - selection / line inspection (no-ops)
    # ==================================================================
    def on_selection(self, xmin: float, xmax: float) -> None:
        """No-op - this view does not respond to span-selector events."""
        pass

    def clear_selection(self) -> None:
        """No-op."""
        pass

    def clear_active_lines(self) -> None:
        """No-op - no active-line markers in this view."""
        pass

    # ==================================================================
    # PlotView - molecule lifecycle (no-ops)
    # ==================================================================
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """No-op."""
        pass

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Refresh molecule model line overlays when a parameter changes."""
        self._refresh_model_line_overlays()
        self._refresh_summed_spectrum()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Refresh model line overlays so a deleted molecule's line is removed."""
        self._refresh_model_line_overlays()
        self._refresh_summed_spectrum()

    # ==================================================================
    # PlotView - toggle helpers (all no-ops for this view)
    # ==================================================================
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """Apply the summed-spectrum toggle state when this view becomes active."""
        summed = toggle_state.get("summed", False)
        if summed:
            self.toggle_summed_spectrum(True)

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill on all fit-lines panel axes."""
        import numpy as _np
        if not self._plot_grids:
            return

        # Check if any panel already has the summed fill rendered
        all_axes = []
        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                ax = getattr(panel, "ax", None)
                if ax is not None:
                    all_axes.append((ax, panel))

        if visible:
            # Compute summed spectrum data once
            wave = getattr(self._islat, "wave_data", None)
            mols = getattr(self._islat, "molecules_dict", None)
            s_wave, s_flux = None, None
            if mols is not None and wave is not None and len(wave) > 0:
                try:
                    s_wave, s_flux = mols.get_summed_flux(wave)
                except Exception as exc:
                    debug_config.warning(
                        "fit_lines_grid_view",
                        f"toggle_summed_spectrum: get_summed_flux failed: {exc}",
                    )

            for ax, panel in all_axes:
                existing = [c for c in ax.collections if hasattr(c, "_islat_summed")]
                if existing:
                    for coll in existing:
                        coll.set_visible(True)
                elif s_wave is not None and s_flux is not None:
                    # Render the fill into this panel's range
                    xlim = getattr(panel, "xlim", None)
                    if xlim is None:
                        xdata = getattr(panel, "wave_data", None)
                        if xdata is not None and len(xdata) > 0:
                            xlim = (float(xdata[0]), float(xdata[-1]))
                    if xlim is not None:
                        mask = (s_wave >= xlim[0]) & (s_wave <= xlim[1])
                        if _np.any(mask):
                            try:
                                panel._plot_summed_spectrum(
                                    ax, s_wave[mask], s_flux[mask], deduplicate=True
                                )
                            except Exception as exc:
                                debug_config.warning(
                                    "fit_lines_grid_view",
                                    f"toggle_summed_spectrum: _plot_summed_spectrum failed: {exc}",
                                )
        else:
            for ax, _ in all_axes:
                for coll in ax.collections:
                    if hasattr(coll, "_islat_summed"):
                        coll.set_visible(False)

        self.draw()

    def _is_summed_visible(self) -> bool:
        """Return ``True`` if the summed-spectrum toggle is currently on."""
        toggle_state = getattr(self._pm, "toggle_state", None)
        if isinstance(toggle_state, dict):
            return bool(toggle_state.get("summed", False))
        return False

    def _refresh_summed_spectrum(self) -> None:
        """Recompute and re-render the summed model fill if it is visible.

        The cached summed-fill collections are removed outright (not just
        hidden) so that :meth:`toggle_summed_spectrum` recomputes the summed flux from the current molecule parameters.
        No-op when the summed toggle is off or no grids are present.
        """
        if not (self._plot_grids and self._is_summed_visible()):
            return
        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                ax = getattr(panel, "ax", None)
                if ax is None:
                    continue
                for coll in [c for c in ax.collections if hasattr(c, "_islat_summed")]:
                    try:
                        coll.remove()
                    except Exception:
                        pass
        self.toggle_summed_spectrum(True)

    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """No-op."""
        pass

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """Show or hide saved-line annotations on every panel."""
        if not self._plot_grids:
            return
        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                if not hasattr(panel, 'plot_saved_lines'):
                    continue
                if show and loaded_lines is not None:
                    panel.plot_saved_lines(loaded_lines)
                else:
                    panel.remove_saved_lines()
        self.draw()

    def toggle_atomic_lines(self, show: bool) -> None:
        """Show or hide atomic-line markers on every panel."""
        if not self._plot_grids:
            return
        islat = getattr(self, '_islat', None)
        atomic_data = getattr(islat, 'atomic_lines', None) if islat is not None else None
        for plot_grid in self._plot_grids:
            for panel in plot_grid.iter_panels():
                if not hasattr(panel, 'plot_atomic_lines'):
                    continue
                if show and atomic_data is not None:
                    panel.plot_atomic_lines(atomic_data)
                else:
                    panel.remove_atomic_lines()
        self.draw()

    # ==================================================================
    # PlotView - canvas / drawing
    # ==================================================================
    def draw(self) -> None:
        """Flush all pending changes to every embedded canvas."""
        for canvas in self._tab_canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass
        # Also flush the placeholder canvas if it's active
        if self._canvas is not None and self._canvas not in self._tab_canvases:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    def get_canvas(self) -> "FigureCanvasTkAgg":
        """Return the primary :class:`FigureCanvasTkAgg` for this view."""
        return self._canvas

    def get_figure(self) -> "Figure":
        """Return the primary :class:`~matplotlib.figure.Figure` for this view."""
        return self._fig