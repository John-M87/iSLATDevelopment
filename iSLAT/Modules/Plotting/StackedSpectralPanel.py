"""
StackedSpectralPanel -- Abstract composer for vertically stacked spectral panels.

Manages a collection of :class:`SpectralPanel` instances laid out in a vertical stack using matplotlib's :class:`~matplotlib.gridspec.GridSpec`.
Each row (cell) may contain an **arbitrary number** of sub-panels arranged vertically with configurable height ratios. 

For example, a single spectrum axes (as in :class:`FullSpectrumPlot`) or a spectrum + residual pair (as in :class:`ResidualSpectrumPlot`).

Concrete subclasses must implement :meth:`_create_cell` to produce the panels and axes for each wavelength row, and may override hooks like :meth:`_post_render_cell` and :meth:`_cell_height_ratios`.

Usage sketch::

    class MyStackedPlot(StackedSpectralPanel):
        def _create_cell(self, idx, xmin, xmax, gs_slot, **kw):
            ax = self.fig.add_subplot(gs_slot)
            panel = MyPanel(wave, flux, xmin, xmax, ax=ax)
            return [panel]

    plot = MyStackedPlot(wave, flux, n_panels=6)
    plot.generate_plot()
    plot.show()
"""
from __future__ import annotations

from abc import abstractmethod
from typing import (
    Callable,
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from .BasePlot import BasePlot
from .CompositePlot import CompositePlot
from .SpectralPanel import SpectralPanel, GapMode, XScaling
from .LegendStrategy import LegendStrategy, MoleculeColorLegend

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from .CompositeStackedPanel import CompositeStackedPanel

class StackedSpectralPanel(CompositePlot):
    """
    Abstract base for vertically stacked :class:`SpectralPanel` figures.

    Each row in the outer GridSpec is called a **cell**.  A cell may
    contain one or more sub-panels (axes) -- the concrete subclass
    decides via :meth:`_create_cell` and :meth:`_cell_height_ratios`.

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array.
    error_data : np.ndarray, optional
        Flux uncertainties.
    molecules : MoleculeDict, optional
        Molecule collection forwarded to each child panel.
    n_panels : int
        Target number of wavelength rows.  Default 5.
    step : float, optional
        Fixed wavelength width per row.  Overrides *n_panels*.
    xlim_range : tuple[float, float], optional
        ``(start, end)`` wavelength sub-range to display.  Defaults to
        the full extent of *wave_data*.
    ymax_factor : float
        Fractional headroom above the peak flux per panel (0.2 = 20 %).
    uniform_ylim : bool
        When *True* all panels share the same vertical scale.
        Default *False*.
    hspace : float
        Vertical spacing between cells in the outer GridSpec.
        Default 0.15.
    row_height : float
        Base height (inches) per cell used for auto-figsize.
        Default 2.0.
    figsize : tuple, optional
        Explicit figure size.  When *None* it is computed from
        ``(14, row_height * n_cells)``.
    **kwargs
        Forwarded to :class:`BasePlot`.
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        error_data: Optional[np.ndarray] = None,
        molecules: Optional["MoleculeDict"] = None,
        n_panels: int = 5,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        uniform_ylim: bool = False,
        hspace: float = 0.15,
        row_height: float = 2.0,
        figsize: Optional[Tuple[float, float]] = None,
        gap_mode: GapMode | str = GapMode.CONNECT,
        gap_threshold: Optional[float] = None,
        x_scaling: XScaling | str = XScaling.WAVELENGTH,
        **kwargs,
    ):
        super().__init__(figsize=figsize, **kwargs)

        # Stacked-panel plots default to the molecule color legend.
        if not isinstance(self.legend_strategy, MoleculeColorLegend):
            # Only override if caller didn't explicitly pass a strategy.
            if 'legend_strategy' not in kwargs:
                self.legend_strategy = MoleculeColorLegend()

        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.error_data = (
            np.asarray(error_data) if error_data is not None else None
        )
        self.molecules = molecules
        self.n_panels = n_panels
        self.ymax_factor = ymax_factor
        self.uniform_ylim = uniform_ylim
        self.hspace = hspace
        self.row_height = row_height

        # Gap handling
        if isinstance(gap_mode, str):
            gap_mode = GapMode(gap_mode)
        self.gap_mode: GapMode = gap_mode
        self.gap_threshold: Optional[float] = gap_threshold

        # Horizontal-axis scaling strategy
        if isinstance(x_scaling, str):
            x_scaling = XScaling(x_scaling)
        self.x_scaling: XScaling = x_scaling

        # Wavelength range
        if xlim_range is not None:
            self._xlim_start, self._xlim_end = xlim_range
        else:
            self._xlim_start = float(np.nanmin(self.wave_data))
            self._xlim_end = float(np.nanmax(self.wave_data))

        # Compute panel edges and ends
        self._compute_panel_layout(step=step)

        # Auto figsize
        if self._figsize is None:
            self._figsize = (14, self.row_height * len(self._panel_edges))

        # Storage: {row_index: list[SpectralPanel]}
        self.panels: Dict[int, List[SpectralPanel]] = {}

    # ------------------------------------------------------------------
    # Panel-layout computation
    # ------------------------------------------------------------------
    def _compute_panel_layout(
        self,
        step: Optional[float] = None,
    ) -> None:
        """Compute :attr:`_panel_edges` and :attr:`_panel_ends`.

        When :attr:`x_scaling` is ``WAVELENGTH`` (default) each panel
        covers an equal wavelength width - the classic behaviour.

        When :attr:`x_scaling` is ``DATA_DENSITY`` the panel boundaries
        are chosen so that each panel contains approximately the same
        number of finite observed data-points.  The resulting panels
        have **variable** wavelength widths but uniform horizontal
        data-point density.

        After this method runs:

        * ``_panel_edges[i]`` - left boundary of panel *i*
        * ``_panel_ends[i]``  - right boundary of panel *i*
        * ``_step``           - uniform step width (``WAVELENGTH`` mode)
          **or** ``None`` (``DATA_DENSITY`` mode).
        """
        start = self._xlim_start
        end = self._xlim_end

        if self.x_scaling is XScaling.DATA_DENSITY:
            self._step = None  # not meaningful for variable-width panels
            self._panel_edges, self._panel_ends = (
                self._density_edges(start, end, self.n_panels)
            )
        else:
            # Equal-wavelength-width mode (original behaviour).
            if step is not None:
                self._step = step
                edges = np.arange(start, end, self._step)
                self._panel_edges = edges
                self._panel_ends = np.append(edges[1:], end)
            else:
                # Use linspace to guarantee exactly n_panels panels
                # (np.arange may produce off-by-one due to float rounding).
                n = max(self.n_panels, 1)
                boundaries = np.linspace(start, end, n + 1)
                self._step = float(boundaries[1] - boundaries[0])
                self._panel_edges = boundaries[:-1]
                self._panel_ends = boundaries[1:]

    def _density_edges(
        self,
        start: float,
        end: float,
        n_panels: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(edges, ends)`` placing boundaries at data-quantiles.

        Only *finite* flux values inside ``[start, end]`` are counted.
        If there are fewer data points than panels the function falls
        back to equal-wavelength spacing.
        """
        mask = (
            (self.wave_data >= start)
            & (self.wave_data <= end)
            & np.isfinite(self.flux_data)
        )
        wave_in_range = np.sort(self.wave_data[mask])

        if len(wave_in_range) < n_panels or n_panels <= 1:
            # Not enough data - fall back to equal-wavelength.
            step = (end - start) / max(n_panels, 1)
            edges = np.arange(start, end, step)
            return edges, np.append(edges[1:], end)

        # Quantile-based boundaries so each panel has ~equal point count.
        quantiles = np.linspace(0, 1, n_panels + 1)
        boundaries = np.quantile(wave_in_range, quantiles)

        # Snap the outermost boundaries to the exact data range.
        boundaries[0] = start
        boundaries[-1] = end

        edges = boundaries[:-1]
        ends = boundaries[1:]
        return edges, ends

    def get_panel_end(self, idx: int) -> float:
        """Return the right wavelength boundary for panel *idx*.

        Works for both equal-width and data-density modes.
        """
        return float(self._panel_ends[idx])

    # ------------------------------------------------------------------
    # CompositePlot abstract-method implementations
    # ------------------------------------------------------------------
    def _build_layout(self) -> "GridSpec":
        """Create the outer ``GridSpec`` for the stacked rows.

        .. note::
           :meth:`StackedSpectralPanel.generate_plot` overrides the
           default :class:`CompositePlot` orchestration and calls the
           equivalent logic internally.
        """
        active_indices, _ = self._active_panel_edges()
        n_active = len(active_indices)
        gs = GridSpec(
            nrows=max(n_active, 1),
            ncols=1,
            figure=self.fig,
            hspace=self.hspace,
        )
        self.fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.06)
        return gs

    def _create_panels(self, layout: "GridSpec") -> None:
        """Cell creation is handled by :meth:`_create_cell` inside
        :meth:`generate_plot`, which overrides the default
        :class:`CompositePlot` orchestration."""

    # ------------------------------------------------------------------
    # Abstract factory -- subclasses produce the concrete cell contents
    # ------------------------------------------------------------------
    @abstractmethod
    def _create_cell(
        self,
        idx: int,
        xmin: float,
        xmax: float,
        gs_slot: "SubplotSpec",
        **kwargs,
    ) -> List[SpectralPanel]:
        """Create and return one or more :class:`SpectralPanel` instances
        for a single cell (row) in the stacked figure.

        The subclass is responsible for creating the axes (via
        ``self.fig.add_subplot`` or a nested ``GridSpecFromSubplotSpec``)
        and attaching them to the panels.

        Parameters
        ----------
        idx : int
            Row index (0-based).
        xmin, xmax : float
            Wavelength bounds for this row.
        gs_slot : SubplotSpec
            The outer-GridSpec slot allocated to this row.  Use it
            directly with ``fig.add_subplot(gs_slot)`` for a single-axes
            cell, or split it further with ``GridSpecFromSubplotSpec``
            for multi-axes cells.
        **kwargs
            Extra keyword arguments forwarded from :meth:`generate_plot`.

        Returns
        -------
        list[SpectralPanel]
            One or more panels that will be rendered in this row.
        """
        ...

    # ------------------------------------------------------------------
    # Optional per-row hook
    # ------------------------------------------------------------------
    def _post_render_cell(
        self,
        idx: int,
        cell_panels: List[SpectralPanel],
        is_last: bool,
    ) -> None:
        """Called after all panels in a cell have been generated.

        Subclasses can override this to add per-row tick formatting,
        labels, annotations, chi-squared boxes, etc.  The default
        implementation applies common tick locators and hides x-axis
        labels on all but the last row.
        """
        fg = self._get_theme_value("foreground", "black")
        for p_idx, panel in enumerate(cell_panels):
            ax = panel.ax
            if ax is None:
                continue
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            is_bottom_ax = (p_idx == len(cell_panels) - 1)
            if is_last and is_bottom_ax:
                ax.set_xlabel(
                    "Wavelength (\u03bcm)", fontsize=8, color=fg,
                )
            else:
                ax.tick_params(axis="x", labelbottom=is_bottom_ax)

        # --- Gap indicators (drawn after y-limits are finalised) -------
        if self.gap_mode is GapMode.SKIP:
            for panel in cell_panels:
                panel.draw_gap_indicators()

    # ------------------------------------------------------------------
    # y-limit computation
    # ------------------------------------------------------------------
    @staticmethod
    def _default_ylim_fn(
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        mask: np.ndarray,
        ymax_factor: float = 0.2,
    ) -> Tuple[float, float]:
        """Compute default ``(ymin, ymax)`` from observed flux."""
        if np.any(mask):
            finite = np.isfinite(flux_data[mask])
            if np.any(finite):
                peak = float(np.nanmax(flux_data[mask][finite]))
                return (-0.005, peak + peak * ymax_factor)
        return (-0.005, 0.1)

    def _compute_panel_ylims(
        self,
        uniform: Optional[bool] = None,
        ylim_fn: Optional[
            Callable[[np.ndarray], Tuple[float, float]]
        ] = None,
    ) -> List[Tuple[float, float]]:
        """Compute per-cell y-limits.

        Parameters
        ----------
        uniform : bool, optional
            When *True* all entries share a global min/max.  Defaults to
            :attr:`uniform_ylim`.
        ylim_fn : callable, optional
            ``fn(mask) -> (ymin, ymax)``.  Defaults to a closure around
            :meth:`_default_ylim_fn`.
        """
        if uniform is None:
            uniform = self.uniform_ylim

        if ylim_fn is None:
            def ylim_fn(mask: np.ndarray) -> Tuple[float, float]:
                return self._default_ylim_fn(
                    self.wave_data, self.flux_data, mask, self.ymax_factor,
                )

        ylims: List[Tuple[float, float]] = []
        for idx, xlim_start in enumerate(self._panel_edges):
            panel_end = self._panel_ends[idx]
            mask = (self.wave_data >= xlim_start) & (
                self.wave_data <= panel_end
            )
            ylims.append(ylim_fn(mask))

        if uniform and ylims:
            g_ymin = min(yl[0] for yl in ylims)
            g_ymax = max(yl[1] for yl in ylims)
            ylims = [(g_ymin, g_ymax)] * len(ylims)

        return ylims

    # ------------------------------------------------------------------
    # Post-render annotation helpers (delegate to each panel)
    # ------------------------------------------------------------------
    def plot_atomic_lines(
        self,
        atomic_df: Any,
        tag: str = "_islat_atomic_line",
    ) -> None:
        """Draw atomic-line markers on every panel that supports it."""
        if atomic_df is None or (hasattr(atomic_df, "__len__") and len(atomic_df) == 0):
            return
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if hasattr(panel, "plot_atomic_lines"):
                    panel.plot_atomic_lines(atomic_df, tag=tag)

    def remove_atomic_lines(
        self,
        tag: str = "_islat_atomic_line",
    ) -> None:
        """Remove atomic-line artists from every panel."""
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if hasattr(panel, "remove_atomic_lines"):
                    panel.remove_atomic_lines(tag=tag)

    def plot_saved_lines(
        self,
        line_data: Any,
        tag: str = "_islat_saved_line",
    ) -> None:
        """Draw saved-line annotations on every panel that supports it."""
        if line_data is None or (hasattr(line_data, "__len__") and len(line_data) == 0):
            return
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if hasattr(panel, "plot_saved_lines"):
                    panel.plot_saved_lines(line_data, tag=tag)

    def remove_saved_lines(
        self,
        tag: str = "_islat_saved_line",
    ) -> None:
        """Remove saved-line artists from every panel."""
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if hasattr(panel, "remove_saved_lines"):
                    panel.remove_saved_lines(tag=tag)

    # ------------------------------------------------------------------
    # Cell / panel label helpers
    # ------------------------------------------------------------------
    _LABEL_TAG = "_islat_cell_label"

    def set_cell_labels(
        self,
        labels: Dict[int, str],
    ) -> None:
        """Assign text labels to specific cells (rows).

        The label is drawn in the **top-left corner** of the first
        panel in the cell.  Previously drawn cell labels are removed
        before new ones are placed.

        Parameters
        ----------
        labels : dict[int, str]
            Mapping of ``{cell_index: label_text}``.  Cells not present
            in the dict are left unlabelled.
        """
        self._cell_labels = dict(labels)
        # If the figure already exists, apply immediately.
        if self.panels:
            self._apply_cell_labels(self._cell_labels)

    def set_panel_labels(
        self,
        labels: Dict[Tuple[int, int], str],
    ) -> None:
        """Assign text labels to specific sub-panels within cells.

        Parameters
        ----------
        labels : dict[tuple[int, int], str]
            Mapping of ``{(cell_index, panel_index): label_text}``.
        """
        self._panel_labels = dict(labels)
        if self.panels:
            self._apply_panel_labels(self._panel_labels)

    def _apply_cell_labels(
        self,
        labels: Dict[int, str],
    ) -> None:
        """Render cell labels on the first panel of each labelled cell."""
        fg = self._get_theme_value("foreground", "black")
        for cell_idx, lbl in labels.items():
            cell_panels = self.panels.get(cell_idx)
            if not cell_panels or not lbl:
                continue
            ax = cell_panels[0].ax
            if ax is None:
                continue
            # Remove any existing label for this cell.
            tag = f"{self._LABEL_TAG}_{cell_idx}"
            self._clear_tagged_artists(
                ax, tag,
                lines=False, collections=False, texts=True,
            )
            txt = ax.text(
                0.01, 0.97, lbl,
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                color=fg,
                alpha=0.75,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=self._get_theme_value("background", "white"),
                    edgecolor="none",
                    alpha=0.6,
                ),
                zorder=100,
            )
            setattr(txt, tag, True)

    def _apply_panel_labels(
        self,
        labels: Dict[Tuple[int, int], str],
    ) -> None:
        """Render labels on individual sub-panels."""
        fg = self._get_theme_value("foreground", "black")
        for (cell_idx, panel_idx), lbl in labels.items():
            cell_panels = self.panels.get(cell_idx)
            if not cell_panels or panel_idx >= len(cell_panels) or not lbl:
                continue
            ax = cell_panels[panel_idx].ax
            if ax is None:
                continue
            tag = f"{self._LABEL_TAG}_{cell_idx}_{panel_idx}"
            self._clear_tagged_artists(
                ax, tag, lines=False, collections=False, texts=True,
            )
            txt = ax.text(
                0.01, 0.97, lbl,
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                color=fg,
                alpha=0.75,
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=self._get_theme_value("background", "white"),
                    edgecolor="none",
                    alpha=0.6,
                ),
                zorder=100,
            )
            setattr(txt, tag, True)

    def clear_cell_labels(self) -> None:
        """Remove all cell / panel labels from every panel."""
        for cell_idx, cell_panels in self.panels.items():
            for p_idx, panel in enumerate(cell_panels):
                ax = panel.ax
                if ax is None:
                    continue
                self._clear_tagged_artists(
                    ax, f"{self._LABEL_TAG}_{cell_idx}",
                    lines=False, collections=False, texts=True,
                )
                self._clear_tagged_artists(
                    ax, f"{self._LABEL_TAG}_{cell_idx}_{p_idx}",
                    lines=False, collections=False, texts=True,
                )
        self._cell_labels = {}
        self._panel_labels = {}

    # ------------------------------------------------------------------
    # Gap-skip helpers
    # ------------------------------------------------------------------
    _GAP_SKIP_TAG = "_islat_gap_skip"

    def _cell_has_data(self, xmin: float, xmax: float) -> bool:
        """Return *True* if the wavelength window ``[xmin, xmax]``
        contains at least one finite observed-flux data point."""
        mask = (self.wave_data >= xmin) & (self.wave_data <= xmax)
        if not np.any(mask):
            return False
        return bool(np.any(np.isfinite(self.flux_data[mask])))

    def _active_panel_edges(
        self,
    ) -> Tuple[List[int], np.ndarray]:
        """Return the indices and edges of cells that have data.

        When :attr:`gap_mode` is not ``SKIP`` this returns all cells.

        Returns
        -------
        active_indices : list[int]
            Indices into :attr:`_panel_edges` that contain data.
        active_edges : np.ndarray
            Corresponding edge values.
        """
        if self.gap_mode is not GapMode.SKIP:
            return list(range(len(self._panel_edges))), self._panel_edges

        active_idx: List[int] = []
        for i, edge in enumerate(self._panel_edges):
            xmin = edge
            xmax = self._panel_ends[i]
            if self._cell_has_data(xmin, xmax):
                active_idx.append(i)
        return active_idx, self._panel_edges[active_idx] if active_idx else np.array([])

    def _draw_gap_skip_annotation(
        self,
        ax_above: Axes,
        ax_below: Axes,
        skipped_start: float,
        skipped_end: float,
    ) -> None:
        """Draw a small annotation between two axes indicating that a
        wavelength region was skipped.

        The annotation is placed at the bottom of *ax_above* in axes
        coordinates.
        """
        fg = self._get_theme_value("foreground", "black")
        lbl = f"\u2702 {skipped_start:.3f}\u2013{skipped_end:.3f} \u03bcm skipped"
        txt = ax_above.text(
            0.5, -0.01, lbl,
            transform=ax_above.transAxes,
            fontsize=6,
            color=fg,
            alpha=0.55,
            ha="center",
            va="top",
            fontstyle="italic",
        )
        setattr(txt, self._GAP_SKIP_TAG, True)

    # ------------------------------------------------------------------
    # Top-level plot generation
    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the stacked multi-panel figure.

        For each panel edge a cell is created (via :meth:`_create_cell`),
        each child panel's :meth:`~SpectralPanel.generate_plot` is
        invoked, and :meth:`_post_render_cell` is called for per-row
        decoration.

        When :attr:`gap_mode` is :attr:`GapMode.SKIP`, cells whose
        wavelength range contains no finite observed data are omitted
        entirely.  A small "skipped" annotation is placed between
        adjacent rows that span a gap.
        """
        # Determine which cells to render.
        active_indices, active_edges = self._active_panel_edges()
        n_active = len(active_indices)
        if n_active == 0:
            # Nothing to draw -- create an empty figure.
            self._ensure_figure()
            self.fig.clf()
            self.panels.clear()
            return

        # Adjust figsize for the number of *active* rows.
        if self.gap_mode is GapMode.SKIP:
            self._figsize = (self._figsize[0], self.row_height * n_active)

        self._ensure_figure()
        self.fig.clf()
        self.panels.clear()
        self.child_panels.clear()

        # Disable constrained_layout for uniform cell heights.
        self.fig.set_layout_engine(None)

        # Compute hspace so x-axis tick labels on each panel are not
        # covered by the axes background of the row below.  hspace is the
        # inter-panel gap expressed as a fraction of the average axes
        # height.  A target gap of 0.22 in accommodates 7-pt tick labels
        # plus padding at typical screen DPIs.  The value is clamped so it
        # is always at least self.hspace and at most 0.8.
        _margin_top = 0.93
        _margin_bottom = 0.06
        figH = (self._figsize or (12, 8.0))[1]
        _avg_ax_h = (_margin_top - _margin_bottom) * figH / max(n_active, 1)
        _target_gap = 0.22  # inches needed for tick labels + padding
        _needed_hspace = _target_gap / _avg_ax_h if _avg_ax_h > 0 else 0.35
        render_hspace = min(max(self.hspace, _needed_hspace), 0.8)

        gs = GridSpec(
            nrows=n_active,
            ncols=1,
            figure=self.fig,
            hspace=render_hspace,
        )

        self.fig.subplots_adjust(
            left=0.06, right=0.94, top=_margin_top, bottom=_margin_bottom,
        )

        fg = self._get_theme_value("foreground", "black")

        # Map from active row position -> original cell index.
        prev_bottom_ax: Optional[Axes] = None
        prev_original_idx: Optional[int] = None

        for row_pos, orig_idx in enumerate(active_indices):
            is_last = row_pos == n_active - 1
            xlim_start = self._panel_edges[orig_idx]
            panel_end = self._panel_ends[orig_idx]
            xmin, xmax = xlim_start, panel_end

            # Delegate cell creation to the concrete subclass.
            cell_panels = self._create_cell(
                orig_idx, xmin, xmax, gs[row_pos, 0], **kwargs,
            )
            self.panels[orig_idx] = cell_panels
            # Register in the CompositePlot child-panel registry.
            for p_idx, panel in enumerate(cell_panels):
                self._register_panel(f"cell_{orig_idx}_{p_idx}", panel)

            # Render each sub-panel in the cell.
            for panel in cell_panels:
                panel.generate_plot(**kwargs)

            # Per-row decoration hook.
            self._post_render_cell(orig_idx, cell_panels, is_last)

            # --- Gap-skip annotation between non-adjacent rows ---------
            if (
                self.gap_mode is GapMode.SKIP
                and prev_original_idx is not None
                and orig_idx != prev_original_idx + 1
                and prev_bottom_ax is not None
            ):
                # There were skipped cells between prev and current row.
                skipped_start = self._panel_ends[prev_original_idx]
                skipped_end = xlim_start
                self._draw_gap_skip_annotation(
                    prev_bottom_ax,
                    cell_panels[0].ax,
                    skipped_start,
                    skipped_end,
                )

            # Track the bottom axes of this cell for annotation.
            prev_bottom_ax = cell_panels[-1].ax
            prev_original_idx = orig_idx

        # --- Global figure labels --------------------------------------
        self.fig.supylabel(
            "Flux Density (Jy)", fontsize=10, color=fg, x=0.01,
        )

        # Apply theme
        self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # Composition – stack two plots together
    # ------------------------------------------------------------------
    def stack_with(
        self,
        other: StackedSpectralPanel,
        *,
        hspace: float = 0.25,
        row_height: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        labels: Optional[Tuple[str, str]] = None,
    ) -> "CompositeStackedPanel":
        """Combine two stacked-spectral plots into a single new panel.

        Cells from *other* are paired with cells from *self* by
        **closest matching wavelength range** (midpoint proximity).
        Each matched pair occupies two consecutive rows - *self*'s cell
        on top, *other*'s cell directly beneath.

        Unmatched cells (those with no close counterpart in the partner
        plot) are appended at the end of the figure.

        Both plots are **re-rendered** into a shared figure so that the
        axes, annotations, and theme all come out correctly.

        Parameters
        ----------
        other : StackedSpectralPanel
            The second plot whose cells will be interleaved beneath the
            matching cells of *self*.
        hspace : float
            Vertical spacing between rows in the composite GridSpec.
            Default 0.25.
        row_height : float, optional
            Base row height (inches) for each cell in the composite.
            Defaults to ``max(self.row_height, other.row_height)``.
        figsize : tuple, optional
            Explicit figure size.  When *None* it is computed from the
            total number of rows and ``row_height``.
        labels : tuple[str, str], optional
            Descriptive labels for *self* and *other* placed as text
            annotations in the top-left corner of each panel row.  For
            example ``("Observation A", "Observation B")``.

        Returns
        -------
        CompositeStackedPanel
            A new :class:`StackedSpectralPanel` subclass instance that
            holds the combined figure.  It supports the full
            ``show`` / ``save`` / ``close`` / annotation API inherited
            from :class:`BasePlot`.

        Examples
        --------
        >>> composite = plot_a.stack_with(plot_b)
        >>> composite.show()

        Using the ``+`` operator:

        >>> composite = plot_a + plot_b
        >>> composite.save("combined.png")
        """
        from .CompositeStackedPanel import CompositeStackedPanel

        return CompositeStackedPanel.from_pair(
            self,
            other,
            hspace=hspace,
            row_height=row_height,
            figsize=figsize,
            labels=labels,
        )

    def __add__(self, other: StackedSpectralPanel) -> "CompositeStackedPanel":
        """Syntactic sugar: ``plot_a + plot_b`` = ``plot_a.stack_with(plot_b)``."""
        if not isinstance(other, StackedSpectralPanel):
            return NotImplemented
        return self.stack_with(other)

# ======================================================================
# Helper – extract the kwargs that _create_cell expects
# ======================================================================
def _extract_cell_kwargs(ssp: StackedSpectralPanel) -> Dict[str, Any]:
    """Collect the keyword arguments a concrete subclass passes to
    ``_create_cell`` during its ``generate_plot`` override.

    For :class:`FullSpectrumPlot` (and its subclass
    :class:`ResidualSpectrumPlot`) this includes the molecule cache and
    summed spectrum arrays.  For unknown subclasses we return an empty
    dict - callers can always extend this function.
    """
    kw: Dict[str, Any] = {}

    # FullSpectrumPlot forwards mol_cache, summed_wave, summed_flux.
    if hasattr(ssp, "_build_mol_cache"):
        mol_cache, _labels, _colors = ssp._build_mol_cache()
        kw["mol_cache"] = mol_cache

    if hasattr(ssp, "molecules") and ssp.molecules is not None:
        try:
            wave_obs = getattr(ssp, "wave_data_obs", ssp.wave_data)
            sw, sf = ssp.molecules.get_summed_flux(
                wave_obs, visible_only=True, extend_to_global_range=True,
            )
            kw["summed_wave"] = sw
            kw["summed_flux"] = sf
        except Exception:
            pass

    return kw