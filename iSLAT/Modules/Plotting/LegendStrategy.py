"""
LegendStrategy - Swappable legend providers for all iSLAT plots.

Defines a :class:`LegendStrategy` abstract base class and two concrete
implementations:

* :class:`StandardLegend` - the default for most plots.  Displays a
  standard matplotlib legend built from the visible labelled artists
  on an axes.
* :class:`MoleculeColorLegend` - the default for stacked-panel plots
  (:class:`FullSpectrumPlot`, :class:`ResidualSpectrumPlot`).  Renders
  a compact, text-only, per-molecule-color legend above the panel
  area.

Example
-------
Swap the legend with a custom implementation::

    class MinimalLegend(LegendStrategy):
        def build_legend(self, ax, fig, labels, colors, **kw):
            ax.legend(labels, loc="upper right", fontsize=6)
        def remove_legend(self, ax):
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        def update_visibility(self, ax, visible):
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(visible)
        def apply_theme(self, ax, theme):
            pass

    plot = FullSpectrumPlot(wave, flux, legend_strategy=MinimalLegend())
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

# ======================================================================
# Abstract base class
# ======================================================================
class LegendStrategy(ABC):
    """Interface for pluggable legend providers.

    Subclasses must implement all four methods.
    Each method receives the *axes* that owns the matplotlib legend object, along with any extra data needed to build, remove, or style it.
    """
    @abstractmethod
    def build_legend(
        self,
        ax: Axes,
        fig: MplFigure,
        labels: List[str],
        colors: List[str],
        *,
        fontsize: int = 10,
        max_ncols: Optional[int] = None,
    ) -> None:
        """Create or replace the legend on *ax*.

        Parameters
        ----------
        ax : Axes
            The axes that will own the legend artist.
        fig : Figure
            The parent figure - used for width / height queries.
        labels : list[str]
            Display names (one per entry).
        colors : list[str]
            Corresponding color specs.
        fontsize : int
            Base font size for legend text.
        max_ncols : int, optional
            Hard upper bound on the number of columns.
        """

    @abstractmethod
    def remove_legend(self, ax: Axes) -> None:
        """Remove the legend from *ax* if one exists."""

    @abstractmethod
    def update_visibility(self, ax: Axes, visible: bool) -> None:
        """Show or hide the legend without rebuilding it."""

    @abstractmethod
    def apply_theme(self, ax: Axes, theme: Dict[str, Any]) -> None:
        """Re-color legend text / frame to match the supplied theme.

        Implementations should **not** overwrite per-molecule colors
        (text entries tagged with ``_islat_mol_color``).
        """

# ======================================================================
# Concrete implementation - standard artist-based legend
# ======================================================================
class StandardLegend(LegendStrategy):
    """Standard matplotlib legend built from visible labelled artists.

    This is the default strategy used by :class:`BasePlot` and all non-stacked plot classes (e.g. :class:`LineInspectionPlot`, :class:`MainPlotGrid`).
    It filters the axes' handles/labels to only visible artists and creates a regular ``ax.legend()``.
    """
    def build_legend(
        self,
        ax: Axes,
        fig: MplFigure,
        labels: List[str],
        colors: List[str],
        *,
        fontsize: int = 10,
        max_ncols: Optional[int] = None,
    ) -> None:
        """Build a legend from the visible labelled artists on *ax*.

        The *labels* and *colors* parameters are **ignored** - the legend
        is derived from the axes' own ``get_legend_handles_labels()``.
        This keeps the call signature compatible with the ABC while
        preserving the standard-plot behaviour.
        """
        handles, ax_labels = ax.get_legend_handles_labels()
        visible_handles: List[Any] = []
        visible_labels: List[str] = []
        for h, lbl in zip(handles, ax_labels):
            try:
                is_visible = h.get_visible()
            except AttributeError:
                is_visible = h[0].get_visible() if len(h) > 0 else True
            if is_visible:
                visible_handles.append(h)
                visible_labels.append(lbl)
        if visible_handles:
            ncols = 2 if len(visible_handles) > 8 else 1
            if max_ncols is not None:
                ncols = min(ncols, max_ncols)
            ax.legend(visible_handles, visible_labels, ncols=ncols)
        else:
            self.remove_legend(ax)

    def remove_legend(self, ax: Axes) -> None:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    def update_visibility(self, ax: Axes, visible: bool) -> None:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(visible)

    def apply_theme(self, ax: Axes, theme: Dict[str, Any]) -> None:
        """Re-color legend frame and text to match *theme*."""
        legend = ax.get_legend()
        if legend is None:
            return
        bg = theme.get("graph_fill_color", theme.get("background", "white"))
        fg = theme.get("foreground", "black")
        frame = legend.get_frame()
        if legend.get_visible() and frame.get_visible():
            frame.set_facecolor(bg)
            frame.set_edgecolor(fg)
        for text in legend.get_texts():
            text.set_color(fg)

# ======================================================================
# Concrete implementation - molecule color legend
# ======================================================================
class MoleculeColorLegend(LegendStrategy):
    """Compact, text-only, per-molecule-colored legend.

    Each entry is rendered as bold colored text with an invisible handle patch, giving a color key above the plot.
    The number of columns is determined at render time so that the legend fits within the *panel* width and does not overlap the panels or the title.

    Legend text objects are tagged with ``_islat_mol_color = True`` so that the theme system does not overwrite them with the foreground color.
    """
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def build_legend(
        self,
        ax: Axes,
        fig: MplFigure,
        labels: List[str],
        colors: List[str],
        *,
        fontsize: int = 10,
        max_ncols: Optional[int] = None,
    ) -> None:
        # Always remove the old legend first.
        self.remove_legend(ax)

        if not labels:
            return

        from matplotlib.patches import Patch

        ncols = self._compute_ncols(
            ax, fig, labels, fontsize=fontsize, max_ncols=max_ncols,
        )

        # Invisible handle patches (text-only appearance)
        handles = [Patch(facecolor="none", edgecolor="none") for _ in colors]

        if fig.get_layout_engine() is not None:
            # Directly above the panel; the layout engine makes room for it.
            placement = dict(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                             bbox_transform=ax.transAxes, borderaxespad=0.2)
        else:
            # Fixed margins: centered at x=0.5 in the margin above the
            # topmost panel, in figure coordinates.
            placement = dict(loc="upper center", bbox_to_anchor=(0.5, self._safe_y_anchor(fig)),
                             bbox_transform=fig.transFigure)

        leg = ax.legend(
            handles,
            labels,
            ncols=min(ncols, len(labels)),
            handlelength=0,
            handletextpad=0,
            fontsize=fontsize,
            prop={"weight": "bold", "size": fontsize},
            frameon=False,
            **placement,
        )

        # Tag each text entry with its molecule color.
        for txt, col in zip(leg.get_texts(), colors):
            txt.set_color(col)
            txt._islat_mol_color = True  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def remove_legend(self, ax: Axes) -> None:
        old = ax.get_legend()
        if old is not None:
            old.remove()

    # ------------------------------------------------------------------
    def update_visibility(self, ax: Axes, visible: bool) -> None:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(visible)

    # ------------------------------------------------------------------
    def apply_theme(self, ax: Axes, theme: Dict[str, Any]) -> None:
        """Re-color legend frame / non-molecule text to match *theme*."""
        legend = ax.get_legend()
        if legend is None:
            return
        bg = theme.get("graph_fill_color", theme.get("background", "white"))
        fg = theme.get("foreground", "black")
        frame = legend.get_frame()
        if legend.get_visible() and frame.get_visible():
            frame.set_facecolor(bg)
            frame.set_edgecolor(fg)
        for text in legend.get_texts():
            if not getattr(text, "_islat_mol_color", False):
                text.set_color(fg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ncols(
        ax: Axes,
        fig: MplFigure,
        labels: List[str],
        *,
        fontsize: int = 10,
        max_ncols: Optional[int] = None,
    ) -> int:
        """Determine the number of columns that fit within the panel width.

        Uses the matplotlib renderer when available for exact text measurement; falls back to a character-width heuristic otherwise.
        """
        import numpy as np

        # --- Attempt renderer-based measurement -----------------------
        renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else None

        if renderer is not None:
            from matplotlib.text import Text

            # Measure each label's width in *display* (pixel) coordinates.
            tmp_texts = []
            widths_px: List[float] = []
            for lbl in labels:
                t = Text(0, 0, lbl, fontsize=fontsize, fontweight="bold")
                t.set_figure(fig)
                bbox = t.get_window_extent(renderer)
                widths_px.append(bbox.width)
                tmp_texts.append(t)

            padding_px = 18.0  # inter-column padding (approx)
            col_widths_px = [w + padding_px for w in widths_px]

            # Available width = panel width in pixels (left->right of
            # the *axes* area, obtained from the axes position).
            ax_bbox = ax.get_position()
            fig_w_px = fig.get_figwidth() * fig.dpi
            panel_width_px = ax_bbox.width * fig_w_px

            # Greedy bin-pack: start with max possible cols and shrink
            # until total row width fits.
            n = len(labels)
            for ncols in range(n, 0, -1):
                # Simulate wrapping: split labels into rows of `ncols`.
                max_row_width = 0.0
                for row_start in range(0, n, ncols):
                    row = col_widths_px[row_start : row_start + ncols]
                    max_row_width = max(max_row_width, sum(row))
                if max_row_width <= panel_width_px:
                    break
            else:
                ncols = 1
        else:
            # --- Heuristic fallback -----------------------------------
            fig_w_in = fig.get_figwidth()
            char_w_in = 0.085 * (fontsize / 10.0)
            avg_label_w = sum(len(lbl) for lbl in labels) / len(labels)
            col_w_in = avg_label_w * char_w_in + 0.3
            max_cols = max(int(fig_w_in / col_w_in), 1)
            ncols = min(max_cols, len(labels))

        if max_ncols is not None:
            ncols = min(ncols, max_ncols)

        return max(ncols, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_y_anchor(fig: MplFigure) -> float:
        """Compute a y-anchor in figure coordinates that avoids the title.

        The legend is placed in the gap between the top of the panel area (``subplots_adjust(top=…)``) and the figure edge (y = 1.0).
        If a ``suptitle`` exists, the anchor is shifted down so the legend sits just below the title.
        """
        # Determine the top edge of the panel area.
        sp = fig.subplotpars
        top = sp.top  # e.g. 0.93

        # Default: midpoint between top-of-panels and figure top.
        default_y = (top + 1.0) / 2.0  # e.g. 0.965

        # Check for a suptitle.
        suptitle = fig._suptitle  # type: ignore[attr-defined]
        if suptitle is not None and suptitle.get_text():
            renderer = (
                fig.canvas.get_renderer()
                if hasattr(fig.canvas, "get_renderer")
                else None
            )
            if renderer is not None:
                title_bbox = suptitle.get_window_extent(renderer)
                fig_h_px = fig.get_figheight() * fig.dpi
                # Bottom of the title in figure coordinates.
                title_bottom_fig = title_bbox.y0 / fig_h_px
                # Place legend just below the title with a small gap.
                return max(title_bottom_fig - 0.01, top + 0.005)
            else:
                # Without a renderer, nudge down conservatively.
                return top + 0.01

        return default_y