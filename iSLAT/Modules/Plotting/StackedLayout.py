"""Tight layout for a vertical stack of equal cells (the full-spectrum style plots).

:class:`StackedPanelLayoutEngine` measures what sticks out of the cells of
the figure's outer one-column grid - tick labels, axis labels, the legend
above the first cell, texts beside the panels - and sets the grid's margins
and the gap between the cells to just fit it, plus a small pad.  Every cell
keeps the same height, including cells split into sub-panels (spectrum +
residual, tau + flux, one row per sample spectrum), which constrained
layout sizes unevenly when only the first or last cell has a legend or an
axis label.

Also handled:

* the figure's ``supylabel``, placed at the left edge;
* figure texts (e.g. the residual plot's total chi^2 box at the bottom),
  which get room at the figure edge they sit nearer to.

Through :class:`~iSLAT.Modules.Plotting.CachedLayout.CachedLayoutMixin` the
measuring runs again only when something that sizes the layout changed.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.layout_engine import LayoutEngine
from matplotlib.transforms import Bbox

from .CachedLayout import CachedLayoutMixin, _artist_key, _legend_key, _text_key

#: Smallest cell height (pixels) the layout accepts; with less room it leaves the figure alone.
MIN_CELL_PX = 8.0


class _StackedPanelLayout(LayoutEngine):
    """The uncached measuring layout (see the module docstring)."""

    _adjust_compatible = True
    _colorbar_gridspec = False  # as constrained layout, so the two can replace each other

    def __init__(self, *, pad: float = 4.0, gap_pad: float = 3.0, min_hspace: float = 0.0, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        pad : float
            Space (points) between the figure edge and anything drawn.
        gap_pad : float
            Space (points) added to what the cells need between them.
        min_hspace : float
            Smallest gap between cells, as a fraction of the cell height.
        """
        super().__init__(**kwargs)
        self._params = {"pad": float(pad), "gap_pad": float(gap_pad), "min_hspace": float(min_hspace)}

    def set(self, *, pad: Optional[float] = None, gap_pad: Optional[float] = None,
            min_hspace: Optional[float] = None) -> None:
        for name, value in (("pad", pad), ("gap_pad", gap_pad), ("min_hspace", min_hspace)):
            if value is not None:
                self._params[name] = float(value)

    def execute(self, fig) -> None:
        # Tick labels can change with the axes size, so measure again once
        # after moving the cells (stops early when nothing moved).
        for _ in range(2):
            if not self._layout_once(fig):
                break

    # ------------------------------------------------------------------
    def _layout_once(self, fig) -> bool:
        """Measure and place once; return whether the grid changed."""
        found = _outer_grid_cells(fig)
        if found is None:
            return False
        gs, cells = found
        renderer = fig._get_renderer()
        width, height = fig.bbox.width, fig.bbox.height
        if width <= 0 or height <= 0:
            return False
        to_px = fig.dpi / 72.0
        pad = self._params["pad"] * to_px

        # What sticks out of each cell (pixels).
        left = right = 0.0
        above: Dict[int, float] = defaultdict(float)
        below: Dict[int, float] = defaultdict(float)
        for row, axes in cells.items():
            box = Bbox.union([ax.bbox for ax in axes])
            tight_boxes = [b for b in (ax.get_tightbbox(renderer, for_layout_only=True) for ax in axes)
                           if b is not None]
            if not tight_boxes:
                continue
            tight = Bbox.union(tight_boxes)
            left = max(left, box.x0 - tight.x0)
            right = max(right, tight.x1 - box.x1)
            above[row] = max(tight.y1 - box.y1, 0.0)
            below[row] = max(box.y0 - tight.y0, 0.0)

        # Figure-level text: the supylabel at the left edge, figure texts at
        # the top or bottom edge they sit nearer to.
        left_edge = pad
        sup = fig._supylabel
        if sup is not None and sup.get_visible() and sup.get_text() and sup.get_in_layout():
            # Shift it so its drawn extent starts at the pad (it is rotated
            # and aligned around its x, so x alone does not say where it ends).
            ext = sup.get_window_extent(renderer)
            sup.set_x(sup.get_position()[0] + (pad - ext.x0) / width)
            left_edge = ext.width + 2 * pad
        top_edge = bottom_edge = pad
        sup_texts = {id(t) for t in (fig._suptitle, fig._supxlabel, fig._supylabel) if t is not None}
        for text in fig.texts:
            if id(text) in sup_texts:  # figure.texts also lists them
                continue
            if not (text.get_visible() and text.get_text() and text.get_in_layout()):
                continue
            ext = text.get_window_extent(renderer)
            if (ext.y0 + ext.y1) / 2 < height / 2:
                bottom_edge = max(bottom_edge, ext.y1 + 2 * pad)
            else:
                top_edge = max(top_edge, height - ext.y0 + 2 * pad)

        n = gs.nrows
        gap = (max((below[r] for r in range(n - 1)), default=0.0)
               + max((above[r] for r in range(1, n)), default=0.0)
               + self._params["gap_pad"] * to_px)
        left_px = left_edge + left
        right_px = pad + right
        top_px = top_edge + above[0]
        bottom_px = bottom_edge + below[n - 1]

        cell_h = (height - top_px - bottom_px - (n - 1) * gap) / n
        if cell_h < MIN_CELL_PX or left_px + right_px >= width - MIN_CELL_PX:
            return False
        hspace = max(gap / cell_h, self._params["min_hspace"]) if n > 1 else 0.0

        new = (left_px / width, 1.0 - right_px / width, 1.0 - top_px / height, bottom_px / height, hspace)
        old = (gs.left, gs.right, gs.top, gs.bottom, gs.hspace)
        scale = (width, width, height, height, cell_h)  # pixels per unit of each value
        if all(o is not None and abs(o - v) * s < 0.5 for o, v, s in zip(old, new, scale)):
            return False
        gs.update(left=new[0], right=new[1], top=new[2], bottom=new[3], hspace=new[4])
        # GridSpec.update only moves the axes of pyplot-managed figures.
        for ax in fig.axes:
            spec = ax.get_subplotspec()
            if spec is not None and spec.get_topmost_subplotspec().get_gridspec() is gs:
                ax._set_position(spec.get_position(fig))
        return True


class StackedPanelLayoutEngine(CachedLayoutMixin, _StackedPanelLayout):
    """:class:`_StackedPanelLayout` that measures again only when its inputs changed.

    Its key is cheaper than constrained layout's, so the check costs little on
    each draw: the y limits stand in for the y tick labels (same limits, size
    and tick settings give the same labels), and the panels prune their edge
    ticks, so no x tick label reaches the figure edge.
    """

    def _layout_key(self, fig) -> Tuple:
        return (
            tuple(fig.get_size_inches()), fig.dpi,
            repr(sorted(self.get().items())),
            tuple(_text_key(t, placed=False) for t in (fig._suptitle, fig._supxlabel, fig._supylabel)
                  if t is not None),
            tuple(_legend_key(leg) for leg in fig.legends),
            tuple(_text_key(t) for t in fig.texts),
            len(fig.subfigs),
            tuple(_stacked_axes_key(ax) for ax in fig.axes),
        )

    def _edges_need_layout(self, fig) -> bool:
        return False


def _stacked_axes_key(ax) -> Tuple:
    """What sizes *ax*'s decorations, without computing its ticks."""
    spec = ax.get_subplotspec()
    grid = None
    if spec is not None:
        gs = spec.get_gridspec()
        grid = (id(gs), gs.get_geometry(), spec.rowspan, spec.colspan,
                repr(gs.get_height_ratios()), getattr(gs, "hspace", None))

    def axis_key(axis):
        return (axis.get_visible(), axis.get_label_position(), axis.labelpad,
                _text_key(axis.label, placed=False),
                repr(sorted(axis.get_tick_params(which="major").items())),
                type(axis.get_major_locator()).__name__, type(axis.get_major_formatter()).__name__)

    return (
        id(ax), ax.get_visible(), ax.get_in_layout(), ax.axison, grid,
        ax.get_yscale(), tuple(ax.get_ylim()),
        tuple(_text_key(t, placed=False) for t in (ax.title, ax._left_title, ax._right_title)),
        axis_key(ax.xaxis), axis_key(ax.yaxis),
        tuple(_artist_key(a) for a in ax.get_default_bbox_extra_artists()),
    )


def _outer_grid_cells(fig) -> Optional[Tuple[Any, Dict[int, List[Any]]]]:
    """The figure's outer one-column grid and its visible axes by row, or ``None``.

    Axes in nested grids belong to the outer row their grid sits in.  Axes
    outside that grid (or not taking part in the layout) are ignored.
    """
    grid = None
    cells: Dict[int, List[Any]] = defaultdict(list)
    for ax in fig.axes:
        if not (ax.get_visible() and ax.get_in_layout()):
            continue
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        top = spec.get_topmost_subplotspec()
        gs = top.get_gridspec()
        if grid is None:
            if gs.ncols != 1:
                return None
            grid = gs
        if gs is not grid:
            continue
        cells[top.rowspan.start].append(ax)
    if grid is None or not cells:
        return None
    return grid, dict(cells)
