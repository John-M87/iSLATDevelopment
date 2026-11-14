"""Constrained layout that is only recomputed when the layout can change.

Matplotlib's constrained layout runs on every figure draw.  In the Three
Panel view that is about a third of each redraw (~170 ms at full-screen
size), even for a pan, where nothing that sizes the margins has changed.

:class:`CachedConstrainedLayoutEngine` keeps the last layout and runs the
real one again only when one of its inputs changed.  It checks in two steps:

1. A cheap key (a few ms) of everything that could size the layout: the
   figure size, dpi and engine settings (pads, spacing); the axes and their
   grid placement, scales and aspect; titles, axis labels and tick settings;
   the y tick labels (their widths set the left / inter-panel margins);
   legends and other unclipped artists that constrained layout makes room
   for.  Unchanged: the layout is kept.
2. When only some axes' decorations changed, what the layout actually uses
   of them is measured for those axes (~20 ms each): their *reach*, how far
   their tick labels, axis labels, titles, legends and unclipped texts stick
   out of them on each side (the layout tight bbox minus the position).  New
   tick label text of the same width, a renamed title of the same height,
   or a legend or text label moved around inside the axes change the cheap
   key but not the reach, so the layout is still kept.  The line inspection
   panel relabels its lines on every parameter change, so this saves a
   layout (~200-300 ms) there.  A change of the figure or of the axes'
   placement always lays out again.

X limits are not an input, so a pan reuses the layout unless an x tick label
would then be cut off at the figure edge: neither key looks at how far x
tick labels stick out sideways (they move with every pan).  Saving always
runs the real layout, and the next screen draw lays out again.

The caching lives in :class:`CachedLayoutMixin`, which also serves the
stacked plots' :class:`~iSLAT.Modules.Plotting.StackedLayout.StackedPanelLayoutEngine`.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
from matplotlib import artist as martist
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox

#: How far (pixels) an x tick label may reach past the figure edge before
#: the layout is redone to make room for it.
EDGE_TOLERANCE_PX = 0.5
#: Decimals (of a pixel) an axes' reach is compared to: changes below a
#: millionth of a pixel are round-off, and move the layout by even less.
REACH_DIGITS = 6

class CachedLayoutMixin:
    """Run a layout engine's ``execute`` only when something that sizes the layout changed.

    Put it before the engine class in the bases.  :meth:`_layout_key` is
    the cheap key.  When it changed, :meth:`_layout_unchanged` may still
    find that the layout would come out the same (by default it never does).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_key = None

    def execute(self, fig):
        if fig.canvas.is_saving():
            # Exports lay out with their own renderer / dpi; re-lay out the
            # screen figure afterwards.
            self._last_key = None
            self._forget_layout()
            return super().execute(fig)
        if self._last_key is not None and not self._edges_need_layout(fig):
            key = self._layout_key(fig)
            if key == self._last_key:
                return None
            if self._layout_unchanged(fig, key):
                # Something changed, but nothing the layout uses of it: keep
                # the layout, and take the new cheap key for the next draws.
                self._last_key = key
                return None
        result = super().execute(fig)
        # Taken after the layout: the y tick labels depend on the new axes heights.
        self._last_key = self._layout_key(fig)
        self._remember_layout(fig)
        return result

    def _edges_need_layout(self, fig) -> bool:
        """Whether something not in the key needs a new layout: an x tick label cut off at the figure edge."""
        return x_tick_labels_cut_off(fig)

    def _layout_key(self, fig) -> Tuple:
        """Everything the layout depends on, except the x limits."""
        return (
            _figure_key(self, fig),
            tuple(_axes_key(ax) for ax in fig.axes),
        )

    def _layout_unchanged(self, fig, key: Tuple) -> bool:
        """Whether the layout would come out the same although the cheap key changed to *key*."""
        return False

    def _remember_layout(self, fig) -> None:
        """Called after each layout run; record what :meth:`_layout_unchanged` compares against."""

    def _forget_layout(self) -> None:
        """Called when the layout must run again regardless."""

class CachedConstrainedLayoutEngine(CachedLayoutMixin, ConstrainedLayoutEngine):
    """A :class:`ConstrainedLayoutEngine` that skips the layout when its inputs are unchanged.

    Besides the cheap key it keeps every axes' reach (:func:`axes_reach`) as
    measured after the last layout.  When the cheap key changed only in some
    axes' decorations, the reach of those axes is measured again (~20 ms
    each); the layout is kept when none of them changed.  Each kept state has
    the same reaches as the last laid-out one, so the other axes' reaches
    still hold.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reaches: Optional[List[Any]] = None

    def _remember_layout(self, fig) -> None:
        renderer = fig._get_renderer()
        self._reaches = [axes_reach(ax, renderer) for ax in fig.axes]

    def _forget_layout(self) -> None:
        self._reaches = None

    def _layout_unchanged(self, fig, key: Tuple) -> bool:
        if self._reaches is None or self._last_key is None:
            return False
        figure_key, axes_keys = key
        last_figure_key, last_axes_keys = self._last_key
        if figure_key != last_figure_key or len(axes_keys) != len(last_axes_keys) \
                or len(axes_keys) != len(self._reaches):
            return False
        changed = []
        for index, (now, before) in enumerate(zip(axes_keys, last_axes_keys)):
            if now[0] != before[0]:
                return False  # an axes was added, moved, hidden, rescaled...
            if now != before:
                changed.append(index)
        if _decorations_depend_on_other_axes(fig):
            changed = range(len(axes_keys))
        renderer = fig._get_renderer()
        return all(axes_reach(fig.axes[index], renderer) == self._reaches[index] for index in changed)

def _decorations_depend_on_other_axes(fig) -> bool:
    """True when an axes' labels are placed with another axes' (twin axes, aligned labels)."""
    groups = list(getattr(fig, "_align_label_groups", {}).values())
    for ax in fig.axes:
        twins = getattr(ax, "_twinned_axes", None)
        if twins is not None and len(twins.get_siblings(ax)) > 1:
            return True
        if any(len(group.get_siblings(ax)) > 1 for group in groups):
            return True
    return False

def _figure_key(engine, fig) -> Tuple:
    return (
        tuple(fig.get_size_inches()), fig.dpi,
        repr(sorted(engine.get().items())),
        tuple(_text_key(t, placed=False) for t in (fig._suptitle, fig._supxlabel, fig._supylabel) if t is not None),
        tuple(_legend_key(leg) for leg in fig.legends),
        tuple(_text_key(t) for t in fig.texts),
        len(fig.subfigs),
    )

def x_tick_labels_cut_off(fig, tolerance: float = EDGE_TOLERANCE_PX) -> bool:
    """True when a drawn x tick label reaches past the left or right figure edge."""
    renderer = fig._get_renderer()
    width = fig.bbox.width
    for ax in fig.axes:
        if not (ax.get_visible() and ax.axison and ax.xaxis.get_visible()):
            continue
        for tick in ax.xaxis._update_ticks():
            for label in (tick.label1, tick.label2):
                if not (label.get_visible() and label.get_text()):
                    continue
                extent = label.get_window_extent(renderer)
                if extent.x0 < -tolerance or extent.x1 > width + tolerance:
                    return True
    return False

def _axes_placement_key(ax) -> Tuple:
    """The axes, its grid placement, scales and aspect."""
    spec = ax.get_subplotspec()
    if spec is None:
        grid = None
    else:
        gs = spec.get_gridspec()
        grid = (
            id(gs), gs.get_geometry(), spec.rowspan, spec.colspan,
            repr(gs.get_width_ratios()), repr(gs.get_height_ratios()),
            getattr(gs, "wspace", None), getattr(gs, "hspace", None),
        )
    return (
        id(ax), ax.get_visible(), ax.get_in_layout(), ax.axison, grid,
        ax.get_xscale(), ax.get_yscale(), repr(ax.get_aspect()), repr(ax.get_box_aspect()),
    )

def _axes_key(ax) -> Tuple:
    """Cheap key: the placement plus the decorations' text and settings (not their extents)."""
    return (
        _axes_placement_key(ax),
        tuple(_text_key(t, placed=False) for t in (ax.title, ax._left_title, ax._right_title)),
        _axis_key(ax.xaxis, with_tick_labels=False),
        _axis_key(ax.yaxis, with_tick_labels=True),
        tuple(_artist_key(a) for a in ax.get_default_bbox_extra_artists()),
    )

def _axis_key(axis, with_tick_labels: bool) -> Tuple:
    ticks = axis._update_ticks()
    labels = ()
    if with_tick_labels:
        labels = tuple(
            (label.get_text(), label.get_visible())
            for tick in ticks for label in (tick.label1, tick.label2)
        )
    return (
        axis.get_visible(), axis.get_label_position(), axis.labelpad,
        _text_key(axis.label, placed=False),
        repr(sorted(axis.get_tick_params(which="major").items())),
        repr(sorted(axis.get_tick_params(which="minor").items())),
        axis.major.formatter.get_offset(),
        labels,
    )

def axes_reach(ax, renderer) -> Optional[Tuple[float, float, float, float]]:
    """How far (pixels) *ax*'s decorations stick out of it: ``(left, right, bottom, top)``.

    Measured like constrained layout measures an axes (``Axes.get_tightbbox``
    with ``for_layout_only=True``: the axis tick labels and labels, the
    titles collapsed to their height, and the visible extra artists that
    are not clipped to the axes), whose difference to the axes position is
    all it uses of them.  Two differences keep a pan from forcing a layout:
    the x axis counts only for how far it reaches below / above the axes
    (x tick labels move sideways with the x limits; see
    :func:`x_tick_labels_cut_off`), and a legend anchored inside the axes
    that fits there counts as not sticking out, without placing it (a
    ``"best"`` legend is placed by searching the plotted data).  ``None``
    for a hidden axes.
    """
    if not ax.get_visible():
        return None
    axbox = ax.get_window_extent(renderer)
    boxes = [axbox]
    for name, axis in ax._axis_map.items():
        if not (ax.axison and axis.get_visible()):
            continue
        box = martist._get_tightbbox_for_layout_only(axis, renderer)
        if not box:
            continue
        if name == "x":
            box = Bbox.from_extents(axbox.x0, box.y0, axbox.x1, box.y1)
        boxes.append(box)
    ax._update_title_position(renderer)
    for title in (ax.title, ax._left_title, ax._right_title):
        if title.get_visible():
            box = title.get_window_extent(renderer)
            if box.width > 0:
                # As Axes.get_tightbbox(for_layout_only=True): only the title's height counts.
                middle = (box.x0 + box.x1) / 2
                box = Bbox.from_extents(middle - 0.5, box.y0, middle + 0.5, box.y1)
            boxes.append(box)
    for artist in ax.get_default_bbox_extra_artists():
        if isinstance(artist, Legend) and _legend_fits_inside(artist, ax, axbox, renderer):
            continue
        box = artist.get_tightbbox(renderer)
        if box is not None and np.all(np.isfinite(box.extents)) and (box.width > 0 or box.height > 0):
            boxes.append(box)
    union = Bbox.union(boxes)
    reach = (axbox.x0 - union.x0, union.x1 - axbox.x1, axbox.y0 - union.y0, union.y1 - axbox.y1)
    # Transform round-off (~1e-13 px, e.g. a spine's edge after a pan) is not a change.
    return tuple(round(float(value), REACH_DIGITS) for value in reach)

def _legend_fits_inside(legend: Legend, ax, axbox: Bbox, renderer) -> bool:
    """True when *legend* is anchored to *ax* and small enough that every placement keeps it inside.

    A legend anchored to its axes (no ``bbox_to_anchor``) with a location
    code (not an ``(x, y)`` position, which a dragged legend also gets) is
    placed in the axes box shrunk by ``borderaxespad``, at its ``loc`` or
    (``"best"``) at one of the nine standard locations; while it fits there
    it cannot stick out.
    """
    if legend._bbox_to_anchor is not None or legend.axes is not ax:
        return False
    if not isinstance(legend._loc, (int, np.integer)):
        return False
    box = legend._legend_box.get_bbox(renderer)
    pad = 2 * legend.borderaxespad * renderer.points_to_pixels(legend._fontsize)
    return box.width <= axbox.width - pad and box.height <= axbox.height - pad

def _artist_key(artist) -> Tuple:
    if isinstance(artist, Legend):
        return _legend_key(artist)
    if isinstance(artist, Text):
        return _text_key(artist)
    return (type(artist).__name__, artist.get_visible())

def _legend_key(legend) -> Tuple:
    return (
        "Legend", legend.get_visible(), repr(getattr(legend, "_loc", None)),
        getattr(legend, "_ncols", None),
        tuple(_text_key(t, placed=False) for t in legend.get_texts()),
        _text_key(legend.get_title(), placed=False),
    )

def _text_key(text, placed: bool = True) -> Tuple:
    """Key for *text*; *placed* = False for texts matplotlib positions itself
    (titles, axis labels, legend entries), whose positions follow the layout."""
    position = None
    if placed:
        try:
            position = tuple(float(v) for v in text.get_position())
        except (TypeError, ValueError):
            position = repr(text.get_position())
    return (
        "Text", text.get_visible(), text.get_text(), text.get_rotation(),
        hash(text.get_fontproperties()), position,
        text.get_horizontalalignment(), text.get_verticalalignment(),
    )