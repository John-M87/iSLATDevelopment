"""Constrained layout that is only recomputed when the layout can change.

Matplotlib's constrained layout runs on every figure draw.  In the Three
Panel view that is about a third of each redraw (~170 ms at full-screen
size), even for a pan, where nothing that sizes the margins has changed.

:class:`CachedConstrainedLayoutEngine` keeps the last layout and runs the
real one again only when one of its inputs changed:

* the figure size, dpi or the engine settings (pads, spacing);
* the axes in the figure and their grid placement, scales and aspect;
* titles, axis labels and tick settings;
* the y tick labels (their widths set the left / inter-panel margins);
* legends and other unclipped artists that constrained layout makes room for.

X limits are not an input, so a pan reuses the layout unless an x tick label
would then be cut off at the figure edge.  Saving always runs the real
layout, and the next screen draw lays out again.

The caching lives in :class:`CachedLayoutMixin`, which also serves the
stacked plots' :class:`~iSLAT.Modules.Plotting.StackedLayout.StackedPanelLayoutEngine`.
"""
from typing import Any, Tuple

from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.text import Text

#: How far (pixels) an x tick label may reach past the figure edge before
#: the layout is redone to make room for it.
EDGE_TOLERANCE_PX = 0.5

class CachedLayoutMixin:
    """Run a layout engine's ``execute`` only when something that sizes the layout changed.

    Put it before the engine class in the bases.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_key = None

    def execute(self, fig):
        if fig.canvas.is_saving():
            # Exports lay out with their own renderer / dpi; re-lay out the
            # screen figure afterwards.
            self._last_key = None
            return super().execute(fig)
        if self._last_key is not None and self._layout_key(fig) == self._last_key \
                and not self._edges_need_layout(fig):
            return None
        result = super().execute(fig)
        # Taken after the layout: the y tick labels depend on the new axes heights.
        self._last_key = self._layout_key(fig)
        return result

    def _edges_need_layout(self, fig) -> bool:
        """Whether something not in the key needs a new layout: an x tick label cut off at the figure edge."""
        return x_tick_labels_cut_off(fig)

    def _layout_key(self, fig) -> Tuple:
        """Everything the layout depends on, except the x limits."""
        return (
            tuple(fig.get_size_inches()), fig.dpi,
            repr(sorted(self.get().items())),
            tuple(_text_key(t, placed=False) for t in (fig._suptitle, fig._supxlabel, fig._supylabel) if t is not None),
            tuple(_legend_key(leg) for leg in fig.legends),
            tuple(_text_key(t) for t in fig.texts),
            len(fig.subfigs),
            tuple(_axes_key(ax) for ax in fig.axes),
        )

class CachedConstrainedLayoutEngine(CachedLayoutMixin, ConstrainedLayoutEngine):
    """A :class:`ConstrainedLayoutEngine` that skips the layout when its inputs are unchanged."""

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

def _axes_key(ax) -> Tuple:
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