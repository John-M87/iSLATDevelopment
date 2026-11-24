"""
HoldableCanvas - the TkAgg canvas used for iSLAT's plots.

:class:`HoldableFigureCanvasTkAgg` adds three things to matplotlib's
``FigureCanvasTkAgg``:

* It can hold its draws until Tk has laid its widget out
  (:meth:`~HoldableFigureCanvasTkAgg.hold_draws_until_laid_out`).  A new
  canvas widget starts at the figure's own size and is resized over the next
  event-loop rounds; every draw before that is at a provisional size.
* It can show its last rendered frame again without re-rendering the figure
  (:meth:`~HoldableFigureCanvasTkAgg.repaint_idle`), for a widget that was
  unpacked and packed again.
* It blits opaque figures to Tk as data without alpha, which spares Tk a
  per-pixel transparency scan (~20% of a full-window blit).
* It redraws only the axes that changed when it can
  (:meth:`~HoldableFigureCanvasTkAgg._draw_changed_axes_only`): a pan or a
  data update in one panel repaints that panel's area, without drawing the
  other panels or laying the figure out again.
"""
from __future__ import annotations

import contextlib
import math
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backend_bases import DrawEvent
from matplotlib.backends import _backend_tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.transforms import Bbox

from .CachedLayout import CachedLayoutMixin

#: ``Tk_PhotoImageBlock`` offsets of red, green, blue and alpha in RGBA8888 data.
RGBA_OFFSETS = (0, 1, 2, 3)
#: The same without alpha: an alpha offset outside the 4-byte pixel tells Tk
#: the block is fully opaque.
RGB_OFFSETS = (0, 1, 2, 4)


class HoldableFigureCanvasTkAgg(FigureCanvasTkAgg):
    """FigureCanvasTkAgg that can hold its draws while the window is still being laid out.

    While held, ``draw_idle`` requests (including the ones Tk resizes make)
    are only remembered; resizes still update the figure size.
    :meth:`release_draws` then makes one draw.  Explicit ``draw()`` calls
    are not held.
    """

    #: How often (ms) to check whether Tk has finished laying out the widget ...
    LAYOUT_CHECK_MS = 50
    #: ... and the longest (s) draws are held while waiting for that.
    LAYOUT_MAX_WAIT_S = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._holding_draws = False
        self._draw_requested = False
        self._idle_repaint_id: Optional[str] = None
        self._frame_drawn = False
        self._cells_key: Any = None
        self._cells: Optional[Dict[Any, Bbox]] = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Holding draws
    # ------------------------------------------------------------------
    @property
    def holding_draws(self) -> bool:
        return self._holding_draws

    def hold_draws(self) -> None:
        """Remember draw requests instead of drawing until :meth:`release_draws`."""
        self._holding_draws = True
        # Take over a draw that is already queued.
        queued = getattr(self, "_idle_draw_id", None)
        if queued:
            self.get_tk_widget().after_cancel(queued)
            self._idle_draw_id = None
            self._draw_requested = True

    def release_draws(self) -> None:
        """Stop holding draws; draw once if any draw was requested meanwhile."""
        if not self._holding_draws:
            return
        self._holding_draws = False
        if self._draw_requested:
            self._draw_requested = False
            super().draw_idle()

    def draw_idle(self) -> None:
        if self._holding_draws:
            self._draw_requested = True
            return
        super().draw_idle()

    def hold_draws_until_laid_out(self) -> None:
        """Hold draws until Tk has laid this canvas's widget out, then draw once.

        The widget counts as laid out once it is mapped, the figure has been
        resized to it and its size did not change since the previous check
        (every :attr:`LAYOUT_CHECK_MS`); after :attr:`LAYOUT_MAX_WAIT_S` the
        draws are released anyway.
        """
        self.hold_draws()
        self._release_when_laid_out(time.monotonic(), None)

    def _release_when_laid_out(self, started: float, last_size: Optional[Tuple[int, int]]) -> None:
        if not self._holding_draws:
            return  # released meanwhile
        widget = self.get_tk_widget()
        try:
            if not widget.winfo_exists():
                return
            size = (widget.winfo_width(), widget.winfo_height())
            mapped = bool(widget.winfo_ismapped())
            # Scheduled on the toplevel so the check survives the widget being destroyed.
            scheduler = widget.winfo_toplevel()
        except tk.TclError:
            return
        figure_size = tuple(int(round(v)) for v in self.figure.bbox.size)
        laid_out = mapped and size == last_size and figure_size == size
        if laid_out or time.monotonic() - started >= self.LAYOUT_MAX_WAIT_S:
            self.release_draws()
        else:
            scheduler.after(self.LAYOUT_CHECK_MS, self._release_when_laid_out, started, size)

    # ------------------------------------------------------------------
    # Showing the last frame again
    # ------------------------------------------------------------------
    def repaint_idle(self) -> None:
        """Show the last rendered frame again once Tk is idle, without re-rendering the figure.

        An unpacked canvas widget keeps its image, but when it is packed
        again Tk on macOS leaves the repaint to the system, which makes it
        only when the next input event (e.g. a mouse move) arrives: until
        then the widget is blank.  Blitting the kept frame changes the image,
        and Tk repaints the widget for that at once.  Call it after packing
        the widget.  Without a frame of the figure's current size this is a
        :meth:`draw_idle`; a draw already due blits anyway.
        """
        if self._holding_draws or getattr(self, "_idle_draw_id", None):
            self.draw_idle()
            return
        if self._idle_repaint_id is None:
            self._idle_repaint_id = self.get_tk_widget().after_idle(self._repaint)

    def _repaint(self) -> None:
        self._idle_repaint_id = None
        if getattr(self, "_idle_draw_id", None):
            return  # a draw was asked for meanwhile; it blits
        try:
            if not self.get_tk_widget().winfo_exists():
                return
        except tk.TclError:
            return
        renderer = getattr(self, "renderer", None)
        size = (*self.figure.bbox.size, self.figure.dpi)
        if renderer is None or (renderer.width, renderer.height, renderer.dpi) != size:
            self.draw_idle()
        else:
            self.blit()

    # ------------------------------------------------------------------
    # Blitting
    # ------------------------------------------------------------------
    def _figure_is_opaque(self) -> bool:
        """True when every pixel is opaque: the figure background is drawn without transparency."""
        patch = self.figure.patch
        return bool(patch.get_visible()) and patch.get_facecolor()[3] >= 1.0

    def blit(self, bbox: Any = None) -> None:
        offsets = RGB_OFFSETS if self._figure_is_opaque() else RGBA_OFFSETS
        _backend_tk.blit(self._tkphoto, self.renderer.buffer_rgba(), offsets, bbox=bbox)

    # ------------------------------------------------------------------
    # Redrawing only the axes that changed
    # ------------------------------------------------------------------
    def draw(self) -> None:
        if not self._draw_changed_axes_only():
            super().draw()
            self._frame_drawn = True

    def _draw_changed_axes_only(self) -> bool:
        """Repaint only the areas of the axes that changed; ``False`` when a full draw is needed.

        When nothing is stale the last frame is shown again.  A partial
        redraw is used when some (not all) axes are stale, the last frame is still in
        the buffer at the current size, the figure background is opaque, the
        figure has no artists of its own besides its background, and the
        layout keeps every axes where it is.  Each axes owns a cell of the
        figure (:meth:`_axes_cells`); a changed axes' cell is cleared to the
        figure background and the axes drawn into it.  Anything an axes draws
        outside its cell is undone, so the buffer stays exactly the full
        figure.  Listeners get the usual ``draw_event``.
        """
        figure = self.figure
        renderer = getattr(self, "renderer", None)
        if (not self._frame_drawn or renderer is None or self.get_renderer() is not renderer
                or not figure.get_visible() or not self._figure_is_opaque()):
            return False
        if not figure.stale:
            # Nothing changed since the last draw (matplotlib marks every
            # change made through its setters): show the frame we have.
            self.blit()
            return True
        axes = list(figure.axes)
        stale = [ax for ax in axes if ax.stale]
        if not stale or len(stale) == len(axes):
            return False
        if any(child is not figure.patch and child not in axes and child.get_visible()
               for child in figure.get_children()):
            return False  # suptitle, figure legend, figure text...
        engine = figure.get_layout_engine()
        if engine is not None and not isinstance(engine, CachedLayoutMixin):
            return False
        get_draw_artists = getattr(figure, "_get_draw_artists", None)
        if get_draw_artists is None:
            return False
        try:
            with getattr(figure, "_render_lock", None) or contextlib.nullcontext():
                positions = [ax.get_position().bounds for ax in axes]
                get_draw_artists(renderer)  # applies fixed aspects, as Figure.draw does
                if engine is not None:
                    engine.execute(figure)
                if [ax.get_position().bounds for ax in axes] != positions:
                    return False  # the layout moved something
                cells = self._axes_cells(axes, renderer)
                dirty = list(dict.fromkeys(cells[ax] for ax in stale if ax in cells))
                if len(dirty) == len(set(cells.values())):
                    return False  # every cell changed: no gain
                for cell in dirty:
                    self._redraw_cell(cell, [ax for ax in axes if cells.get(ax) is cell], renderer)
                # Reading the positions re-applies aspects, which marks every axes
                # stale; the ones not redrawn did not move, so their pixels are current.
                for ax in axes:
                    ax.stale = False
                figure.stale = False
        except Exception:
            return False  # anything unexpected: the full draw repaints everything
        DrawEvent("draw_event", self, renderer)._process()
        for cell in dirty:
            self.blit(cell)
        return True

    def _redraw_cell(self, cell: Bbox, cell_axes: List[Any], renderer: Any) -> None:
        """Clear *cell* to the figure background and draw *cell_axes* in it; leave the rest untouched."""
        figure = self.figure
        full = figure.bbox
        outside = [Bbox.from_extents(*extents) for extents in (
            (full.x0, full.y0, cell.x0, full.y1), (cell.x1, full.y0, full.x1, full.y1),
            (cell.x0, full.y0, cell.x1, cell.y0), (cell.x0, cell.y1, cell.x1, full.y1),
        ) if extents[2] > extents[0] and extents[3] > extents[1]]
        kept = [self.copy_from_bbox(box) for box in outside]
        patch = figure.patch
        clip_box, clip_on = patch.get_clip_box(), patch.get_clip_on()
        patch.set_clip_box(cell)
        patch.set_clip_on(True)
        try:
            patch.draw(renderer)
        finally:
            patch.set_clip_box(clip_box)
            patch.set_clip_on(clip_on)
            patch.stale = False
        for ax in sorted(cell_axes, key=lambda a: (a.get_zorder(), figure.axes.index(a))):
            ax.draw(renderer)
        for region in kept:
            self.restore_region(region)

    def _axes_cells(self, axes: List[Any], renderer: Any) -> Dict[Any, Bbox]:
        """Split the figure into one cell per visible axes (overlapping axes share one).

        Cells are measured from the axes' extents with their decorations, and
        reach halfway into the gaps between them, so they never overlap.
        Kept until an axes is added, removed, shown, hidden or moved.
        """
        figure = self.figure
        key = (tuple((id(ax), ax.get_visible(), ax.get_position().bounds) for ax in axes),
               tuple(figure.bbox.bounds), figure.dpi)
        if key == self._cells_key and self._cells is not None:
            return self._cells
        groups: List[Tuple[List[Any], Bbox]] = []
        for ax in axes:
            if not ax.get_visible():
                continue
            box = ax.get_tightbbox(renderer)
            if box is None:
                continue
            members = [ax]
            for group in [g for g in groups if g[1].overlaps(box)]:
                groups.remove(group)
                members = group[0] + members
                box = Bbox.union([group[1], box])
            groups.append((members, box))
        full = figure.bbox
        cells: Dict[Any, Bbox] = {}
        for members, box in groups:
            x0, y0, x1, y1 = full.extents
            for _, other in (g for g in groups if g[0] is not members):
                gap_x = max(other.x0 - box.x1, box.x0 - other.x1)
                gap_y = max(other.y0 - box.y1, box.y0 - other.y1)
                if gap_x >= gap_y:  # split at a vertical line in the middle of the gap
                    if other.x0 >= box.x1:
                        x1 = min(x1, math.floor((box.x1 + other.x0) / 2))
                    else:
                        x0 = max(x0, math.floor((other.x1 + box.x0) / 2))
                else:
                    if other.y0 >= box.y1:
                        y1 = min(y1, math.floor((box.y1 + other.y0) / 2))
                    else:
                        y0 = max(y0, math.floor((other.y1 + box.y0) / 2))
            cell = Bbox.from_extents(max(x0, full.x0), max(y0, full.y0), min(x1, full.x1), min(y1, full.y1))
            for ax in members:
                cells[ax] = cell
        self._cells_key, self._cells = key, cells
        return cells
