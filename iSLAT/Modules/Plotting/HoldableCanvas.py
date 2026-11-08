"""
HoldableCanvas - the TkAgg canvas used for iSLAT's plots.

:class:`HoldableFigureCanvasTkAgg` adds two things to matplotlib's
``FigureCanvasTkAgg``:

* It can hold its draws until Tk has laid its widget out
  (:meth:`~HoldableFigureCanvasTkAgg.hold_draws_until_laid_out`).  A new
  canvas widget starts at the figure's own size and is resized over the next
  event-loop rounds; every draw before that is at a provisional size.
* It blits opaque figures to Tk as data without alpha, which spares Tk a
  per-pixel transparency scan (~20% of a full-window blit).
"""
from __future__ import annotations

import time
import tkinter as tk
from typing import Any, Optional, Tuple

from matplotlib.backends import _backend_tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    # Blitting
    # ------------------------------------------------------------------
    def _figure_is_opaque(self) -> bool:
        """True when every pixel is opaque: the figure background is drawn without transparency."""
        patch = self.figure.patch
        return bool(patch.get_visible()) and patch.get_facecolor()[3] >= 1.0

    def blit(self, bbox: Any = None) -> None:
        offsets = RGB_OFFSETS if self._figure_is_opaque() else RGBA_OFFSETS
        _backend_tk.blit(self._tkphoto, self.renderer.buffer_rgba(), offsets, bbox=bbox)
