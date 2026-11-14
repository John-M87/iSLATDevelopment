"""
ErrorBars - vertical error bars for long spectra, drawn as one path.

``Axes.errorbar`` gives every error bar its own path.  For a MIRI spectrum
(~10,700 points) creating them takes ~280 ms, and every redraw walks all
those paths again (~50 ms at full-screen size).  :func:`fast_errorbar`
draws the same thing with all the bars in a single compound path: it is
created in ~2 ms and redraws in about half the time.

It returns the same :class:`~matplotlib.container.ErrorbarContainer`
(data line, no caps, one bar collection) and adds it to the axes, so the
legend entry, autoscaling and code that looks at the container are
unchanged.  On screen the result matches ``Axes.errorbar`` to within one
colour level on a few edge pixels, as long as neighbouring bars do not
share a pixel column.  Where they do (a long spectrum squeezed into a few
hundred pixels) overlapping bars are no longer anti-aliased twice, so
those columns come out slightly lighter.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.path import Path


class ErrorBarCollection(LineCollection):
    """A :class:`LineCollection` holding every vertical error bar in one compound path.

    :meth:`get_segments` still returns one ``[[x, low], [x, high]]`` segment
    per bar, as for the collection ``Axes.errorbar`` makes.  Bars with a
    non-finite end (NaN flux or error) are left out, as Agg skips them
    there too.

    The path is always snapped to whole pixels.  Agg snaps each two-point
    path of ``Axes.errorbar`` on its own ("auto" snapping), but it never
    auto-snaps a path of more than 1024 vertices, so without this the bars
    would be drawn blurred across two pixel columns.
    """

    def __init__(self, x: Any, low: Any, high: Any, **kwargs: Any) -> None:
        kwargs.setdefault("snap", True)
        super().__init__([], **kwargs)
        self._bar_path: Optional[Path] = None
        self.set_bars(x, low, high)

    def set_bars(self, x: Any, low: Any, high: Any) -> None:
        """Replace the bars with one from ``(x[i], low[i])`` to ``(x[i], high[i])`` per point."""
        x = np.asarray(x, dtype=float).ravel()
        low = np.asarray(low, dtype=float).ravel()
        high = np.asarray(high, dtype=float).ravel()
        finite = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
        x, low, high = x[finite], low[finite], high[finite]
        if len(x) == 0:
            self._bar_path = None
            self._paths = []
            self.stale = True
            return
        vertices = np.empty((2 * len(x), 2))
        vertices[0::2, 0] = x
        vertices[1::2, 0] = x
        vertices[0::2, 1] = low
        vertices[1::2, 1] = high
        codes = np.empty(2 * len(x), dtype=Path.code_type)
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO
        path = Path(vertices, codes)
        # Each bar of Axes.errorbar is a 2-vertex path, which is never simplified.
        path.should_simplify = False
        self._bar_path = path
        self._paths = [path]
        self.stale = True

    def get_segments(self) -> List[np.ndarray]:
        """One ``(2, 2)`` array ``[[x, low], [x, high]]`` per bar."""
        if self._bar_path is None or self._paths != [self._bar_path]:
            # Set by set_segments / set_paths since: the usual one path per segment.
            return super().get_segments()
        return list(self._bar_path.vertices.reshape(-1, 2, 2))


def _update_datalim_like_vlines(ax: Any, bars: ErrorBarCollection) -> None:
    """Extend the axes data limits to cover *bars*, as ``Axes.vlines`` does for its collection."""
    if not bars.get_paths():
        return
    if ax.name == "rectilinear":
        datalim = bars.get_datalim(ax.transData)
        transform = bars.get_transform()
        # matplotlib < 3.11 spells this "seperately".
        contains = getattr(transform, "contains_branch_separately", None) \
            or transform.contains_branch_seperately
        update_x, update_y = contains(ax.transData)
        corners = (datalim.xmin, datalim.ymin), (datalim.xmax, datalim.ymax)
    else:
        vertices = bars.get_paths()[0].vertices
        update_x = update_y = True
        corners = (vertices[:, 0].min(), vertices[:, 1].min()), (vertices[:, 0].max(), vertices[:, 1].max())
    ax.update_datalim(corners, update_x, update_y)
    ax._request_autoscale_view()


def fast_errorbar(
    ax: Any,
    x: Any,
    y: Any,
    yerr: Any,
    *,
    fmt: str = "-",
    color: Any = None,
    linewidth: float = 1.0,
    elinewidth: float = 0.5,
    zorder: float = 2,
    label: Optional[str] = None,
) -> ErrorbarContainer:
    """Draw *y* with symmetric vertical error bars *yerr*, like ``ax.errorbar``.

    Gives the same artists and styles as ``ax.errorbar(x, y, yerr=yerr,
    fmt=fmt, color=color, linewidth=linewidth, elinewidth=elinewidth,
    capsize=0, zorder=zorder, label=label)``: a data line at *zorder* + 0.1
    and the bars at *zorder*, both unlabelled, in an ``ErrorbarContainer``
    that carries *label* and is added to *ax*.  The bars are one
    :class:`ErrorBarCollection`.  Unlike ``ax.errorbar``, negative errors
    are drawn (as the same bar) instead of raising ``ValueError``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    line_kwargs = {"linewidth": linewidth, "zorder": zorder + 0.1, "label": "_nolegend_"}
    if color is not None:
        line_kwargs["color"] = color
    (data_line,) = ax.plot(x, y, fmt, **line_kwargs)

    bars = ErrorBarCollection(
        x, y - yerr, y + yerr,
        colors=data_line.get_color(), linewidths=elinewidth,
        zorder=zorder, label="_nolegend_",
    )
    ax.add_collection(bars, autolim=False)
    _update_datalim_like_vlines(ax, bars)

    container = ErrorbarContainer((data_line, (), (bars,)), has_xerr=False, has_yerr=True, label=label)
    ax.add_container(container)
    return container
