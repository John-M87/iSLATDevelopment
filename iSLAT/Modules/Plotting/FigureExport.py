"""
FigureExport - write a matplotlib figure to a file the same way everywhere.

Every save path in the plotting package (the views' *Save Figure*, the
full-spectrum PDF export and :meth:`BasePlot.save`) goes through
:func:`save_figure_file`, so they share one rule for the resolution, the
file format and rasterizing.  :func:`save_figures_to_pdf` writes several
figures as the pages of one PDF with the same rules.
:func:`isolated_axes` shows only some axes of a figure for a save, so one
panel of a composite figure can be written on its own.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sequence, Set, Union

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Resolution used when the display settings cannot be read.
FALLBACK_EXPORT_DPI: int = 300


def default_export_dpi() -> int:
    """The DPI exports use unless one is given: the display settings' ``savefig_dpi`` (300 by default)."""
    try:
        from iSLAT.Modules.GUI.DisplayConfig import display_config
        return int(display_config.savefig_dpi)
    except Exception:
        return FALLBACK_EXPORT_DPI


def export_format(path: Union[str, Path], file_format: Optional[str] = None) -> Optional[str]:
    """The format to write *path* in: its extension when matplotlib can write it, else *file_format*.

    ``None`` (no extension matplotlib knows and no *file_format*) leaves the
    choice to matplotlib.
    """
    from matplotlib.backend_bases import FigureCanvasBase
    ext = Path(str(path)).suffix.lstrip(".").lower()
    if ext and ext in FigureCanvasBase.get_supported_filetypes():
        return ext
    return file_format


def save_figure_file(
    fig: "Figure",
    path: Union[str, Path],
    *,
    file_format: Optional[str] = None,
    dpi: Optional[int] = None,
    rasterized: bool = False,
    bbox_inches: Any = "tight",
    **savefig_kwargs: Any,
) -> str:
    """Save *fig* to *path* and return the path as a string.

    Parameters
    ----------
    fig : Figure
        The figure to write.
    path : str or Path
        Destination file.
    file_format : str, optional
        Format used when *path* has no extension matplotlib knows (see
        :func:`export_format`).  An explicit ``format=`` keyword wins over
        both.
    dpi : int, optional
        Resolution; :func:`default_export_dpi` when ``None`` or 0.
    rasterized : bool
        Rasterize every axes for this save only (smaller files for very
        dense data); each axes gets its own setting back afterwards.
    bbox_inches : str, optional
        Forwarded to :meth:`~matplotlib.figure.Figure.savefig` (default
        ``"tight"``).
    **savefig_kwargs
        Any other :meth:`~matplotlib.figure.Figure.savefig` keyword.
    """
    fmt = savefig_kwargs.pop("format", None) or export_format(path, file_format)
    kwargs = {"bbox_inches": bbox_inches, "dpi": dpi or default_export_dpi(), **savefig_kwargs}
    if fmt:
        kwargs["format"] = fmt
    with _rasterized_axes(fig, rasterized):
        fig.savefig(str(path), **kwargs)
    return str(path)


def save_figures_to_pdf(
    figures: Iterable["Figure"],
    path: Union[str, Path],
    *,
    dpi: Optional[int] = None,
    rasterized: bool = False,
    bbox_inches: Any = "tight",
    **savefig_kwargs: Any,
) -> int:
    """Write *figures* as the pages of one PDF at *path*; return the page count.

    *figures* may be a generator that builds each page when asked for it
    (and closes it when asked for the next), so only one page is in memory
    at a time.  The pages use the same resolution and rasterizing rules as
    :func:`save_figure_file`.  If a page fails, the partly written file is
    removed and the error raised.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    kwargs = {"bbox_inches": bbox_inches, "dpi": dpi or default_export_dpi(), **savefig_kwargs}
    iterator = iter(figures)
    pages = 0
    try:
        with PdfPages(str(path)) as pdf:
            for fig in iterator:
                with _rasterized_axes(fig, rasterized):
                    pdf.savefig(fig, **kwargs)
                pages += 1
    except BaseException:
        try:
            Path(str(path)).unlink()
        except OSError:
            pass
        raise
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return pages


@contextmanager
def isolated_axes(fig: "Figure", axes: Sequence["Axes"]) -> Iterator[None]:
    """Show only *axes* (one panel) of *fig* inside the block, for a ``bbox_inches="tight"`` save.

    Every other axes is hidden, except the twins of *axes* and colorbars
    attached to them.  Figure-level artists (legends, texts, the
    ``supxlabel`` / ``supylabel``) are hidden too, as they belong to the
    whole figure.  A shared ``supxlabel`` / ``supylabel`` becomes the
    panel's own x / y label while the panel has none, so the saved panel
    keeps its axis labels.  Everything is restored afterwards.
    """
    keep = _panel_axes(fig, axes)
    with ExitStack() as restore:
        for ax in fig.axes:
            if ax not in keep and ax.get_visible():
                ax.set_visible(False)
                restore.callback(ax.set_visible, True)
        _borrow_sup_label(fig, getattr(fig, "_supxlabel", None), keep, "x", restore)
        _borrow_sup_label(fig, getattr(fig, "_supylabel", None), keep, "y", restore)
        for artist in (*fig.legends, *fig.texts, *fig.artists, *fig.lines, *fig.patches, *fig.images):
            if artist.get_visible():
                artist.set_visible(False)
                restore.callback(artist.set_visible, True)
        yield


def _panel_axes(fig: "Figure", axes: Sequence["Axes"]) -> Set["Axes"]:
    """*axes* plus their twins and the colorbars attached to them."""
    keep: Set[Any] = set(axes)
    for ax in fig.axes:
        if any(ax in other._twinned_axes.get_siblings(other) for other in axes):
            keep.add(ax)
    for ax in fig.axes:
        info = getattr(ax, "_colorbar_info", None)
        if info and any(parent in keep for parent in info.get("parents", ())):
            keep.add(ax)
    return keep


def _borrow_sup_label(fig: "Figure", sup: Any, keep: Set["Axes"], which: str, restore: ExitStack) -> None:
    """Show the figure's *sup* label as the panel's own *which* (``"x"`` / ``"y"``) axis label.

    It goes on the panel's bottom-most (x) or left-most (y) axes whose label
    is empty; nothing happens when the label is not shown or every axis of
    the panel already has its own label.
    """
    if sup is None or not (sup.get_visible() and sup.get_text()) or sup.get_figure(root=False) is not fig:
        return
    candidates: List[Any] = [
        ax for ax in keep
        if ax.get_visible() and not hasattr(ax, "_colorbar_info") and getattr(ax, f"{which}axis").get_visible()
    ]
    if not candidates:
        return
    if which == "x":
        ax = min(candidates, key=lambda a: a.get_position().y0)
    else:
        ax = min(candidates, key=lambda a: a.get_position().x0)
    label = getattr(ax, f"{which}axis").label
    if label.get_text():
        return
    saved = (label.get_fontproperties().copy(), label.get_color())
    label.set_text(sup.get_text())
    label.set_fontproperties(sup.get_fontproperties())
    label.set_color(sup.get_color())

    def _put_back() -> None:
        label.set_text("")
        label.set_fontproperties(saved[0])
        label.set_color(saved[1])

    restore.callback(_put_back)


@contextmanager
def _rasterized_axes(fig: "Figure", rasterized: bool) -> Iterator[None]:
    """Rasterize every axes of *fig* inside the block (when *rasterized*), then restore each one's setting."""
    previous = [(ax, ax.get_rasterized()) for ax in fig.axes] if rasterized else []
    try:
        for ax, _was in previous:
            ax.set_rasterized(True)
        yield
    finally:
        for ax, was in previous:
            ax.set_rasterized(was)
