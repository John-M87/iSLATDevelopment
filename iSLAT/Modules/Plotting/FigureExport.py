"""
FigureExport - write a matplotlib figure to a file the same way everywhere.

Every save path in the plotting package (the views' *Save Figure*, the
full-spectrum PDF export and :meth:`BasePlot.save`) goes through
:func:`save_figure_file`, so they share one rule for the resolution, the
file format and rasterizing.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
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
    previous = [(ax, ax.get_rasterized()) for ax in fig.axes] if rasterized else []
    try:
        for ax, _was in previous:
            ax.set_rasterized(True)
        fig.savefig(str(path), **kwargs)
    finally:
        for ax, was in previous:
            ax.set_rasterized(was)
    return str(path)
