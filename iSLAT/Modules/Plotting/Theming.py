"""
Theming - apply an iSLAT theme dictionary to a figure and its Tk canvas.

One implementation shared by the plot classes (:meth:`BasePlot.apply_theme_to_figure`),
the main controller and the views that own their own figure, so every view
uses the same theme keys:

* ``background``         - figure background and canvas widget
* ``graph_fill_color``   - axes background (falls back to ``background``)
* ``foreground``         - ticks, labels, titles, spines and the observed spectrum
* ``summed_spectra_color`` - the summed-model fill
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from .ArtistTags import OBSERVED_TAG, SUMMED_TAG, iter_tagged

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .LegendStrategy import LegendStrategy

#: Canvas widget background used when the theme has no ``background``.
DEFAULT_CANVAS_BACKGROUND = "#181A1B"


def apply_figure_theme(
    fig: Optional["Figure"],
    theme: Dict[str, Any],
    legend_strategy: Optional["LegendStrategy"] = None,
) -> None:
    """Color *fig* and all its axes from *theme*.

    Sets the figure and axes backgrounds, the tick / label / title / spine
    colors, and re-colors the tagged observed-spectrum lines and summed-model
    fills so they follow a theme change.  When *legend_strategy* is given its
    :meth:`~LegendStrategy.apply_theme` styles each axes' legend; otherwise
    legends are left alone.
    """
    if fig is None:
        return
    fg = theme.get("foreground", "black")
    bg = theme.get("background", "white")
    graph_bg = theme.get("graph_fill_color", bg)
    summed_color = theme.get("summed_spectra_color", "lightgray")

    fig.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(graph_bg)
        ax.tick_params(colors=fg, which="both")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

        for line in iter_tagged(ax, OBSERVED_TAG, collections=False):
            line.set_color(fg)
        for fill in iter_tagged(ax, SUMMED_TAG, lines=False):
            fill.set_facecolor(summed_color)
            fill.set_edgecolor(summed_color)

        if legend_strategy is not None:
            legend_strategy.apply_theme(ax, theme)


def apply_canvas_theme(canvas: Any, theme: Dict[str, Any]) -> None:
    """Set the Tk widget behind *canvas* to the theme background (ignored without a widget)."""
    if canvas is None:
        return
    try:
        canvas.get_tk_widget().configure(bg=theme.get("background", DEFAULT_CANVAS_BACKGROUND))
    except Exception:
        pass
