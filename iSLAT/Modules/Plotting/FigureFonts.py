"""
FigureFonts - set the text sizes of a rendered figure by kind of text.

The full-spectrum exports use it for the font sizes of *Full Spectrum Export
Settings*.  :func:`apply_figure_font_sizes` resizes the text of a figure
after a plot class drew it, so every kind of full-spectrum plot (plain,
residual sub-panels, sample stack) and any plot a subclass swaps in follow
the same settings without the plot classes knowing about them.  The kinds
of text are (see :data:`FONT_SIZE_KINDS`):

* ``"ticks"``  - tick labels (major and minor) and the axis offset texts
* ``"labels"`` - axis labels and the figure's ``supxlabel`` / ``supylabel``
* ``"legend"`` - legend entries and legend titles
* ``"text"``   - every other text: titles, saved / atomic line labels,
  wavelength-mark labels, chi^2 values, sample names and other figure texts

Sizes are in points; ``None`` leaves that kind as the plot drew it.  The
stacked layout measures the text when the figure is drawn, so the margins
and the gaps between panels fit the new sizes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import numpy as np

from .LegendStrategy import MoleculeColorLegend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from .LegendStrategy import LegendStrategy

#: The kinds of text, with the names the export settings dialog and errors use.
FONT_SIZE_KINDS: Dict[str, str] = {
    "ticks": "Tick label",
    "labels": "Axis label",
    "legend": "Legend",
    "text": "Annotation",
}

#: Largest accepted font size (points).
MAX_FONT_SIZE: float = 72.0


def default_font_sizes() -> Dict[str, Optional[float]]:
    """Every kind ``None``: the text keeps the sizes the plot draws it with."""
    return {kind: None for kind in FONT_SIZE_KINDS}


def validate_font_size(name: str, value: Any) -> Optional[float]:
    """*value* as a size in ``(0, MAX_FONT_SIZE]`` points, or ``None`` for ``None`` / a blank string.

    *name* names the size in the error message.

    Raises
    ------
    ValueError
        If *value* is not a number in that range.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} font size must be a number.")
    try:
        size = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} font size must be a number.") from None
    if not np.isfinite(size) or size <= 0:
        raise ValueError(f"{name} font size must be greater than 0.")
    if size > MAX_FONT_SIZE:
        raise ValueError(f"{name} font size must be at most {MAX_FONT_SIZE:g} pt.")
    return size


def validate_font_sizes(sizes: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """A size (or ``None``) for every kind of :data:`FONT_SIZE_KINDS`; kinds left out of *sizes* are ``None``.

    Raises
    ------
    ValueError
        For an invalid size (see :func:`validate_font_size`) or a kind not
        in :data:`FONT_SIZE_KINDS`.
    """
    result = default_font_sizes()
    for kind, value in (sizes or {}).items():
        if kind not in FONT_SIZE_KINDS:
            raise ValueError(
                f"Unknown kind of text {kind!r} (expected one of: {', '.join(FONT_SIZE_KINDS)})."
            )
        result[kind] = validate_font_size(FONT_SIZE_KINDS[kind], value)
    return result


def apply_figure_font_sizes(
    fig: Optional["Figure"],
    sizes: Optional[Mapping[str, Optional[float]]],
    legend_strategy: Optional["LegendStrategy"] = None,
) -> None:
    """Set the size of each kind of text in *fig* that *sizes* gives (points; ``None`` = leave it).

    Call it once the figure is rendered (after ``generate_plot()`` and any
    overlays).  A molecule color legend (:class:`MoleculeColorLegend`, the
    stacked plots' default) chooses its column count for its font size, so
    with *legend_strategy* being one it is built again at the new size,
    keeping its entries, colors and visibility; other legends keep their
    columns and only get the new text size.
    """
    if fig is None or not sizes:
        return
    ticks, labels, legend, text = (sizes.get(kind) for kind in FONT_SIZE_KINDS)
    if ticks is None and labels is None and legend is None and text is None:
        return

    for ax in fig.axes:
        if ticks is not None:
            ax.tick_params(axis="both", which="both", labelsize=ticks)
            ax.xaxis.offsetText.set_fontsize(ticks)
            ax.yaxis.offsetText.set_fontsize(ticks)
        if labels is not None:
            ax.xaxis.label.set_fontsize(labels)
            ax.yaxis.label.set_fontsize(labels)
        if text is not None:
            for artist in (ax.title, ax._left_title, ax._right_title, *ax.texts):
                artist.set_fontsize(text)
        if legend is not None:
            _resize_axes_legend(fig, ax, legend, legend_strategy)

    # Figure-level text; figure.texts also lists the sup labels and the suptitle.
    sup_labels = [t for t in (fig._supxlabel, fig._supylabel) if t is not None]
    if labels is not None:
        for artist in sup_labels:
            artist.set_fontsize(labels)
    if text is not None:
        skip = {id(t) for t in sup_labels}
        for artist in fig.texts:
            if id(artist) not in skip:
                artist.set_fontsize(text)
    if legend is not None:
        for leg in fig.legends:
            _set_legend_text_size(leg, legend)


def _resize_axes_legend(
    fig: "Figure", ax: "Axes", size: float, legend_strategy: Optional["LegendStrategy"],
) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    texts = list(leg.get_texts())
    if (
        isinstance(legend_strategy, MoleculeColorLegend)
        and texts and all(getattr(t, "_islat_mol_color", False) for t in texts)
    ):
        visible = leg.get_visible()
        legend_strategy.build_legend(
            ax, fig, [t.get_text() for t in texts], [t.get_color() for t in texts], fontsize=size,
        )
        legend_strategy.update_visibility(ax, visible)
        return
    _set_legend_text_size(leg, size)


def _set_legend_text_size(leg: "Legend", size: float) -> None:
    for artist in leg.get_texts():
        artist.set_fontsize(size)
    leg.get_title().set_fontsize(size)
