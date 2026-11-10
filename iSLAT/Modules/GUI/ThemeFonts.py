"""
ThemeFonts - the GUI font sizes a theme sets.

A theme (a file in ``DATAFILES/CONFIG/GUIThemes``) can give font sizes in an
optional ``"fonts"`` block::

    "fonts": {
        "size": 14,
        "text_output_size": 12,
        "tooltip_size": 12
    }

* ``size`` - the GUI text: labels, buttons, entries, lists and menus.  It
  sets ``TkDefaultFont``, and every other Tk named font (text, fixed-width,
  headings, captions, ...) is scaled by the same factor, so their sizes
  keep their relation (on macOS the fixed-width and heading fonts stay a
  little smaller).
* ``text_output_size`` - the *iSLAT Text Output* pane (its own fixed-width
  font, :data:`TEXT_OUTPUT_FONT`).
* ``tooltip_size`` - tooltips.

Sizes are in points.  ``null``, a missing key or a missing block keeps the
default: the platform's own sizes for ``size``, the fixed-width font at the
GUI scale for ``text_output_size`` and :data:`DEFAULT_TOOLTIP_FONT_SIZE`
for ``tooltip_size``.  A value that is not a number from
:data:`MIN_THEME_FONT_SIZE` to :data:`MAX_THEME_FONT_SIZE` is ignored.

:func:`apply_theme_fonts` sets the sizes on the Tk named fonts, so every
widget that uses them follows, including widgets created later, and
applying another theme changes the text in place.  Widgets created with a
fixed font of their own (e.g. a few small labels) keep it.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import font as tkfont
from typing import Any, Dict, Optional, Tuple

#: Tk named fonts that the theme's ``size`` scales (Tk defines them on every platform).
SCALED_NAMED_FONTS: Tuple[str, ...] = (
    "TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont",
    "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont",
)
#: Named font of the *iSLAT Text Output* pane: the fixed-width font at ``text_output_size``.
TEXT_OUTPUT_FONT: str = "iSLATTextOutputFont"
#: Tooltip font family, and its size (points) when the theme gives none.
TOOLTIP_FONT_FAMILY: str = "tahoma"
DEFAULT_TOOLTIP_FONT_SIZE: int = 12
#: Accepted theme font sizes (points); other values are ignored.
MIN_THEME_FONT_SIZE: float = 4
MAX_THEME_FONT_SIZE: float = 48

# The platform's sizes of SCALED_NAMED_FONTS, read before a theme first changes them.
_DEFAULT_SIZES_ATTR = "_islat_default_font_sizes"


def theme_font_size(theme: Any, key: str) -> Optional[int]:
    """The theme's ``fonts[key]`` size in whole points, or ``None`` when it is unset or invalid."""
    fonts = theme.get("fonts") if isinstance(theme, dict) else None
    value = fonts.get(key) if isinstance(fonts, dict) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(size) or not MIN_THEME_FONT_SIZE <= size <= MAX_THEME_FONT_SIZE:
        return None
    # Tk 8.6 font sizes are whole numbers.
    return int(round(size))


def tooltip_font(theme: Any) -> Tuple[str, str, str]:
    """The Tk font description for tooltips under *theme* (``tooltip_size``, else the default size)."""
    size = theme_font_size(theme, "tooltip_size") or DEFAULT_TOOLTIP_FONT_SIZE
    return (TOOLTIP_FONT_FAMILY, str(size), "normal")


def text_output_font(widget: tk.Misc) -> str:
    """Name of the *iSLAT Text Output* font, created as a copy of ``TkFixedFont`` the first time."""
    root = widget._root()
    if TEXT_OUTPUT_FONT not in tkfont.names(root):
        # Created through Tcl so that no Python Font object deletes it when garbage-collected.
        options = tkfont.nametofont("TkFixedFont", root=root).configure()
        args = []
        for name, value in options.items():
            args += [f"-{name}", value]
        root.tk.call("font", "create", TEXT_OUTPUT_FONT, *args)
    return TEXT_OUTPUT_FONT


def apply_theme_fonts(widget: tk.Misc, theme: Any) -> None:
    """Set the GUI fonts of *widget*'s Tk application to the sizes in *theme* (see the module docstring).

    A theme without sizes puts the platform's defaults back.
    """
    try:
        root = widget._root()
        defaults = _default_sizes(root)
        factor = 1.0
        size = theme_font_size(theme, "size")
        base = defaults.get("TkDefaultFont")
        if size is not None and base:
            factor = size / _points(base, root)
        for name, default in defaults.items():
            tkfont.nametofont(name, root=root).configure(size=_scaled(default, factor))

        output_size = theme_font_size(theme, "text_output_size")
        if output_size is None:
            output_size = tkfont.nametofont("TkFixedFont", root=root).cget("size")
        tkfont.nametofont(text_output_font(root), root=root).configure(size=output_size)
    except tk.TclError:
        pass  # fonts are cosmetic; never block the GUI


def _default_sizes(root: tk.Misc) -> Dict[str, int]:
    """The platform's sizes of :data:`SCALED_NAMED_FONTS`, read once per Tk application."""
    sizes = getattr(root, _DEFAULT_SIZES_ATTR, None)
    if sizes is None:
        sizes = {}
        for name in SCALED_NAMED_FONTS:
            try:
                sizes[name] = int(tkfont.nametofont(name, root=root).cget("size"))
            except (tk.TclError, ValueError):
                pass
        setattr(root, _DEFAULT_SIZES_ATTR, sizes)
    return sizes


def _points(size: int, root: tk.Misc) -> float:
    """A Tk font size in points (negative sizes are pixels)."""
    if size > 0:
        return float(size)
    try:
        pixels_per_point = float(root.tk.call("tk", "scaling"))
    except (tk.TclError, ValueError):
        pixels_per_point = 96.0 / 72.0
    return -size / pixels_per_point


def _scaled(size: int, factor: float) -> int:
    """*size* times *factor*, rounded, keeping its sign (points / pixels) and never 0 (Tk's "default")."""
    if factor == 1.0 or size == 0:
        return size
    scaled = int(round(size * factor))
    if scaled == 0:
        scaled = 1 if size > 0 else -1
    return scaled
