# -*- coding: utf-8 -*-
"""ThemeFonts: the GUI font sizes of a theme's ``"fonts"`` block.

``size`` sets the GUI text (``TkDefaultFont``) and scales the other Tk named
fonts by the same factor, ``text_output_size`` sets the iSLAT Text Output
pane and ``tooltip_size`` the tooltips; unset sizes keep the defaults, and
applying a theme without sizes puts the platform's sizes back.
"""

import json
import math

import pytest

from iSLAT.Modules.FileHandling import theme_file_path
from iSLAT.Modules.GUI.ThemeFonts import (
    DEFAULT_TOOLTIP_FONT_SIZE,
    MAX_THEME_FONT_SIZE,
    MIN_THEME_FONT_SIZE,
    SCALED_NAMED_FONTS,
    TEXT_OUTPUT_FONT,
    TOOLTIP_FONT_FAMILY,
    apply_theme_fonts,
    text_output_font,
    theme_font_size,
    tooltip_font,
)


def _theme(**fonts):
    return {"background": "#FFFFFF", "fonts": fonts}


@pytest.fixture(scope="module")
def tk_root():
    # One root for the module: creating many Tk() instances is flaky on Windows.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def root(tk_root):
    """The module's root with the platform's font sizes put back after each test."""
    yield tk_root
    apply_theme_fonts(tk_root, {})


def _size(root, name):
    from tkinter import font as tkfont
    return tkfont.nametofont(name, root=root).cget("size")


def _platform_sizes(root):
    apply_theme_fonts(root, {})
    return {name: _size(root, name) for name in SCALED_NAMED_FONTS}


class TestThemeFontSize:
    @pytest.mark.parametrize("theme, expected", [
        (_theme(size=14), 14),
        (_theme(size="13"), 13),
        (_theme(size=12.6), 13),                      # Tk 8.6 sizes are whole points
        (_theme(size=MIN_THEME_FONT_SIZE), MIN_THEME_FONT_SIZE),
        (_theme(size=MAX_THEME_FONT_SIZE), MAX_THEME_FONT_SIZE),
    ])
    def test_valid_sizes(self, theme, expected):
        assert theme_font_size(theme, "size") == expected

    @pytest.mark.parametrize("theme", [
        _theme(size=None), _theme(), {"background": "#FFFFFF"}, {"fonts": "big"}, None,
        _theme(size="big"), _theme(size=True), _theme(size=math.nan),
        _theme(size=MIN_THEME_FONT_SIZE - 1), _theme(size=MAX_THEME_FONT_SIZE + 1),
    ])
    def test_unset_or_invalid_sizes_are_none(self, theme):
        assert theme_font_size(theme, "size") is None

    def test_tooltip_font(self):
        assert tooltip_font(_theme(tooltip_size=15)) == (TOOLTIP_FONT_FAMILY, "15", "normal")
        assert tooltip_font({}) == (TOOLTIP_FONT_FAMILY, str(DEFAULT_TOOLTIP_FONT_SIZE), "normal")


class TestApplyThemeFonts:
    def test_size_sets_the_default_font_and_scales_the_others(self, root):
        platform = _platform_sizes(root)
        base = platform["TkDefaultFont"]
        target = abs(base) * 2 if base > 0 else 20
        apply_theme_fonts(root, _theme(size=target))
        if base > 0:                        # points (negative sizes are pixels, e.g. on X11)
            assert _size(root, "TkDefaultFont") == target
        # The same factor for every font, so the sizes keep their relation (and their sign).
        factor = _size(root, "TkDefaultFont") / base
        assert factor > 1
        for name, default in platform.items():
            assert abs(_size(root, name) - default * factor) <= 0.5

    def test_applying_again_does_not_compound(self, root):
        apply_theme_fonts(root, _theme(size=18))
        once = {name: _size(root, name) for name in SCALED_NAMED_FONTS}
        apply_theme_fonts(root, _theme(size=18))
        assert {name: _size(root, name) for name in SCALED_NAMED_FONTS} == once

    def test_theme_without_sizes_restores_the_platform_sizes(self, root):
        platform = _platform_sizes(root)
        apply_theme_fonts(root, _theme(size=20))
        apply_theme_fonts(root, {"background": "#000000"})
        assert {name: _size(root, name) for name in SCALED_NAMED_FONTS} == platform

    def test_text_output_follows_the_fixed_font_unless_set(self, root):
        apply_theme_fonts(root, _theme(size=18))
        assert text_output_font(root) == TEXT_OUTPUT_FONT
        assert _size(root, TEXT_OUTPUT_FONT) == _size(root, "TkFixedFont")
        fixed = _size(root, "TkFixedFont")
        apply_theme_fonts(root, _theme(size=18, text_output_size=9))
        assert _size(root, TEXT_OUTPUT_FONT) == 9
        assert _size(root, "TkFixedFont") == fixed

    def test_text_output_pane_uses_the_text_output_font(self, root):
        import tkinter as tk
        from iSLAT.Modules.GUI.Widgets.DataField import DataField
        host = tk.Frame(root)
        try:
            field = DataField("", host)
            assert str(field.text.cget("font")) == TEXT_OUTPUT_FONT
            apply_theme_fonts(root, _theme(text_output_size=17))
            assert _size(root, str(field.text.cget("font"))) == 17
        finally:
            host.destroy()


class TestBundledThemes:
    @pytest.mark.parametrize("name", ["DarkTheme", "LightTheme", "PastelBlue"])
    def test_fonts_block(self, name):
        with open(theme_file_path / f"{name}.json") as fh:
            fonts = json.load(fh)["fonts"]
        assert set(fonts) == {"size", "text_output_size", "tooltip_size", "description"}
        for key in ("size", "text_output_size", "tooltip_size"):
            value = fonts[key]
            assert value is None or theme_font_size({"fonts": fonts}, key) == value
