# -*- coding: utf-8 -*-
"""The views' Save Figure dialog offers PDF, PNG and SVG; the selected type decides the format."""

import matplotlib
matplotlib.use("Agg")

import sys
from types import SimpleNamespace

import pytest

from iSLAT.Modules.Plotting.PlotView import ask_save_path, with_format_extension

# The package re-exports the class under the module's name: patch the module itself.
plot_view_module = sys.modules["iSLAT.Modules.Plotting.PlotView"]

from tests.plotting.test_panel_save import _PanelView, _event, _two_panel_figure
from tests.plotting.test_view_plumbing import _LiveFigureView, _figure

FORMATS = ("pdf", "png", "svg")
REAL_TYPE_VARIABLE = plot_view_module._selected_type_variable


class _TypeVariable:
    """Stands in for the Tk variable the dialog writes the selected file type to."""

    def __init__(self, label):
        self.label = label

    def get(self):
        return self.label


@pytest.fixture
def dialog(monkeypatch):
    """Fake save dialog: records its options, returns ``state.answer`` and selects ``state.selected``."""
    import tkinter.filedialog
    state = SimpleNamespace(asked=[], answer="", selected=None)

    def ask(**options):
        state.asked.append(options)
        if state.selected is not None and "typevariable" in options:
            options["typevariable"].label = state.selected
        return state.answer

    monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", ask)
    monkeypatch.setattr(plot_view_module, "_selected_type_variable", _TypeVariable)
    return state


class TestDialog:
    def test_offers_the_requested_format_first(self, dialog):
        ask_save_path("T", "png", initialfile="spec_x.png", formats=FORMATS)
        options = dialog.asked[0]
        assert options["filetypes"] == [("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        assert options["typevariable"].get() == "PNG files"
        assert options["initialfile"] == "spec_x"
        assert "defaultextension" not in options

    def test_one_format_keeps_the_default_extension(self, dialog):
        ask_save_path("T", "pdf", initialfile="spec_x.pdf")
        options = dialog.asked[0]
        assert options["filetypes"] == [("PDF files", "*.pdf")]
        assert options["defaultextension"] == ".pdf" and options["initialfile"] == "spec_x.pdf"
        assert "typevariable" not in options

    @pytest.mark.parametrize("selected, answer, expected", [
        ("SVG files", "C:/out/plot", "C:/out/plot.svg"),
        ("PNG files", "C:/out/plot", "C:/out/plot.png"),
        ("PNG files", "C:/out/plot.svg", "C:/out/plot.svg"),    # a typed extension wins
        ("PDF files", "C:/out/plot.v2", "C:/out/plot.v2.pdf"),  # not a format: the type's is added
        ("All files", "C:/out/plot", "C:/out/plot.pdf"),        # "All files": the requested format
    ])
    def test_the_selected_type_gives_the_extension(self, dialog, selected, answer, expected):
        dialog.selected, dialog.answer = selected, answer
        assert ask_save_path("T", "pdf", formats=FORMATS, any_file_type=True) == expected

    def test_cancelled(self, dialog):
        dialog.selected = "SVG files"
        assert ask_save_path("T", "pdf", formats=FORMATS) == ""

    def test_without_a_tk_root_the_requested_format_is_used(self, dialog, monkeypatch):
        import tkinter
        monkeypatch.setattr(plot_view_module, "_selected_type_variable", REAL_TYPE_VARIABLE)
        monkeypatch.setattr(tkinter, "_default_root", None)
        dialog.answer = "C:/out/plot"
        assert ask_save_path("T", "png", formats=FORMATS) == "C:/out/plot.png"
        assert "typevariable" not in dialog.asked[0]

    def test_with_format_extension(self):
        assert with_format_extension("a/b.PNG", "pdf") == "a/b.PNG"
        assert with_format_extension("a/b", "svg") == "a/b.svg"



class TestViews:
    def test_save_figure_writes_the_selected_type(self, dialog, tmp_path):
        dialog.selected, dialog.answer = "SVG files", str(tmp_path / "view")
        saved = _LiveFigureView(_figure()).save_figure(dpi=20)
        assert saved == str(tmp_path / "view.svg")
        assert b"<svg" in (tmp_path / "view.svg").read_bytes()[:400]

    def test_save_panel_writes_the_selected_type(self, dialog, tmp_path):
        fig, left, _right = _two_panel_figure()
        view = _PanelView(fig)
        dialog.selected, dialog.answer = "PNG files", str(tmp_path / "left")
        saved = view.save_panel(view.panel_at(_event(left)), dpi=20)
        assert saved == str(tmp_path / "left.png")
        assert (tmp_path / "left.png").read_bytes()[:4] == b"\x89PNG"

    def test_save_figure_offers_pdf_png_and_svg(self, dialog):
        _LiveFigureView(_figure()).save_figure()
        assert [pattern for _label, pattern in dialog.asked[0]["filetypes"]] == ["*.pdf", "*.png", "*.svg"]
