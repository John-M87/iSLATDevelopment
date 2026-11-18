# -*- coding: utf-8 -*-
"""Copying a figure to the clipboard as a PNG (right-click > Copy Figure to Clipboard).

The platform helpers (``osascript``, ``wl-copy`` / ``xclip``) are replaced by
a fake ``subprocess.run`` so the tests never touch the real clipboard.
"""

import matplotlib
matplotlib.use("Agg")

import importlib
import io
import struct
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from iSLAT.Modules.Plotting import FigureClipboard as fc
from iSLAT.Modules.Plotting.FigureClipboard import ClipboardError, copy_figure_to_clipboard
from iSLAT.Modules.Plotting.PlotView import FigurePanel

from tests.plotting.test_view_plumbing import _LiveFigureView

# The package exports the PlotView class under the module's name.
plot_view_module = importlib.import_module("iSLAT.Modules.Plotting.PlotView")

PNG_MAGIC = b"\x89PNG"


def _figure():
    fig = Figure(figsize=(4, 3))
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    left.plot([0, 1], [0, 1])
    right.plot([0, 1], [1, 0])
    return fig


@pytest.fixture
def fake_run(monkeypatch):
    """Record each helper call with the PNG it was given (read while the temp file still exists)."""
    calls = []

    def _run(args, input=None, **kwargs):
        png = input
        if args[0] == "osascript":
            path = args[2].split('POSIX file "', 1)[1].split('"', 1)[0]
            with open(path, "rb") as fh:
                png = fh.read()
        calls.append(SimpleNamespace(args=args, png=png, kwargs=kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fc.subprocess, "run", _run)
    return calls


# ======================================================================
# Platform back ends
# ======================================================================
class TestBackEnds:
    def test_macos_puts_the_png_on_the_pasteboard(self, monkeypatch, fake_run):
        monkeypatch.setattr(sys, "platform", "darwin")
        copy_figure_to_clipboard(_figure(), dpi=20)
        (call,) = fake_run
        assert call.args[:2] == ["osascript", "-e"]
        assert "«class PNGf»" in call.args[2]
        assert call.png.startswith(PNG_MAGIC)

    def test_wayland_uses_wl_copy(self, monkeypatch, fake_run):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(fc.shutil, "which", lambda name: f"/usr/bin/{name}")
        copy_figure_to_clipboard(_figure(), dpi=20)
        (call,) = fake_run
        assert call.args == ["wl-copy", "--type", "image/png"]
        assert call.png.startswith(PNG_MAGIC)

    def test_x11_uses_xclip_without_capturing_its_output(self, monkeypatch, fake_run):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setattr(fc.shutil, "which", lambda name: "/usr/bin/xclip" if name == "xclip" else None)
        copy_figure_to_clipboard(_figure(), dpi=20)
        (call,) = fake_run
        assert call.args == ["xclip", "-selection", "clipboard", "-t", "image/png", "-i"]
        assert call.png.startswith(PNG_MAGIC)
        # xclip keeps serving the clipboard in the background: piped output would never close.
        assert call.kwargs["stdout"] is fc.subprocess.DEVNULL
        assert call.kwargs["stderr"] is fc.subprocess.DEVNULL

    def test_no_helper_installed(self, monkeypatch, fake_run):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(fc.shutil, "which", lambda name: None)
        with pytest.raises(ClipboardError, match="xclip"):
            copy_figure_to_clipboard(_figure(), dpi=20)
        assert fake_run == []

    def test_helper_failure_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(fc.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
        with pytest.raises(ClipboardError, match="exit code 1"):
            copy_figure_to_clipboard(_figure(), dpi=20)

    def test_missing_helper_program_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        def _missing(*a, **k):
            raise FileNotFoundError("osascript")

        monkeypatch.setattr(fc.subprocess, "run", _missing)
        with pytest.raises(ClipboardError, match="Could not run osascript"):
            copy_figure_to_clipboard(_figure(), dpi=20)

    def test_windows_bitmap_is_a_headerless_bmp_on_white(self):
        img = Image.new("RGBA", (3, 2), (255, 0, 0, 0))   # fully transparent red
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        dib = fc._png_to_dib(buf.getvalue())
        header_size, width, height = struct.unpack("<Iii", dib[:12])
        assert (header_size, width, height) == (40, 3, 2)
        bmp = Image.open(io.BytesIO(b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + 40) + dib))
        assert bmp.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


# ======================================================================
# PlotView: Copy Figure / Copy <panel>
# ======================================================================
class _PanelView(_LiveFigureView):
    def panel_at(self, event):
        return FigurePanel(label="Left Panel", file_suffix="left", axes=(event.inaxes,))


class TestPlotViewCopy:
    @pytest.fixture
    def copied(self, monkeypatch):
        figures = []
        monkeypatch.setattr(plot_view_module, "copy_figure_to_clipboard",
                            lambda fig, dpi=None: figures.append((fig, [ax.get_visible() for ax in fig.axes])))
        return figures

    def test_copies_the_export_figure_and_reports_it(self, copied):
        data_field = MagicMock()
        fig = _figure()
        view = _LiveFigureView(fig, islat=SimpleNamespace(GUI=SimpleNamespace(data_field=data_field)))
        assert view.copy_figure_to_clipboard() is True
        assert copied == [(fig, [True, True])]
        data_field.insert_text.assert_called_once_with(view.COPY_MESSAGE)

    def test_copies_just_the_panel_then_restores_the_figure(self, copied):
        fig = _figure()
        fig.canvas.draw_idle = MagicMock()
        left, right = fig.axes
        view = _PanelView(fig)
        assert view.copy_panel_to_clipboard(view.panel_at(SimpleNamespace(inaxes=left))) is True
        assert copied == [(fig, [True, False])]            # only the panel while copying
        assert left.get_visible() and right.get_visible()   # everything back afterwards
        fig.canvas.draw_idle.assert_called()

    def test_nothing_to_copy(self, copied):
        view = _LiveFigureView(None)
        assert view.copy_figure_to_clipboard() is False
        assert copied == []

    def test_failure_is_reported_and_the_figure_released(self, monkeypatch):
        def _fail(fig, dpi=None):
            raise ClipboardError("no clipboard here")

        monkeypatch.setattr(plot_view_module, "copy_figure_to_clipboard", _fail)
        data_field = MagicMock()
        fig = _figure()
        fig.canvas.draw_idle = MagicMock()
        left, right = fig.axes
        view = _PanelView(fig, islat=SimpleNamespace(GUI=SimpleNamespace(data_field=data_field)))
        assert view.copy_panel_to_clipboard(view.panel_at(SimpleNamespace(inaxes=left))) is False
        assert right.get_visible()
        (message,), _kw = data_field.insert_text.call_args
        assert "no clipboard here" in message
