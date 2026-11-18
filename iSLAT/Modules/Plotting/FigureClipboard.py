"""
FigureClipboard - copy a matplotlib figure to the system clipboard as a PNG image.
Tk's own clipboard only holds text, so the image is handed to the operating system:

* **macOS** - ``osascript`` puts the PNG on the pasteboard (``«class PNGf»``).
* **Windows** - the Win32 clipboard API (through :mod:`ctypes`) gets both the
  registered ``"PNG"`` format (kept by Office, browsers, ...) and a ``CF_DIB``
  bitmap (for Paint and older programs).
* **Linux / BSD** - ``wl-copy`` on Wayland, else ``xclip``; one of them has
  to be installed.

:func:`copy_figure_to_clipboard` renders the figure the way *Save Figure*
does (:func:`~iSLAT.Modules.Plotting.FigureExport.save_figure_file`: tight
bounding box, the export resolution) and copies the result.  It raises
:class:`ClipboardError` with a message for the user when the copy fails.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .FigureExport import save_figure_file

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Seconds to wait for a clipboard helper program (osascript, wl-copy, xclip).
_HELPER_TIMEOUT: float = 15.0

class ClipboardError(RuntimeError):
    """The image could not be put on the clipboard; the message says why."""

def copy_figure_to_clipboard(
    fig: "Figure",
    *,
    dpi: Optional[int] = None,
    rasterized: bool = False,
    **savefig_kwargs: Any,
) -> None:
    """Render *fig* as a PNG (as *Save Figure* would) and put it on the clipboard.

    *dpi* defaults to the display settings' export resolution; *rasterized*
    and *savefig_kwargs* are as for
    :func:`~iSLAT.Modules.Plotting.FigureExport.save_figure_file`.  Raises
    :class:`ClipboardError` when the figure cannot be rendered or copied.
    """
    with tempfile.TemporaryDirectory(prefix="islat_clipboard_") as tmp:
        path = Path(tmp) / "figure.png"
        try:
            save_figure_file(fig, path, file_format="png", dpi=dpi, rasterized=rasterized,
                             **savefig_kwargs)
        except Exception as exc:
            raise ClipboardError(f"Could not render the figure: {exc}") from exc
        _copy_png_file(path)


# ----------------------------------------------------------------------
# Platform back ends
# ----------------------------------------------------------------------
def _copy_png_file(path: Path) -> None:
    """Copy the PNG file at *path* to the clipboard with the current platform's back end."""
    if sys.platform == "darwin":
        _copy_macos(path)
    elif sys.platform.startswith("win"):
        _copy_windows(path.read_bytes())
    else:
        _copy_unix(path.read_bytes())


def _run_helper(args: list, *, input_bytes: Optional[bytes] = None) -> None:
    """Run a clipboard helper program; raise :class:`ClipboardError` if it fails.

    stdout / stderr go to ``DEVNULL``: ``xclip`` and ``wl-copy`` keep a
    background process alive to serve the clipboard, and waiting for that
    process to close captured pipes would never finish.
    """
    try:
        result = subprocess.run(
            args, input=input_bytes,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=_HELPER_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClipboardError(f"Could not run {args[0]}: {exc}") from exc
    if result.returncode != 0:
        raise ClipboardError(f"{args[0]} could not copy the image (exit code {result.returncode}).")


def _copy_macos(path: Path) -> None:
    """Put the PNG file on the macOS pasteboard with ``osascript``."""
    posix = str(path).replace("\\", "\\\\").replace('"', '\\"')
    script = f'set the clipboard to (read (POSIX file "{posix}") as «class PNGf»)'
    _run_helper(["osascript", "-e", script])


def _copy_unix(png: bytes) -> None:
    """Put the PNG on the clipboard with ``wl-copy`` (Wayland) or ``xclip`` (X11)."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        _run_helper(["wl-copy", "--type", "image/png"], input_bytes=png)
        return
    if shutil.which("xclip"):
        _run_helper(["xclip", "-selection", "clipboard", "-t", "image/png", "-i"], input_bytes=png)
        return
    raise ClipboardError(
        "Copying images needs 'xclip' (X11) or 'wl-clipboard' (Wayland); install one and try again."
    )


def _png_to_dib(png: bytes) -> bytes:
    """The PNG *png* as a ``CF_DIB`` bitmap: a BMP file without its 14-byte file header."""
    import io
    from PIL import Image

    with Image.open(io.BytesIO(png)) as img:
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.split()[-1])
        else:
            rgb = img.convert("RGB")
        out = io.BytesIO()
        rgb.save(out, format="BMP")
    return out.getvalue()[14:]


def _copy_windows(png: bytes) -> None:  # pragma: no cover - needs Windows
    """Put the PNG on the Windows clipboard as ``"PNG"`` and as a ``CF_DIB`` bitmap."""
    import ctypes
    import time
    from ctypes import wintypes

    CF_DIB = 8
    GMEM_MOVEABLE = 0x0002

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    try:
        dib = _png_to_dib(png)
    except Exception as exc:
        raise ClipboardError(f"Could not convert the image for the clipboard: {exc}") from exc

    def _set(fmt: int, data: bytes) -> None:
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            raise ClipboardError("Out of memory while copying the image.")
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            raise ClipboardError("Could not lock clipboard memory.")
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(fmt, handle):
            kernel32.GlobalFree(handle)   # the clipboard owns the memory only on success
            raise ClipboardError("Windows refused the clipboard data.")

    # Another program may hold the clipboard for a moment: retry briefly.
    for _attempt in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        raise ClipboardError("The clipboard is in use by another program; try again.")
    try:
        user32.EmptyClipboard()
        png_format = user32.RegisterClipboardFormatW("PNG")
        if png_format:
            _set(png_format, png)
        _set(CF_DIB, dib)
    finally:
        user32.CloseClipboard()