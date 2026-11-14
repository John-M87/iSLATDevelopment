# -*- coding: utf-8 -*-
"""The Windows window icon is converted from the logo PNG once and kept.

windows_icon_file writes the ``.ico`` (at ICON_SIZES) to the temp folder
under a name made from the PNG's size and modification time, and later calls
(later launches) reuse it without PIL.  A changed PNG gets a new file.  The
file is written under a temporary name and renamed into place; when the
rename fails because another launch has the finished file open, that file is
used.

set_windows_frame_icon puts the icon on the window's frame (WM_SETICON) and on
the Tk toplevel window class, waiting for the root's first <Map> while the
frame does not exist yet; setting it again releases the old icon handles.
After startup (async display complete, then the next plot draw) the GUI sends
the icon again: the taskbar shows Tk's feather when its icon query at window
creation goes unanswered, and a new WM_SETICON makes it ask again.
"""
import os

import pytest

PIL_Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def temp_folder(tmp_path, monkeypatch):
    import tempfile
    folder = tmp_path / "temp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder


@pytest.fixture
def logo(tmp_path):
    path = tmp_path / "logo.png"
    PIL_Image.new("RGBA", (128, 128), (30, 144, 255, 255)).save(path)
    return path


@pytest.fixture
def png_opens(monkeypatch):
    """Count the PNG conversions (PIL.Image.open calls)."""
    opened = []
    original = PIL_Image.open

    def counting(fp, *args, **kwargs):
        opened.append(fp)
        return original(fp, *args, **kwargs)

    monkeypatch.setattr(PIL_Image, "open", counting)
    return opened


def _touch_later(path, seconds=10):
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + int(seconds * 1e9)))


def test_writes_an_icon_with_all_sizes(temp_folder, logo):
    from iSLAT.Modules.GUI.GUI import ICON_SIZES, windows_icon_file
    ico_path = windows_icon_file(logo)
    assert os.path.dirname(ico_path) == str(temp_folder)
    with PIL_Image.open(ico_path) as icon:
        assert icon.format == "ICO"
        assert set(icon.info["sizes"]) == set(ICON_SIZES)
    assert os.listdir(temp_folder) == [os.path.basename(ico_path)]  # no partial file left


def test_later_calls_reuse_the_file_without_converting(temp_folder, logo, png_opens):
    from iSLAT.Modules.GUI.GUI import windows_icon_file
    first = windows_icon_file(logo)
    written = os.stat(first).st_mtime_ns
    assert windows_icon_file(logo) == first
    assert windows_icon_file(logo) == first
    assert len(png_opens) == 1
    assert os.stat(first).st_mtime_ns == written


def test_a_changed_logo_gets_a_new_icon(temp_folder, logo, png_opens):
    from iSLAT.Modules.GUI.GUI import windows_icon_file
    first = windows_icon_file(logo)
    PIL_Image.new("RGBA", (128, 128), (255, 69, 0, 255)).save(logo)
    _touch_later(logo)
    second = windows_icon_file(logo)
    assert second != first and os.path.isfile(second)
    assert len(png_opens) == 2


def test_an_empty_icon_file_is_written_again(temp_folder, logo, png_opens):
    from iSLAT.Modules.GUI.GUI import windows_icon_file
    ico_path = windows_icon_file(logo)
    open(ico_path, "wb").close()
    assert windows_icon_file(logo) == ico_path
    assert os.path.getsize(ico_path) > 0 and len(png_opens) == 2


def test_uses_the_file_another_launch_finished(temp_folder, logo, monkeypatch):
    """The rename fails (Windows: the finished file is open elsewhere); that file is used."""
    from iSLAT.Modules.GUI.GUI import windows_icon_file
    ico_path = windows_icon_file(logo)
    finished = open(ico_path, "rb").read()
    os.remove(ico_path)
    replaced = []

    def other_launch_has_it_open(src, dst):
        replaced.append(dst)
        with open(dst, "wb") as fh:  # the other launch's finished file
            fh.write(finished)
        raise PermissionError(13, "file is in use", dst)

    monkeypatch.setattr(os, "replace", other_launch_has_it_open)
    assert windows_icon_file(logo) == ico_path
    assert replaced == [ico_path]
    assert os.listdir(temp_folder) == [os.path.basename(ico_path)]


def test_a_failed_rename_without_a_file_raises(temp_folder, logo, monkeypatch):
    from iSLAT.Modules.GUI.GUI import windows_icon_file

    def refuse(src, dst):
        raise PermissionError(13, "denied", dst)

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        windows_icon_file(logo)
    assert os.listdir(temp_folder) == []


# ----------------------------------------------------------------------
# The icon on the frame and the window class, and the refresh after startup
# ----------------------------------------------------------------------
class _FakeUser32:
    """Records the Win32 calls set_windows_frame_icon makes; icons are numbered handles."""

    def __init__(self):
        self.next_handle = 100
        self.sent, self.class_icons, self.destroyed = [], [], []

    def LoadImageW(self, instance, path, kind, width, height, flags):
        self.next_handle += 1
        return self.next_handle

    def SendMessageW(self, hwnd, message, which, hicon):
        self.sent.append((hwnd, message, which, hicon))

    def set_class_icon(self, hwnd, index, hicon):
        self.class_icons.append((hwnd, index, hicon))

    def DestroyIcon(self, hicon):
        self.destroyed.append(hicon)


class _FakeWindow:
    """A Tk root stand-in: its frame exists once *mapped* is set."""

    def __init__(self, mapped):
        self.mapped = mapped
        self.map_handlers = []

    def winfo_id(self):
        return 0x10

    def wm_frame(self):
        return hex(0x20 if self.mapped else 0x10)

    def bind(self, sequence, handler, add=None):
        assert sequence == "<Map>"
        self.map_handlers.append(handler)


@pytest.fixture
def user32(monkeypatch):
    import sys
    import iSLAT.Modules.GUI.GUI  # noqa: F401  (the package attribute is the class, not the module)
    fake = _FakeUser32()
    monkeypatch.setattr(sys.modules["iSLAT.Modules.GUI.GUI"], "_windows_user32", lambda: fake)
    return fake


WM_SETICON, GCLP_HICON, GCLP_HICONSM = 0x0080, -14, -34


def test_icon_goes_on_the_frame_and_the_class(user32):
    from iSLAT.Modules.GUI.GUI import set_windows_frame_icon
    window = _FakeWindow(mapped=True)
    set_windows_frame_icon(window, "logo.ico")
    big, small = window._islat_icon_handles
    assert user32.sent == [(0x20, WM_SETICON, 1, big), (0x20, WM_SETICON, 0, small)]
    assert user32.class_icons == [(0x20, GCLP_HICON, big), (0x20, GCLP_HICONSM, small)]
    assert window.map_handlers == []


def test_waits_for_the_frame_until_the_window_is_mapped(user32):
    from types import SimpleNamespace
    from iSLAT.Modules.GUI.GUI import set_windows_frame_icon
    window = _FakeWindow(mapped=False)
    set_windows_frame_icon(window, "logo.ico")
    assert user32.sent == [] and len(window.map_handlers) == 1
    on_map = window.map_handlers[0]
    window.mapped = True
    on_map(SimpleNamespace(widget=object()))     # a child widget's <Map>
    assert user32.sent == []
    on_map(SimpleNamespace(widget=window))
    assert [hwnd for hwnd, *_ in user32.sent] == [0x20, 0x20]
    on_map(SimpleNamespace(widget=window))       # applied once
    assert len(user32.sent) == 2


def test_setting_again_replaces_and_releases_the_old_icons(user32):
    from iSLAT.Modules.GUI.GUI import set_windows_frame_icon
    window = _FakeWindow(mapped=True)
    set_windows_frame_icon(window, "logo.ico")
    old = window._islat_icon_handles
    set_windows_frame_icon(window, "logo.ico")
    new = window._islat_icon_handles
    assert set(new).isdisjoint(old)
    assert user32.sent[-2:] == [(0x20, WM_SETICON, 1, new[0]), (0x20, WM_SETICON, 0, new[1])]
    assert sorted(user32.destroyed) == sorted(old)


class _FakeCanvas:
    def __init__(self):
        self.handlers = {}

    def mpl_connect(self, name, handler):
        self.handlers[len(self.handlers)] = (name, handler)
        return len(self.handlers) - 1

    def mpl_disconnect(self, cid):
        del self.handlers[cid]

    def draw(self):
        for name, handler in list(self.handlers.values()):
            if name == "draw_event":
                handler(None)


def _gui_with_icon(icon_file):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from iSLAT.Modules.GUI.GUI import GUI
    gui = GUI.__new__(GUI)
    gui.master = MagicMock()
    gui.master._islat_icon_file = icon_file
    gui.plot = SimpleNamespace(canvas=_FakeCanvas())
    return gui


def test_icon_is_sent_again_after_the_next_draw():
    gui = _gui_with_icon("logo.ico")
    gui._refresh_window_icon_after_next_draw()
    gui.master.after_idle.assert_not_called()
    gui.plot.canvas.draw()
    gui.master.after_idle.assert_called_once_with(gui._refresh_window_icon)
    gui.plot.canvas.draw()                        # only the first draw
    assert gui.master.after_idle.call_count == 1 and gui.plot.canvas.handlers == {}


def test_no_refresh_without_a_windows_icon():
    gui = _gui_with_icon(None)
    gui._refresh_window_icon_after_next_draw()
    assert gui.plot.canvas.handlers == {}
    gui.master.after_idle.assert_not_called()


def test_refresh_sets_the_icon_again(user32):
    gui = _gui_with_icon("logo.ico")
    gui.master = _FakeWindow(mapped=True)
    gui.master._islat_icon_file = "logo.ico"
    gui._refresh_window_icon()
    assert [hwnd for hwnd, *_ in user32.sent] == [0x20, 0x20]


def test_async_display_complete_schedules_the_refresh(monkeypatch):
    from unittest.mock import MagicMock
    gui = _gui_with_icon("logo.ico")
    refresh = MagicMock()
    monkeypatch.setattr(gui, "_refresh_window_icon_after_next_draw", refresh)
    gui._on_async_display_complete(False)
    refresh.assert_called_once_with()
