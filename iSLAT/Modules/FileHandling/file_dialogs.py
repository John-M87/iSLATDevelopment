"""File-open dialog that needs only tkinter.

Kept out of the GUI package (whose import pulls in matplotlib and the whole
GUI) so the startup can show the spectrum dialog while the rest of iSLAT is
still being imported; see :mod:`iSLAT.startup`.
"""
import os
import platform
import tkinter as tk
from tkinter import filedialog

def file_selector(title: str = None, filetypes=None, initialdir=None, use_abspath=True, allow_multiple=False):
    """Ask for one file (a path, ``""`` when cancelled) or several (a list).

    Without a Tk root (at startup) a hidden temporary root is made for the
    dialog and destroyed afterwards.
    """
    window_title = title if title else "Select File"
    if use_abspath and initialdir:
        initialdir = os.path.abspath(initialdir)
    elif initialdir is None:
        initialdir = os.getcwd()

    if filetypes is None:
        filetypes = [("All Files", "*.*")]
    elif isinstance(filetypes, str):
        filetypes = [(filetypes, "*.*")]

    # If no Tk root window exists yet (initial launch), create one and
    # bring the app to the foreground so the native file dialog is
    # visible and interactive on macOS.
    temp_root = None
    if tk._default_root is None:
        temp_root = tk.Tk()
        temp_root.withdraw()

    # On macOS, nudge the process to the foreground so the file
    # dialog is not hidden behind other application windows.
    if platform.system() == "Darwin":
        try:
            from AppKit import NSApplication, NSApp  # type: ignore[import-untyped]
            NSApplication.sharedApplication()
            NSApp.activateIgnoringOtherApps_(True)
        except ImportError:
            # PyObjC not available — fall back to osascript
            try:
                import subprocess
                subprocess.Popen(
                    ["osascript", "-e",
                     'tell application "System Events" to set frontmost '
                     'of the first process whose unix id is '
                     + str(os.getpid()) + ' to true'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    try:
        if allow_multiple:
            file_paths = filedialog.askopenfilenames(
                title=window_title,
                filetypes=filetypes,
                initialdir=initialdir,
            )
            return list(file_paths)

        return filedialog.askopenfilename(
            title=window_title,
            filetypes=filetypes,
            initialdir=initialdir,
        )
    finally:
        if temp_root is not None:
            temp_root.destroy()