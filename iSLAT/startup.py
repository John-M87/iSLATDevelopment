"""Application start: the spectrum file dialog opens while iSLAT is still being imported.

Both ``iSLAT-launch.py`` and the ``islat`` console script start here.
"""
import json
import os
import platform
import sys
import threading
from typing import Iterable, Optional

#: Modules imported on the background thread: everything the startup needs
#: after the dialog (the iSLAT class, the plotting package, the GUI).
HEAVY_MODULES = (
    "iSLAT.iSLATClass",
    "iSLAT.Modules.Plotting",
    "iSLAT.Modules.GUI",
    "iSLAT.Modules.Plotting.MainPlot",
)

SPECTRUM_DIALOG_TITLE = "Choose Spectrum Data File"

_dpi_awareness_done = False

def enable_windows_dpi_awareness() -> None:
    """Make the process DPI aware on Windows (per-monitor, else system); once, before any window exists.

    Windows decides how a window is scaled when it is created, so this runs
    before the first Tk window (the startup dialog).  Elsewhere a no-op.
    """
    global _dpi_awareness_done
    if _dpi_awareness_done or platform.system() != "Windows":
        return
    _dpi_awareness_done = True
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

class BackgroundImports:
    """Import *modules* on a daemon thread; :meth:`wait` joins it and re-raises an import error."""

    def __init__(self, modules: Iterable[str] = HEAVY_MODULES):
        self.modules = tuple(modules)
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="iSLAT-imports", daemon=True)

    def start(self) -> "BackgroundImports":
        self._thread.start()
        return self

    def _run(self) -> None:
        import importlib
        try:
            for name in self.modules:
                importlib.import_module(name)
        except BaseException as exc:  # re-raised on the main thread by wait()
            self.error = exc

    def wait(self) -> None:
        self._thread.join()
        if self.error is not None:
            raise self.error

def configured_default_spectrum() -> Optional[str]:
    """The ``default_spectra_file_to_load`` user setting, read without loading iSLAT.

    Read from UserSettings.json, or DefaultUserSettings.json when there is no
    user file yet (as the settings loader does).  Only decides whether the
    startup shows the dialog: when it does not, iSLAT reads the settings
    properly and still asks for a file itself if none is configured.
    """
    from iSLAT.Modules.FileHandling import (
        user_configuration_file_path, user_configuration_file_name, default_user_settings_file_name,
    )
    user_file = os.path.join(user_configuration_file_path, user_configuration_file_name)
    settings_file = user_file if os.path.exists(user_file) else \
        os.path.join(user_configuration_file_path, default_user_settings_file_name)
    try:
        with open(settings_file, "r") as fh:
            return json.load(fh).get("default_spectra_file_to_load") or None
    except (OSError, ValueError, AttributeError):
        return None

def print_banner() -> None:
    from iSLAT import __version__
    print("\n" + "=" * 60)
    print(f"iSLAT {__version__} - interactive Spectral-Line Analysis Tool")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("Please select a spectrum file to load.")
    print("=" * 60)

def main() -> None:
    """Start iSLAT: dialog first, imports in the background, then the GUI."""
    enable_windows_dpi_awareness()
    print_banner()
    # Everything the main thread needs before the dialog is imported now,
    # before the import thread starts, so the two never import a module at once.
    from iSLAT.Modules.FileHandling import example_data_folder_path
    from iSLAT.Modules.FileHandling.file_dialogs import file_selector

    imports = BackgroundImports().start()

    spectrum_file = None
    if configured_default_spectrum() is None:
        spectrum_file = file_selector(title=SPECTRUM_DIALOG_TITLE, initialdir=example_data_folder_path)
        if not spectrum_file:
            print("No spectrum selected. Exiting...")
            sys.exit(0)

    imports.wait()
    from iSLAT.iSLATClass import iSLAT
    app = iSLAT()
    app.run(spectrum_file=spectrum_file, show_banner=False)