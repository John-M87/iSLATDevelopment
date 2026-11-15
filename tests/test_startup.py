# -*- coding: utf-8 -*-
"""iSLAT starts with the spectrum file dialog while the rest of iSLAT is imported in the background.

* iSLAT.startup (and what it uses before the dialog: the FileHandling paths
  and the tkinter-only file dialog) imports no numpy, pandas or matplotlib,
  so the dialog can open at once.
* startup.main starts the heavy imports on a background thread, then shows
  the dialog unless a default spectrum is configured, waits for the imports
  and runs iSLAT with the chosen file.  Cancelling the dialog exits without
  building iSLAT.  BackgroundImports re-raises an import error on the main
  thread.
* The chosen file is loaded like a file picked in iSLAT's own dialog: it
  wins over the configured default and counts as chosen by the user (a new
  primary spectrum clears the sample).
* Windows DPI awareness is set once, before the first window; DisplayConfig
  uses the same function.
"""
import json
import os
import subprocess
import sys
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_startup_and_the_dialog_need_no_heavy_modules():
    code = (
        "import sys; import iSLAT.startup; "
        "from iSLAT.Modules.FileHandling import example_data_folder_path; "
        "from iSLAT.Modules.FileHandling.file_dialogs import file_selector; "
        "print(sorted(m for m in ('numpy', 'pandas', 'matplotlib', 'iSLAT.iSLATClass', 'iSLAT.Modules.GUI') "
        "if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
@pytest.fixture
def launch(monkeypatch):
    """startup.main with the imports, dialog, settings and iSLAT replaced by recorders."""
    import iSLAT.startup as startup
    import iSLAT.iSLATClass as islat_module
    events = []
    state = {"default": None, "picked": "C:/data/spectrum.csv", "apps": []}

    class RecordingImports:
        def start(self):
            events.append("imports started")
            return self

        def wait(self):
            events.append("imports waited for")

    def file_selector(title=None, initialdir=None, **kwargs):
        events.append("dialog")
        state["dialog_title"] = title
        return state["picked"]

    class RecordingApp:
        def __init__(self):
            events.append("iSLAT built")
            state["apps"].append(self)

        def run(self, **kwargs):
            events.append("run")
            self.run_kwargs = kwargs

    import iSLAT.Modules.FileHandling.file_dialogs as file_dialogs
    monkeypatch.setattr(startup, "BackgroundImports", RecordingImports)
    monkeypatch.setattr(startup, "configured_default_spectrum", lambda: state["default"])
    monkeypatch.setattr(startup, "enable_windows_dpi_awareness", lambda: events.append("dpi"))
    monkeypatch.setattr(file_dialogs, "file_selector", file_selector)
    monkeypatch.setattr(islat_module, "iSLAT", RecordingApp)
    state["events"] = events
    return startup, state


def test_dialog_opens_while_importing_then_runs_with_the_chosen_file(launch):
    startup, state = launch
    startup.main()
    assert state["events"] == ["dpi", "imports started", "dialog", "imports waited for", "iSLAT built", "run"]
    assert state["dialog_title"] == startup.SPECTRUM_DIALOG_TITLE
    assert state["apps"][0].run_kwargs == {"spectrum_file": state["picked"], "show_banner": False}


def test_no_dialog_with_a_default_spectrum(launch):
    startup, state = launch
    state["default"] = "C:/data/default.csv"
    startup.main()
    assert "dialog" not in state["events"]
    assert state["apps"][0].run_kwargs == {"spectrum_file": None, "show_banner": False}


def test_cancelling_the_dialog_exits(launch):
    startup, state = launch
    state["picked"] = ""
    with pytest.raises(SystemExit) as exit_info:
        startup.main()
    assert exit_info.value.code == 0
    assert state["events"] == ["dpi", "imports started", "dialog"]


# ----------------------------------------------------------------------
# BackgroundImports
# ----------------------------------------------------------------------
@pytest.fixture
def module_dir(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [m for m in sys.modules if m.startswith("islat_startup_probe")]:
        del sys.modules[name]


def test_imports_on_another_thread(module_dir):
    from iSLAT.startup import BackgroundImports
    (module_dir / "islat_startup_probe_ok.py").write_text(
        "import threading\nTHREAD = threading.current_thread()\n")
    imports = BackgroundImports(["islat_startup_probe_ok"]).start()
    imports.wait()
    module = sys.modules["islat_startup_probe_ok"]
    assert module.THREAD is not threading.main_thread()


def test_an_import_error_is_raised_by_wait(module_dir):
    from iSLAT.startup import BackgroundImports
    (module_dir / "islat_startup_probe_bad.py").write_text("raise ValueError('broken module')\n")
    imports = BackgroundImports(["islat_startup_probe_bad"]).start()
    with pytest.raises(ValueError, match="broken module"):
        imports.wait()


# ----------------------------------------------------------------------
# configured_default_spectrum
# ----------------------------------------------------------------------
@pytest.fixture
def settings_folder(tmp_path, monkeypatch):
    import iSLAT.Modules.FileHandling as file_handling
    monkeypatch.setattr(file_handling, "user_configuration_file_path", tmp_path)
    return tmp_path


def _write(folder, name, settings):
    (folder / name).write_text(json.dumps(settings))


@pytest.mark.parametrize("settings, expected", [
    ({"default_spectra_file_to_load": None}, None),
    ({"default_spectra_file_to_load": "C:/data/a.csv"}, "C:/data/a.csv"),
    ({}, None),
], ids=["null", "set", "missing"])
def test_default_spectrum_from_the_user_settings(settings_folder, settings, expected):
    from iSLAT.startup import configured_default_spectrum
    _write(settings_folder, "UserSettings.json", settings)
    _write(settings_folder, "DefaultUserSettings.json", {"default_spectra_file_to_load": "C:/data/other.csv"})
    assert configured_default_spectrum() == expected


def test_default_spectrum_from_the_defaults_without_a_user_file(settings_folder):
    from iSLAT.startup import configured_default_spectrum
    _write(settings_folder, "DefaultUserSettings.json", {"default_spectra_file_to_load": "C:/data/b.csv"})
    assert configured_default_spectrum() == "C:/data/b.csv"


def test_unreadable_settings_mean_no_default(settings_folder):
    from iSLAT.startup import configured_default_spectrum
    (settings_folder / "UserSettings.json").write_text("{not json")
    assert configured_default_spectrum() is None


# ----------------------------------------------------------------------
# iSLAT.run / load_spectrum with the chosen file
# ----------------------------------------------------------------------
@pytest.fixture
def example_spectrum():
    from iSLAT.Modules.FileHandling import example_data_folder_path
    path = os.path.join(example_data_folder_path, "CITau_MIRI_Banzatti+2023b.csv")
    if not os.path.exists(path):
        pytest.skip("example spectrum not available")
    return path


@pytest.fixture
def app(monkeypatch):
    """An iSLAT whose molecule loading and sample saving are stubbed (nothing is written)."""
    from iSLAT.iSLATClass import iSLAT
    app = iSLAT()
    app._user_settings = {"theme": {}, "default_spectra_file_to_load": "C:/does/not/exist.csv"}
    saves = []
    monkeypatch.setattr(app, "_initialize_molecules_for_spectrum", lambda: None)
    monkeypatch.setattr(app, "save_sample_state", lambda: saves.append(1))
    app.sample_saves = saves
    return app


def test_chosen_file_wins_over_the_default_and_counts_as_chosen_by_the_user(app, example_spectrum):
    app.sample_spectra = ["C:/data/old_sample_member.csv"]
    assert app.load_spectrum(selected_file=example_spectrum)
    assert app.loaded_spectrum_file == example_spectrum
    assert app.sample_spectra == [] and app.sample_saves == [1]


def test_run_loads_the_chosen_file(app, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(app, "check_HITRAN", lambda: calls.append("check_HITRAN"))
    monkeypatch.setattr(app, "load_spectrum", lambda **kwargs: calls.append(("load_spectrum", kwargs)) or True)
    monkeypatch.setattr(app, "init_gui", lambda: calls.append("init_gui"))
    app._user_settings["restore_sample_on_startup"] = False
    app.run(spectrum_file="C:/data/picked.csv", show_banner=False)
    assert calls == ["check_HITRAN", ("load_spectrum", {"selected_file": "C:/data/picked.csv"}), "init_gui"]
    assert "iSLAT" not in capsys.readouterr().out

    calls.clear()
    app.run()
    assert calls[1] == ("load_spectrum", {"selected_file": None})
    assert "Please select a spectrum file to load." in capsys.readouterr().out


def test_gui_file_selector_uses_the_light_dialog(monkeypatch):
    import iSLAT.Modules.FileHandling.file_dialogs as file_dialogs
    from iSLAT.Modules.GUI.GUI import GUI
    asked = []

    def file_selector(**kwargs):
        asked.append(kwargs)
        return "C:/data/x.csv"

    monkeypatch.setattr(file_dialogs, "file_selector", file_selector)
    assert GUI.file_selector(title="T", initialdir="C:/data", allow_multiple=False) == "C:/data/x.csv"
    assert asked == [dict(title="T", filetypes=None, initialdir="C:/data", use_abspath=True, allow_multiple=False)]


# ----------------------------------------------------------------------
# DPI awareness
# ----------------------------------------------------------------------
@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPI awareness")
def test_dpi_awareness_is_set_once(monkeypatch):
    import ctypes
    import iSLAT.startup as startup
    from types import SimpleNamespace
    calls = []
    fake = SimpleNamespace(shcore=SimpleNamespace(SetProcessDpiAwareness=lambda value: calls.append(value)),
                           user32=SimpleNamespace(SetProcessDPIAware=lambda: calls.append("system")))
    monkeypatch.setattr(ctypes, "windll", fake)
    monkeypatch.setattr(startup, "_dpi_awareness_done", False)
    startup.enable_windows_dpi_awareness()
    startup.enable_windows_dpi_awareness()
    assert calls == [2]


def test_display_config_uses_the_startup_dpi_awareness():
    import inspect
    from iSLAT.Modules.GUI import DisplayConfig
    assert "enable_windows_dpi_awareness()" in inspect.getsource(DisplayConfig._detect_windows_scale_factor)
