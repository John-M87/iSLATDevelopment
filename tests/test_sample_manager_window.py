# -*- coding: utf-8 -*-
"""Unit tests for the SampleManagerWindow helpers, per-spectrum global
overrides, and sample config persistence."""
import json
import os

import pytest

from iSLAT.Modules.GUI.Widgets.SampleManagerWindow import (
    format_override_value,
    load_global_override_fields,
    parse_override_value,
)
from iSLAT.Modules.FileHandling.iSLATFileHandling import (
    load_sample_config,
    save_sample_config,
)


# ---------------------------------------------------------------------------
# parse_override_value / format_override_value
# ---------------------------------------------------------------------------

def test_parse_override_blank_means_no_override():
    assert parse_override_value("") == (None, True)
    assert parse_override_value("   ") == (None, True)
    assert parse_override_value(None) == (None, True)


def test_parse_override_valid_floats():
    assert parse_override_value("140") == (140.0, True)
    assert parse_override_value(" -12.5 ") == (-12.5, True)
    assert parse_override_value("1e2") == (100.0, True)


def test_parse_override_invalid():
    assert parse_override_value("abc") == (None, False)
    assert parse_override_value("12,5") == (None, False)


def test_format_override_value():
    assert format_override_value(140.0) == "140"
    assert format_override_value(0.123456789) == "0.123457"
    assert format_override_value(5, "{:.4g}") == "5"


# ---------------------------------------------------------------------------
# load_global_override_fields
# ---------------------------------------------------------------------------

def test_load_global_override_fields_contains_expected_keys():
    fields = load_global_override_fields()
    assert "distance" in fields
    assert "stellar_rv" in fields
    for cfg in fields.values():
        assert cfg.get("property")


# ---------------------------------------------------------------------------
# Sample config persistence round-trip
# ---------------------------------------------------------------------------

def test_sample_config_round_trip(tmp_path):
    spec = tmp_path / "spec1.csv"
    spec.write_text("wave,flux\n1,2\n")
    param = tmp_path / "params.csv"
    param.write_text("Molecule Name\n")

    config = {
        "active_index": 0,
        "spectra": [
            {
                "file": str(spec),
                "param_file": str(param),
                "overrides": {"distance": 100.0, "stellar_rv": 5.5},
            }
        ],
    }
    out = tmp_path / "SampleConfig.json"
    save_sample_config(config, str(out))
    loaded = load_sample_config(str(out))

    assert loaded["active_index"] == 0
    assert len(loaded["spectra"]) == 1
    entry = loaded["spectra"][0]
    assert entry["file"] == str(spec)
    assert entry["param_file"] == str(param)
    assert entry["overrides"] == {"distance": 100.0, "stellar_rv": 5.5}


def test_sample_config_missing_file_returns_defaults(tmp_path):
    loaded = load_sample_config(str(tmp_path / "nope.json"))
    assert loaded == {"active_index": 0, "spectra": []}


def test_sample_config_skips_missing_spectra(tmp_path):
    spec = tmp_path / "exists.csv"
    spec.write_text("wave,flux\n1,2\n")
    config = {
        "active_index": 1,
        "spectra": [
            {"file": str(tmp_path / "gone.csv"), "param_file": None, "overrides": {}},
            {"file": str(spec), "param_file": None, "overrides": {}},
        ],
    }
    out = tmp_path / "SampleConfig.json"
    save_sample_config(config, str(out))
    loaded = load_sample_config(str(out))

    assert [e["file"] for e in loaded["spectra"]] == [str(spec)]
    # active_index clamped into range after the drop
    assert loaded["active_index"] == 0


def test_sample_config_clears_missing_param_file(tmp_path):
    spec = tmp_path / "spec.csv"
    spec.write_text("wave,flux\n1,2\n")
    config = {
        "active_index": 0,
        "spectra": [
            {
                "file": str(spec),
                "param_file": str(tmp_path / "gone_params.csv"),
                "overrides": {"distance": 50.0},
            }
        ],
    }
    out = tmp_path / "SampleConfig.json"
    save_sample_config(config, str(out))
    loaded = load_sample_config(str(out))

    assert loaded["spectra"][0]["param_file"] is None
    assert loaded["spectra"][0]["overrides"] == {"distance": 50.0}


def test_sample_config_corrupt_json_returns_defaults(tmp_path):
    out = tmp_path / "SampleConfig.json"
    out.write_text("{not valid json")
    loaded = load_sample_config(str(out))
    assert loaded == {"active_index": 0, "spectra": []}


# ---------------------------------------------------------------------------
# iSLAT-level override application and removal
# ---------------------------------------------------------------------------

class _FakeMoleculeDict(dict):
    def __init__(self):
        super().__init__()
        self.global_distance = 140.0
        self.global_stellar_rv = 0.0
        self.global_model_pixel_res = 0.02


@pytest.fixture
def islat_with_sample(tmp_path, monkeypatch):
    """A real iSLAT instance (no GUI) with two fake spectra in the sample."""
    from iSLAT.iSLATClass import iSLAT

    islat = iSLAT.__new__(iSLAT)  # skip heavy __init__
    islat.molecules_dict = _FakeMoleculeDict()
    islat.sample_spectra = []
    islat.sample_spectra_index = 0
    islat.sample_spectra_params = {}
    islat.sample_spectra_overrides = {}
    islat._global_fields_config = None
    islat._suppress_sample_save = False

    spec_a = tmp_path / "a.csv"
    spec_a.write_text("wave,flux\n1,2\n")
    spec_b = tmp_path / "b.csv"
    spec_b.write_text("wave,flux\n1,2\n")
    islat.sample_spectra = [str(spec_a), str(spec_b)]

    # Redirect persistence to tmp_path
    cfg_path = tmp_path / "SampleConfig.json"
    import iSLAT.iSLATClass as islat_module

    monkeypatch.setattr(
        islat_module, "save_sample_config",
        lambda cfg, file_path=None: save_sample_config(cfg, str(cfg_path)),
    )
    monkeypatch.setattr(
        islat_module, "load_sample_config",
        lambda file_path=None: load_sample_config(str(cfg_path)),
    )
    islat._test_cfg_path = str(cfg_path)
    return islat


def test_apply_sample_overrides_sets_properties(islat_with_sample):
    islat = islat_with_sample
    path = islat.sample_spectra[0]
    islat.sample_spectra_overrides[path] = {"distance": 200.0, "stellar_rv": 12.0}

    islat._apply_sample_overrides(path)

    assert islat.molecules_dict.global_distance == 200.0
    assert islat.molecules_dict.global_stellar_rv == 12.0


def test_apply_sample_overrides_ignores_unknown_key(islat_with_sample):
    islat = islat_with_sample
    path = islat.sample_spectra[0]
    islat.sample_spectra_overrides[path] = {"bogus_key": 1.0, "distance": 90.0}

    islat._apply_sample_overrides(path)

    assert islat.molecules_dict.global_distance == 90.0
    assert not hasattr(islat.molecules_dict, "bogus_key")


def test_apply_sample_overrides_no_entry_is_noop(islat_with_sample):
    islat = islat_with_sample
    islat._apply_sample_overrides(islat.sample_spectra[0])
    assert islat.molecules_dict.global_distance == 140.0


def test_remove_sample_spectrum_purges_params_and_overrides(islat_with_sample):
    islat = islat_with_sample
    a, b = islat.sample_spectra
    islat.sample_spectra_params[a] = "/some/params.csv"
    islat.sample_spectra_overrides[a] = {"distance": 10.0}
    islat.sample_spectra_index = 1

    islat.remove_sample_spectrum(0)

    assert islat.sample_spectra == [b]
    assert a not in islat.sample_spectra_params
    assert a not in islat.sample_spectra_overrides
    assert islat.sample_spectra_index == 0


def test_remove_sample_spectrum_out_of_range_is_noop(islat_with_sample):
    islat = islat_with_sample
    before = list(islat.sample_spectra)
    islat.remove_sample_spectrum(5)
    islat.remove_sample_spectrum(-1)
    assert islat.sample_spectra == before


def test_clear_sample_spectra_purges_everything(islat_with_sample):
    islat = islat_with_sample
    a = islat.sample_spectra[0]
    islat.sample_spectra_params[a] = "/p.csv"
    islat.sample_spectra_overrides[a] = {"distance": 1.0}

    islat.clear_sample_spectra()

    assert islat.sample_spectra == []
    assert islat.sample_spectra_params == {}
    assert islat.sample_spectra_overrides == {}
    assert islat.sample_spectra_index == 0


def test_save_and_restore_sample_state_round_trip(islat_with_sample):
    islat = islat_with_sample
    a, b = islat.sample_spectra
    islat.sample_spectra_params[b] = a  # any existing file works as param stub
    islat.sample_spectra_overrides[b] = {"stellar_rv": -3.0}
    islat.sample_spectra_index = 1

    islat.save_sample_state()

    # Wipe and restore
    islat.sample_spectra = []
    islat.sample_spectra_params = {}
    islat.sample_spectra_overrides = {}
    islat.sample_spectra_index = 0
    islat.restore_sample_state()

    assert islat.sample_spectra == [a, b]
    assert islat.sample_spectra_params == {b: a}
    assert islat.sample_spectra_overrides == {b: {"stellar_rv": -3.0}}
    assert islat.sample_spectra_index == 1


def test_restore_inserts_loaded_spectrum_when_absent(islat_with_sample, tmp_path):
    islat = islat_with_sample
    islat.save_sample_state()

    other = tmp_path / "other.csv"
    other.write_text("wave,flux\n1,2\n")
    islat.loaded_spectrum_file = str(other)

    islat.sample_spectra = []
    islat.restore_sample_state()

    assert islat.sample_spectra[0] == str(other)
    assert islat.sample_spectra_index == 0


def test_restore_syncs_index_to_loaded_spectrum(islat_with_sample):
    islat = islat_with_sample
    a, b = islat.sample_spectra
    islat.sample_spectra_index = 0
    islat.save_sample_state()

    islat.loaded_spectrum_file = b
    islat.restore_sample_state()

    assert islat.sample_spectra_index == islat.sample_spectra.index(b)


def test_save_sample_state_suppressed(islat_with_sample):
    islat = islat_with_sample
    islat._suppress_sample_save = True
    islat.save_sample_state()
    assert not os.path.exists(islat._test_cfg_path)


# ---------------------------------------------------------------------------
# Control panel sync after switching sample spectra
# ---------------------------------------------------------------------------

class _FakeControlPanel:
    def __init__(self):
        self.refresh_count = 0

    def refresh_from_molecules_dict(self):
        self.refresh_count += 1


class _FakeGUI:
    def __init__(self):
        self.control_panel = _FakeControlPanel()
        self.plot = None


def test_switch_to_spectrum_refreshes_control_panel_after_overrides(
    islat_with_sample, monkeypatch
):
    """Regression: control panel must be re-synced *after* param file /
    overrides are applied, not just by load_spectrum's earlier refresh."""
    islat = islat_with_sample
    islat.GUI = _FakeGUI()
    islat.loaded_spectrum_file = islat.sample_spectra[0]
    islat.sample_spectra_index = 0

    target = islat.sample_spectra[1]
    islat.sample_spectra_overrides[target] = {"distance": 250.0}

    events = []
    monkeypatch.setattr(
        type(islat), "load_spectrum",
        lambda self, file_path=None, load_parameters=False, force_dialog=False: (
            events.append("load"), setattr(self, "loaded_spectrum_file", file_path), True
        )[-1],
    )

    orig_refresh = type(islat)._refresh_gui_after_parameter_change
    monkeypatch.setattr(
        type(islat), "_refresh_gui_after_parameter_change",
        lambda self, update_plot=True: (
            events.append("refresh"), orig_refresh(self, update_plot)
        )[-1],
    )

    islat.switch_to_spectrum(1)

    # Override was applied and the control panel refreshed afterwards
    assert islat.molecules_dict.global_distance == 250.0
    assert islat.GUI.control_panel.refresh_count == 1
    assert events == ["load", "refresh"]


def test_switch_to_spectrum_no_extra_refresh_without_overrides(
    islat_with_sample, monkeypatch
):
    islat = islat_with_sample
    islat.GUI = _FakeGUI()
    islat.loaded_spectrum_file = islat.sample_spectra[0]
    islat.sample_spectra_index = 0

    monkeypatch.setattr(
        type(islat), "load_spectrum",
        lambda self, file_path=None, load_parameters=False, force_dialog=False: (
            setattr(self, "loaded_spectrum_file", file_path), True
        )[-1],
    )

    islat.switch_to_spectrum(1)

    # load_spectrum already refreshes; no redundant explicit refresh needed
    assert islat.GUI.control_panel.refresh_count == 0


def test_refresh_gui_helper_handles_missing_gui(islat_with_sample):
    islat = islat_with_sample
    islat.GUI = None
    # Should not raise
    islat._refresh_gui_after_parameter_change()


def test_bottom_buttons_are_packed_before_the_rows_area(monkeypatch):
    """pack shrinks the last-packed widgets first: the button bar must come before the expanding rows area."""
    import tkinter as tk
    from tkinter import ttk
    from types import SimpleNamespace
    from iSLAT.Modules.GUI.Widgets.SampleManagerWindow import SampleManagerWindow
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    original = tk.Toplevel.__init__

    def hidden(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.withdraw()

    monkeypatch.setattr(tk.Toplevel, "__init__", hidden)
    islat = SimpleNamespace(sample_spectra=["/data/a.csv", "/data/b.csv"], sample_spectra_index=0,
                            sample_spectra_params={}, sample_spectra_overrides={})
    try:
        win = SampleManagerWindow(root, islat, {})._win
        slaves = win.pack_slaves()
        bar = next(w for w in slaves if any(isinstance(b, ttk.Button) and b.cget("text") == "Close"
                                            for b in w.winfo_children()))
        rows = next(w for w in slaves if w.pack_info().get("expand") in (1, True, "1"))
        assert bar.pack_info()["side"] == "bottom"
        assert slaves.index(bar) < slaves.index(rows)
        win.destroy()
    finally:
        root.withdraw()
        root.destroy()



# ---------------------------------------------------------------------------
# Window behaviour (hidden windows)
# ---------------------------------------------------------------------------

@pytest.fixture
def manager(monkeypatch):
    """A SampleManagerWindow over a fake iSLAT with 30 spectra; every window stays hidden."""
    import tkinter as tk
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from iSLAT.Modules.GUI.Widgets.SampleManagerWindow import SampleManagerWindow
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    original = tk.Toplevel.__init__

    def hidden(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.withdraw()

    monkeypatch.setattr(tk.Toplevel, "__init__", hidden)
    islat = SimpleNamespace(
        sample_spectra=[f"/data/spec_{i}.csv" for i in range(30)], sample_spectra_index=0,
        sample_spectra_params={}, sample_spectra_overrides={},
        GUI=SimpleNamespace(data_field=MagicMock(), file_interaction_pane=MagicMock()),
    )

    def switch(index):
        islat.sample_spectra_index = index

    islat.switch_to_spectrum = switch
    window = SampleManagerWindow(root, islat, {})
    window._win.update_idletasks()
    yield window, islat
    window._win.destroy()
    root.withdraw()
    root.destroy()


def _bold_rows(window):
    return [i for i, (_, _, lbl, _, _) in enumerate(window._row_widgets) if "bold" in str(lbl.cget("font"))]


def test_activating_a_row_moves_the_highlight_without_rebuilding(manager, monkeypatch):
    window, islat = manager
    first_widgets = list(window._row_widgets)
    rebuilds = []
    monkeypatch.setattr(window, "_rebuild_rows", lambda: rebuilds.append(1))
    window._activate(5)
    assert islat.sample_spectra_index == 5
    assert rebuilds == []
    assert window._row_widgets == first_widgets   # same widgets, not recreated
    assert _bold_rows(window) == [5]
    bg = {i: outer.cget("bg") for i, (outer, *_rest) in enumerate(window._row_widgets)}
    assert bg[5] != bg[0] and bg[0] == bg[1]


def test_switch_errors_reach_the_message_pane(manager):
    window, islat = manager

    def broken(index):
        raise OSError("file vanished")

    islat.switch_to_spectrum = broken
    window._activate(3)
    message = islat.GUI.data_field.insert_text.call_args[0][0]
    assert "spectrum 4" in message and "file vanished" in message
    assert _bold_rows(window) == [0]    # still highlights the spectrum that is really active


def test_rebuild_keeps_the_users_window_size(manager, monkeypatch):
    """Only the first build sizes the window; later rebuilds leave the user's size alone."""
    window, _islat = manager
    sizes = []
    monkeypatch.setattr(window._win, "geometry", lambda *args: sizes.append(args))
    window._rebuild_rows()
    window._activate(2)
    assert sizes == []


def test_rebuild_does_not_stack_touchpad_handlers(manager):
    window, _islat = manager
    if not window._rows_frame.bind("<TouchpadScroll>"):
        pytest.skip("no <TouchpadScroll> in Tk 8.6")
    for _ in range(4):
        window._rebuild_rows()
    script = window._rows_frame.bind("<TouchpadScroll>")
    assert len([line for line in script.splitlines() if line.strip()]) == 1


@pytest.mark.parametrize("system, delta, expected", [
    ("Darwin", 1, -1), ("Darwin", -3, 3), ("Darwin", 0, 0),
    ("Windows", 120, -1), ("Windows", -240, 2), ("Windows", 30, -1),
])
def test_mouse_wheel_scrolls_on_every_platform(manager, monkeypatch, system, delta, expected):
    from types import SimpleNamespace
    window, _islat = manager
    monkeypatch.setattr("platform.system", lambda: system)
    calls = []
    monkeypatch.setattr(window._canvas, "yview_scroll", lambda n, what: calls.append((n, what)))
    window._on_mouse_wheel(SimpleNamespace(delta=delta))
    assert calls == ([(expected, "units")] if expected else [])
