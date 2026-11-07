# -*- coding: utf-8 -*-
"""Tests for choosing the intensity calculation method (Manage Molecules -> Intensity Settings).

Covers the method registry and descriptions, the caches that must follow a
method change (``Intensity`` and ``Molecule``), the propagation through
``MoleculeDict.global_intensity_calculation_method``, the user-setting
mapping, and the Intensity Settings window.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Intensity import Intensity
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
from iSLAT.Modules.GUI.Widgets import IntensitySettingsWindow as isw

from tests.plotting.test_plot_views import _make_test_line_list, _make_test_molecule

METHODS = ["curve_growth", "curve_growth_no_overlap", "radex"]
THICK = dict(t_kin=500.0, n_mol=1e22, dv=1.0)  # optically thick: the methods differ


def _fresh_intensity(method):
    inten = Intensity(_make_test_line_list())
    inten.calc_intensity(method=method, **THICK)
    return inten._intensity.copy()


def _thick_molecule(name="H2O"):
    mol = _make_test_molecule(name=name)
    mol.n_mol = 1e22
    return mol


def _molecule_intensity(mol):
    mol._ensure_intensity_calculated()
    return mol.intensity._intensity.copy()


# ======================================================================
# Registry / descriptions
# ======================================================================

class TestRegistry:
    def test_builtin_methods_default_first(self):
        assert Intensity.available_strategies() == METHODS

    def test_custom_strategy_listed_after_builtins(self):
        from iSLAT.Modules.DataTypes._intensity_strategies import RadexStrategy
        Intensity.register_strategy("my_method", RadexStrategy())
        try:
            assert Intensity.available_strategies() == METHODS + ["my_method"]
            assert Intensity.describe_strategy("my_method") == ("my_method", "")
        finally:
            Intensity._strategy_registry.pop("my_method", None)

    def test_descriptions(self):
        for method in METHODS:
            name, description = Intensity.describe_strategy(method)
            assert name and description
        assert "RADEX" in Intensity.describe_strategy("radex")[0]

    def test_methods_really_differ_for_thick_lines(self):
        results = {m: _fresh_intensity(m) for m in METHODS}
        assert not np.allclose(results["radex"], results["curve_growth"])
        assert not np.allclose(results["curve_growth_no_overlap"], results["radex"])


# ======================================================================
# Caches follow the method
# ======================================================================

class TestCaches:
    def test_intensity_cache_is_keyed_on_method(self):
        inten = Intensity(_make_test_line_list())
        inten.calc_intensity(method="curve_growth", **THICK)
        cog = inten._intensity.copy()
        inten.calc_intensity(method="radex", **THICK)  # same T, N, dv: must not be a cache hit
        assert np.allclose(inten._intensity, _fresh_intensity("radex"))
        assert inten._method == "radex"
        inten.calc_intensity(method="curve_growth", **THICK)
        assert np.allclose(inten._intensity, cog)

    def test_molecule_bulk_update_accepts_method_name(self):
        mol = _thick_molecule()
        mol.bulk_update_parameters({"intensity_calculation_method": "radex"})
        assert mol.intensity_calculation_method == "radex"
        assert np.allclose(_molecule_intensity(mol), _fresh_intensity("radex"))

    def test_global_setter_recalculates_every_molecule(self):
        md = MoleculeDict()
        md["H2O"] = _thick_molecule("H2O")
        md["OH"] = _thick_molecule("OH")
        before = {name: _molecule_intensity(mol) for name, mol in md.items()}
        seen = []
        md.add_global_parameter_change_callback(lambda name, old, new: seen.append((name, old, new)))

        md.global_intensity_calculation_method = "radex"

        assert ("intensity_calculation_method", "curve_growth", "radex") in seen
        for name, mol in md.items():
            assert mol.intensity_calculation_method == "radex"
            after = _molecule_intensity(mol)
            assert not np.allclose(after, before[name])
            assert np.allclose(after, _fresh_intensity("radex"))

    def test_full_range_recalculation_uses_the_current_method(self, monkeypatch):
        inten = Intensity(_make_test_line_list())
        inten.calc_intensity(method="radex", **THICK)
        used = []
        original = Intensity.calc_intensity

        def spy(self, *args, **kwargs):
            used.append(kwargs.get("method"))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Intensity, "calc_intensity", spy)
        inten._compute_full_range_intensity(inten.molecule._raw_lines_data)
        assert used == ["radex"]


# ======================================================================
# User setting
# ======================================================================

class TestUserSetting:
    @pytest.mark.parametrize("value, method", [
        (True, "curve_growth"),
        ("curve_growth", "curve_growth"),
        (False, "curve_growth_no_overlap"),
        ("curve_growth_no_overlap", "curve_growth_no_overlap"),
        ("radex", "radex"),
        (None, "curve_growth"),
        ("bogus", "curve_growth"),
    ])
    def test_setting_maps_to_method(self, value, method):
        assert ifh.intensity_method_from_setting(value) == method

    @pytest.mark.parametrize("method", METHODS)
    def test_saved_choice_is_loaded_next_session(self, tmp_path, method):
        import shutil
        from iSLAT.Modules.FileHandling import user_configuration_file_path
        shutil.copy(user_configuration_file_path / "DefaultUserSettings.json", tmp_path)
        settings = ifh.load_user_settings(file_path=tmp_path, file_name="UserSettings.json")
        assert settings["intensity_calculation_method"] == "curve_growth"  # default: true
        settings[ifh.INTENSITY_METHOD_SETTING_KEY] = method
        settings["intensity_calculation_method"] = method
        ifh.save_user_settings(settings, file_path=tmp_path, file_name="UserSettings.json")
        reloaded = ifh.load_user_settings(file_path=tmp_path, file_name="UserSettings.json")
        assert reloaded["intensity_calculation_method"] == method


# ======================================================================
# Apply logic behind the window
# ======================================================================

def _islat(method="curve_growth"):
    md = MoleculeDict(global_intensity_calculation_method=method)
    md["H2O"] = _make_test_molecule("H2O")
    md["H2O"].intensity_calculation_method = method
    return SimpleNamespace(
        molecules_dict=md,
        user_settings={ifh.INTENSITY_METHOD_SETTING_KEY: True, "intensity_calculation_method": method},
    )


class TestApplyIntensityMethod:
    @pytest.fixture(autouse=True)
    def _no_disk_writes(self, monkeypatch):
        self.saved = []
        monkeypatch.setattr(ifh, "save_user_settings", lambda settings: self.saved.append(dict(settings)))

    def test_apply_sets_method_and_saves_setting(self):
        islat = _islat()
        assert isw.apply_intensity_method(islat, "radex") is True
        assert islat.molecules_dict.global_intensity_calculation_method == "radex"
        assert islat.molecules_dict["H2O"].intensity_calculation_method == "radex"
        assert self.saved[-1][ifh.INTENSITY_METHOD_SETTING_KEY] == "radex"
        assert ifh.intensity_method_from_setting(self.saved[-1][ifh.INTENSITY_METHOD_SETTING_KEY]) == "radex"
        assert islat.user_settings["intensity_calculation_method"] == "radex"

    def test_unchanged_method_does_not_recalculate(self):
        islat = _islat("radex")
        calls = []
        islat.molecules_dict.add_global_parameter_change_callback(lambda *a: calls.append(a))
        assert isw.apply_intensity_method(islat, "radex") is False
        assert calls == []
        assert self.saved  # still stored as the default

    def test_molecule_out_of_step_is_updated(self):
        islat = _islat("radex")
        islat.molecules_dict["H2O"].intensity_calculation_method = "curve_growth"
        assert isw.apply_intensity_method(islat, "radex") is True
        assert islat.molecules_dict["H2O"].intensity_calculation_method == "radex"

    def test_unknown_method_is_rejected(self):
        islat = _islat()
        with pytest.raises(ValueError):
            isw.apply_intensity_method(islat, "bogus")
        assert islat.molecules_dict.global_intensity_calculation_method == "curve_growth"
        assert self.saved == []

    def test_no_save_option(self):
        islat = _islat()
        isw.apply_intensity_method(islat, "curve_growth_no_overlap", save=False)
        assert self.saved == []
        assert islat.user_settings[ifh.INTENSITY_METHOD_SETTING_KEY] == "curve_growth_no_overlap"

    def test_current_method_fallbacks(self):
        assert isw.current_method(_islat("radex")) == "radex"
        assert isw.current_method(SimpleNamespace(user_settings={"intensity_calculation_method": "radex"})) == "radex"
        assert isw.current_method(SimpleNamespace()) == "curve_growth"


# ======================================================================
# Window
# ======================================================================

@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


class TestWindow:
    def test_lists_methods_and_applies_choice(self, tk_root, monkeypatch):
        monkeypatch.setattr(ifh, "save_user_settings", lambda settings: None)
        islat = _islat("curve_growth_no_overlap")
        messages = []
        islat.GUI = SimpleNamespace(data_field=SimpleNamespace(insert_text=messages.append))

        window = isw.IntensitySettingsWindow.open(tk_root, islat)
        assert isw.IntensitySettingsWindow.open(tk_root, islat) is window  # singleton
        radios = [w for w in _walk(window._win) if w.winfo_class() == "TRadiobutton"]
        assert [r.cget("value") for r in radios] == METHODS
        assert window._method_var.get() == "curve_growth_no_overlap"

        window._method_buttons["radex"].invoke()
        next(w for w in _walk(window._win) if w.winfo_class() == "TButton" and w.cget("text") == "Apply").invoke()

        assert islat.molecules_dict.global_intensity_calculation_method == "radex"
        assert messages and "RADEX" in messages[-1]
        assert id(islat) not in isw.IntensitySettingsWindow._instances  # closed

    def test_cancel_changes_nothing(self, tk_root, monkeypatch):
        monkeypatch.setattr(ifh, "save_user_settings", lambda settings: pytest.fail("saved on cancel"))
        islat = _islat()
        window = isw.IntensitySettingsWindow.open(tk_root, islat)
        window._method_buttons["radex"].invoke()
        next(w for w in _walk(window._win) if w.winfo_class() == "TButton" and w.cget("text") == "Cancel").invoke()
        assert islat.molecules_dict.global_intensity_calculation_method == "curve_growth"
        assert id(islat) not in isw.IntensitySettingsWindow._instances
