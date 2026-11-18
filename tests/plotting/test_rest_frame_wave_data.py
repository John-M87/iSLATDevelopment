# -*- coding: utf-8 -*-
"""Three Panel refreshes keep ``islat.wave_data`` when the rest-frame wavelengths are unchanged.

With a non-zero stellar RV, ``MoleculeDict.apply_stellar_rv`` returns a new
array on every call.  The other views compare ``islat.wave_data`` by identity
to tell whether their last frame is still current, so replacing it with an
equal array made each of them redraw when shown again.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from types import SimpleNamespace

from iSLAT.Modules.Plotting.ThreePanelView import _rest_frame_wave_data

from tests.plotting.test_view_reactivation import _round_trip, fs  # noqa: F401  (fixture)

SPEED_OF_LIGHT_KMS = 299792.458


class _StellarRV:
    """Stands in for MoleculeDict: the same formula as ``apply_stellar_rv``."""

    def __init__(self, rv):
        self.rv = rv

    def apply_stellar_rv(self, wave):
        if self.rv == 0:
            return wave
        return wave - (wave / SPEED_OF_LIGHT_KMS * self.rv)


def _islat(wave_obs, rv):
    mol_dict = _StellarRV(rv)
    islat = SimpleNamespace(wave_data_original=wave_obs, wave_data=None)
    islat.wave_data = _rest_frame_wave_data(islat, mol_dict, wave_obs)
    return islat, mol_dict


class TestRestFrameWaveData:
    def test_equal_values_keep_the_current_array(self):
        islat, mol_dict = _islat(np.linspace(5.0, 28.0, 500), rv=17.0)
        first = islat.wave_data
        again = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)
        assert again is first

    def test_changed_rv_gives_the_new_values(self):
        islat, mol_dict = _islat(np.linspace(5.0, 28.0, 500), rv=17.0)
        first = islat.wave_data
        mol_dict.rv = 20.0
        again = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)
        assert again is not first
        np.testing.assert_array_equal(again, mol_dict.apply_stellar_rv(islat.wave_data_original))

    def test_new_observed_data_gives_the_new_values(self):
        islat, mol_dict = _islat(np.linspace(5.0, 28.0, 500), rv=17.0)
        new_obs = np.linspace(5.0, 28.0, 400)
        again = _rest_frame_wave_data(islat, mol_dict, new_obs)
        np.testing.assert_array_equal(again, mol_dict.apply_stellar_rv(new_obs))

    def test_zero_rv_returns_the_observed_array(self):
        wave_obs = np.linspace(5.0, 28.0, 500)
        islat, mol_dict = _islat(wave_obs, rv=0.0)
        assert islat.wave_data is wave_obs
        assert _rest_frame_wave_data(islat, mol_dict, wave_obs) is wave_obs

    def test_no_current_array(self):
        wave_obs = np.linspace(5.0, 28.0, 500)
        islat = SimpleNamespace(wave_data_original=wave_obs)
        out = _rest_frame_wave_data(islat, _StellarRV(17.0), wave_obs)
        np.testing.assert_array_equal(out, _StellarRV(17.0).apply_stellar_rv(wave_obs))


class TestOtherViewsAcrossAThreePanelRefresh:
    def test_view_is_not_redrawn_when_the_refresh_leaves_values_unchanged(self, fs):
        view, _pm, islat, _ = fs
        mol_dict = _StellarRV(17.0)
        islat.wave_data = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)

        def three_panel_refresh():
            islat.wave_data = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)

        assert _round_trip(view, three_panel_refresh) == (False, 0)

    def test_view_redraws_after_a_stellar_rv_change(self, fs):
        view, _pm, islat, _ = fs
        mol_dict = _StellarRV(17.0)
        islat.wave_data = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)

        def rv_change():
            mol_dict.rv = 25.0
            islat.wave_data = _rest_frame_wave_data(islat, mol_dict, islat.wave_data_original)

        synced, _ = _round_trip(view, rv_change)
        assert synced
