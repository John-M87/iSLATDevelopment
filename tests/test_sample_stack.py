# -*- coding: utf-8 -*-
"""SampleStackBuilder: the sample's spectra, each with the parameter set the Sample Manager gives it.

Uses real CO molecules (HITRAN line list) over a narrow range, so the
parameter sets are built exactly as in the GUI.
"""

import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
from iSLAT.Modules.DataProcessing.SampleStack import (
    SampleStackBuilder, molecule_save_row, sample_signature, same_path,
)

ROOT = Path(__file__).resolve().parents[1]
CO_PAR = ROOT / "iSLAT" / "DATAFILES" / "HITRANdata" / "data_Hitran_CO.par"

pytestmark = pytest.mark.skipif(not CO_PAR.exists(), reason="CO HITRAN file not available")

HEADER = ["Molecule Name", "File Path", "Molecule Label", "Temp", "Rad", "N_Mol", "Color", "Vis",
          "Dist", "StellarRV", "FWHM", "KeplerianFWHM", "InstrumentalProfile", "Broad", "RV Shift"]
WAVE = np.linspace(4.9, 5.2, 300)


def _row(temp=600.0, dist=140.0, rv=0.0, color="#ff0000"):
    return {"Molecule Name": "CO", "File Path": str(CO_PAR), "Molecule Label": "CO", "Temp": temp,
            "Rad": 0.5, "N_Mol": 1e17, "Color": color, "Vis": True, "Dist": dist, "StellarRV": rv,
            "FWHM": 130.0, "KeplerianFWHM": 0.0, "InstrumentalProfile": "constant", "Broad": 2.5,
            "RV Shift": 0.0}


def _write_spectrum(path, flux_level):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["wave", "flux", "err"])
        for x in WAVE:
            writer.writerow([x, flux_level, 0.001])


def _write_params(path, **kw):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerow(_row(**kw))


def _release(md):
    Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


@pytest.fixture
def live():
    md = MoleculeDict(global_wavelength_range=(4.9, 5.2), global_model_pixel_res=0.001,
                      global_distance=140.0)
    md.load_molecules([_row()], {}, update_global_parameters=True, strategy="sequential")
    yield md
    _release(md)


@pytest.fixture
def setup(tmp_path, live):
    paths = [str(tmp_path / f"spec{i}.csv") for i in range(3)]
    for i, p in enumerate(paths):
        _write_spectrum(p, 0.01 + 0.001 * i)
    islat = SimpleNamespace(
        molecules_dict=live, loaded_spectrum_file=paths[0], wave_data_original=WAVE.copy(),
        flux_data=np.full_like(WAVE, 0.01), err_data=np.full_like(WAVE, 0.001),
        sample_spectra=list(paths), sample_spectra_params={}, sample_spectra_overrides={},
        user_settings={}, initial_molecule_parameters={},
    )
    builder = SampleStackBuilder()
    yield islat, builder, tmp_path
    builder.release()


def _peak(molecules):
    _w, flux = molecules.get_summed_flux(WAVE, visible_only=True)
    return float(np.max(flux))


class TestEntries:
    def test_order_names_and_active_entry(self, setup):
        islat, builder, _ = setup
        entries = builder.build(islat)
        assert [e.name for e in entries] == ["spec0", "spec1", "spec2"]
        assert [e.is_active for e in entries] == [True, False, False]
        assert entries[0].molecules is islat.molecules_dict
        assert entries[0].parameter_source == "active"
        assert builder.warnings == []

    def test_entries_without_parameter_file_share_the_live_molecules(self, setup):
        islat, builder, _ = setup
        entries = builder.build(islat)
        assert all(e.molecules is islat.molecules_dict for e in entries)
        assert [e.parameter_source for e in entries[1:]] == ["shared", "shared"]
        assert np.allclose(entries[2].flux, 0.012)

    def test_loaded_spectrum_outside_the_sample_comes_first(self, setup, tmp_path):
        islat, builder, _ = setup
        other = str(tmp_path / "other.csv")
        _write_spectrum(other, 0.02)
        islat.loaded_spectrum_file = other
        entries = builder.build(islat)
        assert entries[0].is_active and entries[0].name == "other"
        assert [e.name for e in entries[1:]] == ["spec0", "spec1", "spec2"]

    def test_unreadable_spectrum_is_skipped_with_a_warning(self, setup, tmp_path):
        islat, builder, _ = setup
        islat.sample_spectra.append(str(tmp_path / "missing.csv"))
        entries = builder.build(islat)
        assert [e.name for e in entries] == ["spec0", "spec1", "spec2"]
        assert len(builder.warnings) == 1 and "missing.csv" in builder.warnings[0]

    def test_data_scalars_apply_to_other_spectra(self, setup):
        islat, builder, _ = setup
        islat.user_settings = {"wave_data_scalar": 2.0, "flux_data_scalar": 10.0}
        entry = builder.build(islat)[1]
        assert np.isclose(entry.wave_obs[0], 9.8)
        assert np.allclose(entry.flux, 0.11)


class TestParameterSets:
    def test_parameter_file_gives_its_own_set(self, setup):
        islat, builder, tmp = setup
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0, dist=100.0, rv=30.0)
        islat.sample_spectra_params = {islat.sample_spectra[2]: params}
        entry = builder.build(islat)[2]
        assert entry.parameter_source == "file" and entry.param_file == params
        md = entry.molecules
        assert md is not islat.molecules_dict
        assert md["CO"].temp == 900.0
        assert md.global_distance == 100.0 and md.global_stellar_rv == 30.0
        # Data moved to the rest frame with the file's stellar RV
        assert np.allclose(entry.wave, md.apply_stellar_rv(entry.wave_obs))
        assert entry.wave[0] < entry.wave_obs[0]
        # The model range follows the live molecules
        assert tuple(md.global_wavelength_range) == (4.9, 5.2)

    def test_overrides_win_over_the_parameter_file(self, setup):
        islat, builder, tmp = setup
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0, dist=100.0)
        path = islat.sample_spectra[2]
        islat.sample_spectra_params = {path: params}
        islat.sample_spectra_overrides = {path: {"distance": 250.0}}
        md = builder.build(islat)[2].molecules
        assert md.global_distance == 250.0
        assert md["CO"].distance == 250.0
        assert md["CO"].temp == 900.0

    def test_override_without_parameter_file_copies_the_current_parameters(self, setup):
        islat, builder, _ = setup
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 280.0}}
        entry = builder.build(islat)[1]
        assert entry.parameter_source == "current"
        md = entry.molecules
        assert md is not islat.molecules_dict
        assert md["CO"].temp == islat.molecules_dict["CO"].temp
        # Flux scales with 1 / distance^2
        assert _peak(islat.molecules_dict) / _peak(md) == pytest.approx(4.0)

    def test_override_equal_to_the_current_value_shares_the_live_set(self, setup):
        islat, builder, _ = setup
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 140.0}}
        entry = builder.build(islat)[1]
        assert entry.parameter_source == "shared" and entry.molecules is islat.molecules_dict

    def test_copy_follows_live_edits_without_being_rebuilt(self, setup):
        islat, builder, _ = setup
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 280.0}}
        first = builder.build(islat)[1].molecules
        islat.molecules_dict["CO"].temp = 750.0
        second = builder.build(islat)[1].molecules
        assert second is first
        assert second["CO"].temp == 750.0
        assert second.global_distance == 280.0

    def test_parameter_file_set_ignores_live_edits_but_follows_visibility_and_color(self, setup):
        islat, builder, tmp = setup
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0, color="#00ff00")
        islat.sample_spectra_params = {islat.sample_spectra[2]: params}
        builder.build(islat)
        live_co = islat.molecules_dict["CO"]
        live_co.temp = 750.0
        live_co.is_visible = False
        md = builder.build(islat)[2].molecules
        assert md["CO"].temp == 900.0
        assert md["CO"].is_visible is False
        assert md["CO"].color == live_co.color

    def test_changed_parameter_file_is_reloaded(self, setup):
        islat, builder, tmp = setup
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0)
        islat.sample_spectra_params = {islat.sample_spectra[2]: params}
        first = builder.build(islat)[2].molecules
        _write_params(params, temp=1100.0)
        st = os.stat(params)
        os.utime(params, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        second = builder.build(islat)[2].molecules
        assert second is not first
        assert second["CO"].temp == 1100.0


class TestNotificationsAndRelease:
    def test_building_sends_no_molecule_parameter_notifications(self, setup):
        islat, builder, tmp = setup
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0, dist=100.0)
        islat.sample_spectra_params = {islat.sample_spectra[2]: params}
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 280.0, "stellar_rv": 12.0}}
        fired = []
        callback = lambda *a: fired.append(a)
        Molecule.add_molecule_parameter_change_callback(callback)
        try:
            builder.build(islat)
            builder.build(islat)
        finally:
            Molecule.remove_molecule_parameter_change_callback(callback)
        assert fired == []

    def test_release_and_removed_entries_unregister_their_callbacks(self, setup):
        islat, builder, tmp = setup
        before = len(Molecule._class_callbacks)
        params = str(tmp / "p.csv")
        _write_params(params, temp=900.0)
        islat.sample_spectra_params = {islat.sample_spectra[2]: params}
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 280.0}}
        builder.build(islat)
        assert len(Molecule._class_callbacks) == before + 2
        islat.sample_spectra = islat.sample_spectra[:2]          # entry with the file removed
        builder.build(islat)
        assert len(Molecule._class_callbacks) == before + 1
        builder.release()
        assert len(Molecule._class_callbacks) == before


class TestHelpers:
    def test_sample_signature(self, setup):
        islat, _builder, tmp = setup
        base = sample_signature(islat)
        assert sample_signature(islat) == base
        # The loaded spectrum's own overrides are already in the live molecules
        islat.sample_spectra_overrides = {islat.sample_spectra[0]: {"distance": 200.0}}
        assert sample_signature(islat) == base
        islat.sample_spectra_overrides = {islat.sample_spectra[1]: {"distance": 200.0}}
        assert sample_signature(islat) != base
        islat.sample_spectra_overrides = {}
        islat.loaded_spectrum_file = islat.sample_spectra[1]
        assert sample_signature(islat) != base

    def test_same_path(self, tmp_path):
        p = str(tmp_path / "a.csv")
        assert same_path(p, os.path.join(str(tmp_path), ".", "a.csv"))
        assert not same_path(p, None)

    def test_molecule_save_row_round_trips(self, live):
        row = molecule_save_row(live["CO"], 300.0, 5.0)
        assert row["Molecule Name"] == "CO" and row["Dist"] == 300.0 and row["StellarRV"] == 5.0
        assert set(HEADER) <= set(row)
