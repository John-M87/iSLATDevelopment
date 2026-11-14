# -*- coding: utf-8 -*-
"""The summed model follows every change that alters a molecule's flux.

``MoleculeDict.get_summed_flux`` caches the sum and validates the entry
against each molecule's ``get_parameter_hash('full')``.  That hash used to be
stored once at construction, and the parameter-change handler only dropped
the cache for a fixed list of parameters, so after a Keplerian FWHM,
instrumental profile, pixel resolution or intensity method change the plots
kept drawing the old summed model, although each molecule's own trace was
updated.

The hash is now computed from the current parameters, includes a counter of
line-data replacements (a new line list, a filtered line list applied from
the Line List Filter window, a replaced partition function), and the handler
drops the cache for every parameter in ``Molecule.FLUX_AFFECTING_PARAMS``.

Uses real CO molecules (HITRAN line list) over a narrow range; line-list
cache writes are stubbed out.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

ROOT = Path(__file__).resolve().parents[1]
CO_PAR = ROOT / "iSLAT" / "DATAFILES" / "HITRANdata" / "data_Hitran_CO.par"
WAVE = np.linspace(4.92, 5.18, 400)

line_list_mod = importlib.import_module("iSLAT.Modules.DataTypes.MoleculeLineList")


def _row(name, temp):
    return {"Molecule Name": name, "File Path": str(CO_PAR), "Molecule Label": name, "Temp": temp,
            "Rad": 0.5, "N_Mol": 1e17, "Color": "#ff0000", "Vis": True, "Dist": 140.0,
            "StellarRV": 0.0, "FWHM": 130.0, "KeplerianFWHM": 0.0, "InstrumentalProfile": "constant",
            "Broad": 2.5, "RV Shift": 0.0}


@pytest.fixture
def md(monkeypatch):
    if not CO_PAR.exists():
        pytest.skip("CO HITRAN file not available")
    monkeypatch.setattr(line_list_mod.MoleculeLineList, "_save_to_cache", lambda self, path: True)
    md = MoleculeDict(global_wavelength_range=(4.9, 5.2), global_model_pixel_res=0.001,
                      global_distance=140.0)
    md.load_molecules([_row("CO", 600.0), _row("COb", 900.0)], {},
                      update_global_parameters=True, strategy="sequential")
    yield md
    Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


def _fresh_sum(md):
    """The sum computed from scratch (summed cache bypassed)."""
    md._summed_flux_cache.clear()
    return md.get_summed_flux(WAVE)


CHANGES = [
    ("keplerian_fwhm", 40.0),
    ("instrumental_profile_key", "miri_mrs"),
    ("model_pixel_res", 0.0005),
    ("intensity_calculation_method", "curve_growth_no_overlap"),
    ("temp", 750.0),
]


class TestSummedModelFollowsChanges:
    @pytest.mark.parametrize("param, value", CHANGES, ids=[c[0] for c in CHANGES])
    def test_sum_after_a_change_equals_a_fresh_sum(self, md, param, value):
        _w0, before = md.get_summed_flux(WAVE)
        setattr(md["CO"], param, value)
        w, f = md.get_summed_flux(WAVE)                 # what the plots draw
        w_fresh, f_fresh = _fresh_sum(md)
        np.testing.assert_array_equal(w, w_fresh)
        np.testing.assert_array_equal(f, f_fresh)
        assert before.shape != f.shape or not np.array_equal(before, f)

    def test_each_keplerian_step_is_drawn(self, md):
        md.get_summed_flux(WAVE)
        for kep in (20.0, 60.0, 0.0, 35.0):
            md["COb"].keplerian_fwhm = kep
            _w, f = md.get_summed_flux(WAVE)
            np.testing.assert_array_equal(f, _fresh_sum(md)[1])

    def test_replaced_partition_function(self, md):
        md.get_summed_flux(WAVE)
        pf = md["CO"].partition_function
        md["CO"].set_partition_function(np.asarray(pf.t), np.asarray(pf.q) * 2.0)
        _w, f = md.get_summed_flux(WAVE)
        np.testing.assert_array_equal(f, _fresh_sum(md)[1])

    def test_filtered_line_list_applied_from_the_filter_window(self, md):
        from iSLAT.Modules.DataProcessing.LineListMaker import LineListMaker
        from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
        _w0, before = md.get_summed_flux(WAVE)
        co = md["CO"]
        co._ensure_lines_loaded()
        # As "Apply to molecule" does: the lines that pass the window's filter.
        low_energy = LineListMaker(co.lines).filter_eup(max_val=3000.0).to_linelist()
        LineListFilterWindow._assign_line_list(co, low_energy)
        _w, f = md.get_summed_flux(WAVE)
        np.testing.assert_array_equal(f, _fresh_sum(md)[1])
        assert not np.array_equal(f, before)


class TestParameterHash:
    def _molecule(self):
        return Molecule(name="X", displaylabel="X", filepath=None, temp=500.0, radius=1.0,
                        n_mol=1e18, distance=140.0, fwhm=130.0,
                        initial_molecule_parameters={"t_kin": 500.0, "scale_exponent": 18.0,
                                                     "scale_number": 1.0, "radius_init": 1.0})

    @pytest.mark.parametrize("param, value", CHANGES, ids=[c[0] for c in CHANGES])
    def test_full_hash_follows_the_parameters(self, param, value):
        mol = self._molecule()
        before = mol.get_parameter_hash("full")
        setattr(mol, param, value)
        assert mol.get_parameter_hash("full") != before

    def test_display_only_change_keeps_the_hash(self):
        mol = self._molecule()
        before = mol.get_parameter_hash("full")
        mol.is_bold = True
        assert mol.get_parameter_hash("full") == before

    def test_new_line_list_changes_the_hash(self):
        mol = self._molecule()
        before = mol.get_parameter_hash("full")
        mol.change_line_list("not_read_until_used.par")
        assert mol.get_parameter_hash("full") != before


class TestHandlerDropsTheCache:
    @pytest.fixture
    def empty_md(self):
        md = MoleculeDict()
        yield md
        Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)

    @pytest.mark.parametrize("param", sorted(Molecule.FLUX_AFFECTING_PARAMS) + ["partition_function"])
    def test_flux_affecting_parameters(self, empty_md, param):
        empty_md._summed_flux_cache[1] = (np.ones(3), np.ones(3), 0)
        empty_md._on_molecule_parameter_changed("CO", param, 1.0, 2.0)
        assert empty_md._summed_flux_cache == {}

    @pytest.mark.parametrize("param", ["is_visible", "is_bold", "color"])
    def test_display_parameters(self, empty_md, param):
        empty_md._summed_flux_cache[1] = (np.ones(3), np.ones(3), 0)
        empty_md._on_molecule_parameter_changed("CO", param, 1.0, 2.0)
        assert 1 in empty_md._summed_flux_cache
