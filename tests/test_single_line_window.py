# -*- coding: utf-8 -*-
"""Regression: a model window holding exactly one line.

``Intensity`` used ``squeeze()`` to drop the single-condition axis, which also
dropped the length-1 line axis and left 0-d arrays, so ``Spectrum`` raised
``TypeError: len() of unsized object``.
"""

import numpy as np
import pytest

import iSLAT.Constants as c

_T = np.array([100.0, 300.0, 1000.0, 3000.0])
_Q = np.array([10.0, 50.0, 400.0, 3000.0])


def _line_list(lams):
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    lines = [
        {'nr': i + 1, 'lev_up': f'0|{2*i+2}', 'lev_low': f'0|{2*i+1}',
         'lam': lam, 'freq': c.SPEED_OF_LIGHT_MICRONS / lam, 'a_stein': 0.02,
         'e_up': 3000.0 - i * 200, 'e_low': 2000.0 - i * 200,
         'g_up': 2 * i + 3, 'g_low': 2 * i + 1}
        for i, lam in enumerate(lams)
    ]
    ll = MoleculeLineList(molecule_id='H2O', lines_data=lines)
    ll.partition_function = ll._partition_type(t=_T.copy(), q=_Q.copy())
    ll._molar_mass = 18.015
    return ll


def _molecule(wavelength_range):
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    mol = Molecule(
        name='H2O', displaylabel='H2O', filepath=None, color='#0000FF',
        is_visible=True, temp=800.0, radius=1.0, n_mol=1e18, distance=140.0,
        fwhm=130.0, wavelength_range=wavelength_range,
        initial_molecule_parameters={'t_kin': 800.0, 'scale_exponent': 18.0,
                                     'scale_number': 1.0, 'radius_init': 1.0},
    )
    mol.lines = _line_list([6.0, 10.0, 15.0])
    mol.lines.wavelength_range = wavelength_range
    return mol


@pytest.mark.parametrize("method", ["curve_growth", "curve_growth_no_overlap", "radex"])
def test_intensity_keeps_a_1d_line_axis_for_one_line(method):
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    inten = Intensity(_line_list([10.0]))
    inten.calc_intensity(t_kin=800.0, n_mol=1e18, dv=1.0, method=method)
    assert inten.intensity.shape == (1,)
    assert inten.tau.shape == (1,)
    assert np.all(np.isfinite(inten.intensity)) and inten.intensity[0] > 0


def test_scalar_shapes_unchanged_for_several_lines():
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    inten = Intensity(_line_list([6.0, 10.0, 15.0]))
    inten.calc_intensity(t_kin=800.0, n_mol=1e18, dv=1.0)
    assert inten.intensity.shape == (3,)
    assert inten.tau.shape == (3,)


def test_one_line_window_matches_the_same_line_in_a_wider_window():
    one = _molecule((9.0, 11.0))
    assert len(one.line_list) == 1
    wave = np.linspace(9.5, 10.5, 2001)
    flux_one = one.get_flux(wave)
    assert np.nanmax(flux_one) > 0

    wide = _molecule((5.0, 20.0))
    wide.calculate_intensity()
    idx = int(np.argmin(np.abs(wide.line_list.get_wavelengths() - 10.0)))
    one.calculate_intensity()
    assert one.intensity.intensity[0] == pytest.approx(wide.intensity.intensity[idx])


def test_one_line_window_line_table():
    mol = _molecule((9.0, 11.0))
    mol.calculate_intensity()
    table = mol.intensity.get_table
    df = table() if callable(table) else table
    assert len(df) == 1


def test_slab_fit_pipeline_with_one_line(tmp_path):
    from unittest.mock import MagicMock
    from iSLAT.Modules.DataProcessing.Slabfit import SlabModel
    slab = SlabModel(output_folder=str(tmp_path), source=_line_list([10.0]),
                     input_file=str(tmp_path / 'nonexistent.csv'),
                     min_wavelength=9.0, max_wavelength=11.0)
    slab.chi2_evaluator = MagicMock()
    slab.chi2_evaluator.chi2_total = 1.0
    spectrum = slab._pipeline.evaluate({'temp': np.float64(800.0)}, slab._config)
    assert np.nanmax(spectrum.flux) > 0
