# -*- coding: utf-8 -*-
"""Partition function extension: data layer, molecule API and file output."""

import numpy as np
import pytest

from iSLAT.Modules.DataProcessing.PartitionFunctionExtender import (
    EXTENSION_MODES,
    MAX_NEW_POINTS,
    ExtensionMode,
    extend_partition_function,
    register_extension_mode,
    suggest_step,
)

T = np.array([100.0, 300.0, 500.0, 1000.0, 2000.0])
Q = np.array([10.0, 150.0, 500.0, 2000.0, 8000.0])


def _lines():
    import iSLAT.Constants as c_mod
    return [
        {'nr': i + 1, 'lev_up': f'0|{2*i+2}', 'lev_low': f'0|{2*i+1}',
         'lam': lam, 'freq': c_mod.SPEED_OF_LIGHT_MICRONS / lam, 'a_stein': 0.02,
         'e_up': 3000.0 - i * 200, 'e_low': 2000.0 - i * 200,
         'g_up': 2 * i + 3, 'g_low': 2 * i + 1}
        for i, lam in enumerate([6.0, 10.0, 15.0])
    ]


def _line_list():
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    ll = MoleculeLineList(molecule_id='H2O', lines_data=_lines())
    ll.partition_function = ll._partition_type(t=T.copy(), q=Q.copy())
    ll._molar_mass = 18.015
    return ll


def _molecule(temp=500.0):
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    mol = Molecule(
        name='H2O', displaylabel='H2O', filepath=None, color='#0000FF',
        is_visible=True, temp=temp, radius=1.0, n_mol=1e18, distance=140.0,
        fwhm=130.0, wavelength_range=(5.0, 20.0),
        initial_molecule_parameters={'t_kin': temp, 'scale_exponent': 18.0,
                                     'scale_number': 1.0, 'radius_init': 1.0},
    )
    mol.lines = _line_list()
    mol.lines.wavelength_range = (5.0, 20.0)
    return mol


class TestExtend:
    def test_constant_fill_both_sides(self):
        ext = extend_partition_function(T, Q, t_low=50.0, t_high=2300.0, step=100.0)
        assert ext.t[:2] == pytest.approx([50.0, 100.0])
        assert ext.t[-4:] == pytest.approx([2000.0, 2100.0, 2200.0, 2300.0])
        assert ext.q[0] == Q[0]
        assert np.all(ext.q[-3:] == Q[-1])
        assert ext.n_added_low == 1 and ext.n_added_high == 3
        assert ext.is_extended.sum() == ext.n_added
        # original rows untouched and in place
        assert ext.t[~ext.is_extended] == pytest.approx(T)
        assert ext.q[~ext.is_extended] == pytest.approx(Q)

    def test_partial_step_still_reaches_the_bound(self):
        ext = extend_partition_function(T, Q, t_high=2250.0, step=100.0)
        assert ext.t[-3:] == pytest.approx([2100.0, 2200.0, 2250.0])
        assert np.all(np.diff(ext.t) > 0)

    def test_low_side_spacing_counts_from_the_edge(self):
        ext = extend_partition_function(T, Q, t_low=1.0, step=25.0)
        added = ext.t[ext.is_extended]
        assert added[0] == 1.0
        assert added[1:] == pytest.approx(np.arange(25.0, 100.0, 25.0))

    def test_fractional_step_is_exact(self):
        ext = extend_partition_function(T, Q, t_high=2001.0, step=0.25)
        assert ext.t[-4:] == pytest.approx([2000.25, 2000.5, 2000.75, 2001.0])

    def test_bounds_inside_the_table_change_nothing(self):
        ext = extend_partition_function(T, Q, t_low=200.0, t_high=1500.0, step=10.0)
        assert ext.n_added == 0
        assert ext.t == pytest.approx(T)
        assert len(ext.notes) == 2

    def test_unsorted_input_is_sorted(self):
        ext = extend_partition_function(T[::-1], Q[::-1], t_high=2100.0, step=100.0)
        assert ext.t[:-1] == pytest.approx(T)
        assert ext.q[-1] == Q[-1]

    @pytest.mark.parametrize("kwargs, match", [
        (dict(step=0.0), "step"),
        (dict(step=-1.0), "step"),
        (dict(step=float("nan")), "step"),
        (dict(t_low=0.0), "above 0"),
        (dict(t_low=-5.0), "above 0"),
        (dict(t_low=500.0, t_high=400.0), "below the highest"),
        (dict(mode="nope"), "Unknown extension mode"),
        (dict(t_high=float("inf")), "finite"),
    ])
    def test_invalid_input(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            extend_partition_function(T, Q, **kwargs)

    def test_point_cap(self):
        with pytest.raises(ValueError, match="larger step"):
            extend_partition_function(T, Q, t_high=2000.0 + 10 * MAX_NEW_POINTS, step=1.0)

    def test_empty_table_rejected(self):
        with pytest.raises(ValueError):
            extend_partition_function([], [], t_high=10.0)

    def test_suggest_step(self):
        assert suggest_step([1.0, 2.0, 3.0, 4.0]) == 1.0
        assert suggest_step([5.0]) == 1.0

    def test_registered_modes_are_used(self):
        mode = ExtensionMode("test_double", "Double", "test",
                             lambda t, q, new_t, side: np.full(new_t.shape, 2 * q[-1]))
        register_extension_mode(mode)
        try:
            ext = extend_partition_function(T, Q, t_high=2100.0, step=100.0, mode="test_double")
            assert ext.q[-1] == 2 * Q[-1]
        finally:
            EXTENSION_MODES.pop("test_double", None)

    def test_constant_is_the_only_builtin_mode(self):
        assert list(EXTENSION_MODES) == ["constant"]
        assert EXTENSION_MODES["constant"].label == "Constant Q(T)"


class TestLineListAndMolecule:
    def test_line_list_set_partition_sorts(self):
        ll = _line_list()
        ll.set_partition_function([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
        assert ll.partition.t == pytest.approx([1.0, 2.0, 3.0])
        assert ll.partition.q == pytest.approx([10.0, 20.0, 30.0])

    @pytest.mark.parametrize("t, q", [([], []), ([1.0, 2.0], [1.0]), ([1.0], [np.nan])])
    def test_line_list_set_partition_rejects_bad_tables(self, t, q):
        with pytest.raises(ValueError):
            _line_list().set_partition_function(t, q)

    def test_copy_is_independent_for_partition(self):
        ll = _line_list()
        ll.wavelength_range = (5.0, 12.0)
        cp = ll.copy()
        cp.set_partition_function([1.0, 2.0], [1.0, 2.0])
        assert ll.partition.t == pytest.approx(T)
        assert cp.wavelength_range == (5.0, 12.0)
        assert len(cp) == len(ll) == 2

    def test_extension_lets_the_model_run_hotter(self):
        mol = _molecule(temp=1000.0)
        mol.calculate_intensity()
        mol.temp = 3000.0
        with pytest.raises(ValueError, match="outside partition function range"):
            mol.calculate_intensity()

        ext = extend_partition_function(T, Q, t_high=4000.0, step=100.0)
        mol.set_partition_function(ext.t, ext.q)
        assert mol.partition_function.t[-1] == 4000.0
        mol.calculate_intensity()
        assert mol.intensity.intensity is not None

    def test_set_partition_invalidates_cached_intensity(self):
        mol = _molecule(temp=1000.0)
        mol.calculate_intensity()
        before = np.array(mol.intensity.intensity, copy=True)
        mol.set_partition_function(T, Q * 2.0)
        mol.calculate_intensity()
        # Q doubles -> level populations (and thin-line intensities) halve
        assert np.all(np.asarray(mol.intensity.intensity) < before)

    def test_set_partition_notifies(self):
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        mol = _molecule()
        seen = []

        def cb(name, param, old, new):
            seen.append((name, param))

        Molecule.add_molecule_parameter_change_callback(cb)
        try:
            mol.set_partition_function(np.append(T, 2100.0), np.append(Q, Q[-1]))
        finally:
            Molecule.remove_molecule_parameter_change_callback(cb)
        assert ("H2O", "partition_function") in seen


class TestFiles:
    def test_writer_keeps_one_decimal_for_whole_temperatures(self):
        import io
        import pandas as pd
        from iSLAT.Modules.FileHandling.partition_function_writer import write_partition_function
        fh = io.StringIO()
        write_partition_function(fh, pd.DataFrame({'temp': [1.0, 2.0], 'q': [1.5, 2.5]}))
        rows = fh.getvalue().splitlines()[4:]
        assert rows == ["   1.0         1.500000", "   2.0         2.500000"]

    def test_writer_keeps_fractional_temperatures(self):
        import io
        import pandas as pd
        from iSLAT.Modules.FileHandling.partition_function_writer import write_partition_function
        fh = io.StringIO()
        write_partition_function(fh, pd.DataFrame({'temp': [1.0, 1.25], 'q': [1.0, 1.0]}))
        temps = [float(r.split()[0]) for r in fh.getvalue().splitlines()[4:]]
        assert temps == [1.0, 1.25]

    def test_par_round_trip(self, tmp_path):
        from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
        from iSLAT.Modules.GUI.Widgets.PartitionFunctionWindow import write_partition_file
        mol = _molecule()
        ext = extend_partition_function(T, Q, t_low=50.0, t_high=2500.0, step=250.0)
        out = write_partition_file(mol, ext, tmp_path / "ext.par")
        loaded = MoleculeLineList.from_file(out)
        assert loaded.partition.t == pytest.approx(ext.t)
        assert loaded.partition.q == pytest.approx(ext.q)
        # the whole line list is written, not just the model wavelength range
        assert len(loaded) == 3

    def test_csv_table(self, tmp_path):
        import pandas as pd
        from iSLAT.Modules.GUI.Widgets.PartitionFunctionWindow import write_partition_file
        ext = extend_partition_function(T, Q, t_high=2200.0, step=100.0)
        out = write_partition_file(_molecule(), ext, tmp_path / "q.csv")
        df = pd.read_csv(out)
        assert list(df.columns) == ["temp", "q"]
        assert df["temp"].to_numpy() == pytest.approx(ext.t)
        assert df["q"].to_numpy() == pytest.approx(ext.q)
