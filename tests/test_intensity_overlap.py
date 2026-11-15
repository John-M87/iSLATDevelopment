# -*- coding: utf-8 -*-
"""Overlapping lines in ``Intensity`` (``method="curve_growth"``).

Every line is integrated on its own quadrature window with the opacity of
all neighbours within 7 Doppler parameters summed in
(``iSLAT.Modules.DataTypes._overlap_perline``).  These tests compare that
against the float64 brute-force reference in ``tests/overlap_reference.py``.

Covers:
- the kernel on its own (``build_tables`` / ``perline_fint``)
- ``calc_intensity`` against the reference
- the sum rule of a blend, and the optically thin limit
- line lists without any overlap (the isolated-line fast path)
- ``prepare_overlap_structure`` / ``calc_intensity_batch_fast``
- wavelength range changes
- the thread pool (``ISLAT_INTENSITY_THREADS``)

Each check runs on a synthetic line list (a line-dense band, compact
blends and isolated lines) and on real HITRAN line lists between 10 and
18 microns.  The HITRAN files are not part of the repository, so their
cases are skipped where a file is missing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

import iSLAT.Constants as c
import iSLAT.Modules.DataTypes._overlap_perline as overlap_perline
from iSLAT.Modules.DataTypes.Intensity import Intensity
from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList

from tests.overlap_reference import (
    C_KMS,
    TWO_SQRT_LN2,
    blend_equivalent_width,
    gauss_legendre_nodes,
    isolated_fint_reference,
    overlap_intensity_reference,
    perline_fint_reference,
)

HITRAN_DIR = Path(__file__).resolve().parents[1] / "iSLAT" / "DATAFILES" / "HITRANdata"
REAL_FILES = {
    "HCN": "data_Hitran_HCN.par",
    "C2H2": "data_Hitran_C2H2.par",
    "CO2": "data_Hitran_CO2.par",
    "H2O": "H2O_HITEMP_Combined_v0_2p9-2000p0um.par",
}
WAVELENGTH_RANGE = (10.0, 18.0)


# ---------------------------------------------------------------------------
# Line lists
# ---------------------------------------------------------------------------

def _synthetic_lines_data(seed=7):
    """325 lines near 14 microns, in random order.

    * a line-dense band: 160 lines scattered over 500 km/s (one long chain
      of overlapping lines, like a Q-branch);
    * 40 compact blends of 2-6 lines within 14 km/s, 600 km/s apart;
    * 12 lines 1500 km/s apart (no neighbour at any line width used here).
    """
    rng = np.random.default_rng(seed)
    nu0 = c.SPEED_OF_LIGHT_MICRONS / 14.0
    kms = [rng.uniform(0.0, 500.0, 160)]
    for k in range(40):
        kms.append(2000.0 + 600.0 * k + rng.uniform(0.0, 14.0, int(rng.integers(2, 7))))
    kms.append(40000.0 + 1500.0 * np.arange(12))
    freq = nu0 * (1.0 + rng.permutation(np.concatenate(kms)) / C_KMS)

    n = freq.size
    e_low = rng.uniform(0.0, 4000.0, n)
    e_up = e_low + c.PLANCK_CONSTANT * freq / c.BOLTZMANN_CONSTANT
    a_stein = rng.lognormal(0.0, 1.5, n)
    g_up = 2 * rng.integers(1, 20, n) + 1
    return [
        {
            'nr': i + 1,
            'lev_up': f'1|{i}',
            'lev_low': f'0|{i}',
            'lam': c.SPEED_OF_LIGHT_MICRONS / freq[i],
            'freq': freq[i],
            'a_stein': a_stein[i],
            'e_up': e_up[i],
            'e_low': e_low[i],
            'g_up': int(g_up[i]),
            'g_low': int(g_up[i]) - 2,
        }
        for i in range(n)
    ]


def _with_partition(mll):
    mll.partition_function = mll._partition_type(
        t=np.array([100, 200, 300, 500, 1000, 2000], dtype=np.float64),
        q=np.array([10, 50, 150, 500, 2000, 8000], dtype=np.float64),
    )
    mll._molar_mass = 18.015
    return mll


def _make_line_list(name, wavelength_range=WAVELENGTH_RANGE):
    """A new line list: ``"synthetic"`` or a key of ``REAL_FILES``."""
    if name == "synthetic":
        return _with_partition(MoleculeLineList(
            molecule_id='TestMol', lines_data=_synthetic_lines_data(),
            wavelength_range=wavelength_range))
    path = HITRAN_DIR / REAL_FILES[name]
    if not path.exists():
        pytest.skip(f"{path.name} not available")
    return MoleculeLineList(molecule_id=name, filename=str(path),
                            wavelength_range=wavelength_range)


@pytest.fixture(scope="module")
def line_list():
    """``line_list(name)``: line lists shared by the tests that do not modify them."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = _make_line_list(name)
        return cache[name]

    return get


def _isolated_line_list(n_lines=10):
    """Lines 1 micron apart, as in the other Intensity tests: no overlap."""
    rng = np.random.default_rng(42)
    lines_data = []
    for i in range(n_lines):
        lam = 10.0 + i
        lines_data.append({
            'nr': i + 1,
            'lev_up': f'0|{2 * i + 2}',
            'lev_low': f'0|{2 * i + 1}',
            'lam': lam,
            'freq': c.SPEED_OF_LIGHT_MICRONS / lam,
            'a_stein': rng.uniform(1e-3, 5e-2),
            'e_up': 1000.0 + i * 500,
            'e_low': 500.0 + i * 500,
            'g_up': 2 * i + 3,
            'g_low': 2 * i + 1,
        })
    return _with_partition(MoleculeLineList(molecule_id='TestMol', lines_data=lines_data))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc(mll, t_kin, n_mol, dv, method="curve_growth"):
    inten = Intensity(mll)
    inten.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=dv, method=method)
    return inten


def _freq(mll):
    return np.asarray(mll.lines_as_namedtuple.freq, dtype=np.float64)


def _significant(values):
    """Lines within six orders of magnitude of the strongest one."""
    return values > 1e-6 * np.max(values)


def _overlap_fint(mll, t_kin, n_mol, dv):
    """``(tau, fint)`` of every line as ``calc_intensity`` computes them.

    ``intensity = prefactor * fint`` with the same prefactor for
    "curve_growth" and "curve_growth_no_overlap", and the fint of the
    latter is the isolated integral, so the ratio of the two intensities
    gives the fint of the overlap calculation.
    """
    no_overlap = _calc(mll, t_kin, n_mol, dv, "curve_growth_no_overlap")
    overlap = _calc(mll, t_kin, n_mol, dv)
    tau = np.asarray(no_overlap.tau)
    fint = np.zeros_like(tau)
    live = no_overlap.intensity > 0
    fint[live] = (overlap.intensity[live] / no_overlap.intensity[live]
                  * isolated_fint_reference(tau[live]))
    return tau, fint


def _neighbour_tau_bound(inten):
    """Largest sum of the opacities of a line and its tabulated neighbours.

    An upper bound of ``tau_tot`` at any quadrature node of any line.
    """
    tables = inten._perline_tables[1]
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(inten.tau)[tables['order']])])
    return float(np.max(csum[tables['hi']] - csum[tables['lo']]))


def _isolated_blends(freq, dv, min_gap_x=14.0, max_lines=40):
    """Index arrays of the blends that stand alone.

    A blend is a run of 2..``max_lines`` lines (in frequency order) with
    gaps below ``min_gap_x`` Doppler parameters, so every other line is
    more than ``min_gap_x`` away from each of its lines: outside the
    neighbour radius (7), and without any opacity inside the windows of
    the blend (|x| <= 6, exp(-8^2) beyond).
    """
    order = np.argsort(freq, kind="stable")
    fs = freq[order]
    gap_x = C_KMS * np.diff(fs) / fs[:-1] / (dv / TWO_SQRT_LN2)
    cuts = np.flatnonzero(gap_x > min_gap_x) + 1
    starts = np.concatenate([[0], cuts])
    stops = np.concatenate([cuts, [fs.size]])
    return [order[a:b] for a, b in zip(starts, stops) if 2 <= b - a <= max_lines]


# ---------------------------------------------------------------------------
# The kernel on its own
# ---------------------------------------------------------------------------

class TestPerLineKernel:
    """``_overlap_perline.build_tables`` / ``perline_fint`` without ``Intensity``."""

    NU0 = c.SPEED_OF_LIGHT_MICRONS / 14.0

    def _band(self, rng, n, span_kms):
        return self.NU0 * (1.0 + rng.permutation(rng.uniform(0.0, span_kms, n)) / C_KMS)

    def test_slot_width_ladder(self):
        """Four padded widths per octave, never below the neighbour count."""
        counts = np.arange(1, 70)
        widths = overlap_perline._slot_widths(counts)
        assert sorted(set(widths.tolist())) == [
            2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80]
        assert np.all(widths >= np.maximum(counts, 2))
        assert np.all(widths <= np.maximum(counts, 2) * 1.25)

    def test_no_tables_without_neighbours(self):
        """No line with a neighbour (or fewer than two lines): ``None``."""
        isolated = self.NU0 * (1.0 + 1000.0 * np.arange(6) / C_KMS)
        assert overlap_perline.build_tables(isolated, 20.0) is None
        assert overlap_perline.build_tables(isolated[:1], 20.0) is None
        assert overlap_perline.build_tables(np.array([]), 20.0) is None

    def test_neighbour_ranges(self):
        """Every range holds the line itself and exactly the lines within 7 b."""
        rng = np.random.default_rng(1)
        freq = self._band(rng, 200, 800.0)
        dv = 6.0
        tables = overlap_perline.build_tables(freq, dv)
        order, fs, lo, hi = tables['order'], tables['fs'], tables['lo'], tables['hi']
        assert np.array_equal(freq[order], fs) and np.all(np.diff(fs) >= 0)
        radius_kms = 7.0 * dv / TWO_SQRT_LN2
        sep_kms = C_KMS * np.abs(np.log(fs[:, None] / fs[None, :]))
        in_range = sep_kms <= radius_kms
        position = np.arange(fs.size)
        expected = (position[None, :] >= lo[:, None]) & (position[None, :] < hi[:, None])
        assert np.array_equal(in_range, expected)
        # isolated lines and the buckets partition the line list
        seen = np.concatenate([tables['isolated']] + [order[sel] for _, sel in tables['buckets']])
        assert np.array_equal(np.sort(seen), np.arange(freq.size))
        for width, sel in tables['buckets']:
            assert np.all((hi - lo)[sel] <= width) and np.all((hi - lo)[sel] > 1)

    def test_matches_reference_mixed_dv_batch(self):
        """Four conditions with different dv on tables built for the largest."""
        rng = np.random.default_rng(2)
        freq = np.concatenate([self._band(rng, 300, 1500.0),
                               self.NU0 * (1.0 + (5000.0 + 900.0 * np.arange(8)) / C_KMS)])
        tau = rng.lognormal(0.0, 2.0, (4, freq.size))
        dv = np.array([19.5, 9.1, 2.0, 5.0])
        x_quad, w_quad = gauss_legendre_nodes()
        tables = overlap_perline.build_tables(freq, float(dv.max()))
        assert tables['isolated'].size == 8
        got = overlap_perline.perline_fint(tau, dv, tables, x_quad, w_quad)
        want = perline_fint_reference(tau, freq, dv)
        assert got.shape == tau.shape and got.dtype == np.float64
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-9)

    def test_overcomplete_tables_change_nothing(self):
        """Tables built for a larger dv only add neighbours too far away to matter."""
        rng = np.random.default_rng(3)
        freq = self._band(rng, 300, 1500.0)
        tau = rng.lognormal(0.0, 1.5, (3, freq.size))
        dv = np.array([1.0, 3.0, 5.0])
        x_quad, w_quad = gauss_legendre_nodes()
        tight = overlap_perline.perline_fint(
            tau, dv, overlap_perline.build_tables(freq, 5.0), x_quad, w_quad)
        loose = overlap_perline.perline_fint(
            tau, dv, overlap_perline.build_tables(freq, 30.0), x_quad, w_quad)
        np.testing.assert_allclose(loose, tight, rtol=1e-6, atol=1e-9)

    def test_thick_batch_uses_all_nodes(self):
        """Above tau = 1e4 the outer nodes count; all 20 are used."""
        rng = np.random.default_rng(4)
        freq = self._band(rng, 120, 400.0)
        tau = rng.lognormal(7.0, 2.5, (1, freq.size))
        assert tau.max() > overlap_perline.CORE_TAU_MAX
        x_quad, w_quad = gauss_legendre_nodes()
        tables = overlap_perline.build_tables(freq, 4.0)
        got = overlap_perline.perline_fint(tau, np.array([4.0]), tables, x_quad, w_quad)
        want = perline_fint_reference(tau, freq, np.array([4.0]))
        live = _significant(want[0])
        np.testing.assert_allclose(got[0][live], want[0][live], rtol=5e-5)

    def test_nan_opacity_stays_local(self):
        """A NaN opacity takes the all-node branch; lines out of its reach are unaffected."""
        rng = np.random.default_rng(5)
        freq = np.concatenate([self._band(rng, 60, 200.0),
                               self.NU0 * (1.0 + (5000.0 + np.array([0.0, 3.0])) / C_KMS)])
        tau = rng.lognormal(0.0, 1.0, (1, freq.size))
        tau[0, -1] = np.nan
        x_quad, w_quad = gauss_legendre_nodes()
        tables = overlap_perline.build_tables(freq, 4.0)
        got = overlap_perline.perline_fint(tau, np.array([4.0]), tables, x_quad, w_quad)[0]
        assert np.isnan(got[-1])
        # the band is 4800 km/s from the NaN line and its one neighbour
        clean = perline_fint_reference(tau[:, :-2], freq[:-2], np.array([4.0]))[0]
        np.testing.assert_allclose(got[:-2], clean, rtol=1e-5)


# ---------------------------------------------------------------------------
# calc_intensity against the reference
# ---------------------------------------------------------------------------

# (line list, t_kin, n_mol, dv, rtol).  The largest opacity of a call
# decides between the 12 central nodes (up to 1e4) and all 20.
REFERENCE_CASES = [
    pytest.param("synthetic", 600.0, 1e16, 4.0, 1e-5, id="synthetic-tau1"),
    pytest.param("synthetic", 600.0, 1e18, 4.0, 1e-5, id="synthetic-tau1e2"),
    pytest.param("synthetic", 900.0, 3e16, 12.0, 1e-5, id="synthetic-wide"),
    pytest.param("synthetic", 600.0, 1e20, 1.0, 5e-5, id="synthetic-tau6e4"),
]
for _name in REAL_FILES:
    for _t, _n, _dv in [(942.0, 2.9e17, 19.5), (873.0, 7.9e17, 19.9),
                        (600.0, 1e18, 2.0), (600.0, 1e20, 1.0)]:
        REFERENCE_CASES.append(pytest.param(
            _name, _t, _n, _dv, None, id=f"{_name}-T{_t:g}-N{_n:g}-dv{_dv:g}"))


class TestOverlapMatchesReference:

    @pytest.mark.parametrize("name, t_kin, n_mol, dv, rtol", REFERENCE_CASES)
    def test_calc_intensity(self, line_list, name, t_kin, n_mol, dv, rtol):
        """curve_growth = no-overlap intensity x (reference fint / isolated fint)."""
        mll = line_list(name)
        no_overlap = _calc(mll, t_kin, n_mol, dv, "curve_growth_no_overlap")
        got = _calc(mll, t_kin, n_mol, dv)
        assert got._perline_tables[1] is not None, "no overlapping lines in this case"

        tau_max = float(np.max(no_overlap.tau))
        if rtol is None:
            rtol = 1e-5 if tau_max <= overlap_perline.CORE_TAU_MAX else 5e-5
        elif rtol > 1e-5:
            assert tau_max > overlap_perline.CORE_TAU_MAX   # the all-node branch

        want = overlap_intensity_reference(no_overlap.intensity, no_overlap.tau, _freq(mll), dv)
        live = _significant(want)
        np.testing.assert_allclose(got.intensity[live], want[live], rtol=rtol)
        # the rest: weak lines, no worse than the strongest line's tolerance
        np.testing.assert_allclose(got.intensity, want, rtol=1e-3, atol=rtol * want.max())
        np.testing.assert_array_equal(got.tau, no_overlap.tau)

    @pytest.mark.parametrize("name", ["synthetic", "HCN", "C2H2"])
    def test_overlap_lowers_the_dense_band(self, line_list, name):
        """Blended thick lines share their emission: less flux than without overlap."""
        mll = line_list(name)
        t_kin, n_mol, dv = (600.0, 1e17, 4.0) if name == "synthetic" else (900.0, 5e17, 19.7)
        no_overlap = _calc(mll, t_kin, n_mol, dv, "curve_growth_no_overlap")
        overlap = _calc(mll, t_kin, n_mol, dv)
        assert np.all(overlap.intensity <= no_overlap.intensity * (1.0 + 1e-6))
        assert overlap.intensity.sum() < 0.97 * no_overlap.intensity.sum()

    def test_batch_matches_single_calls(self, line_list):
        """calc_intensity_batch (mixed dv) = one calc_intensity per condition."""
        mll = line_list("synthetic")
        t_kin = np.array([600.0, 900.0, 500.0])
        n_mol = np.array([1e16, 1e18, 3e17])
        dv = np.array([4.0, 12.0, 1.0])
        batch = Intensity(mll).calc_intensity_batch(t_kin, n_mol, dv)
        assert batch.shape == (3, len(_freq(mll)))
        for i in range(3):
            single = _calc(mll, t_kin[i], n_mol[i], dv[i]).intensity
            np.testing.assert_allclose(batch[i], single, rtol=1e-6, atol=1e-9 * single.max())


# ---------------------------------------------------------------------------
# Sum rule
# ---------------------------------------------------------------------------

class TestBlendSumRule:
    """The lines of a blend share its emission: sum_i fint_i = integral(1 - exp(-tau_tot)).

    Independent of any per-line formula (dense trapezoid over the whole
    blend).  Thin blends agree to the window edges; for saturated
    profiles the 20-node rule itself is only good to 1-2 %.
    """

    THIN = 0.1      # largest line-centre opacity of a "thin" blend
    THICK = 30.0

    @pytest.mark.parametrize("name, t_kin, n_mol, dv", [
        ("synthetic", 600.0, 1e14, 4.0),
        ("synthetic", 600.0, 1e16, 4.0),
        ("CO2", 600.0, 1e18, 2.0),
        ("H2O", 600.0, 1e18, 9.1),
        ("HCN", 873.0, 7.9e17, 2.0),
    ])
    def test_sum_of_lines_is_the_blend_emission(self, line_list, name, t_kin, n_mol, dv):
        mll = line_list(name)
        freq = _freq(mll)
        tau, fint = _overlap_fint(mll, t_kin, n_mol, dv)

        n_thin = n_thick = 0
        for idx in _isolated_blends(freq, dv):
            tau_max = tau[idx].max()
            if tau[idx].sum() < 1e-6 or tau_max >= self.THICK:
                continue
            exact = blend_equivalent_width(tau[idx], freq[idx], dv)
            if tau_max < self.THIN:
                n_thin += 1
                assert fint[idx].sum() == pytest.approx(exact, rel=1e-3)
            else:
                n_thick += 1
                assert fint[idx].sum() == pytest.approx(exact, rel=2e-2)
        assert n_thin + n_thick >= 10
        if n_mol >= 1e16:
            assert n_thick >= 5


# ---------------------------------------------------------------------------
# Optically thin limit
# ---------------------------------------------------------------------------

class TestThinLimit:

    @pytest.mark.parametrize("name, t_kin, dv", [
        ("synthetic", 600.0, 4.0),
        ("HCN", 873.0, 19.9),
        ("C2H2", 942.0, 19.5),
    ])
    def test_thin_lines_do_not_interact(self, line_list, name, t_kin, dv):
        """tau_tot < 1e-4: curve_growth = curve_growth_no_overlap to 1e-4."""
        mll = line_list(name)
        probe = _calc(mll, t_kin, 1e12, dv)
        assert probe._perline_tables[1] is not None
        n_mol = 1e12 * 5e-5 / _neighbour_tau_bound(probe)     # tau is linear in n_mol

        overlap = _calc(mll, t_kin, n_mol, dv)
        assert _neighbour_tau_bound(overlap) < 1e-4
        no_overlap = _calc(mll, t_kin, n_mol, dv, "curve_growth_no_overlap")
        live = _significant(no_overlap.intensity)
        np.testing.assert_allclose(overlap.intensity[live], no_overlap.intensity[live], rtol=1e-4)


# ---------------------------------------------------------------------------
# Line lists without overlap
# ---------------------------------------------------------------------------

class TestIsolatedLists:
    """No line has a neighbour: the isolated-line fast path, unchanged."""

    def test_no_tables_are_built(self):
        inten = _calc(_isolated_line_list(), 500.0, 1e18, 1.0)
        key, tables = inten._perline_tables
        assert key == (10, 1.0)
        assert tables is None

    @pytest.mark.parametrize("n_mol", [1e12, 1e17, 1e20])
    def test_fint_multi_is_the_isolated_fast_path(self, n_mol):
        """Bit-identical to ``_cog_with_tau_shortcuts`` times the physical factors.

        ``n_mol`` covers all-thin, mixed and all-thick lists (the thin
        shortcut applies below tau = 0.01).
        """
        mll = _isolated_line_list()
        inten = _calc(mll, 500.0, n_mol, 1.0)
        lines = mll.lines_as_namedtuple
        cls = Intensity
        center_tau = np.asarray(inten.tau)[np.newaxis, :]
        dv = np.array([1.0])
        bb_vals = cls._bb(lines.freq, np.array([500.0]))
        freq_ratio = (lines.freq / c.SPEED_OF_LIGHT_CGS)[np.newaxis, :]

        got = inten._fint_multi(center_tau, dv, bb_vals, freq_ratio, cls._SQRT_LN2_INV, lines.freq)

        expected = np.empty_like(center_tau)
        np.multiply(freq_ratio, bb_vals, out=expected)
        expected *= (cls._SQRT_LN2_INV * 1e5)
        expected *= dv[:, np.newaxis]
        np.multiply(expected, inten._cog_with_tau_shortcuts(
            center_tau, cls._GAUSS_QUAD_EXP, cls._GAUSS_QUAD_W), out=expected)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(inten.intensity, expected[0])

    def test_thin_shortcut_still_applies(self):
        """All lines below tau = 0.01: F = tau * sum_k w_k exp(-x_k^2).

        The per-line overlap integral has no such shortcut; at
        tau = 0.005 the two differ by 0.2 %.
        """
        mll = _isolated_line_list()
        n_mol = 1e12 * 0.005 / _calc(mll, 500.0, 1e12, 1.0).tau.max()   # tau is linear in n_mol
        inten = _calc(mll, 500.0, n_mol, 1.0)
        assert 0.004 < inten.tau.max() < Intensity._TAU_THIN

        freq = _freq(mll)
        prefactor = (Intensity._SQRT_LN2_INV * 1e5 * 1.0 * (freq / c.SPEED_OF_LIGHT_CGS)
                     * Intensity._bb(freq, np.array([500.0])))
        shortcut = prefactor * inten.tau * Intensity._GAUSS_QUAD_SQRT_PI_SUM
        np.testing.assert_allclose(inten.intensity, shortcut, rtol=1e-12)
        exact = prefactor * isolated_fint_reference(inten.tau)
        assert np.max(np.abs(shortcut / exact - 1.0)) > 1e-3

    def test_batch_fast_all_isolated(self):
        inten = _calc(_isolated_line_list(), 500.0, 1e18, 1.0)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=1.0)
        assert ctx['overlap'] == {'all_isolated': True}


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------

class TestBatchFastPath:
    """``prepare_overlap_structure`` + ``calc_intensity_batch_fast`` with overlapping lines."""

    T_KIN = np.array([942.0, 873.0, 600.0, 700.0])
    N_MOL = np.array([2.9e17, 7.9e17, 1e18, 3e17])
    DV = np.array([19.5, 19.9, 2.0, 7.0])

    def _assert_matches_single_calls(self, mll, result):
        assert result.dtype == np.float32
        assert result.shape == (4, len(_freq(mll)))
        for i in range(4):
            want = _calc(mll, self.T_KIN[i], self.N_MOL[i], self.DV[i]).intensity
            live = _significant(want)
            np.testing.assert_allclose(result[i][live], want[live], rtol=1e-5)
            np.testing.assert_allclose(result[i], want, rtol=1e-3, atol=1e-5 * want.max())

    @pytest.mark.parametrize("name", ["synthetic", "HCN", "C2H2", "CO2", "H2O"])
    def test_mixed_dv_batch_matches_calc_intensity(self, line_list, name):
        mll = line_list(name)
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=float(self.DV.max()))
        assert ctx['overlap']['all_isolated'] is False
        assert ctx['overlap']['tables']['dv_max'] == self.DV.max()

        result = inten.calc_intensity_batch_fast(self.T_KIN, self.N_MOL, self.DV, ctx)
        self._assert_matches_single_calls(mll, result)
        assert ctx['overlap']['tables']['dv_max'] == self.DV.max()      # not rebuilt

    @pytest.mark.parametrize("name", ["synthetic", "HCN"])
    def test_dv_above_the_tables_rebuilds_them(self, line_list, name):
        """Tables prepared for too small a dv are rebuilt for the batch's largest."""
        mll = line_list(name)
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=3.0)
        assert ctx['overlap']['all_isolated'] is False
        assert ctx['overlap']['tables']['dv_max'] == 3.0

        slots_before = ctx['overlap']['n_slots']

        result = inten.calc_intensity_batch_fast(self.T_KIN, self.N_MOL, self.DV, ctx)
        self._assert_matches_single_calls(mll, result)
        assert ctx['overlap']['tables']['dv_max'] == self.DV.max()
        assert ctx['overlap']['n_slots'] > slots_before             # summary follows the tables

    def test_overlap_summary(self, line_list):
        """The summary counts stored next to the tables."""
        mll = line_list("synthetic")
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=4.0)
        overlap = ctx['overlap']
        tables = overlap['tables']
        counts = tables['hi'] - tables['lo']

        assert overlap['n_isolated'] == 12                          # the 12 lines 1500 km/s apart
        assert overlap['n_isolated'] + overlap['n_blended_lines'] == ctx['n_lines'] == 325
        assert overlap['max_neighbors'] == counts.max() >= 2
        assert overlap['mean_neighbors'] == pytest.approx(counts.mean())
        assert overlap['n_slots'] == sum(width * sel.size for width, sel in tables['buckets'])
        assert overlap['n_slots'] >= counts[counts > 1].sum()

    def test_single_condition(self, line_list):
        mll = line_list("synthetic")
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=4.0)
        result = inten.calc_intensity_batch_fast(
            np.array([600.0]), np.array([1e16]), np.array([4.0]), ctx)
        want = _calc(mll, 600.0, 1e16, 4.0).intensity
        live = _significant(want)
        np.testing.assert_allclose(result[0][live], want[live], rtol=1e-5)

    def test_line_indices_slice_the_result(self, line_list):
        mll = line_list("synthetic")
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context()
        inten.prepare_overlap_structure(ctx, dv=float(self.DV.max()))
        full = inten.calc_intensity_batch_fast(self.T_KIN, self.N_MOL, self.DV, ctx)
        ctx['line_indices'] = np.array([0, 5, 17, 200])
        sliced = inten.calc_intensity_batch_fast(self.T_KIN, self.N_MOL, self.DV, ctx)
        np.testing.assert_array_equal(sliced, full[:, [0, 5, 17, 200]])

    def test_context_for_a_subset_of_lines(self, line_list):
        """``prepare_batch_context(line_indices=...)``: overlap among the kept lines only."""
        mll = line_list("synthetic")
        freq = _freq(mll)
        keep = np.flatnonzero(np.arange(freq.size) % 3 != 0)
        inten = Intensity(mll)
        ctx = inten.prepare_batch_context(line_indices=keep)
        inten.prepare_overlap_structure(ctx, dv=4.0)
        result = inten.calc_intensity_batch_fast(
            np.array([600.0]), np.array([1e17]), np.array([4.0]), ctx)[0]

        no_overlap = _calc(mll, 600.0, 1e17, 4.0, "curve_growth_no_overlap")
        want = overlap_intensity_reference(
            no_overlap.intensity[keep], no_overlap.tau[keep], freq[keep], 4.0)
        live = _significant(want)
        np.testing.assert_allclose(result[live], want[live], rtol=1e-5)


# ---------------------------------------------------------------------------
# Wavelength range changes
# ---------------------------------------------------------------------------

class TestWavelengthRangeChange:

    @pytest.mark.parametrize("name, t_kin, n_mol, dv, new_range", [
        ("synthetic", 600.0, 1e16, 4.0, (13.0, 14.1)),
        ("HCN", 873.0, 7.9e17, 19.9, (13.5, 14.5)),
    ])
    def test_tables_are_rebuilt(self, name, t_kin, n_mol, dv, new_range):
        inten = _calc(_make_line_list(name), t_kin, n_mol, dv)
        n_before = len(inten.intensity)
        key, tables = inten._perline_tables
        assert key == (n_before, dv) and tables is not None

        inten.wavelength_range = new_range
        assert inten._perline_tables is None
        inten.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=dv)
        n_after = len(inten.intensity)
        assert 0 < n_after < n_before
        assert inten._perline_tables[0] == (n_after, dv)
        assert inten._perline_tables[1]['n_lines'] == n_after

        fresh = _calc(_make_line_list(name, wavelength_range=new_range), t_kin, n_mol, dv)
        np.testing.assert_array_equal(inten.intensity, fresh.intensity)

    def test_tables_follow_dv_and_are_reused_otherwise(self, line_list):
        """One entry per instance: kept while T and N change, rebuilt for a new dv."""
        inten = _calc(line_list("synthetic"), 600.0, 1e16, 4.0)
        tables = inten._perline_tables[1]
        inten.calc_intensity(t_kin=700.0, n_mol=3e16, dv=4.0)
        assert inten._perline_tables[1] is tables
        inten.calc_intensity(t_kin=700.0, n_mol=3e16, dv=6.0)
        assert inten._perline_tables[1] is not tables
        assert inten._perline_tables[1]['dv_max'] == 6.0


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

# Uses the pool in a parent process, then evaluates the same integral in a
# forked child (see TestThreads.test_forked_child_gets_its_own_pool).
_FORK_SCRIPT = r"""
import multiprocessing as mp
import numpy as np
import iSLAT.Modules.DataTypes._overlap_perline as op


def fint_sum():
    rng = np.random.default_rng(0)
    freq = 2.1e13 * (1.0 + np.sort(rng.uniform(0.0, 3000.0, 6000)) / 299792.458)
    tau = rng.lognormal(0.0, 1.0, (1, freq.size))
    x, w = np.polynomial.legendre.leggauss(20)
    tables = op.build_tables(freq, 20.0)
    return float(op.perline_fint(tau, np.array([20.0]), tables, 6.0 * x, 6.0 * w).sum())


def child(queue):
    queue.put(fint_sum())


if __name__ == "__main__":
    parent = fint_sum()
    assert op._POOL is not None, "the parent did not use the thread pool"
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=child, args=(queue,))
    process.start()
    process.join(30)
    if process.is_alive():
        process.terminate()
        raise SystemExit("the forked child did not return from the overlap integral")
    assert queue.get(timeout=5) == parent
    print("OK")
"""


class TestThreads:
    """``ISLAT_INTENSITY_THREADS``: the same numbers on one thread and on the pool."""

    @staticmethod
    def _count_pool_use(monkeypatch):
        used = []
        real_pool = overlap_perline._pool

        def spy():
            used.append(True)
            return real_pool()

        monkeypatch.setattr(overlap_perline, "_pool", spy)
        return used

    @pytest.mark.parametrize("name, t_kin, n_mol, dv", [
        ("synthetic", 600.0, 1e17, 4.0),
        ("HCN", 873.0, 7.9e17, 19.9),
    ])
    def test_one_and_eight_threads_agree(self, monkeypatch, line_list, name, t_kin, n_mol, dv):
        mll = line_list(name)
        if name == "synthetic":
            # too small a job to be split or sent to the pool by default
            monkeypatch.setattr(overlap_perline, "PARALLEL_MIN_WORK", 0)
            monkeypatch.setattr(overlap_perline, "CHUNK_ELEMS", 2000)
        used = self._count_pool_use(monkeypatch)

        monkeypatch.setenv("ISLAT_INTENSITY_THREADS", "1")
        serial = _calc(mll, t_kin, n_mol, dv).intensity
        assert not used

        monkeypatch.setenv("ISLAT_INTENSITY_THREADS", "8")
        threaded = _calc(mll, t_kin, n_mol, dv).intensity
        assert used

        np.testing.assert_array_equal(serial, threaded)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    def test_forked_child_gets_its_own_pool(self):
        """A child forked after the parent used the pool must not wait on the parent's threads.

        Forking is the default start method of process pools on Linux
        before Python 3.14.  Run in a fresh interpreter, so that this
        (multi-threaded, Tk-using) test process is not the one forked.
        """
        result = subprocess.run(
            [sys.executable, "-c", _FORK_SCRIPT], capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**os.environ, "ISLAT_INTENSITY_THREADS": "4"},
        )
        assert result.returncode == 0 and result.stdout.strip().endswith("OK"), (
            result.stdout + result.stderr)

    def test_thread_count_setting(self, monkeypatch):
        monkeypatch.setenv("ISLAT_INTENSITY_THREADS", "3")
        assert overlap_perline._n_threads() == 3
        monkeypatch.setenv("ISLAT_INTENSITY_THREADS", "0")
        assert overlap_perline._n_threads() == 1
        monkeypatch.delenv("ISLAT_INTENSITY_THREADS")
        assert 1 <= overlap_perline._n_threads() <= 8
