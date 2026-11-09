# -*- coding: utf-8 -*-
"""Tests for iSLAT.Modules.DataProcessing.LineIdentity.

A LineRef is what the plots and the line list tables hand each other to say
"this line".  It is built from a plot's info dict or a table row, and
match_line finds that line again in a list of wavelengths (and level labels):
within a relative wavelength tolerance, preferring the line with the same
levels, and taking a saved line's fit window into account when its wavelength
is a fitted center rather than the catalog value.
"""

import numpy as np
import pandas as pd
import pytest

from iSLAT.Modules.DataProcessing.LineIdentity import (
    DEFAULT_HALF_WINDOW,
    LAM_RTOL,
    LineRef,
    match_line,
)


# ---------------------------------------------------------------------------
# Building a LineRef
# ---------------------------------------------------------------------------

def test_from_info_reads_the_plot_keys():
    ref = LineRef.from_info({
        'lam': 17.25, 'molecule_name': 'H2O',
        'up_lev': ' 0_0_0|10_2_9 ', 'low_lev': '0_0_0|9_1_8',
    })
    assert ref == LineRef(lam=17.25, molecule='H2O',
                          lev_up='0_0_0|10_2_9', lev_low='0_0_0|9_1_8')


def test_from_info_accepts_the_table_level_keys_and_blanks():
    ref = LineRef.from_info({'lam': '5.1', 'lev_up': '', 'lev_low': np.nan})
    assert ref == LineRef(lam=5.1)


def test_from_info_without_a_wavelength_is_none():
    assert LineRef.from_info(None) is None
    assert LineRef.from_info({}) is None
    assert LineRef.from_info({'lam': None}) is None
    assert LineRef.from_info({'lam': 'not a number'}) is None
    assert LineRef.from_info({'lam': np.nan}) is None


def test_from_row_takes_the_species_unless_a_molecule_is_given():
    row = pd.Series({'species': 'CO', 'lam': '4.95', 'lev_up': 'v1 J2',
                     'lev_low': 'v0 J1'})
    assert LineRef.from_row(row).molecule == 'CO'
    assert LineRef.from_row(row, molecule='CO_hot').molecule == 'CO_hot'
    assert LineRef.from_row(row).lam == pytest.approx(4.95)


def test_from_row_keeps_a_fit_window_only_around_the_line():
    around = {'lam': 12.40, 'xmin': 12.39, 'xmax': 12.42}
    assert LineRef.from_row(around).window == (12.39, 12.42)
    beside = {'lam': 12.40, 'xmin': 12.41, 'xmax': 12.42}
    assert LineRef.from_row(beside).window is None
    reversed_ = {'lam': 12.40, 'xmin': 12.42, 'xmax': 12.39}
    assert LineRef.from_row(reversed_).window is None
    assert LineRef.from_row({'lam': 12.40, 'xmin': ''}).window is None


def test_describe():
    assert LineRef(17.23456, 'H2O').describe() == 'H2O 17.2346 um'
    assert LineRef(5.0).describe() == '5.0000 um'


def test_default_half_window_is_the_saved_line_default():
    assert DEFAULT_HALF_WINDOW == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# Finding a line
# ---------------------------------------------------------------------------

def test_match_line_takes_the_nearest_wavelength_within_the_tolerance():
    lams = [12.0, 12.4, 12.4 * (1 + 0.5 * LAM_RTOL), 13.0]
    assert match_line(LineRef(12.4), lams) == 1
    assert match_line(LineRef(12.4 * (1 + 0.4 * LAM_RTOL)), lams) == 2
    assert match_line(LineRef(12.4 * (1 + 3 * LAM_RTOL)), lams) is None


def test_match_line_reads_rounded_text_wavelengths():
    # The line search table shows wavelengths with 10 significant digits.
    lams = [f"{v:.10g}" for v in (12.123456789012, 12.5)]
    assert match_line(LineRef(12.123456789012), lams) == 0
    assert match_line(LineRef(12.5), ['', 'x', '12.5']) == 2


def test_match_line_prefers_the_line_with_the_same_levels():
    lams = [17.2, 17.2, 17.2 * (1 + 0.1 * LAM_RTOL)]
    ups = ['a', 'b', 'c']
    lows = ['x', 'y', 'z']
    assert match_line(LineRef(17.2, lev_up='b', lev_low='y'), lams, ups, lows) == 1
    assert match_line(LineRef(17.2, lev_up='c'), lams, ups, lows) == 2
    # Unknown levels: fall back to the nearest wavelength.
    assert match_line(LineRef(17.2, lev_up='q'), lams, ups, lows) == 0


def test_match_line_finds_a_fitted_line_by_levels_inside_its_window():
    lams = [12.398, 12.4, 12.402]
    ups = ['a', 'b', 'c']
    fitted = LineRef(12.4011, lev_up='c', window=(12.39, 12.41))
    assert match_line(fitted, lams, ups) == 2
    # Without the window the fitted center is too far from any line.
    assert match_line(LineRef(12.4011, lev_up='c'), lams, ups) is None


def test_match_line_uses_positions_not_index_labels():
    df = pd.DataFrame({'lam': [5.0, 6.0, 6.0], 'lev_up': ['a', 'b', 'c']},
                      index=[30, 10, 20])
    assert match_line(LineRef(6.0, lev_up='c'), df['lam'], df['lev_up']) == 2


def test_match_line_on_nothing():
    assert match_line(LineRef(6.0), []) is None
    assert match_line(LineRef(6.0, lev_up='a'), [6.0], None) == 0
