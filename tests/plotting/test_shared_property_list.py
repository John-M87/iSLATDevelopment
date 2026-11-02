# -*- coding: utf-8 -*-
"""One shared line-property list for the population diagram and the line-list filter.

The population diagram's X / Y axes, color-by and marker shape (all in the one
Plot Settings dialog) and the line-list filter offer the same properties, under
the same names, in the same order, and each one actually works everywhere.
"""

import matplotlib
matplotlib.use("Agg")

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import iSLAT.Constants as c
import iSLAT.Modules.DataTypes.Intensity  # noqa: F401  (registers the shared list)
import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401  (registers quantum entries)
from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry
from iSLAT.Modules.DataProcessing.LineFilterExpression import (
    Condition,
    MaskContext,
    condition_mask,
    nearest_neighbour_separation_kms,
    vib_band_label,
)
from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import (
    QUICK_ADD_KEYS,
    _DERIVED_KEYS,
    build_property_catalog,
    model_quantity_frame,
    row_state_to_condition,
)
from iSLAT.Modules.Plotting import PopulationDiagramPlot
from iSLAT.Modules.Plotting.PopulationDiagramContextMixin import (
    DEFAULT_PLOT_SETTINGS,
    NONE_LABEL,
    apply_plot_settings,
    capture_plot_settings,
    parse_plot_settings,
)

# H2O-format levels "v1_v2_v3|J_Ka_Kc": two v1-0 lines, one v0-0, one hot band.
_LEVELS = [
    ("1_0_0|9_2_7", "0_0_0|8_0_8"),
    ("1_0_0|7_2_5", "0_0_0|6_1_6"),
    ("0_0_0|10_1_9", "0_0_0|9_0_9"),
    ("0_1_0|5_1_4", "0_1_0|4_0_4"),
    ("0_0_0|3_1_2", "0_0_0|2_0_2"),
]
_LAMS = [6.50, 6.51, 12.40, 14.0, 17.20]


def _h2o_molecule(name="H2O"):
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    lines = []
    for i, ((up, low), lam) in enumerate(zip(_LEVELS, _LAMS)):
        lines.append({
            "nr": i + 1, "lev_up": up, "lev_low": low, "lam": lam,
            "freq": c.SPEED_OF_LIGHT_MICRONS / lam,
            "a_stein": 0.01 + i * 0.005, "e_up": 1000 + i * 600,
            "e_low": 500 + i * 600, "g_up": 2 * i + 3, "g_low": 2 * i + 1,
        })
    mll = MoleculeLineList(molecule_id="H2O", lines_data=lines)
    mll.partition_function = mll._partition_type(
        t=np.array([100, 300, 500, 1000, 2000], dtype=float),
        q=np.array([10, 150, 500, 2000, 8000], dtype=float),
    )
    mll._molar_mass = 18.015
    mol = Molecule(
        name=name, displaylabel=name, filepath=None, color="#3399FF",
        is_visible=True, temp=500.0, radius=1.0, n_mol=1e18, distance=160.0,
        fwhm=130.0,
        initial_molecule_parameters={"t_kin": 500.0, "scale_exponent": 18.0,
                                     "scale_number": 1.0, "radius_init": 1.0},
    )
    mol.lines = mll
    intensity = Intensity(mll)
    intensity.calc_intensity(t_kin=500.0, n_mol=1e18, dv=1.0)
    mol.intensity = intensity
    return mol


@pytest.fixture
def pdp():
    plot = PopulationDiagramPlot(_h2o_molecule())
    plot.generate_plot()
    yield plot
    plot.close()


def _line_frame():
    """The molecule's lines as the line-list filter sees them."""
    return pd.DataFrame({
        "species": ["H2O"] * len(_LAMS),
        "lev_up": [u for u, _ in _LEVELS], "lev_low": [l for _, l in _LEVELS],
        "lam": _LAMS, "freq": [c.SPEED_OF_LIGHT_MICRONS / x for x in _LAMS],
        "a_stein": [0.01 + i * 0.005 for i in range(5)],
        "e_up": [1000.0 + i * 600 for i in range(5)],
        "e_low": [500.0 + i * 600 for i in range(5)],
        "g_up": [2.0 * i + 3 for i in range(5)], "g_low": [2.0 * i + 1 for i in range(5)],
    })


def _shared(species=("H2O",)):
    return PlotAxisRegistry.get_property_options(species=list(species))


# ======================================================================
# The list itself
# ======================================================================
class TestSharedList:
    def test_axes_color_and_shape_offer_the_same_list(self):
        for species in (None, ["H2O"], ["H2O", "CO"], []):
            options = PlotAxisRegistry.get_property_options(species=species)
            assert PlotAxisRegistry.get_axis_options(species) == options
            assert PlotAxisRegistry.get_color_options(species) == options

    def test_names_are_unique(self):
        names = [e.display_name for e in PlotAxisRegistry.get_property_options()]
        assert len(names) == len(set(names))

    def test_contains_the_union_of_former_lists(self):
        keys = {e.key for e in _shared()}
        # formerly plot-only, formerly filter-only, formerly color-only
        assert {"rd_yax", "fwhm_instrumental_kms", "fwhm_convolved_kms"} <= keys
        assert {"freq", "velocity_sep", "vib_band"} <= keys
        assert {"lev_up", "lev_low", "molecule"} <= keys
        assert "component" not in keys  # alias of "molecule", listed once

    def test_every_entry_is_filterable(self):
        assert all(e.filter_kind for e in PlotAxisRegistry.get_property_options())

    @pytest.mark.parametrize("alias, key", [("lam", "wavelength"), ("e_up", "eu"),
                                            ("component", "molecule")])
    def test_legacy_keys_still_resolve(self, alias, key):
        assert PlotAxisRegistry.canonical_key(alias) == key
        assert PlotAxisRegistry.get(alias) is PlotAxisRegistry.get(key)


# ======================================================================
# The filter offers the same list
# ======================================================================
class TestFilterUsesSharedList:
    def test_same_names_same_order(self):
        catalog = build_property_catalog(_line_frame(), "H2O", derived=_DERIVED_KEYS)
        assert [(s.key, s.label) for s in catalog] == [
            (e.key, e.display_name) for e in _shared()]

    def test_extra_columns_follow_the_shared_list(self):
        df = _line_frame().assign(xmin=0.0, fit_flux=1.0)
        labels = [s.label for s in build_property_catalog(df, "H2O", derived=_DERIVED_KEYS)]
        assert labels[-2:] == ["fit_flux", "xmin"]
        assert labels[:-2] == [e.display_name for e in _shared()]

    def test_derived_keys_are_the_model_quantities(self):
        assert set(_DERIVED_KEYS) == {"rd_yax", "intens", "tau",
                                      "fwhm_instrumental_kms", "fwhm_convolved_kms"}

    def test_quick_add_keys_are_in_the_list(self):
        keys = {e.key for e in _shared()}
        assert set(QUICK_ADD_KEYS) <= keys

    @pytest.mark.parametrize("label, op, value, expected", [
        ("Pop. diagram Y [ln(4πF/hνAg)]", "≥", "-2.5", [True, True, False, False, False]),
        ("Convolved FWHM (km/s)", "≥", "131.5", [False, False, True, True, True]),
        ("Molecule (species)", "is one of", "H2O", [True] * 5),
        ("Vib. band", "is band", "v1-0", [True, True, False, False, False]),
        ("Frequency (Hz)", "≤", "2.2e13", [False, False, False, True, True]),
    ])
    def test_new_properties_filter(self, label, op, value, expected):
        df = _line_frame()
        data = {"rd_yax": np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
                "fwhm_convolved_kms": np.array([130.0, 131.0, 132.0, 133.0, 134.0])}
        catalog = build_property_catalog(df, "H2O", derived=_DERIVED_KEYS)
        cond, errors = row_state_to_condition({"prop": label, "op": op, "v1": value}, catalog)
        assert not errors and cond is not None
        ctx = MaskContext(species="H2O",
                          derived_source=lambda: model_quantity_frame(data, df.index))
        assert condition_mask(df, cond, ctx).tolist() == expected


# ======================================================================
# The population diagram plots every property
# ======================================================================
_ALL_KEYS = [e.key for e in PlotAxisRegistry.get_property_options(species=["H2O"])]


class TestPlotUsesSharedList:
    @pytest.mark.parametrize("key", _ALL_KEYS)
    def test_every_property_as_x_axis_color_and_shape(self, pdp, key):
        pdp.set_axes(x_prop=key, y_prop="rd_yax", regenerate=False)
        pdp.color_by(key, regenerate=False)
        pdp.shape_by(key)
        assert pdp.ax.get_title() == "H2O Population diagram"
        assert pdp.ax.collections, f"no points plotted for {key}"

    @pytest.mark.parametrize("key", ["vib_band", "molecule", "lev_up"])
    def test_categorical_property_is_a_category_axis(self, pdp, key):
        pdp.set_axes(x_prop="eu", y_prop=key, y_log=True)
        cats = pdp._categories_for(key)
        assert cats, f"{key} should be drawn as a category axis"
        assert pdp.ax.get_yscale() == "linear"
        assert pdp.ax.get_ylim() == pytest.approx((-0.5, len(cats) - 0.5))
        assert [t.get_text() for t in pdp.ax.get_yticklabels()] == cats

    def test_band_labels_match_the_band_filter(self, pdp):
        pdp.set_axes(x_prop="vib_band", y_prop="eu")
        cdata = pdp.component_data[0]
        bands = PlotAxisRegistry.resolve_array("vib_band", cdata)
        assert pdp._categories_for("vib_band") == ["v0-0", "v1-0", "v1-1"]
        df = pd.DataFrame({"lev_up": cdata["lev_up"], "lev_low": cdata["lev_low"]})
        for band in set(bands):
            selected = condition_mask(df, Condition("vib_band", op="band", value=band),
                                      MaskContext())
            assert selected.tolist() == [b == band for b in bands]

    def test_band_label_helper(self):
        assert vib_band_label("1_0_0|9_2_7", "0_0_0|8_0_8") == "v1-0"
        assert vib_band_label("0_1_0|5_1_4", "0_1_0|4_0_4") == "v1-1"
        assert vib_band_label("X|1", "0|1") == ""

    def test_constant_property_gives_one_marker_group(self, pdp):
        pdp.shape_by("fwhm_instrumental_kms")
        legend_texts = [t.get_text() for t in pdp.ax.get_legend().get_texts()]
        assert "130" in legend_texts

    def test_velocity_separation_matches_the_filter(self, pdp):
        cdata = pdp.component_data[0]
        expected = nearest_neighbour_separation_kms(cdata["wavelength"])
        np.testing.assert_allclose(PlotAxisRegistry.resolve_array("velocity_sep", cdata), expected)

    def test_frequency_and_fwhm_are_per_line(self, pdp):
        cdata = pdp.component_data[0]
        n = len(cdata["wavelength"])
        np.testing.assert_allclose(cdata["freq"], c.SPEED_OF_LIGHT_MICRONS / np.asarray(_LAMS))
        for key in ("fwhm_instrumental_kms", "fwhm_convolved_kms"):
            assert np.shape(PlotAxisRegistry.resolve_array(key, cdata)) == (n,)

    def test_text_quantum_field_resolves_as_labels(self, pdp):
        assert PlotAxisRegistry.get_kind("qn_upper:Sym") == "categorical"
        vals = PlotAxisRegistry.resolve_array("qn_upper:Sym", pdp.component_data[0])
        assert vals is not None and vals.dtype == object

    def test_active_lines_land_on_the_category_axis(self, pdp):
        mol = pdp.molecule
        mol.lines._ensure_lines_created()
        table = mol.intensity.build_table()
        line_data = [(ln, float(table.loc[np.isclose(table["lam"], ln.lam), "intens"].iloc[0]), 0.0)
                     for ln in mol.lines.lines]
        pdp.set_axes(x_prop="lev_low", y_prop="eu")
        sc = pdp.render_active_lines(line_data, [], molecule=mol)
        cats = pdp._categories_for("lev_low")
        expected = sorted(cats.index(low) for _, low in _LEVELS)
        assert sorted(sc.get_offsets()[:, 0].tolist()) == expected

    def test_legacy_component_key_colors_by_molecule(self, pdp):
        pdp.color_by("component")
        legend = pdp.ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["H2O"]


# ======================================================================
# Plot Settings dialog helpers
# ======================================================================
def _form(**overrides):
    labels = {e.display_name: e.key for e in _shared()}
    values = {
        "x_prop": "E_up (K)", "x_log": False, "x_lim_mode": "auto", "x_lim_lo": "", "x_lim_hi": "",
        "y_prop": "Pop. diagram Y [ln(4πF/hνAg)]", "y_log": False, "y_lim_mode": "auto",
        "y_lim_lo": "", "y_lim_hi": "",
        "color_prop": NONE_LABEL, "cmap": "viridis", "vmin": "", "vmax": "", "pmin": "",
        "pmax": "", "color_log": False,
        "shape_prop": NONE_LABEL, "shape_bins": "5", "marker_size": "",
    }
    values.update(overrides)
    return values, labels


class TestPlotSettingsHelpers:
    def test_defaults_parse_to_default_settings(self):
        assert parse_plot_settings(*_form()) == DEFAULT_PLOT_SETTINGS

    def test_full_form(self):
        values, labels = _form(
            x_prop="Vib. band", y_lim_mode="exact", y_lim_lo="-5", y_lim_hi="",
            color_prop="Velocity sep. (km/s)", color_log=True, pmax="150",
            shape_prop="Molecule (species)", shape_bins="3", marker_size="12")
        s = parse_plot_settings(values, labels)
        assert s["x_prop"] == "vib_band" and s["y_lim"] == ("exact", -5.0, None)
        assert s["color"]["prop"] == "velocity_sep" and s["color"]["log_scale"] is True
        assert s["color"]["pmax"] == 100.0  # clamped
        assert s["shape"] == {"prop": "molecule", "n_bins": 3}
        assert s["marker_size"] == 12.0

    @pytest.mark.parametrize("overrides, message", [
        ({"x_prop": "nope"}, "X axis"),
        ({"vmin": "abc", "color_prop": "E_up (K)"}, "vmin"),
        ({"x_lim_mode": "exact", "x_lim_lo": "5", "x_lim_hi": "1"}, "less than"),
        ({"y_lim_mode": "percentile", "y_lim_hi": "120"}, "between 0 and 100"),
        ({"marker_size": "0"}, "greater than 0"),
    ])
    def test_invalid_fields_raise(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            parse_plot_settings(*_form(**overrides))

    def test_apply_round_trips_with_one_redraw(self, pdp):
        values, labels = _form(x_prop="Frequency (Hz)", x_log=True,
                               color_prop="Vib. band", cmap="tab10",
                               shape_prop="E_low (K)", marker_size="9")
        settings = parse_plot_settings(values, labels)
        with patch.object(pdp, "generate_plot", wraps=pdp.generate_plot) as gen:
            apply_plot_settings(pdp, settings)
        assert gen.call_count == 1
        assert capture_plot_settings(pdp) == {
            **settings,
            "color": {**settings["color"]},
            "shape": {**settings["shape"]},
        }

    def test_reset_clears_everything(self, pdp):
        pdp.color_by("eu", regenerate=False)
        pdp.shape_by("vib_band", regenerate=False)
        pdp.set_marker_size(20, regenerate=False)
        pdp.set_axes(x_prop="tau", y_prop="intens", x_log=True)
        apply_plot_settings(pdp, DEFAULT_PLOT_SETTINGS)
        assert capture_plot_settings(pdp) == DEFAULT_PLOT_SETTINGS

    def test_capture_without_plot_is_defaults(self):
        assert capture_plot_settings(None) == DEFAULT_PLOT_SETTINGS
