# -*- coding: utf-8 -*-
"""Full-spectrum export: GUI toggle-state sync and figure-size overrides."""

import matplotlib
matplotlib.use("Agg")

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from iSLAT.Modules.Plotting import FullSpectrumPlot
from iSLAT.Modules.Plotting.FullSpectrumView import (
    DEFAULT_EXPORT_HEIGHT,
    DEFAULT_EXPORT_WIDTH,
    MAX_EXPORT_PAGES,
    MAX_ROWS_PER_PAGE,
    apply_export_size,
    get_export_settings,
    get_page_settings,
    observed_wavelength_bounds,
    rows_per_page_list,
    set_export_settings,
    set_page_settings,
    toggle_state_display_kwargs,
    validate_page_ranges,
)

from tests.plotting import make_atomic_lines

# The package re-exports the FullSpectrumView *class* under the submodule's name.
fsv = importlib.import_module("iSLAT.Modules.Plotting.FullSpectrumView")


WAVE = np.linspace(5.0, 11.0, 300)
FLUX = np.where(WAVE < 8.0, 1.0, 4.0)
MODEL = 0.2 * np.exp(-0.5 * ((WAVE - 8.0) / 2.0) ** 2)


class _Molecules:
    """Minimal MoleculeDict stand-in: no visible molecules, a summed model."""

    _global_wavelength_range = None

    def __init__(self, summed=MODEL):
        self._summed = summed

    def apply_stellar_rv(self, wave):
        return np.asarray(wave)

    def get_summed_flux(self, wave, **_kw):
        return WAVE, self._summed

    def get_visible_molecules(self, return_objects=False):
        return []


def _tagged(plot, tag):
    return [a for ax in plot.subplots.values()
            for a in list(ax.lines) + list(ax.collections) if hasattr(a, tag)]


def _plot(**kw):
    kw.setdefault("n_panels", 3)
    return FullSpectrumPlot(WAVE, FLUX, molecules=_Molecules(), **kw)


# ======================================================================
# FullSpectrumPlot display options
# ======================================================================
class TestFullSpectrumPlotDisplayOptions:
    def test_defaults_keep_layout_limits_and_visible_artists(self):
        plot = _plot()
        plot.generate_plot()
        expected = plot._compute_panel_ylims(ylim_fn=plot._spectrum_ylim_fn)
        for idx, ax in plot.subplots.items():
            assert ax.get_ylim() == pytest.approx(expected[idx])
        assert all(a.get_visible() for a in _tagged(plot, "_islat_observed"))
        assert all(a.get_visible() for a in _tagged(plot, "_islat_summed"))
        plot.close()

    def test_hidden_observed_including_error_bars(self):
        plot = FullSpectrumPlot(WAVE, FLUX, error_data=np.full_like(FLUX, 0.1),
                                n_panels=3, show_observed=False)
        plot.generate_plot()
        observed = _tagged(plot, "_islat_observed")
        assert observed and all(not a.get_visible() for a in observed)
        plot.close()

    def test_hidden_summed(self):
        plot = _plot(show_summed=False)
        plot.generate_plot()
        summed = _tagged(plot, "_islat_summed")
        assert summed and all(not a.get_visible() for a in summed)
        assert all(a.get_visible() for a in _tagged(plot, "_islat_observed"))
        plot.close()

    def test_manual_limits_pin_every_panel_before_annotations(self):
        plot = _plot(y_limits=(0.0, 9.0), atomic_lines=make_atomic_lines([6.0, 9.0]))
        plot.generate_plot()
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())
        # Annotation labels are anchored at the panel's final Y max.
        labels = [t for ax in plot.subplots.values() for t in ax.texts
                  if hasattr(t, "_islat_atomic_line")]
        assert labels and all(t.get_position()[1] == pytest.approx(9.0) for t in labels)
        plot.close()

    def test_model_source_fits_each_panel_to_its_model(self):
        plot = _plot(y_autoscale_source="model")
        plot.generate_plot()
        for ax in plot.subplots.values():
            x0, x1 = ax.get_xlim()
            sel = (WAVE >= x0) & (WAVE <= x1)
            assert ax.get_ylim()[1] == pytest.approx(MODEL[sel].max() * (1 + 1 / 8), rel=1e-6)
        plot.close()

    def test_model_source_ignores_hidden_summed_and_falls_back_to_data(self):
        plot = _plot(y_autoscale_source="model", show_summed=False)
        plot.generate_plot()
        # No visible model: the data rule (min .. max + max/8) applies.
        for ax in plot.subplots.values():
            x0, x1 = ax.get_xlim()
            in_range = FLUX[(WAVE >= x0) & (WAVE <= x1)]
            assert ax.get_ylim() == pytest.approx((in_range.min(), in_range.max() * (1 + 1 / 8)))
        plot.close()

    def test_manual_limits_win_over_model_source(self):
        plot = _plot(y_autoscale_source="model", y_limits=(-1.0, 2.0))
        plot.generate_plot()
        assert all(ax.get_ylim() == (-1.0, 2.0) for ax in plot.subplots.values())
        plot.close()

    def test_data_sync_global_and_view(self):
        # Panels 5-7 / 7-9 / 9-11 um; the data is 1 Jy below 8 um and 4 Jy above.
        for scope in ("global", "view"):
            plot = _plot(y_data_sync=scope)
            plot.generate_plot()
            assert all(ax.get_ylim() == (1.0, 4.0) for ax in plot.subplots.values())
            plot.close()

    def test_data_sync_view_covers_shown_range_only(self):
        plot = _plot(y_data_sync="view", xlim_range=(5.0, 7.0))
        plot.generate_plot()
        assert all(ax.get_ylim() == pytest.approx((0.95, 1.05)) for ax in plot.subplots.values())
        plot.close()

    def test_data_sync_per_panel(self):
        plot = _plot(y_data_sync="panel")
        plot.generate_plot()
        lims = [plot.subplots[i].get_ylim() for i in sorted(plot.subplots)]
        assert lims[0] == pytest.approx((0.95, 1.05))
        assert lims[1] == pytest.approx((1.0, 4.0))
        assert lims[2] == pytest.approx((3.8, 4.2))
        plot.update_panels_inplace()               # in-place model updates keep it
        assert [plot.subplots[i].get_ylim() for i in sorted(plot.subplots)] == pytest.approx(lims)
        plot.close()

    def test_data_sync_wins_over_model_but_not_manual(self):
        plot = _plot(y_data_sync="global", y_autoscale_source="model")
        plot.generate_plot()
        assert all(ax.get_ylim() == (1.0, 4.0) for ax in plot.subplots.values())
        plot.close()
        plot = _plot(y_data_sync="global", y_limits=(-1.0, 2.0))
        plot.generate_plot()
        assert all(ax.get_ylim() == (-1.0, 2.0) for ax in plot.subplots.values())
        plot.close()

    def test_invalid_data_sync_is_off(self):
        assert _plot(y_data_sync="nope").y_data_sync is None
        assert _plot().y_data_sync is None

    def test_inplace_update_keeps_display_options(self):
        plot = _plot(y_limits=(0.0, 9.0), show_summed=False, show_observed=False)
        plot.generate_plot()
        plot.update_data(WAVE, FLUX * 2)
        plot.update_panels_inplace()
        assert all(ax.get_ylim() == (0.0, 9.0) for ax in plot.subplots.values())
        assert all(not a.get_visible() for a in _tagged(plot, "_islat_summed"))
        assert all(not a.get_visible() for a in _tagged(plot, "_islat_observed"))
        plot.close()


# ======================================================================
# Figure-size overrides
# ======================================================================
class TestSetFigureSize:
    def test_width_and_row_height(self):
        plot = _plot(n_panels=4, figsize=(12, 16))
        assert plot.set_figure_size(width=30) == (30.0, 16.0)
        assert plot.set_figure_size(row_height=2.5) == (30.0, 10.0)
        plot.generate_plot()
        assert tuple(plot.fig.get_size_inches()) == pytest.approx((30.0, 10.0))
        plot.close()

    def test_row_height_wins_over_total_height(self):
        plot = _plot(n_panels=2)
        assert plot.set_figure_size(height=20, row_height=3) == pytest.approx((12.0, 6.0))
        assert plot.set_figure_size(height=20) == pytest.approx((12.0, 20.0))
        plot.close()

    def test_resizes_existing_figure(self):
        plot = _plot()
        plot.generate_plot()
        plot.set_figure_size(width=25)
        assert plot.fig.get_size_inches()[0] == pytest.approx(25.0)
        plot.close()

    @pytest.mark.parametrize("bad", [0, -2, float("nan"), float("inf")])
    def test_rejects_non_positive(self, bad):
        plot = _plot()
        with pytest.raises(ValueError):
            plot.set_figure_size(width=bad)
        with pytest.raises(ValueError):
            plot.set_figure_size(row_height=bad)
        plot.close()


class TestExportSettings:
    def test_defaults_without_settings(self):
        assert get_export_settings(SimpleNamespace()) == {"width": None, "panel_height": None}
        assert get_export_settings(None) == {"width": None, "panel_height": None}
        assert get_export_settings(MagicMock()) == {"width": None, "panel_height": None}

    def test_set_round_trips_and_blank_means_default(self):
        pm = SimpleNamespace()
        assert set_export_settings(pm, "30", " ") == {"width": 30.0, "panel_height": None}
        assert get_export_settings(pm) == {"width": 30.0, "panel_height": None}
        set_export_settings(pm, None, 2.5)
        assert get_export_settings(pm) == {"width": None, "panel_height": 2.5}

    @pytest.mark.parametrize("width, height", [
        ("abc", None), ("-1", None), ("0", None), ("1e6", None), (None, "-3"), (None, "500"),
    ])
    def test_invalid_values_raise_and_keep_previous(self, width, height):
        pm = SimpleNamespace()
        set_export_settings(pm, 20, 2)
        with pytest.raises(ValueError):
            set_export_settings(pm, width, height)
        assert get_export_settings(pm) == {"width": 20.0, "panel_height": 2.0}

    def test_invalid_stored_values_fall_back_to_defaults(self):
        pm = SimpleNamespace(full_spectrum_export_settings={"width": -5, "panel_height": "x"})
        assert get_export_settings(pm) == {"width": None, "panel_height": None}

    def test_apply_export_size(self):
        plot = _plot(n_panels=5)
        assert apply_export_size(plot, None) == (DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT)
        assert apply_export_size(plot, {"width": 40.0, "panel_height": None}) == (40.0, DEFAULT_EXPORT_HEIGHT)
        assert apply_export_size(plot, {"width": None, "panel_height": 2.0}) == (DEFAULT_EXPORT_WIDTH, 10.0)
        plot.close()


class TestToggleStateDisplayKwargs:
    def test_defaults(self):
        assert toggle_state_display_kwargs({}) == {
            "show_observed": True, "show_summed": True, "y_autoscale_source": "data",
        }
        assert toggle_state_display_kwargs(None)["show_observed"] is True

    def test_observed_summed_and_model_source(self):
        kw = toggle_state_display_kwargs(
            {"observed": False, "summed": False, "y_autoscale": True,
             "y_autoscale_source": "model", "y_limits": (0.0, 5.0)})
        assert kw == {"show_observed": False, "show_summed": False, "y_autoscale_source": "model"}

    def test_manual_limits_when_autoscale_off(self):
        kw = toggle_state_display_kwargs({"y_autoscale": False, "y_limits": [0, 5]})
        assert kw["y_limits"] == (0.0, 5.0)

    def test_autoscale_off_without_limits_uses_layout_rule(self):
        kw = toggle_state_display_kwargs({"y_autoscale": False, "y_limits": None,
                                          "y_autoscale_source": "model"})
        assert "y_limits" not in kw
        assert kw["y_autoscale_source"] == "data"

    def test_data_sync_only_while_autoscaling(self):
        assert toggle_state_display_kwargs({"y_data_sync": "Panel"})["y_data_sync"] == "panel"
        assert "y_data_sync" not in toggle_state_display_kwargs({"y_data_sync": None})
        assert "y_data_sync" not in toggle_state_display_kwargs({"y_data_sync": "off"})
        kw = toggle_state_display_kwargs({"y_autoscale": False, "y_limits": (0, 5), "y_data_sync": "global"})
        assert "y_data_sync" not in kw and kw["y_limits"] == (0.0, 5.0)


# ======================================================================
# End-to-end export paths
# ======================================================================
@pytest.fixture
def recorder(monkeypatch):
    """Swap FullSpectrumView's FullSpectrumPlot for one that snapshots itself on close."""
    snapshots = []

    class _Recording(FullSpectrumPlot):
        def close(self):
            if self.fig is not None:
                snapshots.append({
                    "size": tuple(self.fig.get_size_inches()),
                    "xlims": [ax.get_xlim() for ax in self.subplots.values()],
                    "ylims": [ax.get_ylim() for ax in self.subplots.values()],
                    "observed": [a.get_visible() for a in _tagged(self, "_islat_observed")],
                    "summed": [a.get_visible() for a in _tagged(self, "_islat_summed")],
                })
            super().close()

    monkeypatch.setattr(fsv, "FullSpectrumPlot", _Recording)
    monkeypatch.setattr(fsv.ifh, "write_molecules_to_csv", lambda *a, **k: None)
    return snapshots


def _toggle_state(**overrides):
    ts = {"atomic_lines": False, "saved_lines": False, "summed": True, "legend": True,
          "observed": True, "y_autoscale": True, "y_autoscale_source": "data", "y_limits": None}
    ts.update(overrides)
    return ts


class TestOutputFullSpectrum:
    @staticmethod
    def _islat(toggle_state, export_settings=None):
        gui_plot = SimpleNamespace(toggle_state=toggle_state, theme=None)
        if export_settings is not None:
            gui_plot.full_spectrum_export_settings = export_settings
        return SimpleNamespace(
            wave_data_original=WAVE, flux_data=FLUX, molecules_dict=_Molecules(),
            loaded_spectrum_file="spec.csv", input_line_list="unused.csv",
            GUI=SimpleNamespace(plot=gui_plot, data_field=MagicMock()),
        )

    @pytest.fixture
    def save_to(self, monkeypatch, tmp_path):
        target = tmp_path / "out.pdf"
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: str(target))
        return target

    def test_syncs_observed_and_manual_y_limits(self, recorder, save_to):
        islat = self._islat(_toggle_state(observed=False, summed=False,
                                          y_autoscale=False, y_limits=(0.0, 9.0)))
        assert fsv.output_full_spectrum(islat) == str(save_to)
        assert save_to.exists()
        (snap,) = recorder
        assert snap["observed"] and not any(snap["observed"])
        assert snap["summed"] and not any(snap["summed"])
        assert all(yl == (0.0, 9.0) for yl in snap["ylims"])

    def test_model_autoscale_source(self, recorder, save_to):
        fsv.output_full_spectrum(self._islat(_toggle_state(y_autoscale_source="model")))
        (snap,) = recorder
        assert all(yl[1] < 1.0 for yl in snap["ylims"])  # model peaks at 0.2 Jy, data at 4 Jy

    def test_data_sync_global(self, recorder, save_to):
        fsv.output_full_spectrum(self._islat(_toggle_state(y_data_sync="global")))
        (snap,) = recorder
        assert snap["ylims"] and all(yl == (1.0, 4.0) for yl in snap["ylims"])

    def test_default_size(self, recorder, save_to):
        fsv.output_full_spectrum(self._islat(_toggle_state()))
        assert recorder[0]["size"] == pytest.approx((DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT))

    def test_uses_session_export_settings(self, recorder, save_to):
        islat = self._islat(_toggle_state(), {"width": 36.0, "panel_height": 2.0})
        fsv.output_full_spectrum(islat)
        assert recorder[0]["size"] == pytest.approx((36.0, 2.0 * len(recorder[0]["ylims"])))

    def test_explicit_overrides_win(self, recorder, save_to):
        islat = self._islat(_toggle_state(), {"width": 36.0, "panel_height": None})
        fsv.output_full_spectrum(islat, figure_width=50)
        assert recorder[0]["size"] == pytest.approx((50.0, DEFAULT_EXPORT_HEIGHT))

    def test_cancelled_dialog_saves_nothing(self, recorder, monkeypatch):
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: "")
        assert fsv.output_full_spectrum(self._islat(_toggle_state())) is None


class TestViewSaveFigure:
    @staticmethod
    def _view(toggle_state, export_settings=None):
        pm = MagicMock()
        pm.islat.wave_data_original = WAVE
        pm.islat.flux_data = FLUX
        pm.islat.molecules_dict = _Molecules()
        pm.theme = None
        pm.toggle_state = toggle_state
        pm.full_spectrum_export_settings = export_settings or {"width": None, "panel_height": None}
        return fsv.FullSpectrumView(pm)

    def test_save_syncs_toggles_and_size(self, recorder, tmp_path):
        view = self._view(_toggle_state(observed=False, y_autoscale=False, y_limits=(0.0, 9.0)),
                          {"width": 30.0, "panel_height": 1.5})
        out = view.save_figure(str(tmp_path / "view.pdf"))
        assert out and (tmp_path / "view.pdf").exists()
        (snap,) = recorder
        assert snap["observed"] and not any(snap["observed"])
        assert all(yl == (0.0, 9.0) for yl in snap["ylims"])
        assert snap["size"] == pytest.approx((30.0, 1.5 * len(snap["ylims"])))


# ======================================================================
# Multi-page PDF export
# ======================================================================
def _pdf_page_count(path):
    """Pages in a PDF written by matplotlib (its page objects, not the page tree)."""
    import re
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


class TestRowsPerPage:
    @pytest.mark.parametrize("value, expected", [
        (8, [8, 8, 8]),
        ("8", [8, 8, 8]),
        (" 4, 6 10 ", [4, 6, 10]),
        ("4;6;10", [4, 6, 10]),
        ([4, 6, 10], [4, 6, 10]),
        ((5.0,), [5, 5, 5]),
    ])
    def test_one_count_for_every_page_or_one_per_page(self, value, expected):
        assert rows_per_page_list(value, 3) == expected

    @pytest.mark.parametrize("value", [
        "", "abc", "2.5", "0", str(MAX_ROWS_PER_PAGE + 1), [1, 2], "1, 2, 3, 4", [3, None, 3],
    ])
    def test_invalid_counts_raise(self, value):
        with pytest.raises(ValueError):
            rows_per_page_list(value, 3)


class TestPageRanges:
    def test_valid_ranges_become_floats(self):
        assert validate_page_ranges([(5, 6), ("7", 8.5)]) == [(5.0, 6.0), (7.0, 8.5)]

    @pytest.mark.parametrize("ranges", [
        [], [(6.0, 5.0)], [(5.0, 5.0)], [(5.0, float("nan"))],
        [(i, i + 1) for i in range(MAX_EXPORT_PAGES + 1)],
    ])
    def test_invalid_ranges_raise(self, ranges):
        with pytest.raises(ValueError):
            validate_page_ranges(ranges)

    def test_observed_bounds_are_rest_frame_and_skip_nans(self):
        class _Shifted(_Molecules):
            def apply_stellar_rv(self, wave):
                return np.asarray(wave) - 0.5

        wave = np.array([np.nan, 5.0, 6.0, 11.0, np.nan])
        islat = SimpleNamespace(wave_data_original=wave, molecules_dict=_Shifted())
        assert observed_wavelength_bounds(islat) == (4.5, 10.5)
        assert observed_wavelength_bounds(SimpleNamespace()) == (None, None)


class TestPageSettings:
    def test_unset_entries_are_blank(self):
        assert get_page_settings(SimpleNamespace()) == {"pages": "", "rows": ""}
        assert get_page_settings(None) == {"pages": "", "rows": ""}
        assert get_page_settings(MagicMock()) == {"pages": "", "rows": ""}

    def test_round_trip_keeps_the_entries_as_typed(self):
        pm = SimpleNamespace()
        set_page_settings(pm, "DATASTART 8\n8 DATAEND", " 8, 6 ")
        assert get_page_settings(pm) == {"pages": "DATASTART 8\n8 DATAEND", "rows": " 8, 6 "}
        set_page_settings(pm, None, None)
        assert get_page_settings(pm) == {"pages": "", "rows": ""}


class TestOutputFullSpectrumPages:
    PAGES = [(5.0, 7.0), (7.0, 9.0), (9.0, 11.0)]

    @pytest.fixture
    def save_to(self, monkeypatch, tmp_path):
        target = tmp_path / "pages.pdf"
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: str(target))
        return target

    @pytest.fixture
    def parameter_files(self, recorder, monkeypatch):
        """Where the companion parameter CSVs were written (after the recorder's no-op)."""
        written = []
        monkeypatch.setattr(fsv.ifh, "write_molecules_to_csv",
                            lambda _mols, file_path, file_name: written.append(file_name))
        return written

    def test_one_page_per_range_split_into_rows(self, recorder, save_to):
        islat = TestOutputFullSpectrum._islat(_toggle_state())
        assert fsv.output_full_spectrum_pages(islat, self.PAGES, 4) == str(save_to)
        assert _pdf_page_count(save_to) == 3
        assert [len(snap["xlims"]) for snap in recorder] == [4, 4, 4]
        for snap, (lo, hi) in zip(recorder, self.PAGES):
            assert snap["xlims"][0][0] == pytest.approx(lo)
            assert snap["xlims"][-1][1] == pytest.approx(hi)

    def test_rows_per_page_list(self, recorder, save_to):
        fsv.output_full_spectrum_pages(TestOutputFullSpectrum._islat(_toggle_state()), self.PAGES, "2, 3, 5")
        assert [len(snap["xlims"]) for snap in recorder] == [2, 3, 5]

    def test_pages_follow_toggles_and_panel_height(self, recorder, save_to):
        islat = TestOutputFullSpectrum._islat(
            _toggle_state(observed=False, y_autoscale=False, y_limits=(0.0, 9.0)),
            {"width": 30.0, "panel_height": 1.5},
        )
        fsv.output_full_spectrum_pages(islat, self.PAGES[:2], [2, 4])
        assert [snap["size"] for snap in recorder] == [pytest.approx((30.0, 3.0)), pytest.approx((30.0, 6.0))]
        for snap in recorder:
            assert snap["observed"] and not any(snap["observed"])
            assert all(yl == (0.0, 9.0) for yl in snap["ylims"])

    def test_writes_parameters_and_reports(self, recorder, save_to, parameter_files):
        islat = TestOutputFullSpectrum._islat(_toggle_state())
        fsv.output_full_spectrum_pages(islat, self.PAGES, 2)
        assert parameter_files == [str(save_to.with_suffix("")) + "_parameters.csv"]
        islat.GUI.data_field.insert_text.assert_called_once()
        assert "(3 pages)" in islat.GUI.data_field.insert_text.call_args.args[0]

    def test_given_path_skips_the_dialog(self, recorder, monkeypatch, tmp_path):
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename",
                            lambda **_k: pytest.fail("the dialog was opened"))
        out = tmp_path / "given.pdf"
        islat = TestOutputFullSpectrum._islat(_toggle_state())
        assert fsv.output_full_spectrum_pages(islat, self.PAGES[:1], 3, save_path=str(out)) == str(out)
        assert _pdf_page_count(out) == 1

    def test_cancelled_dialog_draws_nothing(self, recorder, monkeypatch):
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **_k: "")
        assert fsv.output_full_spectrum_pages(TestOutputFullSpectrum._islat(_toggle_state()), self.PAGES, 2) is None
        assert recorder == []

    @pytest.mark.parametrize("pages, rows", [([], 2), ([(7.0, 5.0)], 2), (PAGES, "2, 3")])
    def test_invalid_input_raises_before_the_dialog(self, recorder, monkeypatch, pages, rows):
        import tkinter.filedialog
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename",
                            lambda **_k: pytest.fail("the dialog was opened"))
        with pytest.raises(ValueError):
            fsv.output_full_spectrum_pages(TestOutputFullSpectrum._islat(_toggle_state()), pages, rows)
        assert recorder == []
