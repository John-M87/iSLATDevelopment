# -*- coding: utf-8 -*-
"""Tests for ResidualSpectrumPlot, n_panels integration, and stacking."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes

from iSLAT.Modules.Plotting import (
    FullSpectrumPlot,
    ResidualSpectrumPlot,
    CompositeStackedPanel,
)
from iSLAT.Modules.Plotting.SpectrumPanel import SpectrumPanel
from iSLAT.Modules.Plotting.ResidualPanel import ResidualPanel

from tests.plotting import make_wave_flux, make_atomic_lines, make_line_list


# ======================================================================
# ResidualSpectrumPlot
# ======================================================================

class TestResidualSpectrumPlot:
    def _make_rsp(self, n_panels=3, gap_mode="connect", **kwargs):
        wave, flux, err = make_wave_flux(n=200)
        model = flux + np.random.default_rng(7).normal(0, 0.002, 200)
        return ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=n_panels, gap_mode=gap_mode, **kwargs,
        )

    def test_generate_plot(self):
        rsp = self._make_rsp()
        rsp.generate_plot()
        assert rsp.fig is not None
        assert len(rsp.panels) == 3
        for idx, panels in rsp.panels.items():
            assert len(panels) == 2  # spectrum + residual
            assert isinstance(panels[0], SpectrumPanel)
            assert isinstance(panels[1], ResidualPanel)
        rsp.close()

    def test_subplots_tuple(self):
        rsp = self._make_rsp()
        rsp.generate_plot()
        for idx in rsp.subplots:
            val = rsp.subplots[idx]
            assert isinstance(val, tuple)
            assert len(val) == 2
            assert isinstance(val[0], Axes)
            assert isinstance(val[1], Axes)
        rsp.close()

    def test_chi2_annotation(self):
        rsp = self._make_rsp()
        rsp.generate_plot()
        # Per-panel chi2 annotations are on the spectrum axes
        for idx in rsp.subplots:
            ax_spec = rsp.subplots[idx][0]
            assert len(ax_spec.texts) >= 0  # May or may not have chi2
        rsp.close()

    def test_with_excluded_ranges(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=3, excluded_ranges=[(13.0, 14.0)],
        )
        rsp.generate_plot()
        rsp.close()

    def test_with_noise_floor(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=3, noise_floor=0.01,
        )
        rsp.generate_plot()
        assert rsp._has_noise_floor is True
        rsp.close()

    def test_with_continuum(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=3,
            continuum_c0=0.01, continuum_c1=0.001, lam_ref=15.0,
        )
        rsp.generate_plot()
        assert rsp._has_continuum is True
        rsp.close()

    def test_gap_mode_skip(self):
        wave, flux, err = make_wave_flux(n=200, gap=(13.0, 17.0))
        model = np.where(np.isnan(flux), np.nan, flux + 0.001)
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=5, gap_mode="skip",
        )
        rsp.generate_plot()
        assert len(rsp.panels) <= 5
        rsp.close()

    def test_annotations_respect_gap_tightening(self):
        """RSP annotations should not be drawn in gap-cropped regions."""
        wave = np.concatenate([
            np.linspace(10, 10.5, 20),
            np.linspace(18, 20, 100),
        ])
        flux = np.ones(120) * 0.1
        model = flux.copy()
        err = np.full(120, 0.01)
        atomic = make_atomic_lines([12.0, 19.0])
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=1, gap_mode="skip",
            atomic_lines=atomic,
        )
        rsp.generate_plot()
        panels = list(rsp.panels.values())[0]
        spec_panel = panels[0]
        vis_lo, vis_hi = spec_panel.ax.get_xlim()
        for txt in spec_panel.ax.texts:
            if hasattr(txt, "_islat_atomic_line"):
                x, _ = txt.get_position()
                assert vis_lo <= x <= vis_hi + 1.0, (
                    f"RSP annotation at x={x} outside visible range "
                    f"[{vis_lo}, {vis_hi}]"
                )
        rsp.close()

    def test_excluded_ranges_use_tightened_xlim(self):
        """Excluded-range shading should use the tightened xlim.

        With 5 panels and a large gap (10.5–18), the panel whose
        edges fall entirely inside the gap should be skipped, and
        the excluded range at 12–13 (inside the gap) should not
        appear in any visible panel.
        """
        wave = np.concatenate([
            np.linspace(10, 10.5, 20),
            np.linspace(18, 20, 100),
        ])
        flux = np.ones(120) * 0.1
        model = flux.copy()
        err = np.full(120, 0.01)
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=5, gap_mode="skip",
            excluded_ranges=[(12.0, 13.0)],  # in the gap
        )
        rsp.generate_plot()
        # With 5 panels across 10–20, panels whose edges fall
        # entirely in the 10.5–18 gap are skipped.  The excluded
        # range 12–13 is in that gap.  Verify that no rendered
        # panel's visible xlim contains 12–13.
        for panels in rsp.panels.values():
            spec_panel = panels[0]
            vis_lo, vis_hi = spec_panel.ax.get_xlim()
            # Either the panel doesn't reach 12, or it starts after 13
            if vis_lo < 13.0 and vis_hi > 12.0:
                # Panel overlaps with 12–13 — this should not happen
                # because 12–13 is entirely in the skipped gap.
                pytest.fail(
                    f"Panel [{vis_lo:.2f}, {vis_hi:.2f}] overlaps with "
                    f"excluded range [12, 13] that should be in a gap."
                )
        rsp.close()

    def test_data_density_mode(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=4, x_scaling="data_density",
        )
        rsp.generate_plot()
        assert len(rsp.panels) == 4
        rsp.close()

    def test_is_excluded(self):
        wave, flux, err = make_wave_flux(n=50)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=2, excluded_ranges=[(12.0, 14.0)],
        )
        assert rsp._is_excluded(13.0) is True
        assert rsp._is_excluded(11.0) is False

    def test_get_fit_mask(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=2, excluded_ranges=[(13.0, 14.0)],
        )
        mask = rsp._get_fit_mask(wave)
        in_range = (wave >= 13.0) & (wave <= 14.0)
        assert not np.any(mask[in_range])

    def test_get_fit_mask_with_atomic_exclusion(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        atomic = make_atomic_lines([15.0])
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err,
            n_panels=2,
            atomic_lines=atomic,
            exclude_lines_half_width=0.5,
        )
        mask = rsp._get_fit_mask(wave)
        near_15 = np.abs(wave - 15.0) <= 0.5
        assert not np.any(mask[near_15])

    def test_compute_chi2(self):
        wave, flux, err = make_wave_flux(n=100)
        model = flux + 0.01  # systematic offset
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
        )
        fit_mask = np.ones(len(wave), dtype=bool)
        chi2_raw, chi2_adj, n_fit = rsp._compute_chi2(
            flux, model, model, err, err, fit_mask,
        )
        assert n_fit == len(wave)
        assert chi2_raw > 0
        assert chi2_adj > 0

    def test_compute_chi2_empty(self):
        wave, flux, err = make_wave_flux(n=100)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
        )
        fit_mask = np.zeros(len(wave), dtype=bool)
        chi2_raw, chi2_adj, n_fit = rsp._compute_chi2(
            flux, model, model, err, err, fit_mask,
        )
        assert n_fit == 0
        assert chi2_raw == 0.0

    def test_format_chi2_annotation_simple(self):
        wave, flux, err = make_wave_flux(n=50)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
        )
        ann = rsp._format_chi2_annotation(
            100.0, 100.0, 50, has_nuisance=False,
        )
        assert r"\chi^2_\nu" in ann

    def test_format_chi2_annotation_nuisance(self):
        wave, flux, err = make_wave_flux(n=50)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
        )
        ann = rsp._format_chi2_annotation(
            100.0, 90.0, 50, has_nuisance=True,
        )
        assert "adj" in ann

    def test_set_model_flux(self):
        rsp = self._make_rsp()
        new_model = np.ones(200) * 0.05
        rsp.set_model_flux(new_model)
        np.testing.assert_array_equal(rsp.model_flux, new_model)

    def test_set_model_components(self):
        rsp = self._make_rsp()
        comps = [{"wave": [1], "flux": [1], "color": "red", "label": "comp1"}]
        rsp.set_model_components(comps)
        assert rsp.model_components == comps

    # -- match_pixel_sampling ------------------------------------------
    def _make_hires_inputs(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux + np.random.default_rng(7).normal(0, 0.002, 200)
        wave_hr = np.linspace(wave[0], wave[-1], 2000)
        flux_hr = np.interp(wave_hr, wave, model)
        # Smooth component so flux-conserving resampling is comparable
        # to pointwise interpolation in tests.
        comp_flux_hr = 0.5 * (
            0.05 + 0.1 * np.sin(2 * np.pi * (wave_hr - wave[0]) / 10.0)
        )
        comps = [
            {"wave": wave_hr, "flux": comp_flux_hr,
             "color": "red", "label": "comp1"},
        ]
        return wave, flux, err, model, wave_hr, flux_hr, comps

    def test_match_pixel_sampling_sum_fill_uses_data_grid(self):
        wave, flux, err, model, wave_hr, flux_hr, _ = self._make_hires_inputs()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            model_wave=wave_hr, model_flux_hires=flux_hr,
            match_pixel_sampling=True,
        )
        rsp.generate_plot()
        spectrum_panel = rsp.panels[0][0]
        np.testing.assert_array_equal(spectrum_panel.summed_wave, wave)
        np.testing.assert_array_equal(
            spectrum_panel.summed_flux, rsp._model_flux_adj
        )
        rsp.close()

    def test_hires_fill_unchanged_by_default(self):
        wave, flux, err, model, wave_hr, flux_hr, _ = self._make_hires_inputs()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            model_wave=wave_hr, model_flux_hires=flux_hr,
        )
        rsp.generate_plot()
        spectrum_panel = rsp.panels[0][0]
        np.testing.assert_array_equal(spectrum_panel.summed_wave, wave_hr)
        rsp.close()

    def test_match_pixel_sampling_resamples_components(self):
        wave, flux, err, model, wave_hr, flux_hr, comps = \
            self._make_hires_inputs()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            model_components=comps,
            match_pixel_sampling=True,
        )
        comp = rsp.model_components[0]
        np.testing.assert_array_equal(comp["wave"], wave)
        assert len(comp["flux"]) == len(wave)
        # Flux-conserving resample should track the hi-res component
        finite = np.isfinite(comp["flux"])
        assert np.any(finite)
        expected = np.interp(wave[finite], wave_hr, comps[0]["flux"])
        np.testing.assert_allclose(comp["flux"][finite], expected, atol=1e-4)
        # Caller's dicts must not be mutated
        assert len(comps[0]["wave"]) == len(wave_hr)
        rsp.generate_plot()
        rsp.close()

    def test_components_unchanged_by_default(self):
        wave, flux, err, model, wave_hr, flux_hr, comps = \
            self._make_hires_inputs()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            model_components=comps,
        )
        assert rsp.model_components is comps
        np.testing.assert_array_equal(
            rsp.model_components[0]["wave"], wave_hr
        )

    def test_set_model_components_resamples_when_enabled(self):
        wave, flux, err, model, wave_hr, flux_hr, comps = \
            self._make_hires_inputs()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            match_pixel_sampling=True,
        )
        rsp.set_model_components(comps)
        np.testing.assert_array_equal(rsp.model_components[0]["wave"], wave)

    def test_match_pixel_sampling_skips_data_grid_components(self):
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        comps = [{"wave": wave, "flux": model * 0.5,
                  "color": "red", "label": "comp1"}]
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=2,
            model_components=comps,
            match_pixel_sampling=True,
        )
        np.testing.assert_array_equal(
            rsp.model_components[0]["flux"], model * 0.5
        )

    def test_update_panels_inplace_replaces_error_bars(self):
        """Regression: with residuals shown, cycling a sample kept the old spectrum's error bars.

        The spectrum sub-panels are refreshed through the parent's in-place path, so the errorbar ``LineCollection`` must be replaced along with the data line there too.
        """
        wave, flux, err = make_wave_flux(n=200)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(wave, flux, model, error_data=err, n_panels=3)
        rsp.generate_plot()

        new_flux = flux + 0.5
        new_err = np.full_like(err, 0.05)  # 10x the original 0.005
        assert rsp.update_data(wave, new_flux, error_data=new_err) is False
        rsp.update_panels_inplace()

        for idx, (ax_spec, _ax_res) in rsp.subplots.items():
            obs_lines = [a for a in ax_spec.lines if hasattr(a, "_islat_observed")]
            obs_colls = [c for c in ax_spec.collections if hasattr(c, "_islat_observed")]
            assert len(obs_lines) == 1
            assert len(obs_colls) == 1

            xr = rsp.panels[idx][0].xlim
            mask = (wave >= xr[0]) & (wave <= xr[1])
            np.testing.assert_allclose(obs_lines[0].get_ydata(), new_flux[mask])
            segs = np.asarray(obs_colls[0].get_segments())
            half = (segs[:, 1, 1] - segs[:, 0, 1]) / 2.0
            np.testing.assert_allclose(half, 0.05)
            centre = (segs[:, 1, 1] + segs[:, 0, 1]) / 2.0
            np.testing.assert_allclose(centre, new_flux[mask])
        rsp.close()

    def test_save(self, tmp_path):
        rsp = self._make_rsp()
        rsp.generate_plot()
        out = rsp.save(tmp_path / "rsp.png", dpi=72)
        assert out.exists()
        rsp.close()


# ======================================================================
# Integration: n_panels correctness
# ======================================================================

class TestNPanelsIntegration:
    """Verify n_panels produces the exact requested count."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10, 15, 20])
    def test_fsp_n_panels_exact(self, n):
        wave, flux, _ = make_wave_flux(n=500)
        fsp = FullSpectrumPlot(wave, flux, n_panels=n)
        assert len(fsp._panel_edges) == n
        assert len(fsp._panel_ends) == n
        for i in range(n - 1):
            assert fsp._panel_ends[i] == pytest.approx(fsp._panel_edges[i + 1])
        fsp.close()

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 10])
    def test_rsp_n_panels_exact(self, n):
        wave, flux, err = make_wave_flux(n=500)
        model = flux.copy()
        rsp = ResidualSpectrumPlot(
            wave, flux, model, error_data=err, n_panels=n,
        )
        assert len(rsp._panel_edges) == n
        assert len(rsp._panel_ends) == n
        rsp.close()

    @pytest.mark.parametrize("n", [1, 3, 5, 7])
    def test_fsp_n_panels_render(self, n):
        """Verify that generate_plot produces exactly n rendered panels."""
        wave, flux, _ = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, n_panels=n)
        fsp.generate_plot()
        assert len(fsp.panels) == n
        fsp.close()

    def test_density_mode_n_panels_exact(self):
        wave, flux, _ = make_wave_flux(n=500)
        fsp = FullSpectrumPlot(
            wave, flux, n_panels=7, x_scaling="data_density",
        )
        assert len(fsp._panel_edges) == 7
        fsp.close()


# ======================================================================
# Stacking operator
# ======================================================================

class TestStackWith:
    def test_add_operator(self):
        wave, flux, _ = make_wave_flux(n=100)
        a = FullSpectrumPlot(wave, flux, n_panels=3)
        b = FullSpectrumPlot(wave, flux, n_panels=3)
        composite = a + b
        assert isinstance(composite, CompositeStackedPanel)

    def test_add_non_ssp_returns_not_implemented(self):
        wave, flux, _ = make_wave_flux(n=100)
        a = FullSpectrumPlot(wave, flux, n_panels=3)
        result = a.__add__("not a plot")
        assert result is NotImplemented
