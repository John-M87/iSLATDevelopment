# -*- coding: utf-8 -*-
"""Tests for OpticalDepthPanel and OpticalDepthSpectrumPlot."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes

from iSLAT.Modules.Plotting import OpticalDepthPanel, OpticalDepthSpectrumPlot

from tests.plotting import make_wave_flux


def _make_tau_cache(wave, n_molecules=1):
    """Build a synthetic mol_tau_cache for testing."""
    colors = ["blue", "red", "green", "orange", "purple"]
    cache = []
    for i in range(n_molecules):
        tau = 0.1 * (i + 1) * np.abs(np.sin(
            2 * np.pi * (wave - wave[0]) / (wave[-1] - wave[0]) * (i + 1)
        ))
        cache.append((
            wave, tau,
            colors[i % len(colors)],
            f"Mol{i}",
            f"mol_{i}",
        ))
    return cache


class TestOpticalDepthPanel:
    """Tests for the single-panel OpticalDepthPanel."""

    def _make_panel(self, display_mode="profile", log_scale=False,
                    n_molecules=1, **kwargs):
        wave, flux, err = make_wave_flux(n=200)
        tau_cache = _make_tau_cache(wave, n_molecules=n_molecules)
        return OpticalDepthPanel(
            wave, flux,
            xmin=float(wave[0]), xmax=float(wave[-1]),
            error_data=err,
            mol_tau_cache=tau_cache,
            display_mode=display_mode,
            log_scale=log_scale,
            **kwargs,
        )

    def test_generate_profile_mode(self):
        panel = self._make_panel(display_mode="profile")
        panel.generate_plot()
        assert panel.ax is not None
        panel.close()

    def test_generate_stem_mode(self):
        """Stem mode with no molecules still renders without error."""
        wave, flux, err = make_wave_flux(n=200)
        panel = OpticalDepthPanel(
            wave, flux,
            xmin=float(wave[0]), xmax=float(wave[-1]),
            display_mode="stem",
        )
        panel.generate_plot()
        assert panel.ax is not None
        panel.close()

    def test_multiple_molecules_profile(self):
        panel = self._make_panel(n_molecules=3)
        panel.generate_plot()
        # Should have fills rendered (check that axes has collections)
        assert len(panel.ax.collections) > 0 or len(panel.ax.lines) > 0
        panel.close()

    def test_stacked_fill_false(self):
        panel = self._make_panel(n_molecules=2, stacked_fill=False)
        panel.generate_plot()
        # Should have line artists instead of fills
        assert panel.ax is not None
        panel.close()

    def test_show_total_line(self):
        panel = self._make_panel(n_molecules=2, show_total=True)
        panel.generate_plot()
        assert panel.ax is not None
        panel.close()

    def test_log_scale(self):
        panel = self._make_panel(log_scale=True)
        panel.generate_plot()
        assert panel.ax.get_yscale() == "log"
        panel.close()

    def test_linear_scale(self):
        panel = self._make_panel(log_scale=False)
        panel.generate_plot()
        assert panel.ax.get_yscale() == "linear"
        panel.close()

    def test_compute_ylim_default(self):
        panel = self._make_panel()
        ymin, ymax = panel.compute_ylim()
        assert ymin == 0.0
        assert ymax > 0.0

    def test_compute_ylim_log(self):
        panel = self._make_panel(log_scale=True)
        ymin, ymax = panel.compute_ylim()
        assert ymin > 0.0  # Floor for log scale
        assert ymax > ymin

    def test_compute_ylim_empty_cache(self):
        wave, flux, err = make_wave_flux(n=200)
        panel = OpticalDepthPanel(
            wave, flux,
            xmin=float(wave[0]), xmax=float(wave[-1]),
            mol_tau_cache=[],
        )
        ymin, ymax = panel.compute_ylim()
        assert ymax > 0  # Should provide a sensible default

    def test_ylabel_set(self):
        panel = self._make_panel()
        panel.generate_plot()
        ylabel = panel.ax.get_ylabel()
        assert "\u03c4" in ylabel or "Optical" in ylabel
        panel.close()


class TestOpticalDepthSpectrumPlot:
    """Tests for the multi-panel stacked OpticalDepthSpectrumPlot."""

    def _make_odsp(self, n_panels=3, **kwargs):
        wave, flux, err = make_wave_flux(n=200)
        return OpticalDepthSpectrumPlot(
            wave, flux, error_data=err, n_panels=n_panels, **kwargs,
        )

    def test_generate_plot(self):
        odsp = self._make_odsp()
        odsp.generate_plot()
        assert odsp.fig is not None
        assert len(odsp.panels) == 3
        for idx, panels in odsp.panels.items():
            assert len(panels) == 1
            assert isinstance(panels[0], OpticalDepthPanel)
        odsp.close()

    def test_subplots_populated(self):
        odsp = self._make_odsp()
        odsp.generate_plot()
        assert len(odsp.subplots) == 3
        for idx in range(3):
            assert isinstance(odsp.subplots[idx], Axes)
        odsp.close()

    def test_panel_count(self):
        for n in (2, 5, 8):
            odsp = self._make_odsp(n_panels=n)
            odsp.generate_plot()
            assert len(odsp.panels) == n
            odsp.close()

    def test_log_scale_propagated(self):
        odsp = self._make_odsp(log_scale=True)
        odsp.generate_plot()
        for panels in odsp.panels.values():
            assert panels[0].ax.get_yscale() == "log"
        odsp.close()

    def test_display_mode_propagated(self):
        odsp = self._make_odsp(display_mode="stem")
        odsp.generate_plot()
        for panels in odsp.panels.values():
            assert panels[0].display_mode == "stem"
        odsp.close()

    def test_save_and_close(self, tmp_path):
        odsp = self._make_odsp()
        odsp.generate_plot()
        out = odsp.save(tmp_path / "odsp.png", dpi=72)
        assert out.exists()
        odsp.close()
        assert odsp.fig is None

    def test_uniform_ylim(self):
        odsp = self._make_odsp(n_panels=3, uniform_ylim=True)
        odsp.generate_plot()
        ylims = [
            list(odsp.panels.values())[i][0].ax.get_ylim()
            for i in odsp.panels
        ]
        for yl in ylims[1:]:
            assert yl == pytest.approx(ylims[0], abs=1e-6)
        odsp.close()

    def test_composite_stacking(self):
        """OpticalDepthSpectrumPlot + FullSpectrumPlot via stack_with."""
        from iSLAT.Modules.Plotting import FullSpectrumPlot
        wave, flux, err = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, error_data=err, n_panels=3)
        odsp = OpticalDepthSpectrumPlot(
            wave, flux, error_data=err, n_panels=3,
        )
        composite = fsp + odsp
        composite.generate_plot()
        assert composite.fig is not None
        # Should have panels from both sources
        assert len(composite.panels) >= 3
        composite.close()


class TestOpticalDepthTagsAndLabels:
    """Artist tags and labels the Optical Depth view relies on."""

    def test_total_tau_line_is_tagged(self):
        from iSLAT.Modules.Plotting.OpticalDepthPanel import TOTAL_TAU_TAG
        wave, flux, err = make_wave_flux(n=200)
        panel = OpticalDepthPanel(
            wave, flux, xmin=float(wave[0]), xmax=float(wave[-1]),
            mol_tau_cache=_make_tau_cache(wave, n_molecules=2), show_total=True,
        )
        panel.generate_plot()
        tagged = [ln for ln in panel.ax.lines if hasattr(ln, TOTAL_TAU_TAG)]
        assert len(tagged) == 1 and tagged[0].get_label() == "Total \u03c4"
        panel.close()

    def test_stacked_plot_y_label_is_optical_depth(self):
        wave, flux, err = make_wave_flux(n=200)
        odsp = OpticalDepthSpectrumPlot(wave, flux, n_panels=3)
        odsp.generate_plot()
        assert odsp.fig._supylabel.get_text() == "Optical Depth (\u03c4)"
        assert all(ax.get_ylabel() == "" for ax in odsp.subplots.values())
        odsp.close()

    def test_flux_plots_keep_flux_label(self):
        from iSLAT.Modules.Plotting import FullSpectrumPlot
        wave, flux, err = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, n_panels=2)
        fsp.generate_plot()
        assert fsp.fig._supylabel.get_text() == "Flux Density (Jy)"
        fsp.close()


class TestOpticalDepthUpdateData:
    def test_keeps_range_and_reports_layout_change(self):
        wave, flux, err = make_wave_flux(n=200)
        odsp = OpticalDepthSpectrumPlot(wave, flux, n_panels=4, xlim_range=(12.0, 16.0))
        new_flux = flux * 2
        assert odsp.update_data(wave, new_flux) is False
        assert (odsp._xlim_start, odsp._xlim_end) == (12.0, 16.0)
        assert np.array_equal(odsp.flux_data, new_flux)
        odsp.n_panels = 2
        assert odsp.update_data(wave, new_flux) is True
        assert len(odsp._panel_edges) == 2


class TestOpticalDepthStemLimits:
    """Stem mode scales each panel to its per-line τ, not the convolved profile."""

    @staticmethod
    def _molecules():
        from tests.plotting.test_plot_views import _make_test_molecule

        class _Dict(dict):
            def get_visible_molecules(self, return_objects=False):
                return list(self.values()) if return_objects else list(self)

            def get_matched_sampling_wavelengths(self, w):
                return False, w

        return _Dict(H2O=_make_test_molecule())

    def _stem_plot(self, **kwargs):
        wave = np.linspace(10.0, 20.0, 2000)
        odsp = OpticalDepthSpectrumPlot(
            wave, np.ones_like(wave), molecules=self._molecules(),
            n_panels=3, display_mode="stem", **kwargs,
        )
        odsp.generate_plot()
        return odsp

    def test_limits_fit_stems(self):
        odsp = self._stem_plot()
        checked = 0
        for cell in odsp.panels.values():
            peak = cell[0].stem_tau_max()
            if peak > 0:
                checked += 1
                assert cell[0].ax.get_ylim() == pytest.approx((0.0, peak * 1.1))
        assert checked > 0
        odsp.close()

    def test_uniform_limits_share_tallest_stem(self):
        odsp = self._stem_plot(uniform_ylim=True)
        top = max(cell[0].stem_tau_max() for cell in odsp.panels.values())
        for cell in odsp.panels.values():
            assert cell[0].ax.get_ylim() == pytest.approx((0.0, top * 1.1))
        odsp.close()

    def test_log_stems_start_at_floor(self):
        from iSLAT.Modules.Plotting.OpticalDepthPanel import TAU_STEM_TAG
        odsp = self._stem_plot(log_scale=True, tau_floor=1e-6)
        bottoms = [
            seg[:, 1].min()
            for cell in odsp.panels.values()
            for coll in cell[0].ax.collections if hasattr(coll, TAU_STEM_TAG)
            for seg in coll.get_segments()
        ]
        assert bottoms and min(bottoms) == pytest.approx(1e-6)
        odsp.close()


def test_stacked_panels_use_the_plot_theme():
    """The total τ line follows the plot's theme foreground (e.g. light on a dark theme)."""
    from iSLAT.Modules.Plotting.OpticalDepthPanel import TOTAL_TAU_TAG
    from tests.plotting.test_plot_views import _make_test_molecule
    molecules = TestOpticalDepthStemLimits._molecules()
    molecules["CO2"] = _make_test_molecule(name="CO2", lam_start=12.3)
    wave = np.linspace(10.0, 20.0, 2000)
    odsp = OpticalDepthSpectrumPlot(
        wave, np.ones_like(wave), molecules=molecules, n_panels=2, theme={"foreground": "#eeeeee"},
    )
    odsp.generate_plot()
    assert all(cell[0].theme is odsp.theme for cell in odsp.panels.values())
    totals = [ln for ax in odsp.subplots.values() for ln in ax.lines if hasattr(ln, TOTAL_TAU_TAG)]
    assert totals and {ln.get_color() for ln in totals} == {"#eeeeee"}
    odsp.close()


class TestOpticalDepthFluxSubPanels:
    """``show_flux=True``: a flux sub-panel below every τ panel (residual-plot style)."""

    @staticmethod
    def _plot(**kwargs):
        from tests.plotting.test_plot_views import _make_test_molecule

        class _Dict(dict):
            def get_visible_molecules(self, return_objects=False):
                return list(self.values()) if return_objects else list(self)

            def get_matched_sampling_wavelengths(self, w):
                return False, w

            def get_summed_flux(self, w, visible_only=True, extend_to_global_range=False):
                return self["H2O"].get_flux(return_wavelengths=True)

        wave = np.linspace(10.0, 20.0, 2000)
        flux = np.full_like(wave, 0.05)
        return OpticalDepthSpectrumPlot(
            wave, flux, molecules=_Dict(H2O=_make_test_molecule()),
            error_data=np.full_like(wave, 0.005), n_panels=3, show_flux=True, **kwargs,
        )

    def test_layout_pairs_tau_and_flux(self):
        from iSLAT.Modules.Plotting import SpectrumPanel
        odsp = self._plot()
        odsp.generate_plot()
        assert set(odsp.subplots) == {0, 1, 2}
        for idx, cell in odsp.panels.items():
            assert isinstance(cell[0], OpticalDepthPanel) and isinstance(cell[1], SpectrumPanel)
            tau_ax, flux_ax = odsp.subplots[idx]
            assert (cell[0].ax, cell[1].ax) == (tau_ax, flux_ax)
            assert flux_ax.get_shared_x_axes().joined(tau_ax, flux_ax)
            # τ above flux, touching (no gap), wavelength ticks under the flux panel only
            assert tau_ax.get_position().y0 == pytest.approx(flux_ax.get_position().y1)
            assert not any(t.get_visible() for t in tau_ax.get_xticklabels())
            assert (tau_ax.get_ylabel(), flux_ax.get_ylabel()) == ("\u03c4", "Jy")
        assert odsp.get_tau_axes() == {i: v[0] for i, v in odsp.subplots.items()}
        assert odsp.get_flux_axes() == {i: v[1] for i, v in odsp.subplots.items()}
        # Each sub-panel is labelled; a figure-wide label would collide with them.
        assert odsp.fig._supylabel.get_text() == ""
        assert odsp.subplots[2][1].get_xlabel().startswith("Wavelength")
        odsp.close()

    def test_flux_panels_draw_spectrum_models_and_sum(self):
        odsp = self._plot()
        odsp.generate_plot()
        expected = odsp.compute_flux_ylims()
        for idx, flux_ax in odsp.get_flux_axes().items():
            assert any(hasattr(a, "_islat_observed") for a in flux_ax.lines)
            assert flux_ax.get_ylim() == pytest.approx(expected[idx])
        models = {getattr(ln, "_molecule_name", None) for ax in odsp.get_flux_axes().values() for ln in ax.lines}
        assert "H2O" in models
        assert any(hasattr(c, "_islat_summed") for ax in odsp.get_flux_axes().values() for c in ax.collections)
        odsp.close()

    def test_height_ratio_and_auto_figsize(self):
        odsp = self._plot(flux_height_ratio=0.5)
        assert odsp._figsize == pytest.approx((12, 1.6 * 1.5 * 3))
        odsp.generate_plot()
        tau_ax, flux_ax = odsp.subplots[0]
        assert flux_ax.get_position().height == pytest.approx(0.5 * tau_ax.get_position().height)
        odsp.close()

    @pytest.mark.parametrize("ratio", [0, -1, float("nan")])
    def test_invalid_height_ratio(self, ratio):
        with pytest.raises(ValueError):
            self._plot(flux_height_ratio=ratio)

    def test_without_flux_axes_accessors(self):
        wave, flux, err = make_wave_flux(n=200)
        odsp = OpticalDepthSpectrumPlot(wave, flux, n_panels=2)
        odsp.generate_plot()
        assert odsp.get_flux_axes() == {}
        assert odsp.get_tau_axes() == odsp.subplots
        odsp.close()


class TestFractionMode:
    """``display_mode="fraction"``: each molecule's share of the total τ per pixel."""

    @staticmethod
    def _panel(cache, **kwargs):
        wave, flux, err = make_wave_flux(n=200)
        return OpticalDepthPanel(
            wave, flux, xmin=float(wave[0]), xmax=float(wave[-1]),
            mol_tau_cache=cache, display_mode="fraction", **kwargs,
        )

    def test_shares_are_tau_over_total(self):
        wave, flux, err = make_wave_flux(n=200)
        cache = _make_tau_cache(wave, n_molecules=3)
        panel = self._panel(cache)
        lam, shares = panel.compute_fractional_shares()
        assert np.array_equal(lam, wave)
        assert [s[3] for s in shares] == ["mol_0", "mol_1", "mol_2"]
        total = sum(entry[1] for entry in cache)
        defined = total > panel.fraction_min_tau
        for (share, *_), entry in zip(shares, cache):
            assert np.allclose(share[defined], entry[1][defined] / total[defined])
            assert np.all((share[defined] >= 0) & (share[defined] <= 1))
        assert np.allclose(np.sum([s[0] for s in shares], axis=0)[defined], 1.0)

    def test_negligible_total_tau_is_undefined(self):
        wave, flux, err = make_wave_flux(n=200)
        cache = _make_tau_cache(wave, n_molecules=2)  # sin(0): total τ is 0 at the first pixel
        lam, shares = self._panel(cache, fraction_min_tau=0.05).compute_fractional_shares()
        total = cache[0][1] + cache[1][1]
        for share, *_ in shares:
            assert np.all(np.isnan(share[total <= 0.05]))
            assert not np.any(np.isnan(share[total > 0.05]))

    def test_molecules_on_other_grids_are_interpolated(self):
        wave, flux, err = make_wave_flux(n=200)
        coarse = np.linspace(wave[0], wave[-1], 50)
        cache = [
            (wave, np.ones_like(wave), "blue", "A", "a"),
            (coarse, np.full_like(coarse, 3.0), "red", "B", "b"),
        ]
        lam, shares = self._panel(cache).compute_fractional_shares()
        assert len(lam) == len(wave)
        assert np.allclose(shares[0][0], 0.25) and np.allclose(shares[1][0], 0.75)

    def test_stacked_fills_reach_one(self):
        wave, flux, err = make_wave_flux(n=200)
        panel = self._panel(_make_tau_cache(wave, n_molecules=3))
        panel.generate_plot()
        fills = panel.ax.collections
        assert len(fills) == 3
        top = max(np.nanmax(path.vertices[:, 1]) for path in fills[-1].get_paths())
        assert top == pytest.approx(1.0)
        assert panel.ax.get_ylim() == pytest.approx((0.0, 1.05))
        assert panel.ax.get_ylabel() == "Fraction of total \u03c4"
        assert not [ln for ln in panel.ax.lines]  # no total line: it would be 1
        panel.close()

    def test_unstacked_draws_one_share_line_per_molecule(self):
        wave, flux, err = make_wave_flux(n=200)
        panel = self._panel(_make_tau_cache(wave, n_molecules=2), stacked_fill=False)
        panel.generate_plot()
        assert [ln.get_label() for ln in panel.ax.lines] == ["Mol0", "Mol1"]
        assert np.nanmax(panel.ax.lines[0].get_ydata()) <= 1.0
        panel.close()

    def test_axis_stays_linear_with_log_scale(self):
        wave, flux, err = make_wave_flux(n=200)
        panel = self._panel(_make_tau_cache(wave, n_molecules=2), log_scale=True)
        panel.generate_plot()
        assert panel.ax.get_yscale() == "linear"
        assert panel.compute_ylim() == (0.0, 1.05)
        panel.close()

    def test_empty_cache(self):
        lam, shares = self._panel([]).compute_fractional_shares()
        assert lam.size == 0 and shares == []

    def test_stacked_plot_labels_and_limits(self):
        molecules = TestOpticalDepthStemLimits._molecules()
        from tests.plotting.test_plot_views import _make_test_molecule
        molecules["CO2"] = _make_test_molecule(name="CO2", lam_start=12.3)
        wave = np.linspace(10.0, 20.0, 2000)
        for show_flux, sup, tau_label in ((False, "Fraction of total \u03c4", ""), (True, "", r"$\tau_i/\tau$")):
            odsp = OpticalDepthSpectrumPlot(
                wave, np.ones_like(wave), molecules=molecules, n_panels=2,
                display_mode="fraction", log_scale=True, show_flux=show_flux,
            )
            odsp.generate_plot()
            assert odsp.fig._supylabel.get_text() == sup
            for ax in odsp.get_tau_axes().values():
                assert ax.get_ylim() == pytest.approx((0.0, 1.05))
                assert ax.get_yscale() == "linear"
                assert ax.get_ylabel() == tau_label
            odsp.close()
