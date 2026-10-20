# -*- coding: utf-8 -*-
"""Tests for FullSpectrumPlot."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes

from iSLAT.Modules.Plotting import FullSpectrumPlot
from iSLAT.Modules.Plotting.SpectrumPanel import SpectrumPanel

from tests.plotting import make_wave_flux, make_atomic_lines, make_line_list


class TestFullSpectrumPlot:
    def _make_fsp(self, n_panels=3, gap_mode="connect", **kwargs):
        wave, flux, err = make_wave_flux(n=200)
        return FullSpectrumPlot(
            wave, flux, error_data=err, n_panels=n_panels,
            gap_mode=gap_mode, **kwargs,
        )

    def test_generate_plot(self):
        fsp = self._make_fsp()
        fsp.generate_plot()
        assert fsp.fig is not None
        assert len(fsp.panels) == 3
        for idx, panels in fsp.panels.items():
            assert len(panels) == 1
            assert isinstance(panels[0], SpectrumPanel)
        fsp.close()

    def test_subplots_populated(self):
        fsp = self._make_fsp()
        fsp.generate_plot()
        assert len(fsp.subplots) == 3
        for idx in range(3):
            assert isinstance(fsp.subplots[idx], Axes)
        fsp.close()

    def test_gap_mode_skip_rendering(self):
        wave, flux, err = make_wave_flux(n=200, gap=(13.0, 17.0))
        fsp = FullSpectrumPlot(
            wave, flux, error_data=err, n_panels=5, gap_mode="skip",
        )
        fsp.generate_plot()
        assert fsp.fig is not None
        # Some panels should have been skipped
        assert len(fsp.panels) <= 5
        fsp.close()

    def test_with_annotations(self):
        wave, flux, err = make_wave_flux(n=200)
        atomic = make_atomic_lines([12.0, 15.0, 18.0])
        lines = make_line_list([11.0, 14.0])
        fsp = FullSpectrumPlot(
            wave, flux, error_data=err, n_panels=3,
            atomic_lines=atomic, line_list=lines,
        )
        fsp.generate_plot()
        # Check that some annotations were drawn
        total_texts = 0
        for panels in fsp.panels.values():
            for p in panels:
                total_texts += len(p.ax.texts)
        assert total_texts > 0
        fsp.close()

    def test_annotations_respect_gap_tightening(self):
        """Annotations should not be drawn in gap-cropped regions."""
        wave = np.concatenate([
            np.linspace(10, 10.5, 20),
            np.linspace(18, 20, 100),
        ])
        flux = np.ones(120)
        # Atomic line at 12.0 is in the gap
        atomic = make_atomic_lines([12.0, 19.0])
        fsp = FullSpectrumPlot(
            wave, flux, n_panels=1,
            gap_mode="skip",
            atomic_lines=atomic,
        )
        fsp.generate_plot()
        # Check the single panel's annotations
        panel = list(fsp.panels.values())[0][0]
        vis_lo, vis_hi = panel.ax.get_xlim()
        # All annotation x-positions should be within visible range
        for txt in panel.ax.texts:
            if hasattr(txt, "_islat_atomic_line"):
                x, _ = txt.get_position()
                assert vis_lo <= x <= vis_hi + 1.0, (
                    f"Annotation at x={x} outside visible range "
                    f"[{vis_lo}, {vis_hi}]"
                )
        fsp.close()

    def test_update_data(self):
        fsp = self._make_fsp(n_panels=3)
        fsp.generate_plot()
        new_wave, new_flux, _ = make_wave_flux(wmin=5, wmax=25, n=300)
        changed = fsp.update_data(new_wave, new_flux)
        assert changed is True  # Range changed

    @staticmethod
    def _observed_artists(ax):
        lines = [a for a in ax.lines if hasattr(a, "_islat_observed")]
        colls = [c for c in ax.collections if hasattr(c, "_islat_observed")]
        return lines, colls

    def test_update_panels_inplace_replaces_error_bars(self):
        """Regression: cycling through a sample kept the previous spectrum's error bars.

        ``ax.errorbar`` draws a ``Line2D`` plus ``LineCollection`` error bars, all tagged ``_islat_observed``.
        The in-place fast path used to ``set_data`` only the line, so the collections still showed the old errors around the old flux.
        """
        wave, flux, err = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, error_data=err, n_panels=3)
        fsp.generate_plot()

        # Same wavelength grid -> same panel layout -> the view takes the fast path.
        new_flux = flux + 0.5
        new_err = np.full_like(err, 0.05)  # 10x the original 0.005
        assert fsp.update_data(wave, new_flux, error_data=new_err) is False
        fsp.update_panels_inplace()

        for idx, ax in fsp.subplots.items():
            obs_lines, obs_colls = self._observed_artists(ax)
            assert len(obs_lines) == 1
            assert len(obs_colls) == 1  # one set of error bars, no stale copy

            xr = fsp.panels[idx][0].xlim
            mask = (wave >= xr[0]) & (wave <= xr[1])
            np.testing.assert_allclose(obs_lines[0].get_ydata(), new_flux[mask])

            # Every bar spans +/- the *new* error around the *new* flux.
            segs = np.asarray(obs_colls[0].get_segments())  # (n, 2, 2): [[x, y-e], [x, y+e]]
            assert segs.shape[0] == mask.sum()
            np.testing.assert_allclose(segs[:, 0, 0], wave[mask])
            half = (segs[:, 1, 1] - segs[:, 0, 1]) / 2.0
            np.testing.assert_allclose(half, 0.05)
            centre = (segs[:, 1, 1] + segs[:, 0, 1]) / 2.0
            np.testing.assert_allclose(centre, new_flux[mask])
        fsp.close()

    def test_update_panels_inplace_drops_error_bars_when_new_data_has_none(self):
        wave, flux, err = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, error_data=err, n_panels=2)
        fsp.generate_plot()
        fsp.update_data(wave, flux, error_data=None)
        fsp.update_panels_inplace()
        for ax in fsp.subplots.values():
            obs_lines, obs_colls = self._observed_artists(ax)
            assert len(obs_lines) == 1
            assert obs_colls == []
        fsp.close()

    def test_update_panels_inplace_preserves_hidden_observed_data(self):
        wave, flux, err = make_wave_flux(n=200)
        fsp = FullSpectrumPlot(wave, flux, error_data=err, n_panels=2)
        fsp.generate_plot()
        for ax in fsp.subplots.values():
            for art in list(ax.lines) + list(ax.collections):
                if hasattr(art, "_islat_observed"):
                    art.set_visible(False)

        fsp.update_data(wave, flux + 0.1, error_data=err)
        fsp.update_panels_inplace()

        for ax in fsp.subplots.values():
            obs_lines, obs_colls = self._observed_artists(ax)
            tagged = obs_lines + obs_colls
            assert tagged
            assert all(not a.get_visible() for a in tagged)
        fsp.close()

    def test_save_and_close(self, tmp_path):
        fsp = self._make_fsp()
        fsp.generate_plot()
        out = fsp.save(tmp_path / "fsp.png", dpi=72)
        assert out.exists()
        fsp.close()
        assert fsp.fig is None

    def test_uniform_ylim(self):
        fsp = self._make_fsp(n_panels=3, uniform_ylim=True)
        fsp.generate_plot()
        # All panels should have the same y-limits
        ylims = [list(fsp.panels.values())[i][0].ax.get_ylim()
                 for i in fsp.panels]
        for yl in ylims[1:]:
            assert yl[0] == pytest.approx(ylims[0][0], abs=0.001)
            assert yl[1] == pytest.approx(ylims[0][1], abs=0.001)
        fsp.close()
