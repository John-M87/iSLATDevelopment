# -*- coding: utf-8 -*-
"""fast_errorbar: the observed spectrum's error bars drawn as one path.

``Axes.errorbar`` makes one path per error bar, which takes ~280 ms for a
MIRI spectrum and slows every redraw.  ErrorBars.fast_errorbar puts all the
bars in one compound path (an ErrorBarCollection).  It must give the same
artists, styles, legend entry, data limits and segments as ``Axes.errorbar``,
and the same pixels to within one color level while the bars do not share
pixel columns (the zoomed-in views the GUI shows).
"""

import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting.ArtistTags import OBSERVED_TAG, iter_tagged
from iSLAT.Modules.Plotting.ErrorBars import ErrorBarCollection, fast_errorbar
from iSLAT.Modules.Plotting.SpectrumPanel import SpectrumPanel

STYLE = dict(fmt="-", color="black", linewidth=1, elinewidth=0.5, zorder=2, label="Data")


def _spectrum(n=4000, seed=0):
    """A noisy line spectrum sampled like a MIRI channel."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(14.0, 17.0, n)
    flux = 0.05 + 0.4 * np.exp(-((wave % 0.11) - 0.05) ** 2 / 2e-5) + rng.normal(0, 0.01, n)
    err = np.abs(rng.normal(0.01, 0.004, n))
    return wave, flux, err


def _axes(size=(14, 4), dpi=100):
    fig = Figure(figsize=size, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig.add_axes([0.06, 0.12, 0.9, 0.8])


def _stock(ax, wave, flux, err):
    return ax.errorbar(wave, flux, yerr=err, capsize=0, **STYLE)


def _fast(ax, wave, flux, err):
    return fast_errorbar(ax, wave, flux, err, **STYLE)


def _pair(wave, flux, err, xlim=None, legend=True):
    """(stock, fast) axes with the same spectrum drawn, both rendered."""
    out = []
    for draw in (_stock, _fast):
        ax = _axes()
        draw(ax, wave, flux, err)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if legend:
            ax.legend()
        ax.figure.canvas.draw()
        out.append(ax)
    return out


def _pixels(ax):
    return np.asarray(ax.figure.canvas.buffer_rgba()).astype(int)


class TestSameArtistsAsErrorbar:
    def test_returns_a_container_added_to_the_axes(self):
        ax = _axes()
        container = _fast(ax, *_spectrum())
        assert isinstance(container, ErrorbarContainer)
        assert container in ax.containers
        assert container.get_label() == "Data"
        assert container.has_yerr and not container.has_xerr

    def test_data_line_and_bars_like_errorbar(self):
        wave, flux, err = _spectrum()
        stock, fast = _stock(_axes(), wave, flux, err), _fast(_axes(), wave, flux, err)
        (s_line, s_caps, (s_bars,)), (f_line, f_caps, (f_bars,)) = stock, fast
        assert f_caps == s_caps == ()
        assert isinstance(f_bars, LineCollection)
        np.testing.assert_array_equal(f_line.get_xydata(), s_line.get_xydata())
        for getter in ("get_zorder", "get_color", "get_linewidth", "get_linestyle", "get_label"):
            assert getattr(f_line, getter)() == getattr(s_line, getter)(), getter
        for getter in ("get_zorder", "get_label"):
            assert getattr(f_bars, getter)() == getattr(s_bars, getter)(), getter
        np.testing.assert_array_equal(f_bars.get_linewidth(), s_bars.get_linewidth())
        np.testing.assert_array_equal(f_bars.get_color(), s_bars.get_color())

    def test_bars_are_one_path(self):
        ax = _axes()
        _, _, (bars,) = _fast(ax, *_spectrum())
        assert isinstance(bars, ErrorBarCollection)
        assert len(bars.get_paths()) == 1

    def test_segments_match_errorbar(self):
        wave, flux, err = _spectrum()
        _, _, (s_bars,) = _stock(_axes(), wave, flux, err)
        _, _, (f_bars,) = _fast(_axes(), wave, flux, err)
        np.testing.assert_array_equal(np.asarray(f_bars.get_segments()), np.asarray(s_bars.get_segments()))

    def test_segments_leave_out_non_finite_bars(self):
        wave, flux, err = _spectrum(n=50)
        flux[[3, 10]] = np.nan
        err[20] = np.nan
        _, _, (f_bars,) = _fast(_axes(), wave, flux, err)
        _, _, (s_bars,) = _stock(_axes(), wave, flux, err)
        # Stock segments of NaN bars come back shorter (non-finite points dropped).
        finite = [s for s in s_bars.get_segments() if len(s) == 2 and np.all(np.isfinite(s))]
        assert len(finite) == 47
        np.testing.assert_array_equal(np.asarray(f_bars.get_segments()), np.asarray(finite))

    def test_set_segments_still_works(self):
        _, _, (bars,) = _fast(_axes(), *_spectrum(n=20))
        new = [np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 2.0]])]
        bars.set_segments(new)
        assert len(bars.get_paths()) == 2
        np.testing.assert_array_equal(np.asarray(bars.get_segments()), np.asarray(new))

    def test_legend_entry_is_the_container(self):
        ax = _axes()
        container = _fast(ax, *_spectrum())
        handles, labels = ax.get_legend_handles_labels()
        assert handles == [container]
        assert labels == ["Data"]


class TestDataLimits:
    @pytest.mark.parametrize("with_nan", [False, True])
    def test_autoscaled_limits_match_errorbar(self, with_nan):
        wave, flux, err = _spectrum()
        if with_nan:
            flux[[0, 100, -1]] = np.nan
            err[50] = np.nan
        stock, fast = _pair(wave, flux, err, legend=False)
        np.testing.assert_array_equal(fast.dataLim.get_points(), stock.dataLim.get_points())
        assert fast.get_xlim() == stock.get_xlim()
        assert fast.get_ylim() == stock.get_ylim()


class TestPixels:
    @pytest.mark.parametrize("xlim", [(15.0, 15.6), (14.2, 14.5), (16.4, 17.0)])
    def test_zoomed_view_matches_errorbar(self, xlim):
        """The GUI's views show the bars at least a pixel apart: at most one color level differs."""
        stock, fast = _pair(*_spectrum(), xlim=xlim)
        diff = np.abs(_pixels(fast) - _pixels(stock))
        assert diff.max() <= 1

    def test_whole_spectrum_is_close(self):
        """Squeezed so bars share pixel columns, overlaps are anti-aliased once: close, not equal."""
        stock, fast = _pair(*_spectrum())
        diff = np.abs(_pixels(fast) - _pixels(stock)).max(axis=2)
        assert np.count_nonzero(diff > 1) < 0.02 * diff.size


class TestEdgeCases:
    def test_all_errors_non_finite_draws_the_line_only(self):
        wave, flux, _ = _spectrum(n=30)
        ax = _axes()
        line, _, (bars,) = fast_errorbar(ax, wave, flux, np.full(30, np.nan), **STYLE)
        assert bars.get_paths() == []
        assert bars.get_segments() == []
        ax.figure.canvas.draw()
        assert ax.get_ylim() == pytest.approx(_axes_with_line(wave, flux).get_ylim())

    def test_negative_errors_draw_instead_of_raising(self):
        wave, flux, err = _spectrum(n=30)
        ax = _axes()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, _, (bars,) = fast_errorbar(ax, wave, flux, -err, **STYLE)
        assert len(bars.get_segments()) == 30

    def test_empty_spectrum(self):
        ax = _axes()
        line, _, (bars,) = fast_errorbar(ax, [], [], [], **STYLE)
        assert len(line.get_xdata()) == 0
        assert bars.get_segments() == []
        ax.figure.canvas.draw()


def _axes_with_line(wave, flux):
    ax = _axes()
    ax.plot(wave, flux, "-")
    ax.figure.canvas.draw()
    return ax


class TestSpectrumPanel:
    def test_observed_spectrum_uses_one_path_bars(self):
        wave, flux, err = _spectrum()
        ax = _axes()
        panel = SpectrumPanel(wave_data=wave, flux_data=flux, xmin=15.0, xmax=15.6, error_data=err, ax=ax)
        panel.generate_plot()
        observed = list(iter_tagged(ax, OBSERVED_TAG))
        bars = [a for a in observed if isinstance(a, ErrorBarCollection)]
        lines = [a for a in observed if not isinstance(a, ErrorBarCollection)]
        assert len(bars) == 1 and len(lines) == 1
        assert len(bars[0].get_paths()) == 1
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ["Data"] and isinstance(handles[0], ErrorbarContainer)

    def test_without_errors_draws_a_plain_line(self):
        wave, flux, _ = _spectrum(n=100)
        ax = _axes()
        panel = SpectrumPanel(wave_data=wave, flux_data=flux, xmin=14.0, xmax=17.0, ax=ax)
        panel.generate_plot()
        assert not any(isinstance(a, LineCollection) for a in iter_tagged(ax, OBSERVED_TAG))
