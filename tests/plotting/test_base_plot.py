# -*- coding: utf-8 -*-
"""Tests for BasePlot class methods and infrastructure."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure as MplFigure

from iSLAT.Modules.Plotting import BasePlot, DEFAULT_THEME
from iSLAT.Modules.Plotting.SpectrumPanel import SpectrumPanel

from tests.plotting import ConcreteSpectralPanel, make_wave_flux, make_atomic_lines, make_line_list


class TestBasePlotTheme:
    """Theme helper tests."""

    def test_default_theme_keys(self):
        assert "foreground" in DEFAULT_THEME
        assert "background" in DEFAULT_THEME
        assert "summed_spectra_color" in DEFAULT_THEME

    def test_get_theme_value_returns_from_theme(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
            theme={"foreground": "red"},
        )
        assert panel._get_theme_value("foreground") == "red"

    def test_get_theme_value_default(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        assert panel._get_theme_value("nonexistent_key", "fallback") == "fallback"


class TestBasePlotFigureLifecycle:
    """Figure creation, ownership, and teardown."""

    def test_ensure_figure_creates_figure(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        assert panel.fig is None
        panel._ensure_figure()
        assert panel.fig is not None
        assert isinstance(panel.fig, MplFigure)
        panel.close()

    def test_ensure_figure_respects_figsize(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
            figsize=(6, 3),
        )
        panel._ensure_figure()
        w, h = panel.fig.get_size_inches()
        assert abs(w - 6.0) < 0.01
        assert abs(h - 3.0) < 0.01
        panel.close()

    def test_close_releases_figure(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel.close()
        assert panel.fig is None

    def test_get_figure_returns_fig(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        assert panel.get_figure() is panel.fig
        panel.close()

    def test_external_figure_not_closed(self):
        fig = MplFigure(figsize=(4, 2))
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
            fig=fig,
        )
        assert panel.fig is fig
        assert not panel._owns_figure
        panel.close()

    def test_apply_theme_sets_colors(self):
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
            theme={"foreground": "#111111", "background": "#eeeeee",
                    "graph_fill_color": "#dddddd"},
        )
        panel.generate_plot()
        panel.apply_theme_to_figure()
        assert panel.fig.get_facecolor() is not None
        panel.close()


class TestClearTaggedArtists:
    def test_removes_lines(self):
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [0, 1])
        line._test_tag = True
        assert len(ax.lines) == 1
        BasePlot._clear_tagged_artists(ax, "_test_tag", lines=True)
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_removes_texts(self):
        fig, ax = plt.subplots()
        txt = ax.text(0.5, 0.5, "hello")
        txt._test_tag = True
        assert len(ax.texts) == 1
        BasePlot._clear_tagged_artists(ax, "_test_tag", texts=True)
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_ignores_untagged(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])  # no tag
        BasePlot._clear_tagged_artists(ax, "_test_tag", lines=True)
        assert len(ax.lines) == 1
        plt.close(fig)


class TestSpectrumPanelRenderingHelpers:
    """SpectrumPanel concrete rendering helpers: _plot_observed_spectrum,
    _plot_summed_spectrum."""

    def test_plot_observed_spectrum(self):
        panel = SpectrumPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel.generate_plot()
        ax = panel.ax
        n_before = len(ax.lines)
        panel._plot_observed_spectrum(ax, np.arange(5.0), np.ones(5))
        assert len(ax.lines) > n_before
        panel.close()

    def test_plot_observed_spectrum_with_error(self):
        panel = SpectrumPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel.generate_plot()
        ax = panel.ax
        panel._plot_observed_spectrum(
            ax, np.arange(5.0), np.ones(5), error_data=np.full(5, 0.01),
        )
        panel.close()

    def test_plot_observed_spectrum_deduplicate(self):
        panel = SpectrumPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel.generate_plot()
        ax = panel.ax
        panel._plot_observed_spectrum(
            ax, np.arange(5.0), np.ones(5), deduplicate=True,
        )
        tagged_before = sum(1 for l in ax.lines if hasattr(l, '_islat_observed'))
        panel._plot_observed_spectrum(
            ax, np.arange(5.0), np.ones(5), deduplicate=True,
        )
        tagged_after = sum(1 for l in ax.lines if hasattr(l, '_islat_observed'))
        assert tagged_after == tagged_before
        panel.close()

    def test_plot_summed_spectrum(self):
        panel = SpectrumPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel.generate_plot()
        ax = panel.ax
        panel._plot_summed_spectrum(
            ax, np.arange(5.0), np.array([0, 0.1, 0.2, 0.1, 0]),
        )
        tagged = [c for c in ax.collections if hasattr(c, '_islat_summed')]
        assert len(tagged) == 1
        panel.close()

    def test_plot_summed_spectrum_empty_data(self):
        panel = SpectrumPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel.generate_plot()
        panel._plot_summed_spectrum(panel.ax, np.array([]), np.array([]))
        panel.close()

    def test_plot_line_annotations(self):
        fig, ax = plt.subplots()
        ax.set_xlim(10, 20)
        ax.set_ylim(0, 1)
        df = make_line_list([12.0, 15.0, 25.0])
        BasePlot._plot_line_annotations(ax, df, (10, 20), 0, 1, tag="_test")
        tagged_texts = [t for t in ax.texts if hasattr(t, "_test")]
        assert len(tagged_texts) == 2
        plt.close(fig)

    def test_plot_atomic_lines(self):
        fig, ax = plt.subplots()
        ax.set_xlim(10, 20)
        ax.set_ylim(0, 1)
        df = make_atomic_lines([11.0, 15.0, 22.0])
        BasePlot._plot_atomic_lines(ax, df, xr=(10, 20), tag="_atag")
        tagged = [t for t in ax.texts if hasattr(t, "_atag")]
        assert len(tagged) == 2
        plt.close(fig)

    def test_plot_saved_line_markers(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame({"lam": [5.0, 10.0], "xmin": [4.5, 9.5], "xmax": [5.5, 10.5]})
        BasePlot._plot_saved_line_markers(ax, df, tag="_stag")
        tagged_lines = [l for l in ax.lines if hasattr(l, "_stag")]
        assert len(tagged_lines) == 6
        plt.close(fig)


class TestMoleculeLinewidth:
    """BasePlot.get_molecule_linewidth / get_molecule_linewidth_by_name."""

    def test_plain_molecule_keeps_base_width(self):
        from types import SimpleNamespace
        mol = SimpleNamespace(name="A", is_bold=False)
        assert BasePlot.get_molecule_linewidth(mol, 1.2) == 1.2

    def test_bold_molecule_is_scaled(self):
        from types import SimpleNamespace
        mol = SimpleNamespace(name="A", is_bold=True)
        assert BasePlot.get_molecule_linewidth(mol, 1.2) == pytest.approx(1.2 * BasePlot.BOLD_LINEWIDTH_FACTOR)

    def test_missing_or_non_bool_flag_is_not_bold(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        assert BasePlot.get_molecule_linewidth(None, 2.0) == 2.0
        assert BasePlot.get_molecule_linewidth(SimpleNamespace(name="A"), 2.0) == 2.0
        assert BasePlot.get_molecule_linewidth(MagicMock(), 2.0) == 2.0

    def test_by_name_looks_up_molecules_collection(self):
        from types import SimpleNamespace
        panel = ConcreteSpectralPanel(np.arange(5.0), np.ones(5), 0.0, 4.0)
        panel.molecules = {"A": SimpleNamespace(name="A", is_bold=True),
                           "B": SimpleNamespace(name="B", is_bold=False)}
        assert panel.get_molecule_linewidth_by_name("A", 1.0) == pytest.approx(BasePlot.BOLD_LINEWIDTH_FACTOR)
        assert panel.get_molecule_linewidth_by_name("B", 1.0) == 1.0
        assert panel.get_molecule_linewidth_by_name("missing", 1.0) == 1.0

    def test_by_name_without_molecules_uses_base(self):
        panel = ConcreteSpectralPanel(np.arange(5.0), np.ones(5), 0.0, 4.0)
        panel.molecules = None
        assert panel.get_molecule_linewidth_by_name("A", 0.8) == 0.8


class TestBuildMoleculeLegend:
    def test_empty(self):
        fig, ax = plt.subplots()
        BasePlot.build_molecule_legend(ax, [], [])
        assert ax.get_legend() is None
        plt.close(fig)

    def test_creates_legend(self):
        fig, ax = plt.subplots()
        BasePlot.build_molecule_legend(
            ax, ["H2O", "CO"], ["blue", "red"],
            use_figure_transform=False,
        )
        leg = ax.get_legend()
        assert leg is not None
        texts = leg.get_texts()
        assert len(texts) == 2
        assert texts[0].get_text() == "H2O"
        plt.close(fig)


class TestRegisterWithPyplot:
    """Ensure _register_with_pyplot produces a properly managed figure.

    This test class guards against the _cidgcf regression: when a
    figure is created via ``matplotlib.figure.Figure()`` (not
    ``plt.figure()``) and then registered with pyplot's Gcf, the
    resulting FigureManager must have a ``_cidgcf`` attribute.
    Without it, ``Gcf.destroy_all()`` — called by ``plt.show()`` in
    the inline / notebook backend — raises ``AttributeError``.
    """

    @pytest.fixture(autouse=True)
    def _close_all_figures(self):
        """Clean pyplot state before and after every test."""
        plt.close("all")
        yield
        plt.close("all")

    def test_figure_registered_in_pyplot(self):
        """After _register_with_pyplot the figure should be in Gcf."""
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()

        # The figure should now appear in pyplot's figure list.
        assert len(plt.get_fignums()) >= 1
        panel.close()

    def test_manager_has_cidgcf(self):
        """The registered FigureManager must have _cidgcf.

        This is the exact attribute whose absence caused the
        'FigureManagerBase has no attribute _cidgcf' crash when
        plt.show() → Gcf.destroy_all() was called in notebooks.
        """
        from matplotlib._pylab_helpers import Gcf

        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()

        # Find the manager for our figure
        fig_num = panel.fig.number
        manager = Gcf.figs.get(fig_num)
        assert manager is not None, (
            f"Figure {fig_num} not found in Gcf.figs after registration"
        )
        assert hasattr(manager, "_cidgcf"), (
            "FigureManager registered via _register_with_pyplot is missing "
            "_cidgcf — Gcf.destroy_all() will crash (the original bug)."
        )
        panel.close()

    def test_destroy_all_after_register(self):
        """Gcf.destroy_all() must not raise after _register_with_pyplot.

        This is the high-level regression test: plt.show() on the
        inline backend calls Gcf.destroy_all() internally.  If the
        manager was registered incorrectly this raises AttributeError.
        """
        from matplotlib._pylab_helpers import Gcf

        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()

        # Must not raise
        Gcf.destroy_all()
        panel.fig = None  # Already destroyed — avoid double-close

    def test_plt_close_all_after_register(self):
        """plt.close('all') must work after _register_with_pyplot."""
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()

        # Must not raise
        plt.close("all")
        assert len(plt.get_fignums()) == 0
        panel.fig = None  # Already closed

    def test_register_idempotent(self):
        """Calling _register_with_pyplot twice must be safe."""
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()
        fignums_first = list(plt.get_fignums())
        panel._register_with_pyplot()
        fignums_second = list(plt.get_fignums())
        assert fignums_first == fignums_second
        panel.close()

    def test_register_multiple_figures(self):
        """Register several independent figures and destroy all."""
        from matplotlib._pylab_helpers import Gcf

        panels = []
        for _ in range(5):
            p = ConcreteSpectralPanel(
                np.arange(5.0), np.ones(5), 0.0, 4.0,
            )
            p._ensure_figure()
            p._register_with_pyplot()
            panels.append(p)

        assert len(plt.get_fignums()) >= 5
        # All managers must have _cidgcf
        for fig_num in plt.get_fignums():
            mgr = Gcf.figs.get(fig_num)
            assert hasattr(mgr, "_cidgcf"), (
                f"Manager for figure {fig_num} missing _cidgcf"
            )

        # Must not raise
        Gcf.destroy_all()
        for p in panels:
            p.fig = None

    def test_close_after_register_deregisters(self):
        """BasePlot.close() should deregister the figure from pyplot."""
        panel = ConcreteSpectralPanel(
            np.arange(5.0), np.ones(5), 0.0, 4.0,
        )
        panel._ensure_figure()
        panel._register_with_pyplot()
        assert len(plt.get_fignums()) >= 1
        panel.close()
        # The figure should no longer be tracked
        assert panel.fig is None

    def test_show_does_not_crash(self):
        """show() on a concrete plot must not raise AttributeError.

        This simulates the exact user-facing action that triggered the
        original notebook crash.
        """
        wave, flux, _ = make_wave_flux(n=50)
        panel = ConcreteSpectralPanel(wave, flux, 10.0, 20.0)
        panel.generate_plot()
        # show() calls _register_with_pyplot() internally.
        # On the Agg backend it won't display anything, but it must
        # not crash.
        panel.show()
        panel.close()


class TestBasePlotIO:
    """Save and IPython rich-display."""

    def test_save(self, tmp_path):
        wave, flux, _ = make_wave_flux(n=50)
        panel = ConcreteSpectralPanel(wave, flux, 10.0, 20.0)
        panel.generate_plot()
        out = panel.save(tmp_path / "test.png", dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0
        panel.close()

    def test_repr_png(self):
        wave, flux, _ = make_wave_flux(n=50)
        panel = ConcreteSpectralPanel(wave, flux, 10.0, 20.0)
        panel.generate_plot()
        png = panel._repr_png_()
        assert png is not None
        assert len(png) > 100
        panel.close()

    def test_repr_html(self):
        wave, flux, _ = make_wave_flux(n=50)
        panel = ConcreteSpectralPanel(wave, flux, 10.0, 20.0)
        panel.generate_plot()
        html = panel._repr_html_()
        assert html is not None
        assert "<img" in html
        panel.close()
