# -*- coding: utf-8 -*-
"""Saving one panel of a composite view (right-click > Save <panel>...).

Views made of several panels say which panel a right-click is on
(``PlotView.panel_at``) and the context menu then offers to save just that
panel next to *Save Figure...*:

* the stacked views (Full Spectrum, Optical Depth, the Full Spectrum window)
  build a fresh one-panel figure of the panel's wavelength range, with its own
  legend, axis labels and sub-panels;
* the Three Panel view and the Line Grid save the panel's axes as shown, with
  everything else hidden for the save (``FigureExport.isolated_axes``).
"""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from iSLAT.Modules.Plotting.FigureExport import isolated_axes
from iSLAT.Modules.Plotting.FullSpectrumView import DEFAULT_PANEL_EXPORT_HEIGHT, set_export_settings
from iSLAT.Modules.Plotting.PlotView import FigurePanel
from iSLAT.Modules.Plotting.ResidualSpectrumPlot import ResidualSpectrumPlot

from tests.plotting.test_line_inspection_population_diagram import _build_three_panel_view, _make_molecule
from tests.plotting.test_optical_depth_view import _make_view as _make_optical_depth_view
from tests.plotting.test_panel_ranges import _render
from tests.plotting.test_residual_toggle import _make_view as _make_full_spectrum_view
from tests.plotting.test_view_plumbing import _LiveFigureView


def _event(ax):
    return SimpleNamespace(inaxes=ax, ydata=None)


def _first_axes(cell):
    return cell[0] if isinstance(cell, tuple) else cell


def _png_size(path):
    with Image.open(path) as im:
        return im.size


def _two_panel_figure():
    fig = Figure(figsize=(8, 4))
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    left.plot([0, 1], [0, 1])
    right.plot([0, 1], [1, 0])
    return fig, left, right


@pytest.fixture(scope="module")
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


def _labels(menu):
    end = menu.index("end")
    return [menu.entrycget(i, "label") for i in range(end + 1) if menu.type(i) != "separator"]


# ======================================================================
# isolated_axes
# ======================================================================
class TestIsolatedAxes:
    def test_shows_only_the_panel_then_restores(self):
        fig, left, right = _two_panel_figure()
        legend = fig.legend(["a"])
        text = fig.text(0.5, 0.01, "note")
        with isolated_axes(fig, [left]):
            assert left.get_visible() and not right.get_visible()
            assert not legend.get_visible() and not text.get_visible()
        assert left.get_visible() and right.get_visible()
        assert legend.get_visible() and text.get_visible()

    def test_keeps_twins_and_colorbars(self):
        fig, left, right = _two_panel_figure()
        twin = left.twinx()
        mappable = left.scatter([0, 1], [0, 1], c=[0, 1])
        cbar = fig.colorbar(mappable, ax=left)
        with isolated_axes(fig, [left]):
            assert twin.get_visible() and cbar.ax.get_visible()
            assert not right.get_visible()

    def test_shared_figure_labels_become_the_panel_labels(self):
        fig, left, right = _two_panel_figure()
        fig.supxlabel("Wavelength", fontsize=13)
        fig.supylabel("Flux")
        right.set_ylabel("Own label")
        with isolated_axes(fig, [left]):
            assert left.get_xlabel() == "Wavelength" and left.xaxis.label.get_fontsize() == 13
            assert left.get_ylabel() == "Flux"
            assert not fig._supxlabel.get_visible()
        with isolated_axes(fig, [right]):
            assert right.get_ylabel() == "Own label"  # its own label wins
        assert left.get_xlabel() == "" and left.get_ylabel() == ""
        assert left.xaxis.label.get_fontsize() != 13
        assert fig._supxlabel.get_visible() and fig._supylabel.get_visible()

    def test_restores_after_an_error(self):
        fig, left, right = _two_panel_figure()
        with pytest.raises(RuntimeError):
            with isolated_axes(fig, [left]):
                raise RuntimeError("save failed")
        assert right.get_visible()


# ======================================================================
# PlotView: the menu item and save_panel
# ======================================================================
class _PanelView(_LiveFigureView):
    SAVE_FILE_SUFFIX = "view"

    def panel_at(self, event):
        if event.inaxes is None:
            return None
        return FigurePanel(label="Left Panel", file_suffix="left", axes=(event.inaxes,))


class TestPlotViewPanelSave:
    def test_menu_offers_the_clicked_panel(self, tk_root):
        import tkinter as tk
        fig, left, _right = _two_panel_figure()
        view = _PanelView(fig)
        menu = tk.Menu(tk_root, tearoff=0)
        view._append_save_figure_item(menu, _event(left))
        assert menu.type(0) != "separator"  # no separator at the top of an empty menu
        assert _labels(menu) == ["Save Figure…", "Save Left Panel…"]

        no_panel = tk.Menu(tk_root, tearoff=0)
        no_panel.add_command(label="Other")
        view._append_save_figure_item(no_panel, _event(None))
        assert _labels(no_panel) == ["Other", "Save Figure…"]
        assert no_panel.type(1) == "separator"

    def test_save_panel_crops_to_the_panel_and_restores_the_figure(self, tmp_path):
        fig, left, right = _two_panel_figure()
        fig.canvas.draw_idle = MagicMock()
        view = _PanelView(fig)
        whole, part = tmp_path / "whole.png", tmp_path / "left.png"
        view.save_figure(str(whole), dpi=50)
        assert view.save_panel(view.panel_at(_event(left)), str(part), dpi=50) == str(part)
        (whole_w, whole_h), (part_w, part_h) = _png_size(whole), _png_size(part)
        assert part_w < 0.65 * whole_w and part_h == pytest.approx(whole_h, abs=6)
        assert right.get_visible()
        fig.canvas.draw_idle.assert_called()  # the screen shows everything again

    def test_dialog_suggests_the_panel_name(self, monkeypatch):
        import tkinter.filedialog
        asked = []
        monkeypatch.setattr(tkinter.filedialog, "asksaveasfilename", lambda **k: asked.append(k) or "")
        fig, left, _right = _two_panel_figure()
        view = _PanelView(fig, islat=SimpleNamespace(loaded_spectrum_file="/data/citau.csv"))
        assert view.save_panel(view.panel_at(_event(left)), file_format="png") is None
        assert asked[0]["initialfile"] == "citau_view_left.png"

    def test_views_without_panels_offer_none(self):
        fig, left, _right = _two_panel_figure()
        assert _LiveFigureView(fig).panel_at(_event(left)) is None


# ======================================================================
# Stacked views: a fresh one-panel figure
# ======================================================================
@pytest.fixture(params=[False, True], ids=["spectrum", "residuals"])
def full_spectrum(request):
    view, pm, *_ = _make_full_spectrum_view(show_residuals=request.param)
    _render(view)
    yield view, pm
    if view._plot is not None:
        view._plot.close()


class TestFullSpectrumPanel:
    def test_panel_at_names_the_clicked_cell(self, full_spectrum):
        view, _pm = full_spectrum
        lo, hi = view.shown_panel_ranges()[1]
        for ax in np.atleast_1d(view._plot.subplots[1]):  # any sub-panel of the cell
            panel = view.panel_at(_event(ax))
            assert panel.key == (lo, hi)
            assert panel.label == f"Panel 2 ({lo:.2f}–{hi:.2f} μm)"
            assert panel.file_suffix == f"panel2_{lo:.2f}-{hi:.2f}um"
        assert view.panel_at(_event(None)) is None

    def test_export_is_one_panel_of_that_range(self, full_spectrum):
        view, _pm = full_spectrum
        lo, hi = view.shown_panel_ranges()[1]
        plot = view._create_panel_export_plot((lo, hi))
        try:
            assert list(zip(plot._panel_edges, plot._panel_ends)) == [(lo, hi)]
            assert plot.fig is not view._plot.fig
            assert _first_axes(plot.subplots[0]).get_xlim() == pytest.approx((lo, hi))
            assert plot.fig.get_size_inches()[1] == pytest.approx(DEFAULT_PANEL_EXPORT_HEIGHT)
            if isinstance(plot, ResidualSpectrumPlot):
                assert not plot.show_total_chi2
                assert [t for t in plot.fig.texts if t is not plot.fig._supylabel] == []
        finally:
            plot.close()

    def test_export_settings_set_the_size(self, full_spectrum):
        view, pm = full_spectrum
        set_export_settings(pm, width=9.0, panel_height=2.5)
        plot = view._create_panel_export_plot(view.shown_panel_ranges()[0])
        try:
            assert tuple(plot.fig.get_size_inches()) == pytest.approx((9.0, 2.5))
        finally:
            plot.close()

    def test_save_panel_writes_the_file_and_leaves_the_view_alone(self, full_spectrum, tmp_path):
        view, _pm = full_spectrum
        live_axes = list(view._plot.fig.axes)
        panel = view.panel_at(_event(_first_axes(view._plot.subplots[0])))
        out = tmp_path / "panel.png"
        assert view.save_panel(panel, str(out), dpi=30) == str(out)
        width, height = _png_size(out)
        assert width > 2 * height  # one wide panel, not the whole stack
        assert view._plot.fig.axes == live_axes and all(ax.get_visible() for ax in live_axes)

    def test_context_menu(self, full_spectrum, tk_root):
        view, _pm = full_spectrum
        menu = view.build_context_menu(_event(_first_axes(view._plot.subplots[0])), tk_root)
        lo, hi = view.shown_panel_ranges()[0]
        assert _labels(menu)[-2:] == ["Save Figure…", f"Save Panel 1 ({lo:.2f}–{hi:.2f} μm)…"]


class TestOpticalDepthPanel:
    def test_export_is_one_panel_of_that_range(self):
        view, *_ = _make_optical_depth_view()
        _render(view)
        lo, hi = view.shown_panel_ranges()[2]
        panel = view.panel_at(_event(_first_axes(view._plot.subplots[2])))
        assert panel.key == (lo, hi)
        plot = view._create_panel_export_plot(panel.key)
        try:
            assert list(zip(plot._panel_edges, plot._panel_ends)) == [(lo, hi)]
            assert plot.fig.get_size_inches()[1] == pytest.approx(DEFAULT_PANEL_EXPORT_HEIGHT)
        finally:
            plot.close()
            view._plot.close()


class TestFullSpectrumWindowMenu:
    def test_offers_the_panel(self, tk_root):
        from iSLAT.Modules.GUI.FullSpectrumWindow import WindowFullSpectrumView
        source, pm, *_ = _make_full_spectrum_view()
        view = WindowFullSpectrumView(pm)
        _render(view)
        try:
            menu = view.build_context_menu(_event(_first_axes(view._plot.subplots[0])), tk_root)
            labels = _labels(menu)
            assert labels[0] == "Save Figure…" and labels[1].startswith("Save Panel 1 (")
            assert menu.type(0) != "separator"
        finally:
            view._plot.close()


# ======================================================================
# Three Panel view and Line Grid: the panel as shown
# ======================================================================
class TestThreePanel:
    @pytest.fixture
    def three_panel(self):
        view, pm, fig = _build_three_panel_view(_make_molecule())
        FigureCanvasAgg(fig)
        fig.set_size_inches(12, 8)
        fig.canvas.draw()
        return view, pm, fig

    def test_each_panel(self, three_panel):
        view, pm, _fig = three_panel
        expected = {
            "ax1": ("Spectrum Panel", "spectrum"),
            "ax2": ("Line Inspection Panel", "line_inspection"),
            "ax3": ("Population Diagram Panel", "population_diagram"),
        }
        for attr, (label, suffix) in expected.items():
            ax = getattr(pm, attr)
            assert view.panel_at(_event(ax)) == FigurePanel(label=label, file_suffix=suffix, axes=(ax,))
        assert view.panel_at(_event(None)) is None

    def test_saves_just_that_panel(self, three_panel, tmp_path):
        view, pm, fig = three_panel
        whole, part = tmp_path / "whole.png", tmp_path / "pop.png"
        view.save_figure(str(whole), dpi=30)
        view.save_panel(view.panel_at(_event(pm.ax3)), str(part), dpi=30)
        (whole_w, whole_h), (part_w, part_h) = _png_size(whole), _png_size(part)
        assert part_w < 0.6 * whole_w and part_h < 0.75 * whole_h
        assert all(ax.get_visible() for ax in (pm.ax1, pm.ax2, pm.ax3))

    def test_spectrum_panel_menu(self, three_panel, tk_root):
        view, pm, _fig = three_panel
        menu = view.build_context_menu(_event(pm.ax1), tk_root)
        assert _labels(menu)[-2:] == ["Save Figure…", "Save Spectrum Panel…"]


class TestLineGrid:
    def test_panel_at_finds_the_fit_in_any_grid(self):
        from iSLAT.Modules.Plotting.FitLinesPlotGridView import FitLinesPlotGridView

        def grid(lams):
            fig = Figure()
            axs = fig.subplots(1, len(lams), squeeze=False)
            entries = {i: {"xmin": lam - 0.01, "xmax": lam + 0.01, "lam": lam, "species": "H2O"}
                       for i, lam in enumerate(lams)}
            return SimpleNamespace(axs=axs, fit_csv_dict=entries, rows=1, cols=len(lams))

        first, second = grid([12.0, 12.5]), grid([13.0, 13.25])
        view = FitLinesPlotGridView(MagicMock())
        view._plot_grids = [first, second]
        ax = second.axs[0, 1]
        assert view.panel_at(_event(ax)) == FigurePanel(
            label="Panel [H2O 13.2500 μm]", file_suffix="line2_13.2500um", axes=(ax,),
        )
        assert view.panel_at(_event(Figure().add_subplot())) is None
