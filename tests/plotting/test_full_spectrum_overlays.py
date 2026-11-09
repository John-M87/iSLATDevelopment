# -*- coding: utf-8 -*-
"""Full Spectrum view: overlays that are still current survive a sync, and the panels' span selectors.

``sync_toggle_state`` runs on every activation, model update and layout
change.  It used to remove and re-create the atomic and saved-line overlays
(and re-read the saved-line file) each time; FullSpectrumView now reports
through ``_overlay_drawn`` whether what is drawn is still current, so only
stale overlays are redrawn.  The view's per-panel span selectors are
:class:`SingleDrawSpanSelector`, so a span left visible in one panel does not
make every draw render the figure once per panel.

Rendered headlessly (Agg) with real test molecules.
"""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import MagicMock
from matplotlib.backends.backend_agg import FigureCanvasAgg

from iSLAT.Modules.GUI.InteractionHandler import SingleDrawSpanSelector
from iSLAT.Modules.Plotting import FullSpectrumView
from iSLAT.Modules.Plotting.ArtistTags import ATOMIC_LINE_TAG, SAVED_LINE_TAG
from iSLAT.Modules.Plotting.SavedLineLabels import (
    reset_saved_line_label_settings, set_saved_line_label_settings,
)

from tests.plotting import make_line_list
from tests.plotting.test_optical_depth_view import _MolDict
from tests.plotting.test_plot_views import _make_test_molecule


# ======================================================================
# Helpers
# ======================================================================

def _make_view(line_list, **toggles):
    wave = np.linspace(10.0, 20.0, 2000)
    mols = _MolDict()
    for i, name in enumerate(("H2O", "CO2")):
        mols[name] = _make_test_molecule(name=name, lam_start=12.0 + 0.3 * i)

    islat = MagicMock()
    islat.wave_data_original = wave
    islat.flux_data = np.full_like(wave, 0.05)
    islat.err_data = np.full_like(wave, 0.005)
    islat.molecules_dict = mols
    islat.input_line_list = str(line_list)
    islat.root = None

    pm = MagicMock()
    pm.islat = islat
    pm.theme = {}
    pm.legend_toggle = True
    pm.control_bus = None
    pm.active_view = None
    pm.toggle_state = {
        "atomic_lines": True,
        "saved_lines": True,
        "summed": True,
        "legend": True,
        "data_density": False,
        "y_autoscale": True,
        "y_limits": None,
        "observed": True,
        "wavelength_markers": [],
        **toggles,
    }
    return FullSpectrumView(pm), pm, islat


def _render(view):
    """What activate() does, minus the Tk canvas (an Agg canvas, so the selectors can blit)."""
    view._plot = view._create_plot()
    view._plot.generate_plot()
    FigureCanvasAgg(view.fig)
    view._install_span_selectors()
    view._initialised = True
    view._needs_refresh = False
    view.sync_toggle_state(view._pm.toggle_state)


def _overlay(view, tag):
    return [a for ax in view._iter_all_axes()
            for a in list(ax.lines) + list(ax.collections) + list(ax.texts)
            if hasattr(a, tag)]


def _texts(view, tag):
    return [t for ax in view._iter_all_axes() for t in ax.texts if hasattr(t, tag)]


def _same_artists(a, b):
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def lines_csv(tmp_path):
    csv = tmp_path / "lines.csv"
    make_line_list([12.5, 15.5]).to_csv(csv, index=False)  # inside panels, not on an edge
    return csv


@pytest.fixture
def rendered(lines_csv):
    view, pm, islat = _make_view(lines_csv)
    _render(view)
    yield view, pm, islat
    view.destroy()


# ======================================================================
# Overlays kept by sync_toggle_state
# ======================================================================

class TestSyncKeepsCurrentOverlays:
    def test_both_overlays_are_drawn(self, rendered):
        view, *_ = rendered
        assert _overlay(view, ATOMIC_LINE_TAG)
        assert len(_texts(view, SAVED_LINE_TAG)) == 2

    def test_sync_keeps_the_same_artists(self, rendered):
        view, pm, _ = rendered
        atomic, saved = _overlay(view, ATOMIC_LINE_TAG), _overlay(view, SAVED_LINE_TAG)
        view.sync_toggle_state(pm.toggle_state)
        assert _same_artists(_overlay(view, ATOMIC_LINE_TAG), atomic)
        assert _same_artists(_overlay(view, SAVED_LINE_TAG), saved)

    def test_sync_does_not_reread_the_line_list(self, rendered, monkeypatch):
        view, pm, _ = rendered
        reads = []
        monkeypatch.setattr(view, "_load_line_data", lambda: reads.append(1))
        view.sync_toggle_state(pm.toggle_state)
        view.sync_toggle_state(pm.toggle_state)
        assert reads == []

    def test_kept_overlays_look_like_redrawn_ones(self, rendered):
        view, pm, _ = rendered
        view.fig.canvas.draw()
        kept = np.asarray(view.fig.canvas.buffer_rgba()).copy()
        view._remove_atomic_line_artists()
        view._remove_saved_line_artists()
        view.sync_toggle_state(pm.toggle_state)
        assert _overlay(view, ATOMIC_LINE_TAG) and _overlay(view, SAVED_LINE_TAG)
        view.fig.canvas.draw()
        np.testing.assert_array_equal(kept, np.asarray(view.fig.canvas.buffer_rgba()))


class TestSyncRedrawsStaleOverlays:
    def test_edited_line_list(self, rendered, lines_csv):
        view, pm, _ = rendered
        saved = _overlay(view, SAVED_LINE_TAG)
        make_line_list([12.5, 15.5, 17.5]).to_csv(lines_csv, index=False)
        _bump_mtime(lines_csv)
        view.sync_toggle_state(pm.toggle_state)
        assert len(_texts(view, SAVED_LINE_TAG)) == 3
        assert not any(a is b for a in _overlay(view, SAVED_LINE_TAG) for b in saved)

    def test_other_line_list(self, rendered, tmp_path):
        view, pm, islat = rendered
        other = tmp_path / "other.csv"
        make_line_list([13.5]).to_csv(other, index=False)
        islat.input_line_list = str(other)
        view.sync_toggle_state(pm.toggle_state)
        assert len(_texts(view, SAVED_LINE_TAG)) == 1

    def test_changed_panel_limits_put_labels_at_the_new_top(self, rendered):
        view, pm, _ = rendered
        ax = next(ax for ax in view._iter_all_axes()
                  if any(hasattr(t, ATOMIC_LINE_TAG) for t in ax.texts))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, 2 * hi)
        view.sync_toggle_state(pm.toggle_state)
        tops = [t.get_position()[1] for t in ax.texts if hasattr(t, ATOMIC_LINE_TAG)]
        assert tops and all(y == pytest.approx(2 * hi) for y in tops)

    def test_partly_removed_overlay_is_restored(self, rendered):
        # ResidualSpectrumPlot.update_panels_inplace removes every text on the spectrum axes.
        view, pm, _ = rendered
        n_texts = len(_texts(view, SAVED_LINE_TAG)) + len(_texts(view, ATOMIC_LINE_TAG))
        n_saved = len(_overlay(view, SAVED_LINE_TAG))
        for ax in view._iter_all_axes():
            for txt in list(ax.texts):
                txt.remove()
        view.sync_toggle_state(pm.toggle_state)
        assert len(_texts(view, SAVED_LINE_TAG)) + len(_texts(view, ATOMIC_LINE_TAG)) == n_texts
        assert len(_overlay(view, SAVED_LINE_TAG)) == n_saved  # not drawn twice

    def test_label_setting_change(self, rendered):
        view, pm, _ = rendered
        try:
            set_saved_line_label_settings(hidden_views=["Full Spectrum"])
            view.sync_toggle_state(pm.toggle_state)
            assert _texts(view, SAVED_LINE_TAG) == []
            assert _overlay(view, SAVED_LINE_TAG)  # the markers stay
        finally:
            reset_saved_line_label_settings()

    def test_regenerated_layout_gets_new_overlays(self, rendered):
        view, *_ = rendered
        old = _overlay(view, ATOMIC_LINE_TAG)
        view._regenerate_layout()
        new = _overlay(view, ATOMIC_LINE_TAG)
        assert len(new) == len(old)
        assert not any(a is b for a in new for b in old)


class TestTurningOverlaysOff:
    def test_toggle_off_then_sync_draws_nothing(self, rendered):
        view, pm, _ = rendered
        pm.toggle_state["atomic_lines"] = False
        view.toggle_atomic_lines(False)
        view.sync_toggle_state(pm.toggle_state)
        assert _overlay(view, ATOMIC_LINE_TAG) == []

    def test_sync_with_overlay_off_removes_it(self, rendered):
        view, pm, _ = rendered
        pm.toggle_state["saved_lines"] = False
        view.sync_toggle_state(pm.toggle_state)
        assert _overlay(view, SAVED_LINE_TAG) == []

    def test_stale_record_is_dropped(self, rendered):
        # A record of a replaced layout must not keep its artists (and figure) alive.
        view, pm, _ = rendered
        pm.toggle_state["atomic_lines"] = False
        view._regenerate_layout()
        assert _overlay(view, ATOMIC_LINE_TAG) == []
        assert "atomic_lines" not in view._overlay_records


# ======================================================================
# Span selectors
# ======================================================================

class TestPanelSpanSelectors:
    def test_selectors_draw_once(self, rendered):
        view, *_ = rendered
        assert view.span_selectors
        assert all(isinstance(s, SingleDrawSpanSelector) for s in view.span_selectors.values())

    def test_span_in_one_panel_renders_figure_once(self, rendered):
        view, *_ = rendered
        fig = view.fig
        fig.canvas.draw()
        span = list(view.span_selectors.values())[1]
        lo, hi = span.ax.get_xlim()
        span.set_visible(True)
        span.extents = (lo + 0.3 * (hi - lo), lo + 0.4 * (hi - lo))
        renders = []
        original = fig.draw
        fig.draw = lambda renderer: (renders.append(1), original(renderer))[1]
        fig.canvas.draw()
        assert len(renders) == 1
