"""Theme grid keys reach their axes; the inspection Total Model toggle redraws its own canvas."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from iSLAT.Modules.Plotting.ArtistTags import SUMMED_TAG, tag_artist
from iSLAT.Modules.Plotting.LineInspectionContextMixin import LineInspectionContextMixin
from iSLAT.Modules.Plotting.MainPlot import iSLATPlot


def test_theme_grid_keys_apply_to_their_axes():
    fig = Figure()
    axes = [fig.add_subplot(3, 1, i) for i in (1, 2, 3)]
    pm = SimpleNamespace(fig=fig, ax1=axes[0], ax2=axes[1], ax3=axes[2], canvas=MagicMock(),
                         theme={"ax2_grid": True}, _three_panel_view=SimpleNamespace(_grid=None))
    iSLATPlot._apply_plot_theming(pm)
    assert [ax.xaxis.get_gridlines()[0].get_visible() for ax in axes] == [False, True, False]


def test_inspection_total_model_toggle_redraws_the_axes_own_canvas():
    fig = Figure()
    ax = fig.add_subplot()
    tag_artist(ax.fill_between([1, 2], [0, 1]), SUMMED_TAG)
    fig.canvas = MagicMock()
    other = MagicMock()
    view = SimpleNamespace(_pm=SimpleNamespace(canvas=other))
    LineInspectionContextMixin._toggle_inspection_summed(view, ax)
    fig.canvas.draw_idle.assert_called_once_with()
    other.draw_idle.assert_not_called()
