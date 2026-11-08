# -*- coding: utf-8 -*-
"""Selections handed to the main plot by code reach its span selector intact.

The Full Spectrum (and Optical Depth) view moves the main plot to a new range
and sets the selection there; the Three Panel view restores its selection when
re-activated.  Both go through ``InteractionHandler.show_span``.  Until the
main plot is redrawn at the new range, nothing may be blitted onto it: the span
selector's background still shows the old range, and blitting it (for the
handed-over span, or for a drag the user starts before the redraw) puts the
old wavelength range back on screen.
"""
import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler


def _main_plot():
    """A main-plot stub with a real ax1, canvas and InteractionHandler span selector.

    ``draw_idle`` waits for ``flush()`` as on the Tk canvas (Agg would draw at once).
    """
    fig = Figure(figsize=(8, 3), dpi=80)
    canvas = FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(111)
    x = np.linspace(10.0, 20.0, 2000)
    ax1.plot(x, np.sin(3 * x))
    ih = InteractionHandler.__new__(InteractionHandler)
    ih.ax1 = ax1
    ih._setup_span_selector()
    canvas.draw()  # background captured for 10-20 um

    pending = []
    canvas.draw_idle = lambda *a, **k: pending.append(1)

    def flush():
        if pending:
            pending.clear()
            canvas.draw()

    return SimpleNamespace(
        fig=fig, ax1=ax1, canvas=canvas, interaction_handler=ih,
        onselect=MagicMock(), current_selection=None, pending=pending, flush=flush,
    )


def _frame(fig):
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _true_frame(fig):
    fig.canvas.draw()
    return _frame(fig)


def _full_spectrum_view(main_plot):
    from iSLAT.Modules.Plotting.FullSpectrumView import FullSpectrumView

    view = FullSpectrumView.__new__(FullSpectrumView)
    view._pm = SimpleNamespace(islat=SimpleNamespace(GUI=SimpleNamespace(plot=main_plot)))
    view._plot = SimpleNamespace(_xlim_start=10.0, _xlim_end=20.0)
    return view


def test_full_spectrum_selection_moves_range_and_shows_span():
    main_plot = _main_plot()
    old_frame = _frame(main_plot.fig)

    _full_spectrum_view(main_plot)._apply_selection(16.0, 16.2)

    span = main_plot.interaction_handler.span_selector
    lo, hi = main_plot.ax1.get_xlim()
    assert lo < 16.0 and hi > 16.2 and (hi - lo) < 5.0  # moved to the selection
    assert span.extents == (16.0, 16.2)
    assert span.get_visible() and span._selection_completed
    main_plot.onselect.assert_called_once_with(16.0, 16.2)
    # Nothing blitted over the old frame; a redraw is pending instead.
    np.testing.assert_array_equal(_frame(main_plot.fig), old_frame)
    assert main_plot.pending
    main_plot.flush()
    np.testing.assert_array_equal(_frame(main_plot.fig), _true_frame(main_plot.fig))


def test_drag_before_redraw_does_not_blit_old_range():
    """A drag started before the redraw lands must not show the old range either."""
    main_plot = _main_plot()
    old_frame = _frame(main_plot.fig)
    _full_spectrum_view(main_plot)._apply_selection(16.0, 16.2)
    span = main_plot.interaction_handler.span_selector

    span.extents = (15.9, 16.4)  # what each drag step does
    np.testing.assert_array_equal(_frame(main_plot.fig), old_frame)

    main_plot.flush()
    np.testing.assert_array_equal(_frame(main_plot.fig), _true_frame(main_plot.fig))


def test_drag_after_redraw_blits_new_range():
    main_plot = _main_plot()
    _full_spectrum_view(main_plot)._apply_selection(16.0, 16.2)
    main_plot.flush()
    span = main_plot.interaction_handler.span_selector

    span.extents = (15.9, 16.4)

    assert not main_plot.pending  # a plain blit, no redraw
    np.testing.assert_array_equal(_frame(main_plot.fig), _true_frame(main_plot.fig))


def test_three_panel_restore_shows_span():
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView

    main_plot = _main_plot()
    pm = MagicMock()
    pm.canvas = main_plot.canvas
    pm.interaction_handler = main_plot.interaction_handler
    pm.toggle_state = {"current_selection": (12.0, 12.5)}
    view = ThreePanelView(pm)
    view.on_selection = MagicMock()

    view._restore_line_selection()

    span = main_plot.interaction_handler.span_selector
    assert span.extents == (12.0, 12.5)
    assert span.get_visible() and span._selection_completed
    view.on_selection.assert_called_once_with(12.0, 12.5)
    # Same range as the background: the span is blitted on it directly.
    np.testing.assert_array_equal(_frame(main_plot.fig), _true_frame(main_plot.fig))
