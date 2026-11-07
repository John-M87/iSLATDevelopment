"""The main navigation toolbar follows the active plot view's canvas (Pan / Zoom / Home in every view)."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from matplotlib.backend_bases import MouseEvent, _Mode
from matplotlib.figure import Figure


@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display not available (headless environment)")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvases(tk_root):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    def _make():
        fig = Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot([0, 10], [0, 10])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        canvas = FigureCanvasTkAgg(fig, master=tk_root)
        canvas.get_tk_widget().pack()
        canvas.draw()
        return canvas, ax

    return _make(), _make()


@pytest.fixture
def toolbar(tk_root, canvases):
    import tkinter as tk
    from iSLAT.Modules.GUI.Widgets.iSLATToolbar import iSLATNavigationToolbar
    frame = tk.Frame(tk_root)
    return iSLATNavigationToolbar(canvases[0][0], window=frame)


def _drag(canvas, ax, start, end, button=1):
    x0, y0 = ax.transData.transform(start)
    x1, y1 = ax.transData.transform(end)
    MouseEvent("button_press_event", canvas, x0, y0, button)._process()
    MouseEvent("motion_notify_event", canvas, x1, y1, button, buttons={button})._process()
    MouseEvent("button_release_event", canvas, x1, y1, button)._process()


class TestSetCanvas:
    def test_retargets_handlers(self, toolbar, canvases):
        (c1, _), (c2, _) = canvases
        toolbar.set_canvas(c2)
        assert toolbar.canvas is c2
        assert c2.toolbar is toolbar and c1.toolbar is None
        handlers = lambda c: {ref() for refs in c.callbacks.callbacks.values() for ref in refs.values()}
        assert toolbar._zoom_pan_handler in handlers(c2)
        assert toolbar._zoom_pan_handler not in handlers(c1)

    def test_pan_and_home_work_on_new_canvas(self, toolbar, canvases):
        (_, _), (c2, ax2) = canvases
        toolbar.set_canvas(c2)
        toolbar.pan()
        _drag(c2, ax2, (5, 5), (3, 5))
        assert ax2.get_xlim() == pytest.approx((2, 12), abs=0.2)
        toolbar.home()
        assert ax2.get_xlim() == pytest.approx((0, 10))

    def test_zoom_on_new_canvas(self, toolbar, canvases):
        (_, _), (c2, ax2) = canvases
        toolbar.set_canvas(c2)
        toolbar.zoom()
        _drag(c2, ax2, (2, 2), (6, 6))
        assert ax2.get_xlim() == pytest.approx((2, 6), abs=0.1)

    def test_mode_carries_over_and_lock_moves(self, toolbar, canvases):
        (c1, ax1), (c2, ax2) = canvases
        toolbar.pan()
        toolbar.set_canvas(c2)
        assert toolbar.mode == _Mode.PAN
        assert c2.widgetlock.isowner(toolbar) and not c1.widgetlock.locked()
        if hasattr(_Mode.NONE, "_navigate_mode"):  # matplotlib < 3.11 mirrors the mode on each axes
            assert ax2.get_navigate_mode() == "PAN" and ax1.get_navigate_mode() is None
        assert toolbar._buttons["Pan"].var.get() == 1

    def test_locked_canvas_drops_mode(self, toolbar, canvases):
        (_, _), (c2, _) = canvases
        c2.widgetlock(object())
        toolbar.pan()
        toolbar.set_canvas(c2)
        assert toolbar.mode == _Mode.NONE
        assert toolbar._buttons["Pan"].var.get() == 0

    def test_history_is_per_canvas_and_resettable(self, toolbar, canvases):
        (c1, ax1), (c2, _) = canvases
        toolbar.push_current()
        ax1.set_xlim(3, 4)
        toolbar.set_canvas(c2)
        assert toolbar._nav_stack() is None
        toolbar.set_canvas(c1)
        toolbar.home()
        assert ax1.get_xlim() == pytest.approx((0, 10))
        toolbar.set_canvas(c1, reset_history=True)
        assert toolbar._nav_stack() is None

    def test_none_is_ignored(self, toolbar, canvases):
        toolbar.set_canvas(None)
        assert toolbar.canvas is canvases[0][0]


class TestPlotManagerSync:
    def test_sync_follows_active_view(self):
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        canvas = object()
        view = SimpleNamespace(get_canvas=lambda: canvas)
        pm = SimpleNamespace(toolbar=MagicMock(), active_view=view)
        iSLATPlot.sync_toolbar_to_active_view(pm)
        pm.toolbar.set_canvas.assert_called_once_with(canvas, reset_history=False)

    def test_only_active_view_resets(self):
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        active, other = object(), object()
        pm = SimpleNamespace(active_view=active, sync_toolbar_to_active_view=MagicMock())
        iSLATPlot.on_view_canvas_changed(pm, other)
        pm.sync_toolbar_to_active_view.assert_not_called()
        iSLATPlot.on_view_canvas_changed(pm, active)
        pm.sync_toolbar_to_active_view.assert_called_once_with(reset_history=True)


class TestViewsNotify:
    def _pm(self):
        return SimpleNamespace(
            islat=SimpleNamespace(active_molecule=None, molecules_dict=None),
            theme={}, control_bus=None, toggle_state={},
            on_view_canvas_changed=MagicMock(),
        )

    def test_line_inspection_new_canvas_notifies(self, tk_root):
        from iSLAT.Modules.Plotting.LineInspectionView import LineInspectionView
        pm = self._pm()
        view = LineInspectionView.__new__(LineInspectionView)
        view._pm, view._fig, view._parent_frame = pm, Figure(), tk_root
        view._canvas = view._right_click_cid = view._pick_cid = None
        view._ensure_canvas()
        pm.on_view_canvas_changed.assert_called_once_with(view)
        view._ensure_canvas()  # same figure: no new canvas, no notification
        pm.on_view_canvas_changed.assert_called_once()

    def test_population_diagram_refresh_notifies(self):
        from iSLAT.Modules.Plotting.PopulationDiagramView import PopulationDiagramView
        pm = self._pm()
        view = PopulationDiagramView(pm)
        view._initialised, view._fig = True, Figure()
        view.update_model_plot()
        pm.on_view_canvas_changed.assert_called_once_with(view)
        view.on_molecule_parameter_changed("CO", "temp")
        assert pm.on_view_canvas_changed.call_count == 2

    def test_population_diagram_menu_redraw_notifies(self):
        from iSLAT.Modules.Plotting.PopulationDiagramView import PopulationDiagramView
        pm = self._pm()
        view = PopulationDiagramView(pm)
        captured = {}
        view._build_population_diagram_menu = lambda plot, widget, draw_idle: captured.setdefault("f", draw_idle)
        view._append_save_figure_item = lambda menu: None
        view.build_context_menu(None, None)
        captured["f"]()
        pm.on_view_canvas_changed.assert_called_once_with(view)

    def test_window_controller_without_hook_is_fine(self):
        from iSLAT.Modules.Plotting.PopulationDiagramView import PopulationDiagramView
        pm = self._pm()
        del pm.on_view_canvas_changed
        view = PopulationDiagramView(pm)
        view._notify_canvas_changed()
