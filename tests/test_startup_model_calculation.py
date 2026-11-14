# -*- coding: utf-8 -*-
"""The startup model calculation runs while the window is built, so the first draw shows the models.

* iSLAT.init_gui starts GUI.start_model_calculation (the visible molecules'
  summed flux, on a background thread) before it builds the GUI, and hands
  the result queue to the GUI as ``model_calculation``.
* GUI.display_spectrum_async (run once the window is shown) puts a finished
  calculation's models on the plot at once, without the loading indicator:
  the plot's held first draw then already shows them, instead of a draw of
  the observed spectrum alone followed by a second draw with the models.
* A calculation still running is waited for as before (loading indicator,
  polling), without starting a second one; with no calculation handed over,
  the GUI starts its own.
"""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def _gui(pending=None):
    """A GUI with mocked window and plot; *pending* is the handed-over result queue."""
    from iSLAT.Modules.GUI.GUI import GUI
    gui = GUI.__new__(GUI)
    gui.master = MagicMock()
    gui.plot = MagicMock()
    gui.islat_class = SimpleNamespace()
    gui._pending_model_calculation = pending
    return gui


def _result_queue(*results):
    q = queue.Queue()
    for result in results:
        q.put(result)
    return q


@pytest.fixture
def started_threads(monkeypatch):
    """Record the targets of threads the GUI module starts (the threads do not run)."""
    import sys
    import iSLAT.Modules.GUI.GUI  # noqa: F401  (the package attribute is the class, not the module)
    gui_module = sys.modules["iSLAT.Modules.GUI.GUI"]
    targets = []

    class RecordingThread:
        def __init__(self, target=None, args=(), daemon=None, **kwargs):
            self.target = target

        def start(self):
            targets.append(self.target)

    monkeypatch.setattr(gui_module.threading, "Thread", RecordingThread)
    return targets


@pytest.fixture
def callback_results():
    results = []

    def callback(success):
        results.append(success)

    callback.results = results
    return callback


class TestDisplaySpectrumAsync:
    def test_finished_calculation_goes_on_the_plot_without_the_indicator(self, started_threads, callback_results):
        gui = _gui(_result_queue(("success", None)))
        gui.display_spectrum_async(callback=callback_results)
        gui.plot.show_loading_indicator.assert_not_called()
        gui.plot.update_model_plot.assert_called_once()
        gui.plot._set_initial_zoom_range.assert_called_once()
        gui.plot.canvas.draw.assert_not_called()   # asks for a draw; the held canvas makes it
        gui.master.after.assert_not_called()       # nothing left to poll
        assert started_threads == []
        assert callback_results.results == [True]

    def test_running_calculation_is_polled_with_the_indicator(self, started_threads, callback_results):
        pending = _result_queue()
        gui = _gui(pending)
        gui.display_spectrum_async(callback=callback_results)
        gui.plot.show_loading_indicator.assert_called_once()
        gui.plot.update_model_plot.assert_not_called()
        gui.master.after.assert_called_once_with(50, gui._poll_async_result)
        assert started_threads == []               # no second calculation
        assert gui._async_result_queue is pending

        pending.put(("success", None))
        gui._poll_async_result()
        gui.plot.hide_loading_indicator.assert_called_once()
        gui.plot.update_model_plot.assert_called_once()
        assert callback_results.results == [True]

    def test_failed_calculation_is_reported(self, started_threads, callback_results):
        gui = _gui(_result_queue(("error", "boom")))
        gui.display_spectrum_async(callback=callback_results)
        gui.plot.update_model_plot.assert_not_called()
        assert callback_results.results == [False]

    def test_without_a_calculation_it_starts_its_own(self, started_threads):
        gui = _gui(None)
        gui.display_spectrum_async()
        gui.plot.show_loading_indicator.assert_called_once()
        assert started_threads == [gui._async_spectrum_calculation]

    def test_handed_over_calculation_is_used_once(self, started_threads):
        gui = _gui(_result_queue(("success", None)))
        gui.display_spectrum_async()
        gui.display_spectrum_async()
        assert started_threads == [gui._async_spectrum_calculation]


class TestStartModelCalculation:
    @staticmethod
    def _islat(side_effect=None):
        threads = []

        def summed_flux(wave_data, visible_only=True):
            threads.append(threading.current_thread())
            if side_effect is not None:
                raise side_effect
            return wave_data, np.zeros_like(wave_data)

        molecules_dict = MagicMock()
        molecules_dict.get_summed_flux_parallel.side_effect = summed_flux
        islat = SimpleNamespace(molecules_dict=molecules_dict, wave_data_original=np.linspace(5.0, 6.0, 4))
        return islat, threads

    def test_computes_the_visible_summed_flux_on_another_thread(self):
        from iSLAT.Modules.GUI.GUI import GUI
        islat, threads = self._islat()
        result_queue = GUI.start_model_calculation(islat)
        assert result_queue.get(timeout=10) == ("success", None)
        islat.molecules_dict.get_summed_flux_parallel.assert_called_once_with(
            islat.wave_data_original, visible_only=True)
        assert len(threads) == 1 and threads[0] is not threading.main_thread()

    def test_reports_an_error(self):
        from iSLAT.Modules.GUI.GUI import GUI
        islat, _threads = self._islat(side_effect=RuntimeError("boom"))
        assert GUI.start_model_calculation(islat).get(timeout=10) == ("error", "boom")


class TestInitGui:
    @pytest.fixture
    def recording_gui_class(self, monkeypatch):
        """Replace the GUI class init_gui builds; record the order of calculation start and construction."""
        import iSLAT.Modules.GUI as gui_package
        events = []
        handed_over = queue.Queue()

        class RecordingGUI:
            @staticmethod
            def start_model_calculation(islat_class):
                events.append("start calculation")
                return handed_over

            def __init__(self, **kwargs):
                events.append("build GUI")
                self.kwargs = kwargs

            def start(self, display_spectrum_async=True):
                events.append("start GUI")

        monkeypatch.setattr(gui_package, "GUI", RecordingGUI)
        RecordingGUI.events = events
        RecordingGUI.handed_over = handed_over
        return RecordingGUI

    @staticmethod
    def _app():
        from iSLAT.iSLATClass import iSLAT
        app = iSLAT()
        app._user_settings = {"theme": {}}
        return app

    def test_starts_the_calculation_before_building_the_window(self, recording_gui_class):
        app = self._app()
        app.wave_data_original = np.linspace(5.0, 6.0, 4)
        app.init_gui()
        assert recording_gui_class.events == ["start calculation", "build GUI", "start GUI"]
        assert app.GUI.kwargs["model_calculation"] is recording_gui_class.handed_over

    def test_without_a_spectrum_nothing_is_started(self, recording_gui_class):
        app = self._app()
        app.init_gui()
        assert recording_gui_class.events == ["build GUI", "start GUI"]
        assert app.GUI.kwargs["model_calculation"] is None
