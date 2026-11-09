# -*- coding: utf-8 -*-
"""Process Tk events in tests without the risk of ``update()`` never returning.

On macOS, once a hidden (withdrawn) window has something to redraw, Tk keeps
queuing a redraw task that can only finish when the window is drawn, which a
hidden window never is.  ``update()`` handles events until none are left, so
it then loops forever.  This affects every later ``update()`` in the
process, whichever test created the window.  :func:`pump_tk` processes
events in short slices and always returns.
"""
import time

import _tkinter

#: Longest time (s) events are processed before ``until`` is checked again.
SLICE_S = 0.05

_EVENTS = _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT


def pump_tk(widget, seconds, until=None):
    """Process Tk events (timers, window and idle events) for *seconds*, or until ``until()`` is true.

    Returns whether *until* came true (``True`` when no *until* is given).
    """
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if until is not None and until():
            return True
        slice_end = min(end, time.monotonic() + SLICE_S)
        while time.monotonic() < slice_end and widget.tk.dooneevent(_EVENTS):
            pass
        time.sleep(0.01)
    return until is None or bool(until())
