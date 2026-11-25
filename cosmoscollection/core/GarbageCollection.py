#!/usr/bin/env python3
"""
When Python's cyclic garbage collector runs.

Left to itself, the collector runs on whichever thread happens to allocate past
its threshold. In this app that has crashed the whole process natively ("Fatal
Python error: Segmentation fault" / "Illegal instruction", with
"Garbage-collecting" at the top of the trace) in two ways:

- A pass on a worker thread frees Qt objects that belong to the GUI thread.
- A pass (on any thread) lands while worker threads are partway through astropy
  or numpy calculations (Best DSO Tonight, the Weather Forecast).

So in the app, automatic collection is off, and the main thread collects on a
timer instead, skipping its turn while a calculation that is unsafe to collect
during is running:

    start_main_thread_collection(app)    # once, before app.exec()

    with collection_paused():            # around such a calculation, any thread
        ...

Outside the app (the command line, tests), collection_paused() turns automatic
collection off for its duration and back on afterwards.
"""

import gc
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

YOUNG_INTERVAL_MS = 1000         # Quick pass (young objects plus an old increment)
FULL_INTERVAL_MS = 5 * 60 * 1000  # Full pass - the quick ones may not reach old cycles

_lock = threading.Lock()
_paused = 0            # Calculations currently running inside collection_paused()
_reenable = False      # Whether the last one out turns automatic collection back on
_full_due = False      # A full pass was skipped while paused - run it next turn
_started = False


@contextmanager
def collection_paused():
    """No garbage collection while this block runs (nests, and works across threads)."""
    global _paused, _reenable
    with _lock:
        if _paused == 0:
            _reenable = gc.isenabled()
            gc.disable()
        _paused += 1
    try:
        yield
    finally:
        with _lock:
            _paused -= 1
            if _paused == 0 and _reenable:
                gc.enable()


def _collect(generation):
    global _full_due
    if _paused:
        if generation == 2:
            _full_due = True
        return
    if _full_due:
        generation, _full_due = 2, False
    gc.collect(generation)


def start_main_thread_collection(app):
    """Collect garbage only on the main thread, on a timer (once, before app.exec())."""
    global _started, _reenable
    if _started:
        return
    _started = True
    try:
        from PySide6.QtCore import QTimer
        young = QTimer(app)
        young.timeout.connect(lambda: _collect(1))
        young.start(YOUNG_INTERVAL_MS)
        full = QTimer(app)
        full.timeout.connect(lambda: _collect(2))
        full.start(FULL_INTERVAL_MS)
        with _lock:
            if _paused:
                _reenable = False  # Stays off when the current pause ends
            gc.disable()
    except Exception as e:
        logger.warning(f"Could not move garbage collection to the main thread: {e}")
