#!/usr/bin/env python3
"""
Notes in the log file (Settings > Enable log file) when the app freezes or
keeps growing, for troubleshooting slowness anywhere in the app:

- A freeze: the app didn't respond for FREEZE_SECONDS or more. Logged with
  where the main thread was stuck (its Python stack, file and line), and when
  it recovers, how long it took and how many memory pages had to be read back
  in meanwhile. A big number there means the app's memory had been moved out
  to disk while it sat idle (e.g. in the tray).
- A slow memory cleanup pass (Python's garbage collection) of SLOW_GC_SECONDS
  or more. Every thread waits while one runs.
- Memory use every MEMORY_LOG_MINUTES and when the app goes to or comes back
  from the tray: how much it holds, how much of that is in RAM, and how many
  windows exist. A window count that keeps climbing points at windows that
  aren't freed after closing.

    start_diagnostics(app)       # once, just before app.exec()
    log_memory("minimized to tray")
"""

import gc
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

FREEZE_SECONDS = 2.0      # Unresponsive this long is logged as a freeze
SLOW_GC_SECONDS = 1.0     # Garbage collection passes this long are logged
MEMORY_LOG_MINUTES = 60   # How often memory use is logged
HEARTBEAT_MS = 250        # How often the main thread checks in
STACK_LINES = 12          # Innermost stack entries logged for a freeze

_started = False
_beat = 0.0               # When the main thread last checked in (_clock())
_beat_faults = 0          # Page faults at that check-in
_slow_gc = []             # (seconds, generation, collected) not logged yet
_gc_start = 0.0


# --------------------------------------------------------------------
# Clock and memory counters
# --------------------------------------------------------------------
def _make_clock():
    """Seconds, not counting time the computer was asleep - waking the PC
    isn't a freeze."""
    if sys.platform == "win32":
        try:
            import ctypes
            query = ctypes.windll.kernel32.QueryUnbiasedInterruptTime
            value = ctypes.c_ulonglong()

            def clock():
                query(ctypes.byref(value))
                return value.value / 1e7  # 100 ns units
            clock()
            return clock
        except Exception:
            pass
    return time.monotonic  # Already leaves out sleep on macOS and Linux


_clock = _make_clock()


def _make_memory_reader():
    """A function returning (held MB, in-RAM MB, page faults so far), or None."""
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes as wt

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wt.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
            process = kernel32.GetCurrentProcess()
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)

            def read():
                if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                    return None
                return (counters.PrivateUsage / 2**20, counters.WorkingSetSize / 2**20,
                        counters.PageFaultCount)
            read()
            return read
        except Exception:
            return lambda: None
    if sys.platform.startswith("linux"):
        import resource

        def read():
            try:
                held = ram = None
                with open("/proc/self/status") as status:
                    for line in status:
                        if line.startswith("VmData:"):
                            held = int(line.split()[1]) / 1024
                        elif line.startswith("VmRSS:"):
                            ram = int(line.split()[1]) / 1024
                return held, ram, resource.getrusage(resource.RUSAGE_SELF).ru_majflt
            except Exception:
                return None
        return read
    return lambda: None


_read_memory = _make_memory_reader()


def _faults():
    info = _read_memory()
    return info[2] if info else 0


# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
def log_memory(reason):
    """Log memory use and the window count (call on the main thread)."""
    info = _read_memory()
    parts = []
    if info:
        held, ram, _ = info
        if held is not None:
            parts.append(f"{held:,.0f} MB held")
        if ram is not None:
            parts.append(f"{ram:,.0f} MB in RAM")
    try:
        from PySide6.QtWidgets import QApplication
        windows = QApplication.topLevelWidgets()
        parts.append(f"{len(windows)} windows ({sum(1 for w in windows if w.isVisible())} on screen)")
    except Exception:
        pass
    if parts:
        logger.info(f"Memory ({reason}): {', '.join(parts)}")


def _on_gc(phase, info):
    # Only note it here: logging from inside a collection could take locks mid-collection
    global _gc_start
    if phase == "start":
        _gc_start = time.perf_counter()
    else:
        seconds = time.perf_counter() - _gc_start
        if seconds >= SLOW_GC_SECONDS:
            _slow_gc.append((seconds, info.get("generation"), info.get("collected")))


def _flush_slow_gc():
    while _slow_gc:
        seconds, generation, collected = _slow_gc.pop(0)
        logger.warning(f"Slow memory cleanup: Python's garbage collection (generation {generation}) "
                       f"took {seconds:.1f} s and freed {collected} objects - every thread waits for it")


def _heartbeat():
    """On the main thread, every HEARTBEAT_MS."""
    global _beat, _beat_faults
    _beat_faults = _faults()
    _beat = _clock()
    _flush_slow_gc()


def _watch(main_thread_id):
    """Background thread: notices when the main thread stops checking in."""
    frozen_since = None
    frozen_faults = 0
    while True:
        time.sleep(HEARTBEAT_MS / 1000)
        beat = _beat
        now = _clock()
        if frozen_since is None:
            if now - beat >= FREEZE_SECONDS:
                frozen_since = beat
                frozen_faults = _beat_faults
                frame = sys._current_frames().get(main_thread_id)
                stack = "".join(traceback.format_stack(frame)[-STACK_LINES:]) if frame else "(unknown)\n"
                logger.warning(f"App not responding for {now - beat:.1f} s so far. "
                               f"The main thread is at:\n{stack.rstrip()}")
        elif beat > frozen_since:
            faults = max(0, _beat_faults - frozen_faults)
            info = _read_memory()
            memory = f", {info[0]:,.0f} MB held, {info[1]:,.0f} MB in RAM" if info and None not in info[:2] else ""
            logger.warning(f"App responding again after {beat - frozen_since:.1f} s "
                           f"({faults:,} memory pages read back in meanwhile{memory})")
            frozen_since = None


def start_diagnostics(app):
    """Start freeze, slow-cleanup and memory logging (once, just before app.exec())."""
    global _started, _beat
    if _started:
        return
    _started = True
    try:
        from PySide6.QtCore import QTimer
        gc.callbacks.append(_on_gc)

        _beat = _clock()
        heartbeat = QTimer(app)
        heartbeat.timeout.connect(_heartbeat)
        heartbeat.start(HEARTBEAT_MS)

        memory_timer = QTimer(app)
        memory_timer.timeout.connect(lambda: log_memory("hourly" if MEMORY_LOG_MINUTES == 60
                                                        else f"every {MEMORY_LOG_MINUTES} min"))
        memory_timer.start(MEMORY_LOG_MINUTES * 60 * 1000)

        threading.Thread(target=_watch, args=(threading.main_thread().ident,),
                         name="FreezeWatch", daemon=True).start()
        log_memory("at start")
    except Exception as e:
        logger.warning(f"Could not start diagnostics logging: {e}")
