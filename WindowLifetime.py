#!/usr/bin/env python3
"""
Tool windows (Gallery, Session Manager, Target List, ...) free their memory
once they're closed.

A window's signals connected to its own lambdas and methods keep it alive
through Qt's side, where Python's garbage collector can't see. So a closed
window dropped from Python was never freed, and every reopen left another full
copy behind - most of the memory a long-running tray session held (2 GB after
a day and a half). release_on_close() deletes the window once it has closed
and nothing it started is still running.

    class MyToolWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            release_on_close(self)

Code that keeps a window to reuse while it's open checks it with
window_is_open(), which is False once the window is closed or deleted:

    if not window_is_open(getattr(self, 'gallery_window', None)):
        self.gallery_window = DSOGalleryWindow()
"""

import logging

import shiboken6
from PySide6.QtCore import QEvent, QObject, QThread, QThreadPool, QTimer
from PySide6.QtWidgets import QApplication, QDialog

logger = logging.getLogger(__name__)

RETRY_MS = 1000  # How often to check again while a closed window is still busy

_open = set()        # Windows set to release - kept alive here until released
_kept_hidden = set()  # Windows to keep when closed, for reuse (see keep_when_closed)


def window_is_open(window):
    """Whether window is a live window on screen (False for None, closed or deleted)."""
    try:
        return window is not None and shiboken6.isValid(window) and window.isVisible()
    except RuntimeError:
        return False


def release_on_close(window):
    """Delete window once it's closed and idle (see the module notes)."""
    _open.add(window)
    window.installEventFilter(_close_watcher())
    if isinstance(window, QDialog):
        # Esc and the dialog's own buttons hide it without a close event
        window.finished.connect(lambda *_: QTimer.singleShot(0, lambda: _release_if_closed(window)))


def keep_when_closed(window):
    """Keep window when it's closed instead, for code that shows it again later
    (one shared instance, so it can't pile up)."""
    _kept_hidden.add(window)


def _busy_reason(window):
    """Why window can't be deleted yet, or None when it can."""
    # Its own threads: deleting a running QThread takes the app down
    candidates = list(window.findChildren(QThread))
    for value in list(vars(window).values()):
        if isinstance(value, (list, tuple, set)):
            candidates.extend(value)
        elif isinstance(value, dict):
            candidates.extend(value.values())
        else:
            candidates.append(value)
    for value in candidates:
        try:
            if isinstance(value, QThread) and shiboken6.isValid(value) and value.isRunning():
                return f"thread {type(value).__name__} still running"
            if isinstance(value, QThreadPool) and shiboken6.isValid(value) and value.activeThreadCount():
                return "thread pool still busy"
        except RuntimeError:
            continue
    # Windows it opened as its children still on screen would close with it
    for widget in QApplication.topLevelWidgets():
        if widget is window or not widget.isVisible():
            continue
        parent = widget.parentWidget()
        while parent is not None:
            if parent is window:
                return f"child window {type(widget).__name__} still open"
            parent = parent.parentWidget()
    return None


def _release_if_closed(window):
    if not shiboken6.isValid(window):
        _open.discard(window)
        return
    if window.isVisible() or window in _kept_hidden:
        return  # Its close was cancelled, or it's kept for reuse
    reason = _busy_reason(window)
    if reason:
        logger.debug(f"Closed {type(window).__name__} kept for now: {reason}")
        QTimer.singleShot(RETRY_MS, lambda: _release_if_closed(window))
        return
    logger.debug(f"Releasing closed {type(window).__name__}")
    window.removeEventFilter(_close_watcher())
    window.destroyed.connect(lambda *_: QTimer.singleShot(0, lambda: _forget(window)))
    window.deleteLater()


def _forget(window):
    """After Qt deleted the window: drop its Python attributes too (data lists,
    images, workers), which would otherwise wait for whoever still holds the
    window object to replace it."""
    _open.discard(window)
    try:
        vars(window).clear()
    except Exception as e:
        logger.debug(f"Could not clear a released window's attributes: {e}")


class _CloseWatcher(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close:
            # After the close finishes: closeEvent may still cancel it
            QTimer.singleShot(0, lambda window=obj: _release_if_closed(window))
        return False


_watcher = None


def _close_watcher():
    global _watcher
    if _watcher is None:
        _watcher = _CloseWatcher(QApplication.instance())
    return _watcher
