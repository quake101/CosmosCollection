#!/usr/bin/env python3
"""
Limits on how many requests go to each online astronomy service at once,
shared by every window and thread. CDS, which runs SIMBAD and VizieR, asks
clients to keep their request rate low and temporarily blocks ones that
flood it - e.g. star names looked up for a whole gallery at once.

    with online_request("simbad"):
        result = simbad.query_region(...)

Also offline mode (Settings > Application Settings): while it's on, nothing
in the app goes online. Background work skips quietly; what the user asked
for says why with OFFLINE_NOTE / notify_offline(). Links that open the
user's web browser, and NINA (on the local network), still work.
"""

import threading

LIMITS = {"simbad": 2, "vizier": 2}   # Requests at once, per service

_slots = {service: threading.BoundedSemaphore(count) for service, count in LIMITS.items()}

OFFLINE_SETTING = "offline_mode"
OFFLINE_NOTE = "Offline mode is on (Settings > Application Settings)."


class OfflineError(ConnectionError):
    """An online request was made while offline mode is on."""

    def __init__(self, message=OFFLINE_NOTE):
        super().__init__(message)


def is_offline():
    """Whether offline mode is on - read each time, so a change in Settings
    takes effect at once, from any thread."""
    try:
        from PySide6.QtCore import QSettings
        return QSettings("CosmosCollection", "CosmosCollection").value(OFFLINE_SETTING, False, type=bool)
    except Exception:
        return False


def notify_offline(parent, feature):
    """Tell the user feature (e.g. "The weather forecast") needs the internet."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(parent, "Offline Mode", f"{feature} needs the internet.\n\n{OFFLINE_NOTE}")


class _Request:
    """One of a service's slots, held for a request - OfflineError instead
    while offline mode is on."""

    def __init__(self, slot):
        self._slot = slot

    def __enter__(self):
        if is_offline():
            raise OfflineError()
        self._slot.acquire()
        return self

    def __exit__(self, *exc):
        self._slot.release()
        return False


def online_request(service):
    """A context manager that waits for one of the service's slots and holds
    it for the request."""
    return _Request(_slots[service])
