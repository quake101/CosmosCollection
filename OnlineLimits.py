#!/usr/bin/env python3
"""
Limits on how many requests go to each online astronomy service at once,
shared by every window and thread. CDS, which runs SIMBAD and VizieR, asks
clients to keep their request rate low and temporarily blocks ones that
flood it - e.g. star names looked up for a whole gallery at once.

    with online_request("simbad"):
        result = simbad.query_region(...)
"""

import threading

LIMITS = {"simbad": 2, "vizier": 2}   # Requests at once, per service

_slots = {service: threading.BoundedSemaphore(count) for service, count in LIMITS.items()}


def online_request(service):
    """A context manager that waits for one of the service's slots and holds
    it for the request."""
    return _slots[service]
