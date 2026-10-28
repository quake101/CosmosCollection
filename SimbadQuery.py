#!/usr/bin/env python3
"""
SIMBAD lookups with a backup server. CDS (Strasbourg) runs SIMBAD, and
Harvard (CfA) runs a mirror of it. When CDS doesn't answer, a lookup goes to
the mirror, which is then tried first for the next 15 minutes.

    result = query(lambda simbad: simbad.query_region(coords, radius=3 * u.arcmin),
                   fields=('otype', 'V'))

SimbadUnavailable's message tells the user why: no internet, or no SIMBAD
server answering.
"""

import logging
import sys
import time

import requests

from OnlineLimits import online_request

logger = logging.getLogger(__name__)

# astroquery.simbad.conf.servers_list - the first is the main one
MAIN_SERVER = "simbad.cds.unistra.fr"
BACKUP_SERVER = "simbad.harvard.edu"

TIMEOUT = (8, 30)          # Seconds to connect, and to wait for more of the answer
REMEMBER_BACKUP = 15 * 60  # Seconds the backup is tried first, after CDS didn't answer

# Text in a network error that means this computer couldn't look up the
# server's address or has no route out - no internet, rather than a server down
_NO_INTERNET_MARKERS = ("NameResolutionError", "getaddrinfo failed", "Failed to resolve",
                        "Name or service not known", "Network is unreachable", "WinError 10051",
                        "WinError 10065")

_backup_until = 0.0  # time.monotonic() until which the backup is tried first


class SimbadUnavailable(ConnectionError):
    """No SIMBAD server answered. The message is for the user."""


class _NetworkWatch:
    """Watches the requests made through one Simbad's session: each gets a
    timeout, and a failure to get an answer (or a server error) is noted -
    pyvo wraps those in its own errors, sometimes dropping the cause."""

    def __init__(self, session, verify=True):
        self.failure = None
        original_request = session.request

        def request(method, url, *args, **kwargs):
            kwargs.setdefault('timeout', TIMEOUT)
            if not verify:
                # Not session.verify: the REQUESTS_CA_BUNDLE environment variable
                # (set at startup for PyInstaller builds) overrides it
                kwargs['verify'] = False
            try:
                response = original_request(method, url, *args, **kwargs)
            except requests.RequestException as e:
                self.failure = e
                raise
            if response.status_code >= 500:
                self.failure = f"HTTP {response.status_code} {response.reason} from {url}"
            return response

        session.request = request


def _try(server, run, fields, verify=True):
    """run(simbad) on server: (result, None), or (None, failure) when the
    server didn't answer. Other errors - SIMBAD answered with one - are raised."""
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.server = server
    network = _NetworkWatch(simbad._session, verify)
    try:
        with online_request("simbad"):
            if fields:
                # Asks the server which fields it has
                simbad.add_votable_fields(*fields)
            return run(simbad), None
    except Exception:
        if network.failure is None:
            raise
        return None, network.failure


def query(run, fields=()):
    """run(simbad) - a Simbad with fields added - on CDS, or on its Harvard
    mirror when CDS doesn't answer. Raises SimbadUnavailable when neither
    answers, OnlineLimits.OfflineError in offline mode."""
    global _backup_until

    servers = [MAIN_SERVER, BACKUP_SERVER]
    if time.monotonic() < _backup_until:
        servers.reverse()

    failures = []
    for server in servers:
        result, failure = _try(server, run, fields)
        if getattr(sys, 'frozen', False) and isinstance(failure, requests.exceptions.SSLError):
            # PyInstaller builds have hit certificate-chain failures. Retry
            # unverified on this lookup's session only - never patch requests
            # globally, which would disable verification app-wide.
            logger.warning(f"SIMBAD SSL verification failed at {server} ({failure}); retrying without verification")
            result, failure = _try(server, run, fields, verify=False)

        if failure is None:
            if server == MAIN_SERVER:
                _backup_until = 0.0
            elif failures:
                logger.info(f"Using the SIMBAD mirror at {server} for the next {REMEMBER_BACKUP // 60} min")
                _backup_until = time.monotonic() + REMEMBER_BACKUP
            return result

        logger.warning(f"SIMBAD at {server} didn't answer: {failure}")
        failures.append(failure)

    if all(any(marker in str(failure) for marker in _NO_INTERNET_MARKERS) for failure in failures):
        raise SimbadUnavailable("No internet connection - couldn't reach SIMBAD.")
    raise SimbadUnavailable("SIMBAD isn't answering right now (tried CDS in Strasbourg "
                            "and its mirror at Harvard). Try again later.")
