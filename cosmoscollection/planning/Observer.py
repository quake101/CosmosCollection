"""
Observing locations: the user's saved locations (usersettings rows, one of
them active) and the observer a calculation uses.

No UI imports - used by the planning windows and the command line.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Site:
    """One saved location (or a typed-in one: id None)."""
    id: object
    name: str
    lat: float
    lon: float
    timezone: str = None
    is_active: bool = False
    horizon_name: str = None
    has_horizon: bool = False

    def to_dict(self):
        return {"id": self.id, "name": self.name, "lat": self.lat, "lon": self.lon, "timezone": self.timezone,
                "active": self.is_active, "horizon": self.horizon_name if self.has_horizon else None}


def load_observer(conn):
    """(lat, lon, timezone name or None) of the active location (see
    active_site), or None."""
    site = active_site(conn)
    return (site.lat, site.lon, site.timezone) if site else None


def coordinates_text(lat, lon):
    """'40.71°N, 74.01°W'"""
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'W' if lon < 0 else 'E'}"


def local_zone(tz_name):
    """tzinfo for the location (zoneinfo, which unlike pytz doesn't scan every
    zone file on first use); this computer's zone if unknown."""
    from datetime import timezone as dt_timezone
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name) if tz_name else datetime.now().astimezone().tzinfo
    except Exception:
        return datetime.now().astimezone().tzinfo or dt_timezone.utc


def load_sites(conn):
    """Every saved location with coordinates, in the order they were added."""
    rows = conn.execute("""
        SELECT id, location_name, location_lat, location_lon, timezone, is_active, horizon_name,
               horizon_points IS NOT NULL AND horizon_points <> ''
        FROM usersettings WHERE location_lat IS NOT NULL AND location_lon IS NOT NULL ORDER BY id
    """).fetchall()
    return [Site(row[0], row[1] or f"Location {row[0]}", row[2], row[3], row[4], bool(row[5]), row[6], bool(row[7]))
            for row in rows]


def active_site(conn):
    """The active saved location - or, as load_observer does, the newest one
    when none is marked active; None when there are none."""
    sites = load_sites(conn)
    return next((s for s in sites if s.is_active), sites[-1] if sites else None)


def find_site(conn, name):
    """The saved location named name (case ignored) or with that id; None if none."""
    text = str(name or "").strip()
    for site in load_sites(conn):
        if site.name.strip().lower() == text.lower() or str(site.id) == text:
            return site
    return None
