"""
HorizonProfile.py - Custom horizon support for visibility calculations.

Parses horizon files exported from NINA (.hrz) and Stellarium (polygonal
landscape: landscape.ini + horizon list, or a packaged landscape .zip) into
a HorizonProfile: a set of azimuth/altitude points describing where trees,
houses, etc. block the sky. The profile is stored per location in the
usersettings table and applied to the active location's visibility checks.
"""

import configparser
import io
import json
import logging
import math
import posixpath
import sqlite3
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Stellarium polygonal_horizon_list_mode values -> (azimuth unit, second column)
_STELLARIUM_MODES = {
    "azdeg_altdeg": ("deg", "alt"),
    "azdeg_zddeg": ("deg", "zd"),
    "azrad_altrad": ("rad", "alt"),
    "azrad_zdrad": ("rad", "zd"),
    "azgrad_altgrad": ("grad", "alt"),
    "azgrad_zdgrad": ("grad", "zd"),
}

_UNIT_TO_DEG = {
    "deg": 1.0,
    "rad": 180.0 / math.pi,
    "grad": 0.9,
}


class HorizonParseError(ValueError):
    """Raised when a horizon file can't be read or contains invalid data."""


class HorizonProfile:
    """Horizon altitude as a function of azimuth for one observing location.

    Points are (azimuth, altitude) in degrees, azimuth measured from North
    through East (0-360). Altitude between points is linearly interpolated,
    wrapping around North.
    """

    def __init__(self, points, name=None):
        """
        Args:
            points: Iterable of (azimuth, altitude) pairs in degrees.
            name: Display name for the profile, usually the source file name.
        """
        self.name = name
        self.points = _normalize_points(points)

        # Pad one point past each end so interpolation wraps around North
        az = np.array([p[0] for p in self.points])
        alt = np.array([p[1] for p in self.points])
        self._az = np.concatenate(([az[-1] - 360.0], az, [az[0] + 360.0]))
        self._alt = np.concatenate(([alt[-1]], alt, [alt[0]]))

    def altitude_at(self, azimuth):
        """Horizon altitude in degrees at the given azimuth(s).

        Args:
            azimuth: Azimuth in degrees, scalar or numpy array.

        Returns:
            float or numpy array matching the input shape.
        """
        result = np.interp(np.mod(azimuth, 360.0), self._az, self._alt)
        return float(result) if np.ndim(result) == 0 else result

    @property
    def max_altitude(self):
        """Highest altitude in the profile, in degrees."""
        return max(alt for _, alt in self.points)

    def summary(self):
        """Short description for display, e.g. 'backyard.hrz, 72 points, max 34°'."""
        parts = [self.name] if self.name else []
        parts.append(f"{len(self.points)} points")
        parts.append(f"max {self.max_altitude:.0f}°")
        return ", ".join(parts)

    def to_json(self):
        """Serialize the points for storage in usersettings.horizon_points."""
        return json.dumps([[round(az, 4), round(alt, 4)] for az, alt in self.points])

    @classmethod
    def from_json(cls, text, name=None):
        """Rebuild a profile from to_json() output. Returns None if text is empty."""
        if not text:
            return None
        return cls(json.loads(text), name=name)


def _normalize_points(points):
    """Validate points, wrap azimuths into 0-360, sort, and merge duplicates.

    Where several points share an azimuth (a vertical edge such as a wall),
    the highest altitude is kept so the horizon errs on the blocked side.
    """
    merged = {}
    for az, alt in points:
        az, alt = float(az), float(alt)
        if not (math.isfinite(az) and math.isfinite(alt)):
            raise HorizonParseError("Horizon contains a non-numeric azimuth or altitude.")
        if not -90.0 <= alt <= 90.0:
            raise HorizonParseError(f"Altitude {alt:g}° at azimuth {az:g}° is outside -90° to 90°.")
        az = round(az % 360.0, 6) % 360.0
        merged[az] = max(alt, merged.get(az, -90.0))

    if len(merged) < 2:
        raise HorizonParseError("A horizon needs at least two points with different azimuths.")
    return sorted(merged.items())


def _parse_point_lines(text, source, az_scale=1.0, alt_scale=1.0, zenith_distance=False, rotation=0.0):
    """Parse 'azimuth altitude' pairs, one per line.

    Columns may be separated by whitespace, commas, or semicolons; extra
    columns are ignored. Blank lines and lines starting with '#' or ';' are
    skipped, as are non-numeric lines before the first point (CSV headers).
    """
    points = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line[0] in "#;":
            continue
        columns = line.replace(",", " ").replace(";", " ").split()
        try:
            first, second = float(columns[0]), float(columns[1])
        except (ValueError, IndexError):
            if not points:
                continue  # Header line
            raise HorizonParseError(f"{source}, line {line_number}: expected 'azimuth altitude', got '{line}'.")
        az = first * az_scale + rotation
        alt = second * alt_scale
        if zenith_distance:
            alt = 90.0 - alt
        points.append((az, alt))

    if not points:
        raise HorizonParseError(f"{source} doesn't contain any azimuth/altitude points.")
    return points


def _read_text(data):
    """Decode file bytes, tolerating a UTF-8 BOM and non-UTF-8 exports."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _parse_stellarium_ini(ini_text, read_file, source):
    """Parse a Stellarium landscape.ini and the horizon list it references.

    Args:
        ini_text: Contents of landscape.ini.
        read_file: Callable taking the list file name (relative to the ini)
            and returning its bytes.
        source: Name used in error messages.

    Returns:
        tuple: (points, landscape name or None)
    """
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(ini_text)
    except configparser.Error as e:
        raise HorizonParseError(f"{source} is not a valid landscape.ini: {e}")

    if not parser.has_section("landscape"):
        raise HorizonParseError(f"{source} has no [landscape] section.")
    section = parser["landscape"]

    def value(key, default=""):
        return section.get(key, default).strip().strip('"')

    list_name = value("polygonal_horizon_list")
    if not list_name:
        raise HorizonParseError(
            f"{source} doesn't define a horizon line (polygonal_horizon_list). "
            "Only Stellarium landscapes with a polygonal horizon can be imported."
        )

    mode = value("polygonal_horizon_list_mode", "azDEG_altDEG")
    if mode.lower() not in _STELLARIUM_MODES:
        raise HorizonParseError(f"{source}: unsupported polygonal_horizon_list_mode '{mode}'.")
    unit, second_column = _STELLARIUM_MODES[mode.lower()]

    try:
        rotation = float(value("polygonal_angle_rotatez", "0") or 0)
    except ValueError:
        raise HorizonParseError(f"{source}: polygonal_angle_rotatez is not a number.")

    try:
        list_data = read_file(list_name)
    except (OSError, KeyError):
        raise HorizonParseError(f"Horizon list '{list_name}' referenced by {source} was not found.")

    scale = _UNIT_TO_DEG[unit]
    points = _parse_point_lines(
        _read_text(list_data), list_name,
        az_scale=scale, alt_scale=scale,
        zenith_distance=(second_column == "zd"), rotation=rotation,
    )
    return points, value("name") or None


def _parse_stellarium_zip(path):
    """Parse a packaged Stellarium landscape (.zip containing landscape.ini)."""
    try:
        with zipfile.ZipFile(path) as archive:
            ini_names = [n for n in archive.namelist() if posixpath.basename(n).lower() == "landscape.ini"]
            if not ini_names:
                raise HorizonParseError(f"{path.name} doesn't contain a Stellarium landscape.ini.")
            ini_name = min(ini_names, key=len)  # Prefer the top-most one
            ini_dir = posixpath.dirname(ini_name)

            def read_file(name):
                return archive.read(posixpath.join(ini_dir, name.replace("\\", "/")))

            return _parse_stellarium_ini(_read_text(archive.read(ini_name)), read_file, path.name)
    except zipfile.BadZipFile:
        raise HorizonParseError(f"{path.name} is not a valid zip file.")


def parse_horizon_file(path):
    """Read a NINA or Stellarium horizon file into a HorizonProfile.

    Supported inputs:
        - NINA .hrz, or any text/CSV file of 'azimuth altitude' pairs in degrees
        - Stellarium landscape.ini (reads the polygonal horizon list it references)
        - Stellarium landscape .zip

    Raises:
        HorizonParseError: The file is missing, unreadable, or has invalid data.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".zip":
            points, name = _parse_stellarium_zip(path)
        else:
            text = _read_text(path.read_bytes())
            if path.suffix.lower() == ".ini" or "[landscape]" in text.lower():
                points, name = _parse_stellarium_ini(
                    text, lambda list_name: (path.parent / list_name).read_bytes(), path.name)
            else:
                points, name = _parse_point_lines(text, path.name), None
    except OSError as e:
        raise HorizonParseError(f"Could not read {path.name}: {e.strerror or e}")

    # Stellarium's landscape name is friendlier than "landscape.ini"
    display_name = f"{name} ({path.name})" if name else path.name
    profile = HorizonProfile(points, name=display_name)
    logger.debug(f"Parsed horizon {profile.summary()}")
    return profile


def _load_horizon(conn, queries, description):
    """Run queries in order until one returns a row, and build its HorizonProfile.

    Opens its own sqlite3 connection when conn is None, so it's safe to call
    from worker threads (the DatabaseManager singleton's connection is bound
    to the thread that created it).

    Args:
        conn: sqlite3 connection, or None to open a fresh one.
        queries: List of (sql, params) selecting (horizon_points, horizon_name).
        description: Which location, for the warning logged on failure.
    """
    own_conn = conn is None
    if own_conn:
        from ResourceManager import ResourceManager
        conn = sqlite3.connect(str(ResourceManager.get_database_path()))
    try:
        cursor = conn.cursor()
        row = None
        for sql, params in queries:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            if row:
                break
        if not row or not row[0]:
            return None
        return HorizonProfile.from_json(row[0], name=row[1])
    except (sqlite3.Error, ValueError, TypeError) as e:
        logger.warning(f"Could not load custom horizon for {description}: {e}")
        return None
    finally:
        if own_conn:
            conn.close()


def load_active_horizon(conn=None):
    """Load the custom horizon of the active location from the database.

    Returns:
        HorizonProfile or None if the active location has no custom horizon.
    """
    return _load_horizon(conn, [
        ("SELECT horizon_points, horizon_name FROM usersettings WHERE is_active = 1 LIMIT 1", ()),
        ("SELECT horizon_points, horizon_name FROM usersettings ORDER BY id DESC LIMIT 1", ()),
    ], "the active location")


def load_horizon_for_coordinates(lat, lon, conn=None):
    """Load the custom horizon of the saved location at the given coordinates.

    Sessions store a copy of their location's lat/lon rather than a reference
    to the usersettings row, so the saved location is found by coordinates.
    If several saved locations share the coordinates, the active one wins,
    then one that has a horizon.

    Returns:
        HorizonProfile or None if no saved location matches (e.g. a session's
        custom location) or the matching location has no custom horizon.
    """
    if lat is None or lon is None:
        return None
    return _load_horizon(conn, [(
        """SELECT horizon_points, horizon_name FROM usersettings
           WHERE ABS(location_lat - ?) < 1e-6 AND ABS(location_lon - ?) < 1e-6
           ORDER BY is_active DESC, horizon_points IS NULL, id LIMIT 1""",
        (lat, lon),
    )], f"location {lat}, {lon}")
