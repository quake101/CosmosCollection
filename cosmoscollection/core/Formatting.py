"""Shared text formatting: coordinates, durations, numbers, counts and sort keys.

No Qt: the command line uses these too. Every window and command formats these
values through here so they read the same everywhere."""

import re


def format_ra(ra_deg):
    """RA in degrees as '05h 35m 17.3s' ('' for None). Rounded before splitting,
    so it never shows 60.0s."""
    if ra_deg is None:
        return ""
    total_seconds = round((float(ra_deg) % 360.0) / 15.0 * 3600.0, 1) % 86400.0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02.0f}h {minutes:02.0f}m {seconds:04.1f}s"


def format_dec(dec_deg):
    """Dec in degrees as '+22° 00' 52"' ('' for None)."""
    if dec_deg is None:
        return ""
    dec_deg = float(dec_deg)
    total_arcsec = round(abs(dec_deg) * 3600.0)
    sign = '-' if dec_deg < 0 and total_arcsec else '+'  # A hair below 0 rounds to +00° 00' 00"
    degrees, remainder = divmod(total_arcsec, 3600)
    arcmin, arcsec = divmod(remainder, 60)
    return f"{sign}{degrees:02d}° {arcmin:02d}' {arcsec:02d}\""


def trim_number(value, decimals):
    """'300' or '0.25': at most `decimals` places, without trailing zeros."""
    return f"{float(value):.{decimals}f}".rstrip('0').rstrip('.')


def format_duration(seconds):
    """'12h 30m', '45m', '30s': an integration (or any) time span."""
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{trim_number(seconds, 1)}s"
    hours, minutes = divmod(round(seconds / 60.0), 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h" if hours else f"{minutes}m"


# The catalog writes 79.9 (dark nebulae) or 99.9 for "magnitude unknown", and a
# Target List entry added without one has 0
UNKNOWN_MAGNITUDE_FROM = 50.0


def known_magnitude(value):
    """An object's magnitude as a number, or None when it isn't known (blank,
    0, or the catalog's 79.9 / 99.9 stand-ins)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0 < value < UNKNOWN_MAGNITUDE_FROM else None


def format_magnitude(value, decimals=1, unknown=""):
    """'6.5' - or `unknown` (default blank) when it isn't known."""
    value = known_magnitude(value)
    return f"{value:.{decimals}f}" if value is not None else unknown


def format_exposure(seconds):
    """'300s', '0.5s': one sub's exposure."""
    return f"{trim_number(seconds or 0, 3)}s"


def count_text(count, word, words=None, bold=False):
    """'1 image', '3 images', '1,204 files'; bold puts the number in <b> for rich text."""
    number = f"<b>{count:,}</b>" if bold else f"{count:,}"
    return f"{number} {word if count == 1 else (words or word + 's')}"


_NATURAL_SPLIT_RE = re.compile(r"(\d+)")


def natural_key(text):
    """Sort key that orders embedded numbers numerically ("M 2" before "M 10")."""
    return tuple((0, int(part)) if part.isdecimal() else (1, part.lower())
                 for part in _NATURAL_SPLIT_RE.split(text or "") if part)
