"""
TimeFormatHelper.py - Utility module for consistent time formatting throughout the app.

Reads the user's time format preference from QSettings and provides helper
functions to format time and datetime objects accordingly.
"""

from PySide6.QtCore import QSettings


def get_time_format_24h():
    """Check if the user prefers 24-hour time format.

    Returns:
        bool: True if 24-hour format, False if 12-hour format.
    """
    settings = QSettings("CosmosCollection", "CosmosCollection")
    time_format = settings.value("time_format", "12-hour", type=str)
    return time_format == "24-hour"


def format_time(dt, seconds=False):
    """Format only the time portion of a datetime object.

    Args:
        dt: A datetime object.
        seconds: If True, include seconds in the output.

    Returns:
        str: Formatted time string (e.g., "2:30 PM" or "14:30").
    """
    if get_time_format_24h():
        fmt = "%H:%M:%S" if seconds else "%H:%M"
    else:
        fmt = "%I:%M:%S %p" if seconds else "%I:%M %p"
    return dt.strftime(fmt)


def format_datetime(dt, seconds=False):
    """Format a datetime object with both date and time.

    Args:
        dt: A datetime object.
        seconds: If True, include seconds in the output.

    Returns:
        str: Formatted datetime string (e.g., "2024-01-15 2:30 PM" or "2024-01-15 14:30").
    """
    if get_time_format_24h():
        fmt = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    else:
        fmt = "%Y-%m-%d %I:%M:%S %p" if seconds else "%Y-%m-%d %I:%M %p"
    return dt.strftime(fmt)
