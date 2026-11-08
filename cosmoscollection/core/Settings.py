"""The app's settings store (QtCore only, so the command line shares it)."""

from PySide6.QtCore import QSettings

SETTINGS_ORG = "CosmosCollection"
SETTINGS_APP = "CosmosCollection"


def app_settings():
    """The QSettings every window and command reads and writes."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)
