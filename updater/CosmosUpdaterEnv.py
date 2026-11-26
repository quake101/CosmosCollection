"""
The environment the updater gives the programs it starts: the relaunched app,
the progress window, and zenity / kdialog / dbus-send / osascript.

The updater inherits the old app's environment, then its own PyInstaller
start-up adds to it. Passed on unchanged, a relaunched app on Linux would
look for libraries first in folders on LD_LIBRARY_PATH that belong to the
two bundles: the old app's _internal folder and the updater's temporary
extraction folder (deleted moments later). Those copies come from the build
machine; a system library such as Mesa's GL driver picking them up instead
of the system's own can fail on a distribution older than the build machine.

So this takes back out what the bundles and the app put in:
  - LD_LIBRARY_PATH entries inside either bundle (others are the user's
    own and stay), and PyInstaller's LD_LIBRARY_PATH_ORIG
  - PyInstaller's onefile hand-off variables (_PYI_*, _MEIPASS2), which
    must not reach a different frozen program
  - the variables the app sets for its own use (the same list as the app's
    core/UrlOpener.clean_subprocess_env; the updater can't import app code)

Standard library only.
"""

import os
import sys

# Set by the app for itself: app/Bootstrap.py and core/ProcessSetup.py
APP_VARIABLES = (
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "XDG_CACHE_HOME",
)


def _inside(path, folders):
    try:
        path = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for folder in folders:
        if path == folder or path.startswith(folder + os.sep):
            return True
    return False


_bundle_dirs = []


def register_bundle_dirs(*dirs):
    """Folders whose libraries must not leak into started programs: the
    app's install folder and the staged update. This updater's own
    extraction folder is always included."""
    for folder in dirs:
        if folder:
            _bundle_dirs.append(os.path.realpath(folder))


def clean_environment():
    """A copy of os.environ for starting another program."""
    env = os.environ.copy()
    folders = list(_bundle_dirs)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        folders.append(os.path.realpath(meipass))

    library_path = env.get("LD_LIBRARY_PATH")
    if library_path is not None:
        kept = [entry for entry in library_path.split(os.pathsep)
                if entry and not _inside(entry, folders)]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            env.pop("LD_LIBRARY_PATH")
    env.pop("LD_LIBRARY_PATH_ORIG", None)

    for name in list(env):
        if name.startswith("_PYI_") or name == "_MEIPASS2":
            env.pop(name)
    for name in APP_VARIABLES:
        env.pop(name, None)
    return env
