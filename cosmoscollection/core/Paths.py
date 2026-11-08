"""File path helpers shared by the app and the command line (no Qt)."""

import os


def path_key(path):
    """How file paths are matched: '/' or '\\' and (on Windows) any case compare
    equal. '' for an empty path. Stored paths mix separators, so compare these."""
    return os.path.normcase(os.path.normpath(path)) if path else ""
