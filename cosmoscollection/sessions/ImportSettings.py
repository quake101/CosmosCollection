"""The Import Wizard's saved settings: the library folders it scans and the
scan rules changed under Advanced. Shared by the wizard (ImportWizard.py) and
the command line's 'library scan', so both scan the same way.

Only PySide6.QtCore (QSettings), no windows."""

import re


from cosmoscollection.sessions.LibraryScanner import ScanRules
from cosmoscollection.core.Settings import app_settings

SETTINGS_PREFIX = "import_wizard"


def saved_folders():
    folders = app_settings().value(f"{SETTINGS_PREFIX}/folders", [], type=list)
    return [f for f in folders if f]


def save_folders(folders):
    app_settings().setValue(f"{SETTINGS_PREFIX}/folders", list(folders))


def _split_words(text):
    return tuple(w.strip() for w in re.split(r"[,;]", text or "") if w.strip())


def load_rules():
    """The scan rules, with the user's Advanced changes."""
    settings = app_settings()
    rules = ScanRules()
    rules.session_gap_days = settings.value(f"{SETTINGS_PREFIX}/session_gap_days", rules.session_gap_days, type=int)
    finals = _split_words(settings.value(f"{SETTINGS_PREFIX}/final_keywords", "", type=str))
    work = _split_words(settings.value(f"{SETTINGS_PREFIX}/work_folders", "", type=str))
    if finals:
        rules.final_keywords = finals
    if work:
        rules.work_folders = work
    return rules


def save_rules(session_gap_days, final_keywords, work_folders):
    """Save the Advanced scan rules (the word lists as typed, comma separated).
    The lists are stored only when changed, so improved defaults reach everyone else."""
    settings = app_settings()
    defaults = ScanRules()
    settings.setValue(f"{SETTINGS_PREFIX}/session_gap_days", session_gap_days)
    finals, work = _split_words(final_keywords), _split_words(work_folders)
    settings.setValue(f"{SETTINGS_PREFIX}/final_keywords",
                      "" if finals == defaults.final_keywords else ", ".join(finals))
    settings.setValue(f"{SETTINGS_PREFIX}/work_folders", "" if work == defaults.work_folders else ", ".join(work))


def common_names_for_new_sessions():
    """Whether new sessions are named "M 31 (Andromeda Galaxy)" rather than "M 31"
    (the review's checkbox, shared with the Session Manager)."""
    from cosmoscollection.catalog.DSONames import SESSION_COMMON_NAMES_SETTING
    return app_settings().value(SESSION_COMMON_NAMES_SETTING, True, type=bool)


def set_common_names_for_new_sessions(on):
    from cosmoscollection.catalog.DSONames import SESSION_COMMON_NAMES_SETTING
    app_settings().setValue(SESSION_COMMON_NAMES_SETTING, bool(on))
