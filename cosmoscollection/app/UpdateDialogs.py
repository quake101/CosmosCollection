"""Update check, the update-available dialog, and downloading/applying an update."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from cosmoscollection.ui.Theme import font_size, themed_style


def _get_quit_callback(widget):
    """Walk up the widget's parent chain looking for MainWindow's
    _quit_application (which stops timers/tray/db cleanly before quitting).
    Falls back to a bare QApplication.quit if it can't be found - e.g. an
    update offered before MainWindow exists."""
    while widget is not None:
        if hasattr(widget, '_quit_application'):
            return widget._quit_application
        widget = widget.parent() if hasattr(widget, 'parent') else None
    return QApplication.instance().quit


def run_update_check(owner, on_result, refresh=False):
    """Check GitHub for a newer release on a background thread, then call
    on_result(version_info) on the UI thread - unless owner (a QObject) has
    been closed and deleted by then. refresh=True skips the cached answer."""
    from cosmoscollection.core.AppUpdater import UpdateChecker
    checker = UpdateChecker(owner)
    checker.checked.connect(on_result)
    checker.checked.connect(checker.deleteLater)
    checker.start(refresh)


def offer_update_and_apply(parent, version_info):
    """Shared 'update is available' flow used by the About dialog's Check
    for Updates button, the silent startup check, and the tray check.

    On a packaged build for a platform we publish a release for, offers to
    download, verify, and install the update in place, then restart. In
    every other case (source/dev runs, unsupported OS, or a release missing
    the matching asset), falls back to the original "visit the download
    page" behavior.
    """
    from cosmoscollection.core.AppUpdater import update_manager
    from cosmoscollection.core.OnlineLimits import is_offline, notify_offline

    if is_offline():
        notify_offline(parent, "Updating")
        return
    if not update_manager.is_supported():
        UpdateAvailableDialog(parent, version_info, "page").exec()
        return

    release_data = version_info.get('release_data') or {}  # From the (background) check
    zip_asset, sha256_asset, asset_error = update_manager.get_pending_asset(release_data)

    dialog = UpdateAvailableDialog(parent, version_info, None if asset_error else "install", asset_error)
    if dialog.exec() == QDialog.Accepted:
        _download_and_install_update(parent, zip_asset, sha256_asset)


class UpdateAvailableDialog(QDialog):
    """'A new version is available': install it (or open the download page),
    read the release notes, or choose when to be reminded - or skip this
    version. Accepted means Download & Install was chosen."""

    def __init__(self, parent, version_info, action, note=None):
        """action: "install", "page" (open the download page) or None."""
        super().__init__(parent)
        from cosmoscollection.core.AppUpdater import REMIND_CHOICES, remind_choice_index

        self.setWindowTitle("Update Available")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.version = version_info.get('github_version') or ""
        self.github_url = version_info.get('github_url')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)
        title = QLabel("A new version is available!")
        themed_style(title, lambda: f"font-size: {font_size(13)}; font-weight: bold;")
        layout.addWidget(title)
        info_text = (
            f"Current version: {version_info.get('local_version')}\n"
            f"Latest version: {self.version}"
        )
        if note:
            info_text += f"\n\n{note}"
        info = QLabel(info_text)
        info.setWordWrap(True)
        layout.addWidget(info)

        remind_row = QHBoxLayout()
        remind_row.addWidget(QLabel("Remind me:"))
        self.remind_combo = QComboBox()
        for label, _days in REMIND_CHOICES:
            self.remind_combo.addItem(label)
        self.remind_combo.setCurrentIndex(remind_choice_index())
        self.remind_combo.setToolTip("When to hear about this update again if you don't install it now")
        remind_row.addWidget(self.remind_combo)
        remind_row.addStretch()
        layout.addLayout(remind_row)

        buttons = QHBoxLayout()
        notes_button = QPushButton("Release Notes")
        notes_button.setAutoDefault(False)
        notes_button.clicked.connect(self._open_release_notes)
        notes_button.setEnabled(bool(self.github_url))
        buttons.addWidget(notes_button)
        buttons.addStretch()
        self.later_button = QPushButton()
        self.later_button.setAutoDefault(False)
        self.later_button.clicked.connect(self._remind_later)
        buttons.addWidget(self.later_button)
        if action == "install":
            main_button = QPushButton("Download && Install")
            main_button.clicked.connect(self.accept)
        elif action == "page":
            main_button = QPushButton("Download Page")
            main_button.setEnabled(bool(self.github_url))
            main_button.clicked.connect(self._open_download_page)
        else:
            main_button = None
        if main_button is not None:
            main_button.setDefault(True)
            buttons.addWidget(main_button)
        layout.addLayout(buttons)

        self.remind_combo.currentIndexChanged.connect(self._update_later_button)
        self._update_later_button()

    def _update_later_button(self):
        from cosmoscollection.core.AppUpdater import REMIND_CHOICES
        skip = REMIND_CHOICES[self.remind_combo.currentIndex()][1] is None
        self.later_button.setText("Skip This Version" if skip else "Remind Me Later")

    def _remind_later(self):
        from cosmoscollection.core.AppUpdater import remind_later
        remind_later(self.version, self.remind_combo.currentIndex())
        self.reject()

    def _open_release_notes(self):
        from cosmoscollection.core.UrlOpener import open_url
        open_url(self.github_url)  # The dialog stays open to decide afterwards

    def _open_download_page(self):
        from cosmoscollection.core.UrlOpener import open_url
        open_url(self.github_url)
        self.reject()


def _download_and_install_update(parent, zip_asset, sha256_asset):
    """Downloads (with a progress dialog), verifies, and applies the update,
    then restarts the app. Any failure is reported and simply leaves the
    running app untouched - nothing here modifies the install directory
    directly (that only happens in the separate updater helper process)."""
    from cosmoscollection.core.AppUpdater import update_manager, UpdateError

    progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, parent)
    progress.setWindowTitle("Updating Cosmos Collection")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)

    quit_callback = _get_quit_callback(parent)

    def on_progress(done, total):
        mb_done = done / (1024 * 1024)
        if total > 0:
            progress.setMaximum(total)
            progress.setValue(done)
            progress.setLabelText(f"Downloading update... {mb_done:.1f} / {total / (1024 * 1024):.1f} MB")
        else:
            progress.setMaximum(0)  # indeterminate
            progress.setLabelText(f"Downloading update... {mb_done:.1f} MB")

    def on_error(message):
        progress.close()
        QMessageBox.warning(parent, "Update Failed", f"Could not download the update:\n\n{message}")

    def on_ready(zip_path):
        progress.setLabelText("Installing update...")
        progress.setMaximum(0)
        try:
            update_manager.apply_update(zip_path)
        except UpdateError as e:
            progress.close()
            QMessageBox.warning(parent, "Update Failed", f"Could not install the update:\n\n{e}")
            return
        progress.close()
        # No "Update Ready" confirmation here - apply_update() has already
        # launched the standalone updater helper, which is now waiting for
        # this process to exit before it can proceed. A blocking dialog at
        # this point would just sit there until the user notices and
        # dismisses it, making the update look hung. The helper's own
        # progress dialog (CosmosUpdaterWorker.py) picks up from here to
        # narrate the rest of the restart.
        quit_callback()

    worker = update_manager.start_download(zip_asset, sha256_asset, on_progress, on_error, on_ready)
    progress.canceled.connect(worker.cancel)
    progress.show()
