#!/usr/bin/env python3
"""
Session Completion
Shown when a session is saved as Completed: offers the next step - stack it in
Siril, hand it to PixInsight WBPP, add a finished image to the gallery, or
nothing - and, for a session linked to a target list entry, marking that
target Completed too. The handoff logic lives in ProcessingHandoff.py.
"""

import os
import time
import logging
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QProcess, QProcessEnvironment, QUrl, QTimer, QSettings
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QButtonGroup, QGroupBox, QCheckBox, QLineEdit,
                               QFileDialog, QMessageBox, QPlainTextEdit, QProgressBar, QListWidget,
                               QListWidgetItem, QProgressDialog, QGridLayout, QWidget, QAbstractItemView)

import ProcessingHandoff as handoff
from DatabaseManager import DatabaseManager
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from SessionManager import format_duration, _rollback, _retire_thread

logger = logging.getLogger(__name__)


def add_session_image_to_gallery(parent, session, file_path=None):
    """Open the gallery's Add Image dialog pre-filled from the session (and
    the given file, e.g. a Siril stack), then save it. True if an image was added."""
    from DSOGallery import AddImageDialog, insert_user_image

    dialog = AddImageDialog(parent)
    if file_path:
        dialog.set_file_path(file_path)
    dialog.prefill_from_session(session)
    if dialog.exec() != QDialog.Accepted:
        return False
    try:
        insert_user_image(DatabaseManager(), dialog.get_image_data())
    except Exception as e:
        logger.error(f"Error adding session image to gallery: {e}", exc_info=True)
        QMessageBox.critical(parent, "Error", f"Failed to add the image to the gallery:\n{e}")
        return False
    _refresh_open_image_views()
    QMessageBox.information(parent, "Image Added", "The image was added to your gallery.")
    return True


def _refresh_open_image_views():
    """The main window's image counts and an open gallery don't watch the
    database, so refresh whichever of them are open."""
    for widget in QApplication.topLevelWidgets():
        name = type(widget).__name__
        try:
            if name == "DSOGalleryWindow" and widget.isVisible():
                widget._refresh_gallery()
            elif name == "MainWindow":
                widget._refresh_data()
        except Exception as e:
            logger.warning(f"Could not refresh {name} after adding an image: {e}")


def _format_clock(seconds):
    """'4:07' or '1:02:15'."""
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _open_folder(path):
    from UrlOpener import open_url
    open_url(QUrl.fromLocalFile(path))


class StageWorker(QThread):
    """Links/copies a session's frames into its workspace off the UI thread -
    copying a few thousand subs across drives can take minutes."""

    progress = Signal(int, int)
    staged = Signal(object, int, int, object)  # layout, linked, copied, unfixed paths
    failed = Signal(str)

    def __init__(self, frames, frames_dir, copy_files, fix_frame_types, siril_sequences):
        super().__init__()
        self.frames = frames
        self.frames_dir = frames_dir
        self.copy_files = copy_files
        self.fix_frame_types = fix_frame_types
        self.siril_sequences = siril_sequences
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            layout, linked, copied, unfixed = handoff.stage_workspace(
                self.frames, self.frames_dir, self.copy_files,
                progress_callback=self.progress.emit, is_cancelled=lambda: self.cancelled,
                fix_frame_types=self.fix_frame_types, siril_sequences=self.siril_sequences)
            if not self.cancelled:
                self.staged.emit(layout, linked, copied, unfixed)
        except Exception as e:
            logger.error(f"Error staging workspace {self.frames_dir}: {e}", exc_info=True)
            self.failed.emit(str(e))


class FrameCheckWorker(QThread):
    """Finds attached subs that no longer exist and whether the lights are from a
    colour camera - both read the disk, which for hundreds of subs on a spinning
    drive can take seconds, so the completion dialog shows first and fills these in."""

    checked = Signal(object, bool)  # missing paths, osc

    def __init__(self, frames):
        super().__init__()
        # Work on a snapshot: the dialog owns (and may change) the real FrameSet.
        self.paths = list(frames.all_paths())
        self.lights = {key: list(paths) for key, paths in frames.lights.items()}

    def run(self):
        missing = [p for p in self.paths if not os.path.isfile(p)]
        gone = set(missing)
        present = handoff.FrameSet(lights={k: [p for p in v if p not in gone] for k, v in self.lights.items()})
        try:
            osc = handoff.detect_osc(present)
        except Exception as e:
            logger.debug(f"Colour camera detection failed: {e}")
            osc = False
        self.checked.emit(missing, osc)


class SessionCompletionDialog(WindowPositionMixin, QDialog):
    """What to do now that a session is Completed. The session itself is
    already saved as Completed, so Skip just closes."""

    WINDOW_POSITION_KEY = "SessionCompletionDialog"
    CHOICE_SIRIL, CHOICE_PIXINSIGHT, CHOICE_GALLERY, CHOICE_NOTHING = range(4)
    CALIBRATION_KINDS = (("Dark", "Darks"), ("Flat", "Flats"), ("Bias", "Bias"))
    OSC_LABEL = "Color camera (debayer the subs)"

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = dict(session)
        self.db_manager = DatabaseManager()
        self.frames = handoff.FrameSet()
        self.target = None  # {"id", "name", "status"} when linked
        self._osc_detected = None
        self._workspace_edited = False
        self._stage_worker = None
        self._frame_check = None
        self._osc_touched = False

        self.setWindowTitle("Session Completed")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(560, 520)  # default size the first time this dialog is ever opened

        self._load()
        self._setup_ui()
        self._restore_choices()
        self._on_choice_changed()
        self.setup_window_position()
        # Checking hundreds of subs on disk can take seconds on a spinning drive -
        # do it once the dialog is on screen instead of before.
        QTimer.singleShot(0, self._start_frame_check)

    # ---- Data -------------------------------------------------------------

    def _load(self):
        try:
            with self.db_manager.get_connection() as conn:
                # The per-file disk check runs in FrameCheckWorker after the dialog shows.
                self.frames = handoff.collect_session_frames(conn, self.session["id"], check_exists=False)
                if self.session.get("target_id"):
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, name, status FROM usertargetlist WHERE id = ?",
                                   (self.session["target_id"],))
                    row = cursor.fetchone()
                    if row:
                        self.target = {"id": row[0], "name": row[1], "status": row[2]}
        except Exception as e:
            logger.error(f"Error loading session {self.session.get('id')} for completion: {e}")

    # ---- UI ---------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{self.session.get('dso_name', '')}</b> — {self.session.get('session_date', '')} "
                       "is marked Completed.")
        title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(title)

        self.summary_label = QLabel(self._summary_text())
        self.summary_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Filled in by the background FrameCheckWorker once the dialog is showing.
        self.missing_label = QLabel()
        self.missing_label.setStyleSheet(f"color: {COLORS['warning']};")
        self.missing_label.setWordWrap(True)
        self.missing_label.hide()
        layout.addWidget(self.missing_label)

        choice_box = QGroupBox("What's next?")
        choice_layout = QVBoxLayout(choice_box)
        self.choice_group = QButtonGroup(self)
        choices = [
            (self.CHOICE_SIRIL, "Stack in Siril",
             "Calibrate, register and stack the subs with a generated Siril script."),
            (self.CHOICE_PIXINSIGHT, "Process in PixInsight (WBPP)",
             "Open WeightedBatchPreprocessing with the session's frames loaded."),
            (self.CHOICE_GALLERY, "Add a finished image to the gallery",
             "Already stacked and processed? Bring the final image in."),
            (self.CHOICE_NOTHING, "Just mark as complete", ""),
        ]
        self._app_for_choice = {self.CHOICE_SIRIL: handoff.SIRIL, self.CHOICE_PIXINSIGHT: handoff.PIXINSIGHT}
        for choice_id, label, hint in choices:
            radio = QRadioButton(label)
            radio.setToolTip(hint)
            app = self._app_for_choice.get(choice_id)
            if app:
                ready, message = handoff.integration_status(app)
                if not ready:
                    radio.setEnabled(False)
                    radio.setText(f"{label}  (not set up)")
                    radio.setToolTip(message)
            self.choice_group.addButton(radio, choice_id)
            choice_layout.addWidget(radio)
        self._disable_apps_without_lights()
        self.choice_group.button(self.CHOICE_NOTHING).setChecked(True)
        self.choice_group.idToggled.connect(lambda _id, checked: checked and self._on_choice_changed())
        layout.addWidget(choice_box)

        layout.addWidget(self._build_handoff_options())

        self.target_checkbox = QCheckBox()
        if self.target and self.target.get("status") != "Completed":
            self.target_checkbox.setText(f"Also mark target '{self.target['name']}' as Completed in the Target List")
            self.target_checkbox.setChecked(True)
        else:
            self.target_checkbox.hide()
        layout.addWidget(self.target_checkbox)

        layout.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        skip_btn = QPushButton("Skip")
        skip_btn.setToolTip("Close without doing anything else - the session stays Completed.")
        skip_btn.clicked.connect(self.reject)
        buttons.addWidget(skip_btn)
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setDefault(True)
        self.continue_btn.clicked.connect(self._on_continue)
        buttons.addWidget(self.continue_btn)
        layout.addLayout(buttons)

    def _disable_apps_without_lights(self):
        if self.frames.light_count:
            return
        for choice_id in self._app_for_choice:
            radio = self.choice_group.button(choice_id)
            if radio.isEnabled():
                radio.setEnabled(False)
                radio.setToolTip("This session has no light frames (on disk) to hand off.")
                if radio.isChecked():
                    self.choice_group.button(self.CHOICE_NOTHING).setChecked(True)

    def _start_frame_check(self):
        self._frame_check = FrameCheckWorker(self.frames)
        self._frame_check.checked.connect(self._on_frame_check)
        self._frame_check.start()

    def _on_frame_check(self, missing, osc):
        """Background check results: drop subs that are gone, record the colour-camera detection."""
        if self._stage_worker is not None:
            return  # staging already started - it skips missing files itself
        if missing:
            handoff.remove_frames(self.frames, missing)
            self.missing_label.setText(f"{len(missing)} attached file(s) no longer exist on disk and will be "
                                       "left out of any handoff.")
            self.missing_label.show()
            self.summary_label.setText(self._summary_text())
            self._refresh_calibration_labels()
            self._disable_apps_without_lights()
        self._osc_detected = osc
        self.osc_checkbox.setEnabled(True)
        self.osc_checkbox.setText(self.OSC_LABEL)
        if not self._osc_touched:
            self.osc_checkbox.setChecked(osc)

    def done(self, result):
        # Never let the frame-check thread be garbage-collected mid-run with the dialog.
        if self._frame_check is not None:
            _retire_thread(self._frame_check)
        super().done(result)

    def _summary_text(self):
        lights = self.frames.lights
        if not lights:
            return "No light frames are attached to this session."
        per_filter = ", ".join(f"{key} {len(paths)}" for key, paths in sorted(lights.items()))
        text = f"{self.frames.light_count} light frames ({per_filter})"
        if self.session.get("integration_seconds"):
            text += f" · {format_duration(self.session['integration_seconds'])} integration"
        return text

    def _build_handoff_options(self):
        self.options_box = QGroupBox("Handoff options")
        grid = QGridLayout(self.options_box)

        grid.addWidget(QLabel("Workspace:"), 0, 0)
        self.workspace_edit = QLineEdit()
        self.workspace_edit.setToolTip("Frames are linked (or copied) into this folder and the app works "
                                       "there. Your original subs are never moved or changed.")
        self.workspace_edit.textEdited.connect(lambda _: setattr(self, "_workspace_edited", True))
        grid.addWidget(self.workspace_edit, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_workspace)
        grid.addWidget(browse_btn, 0, 2)

        self.calibration_labels = {}
        for row, (kind, label) in enumerate(self.CALIBRATION_KINDS, start=1):
            grid.addWidget(QLabel(f"{label}:"), row, 0)
            count_label = QLabel()
            self.calibration_labels[kind] = count_label
            grid.addWidget(count_label, row, 1)
            add_btn = QPushButton("Add Folder...")
            add_btn.setToolTip(f"Add {label.lower()} from a folder (e.g. your calibration library).")
            add_btn.clicked.connect(lambda _=False, k=kind: self._add_calibration_folder(k))
            grid.addWidget(add_btn, row, 2)
        self._refresh_calibration_labels()

        # Disabled until FrameCheckWorker has read a sub's BAYERPAT header.
        self.osc_checkbox = QCheckBox(f"{self.OSC_LABEL}  (detecting...)")
        self.osc_checkbox.setEnabled(False)
        self.osc_checkbox.setToolTip("Detected from the subs' BAYERPAT header. Siril needs to know "
                                     "whether to debayer; WBPP detects it itself.")
        self.osc_checkbox.clicked.connect(lambda: setattr(self, "_osc_touched", True))
        grid.addWidget(self.osc_checkbox, 4, 0, 1, 3)

        self.run_now_checkbox = QCheckBox("Start WBPP right away (otherwise it opens for review first)")
        grid.addWidget(self.run_now_checkbox, 5, 0, 1, 3)

        self.copy_checkbox = QCheckBox("Copy the files instead of hard-linking them")
        self.copy_checkbox.setToolTip("Hard links take no extra disk space and are used whenever possible. "
                                      "Files on another drive are always copied.")
        grid.addWidget(self.copy_checkbox, 6, 0, 1, 3)

        grid.setColumnStretch(1, 1)
        return self.options_box

    def _refresh_calibration_labels(self):
        counts = self.frames.counts()
        for kind, label in self.calibration_labels.items():
            added = self.frames.added_counts.get(kind, 0)
            attached = counts[kind] - added
            text = f"{attached} attached" + (f" + {added} added" if added else "")
            if not counts[kind]:
                text += " (step skipped)"
            label.setText(text)

    def _selected_app(self):
        return self._app_for_choice.get(self.choice_group.checkedId())

    def _on_choice_changed(self):
        app = self._selected_app()
        self.options_box.setVisible(app is not None)
        if app is None:
            return
        self.osc_checkbox.setVisible(app == handoff.SIRIL)
        self.run_now_checkbox.setVisible(app == handoff.PIXINSIGHT)
        if not self._workspace_edited:
            self.workspace_edit.setText(handoff.default_workspace(
                app, self.session.get("dso_name"), self.session.get("session_date"), self.frames))

    def _browse_workspace(self):
        start = os.path.dirname(self.workspace_edit.text()) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Choose a Workspace Folder", start)
        if folder:
            self.workspace_edit.setText(os.path.normpath(folder))
            self._workspace_edited = True

    def _add_calibration_folder(self, kind):
        # Calibration frames usually live in a library folder, so start where the
        # last folder of this kind was picked (e.g. .../_Darks/300s).
        settings = QSettings("CosmosCollection", "CosmosCollection")
        setting_key = f"last_calibration_folder_{kind.lower()}"
        last_folder = settings.value(setting_key, "", type=str)
        start = last_folder if last_folder and os.path.isdir(last_folder) else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, f"Add {kind} Frames From Folder", start)
        if not folder:
            return
        settings.setValue(setting_key, os.path.normpath(folder))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            added = handoff.add_calibration_folder(self.frames, kind, folder)
        finally:
            QApplication.restoreOverrideCursor()
        if not added:
            QMessageBox.information(self, "No Frames Added", "No new FITS/XISF files were found in that folder.")
        self._refresh_calibration_labels()

    # ---- Remembered choices ---------------------------------------------------

    # QSettings keys for what the user picked last time (saved on Continue only).
    CHOICE_SETTING = "session_completion/choice"
    RUN_NOW_SETTING = "session_completion/wbpp_run_now"
    COPY_SETTING = "session_completion/copy_files"
    MARK_TARGET_SETTING = "session_completion/mark_target_completed"
    CHOICE_NAMES = {CHOICE_SIRIL: "siril", CHOICE_PIXINSIGHT: "pixinsight",
                    CHOICE_GALLERY: "gallery", CHOICE_NOTHING: "nothing"}

    def _restore_choices(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        name = settings.value(self.CHOICE_SETTING, "nothing", type=str)
        choice = next((c for c, n in self.CHOICE_NAMES.items() if n == name), self.CHOICE_NOTHING)
        button = self.choice_group.button(choice)
        # A remembered app that's since been disabled (or has no lights here)
        # falls back to the default.
        (button if button.isEnabled() else self.choice_group.button(self.CHOICE_NOTHING)).setChecked(True)
        self.run_now_checkbox.setChecked(settings.value(self.RUN_NOW_SETTING, False, type=bool))
        self.copy_checkbox.setChecked(settings.value(self.COPY_SETTING, False, type=bool))
        if not self.target_checkbox.isHidden():
            self.target_checkbox.setChecked(settings.value(self.MARK_TARGET_SETTING, True, type=bool))

    def _save_choices(self):
        settings = QSettings("CosmosCollection", "CosmosCollection")
        settings.setValue(self.CHOICE_SETTING, self.CHOICE_NAMES[self.choice_group.checkedId()])
        settings.setValue(self.RUN_NOW_SETTING, self.run_now_checkbox.isChecked())
        settings.setValue(self.COPY_SETTING, self.copy_checkbox.isChecked())
        if not self.target_checkbox.isHidden():
            settings.setValue(self.MARK_TARGET_SETTING, self.target_checkbox.isChecked())

    # ---- Actions ----------------------------------------------------------

    def _on_continue(self):
        choice = self.choice_group.checkedId()
        app = self._selected_app()
        if app and not self._validate_handoff(app):
            return
        if app == handoff.SIRIL and self._osc_detected is None and not self._osc_touched:
            # Continue pressed before the background check finished - Siril must know now.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._osc_detected = handoff.detect_osc(self.frames)
            finally:
                QApplication.restoreOverrideCursor()
            self.osc_checkbox.setChecked(self._osc_detected)
        self._save_choices()
        if self.target_checkbox.isVisible() and self.target_checkbox.isChecked():
            self._mark_target_completed()

        if choice == self.CHOICE_GALLERY:
            self.hide()
            add_session_image_to_gallery(self.parentWidget(), self.session)
            self.accept()
        elif app:
            self._start_staging(app)
        else:
            self.accept()

    def _mark_target_completed(self):
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    UPDATE usertargetlist SET status = 'Completed', date_observed = COALESCE(date_observed, ?)
                    WHERE id = ?
                """, (self.session.get("session_date") or datetime.now().strftime("%Y-%m-%d"), self.target["id"]))
                conn.commit()
            self.target_checkbox.hide()  # applied - don't apply it twice if staging is retried
        except Exception as e:
            _rollback(self.db_manager)
            logger.error(f"Error marking target {self.target['id']} completed: {e}")
            QMessageBox.warning(self, "Target Not Updated", f"Couldn't mark the target Completed: {e}")

    def _validate_handoff(self, app):
        workdir = os.path.normpath(self.workspace_edit.text().strip()) if self.workspace_edit.text().strip() else ""
        problem = handoff.workspace_problem(workdir, self.frames)
        if not problem and app == handoff.PIXINSIGHT and handoff.unsafe_wbpp_path(workdir):
            problem = "WBPP can't use a folder whose path contains ',' or '='. Choose another workspace."
        if problem:
            QMessageBox.warning(self, "Workspace", problem)
            return False
        if app == handoff.SIRIL and handoff.has_xisf_lights(self.frames):
            version = handoff.siril_version(handoff.resolve_executable(handoff.SIRIL))
            if version and version < handoff.SIRIL_XISF_VERSION:
                QMessageBox.warning(self, "Siril Too Old for XISF",
                                    f"This session's subs are XISF files, which Siril reads from version 1.4 "
                                    f"(installed: {'.'.join(map(str, version))}). Update Siril, or use PixInsight.")
                return False
        return True

    def _start_staging(self, app):
        self.continue_btn.setEnabled(False)
        workdir = os.path.normpath(self.workspace_edit.text().strip())
        # Siril works in the workspace itself (results land at its top level);
        # WBPP scans a frames/ subfolder recursively, so its output goes beside it.
        frames_dir = workdir if app == handoff.SIRIL else os.path.join(workdir, handoff.FRAMES_DIR)

        total = sum(1 for _ in self.frames.all_paths())
        self._progress = QProgressDialog("Linking frames into the workspace...", "Cancel", 0, total, self)
        self._progress.setWindowTitle("Preparing Workspace")
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(300)

        # WBPP sorts frames by IMAGETYP, so frames added as another type get it
        # corrected. For Siril, FITS frames are staged as ready-made sequences so
        # its convert step (which copies every sub without Developer Mode) is skipped.
        self._stage_worker = StageWorker(self.frames, frames_dir, self.copy_checkbox.isChecked(),
                                         fix_frame_types=(app == handoff.PIXINSIGHT),
                                         siril_sequences=(app == handoff.SIRIL))
        self._stage_worker.progress.connect(lambda done, _total: self._progress.setValue(done))
        self._progress.canceled.connect(self._stage_worker.cancel)
        self._stage_worker.staged.connect(
            lambda layout, linked, copied, unfixed:
                self._on_staged(app, workdir, frames_dir, layout, linked, copied, unfixed))
        self._stage_worker.failed.connect(self._on_stage_failed)
        self._stage_worker.finished.connect(self._on_stage_thread_finished)
        self._stage_worker.start()

    def _close_progress(self):
        # Closing a QProgressDialog emits canceled - disconnect first so a
        # finished staging run isn't flagged as cancelled.
        try:
            self._progress.canceled.disconnect()
        except (RuntimeError, TypeError):
            pass  # already disconnected
        self._progress.close()

    def _on_stage_thread_finished(self):
        if self._stage_worker and self._stage_worker.cancelled:
            self._close_progress()
            self.continue_btn.setEnabled(True)
            QMessageBox.information(self, "Cancelled",
                                    "Staging was cancelled. Choose a new, empty workspace folder to try again.")
            self._workspace_edited = False
            self._on_choice_changed()

    def _on_stage_failed(self, message):
        self._close_progress()
        self.continue_btn.setEnabled(True)
        QMessageBox.critical(self, "Couldn't Prepare the Workspace", message)

    def _on_staged(self, app, workdir, frames_dir, layout, linked, copied, unfixed):
        self._close_progress()
        # staged is emitted at the very end of run(); let the thread exit before
        # this dialog closes and may be garbage-collected along with it.
        self._stage_worker.wait()
        logger.info(f"Staged session {self.session['id']} into {frames_dir}: {linked} linked, {copied} copied, "
                    f"{len(unfixed)} with a mismatched frame type left as-is")
        self._unfixed_frames = unfixed
        base_name = (f"CosmosCollection_{handoff.sanitize_name(self.session.get('dso_name'))}_"
                     f"{handoff.sanitize_name(self.session.get('session_date'), 'undated')}")
        if app == handoff.SIRIL:
            script_path = os.path.join(workdir, f"{base_name}_Siril.ssf")
            script = handoff.build_siril_script(layout, osc=self.osc_checkbox.isChecked())
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)
            except OSError as e:
                self.continue_btn.setEnabled(True)
                QMessageBox.critical(self, "Couldn't Write the Siril Script", str(e))
                return
            steps = handoff.siril_steps(script, handoff.layout_counts(layout, self.frames))
            run_dialog = SirilRunDialog(self.session, workdir, script_path, steps, parent=self.parentWidget())
            run_dialog.show()
            run_dialog.start()
        else:
            self._launch_wbpp(workdir, frames_dir)
        self.accept()

    def _launch_wbpp(self, workdir, frames_dir):
        exe = handoff.resolve_executable(handoff.PIXINSIGHT)
        output_dir = os.path.join(workdir, handoff.WBPP_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        args = handoff.pixinsight_wbpp_args(exe, frames_dir, output_dir, self.run_now_checkbox.isChecked())
        if handoff.launch_detached(args, cwd=workdir):
            fixed = len(self.frames.retype) - len(self._unfixed_frames)
            notes = ""
            if fixed:
                notes += (f"\n\n{fixed} added calibration frame(s) had a FITS IMAGETYP header that didn't match "
                          "the type you added them as (e.g. darks saved as LIGHT). Their workspace copies were "
                          "corrected; your originals are unchanged.")
            if self._unfixed_frames:
                notes += (f"\n\n{len(self._unfixed_frames)} added XISF frame(s) have an IMAGETYP header that "
                          "doesn't match the type you added them as and couldn't be corrected. In WBPP, move "
                          "them to the right tab yourself.")
            QMessageBox.information(
                self.parentWidget(), "Opening PixInsight",
                f"PixInsight is starting WBPP with {sum(1 for _ in self.frames.all_paths())} frames from:\n"
                f"{frames_dir}\n\nWBPP output folder:\n{output_dir}{notes}\n\n"
                "When you've finished processing, add the final image from the Image Gallery.")
        else:
            QGuiApplication.clipboard().setText(frames_dir)
            QMessageBox.warning(
                self.parentWidget(), "Couldn't Start PixInsight",
                "PixInsight couldn't be started. The frames are ready in:\n"
                f"{frames_dir}\n\n(path copied to the clipboard) - open WBPP yourself and use "
                "'Add Directory' with that folder.")


class SirilRunDialog(WindowPositionMixin, QDialog):
    """Runs the generated script with siril-cli and follows it: Siril logs
    'Running command: <name>' for each script command and 'progress: ..., N%'
    while one works, which (matched against the script's own steps) drive a
    stage list, a current-step bar and an overall bar with time remaining.
    Siril's raw output can be shown or hidden. Non-modal, so the rest of the
    app stays usable during a long stack."""

    WINDOW_POSITION_KEY = "SirilRunDialog"
    SHOW_OUTPUT_SETTING = "siril_show_output"
    COMMAND_LABELS = {
        "convert": "Converting frames", "calibrate": "Calibrating", "register": "Registering (aligning stars)",
        "stack": "Stacking", "load": "Saving result", "mirrorx": "Saving result", "save": "Saving result",
    }
    DONE_MARK, ACTIVE_MARK, PENDING_MARK = "✓", "▶", "•"
    STACK_NORMALIZATION_SHARE = 0.3
    ACTIVITY_MAX_CHARS = 110
    QUIET_NOTICE_SECONDS = 15

    def __init__(self, session, workdir, script_path, steps, parent=None):
        super().__init__(parent)
        self.session = dict(session)
        self.workdir = workdir
        self.script_path = script_path
        self.siril_cli = handoff.resolve_executable(handoff.SIRIL)
        self._cancelled = False
        self._partial_line = ""
        self._run_log = None
        self.run_log_path = None

        self.steps = steps
        self.sections = list(dict.fromkeys(s["section"] for s in steps if s["section"]))
        self._total_weight = sum(s["weight"] for s in steps) or 1.0
        self._step_index = -1
        self._phase = ""
        self._done_weight = 0.0
        self._overall = 0.0
        self._started_at = None
        self._last_output_at = None

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"Siril — {self.session.get('dso_name', '')}")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.resize(720, 560)  # default size the first time this dialog is ever opened

        self._setup_ui()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_time_label)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(workdir)
        env = QProcessEnvironment()
        for key, value in handoff.clean_subprocess_env().items():
            env.insert(key, value)
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.setup_window_position()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Starting Siril...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(self.status_label)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.detail_label)
        self.step_bar = QProgressBar()
        self.step_bar.setRange(0, 1000)
        self.step_bar.setFormat("%p%")
        layout.addWidget(self.step_bar)

        # Siril's latest message (or how long it's been quiet), so a long step
        # never looks frozen even before it reports a percentage.
        self.activity_label = QLabel("")
        self.activity_label.setStyleSheet(f"color: {COLORS['text_disabled']}; font-size: 9pt;")
        self.activity_label.setMinimumWidth(1)  # long messages mustn't widen the window
        layout.addWidget(self.activity_label)

        self.python_label = QLabel(
            "Siril is setting up its Python environment. This happens occasionally (e.g. the first run "
            "after installing or updating Siril) and can take a few minutes - stacking continues when it's done.")
        self.python_label.setWordWrap(True)
        self.python_label.setStyleSheet(f"color: {COLORS['info']};")
        self.python_label.hide()
        layout.addWidget(self.python_label)

        overall_row = QHBoxLayout()
        overall_row.addWidget(QLabel("Overall:"))
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setFormat("%p%")
        overall_row.addWidget(self.overall_bar, 1)
        layout.addLayout(overall_row)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.time_label)

        # Shown when Siril says it has to copy frames because it can't symlink them.
        self.symlink_hint = QWidget()
        self.symlink_hint.setStyleSheet(
            f"QWidget#symlinkHint {{ border: 1px solid {COLORS['warning']}; border-radius: 4px; }}")
        self.symlink_hint.setObjectName("symlinkHint")
        hint_layout = QVBoxLayout(self.symlink_hint)
        self.symlink_hint_label = QLabel()
        self.symlink_hint_label.setWordWrap(True)
        hint_layout.addWidget(self.symlink_hint_label)
        hint_buttons = QHBoxLayout()
        open_settings_btn = QPushButton("Open Developer Settings")
        open_settings_btn.setToolTip("Windows Settings → System → For developers → Developer Mode")
        open_settings_btn.clicked.connect(self._open_developer_settings)
        hint_buttons.addWidget(open_settings_btn)
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(self.symlink_hint.hide)
        hint_buttons.addWidget(dismiss_btn)
        hint_buttons.addStretch()
        hint_layout.addLayout(hint_buttons)
        self.symlink_hint.hide()
        self._symlink_hint_shown = False
        layout.addWidget(self.symlink_hint)

        self.stages_list = QListWidget()
        self.stages_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.stages_list.setFocusPolicy(Qt.NoFocus)
        for section in self.sections:
            self.stages_list.addItem(f"{self.PENDING_MARK}  {section}")
        layout.addWidget(self.stages_list, 1)

        self.output_checkbox = QCheckBox("Show Siril output")
        self.output_checkbox.setChecked(QSettings("CosmosCollection", "CosmosCollection")
                                        .value(self.SHOW_OUTPUT_SETTING, False, type=bool))
        self.output_checkbox.toggled.connect(self._on_output_toggled)
        layout.addWidget(self.output_checkbox)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setVisible(self.output_checkbox.isChecked())
        layout.addWidget(self.log_view, 2)

        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addWidget(QLabel("Stacked results:"))
        self.results_list = QListWidget()
        self.results_list.setMaximumHeight(90)
        self.results_list.itemDoubleClicked.connect(lambda _: self._add_to_gallery())
        results_layout.addWidget(self.results_list)
        result_buttons = QHBoxLayout()
        self.open_siril_btn = QPushButton("Open in Siril")
        self.open_siril_btn.clicked.connect(self._open_in_siril)
        result_buttons.addWidget(self.open_siril_btn)
        gallery_btn = QPushButton("Add to Gallery...")
        gallery_btn.clicked.connect(self._add_to_gallery)
        result_buttons.addWidget(gallery_btn)
        result_buttons.addStretch()
        results_layout.addLayout(result_buttons)
        self.results_widget.hide()
        layout.addWidget(self.results_widget)

        buttons = QHBoxLayout()
        folder_btn = QPushButton("Open Workspace Folder")
        folder_btn.clicked.connect(lambda: _open_folder(self.workdir))
        buttons.addWidget(folder_btn)
        buttons.addStretch()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        buttons.addWidget(self.stop_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def start(self):
        self._append(f"Workspace: {self.workdir}\nScript: {self.script_path}\n")
        self._started_at = time.monotonic()
        self._open_run_log()
        self._set_step_busy(True)
        self._elapsed_timer.start(1000)
        self.process.start(self.siril_cli, ["-d", self.workdir, "-s", self.script_path])

    def _open_run_log(self):
        """Everything Siril prints - including the progress and chatter lines the
        window hides - with the time since launch, saved next to the script as
        <script>.log for troubleshooting (e.g. where a long silence happened)."""
        self.run_log_path = os.path.splitext(self.script_path)[0] + ".log"
        try:
            self._run_log = open(self.run_log_path, "w", encoding="utf-8")
            self._run_log.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  {self.siril_cli} -d {self.workdir} "
                                f"-s {self.script_path}\n")
            self._run_log.flush()
        except OSError as e:
            logger.warning(f"Could not create Siril run log {self.run_log_path}: {e}")
            self._run_log = None

    def _write_run_log(self, line):
        if self._run_log:
            try:
                self._run_log.write(f"[{time.monotonic() - self._started_at:8.1f}s] {line}\n")
                self._run_log.flush()
            except (OSError, ValueError):
                self._run_log = None

    def _close_run_log(self):
        if self._run_log:
            self._run_log.close()
            self._run_log = None

    def _on_output_toggled(self, checked):
        self.log_view.setVisible(checked)
        QSettings("CosmosCollection", "CosmosCollection").setValue(self.SHOW_OUTPUT_SETTING, checked)

    def _append(self, text):
        self.log_view.appendPlainText(text.rstrip("\n"))

    def _set_step_busy(self, busy):
        """A moving bar until the current step reports a percentage."""
        if busy:
            self.step_bar.setRange(0, 0)
        elif self.step_bar.maximum() == 0:
            self.step_bar.setRange(0, 1000)
            self.step_bar.setValue(0)

    def _show_activity(self, message):
        if len(message) > self.ACTIVITY_MAX_CHARS:
            message = message[:self.ACTIVITY_MAX_CHARS - 1] + "…"
        self.activity_label.setText(message)

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        if data:
            self._last_output_at = time.monotonic()
        text = self._partial_line + data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._partial_line = lines.pop()  # a line still being written arrives in the next chunk
        for raw in lines:
            self._handle_line(handoff.clean_siril_line(raw))

    def _handle_line(self, line):
        if not line:
            return
        self._write_run_log(line)
        progress = handoff.parse_siril_progress(line)
        if progress is not None:
            self._on_progress(*progress)
            return
        if "Running command:" in line:
            self._on_command(line.split("Running command:", 1)[1].strip())
        if handoff.SIRIL_SYMLINK_WARNING in line and not self._symlink_hint_shown:
            self._show_symlink_hint()
        python_state = handoff.siril_python_setup_state(line)
        if python_state is not None:
            self.python_label.setVisible(python_state)
        if handoff.is_siril_noise(line):
            return
        message = line[5:] if line.startswith("log: ") else line
        self._append(message)
        self._show_activity(message)

    def _show_symlink_hint(self):
        """Without Windows Developer Mode, Siril can't symlink FITS subs into its
        process folder during 'convert', so it copies every one of them."""
        self._symlink_hint_shown = True
        size = handoff.staged_frames_size(self.workdir)
        size_text = f"about {size / 1024 ** 3:.1f} GB of" if size >= 1024 ** 3 else "extra"
        self.symlink_hint_label.setText(
            f"<b>Siril is copying every frame</b> into the workspace's process folder, which uses {size_text} "
            "disk space and takes longer. Windows only lets Siril link the files instead of copying them when "
            "<b>Developer Mode</b> is on (Settings → System → For developers). Turning it on won't change "
            "this run, but future stacks will be faster and use far less space.")
        self.symlink_hint.show()

    def _open_developer_settings(self):
        from UrlOpener import open_url
        if not open_url("ms-settings:developers"):
            QMessageBox.information(self, "Developer Mode",
                                    "Open Windows Settings → System → For developers and turn on Developer Mode.")

    # ---- Progress tracking --------------------------------------------------

    def _on_command(self, name):
        """Advance to the next script step running `name` (Siril runs them in order)."""
        for index in range(self._step_index + 1, len(self.steps)):
            if self.steps[index]["command"] == name:
                break
        else:
            return
        self._done_weight = sum(s["weight"] for s in self.steps[:index])
        self._step_index = index
        self._phase = ""
        step = self.steps[index]
        if step["section"]:
            self.status_label.setText(step["section"])
            self._mark_sections(step["section"])
        self.detail_label.setText(self.COMMAND_LABELS.get(name, ""))  # blank for cd/requires/close
        self._set_step_busy(True)
        self._set_overall(0.0)

    def _on_progress(self, text, percent):
        command = self.steps[self._step_index]["command"] if 0 <= self._step_index < len(self.steps) else ""
        if text:
            self._phase = text  # percent-only lines that follow belong to this phase
            label = self.COMMAND_LABELS.get(command, "")
            self.detail_label.setText(text if not label else f"{label} — {text}")
        if percent is not None:
            fraction = max(0.0, min(percent / 100.0, 1.0))
            self._set_step_busy(False)
            self.step_bar.setValue(int(fraction * 1000))
            self._set_overall(self._step_fraction(command, fraction))

    def _step_fraction(self, command, phase_fraction):
        """stack reports 0-100% twice - normalization, then the (slower)
        rejection stacking - so give each phase its own part of the step."""
        if command != "stack":
            return phase_fraction
        phase = self._phase.lower()
        share = self.STACK_NORMALIZATION_SHARE
        if "normaliz" in phase:
            return share * phase_fraction
        if "opening" in phase:  # 'Opening images for stacking, 100%' comes before stacking starts
            return share
        if "stack" in phase:
            return share + (1 - share) * phase_fraction
        return 0.0

    def _set_overall(self, step_fraction):
        weight = self.steps[self._step_index]["weight"] if 0 <= self._step_index < len(self.steps) else 0.0
        # Parallel workers report out of order and some commands restart
        # their percentage per phase - never let the overall bar go backwards.
        self._overall = max(self._overall, (self._done_weight + weight * step_fraction) / self._total_weight)
        self.overall_bar.setValue(int(self._overall * 1000))

    def _mark_sections(self, active_section):
        active = self.sections.index(active_section) if active_section in self.sections else -1
        for row, section in enumerate(self.sections):
            mark = self.DONE_MARK if row < active else self.ACTIVE_MARK if row == active else self.PENDING_MARK
            self.stages_list.item(row).setText(f"{mark}  {section}")

    def _update_time_label(self):
        if self._started_at is None:
            return
        elapsed = time.monotonic() - self._started_at
        text = f"Elapsed {_format_clock(elapsed)}"
        if self._running() and self._overall >= 0.03 and elapsed >= 20:
            remaining = elapsed / self._overall - elapsed
            text += f"  ·  about {_format_clock(remaining)} left"
        self.time_label.setText(text)

        quiet = time.monotonic() - (self._last_output_at or self._started_at)
        if quiet < self.QUIET_NOTICE_SECONDS and self.activity_label.text().startswith("No new output"):
            self.activity_label.setText("")  # Siril is talking again (maybe only progress lines)
        if self._running() and quiet >= self.QUIET_NOTICE_SECONDS:
            if self._step_index < 0:
                self.detail_label.setText("Siril is starting up - the first run after installing or updating "
                                          "Siril can take a few minutes.")
            self._show_activity(f"No new output from Siril for {_format_clock(quiet)} - it's still working.")

    def _running(self):
        return self.process.state() != QProcess.NotRunning

    def _stop(self):
        if self._running():
            self._cancelled = True
            self.process.kill()

    def _on_error(self, error):
        if error == QProcess.FailedToStart:
            self._finish_ui()
            self.status_label.setText("Siril couldn't be started.")
            self._append(f"Failed to start {self.siril_cli}: {self.process.errorString()}")
            self._write_run_log(f"# Failed to start: {self.process.errorString()}")
            self._close_run_log()

    def _on_finished(self, exit_code, _exit_status):
        self._read_output()
        if self._partial_line:
            self._handle_line(handoff.clean_siril_line(self._partial_line))
            self._partial_line = ""
        self._write_run_log(f"# Siril exited with code {exit_code}" + (" (stopped by user)" if self._cancelled else ""))
        self._close_run_log()
        self._finish_ui()
        results = handoff.find_siril_results(self.workdir)
        if self._cancelled:
            self.status_label.setText("Stopped.")
            self.detail_label.setText("")
        elif exit_code == 0 and results:
            self.status_label.setText(f"Done - {len(results)} stacked image(s).")
            self.status_label.setStyleSheet(f"font-weight: bold; font-size: 11pt; color: {COLORS['success']};")
            self.detail_label.setText("")
            self.step_bar.setValue(1000)
            self.overall_bar.setValue(1000)
            for row in range(self.stages_list.count()):
                self.stages_list.item(row).setText(f"{self.DONE_MARK}  {self.sections[row]}")
        else:
            self.status_label.setText("Siril reported a problem - see its output below.")
            self.status_label.setStyleSheet(f"font-weight: bold; font-size: 11pt; color: {COLORS['error']};")
            # Show the error without changing the saved show/hide preference.
            self.output_checkbox.blockSignals(True)
            self.output_checkbox.setChecked(True)
            self.output_checkbox.blockSignals(False)
            self.log_view.show()
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        if results:
            for path in results:
                item = QListWidgetItem(os.path.basename(path))
                item.setData(Qt.UserRole, path)
                item.setToolTip(path)
                self.results_list.addItem(item)
            self.results_list.setCurrentRow(0)
            self.open_siril_btn.setEnabled(bool(handoff.find_siril_gui(self.siril_cli)))
            self.results_widget.show()

    def _finish_ui(self):
        self._elapsed_timer.stop()
        self._update_time_label()
        self._set_step_busy(False)
        self.activity_label.setText("")
        self.python_label.hide()
        self.stop_btn.setEnabled(False)

    def _selected_result(self):
        item = self.results_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _open_in_siril(self):
        path = self._selected_result()
        gui = handoff.find_siril_gui(self.siril_cli)
        if path and gui and not handoff.launch_detached([gui, path], cwd=self.workdir):
            QMessageBox.warning(self, "Couldn't Start Siril", "Siril's GUI couldn't be started.")

    def _add_to_gallery(self):
        path = self._selected_result()
        if path:
            add_session_image_to_gallery(self, self.session, path)

    def _confirm_stop(self):
        """False if Siril is running and the user wants to keep it running."""
        if not self._running():
            return True
        reply = QMessageBox.question(self, "Stop Siril?", "Siril is still running. Stop it and close?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return False
        self._cancelled = True
        self.process.kill()
        self.process.waitForFinished(3000)
        return True

    def reject(self):
        # Esc comes here directly, bypassing closeEvent.
        if self._confirm_stop():
            super().reject()

    def closeEvent(self, event):
        if not self._confirm_stop():
            event.ignore()
            return
        super().closeEvent(event)
