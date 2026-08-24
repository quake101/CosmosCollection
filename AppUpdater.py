#!/usr/bin/env python3
"""
Download-and-apply update support for Cosmos Collection.

Downloads the platform-matching release zip from GitHub, verifies it against
the release's published SHA256 checksum, extracts it to a staging
directory, then hands off to the standalone CosmosCollectionUpdater helper
(see updater/cosmos_updater.py) to swap the install directory and relaunch
the app. A running onedir build can't overwrite its own executable/DLLs, so
the actual file swap always happens in that separate process, after this
one has quit.

version.py's VersionManager still owns "is an update available" detection;
this module only takes over once the user decides to install one.
"""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from PySide6.QtCore import QThread, Signal

from ResourceManager import ResourceManager

logger = logging.getLogger(__name__)

_PLATFORM_ASSET_LABELS = {
    'Windows': 'Windows',
    'Linux': 'Linux',
    'Darwin': 'macOS',
}


class UpdateError(Exception):
    """Raised when an update can't be downloaded or applied."""


def get_platform_asset_name() -> Optional[str]:
    """Release asset filename expected for the current OS, or None if
    CosmosCollection doesn't publish a build for this platform."""
    label = _PLATFORM_ASSET_LABELS.get(platform.system())
    return f"CosmosCollection-{label}.zip" if label else None


def find_release_assets(release_data: dict):
    """Given a GitHub release API payload (as returned by
    version_manager.get_github_latest_release()), return
    (zip_asset, sha256_asset) dicts for the current platform - either may be
    None if not found."""
    asset_name = get_platform_asset_name()
    if not asset_name:
        return None, None

    zip_asset = None
    sha256_asset = None
    for asset in release_data.get('assets', []):
        name = asset.get('name', '')
        if name == asset_name:
            zip_asset = asset
        elif name == f"{asset_name}.sha256":
            sha256_asset = asset
    return zip_asset, sha256_asset


class DownloadWorker(QThread):
    """Downloads and checksum-verifies the update zip in the background."""

    progress = Signal(int, int)  # bytes_downloaded, bytes_total (total may be 0 if unknown)
    finished_ok = Signal(str)    # path to the verified zip on disk
    error_occurred = Signal(str)

    def __init__(self, zip_asset: dict, sha256_asset: Optional[dict], dest_dir: Path):
        super().__init__()
        self.zip_asset = zip_asset
        self.sha256_asset = sha256_asset
        self.dest_dir = dest_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
            zip_path = self.dest_dir / self.zip_asset['name']

            # Disable SSL verification for PyInstaller builds, matching the
            # pattern used elsewhere in this codebase (version.py,
            # WeatherForecast.py) to work around bundled-cert issues.
            verify_ssl = not getattr(sys, 'frozen', False)

            expected_sha256 = self._fetch_expected_sha256(verify_ssl)
            self._download(self.zip_asset['browser_download_url'], zip_path, verify_ssl)

            if self._cancelled:
                return

            if expected_sha256:
                self._verify_checksum(zip_path, expected_sha256)
            else:
                logger.warning("No checksum published for this release asset; skipping verification")

            self.finished_ok.emit(str(zip_path))

        except UpdateError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            logger.error(f"Update download failed: {e}")
            self.error_occurred.emit(str(e))

    def _fetch_expected_sha256(self, verify_ssl: bool) -> Optional[str]:
        if not self.sha256_asset:
            return None
        response = requests.get(
            self.sha256_asset['browser_download_url'], timeout=15, verify=verify_ssl
        )
        response.raise_for_status()
        # Checksum files are conventionally "<hash>  <filename>" or just "<hash>"
        text = response.text.strip()
        return text.split()[0].lower() if text else None

    def _download(self, url: str, dest: Path, verify_ssl: bool):
        response = requests.get(url, stream=True, timeout=30, verify=verify_ssl)
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0)) or self.zip_asset.get('size', 0)
        downloaded = 0

        tmp_dest = dest.with_name(dest.name + '.part')
        with open(tmp_dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=262144):
                if self._cancelled:
                    break
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.progress.emit(downloaded, total)

        if self._cancelled:
            tmp_dest.unlink(missing_ok=True)
            return

        tmp_dest.replace(dest)

    def _verify_checksum(self, path: Path, expected_sha256: str):
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(262144), b''):
                sha256.update(chunk)
        actual = sha256.hexdigest().lower()
        if actual != expected_sha256:
            path.unlink(missing_ok=True)
            raise UpdateError(
                "The downloaded update failed checksum verification "
                "(the file may be corrupt or the download may have been tampered "
                "with). Please try again."
            )


class UpdateManager:
    """Orchestrates downloading, verifying, and applying an update."""

    def __init__(self):
        self._worker: Optional[DownloadWorker] = None

    @property
    def updates_dir(self) -> Path:
        return ResourceManager.get_data_dir() / 'updates'

    def is_supported(self) -> bool:
        """Self-update only applies to a packaged build running on a
        platform we publish a release asset for - source/dev runs and any
        other OS fall back to the manual download-page flow."""
        return ResourceManager.is_bundled and get_platform_asset_name() is not None

    def get_pending_asset(self, release_data: dict):
        """Return (zip_asset, sha256_asset, error_message) for the current
        platform from a GitHub release payload. error_message is set (and
        the assets are None) when this release has no build for this OS."""
        zip_asset, sha256_asset = find_release_assets(release_data)
        if not zip_asset:
            return None, None, (
                f"No {get_platform_asset_name() or 'compatible'} build was found "
                f"in this release."
            )
        return zip_asset, sha256_asset, None

    def start_download(self, zip_asset: dict, sha256_asset: Optional[dict],
                        on_progress, on_error, on_ready) -> DownloadWorker:
        """Kick off a background download. Caller owns the worker's
        lifetime/signal wiring beyond what's passed here."""
        worker = DownloadWorker(zip_asset, sha256_asset, self.updates_dir)
        worker.progress.connect(on_progress)
        worker.error_occurred.connect(on_error)
        worker.finished_ok.connect(on_ready)
        self._worker = worker
        worker.start()
        return worker

    def apply_update(self, zip_path: str) -> None:
        """Extract the verified zip and launch the standalone updater
        helper; the caller should quit the application immediately after
        this returns. Raises UpdateError if staging fails - nothing in the
        install directory is touched until the helper process takes over,
        so a failure here always leaves the running app untouched."""
        zip_path = Path(zip_path)
        staged_dir = self.updates_dir / f"staged-{os.getpid()}"

        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)
        staged_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(staged_dir)
        except Exception as e:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise UpdateError(f"Could not extract the update archive: {e}")

        updater_name = 'CosmosCollectionUpdater.exe' if platform.system() == 'Windows' \
            else 'CosmosCollectionUpdater'
        updater_path = staged_dir / updater_name
        if not updater_path.exists():
            # Fall back to the currently-installed updater if the new
            # release's zip didn't include one for some reason.
            fallback = ResourceManager.base_path / updater_name
            if fallback.exists():
                updater_path = fallback
            else:
                shutil.rmtree(staged_dir, ignore_errors=True)
                raise UpdateError("Updater helper was not found in the downloaded release.")

        if platform.system() != 'Windows':
            try:
                os.chmod(updater_path, 0o755)
            except OSError:
                pass

        install_dir = ResourceManager.base_path
        relaunch_path = sys.executable
        log_path = self.updates_dir / 'update_log.txt'

        args = [
            str(updater_path),
            '--pid', str(os.getpid()),
            '--staged-dir', str(staged_dir),
            '--install-dir', str(install_dir),
            '--relaunch', str(relaunch_path),
            '--log', str(log_path),
            '--cleanup', str(zip_path),
        ]

        popen_kwargs = {'cwd': str(self.updates_dir), 'close_fds': True}
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            popen_kwargs['start_new_session'] = True

        subprocess.Popen(args, **popen_kwargs)
        logger.info(f"Launched updater helper: {updater_path}")

    def check_previous_update_failure(self) -> Optional[str]:
        """Return the failure message left by a prior failed update apply,
        if any, clearing the marker so it's only surfaced once."""
        marker = self.updates_dir / 'update_failed.txt'
        if not marker.exists():
            return None
        try:
            message = marker.read_text(encoding='utf-8')
        except OSError:
            message = "The previous update did not complete successfully."
        try:
            marker.unlink()
        except OSError:
            pass
        return message


# Global instance for easy access, mirroring version.py's version_manager
update_manager = UpdateManager()
