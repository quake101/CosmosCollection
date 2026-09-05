#!/usr/bin/env python3
"""
Cosmos Collection standalone updater.

This runs as its own process, separate from the main application, because a
running app can't overwrite its own executable/DLL files (especially on
Windows). The main app downloads and verifies a new release, extracts it to
a staging directory, then launches this helper and quits. This helper waits
for the main app's process to fully exit, mirrors the staged files over the
install directory, relaunches the app, and cleans up after itself.

The actual wait/copy/relaunch logic (_run_update below) is stdlib-only by
design, so the update itself can never be blocked by a GUI problem - it must
be able to run standalone even while the main app's own bundled libraries
are mid-replacement. main() additionally shows a PySide6 progress dialog
(see CosmosUpdaterWorker.py) reporting each stage, imported lazily and
wrapped so any failure to create it just falls back to running the update
headless.

Usage:
    CosmosCollectionUpdater --pid <main_app_pid> --staged-dir <path>
        --install-dir <path> --relaunch <path_to_exe>
        [--log <path>] [--cleanup <path> ...]

This is a windowed (console=False) build with no visible stdout/stderr, so
running it directly with no/bad arguments or --help shows a message box
with the usage text above instead of printing it (see
_show_standalone_help).
"""

import argparse
import ctypes
import io
import os
import shutil
import subprocess
import sys
import time


def _log(log_path, message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _wait_for_pid_exit(pid, timeout=60):
    """Block until the given process id has exited, or timeout seconds pass."""
    deadline = time.time() + timeout

    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid
        )
        if not handle:
            # Already gone (or never existed)
            return True
        try:
            remaining_ms = max(0, int((deadline - time.time()) * 1000))
            result = kernel32.WaitForSingleObject(handle, remaining_ms)
            return result != WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    else:
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass  # exists, owned by someone else - treat as still running
            time.sleep(0.25)
        return False


def _retry(func, attempts=10, delay=0.5):
    """Run func() with retries, swallowing PermissionError/OSError between
    attempts. Windows can hold file handles open briefly after a process
    exits, so a straight copy/remove can fail the first try or two."""
    last_err = None
    for _ in range(attempts):
        try:
            return func()
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(delay)
    raise last_err


# Top-level app executables that must be able to run via execve() after an update.
# shutil.copy2() below preserves whatever mode the staged source file already has,
# which is normally correct -- except the *main app's* own extraction step
# (AppUpdater.py, a separate, older copy of the code doing the staging) has at times
# shipped with a bug where Python's zipfile.extractall() silently drops the Unix
# executable bit on every extracted file. This helper can't assume the app that
# staged the files it's copying already has that fixed: an old, buggy app version is
# exactly the one that needs an update most. So explicitly re-assert +x on these by
# name rather than trusting the source mode -- that makes the fix self-healing on the
# very next update for every user, regardless of which app version initiated it.
_EXECUTABLE_NAMES = {'CosmosCollection', 'CosmosCollection-CLI', 'CosmosCollectionUpdater'}


def _mirror_sync(src_dir, dst_dir, log_path):
    """Make dst_dir's contents match src_dir's: copy every file from
    src_dir into dst_dir (overwriting), then delete anything in dst_dir
    that isn't present in src_dir (handles files removed between
    releases)."""

    # os.walk() silently yields nothing for a missing/empty directory rather
    # than raising - without this check, a wrong/missing staged-dir would
    # copy nothing *and* then delete every existing install file in the
    # "remove stale entries" pass below, destroying the working install.
    if not os.path.isdir(src_dir) or not os.listdir(src_dir):
        raise RuntimeError(f"staged update directory is missing or empty: {src_dir}")

    # Copy/overwrite new + changed files
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        dst_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        if not os.path.isdir(dst_root):
            _retry(lambda: os.makedirs(dst_root, exist_ok=True))
        for name in files:
            src_file = os.path.join(root, name)
            dst_file = os.path.join(dst_root, name)
            _retry(lambda sf=src_file, df=dst_file: shutil.copy2(sf, df))
            if sys.platform != "win32" and rel == "." and name in _EXECUTABLE_NAMES:
                try:
                    os.chmod(dst_file, 0o755)
                except OSError as e:
                    _log(log_path, f"WARN: could not set executable bit on {dst_file}: {e}")

    # Remove files/dirs present in dst_dir but no longer in src_dir
    for root, dirs, files in os.walk(dst_dir, topdown=False):
        rel = os.path.relpath(root, dst_dir)
        src_root = src_dir if rel == "." else os.path.join(src_dir, rel)
        for name in files:
            if not os.path.exists(os.path.join(src_root, name)):
                dst_file = os.path.join(root, name)
                try:
                    _retry(lambda df=dst_file: os.remove(df))
                except OSError as e:
                    _log(log_path, f"WARN: could not remove stale file {dst_file}: {e}")
        for name in dirs:
            if not os.path.isdir(os.path.join(src_root, name)):
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _popen_relaunch(args):
    """Launch args.relaunch, escaping any systemd unit/cgroup this helper
    process might itself be a member of, when possible (see
    _relaunch_app's docstring for why that matters)."""
    cmd = [args.relaunch]
    if sys.platform != "win32":
        systemd_run = shutil.which("systemd-run")
        if systemd_run:
            # --scope adopts this exact command into a brand-new,
            # independent transient unit instead of leaving it in whatever
            # cgroup this helper inherited. --collect cleans that unit up
            # once it exits instead of leaving a "failed"/"exited" unit
            # behind forever (as the app's own KDE/GNOME-created units do).
            cmd = [systemd_run, '--user', '--scope', '--quiet', '--collect', '--'] + cmd
    return subprocess.Popen(cmd, cwd=args.install_dir, close_fds=True)


def _relaunch_app(args, log_path, attempts=2, settle=1.5, healthcheck=0.75):
    """Launch args.relaunch and confirm it's actually still running a beat
    later, retrying once if not. Returns True if the app appears to have
    started successfully.

    A bare Popen() only raises when execve() itself fails outright (bad
    path, missing +x) - it says nothing about what happens a moment later.
    On a systemd-managed desktop (KDE and GNOME both do this), launching
    this app wraps it in its own per-launch systemd scope/service with
    KillMode=control-group. The instant that unit's tracked process exits
    - which is exactly what just happened, since that's the whole reason
    this updater is running - systemd tears down its entire cgroup. This
    helper, and the app it's about to relaunch, can still be sitting in
    that very cgroup at that moment: start_new_session (setsid) moves a
    process to a new *session*, not a new *cgroup*, so it does nothing to
    protect against this. The result is exactly what was observed in the
    wild: Popen() reports success (the exec did succeed), the log says
    "Relaunched ...", and the process is killed a moment later - before it
    can render anything or write so much as its first log line - so
    nothing else ever shows it failed either. Confirming the process is
    still alive a fraction of a second later, and retrying after a short
    settle delay if it isn't, turns that silent lie into an honest,
    self-healing result.
    """
    for attempt in range(1, attempts + 1):
        try:
            proc = _popen_relaunch(args)
        except OSError as e:
            _log(log_path, f"ERROR: could not relaunch app (attempt {attempt}/{attempts}): {e}")
            time.sleep(settle)
            continue

        time.sleep(healthcheck)
        exit_code = proc.poll()
        if exit_code is None:
            _log(log_path, f"Relaunched {args.relaunch} (pid {proc.pid})")
            return True

        _log(
            log_path,
            f"WARN: relaunched process exited immediately (code {exit_code}) "
            f"on attempt {attempt}/{attempts}"
        )
        time.sleep(settle)

    _log(log_path, f"ERROR: app did not stay running after {attempts} relaunch attempts")
    return False


def _write_failure_marker(log_path, install_dir, staged_dir, error):
    marker_dir = os.path.dirname(log_path) if log_path else install_dir
    marker = os.path.join(marker_dir, "update_failed.txt")
    try:
        with open(marker, "w", encoding="utf-8") as f:
            f.write(
                "Cosmos Collection update failed to apply.\n"
                f"Error: {error}\n"
                f"Downloaded update files were left at: {staged_dir}\n"
            )
    except OSError:
        pass


def _run_update(args, status_cb):
    """Runs the wait/copy/relaunch steps, reporting each stage through
    status_cb(text) (in addition to the log file) so a progress dialog can
    show live status. Returns the process exit code; failures are reported
    rather than raised so the caller can still relaunch the (possibly old)
    app either way, matching the original behavior."""

    def _report(text):
        _log(args.log, text)
        status_cb(text)

    _report("Waiting for Cosmos Collection to close...")
    _wait_for_pid_exit(args.pid, timeout=60)
    time.sleep(1.0)  # grace period for Windows to release file handles

    try:
        _report("Copying updated files...")
        _mirror_sync(args.staged_dir, args.install_dir, args.log)
        _log(args.log, "Mirror sync complete")
    except Exception as e:
        _report(f"Update failed: {e}")
        _log(args.log, f"ERROR: update failed during file sync: {e}")
        _write_failure_marker(args.log, args.install_dir, args.staged_dir, e)
        # Try to relaunch the (old, possibly partially-updated) app anyway so
        # the user isn't left with nothing running.
        _relaunch_app(args, args.log)
        return 1

    _report("Restarting Cosmos Collection...")
    if not _relaunch_app(args, args.log):
        _report("Cosmos Collection did not restart - please launch it manually.")

    shutil.rmtree(args.staged_dir, ignore_errors=True)
    for path in args.cleanup:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    _log(args.log, "Update applied successfully")
    return 0


def _show_standalone_help(usage_text):
    """Show argparse's usage/help text in a message box.

    This is a windowed (console=False) build, so PyInstaller sets
    sys.stdout/sys.stderr to None - argparse's normal print-and-exit for
    --help or missing/bad arguments would otherwise crash silently the
    moment it tries to write, leaving someone who double-clicks the exe
    directly with no explanation at all. A QMessageBox works regardless,
    since it doesn't depend on either stream."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setWindowTitle("Cosmos Collection Updater")
        box.setIcon(QMessageBox.Information)
        box.setText(
            "This is an internal helper that Cosmos Collection launches "
            "automatically to apply an update. It isn't meant to be run "
            "directly."
        )
        box.setDetailedText(usage_text)
        box.exec()
    except Exception:
        pass  # Qt unavailable - nothing more we can do in a windowed build


def main():
    parser = argparse.ArgumentParser(
        prog="CosmosCollectionUpdater",
        description=(
            "Cosmos Collection's standalone update helper. Cosmos Collection "
            "launches this automatically after downloading an update - it "
            "isn't meant to be run directly."
        ),
    )
    parser.add_argument(
        "--pid", type=int, required=True,
        help="Process ID of the running Cosmos Collection instance to wait for",
    )
    parser.add_argument(
        "--staged-dir", required=True,
        help="Directory holding the already-extracted update to copy into place",
    )
    parser.add_argument(
        "--install-dir", required=True,
        help="Cosmos Collection installation directory to update",
    )
    parser.add_argument(
        "--relaunch", required=True,
        help="Executable to relaunch once the update has been applied",
    )
    parser.add_argument(
        "--log", default=None,
        help="Optional log file to append progress/status to",
    )
    parser.add_argument(
        "--cleanup", action="append", default=[],
        help="Extra file/dir to delete after a successful update (repeatable)",
    )

    # argparse writes --help/usage-error text to sys.stdout/sys.stderr,
    # which are None in this windowed build - substitute an in-memory
    # buffer so that write succeeds and we can show its contents in a
    # dialog, instead of the write itself crashing the process first.
    captured = io.StringIO()
    if sys.stdout is None:
        sys.stdout = captured
    if sys.stderr is None:
        sys.stderr = captured

    try:
        args = parser.parse_args()
    except SystemExit as e:
        _show_standalone_help(captured.getvalue().strip() or parser.format_help())
        return e.code if isinstance(e.code, int) else 2

    _log(args.log, f"Updater starting, waiting for pid {args.pid} to exit")

    try:
        from CosmosUpdaterWorker import run_with_progress
        return run_with_progress(lambda status_cb: _run_update(args, status_cb))
    except Exception as e:
        # A GUI problem (missing display, Qt failure, etc.) must never block
        # the actual update - fall back to running it headless.
        _log(args.log, f"WARN: progress dialog unavailable, continuing headless: {e}")
        return _run_update(args, lambda _text: None)


if __name__ == "__main__":
    sys.exit(main())
