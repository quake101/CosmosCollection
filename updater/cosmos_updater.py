#!/usr/bin/env python3
"""
Cosmos Collection standalone updater.

This runs as its own process, separate from the main application, because a
running app can't overwrite its own executable/DLL files (especially on
Windows). The main app downloads and verifies a new release, extracts it to
a staging directory, then launches this helper and quits. This helper waits
for the main app's process to fully exit, mirrors the staged files over the
install directory, relaunches the app, and cleans up after itself.

Stdlib-only by design so it stays small and dependency-free when compiled by
PyInstaller - it must be able to run standalone even while the main app's
own bundled libraries are mid-replacement.

Usage:
    CosmosCollectionUpdater --pid <main_app_pid> --staged-dir <path>
        --install-dir <path> --relaunch <path_to_exe>
        [--log <path>] [--cleanup <path> ...]
"""

import argparse
import ctypes
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


def main():
    parser = argparse.ArgumentParser(description="Cosmos Collection updater")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staged-dir", required=True)
    parser.add_argument("--install-dir", required=True)
    parser.add_argument("--relaunch", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument(
        "--cleanup", action="append", default=[],
        help="Extra file/dir to delete after a successful update (repeatable)",
    )
    args = parser.parse_args()

    _log(args.log, f"Updater starting, waiting for pid {args.pid} to exit")
    _wait_for_pid_exit(args.pid, timeout=60)
    time.sleep(1.0)  # grace period for Windows to release file handles

    try:
        _mirror_sync(args.staged_dir, args.install_dir, args.log)
        _log(args.log, "Mirror sync complete")
    except Exception as e:
        _log(args.log, f"ERROR: update failed during file sync: {e}")
        _write_failure_marker(args.log, args.install_dir, args.staged_dir, e)
        # Try to relaunch the (old, possibly partially-updated) app anyway so
        # the user isn't left with nothing running.
        try:
            subprocess.Popen([args.relaunch], cwd=args.install_dir, close_fds=True)
        except OSError:
            pass
        return 1

    try:
        subprocess.Popen([args.relaunch], cwd=args.install_dir, close_fds=True)
        _log(args.log, f"Relaunched {args.relaunch}")
    except OSError as e:
        _log(args.log, f"ERROR: could not relaunch app: {e}")

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


if __name__ == "__main__":
    sys.exit(main())
