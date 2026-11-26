"""
Native progress window for the standalone updater (see CosmosUpdater.py).

The updater used to show a PySide6 dialog, which meant bundling Qt into its
one-file build - most of a 45 MB file, for one label and one progress bar.
This draws the window with what each OS already has instead:

  - Windows: a plain Win32 window (CosmosUpdaterWin.py, ctypes)
  - macOS:   an AppKit window (CosmosUpdaterMac.py, ctypes)
  - Linux:   zenity or kdialog, whichever fits the desktop, so the window
             matches GTK or KDE themes

The window always runs in its own process, fed one command per line over
its stdin. If it fails to start or crashes, the writes to it just stop
landing and the update carries on headless: a GUI problem can never leave
the install half-copied. When the updater closes the pipe (or dies), the
window closes.

Commands sent to the window process (UTF-8, one per line):
    text <message>    status line
    progress <0-100>  determinate bar
    busy              indeterminate (animated) bar
"""

import html
import os
import shutil
import subprocess
import sys
import threading

TITLE = "Updating Cosmos Collection"
UI_FLAG = "--progress-ui"


def resource_path(*parts):
    """Path to a bundled resource, working both frozen (one-file build -
    files are extracted under sys._MEIPASS) and running from source (this
    file lives in updater/, resources are one directory up)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return os.path.join(base, *parts)


def icon_path():
    path = resource_path("images", "CosmosCollectionUpdater.png")
    return path if os.path.isfile(path) else None


def _system_tool_env():
    """Environment for running the desktop's own programs (zenity, kdialog,
    dbus-send, osascript). A frozen Linux build can point LD_LIBRARY_PATH at
    its own extracted libraries; a system program loading those instead of
    the system's copies can crash, so put back the original value."""
    env = os.environ.copy()
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and meipass in env.get("LD_LIBRARY_PATH", ""):
        original = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if original is not None:
            env["LD_LIBRARY_PATH"] = original
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _self_command():
    """Command that starts this updater again in window-only mode."""
    if getattr(sys, "frozen", False):
        return [sys.executable, UI_FLAG]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CosmosUpdater.py")
    return [sys.executable, script, UI_FLAG]


def _self_env():
    """Environment for the window process: a fresh, independent copy of this
    one-file build rather than one sharing this process's extracted files,
    which the bootloader deletes as soon as this process exits."""
    env = os.environ.copy()
    if getattr(sys, "frozen", False):
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"  # PyInstaller 6.9+
        env.pop("_MEIPASS2", None)  # older PyInstaller
    return env


def _clean_line(text):
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------
# Updater side: the objects CosmosUpdater._run_update reports through
# ---------------------------------------------------------------------------

class NullProgress:
    """No window: every call is a no-op."""

    def text(self, message):
        pass

    def progress(self, percent):
        """percent 0-100, or None for an indeterminate (busy) bar."""
        pass

    def close(self):
        pass


class _PipeProgress(NullProgress):
    """Writes commands to a child process's stdin. Any failure (the window
    process crashed, was closed, never started properly) turns this into a
    no-op instead of raising."""

    def __init__(self, cmd, env):
        kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
            "close_fds": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(cmd, **kwargs)
        self._alive = True
        self._last_percent = object()

    def _send(self, line):
        if not self._alive:
            return
        try:
            self._proc.stdin.write((line + "\n").encode("utf-8", "replace"))
            self._proc.stdin.flush()
        except (OSError, ValueError):
            self._alive = False

    def text(self, message):
        self._send("text " + _clean_line(message))

    def progress(self, percent):
        if percent is not None:
            percent = max(0, min(100, int(percent)))
        if percent == self._last_percent:
            return
        self._last_percent = percent
        self._send("busy" if percent is None else f"progress {percent}")

    def close(self):
        self._alive = False
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
            except OSError:
                pass


class _ZenityProgress(_PipeProgress):
    """zenity --progress reads '# text' and bare percentages from stdin.
    It has no way to switch to a busy bar once started, so busy keeps the
    last value. --auto-close closes it at 100%, so the bar tops out at 99
    until the pipe closes, which also closes it."""

    def __init__(self, zenity, first_text):
        cmd = [
            zenity, "--progress",
            f"--title={TITLE}",
            f"--text={html.escape(first_text, quote=False)}",
            "--percentage=0",
            "--auto-close",
            "--no-cancel",
            "--width=440",
        ]
        icon = icon_path()
        attempts = [cmd + [f"--window-icon={icon}"], cmd] if icon else [cmd]
        for attempt in attempts:
            super().__init__(attempt, _system_tool_env())
            # An option this zenity doesn't know makes it exit at once
            # (--window-icon is deprecated in zenity 4): try the next form
            try:
                self._proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                return
        raise RuntimeError(f"zenity exited with code {self._proc.returncode}")

    def text(self, message):
        # zenity reads text as Pango markup
        self._send("# " + html.escape(_clean_line(message), quote=False))

    def progress(self, percent):
        if percent is None:
            return
        percent = max(0, min(99, int(percent)))
        if percent == self._last_percent:
            return
        self._last_percent = percent
        self._send(str(percent))


class _KdialogProgress(NullProgress):
    """kdialog --progressbar starts a KDE progress dialog and prints the
    D-Bus address it can be driven through; each update is a short
    dbus-send (or qdbus) call. Copy progress arrives one step per percent
    at most (it's deduplicated), so that's at most ~100 short calls."""

    _IFACE = "org.kde.kdialog.ProgressDialog"

    def __init__(self, kdialog, first_text):
        self._env = _system_tool_env()
        cmd = [kdialog, "--title", TITLE]
        icon = icon_path()
        if icon:
            cmd += ["--icon", icon]
        cmd += ["--progressbar", first_text, "100"]
        out = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=self._env, timeout=10, check=True,
        ).stdout.decode("utf-8", "replace").split()
        if len(out) < 2:
            raise RuntimeError("kdialog did not report its D-Bus address")
        self._service, self._path = out[0], out[1]
        self._dbus_send = shutil.which("dbus-send")
        self._qdbus = None
        if not self._dbus_send:
            for name in ("qdbus6", "qdbus-qt6", "qdbus", "qdbus-qt5"):
                self._qdbus = shutil.which(name)
                if self._qdbus:
                    break
        if not (self._dbus_send or self._qdbus):
            self._call("close")  # can't drive it, so don't leave it stuck open
            raise RuntimeError("neither dbus-send nor qdbus is installed")
        self._alive = True
        self._last_percent = object()
        self._busy = False

    def _run(self, cmd):
        try:
            subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=self._env, timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            self._alive = False

    def _call(self, method, text=None):
        if self._dbus_send:
            cmd = [self._dbus_send, "--session", "--type=method_call",
                   f"--dest={self._service}", self._path, f"{self._IFACE}.{method}"]
            if text is not None:
                cmd.append(f"string:{text}")
        elif self._qdbus:
            cmd = [self._qdbus, self._service, self._path, f"{self._IFACE}.{method}"]
            if text is not None:
                cmd.append(text)
        else:
            return
        self._run(cmd)

    def _set(self, prop, value):
        if self._dbus_send:
            self._run([self._dbus_send, "--session", "--type=method_call",
                       f"--dest={self._service}", self._path,
                       "org.freedesktop.DBus.Properties.Set",
                       f"string:{self._IFACE}", f"string:{prop}", f"variant:int32:{value}"])
        elif self._qdbus:
            # kdialog's documented form; qdbus converts the value to the
            # property's own type
            self._run([self._qdbus, self._service, self._path, "Set", "", prop, str(value)])

    def text(self, message):
        if self._alive:
            self._call("setLabelText", _clean_line(message))

    def progress(self, percent):
        if not self._alive:
            return
        if percent is not None:
            # 99 at most: like QProgressDialog, which it's built on, it
            # closes itself on reaching its maximum
            percent = max(0, min(99, int(percent)))
        if percent == self._last_percent:
            return
        self._last_percent = percent
        if percent is None:
            self._set("maximum", 0)  # a 0 maximum makes it a busy bar
            self._busy = True
            return
        if self._busy:
            self._set("maximum", 100)
            self._busy = False
        self._set("value", percent)

    def close(self):
        if self._alive:
            self._alive = False
            self._call("close")


def _linux_progress(first_text):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("no display")
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    kde_like = any(name in desktop for name in ("KDE", "LXQT", "TRINITY"))
    order = ("kdialog", "zenity") if kde_like else ("zenity", "kdialog")
    errors = []
    for name in order:
        tool = shutil.which(name)
        if not tool:
            continue
        try:
            if name == "zenity":
                return _ZenityProgress(tool, first_text)
            return _KdialogProgress(tool, first_text)
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("; ".join(errors) or "neither zenity nor kdialog is installed")


def open_progress(first_text, log=None):
    """Open the progress window showing first_text and a busy bar. Never
    raises: if no window can be shown, returns a NullProgress and the
    update runs headless. log(message) receives the reason."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_progress(first_text)
        ui = _PipeProgress(_self_command(), _self_env())
        ui.text(first_text)
        ui.progress(None)
        return ui
    except Exception as e:
        if log:
            log(f"WARN: progress window unavailable, continuing headless: {e}")
        return NullProgress()


# ---------------------------------------------------------------------------
# Window side: the "--progress-ui" process (Windows and macOS)
# ---------------------------------------------------------------------------

def _stdin_binary():
    if sys.stdin is not None:
        return sys.stdin.buffer
    if sys.platform == "win32":
        # A windowed build can start with sys.stdin unset even when it was
        # given a pipe; open the handle it was given directly.
        import ctypes
        import msvcrt
        handle = ctypes.windll.kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if handle and handle != -1:
            return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb")
    return None


def start_command_reader(on_command):
    """Read commands from stdin on a background thread, calling
    on_command(kind, value) for each and on_command("close", "") once the
    pipe closes."""

    def _read():
        try:
            stream = _stdin_binary()
            if stream is not None:
                for raw in stream:
                    kind, _sep, value = raw.decode("utf-8", "replace").rstrip("\r\n").partition(" ")
                    if kind:
                        on_command(kind, value)
        except Exception:
            pass
        on_command("close", "")

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return thread


def run_ui_process():
    """Entry point for "--progress-ui": show the window until stdin closes."""
    try:
        if sys.platform == "win32":
            import CosmosUpdaterWin as ui
        elif sys.platform == "darwin":
            import CosmosUpdaterMac as ui
        else:
            return 1
        ui.run(TITLE, icon_path())
        return 0
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Message box for someone running the updater directly
# ---------------------------------------------------------------------------

def show_message(title, message, details=""):
    """Native information box. Best effort: does nothing if it can't."""
    body = f"{message}\n\n{details}" if details else message
    try:
        if sys.platform == "win32":
            import ctypes
            MB_ICONINFORMATION = 0x40
            ctypes.windll.user32.MessageBoxW(None, body, title, MB_ICONINFORMATION)
        elif sys.platform == "darwin":
            subprocess.run(
                ["osascript",
                 "-e", "on run argv",
                 "-e", 'display dialog (item 1 of argv) with title (item 2 of argv) '
                       'buttons {"OK"} default button 1 with icon note',
                 "-e", "end run",
                 body, title],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=_system_tool_env(),
            )
        else:
            env = _system_tool_env()
            zenity, kdialog = shutil.which("zenity"), shutil.which("kdialog")
            if zenity:
                cmd = [zenity, "--info", f"--title={title}",
                       f"--text={html.escape(body, quote=False)}", "--width=440"]
            elif kdialog:
                cmd = [kdialog, "--title", title, "--msgbox", body]
            else:
                return
            subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, env=env)
    except Exception:
        pass
