"""Linux: find the system libraries Qt needs that aren't installed, and say
which package installs them on this distribution.

Two places can fail:
* Qt's X11 (xcb) platform plugin, while QApplication is created. Qt aborts
  there, so check_before_qt() runs first, reports with zenity/kdialog/xmessage
  (no Qt dialog is possible yet) and exits.
* QtWebEngine (Map Picker, FOV Simulator): its import raises ImportError
  naming the missing file (e.g. libxml2.so.2 on CachyOS/Arch).
  missing_from_import_error() reads the name from it.

Both messages can copy the install command and open the README's Linux
section (HELP_URL). Uses the standard library only - check_before_qt() runs
before Qt loads."""

import ctypes
import logging
import os
import re
import shutil
import struct
import subprocess
import sys

logger = logging.getLogger(__name__)

# The README's "Linux: required system libraries" section (its anchor is in README.md)
HELP_URL = "https://github.com/quake101/CosmosCollection#linux-libraries"

# Libraries known to be missing on some distributions: the file -> what needs
# it and the package that installs it per distribution family
KNOWN_LIBRARIES = {
    "libxcb-cursor.so.0": {
        "needed_by": "the app's window (Qt's X11 support)",
        "packages": {"debian": "libxcb-cursor0", "fedora": "xcb-util-cursor",
                     "arch": "xcb-util-cursor", "suse": "libxcb-cursor0"},
    },
    "libxml2.so.2": {
        "needed_by": "the Map Picker and FOV Simulator (QtWebEngine)",
        "packages": {"debian": "libxml2", "fedora": "libxml2",
                     "arch": "libxml2-legacy", "suse": "libxml2-2"},
    },
}

# Distribution family -> (install command, command that finds the package with a file)
_FAMILY_COMMANDS = {
    "debian": ("sudo apt-get install {packages}",
               "sudo apt-get install apt-file && sudo apt-file update && apt-file search {library}"),
    "fedora": ("sudo dnf install {packages}", "dnf provides '*/{library}'"),
    "arch": ("sudo pacman -S --needed {packages}", "sudo pacman -Fy && pacman -F {library}"),
    "suse": ("sudo zypper install {packages}", "zypper search --provides {library}"),
}

_FAMILY_IDS = {
    "debian": {"debian", "ubuntu", "linuxmint", "pop", "elementary", "zorin", "kali", "raspbian"},
    "fedora": {"fedora", "rhel", "centos", "rocky", "almalinux", "nobara", "ol"},
    "arch": {"arch", "cachyos", "manjaro", "endeavouros", "garuda", "artix"},
    "suse": {"suse", "opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles", "sled"},
}

_NOT_FOUND_RE = re.compile(r"([\w.+-]+\.so[\w.]*): cannot open shared object file")


def _clean_env():
    """The environment for other programs: a packaged build's LD_LIBRARY_PATH
    points at its bundled libraries, which break system programs (bash's
    readline, GTK in zenity). Same as UrlOpener.clean_subprocess_env, which
    can't be used before Qt loads."""
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "QTWEBENGINE_CHROMIUM_FLAGS"):
        env.pop(var, None)
    return env


def os_release():
    """/etc/os-release as a dict (NAME, ID, ID_LIKE, ...); empty if unreadable."""
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            with open(path, encoding="utf-8") as f:
                values = {}
                for line in f:
                    key, sep, value = line.strip().partition("=")
                    if sep:
                        values[key] = value.strip().strip('"\'')
                return values
        except OSError:
            continue
    return {}


def distro_family(release=None):
    """'debian', 'fedora', 'arch', 'suse' or None: from ID, then ID_LIKE."""
    release = os_release() if release is None else release
    ids = [release.get("ID", "").lower()] + release.get("ID_LIKE", "").lower().split()
    for distro_id in ids:
        for family, members in _FAMILY_IDS.items():
            if distro_id in members or (family == "suse" and "suse" in distro_id):
                return family
    return None


def _loads(library):
    """Whether the dynamic loader finds the library (the bundle's folder first)."""
    candidates = [library]
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.insert(0, os.path.join(bundle, library))
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
            return True
        except OSError:
            continue
    return False


def _qt_dirs():
    """(PySide6 package folder, Qt's lib folder) without importing Qt."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("PySide6")
        if spec is None or not spec.submodule_search_locations:
            return None, None
        package = list(spec.submodule_search_locations)[0]
        return package, os.path.join(package, "Qt", "lib")
    except Exception:
        return None, None


def _elf_interpreter(path):
    """The dynamic loader an ELF file names (PT_INTERP), e.g.
    /lib64/ld-linux-x86-64.so.2; None if it names none or can't be read.
    A shared library usually names none, so the caller tries this Python's."""
    try:
        with open(path, "rb") as f:
            header = f.read(64)
            if header[:4] != b"\x7fELF":
                return None
            is_64 = header[4] == 2
            endian = "<" if header[5] == 1 else ">"
            if is_64:
                phoff, = struct.unpack_from(endian + "Q", header, 0x20)
                phentsize, phnum = struct.unpack_from(endian + "HH", header, 0x36)
            else:
                phoff, = struct.unpack_from(endian + "I", header, 0x1C)
                phentsize, phnum = struct.unpack_from(endian + "HH", header, 0x2A)
            for i in range(phnum):
                f.seek(phoff + i * phentsize)
                entry = f.read(phentsize)
                if struct.unpack_from(endian + "I", entry, 0)[0] != 3:  # PT_INTERP
                    continue
                if is_64:
                    offset, = struct.unpack_from(endian + "Q", entry, 8)
                    size, = struct.unpack_from(endian + "Q", entry, 32)
                else:
                    offset, = struct.unpack_from(endian + "I", entry, 4)
                    size, = struct.unpack_from(endian + "I", entry, 16)
                f.seek(offset)
                return f.read(size).rstrip(b"\0").decode() or None
    except (OSError, struct.error, UnicodeDecodeError):
        pass
    return None


def _ldd_missing(path, library_dirs):
    """The libraries the dynamic loader can't find for path, or None when it
    can't be asked. Runs the loader itself in trace mode (what ldd does,
    without ldd's bash script), with library_dirs searched first so the
    libraries Qt and the bundle ship aren't reported."""
    if not os.path.exists(path):
        return None
    dirs = [d for d in library_dirs if d and os.path.isdir(d)]
    env = _clean_env()
    env["LD_TRACE_LOADED_OBJECTS"] = "1"
    loader = _elf_interpreter(sys.executable)
    if loader and os.path.exists(loader):
        command = [loader] + (["--library-path", os.pathsep.join(dirs)] if dirs else []) + [path]
    elif shutil.which("ldd"):
        env.pop("LD_TRACE_LOADED_OBJECTS")
        if dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs)
        command = [shutil.which("ldd"), path]
    else:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 and "=>" not in result.stdout:
        return None
    missing = []
    for line in result.stdout.splitlines():
        name, sep, target = line.strip().partition("=>")
        if sep and target.strip() == "not found":
            missing.append(name.strip())
    return missing


def _uses_xcb():
    """Whether Qt will start on its X11 (xcb) plugin: QT_QPA_PLATFORM's first
    choice, or - when unset - anything but a Wayland session (Qt 6 uses its
    Wayland plugin there)."""
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform:
        return platform.split(";")[0].split(":")[0] == "xcb"
    return os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland" or not os.environ.get("WAYLAND_DISPLAY")


def missing_for_window():
    """The libraries Qt's X11 plugin needs that aren't installed (empty when
    Qt won't use it). Asks the dynamic loader about the plugin, so libraries
    not in KNOWN_LIBRARIES are found too; failing that, checks the known one."""
    if not sys.platform.startswith("linux") or not _uses_xcb():
        return []
    package, qt_lib = _qt_dirs()
    if package:
        plugin = os.path.join(package, "Qt", "plugins", "platforms", "libqxcb.so")
        found = _ldd_missing(plugin, [qt_lib, getattr(sys, "_MEIPASS", None)])
        if found is not None:
            return found
    return [] if _loads("libxcb-cursor.so.0") else ["libxcb-cursor.so.0"]


def missing_from_import_error(error):
    """The library an ImportError says can't be opened (as a list), or []."""
    return _NOT_FOUND_RE.findall(str(error))


def _advice(libraries, release):
    """(message text, the commands in it)."""
    family = distro_family(release)
    distro = release.get("PRETTY_NAME") or release.get("NAME") or "your Linux distribution"
    lines = [f"Missing system {'library' if len(libraries) == 1 else 'libraries'}:"]
    packages, unknown = [], []
    for library in libraries:
        info = KNOWN_LIBRARIES.get(library)
        lines.append(f"  • {library}" + (f" - needed by {info['needed_by']}" if info else ""))
        package = info["packages"].get(family) if info and family else None
        if package:
            if package not in packages:
                packages.append(package)
        else:
            unknown.append(library)
    install, search = _FAMILY_COMMANDS.get(family, (None, None))
    commands = []
    if packages:
        commands.append(install.format(packages=" ".join(packages)))
        lines += ["", f"To install on {distro}, run in a terminal:", "  " + commands[-1]]
    if unknown:
        lines.append("")
        if search:
            lines.append("To find the package that provides "
                         + ("it" if len(unknown) == 1 else "each one") + ", run:")
            for library in unknown:
                commands.append(search.format(library=library))
                lines.append("  " + commands[-1])
        else:
            lines.append(f"Install the package that provides {', '.join(unknown)} "
                         f"with {distro}'s package manager.")
    lines += ["", "Then start Cosmos Collection again.", "", f"Help: {HELP_URL}"]
    return "\n".join(lines), commands


def install_advice(libraries, release=None):
    """Plain text: each missing library, what needs it, and the command that
    installs it on this distribution - or how to find the package for one
    this doesn't know - and the README's link."""
    return _advice(libraries, os_release() if release is None else release)[0]


def install_commands(libraries, release=None):
    """The commands install_advice() gives, one per line, for copying."""
    return "\n".join(_advice(libraries, os_release() if release is None else release)[1])


# --- Before Qt: desktop dialog tools ---

def _copy_without_qt(text):
    """Put text on the clipboard with wl-copy, xclip or xsel. False if none is
    installed or it failed. (They stay running to serve the clipboard after
    the app exits.)"""
    tools = []
    if os.environ.get("WAYLAND_DISPLAY"):
        tools.append(["wl-copy"])
    if os.environ.get("DISPLAY"):
        tools += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for command in tools:
        if shutil.which(command[0]):
            try:
                subprocess.run(command, input=text, text=True, timeout=5, env=_clean_env(),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def _can_copy_without_qt():
    return any(shutil.which(tool) for tool in
               (["wl-copy"] if os.environ.get("WAYLAND_DISPLAY") else [])
               + (["xclip", "xsel"] if os.environ.get("DISPLAY") else []))


def _open_without_qt(url):
    for tool in ("xdg-open", "gio"):
        if shutil.which(tool):
            try:
                subprocess.Popen([tool, url] if tool == "xdg-open" else [tool, "open", url],
                                 env=_clean_env(), start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue


COPY_LABEL = "Copy Command"
HELP_LABEL = "Open Help"


def _ask_dialog_tool(title, text, buttons):
    """Show text in zenity, kdialog or xmessage with buttons (labels besides
    Close). The label clicked, "" for Close, or None when no tool is installed."""
    env = _clean_env()

    def run(command):
        return subprocess.run(command, capture_output=True, text=True, timeout=3600, env=env)

    if shutil.which("zenity"):
        command = ["zenity", "--error", "--no-wrap", f"--title={title}", f"--text={text}",
                   "--ok-label=Close"] + [f"--extra-button={label}" for label in buttons]
        return run(command).stdout.strip()  # The extra button's label, or nothing
    if shutil.which("kdialog"):
        labels = (buttons + [None, None])[:2]
        command = ["kdialog", "--title", title, "--warningyesnocancel", text,
                   "--yes-label", labels[0] or "Close", "--no-label", labels[1] or "Close",
                   "--cancel-label", "Close"]
        code = run(command).returncode
        return {0: labels[0], 1: labels[1]}.get(code) or ""
    if shutil.which("xmessage"):
        spec = ",".join([f"{label}:{i + 10}" for i, label in enumerate(buttons)] + ["Close:0"])
        code = run(["xmessage", "-center", "-buttons", spec, f"{title}\n\n{text}"]).returncode
        return buttons[code - 10] if 10 <= code < 10 + len(buttons) else ""
    return None


def _show_without_qt(title, text, commands):
    """Show a message before Qt can: always on the terminal / log, and in a
    desktop dialog tool if one is installed, with Copy Command and Open Help."""
    try:
        os.write(2, f"\n{title}\n\n{text}\n\n".encode("utf-8", "replace"))
    except OSError:
        pass
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    buttons = ([COPY_LABEL] if commands and _can_copy_without_qt() else []) + [HELP_LABEL]
    shown = text
    for _ in range(10):  # Each button closes the dialog: show it again after Copy / Open Help
        try:
            clicked = _ask_dialog_tool(title, shown, buttons)
        except (OSError, subprocess.SubprocessError):
            return
        if clicked == COPY_LABEL:
            copied = _copy_without_qt(commands)
            shown = text + ("\n\n✓ Copied - paste it into a terminal." if copied
                            else "\n\nCouldn't copy - select the command above instead.")
        elif clicked == HELP_LABEL:
            _open_without_qt(HELP_URL)
        else:
            return


def check_before_qt():
    """Before QApplication: if Qt's window support can't load, say what to
    install and exit instead of letting Qt abort with its own message."""
    try:
        missing = missing_for_window()
    except Exception as e:
        logger.debug(f"Could not check the system libraries: {e}")
        return
    if not missing:
        return
    advice, commands = _advice(missing, os_release())
    text = "Cosmos Collection can't open its window.\n\n" + advice
    logger.error(text)
    _show_without_qt("Cosmos Collection - Missing Libraries", text, "\n".join(commands))
    sys.exit(1)


# --- After Qt has started ---

def show_missing_library_dialog(parent, feature, libraries):
    """A warning saying feature needs libraries, with Copy Command (to the
    clipboard) and Open Help (the README's Linux section) buttons. The text
    is selectable too."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    from cosmoscollection.core.UrlOpener import open_url

    advice, commands = _advice(libraries, os_release())
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Missing Libraries")
    msg.setText(f"{feature} needs a system library that isn't installed.")
    msg.setInformativeText(advice)
    msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
    copy_btn = msg.addButton(COPY_LABEL, QMessageBox.ActionRole) if commands else None
    help_btn = msg.addButton(HELP_LABEL, QMessageBox.HelpRole)
    close_btn = msg.addButton(QMessageBox.Close)
    msg.setDefaultButton(copy_btn or close_btn)
    while True:  # A QMessageBox closes on any button: show it again after Copy / Open Help
        msg.exec()
        clicked = msg.clickedButton()
        if copy_btn is not None and clicked is copy_btn:
            QApplication.clipboard().setText("\n".join(commands))
            msg.setInformativeText(advice + "\n\n✓ Copied - paste it into a terminal.")
        elif clicked is help_btn:
            open_url(HELP_URL)
        else:
            return
