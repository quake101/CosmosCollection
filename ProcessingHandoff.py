#!/usr/bin/env python3
"""
Processing Handoff
Hands a completed session's subs to an external stacking/processing app:
Siril (a generated preprocessing script run headless via siril-cli) or
PixInsight's WeightedBatchPreprocessing (WBPP, via its command-line
automation mode). Logic only - the dialogs live in SessionCompletion.py.

Source subs are never moved or modified: they are hard-linked (or copied
when linking isn't possible) into a per-session workspace folder that the
external app works in.
"""

import os
import re
import sys
import shutil
import logging
import subprocess
from dataclasses import dataclass, field

from PySide6.QtCore import QSettings

import SessionFileScanner
import SessionObservations
from UrlOpener import clean_subprocess_env

logger = logging.getLogger(__name__)

SIRIL = "siril"
PIXINSIGHT = "pixinsight"
APP_LABELS = {SIRIL: "Siril", PIXINSIGHT: "PixInsight"}

# Minimum Siril version for the generated script's syntax (matches the
# 'requires' line of Siril's own bundled 1.4 preprocessing scripts), and the
# first version that can read XISF subs.
SIRIL_MIN_VERSION = (1, 3, 4)
SIRIL_XISF_VERSION = (1, 4, 0)

# Folder names inside a workspace. Frames are staged under FRAMES_DIR so a
# recursive scan of it (WBPP's dir= parameter) never picks up the outputs.
FRAMES_DIR = "frames"
WBPP_OUTPUT_DIR = "WBPP"


def _settings():
    return QSettings("CosmosCollection", "CosmosCollection")


# ---- Executable detection --------------------------------------------------

def _first_existing(paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def find_siril_cli(configured_path=""):
    """siril-cli (the headless console binary) - the configured path first
    (which may point at either siril or siril-cli), then the default install
    locations, then PATH. Mirrors PlateSolver's ASTAP auto-detection."""
    configured_path = (configured_path or "").strip()
    if configured_path and os.path.isfile(configured_path):
        folder, name = os.path.split(configured_path)
        # Pointed at the GUI binary - prefer its console sibling when present.
        if "siril-cli" not in name.lower():
            sibling = _first_existing([os.path.join(folder, "siril-cli.exe"), os.path.join(folder, "siril-cli")])
            if sibling:
                return sibling
        return configured_path

    if sys.platform == "win32":
        candidates = [r"C:\Program Files\Siril\bin\siril-cli.exe",
                      r"C:\Program Files (x86)\Siril\bin\siril-cli.exe"]
    elif sys.platform == "darwin":
        candidates = ["/Applications/Siril.app/Contents/MacOS/siril-cli"]
    else:
        candidates = ["/usr/bin/siril-cli", "/usr/local/bin/siril-cli"]
    return _first_existing(candidates) or shutil.which("siril-cli")


def find_siril_gui(siril_cli_path):
    """The Siril GUI binary next to siril-cli, for 'Open in Siril'."""
    if not siril_cli_path:
        return None
    folder = os.path.dirname(siril_cli_path)
    return _first_existing([os.path.join(folder, "siril.exe"), os.path.join(folder, "siril")]) or shutil.which("siril")


def find_pixinsight(configured_path=""):
    configured_path = (configured_path or "").strip()
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    if sys.platform == "win32":
        candidates = [r"C:\Program Files\PixInsight\bin\PixInsight.exe"]
    elif sys.platform == "darwin":
        candidates = ["/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight"]
    else:
        candidates = ["/opt/PixInsight/bin/PixInsight.sh", "/opt/PixInsight/bin/PixInsight"]
    return _first_existing(candidates) or shutil.which("PixInsight")


def find_wbpp_script(pixinsight_path):
    """WBPP.js lives under <install root>/src/scripts/BatchPreprocessing/. The
    executable sits a different depth below the install root on each OS
    (bin/ on Windows/Linux, PixInsight.app/Contents/MacOS/ on macOS), so walk
    up from it looking for the script."""
    if not pixinsight_path:
        return None
    folder = os.path.dirname(os.path.abspath(pixinsight_path))
    for _ in range(5):
        candidate = os.path.join(folder, "src", "scripts", "BatchPreprocessing", "WBPP.js")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return None


def resolve_executable(app):
    """The executable to run for an app given its saved path setting (or
    auto-detection when that's empty), or None."""
    configured = _settings().value(f"{app}_path", "", type=str)
    return find_siril_cli(configured) if app == SIRIL else find_pixinsight(configured)


def integration_status(app):
    """(ready, message) - ready only when the integration is enabled in Settings
    and its executable (and for PixInsight, WBPP.js) can be found."""
    label = APP_LABELS[app]
    if not _settings().value(f"{app}_integration_enabled", False, type=bool):
        return False, f"Enable {label} in Settings → Integrations to use this."
    exe = resolve_executable(app)
    if not exe:
        return False, f"{label} wasn't found. Set its path in Settings → Integrations."
    if app == PIXINSIGHT and not find_wbpp_script(exe):
        return False, "WBPP.js wasn't found in the PixInsight installation."
    return True, exe


def siril_version(siril_cli_path):
    """Siril's version as a tuple (e.g. (1, 4, 4)), or None if it can't be read."""
    try:
        result = subprocess.run([siril_cli_path, "--version"], capture_output=True, text=True, timeout=30,
                                env=clean_subprocess_env(), creationflags=_no_window_flag())
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout + result.stderr)
        return tuple(int(part) for part in match.groups()) if match else None
    except Exception as e:
        logger.warning(f"Could not read Siril version from {siril_cli_path}: {e}")
        return None


# ---- Launching --------------------------------------------------------------

def _no_window_flag():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def launch_detached(args, cwd=None):
    """Start an external GUI app that outlives this one, without any console
    windows flashing up. Returns True if it started."""
    kwargs = {"cwd": cwd, "env": clean_subprocess_env(), "close_fds": True,
              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        # Not DETACHED_PROCESS: siril.exe is a console-subsystem program, and with
        # no console at all every console helper it starts (gdbus, gspawn, ...)
        # gets a visible window of its own. CREATE_NO_WINDOW gives it a hidden
        # console they share instead; GUI-subsystem apps (PixInsight) ignore it.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(args, **kwargs)
        logger.info(f"Launched: {args}")
        return True
    except Exception as e:
        logger.error(f"Failed to launch {args}: {e}")
        return False


def pixinsight_wbpp_args(pixinsight_path, frames_dir, output_dir, run_immediately=False):
    """PixInsight command line that opens WBPP with every sub under frames_dir
    loaded (WBPP groups them by IMAGETYP/FILTER itself). -n starts a new
    instance so this works even while PixInsight is already open; loadOnly
    opens the WBPP dialog for review instead of running straight away.
    See BPP-automation.js in the PixInsight install for the parameters."""
    wbpp = find_wbpp_script(pixinsight_path)
    params = [wbpp, "automationMode=true", f"dir={frames_dir}", f"outputDirectory={output_dir}"]
    if not run_immediately:
        params.append("loadOnly")
    return [pixinsight_path, "-n", "--automation-mode", "-r=" + ",".join(params)]


def unsafe_wbpp_path(path):
    """WBPP splits its arguments on ',' and '=', so a path containing either
    can't be passed to it."""
    return "," in path or "=" in path


# ---- Session frames ---------------------------------------------------------

def sanitize_name(text, fallback="Unnamed"):
    """Safe for a file/folder name: runs of anything but letters, digits, '.',
    '-' and '_' become '_'."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._")
    return cleaned or fallback


def filter_key(filter_name):
    """Letters/digits only - also used inside Siril sequence names."""
    return re.sub(r"[^A-Za-z0-9]", "", str(filter_name or "")) or "NoFilter"


@dataclass
class FrameSet:
    """A session's frames, ready to stage. lights/flats map a filter key to
    source paths; darks/biases are shared across filters."""
    lights: dict = field(default_factory=dict)
    flats: dict = field(default_factory=dict)
    darks: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    missing: list = field(default_factory=list)   # attached files no longer on disk
    added_counts: dict = field(default_factory=dict)  # frames added from calibration folders, by kind
    # Added files whose IMAGETYP header names a different frame type than the
    # one they were added as (e.g. darks NINA saved as LIGHT), path -> kind.
    # WBPP sorts frames by that header, so their staged copies get it corrected.
    retype: dict = field(default_factory=dict)

    @property
    def light_count(self):
        return sum(len(paths) for paths in self.lights.values())

    def counts(self):
        return {
            "Light": self.light_count,
            "Flat": sum(len(paths) for paths in self.flats.values()),
            "Dark": len(self.darks),
            "Bias": len(self.biases),
        }

    def all_paths(self):
        for paths in list(self.lights.values()) + list(self.flats.values()) + [self.darks, self.biases]:
            yield from paths

    def light_folders(self):
        return sorted({os.path.dirname(p) for paths in self.lights.values() for p in paths})


def collect_session_frames(conn, session_id, check_exists=True):
    """Group a session's attached files by frame type (and filter, for lights
    and flats). Files without a recognized IMAGETYP only count as lights when
    the session has no typed Light frames at all. check_exists=False skips the
    per-file disk check (slow for hundreds of subs on a spinning drive) - use
    find_missing_frames()/remove_frames() off the UI thread instead."""
    files = SessionObservations.load_session_files(conn, session_id)
    frames = FrameSet()
    has_typed_lights = any(f["frame_type"] == "Light" for f in files)
    seen = set()
    for f in files:
        path = f["file_path"]
        if not path or path in seen:
            continue
        seen.add(path)
        frame_type = f["frame_type"]
        if frame_type == "Unknown" and not has_typed_lights:
            frame_type = "Light"
        if frame_type not in ("Light", "Flat", "Dark", "Bias"):
            continue
        if check_exists and not os.path.isfile(path):
            frames.missing.append(path)
            continue
        _add_frame(frames, frame_type, path, f["filter_name"])
    return frames


def find_missing_frames(frames):
    """Paths in the set that no longer exist on disk."""
    return [p for p in frames.all_paths() if not os.path.isfile(p)]


def remove_frames(frames, paths):
    """Drop paths from every group (and any group left empty), noting them as missing."""
    gone = set(paths)
    if not gone:
        return
    for groups in (frames.lights, frames.flats):
        for key in list(groups):
            groups[key] = [p for p in groups[key] if p not in gone]
            if not groups[key]:
                del groups[key]
    frames.darks[:] = [p for p in frames.darks if p not in gone]
    frames.biases[:] = [p for p in frames.biases if p not in gone]
    frames.missing.extend(p for p in paths if p not in frames.missing)


def _add_frame(frames, frame_type, path, filter_name=None):
    if frame_type == "Light":
        frames.lights.setdefault(filter_key(filter_name), []).append(path)
    elif frame_type == "Flat":
        frames.flats.setdefault(filter_key(filter_name), []).append(path)
    elif frame_type == "Dark":
        frames.darks.append(path)
    elif frame_type == "Bias":
        frames.biases.append(path)


def add_calibration_folder(frames, kind, folder):
    """Add every FITS/XISF file in a folder (recursively) as `kind` frames
    ('Dark', 'Flat' or 'Bias') - calibration frames are often kept in a
    library rather than attached to each session. Flats are grouped by their
    FILTER header. The user's choice of kind wins over the files' IMAGETYP
    header; files whose header says otherwise are noted in frames.retype.
    Returns how many new files were added."""
    existing = set(frames.all_paths())
    added = 0
    for root, _dirs, names in os.walk(folder):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() not in SessionFileScanner.SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            if path in existing:
                continue
            scanned = SessionFileScanner.scan_file(path)
            filter_name = None
            if scanned:
                if kind == "Flat":
                    filter_name = SessionFileScanner._clean_str(scanned.get("FILTER"))
                if scanned["_frame_type"] not in ("Unknown", kind):
                    frames.retype[path] = kind
            _add_frame(frames, kind, path, filter_name)
            existing.add(path)
            added += 1
    frames.added_counts[kind] = frames.added_counts.get(kind, 0) + added
    return added


def detect_osc(frames):
    """True if the first readable light sub has a Bayer pattern (a one-shot
    colour camera), which the Siril script must debayer."""
    for paths in frames.lights.values():
        for path in paths[:3]:
            scanned = SessionFileScanner.scan_file(path)
            if scanned is not None:
                return bool(scanned.get("BAYERPAT"))
    return False


def has_xisf_lights(frames):
    return any(p.lower().endswith(".xisf") for paths in frames.lights.values() for p in paths)


# ---- Workspace --------------------------------------------------------------

def default_workspace(app, dso_name, session_date, frames):
    """<base>/<DSO>_<date>_<App>, where base is the app's workspace setting or
    else the folder above the lights. Never an existing non-empty folder - a
    numeric suffix is added instead, so a re-run never mixes with an old one."""
    base = _settings().value(f"{app}_workspace_dir", "", type=str).strip()
    if not base:
        folders = frames.light_folders()
        if folders:
            try:
                common = os.path.commonpath(folders)
            except ValueError:  # lights on different drives
                common = folders[0]
            base = os.path.dirname(common) or common
        else:
            base = os.path.expanduser("~")
    name = f"{sanitize_name(dso_name)}_{sanitize_name(session_date, 'undated')}_{APP_LABELS[app]}"
    candidate = os.path.join(base, name)
    suffix = 2
    while os.path.exists(candidate) and os.listdir(candidate):
        candidate = os.path.join(base, f"{name}_{suffix}")
        suffix += 1
    return candidate


def workspace_problem(workdir, frames):
    """Why this workspace can't be used, or None. It must be empty (or new) -
    staging never deletes anything, so an old run's folders would get mixed in -
    and must not be one of the folders holding the source subs."""
    if not workdir:
        return "Choose a workspace folder."
    workdir = os.path.abspath(workdir)
    if os.path.isfile(workdir):
        return "The workspace path is a file, not a folder."
    if os.path.isdir(workdir) and os.listdir(workdir):
        return "The workspace folder isn't empty. Choose a new or empty folder."
    source_dirs = {os.path.abspath(os.path.dirname(p)) for p in frames.all_paths()}
    if workdir in source_dirs:
        return "The workspace can't be a folder that holds the subs themselves."
    return None


def plan_layout(frames):
    """Staged folder name (relative to the frames folder) for each group. The
    names double as a frame-type hint for WBPP, which falls back to the path
    for files with no IMAGETYP - it recognizes lights/flats/darks/bias
    (but not 'biases'). 'prestaged' lists the Siril sequences staged straight
    into process/ (see stage_workspace's siril_sequences)."""
    return {
        "lights": {key: f"lights_{key}" for key in sorted(frames.lights)},
        "flats": {key: f"flats_{key}" for key in sorted(frames.flats)},
        "darks": "darks" if frames.darks else None,
        "biases": "bias" if frames.biases else None,
        "prestaged": set(),
    }


def _frame_groups(frames, layout):
    """(staged folder, Siril sequence name, source paths) for every group."""
    for key, paths in sorted(frames.lights.items()):
        yield layout["lights"][key], f"light_{key}", paths
    for key, paths in sorted(frames.flats.items()):
        yield layout["flats"][key], f"flat_{key}", paths
    if frames.darks:
        yield layout["darks"], "dark", frames.darks
    if frames.biases:
        yield layout["biases"], "bias", frames.biases


# Siril sequences staged directly into process/ are named the way its convert
# command would name them: <sequence>_00001.fit, <sequence>_00002.fit, ...
SIRIL_PROCESS_DIR = "process"
SIRIL_SEQUENCE_EXT = "fit"


# IMAGETYP values WBPP (and most capture software) recognize.
IMAGETYP_VALUES = {"Light": "Light Frame", "Dark": "Dark Frame", "Flat": "Flat Field", "Bias": "Bias Frame"}


def _set_fits_frame_type(path, kind):
    """Rewrite IMAGETYP in a staged *copy* (never a hard link - that would
    change the original sub too)."""
    from astropy.io import fits
    fits.setval(path, "IMAGETYP", value=IMAGETYP_VALUES[kind])


def stage_workspace(frames, frames_dir, copy_files=False, progress_callback=None, is_cancelled=None,
                    fix_frame_types=False, siril_sequences=False):
    """Link (or copy) every frame into frames_dir/<group folder>/. Hard links
    cost no disk space and leave the originals untouched; a file on another
    volume (or a filesystem without hard links) is copied instead.

    With siril_sequences, all-FITS groups are instead linked straight into
    frames_dir/process/ under the names Siril's convert would give them, so
    the script can skip convert for them. On Windows convert can't symlink FITS
    subs without Developer Mode and silently copies every one first (minutes
    for a few hundred subs); hard links need no special permission. XISF groups
    still go through convert, which reads XISF.

    With fix_frame_types (for WBPP, which sorts frames by IMAGETYP), FITS files
    in frames.retype are copied instead and the copy's IMAGETYP corrected;
    XISF ones can't be and are returned as unfixed.

    Subs that have disappeared since the session was loaded are left out
    (and moved to frames.missing) rather than failing the whole staging.

    Returns (layout, linked_count, copied_count, unfixed_paths); stops early if
    is_cancelled()."""
    remove_frames(frames, find_missing_frames(frames))
    layout = plan_layout(frames)
    jobs = []  # (destination folder, destination name or None to keep the source's, source)
    for folder, sequence, paths in _frame_groups(frames, layout):
        all_fits = all(os.path.splitext(p)[1].lower() in SessionFileScanner.FITS_EXTENSIONS for p in paths)
        if siril_sequences and all_fits:
            layout["prestaged"].add(sequence)
            jobs.extend((SIRIL_PROCESS_DIR, f"{sequence}_{n:05d}.{SIRIL_SEQUENCE_EXT}", p)
                        for n, p in enumerate(paths, start=1))
        else:
            jobs.extend((folder, None, p) for p in paths)

    linked = copied = 0
    unfixed = []
    used_names = {}
    for i, (folder, fixed_name, src) in enumerate(jobs):
        if is_cancelled and is_cancelled():
            break
        dest_dir = os.path.join(frames_dir, folder)
        os.makedirs(dest_dir, exist_ok=True)
        if fixed_name:
            name = fixed_name
        else:
            # Subs from different nights often share file names; number them so
            # nothing is overwritten (and sort order stays chronological).
            names = used_names.setdefault(folder, set())
            base = os.path.basename(src)
            name = base if base.lower() not in names else f"{len(names):04d}_{base}"
            names.add(name.lower())
        dest = os.path.join(dest_dir, name)
        retype_kind = frames.retype.get(src) if fix_frame_types else None
        if retype_kind and src.lower().endswith(".xisf"):
            unfixed.append(src)
            retype_kind = None
        if retype_kind:
            shutil.copy2(src, dest)
            _set_fits_frame_type(dest, retype_kind)
            copied += 1
        elif not copy_files:
            try:
                os.link(src, dest)
                linked += 1
            except OSError:
                shutil.copy2(src, dest)
                copied += 1
        else:
            shutil.copy2(src, dest)
            copied += 1
        if progress_callback:
            progress_callback(i + 1, len(jobs))
    return layout, linked, copied, unfixed


# ---- Siril script -----------------------------------------------------------

def _flat_master_for(light_key, layout, flat_masters):
    if light_key in flat_masters:
        return flat_masters[light_key]
    # A single flat set with lights of a single filter - assume they belong
    # together even when their FILTER headers differ (or one is missing).
    if len(flat_masters) == 1 and len(layout["lights"]) == 1:
        return next(iter(flat_masters.values()))
    return None


def build_siril_script(layout, osc=False):
    """A Siril script (based on Siril 1.4's bundled Mono/OSC_Preprocessing.ssf)
    that only includes the steps whose frames exist: master bias, a master
    flat per filter, master dark, then calibrate → register → stack for each
    light filter. Sequences already staged in process/ (layout['prestaged'])
    skip Siril's convert. Starts in the frames folder and works in process/;
    each filter's stack is saved in the frames folder as result_<filter>.fit."""
    prestaged = layout.get("prestaged", set())
    lines = [
        "# Generated by Cosmos Collection",
        f"requires {'.'.join(str(n) for n in SIRIL_MIN_VERSION)}",
    ]
    if prestaged:
        # The staged sequence files are .fit; Siril only recognizes sequences
        # with its configured FITS extension, which users can change.
        lines.append(f"setext {SIRIL_SEQUENCE_EXT}")
    lines.append("")
    in_process = False  # the script starts in the frames folder

    def open_sequence(sequence, folder):
        """Lines that leave Siril in process/ with `sequence` available there:
        just a cd for a pre-staged sequence, else a convert from its folder."""
        nonlocal in_process
        was_in_process, in_process = in_process, True
        if sequence in prestaged:
            return [] if was_in_process else [f"cd {SIRIL_PROCESS_DIR}"]
        return [f"cd {'../' if was_in_process else ''}{folder}",
                f"convert {sequence} -out=../{SIRIL_PROCESS_DIR}", f"cd ../{SIRIL_PROCESS_DIR}"]

    has_bias = bool(layout["biases"])
    has_dark = bool(layout["darks"])

    if has_bias:
        lines += ["# Master bias"] + open_sequence("bias", layout["biases"]) + [
            "stack bias rej 3 3 -nonorm -out=../masters/bias_stacked",
            "",
        ]

    flat_masters = {}
    for key, folder in layout["flats"].items():
        seq = f"flat_{key}"
        lines += [f"# Master flat ({key})"] + open_sequence(seq, folder)
        if has_bias:
            lines.append(f"calibrate {seq} -bias=../masters/bias_stacked")
            seq = f"pp_{seq}"
        master = f"flat_stacked_{key}"
        lines += [f"stack {seq} rej 3 3 -norm=mul -out=../masters/{master}", ""]
        flat_masters[key] = master

    if has_dark:
        lines += ["# Master dark"] + open_sequence("dark", layout["darks"]) + [
            "stack dark rej 3 3 -nonorm -out=../masters/dark_stacked",
            "",
        ]

    for key, folder in layout["lights"].items():
        seq = f"light_{key}"
        lines += [f"# Lights ({key})"] + open_sequence(seq, folder)
        options = []
        if has_dark:
            options += ["-dark=../masters/dark_stacked", "-cc=dark"]
            if osc:
                options.append("-cfa")
        flat = _flat_master_for(key, layout, flat_masters)
        if flat:
            options.append(f"-flat=../masters/{flat}")
            if osc:
                options.append("-equalize_cfa")
        if osc:
            options.append("-debayer")
        if options:
            lines.append(f"calibrate {seq} {' '.join(options)}")
            seq = f"pp_{seq}"
        stack_options = "-norm=addscale -output_norm" + (" -rgb_equal" if osc else "") + " -32b"
        lines += [
            f"register {seq}",
            f"stack r_{seq} rej 3 3 {stack_options} -out=result_{key}",
            f"load result_{key}",
            "mirrorx -bottomup",
            f"save ../result_{key}",
            "",
        ]

    lines.append("close")
    return "\n".join(lines) + "\n"


def layout_counts(layout, frames):
    """Frames per Siril sequence (light_<filter>, flat_<filter>, dark, bias),
    for weighting the Siril steps."""
    return {sequence: len(paths) for _folder, sequence, paths in _frame_groups(frames, layout)}


# Commands that do real work (and report progress); everything else is instant.
# Values are rough relative cost per frame - registration is the slowest.
SIRIL_HEAVY_COMMANDS = {"convert": 1.0, "calibrate": 1.0, "register": 2.0, "stack": 1.5}


def siril_steps(script, sequence_counts):
    """Every command in a generated script, in order, as dicts with 'command',
    'section' (the '# ...' comment it falls under) and 'weight' (its share of
    the run - zero for instant commands; heavy ones scale with the frame count
    of the sequence they work on). Siril logs 'Running command: <name>' for
    each, which is how SirilRunDialog follows along."""
    steps = []
    section = ""
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            text = line.lstrip("# ").strip()
            if text and not text.startswith("Generated by"):
                section = text
            continue
        command, _, arguments = line.partition(" ")
        weight = SIRIL_HEAVY_COMMANDS.get(command, 0.0)
        if weight:
            # 'stack r_pp_light_L ...' works on light_L's frames
            sequence = re.sub(r"^(?:r_)?(?:pp_)?", "", arguments.split(" ", 1)[0])
            weight *= max(1, sequence_counts.get(sequence, 1))
        steps.append({"command": command, "section": section, "weight": weight})
    return steps


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PROGRESS_RE = re.compile(r"^progress:\s*(?P<text>.*?)(?:,?\s*(?P<pct>\d+(?:\.\d+)?)%)?\s*$")


def clean_siril_line(line):
    """Siril redraws its console progress line with terminal escape codes
    (cursor-up, erase-line, then '\\r'), which show up as garbage outside a
    terminal. Keep only the text after the last carriage return."""
    line = _ANSI_RE.sub("", line).rstrip("\r\n")  # a CRLF line ending isn't a redraw
    return line.rsplit("\r", 1)[-1].strip()


def parse_siril_progress(line):
    """(description or '', percent or None) for a 'progress:' line, else None."""
    match = _PROGRESS_RE.match(line)
    if not match:
        return None
    pct = match.group("pct")
    return match.group("text").strip(" ,"), (float(pct) if pct is not None else None)


# Siril's internal/debug chatter - hidden from the output view.
_SIRIL_NOISE_RE = re.compile(
    r"^(HDU \d+: type=|\d+: running command|Block \d+: channel|bitpix for the sequence|"
    r"number of filtered-in images|Reading sequence file|Writing sequence file|"
    r"Reading sequence failed|allocating data for|image size:|Memory per image|cfitsio was compiled)")


def is_siril_noise(line):
    return bool(_SIRIL_NOISE_RE.match(line))


# Logged by Siril's convert on Windows when it has to copy FITS subs into the
# process folder because creating symbolic links needs Developer Mode.
SIRIL_SYMLINK_WARNING = "enable the Developer Mode in order to create symbolic links"


def siril_python_setup_state(line):
    """Siril 1.4 keeps its own Python environment for its scripting module and
    occasionally (re)builds or updates it, which can take minutes. True when a
    line says that's starting, False when it says it's finished, else None."""
    lower = line.lower()
    if "python" not in lower:
        return None
    if any(word in lower for word in ("preparing", "checking", "installing", "updating", "creating")):
        return True
    if any(word in lower for word in ("is up-to-date", "up to date", "updated", "installed", "ready")):
        return False
    return None


def staged_frames_size(workdir):
    """Total bytes of the staged frames - roughly what Siril's convert copies
    when it can't symlink them. Siril's own process/masters output is skipped."""
    total = 0
    try:
        for entry in os.scandir(workdir):
            if not entry.is_dir() or entry.name in ("process", "masters"):
                continue
            for sub in os.scandir(entry.path):
                if sub.is_file():
                    total += sub.stat().st_size
    except OSError as e:
        logger.debug(f"Could not size staged frames in {workdir}: {e}")
    return total


def find_siril_results(frames_dir):
    """result_<filter>.fit(s) files the Siril script saved."""
    if not os.path.isdir(frames_dir):
        return []
    return sorted(
        os.path.join(frames_dir, name) for name in os.listdir(frames_dir)
        if name.lower().startswith("result_") and os.path.splitext(name)[1].lower() in (".fit", ".fits", ".fts")
    )
