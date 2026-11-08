"""Command line tests: every command runs, and none loads the app's GUI.

    python -m unittest discover tests -v

Runs against a copy of your database (or COSMOS_TEST_DATA_DIR: a folder with a
DSO.sqlite you don't mind being changed), never the real one - the write
commands' tests add and remove rows. Doesn't touch the app's settings.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO, "main.py")

# Never imported by a command: the command line must stay lightweight
FORBIDDEN = ("PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
             "matplotlib")
_IMPORT_LINE = re.compile(r"^import time:\s+\d+ \|\s+\d+ \|\s*(\S+)")


def _user_database():
    from cosmoscollection.core.ResourceManager import ResourceManager
    saved = os.environ.pop("COSMOS_DATA_DIR", None)
    try:
        return str(ResourceManager.get_data_dir() / "DSO.sqlite")
    finally:
        if saved is not None:
            os.environ["COSMOS_DATA_DIR"] = saved


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = None
        cls.data_dir = os.environ.get("COSMOS_TEST_DATA_DIR")
        if not cls.data_dir:
            sys.path.insert(0, REPO)
            cls._temp = tempfile.mkdtemp(prefix="cosmos_cli_test_")
            source = _user_database()
            if not os.path.exists(source):
                source = os.path.join(REPO, "catalogs", "DSO.sqlite")  # A fresh install's database
            shutil.copy2(source, os.path.join(cls._temp, "DSO.sqlite"))
            cls.data_dir = cls._temp

    @classmethod
    def tearDownClass(cls):
        if cls._temp:
            shutil.rmtree(cls._temp, ignore_errors=True)

    def run_cli(self, *args, expect=0):
        """(stdout, seconds) of a command, checking its exit code and imports."""
        env = dict(os.environ, COSMOS_DATA_DIR=self.data_dir, PYTHONIOENCODING="utf-8")
        started = time.perf_counter()
        result = subprocess.run([sys.executable, "-X", "importtime", MAIN, *args], cwd=REPO, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        seconds = time.perf_counter() - started
        stderr_lines = result.stderr.splitlines()
        imported = {m.group(1) for line in stderr_lines if (m := _IMPORT_LINE.match(line))}
        messages = "\n".join(line for line in stderr_lines if not _IMPORT_LINE.match(line))
        self.assertEqual(result.returncode, expect, f"{args}: exit {result.returncode}\n{messages}")
        loaded = sorted(m for m in imported if m.startswith(FORBIDDEN))
        self.assertEqual(loaded, [], f"{args} loaded GUI modules: {loaded}")
        return result.stdout, seconds

    def json_of(self, *args):
        return json.loads(self.run_cli("--json", *args)[0])

    # ---- startup

    def test_help_is_quick(self):
        out, seconds = self.run_cli("--help")
        self.assertIn("dso", out)
        print(f"\n  --help: {seconds:.2f} s", end="")
        self.assertLess(seconds, 5)

    def test_group_without_action_shows_help(self):
        out, _ = self.run_cli("dso", expect=2)
        self.assertIn("search", out)

    # ---- queries

    def test_dso_search(self):
        out, seconds = self.run_cli("dso", "search", "M31")
        self.assertIn("M 31", out)
        print(f"\n  dso search: {seconds:.2f} s", end="")

    def test_dso_search_by_common_name(self):
        names = [d["name"] for d in self.json_of("dso", "search", "Orion Nebula")]
        self.assertIn("M 42", names)

    def test_dso_show(self):
        data = self.json_of("dso", "show", "m31")
        self.assertEqual(data["name"], "M 31")
        self.assertEqual(data["common_name"], "Andromeda Galaxy")
        self.assertIn("sessions", data)

    def test_dso_show_unknown(self):
        self.run_cli("dso", "show", "no such object", expect=1)

    def test_dso_list_filters(self):
        data = self.json_of("dso", "list", "--catalog", "m", "--type", "galaxy")
        self.assertTrue(data)
        self.assertTrue(all(d["type"] == "GALXY" for d in data))

    def test_sessions(self):
        sessions = self.json_of("sessions", "list")
        if sessions:
            detail = self.json_of("sessions", "show", str(sessions[0]["id"]))
            self.assertEqual(detail["id"], sessions[0]["id"])

    def test_images_targets_stats_sites(self):
        for command in (["images", "list"], ["targets", "list"], ["stats"], ["sites"]):
            self.json_of(*command)

    def test_legacy_options(self):
        out, _ = self.run_cli("--search-dso", "M31")
        self.assertIn("M 31", out)

    # ---- changes (on the copy)

    def test_images_add_with_favorite(self):
        image = os.path.join(REPO, "images", "CosmosCollection.png")
        added = self.json_of("images", "add", "M 33", image, "--integration", "3600", "--favorite", "--force")
        self.assertEqual(added["integration"], "1h")
        images = self.json_of("images", "list", "--dso", "M33")
        self.assertEqual([i["id"] for i in images if i["favorite"]], [added["id"]])

    def test_targets_add_and_remove(self):
        before = {t["name"] for t in self.json_of("targets", "list")}
        added = self.json_of("targets", "add", "NGC 891", "--force")
        self.run_cli("targets", "remove", str(added["id"]), "--no-input", expect=2)  # Asks first
        self.run_cli("targets", "remove", str(added["id"]), "--yes")
        self.assertEqual({t["name"] for t in self.json_of("targets", "list")}, before)


def _write_light(path, when, filter_name="L-eXtreme"):
    """A tiny FITS light frame with the headers a capture program writes."""
    import numpy as np
    from astropy.io import fits
    header = fits.Header()
    for key, value in (("IMAGETYP", "LIGHT"), ("EXPTIME", 300.0), ("DATE-OBS", when), ("OBJECT", "M 31"),
                       ("FILTER", filter_name), ("INSTRUME", "Test Camera 2600"), ("TELESCOP", "Test Scope 61"),
                       ("FOCALLEN", 360.0), ("XPIXSZ", 3.76), ("XBINNING", 1), ("GAIN", 100),
                       ("RA", 10.6847), ("DEC", 41.2687)):
        header[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.uint16), header).writeto(path)


class LibraryCommandTests(unittest.TestCase):
    """library scan / sessions check on a small made-up library, imported into a
    new collection (the database a fresh install starts with)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.mkdtemp(prefix="cosmos_cli_library_")
        cls.data_dir = os.path.join(cls.temp, "data")
        os.makedirs(cls.data_dir)
        shutil.copy2(os.path.join(REPO, "catalogs", "DSO.sqlite"), os.path.join(cls.data_dir, "DSO.sqlite"))
        cls.library = os.path.join(cls.temp, "Astro")
        cls.night = os.path.join(cls.library, "M 31", "2026-09-01", "LIGHT")
        for n, minute in enumerate(("10", "16", "22")):
            _write_light(os.path.join(cls.night, f"M31_Light_{n + 1:03d}.fits"), f"2026-09-02T03:{minute}:00.000")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    run_cli = CommandLineTests.run_cli
    json_of = CommandLineTests.json_of

    def test_scan_import_and_check(self):
        scan = self.json_of("library", "scan", self.library, "--no-mounts")
        self.assertEqual([(s["object"], s["subs"], s["to_import"]) for s in scan["sessions"]], [("M 31", 3, True)])
        self.assertEqual(len(scan["unmatched_equipment"]), 2)
        self.assertIsNone(scan["imported"])

        # Unmatched equipment and no terminal: it won't guess
        self.run_cli("library", "scan", self.library, "--no-mounts", "--import", expect=2)
        imported = self.json_of("library", "scan", self.library, "--no-mounts", "--import", "--equipment", "keep",
                                "--yes")["imported"]
        self.assertEqual((imported["sessions_created"], imported["subs_added"]), (1, 3))
        again = self.json_of("library", "scan", self.library, "--no-mounts", "--import", "--yes")["imported"]
        self.assertTrue(again.get("nothing_to_import"))

        sessions = self.json_of("sessions", "list")
        self.assertEqual([(s["sub_count"], s["integration_seconds"]) for s in sessions], [(3, 900.0)])
        session_id = str(sessions[0]["id"])
        self.assertEqual(self.json_of("sessions", "check")["sessions"], [])

        # The night folder renamed: found again and relinked
        renamed = os.path.join(os.path.dirname(self.night), "LIGHTS")
        os.rename(self.night, renamed)
        problems = self.json_of("sessions", "check", session_id)["sessions"][0]["problems"]
        self.assertEqual([(p["kind"], p["files"]) for p in problems], [("moved", 3)])
        self.assertEqual(self.json_of("sessions", "check", "--relink")["relinked"], 3)
        self.assertEqual(self.json_of("sessions", "check")["sessions"], [])

        # A sub deleted: removed from the session only when confirmed
        os.remove(os.path.join(renamed, "M31_Light_003.fits"))
        self.run_cli("sessions", "check", "--remove-missing", "--no-input", expect=2)
        self.assertEqual(self.json_of("sessions", "check", "--remove-missing", "--yes")["removed"], 1)
        self.assertEqual(self.json_of("sessions", "show", session_id)["sub_count"], 2)


def _write_star_field(path, when, sigma, seed=7, filter_name="L-eXtreme"):
    """A small FITS light frame of the same stars (blurred by sigma, pixels),
    with sky noise - a sharp frame, or a bloated one (bad seeing or focus)."""
    import numpy as np
    from astropy.io import fits
    stars = np.random.default_rng(seed)  # The same stars in every frame
    count, size = 160, 512
    xs, ys = stars.uniform(40, size - 40, count), stars.uniform(40, size - 40, count)
    fluxes = stars.uniform(3000, 60000, count)
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.full((size, size), 1000.0)
    for x, y, flux in zip(xs, ys, fluxes):
        box = (slice(max(int(y) - 20, 0), int(y) + 21), slice(max(int(x) - 20, 0), int(x) + 21))
        image[box] += flux / (2 * np.pi * sigma ** 2) * np.exp(
            -((xx[box] - x) ** 2 + (yy[box] - y) ** 2) / (2 * sigma ** 2))
    noise = np.random.default_rng(abs(hash(path)) % 2 ** 32)
    image = noise.poisson(image).astype(np.float64)
    header = fits.Header()
    for key, value in (("IMAGETYP", "LIGHT"), ("EXPTIME", 120.0), ("DATE-OBS", when), ("OBJECT", "M 31"),
                       ("FILTER", filter_name), ("INSTRUME", "Test Camera 2600"), ("TELESCOP", "Test Scope 61"),
                       ("FOCALLEN", 360.0), ("XPIXSZ", 3.76), ("XBINNING", 1), ("GAIN", 100),
                       ("RA", 10.6847), ("DEC", 41.2687)):
        header[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.PrimaryHDU(np.clip(image, 0, 65535).astype(np.uint16), header).writeto(path)


class ImagingCommandTests(unittest.TestCase):
    """quality and solve on made-up frames: nine sharp ones and three bloated
    ones in one night, imported as a session into a new collection."""

    BLOATED = (4, 8, 11)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.mkdtemp(prefix="cosmos_cli_imaging_")
        cls.data_dir = os.path.join(cls.temp, "data")
        os.makedirs(cls.data_dir)
        shutil.copy2(os.path.join(REPO, "catalogs", "DSO.sqlite"), os.path.join(cls.data_dir, "DSO.sqlite"))
        cls.night = os.path.join(cls.temp, "Astro", "M 31", "2026-09-01")
        for n in range(12):
            _write_star_field(os.path.join(cls.night, f"M31_Light_{n + 1:03d}.fits"),
                              f"2026-09-02T03:{10 + 3 * n:02d}:00.000", 3.2 if n in cls.BLOATED else 1.3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    run_cli = CommandLineTests.run_cli
    json_of = CommandLineTests.json_of

    def bloated_names(self):
        return {f"M31_Light_{n + 1:03d}.fits" for n in self.BLOATED}

    def test_quality_folder_then_session_move(self):
        # Graded straight from the folder
        graded = self.json_of("quality", self.night)
        self.assertEqual(graded["lights"], 12)
        self.assertEqual(graded["analyzed"], 12)
        not_good = {os.path.basename(f["path"]) for f in graded["frames"] if f["grade"] != "Good"}
        self.assertEqual(not_good, self.bloated_names())
        self.assertEqual(graded["selected"]["frames"], 3)

        # Imported as a session: the measurements are reused, the grades are the same
        self.json_of("library", "scan", os.path.dirname(os.path.dirname(self.night)), "--no-mounts", "--import",
                     "--equipment", "keep", "--yes")
        session_id = str(self.json_of("sessions", "list")[0]["id"])
        graded = self.json_of("quality", session_id)
        self.assertEqual((graded["analyzed"], graded["from_saved"]), (0, 12))
        self.assertEqual({os.path.basename(f["path"]) for f in graded["frames"] if f["grade"] == "Reject"},
                         self.bloated_names())

        # Moving the Rejects asks first, then takes them out of the session
        self.run_cli("quality", session_id, "--move", "--no-input", expect=2)
        moved = self.json_of("quality", session_id, "--move", "--yes")["action"]
        self.assertEqual(moved["done"], 3)
        rejected = os.path.join(self.night, "Rejected")
        self.assertEqual(set(os.listdir(rejected)), self.bloated_names())
        self.assertEqual(self.json_of("sessions", "show", session_id)["sub_count"], 9)
        self.assertEqual(self.json_of("quality", session_id)["lights"], 9)

    def test_solve_without_a_position(self):
        self.run_cli("solve", os.path.join(self.temp, "no such image.fits"), expect=1)
        image = os.path.join(self.temp, "blank", "blank.fits")
        _write_light(image, "2026-09-02T03:00:00.000")
        result = self.json_of_failing("solve", image, "--solver", "builtin", "--ra", "10.68", "--dec", "41.27")
        self.assertFalse(result[0]["solved"])
        self.assertIn("stars", result[0]["error"])

    def json_of_failing(self, *args):
        return json.loads(self.run_cli("--json", *args, expect=1)[0])


class PlanningCommandTests(unittest.TestCase):
    """tonight, visibility and weather for made-up locations in a new
    collection: one at 45° N, and one walled in by a custom horizon."""

    NIGHT = "2026-09-15"

    @classmethod
    def setUpClass(cls):
        import sqlite3
        cls.temp = tempfile.mkdtemp(prefix="cosmos_cli_planning_")
        shutil.copy2(os.path.join(REPO, "catalogs", "DSO.sqlite"), os.path.join(cls.temp, "DSO.sqlite"))
        cls.data_dir = cls.temp
        # Any command sets the database up (the location table's horizon columns)
        subprocess.run([sys.executable, MAIN, "sites"], cwd=REPO, env=dict(os.environ, COSMOS_DATA_DIR=cls.temp),
                       capture_output=True, timeout=120, check=True)
        conn = sqlite3.connect(os.path.join(cls.temp, "DSO.sqlite"))
        with conn:
            conn.execute("UPDATE usersettings SET is_active = 0")
            conn.execute("INSERT INTO usersettings (location_name, location_lat, location_lon, timezone, is_active) "
                         "VALUES ('Test Field', 45.0, -75.0, 'America/Toronto', 1)")
            walls = json.dumps([[az, 89.0] for az in range(0, 360, 30)])
            conn.execute("INSERT INTO usersettings (location_name, location_lat, location_lon, timezone, is_active, "
                         "horizon_points, horizon_name) VALUES ('Walled In', 45.0, -75.0, 'America/Toronto', 0, ?, "
                         "'walls.hrz')", (walls,))
        conn.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    def run_cli(self, *args, expect=0):
        return CommandLineTests.run_cli(self, *args, expect=expect)

    def json_of(self, *args):
        return json.loads(self.run_cli("--json", *args)[0])

    def test_tonight(self):
        data = self.json_of("tonight", "--date", self.NIGHT, "--catalog", "M", "--limit", "0")
        self.assertEqual(data["site"]["name"], "Test Field")
        self.assertTrue(data["time"]["start"].startswith(self.NIGHT))
        names = [o["name"] for o in data["objects"]]
        self.assertIn("M 31", names)
        self.assertNotIn("M 7", names)  # Too far south from 45° N in September
        self.assertTrue(all(o["max_altitude"] >= 30 for o in data["objects"]))
        hours = [o["visible_hours"] for o in data["objects"]]
        self.assertEqual(hours, sorted(hours, reverse=True))  # Sorted as the window sorts

        # A custom horizon of 89° all round hides everything - and says so
        walled = self.json_of("--site", "Walled In", "tonight", "--date", self.NIGHT, "--catalog", "M")
        self.assertEqual((walled["found"], walled["custom_horizon"]), (0, "walls.hrz"))
        self.assertEqual(walled["blocked_by_horizon"], data["found"])

    def test_tonight_time_frame_and_bad_time(self):
        data = self.json_of("tonight", "--date", self.NIGHT, "--catalog", "M", "--time", "01:00-02:00")
        self.assertTrue(data["time"]["whole_window"])
        self.assertTrue(data["time"]["start"].startswith("2026-09-16T01:00"))
        self.run_cli("tonight", "--time", "25:00", expect=2)

    def test_visibility(self):
        data = self.json_of("visibility", "M31", "--date", self.NIGHT, "--hourly")
        self.assertEqual(data["object"], "M 31 (Andromeda Galaxy)")
        self.assertAlmostEqual(data["transit"]["altitude"], 90 - (45 - 41.27), delta=0.5)
        self.assertTrue(data["windows"])
        self.assertLess(data["sun"]["sunset"], data["sun"]["sunrise"])
        self.assertEqual(len(data["hourly"]), 25)

        # Behind the walled-in location's custom horizon all night (unless it's ignored)
        walled = self.json_of("--site", "Walled In", "visibility", "M31", "--date", self.NIGHT)
        self.assertEqual((walled["custom_horizon"], walled["windows"]), ("walls.hrz", []))
        ignored = self.json_of("--site", "Walled In", "visibility", "M31", "--date", self.NIGHT, "--no-horizon")
        self.assertEqual(ignored["windows"], data["windows"])

        # Never above the horizon from 45° N
        south = self.json_of("visibility", "--ra", "120", "--dec", "-80", "--date", self.NIGHT)
        self.assertTrue(south["never_up"])
        self.assertEqual(south["windows"], [])

    def test_weather_checks_its_options_first(self):
        self.run_cli("weather", "--days", "0", expect=2)


if __name__ == "__main__":
    unittest.main()
