# Command Line Interface

Cosmos Collection has a command line for working with your collection from a terminal or a script. It uses the same database and settings as the app, so anything it adds shows up in the app, and the other way round.

It starts quickly because it doesn't load any of the app's windows.

## Running It

**Release:** use the `CosmosCollection-CLI` executable included in the release.
```bash
# Windows
CosmosCollection-CLI.exe dso search M31

# Linux
./CosmosCollection-CLI dso search M31

# macOS (bundled inside the .app)
./CosmosCollection.app/Contents/MacOS/CosmosCollection-CLI dso search M31
```

**From source:** use `python main.py` (or `python -m cosmoscollection.cli`) with the same commands.
```bash
python main.py dso search M31
```

The examples below use `CosmosCollection-CLI`. From source, replace it with `python main.py`.

```bash
CosmosCollection-CLI --help              # all commands
CosmosCollection-CLI dso --help          # one command's actions
CosmosCollection-CLI dso search --help   # one action's options
```

## Objects

```bash
# Search the way the main window's search box does: designations, common names, part of an RA/Dec
CosmosCollection-CLI dso search M31
CosmosCollection-CLI dso search "Orion Nebula"
CosmosCollection-CLI dso search 00h42m

# List objects, with filters
CosmosCollection-CLI dso list --catalog M --type galaxy --sort magnitude
CosmosCollection-CLI dso list --constellation Cygnus --imaged
CosmosCollection-CLI dso list --catalog Sh2 --not-imaged

# One object's details, with your sessions, gallery images and Target List entry
CosmosCollection-CLI dso show "NGC 7000"
```

| Option | Description |
|--------|-------------|
| `--catalog NAME` | Only this catalog's objects (M, NGC, IC, Sh2, ...) |
| `--type TYPE` | Only this type: a code (`GALXY`) or name (`galaxy`, `"open cluster"`) |
| `--constellation NAME` | Only this constellation: abbreviation (`ORI`) or name (`Orion`) |
| `--imaged` / `--not-imaged` | Only objects with (or without) images in your gallery (`dso list`) |
| `--sort` | `name` (default), `magnitude`, `images`, `ra` or `dec` (`dso list`) |
| `--limit N` | Show at most N objects (`dso search` shows 50 by default; 0 = all) |

## Gallery Images

```bash
CosmosCollection-CLI images list
CosmosCollection-CLI images list --dso M31

# Add an image to an object's gallery
CosmosCollection-CLI images add M31 "/path/to/image.jpg"

# With details
CosmosCollection-CLI images add "NGC 7000" "/path/to/image.tif" \
    --equipment "Telescope: AT72ED, Camera: ASI2600" \
    --integration 3600 \
    --date 2024-01-15 \
    --notes "First light!" \
    --favorite
```

| Option | Description |
|--------|-------------|
| `--equipment TEXT` | Telescope and camera, linked to your Equipment where they match |
| `--integration TIME` | Integration time: seconds (`3600`, shown as "1h") or as the app writes it (`"5h 6m"`) |
| `--date YYYY-MM-DD` | Date the image was taken |
| `--notes TEXT` | Notes about the image |
| `--favorite` | Make it the object's favorite image |
| `--force` | Add it even if the object already has this file |

## Sessions

```bash
CosmosCollection-CLI sessions list
CosmosCollection-CLI sessions list --dso M31
CosmosCollection-CLI sessions list --since 2026-01-01 --telescope "ZenithStar"

# One session's nights, totals per filter and linked images
CosmosCollection-CLI sessions show 37
```

`sessions list` filters: `--dso NAME`, `--status STATUS`, `--since YYYY-MM-DD`, `--until YYYY-MM-DD`, `--telescope TEXT`, `--camera TEXT`, `--limit N`.

## Importing Your Library

`library scan` does what the Import Wizard does. It finds the subs in your folders, groups them into sessions by object and equipment, and pairs finished images with their sessions. On its own it only lists what's new. Add `--import` to import it with the wizard's default choices. Your files are never moved or changed; only references to them are added.

```bash
# What's new in the Import Wizard's saved folders
CosmosCollection-CLI library scan

# What's new in these folders
CosmosCollection-CLI library scan "D:/Astro/M 31" "D:/Astro/NGC 7000"

# Import it (asks first; --yes for a script)
CosmosCollection-CLI library scan "D:/Astro" --import

# The folders the Import Wizard scans, and when each was last scanned
CosmosCollection-CLI library folders
```

| Option | Description |
|--------|-------------|
| `--import` | Import the new sessions and finished images |
| `--sessions-only` / `--images-only` | Import only sessions, or only finished images |
| `--equipment CHOICE` | For telescope and camera names in your files that aren't matched to your Equipment yet: `ask` (default), `suggested` (use the suggested match, else keep the name), `add` (add new equipment when there's no match) or `keep` (keep the names unmatched) |
| `--all` | List everything found, not just what's new |
| `--full` | Read every file's header again, ignoring what earlier scans found |
| `--no-mounts` | Don't measure each night's mount (faster) |
| `--gap-days DAYS` | Days without imaging that start a new session (default: the wizard's setting) |
| `--report FILE` | Also write a detailed report of the scan to FILE |

Your choices for equipment names are saved, so each name is only asked about once. The wizard and the command line share them.

Objects the scan can't identify from the file headers or folder names aren't imported. To bring those in, pick the object in the Import Wizard.

Like the wizard, a scan also updates sessions you've already imported. Subs whose folder was renamed or moved are pointed at their new place, and nights get their measured mount.

## Checking Session Files

`sessions check` does what the Session File Check does. It looks for attached files that are no longer on disk, and searches for missing ones near where they were (a renamed or moved folder). It also finds calibration frames that aren't on any night. On its own it only lists problems. The options below fix them.

```bash
CosmosCollection-CLI sessions check            # every session
CosmosCollection-CLI sessions check 37 40      # these sessions

# Point moved files at their new folder
CosmosCollection-CLI sessions check --relink

# Look for missing files in another folder too
CosmosCollection-CLI sessions check --find-in "E:/Astro backup" --relink

# Remove files that are gone from their sessions (asks first)
CosmosCollection-CLI sessions check --remove-missing
```

| Option | Description |
|--------|-------------|
| `--relink` | Point moved files at the folder they were found in |
| `--find-in FOLDER` | Also look for missing files anywhere in FOLDER |
| `--remove-missing` | Remove files that aren't on disk from their sessions |
| `--remove-no-night` | Remove calibration frames that aren't on any night |

Files on a drive that isn't connected (an unplugged disk or offline NAS) are reported but never removed. The files on disk are never changed.

## Grading Light Frames

`quality` does what the Light Quality review does. It measures each light's stars, sky and trails, then grades it against the other frames with the same filter, exposure and binning on its night. Each night is also checked against a typical one. It uses the review's thresholds and the app's Maximum Threads setting. Measurements are saved, so only new or changed frames are measured again, and the app's review reuses them too.

```bash
# A session's lights (the id from 'sessions list')
CosmosCollection-CLI quality 37

# Subs in folders, before importing them (subfolders too, except 'Rejected' ones)
CosmosCollection-CLI quality "D:/Astro/M 31/2026-09-01"

# Every frame, not just those that aren't Good
CosmosCollection-CLI quality 37 --all

# Move the Rejects into a 'Rejected' folder beside each one, and out of the session (asks first)
CosmosCollection-CLI quality 37 --move

# Keep the best 80% of each filter; take the rest out of the session (the files stay)
CosmosCollection-CLI quality 37 --keep-top 80 --remove
```

| Option | Description |
|--------|-------------|
| `--night YYYY-MM-DD` | Only this night's frames |
| `--filter NAME` | Only this filter's frames |
| `--all` | List every frame |
| `--reanalyze` | Measure every frame again |
| `--workers N` | Frames measured at once (default: Settings > Maximum Threads) |
| `--remove` | Remove the frames from the session. The files stay where they are |
| `--move [FOLDER]` | Move the frames into FOLDER, or a 'Rejected' folder beside each one, and remove them from the session |
| `--delete` | Delete the files permanently (not to the Recycle Bin) and remove them from the session |
| `--include-marginal` | Act on Marginal frames too |
| `--keep-top [PCT]` | Act on every frame outside the best PCT% of each filter (default: the percentage last used in the app) |

`--remove`, `--move` and `--delete` act on the Reject frames, limited by `--night` and `--filter`. They ask first; pass `--yes` in a script. Like the review, when a night loses all its lights, it asks whether to remove that night's calibration frames too.

A file attached to other sessions stays attached to them at its new place when moved. When deleted, it's removed from them too.

With folders, `--move` and `--delete` work too, and sessions those files belong to are kept up to date. Nights are dated in your observing location's time zone.

## Plate Solving

`solve` finds where images are in the sky, as the image viewer's plate solve does. It uses the solvers set up in the app (Settings > Plate Solving), in the same order. Each solution is saved next to its image as a `.wcs` file, where every window of the app finds it.

```bash
CosmosCollection-CLI solve "D:/Astro/M 31/M31_stack.fits"

# Every image in a folder (not its subfolders)
CosmosCollection-CLI solve "D:/Astro/Finished"

# Tell the solvers where to look, for an image that isn't in your gallery and has no RA/Dec in its header
CosmosCollection-CLI solve photo.jpg --object M44
```

| Option | Description |
|--------|-------------|
| `--object NAME` | The object the images show, to search near |
| `--ra DEG --dec DEG` | The position to search near, in degrees |
| `--solver NAME` | `builtin`, `astap` or `astrometry` (repeat for several, tried in the order given) |
| `--fresh` | Solve again even if an image already has a solution, and replace it |

Without `--object` or `--ra`/`--dec`, the solvers search near the gallery image's object, or the RA/Dec in the file's header. With none of those, only ASTAP and astrometry.net can solve it (searching the whole sky). Images that already have a solution are listed as such without being solved again.

## Planning a Night

These use your active observing location (or `--site NAME`) and its custom horizon, as the app's planning tools do. Times are shown in the location's time zone, as 12- or 24-hour times per Settings. "Tonight" means the current night until dawn, then the coming one.

### Best Objects Tonight

`tonight` does what Best DSO Tonight does. It lists the objects that are above the minimum altitude and your custom horizon while the sun is below -12°. It checks the brightest catalog objects, or your Target List with `--targets`, and sorts them by how many hours each is up.

```bash
CosmosCollection-CLI tonight
CosmosCollection-CLI tonight --type galaxy --min-alt 40

# Up for all of 1 to 3 AM
CosmosCollection-CLI tonight --time 01:00-03:00

# Your Target List on a night next month
CosmosCollection-CLI tonight --targets --hide-completed --date 2026-12-12
```

| Option | Description |
|--------|-------------|
| `--time WHEN` | `night` (dusk to dawn, the default), `dusk`, `midnight`, `dawn`, `now`, a time (`23:30`), or a time frame (`01:00-03:00`, objects up for all of it) |
| `--date YYYY-MM-DD` | The night starting on this date |
| `--min-alt DEG` | Minimum altitude (default 30) |
| `--max-mag MAG` | Faintest magnitude to check (default 12) |
| `--objects N` | How many of the brightest objects to check (default 200, the window's DSO Limit) |
| `--catalog NAME` / `--type TYPE` | Only this catalog's objects, or this type |
| `--targets` | Check your Target List instead of the catalog (`--hide-completed` leaves out Completed targets) |
| `--no-horizon` | Ignore the location's custom horizon |
| `--sort` | `hours` (default), `altitude`, `magnitude`, `time` (best time), `moon` (farthest from the Moon first) or `rank` |
| `--limit N` | Show at most N objects (default 30; 0 = all) |

Each object's best time is the middle of its time up (with `--time night`) or when it's highest. The Direction column shows where it is at the start and end of the time, and Moon is its distance from the Moon at its best time.

### One Object's Night

`visibility` does what the DSO Visibility Calculator does for one night: the sun's times, when the object rises, crosses the meridian (when a German equatorial mount flips) and sets, the times it's up in a dark sky, and the Moon.

```bash
CosmosCollection-CLI visibility M31
CosmosCollection-CLI visibility "Orion Nebula" --date 2026-12-15 --min-alt 40

# A position instead of a name, with its altitude on each hour
CosmosCollection-CLI visibility --ra 83.82 --dec -5.39 --hourly
```

| Option | Description |
|--------|-------------|
| `--date YYYY-MM-DD` | The night starting on this date |
| `--min-alt DEG` | Minimum altitude (default 30) |
| `--ra DEG --dec DEG` | A position instead of a name, in degrees |
| `--no-horizon` | Ignore the location's custom horizon |
| `--hourly` | Also list its altitude, direction, and the sun's and Moon's altitudes on each hour of the night |

The name can be a catalog object or one of your Target List entries.

### Weather

`weather` shows the Weather Forecast's 7 nights with their astro scores. It uses Open-Meteo plus the weather sources turned on in Settings, and the units set there. Each night's score, clouds and seeing are for its dark hours, from the evening into the next morning.

```bash
CosmosCollection-CLI weather

# Each hour of tonight, or of another night
CosmosCollection-CLI weather --hourly
CosmosCollection-CLI weather --hourly 2026-11-12
```

| Option | Description |
|--------|-------------|
| `--days N` | Nights to show (default 7) |
| `--hourly [YYYY-MM-DD]` | Each hour of one night (default: tonight): score, clouds, humidity, wind, chance of rain and temperature |

`weather` needs the internet; with offline mode on, it stops with exit code 3.

## Target List

```bash
CosmosCollection-CLI targets list
CosmosCollection-CLI targets list --priority High

CosmosCollection-CLI targets add "Orion Nebula" --priority High --telescope "ZenithStar 61"
CosmosCollection-CLI targets remove "M 42"
```

`targets add` names the entry the way the app does, e.g. "M 42 (Orion Nebula)". If the object is already on the list it stops and says so; `--force` adds it anyway. `targets remove` asks first. Pass `--yes` to skip the question, for example in a script. The target's sessions are kept but no longer linked to it.

## Totals and Locations

```bash
CosmosCollection-CLI stats     # objects imaged, sessions, nights, integration by year and filter
CosmosCollection-CLI sites     # your saved observing locations (* = active)
```

## For Scripts

These global options work before or after the command:

| Option | Description |
|--------|-------------|
| `--json` | Print the result as JSON |
| `-y`, `--yes` | Answer yes to confirmations |
| `--no-input` | Never ask a question; stop with an error instead |
| `-q`, `--quiet` | Only print the result and errors |
| `-v`, `--verbose` | Show progress details (`-vv` for debug logging) |
| `--site NAME` | Use this saved observing location instead of the active one |

```bash
# Total integration in hours
CosmosCollection-CLI --json stats | python -c "import json,sys; print(json.load(sys.stdin)['integration_seconds'] / 3600)"

# Tonight's astro score
CosmosCollection-CLI --json weather --days 1 | python -c "import json,sys; print(json.load(sys.stdin)['nights'][0]['astro_score'])"
```

The result goes to standard output. Notes and errors go to standard error, so a pipe gets only the result.

Without a terminal (a scheduled task, a NINA sequence, another program), the command line never waits for an answer. A command that would ask a question stops with exit code 2 unless `--yes` is given. For `library scan --import`, also pass `--equipment` if your files might name new equipment.

```bash
# Nightly: import what's new in the wizard's folders, then relink anything moved
CosmosCollection-CLI library scan --import --equipment suggested --yes
CosmosCollection-CLI sessions check --relink
```

Ctrl+C stops a long scan cleanly, and what was read so far is kept for the next scan.

**Exit codes:** `0` done · `1` error (e.g. object not found, or an image that didn't solve) · `2` usage (bad option, or a question with no one to answer it) · `3` needs the internet while offline mode is on

**Changes while the app is open:** the app shows what the command line added or removed after its window is refreshed or reopened.

## Old Options

The original options still work. They run the commands that replaced them:

| Old | New |
|-----|-----|
| `--list-dsos` | `dso list` |
| `--search-dso NAME` | `dso search NAME` |
| `--add-image --dso NAME --image PATH ...` | `images add NAME PATH ...` |
| `--set-favorite` | `--favorite` |
