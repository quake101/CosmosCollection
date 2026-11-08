# Building from Source

## Prerequisites

- Python 3.x (3.12 recommended)
- The dependencies in `requirements.txt`
- **Linux only**: the system Qt libraries listed in the [README](../README.md#getting-started)

## Install and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/quake101/CosmosCollection.git
   cd CosmosCollection
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Project Layout

```
main.py                 Launcher: the command line (cli/) when given a command, else
                        app/Bootstrap.py, then app/Startup.py
CosmosCollection.spec   PyInstaller build for the app
build_tools/            Build-time scripts (release version stamping, Windows version info)
catalogs/  images/  web/   Data shipped with the app (DSO catalog, icons, Aladin Lite)
updater/                The separate updater program and its PyInstaller spec
tests/                  Command line tests (python -m unittest discover tests)
cosmoscollection/       The app's code
  cli/        The command line (docs/CLI.md): no GUI modules, only PySide6.QtCore for settings
  app/        Startup, main window, Settings, About/update dialogs, tray, the old CLI options
  ui/         Theme, window position/lifetime helpers, shared dialogs (map location picker)
  core/       Database, resource paths, network limits/offline mode, diagnostics, app updater
  catalog/    DSO names, SIMBAD lookups, the DSO detail window
  planning/   Best DSO Tonight, Visibility Calculator, Target List, Weather, horizon, FOV Simulator
  sessions/   Session Manager, library scan/import, session files and observations, mount detection
  imaging/    Gallery, image viewer/details/export, collages, annotations, image loading, frame quality
  solving/    Plate solvers (built-in and external) and the offline star tile cache
  equipment/  Equipment Manager and equipment name matching
  nina/       NINA integration and dashboard
```

Imports are absolute, e.g. `from cosmoscollection.core.DatabaseManager import DatabaseManager`. `app/Bootstrap.py` must be imported before anything that loads numpy or Qt, so keep the package `__init__.py` files empty.

The command line imports logic modules directly, so keep work that a command could use out of modules that import `QtGui`, `QtWidgets` or matplotlib, and import numpy and other heavy packages inside the functions that need them. `tests/test_cli.py` fails if a command loads a GUI module. The `COSMOS_DATA_DIR` environment variable points the app and the command line at another data folder (database and logs), for testing on a copy.

A few modules also run on their own, for testing. Run them as modules from the repo root:

```bash
python -m cosmoscollection.solving.NativePlateSolver --help
```

The same works for `imaging.FrameQuality`, `sessions.LibraryScanner` and `sessions.MountDetection`.

## Building the Packaged App

Releases are built by GitHub Actions (`.github/workflows/build-release.yml`). To build locally:

```bash
pyinstaller CosmosCollection.spec
pyinstaller updater/CosmosUpdater.spec
```

## CachyOS / Arch Linux: Python Version

Rolling-release distros like CachyOS often ship the newest Python (e.g. 3.14) before pinned dependencies such as `PySide6==6.9.0` and `numpy==2.2.5` publish wheels for it. Then `pip install -r requirements.txt` fails with "No matching distribution found" or a build error.

Use [uv](https://github.com/astral-sh/uv) to create the virtual environment with a compatible Python (3.12):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
