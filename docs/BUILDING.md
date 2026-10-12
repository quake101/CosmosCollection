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

## CachyOS / Arch Linux: Python Version

Rolling-release distros like CachyOS often ship the newest Python (e.g. 3.14) before pinned dependencies such as `PySide6==6.9.0` and `numpy==2.2.5` publish wheels for it. Then `pip install -r requirements.txt` fails with "No matching distribution found" or a build error.

Use [uv](https://github.com/astral-sh/uv) to create the virtual environment with a compatible Python (3.12):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
