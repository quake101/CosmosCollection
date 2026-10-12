# Command Line Interface

Cosmos Collection supports command line operations for scripting and automation.

## Running It

**Release:** use the `CosmosCollection-CLI` executable included in the release.
```bash
# Windows
CosmosCollection-CLI.exe --list-dsos

# Linux
./CosmosCollection-CLI --list-dsos

# macOS (bundled inside the .app)
./CosmosCollection.app/Contents/MacOS/CosmosCollection-CLI --list-dsos
```

**From source:** use `python main.py` with the same options.
```bash
python main.py --list-dsos
```

The examples below use `CosmosCollection-CLI`; from source, replace it with `python main.py`.

## Add an Image

```bash
# Basic
CosmosCollection-CLI --add-image --dso "M31" --image "/path/to/image.jpg"

# With full metadata
CosmosCollection-CLI --add-image --dso "NGC 7000" --image "/path/to/image.tif" \
    --equipment "Telescope: AT72ED, Camera: ASI2600" \
    --integration "3600" \
    --date "2024-01-15" \
    --notes "First light!" \
    --set-favorite
```

| Option | Description |
|--------|-------------|
| `--dso NAME` | DSO name (required), e.g. M31, NGC 7000, IC 1396 |
| `--image PATH` | Path to the image file (required) |
| `--equipment TEXT` | Equipment used (optional) |
| `--integration SECONDS` | Integration time in seconds (optional) |
| `--date YYYY-MM-DD` | Date the image was taken (optional) |
| `--notes TEXT` | Notes about the image (optional) |
| `--set-favorite` | Set this image as the favorite for the DSO |

## Search for DSOs

```bash
CosmosCollection-CLI --search-dso "M31"
CosmosCollection-CLI --search-dso "Orion"
CosmosCollection-CLI --search-dso "NGC 70"
```

## List All DSOs

```bash
CosmosCollection-CLI --list-dsos
```

## Help

```bash
CosmosCollection-CLI --help
```
