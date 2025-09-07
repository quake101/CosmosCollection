# CosmosCollection Installation Guide

## Python Requirements

CosmosCollection requires Python 3.8 or higher. It has been tested with Python 3.12.

## Installation Options

### Option 1: Production Installation (Recommended for Users)

Install only the packages needed to run the application:

```bash
pip install -r requirements-prod.txt
```

### Option 2: Full Installation (For Developers)

Install all packages including development and build tools:

```bash
pip install -r requirements.txt
```

### Option 3: Development Installation

Install production packages plus development tools:

```bash
pip install -r requirements-dev.txt
```

## Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements-prod.txt
```

## Key Dependencies

- **PySide6** (6.9.0): Qt-based GUI framework
- **astropy** (7.0.2): Core astronomy library for coordinates and calculations
- **astroplan** (0.10.1): Observation planning tools
- **numpy** (2.2.5): Numerical computing
- **matplotlib** (3.10.3): Plotting and visualization
- **pytz** (2025.2): Timezone handling

## Building Executable

If you want to build a standalone executable:

1. Install development requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run the build script (if available) or use PyInstaller directly:
   ```bash
   pyinstaller CosmosCollection.spec
   ```

## Troubleshooting

### Common Issues

1. **PySide6 Installation Issues**: Make sure you have the latest pip version:
   ```bash
   pip install --upgrade pip
   pip install PySide6
   ```

2. **Astropy Installation Issues**: Some systems may need additional development tools. On Windows, ensure you have Microsoft C++ Build Tools installed.

3. **Import Errors**: Make sure your virtual environment is activated and all requirements are installed.

### Verify Installation

Run this command to verify all key packages are properly installed:

```bash
python -c "import PySide6, astropy, astroplan, numpy, matplotlib, pytz; print('All packages imported successfully!')"
```