# PyInstaller hook for astroquery
# This hook helps PyInstaller properly bundle astroquery and its dependencies

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all astroquery submodules
hiddenimports = collect_submodules('astroquery')

# Collect all data files, binaries, and submodules
datas, binaries, additional_hiddenimports = collect_all('astroquery')

# Add critical astropy dependencies that astroquery needs
hiddenimports += [
    'astropy.io.ascii',
    'astropy.io.votable',
    'astropy.table',
    'astropy.units',
    'astropy.coordinates',
    'astropy.time',
    'astropy.config',
    'astropy.utils',
    'astropy.utils.data',
    'astropy.utils.iers',
]

# Add network-related dependencies
hiddenimports += [
    'requests',
    'urllib3',
    'certifi',
    'html.parser',
    'keyring',
]

# Merge with additional hidden imports found by collect_all
hiddenimports += additional_hiddenimports