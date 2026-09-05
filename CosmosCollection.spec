# -*- mode: python ; coding: utf-8 -*-
import certifi
import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get certifi certificate bundle path
cert_file = certifi.where()
cert_dir = os.path.dirname(cert_file)

# Collect astroquery data files (CITATION, configs, etc.)
astroquery_datas = collect_data_files('astroquery')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('catalogs', 'catalogs'),
        ('images', 'images'),
        (cert_file, 'certifi'),
    ] + astroquery_datas,
    hiddenimports=[
        # Astroquery - top-level import in main.py should auto-detect, but list key modules as backup
        'astroquery',
        'astroquery.simbad',
        'astroquery.query',
        # Astropy - top-level import in main.py should auto-detect, but list key modules as backup
        'astropy',
        'astropy.coordinates',
        'astropy.units',
        # Other dependencies
        'pyvo',
        'bs4',
        'certifi',
        'html.parser',
        'requests',
        'urllib3',
        'keyring',
        'BestDSOTonight',
        'DSOVisibilityCalculator',
        'DSOTargetList',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'scipy',
        'tkinter',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

if sys.platform.startswith('linux'):
    # These are Linux desktop/OS-integration libraries (GTK, NSS, cairo, fontconfig,
    # freetype, OpenSSL, D-Bus, CUPS, avahi, ...) that PyInstaller's dependency walker
    # bundles from whatever happens to be installed on the BUILD machine, because Qt
    # WebEngine's Chromium dlopens many of them optionally. Bundling a snapshot from one
    # machine (e.g. GitHub Actions' Ubuntu runner) and running it on another (e.g.
    # CachyOS) reproducibly crashed the WebEngine renderer process (map picker, FOV
    # Simulator - "renderer process terminated ... exit code: 1002", no import-time
    # error, only at first render) even though the actual Qt/Chromium binaries were
    # byte-identical between the working and broken builds. These libraries are
    # near-universally present on any Linux desktop capable of running a Qt GUI at all,
    # so excluding them here and letting normal dynamic linking find the target
    # machine's own copies is safer than freezing in whatever CI happened to have.
    # Confirmed fix: stripping exactly this set from a broken build made the crash go
    # away; the same build with them present crashed 5/5 runs.
    _EXCLUDED_SYSTEM_LIB_PREFIXES = (
        'libgtk-3', 'libgdk-3', 'libatk-1.0', 'libatk-bridge-2.0', 'libatspi',
        'libcairo', 'libpango', 'libnss3', 'libnssutil3', 'libnspr4', 'libnssckbi',
        'libfreebl3', 'libfreeblpriv3', 'libfontconfig', 'libfreetype',
        'libglib-2.0', 'libgobject-2.0', 'libgio-2.0', 'libgmodule-2.0',
        'libgthread-2.0', 'libgdk_pixbuf-2.0', 'libharfbuzz', 'libgraphite2',
        'libfribidi', 'libdatrie', 'libthai', 'libcrypto.so', 'libssl.so',
        'libp11-kit', 'libgnutls', 'libavahi-client', 'libavahi-common',
        'libcups', 'libdbus-1', 'libepoxy', 'libexpat', 'libffi',
    )
    a.binaries = [
        b for b in a.binaries
        if not os.path.basename(b[0]).startswith(_EXCLUDED_SYSTEM_LIB_PREFIXES)
    ]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CosmosCollection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/CosmosCollection.png',
)

# Separate CLI executable with console enabled
exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CosmosCollection-CLI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/CosmosCollection.png',
)

coll = COLLECT(
    exe,
    exe_cli,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CosmosCollection',
)
