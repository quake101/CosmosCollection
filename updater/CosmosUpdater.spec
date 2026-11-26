# -*- mode: python ; coding: utf-8 -*-
# Standalone updater helper, built as a single-file executable so it has no
# external DLLs/data directory of its own to manage while it's swapping out
# the main app's install directory. Build from the repo root:
#   pyinstaller updater/CosmosUpdater.spec

import os
import sys

block_cipher = None

# SPECPATH is injected by PyInstaller as the directory containing this spec
# file - use it rather than a bare relative path so the build works
# regardless of the CWD it's invoked from.
script_path = os.path.join(SPECPATH, 'CosmosUpdater.py')

a = Analysis(
    [script_path],
    pathex=[],
    binaries=[],
    # The progress window's icon on macOS (Dock) and Linux (zenity /
    # kdialog). Windows doesn't need it: its window uses the .exe's own
    # icon, set by icon= below.
    datas=[] if sys.platform == 'win32' else [
        (os.path.join(SPECPATH, '..', 'images', 'CosmosCollectionUpdater.png'), 'images'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The updater is stdlib-only and draws its progress window natively
    # (CosmosUpdaterProgress.py), so no GUI toolkit belongs in it. These
    # excludes guard against one being pulled in by accident, which would
    # grow this one-file build several times over (bundling Qt made it 45 MB).
    excludes=[
        'PySide6',
        'shiboken6',
        'tkinter',
        '_tkinter',
        'numpy',
        'scipy',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
        # Unused stdlib parts that bring DLLs along or are large. hashlib
        # works without _hashlib (it falls back to the built-in hashes), and
        # dropping it drops OpenSSL's 5 MB libcrypto. shutil works without
        # _bz2/_lzma (only its archive helpers use them).
        'ssl',
        '_ssl',
        '_hashlib',
        '_bz2',
        '_lzma',
        'sqlite3',
        '_sqlite3',
        'unittest',
        'pydoc',
        'pydoc_data',
        'doctest',
        'xmlrpc',
        'asyncio',
        'multiprocessing',
        'lib2to3',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Product name/version in the .exe's details, which SignPath code signing
# checks (see build_tools/version_resource.py)
repo_dir = os.path.join(SPECPATH, '..')
sys.path.insert(0, os.path.join(repo_dir, 'build_tools'))
from version_resource import version_info

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CosmosCollectionUpdater',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, '..', 'images', 'CosmosCollectionUpdater.png'),
    version=(version_info(repo_dir, 'CosmosCollectionUpdater', 'Cosmos Collection updater')
             if sys.platform == 'win32' else None),
)
