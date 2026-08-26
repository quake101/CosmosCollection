# -*- mode: python ; coding: utf-8 -*-
# Standalone updater helper, built as a single-file executable so it has no
# external DLLs/data directory of its own to manage while it's swapping out
# the main app's install directory. Build from the repo root:
#   pyinstaller updater/cosmos_updater.spec

import os

block_cipher = None

# SPECPATH is injected by PyInstaller as the directory containing this spec
# file - use it rather than a bare relative path so the build works
# regardless of the CWD it's invoked from.
script_path = os.path.join(SPECPATH, 'cosmos_updater.py')

a = Analysis(
    [script_path],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
)
