# -*- mode: python ; coding: utf-8 -*-
# Standalone updater helper, built as a single-file executable so it has no
# external DLLs/data directory of its own to manage while it's swapping out
# the main app's install directory. Build from the repo root:
#   pyinstaller updater/CosmosUpdater.spec

import os

block_cipher = None

# SPECPATH is injected by PyInstaller as the directory containing this spec
# file - use it rather than a bare relative path so the build works
# regardless of the CWD it's invoked from.
script_path = os.path.join(SPECPATH, 'CosmosUpdater.py')

a = Analysis(
    [script_path],
    pathex=[],
    binaries=[],
    datas=[
        # Bundled so CosmosUpdaterWorker.py can load it into a QIcon at
        # runtime - separate from the icon= below, which only sets the
        # .exe's own file/taskbar icon.
        (os.path.join(SPECPATH, '..', 'images', 'CosmosCollectionUpdater.png'), 'images'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # PySide6 (QtCore/QtGui/QtWidgets, for CosmosUpdaterWorker.py's status
    # dialog) is auto-detected via its own bundled PyInstaller hooks - excluding
    # the unused Qt modules the main app also excludes/doesn't need here
    # keeps this onefile build from ballooning.
    excludes=[
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineQuick',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtNetwork',
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
    icon=os.path.join(SPECPATH, '..', 'images', 'CosmosCollectionUpdater.png'),
)
