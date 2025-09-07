# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['c:\\Users\\Joe\\PycharmProjects\\CosmosCollection\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\Users\\Joe\\PycharmProjects\\CosmosCollection\\catalogs', 'catalogs'), ('c:\\Users\\Joe\\PycharmProjects\\CosmosCollection\\images', 'images')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CosmosCollection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['c:\\Users\\Joe\\PycharmProjects\\CosmosCollection\\images\\CosmosCollection.png'],
)
