# -*- mode: python ; coding: utf-8 -*-
import certifi
import os

block_cipher = None

# Get certifi certificate bundle path
cert_file = certifi.where()
cert_dir = os.path.dirname(cert_file)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('catalogs', 'catalogs'),
        ('images', 'images'),
        (cert_file, 'certifi'),
    ],
    hiddenimports=[
        'astroquery',
        'astroquery.simbad',
        'astroquery.simbad.core',
        'astroquery.query',
        'astroquery.utils',
        'astroquery.utils.tap',
        'astroquery.utils.tap.core',
        'astroquery.utils.commons',
        'astropy',
        'astropy.coordinates',
        'astropy.units',
        'astropy.table',
        'astropy.io.votable',
        'astropy.io.ascii',
        'pyvo',
        'pyvo.dal',
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CosmosCollection',
)
