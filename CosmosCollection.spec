# -*- mode: python ; coding: utf-8 -*-
import certifi
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Get certifi certificate bundle path
cert_file = certifi.where()
cert_dir = os.path.dirname(cert_file)

# Collect all astroquery and astropy modules, data, and binaries
astroquery_datas, astroquery_binaries, astroquery_hiddenimports = collect_all('astroquery')
astropy_datas, astropy_binaries, astropy_hiddenimports = collect_all('astropy')

# Combine with existing data files
datas_list = [
    ('catalogs', 'catalogs'),
    ('images', 'images'),
    (cert_file, 'certifi'),
] + astroquery_datas + astropy_datas

binaries_list = astroquery_binaries + astropy_binaries

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries_list,
    datas=datas_list,
    hiddenimports=astroquery_hiddenimports + astropy_hiddenimports + [
        'pyvo',
        'pyvo.dal',
        'bs4',
        'certifi',
        'html.parser',
        'requests',
        'requests.adapters',
        'requests.packages',
        'requests.packages.urllib3',
        'urllib3',
        'urllib3.util',
        'urllib3.util.ssl_',
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
