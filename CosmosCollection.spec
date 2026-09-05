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
    # These are base Linux/X11/desktop-integration libraries (GTK, NSS, cairo,
    # fontconfig, freetype, OpenSSL, libstdc++, X11/xcb, GBM, zlib, SQLite, Kerberos,
    # ...) that PyInstaller's dependency walker bundles from whatever happens to be
    # installed on the BUILD machine, because Qt WebEngine's Chromium dynamically
    # depends on (or optionally dlopens) many of them. Bundling a snapshot from one
    # machine (e.g. GitHub Actions' Ubuntu runner) and running it on another (e.g.
    # CachyOS) reproducibly crashed the WebEngine renderer process (map picker, FOV
    # Simulator - "renderer process terminated ... exit code: 1002", no import-time
    # error, only at first render) even though the actual Qt/Chromium binaries were
    # byte-identical between the working and broken builds -- confirmed by diffing a
    # working build against a crashing one built from the exact same source: only
    # these ~95 base-OS libraries actually differed in content. Excluding a narrower
    # first attempt (just GTK/NSS) did not fully fix it, so this list was widened to
    # every base-OS library the diff turned up (libstdc++.so.6 and libgcc_s.so.1 -
    # an ABI mismatch there against a Chromium built on a different toolchain baseline
    # is the single most likely culprit of the two attempts). All of these are
    # near-universally present on any Linux desktop capable of running a Qt GUI at
    # all, so excluding them here and letting normal dynamic linking find the target
    # machine's own copies is safer than freezing in whatever CI happened to have.
    # Every prefix below is anchored with the literal '.so' that starts a soname's
    # version suffix (e.g. 'libfreetype.so', not just 'libfreetype'). Several packages
    # -- Pillow in particular -- vendor their OWN private, auditwheel-renamed copies of
    # some of these same library families as e.g. 'libfreetype-5d47eaee.so.6.20.2', and
    # an earlier unanchored version of this list matched those too (basename.startswith
    # doesn't care about the '-hash' in between), silently deleting Pillow's own
    # required copies and breaking image loading. The '.so' anchor makes that
    # impossible: a hash-suffixed name never has '.so' immediately after the family
    # name, only after '-<hash>'.
    _EXCLUDED_SYSTEM_LIB_PREFIXES = (
        # GTK/GLib/accessibility/theming (optional dlopen targets for native dialogs)
        'libgtk-3.so', 'libgdk-3.so', 'libgdk_pixbuf-2.0.so', 'libatk-1.0.so',
        'libatk-bridge-2.0.so', 'libatspi.so', 'libglib-2.0.so', 'libgobject-2.0.so',
        'libgio-2.0.so', 'libgmodule-2.0.so', 'libgthread-2.0.so',
        # Font/text shaping/rendering
        'libcairo.so', 'libcairo-gobject.so', 'libpango-1.0.so', 'libpangocairo-1.0.so',
        'libpangoft2-1.0.so', 'libfontconfig.so', 'libfreetype.so', 'libharfbuzz.so',
        'libgraphite2.so', 'libfribidi.so', 'libdatrie.so', 'libthai.so', 'libpixman-1.so',
        # NSS/crypto/TLS
        'libnss3.so', 'libnssutil3.so', 'libnspr4.so', 'libnssckbi.so', 'libfreebl3.so',
        'libfreeblpriv3.so', 'libsoftokn3.so', 'libsmime3.so', 'libplc4.so', 'libplds4.so',
        'libcrypto.so', 'libssl.so', 'libp11-kit.so', 'libgnutls.so', 'libtasn1.so',
        'libunistring.so', 'libidn2.so', 'libgmp.so',
        # C/C++ runtime and low-level base libs - an ABI mismatch here (Chromium built
        # against a different toolchain baseline than the bundled copy) is the prime
        # suspect for a hard renderer crash with no Python-level error
        'libstdc++.so', 'libgcc_s.so', 'libatomic.so',
        # Compression / misc data formats
        'libz.so', 'libzstd.so', 'libbz2.so', 'liblzma.so', 'libpng16.so', 'libpcre2-8.so',
        'libbrotlicommon.so', 'libbrotlidec.so',
        # X11/xcb/GPU-buffer (Chromium's Linux windowing + GPU compositing path)
        'libX11.so', 'libX11-xcb.so', 'libXau.so', 'libXcomposite.so', 'libXcursor.so',
        'libXdamage.so', 'libXdmcp.so', 'libXext.so', 'libXfixes.so', 'libXinerama.so',
        'libXi.so', 'libXrandr.so', 'libXrender.so', 'libXtst.so',
        'libxcb-glx.so', 'libxcb-randr.so', 'libxcb-render.so', 'libxcb-shm.so',
        'libxcb-sync.so', 'libxcb-xfixes.so',
        'libxkbcommon.so', 'libxkbfile.so', 'libxshmfence.so', 'libgbm.so',
        # D-Bus/system/session integration
        'libdbus-1.so', 'libepoxy.so', 'libexpat.so', 'libffi.so', 'libavahi-client.so',
        'libavahi-common.so', 'libcups.so', 'libsystemd.so', 'libmount.so', 'libblkid.so',
        'libcom_err.so', 'libasound.so',
        # Kerberos/GSSAPI (pulled in transitively via requests/keyring's optional auth backends)
        'libgssapi_krb5.so', 'libk5crypto.so', 'libkrb5support.so', 'libkrb5.so',
        'libkeyutils.so',
        # XML/misc
        'libxml2.so', 'libxslt.so', 'libsqlite3.so',
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
