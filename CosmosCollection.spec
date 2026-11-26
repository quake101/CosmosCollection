# -*- mode: python ; coding: utf-8 -*-
import certifi
import os
import re
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get certifi certificate bundle path
cert_file = certifi.where()
cert_dir = os.path.dirname(cert_file)

# Collect astroquery data files (CITATION, configs, etc.), without the ~15 MB
# of test data its tests/ folders hold
astroquery_datas = collect_data_files('astroquery', excludes=['**/tests/**'])

# Left out of both executables - nothing the app runs needs them:
_EXCLUDES = [
    'scipy',
    'tkinter',
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    # Only astropy's optional table converters (Table.to_pandas) and its tests
    # import pandas, and the app never calls them (~22 MB)
    'pandas',
    # Build tools: pulled in by astropy's build-time setup_package.py files
    # and by version fallbacks that already handle their absence
    'setuptools',
    'setuptools_scm',
    'pkg_resources',
    'wheel',
    '_distutils_hack',
    # Qt modules the app never loads. They were bundled only because Qt
    # WebEngine's C++ library links QtQuick, and PyInstaller's Qt hooks then
    # brought along every QML plugin and the Quick 3D/Graphs/Charts/Controls
    # libraries those depend on (~70 MB). WebEngine still gets the Qt6Quick
    # and Qt6Qml libraries it links: binary dependencies are collected on
    # their own, whatever the Python modules.
    'PySide6.QtQml',
    'PySide6.QtQuick',
    'PySide6.QtQuickWidgets',
    'PySide6.QtOpenGL',
    'PySide6.QtPositioning',
]

# Tests inside these packages (modules and their data), and astropy's
# build-time setup_package modules. astropy.tests itself stays: importing
# astropy loads astropy.tests.runner.
_TEST_PACKAGES = r'(astropy|astroquery|pyvo|astroplan)'
_TEST_MODULE = re.compile('^' + _TEST_PACKAGES + r'\..*\.(tests|conftest)(\.|$)|\.setup_package$')
_TEST_DATA = re.compile(_TEST_PACKAGES + r'[/\\](.+[/\\])?tests[/\\]')

# Qt translations: the app is English only. Qt's own strings fall back to
# English without any .qm file, and Qt WebEngine falls back to its en-US
# locale pack. Also Qt WebEngine's developer tools, which the app never
# opens (~10 MB). Matches the Windows and Linux layouts; macOS keeps these
# inside QtWebEngineCore.framework, which is left as it is.
_QT_DIR = r'PySide6[/\\](Qt[/\\])?'
_QT_UNUSED_DATA = re.compile(
    _QT_DIR + r'translations[/\\](?!qtwebengine_locales[/\\]en-US\.pak$)'
    + '|' + _QT_DIR + r'resources[/\\]qtwebengine_devtools_resources\.pak$'
)


def _keep_module(entry):
    return not _TEST_MODULE.search(entry[0])


def _keep_file(entry):
    dest = entry[0]
    return not (_TEST_DATA.match(dest) or _QT_UNUSED_DATA.match(dest))


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('catalogs', 'catalogs'),
        ('images', 'images'),
        ('web', 'web'),  # Aladin Lite for the FOV Simulator (LGPL-3.0, see web/aladin/README.txt)
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
        'cosmoscollection.planning.BestDSOTonight',
        'cosmoscollection.planning.DSOVisibilityCalculator',
        'cosmoscollection.planning.DSOTargetList',
        'cosmoscollection.planning.HorizonProfile',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# The command line gets its own, smaller code archive: it starts from the
# CLI package alone, so it leaves out the app's windows and matplotlib (see
# cosmoscollection/cli/Main.py). Both executables still share the one
# _internal folder below. QtGui and QtWidgets stay in its graph: a few
# shared modules the CLI uses import them for images and messages.
a_cli = Analysis(
    [os.path.join('cosmoscollection', 'cli', '__main__.py')],
    pathex=[SPECPATH],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES + [
        'PySide6.QtNetwork',
        'PySide6.QtPrintSupport',
        'PySide6.QtWebChannel',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebEngineWidgets',
        'matplotlib',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

for _analysis in (a, a_cli):
    _analysis.pure = [entry for entry in _analysis.pure if _keep_module(entry)]
    _analysis.datas = [entry for entry in _analysis.datas if _keep_file(entry)]
    _analysis.binaries = [entry for entry in _analysis.binaries if _keep_file(entry)]

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
    for _analysis in (a, a_cli):
        _analysis.binaries = [
            b for b in _analysis.binaries
            if not os.path.basename(b[0]).startswith(_EXCLUDED_SYSTEM_LIB_PREFIXES)
        ]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

# Product name/version in the .exe files' details, which SignPath code
# signing checks (see build_tools/version_resource.py)
sys.path.insert(0, os.path.join(SPECPATH, 'build_tools'))
from version_resource import version_info

def _windows_version(exe_name, description):
    return version_info(SPECPATH, exe_name, description) if sys.platform == 'win32' else None

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
    version=_windows_version('CosmosCollection', 'Cosmos Collection'),
)

# Separate CLI executable with console enabled
exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
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
    version=_windows_version('CosmosCollection-CLI', 'Cosmos Collection command line'),
)

coll = COLLECT(
    exe,
    exe_cli,
    a.binaries,
    a.zipfiles,
    a.datas,
    # Already in a's lists, which COLLECT de-duplicates; listed in case the
    # command line ever needs a file the app doesn't
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CosmosCollection',
)

if sys.platform == 'darwin':
    # Without this, macOS release builds were a bare COLLECT() folder full of loose
    # Mach-O binaries/.so files instead of a proper .app bundle. Unsigned loose
    # binaries get Gatekeeper-challenged one at a time as the dynamic linker touches
    # each one (the "have to authorize every file" report), instead of Gatekeeper
    # evaluating one signed bundle as a single unit. The CI codesign step (ad-hoc,
    # see build-release.yml) is what actually consolidates that into one prompt --
    # this BUNDLE() just gives it something bundle-shaped to sign.
    import re as _re
    with open(os.path.join(SPECPATH, 'cosmoscollection', 'version.py'), 'r', encoding='utf-8') as _vf:
        _version_match = _re.search(r'_FALLBACK_VERSION = "([^"]*)"', _vf.read())
    _app_version = _version_match.group(1) if _version_match else '0.0.0'

    app = BUNDLE(
        coll,
        name='CosmosCollection.app',
        icon='images/CosmosCollection.png',
        bundle_identifier='io.github.quake101.cosmoscollection',
        info_plist={
            'CFBundleName': 'CosmosCollection',
            'CFBundleDisplayName': 'CosmosCollection',
            'CFBundleShortVersionString': _app_version,
            'CFBundleVersion': _app_version,
            'NSHighResolutionCapable': True,
        },
    )
