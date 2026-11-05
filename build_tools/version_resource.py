#!/usr/bin/env python3
"""
Windows version resource for the PyInstaller builds (CosmosCollection.spec,
updater/CosmosUpdater.spec), so Explorer's Properties > Details shows the
product name and version. SignPath code signing checks these: ProductName
must be the project's name and ProductVersion the release being signed.

The version comes from cosmoscollection/version.py's _FALLBACK_VERSION, which
update_version.py sets from the release tag before the build.
"""

import os
import re

PRODUCT_NAME = "Cosmos Collection"
COMPANY_NAME = "Cosmos Collection"
COPYRIGHT = "Copyright (c) 2025-2026 CosmosCollection"


def app_version(repo_dir):
    """The version in repo_dir's cosmoscollection/version.py, e.g. "1.0.10"."""
    with open(os.path.join(repo_dir, 'cosmoscollection', 'version.py'), 'r', encoding='utf-8') as f:
        match = re.search(r'_FALLBACK_VERSION = "([^"]*)"', f.read())
    return match.group(1) if match else '0.0.0'


def version_info(repo_dir, exe_name, description):
    """A VSVersionInfo for exe_name (without .exe), for EXE(version=...)."""
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable,
        VarFileInfo, VarStruct, VSVersionInfo)

    version = app_version(repo_dir)
    # The fixed part needs four numbers: "1.0.10" -> (1, 0, 10, 0)
    numbers = [int(n) for n in re.findall(r'\d+', version)[:4]]
    numbers += [0] * (4 - len(numbers))

    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=tuple(numbers), prodvers=tuple(numbers)),
        kids=[
            StringFileInfo([StringTable('040904B0', [
                StringStruct('CompanyName', COMPANY_NAME),
                StringStruct('FileDescription', description),
                StringStruct('FileVersion', version),
                StringStruct('InternalName', exe_name),
                StringStruct('LegalCopyright', COPYRIGHT),
                StringStruct('OriginalFilename', f'{exe_name}.exe'),
                StringStruct('ProductName', PRODUCT_NAME),
                StringStruct('ProductVersion', version),
            ])]),
            VarFileInfo([VarStruct('Translation', [0x0409, 1200])]),
        ],
    )
