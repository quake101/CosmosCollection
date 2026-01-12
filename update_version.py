#!/usr/bin/env python3
"""
Update version.py with the current Git tag
Used by GitHub Actions during release builds
"""

import re
import sys
from datetime import datetime

def update_version(tag):
    """Update version.py with the provided tag"""
    try:
        # Remove 'v' prefix if present
        clean_tag = tag.lstrip('v')

        # Read current version.py
        with open('version.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Update fallback version and build date (used for packaged builds)
        content = re.sub(r'_FALLBACK_VERSION = "[^"]*"', f'_FALLBACK_VERSION = "{clean_tag}"', content)
        content = re.sub(r'_FALLBACK_BUILD_DATE = "[^"]*"', f'_FALLBACK_BUILD_DATE = "{datetime.now().strftime("%Y-%m-%d")}"', content)

        # Write back
        with open('version.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'Successfully updated version to {clean_tag}')
        return True

    except Exception as e:
        print(f'Error updating version: {e}')
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python update_version.py <tag>')
        sys.exit(1)

    tag = sys.argv[1]
    success = update_version(tag)
    sys.exit(0 if success else 1)