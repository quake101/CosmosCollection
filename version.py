#!/usr/bin/env python3
"""
Version management for Cosmos Collection
Handles version information from local fallback and GitHub releases
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Local fallback version - update this manually or via build process
__version__ = "1.0.7"
__build_date__ = "2025-09-20"

class VersionManager:
    """Manages version information for the application"""

    def __init__(self):
        self.github_repo = "quake101/CosmosCollection"
        self.cache_duration = timedelta(hours=6)  # Cache GitHub API response for 6 hours
        self._cached_release_info = None
        self._cache_timestamp = None

    def get_local_version(self) -> str:
        """Get the local fallback version"""
        return __version__

    def get_build_date(self) -> str:
        """Get the build date"""
        return __build_date__

    def get_github_latest_release(self) -> Optional[Dict[Any, Any]]:
        """
        Fetch the latest release information from GitHub API
        Returns None if unable to fetch or if cached data is still valid
        """
        try:
            # Check if we have valid cached data
            if (self._cached_release_info and self._cache_timestamp and
                datetime.now() - self._cache_timestamp < self.cache_duration):
                return self._cached_release_info

            # Fetch from GitHub API
            url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'CosmosCollection'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            release_data = response.json()

            # Cache the response
            self._cached_release_info = release_data
            self._cache_timestamp = datetime.now()

            logger.debug(f"Fetched latest release: {release_data.get('tag_name', 'Unknown')}")
            return release_data

        except requests.exceptions.RequestException as e:
            logger.debug(f"Could not fetch GitHub release info: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"Could not parse GitHub API response: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error fetching release info: {e}")
            return None

    def get_version_info(self) -> Dict[str, Any]:
        """
        Get comprehensive version information
        Returns a dictionary with version details
        """
        version_info = {
            'local_version': self.get_local_version(),
            'build_date': self.get_build_date(),
            'github_available': False,
            'github_version': None,
            'github_url': None,
            'github_published_date': None,
            'is_latest': None,
            'update_available': False
        }

        # Try to get GitHub release info
        github_release = self.get_github_latest_release()
        if github_release:
            version_info['github_available'] = True
            version_info['github_version'] = github_release.get('tag_name', 'Unknown')
            version_info['github_url'] = github_release.get('html_url')

            # Parse published date
            published_at = github_release.get('published_at')
            if published_at:
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    version_info['github_published_date'] = pub_date.strftime('%Y-%m-%d')
                except:
                    version_info['github_published_date'] = published_at

            # Compare versions
            local_version = self.get_local_version()
            github_version = version_info['github_version']

            if github_version and local_version:
                version_info['is_latest'] = self._compare_versions(local_version, github_version)
                version_info['update_available'] = not version_info['is_latest']

        return version_info

    def _compare_versions(self, local_version: str, github_version: str) -> bool:
        """
        Compare local version with GitHub version
        Returns True if local version is the same or newer, False if update available
        """
        try:
            # Remove 'v' prefix if present
            local_clean = local_version.lstrip('v').strip()
            github_clean = github_version.lstrip('v').strip()

            # Split into parts and compare
            local_parts = [int(x) for x in local_clean.split('.')]
            github_parts = [int(x) for x in github_clean.split('.')]

            # Pad shorter version with zeros
            max_len = max(len(local_parts), len(github_parts))
            local_parts.extend([0] * (max_len - len(local_parts)))
            github_parts.extend([0] * (max_len - len(github_parts)))

            # Compare versions
            for local_part, github_part in zip(local_parts, github_parts):
                if local_part < github_part:
                    return False  # Update available
                elif local_part > github_part:
                    return True   # Local is newer

            return True  # Versions are equal

        except (ValueError, AttributeError) as e:
            logger.debug(f"Error comparing versions {local_version} vs {github_version}: {e}")
            return True  # Assume no update needed if comparison fails

    def get_version_display_string(self) -> str:
        """Get a formatted version string for display in the UI"""
        version_info = self.get_version_info()

        if version_info['github_available'] and version_info['github_version']:
            if version_info['update_available']:
                return f"v{version_info['local_version']} (Update available: {version_info['github_version']})"
            else:
                return f"v{version_info['local_version']} (Latest)"
        else:
            return f"v{version_info['local_version']}"

    def get_detailed_version_info(self) -> str:
        """Get detailed version information for the About dialog"""
        version_info = self.get_version_info()

        details = [f"Version: {version_info['local_version']}"]

        if version_info['build_date']:
            details.append(f"Build Date: {version_info['build_date']}")

        if version_info['github_available']:
            if version_info['update_available']:
                details.append(f"Latest Release: {version_info['github_version']} (Update Available)")
            else:
                details.append(f"Latest Release: {version_info['github_version']} ✓")

            if version_info['github_published_date']:
                details.append(f"Release Date: {version_info['github_published_date']}")
        else:
            details.append("GitHub status: Offline")

        return "\n".join(details)


# Global instance for easy access
version_manager = VersionManager()

# Convenience functions
def get_version() -> str:
    """Get the current application version"""
    return version_manager.get_local_version()

def get_version_info() -> Dict[str, Any]:
    """Get comprehensive version information"""
    return version_manager.get_version_info()

def get_version_display() -> str:
    """Get formatted version string for UI display"""
    return version_manager.get_version_display_string()

def check_for_updates() -> bool:
    """Check if updates are available"""
    version_info = version_manager.get_version_info()
    return version_info.get('update_available', False)