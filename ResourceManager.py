#!/usr/bin/env python3
"""
Cross-platform resource manager for Cosmos Collection
Handles resource paths for both development and PyInstaller bundled environments
"""

import os
import sys
import platform
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages application resources in a cross-platform way"""
    
    def __init__(self):
        self._base_path = self._get_base_path()
        self._is_bundled = self._check_if_bundled()
        
    def _get_base_path(self):
        """Get the base application path"""
        if getattr(sys, 'frozen', False):
            # PyInstaller bundle
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller temporary directory
                return Path(sys._MEIPASS)
            else:
                # PyInstaller one-file mode fallback
                return Path(sys.executable).parent
        else:
            # Development mode
            return Path(__file__).parent.absolute()
    
    def _check_if_bundled(self):
        """Check if running as a PyInstaller bundle"""
        return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    def get_resource_path(self, *path_parts):
        """Get a cross-platform resource path"""
        return self._base_path.joinpath(*path_parts)
    
    def get_data_dir(self):
        """Get the user data directory based on platform"""
        app_name = "CosmosCollection"
        
        if platform.system() == "Windows":
            # Windows: Use AppData/Roaming
            data_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / app_name
        elif platform.system() == "Darwin":
            # macOS: Use ~/Library/Application Support
            data_dir = Path.home() / 'Library' / 'Application Support' / app_name
        else:
            # Linux/Unix: Use ~/.local/share or XDG_DATA_HOME
            xdg_data = os.environ.get('XDG_DATA_HOME')
            if xdg_data:
                data_dir = Path(xdg_data) / app_name.lower()
            else:
                data_dir = Path.home() / '.local' / 'share' / app_name.lower()
        
        # Create directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_config_dir(self):
        """Get the user configuration directory based on platform"""
        app_name = "CosmosCollection"
        
        if platform.system() == "Windows":
            # Windows: Use AppData/Roaming (same as data)
            config_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / app_name
        elif platform.system() == "Darwin":
            # macOS: Use ~/Library/Preferences
            config_dir = Path.home() / 'Library' / 'Preferences' / app_name
        else:
            # Linux/Unix: Use ~/.config or XDG_CONFIG_HOME
            xdg_config = os.environ.get('XDG_CONFIG_HOME')
            if xdg_config:
                config_dir = Path(xdg_config) / app_name.lower()
            else:
                config_dir = Path.home() / '.config' / app_name.lower()
        
        # Create directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def open_file_manager(self, path):
        """Open the system file manager at the specified path"""
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Path does not exist: {path}")
                return False
                
            # Use the directory if path is a file
            if path.is_file():
                path = path.parent
                
            system = platform.system()
            
            if system == "Windows":
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                import subprocess
                subprocess.run(["open", str(path)], check=True)
            else:  # Linux and other Unix-like systems
                import subprocess
                # Try common Linux file managers
                for cmd in ["xdg-open", "nautilus", "dolphin", "thunar", "pcmanfm"]:
                    try:
                        subprocess.run([cmd, str(path)], check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    logger.warning(f"Could not find a suitable file manager to open {path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error opening file manager: {e}")
            return False
    
    def get_database_path(self):
        """Get the user database path, copying from catalogs folder if needed"""
        # Always use user data directory for the user's database
        user_db_path = self.get_data_dir() / 'DSO.sqlite'
        
        # Check if we need to copy the database from catalogs folder
        source_db = self.get_source_database_path()
        if source_db.exists() and not user_db_path.exists():
            # Copy the database from catalogs to user data directory
            import shutil
            shutil.copy2(source_db, user_db_path)
            logger.info(f"Copied initial database from {source_db} to {user_db_path}")
        elif not user_db_path.exists() and not source_db.exists():
            logger.warning(f"No database found at {source_db} and no user database at {user_db_path}")
            
        return user_db_path
    
    def get_source_database_path(self):
        """Get the source/template database path from catalogs folder"""
        return self.get_resource_path('catalogs', 'DSO.sqlite')
    
    def get_icon_path(self):
        """Get the application icon path"""
        return self.get_resource_path('images', 'CosmosCollection.png')
    
    @property
    def is_bundled(self):
        """Check if running as a bundle"""
        return self._is_bundled
    
    @property
    def base_path(self):
        """Get the base application path"""
        return self._base_path


# Global instance
ResourceManager = ResourceManager()