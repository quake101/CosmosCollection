#!/usr/bin/env python3
"""
Database Manager for Cosmos Collection
Handles database connections and table initialization
"""

import sqlite3
import logging
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton database manager for the Cosmos Collection application"""
    
    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize_database()
        return cls._instance

    def _initialize_database(self):
        """Create necessary tables if they do not exist, e.g. usersettings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create usersettings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usersettings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        location_lat REAL,
                        location_lon REAL,
                        location_name TEXT,
                        timezone TEXT
                    )
                """)
                
                # Create usertelescopes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usertelescopes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        aperture REAL,
                        focal_length REAL,
                        mount_type TEXT,
                        notes TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)

                # Create userequipment table for cameras, eyepieces, barlows, and reducers
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS userequipment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_type TEXT NOT NULL,
                        name TEXT NOT NULL,
                        sensor_width REAL,
                        sensor_height REAL,
                        focal_length REAL,
                        apparent_fov REAL,
                        factor REAL,
                        notes TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create junction table for telescope-equipment many-to-many relationship
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS telescope_equipment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        telescope_id INTEGER NOT NULL REFERENCES usertelescopes(id) ON DELETE CASCADE,
                        equipment_id INTEGER NOT NULL REFERENCES userequipment(id) ON DELETE CASCADE,
                        UNIQUE(telescope_id, equipment_id)
                    )
                """)
                
                # Create userimages table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS userimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dsodetailid INTEGER,
                        image_path TEXT,
                        integration_time TEXT,
                        equipment TEXT,
                        date_taken TEXT,
                        notes TEXT,
                        is_favorite BOOLEAN DEFAULT 0,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dsodetailid) REFERENCES dso_detail (id)
                    )
                """)

                # Add is_favorite column if it doesn't exist (for existing databases)
                try:
                    cursor.execute("ALTER TABLE userimages ADD COLUMN is_favorite BOOLEAN DEFAULT 0")
                    logger.debug("Added is_favorite column to userimages table")
                except sqlite3.OperationalError:
                    # Column already exists, ignore
                    pass

                # Add created_date column if it doesn't exist (for existing databases)
                # Note: ALTER TABLE cannot use CURRENT_TIMESTAMP as default, so we use NULL
                try:
                    cursor.execute("ALTER TABLE userimages ADD COLUMN created_date TEXT")
                    logger.debug("Added created_date column to userimages table")
                except sqlite3.OperationalError:
                    # Column already exists, ignore
                    pass

                # Add is_active column to usersettings for multi-location support
                try:
                    cursor.execute("ALTER TABLE usersettings ADD COLUMN is_active INTEGER DEFAULT 0")
                    # Mark most recent record as active for migration
                    cursor.execute("""
                        UPDATE usersettings SET is_active = 1
                        WHERE id = (SELECT id FROM usersettings ORDER BY id DESC LIMIT 1)
                    """)
                    logger.debug("Added is_active column to usersettings table")
                except sqlite3.OperationalError:
                    # Column already exists, ignore
                    pass

                # Create usertargetlist table for user's observing target list
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usertargetlist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        dso_type TEXT,
                        constellation TEXT,
                        ra_deg REAL,
                        dec_deg REAL,
                        magnitude REAL,
                        size_info TEXT,
                        priority TEXT DEFAULT 'Medium',
                        status TEXT DEFAULT 'Not Observed',
                        best_months TEXT,
                        notes TEXT,
                        date_added TEXT,
                        date_observed TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        telescope_id INTEGER REFERENCES usertelescopes(id) ON DELETE SET NULL
                    )
                """)

                # Add telescope_id column if it doesn't exist (for existing databases)
                try:
                    cursor.execute("ALTER TABLE usertargetlist ADD COLUMN telescope_id INTEGER REFERENCES usertelescopes(id) ON DELETE SET NULL")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                # Create usercollages table for user's collage projects
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usercollages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dsodetailid INTEGER,
                        name TEXT NOT NULL,
                        grid_width INTEGER NOT NULL DEFAULT 3,
                        grid_height INTEGER NOT NULL DEFAULT 3,
                        cell_size INTEGER NOT NULL DEFAULT 400,
                        spacing INTEGER NOT NULL DEFAULT 20,
                        background_color TEXT NOT NULL DEFAULT 'black',
                        created_date TEXT NOT NULL,
                        modified_date TEXT NOT NULL,
                        FOREIGN KEY (dsodetailid) REFERENCES dsodetail(id)
                    )
                """)
                
                # Create usercollageimages table for collage image associations
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usercollageimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collage_id INTEGER NOT NULL,
                        userimage_id INTEGER NOT NULL,
                        position_index INTEGER NOT NULL,
                        FOREIGN KEY (collage_id) REFERENCES usercollages(id) ON DELETE CASCADE,
                        FOREIGN KEY (userimage_id) REFERENCES userimages(id) ON DELETE CASCADE
                    )
                """)
                
                conn.commit()
                logger.debug("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling"""
        if self._connection is None:
            # Import here to avoid circular imports
            from ResourceManager import ResourceManager, attach_update_catalogs
            db_path = ResourceManager.get_database_path()
            self._connection = sqlite3.connect(str(db_path))
            self._connection.row_factory = sqlite3.Row
            attach_update_catalogs(self._connection)
            
        try:
            yield self._connection
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise

    def close(self):
        """Close the database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {query}, error: {str(e)}")
            raise

    def execute_update(self, query, params=None):
        """Execute an update/insert/delete query"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing update: {query}, error: {str(e)}")
            raise