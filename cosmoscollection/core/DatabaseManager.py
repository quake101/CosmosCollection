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


def set_favorite_image(cursor, image_id, favorite=True):
    """Make an image its object's favorite - one per object, so the object's
    others stop being it - or stop it being one. Does not commit."""
    if favorite:
        cursor.execute("UPDATE userimages SET is_favorite = 0 WHERE id != ? AND dsodetailid = "
                       "(SELECT dsodetailid FROM userimages WHERE id = ?)", (image_id, image_id))
    cursor.execute("UPDATE userimages SET is_favorite = ? WHERE id = ?", (1 if favorite else 0, image_id))


def add_user_image_row(cursor, dsodetailid, image_path, integration_time="", date_taken="", notes="",
                       session_id=None, equipment=None):
    """Insert one userimages row (the gallery's image of a DSO); its id. Does not commit.
    equipment: {"telescope_id", "telescope_name", "camera_id", "camera_name"} - see EquipmentMapper.py."""
    equipment = equipment or {}
    cursor.execute("""
        INSERT INTO userimages (
            dsodetailid, image_path, integration_time, date_taken, notes, session_id,
            telescope_id, telescope_name, camera_id, camera_name, created_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (dsodetailid, image_path, integration_time, date_taken, notes, session_id,
          equipment.get("telescope_id"), equipment.get("telescope_name"),
          equipment.get("camera_id"), equipment.get("camera_name")))
    return cursor.lastrowid


def open_connection(timeout=30, rows=False, catalogs=True):
    """A new connection to the app's database - for a worker thread or a command
    (DatabaseManager's shared connection belongs to the UI thread). timeout:
    seconds to wait while another connection writes. rows: rows as sqlite3.Row
    (by name) instead of tuples. catalogs: attach the update catalogs, so objects
    have the same names as everywhere else in the app."""
    from cosmoscollection.core.ResourceManager import ResourceManager, attach_update_catalogs
    conn = sqlite3.connect(str(ResourceManager.get_database_path()), timeout=timeout)
    if rows:
        conn.row_factory = sqlite3.Row
    if catalogs:
        attach_update_catalogs(conn)
    return conn


def add_columns(cursor, table, columns):
    """Add columns ("name TYPE ...") a table made by an older version lacks.
    True if any was added. Does not commit."""
    added = False
    for column in columns:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            added = True
        except sqlite3.OperationalError:
            pass  # Column already exists
    return added


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

                add_columns(cursor, "userimages", (
                    "is_favorite BOOLEAN DEFAULT 0",
                    # ALTER TABLE can't default to CURRENT_TIMESTAMP, so older rows have NULL
                    "created_date TEXT",
                    # The Session Manager session an image came from (set by the Add Image
                    # dialog's session prefill and the Import Wizard). No foreign key
                    # enforcement - SessionObservations.delete_session_children clears it.
                    "session_id INTEGER REFERENCES usersessions(id) ON DELETE SET NULL",
                    # The nights of that session the image was stacked from - night
                    # dates, comma separated (ImageDetails.py); NULL for all of them
                    "session_nights TEXT",
                ))

                # The telescope and camera each image was taken with, linked to
                # the user's Equipment (EquipmentMapper.py) - converted below
                from cosmoscollection.equipment import EquipmentMapper
                first_equipment_run = EquipmentMapper.ensure_columns(cursor)

                # Several locations, one of them active: the newest becomes the active one
                if add_columns(cursor, "usersettings", ("is_active INTEGER DEFAULT 0",)):
                    cursor.execute("""
                        UPDATE usersettings SET is_active = 1
                        WHERE id = (SELECT id FROM usersettings ORDER BY id DESC LIMIT 1)
                    """)
                # Each location's custom horizon, JSON [[az, alt], ...] (HorizonProfile.py)
                add_columns(cursor, "usersettings", ("horizon_points TEXT", "horizon_name TEXT"))

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

                add_columns(cursor, "usertargetlist",
                            ("telescope_id INTEGER REFERENCES usertelescopes(id) ON DELETE SET NULL",))
                
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

                # Canvas collages (Collage Builder): canvas size, collage-wide settings
                # (JSON) and each image's frame (JSON). layout_version NULL marks a
                # collage from the old grid editor, converted when it's opened.
                add_columns(cursor, "usercollages",
                            ("canvas_width INTEGER", "canvas_height INTEGER", "settings TEXT", "layout_version INTEGER"))
                add_columns(cursor, "usercollageimages", ("layout TEXT",))

                # Create usersessions table for planned/logged observing sessions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usersessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        target_id INTEGER REFERENCES usertargetlist(id) ON DELETE SET NULL,
                        dso_name TEXT NOT NULL,
                        ra_deg REAL,
                        dec_deg REAL,
                        status TEXT NOT NULL DEFAULT 'Planned',
                        session_date TEXT NOT NULL,
                        start_time TEXT,
                        end_time TEXT,
                        location_lat REAL,
                        location_lon REAL,
                        location_name TEXT,
                        location_timezone TEXT,
                        location_source TEXT DEFAULT 'active',
                        telescope_id INTEGER REFERENCES usertelescopes(id) ON DELETE SET NULL,
                        camera TEXT,
                        filters_used TEXT,
                        sub_count INTEGER DEFAULT 0,
                        integration_seconds REAL DEFAULT 0,
                        earliest_sub_date TEXT,
                        latest_sub_date TEXT,
                        notes TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        modified_date TEXT
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessions_target_id ON usersessions(target_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessions_session_date ON usersessions(session_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessions_status ON usersessions(status)")
                add_columns(cursor, "userequipment", (
                    "is_active INTEGER DEFAULT 1",  # Disabled equipment is kept, but not offered
                    "pixel_size REAL",              # A camera's (microns), for the FOV Simulator
                ))
                # The session's framing saved from the FOV Simulator (mosaic, rotation,
                # equipment - JSON, see FOVSimulator.py)
                add_columns(cursor, "usersessions", ("framing TEXT",))
                # The session's camera linked to the user's Equipment (EquipmentMapper.py) - converted below
                first_session_equipment_run = EquipmentMapper.ensure_session_columns(cursor)

                # Create usersessionfiles table for FITS/XISF subs scanned into a session
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usersessionfiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL REFERENCES usersessions(id) ON DELETE CASCADE,
                        file_path TEXT NOT NULL,
                        file_type TEXT,
                        frame_type TEXT,
                        object_name TEXT,
                        date_obs TEXT,
                        exptime_seconds REAL,
                        filter_name TEXT,
                        camera TEXT,
                        telescope TEXT,
                        gain REAL,
                        offset_value REAL,
                        ccd_temp REAL,
                        xbinning INTEGER,
                        ybinning INTEGER,
                        header_json TEXT,
                        added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(session_id, file_path)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usersessionfiles_session_id ON usersessionfiles(session_id)")

                # Observations (one per observing night) + usersessionfiles.observation_id/sub_key
                from cosmoscollection.sessions import SessionObservations
                SessionObservations.ensure_schema(conn)
                # Each night's detected mount (usersessionmounts, usersessionfiles.mount_id) and
                # the user's own choice (usersessions.mount_override)
                from cosmoscollection.sessions import MountDetection
                MountDetection.ensure_schema(conn)

                conn.commit()

                # Images and sessions from before the equipment columns: link their names
                try:
                    EquipmentMapper.migrate(conn, first_equipment_run, first_session_equipment_run)
                except Exception as e:
                    logger.error(f"Couldn't link the gallery images' equipment: {e}", exc_info=True)
                logger.debug("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling"""
        if self._connection is None:
            self._connection = open_connection(timeout=5.0, rows=True)
            
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