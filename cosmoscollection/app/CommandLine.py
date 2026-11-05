"""Command-line mode (python main.py --help)."""

import argparse
import os
import sys
from typing import Dict, Optional

from cosmoscollection.core.DatabaseManager import DatabaseManager


# --- Command Line Interface ---
def parse_cli_arguments():
    """Parse command line arguments for CLI operations"""
    # Determine the program name for help text
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        prog_name = os.path.basename(sys.executable)
    else:
        # Running from source
        prog_name = "python main.py"

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description='Cosmos Collection - Deep Sky Object Image Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Add an image to a DSO
  {prog_name} --add-image --dso "M31" --image "/path/to/image.jpg"

  # Add an image with metadata
  {prog_name} --add-image --dso "NGC 7000" --image "/path/to/image.tif" \\
      --equipment "Telescope: AT72ED, Camera: ASI2600" \\
      --integration "3600" --date "2024-01-15" --notes "First light!"

  # List all DSOs in the database
  {prog_name} --list-dsos

  # Search for a DSO
  {prog_name} --search-dso "M31"
"""
    )

    # CLI operation flags
    parser.add_argument('--add-image', action='store_true',
                        help='Add an image to a DSO in the database')
    parser.add_argument('--list-dsos', action='store_true',
                        help='List all DSOs in the database')
    parser.add_argument('--search-dso', type=str, metavar='NAME',
                        help='Search for a DSO by name')

    # Image addition arguments
    parser.add_argument('--dso', type=str, metavar='NAME',
                        help='DSO name (e.g., M31, NGC 7000, IC 1396)')
    parser.add_argument('--image', type=str, metavar='PATH',
                        help='Path to the image file')
    parser.add_argument('--equipment', type=str, default='',
                        help='Equipment used: "telescope, camera" (optional)')
    parser.add_argument('--integration', type=str, default='',
                        help='Integration time in seconds (optional)')
    parser.add_argument('--date', type=str, default='',
                        help='Date taken YYYY-MM-DD (optional)')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes about the image (optional)')
    parser.add_argument('--set-favorite', action='store_true',
                        help='Set this image as the favorite for the DSO')

    # Only parse known args to avoid conflicts with Qt arguments
    args, unknown = parser.parse_known_args()

    return args, unknown


def find_dso_by_name(dso_name: str) -> Optional[Dict]:
    """Find a DSO in the database by name or catalog number"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Normalize the search name
            search_name = dso_name.strip().upper()

            # Try matching via cataloguenr table (e.g., M31, NGC 7000)
            import re
            match = re.match(r'^([A-Z]+)\s*(\d+)([A-Z]?)$', search_name)
            if match:
                catalog = match.group(1)
                designation = match.group(2)
                suffix = match.group(3) or ''

                cursor.execute("""
                    SELECT d.id,
                           c.catalogue || ' ' || c.designation as name,
                           d.dsotype,
                           d.constellation
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    WHERE UPPER(c.catalogue) = ? AND c.designation = ?
                    LIMIT 1
                """, (catalog, designation + suffix))

                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'commonnames': None,
                        'type': row[2],
                        'constellation': row[3]
                    }

            # Try partial match on catalog entries
            cursor.execute("""
                SELECT d.id,
                       c.catalogue || ' ' || c.designation as name,
                       d.dsotype,
                       d.constellation
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE UPPER(c.catalogue || c.designation) LIKE ?
                   OR UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                LIMIT 1
            """, (f'%{search_name}%', f'%{search_name}%'))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'commonnames': None,
                    'type': row[2],
                    'constellation': row[3]
                }

            return None

    except Exception as e:
        print(f"Error searching for DSO: {e}")
        return None


def add_image_cli(args) -> bool:
    """Add an image to the database via command line"""
    # Validate required arguments
    if not args.dso:
        print("Error: --dso argument is required")
        return False

    if not args.image:
        print("Error: --image argument is required")
        return False

    # Check if image file exists
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False

    # Find the DSO in the database
    print(f"Searching for DSO: {args.dso}")
    dso = find_dso_by_name(args.dso)

    if not dso:
        print(f"Error: DSO '{args.dso}' not found in database")
        print("Use --search-dso to find available DSOs or --list-dsos to see all")
        return False

    print(f"Found DSO: {dso['name']}", end="")
    if dso['commonnames']:
        print(f" ({dso['commonnames']})", end="")
    print(f" - {dso['type']} in {dso['constellation']}")

    # Add the image to the database
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # "telescope, camera" - linked to the user's Equipment where it names it
            from cosmoscollection.equipment import EquipmentMapper
            from cosmoscollection.sessions import LibraryScanner
            from cosmoscollection.core.DatabaseManager import add_user_image_row
            telescope_id, telescope_name, camera_id, camera_name = EquipmentMapper.parse_equipment_text(
                args.equipment, LibraryScanner.load_equipment_info(conn))
            image_id = add_user_image_row(cursor, dso['id'], image_path, args.integration, args.date, args.notes,
                                          equipment={"telescope_id": telescope_id, "telescope_name": telescope_name,
                                                     "camera_id": camera_id, "camera_name": camera_name})

            # Set as favorite if requested
            if args.set_favorite:
                cursor.execute("""
                    UPDATE dsodetail SET favourite_image = ? WHERE id = ?
                """, (image_id, dso['id']))
                print(f"Set as favorite image for {dso['name']}")

            conn.commit()

        print(f"Successfully added image to {dso['name']}")
        print(f"  Image path: {image_path}")
        if args.equipment:
            print(f"  Equipment: {args.equipment}")
        if args.integration:
            print(f"  Integration: {args.integration}s")
        if args.date:
            print(f"  Date: {args.date}")

        return True

    except Exception as e:
        print(f"Error adding image to database: {e}")
        return False


def list_dsos_cli() -> bool:
    """List all DSOs in the database"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                           ORDER BY CASE c.catalogue
                               WHEN 'M' THEN 1
                               WHEN 'NGC' THEN 2
                               WHEN 'IC' THEN 3
                               ELSE 4
                           END) as name,
                       d.dsotype,
                       d.constellation,
                       (SELECT COUNT(*) FROM userimages u WHERE u.dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                GROUP BY d.id
                ORDER BY
                    CASE
                        WHEN MIN(c.catalogue) = 'M' THEN 1
                        WHEN MIN(c.catalogue) = 'NGC' THEN 2
                        WHEN MIN(c.catalogue) = 'IC' THEN 3
                        ELSE 4
                    END,
                    MIN(CAST(c.designation AS INTEGER))
            """)

            rows = cursor.fetchall()

            print(f"Found {len(rows)} DSOs in database:\n")
            print(f"{'Name':<30} {'Type':<20} {'Constellation':<15} {'Images':<8}")
            print("-" * 80)

            for row in rows:
                name = (row[0] or '')[:29]
                dso_type = (row[1] or '')[:19]
                constellation = (row[2] or '')[:14]
                images = row[3] or 0

                print(f"{name:<30} {dso_type:<20} {constellation:<15} {images:<8}")

            return True

    except Exception as e:
        print(f"Error listing DSOs: {e}")
        return False


def search_dso_cli(search_term: str) -> bool:
    """Search for DSOs matching a term"""
    db_manager = DatabaseManager()

    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            search_pattern = f'%{search_term.upper()}%'

            cursor.execute("""
                SELECT GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', '
                           ORDER BY CASE c.catalogue
                               WHEN 'M' THEN 1
                               WHEN 'NGC' THEN 2
                               WHEN 'IC' THEN 3
                               ELSE 4
                           END) as name,
                       d.dsotype,
                       d.constellation,
                       (SELECT COUNT(*) FROM userimages u WHERE u.dsodetailid = d.id) as image_count
                FROM dsodetail d
                JOIN cataloguenr c ON d.id = c.dsodetailid
                WHERE UPPER(c.catalogue || c.designation) LIKE ?
                   OR UPPER(c.catalogue || ' ' || c.designation) LIKE ?
                GROUP BY d.id
                ORDER BY
                    CASE
                        WHEN MIN(c.catalogue) = 'M' THEN 1
                        WHEN MIN(c.catalogue) = 'NGC' THEN 2
                        WHEN MIN(c.catalogue) = 'IC' THEN 3
                        ELSE 4
                    END,
                    MIN(CAST(c.designation AS INTEGER))
                LIMIT 50
            """, (search_pattern, search_pattern))

            rows = cursor.fetchall()

            if not rows:
                print(f"No DSOs found matching '{search_term}'")
                return True

            print(f"Found {len(rows)} DSOs matching '{search_term}':\n")
            print(f"{'Name':<30} {'Type':<20} {'Constellation':<15} {'Images':<8}")
            print("-" * 80)

            for row in rows:
                name = (row[0] or '')[:29]
                dso_type = (row[1] or '')[:19]
                constellation = (row[2] or '')[:14]
                images = row[3] or 0

                print(f"{name:<30} {dso_type:<20} {constellation:<15} {images:<8}")

            return True

    except Exception as e:
        print(f"Error searching DSOs: {e}")
        return False


def run_cli_command(args) -> Optional[bool]:
    """
    Run a CLI command if specified.
    Returns True/False for success/failure, or None if no CLI command was requested.
    """
    if args.add_image:
        return add_image_cli(args)

    if args.list_dsos:
        return list_dsos_cli()

    if args.search_dso:
        return search_dso_cli(args.search_dso)

    # No CLI command requested
    return None
