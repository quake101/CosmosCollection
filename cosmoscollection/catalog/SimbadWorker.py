"""Background SIMBAD lookup of one object, and the wait dialog shown meanwhile."""

import logging

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from cosmoscollection.ui.Theme import font_size, themed_style

logger = logging.getLogger(__name__)


# --- SIMBAD Query Worker Thread ---
class SimbadQueryWorker(QThread):
    """Worker thread for querying SIMBAD and adding object to database"""
    object_found = Signal(dict)  # Signal with object data
    object_not_found = Signal()  # Signal when object not found
    error_occurred = Signal(str)  # Signal with error message

    def __init__(self, search_term, parent=None):
        super().__init__(parent)
        self.search_term = search_term

    def run(self):
        """Query SIMBAD in background thread"""
        from cosmoscollection.catalog.SimbadQuery import SimbadUnavailable
        try:
            from cosmoscollection.catalog.SimbadQuery import query
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            logger.debug(f"Querying SIMBAD for: {self.search_term}")

            # Query by identifier, with comprehensive data
            result = query(lambda simbad: simbad.query_object(self.search_term),
                           fields=(
                               'otype',  # Object type
                               'dimensions',  # Dimensions
                               'V',  # V magnitude
                               'B',  # B magnitude
                           ))

            if result is None or len(result) == 0:
                logger.debug(f"No SIMBAD data found for {self.search_term}")
                self.object_not_found.emit()
                return

            # Parse the first result
            row = result[0]

            # Debug: print column names
            logger.debug(f"SIMBAD columns: {result.colnames}")

            # Extract coordinates - try different column name variations
            ra_str = None
            dec_str = None

            # Try common RA/DEC column names
            for ra_col in ['RA', 'ra', 'RA_d', 'ra_d', 'RA_ICRS', 'ra_icrs']:
                if ra_col in row.colnames:
                    ra_str = row[ra_col]
                    logger.debug(f"Found RA in column: {ra_col} = {ra_str}")
                    break

            for dec_col in ['DEC', 'dec', 'DEC_d', 'dec_d', 'DEC_ICRS', 'dec_icrs']:
                if dec_col in row.colnames:
                    dec_str = row[dec_col]
                    logger.debug(f"Found DEC in column: {dec_col} = {dec_str}")
                    break

            if ra_str is None or dec_str is None:
                logger.error(f"Could not find RA/DEC in SIMBAD result. Available columns: {result.colnames}")
                self.error_occurred.emit(f"Could not extract coordinates from SIMBAD result")
                return

            # Extract coordinates - SIMBAD returns RA/DEC in degrees
            coords = SkyCoord(
                ra=ra_str,
                dec=dec_str,
                unit=(u.deg, u.deg),
                frame='icrs'
            )

            ra_deg = coords.ra.degree
            dec_deg = coords.dec.degree

            # Extract other fields - use lowercase column names
            main_id = str(row['main_id']) if 'main_id' in row.colnames else str(row.get('MAIN_ID', 'Unknown'))
            obj_type = str(row['otype']) if 'otype' in row.colnames else (str(row['OTYPE']) if 'OTYPE' in row.colnames else None)

            # Get magnitude - try V magnitude first, then B
            magnitude = None
            for mag_col in ['V', 'v', 'B', 'b', 'FLUX_V', 'flux_V', 'FLUX_B', 'flux_B']:
                if mag_col in row.colnames and row[mag_col] is not None:
                    try:
                        magnitude = float(row[mag_col])
                        logger.debug(f"Found magnitude in column {mag_col}: {magnitude}")
                        break
                    except (ValueError, TypeError):
                        pass

            # Get dimensions (in arcminutes)
            size_maj = None
            size_min = None
            for maj_col in ['galdim_majaxis', 'GALDIM_MAJAXIS']:
                if maj_col in row.colnames and row[maj_col] is not None:
                    try:
                        size_maj = float(row[maj_col])
                        logger.debug(f"Found major axis in column {maj_col}: {size_maj}")
                        break
                    except (ValueError, TypeError):
                        pass

            for min_col in ['galdim_minaxis', 'GALDIM_MINAXIS']:
                if min_col in row.colnames and row[min_col] is not None:
                    try:
                        size_min = float(row[min_col])
                        logger.debug(f"Found minor axis in column {min_col}: {size_min}")
                        break
                    except (ValueError, TypeError):
                        pass

            # Map SIMBAD object type to DSO type
            dso_type = self._map_simbad_type(obj_type)

            logger.debug(f"SIMBAD found: {main_id} at RA={ra_deg}, Dec={dec_deg}, Type={obj_type}")

            # Create object data dictionary
            object_data = {
                'main_id': main_id,
                'ra_deg': ra_deg,
                'dec_deg': dec_deg,
                'magnitude': magnitude,
                'size_maj': size_maj,
                'size_min': size_min,
                'dso_type': dso_type,
                'obj_type': obj_type
            }

            self.object_found.emit(object_data)

        except SimbadUnavailable as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            logger.error(f"Error querying SIMBAD: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error querying SIMBAD:\n\n{str(e)}")

    # SIMBAD object type codes (current short codes and legacy long forms) -> DSO type
    _SIMBAD_TYPE_MAP = {
        **dict.fromkeys(['g', 'gig', 'gic', 'bic', 'gip', 'hzg', 'agn', 'syg', 'sy1', 'sy2', 'lin',
                         'liner', 'seyfert', 'qso', 'bla', 'bll', 'rg', 'h2g', 'lsb', 'emg', 'sbg',
                         'bcg', 'ig', 'pag', 'galaxy'], 'GALXY'),
        **dict.fromkeys(['clg', 'grg', 'cgg', 'scg', 'pcg'], 'GALCL'),
        **dict.fromkeys(['pn', 'plnnb'], 'PLNNB'),
        **dict.fromkeys(['snr', 'snrem'], 'SNREM'),
        **dict.fromkeys(['hii', 'gne', 'brtnb', 'emobj'], 'BRTNB'),
        **dict.fromkeys(['rne', 'refnb'], 'REFNB'),
        **dict.fromkeys(['dne', 'drknb'], 'DRKNB'),
        **dict.fromkeys(['opc', 'opcl', 'cl*', 'opncl'], 'OPNCL'),
        **dict.fromkeys(['glc', 'glcl', 'globc', 'glocl'], 'GLOCL'),
        **dict.fromkeys(['as*', 'assoc', 'assc'], 'ASSC'),
    }

    def _map_simbad_type(self, simbad_type):
        """Map SIMBAD object type to DSO type used in database.

        Matches whole type codes: substring checks misfire on SIMBAD's short
        codes (e.g. 'G' would match GlC globular clusters and GNe nebulae)."""
        if not simbad_type:
            return None

        # Candidates are suffixed with '?' (e.g. "PN?")
        code = simbad_type.strip().rstrip('?').lower()
        dso_type = self._SIMBAD_TYPE_MAP.get(code)
        if dso_type is None and 'neb' in code and 'cl' in code:
            dso_type = 'CL+NB'
        return dso_type


# --- Loading Dialog ---
class SimbadLoadingDialog(QDialog):
    """Simple loading dialog to show while querying SIMBAD"""

    def __init__(self, search_term, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Searching SIMBAD")
        self.setModal(True)
        self.setFixedSize(400, 100)

        # Remove close button
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        layout = QVBoxLayout()

        # Message label
        message = QLabel(f"Searching for '{search_term}' in SIMBAD database...\n\nPlease wait...")
        message.setAlignment(Qt.AlignCenter)
        themed_style(message, lambda: f"font-size: {font_size(12)};")
        layout.addWidget(message)

        self.setLayout(layout)
