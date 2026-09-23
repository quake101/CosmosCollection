#!/usr/bin/env python3
"""
Plate Solver Module for Cosmos Collection
Provides plate solving functionality using ASTAP (local) or Astrometry.net API (online)
"""

import os
import sys
import shutil
import subprocess
import tempfile
import time
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from PySide6.QtCore import QObject, Signal, QThread

logger = logging.getLogger(__name__)


def test_astrometry_key(api_key: str, timeout: int = 30) -> Tuple[bool, str]:
    """Test whether an Astrometry.net API key is valid via the login endpoint.
    Used by the Settings dialog's "Test API Key" button.

    Returns:
        tuple: (success: bool, message: str)
    """
    import requests

    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Please enter an API key first."

    try:
        login_url = PlateSolver.ASTROMETRY_NET_API_URL + "login"
        response = requests.post(login_url, data={
            'request-json': json.dumps({"apikey": api_key})
        }, timeout=timeout)

        try:
            login_data = response.json()
        except Exception:
            return False, f"Failed to parse response: {response.text[:200]}"

        if login_data.get('status') != 'success':
            error_msg = login_data.get('errormessage', 'Unknown error')
            if 'apikey' in error_msg.lower():
                return False, "Invalid API key. Please double-check your key."
            return False, f"Astrometry.net login failed: {error_msg}"

        if not login_data.get('session'):
            return False, f"No session key in login response: {login_data}"

        return True, "Connection successful! Your Astrometry.net API key is valid."

    except requests.exceptions.Timeout:
        return False, "Connection timed out. Please check your network connection."
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"


# Formats neither ASTAP nor astrometry.net can read - converted to a temporary
# FITS file before solving
_CONVERT_TO_FITS_EXTENSIONS = ('.xisf',)


def _write_solver_fits(image_path: str, out_dir: str) -> str:
    """Convert an image the solvers can't read (XISF) into a 16-bit mono FITS
    file in out_dir and return its path. Luminance is enough for star matching
    and keeps the file small (a 5000x2850 RGB float stack becomes ~28 MB)."""
    import numpy as np
    from astropy.io import fits
    from XISFReader import read_xisf_pixels

    data = read_xisf_pixels(image_path)
    if data is None:
        raise ValueError("Could not read image pixels")

    lum = data.mean(axis=2) if data.ndim == 3 else data
    lum = np.nan_to_num(lum.astype(np.float32))
    lo, hi = float(lum.min()), float(lum.max())
    if hi > lo:
        lum = (lum - lo) / (hi - lo)
    u16 = (np.clip(lum, 0, 1) * 65535).astype(np.uint16)

    # XISF rows run top-down, FITS rows bottom-up. Flip so the solved WCS uses
    # the standard FITS orientation AnnotationOverlay expects (display top row
    # = highest FITS y), same as when ASTAP solves a PNG/JPG directly.
    out_path = os.path.join(out_dir, Path(image_path).stem + ".fits")
    fits.PrimaryHDU(np.flipud(u16)).writeto(out_path, overwrite=True)
    return out_path


class PlateSolveResult:
    """Container for plate solve results"""

    def __init__(self):
        self.success = False
        self.ra_center = None  # RA in degrees
        self.dec_center = None  # Dec in degrees
        self.field_width = None  # Field width in degrees
        self.field_height = None  # Field height in degrees
        self.rotation = None  # Rotation angle in degrees
        self.pixel_scale = None  # arcsec/pixel
        self.wcs_header = None  # WCS header dict
        self.solver_used = None  # 'ASTAP' or 'astrometry.net'
        self.error_message = None

    def __repr__(self):
        if self.success:
            return f"PlateSolveResult(RA={self.ra_center:.4f}, Dec={self.dec_center:.4f}, scale={self.pixel_scale:.2f}\"/px)"
        return f"PlateSolveResult(failed: {self.error_message})"


class PlateSolverWorker(QThread):
    """Background worker for plate solving"""

    solve_finished = Signal(object)  # Emits PlateSolveResult
    progress = Signal(str)  # Progress messages

    def __init__(self, image_path: str, hints: Dict[str, Any] = None):
        super().__init__()
        self.image_path = image_path
        self.hints = hints or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Run plate solving in background"""
        solver = PlateSolver()
        astap_error = None

        # Try ASTAP first
        self.progress.emit("Checking for ASTAP solver...")
        if solver.is_astap_available():
            self.progress.emit("Solving with ASTAP...")
            result = solver.solve_with_astap(self.image_path, self.hints)
            if result.success:
                self.solve_finished.emit(result)
                return
            astap_error = result.error_message
            self.progress.emit(f"ASTAP failed: {astap_error}")
        else:
            astap_error = "ASTAP not installed or not found"
            self.progress.emit("ASTAP not found, trying online solver...")

        if self._cancelled:
            result = PlateSolveResult()
            result.error_message = "Cancelled"
            self.solve_finished.emit(result)
            return

        # Fall back to astrometry.net - only with a usable API key
        key_problem = solver.astrometry_key_problem()
        if key_problem:
            result = PlateSolveResult()
            result.solver_used = 'astrometry.net'
            result.error_message = f"ASTAP: {astap_error}\n\nAstrometry.net: {key_problem}"
            logger.info(f"Skipping astrometry.net: {key_problem}")
            self.solve_finished.emit(result)
            return

        self.progress.emit("Solving with Astrometry.net (this may take a few minutes)...")
        result = solver.solve_with_astrometry_net(self.image_path, self.hints)

        # If astrometry.net also failed, include both error messages
        if not result.success and astap_error:
            astrometry_error = result.error_message
            result.error_message = f"ASTAP: {astap_error}\n\nAstrometry.net: {astrometry_error}"

        self.solve_finished.emit(result)


class PlateSolver:
    """
    Plate solver that tries ASTAP first, then falls back to astrometry.net API
    """

    ASTROMETRY_NET_API_URL = "http://nova.astrometry.net/api/"

    def __init__(self):
        self.astap_path = self._find_astap()
        self.astrometry_api_key = self._get_astrometry_api_key()
        if self.astap_path:
            logger.info(f"ASTAP found at: {self.astap_path}")
        else:
            logger.info("ASTAP not found - will use online solver only")

    def _get_astrometry_api_key(self) -> str:
        """Get astrometry.net API key from settings"""
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("CosmosCollection", "CosmosCollection")
            return settings.value("astrometry_api_key", "", type=str).strip()
        except Exception:
            return ""

    def astrometry_key_problem(self) -> Optional[str]:
        """Return why astrometry.net can't be used (no key, or a key it already
        rejected), or None if it's worth trying. Avoids a pointless login call -
        astrometry.net no longer accepts anonymous logins."""
        if not self.astrometry_api_key:
            return ("No API key set. Register free at nova.astrometry.net "
                    "and add your key in Settings to use the online solver.")
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("CosmosCollection", "CosmosCollection")
            if settings.value("astrometry_invalid_api_key", "", type=str) == self.astrometry_api_key:
                return ("Your API key was rejected by astrometry.net. "
                        "Check the key in Settings.")
        except Exception:
            pass
        return None

    def _remember_invalid_api_key(self):
        """Record that astrometry.net rejected the current key, so it isn't
        retried until the key is changed in Settings"""
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("CosmosCollection", "CosmosCollection")
            settings.setValue("astrometry_invalid_api_key", self.astrometry_api_key)
        except Exception:
            pass

    def _find_astap(self) -> Optional[str]:
        """Find ASTAP executable"""
        # First check user settings for custom path
        try:
            from PySide6.QtCore import QSettings
            settings = QSettings("CosmosCollection", "CosmosCollection")
            custom_path = settings.value("astap_path", "", type=str)
            if custom_path and os.path.isfile(custom_path):
                # If user specified astap.exe, check if astap_cli.exe exists in same directory
                # (astap_cli.exe is the command-line version we need)
                if custom_path.lower().endswith('astap.exe'):
                    cli_path = custom_path[:-9] + 'astap_cli.exe'  # Replace astap.exe with astap_cli.exe
                    if os.path.isfile(cli_path):
                        logger.info(f"Using astap_cli.exe instead of astap.exe: {cli_path}")
                        return cli_path
                return custom_path
        except Exception:
            pass

        # Common installation paths
        if sys.platform == 'win32':
            possible_paths = [
                r"C:\Program Files\astap\astap_cli.exe",
                r"C:\Program Files\astap\astap.exe",
                r"C:\Program Files (x86)\astap\astap_cli.exe",
                r"C:\Program Files (x86)\astap\astap.exe",
                os.path.expanduser(r"~\AppData\Local\astap\astap_cli.exe"),
                os.path.expanduser(r"~\AppData\Local\astap\astap.exe"),
            ]
        elif sys.platform == 'darwin':
            possible_paths = [
                "/Applications/ASTAP.app/Contents/MacOS/astap_cli",
                "/Applications/ASTAP.app/Contents/MacOS/astap",
                "/usr/local/bin/astap_cli",
                "/usr/local/bin/astap",
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/astap_cli",
                "/usr/bin/astap",
                "/usr/local/bin/astap_cli",
                "/usr/local/bin/astap",
                "/opt/astap/astap_cli",
                "/opt/astap/astap",
                os.path.expanduser("~/astap/astap_cli"),
                os.path.expanduser("~/astap/astap"),
            ]

        # Check PATH first
        import shutil
        for exe_name in ["astap_cli", "astap"]:
            astap_in_path = shutil.which(exe_name)
            if astap_in_path:
                return astap_in_path

        # Check common locations
        for path in possible_paths:
            if os.path.isfile(path):
                return path

        return None

    def is_astap_available(self) -> bool:
        """Check if ASTAP is available"""
        return self.astap_path is not None

    def solve_with_astap(self, image_path: str, hints: Dict[str, Any] = None) -> PlateSolveResult:
        """
        Solve image using ASTAP

        Args:
            image_path: Path to image file
            hints: Optional dict with 'ra', 'dec', 'radius' (search radius in degrees)

        Returns:
            PlateSolveResult
        """
        result = PlateSolveResult()
        result.solver_used = 'ASTAP'

        # Check for existing WCS file first (cached solution)
        wcs_file = Path(image_path).with_suffix('.wcs')
        if wcs_file.exists():
            logger.info(f"Found existing WCS file: {wcs_file}")
            result.wcs_header = self._parse_wcs_file(wcs_file)
            if result.wcs_header and result.wcs_header.get('CRVAL1') is not None:
                result.success = True
                result.ra_center = result.wcs_header.get('CRVAL1')
                result.dec_center = result.wcs_header.get('CRVAL2')

                result.pixel_scale = self._pixel_scale_from_wcs(result.wcs_header)

                logger.info(f"Loaded cached WCS: RA={result.ra_center}, Dec={result.dec_center}, scale={result.pixel_scale}")
                return result

        if not self.astap_path:
            result.error_message = "ASTAP not found"
            return result

        temp_dir = None
        try:
            # ASTAP can't read XISF - solve a temporary FITS copy instead. ASTAP
            # writes its .wcs/.ini next to the file it solves, so those land in
            # the temp dir and the .wcs gets copied next to the original.
            solve_path = image_path
            if Path(image_path).suffix.lower() in _CONVERT_TO_FITS_EXTENSIONS:
                temp_dir = tempfile.mkdtemp(prefix="cosmos_solve_")
                logger.info(f"Converting {image_path} to FITS for ASTAP")
                solve_path = _write_solver_fits(image_path, temp_dir)

            # Build command
            cmd = [self.astap_path, "-f", solve_path]

            # Add hints if provided
            hints = hints or {}
            if 'ra' in hints and 'dec' in hints:
                cmd.extend(["-ra", str(hints['ra'] / 15.0)])  # Convert to hours
                cmd.extend(["-spd", str(hints['dec'] + 90)])  # Convert to SPD
            if 'radius' in hints:
                cmd.extend(["-r", str(hints['radius'])])
            else:
                cmd.extend(["-r", "30"])  # Default 30 degree search radius

            # Run ASTAP
            logger.info(f"Running ASTAP: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Log ASTAP output for debugging
            logger.info(f"ASTAP return code: {process.returncode}")
            if process.stdout:
                logger.info(f"ASTAP stdout: {process.stdout[:500]}")
            if process.stderr:
                logger.warning(f"ASTAP stderr: {process.stderr[:500]}")

            # Check for WCS file
            ini_file = Path(solve_path).with_suffix('.ini')
            solved_wcs = Path(solve_path).with_suffix('.wcs')
            if temp_dir and solved_wcs.exists():
                # Cache the solution next to the original image
                shutil.copyfile(solved_wcs, wcs_file)

            logger.info(f"Checking for WCS file: {wcs_file} - exists: {wcs_file.exists()}")

            if wcs_file.exists():
                result.success = True
                result.wcs_header = self._parse_wcs_file(wcs_file)

                # Extract key values from WCS
                if result.wcs_header:
                    result.ra_center = result.wcs_header.get('CRVAL1')
                    result.dec_center = result.wcs_header.get('CRVAL2')

                    result.pixel_scale = self._pixel_scale_from_wcs(result.wcs_header)

                    logger.info(f"ASTAP solve successful: RA={result.ra_center}, Dec={result.dec_center}, scale={result.pixel_scale}")

                # Clean up INI file only (keep WCS for caching)
                try:
                    if ini_file.exists():
                        ini_file.unlink()
                except:
                    pass
            else:
                # astap_cli usually prints nothing - the reason is in the ERROR=
                # line of the .ini it writes, which would otherwise be left behind
                error_detail = self._read_astap_ini_error(ini_file)
                if not error_detail:
                    error_detail = (process.stderr.strip() or process.stdout.strip()
                                    or f"No solution (exit code {process.returncode})")
                result.error_message = f"ASTAP solve failed: {error_detail}"
                logger.warning(f"ASTAP failed - no WCS file produced. Error: {error_detail}")
                try:
                    if ini_file.exists():
                        ini_file.unlink()
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            result.error_message = "ASTAP solve timed out"
        except Exception as e:
            result.error_message = f"ASTAP error: {str(e)}"
            logger.exception("ASTAP solve failed")
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    @staticmethod
    def _pixel_scale_from_wcs(wcs_header: Dict[str, Any]) -> Optional[float]:
        """Pixel scale in arcsec/pixel. Uses the CD matrix determinant, which
        holds at any rotation - abs(CD1_1) alone is only right for unrotated
        images (at ~90 degrees rotation it's near zero)."""
        if 'CD1_1' in wcs_header:
            det = (wcs_header['CD1_1'] * wcs_header.get('CD2_2', 0)
                   - wcs_header.get('CD1_2', 0) * wcs_header.get('CD2_1', 0))
            return (abs(det) ** 0.5) * 3600
        if 'CDELT1' in wcs_header:
            return abs(wcs_header['CDELT1']) * 3600
        return None

    @staticmethod
    def _read_astap_ini_error(ini_file: Path) -> Optional[str]:
        """Return the ERROR= (or WARNING=) message from an ASTAP .ini file, if any"""
        try:
            with open(ini_file, 'r', errors='replace') as f:
                values = dict(line.strip().split('=', 1) for line in f if '=' in line)
            return values.get('ERROR') or values.get('WARNING')
        except OSError:
            return None

    def _parse_wcs_file(self, wcs_path: Path) -> Dict[str, Any]:
        """Parse ASTAP WCS output file"""
        wcs_header = {}
        try:
            with open(wcs_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.split('/')[0].strip()  # Remove comments
                        value = value.strip("'\" ")

                        # Try to convert to number
                        try:
                            if '.' in value:
                                wcs_header[key] = float(value)
                            else:
                                wcs_header[key] = int(value)
                        except ValueError:
                            wcs_header[key] = value
        except Exception as e:
            logger.error(f"Failed to parse WCS file: {e}")

        return wcs_header

    def solve_with_astrometry_net(self, image_path: str, hints: Dict[str, Any] = None) -> PlateSolveResult:
        """
        Solve image using astrometry.net online API

        Args:
            image_path: Path to image file
            hints: Optional dict with 'ra', 'dec', 'radius', 'scale_low', 'scale_high'

        Returns:
            PlateSolveResult
        """
        import requests

        result = PlateSolveResult()
        result.solver_used = 'astrometry.net'

        key_problem = self.astrometry_key_problem()
        if key_problem:
            result.error_message = key_problem
            return result

        temp_dir = None
        try:
            # Step 1: Login
            api_key = self.astrometry_api_key
            self._log("Logging in to astrometry.net with API key...")
            login_url = self.ASTROMETRY_NET_API_URL + "login"
            login_response = requests.post(login_url, data={
                'request-json': json.dumps({"apikey": api_key})
            }, timeout=30)

            try:
                login_data = login_response.json()
                logger.info(f"Astrometry.net login response: {login_data}")
            except Exception as json_err:
                result.error_message = f"Failed to parse login response: {login_response.text[:200]}"
                logger.error(f"Login response parse error: {json_err}, response: {login_response.text[:500]}")
                return result

            if login_data.get('status') != 'success':
                error_msg = login_data.get('errormessage', 'Unknown error')
                if 'apikey' in error_msg.lower():
                    self._remember_invalid_api_key()
                    result.error_message = ("Your API key was rejected by astrometry.net. "
                                            "Check the key in Settings.")
                else:
                    result.error_message = f"Astrometry.net login failed: {error_msg}"
                logger.warning(f"Astrometry.net login failed: {login_data}")
                return result

            session_key = login_data.get('session')
            if not session_key:
                result.error_message = f"No session key in login response: {login_data}"
                logger.error(f"No session key in login response: {login_data}")
                return result
            logger.info(f"Astrometry.net session: {session_key[:20]}...")

            # Step 2: Upload image (astrometry.net can't read XISF - upload a
            # temporary FITS copy, which is also much smaller)
            if Path(image_path).suffix.lower() in _CONVERT_TO_FITS_EXTENSIONS:
                temp_dir = tempfile.mkdtemp(prefix="cosmos_solve_")
                logger.info(f"Converting {image_path} to FITS for upload")
                image_path = _write_solver_fits(image_path, temp_dir)

            self._log("Uploading image...")
            logger.info(f"Uploading image: {image_path}")
            upload_url = self.ASTROMETRY_NET_API_URL + "upload"

            # Check file size
            file_size = os.path.getsize(image_path)
            logger.info(f"Image file size: {file_size / 1024 / 1024:.2f} MB")

            # Prepare submission options
            submission_opts = {
                'session': session_key,
                'allow_commercial_use': 'n',
                'allow_modifications': 'n',
                'publicly_visible': 'n',
            }

            # Add hints
            hints = hints or {}
            if 'ra' in hints and 'dec' in hints:
                submission_opts['center_ra'] = hints['ra']
                submission_opts['center_dec'] = hints['dec']
                submission_opts['radius'] = hints.get('radius', 5)

            if 'scale_low' in hints and 'scale_high' in hints:
                submission_opts['scale_lower'] = hints['scale_low']
                submission_opts['scale_upper'] = hints['scale_high']
                submission_opts['scale_units'] = 'arcsecperpix'

            logger.info(f"Submission options: {submission_opts}")

            try:
                with open(image_path, 'rb') as f:
                    files = {'file': f}
                    data = {'request-json': json.dumps(submission_opts)}
                    logger.info("Sending upload request...")
                    upload_response = requests.post(upload_url, files=files, data=data, timeout=300)
                    logger.info(f"Upload response status: {upload_response.status_code}")
            except Exception as upload_error:
                result.error_message = f"Upload request failed: {str(upload_error)}"
                logger.exception("Upload request failed")
                return result

            try:
                upload_data = upload_response.json()
                logger.info(f"Upload response: {upload_data}")
            except Exception as json_error:
                result.error_message = f"Failed to parse upload response: {upload_response.text[:200]}"
                logger.error(f"Failed to parse upload response: {upload_response.text[:500]}")
                return result

            if upload_data.get('status') != 'success':
                result.error_message = f"Upload failed: {upload_data}"
                logger.warning(f"Upload failed: {upload_data}")
                return result

            submission_id = upload_data['subid']
            self._log(f"Submission ID: {submission_id}")
            logger.info(f"Submission ID: {submission_id}")

            # Step 3: Wait for job to complete
            self._log("Waiting for job assignment...")
            logger.info("Waiting for astrometry.net to assign a job...")
            job_id = None
            for attempt in range(60):  # Wait up to 5 minutes
                time.sleep(5)

                try:
                    status_url = self.ASTROMETRY_NET_API_URL + f"submissions/{submission_id}"
                    status_response = requests.get(status_url, timeout=30)
                    status_data = status_response.json()

                    jobs = status_data.get('jobs', [])
                    logger.debug(f"Poll {attempt+1}: jobs={jobs}, processing_started={status_data.get('processing_started')}")

                    if jobs:
                        job_id = jobs[0]
                        if job_id:
                            logger.info(f"Job assigned: {job_id}")
                            break

                    job_status = status_data.get('job_calibrations', [])
                    if job_status:
                        logger.info(f"Job calibrations found: {job_status}")
                        break

                    if attempt > 0 and attempt % 6 == 0:  # Every 30 seconds
                        self._log(f"Still waiting for job... ({attempt * 5}s)")
                except Exception as poll_error:
                    logger.warning(f"Poll error: {poll_error}")

            if not job_id:
                result.error_message = "Solve timed out waiting for job assignment"
                logger.warning("Timed out waiting for job assignment")
                return result

            # Step 4: Check job status
            self._log(f"Processing job {job_id}...")
            logger.info(f"Checking job {job_id} status...")
            for attempt in range(60):
                time.sleep(5)

                try:
                    job_url = self.ASTROMETRY_NET_API_URL + f"jobs/{job_id}"
                    job_response = requests.get(job_url, timeout=30)
                    job_data = job_response.json()

                    status = job_data.get('status')
                    logger.debug(f"Job {job_id} status: {status}")

                    if status == 'success':
                        logger.info(f"Job {job_id} completed successfully")
                        break
                    elif status == 'failure':
                        result.error_message = "Plate solve failed - no solution found"
                        logger.warning(f"Job {job_id} failed")
                        return result

                    if attempt > 0 and attempt % 6 == 0:  # Every 30 seconds
                        self._log(f"Still solving... ({attempt * 5}s)")
                except Exception as job_error:
                    logger.warning(f"Job status check error: {job_error}")

            # Step 5: Get calibration data
            self._log("Retrieving solution...")
            calib_url = self.ASTROMETRY_NET_API_URL + f"jobs/{job_id}/calibration"
            calib_response = requests.get(calib_url, timeout=30)
            calib_data = calib_response.json()

            result.success = True
            result.ra_center = calib_data.get('ra')
            result.dec_center = calib_data.get('dec')
            result.field_width = calib_data.get('width_arcsec', 0) / 3600
            result.field_height = calib_data.get('height_arcsec', 0) / 3600
            result.rotation = calib_data.get('orientation')
            result.pixel_scale = calib_data.get('pixscale')

            # Get WCS header
            wcs_url = f"http://nova.astrometry.net/wcs_file/{job_id}"
            wcs_response = requests.get(wcs_url, timeout=30)
            if wcs_response.status_code == 200:
                # Parse FITS WCS header
                result.wcs_header = self._parse_fits_wcs(wcs_response.content)

            logger.info(f"Astrometry.net solve successful: RA={result.ra_center}, Dec={result.dec_center}, scale={result.pixel_scale}")

        except requests.Timeout:
            result.error_message = "Astrometry.net request timed out"
        except Exception as e:
            result.error_message = f"Astrometry.net error: {str(e)}"
            logger.exception("Astrometry.net solve failed")
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return result

    def _parse_fits_wcs(self, wcs_content: bytes) -> Dict[str, Any]:
        """Parse FITS WCS header from astrometry.net"""
        wcs_header = {}
        try:
            from astropy.io import fits
            from io import BytesIO

            with fits.open(BytesIO(wcs_content)) as hdul:
                header = hdul[0].header
                for key in header:
                    if key and key.strip():
                        wcs_header[key] = header[key]
        except Exception as e:
            logger.error(f"Failed to parse FITS WCS: {e}")

        return wcs_header

    def _log(self, message: str):
        """Log a message"""
        logger.info(message)
