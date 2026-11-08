"""Process setup shared by the app and the command line, done before numpy or
any network module loads: importing this does it.

The app's own setup (Chromium flags, stderr log, ...) is in app/Bootstrap.py,
which imports this first."""

import logging
import os
import sys

logger = logging.getLogger(__name__)


# numpy's math library (OpenBLAS) reserves a buffer for every CPU thread when numpy is
# first imported: about 750 MB on a 32-thread PC, held for as long as the app runs. The
# app's numpy work is small matrices and per-pixel math, which doesn't use more than a
# few threads, so cap it at 8 (about 240 MB). Set before anything imports numpy; a value
# already in the environment wins.
if (os.cpu_count() or 1) > 8:
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

# Configure SSL certificates BEFORE any network imports (CRITICAL for PyInstaller)
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle - configure SSL first
    try:
        import certifi
        cert_path = os.path.join(sys._MEIPASS, 'certifi', 'cacert.pem')
        if not os.path.exists(cert_path):
            cert_path = certifi.where()  # Fallback to certifi's default path
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['CURL_CA_BUNDLE'] = cert_path
    except Exception as e:
        print(f"Warning: Could not configure SSL certificates: {e}", file=sys.stderr)

    # astroquery's cache, in a folder a bundle can write to
    try:
        import tempfile
        _cache_dir = os.path.join(tempfile.gettempdir(), 'astroquery_cache')
        os.makedirs(_cache_dir, exist_ok=True)
        os.environ['XDG_CACHE_HOME'] = _cache_dir
    except Exception as e:
        print(f"Warning: Could not configure astroquery cache: {e}", file=sys.stderr)
