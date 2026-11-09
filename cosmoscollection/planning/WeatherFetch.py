"""
Weather forecasts for the planning tools: fetching from Open-Meteo (and the
optional OpenWeather, VisualCrossing and WeatherAPI), the astro score and
seeing estimate per night, sun altitudes and the Moon's phase.

Used by the Weather Forecast window, the tray, Session Manager, the DSO
Visibility Calculator's calendar and the command line ('weather'). No UI
imports - QSettings only, for the API keys.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time

from cosmoscollection.core.OnlineLimits import OFFLINE_NOTE, is_offline
from cosmoscollection.core.Settings import app_settings
from cosmoscollection.planning.NightTimes import DARK_SKY_ALTITUDE

logger = logging.getLogger(__name__)


def _get_openweather_signature() -> Tuple[bool, str]:
    """Return (enabled, api_key) reflecting the current OpenWeather integration
    settings. Used to detect when cached weather data was fetched under a
    different OpenWeather configuration (e.g. the user just enabled the
    integration and added an API key) so a stale cache doesn't hide the change."""
    settings = app_settings()
    enabled = settings.value("openweather_integration_enabled", False, type=bool)
    api_key = settings.value("openweather_api_key", "", type=str)
    return (enabled, api_key)


def _get_visualcrossing_signature() -> Tuple[bool, str]:
    """Return (enabled, api_key) reflecting the current VisualCrossing integration
    settings. Used to detect when cached weather data was fetched under a
    different VisualCrossing configuration (e.g. the user just enabled the
    integration and added an API key) so a stale cache doesn't hide the change."""
    settings = app_settings()
    enabled = settings.value("visualcrossing_integration_enabled", False, type=bool)
    api_key = settings.value("visualcrossing_api_key", "", type=str)
    return (enabled, api_key)


def _get_weatherapi_signature() -> Tuple[bool, str]:
    """Return (enabled, api_key) reflecting the current WeatherAPI.com integration
    settings. Used to detect when cached weather data was fetched under a
    different WeatherAPI configuration (e.g. the user just enabled the
    integration and added an API key) so a stale cache doesn't hide the change."""
    settings = app_settings()
    enabled = settings.value("weatherapi_integration_enabled", False, type=bool)
    api_key = settings.value("weatherapi_api_key", "", type=str)
    return (enabled, api_key)


def test_openweather_key(api_key: str, timeout: int = 10) -> Tuple[bool, str]:
    """Test whether an OpenWeather API key is valid via a lightweight current-weather
    request. Used by the Settings dialog's "Test API Key" button.

    Returns:
        tuple: (success: bool, message: str)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Please enter an API key first."
    if is_offline():
        return False, OFFLINE_NOTE

    try:
        # Any fixed coordinates work here - we only care whether the key is accepted.
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=51.5074&lon=-0.1278&appid={api_key}"
        # See the comment on the Open-Meteo request in WeatherWorker.run() -- verification
        # is left on its default (True) deliberately.
        response = requests.get(url, timeout=timeout)

        if response.status_code == 401:
            return False, (
                "Invalid API key. Please double-check your key.\n\n"
                "Note: newly created OpenWeather keys can take a few minutes to activate."
            )
        response.raise_for_status()
        response.json()  # confirm the response is valid JSON
        return True, "Connection successful! Your OpenWeather API key is valid."

    except requests.exceptions.Timeout:
        return False, "Connection timed out. Please check your network connection."
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"


def test_visualcrossing_key(api_key: str, timeout: int = 10) -> Tuple[bool, str]:
    """Test whether a VisualCrossing API key is valid via a lightweight current-conditions
    request. Used by the Settings dialog's "Test API Key" button.

    Returns:
        tuple: (success: bool, message: str)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Please enter an API key first."
    if is_offline():
        return False, OFFLINE_NOTE

    try:
        # Any fixed coordinates work here - we only care whether the key is accepted.
        # include=current keeps this to the smallest possible record cost against the
        # free tier's daily budget (a "today"/hours request would cost ~24 records).
        url = (
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/51.5074,-0.1278/today?unitGroup=metric&include=current&key={api_key}&contentType=json"
        )
        # See the comment on the Open-Meteo request in WeatherWorker.run() -- verification
        # is left on its default (True) deliberately.
        response = requests.get(url, timeout=timeout)

        # VisualCrossing returns a plain-text error body (not JSON) on failure, and an
        # invalid/unauthorized key has been observed as either 401 or 400 depending on
        # the failure reason, so treat any non-2xx as a failed key check rather than
        # only checking for one specific status code.
        if not response.ok:
            body_preview = (response.text or "").strip()[:200]
            if response.status_code in (401, 403) or "key" in body_preview.lower():
                return False, (
                    "Invalid API key. Please double-check your key.\n\n"
                    "Note: newly created VisualCrossing keys can take a few minutes to activate."
                )
            return False, f"Connection failed: {response.status_code} {body_preview}"

        response.json()  # confirm the response is valid JSON
        return True, "Connection successful! Your VisualCrossing API key is valid."

    except requests.exceptions.Timeout:
        return False, "Connection timed out. Please check your network connection."
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"


def test_weatherapi_key(api_key: str, timeout: int = 10) -> Tuple[bool, str]:
    """Test whether a WeatherAPI.com API key is valid via a lightweight current-conditions
    request. Used by the Settings dialog's "Test API Key" button.

    Returns:
        tuple: (success: bool, message: str)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Please enter an API key first."
    if is_offline():
        return False, OFFLINE_NOTE

    try:
        # Any fixed coordinates work here - we only care whether the key is accepted.
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q=51.5074,-0.1278"
        # See the comment on the Open-Meteo request in WeatherWorker.run() -- verification
        # is left on its default (True) deliberately.
        response = requests.get(url, timeout=timeout)

        if not response.ok:
            # WeatherAPI returns a JSON body of the form {"error": {"code": ..., "message": ...}}
            # on failure. Error code 2006 is specifically an invalid key; other 4xx codes
            # (missing key, quota exceeded, disabled key, etc.) are reported as-is.
            error_message = None
            try:
                error_message = response.json().get("error", {}).get("message")
            except ValueError:
                pass
            if response.status_code == 401 or "invalid" in (error_message or "").lower():
                return False, (
                    "Invalid API key. Please double-check your key.\n\n"
                    "Note: newly created WeatherAPI keys can take a few minutes to activate."
                )
            return False, f"Connection failed: {error_message or f'HTTP {response.status_code}'}"

        response.json()  # confirm the response is valid JSON
        return True, "Connection successful! Your WeatherAPI API key is valid."

    except requests.exceptions.Timeout:
        return False, "Connection timed out. Please check your network connection."
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"


@dataclass
class MoonPhaseData:
    """Moon phase information for a date"""
    phase_angle: float      # 0-180 degrees (elongation from sun)
    illumination: float     # 0-100 percentage
    phase_name: str         # "New Moon", "Waxing Crescent", etc.
    phase_emoji: str        # Moon phase emoji


@dataclass
class HourlyWeatherData:
    """Dataclass for hourly weather data"""
    time: datetime
    cloud_cover: float  # Total cloud cover %
    cloud_cover_low: float
    cloud_cover_mid: float
    cloud_cover_high: float
    temperature: float  # Celsius
    dew_point: float  # Celsius
    humidity: float  # %
    wind_speed: float  # km/h
    precipitation_probability: float  # %
    wind_gusts: float = 0.0  # km/h — peak gust in the hour
    visibility: Optional[float] = None  # meters
    surface_pressure: Optional[float] = None  # hPa
    openweather_blended: bool = False  # True if this hour was averaged with OpenWeather data
    visualcrossing_blended: bool = False  # True if this hour was averaged with VisualCrossing data
    weatherapi_blended: bool = False  # True if this hour was averaged with WeatherAPI data


@dataclass
class DailyWeatherSummary:
    """Dataclass for daily aggregated weather data"""
    date: datetime
    hourly_data: List[HourlyWeatherData]
    avg_cloud_cover: float
    min_cloud_cover: float
    max_cloud_cover: float
    avg_temperature: float
    min_temperature: float
    max_temperature: float
    avg_humidity: float
    avg_wind_speed: float
    max_wind_speed: float
    avg_precipitation_prob: float
    tonight_avg_cloud_cover: float  # avg cloud cover for dark hours only (sun_alt < -12°)
    astro_score: int  # 0-100
    seeing_estimate: str  # Excellent/Good/Moderate/Poor
    moon_phase: Optional[MoonPhaseData] = None
    dark_hours_start: Optional[datetime] = None  # First dark hour (sun_alt < -12°)
    dark_hours_end: Optional[datetime] = None  # Last dark hour (sun_alt < -12°)
    openweather_blended: bool = False  # True if any hour this day was blended with OpenWeather data
    visualcrossing_blended: bool = False  # True if any hour this day was blended with VisualCrossing data
    weatherapi_blended: bool = False  # True if any hour this day was blended with WeatherAPI data


class WeatherUnavailable(Exception):
    """No forecast: offline, or no weather source answered with usable data."""


class WeatherFetch:
    """Fetches the 7-day forecast for a location from Open-Meteo, blended
    with OpenWeather, VisualCrossing and WeatherAPI when they're turned on in
    Settings, and turns it into DailyWeatherSummary nights with astro scores.

        summaries = WeatherFetch(lat, lon, 'America/New_York').fetch()

    After fetching, the *_error attributes say which sources failed and
    baseline_source which one the hourly grid came from."""

    def __init__(self, lat: float, lon: float, timezone: str = None, progress=None):
        self.lat = lat
        self.lon = lon
        self.timezone = timezone
        self.progress = progress  # progress(message), or None
        # Set by _fetch_openweather_data() / _fetch_visualcrossing_data() / _fetch_weatherapi_data()
        # when that integration is enabled but a fetch attempt fails (bad key, network
        # error, etc.) - None otherwise, including when the integration is simply
        # disabled/unconfigured (not an error).
        self.openweather_error: Optional[str] = None
        self.visualcrossing_error: Optional[str] = None
        self.weatherapi_error: Optional[str] = None
        # Set if the primary Open-Meteo fetch itself fails (network error, invalid
        # response, etc.). Unlike the three above, Open-Meteo failing doesn't stop
        # the fetch - the other enabled sources are still tried, and if any of them
        # succeeded, one gets promoted to the baseline hourly grid (see
        # _process_weather_data / baseline_source) instead of leaving the forecast
        # blank just because Open-Meteo was unreachable.
        self.openmeteo_error: Optional[str] = None
        self.baseline_source: str = "openmeteo"

    def _say(self, message):
        if self.progress:
            self.progress(message)

    def fetch_sources(self):
        """(Open-Meteo, OpenWeather, VisualCrossing, WeatherAPI) raw data, each
        None if turned off or it failed - falling back to whichever other
        enabled source(s) are reachable if Open-Meteo itself fails: one source
        failing (bad DNS, network outage, etc.) should never blank out the whole
        forecast when another enabled source could still serve it. Raises
        WeatherUnavailable when offline or none answered."""
        if is_offline():
            raise WeatherUnavailable(f"The weather forecast needs the internet. {OFFLINE_NOTE}")
        data = self._fetch_openmeteo_data()

        # Optionally supplement with OpenWeather, VisualCrossing, and/or WeatherAPI
        # data (all opt-in, requires an API key each). Each returns None if
        # disabled/unconfigured/unavailable. These are always attempted regardless
        # of whether the Open-Meteo fetch above succeeded - if it didn't, the best
        # of these gets promoted to the baseline hourly grid instead of merely
        # being blended into it (see _process_weather_data / baseline_source).
        openweather_data = self._fetch_openweather_data()
        visualcrossing_data = self._fetch_visualcrossing_data()
        weatherapi_data = self._fetch_weatherapi_data()

        if data is None and not (openweather_data or visualcrossing_data or weatherapi_data):
            raise WeatherUnavailable(
                f"Open-Meteo unavailable ({self.openmeteo_error}) and no other "
                "weather source is enabled or reachable."
            )
        return data, openweather_data, visualcrossing_data, weatherapi_data

    def process(self, data, openweather_data=None, visualcrossing_data=None, weatherapi_data=None):
        """The sources' data as daily summaries; WeatherUnavailable if none is usable."""
        daily_summaries = self._process_weather_data(data, openweather_data, visualcrossing_data, weatherapi_data)
        if not daily_summaries:
            raise WeatherUnavailable(
                f"Open-Meteo unavailable ({self.openmeteo_error}) and no other "
                "weather source returned usable forecast data."
            )
        return daily_summaries

    def fetch(self) -> List["DailyWeatherSummary"]:
        """The forecast's daily summaries; raises WeatherUnavailable, or
        requests' RequestException."""
        sources = self.fetch_sources()
        self._say("Processing weather data...")
        return self.process(*sources)

    def _fetch_openmeteo_data(self) -> Optional[Dict[str, Any]]:
        """Fetch the primary Open-Meteo forecast. Returns None (and sets
        self.openmeteo_error to a short reason) on any failure instead of raising,
        so a failed Open-Meteo fetch doesn't prevent run() from still trying the
        other enabled sources."""
        self.openmeteo_error = None
        try:
            self._say("Connecting to Open-Meteo API...")

            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={self.lat}&longitude={self.lon}&"
                f"hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
                f"temperature_2m,dew_point_2m,relative_humidity_2m,"
                f"wind_speed_10m,wind_gusts_10m,precipitation_probability,visibility,surface_pressure&"
                f"forecast_days=7&timezone=auto"
            )

            # Cert verification is left on its default (True) here and everywhere else in
            # this file. app/Bootstrap.py points SSL_CERT_FILE/REQUESTS_CA_BUNDLE at the bundled
            # certifi CA file before any network imports, so requests picks that up as an
            # explicit ca_certs path in both dev and frozen builds. A previous "verify =
            # not frozen" workaround here disabled verification in frozen builds without
            # even avoiding this call path: urllib3 still calls SSLContext.load_default_certs()
            # whenever no explicit ca_certs/ca_cert_dir/ca_cert_data was given, regardless of
            # verify=True/False, and that's exactly where a "Fatal Python error: Illegal
            # instruction" crash (in ssl.py load_default_certs, native/unrecoverable) hit
            # this thread in the field. Passing an explicit ca_certs path (via verify=True
            # + the env vars above) skips load_default_certs() entirely.
            self._say("Downloading weather forecast data...")

            # Open-Meteo occasionally returns a 200 with an empty/invalid body
            # (transient upstream hiccup). Retry once before giving up, since
            # a fresh request a moment later typically succeeds.
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError:
                    body_preview = response.text[:200] if response.text else "<empty>"
                    logger.warning(
                        f"Open-Meteo returned an unparseable response on attempt "
                        f"{attempt}/{max_attempts} (status {response.status_code}): {body_preview!r}"
                    )
                    if attempt == max_attempts:
                        self.openmeteo_error = "invalid response"
                        return None
                    time.sleep(1)

        except requests.exceptions.Timeout:
            self.openmeteo_error = "request timed out"
            logger.warning("Open-Meteo fetch failed: timed out, continuing with remaining source(s)")
        except requests.exceptions.RequestException as e:
            self.openmeteo_error = "network error"
            logger.warning(f"Open-Meteo fetch failed, continuing with remaining source(s): {e}")
        except Exception as e:
            self.openmeteo_error = "unexpected error"
            logger.warning(f"Open-Meteo fetch failed, continuing with remaining source(s): {e}")
        return None

    def _fetch_openweather_data(self) -> Optional[Dict[datetime, Dict[str, Optional[float]]]]:
        """Fetch supplemental forecast data from OpenWeather's free 5 day / 3 hour
        forecast API, if the integration is enabled and an API key is configured.

        Returns a dict keyed by forecast timestamp -> field values, or None if the
        integration is disabled/unconfigured (not an error - self.openweather_error
        stays None) or the request fails for any reason such as a bad key, network
        error, or rate limit (self.openweather_error is set to a short reason so
        callers can surface it, e.g. in the Weather Forecast window's status line).
        Either way, callers should treat None as "no supplemental data available"
        and fall back to Open-Meteo-only data.
        """
        self.openweather_error = None
        try:
            enabled, api_key = _get_openweather_signature()
            if not enabled or not api_key:
                return None

            url = (
                f"https://api.openweathermap.org/data/2.5/forecast?"
                f"lat={self.lat}&lon={self.lon}&units=metric&appid={api_key}"
            )
            # See the comment on the Open-Meteo request in run() -- verification is left
            # on its default (True) deliberately.
            # OpenWeather, like Open-Meteo, occasionally returns a 200 with an
            # empty/invalid body (transient upstream hiccup). Retry once before
            # giving up, since a fresh request a moment later typically succeeds.
            max_attempts = 2
            data = None
            for attempt in range(1, max_attempts + 1):
                response = requests.get(url, timeout=20)

                if response.status_code == 401:
                    self.openweather_error = "invalid API key"
                    logger.warning("OpenWeather fetch failed: invalid API key")
                    return None

                response.raise_for_status()
                try:
                    data = response.json()
                    break
                except ValueError:
                    body_preview = response.text[:200] if response.text else "<empty>"
                    logger.warning(
                        f"OpenWeather returned an unparseable response on attempt "
                        f"{attempt}/{max_attempts} (status {response.status_code}): {body_preview!r}"
                    )
                    if attempt == max_attempts:
                        self.openweather_error = "invalid response"
                        return None
                    time.sleep(1)

            result: Dict[datetime, Dict[str, Optional[float]]] = {}
            for entry in data.get("list", []):
                dt = datetime.fromtimestamp(entry["dt"])
                main = entry.get("main", {})
                wind = entry.get("wind", {})
                wind_speed = wind.get("speed")
                wind_gust = wind.get("gust")
                pop = entry.get("pop")
                result[dt] = {
                    "cloud_cover": entry.get("clouds", {}).get("all"),
                    "temperature": main.get("temp"),
                    "humidity": main.get("humidity"),
                    "surface_pressure": main.get("pressure"),
                    "wind_speed": wind_speed * 3.6 if wind_speed is not None else None,  # m/s -> km/h
                    "wind_gusts": wind_gust * 3.6 if wind_gust is not None else None,  # m/s -> km/h
                    "precipitation_probability": pop * 100 if pop is not None else None,
                    "visibility": entry.get("visibility"),
                }
            return result

        except requests.exceptions.Timeout:
            self.openweather_error = "request timed out"
            logger.warning("OpenWeather fetch failed: timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.openweather_error = "network error"
            logger.warning(f"OpenWeather fetch failed: network error: {e}")
            return None
        except Exception as e:
            self.openweather_error = "unexpected error"
            logger.warning(f"OpenWeather fetch failed, continuing with Open-Meteo only: {e}")
            return None

    def _fetch_visualcrossing_data(self) -> Optional[Dict[datetime, Dict[str, Optional[float]]]]:
        """Fetch supplemental forecast data from VisualCrossing's Timeline Weather API,
        if the integration is enabled and an API key is configured.

        Returns a dict keyed by forecast timestamp -> field values, or None if the
        integration is disabled/unconfigured (not an error - self.visualcrossing_error
        stays None) or the request fails for any reason such as a bad key, network
        error, or exhausted daily quota (self.visualcrossing_error is set to a short
        reason so callers can surface it, e.g. in the Weather Forecast window's status
        line). Either way, callers should treat None as "no supplemental data available"
        and fall back to whatever other source(s) are available.
        """
        self.visualcrossing_error = None
        try:
            enabled, api_key = _get_visualcrossing_signature()
            if not enabled or not api_key:
                return None

            # next7days matches Open-Meteo's forecast_days=7 window above. elements=
            # restricts the response to only the fields we actually use, keeping the
            # payload (and the free tier's daily record budget) small.
            url = (
                "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
                f"timeline/{self.lat},{self.lon}/next7days?"
                f"unitGroup=metric&include=hours&"
                f"elements=datetimeEpoch,cloudcover,temp,humidity,windspeed,windgust,"
                f"precipprob,visibility,pressure&"
                f"key={api_key}&contentType=json"
            )
            # See the comment on the Open-Meteo request in run() -- verification is left
            # on its default (True) deliberately.
            # VisualCrossing, like Open-Meteo, occasionally returns a 200 with an
            # empty/invalid body (transient upstream hiccup). Retry once before
            # giving up, since a fresh request a moment later typically succeeds.
            max_attempts = 2
            data = None
            for attempt in range(1, max_attempts + 1):
                response = requests.get(url, timeout=20)

                if not response.ok:
                    # VisualCrossing returns a plain-text error body (not JSON) on
                    # failure. An invalid/unauthorized key has been observed as either
                    # 401 or 400 depending on the failure reason.
                    body_preview = (response.text or "").strip()[:200]
                    if response.status_code in (401, 403) or "key" in body_preview.lower():
                        self.visualcrossing_error = "invalid API key"
                        logger.warning("VisualCrossing fetch failed: invalid API key")
                    elif response.status_code == 429:
                        self.visualcrossing_error = "daily quota exceeded"
                        logger.warning("VisualCrossing fetch failed: daily quota exceeded")
                    else:
                        self.visualcrossing_error = f"HTTP {response.status_code}"
                        logger.warning(f"VisualCrossing fetch failed: {response.status_code}: {body_preview!r}")
                    return None

                try:
                    data = response.json()
                    break
                except ValueError:
                    body_preview = response.text[:200] if response.text else "<empty>"
                    logger.warning(
                        f"VisualCrossing returned an unparseable response on attempt "
                        f"{attempt}/{max_attempts} (status {response.status_code}): {body_preview!r}"
                    )
                    if attempt == max_attempts:
                        self.visualcrossing_error = "invalid response"
                        return None
                    time.sleep(1)

            result: Dict[datetime, Dict[str, Optional[float]]] = {}
            for day in data.get("days", []):
                for entry in day.get("hours", []):
                    epoch = entry.get("datetimeEpoch")
                    if epoch is None:
                        continue
                    dt = datetime.fromtimestamp(epoch)
                    visibility_km = entry.get("visibility")
                    result[dt] = {
                        "cloud_cover": entry.get("cloudcover"),
                        "temperature": entry.get("temp"),
                        "humidity": entry.get("humidity"),
                        "surface_pressure": entry.get("pressure"),
                        "wind_speed": entry.get("windspeed"),  # already km/h (unitGroup=metric)
                        "wind_gusts": entry.get("windgust"),  # already km/h
                        "precipitation_probability": entry.get("precipprob"),  # already 0-100
                        "visibility": visibility_km * 1000 if visibility_km is not None else None,  # km -> m
                    }
            return result

        except requests.exceptions.Timeout:
            self.visualcrossing_error = "request timed out"
            logger.warning("VisualCrossing fetch failed: timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.visualcrossing_error = "network error"
            logger.warning(f"VisualCrossing fetch failed: network error: {e}")
            return None
        except Exception as e:
            self.visualcrossing_error = "unexpected error"
            logger.warning(f"VisualCrossing fetch failed, continuing without it: {e}")
            return None

    def _fetch_weatherapi_data(self) -> Optional[Dict[datetime, Dict[str, Optional[float]]]]:
        """Fetch supplemental forecast data from WeatherAPI.com's Forecast API, if the
        integration is enabled and an API key is configured.

        Returns a dict keyed by forecast timestamp -> field values, or None if the
        integration is disabled/unconfigured (not an error - self.weatherapi_error
        stays None) or the request fails for any reason such as a bad key, network
        error, or exceeded quota (self.weatherapi_error is set to a short reason so
        callers can surface it, e.g. in the Weather Forecast window's status line).
        Either way, callers should treat None as "no supplemental data available"
        and fall back to whatever other source(s) are available.
        """
        self.weatherapi_error = None
        try:
            enabled, api_key = _get_weatherapi_signature()
            if not enabled or not api_key:
                return None

            # days=7 matches Open-Meteo's forecast_days=7 window above. WeatherAPI's
            # free tier is limited to a 3-day forecast; it silently returns however
            # many days the key's plan allows rather than erroring, so a free-tier key
            # here simply yields fewer hours to blend for the later days (handled
            # gracefully below via find_nearest_match's tolerance window).
            url = (
                f"https://api.weatherapi.com/v1/forecast.json?"
                f"key={api_key}&q={self.lat},{self.lon}&days=7&aqi=no&alerts=no"
            )
            # See the comment on the Open-Meteo request in run() -- verification is left
            # on its default (True) deliberately.
            # WeatherAPI, like Open-Meteo, occasionally returns a 200 with an
            # empty/invalid body (transient upstream hiccup). Retry once before
            # giving up, since a fresh request a moment later typically succeeds.
            max_attempts = 2
            data = None
            for attempt in range(1, max_attempts + 1):
                response = requests.get(url, timeout=20)

                if not response.ok:
                    # WeatherAPI returns a JSON body of the form
                    # {"error": {"code": ..., "message": ...}} on failure.
                    error_message = None
                    try:
                        error_message = response.json().get("error", {}).get("message")
                    except ValueError:
                        pass
                    if response.status_code == 401 or "invalid" in (error_message or "").lower():
                        self.weatherapi_error = "invalid API key"
                        logger.warning("WeatherAPI fetch failed: invalid API key")
                    elif response.status_code == 403:
                        self.weatherapi_error = error_message or "quota exceeded"
                        logger.warning(f"WeatherAPI fetch failed: {self.weatherapi_error}")
                    else:
                        self.weatherapi_error = error_message or f"HTTP {response.status_code}"
                        logger.warning(f"WeatherAPI fetch failed: {self.weatherapi_error}")
                    return None

                try:
                    data = response.json()
                    break
                except ValueError:
                    body_preview = response.text[:200] if response.text else "<empty>"
                    logger.warning(
                        f"WeatherAPI returned an unparseable response on attempt "
                        f"{attempt}/{max_attempts} (status {response.status_code}): {body_preview!r}"
                    )
                    if attempt == max_attempts:
                        self.weatherapi_error = "invalid response"
                        return None
                    time.sleep(1)

            result: Dict[datetime, Dict[str, Optional[float]]] = {}
            for day in data.get("forecast", {}).get("forecastday", []):
                for entry in day.get("hour", []):
                    epoch = entry.get("time_epoch")
                    if epoch is None:
                        continue
                    dt = datetime.fromtimestamp(epoch)
                    visibility_km = entry.get("vis_km")
                    chance_of_rain = entry.get("chance_of_rain")
                    chance_of_snow = entry.get("chance_of_snow")
                    if chance_of_rain is None and chance_of_snow is None:
                        precip_prob = None
                    else:
                        precip_prob = max(chance_of_rain or 0, chance_of_snow or 0)
                    result[dt] = {
                        "cloud_cover": entry.get("cloud"),
                        "temperature": entry.get("temp_c"),
                        "humidity": entry.get("humidity"),
                        "surface_pressure": entry.get("pressure_mb"),
                        "wind_speed": entry.get("wind_kph"),  # already km/h
                        "wind_gusts": entry.get("gust_kph"),  # already km/h
                        "precipitation_probability": precip_prob,
                        "visibility": visibility_km * 1000 if visibility_km is not None else None,  # km -> m
                    }
            return result

        except requests.exceptions.Timeout:
            self.weatherapi_error = "request timed out"
            logger.warning("WeatherAPI fetch failed: timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.weatherapi_error = "network error"
            logger.warning(f"WeatherAPI fetch failed: network error: {e}")
            return None
        except Exception as e:
            self.weatherapi_error = "unexpected error"
            logger.warning(f"WeatherAPI fetch failed, continuing without it: {e}")
            return None

    def _process_weather_data(
        self,
        data: Optional[Dict[str, Any]],
        openweather_data: Optional[Dict[datetime, Dict[str, Optional[float]]]] = None,
        visualcrossing_data: Optional[Dict[datetime, Dict[str, Optional[float]]]] = None,
        weatherapi_data: Optional[Dict[datetime, Dict[str, Optional[float]]]] = None
    ) -> List[DailyWeatherSummary]:
        """Process raw API data into daily summaries, optionally blending in
        supplemental OpenWeather, VisualCrossing, and/or WeatherAPI data (simple
        average across whichever of Open-Meteo/OpenWeather/VisualCrossing/WeatherAPI
        have a value for a given field/hour). data may be None if the Open-Meteo
        fetch failed - in that case, the best available supplemental source is
        promoted to the baseline hourly grid instead (see self.baseline_source)."""
        hourly = data.get("hourly", {}) if data else {}
        times = hourly.get("time", [])

        self.baseline_source = "openmeteo"
        if not times:
            # Open-Meteo is unavailable (or returned no hours) - promote the best
            # remaining enabled source to the baseline grid rather than blending it
            # into an empty one, so a single failed API doesn't blank the forecast.
            # VisualCrossing/WeatherAPI are hourly (like Open-Meteo); OpenWeather's
            # free tier is only 3-hourly, so it's used only as a last resort.
            for name, source in (("visualcrossing", visualcrossing_data),
                                  ("weatherapi", weatherapi_data),
                                  ("openweather", openweather_data)):
                if not source:
                    continue
                sorted_dts = sorted(source.keys())
                times = [dt.isoformat() for dt in sorted_dts]
                hourly = {
                    "time": times,
                    "cloud_cover": [source[dt].get("cloud_cover") for dt in sorted_dts],
                    "temperature_2m": [source[dt].get("temperature") for dt in sorted_dts],
                    "relative_humidity_2m": [source[dt].get("humidity") for dt in sorted_dts],
                    "wind_speed_10m": [source[dt].get("wind_speed") for dt in sorted_dts],
                    "wind_gusts_10m": [source[dt].get("wind_gusts") for dt in sorted_dts],
                    "precipitation_probability": [source[dt].get("precipitation_probability") for dt in sorted_dts],
                    "visibility": [source[dt].get("visibility") for dt in sorted_dts],
                    "surface_pressure": [source[dt].get("surface_pressure") for dt in sorted_dts],
                    # cloud_cover_low/mid/high and dew_point have no equivalent in the
                    # other APIs and are left out here - they default to 0 below via
                    # the existing hourly.get(..., [0] * len(times)) fallback.
                }
                self.baseline_source = name
                # This source is now the baseline itself, not a supplemental match -
                # clear it so it isn't blended against its own values below.
                if name == "visualcrossing":
                    visualcrossing_data = None
                elif name == "weatherapi":
                    weatherapi_data = None
                elif name == "openweather":
                    openweather_data = None
                break

        if not times:
            return []

        def blend(om_value, field: str, *source_matches: Tuple[str, Optional[Dict[str, Optional[float]]]]):
            """Average an Open-Meteo value with the matching value from whichever
            supplemental sources have a value for this field/hour. Returns
            (value, {names of sources that contributed})."""
            values = [om_value]
            contributors = set()
            for source_name, match in source_matches:
                if match is None:
                    continue
                source_value = match.get(field)
                if source_value is None:
                    continue
                values.append(source_value)
                contributors.add(source_name)
            if not contributors:
                return om_value, contributors
            return sum(values) / len(values), contributors

        def find_nearest_match(source_data: Optional[Dict[datetime, Dict[str, Optional[float]]]],
                               dt: datetime, tolerance_minutes: int):
            """Find the nearest supplemental-source forecast entry to an Open-Meteo
            hourly timestamp, within a tolerance window."""
            if not source_data:
                return None
            best_match = None
            best_diff = None
            for source_dt, values in source_data.items():
                diff = abs((source_dt - dt).total_seconds())
                if diff <= tolerance_minutes * 60 and (best_diff is None or diff < best_diff):
                    best_match = values
                    best_diff = diff
            return best_match

        # Parse hourly data
        hourly_records: List[HourlyWeatherData] = []
        for i, time_str in enumerate(times):
            try:
                dt = datetime.fromisoformat(time_str)
                # OpenWeather's free tier is only 3-hourly, so it needs a wide
                # tolerance to find a nearby entry. VisualCrossing and WeatherAPI are
                # hourly (like Open-Meteo), so a tight tolerance is enough for a
                # precise match.
                ow_match = find_nearest_match(openweather_data, dt, tolerance_minutes=90)
                vc_match = find_nearest_match(visualcrossing_data, dt, tolerance_minutes=5)
                wa_match = find_nearest_match(weatherapi_data, dt, tolerance_minutes=5)
                sources = (("openweather", ow_match), ("visualcrossing", vc_match), ("weatherapi", wa_match))

                cloud_cover, c1 = blend(hourly.get("cloud_cover", [0] * len(times))[i] or 0, "cloud_cover", *sources)
                temperature, c2 = blend(hourly.get("temperature_2m", [0] * len(times))[i] or 0, "temperature", *sources)
                humidity, c3 = blend(hourly.get("relative_humidity_2m", [0] * len(times))[i] or 0, "humidity", *sources)
                wind_speed, c4 = blend(hourly.get("wind_speed_10m", [0] * len(times))[i] or 0, "wind_speed", *sources)
                wind_gusts, c5 = blend(hourly.get("wind_gusts_10m", [0] * len(times))[i] or 0, "wind_gusts", *sources)
                precip_prob, c6 = blend(
                    hourly.get("precipitation_probability", [0] * len(times))[i] or 0, "precipitation_probability", *sources
                )
                visibility_om = hourly.get("visibility", [None] * len(times))[i]
                visibility, c7 = blend(visibility_om or 0, "visibility", *sources) if visibility_om is not None else (visibility_om, set())
                pressure_om = hourly.get("surface_pressure", [None] * len(times))[i]
                surface_pressure, c8 = blend(pressure_om or 0, "surface_pressure", *sources) if pressure_om is not None else (pressure_om, set())

                contributors = c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8

                hourly_records.append(HourlyWeatherData(
                    time=dt,
                    cloud_cover=cloud_cover,
                    cloud_cover_low=hourly.get("cloud_cover_low", [0] * len(times))[i] or 0,
                    cloud_cover_mid=hourly.get("cloud_cover_mid", [0] * len(times))[i] or 0,
                    cloud_cover_high=hourly.get("cloud_cover_high", [0] * len(times))[i] or 0,
                    temperature=temperature,
                    dew_point=hourly.get("dew_point_2m", [0] * len(times))[i] or 0,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    precipitation_probability=precip_prob,
                    wind_gusts=wind_gusts,
                    visibility=visibility,
                    surface_pressure=surface_pressure,
                    openweather_blended="openweather" in contributors,
                    visualcrossing_blended="visualcrossing" in contributors,
                    weatherapi_blended="weatherapi" in contributors
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing hourly data at index {i}: {e}")
                continue

        # Group by date
        daily_data: Dict[datetime.date, List[HourlyWeatherData]] = {}
        for record in hourly_records:
            date = record.time.date()
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(record)

        # Calculate sun altitudes for all hourly records (vectorized for efficiency)
        all_times = [record.time for record in hourly_records]
        sun_altitudes = calculate_sun_altitudes(self.lat, self.lon, all_times, self.timezone)
        sun_alt_map = dict(zip(all_times, sun_altitudes))

        # Create daily summaries
        sorted_dates = sorted(daily_data.keys())
        daily_summaries: List[DailyWeatherSummary] = []
        for date_idx, date in enumerate(sorted_dates):
            hours = daily_data[date]
            if not hours:
                continue

            # Calculate aggregates for all hours (used for display)
            cloud_covers = [h.cloud_cover for h in hours]
            temps = [h.temperature for h in hours]
            humidities = [h.humidity for h in hours]
            winds = [h.wind_speed for h in hours]
            precip_probs = [h.precipitation_probability for h in hours]

            avg_cloud = sum(cloud_covers) / len(cloud_covers)
            avg_humidity = sum(humidities) / len(humidities)
            avg_wind = sum(winds) / len(winds)
            avg_precip = sum(precip_probs) / len(precip_probs)

            # Build "tonight" hours: evening dark hours of this day + morning dark hours of next day
            next_hours = daily_data[sorted_dates[date_idx + 1]] if date_idx + 1 < len(sorted_dates) else None
            tonight_hours = [h for h in night_hours(hours, next_hours) if sun_alt_map.get(h.time, 0) < DARK_SKY_ALTITUDE]

            # Determine dark hours start and end times
            dark_start = tonight_hours[0].time if tonight_hours else None
            dark_end = tonight_hours[-1].time if tonight_hours else None

            # Calculate astro score as the average of individual hourly scores
            if tonight_hours:
                hourly_scores = [
                    calculate_astro_score(h.cloud_cover, h.humidity, h.wind_speed,
                                          h.precipitation_probability, h.visibility,
                                          h.wind_gusts)
                    for h in tonight_hours
                ]
                astro_score = int(sum(hourly_scores) / len(hourly_scores))
            else:
                # No dark hours (e.g., polar summer) - score is 0
                astro_score = 0

            # Calculate tonight's average cloud cover (dark hours only)
            if tonight_hours:
                tonight_avg_cloud = sum(h.cloud_cover for h in tonight_hours) / len(tonight_hours)
            else:
                tonight_avg_cloud = avg_cloud  # Fallback to full day average

            # Estimate seeing based on tonight's hours
            if tonight_hours:
                night_humidity = sum(h.humidity for h in tonight_hours) / len(tonight_hours)
                night_wind = sum(h.wind_speed for h in tonight_hours) / len(tonight_hours)
                night_gusts = max(h.wind_gusts for h in tonight_hours)
                night_temp_dew_spread = sum(h.temperature - h.dew_point for h in tonight_hours) / len(tonight_hours)
                # Average surface pressure for tonight (filter out None values)
                pressure_values = [h.surface_pressure for h in tonight_hours if h.surface_pressure is not None]
                night_pressure = sum(pressure_values) / len(pressure_values) if pressure_values else None
            else:
                night_humidity = avg_humidity
                night_wind = avg_wind
                night_gusts = max(h.wind_gusts for h in hours)
                night_temp_dew_spread = sum(h.temperature - h.dew_point for h in hours) / len(hours)
                pressure_values = [h.surface_pressure for h in hours if h.surface_pressure is not None]
                night_pressure = sum(pressure_values) / len(pressure_values) if pressure_values else None

            seeing = estimate_seeing(night_humidity, night_wind, night_temp_dew_spread, night_pressure, night_gusts)

            # Calculate moon phase for this date
            moon_phase = calculate_moon_phase(datetime.combine(date, datetime.min.time()), self.timezone)

            summary = DailyWeatherSummary(
                date=datetime.combine(date, datetime.min.time()),
                hourly_data=hours,
                avg_cloud_cover=avg_cloud,
                min_cloud_cover=min(cloud_covers),
                max_cloud_cover=max(cloud_covers),
                avg_temperature=sum(temps) / len(temps),
                min_temperature=min(temps),
                max_temperature=max(temps),
                avg_humidity=avg_humidity,
                avg_wind_speed=avg_wind,
                max_wind_speed=max(winds),
                avg_precipitation_prob=avg_precip,
                tonight_avg_cloud_cover=tonight_avg_cloud,
                astro_score=astro_score,
                seeing_estimate=seeing,
                moon_phase=moon_phase,
                dark_hours_start=dark_start,
                dark_hours_end=dark_end,
                openweather_blended=any(h.openweather_blended for h in hours),
                visualcrossing_blended=any(h.visualcrossing_blended for h in hours),
                weatherapi_blended=any(h.weatherapi_blended for h in hours)
            )
            daily_summaries.append(summary)

        return daily_summaries


def calculate_astro_score(cloud_cover: float, humidity: float, wind_speed: float,
                          precip_prob: float, visibility: Optional[float] = None,
                          wind_gusts: float = 0.0) -> int:
    """
    Calculate an astrophotography suitability score (0-100).

    Cloud cover is the dominant factor - high cloud cover caps the maximum possible score
    since you cannot do astrophotography through clouds regardless of other conditions.

    Score caps based on cloud cover:
    - >80% clouds: max score 30 (Poor)
    - >60% clouds: max score 50 (Moderate)
    - >40% clouds: max score 70 (Good)

    Score caps based on wind gusts (applied after cloud caps):
    - >40 km/h gusts: max score 35 (severe — tracking essentially impossible)
    - >25 km/h gusts: max score 60 (moderate — long exposures degraded)

    Score caps based on precipitation probability (rain can cut a session short):
    - >50% chance: max score 40 (high risk of disruption)
    - >30% chance: max score 60 (moderate risk of disruption)

    Score caps based on humidity (dew/fogging risk on optics):
    - >90% humidity: max score 40 (severe dew/fog risk)
    - >80% humidity: max score 60 (elevated dew/fog risk)

    Base scoring weights:
    - Cloud cover (40%): lower is better
    - Transparency/Visibility (15%): higher is better (affects deep sky objects)
    - Humidity (15%): ideal 30-50%, affects transparency and dew risk
    - Wind speed (15%): under 15 km/h is good for tracking/guiding
    - Precipitation (15%): 0% is ideal

    Args:
        cloud_cover: Cloud cover percentage (0-100)
        humidity: Relative humidity percentage (0-100)
        wind_speed: Wind speed in km/h
        precip_prob: Precipitation probability percentage (0-100)
        visibility: Visibility in meters (None if unavailable)
        wind_gusts: Peak wind gust speed in km/h (gusts weighted 75% — intermittent)
    """
    # Cloud cover score (0-100, lower clouds = higher score)
    cloud_score = max(0, 100 - cloud_cover)

    # Visibility/Transparency score (0-100, higher visibility = better transparency)
    # Open-Meteo returns visibility in meters
    # Excellent: > 40km, Good: 20-40km, Moderate: 10-20km, Poor: < 10km
    if visibility is not None:
        visibility_km = visibility / 1000.0
        if visibility_km >= 40:
            visibility_score = 100
        elif visibility_km >= 20:
            # Linear interpolation from 70 to 100 between 20-40km
            visibility_score = 70 + (visibility_km - 20) * 1.5
        elif visibility_km >= 10:
            # Linear interpolation from 40 to 70 between 10-20km
            visibility_score = 40 + (visibility_km - 10) * 3
        else:
            # Below 10km, score drops more steeply
            visibility_score = max(0, visibility_km * 4)
    else:
        # If visibility data unavailable, use a neutral score
        visibility_score = 70

    # Humidity score (0-100, ideal around 40%)
    # High humidity reduces transparency and increases dew risk
    if humidity < 30:
        humidity_score = 70 + humidity  # Slightly penalize very dry
    elif humidity <= 50:
        humidity_score = 100  # Ideal range
    elif humidity <= 70:
        humidity_score = 100 - (humidity - 50) * 2  # 50-100 as humidity goes from 50-70
    else:
        humidity_score = max(0, 60 - (humidity - 70))  # Penalize high humidity

    # Wind score (0-100, under 15 km/h is good)
    # Gusts are intermittent so weighted at 75%; effective_wind >= sustained speed
    effective_wind = max(wind_speed, wind_gusts * 0.75)
    if effective_wind <= 10:
        wind_score = 100
    elif effective_wind <= 15:
        wind_score = 100 - (effective_wind - 10) * 4  # 80-100 range
    elif effective_wind <= 25:
        wind_score = 80 - (effective_wind - 15) * 4  # 40-80 range
    else:
        wind_score = max(0, 40 - (effective_wind - 25) * 2)

    # Precipitation score (0-100, 0% is ideal)
    precip_score = max(0, 100 - precip_prob * 2)

    # Weighted average
    total_score = (
        cloud_score * 0.40 +
        visibility_score * 0.15 +
        humidity_score * 0.15 +
        wind_score * 0.15 +
        precip_score * 0.15
    )

    # Apply cloud cover caps - high clouds should hard-limit the score
    # since astrophotography is impossible through heavy cloud cover
    if cloud_cover > 80:
        total_score = min(total_score, 30)  # Cap at Poor
    elif cloud_cover > 60:
        total_score = min(total_score, 50)  # Cap at low Moderate
    elif cloud_cover > 40:
        total_score = min(total_score, 70)  # Cap at Good

    # Apply wind gust caps — severe gusts ruin tracking regardless of sky clarity
    if wind_gusts > 40:
        total_score = min(total_score, 35)  # Severe gusts — tracking essentially impossible
    elif wind_gusts > 25:
        total_score = min(total_score, 60)  # Moderate gusts — long exposures degraded

    # Apply precipitation caps — meaningful rain risk should hard-limit the score
    # since a session can be cut short or ruined regardless of sky clarity elsewhere
    if precip_prob > 50:
        total_score = min(total_score, 40)  # High rain risk — session likely disrupted
    elif precip_prob > 30:
        total_score = min(total_score, 60)  # Moderate rain risk — plan for interruptions

    # Apply humidity caps — near-saturation humidity brings dew/fogging risk
    # that degrades a session regardless of sky clarity
    if humidity > 90:
        total_score = min(total_score, 40)  # Severe dew/fog risk
    elif humidity > 80:
        total_score = min(total_score, 60)  # Elevated dew/fog risk

    return int(min(100, max(0, total_score)))


def estimate_seeing(humidity: float, wind_speed: float, temp_dew_spread: float,
                    surface_pressure: Optional[float] = None,
                    wind_gusts: float = 0.0) -> str:
    """
    Estimate seeing quality based on atmospheric conditions.

    Seeing (atmospheric turbulence) is affected by:
    - Humidity: High humidity can indicate atmospheric instability
    - Wind: High winds cause turbulence, but moderate wind can indicate stable laminar flow
    - Temperature-dew spread: Indicates moisture content and potential for ground-level turbulence
    - Surface pressure: High, stable pressure generally means better seeing;
      low pressure systems bring unstable air masses

    Args:
        humidity: Relative humidity %
        wind_speed: Wind speed in km/h
        temp_dew_spread: Temperature minus dew point (larger = less moisture)
        surface_pressure: Surface pressure in hPa (None if unavailable)

    Returns:
        Seeing quality string: Excellent/Good/Moderate/Poor
    """
    score = 100

    # Humidity factor (lower is better for seeing)
    if humidity > 80:
        score -= 30
    elif humidity > 65:
        score -= 15
    elif humidity > 50:
        score -= 5

    # Wind factor (calm to light is best, strong winds cause turbulence)
    # However, very calm conditions can lead to ground-layer turbulence
    # Use the worse of sustained speed or gusts for seeing penalty
    gust_wind = max(wind_speed, wind_gusts)
    if gust_wind > 30:
        score -= 40  # Very high winds - severe turbulence
    elif gust_wind > 25:
        score -= 30
    elif gust_wind > 15:
        score -= 15
    elif gust_wind > 10:
        score -= 5
    elif wind_speed < 3:
        score -= 5  # Very calm - potential ground layer issues

    # Dew risk factor (larger spread = better, also indicates drier air column)
    if temp_dew_spread < 2:
        score -= 25  # High dew risk and moist air
    elif temp_dew_spread < 5:
        score -= 10
    elif temp_dew_spread > 15:
        score += 10  # Very dry air column - excellent
    elif temp_dew_spread > 10:
        score += 5  # Very safe from dew

    # Surface pressure factor
    # High pressure (>1020 hPa) typically indicates stable air mass = better seeing
    # Low pressure (<1000 hPa) indicates unstable weather systems = worse seeing
    # Normal range is roughly 980-1040 hPa
    if surface_pressure is not None:
        if surface_pressure >= 1025:
            score += 15  # Strong high pressure - excellent stability
        elif surface_pressure >= 1015:
            score += 10  # High pressure - good stability
        elif surface_pressure >= 1005:
            score += 0   # Normal pressure - neutral
        elif surface_pressure >= 995:
            score -= 10  # Lowish pressure - some instability
        else:
            score -= 20  # Low pressure system - unstable air

    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Moderate"
    else:
        return "Poor"


# (lowest astro score, rating) from best to worst
RATINGS = ((80, "Excellent"), (60, "Good"), (40, "Moderate"), (None, "Poor"))


def get_rating_label(score: int) -> str:
    """'Excellent', 'Good', 'Moderate' or 'Poor' for an astro score (RATINGS)."""
    return next(label for lowest, label in RATINGS if lowest is None or score >= lowest)


def weather_units():
    """(temperature, wind) units from Settings > Weather Units: "Celsius" or
    "Fahrenheit", "km/h", "mph" or "m/s"."""
    settings = app_settings()
    return (settings.value("temperature_unit", "Celsius", type=str),
            settings.value("wind_speed_unit", "km/h", type=str))


def convert_temperature(celsius, unit):
    """A forecast temperature (Celsius) in the unit weather_units() gives."""
    return celsius * 9 / 5 + 32 if unit == "Fahrenheit" else celsius


def convert_wind(kmh, unit):
    """A forecast wind speed (km/h) in the unit weather_units() gives."""
    if unit == "mph":
        return kmh * 0.621371
    if unit == "m/s":
        return kmh / 3.6
    return kmh


def night_hours(day_hours, next_day_hours=None):
    """A night's forecast hours: noon of its day to noon the next."""
    return [h for h in day_hours if h.time.hour >= 12] + [h for h in next_day_hours or () if h.time.hour < 12]


def calculate_sun_altitudes(lat: float, lon: float, times: List[datetime], timezone: str = None) -> List[float]:
    """
    Calculate sun altitude for each time point.

    Args:
        lat: Observer latitude in degrees
        lon: Observer longitude in degrees
        times: List of datetime objects for each hour (in local time)
        timezone: Optional timezone string (e.g., 'America/New_York') for converting local times to UTC

    Returns:
        List of sun altitudes in degrees for each time point
    """
    if not times:
        return []

    # Convert naive local times to UTC if timezone is provided
    if timezone:
        try:
            import pytz
            local_tz = pytz.timezone(timezone)
            utc_times = []
            for t in times:
                if t.tzinfo is None:
                    local_dt = local_tz.localize(t)
                    utc_dt = local_dt.astimezone(pytz.UTC)
                    utc_times.append(utc_dt.replace(tzinfo=None))  # astropy works with naive UTC
                else:
                    utc_times.append(t)
            times = utc_times
        except Exception as e:
            logger.warning(f"Could not convert times to UTC: {e}")

    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    astropy_times = Time([t.isoformat() for t in times])
    altaz_frame = AltAz(obstime=astropy_times, location=location)
    sun_altaz = get_sun(astropy_times).transform_to(altaz_frame)

    return sun_altaz.alt.deg.tolist()


def _get_moon_phase_name(phase_angle: float, is_waxing: bool) -> tuple:
    """
    Map phase angle (elongation) to moon phase name and emoji.

    Args:
        phase_angle: Elongation from sun in degrees (0-180)
        is_waxing: True if moon is waxing (getting brighter), False if waning

    Returns:
        Tuple of (phase_name, phase_emoji)
    """
    if phase_angle < 22.5:
        return ("New Moon", "🌑")
    elif phase_angle < 67.5:
        if is_waxing:
            return ("Waxing Crescent", "🌒")
        else:
            return ("Waning Crescent", "🌘")
    elif phase_angle < 112.5:
        if is_waxing:
            return ("First Quarter", "🌓")
        else:
            return ("Last Quarter", "🌗")
    elif phase_angle < 157.5:
        if is_waxing:
            return ("Waxing Gibbous", "🌔")
        else:
            return ("Waning Gibbous", "🌖")
    else:
        return ("Full Moon", "🌕")


def calculate_moon_phase(date: datetime, timezone: str = None) -> MoonPhaseData:
    """
    Calculate moon phase information for a given date.

    Args:
        date: The date to calculate moon phase for (naive datetime in local time)
        timezone: Timezone string (e.g., 'America/New_York') for converting to UTC

    Returns:
        MoonPhaseData with phase angle, illumination, name, and emoji
    """
    from astropy.coordinates import GeocentricTrueEcliptic

    # Convert local midnight to UTC if timezone is provided
    if timezone:
        try:
            import pytz
            local_tz = pytz.timezone(timezone)
            if date.tzinfo is None:
                local_dt = local_tz.localize(date)
                utc_dt = local_dt.astimezone(pytz.UTC)
                date = utc_dt.replace(tzinfo=None)  # astropy works with naive UTC
        except Exception as e:
            logger.warning(f"Could not convert moon phase time to UTC: {e}")

    # Use midnight of the date for calculation
    obs_time = Time(date.isoformat())

    # Get sun and moon positions
    sun = get_sun(obs_time)
    moon = get_body('moon', obs_time)

    # Calculate elongation (angular separation between sun and moon)
    elongation = sun.separation(moon)
    phase_angle = elongation.deg

    # Lit share: 0% at new moon (0°), 100% at full moon (180°)
    from cosmoscollection.planning.Visibility import DSOVisibilityCalculator
    illumination = DSOVisibilityCalculator.get_moon_illumination(obs_time, sun, moon) * 100

    # Determine if waxing or waning by comparing ecliptic longitudes
    # Moon is waxing when its ecliptic longitude is ahead of (greater than) the sun's
    sun_ecliptic = sun.transform_to(GeocentricTrueEcliptic(equinox=obs_time))
    moon_ecliptic = moon.transform_to(GeocentricTrueEcliptic(equinox=obs_time))

    sun_lon = sun_ecliptic.lon.deg
    moon_lon = moon_ecliptic.lon.deg

    # Calculate the difference (moon - sun), normalized to 0-360
    lon_diff = (moon_lon - sun_lon) % 360

    # If difference is 0-180, moon is ahead of sun = waxing
    # If difference is 180-360, moon is behind sun = waning
    is_waxing = lon_diff < 180

    # Get phase name and emoji
    phase_name, phase_emoji = _get_moon_phase_name(phase_angle, is_waxing)

    return MoonPhaseData(
        phase_angle=phase_angle,
        illumination=illumination,
        phase_name=phase_name,
        phase_emoji=phase_emoji
    )
