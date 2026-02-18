"""app.tools.get_weather
=========================

This module implements a very small weather lookup tool that can be
invoked by the OpenAI function-calling interface.  The tool uses the
`OpenMeteo <https://open-meteo.com/>`_ API, which is free, open source
and does not require an API key.  Two endpoints are used:

``geocoding-api.open-meteo.com/v1/search``
    Convert a city name into latitude/longitude coordinates.
``api.open-meteo.com/v1/forecast``
    Retrieve current weather and a daily forecast for the provided
    coordinates.

The public API of this module follows the same pattern as the
``create_file`` tool – a callable named :data:`func` that returns a JSON
string.  On success the JSON contains a ``result`` key; on failure it
contains an ``error`` key.  The tool is automatically discovered by
``app.tools.__init__``.

Example usage:
--------------

>>> import json
>>> from app.tools.get_weather import func as get_weather
>>> json.loads(get_weather("London", "2024-12-01"))
{'result': {'city': 'London', 'date': '2024-12-01', 'current': {'temperature': 6.2, 'windspeed': 5.5, 'winddirection': 210}, 'forecast': {'temperature_2m_max': 12.5, 'temperature_2m_min': 3.8, 'precipitation_sum': 0.0}}

The function accepts ``city`` as a free-form string and ``date`` as an
ISO 8601 date (e.g. ``YYYY-MM-DD``).  If ``date`` is omitted or empty
the current date is used.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _geocode_city(city: str) -> Tuple[float, float]:
    """Return latitude and longitude for a given city name.

    The function queries the OpenMeteo geocoding API and returns the
    first result.  It raises a :class:`ValueError` if the city cannot be
    found.
    """
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    if "results" not in data or not data["results"]:
        raise ValueError(f"City '{city}' not found")
    return data["results"][0]["latitude"], data["results"][0]["longitude"]


def _fetch_weather(lat: float, lon: float, date: str) -> Dict:
    """Fetch current and daily forecast weather data for the given coordinates and date.

    Parameters
    ----------
    lat, lon: float
        Latitude and longitude of the location.
    date: str
        ISO 8601 formatted date string (YYYY-MM-DD).  The API expects a
        single day, so ``start_date`` and ``end_date`` are identical.
    """
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "current_weather": "true",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
    }
    query = "?" + "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
    url = f"https://api.open-meteo.com/v1/forecast{query}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))

# ---------------------------------------------------------------------------
# The actual tool implementation
# ---------------------------------------------------------------------------

def _get_weather(city: str, date: str = "") -> str:
    """Retrieve current and forecast weather information for a given city and date.

    Parameters
    ----------
    city: str
        The name of the city to look up.
    date: str, optional
        The date for which to retrieve forecast data (ISO format YYYY-MM-DD).
        If omitted or empty, today's date is used.
    """
    try:
        if not date:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        lat, lon = _geocode_city(city)
        data = _fetch_weather(lat, lon, date)
        current = data.get("current_weather", {})
        daily = data.get("daily", {})
        result = {
            "city": city,
            "date": date,
            "current": current,
            "forecast": daily,
        }
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# Public attributes for auto-discovery
# ---------------------------------------------------------------------------

func = _get_weather
name = "get_weather"
description = "Retrieve current and forecast weather for a given city and date."
