# app/tools/get_weather.py
"""Weather lookup using the public wttr.in service.

The original implementation only returned the current weather in a
compact one-line format.  The updated version adds an optional
``date`` parameter that supports three values:

* ``"today"`` (default) – current weather
* ``"tomorrow"`` – next day forecast
* ``"YYYY-MM-DD"`` – any date within the next seven days (inclusive)

The function keeps the same minimal interface (no API key, no external
packages) and returns a JSON string.  The JSON contains the city name,
the requested date, the weather summary and, if an error occurred,
an ``error`` field.

The tool is intentionally tolerant: if the requested date is out of
range or the API call fails, the function returns an error message
instead of raising an exception.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict

# ---------------------------------------------------------------------------
# Helper – build a URL for wttr.in with the desired format
# ---------------------------------------------------------------------------

def _build_url(city: str, fmt: str, day: str | None = None) -> str:
    """Return a fully-qualified wttr.in URL.

    Parameters
    ----------
    city
        City name to query.
    fmt
        wttr.in format specifier (``1`` for a one-line summary, ``2``
        for a slightly more detailed format that also accepts a ``day``
        parameter).
    day
        Optional day offset (``"0"`` for today, ``"1"`` for tomorrow,
        etc.).  Only used when ``fmt == "2"``.
    """
    base = f"https://wttr.in/{urllib.parse.quote_plus(city)}"
    if day is not None:
        query = f"?format={fmt}&day={day}"
    else:
        query = f"?format={fmt}"
    return base + query

# ---------------------------------------------------------------------------
# Public API – the function that the tool loader will expose
# ---------------------------------------------------------------------------

def func(city: str, date: str = "today") -> str:
    """Return weather information for *city*.

    The function supports three *date* values:

    * ``"today"`` (default) – current weather
    * ``"tomorrow"`` – next day forecast
    * ``"YYYY-MM-DD"`` – any date within the next 7 days (inclusive)

    The result is a JSON string with the following keys:

    ``city``
        The requested city name.
    ``date``
        The date that was requested (converted to ``YYYY-MM-DD`` for
        explicit dates, ``"today"`` or ``"tomorrow"`` for the other
        cases).
    ``weather``
        The weather summary returned by wttr.in.
    ``error`` (optional)
        Human-readable error message if something went wrong.
    """
    # Normalise the *date* argument
    date_lower = date.strip().lower()
    now = datetime.utcnow().date()

    try:
        if date_lower in ("today", "now", ""):
            fmt = "1"
            url = _build_url(city, fmt)
            requested_date = now
        elif date_lower == "tomorrow":
            fmt = "2"
            url = _build_url(city, fmt, day="1")
            requested_date = now + timedelta(days=1)
        else:
            # Try to parse an explicit date
            target = datetime.strptime(date, "%Y-%m-%d").date()
            if target < now or target > now + timedelta(days=7):
                raise ValueError("Date out of allowable range (today–7 days)")
            fmt = "2"
            day_offset = str((target - now).days)
            url = _build_url(city, fmt, day=day_offset)
            requested_date = target

        # Perform the request
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode().strip()

        result: Dict[str, str] = {
            "city": city,
            "date": requested_date.isoformat(),
            "weather": body,
        }
        return json.dumps(result)
    except Exception as exc:  # pragma: no cover – network errors
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# Metadata used by the tool loader
# ---------------------------------------------------------------------------
name = "get_weather"
# Updated description to mention the optional *date* parameter
# NOTE: The original string used double quotes for the outer
# string and also contained unescaped double quotes around the
# sample values ("today", "tomorrow", "YYYY-MM-DD").  That caused a
# syntax error during import, which prevented the tool from being
# discovered.  We switch to a single-quoted outer string and keep
# the inner quotes as literal text.
description = (
    'Return a concise, human-readable weather summary for a city using wttr.in. '
    'Supports an optional *date* argument ("today", "tomorrow", or "YYYY-MM-DD") to fetch forecasts. '
)

__all__ = ["func", "name", "description"]

# ---------------------------------------------------------------------------
# Debug entry point – useful during development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(func("Taipei", "tomorrow"))