import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.get_weather import func as get_weather


def test_get_weather_success():
    """Test that the get_weather tool returns a result for a valid city and date."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    result_json = get_weather("London", today)
    result = json.loads(result_json)
    assert "result" in result
    data = result["result"]
    assert data["city"] == "London"
    assert data["date"] == today
    assert "current" in data
    assert "forecast" in data


def test_get_weather_error():
    """Test that the get_weather tool returns an error for an invalid city."""
    result_json = get_weather("NonExistentCityXYZ", "")
    result = json.loads(result_json)
    assert "error" in result


def test_get_weather_date_omitted():
    """Test that omitting the date defaults to today."""
    result_json = get_weather("London")
    result = json.loads(result_json)
    assert "result" in result
    data = result["result"]
    assert data["city"] == "London"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    assert data["date"] == today
    assert "current" in data
