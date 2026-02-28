"""
Harvest metadata from Semantic Scholar.

The function ``harvest_semanticscholar`` accepts a query string and returns a list of
paper metadata dictionaries.  The Semantic Scholar API (free tier) is used.

The function performs a search request and returns the first ``max_results``
papers with the following keys:

- ``paperId``
- ``title``
- ``authors`` (comma separated string)
- ``year``
- ``abstract``
- ``venue``
- ``referenceCount``
- ``citationCount``
"""

import os
import json
from typing import List, Dict
import requests

# Base URL for Semantic Scholar API
BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Default max results per query
DEFAULT_MAX = 10


def _format_authors(authors: List[Dict]) -> str:
    """Return comma separated author names from Semantic Scholar author list."""
    return ", ".join([a.get("name", "") for a in authors])


def harvest_semanticscholar(query: str, max_results: int = DEFAULT_MAX) -> List[Dict[str, str]]:
    """Search Semantic Scholar for ``query`` and return metadata.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int, optional
        Maximum number of results to return.
    """
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,abstract,venue,referenceCount,citationCount",
        "limit": max_results,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Semantic Scholar API returned {resp.status_code}")
    data = resp.json()
    results = []
    for entry in data.get("data", []):
        results.append(
            {
                "paperId": entry.get("paperId", ""),
                "title": entry.get("title", ""),
                "authors": _format_authors(entry.get("authors", [])),
                "year": str(entry.get("year", "")),
                "abstract": entry.get("abstract", ""),
                "venue": entry.get("venue", ""),
                "referenceCount": str(entry.get("referenceCount", "")),
                "citationCount": str(entry.get("citationCount", "")),
            }
        )
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python harvest_semanticscholar.py <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    meta = harvest_semanticscholar(q)
    print(json.dumps(meta, indent=2))
"""
