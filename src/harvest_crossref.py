"""
Harvest metadata from Crossref.

The function ``harvest_crossref`` accepts a query string and returns a list of
publication metadata dictionaries.  The Crossref REST API is used.

The function performs a search request and returns the first ``max_results``
papers with the following keys:

- ``DOI``
- ``title``
- ``authors`` (comma separated string)
- ``issued`` (year)
- ``abstract``
- ``publisher``
- ``URL``
"""

import requests
from typing import List, Dict

BASE_URL = "https://api.crossref.org/works"


def _format_authors(authors):
    return ", ".join([a.get("given", "") + " " + a.get("family", "") for a in authors if a.get("family")])


def harvest_crossref(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    params = {
        "query": query,
        "rows": max_results,
    }
    resp = requests.get(BASE_URL, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Crossref API returned {resp.status_code}")
    data = resp.json()
    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        title_list = item.get("title", [""])
        title = title_list[0] if title_list else ""
        authors = _format_authors(item.get("author", []))
        issued = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        results.append(
            {
                "DOI": item.get("DOI", ""),
                "title": title,
                "authors": authors,
                "issued": str(issued) if issued else "",
                "abstract": item.get("abstract", ""),
                "publisher": item.get("publisher", ""),
                "URL": item.get("URL", ""),
            }
        )
    return results

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python harvest_crossref.py <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    meta = harvest_crossref(q)
    print(json.dumps(meta, indent=2))
"""
