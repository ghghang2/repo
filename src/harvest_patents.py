"""
Harvest patent metadata using the PatentsView public API.

The function ``harvest_patents`` accepts a query string and returns a list of
patent metadata dictionaries.  PatentsView (https://patentsview.org/) provides
an open JSON API for patent data.

The function performs a search request and returns the first ``max_results``
patents with the following keys:

- ``patent_number``
- ``title``
- ``inventors`` (comma separated string)
- ``assignees`` (comma separated string)
- ``abstract``
- ``inventor_city``
- ``inventor_state``
- ``inventor_country``
- ``date_filed``
- ``date_publication``
"""

import json
from typing import List, Dict
import requests

API_URL = "https://api.patentsview.org/patents/query"
DEFAULT_MAX = 10


def _format_list(items: List[Dict], key: str) -> str:
    return ", ".join([i.get(key, "") for i in items])


def harvest_patents(query: str, max_results: int = DEFAULT_MAX) -> List[Dict[str, str]]:
    """Search PatentsView for ``query`` and return metadata.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int, optional
        Maximum number of results to return.
    """
    payload = {
        "q": {"_text_query": {"patent_title": query}},
        "f": [
            "patent_number",
            "patent_title",
            "inventor_name",
            "assignee_name",
            "inventor_city",
            "inventor_state",
            "inventor_country",
            "patent_date_filed",
            "patent_date_publication",
            "patent_abstract",
        ],
        "o": {"per_page": max_results},
    }
    resp = requests.post(API_URL, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"PatentsView API returned {resp.status_code}")
    data = resp.json()
    results = []
    for patent in data.get("patents", []):
        inventors = _format_list(patent.get("inventor", []), "inventor_name")
        assignees = _format_list(patent.get("assignee", []), "assignee_name")
        results.append(
            {
                "patent_number": patent.get("patent_number", ""),
                "title": patent.get("patent_title", ""),
                "inventors": inventors,
                "assignees": assignees,
                "abstract": patent.get("patent_abstract", ""),
                "inventor_city": patent.get("inventor_city", ""),
                "inventor_state": patent.get("inventor_state", ""),
                "inventor_country": patent.get("inventor_country", ""),
                "date_filed": patent.get("patent_date_filed", ""),
                "date_publication": patent.get("patent_date_publication", ""),
            }
        )
    return results

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python harvest_patents.py <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    meta = harvest_patents(q)
    print(json.dumps(meta, indent=2))
"""
