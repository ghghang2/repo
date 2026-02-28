"""
Harvest repository metadata from Bitbucket public API.

The function ``harvest_bitbucket`` accepts a query string and returns a list of
repository metadata dictionaries.  It uses the Bitbucket API endpoint
`/repositories` with a search query.

Returned keys:
- ``name``
- ``full_name``
- ``description``
- ``language``
- ``created_on``
- ``updated_on``
- ``size``
- ``uuid``
- ``links`` (clone URLs)
"""

import requests
from typing import List, Dict

BITBUCKET_API_URL = "https://api.bitbucket.org/2.0/repositories"


def harvest_bitbucket(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search Bitbucket for repositories matching ``query``.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of repositories to return.
    """
    params = {
        "q": f"name~\"{query}\"",  # approximate search
        "pagelen": max_results,
    }
    resp = requests.get(BITBUCKET_API_URL, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Bitbucket API returned {resp.status_code}")
    data = resp.json()
    results = []
    for item in data.get("values", []):
        clone_links = [l.get("href") for l in item.get("links", {}).get("clone", [])]
        results.append(
            {
                "name": item.get("name", ""),
                "full_name": item.get("full_name", ""),
                "description": item.get("description", ""),
                "language": item.get("language", ""),
                "created_on": item.get("created_on", ""),
                "updated_on": item.get("updated_on", ""),
                "size": str(item.get("size", "")),
                "uuid": item.get("uuid", ""),
                "clone_links": ", ".join(clone_links),
            }
        )
    return results

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python harvest_bitbucket.py <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    meta = harvest_bitbucket(q)
    print(json.dumps(meta, indent=2))
"""
