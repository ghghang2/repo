"""
Harvest repository metadata from GitLab public API.

The function ``harvest_gitlab`` accepts a query string and returns a list of
repository metadata dictionaries.  It uses the GitLab public API endpoint
`/api/v4/projects` with a search query.

Returned keys:
- ``id``
- ``name``
- ``description``
- ``web_url``
- ``ssh_url_to_repo``
- ``visibility``
- ``star_count``
- ``forks_count``
- ``last_activity_at``
"""

import requests
from typing import List, Dict

GITLAB_API_URL = "https://gitlab.com/api/v4/projects"


def harvest_gitlab(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search GitLab for repositories matching ``query``.

    Parameters
    ----------
    query : str
        Search query string.
    max_results : int, optional
        Maximum number of repositories to return.
    """
    params = {
        "search": query,
        "per_page": max_results,
    }
    resp = requests.get(GITLAB_API_URL, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"GitLab API returned {resp.status_code}")
    data = resp.json()
    results = []
    for item in data:
        results.append(
            {
                "id": str(item.get("id", "")),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "web_url": item.get("web_url", ""),
                "ssh_url_to_repo": item.get("ssh_url_to_repo", ""),
                "visibility": item.get("visibility", ""),
                "star_count": str(item.get("star_count", "")),
                "forks_count": str(item.get("forks_count", "")),
                "last_activity_at": item.get("last_activity_at", ""),
            }
        )
    return results

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python harvest_gitlab.py <query>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    meta = harvest_gitlab(q)
    print(json.dumps(meta, indent=2))
"""
