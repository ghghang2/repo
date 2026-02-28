"""
Search engine abstraction using the browser tool.

The implementation performs a search query against DuckDuckGo (public, no auth) and returns
a list of dictionaries containing ``title``, ``url`` and ``snippet``.

The function ``perform_search`` is intentionally simple – it relies on the
``browser`` tool which will click the search box, type the query and press
enter.  The result page is parsed with BeautifulSoup.
"""

import re
from typing import List, Dict

import bs4

# The browser tool is injected via the runtime environment.
# In this repository we simply declare it as a placeholder.
# In practice the caller will provide the tool.
browser = None  # type: ignore

SEARCH_URL = "https://duckduckgo.com/?q={query}&t=h_&ia=web"


def _extract_results(html: str) -> List[Dict[str, str]]:
    """Parse DuckDuckGo result page and return a list of result dicts."""
    soup = bs4.BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    for item in soup.select(".result"):  # DuckDuckGo uses .result class
        title_el = item.select_one("a.result__a")
        url_el = item.select_one("a.result__a")
        snippet_el = item.select_one("a.result__snippet")
        if not title_el or not url_el:
            continue
        title = title_el.get_text(strip=True)
        url = url_el.get("href", "")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def perform_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo search and return the top ``num_results``.

    Parameters
    ----------
    query: str
        The search query.
    num_results: int
        The maximum number of results to return.
    """
    # Build the URL
    url = SEARCH_URL.format(query=query.replace(" ", "+"))
    # Use the browser tool to fetch the page.
    if browser is None:
        raise RuntimeError("browser tool not provided")
    resp = browser(url=url, selector="")  # full page text
    raw_html = resp.get("text", "") if isinstance(resp, dict) else resp
    results = _extract_results(raw_html)
    return results[:num_results]

# Example usage (would be executed by the main script):
# if __name__ == "__main__":
#     import os
#     browser = os.getenv("BROWSER_TOOL")  # placeholder
#     print(perform_search("recursive self‑improvement"))
"""
