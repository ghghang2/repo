"""
Harvest metadata from arXiv using the public API.

The function ``harvest_arxiv`` accepts an arXiv URL or ID and returns a
dictionary with the following keys (all strings):

- ``id``
- ``title``
- ``authors`` (comma separated)
- ``summary``
- ``published``
- ``updated``
- ``pdf_url``
"""

import re
from urllib.parse import urlparse
import feedparser
import requests

ARXIV_BASE = "http://export.arxiv.org/api/query"


def _extract_id(url_or_id: str) -> str:
    """Extract the arXiv ID from a URL or return the ID if given directly."""
    # If it looks like a URL, parse the path
    if re.match(r"^https?://", url_or_id):
        parsed = urlparse(url_or_id)
        # Expected path: /abs/<id> or /pdf/<id>.pdf
        match = re.search(r"/abs/(.+)$", parsed.path)
        if not match:
            match = re.search(r"/pdf/(.+)\.pdf$", parsed.path)
        if match:
            return match.group(1)
        raise ValueError(f"Could not parse arXiv ID from URL {url_or_id}")
    # Assume it is already an ID
    return url_or_id


def harvest_arxiv(url_or_id: str) -> dict:
    """Return metadata dictionary for an arXiv paper."""
    paper_id = _extract_id(url_or_id)
    query = f"search_query=id:{paper_id}&max_results=1"
    resp = requests.get(ARXIV_BASE, params={"search_query": f"id:{paper_id}", "max_results": 1})
    if resp.status_code != 200:
        raise RuntimeError(f"arXiv API returned {resp.status_code}")
    feed = feedparser.parse(resp.text)
    if not feed.entries:
        raise RuntimeError(f"No entry found for arXiv ID {paper_id}")
    entry = feed.entries[0]
    authors = ", ".join([a.name for a in entry.authors]) if hasattr(entry, "authors") else ""
    pdf_url = next((link.href for link in entry.links if link.type == "application/pdf"), "")
    return {
        "id": paper_id,
        "title": entry.title,
        "authors": authors,
        "summary": entry.summary,
        "published": entry.published,
        "updated": entry.updated,
        "pdf_url": pdf_url,
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python harvest_arxiv.py <arxiv_url_or_id>")
        sys.exit(1)
    meta = harvest_arxiv(sys.argv[1])
    import json
    print(json.dumps(meta, indent=2))
"""
