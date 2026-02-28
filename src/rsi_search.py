"""
Simple command‑line interface to run the RSI pipeline.

Usage:
    python -m src.rsi_search "search query"

The script will:
1. Load configuration from ``config.yaml``.
2. Perform a DuckDuckGo search using :func:`search_engine.perform_search`.
3. Classify each result URL with :func:`classifier.classify_url`.
4. Harvest metadata for each category using the corresponding harvester.
5. Persist results with :func:`store_results.save_query_results`.
6. Generate a Markdown summary in ``summary.md``.
7. Send a completion email via the :func:`send_email` tool.
"""

import yaml
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List

# Import local modules
from .search_engine import perform_search
from .classifier import classify_url
from .harvest_arxiv import harvest_arxiv
from .harvest_semanticscholar import harvest_semanticscholar
from .harvest_crossref import harvest_crossref
from .harvest_patents import harvest_patents
from .store_results import save_query_results

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def harvest_for_category(category: str, url: str, query: str) -> List[Dict]:
    """Dispatch to the appropriate harvester based on category."""
    if category == "paper":
        # Try arXiv first by detecting arXiv URL
        if "arxiv.org" in url:
            return [harvest_arxiv(url)]
        # Fallback to Semantic Scholar search
        return harvest_semanticscholar(query)
    if category == "patent":
        return harvest_patents(query)
    if category == "repo":
        # Not implemented yet – placeholder
        return []
    if category == "blog":
        # Not implemented yet – placeholder
        return []
    return []


def generate_summary_file(results_path: Path, query: str):
    """Create a Markdown summary of the run."""
    summary_path = Path("summary.md")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# RSI Search Summary\n\n")
        f.write(f"## Query\n`{query}`\n\n")
        f.write(f"## Results stored in {results_path}\n")
    return summary_path


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        logger.error("No search query provided.")
        sys.exit(1)
    query = " ".join(argv)
    logger.info(f"Running RSI search for: {query}")

    # Load config
    cfg = load_config()
    search_cfg = cfg.get("search", {})
    max_results = search_cfg.get("max_results", 10)

    # Perform search
    try:
        raw_results = perform_search(query, num_results=max_results)
    except Exception as e:
        logger.exception("Search failed")
        sys.exit(1)

    # Classify URLs and harvest
    harvested: Dict[str, List[Dict]] = {"paper": [], "patent": [], "repo": [], "blog": []}
    for res in raw_results:
        url = res.get("url")
        if not url:
            continue
        category = classify_url(url)
        logger.debug(f"URL {url} classified as {category}")
        data = harvest_for_category(category, url, query)
        if data:
            harvested[category].extend(data)

    # Persist results
    results_path = save_query_results(query, raw_results, harvested)
    logger.info(f"Results saved to {results_path}")

    # Generate summary
    generate_summary_file(results_path, query)
    logger.info("Summary generated in summary.md")

    # Send notification email via tool
    try:
        from functions import send_email
        send_email(
            subject=f"RSI Search Completed: {query}",
            body=f"The search for '{query}' has finished. Results are stored in {results_path}",
        )
        logger.info("Completion email sent")
    except Exception as e:
        logger.warning(f"Failed to send email: {e}")


if __name__ == "__main__":
    main()
"""
