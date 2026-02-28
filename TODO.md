# RSI Research Surface Expansion

## Overview
This repository contains a proof‑of‑concept to broaden the RSI research surface beyond arXiv and GitHub. The goal is to gather metadata from a variety of public sources (search engines, preprint servers, patents, blogs, code hosting platforms) while keeping the stack free and account‑free.

---

## Implementation Plan
Each task is marked with a status:
- `❌ Pending`
- `✅ Completed`
- `🚧 Blocked`

---

### Phase 0 – Project Setup
| # | Task | Status |
|---|------|--------|
| 0.1 | Initialise a new Python project (virtualenv, requirements.txt) | ❌ Pending |
| 0.2 | Create a basic directory layout (`src/`, `results/`, `tests/`) | ❌ Pending |
| 0.3 | Add a `config.yaml` template with default settings | ❌ Pending |

### Phase 1 – Search Engine & Classification
| # | Task | Status |
|---|------|--------|
| 1.1 | Write `search_engine.py` to perform Google/DuckDuckGo search using `browser` tool | ❌ Pending |
| 1.2 | Implement result extraction (title, URL, snippet) | ❌ Pending |
| 1.3 | Create a simple domain‑based classifier to tag URLs as `paper`, `patent`, `repo`, `blog` | ❌ Pending |
| 1.4 | Unit‑test the search engine with mock responses | ❌ Pending |

### Phase 2 – Academic & Patent Harvesters
| # | Task | Status |
|---|------|--------|
| 2.1 | Implement `harvest_arxiv.py` using arXiv API | ❌ Pending |
| 2.2 | Implement `harvest_semanticscholar.py` (free tier) | ❌ Pending |
| 2.3 | Implement `harvest_crossref.py` for non‑arXiv preprints | ❌ Pending |
| 2.4 | Implement `harvest_patents.py` using USPTO Open Patent Services | ❌ Pending |
| 2.5 | Unit‑test each harvester with sample IDs | ❌ Pending |

### Phase 3 – Code Repository Harvesters
| # | Task | Status |
|---|------|--------|
| 3.1 | Implement `harvest_gitlab.py` using GitLab public API | ❌ Pending |
| 3.2 | Implement `harvest_bitbucket.py` using Bitbucket public API | ❌ Pending |
| 3.3 | Unit‑test repository harvesters | ❌ Pending |

### Phase 4 – Result Storage & Logging
| # | Task | Status |
|---|------|--------|
| 4.1 | Design a JSON schema for harvested metadata | ❌ Pending |
| 4.2 | Write `store_results.py` to persist query, raw search, and harvested data | ❌ Pending |
| 4.3 | Implement simple duplicate detection (URL hash) | ❌ Pending |
| 4.4 | Add timestamp & logging to each run | ❌ Pending |

### Phase 5 – Packaging & Demo
| # | Task | Status |
|---|------|--------|
| 5.1 | Create a command‑line interface (`rsi_search.py`) that orchestrates all modules | ❌ Pending |
| 5.2 | Generate a Markdown summary (`summary.md`) per query | ❌ Pending |
| 5.3 | Write a README with usage instructions | ❌ Pending |
| 5.4 | Add a sample run output in the repository | ❌ Pending |

### Phase 6 – Validation & Testing
| # | Task | Status |
|---|------|--------|
| 6.1 | Run a full pipeline on the test keyword `"recursive self‑improvement"` | ❌ Pending |
| 6.2 | Verify that at least one result is harvested from each source type | ❌ Pending |
| 6.3 | Ensure no rate‑limit errors occur | ❌ Pending |

### Phase 7 – Finalisation
| # | Task | Status |
|---|------|--------|
| 7.1 | Mark all completed tasks and remove `TODO.md` or archive it | ❌ Pending |
| 7.2 | Notify via email that implementation is finished | ❌ Pending |

---

## Notes
- All external calls are rate‑limited; implement exponential back‑off where necessary.
- The project is intentionally minimalistic to keep dependencies light (requests, beautifulsoup4, pyyaml, tqdm).
- For privacy, the script will only scrape publicly accessible pages.
- Future enhancements (translation, PDF extraction) will be added in later phases.
