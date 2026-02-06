"""Streamlit UI for displaying Llama metrics.

The original implementation used an infinite `while True` loop which blocks
Streamlit's execution, preventing the app from rendering.  The updated
version performs a single fetch and displays the data.  For real‑time
updates users can call :func:`display_metrics_panel` again or use a
Streamlit timer.
"""

import streamlit as st
import requests
import time
import argparse
import datetime
import re
import sys
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# Regular‑expression helpers
# --------------------------------------------------------------------------- #
LOG_LINE_RE = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (?P<msg>.+)',
    re.IGNORECASE,
)

# Example: 2024-08-28 12:34:56.123 | eval: prompt 200 tokens (150 ms)
PROMPT_RE = re.compile(r'prompt\s+(\d+)\s+tokens\s+\((\d+)\s+ms\)', re.IGNORECASE)
PREDICT_RE = re.compile(r'predict\s+(\d+)\s+tokens\s+\((\d+)\s+ms\)', re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def parse_log(path_string) -> Tuple[bool, float]:
    """
    Returns (processing, prediction_tps)

    *processing* – True if the server is still busy with the latest request.
    *prediction_tps* – tokens per second for the prediction phase of the
                       most recent eval that we encounter.
    """
    log_path = Path(path_string)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # We'll walk the file from the end to the start.
    # Python's `reversed(list(fp))` is simplest for small-to‑medium logs.
    # For very large logs you might want to use `mmap` or a custom tailer.
    with log_path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()

    # Variables that will hold the last *seen* eval data
    last_predict_tokens = 0
    last_predict_ms = 0

    # Scan from bottom to top
    for raw_line in reversed(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        m = LOG_LINE_RE.match(raw_line)
        if not m:
            continue

        msg = m.group("msg").lower()

        # 1. If we hit an eval line first – the server is still processing.
        if "eval:" in msg:
            # Grab prediction tokens & time (if present)
            m_q = PREDICT_RE.search(msg)
            if m_q:
                last_predict_tokens = int(m_q.group(1))
                last_predict_ms = int(m_q.group(2))
            # We can stop – the request is still in flight.
            return True, _calc_tps(last_predict_tokens, last_predict_ms)

        # 2. If we hit an update_slots line first – the request is done.
        if "update_slots:" in msg or "all slots idle" in msg:
            # No processing; we may still want to know the prediction TPS
            # from the *most recent* eval that came before this line.
            return False, _calc_tps(last_predict_tokens, last_predict_ms)

    # Fallback – file had no eval or update_slots lines
    return False, 0.0


def _calc_tps(tokens: int, ms: int) -> float:
    return tokens / (ms / 1000.0) if ms else 0.0

# # Cache the metrics for 3 seconds to avoid excessive HTTP requests.
# # @st.cache_data(ttl=3)
# def get_llama_metrics(url: str) -> dict[str, str]:
#     """Fetch metrics from the Llama server.

#     Parameters
#     ----------
#     url: str
#         The URL to query for metrics.

#     Returns
#     -------
#     dict[str, str]
#         Mapping of metric name to value.
#     """
#     try:
#         resp = requests.get(url, timeout=2)
#         resp.raise_for_status()
#     except Exception as exc:  # pragma: no cover - defensive
#         st.error(f"Could not fetch metrics: {exc}")
#         return {}

#     metrics: dict[str, str] = {}
#     for line in resp.text.splitlines():
#         if line.startswith("#") or not line:
#             continue
#         parts = line.split()
#         if len(parts) >= 2:
#             metrics[parts[0]] = parts[1]
#     return metrics

@st.fragment(run_every=1.0)
def display_metrics_panel() -> None:
    """Display a sidebar panel with the latest metrics.

    The function fetches metrics once and renders them.  It does not
    block the UI, making it safe to call from a Streamlit app.
    """
    processing, prediction_tps = parse_log("llama_server.log")
    st.metric(label="processing", value=processing)
    st.metric(label="tps", value=prediction_tps)
    st.write(f"Updated: {time.strftime('%H:%M:%S')}")
    # placeholder = st.sidebar.empty()
    # base_url = "http://localhost:8000/metrics"
    # metrics = get_llama_metrics(base_url)

    # if metrics:
        
    #     st.metric(label="processing", value=metrics['llamacpp:requests_processing'])
    #     st.metric(label="tps", value=metrics['llamacpp:predicted_tokens_seconds'])
    #     st.write(f"Updated: {time.strftime('%H:%M:%S')}")
    # else:
    #     placeholder.info("No metrics returned from server.")
