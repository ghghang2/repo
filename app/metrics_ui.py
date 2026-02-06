import re
import time
import threading
from pathlib import Path
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

TOKENS_PER_SEC_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s+tokens per second", re.IGNORECASE)

def parse_log(path_string: str = "llama_server.log"):
    """Reads the log and returns a formatted markdown string."""
    log_path = Path(path_string)
    if not log_path.exists():
        return "**Log not found**"
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            lines = f.read().decode("utf-8", errors="ignore").splitlines()
        
        proc = any("slot update_slots:" in l.lower() for l in lines[-10:])
        if any("all slots are idle" in l.lower() for l in lines[-5:]):
            proc = False
            
        tps = 0.0
        for line in reversed(lines):
            if "eval time" in line.lower():
                m = TOKENS_PER_SEC_RE.search(line)
                if m:
                    tps = float(m.group("value"))
                    break
        
        emoji = "🟢" if proc else "⚫"
        return f"**Server:** {emoji}\n\n**TPS:** `{tps}`\n\n*{time.strftime('%H:%M:%S')}*"
    except:
        return "**Error reading log**"

def _background_parser():
    """Constantly updates the state so it's ready for any UI component."""
    while True:
        st.session_state.latest_metrics_md = parse_log()
        time.sleep(1)

@st.fragment(run_every=1.0)
def metrics_fragment():
    """This refreshes every second automatically when the script is IDLE."""
    if "latest_metrics_md" in st.session_state:
        # We wrap this in a unique keyed container so app.py can target it
        content = st.session_state.latest_metrics_md
        st.session_state.metrics_placeholder = st.empty()
        st.session_state.metrics_placeholder.markdown(content)

def display_metrics_panel():
    """Main entry point for app.py sidebar."""
    # 1. Start background thread if not running
    if "latest_metrics_md" not in st.session_state:
        st.session_state.latest_metrics_md = "Initializing..."
        ctx = get_script_run_ctx()
        t = threading.Thread(target=_background_parser, daemon=True)
        add_script_run_ctx(t)
        t.start()

    # 2. Render the fragment
    with st.container(border=True):
        metrics_fragment()