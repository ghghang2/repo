#!/usr/bin/env python3
"""Streamlit chat UI backed by a lightweight SQLite persistence.

The original application kept all messages only in ``st.session_state``.
To enable users to revisit older conversations we store each chat line
in a file‑based SQLite database.  The database lives in the repository
root as ``chat_history.db`` and contains a single ``chat_log`` table.

The UI now:

* shows a sidebar selector to pick an existing session or start a new one;
* loads the selected conversation on page load; and
* writes every user and assistant message to the DB after it is rendered.

Only the minimal amount of code needed for persistence is added – the
rest of the logic (model calls, tool handling, docs extraction, GitHub
push, etc.) remains unchanged.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import uuid

import streamlit as st
import streamlit.components.v1 as components
from git import InvalidGitRepositoryError, Repo

from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.tools import get_tools, TOOLS
from app.docs_extractor import extract
from app.chat import build_messages, stream_and_collect, process_tool_calls
# Persistence helpers
from app.db import init_db, log_message, load_history, get_session_ids


# ---------------------------------------------------------------------------
#  Initialise the database on first run
# ---------------------------------------------------------------------------
init_db()


# ---------------------------------------------------------------------------
#  Helper – refresh docs from the repo
# ---------------------------------------------------------------------------

def refresh_docs() -> str:
    """Run the repository extractor and return its Markdown output."""
    return extract().read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
#  Helper – check if local repo is up to date with GitHub
# ---------------------------------------------------------------------------

def is_repo_up_to_date(repo_path: Path) -> bool:
    """Return ``True`` if the local HEAD equals ``origin/main`` and the
    working tree is clean.
    """
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        return False

    if not repo.remotes:
        return False

    try:
        repo.remotes.origin.fetch()
    except Exception:
        return False

    for branch_name in ("main", "master"):
        try:
            remote_branch = repo.remotes.origin.refs[branch_name]
            break
        except IndexError:
            continue
    else:
        return False

    return (
        repo.head.commit.hexsha == remote_branch.commit.hexsha
        and not repo.is_dirty(untracked_files=True)
    )


# ---------------------------------------------------------------------------
#  Streamlit UI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Chat with GPT‑OSS", layout="wide")
    REPO_PATH = Path(__file__).parent

    # ---------------------------------------------------------------------
    #  Session state – keep in memory for the current browser session.
    # ---------------------------------------------------------------------
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("repo_docs", "")
    st.session_state.has_pushed = is_repo_up_to_date(REPO_PATH)

    # ---------------------------------------------------------------------
    #  Sidebar – new chat / docs / GitHub push + session selector
    # ---------------------------------------------------------------------
    with st.sidebar:
        # 1️⃣  New chat
        if st.button("New Chat"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.history = []
            st.session_state.repo_docs = ""
            st.success("Chat history cleared. Start fresh!")

        # 2️⃣  Refresh codebase docs
        if st.button("Ask Codebase"):
            st.session_state.repo_docs = refresh_docs()
            st.success("Codebase docs updated!")

        # 3️⃣  Push to GitHub
        if st.button("Push to GitHub"):
            with st.spinner("Pushing to GitHub…"):
                try:
                    from app.push_to_github import main as push_main

                    push_main()
                    st.session_state.has_pushed = True
                    st.success("✅  Repository pushed to GitHub.")
                except Exception as exc:
                    st.error(f"❌  Push failed: {exc}")

        # 4️⃣  Push status
        status = "✅  Pushed" if st.session_state.has_pushed else "⚠️  Not pushed"
        st.markdown(f"**Push status:** {status}")

        # 5️⃣  Session selector
        st.subheader("Session selector")
        session_options = ["new"] + get_session_ids()
        selected = st.selectbox("Open a conversation", session_options)
        if selected != "new":
            st.session_state["session_id"] = selected
            # st.rerun()

        # 6️⃣  List available tools
        st.subheader("Available tools")
        for t in TOOLS:
            st.markdown(f"*{t.name}*")

    # ---------------------------------------------------------------------
    #  Load conversation for the chosen session (if any)
    # ---------------------------------------------------------------------
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    history = load_history(session_id)
    st.session_state.history = history  # keep in sync with DB

    # ---------------------------------------------------------------------
    #  Render past messages
    # ---------------------------------------------------------------------
    for user_msg, bot_msg in history:
        st.chat_message("user").markdown(user_msg)
        st.chat_message("assistant").markdown(bot_msg)

    # ---------------------------------------------------------------------
    #  User input – new message
    # ---------------------------------------------------------------------
    if user_input := st.chat_input("Enter request…"):
        st.chat_message("user").markdown(user_input)
        log_message(session_id, "user", user_input)  # persist

        client = get_client()
        tools = get_tools()
        msgs = build_messages(history, st.session_state.system_prompt, st.session_state.repo_docs, user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            assistant_text, tool_calls = stream_and_collect(client, msgs, tools, placeholder)
            log_message(session_id, "assistant", assistant_text)  # persist

        # If the model invoked tools, process them in a loop
        if tool_calls:
            full_text, remaining_calls = process_tool_calls(client, msgs, tools, placeholder, tool_calls)
            log_message(session_id, "assistant", full_text)
            history.append((user_input, full_text))
            while remaining_calls:
                full_text, remaining_calls = process_tool_calls(client, msgs, tools, placeholder, remaining_calls)
                log_message(session_id, "assistant", full_text)
                history[-1] = (user_input, full_text)
        else:
            history.append((user_input, assistant_text))

        # Update session state history to keep UI in sync
        st.session_state.history = history

    # ---------------------------------------------------------------------
    #  Browser‑leaving guard
    # ---------------------------------------------------------------------
    has_pushed = st.session_state.get("has_pushed", False)
    components.html(
        f"""
        <script>
        window.top.hasPushed = {str(has_pushed).lower()};
        window.top.onbeforeunload = function (e) {{
            if (!window.top.hasPushed) {{
                e.preventDefault(); e.returnValue = '';
                return 'You have not pushed to GitHub yet.\nDo you really want to leave?';
            }}
        }};
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()

