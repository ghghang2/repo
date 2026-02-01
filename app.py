#!/usr/bin/env python3
"""
app.py – Streamlit UI that talks to a local llama‑server and can push the repo
to GitHub.  The code now supports **multiple** tool calls per request.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import streamlit.components.v1 as components
from git import InvalidGitRepositoryError, Repo

from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.tools import get_tools, TOOLS
from app.docs_extractor import extract
from app.chat import build_messages, stream_and_collect, process_tool_calls


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def refresh_docs() -> str:
    """Run the extractor and return the Markdown content."""
    return extract().read_text(encoding="utf-8")


def is_repo_up_to_date(repo_path: Path) -> bool:
    """Return True iff the local HEAD matches origin/main and the working tree
    has no dirty files."""
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

# --------------------------------------------------------------------------- #
#  Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    # tab_chat, tab_log = st.tabs(["Chat", "Log"])
    st.set_page_config(page_title="Chat with GPT‑OSS", layout="wide")
    REPO_PATH = Path(__file__).parent

    # Session state
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("repo_docs", "")
    st.session_state.has_pushed = is_repo_up_to_date(REPO_PATH)

    # -------------------------------------------------------------------- #
    #  Sidebar
    # -------------------------------------------------------------------- #
    with st.sidebar:

        # System prompt editor
        prompt = st.text_area(
            "System prompt",
            st.session_state.system_prompt,
            height=120,
        )
        if prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = prompt

        # New chat button
        if st.button("New Chat"):
            st.session_state.history = []
            st.session_state.repo_docs = ""
            st.success("Chat history cleared. Start fresh!")

        # Refresh docs button
        if st.button("Refresh Docs"):
            st.session_state.repo_docs = refresh_docs()
            st.success("Codebase docs updated!")

        # Push to GitHub button
        if st.button("Push to GitHub"):
            with st.spinner("Pushing to GitHub…"):
                try:
                    from app.push_to_github import main as push_main

                    push_main()
                    st.session_state.has_pushed = True
                    st.success("✅  Repository pushed to GitHub.")
                except Exception as exc:
                    st.error(f"❌  Push failed: {exc}")

        # Push status
        status = "✅  Pushed" if st.session_state.has_pushed else "⚠️  Not pushed"
        st.markdown(f"**Push status:** {status}")

        # Available tools
        st.subheader("Available tools")
        for t in TOOLS:
            st.markdown(f"- **{t.name}**: {t.description}")
    # -------------------------------------------------------------------- #
    #  Chat history
    # -------------------------------------------------------------------- #
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # -------------------------------------------------------------------- #
    #  User input
    # -------------------------------------------------------------------- #
    if user_input := st.chat_input("Enter request…"):
        with st.chat_message("user"):
            st.markdown(user_input)

        client = get_client()
        tools = get_tools()
        msgs = build_messages(
            st.session_state.history,
            st.session_state.system_prompt,
            st.session_state.repo_docs,
            user_input,
        )
        with st.chat_message("assistant"):
            placeholder = st.empty()

        # First assistant turn – may contain tool calls
        final_text, tool_calls = stream_and_collect(client, msgs, tools, placeholder)

        # --------------------------------------------------------------------
        #  Show the tool calls that were detected
        # --------------------------------------------------------------------
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc.get("arguments") or "{}")
                st.markdown(
                    f"**Tool call**: `{tc.get('name')}`({', '.join(f'{k}={v}' for k, v in args.items())})",
                    unsafe_allow_html=True,
                )

        # Add the partial assistant reply to the message history
        msgs.append({"role": "assistant", "content": final_text})

        # If the model invoked tools, let the helper process all of them
        if tool_calls:
            full_text, remaining_calls = process_tool_calls(
                client, msgs, tools, placeholder, tool_calls
            )
            st.session_state.history.append((user_input, full_text))
            # Process any nested calls that might have been triggered by a tool
            while remaining_calls:
                full_text, remaining_calls = process_tool_calls(
                    client, msgs, tools, placeholder, remaining_calls
                )
                st.session_state.history[-1] = (user_input, full_text)
        else:
            # No tool calls – just store what we already got
            st.session_state.history.append((user_input, final_text))
    
    # -------------------------------------------------------------------- #
    #  Browser‑leaving guard
    # -------------------------------------------------------------------- #
    has_pushed = st.session_state.get("has_pushed", False)
    components.html(
        f"""
        <script>
        window.top.hasPushed = {str(has_pushed).lower()};
        window.top.onbeforeunload = function (e) {{
            if (!window.top.hasPushed) {{
                e.preventDefault(); e.returnValue = '';
                return 'You have not pushed to GitHub yet.\\nDo you really want to leave?';
            }}
        }};
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()