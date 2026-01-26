## app/__init__.py

```python
__all__ = ["client", "config", "utils", "docs_extractor"]
```

## app/client.py

```python
from openai import OpenAI
from .config import NGROK_URL

def get_client() -> OpenAI:
    """Return a client that talks to the local OpenAI‑compatible server."""
    return OpenAI(base_url=f"{NGROK_URL}/v1", api_key="token")
```

## app/config.py

```python
# Configuration – tweak these values as needed
NGROK_URL = "http://localhost:8000"
MODEL_NAME = "unsloth/gpt-oss-20b-GGUF:F16"
DEFAULT_SYSTEM_PROMPT = "Be concise and accurate at all times"

USER_NAME = "ghghang2"          # GitHub user / org name
REPO_NAME = "v1.1"              # Repository to push to
IGNORED_ITEMS = [
    ".config",
    ".ipynb_checkpoints",
    "sample_data",
    "llama-server",
    "nohup.out",
    "__pycache__",
]
```

## app/docs_extractor.py

```python
# app/docs_extractor.py
"""
extract_docs.py → docs_extractor.py
-----------------------------------
Walk a directory tree and write a single Markdown file that contains:

* The relative path of each file (as a level‑2 heading)
* The raw source code of that file (inside a fenced code block)
"""

from __future__ import annotations

import pathlib
import sys
import logging

log = logging.getLogger(__name__)

def walk_python_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return all *.py files sorted alphabetically."""
    return sorted(root.rglob("*.py"))

def write_docs(root: pathlib.Path, out: pathlib.Path) -> None:
    """Append file path + code to *out*."""
    with out.open("w", encoding="utf-8") as f_out:
        for p in walk_python_files(root):
            rel = p.relative_to(root)
            f_out.write(f"## {rel}\n\n")
            f_out.write("```python\n")
            f_out.write(p.read_text(encoding="utf-8"))
            f_out.write("\n```\n\n")

def extract(repo_root: pathlib.Path | str = ".", out_file: pathlib.Path | str | None = None) -> pathlib.Path:
    """
    Extract the repo into a Markdown file and return the path.

    Parameters
    ----------
    repo_root : pathlib.Path | str
        Root of the repo to walk.  Defaults to the current dir.
    out_file : pathlib.Path | str | None
        Path to write the Markdown.  If ``None`` uses ``repo_docs.md``.
    """
    root = pathlib.Path(repo_root).resolve()
    out = pathlib.Path(out_file or "repo_docs.md").resolve()

    log.info("Extracting docs from %s → %s", root, out)
    write_docs(root, out)
    log.info("✅  Wrote docs to %s", out)
    return out

def main() -> None:  # CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Extract a repo into Markdown")
    parser.add_argument("repo_root", nargs="?", default=".", help="Root of the repo")
    parser.add_argument("output", nargs="?", default="repo_docs.md", help="Output Markdown file")
    args = parser.parse_args()

    extract(args.repo_root, args.output)

if __name__ == "__main__":
    main()
```

## app/push_to_github.py

```python
# scripts/push_to_github.py
"""
Entry point that wires the `RemoteClient` together.
"""

from pathlib import Path
from remote import RemoteClient, REPO_NAME

def main() -> None:
    """Create/attach the remote, pull, commit and push."""
    client = RemoteClient(Path(__file__).resolve().parent.parent)  # repo root

    # 1️⃣  Ensure the GitHub repo exists
    client.ensure_repo(REPO_NAME)

    # 2️⃣  Attach (or re‑attach) the HTTPS remote
    client.attach_remote()

    # 3️⃣  Pull the latest changes
    client.fetch()
    client.pull()

    # 4️⃣  Write .gitignore (idempotent)
    client.write_gitignore()

    # 5️⃣  Commit everything that is new / changed
    client.commit_all("Initial commit")

    # 6️⃣  Make sure we are on the main branch
    if "main" not in [b.name for b in client.repo.branches]:
        client.repo.git.checkout("-b", "main")
        client.repo.git.reset("--hard")
    else:
        client.repo.git.checkout("main")
        client.repo.git.reset("--hard")

    # 7️⃣  Push to GitHub
    client.push()

if __name__ == "__main__":
    main()
```

## app/remote.py

```python
# remote/remote.py
"""
A single, self‑contained adapter that knows how to talk to:
  * a local Git repository (via gitpython)
  * GitHub (via PyGithub)
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import logging
from typing import Optional

from git import Repo, GitCommandError, InvalidGitRepositoryError
from github import Github, GithubException
from github.Auth import Token
from github.Repository import Repository

from config import USER_NAME, REPO_NAME, IGNORED_ITEMS

log = logging.getLogger(__name__)

def _token() -> str:
    """Return the GitHub PAT from the environment."""
    t = os.getenv("GITHUB_TOKEN")
    if not t:
        raise RuntimeError("GITHUB_TOKEN env variable not set")
    return t

def _remote_url() -> str:
    """HTTPS URL that contains the PAT – used only for git push."""
    return f"https://{USER_NAME}:{_token()}@github.com/{USER_NAME}/{REPO_NAME}.git"

class RemoteClient:
    """Thin wrapper around gitpython + PyGithub that knows how to create
    a repo, fetch/pull/push and keep the local repo up‑to‑date.
    """

    def __init__(self, local_path: Path | str):
        self.local_path = Path(local_path).resolve()
        try:
            self.repo = Repo(self.local_path)
            if self.repo.bare:
                raise InvalidGitRepositoryError(self.local_path)
        except (InvalidGitRepositoryError, GitCommandError):
            log.info("Initializing a fresh git repo at %s", self.local_path)
            self.repo = Repo.init(self.local_path)

        self.github = Github(auth=Token(_token()))
        self.user = self.github.get_user()

    # ------------------------------------------------------------------ #
    #  Local‑repo helpers
    # ------------------------------------------------------------------ #
    def is_clean(self) -> bool:
        """Return True if there are no uncommitted changes."""
        return not self.repo.is_dirty(untracked_files=True)

    def fetch(self) -> None:
        """Fetch from the remote (if it exists)."""
        if "origin" in self.repo.remotes:
            log.info("Fetching from origin…")
            self.repo.remotes.origin.fetch()
        else:
            log.info("No remote configured – skipping fetch")

    def pull(self, rebase: bool = True) -> None:
        """Pull the `main` branch from origin, optionally rebasing."""
        if "origin" not in self.repo.remotes:
            raise RuntimeError("No remote named 'origin' configured")

        branch = "main"
        log.info("Pulling %s%s…", branch, " (rebase)" if rebase else "")
        try:
            if rebase:
                self.repo.remotes.origin.pull(refspec=branch, rebase=True, progress=None)
            else:
                self.repo.remotes.origin.pull(branch)
        except GitCommandError as exc:
            log.warning("Rebase failed: %s – falling back to merge", exc)
            self.repo.git.merge(f"origin/{branch}")

    def push(self, remote: str = "origin") -> None:
        """Push the local `main` branch to the given remote."""
        if remote not in self.repo.remotes:
            raise RuntimeError(f"No remote named '{remote}'")
        log.info("Pushing to %s…", remote)
        self.repo.remotes[remote].push("main")

    def reset_hard(self) -> None:
        """Discard any uncommitted or stale merge‑conflict data."""
        self.repo.git.reset("--hard")

    # ------------------------------------------------------------------ #
    #  GitHub helpers
    # ------------------------------------------------------------------ #
    def ensure_repo(self, name: str = REPO_NAME) -> Repository:
        """Create the GitHub repo if it does not exist and return it."""
        try:
            repo = self.user.get_repo(name)
            log.info("Repo '%s' already exists on GitHub", name)
        except GithubException:
            log.info("Creating new repo '%s' on GitHub", name)
            repo = self.user.create_repo(name, private=False)
        return repo

    def attach_remote(self, url: Optional[str] = None) -> None:
        """Delete any existing `origin` remote and add a fresh one."""
        if url is None:
            url = _remote_url()
        if "origin" in self.repo.remotes:
            log.info("Removing old origin remote")
            self.repo.delete_remote("origin")
        log.info("Adding new origin remote: %s", url)
        self.repo.create_remote("origin", url)

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    def write_gitignore(self) -> None:
        """Create a .gitignore that matches the constants in config.py."""
        path = self.local_path / ".gitignore"
        content = "\n".join(IGNORED_ITEMS) + "\n"
        path.write_text(content, encoding="utf-8")
        log.info("Wrote %s", path)

    def commit_all(self, message: str = "Initial commit") -> None:
        """Stage everything and commit (ignoring the 'nothing to commit' error)."""
        self.repo.git.add(A=True)
        try:
            self.repo.index.commit(message)
            log.info("Committed: %s", message)
        except GitCommandError as exc:
            if "nothing to commit" in str(exc):
                log.info("Nothing new to commit")
            else:
                raise
```

## app/utils.py

```python
# app/utils.py  (only the added/modified parts)
from typing import List, Tuple, Dict, Optional
from .config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME
from .client import get_client
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Build the messages list that the OpenAI API expects
# --------------------------------------------------------------------------- #
def build_api_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str] = None,
) -> List[Dict]:
    """
    Convert local chat history into the format expected by the OpenAI API.

    Parameters
    ----------
    history : List[Tuple[str, str]]
        (user, assistant) pairs.
    system_prompt : str
        Prompt given to the model.
    repo_docs : str | None
        Full code‑base text.  If supplied it is sent as the *first* assistant
        message so the model can read it before answering.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": repo_docs})
    for user_msg, bot_msg in history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    return msgs

# --------------------------------------------------------------------------- #
# Stream the assistant reply token‑by‑token
# --------------------------------------------------------------------------- #
def stream_response(
    history: List[Tuple[str, str]],
    user_msg: str,
    client: OpenAI,
    system_prompt: str,
    repo_docs: Optional[str] = None,
):
    """Yield the cumulative assistant reply while streaming."""
    new_hist = history + [(user_msg, "")]
    api_msgs = build_api_messages(new_hist, system_prompt, repo_docs)

    stream = client.chat.completions.create(
        model=MODEL_NAME, messages=api_msgs, stream=True
    )

    full_resp = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_resp += token
        yield full_resp
```

## app.py

```python
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from app.config import DEFAULT_SYSTEM_PROMPT
from app.client import get_client
from app.utils import stream_response
from app.docs_extractor import extract


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def refresh_docs() -> str:
    """Run the extractor once (same folder as app.py)."""
    return extract().read_text(encoding="utf-8")


def is_repo_up_to_date(repo_path: Path) -> bool:
    """Return True iff local HEAD == remote `origin/main` AND no dirty files."""
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        return False

    if not repo.remotes:
        return False

    origin = repo.remotes.origin
    try:
        origin.fetch()
    except Exception:
        return False

    # try common branch names
    for branch_name in ("main", "master"):
        try:
            remote_branch = origin.refs[branch_name]
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
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Chat with GPT‑OSS", layout="wide")

    REPO_PATH = Path(__file__).parent

    # session state
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("repo_docs", "")
    st.session_state.has_pushed = is_repo_up_to_date(REPO_PATH)

    with st.sidebar:
        st.header("Settings")

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

    # Render chat history
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # User input
    if user_input := st.chat_input("Enter request…"):
        st.chat_message("user").markdown(user_input)

        client = get_client()
        bot_output = ""

        with st.chat_message("assistant") as assistant_msg:
            placeholder = st.empty()
            for partial in stream_response(
                st.session_state.history,
                user_input,
                client,
                st.session_state.system_prompt,
                st.session_state.repo_docs,
            ):
                bot_output = partial
                placeholder.markdown(bot_output, unsafe_allow_html=True)

        st.session_state.history.append((user_input, bot_output))

    # Browser‑leaving guard
    has_pushed = st.session_state.get("has_pushed", False)
    components.html(
        f"""
        <script>
        // Make the flag visible to the outer window
        window.top.hasPushed = {str(has_pushed).lower()};

        // Attach the unload guard to the outer window
        window.top.onbeforeunload = function (e) {{
            if (!window.top.hasPushed) {{
            // Modern browsers require e.preventDefault() + e.returnValue
            e.preventDefault();
            e.returnValue = '';
            return 'You have not pushed to GitHub yet.\\nDo you really want to leave?';
            }}
        }};
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()
```

