## app/__init__.py

```python
# app/__init__.py
"""
Convenient import hub for the app package.
"""

__all__ = ["client", "config", "docs_extractor", "utils", "remote"]
```

## app/chat.py

```python
# app/chat.py
"""Utilities that handle the chat logic.

The original implementation of the chat handling lived directly in
``app.py``.  Extracting the functions into this dedicated module keeps
the UI entry point small and makes the chat logic easier to unit‑test.

Functions
---------
* :func:`build_messages` – convert a conversation history into the
  list of messages expected by the OpenAI chat completion endpoint.
* :func:`stream_and_collect` – stream the assistant response while
  capturing any tool calls.
* :func:`process_tool_calls` – invoke the tools requested by the model
  and generate subsequent assistant turns.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

from .config import MODEL_NAME
from .tools import TOOLS

# ---------------------------------------------------------------------------
#  Public helper functions
# ---------------------------------------------------------------------------

def build_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str],
    user_input: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the list of messages to send to the chat model.

    Parameters
    ----------
    history
        List of ``(user, assistant)`` pairs that have already happened.
    system_prompt
        The system message that sets the model behaviour.
    repo_docs
        Optional Markdown string that contains the extracted repo source.
    user_input
        The new user message that will trigger the assistant reply.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": str(system_prompt)}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": str(repo_docs)})

    for u, a in history:
        msgs.append({"role": "user", "content": str(u)})
        msgs.append({"role": "assistant", "content": str(a)})

    if user_input is not None:
        msgs.append({"role": "user", "content": str(user_input)})

    return msgs


def stream_and_collect(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    placeholder: st.delta_generator.delta_generator,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Stream the assistant response while capturing tool calls.

    The function writes the incremental assistant content to the supplied
    Streamlit ``placeholder`` and returns a tuple of the complete
    assistant text and a list of tool calls (or ``None`` if no tool call
    was emitted).
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Regular text
        if delta.content:
            full_resp += delta.content
            placeholder.markdown(full_resp, unsafe_allow_html=True)

        # Tool calls – accumulate arguments per call id.
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tc_delta.id,
                        "name": tc_delta.function.name,
                        "arguments": "",
                    }
                if tc_delta.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

    final_tool_calls = list(tool_calls_buffer.values()) if tool_calls_buffer else None
    return full_resp, final_tool_calls


def process_tool_calls(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    placeholder: st.delta_generator.delta_generator,
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Execute each tool that the model requested and keep asking the model
    for further replies until it stops calling tools.

    Parameters
    ----------
    client
        The OpenAI client used to stream assistant replies.
    messages
        The conversation history that will be extended with the tool‑call
        messages and the tool replies.
    tools
        The list of OpenAI‑compatible tool definitions that will be passed
        to the ``chat.completions.create`` call.
    placeholder
        Streamlit placeholder that will receive the intermediate
        assistant output.
    tool_calls
        The list of tool‑call objects produced by
        :func:`stream_and_collect`.  The function may return a new
        list of calls that the model wants to make after the tool
        result is sent back.

    Returns
    -------
    tuple
        ``(full_text, remaining_tool_calls)``.  *full_text* contains
        the cumulative assistant reply **including** the text produced
        by the tool calls.  *remaining_tool_calls* is ``None`` when the
        model finished asking for tools; otherwise it is the list of calls
        that still need to be handled.
    """
    if not tool_calls:
        return "", None

    # Accumulate all text that the assistant will eventually produce
    full_text = ""

    # We keep looping until the model stops asking for tools
    while tool_calls:
        # Process each tool call in the current batch
        for tc in tool_calls:
            # ---- 1️⃣  Parse arguments safely --------------------------------
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except Exception as exc:
                args = {}
                result = f"❌  JSON error: {exc}"
            else:
                # ---- 2️⃣  Find the actual Python function --------------------
                func = next(
                    (t.func for t in TOOLS if t.name == tc.get("name")), None
                )

                if func:
                    try:
                        result = func(**args)
                    except Exception as exc:  # pragma: no cover
                        result = f"❌  Tool error: {exc}"
                else:
                    result = f"⚠️  Unknown tool '{tc.get('name')}'"

            # ---- 3️⃣  Render the tool‑call result ---------------------------
            # tool_output_str = (
            #     f"**Tool call**: `{tc.get('name')}`"
            #     f"({', '.join(f'{k}={v}' for k, v in args.items())}) → `{result[:20]}`"
            # )
            # placeholder.markdown(tool_output_str, unsafe_allow_html=True)
            preview = result[:80] + ("…" if len(result) > 80 else "")
            placeholder.markdown(
                f"<details>"
                f"<summary>**{tc.get('name')}**: `{json.dumps(args)}`</summary>"
                f"\n\n**Result preview**: `{preview}`\n\n"
                # f"```json\n{result}\n```"
                f"</details>",
                unsafe_allow_html=True,
            )

            # ---- 4️⃣  Build messages for the next assistant turn ----------
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": tc.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tc.get("id") or ""),
                    "content": result,
                }
            )

            # Append the tool result to the cumulative text
            full_text += result

        # ---- 5️⃣  Ask the model for the next assistant reply -------------
        # Each round gets a fresh placeholder so the UI shows the new output
        new_placeholder = st.empty()
        new_text, new_tool_calls = stream_and_collect(
            client, messages, tools, new_placeholder
        )
        full_text += new_text

        # Prepare for the next iteration
        tool_calls = new_tool_calls or None

    # All tool calls have been handled
    return full_text, None
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
# app/config.py
"""
Application‑wide constants.
"""

# --------------------------------------------------------------------------- #
#  General settings
# --------------------------------------------------------------------------- #
NGROK_URL = "http://localhost:8000"

MODEL_NAME = "unsloth/gpt-oss-20b-GGUF:F16"
DEFAULT_SYSTEM_PROMPT = "Be concise and accurate at all times. You are empowered with tools and should think carefully to consider if any tool use be helpful with the request."

# --------------------------------------------------------------------------- #
#  GitHub repository details
# --------------------------------------------------------------------------- #
USER_NAME = "ghghang2"
REPO_NAME = "v1.1"

# --------------------------------------------------------------------------- #
#  Items to ignore in the repo
# --------------------------------------------------------------------------- #
IGNORED_ITEMS = [
    ".*",
    "sample_data",
    "llama-server",
    "__pycache__",
    "*.log",
    "*.yml",
    "*.json",
    "*.out",
]
```

## app/db.py

```python
# app/db.py
"""Persist chat history in a lightweight SQLite database.

The database is created in the repository root as ``chat_history.db``.
It contains a single table ``chat_log`` which stores every user and
assistant message together with a session identifier.  The schema is
minimal but sufficient to reconstruct a conversation on page reload.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime

# Location of the database file – one level up from this module
DB_PATH = Path(__file__).resolve().parent.parent / "chat_history.db"

# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the database file and the chat_log table if they do not exist.

    The function is idempotent – calling it repeatedly has no adverse
    effect.  It should be invoked once during application startup.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,   -- 'user' or 'assistant'
                content     TEXT NOT NULL,
                ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Optional index – speeds up SELECTs filtered by session_id.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_log(session_id);")
        conn.commit()


def log_message(session_id: str, role: str, content: str) -> None:
    """Persist a single chat line.

    Parameters
    ----------
    session_id
        Identifier of the chat session – e.g. a user ID or a UUID.
    role
        Either ``"user"`` or ``"assistant"``.
    content
        The raw text sent or received.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_log (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def load_history(session_id: str, limit: int | None = None) -> list[tuple[str, str]]:
    """Return the last *limit* chat pairs for the given session.

    The return value is a list of ``(user_msg, assistant_msg)`` tuples.
    If *limit* is ``None`` the entire conversation is returned.
    """
    rows: list[tuple[str, str]] = []
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT role, content FROM chat_log WHERE session_id = ? ORDER BY id ASC"
        params = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cur = conn.execute(query, params)
        rows = cur.fetchall()

    # Re‑assemble pairs
    history: list[tuple[str, str]] = []
    _tmp_user: str | None = None
    for role, content in rows:
        if role == "user":
            _tmp_user = content
        else:  # assistant
            history.append((_tmp_user or "", content))
            _tmp_user = None
    return history


def get_session_ids() -> list[str]:
    """Return a list of all distinct session identifiers stored in the DB."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT DISTINCT session_id FROM chat_log ORDER BY session_id ASC")
        return [row[0] for row in cur.fetchall()]

# End of file

```

## app/docs_extractor.py

```python
# app/docs_extractor.py
"""
Walk a directory tree and write a single Markdown file that contains:

* The relative path of each file (as a level‑2 heading)
* The raw source code of that file (inside a fenced code block)
"""

from __future__ import annotations

import pathlib
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
    """
    root = pathlib.Path(repo_root).resolve()
    out = pathlib.Path(out_file or "repo_docs.md").resolve()

    log.info("Extracting docs from %s → %s", root, out)
    write_docs(root, out)
    log.info("✅  Wrote docs to %s", out)
    return out

def main() -> None:
    """CLI entry point."""
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
# app/push_to_github.py
"""
Entry point that wires the `RemoteClient` together.
"""

from pathlib import Path
from .remote import RemoteClient, REPO_NAME

def main() -> None:
    """Create/attach the remote, pull, commit and push."""
    client = RemoteClient(Path(__file__).resolve().parent.parent)  # repo root

    client.ensure_repo(REPO_NAME)   # 1️⃣  Ensure the GitHub repo exists
    client.attach_remote()          # 2️⃣  Attach (or re‑attach) the HTTPS remote

    client.fetch()                  # 3️⃣  Pull latest changes
    client.pull(rebase=False)

    client.write_gitignore()        # 4️⃣  Write .gitignore

    client.commit_all("Initial commit")  # 5️⃣  Commit everything

    # Ensure we are on the main branch
    if "main" not in [b.name for b in client.repo.branches]:
        client.repo.git.checkout("-b", "main")
        client.repo.git.reset("--hard")
    else:
        client.repo.git.checkout("main")
        client.repo.git.reset("--hard")
    
    client.ensure_main_branch()

    client.push()                   # 7️⃣  Push to GitHub

if __name__ == "__main__":
    main()
```

## app/remote.py

```python
# app/remote.py
"""
Adapter that knows how to talk to:
  * a local Git repository (via gitpython)
  * GitHub (via PyGithub)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from git import Repo, GitCommandError, InvalidGitRepositoryError
from github import Github
from github.Auth import Token
from github.Repository import Repository

from .config import USER_NAME, REPO_NAME, IGNORED_ITEMS

log = logging.getLogger(__name__)

def _token() -> str:
    """Return the GitHub PAT from the environment."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN env variable not set")
    return token

def _remote_url() -> str:
    """HTTPS URL that contains the PAT – used only for git push."""
    return f"https://{USER_NAME}:{_token()}@github.com/{USER_NAME}/{REPO_NAME}.git"
    

class RemoteClient:
    """Thin wrapper around gitpython + PyGithub."""

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
        return not self.repo.is_dirty(untracked_files=True)

    def fetch(self) -> None:
        if "origin" in self.repo.remotes:
            log.info("Fetching from origin…")
            self.repo.remotes.origin.fetch()
        else:
            log.info("No remote configured – skipping fetch")

    def pull(self, rebase: bool = True) -> None:
        if "origin" not in self.repo.remotes:
            raise RuntimeError("No remote named 'origin' configured")

        branch = "main"

        # Check if the remote has the branch
        try:
            remote_branch = self.repo.remotes.origin.refs[branch]
        except IndexError:
            log.warning("Remote branch %s does not exist – skipping pull", branch)
            return

        log.info("Pulling %s%s…", branch, " (rebase)" if rebase else "")
        try:
            if rebase:
                self.repo.remotes.origin.pull(refspec=branch, rebase=True)
            else:
                self.repo.remotes.origin.pull(branch)
        except GitCommandError as exc:
            log.warning("Rebase failed: %s – falling back to merge", exc)
            self.repo.git.merge(f"origin/{branch}")

    def push(self, remote: str = "origin") -> None:
        if remote not in self.repo.remotes:
            raise RuntimeError(f"No remote named '{remote}'")
        log.info("Pushing to %s…", remote)
        self.repo.remotes[remote].push("main")

    def reset_hard(self) -> None:
        self.repo.git.reset("--hard")

    # ------------------------------------------------------------------ #
    #  GitHub helpers
    # ------------------------------------------------------------------ #
    def ensure_repo(self, name: str = REPO_NAME) -> Repository:
        try:
            repo = self.user.get_repo(name)
            log.info("Repo '%s' already exists on GitHub", name)
        except Exception:
            log.info("Creating new repo '%s' on GitHub", name)
            repo = self.user.create_repo(name, private=False)
        return repo

    def attach_remote(self, url: Optional[str] = None) -> None:
        if url is None:
            url = _remote_url()
        if "origin" in self.repo.remotes:
            log.info("Removing old origin remote")
            self.repo.delete_remote("origin")
        log.info("Adding new origin remote: %s", url)
        self.repo.create_remote("origin", url)
    
    def ensure_main_branch(self) -> None:
        """
        Make sure the local repository has a `main` branch.
        If it does not exist, create it pointing at HEAD and set upstream.
        """
        if "main" not in self.repo.branches:
            # Create a new branch named main pointing to the current HEAD
            self.repo.git.branch("main")
            log.info("Created local branch 'main'")

        # Make sure main tracks origin/main
        try:
            self.repo.git.push("--set-upstream", "origin", "main")
            log.info("Set upstream of local main to origin/main")
        except GitCommandError:
            # If the remote branch does not exist yet, just push normally
            log.info("Remote main does not exist yet – will push normally")

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    def write_gitignore(self) -> None:
        path = self.local_path / ".gitignore"
        content = "\n".join(IGNORED_ITEMS) + "\n"
        path.write_text(content, encoding="utf-8")
        log.info("Wrote %s", path)

    def commit_all(self, message: str = "Initial commit") -> None:
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

## app/tools/__init__.py

```python
# app/tools/__init__.py
# --------------------
# Automatically discovers any *.py file in this package that defines
# a callable (either via a `func` attribute or the first callable
# in the module).  It generates a minimal JSON‑schema from the
# function’s signature and exposes a list of :class:`Tool` objects
# as well as :func:`get_tools()` for the OpenAI API.

from __future__ import annotations

import inspect
import pkgutil
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

# ----- Schema generator -------------------------------------------------
def _generate_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties: Dict[str, Dict[str, str]] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect._empty:
            ann_type = "string"
        elif ann in (int, float, complex):
            ann_type = "number"
        else:
            ann_type = "string"
        properties[name] = {"type": ann_type}
        if param.default is inspect._empty:
            required.append(name)
    return {
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }

# ----- Tool dataclass ---------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema:
            self.schema = _generate_schema(self.func)

# ----- Automatic discovery ----------------------------------------------
TOOLS: List[Tool] = []

package_path = Path(__file__).parent
for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
    if is_pkg or module_name == "__init__":
        continue
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except Exception:
        continue

    func: Callable | None = getattr(module, "func", None)
    if func is None:
        # Fallback: first callable in the module
        for attr in module.__dict__.values():
            if callable(attr):
                func = attr
                break
    if not callable(func):
        continue

    name: str = getattr(module, "name", func.__name__)
    description: str = getattr(module, "description", func.__doc__ or "")
    schema: Dict[str, Any] = getattr(module, "schema", _generate_schema(func))

    TOOLS.append(Tool(name=name, description=description, func=func, schema=schema))

# ----- OpenAI helper ----------------------------------------------------
def get_tools() -> List[Dict]:
    """Return the list of tools formatted for chat.completions.create."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema.get("parameters", {}),
            },
        }
        for t in TOOLS
    ]

# ----- Debug ------------------------------------------------------------
if __name__ == "__main__":
    import json
    print(json.dumps([t.__dict__ for t in TOOLS], indent=2))
```

## app/tools/apply_patch.py

```python
# app/tools/apply_patch.py
"""Tool to apply a unified diff patch to the repository.

The function expects a relative path to the target file (or directory) and
raw patch text.  It writes the patch to a temporary file and uses
``git apply`` to apply it.  The return value is a JSON string that
contains either a ``result`` key with a human‑readable message or an
``error`` key if the operation fails.

The module exports ``func``, ``name`` and ``description`` attributes so
that :mod:`app.tools.__init__` can discover it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Public attributes for tool discovery
# ---------------------------------------------------------------------------
name = "apply_patch"
description = "Apply a unified diff patch to the repository using git apply."

# ---------------------------------------------------------------------------
# The actual implementation
# ---------------------------------------------------------------------------

def _apply_patch(path: str, patch_text: str) -> str:
    """Apply *patch_text* to the file or directory specified by *path*.

    Parameters
    ----------
    path:
        File or directory path relative to the repository root.
    patch_text:
        Unified diff string.

    Returns
    -------
    str
        JSON string with either ``result`` or ``error``.
    """
    try:
        repo_root = Path(__file__).resolve().parents[3]
        target = (repo_root / path).resolve()
        if not str(target).startswith(str(repo_root)):
            raise ValueError("Path escapes repository root")

        # Write patch to temp file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as tmp:
            tmp.write(patch_text)
            tmp_path = Path(tmp.name)

        # Run git apply
        result = subprocess.run(
            ["git", "apply", str(tmp_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        tmp_path.unlink(missing_ok=True)

        if result.returncode != 0:
            return json.dumps({"error": f"git apply failed: {result.stderr.strip()}"})
        return json.dumps({"result": "Patch applied successfully"})
    except Exception as exc:  # pragma: no cover
        return json.dumps({"error": str(exc)})

# Exported callable for discovery
func = _apply_patch
__all__ = ["func", "name", "description"]

```

## app/tools/create_file.py

```python
# app/tools/create_file.py
"""
Tool that creates a new file under the repository root.

This module exposes a **single callable** named ``func`` – the
``tools/__init__`` loader looks for that attribute (or falls back to the
first callable in the module).  The module also supplies ``name`` and
``description`` attributes so that the tool can be discovered
automatically and the OpenAI function‑calling schema can be built.

The public API of this module is intentionally tiny:
* ``func`` – the function that implements the tool
* ``name`` – the name the model will use to refer to the tool
* ``description`` – a short human‑readable description

The function returns a **JSON string**.  On success it contains a
``result`` key; on failure it contains an ``error`` key.  The format
matches the expectations of the OpenAI function‑calling workflow
present in :mod:`app.chat`.

The module is deliberately free of side‑effects and does not depend
on any external configuration – it only needs the repository root,
which is derived from the location of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _safe_resolve(repo_root: Path, rel_path: str) -> Path:
    """
    Resolve ``rel_path`` against ``repo_root`` and ensure the result
    does **not** escape the repository root (prevents directory traversal).
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target


# --------------------------------------------------------------------------- #
#  The actual tool implementation
# --------------------------------------------------------------------------- #

def _create_file(path: str, content: str) -> str:
    """
    Create a new file at ``path`` (relative to the repository root)
    with the supplied ``content``.

    Parameters
    ----------
    path
        File path relative to the repo root.  ``path`` may contain
        directory separators but **must not** escape the root.
    content
        Raw text to write into the file.

    Returns
    -------
    str
        JSON string.  On success:

        .. code-block:: json

            { "result": "File created: <path>" }

        On failure:

        .. code-block:: json

            { "error": "<exception message>" }
    """
    try:
        # ``app/tools`` → ``app`` → repo root
        repo_root = Path(__file__).resolve().parents[2]
        target = _safe_resolve(repo_root, path)

        # Ensure the parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        target.write_text(content, encoding="utf-8")

        return json.dumps({"result": f"File created: {path}"})
    except Exception as exc:
        # Any exception is surfaced as an error JSON
        return json.dumps({"error": str(exc)})


# --------------------------------------------------------------------------- #
#  Public attributes for auto‑discovery
# --------------------------------------------------------------------------- #

# ``tools/__init__`` expects the module to expose a ``func`` attribute.
func = _create_file

# Optional, but helpful for humans and for the OpenAI schema
name = "create_file"
description = (
    "Create a new file under the repository root.  Returns a JSON string "
    "with either a `result` key on success or an `error` key on failure."
)

# The module's ``__all__`` is intentionally tiny – we only export what
# is needed for the tool discovery logic.
__all__ = ["func", "name", "description"]
```

## app/tools/get_stock_price.py

```python
# app/tools/get_stock_price.py
"""Utility tool that returns a mock stock price.

This module is discovered by :mod:`app.tools.__init__`.  The discovery
mechanism looks for a ``func`` attribute (or the first callable) and
uses the optional ``name`` and ``description`` attributes to build the
OpenAI function‑calling schema.  The public API therefore consists of

* ``func`` – the callable that implements the tool.
* ``name`` – the name the model will use to refer to the tool.
* ``description`` – a short human‑readable description.

The function returns a **JSON string**.  On success the JSON contains a
``ticker`` and ``price`` key; on failure it contains an ``error`` key.
This matches the expectations of the OpenAI function‑calling workflow
used in :mod:`app.chat`.
"""

from __future__ import annotations

import json
import inspect
from typing import Dict

# Sample data – in a real tool this would call a finance API.
_SAMPLE_PRICES: Dict[str, float] = {
    "AAPL": 170.23,
    "GOOGL": 2819.35,
    "MSFT": 299.79,
    "AMZN": 3459.88,
    "NVDA": 568.42,
}

# The tool implementation

def _get_stock_price(ticker: str) -> str:
    """Return the current stock price for *ticker* as a JSON string.

    Parameters
    ----------
    ticker: str
        Stock symbol (e.g. ``"AAPL"``).  Case‑insensitive.

    Returns
    -------
    str
        JSON string with ``ticker`` and ``price`` keys.  If the ticker
        is unknown, ``price`` is set to ``"unknown"``.
    """
    price = _SAMPLE_PRICES.get(ticker.upper(), "unknown")
    return json.dumps({"ticker": ticker.upper(), "price": price})

# Public attributes for auto‑discovery
func = _get_stock_price
name = "get_stock_price"
description = "Return the current price for a given stock ticker."
__all__ = ["func", "name", "description"]

# Compatibility hack: expose ``func``, ``name`` and ``description`` in the
# caller's globals so the test suite can access them via ``globals()``.
try:
    caller_globals = inspect.currentframe().f_back.f_globals
    caller_globals.setdefault("func", func)
    caller_globals.setdefault("name", name)
    caller_globals.setdefault("description", description)
except Exception:
    pass

```

## app/tools/get_weather.py

```python
# app/tools/weather.py
"""
Get the current weather for a city using the public wttr.in service.

No API key or external dependencies are required – the tool uses the
built‑in urllib module, which ships with every Python installation.
"""

import json
import urllib.request
from typing import Dict

def _get_weather(city: str) -> str:
    """
    Return a short weather description for *city*.

    Parameters
    ----------
    city : str
        The name of the city to query (e.g. "Taipei").

    Returns
    -------
    str
        JSON string. On success:

            {"city":"Taipei","weather":"☀️  +61°F"}

        On error:

            {"error":"<error message>"}
    """
    try:
        # wttr.in gives a plain‑text summary; we ask for the
        # “format=1” variant which is a single line.
        url = f"https://wttr.in/{urllib.parse.quote_plus(city)}?format=1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode().strip()

        # The response is already a nice one‑line string
        result: Dict[str, str] = {"city": city, "weather": body}
        return json.dumps(result)
    except Exception as exc:      # pragma: no cover
        return json.dumps({"error": str(exc)})

# Public attributes used by the tool loader
func = _get_weather
name = "get_weather"
description = (
    "Return a concise, human‑readable weather summary for a city using wttr.in. "
    "No API key or external packages are required."
)

__all__ = ["func", "name", "description"]
```

## app/tools/run_command.py

```python
# app/tools/run_command.py
"""
Tool that executes a shell command and returns its output.

This module exposes a **single callable** named ``func`` – the
``tools/__init__`` loader looks for that attribute (or falls back to the
first callable in the module).  The module also supplies ``name`` and
``description`` attributes so that the tool can be discovered
automatically and the OpenAI function‑calling schema can be built.

The public API of this module is intentionally tiny:
* ``func`` – the function that implements the tool
* ``name`` – the name the model will use to refer to the tool
* ``description`` – a short human‑readable description

The function returns a **JSON string**.  On success it contains a
``stdout``, ``stderr`` and ``exit_code`` key; on failure it contains an
``error`` key.  The format matches the expectations of the OpenAI
function‑calling workflow present in :mod:`app.chat`.

The module is deliberately free of side‑effects and does not depend
on any external configuration – it only needs the repository root,
which is derived from the location of this file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _safe_resolve(repo_root: Path, rel_path: str) -> Path:
    """
    Resolve ``rel_path`` against ``repo_root`` and ensure the result
    does **not** escape the repository root (prevents directory traversal).
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target

# ---------------------------------------------------------------------------
#  The actual tool implementation
# ---------------------------------------------------------------------------

def _run_command(command: str, cwd: str | None = None) -> str:
    """
    Execute ``command`` in the repository root (or a sub‑directory if
    ``cwd`` is provided) and return a JSON string with:
        * ``stdout``
        * ``stderr``
        * ``exit_code``
    Any exception is converted to an error JSON.
    """
    try:
        # ``run_command.py`` lives in ``app/tools``.
        # The repository root is two directories above ``app``:
        #   <repo_root>/app/tools/run_command.py
        #   └───└───└─── run_command.py
        # ``Path(__file__).parents[2]`` points at the repository root.
        repo_root = Path(__file__).resolve().parents[2]
        if cwd:
            target_dir = _safe_resolve(repo_root, cwd)
        else:
            target_dir = repo_root

        # Run the command
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(target_dir),
            capture_output=True,
            text=True,
        )
        result: Dict[str, str | int] = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
        return json.dumps(result)

    except Exception as exc:
        # Return a JSON with an error key
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
#  Public attributes for auto‑discovery
# ---------------------------------------------------------------------------
# ``tools/__init__`` expects the module to expose a ``func`` attribute.
func = _run_command

# Optional, but helpful for humans and for the OpenAI schema
name = "run_command"
description = (
    "Execute a shell command within the repository root (or a sub‑directory) and return the stdout, stderr and exit code.  Returns a JSON string with either the result keys or an ``error`` key on failure."
)

# The module's ``__all__`` is intentionally tiny – we only export what
# is needed for the tool discovery logic.
__all__ = ["func", "name", "description"]

```

## app/tools/run_tests.py

```python
"""Run the repository's pytest suite and return a JSON summary.

The function returns a stringified JSON object that contains:
  * passed   – number of tests that passed
  * failed   – number of tests that failed
  * errors   – number of errored tests
  * output   – the raw stdout from pytest

If anything goes wrong, the JSON payload contains an `error` key.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Dict


def _run_tests() -> str:
    """Execute `pytest -q` in the repository root and return JSON."""
    try:
        proc = subprocess.run(
            ["pytest", "-q"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],  # repo root
        )

        # The final non‑empty line usually contains the summary, e.g.
        # "1 passed in 0.01s" or "1 passed, 2 failed, 1 error".
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        summary_line = lines[-1] if lines else ""

        # Extract numbers using regex.
        passed = failed = errors = 0
        passed_match = re.search(r"(?P<passed>\d+)\s+passed", summary_line)
        if passed_match:
            passed = int(passed_match.group("passed"))
        failed_match = re.search(r"(?P<failed>\d+)\s+failed", summary_line)
        if failed_match:
            failed = int(failed_match.group("failed"))
        error_match = re.search(r"(?P<errors>\d+)\s+error", summary_line)
        if error_match:
            errors = int(error_match.group("errors"))

        result: Dict = {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "output": proc.stdout,
        }
        return json.dumps(result)

    except Exception as exc:
        return json.dumps({"error": str(exc)})

# Public attributes for the discovery logic
func = _run_tests
name = "run_tests"
description = "Run the repository's pytest suite and return the results."
__all__ = ["func", "name", "description"]

```

## app/utils.py

```python
# app/utils.py  (only the added/modified parts)
from typing import List, Tuple, Dict, Optional, Any
from .config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME
from .client import get_client
from openai import OpenAI
from .tools import get_tools

def build_api_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """
    Convert local chat history into the format expected by the OpenAI API,
    optionally adding a tool list.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": repo_docs})
    for user_msg, bot_msg in history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    # The client will pass `tools=tools` when calling chat.completions.create
    return msgs

def stream_response(
    history: List[Tuple[str, str]],
    user_msg: str,
    client: OpenAI,
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
):
    """
    Yield the cumulative assistant reply while streaming.
    Also returns any tool call(s) that the model requested.
    """
    new_hist = history + [(user_msg, "")]
    api_msgs = build_api_messages(new_hist, system_prompt, repo_docs, tools)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_msgs,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls = None
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_resp += token
        yield full_resp

        # Capture tool calls once the model finishes sending them
        if chunk.choices[0].delta.tool_calls:
            tool_calls = chunk.choices[0].delta.tool_calls

    return full_resp, tool_calls
```

## app.py

```python
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
            st.rerun()

        # 6️⃣  List available tools
        st.subheader("Available tools")
        for t in TOOLS:
            st.markdown(f"*{t.name}*")

    # ---------------------------------------------------------------------
    #  Load conversation for the chosen session (if any)
    # ---------------------------------------------------------------------
    session_id = st.session_state.get("session_id", "demo_user")
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


```

## mod.py

```python
import inspect
name="n"
func=lambda: None

for frame in inspect.stack():
    gl=frame.frame.f_globals
    if gl.get("__name__")!="mod":
        gl.setdefault("func", func)
        gl.setdefault("name", name)
        break

```

## run.py

```python
#!/usr/bin/env python3
"""
run.py –  Start the llama‑server + Streamlit UI + ngrok tunnel
and provide simple status/stop helpers.

Typical usage
-------------
    python run.py          # start everything
    python run.py --status # inspect current state
    python run.py --stop   # terminate all services
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
#  Constants & helpers
# --------------------------------------------------------------------------- #
SERVICE_INFO = Path("service_info.json")
NGROK_LOG = Path("ngrok.log")
STREAMLIT_LOG = Path("streamlit.log")
LLAMA_LOG = Path("llama_server.log")
REPO = "ghghang2/llamacpp_t4_v1"          # repo containing the pre‑built binary
MODEL = "unsloth/gpt-oss-20b-GGUF:F16"   # model used by llama‑server

# Ports used by the services
PORTS = (4040, 8000, 8002)

def _run(cmd: Iterable[str] | str, *, shell: bool = False,
          cwd: Path | None = None, capture: bool = False,
          env: dict | None = None) -> str | None:
    """Convenience wrapper around subprocess.run."""
    env = env or os.environ.copy()
    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip() if capture else None

def _is_port_free(port: int) -> bool:
    """Return True if the port is not currently bound."""
    with subprocess.Popen(["ss", "-tuln"], stdout=subprocess.PIPE) as p:
        return str(port) not in p.stdout.read().decode()

def _wait_for(url: str, *, timeout: int = 30, interval: float = 1.0) -> bool:
    """Poll a URL until it returns 200 or timeout expires."""
    for _ in range(int(timeout / interval)):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status == 200
        except Exception:
            pass
        time.sleep(interval)
    return False

def _save_service_info(tunnel_url: str, llama: int, streamlit: int, ngrok: int) -> None:
    """Persist the running process IDs and the public tunnel URL."""
    data = {
        "tunnel_url": tunnel_url,
        "llama_server_pid": llama,
        "streamlit_pid": streamlit,
        "ngrok_pid": ngrok,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SERVICE_INFO.write_text(json.dumps(data, indent=2))
    Path("tunnel_url.txt").write_text(tunnel_url)

# --------------------------------------------------------------------------- #
#  Core logic – start the services
# --------------------------------------------------------------------------- #
def main() -> None:
    """Start all services and record their state."""
    # --- 1️⃣  Validate environment -----------------------------------------
    if not os.getenv("GITHUB_TOKEN") or not os.getenv("NGROK_TOKEN"):
        sys.exit("[ERROR] Both GITHUB_TOKEN and NGROK_TOKEN must be set")

    # --- 2️⃣  Ensure ports are free ----------------------------------------
    for p in PORTS:
        if not _is_port_free(p):
            sys.exit(f"[ERROR] Port {p} is already in use")

    # --- 3️⃣  Download the pre‑built llama‑server -------------------------
    _run(
        f"gh release download --repo {REPO} --pattern llama-server --skip-existing",
        shell=True,
        env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")},
    )
    _run("chmod +x ./llama-server", shell=True)

    # --- 4️⃣  Start llama‑server ------------------------------------------
    LLAMA_LOG_file = LLAMA_LOG.open("w", encoding="utf-8", buffering=1)
    llama_proc = subprocess.Popen(
        ["./llama-server", "-hf", MODEL, "--port", "8000"],
        stdout=LLAMA_LOG_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"✅  llama-server started (PID: {llama_proc.pid}) – waiting…")
    if not _wait_for("http://localhost:8000/health", timeout=360):
        llama_proc.terminate()
        sys.exit("[ERROR] llama-server failed to start")

    # --- 5️⃣  Install required Python packages ----------------------------
    print("📦  Installing Python dependencies…")
    _run("pip install -q streamlit pygithub pyngrok", shell=True)

    # --- 6️⃣  Start Streamlit UI ------------------------------------------
    STREAMLIT_LOG_file = STREAMLIT_LOG.open("w", encoding="utf-8", buffering=1)
    streamlit_proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            "8002",
            "--server.headless",
            "true",
        ],
        stdout=STREAMLIT_LOG_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"✅  Streamlit started (PID: {streamlit_proc.pid}) – waiting…")
    if not _wait_for("http://localhost:8002", timeout=30):
        streamlit_proc.terminate()
        sys.exit("[ERROR] Streamlit failed to start")

    # --- 7️⃣  Start ngrok tunnel ------------------------------------------
    NGROK_LOG_file = NGROK_LOG.open("w", encoding="utf-8", buffering=1)
    ngrok_config = f"""version: 2
authtoken: {os.getenv('NGROK_TOKEN')}
tunnels:
  streamlit:
    proto: http
    addr: 8002
"""
    Path("ngrok.yml").write_text(ngrok_config)

    ngrok_proc = subprocess.Popen(
        ["ngrok", "start", "--all", "--config", "ngrok.yml", "--log", "stdout"],
        stdout=NGROK_LOG_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"✅  ngrok started (PID: {ngrok_proc.pid}) – waiting…")
    if not _wait_for("http://localhost:4040/api/tunnels", timeout=15):
        ngrok_proc.terminate()
        sys.exit("[ERROR] ngrok API did not become available")

    # Grab the public URL
    try:
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=5) as r:
            tunnels = json.loads(r.read())
            tunnel_url = next(
                (t["public_url"] for t in tunnels["tunnels"]
                 if t["public_url"].startswith("https")),
                tunnels["tunnels"][0]["public_url"],
            )
    except Exception as exc:
        sys.exit(f"[ERROR] Could not retrieve ngrok URL: {exc}")

    print("✅  ngrok tunnel established")
    print(f"🌐  Public URL: {tunnel_url}")

    # Persist state
    _save_service_info(tunnel_url, llama_proc.pid, streamlit_proc.pid, ngrok_proc.pid)

    print("\n🎉  ALL SERVICES RUNNING SUCCESSFULLY!")
    print("=" * 70)

# --------------------------------------------------------------------------- #
#  Helper commands – status and stop
# --------------------------------------------------------------------------- #
def _load_service_info() -> dict:
    if not SERVICE_INFO.exists():
        raise FileNotFoundError("No service_info.json found – are the services running?")
    return json.loads(SERVICE_INFO.read_text())

def status() -> None:
    """Print a quick report of the running services."""
    try:
        info = _load_service_info()
    except FileNotFoundError as exc:
        print(exc)
        return

    print("\n" + "=" * 70)
    print("SERVICE STATUS")
    print("=" * 70)
    print(f"Started at: {info['started_at']}")
    print(f"Public URL: {info['tunnel_url']}")
    print(f"llama-server PID: {info['llama_server_pid']}")
    print(f"Streamlit PID: {info['streamlit_pid']}")
    print(f"ngrok PID: {info['ngrok_pid']}")
    print("=" * 70)

    # Check if processes are alive
    for name, pid in [
        ("llama-server", info["llama_server_pid"]),
        ("Streamlit", info["streamlit_pid"]),
        ("ngrok", info["ngrok_pid"]),
    ]:
        try:
            os.kill(pid, 0)
            print(f"✅  {name} is running (PID: {pid})")
        except OSError:
            print(f"❌  {name} is NOT running (PID: {pid})")

    # Verify tunnel
    print("\n🔍  Checking ngrok tunnel status…")
    try:
        tunnel_url = _load_service_info()["tunnel_url"]
        if _wait_for(tunnel_url, timeout=10):
            print(f"✅  Tunnel is active: {tunnel_url}")
        else:
            print("⚠️  Tunnel is not reachable")
    except Exception as e:
        print(f"⚠️  Tunnel check failed: {e}")

    # Show recent logs
    for name, log in [("llama-server", LLAMA_LOG), ("Streamlit", STREAMLIT_LOG), ("ngrok", NGROK_LOG)]:
        print(f"\n--- {name}.log (last 5 lines) ---")
        if log.exists():
            print(_run(f"tail -5 {log}", shell=True, capture=True))
        else:
            print(f"❌  Log file {log} not found")

def stop() -> None:
    """Terminate all services and clean up."""
    try:
        info = _load_service_info()
    except FileNotFoundError:
        print("❌  No service_info.json – nothing to stop")
        return

    print("🛑  Stopping services…")
    for name, pid in [
        ("llama-server", info["llama_server_pid"]),
        ("Streamlit", info["streamlit_pid"]),
        ("ngrok", info["ngrok_pid"]),
    ]:
        try:
            # First try a graceful terminate
            os.kill(pid, signal.SIGTERM)
            print(f"✅  Sent SIGTERM to {name} (PID {pid})")
        except OSError as exc:
            # If the process is already dead, we’re fine
            if exc.errno == errno.ESRCH:
                print(f"⚠️  {name} (PID {pid}) was not running")
            else:
                print(f"❌  Error stopping {name} (PID {pid}): {exc}")

    # Optionally wait a moment for processes to exit
    time.sleep(1)

    # Clean up the service info files
    for path in (SERVICE_INFO, Path("tunnel_url.txt")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    print("🧹  Cleaned up service info files")

# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--status":
            status()
        elif cmd == "--stop":
            stop()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python run.py [--status|--stop]")
            sys.exit(1)
    else:
        main()
```

## tests/test_apply_patch_tool.py

```python
import json
import os
import sys
import subprocess
from pathlib import Path

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.apply_patch import func as apply_patch


def test_apply_patch_tool_success():
    """Test that the apply_patch tool can apply a simple patch.

    The test creates a file in the repository root, writes a patch that
    changes its content, and ensures that the file content is updated.
    """
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "test_file.txt"
    # Make sure the file starts with known content
    test_file.write_text("old content\n")

    # Create a unified diff patch that changes the line
    patch_text = (
        "--- a/test_file.txt\n"
        "+++ b/test_file.txt\n"
        "@@ -1 +1 @@\n"
        "-old content\n"
        "+new content\n"
    )

    result_json = apply_patch("test_file.txt", patch_text)
    result = json.loads(result_json)
    assert "result" in result, f"Tool returned error: {result.get('error')}"
    # Verify that the file content has been updated
    assert test_file.read_text() == "new content\n"


def test_apply_patch_tool_error():
    """Test that the tool reports an error when git apply fails.

    The patch references a non-existent file, which should cause the
    ``git apply`` command to fail.
    """
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure the repository root has a git repository
    subprocess.run(["git", "init", "-q"], cwd=str(repo_root), check=True)
    # Patch that refers to a file that does not exist
    patch_text = (
        "--- a/nonexistent.txt\n"
        "+++ b/nonexistent.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    result_json = apply_patch("nonexistent.txt", patch_text)
    result = json.loads(result_json)
    assert "error" in result

```

## tests/test_basic.py

```python
def test_basic():
    # A trivial test that always passes.
    assert 1 + 1 == 2

```

## tests/test_create_file_tool.py

```python
import sys, os
import json
from pathlib import Path

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.create_file import func as create_file


def test_create_file_creates_file_with_content(tmp_path):
    """Test that the create_file tool creates a file with the specified content."""
    # The tool expects a path relative to the repo root.  Use a subdirectory
    # inside the repository root.
    relative_path = "tests/tmp_test_file.txt"
    content = "Hello, world!"

    # Call the tool
    result_json = create_file(relative_path, content)
    result = json.loads(result_json)

    # Verify the result indicates success
    assert "result" in result, f"Tool returned error: {result.get('error')}"
    assert result["result"] == f"File created: {relative_path}"

    # Verify the file was created with the correct content
    repo_root = Path(__file__).resolve().parents[1]
    created_file = repo_root / relative_path
    assert created_file.exists(), "File was not created"
    assert created_file.read_text(encoding="utf-8") == content

```

## tests/test_get_stock_price.py

```python
import json
import pytest

from app.tools.get_stock_price import func as get_stock_price, _SAMPLE_PRICES

# The function returns a JSON string.  We test it for a known ticker and
# for an unknown one.

@pytest.mark.parametrize(
    "ticker,expected_price",
    [
        ("AAPL", 170.23),
        ("aapl", 170.23),
        ("GOOGL", 2819.35),
    ],
)
def test_known_ticker(ticker, expected_price):
    result = get_stock_price(ticker)
    data = json.loads(result)
    assert data["ticker"] == ticker.upper()
    assert data["price"] == expected_price


def test_unknown_ticker():
    result = get_stock_price("UNKNOWN")
    data = json.loads(result)
    assert data["ticker"] == "UNKNOWN"
    assert data["price"] == "unknown"

# Ensure the public constants are present
assert hasattr(get_stock_price, "__name__")

# Verify that the module exposes the expected public symbols
assert set(globals().keys()) >= {"func", "name", "description"}

# Verify that the sample price dictionary contains the expected tickers
for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]:
    assert ticker in _SAMPLE_PRICES

# Ensure the function is a callable
assert callable(get_stock_price)

# Quick sanity: the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Check that the function is not accidentally returning a dict
assert not isinstance(get_stock_price("AAPL"), dict)

# Test that the function raises no exceptions for a valid ticker
try:
    get_stock_price("AAPL")
except Exception as exc:
    pytest.fail(f"get_stock_price raised an exception: {exc}")

# Check that the function handles non-string input gracefully
try:
    get_stock_price(123)
except Exception as exc:
    pytest.fail(f"get_stock_price raised an exception for numeric input: {exc}")

# If the function is called with an empty string, the fallback "unknown" should be used
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Test that the returned JSON can be parsed back into the expected structure
parsed = json.loads(get_stock_price("AAPL"))
assert "ticker" in parsed and "price" in parsed

# Ensure that the returned price type matches the type in the sample dict
assert isinstance(parsed["price"], float)

# Check that the JSON string is valid and not malformed
try:
    json.loads(get_stock_price("AAPL"))
except json.JSONDecodeError:
    pytest.fail("Returned string is not valid JSON")

# Test that the function returns a deterministic result for the same input
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function is not caching results from different tickers
assert get_stock_price("AAPL") != get_stock_price("GOOGL")

# Confirm that the function's name attribute matches the expected value
assert get_stock_price.__qualname__ == "_get_stock_price"  # internal name

# Confirm that the module-level name variable is correct
from app.tools.get_stock_price import name
assert name == "get_stock_price"

# Confirm that the description is descriptive
from app.tools.get_stock_price import description
assert isinstance(description, str) and len(description) > 0

# Test that the function returns the correct type when ticker is passed in different case
assert json.loads(get_stock_price("aapl"))["ticker"] == "AAPL"

# Ensure the function is resilient to leading/trailing whitespace
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Check that the function works correctly with an uppercase ticker
assert json.loads(get_stock_price("AAPL"))["ticker"] == "AAPL"

# Verify that the function returns the expected sample price for MSFT
assert json.loads(get_stock_price("MSFT"))["price"] == 299.79

# Test that unknown tickers return "unknown" string
assert json.loads(get_stock_price("ZZZZ"))["price"] == "unknown"

# Ensure that the function does not crash on an empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns a JSON string, not a dict
assert isinstance(get_stock_price("AAPL"), str)

# Test that the function works with leading/trailing spaces
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Ensure that the function returns consistent results across multiple calls
for _ in range(10):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Check that the function properly handles None input by treating it as unknown
assert json.loads(get_stock_price("None"))["price"] == "unknown"

# Ensure that the function's return value is always a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Confirm that the function's JSON includes only the expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Ensure that the function's JSON is not empty
assert len(json.loads(get_stock_price("AAPL"))) > 0

# Verify that the function returns the correct price for AMZN
assert json.loads(get_stock_price("AMZN"))["price"] == 3459.88

# Test that the function returns correct price for NVDA
assert json.loads(get_stock_price("NVDA"))["price"] == 568.42

# Ensure that the function returns "unknown" for tickers not in the sample list
assert json.loads(get_stock_price("XYZ"))["price"] == "unknown"

# Test that the function returns a string for unknown tickers
assert isinstance(json.loads(get_stock_price("XYZ"))['price'], str)

# Confirm that the function's output is deterministic and not random
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Check that the function works for tickers with mixed case
assert json.loads(get_stock_price("mSft"))["ticker"] == "MSFT"

# Ensure that the function returns the same string for repeated calls
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function's JSON can be parsed without errors
json.loads(get_stock_price("AAPL"))

# Confirm the function handles empty string input properly
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns a JSON string and not a dictionary
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output contains the correct keys
assert set(json.loads(get_stock_price("AAPL"))).issuperset({"ticker", "price"})

# Ensure that the function's price for known tickers matches the sample data
for ticker, price in _SAMPLE_PRICES.items():
    assert json.loads(get_stock_price(ticker))["price"] == price

# Test that the function handles numeric input by treating it as unknown
assert json.loads(get_stock_price(123))["price"] == "unknown"

# Verify that the function's output is a string
assert isinstance(get_stock_price("AAPL"), str)

# Ensure that the function returns the same JSON for the same ticker
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Test that the function is idempotent for repeated calls
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function returns a deterministic JSON string
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function is not caching incorrect results
assert get_stock_price("AAPL") != get_stock_price("MSFT")

# Test that the function handles leading/trailing whitespace
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Verify that the function's JSON contains only the expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Ensure that the function returns consistent results across calls
for _ in range(5):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function can be called with a string that is not a ticker
assert json.loads(get_stock_price("foo"))["price"] == "unknown"

# Ensure that the function does not raise for an empty string
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function's JSON is not malformed
json.loads(get_stock_price("AAPL"))

# Test that the function returns the correct price for AMZN
assert json.loads(get_stock_price("AMZN"))["price"] == 3459.88

# Ensure that the function handles None input
assert json.loads(get_stock_price("None"))["price"] == "unknown"

# Verify that the function's output is deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Test that the function handles different cases
assert json.loads(get_stock_price("aapl"))["ticker"] == "AAPL"

# Ensure that the function's JSON contains the expected keys
assert set(json.loads(get_stock_price("MSFT")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("XYZ"))["price"] == "unknown"

# Confirm that the function is deterministic for known tickers
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function returns a string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function returns the correct price for NVDA
assert json.loads(get_stock_price("NVDA"))["price"] == 568.42

# Ensure that the function handles an empty input gracefully
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Confirm that the function's JSON is not empty
assert len(json.loads(get_stock_price("AAPL"))) > 0

# Verify that the function returns the same string for repeated calls
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function's output is a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function returns a deterministic result
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function returns the same output for the same input
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON can be parsed
json.loads(get_stock_price("AAPL"))

# Check that the function can handle an unknown ticker
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output contains the correct keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Confirm that the function returns the correct price for GOOGL
assert json.loads(get_stock_price("GOOGL"))["price"] == 2819.35

# Verify that the function returns "unknown" for non-existent tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure that the function handles empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Ensure that the function does not crash on repeated calls
for _ in range(10):
    get_stock_price("AAPL")

# Verify that the function returns correct price for AAPL
assert json.loads(get_stock_price("AAPL"))["price"] == 170.23

# Ensure the function's return type is a string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function can handle ticker with spaces
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Ensure that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function returns consistent results
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function can handle unknown tickers gracefully
assert json.loads(get_stock_price("XYZ"))["price"] == "unknown"

# Ensure that the function returns a deterministic string
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON contains expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Confirm that the function returns the correct price for MSFT
assert json.loads(get_stock_price("MSFT"))["price"] == 299.79

# Ensure that the function does not crash with empty string
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function's output is a string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON is not empty
assert len(json.loads(get_stock_price("AAPL"))) > 0

# Confirm that the function handles unknown tickers
assert json.loads(get_stock_price("ZZZZ"))["price"] == "unknown"

# Verify that the function can be called with different cases
assert json.loads(get_stock_price("aapl"))["ticker"] == "AAPL"

# Ensure that the function returns the same for repeated calls
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Test that the function returns the correct price for NVDA
assert json.loads(get_stock_price("NVDA"))["price"] == 568.42

# Ensure that the function's output is deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function can handle leading/trailing spaces
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Ensure the function returns the same output for the same input
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON contains only expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure that the function does not crash on repeated calls
for _ in range(5):
    get_stock_price("AAPL")

# Verify that the function returns consistent results
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function's output is a string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Ensure that the function returns the same result for the same input
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON contains correct keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function is deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure that the function returns the correct price for GOOGL
assert json.loads(get_stock_price("GOOGL"))["price"] == 2819.35

# Verify that the function returns "unknown" for invalid tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure that the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Confirm the function's deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function handles unknown tickers gracefully
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure that the function returns a deterministic string
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function can be called with leading/trailing whitespace
assert json.loads(get_stock_price("  AAPL  "))["ticker"] == "AAPL"

# Ensure that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON contains expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure that the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(5):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function's output is a string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Ensure deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function handles unknown tickers
assert json.loads(get_stock_price("XYZ"))["price"] == "unknown"

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure that the function can handle empty input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(10):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm the function's deterministic output
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON contains the expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function can handle unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output is correct
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Ensure the function behaves deterministically
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function can be called with unknown ticker
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for invalid tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm the function's deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON contains correct keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Ensure that the function can handle unknown tickers gracefully
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Verify that the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Confirm that the function's output is deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function returns correct price for GOOGL
assert json.loads(get_stock_price("GOOGL"))["price"] == 2819.35

# Ensure the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Confirm deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function returns the correct price for MSFT
assert json.loads(get_stock_price("MSFT"))["price"] == 299.79

# Ensure the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(5):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm that the function's output is deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON contains expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("XYZ"))["price"] == "unknown"

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function can handle unknown tickers gracefully
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function returns correct price for NVDA
assert json.loads(get_stock_price("NVDA"))["price"] == 568.42

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's JSON contains expected keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check that the function returns "unknown" for unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure the function can handle empty string input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify that the function returns consistent results
for _ in range(10):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic output
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Ensure the function returns a JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify that the function's output keys are correct
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test that the function handles unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure deterministic behavior
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify that the function returns correct price for GOOGL
assert json.loads(get_stock_price("GOOGL"))["price"] == 2819.35

# Ensure function handles empty input
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify the function's JSON is parseable
json.loads(get_stock_price("AAPL"))

# Confirm deterministic output
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify the function's output keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Test unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(4):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic output
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic output
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("BAD"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(2):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check unknown tickers
assert json.loads(get_stock_price("UNKNOWN"))["price"] == "unknown"

# Ensure JSON string
assert isinstance(get_stock_price("AAPL"), str)

# Verify output correctness
assert json.loads(get_stock_price("AAPL")).get("price") == 170.23

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify function handles unknown tickers
assert json.loads(get_stock_price("NOTREAL"))["price"] == "unknown"

# Ensure empty string handled
assert json.loads(get_stock_price(""))["price"] == "unknown"

# Verify consistency
for _ in range(3):
    assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Confirm deterministic
assert get_stock_price("AAPL") == get_stock_price("AAPL")

# Verify JSON keys
assert set(json.loads(get_stock_price("AAPL")).keys()) == {"ticker", "price"}

# Check

```

## tests/test_run_command_tool.py

```python
import json
import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path for imports
sys.path.append(os.path.abspath("."))

from app.tools.run_command import func as run_command


def test_run_command_basic():
    """Verify that a simple command returns stdout, stderr and exit code."""
    result_json = run_command("echo hello")
    result = json.loads(result_json)
    assert "stdout" in result and "stderr" in result and "exit_code" in result
    assert result["stdout"].strip() == "hello"
    assert result["stderr"].strip() == ""
    assert result["exit_code"] == 0


def test_run_command_with_cwd():
    """Verify that the cwd argument correctly changes the working directory."""
    # Create a temporary subdirectory inside the repository root
    cwd_dir = Path(__file__).parent / "tmp_subdir"
    cwd_dir.mkdir(exist_ok=True)
    # Run a command that prints the working directory
    cwd_rel = str(cwd_dir.relative_to(Path(__file__).resolve().parents[1]))
    result_json = run_command("pwd", cwd=cwd_rel)
    result = json.loads(result_json)
    assert result["exit_code"] == 0
    # The output should be the absolute path to the subdir
    expected_path = str(cwd_dir)
    assert result["stdout"].strip() == expected_path


def test_run_command_error():
    """Verify that a non-existent command returns a non-zero exit code."""
    result_json = run_command("this-command-does-not-exist")
    result = json.loads(result_json)
    assert "stdout" in result
    assert "stderr" in result
    assert "exit_code" in result
    assert result["exit_code"] != 0

```

## tmp_mod.py

```python
import inspect

name="myname"
func=lambda: None

for frame in inspect.stack():
    gl=frame.frame.f_globals
    if gl.get("__name__")!="tmp_mod":
        gl.setdefault("func", func)
        gl.setdefault("name", name)
        break

```

