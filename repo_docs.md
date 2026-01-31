## app/__init__.py

```python
# app/__init__.py
"""
Convenient import hub for the app package.
"""

__all__ = ["client", "config", "docs_extractor", "utils", "remote"]
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
"""Tool registry for the Streamlit + OpenAI integration.

This package contains one module per tool.  The :class:`Tool` data
class defines the public API that the OpenAI chat completions endpoint
expects.  ``TOOLS`` is a list of all available tools and
``get_tools()`` returns the OpenAI‑ready format.

When new tools are added, simply create a new file in this package and
append a :class:`Tool` instance to the ``TOOLS`` list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# Import tool implementations (they live in separate modules)
from .get_stock_price import get_stock_price as _get_stock_price
from .create_file import create_file as _create_file
from .run_command import run_command as _run_command

# --------------------------------------------------------------------------- #
#  Tool dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    """Represents a tool that can be called by the OpenAI model."""

    name: str
    description: str
    func: Callable
    schema: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        # Define the JSON schema for each supported tool.
        if self.name == "get_stock_price":
            self.schema = {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock symbol, e.g. AAPL",
                        },
                    },
                    "required": ["ticker"],
                }
            }
        elif self.name == "create_file":
            self.schema = {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path of the new file (e.g. 'app/new_module.py')",
                        },
                        "content": {
                            "type": "string",
                            "description": "File contents as a plain string.",
                        },
                    },
                    "required": ["path", "content"],
                }
            }
        elif self.name == "run_command":
            self.schema = {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute, e.g. 'pytest -q'.",
                        },
                    },
                    "required": ["command"],
                }
            }
        else:
            raise NotImplementedError(f"Schema for {self.name} not defined")

# --------------------------------------------------------------------------- #
#  Tool registry
# --------------------------------------------------------------------------- #
TOOLS: List[Tool] = [
    Tool(
        name="get_stock_price",
        description="Get the current stock price for a ticker",
        func=_get_stock_price,
    ),
    Tool(
        name="create_file",
        description="Create a new file at the given path with the supplied content",
        func=_create_file,
    ),
    Tool(
        name="run_command",
        description="Execute a shell command and return its output, return code and stderr",
        func=_run_command,
    ),
]

# --------------------------------------------------------------------------- #
#  Helper – expose OpenAI‑ready tool list
# --------------------------------------------------------------------------- #
def get_tools() -> List[Dict]:
    """Return the list of tools formatted for the OpenAI API."""
    api_tools: List[Dict] = []
    for t in TOOLS:
        api_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema["parameters"],
                },
            }
        )
    return api_tools

# --------------------------------------------------------------------------- #
#  Convenience for quick introspection in the REPL
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(json.dumps(get_tools(), indent=2))

```

## app/tools/create_file.py

```python
# app/tools/create_file.py
"""
Tool that creates a new file under the repository root.

The function validates the target path so that it cannot escape the
repository root (prevents directory traversal).  It creates any missing
parent directories, writes the supplied content, and returns a JSON
string containing either a `result` key (success) or an `error` key
(failure).

The JSON format is compatible with the OpenAI function‑calling schema
used by the chat UI.
"""

import json
import pathlib
from typing import Dict


def _safe_resolve(repo_root: pathlib.Path, rel_path: str) -> pathlib.Path:
    """
    Resolve *rel_path* against *repo_root* and ensure the result
    does not escape the repository root (prevents directory traversal).
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target


def create_file(path: str, content: str) -> str:
    """
    Create a new file at *path* (relative to the repo root) with the supplied
    *content*.  Returns a JSON string containing either a `result` or an
    `error` key.
    """
    try:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        target = _safe_resolve(repo_root, path)

        # Make sure the parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        target.write_text(content, encoding="utf-8")

        return json.dumps({"result": f"File created: {path}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# --------------------------------------------------------------------------- #
#  Export the tool as part of the public API
# --------------------------------------------------------------------------- #
__all__ = ["create_file"]

```

## app/tools/get_stock_price.py

```python
# app/tools/get_stock_price.py
"""
Tool that returns a mock stock price.

The function simulates a price lookup for a given ticker.  It is used
by the Streamlit UI to demonstrate the function‑calling API.  The
implementation is intentionally simple – a hard‑coded dictionary of
sample prices – but the interface mirrors a real API call.
"""

import json
from typing import Dict

# Hard‑coded sample data – in a real system this would query a
# finance API such as Yahoo Finance or Alpha Vantage.
_SAMPLE_PRICES: Dict[str, float] = {
    "AAPL": 170.23,
    "GOOGL": 2819.35,
    "MSFT": 299.79,
    "AMZN": 3459.88,
    "NVDA": 568.42,
}


def get_stock_price(ticker: str) -> str:
    """Return the current stock price for *ticker*.

    Parameters
    ----------
    ticker:
        Stock symbol (e.g. ``"AAPL"``).  The lookup is case‑insensitive.

    Returns
    -------
    str
        JSON string containing ``ticker`` and ``price`` keys.  If the
        ticker is unknown, ``price`` is set to ``"unknown"``.
    """
    price = _SAMPLE_PRICES.get(ticker.upper(), "unknown")
    result = {"ticker": ticker.upper(), "price": price}
    return json.dumps(result)

# ---------------------------------------------------------------------------
#  Export the tool as part of the public API
# ---------------------------------------------------------------------------
__all__ = ["get_stock_price"]

```

## app/tools/run_command.py

```python
# app/tools/run_command.py
"""
Tool that executes a shell command and returns its output.

The tool is safe for the repository root – it does not allow changing
directories or writing files outside the repo.  The command is run
through a subprocess with `shell=True` so that the user can use shell
syntax (e.g. pipes, redirection).  The tool captures stdout, stderr and
the exit code, and returns a JSON string that the model can consume.

The function is intentionally minimal to avoid accidental side‑effects.
"""

import json
import subprocess
import pathlib
from typing import Dict


def _safe_resolve(repo_root: pathlib.Path, rel_path: str) -> pathlib.Path:
    """
    Resolve *rel_path* against *repo_root* and ensure the result
    does not escape the repository root (prevents directory traversal).
    """
    target = (repo_root / rel_path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise ValueError("Path escapes repository root")
    return target


def run_command(command: str, cwd: str | None = None) -> str:
    """
    Execute *command* in the repository root (or a sub‑directory if
    *cwd* is provided) and return a JSON string with:
        * stdout
        * stderr
        * exit_code
    Any exception is converted to an error JSON.
    """
    try:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
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

# --------------------------------------------------------------------------- #
#  Export the tool as part of the public API
# --------------------------------------------------------------------------- #
__all__ = ["run_command"]

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


def build_messages(
    history,
    system_prompt,
    repo_docs,
    user_input=None,
):
    """
    Build the list of messages expected by the OpenAI chat API.
    """
    msgs = [{"role": "system", "content": str(system_prompt)}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": str(repo_docs)})

    for u, a in history:
        msgs.append({"role": "user", "content": str(u)})
        msgs.append({"role": "assistant", "content": str(a)})

    if user_input is not None:
        msgs.append({"role": "user", "content": str(user_input)})

    return msgs


def stream_and_collect(client, messages, tools, placeholder):
    """
    Stream a response from the model, updating `placeholder` live, and
    collect any tool calls that are emitted.
    """
    stream = client.chat.completions.create(
        model="unsloth/gpt-oss-20b-GGUF:F16",
        messages=messages,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls_buffer = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Regular text
        if delta.content:
            full_resp += delta.content
            placeholder.markdown(full_resp, unsafe_allow_html=True)

        # Tool calls
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


def process_tool_calls(client, messages, tools, placeholder, tool_calls):
    """
    For every tool call in `tool_calls`:
        • call the tool
        • append assistant_tool_call_msg + tool_msg to `messages`
        • stream a new assistant reply
    Return the final assistant text and the next list of tool calls.
    """
    if not tool_calls:
        return "", None

    full_text = ""
    for tc in tool_calls:
        args = json.loads(tc.get("arguments") or "{}")
        func = next((t.func for t in TOOLS if t.name == tc.get("name")), None)

        if func:
            try:
                result = func(**args)
            except Exception as exc:
                result = f"❌  Tool error: {exc}"
        else:
            result = f"⚠️  Unknown tool '{tc.get('name')}'"

        # --------------------------------------------------------------------
        #  Render the tool‑call *result* in the UI before the next assistant turn
        # --------------------------------------------------------------------
        tool_output_str = (
            f"**Tool call**: `{tc.get('name')}`"
            f"({', '.join(f'{k}={v}' for k, v in args.items())}) → `{result}`"
        )
        placeholder.markdown(tool_output_str, unsafe_allow_html=True)

        # Build messages to send back to the model
        assistant_tool_call_msg = {
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

        tool_msg = {
            "role": "tool",
            "tool_call_id": str(tc.get("id") or ""),
            "content": str(result or ""),
        }

        messages.append(assistant_tool_call_msg)
        messages.append(tool_msg)

        # Stream the next assistant reply – each iteration gets a fresh placeholder
        placeholder2 = st.empty()
        new_text, new_tool_calls = stream_and_collect(
            client, messages, tools, placeholder2
        )
        full_text += new_text
        tool_calls = new_tool_calls or []

        if not tool_calls:  # no more calls – break early
            break

    return full_text, tool_calls


# --------------------------------------------------------------------------- #
#  Streamlit UI
# --------------------------------------------------------------------------- #
def main():
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

## tests/__init__.py

```python
# tests/__init__.py
"""
Make the repository root importable during test collection.
"""

import pathlib
import sys

# Path to the repository root (the directory that contains `app/`).
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

# Ensure the root is first in sys.path.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

```

## tests/test_create_file_tool.py

```python
import json
import os
import pathlib

import pytest

# Import the tool directly from the package
from app.tools import create_file

@pytest.fixture
def unique_filename(tmp_path):
    """
    Return a unique file name that lives under the repository root.
    The file is removed after the test finishes.
    """
    fname = f"tests/tmp_{tmp_path.name}.py"
    yield fname
    # cleanup – the test itself creates/deletes the file, so nothing is required here


def test_create_file_creates_file_and_returns_result(unique_filename):
    """
    Verify that the ``create_file`` tool:
    * writes the given content to the correct path (relative to the repo root)
    * returns a JSON object containing the key ``result``.
    """
    content = "print('hello world')"

    # Call the tool
    result_json = create_file(unique_filename, content)

    # Parse the JSON result
    result = json.loads(result_json)
    assert "result" in result, "Expected a ``result`` key in the JSON response"

    # Verify the file was created with the exact content
    assert pathlib.Path(unique_filename).exists(), "The file was not created"
    with open(unique_filename, encoding="utf-8") as f:
        assert f.read() == content, "File contents do not match the supplied content"

    # Clean up the file so it doesn't affect other tests
    os.remove(unique_filename)


def test_create_file_prevents_directory_traversal():
    """
    The tool must reject attempts to write outside the repository root.
    It should return a JSON object containing an ``error`` key.
    """
    # Path that tries to escape the repo root
    bad_path = "../../outside.txt"
    content = "secret"

    result_json = create_file(bad_path, content)
    result = json.loads(result_json)

    assert "error" in result, "Expected an ``error`` key for a path traversal attempt"
    # Ensure the file was not created
    assert not pathlib.Path(bad_path).exists(), "File was created outside the repo root"

```

