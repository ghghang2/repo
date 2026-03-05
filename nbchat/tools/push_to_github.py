# app/push_to_github.py
"""A minimal OpenAI function‑calling tool that pushes the local
repository to GitHub.

The tool is deliberately lightweight: it relies on the existing
``RemoteClient`` and the configuration found in ``repo_config.yaml``.
It performs the following steps:

1. Load the target repository name from ``config.py`` (or from the
   optional ``repo_name`` argument).
2. Ensure the repository exists on GitHub.
3. Attach the HTTPS remote.
4. Ensure a local ``main`` branch exists and is set to track
   ``origin/main``.
5. Pull the latest changes (optionally rebasing).
6. Commit all staged changes with the supplied ``commit_message``.
7. Push the local ``main`` branch to GitHub.

The function returns a JSON string suitable for use with OpenAI
function‑calling: ``{"status": "success", "repo": "<user>/<repo>"}``.
On error a JSON object with an ``error`` key is returned.
"""

from pathlib import Path
import json
from typing import Optional

from nbchat.core.remote import RemoteClient
from nbchat.core.config import REPO_NAME


def push_to_github(
    repo_name: Optional[str] = None,
    commit_message: str = "Auto commit",
    rebase: bool = False,
) -> str:
    """Push the current repository to GitHub.

    Parameters
    ----------
    repo_name:
        Optional override for the repository name.  If ``None`` the
        default from :data:`nbchat.core.config.REPO_NAME` is used.
    commit_message:
        Commit message for the auto commit.  Defaults to ``"Auto commit"``.
    rebase:
        Whether to rebase during pull.  Defaults to ``False`` to mirror
        the original behaviour.
    """
    try:
        target_repo = repo_name or REPO_NAME
        client = RemoteClient(Path("."))
        # 1. Ensure the GitHub repo exists
        client.ensure_repo(target_repo)
        # 2. Attach the HTTPS remote
        client.attach_remote(target_repo)
        # 3. Ensure local main branch exists and tracks remote
        client.ensure_main_branch()
        # 4. Fetch & pull
        client.fetch()
        client.pull(rebase=rebase)
        # 5. Commit all changes
        client.commit_all(commit_message)
        # 6. Push to GitHub
        client.push()
        return json.dumps({"status": "success", "repo": f"{client.user.login}/{target_repo}"})
    except Exception as exc:  # pragma: no cover - defensive
        return json.dumps({"error": str(exc)})

# ---------------------------------------------------------------------------
# Tool Definition for OpenAI function‑calling
# ---------------------------------------------------------------------------
func = push_to_github
name = "push_to_github"
description = "Push the current local repository to GitHub. Optionally specify a repository name, commit message, and whether to rebase during pull."

schema = {
    "parameters": {
        "type": "object",
        "properties": {
            "repo_name": {"type": "string"},
            "commit_message": {"type": "string"},
            "rebase": {"type": "boolean"},
        },
        "required": [],
    }
}

__all__ = ["push_to_github", "func", "name", "description", "schema"]
