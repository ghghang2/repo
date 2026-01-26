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

from .config import USER_NAME, REPO_NAME, IGNORED_ITEMS

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