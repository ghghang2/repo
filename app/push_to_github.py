# scripts/push_to_github.py
"""
Entry point that wires the `RemoteClient` together.
"""

from pathlib import Path
from .remote import RemoteClient, REPO_NAME

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