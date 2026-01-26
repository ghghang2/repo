# remote/__init__.py
"""
Public API for the remote adapter.

Only the `RemoteClient` class is meant to be imported by other packages.
"""

from .remote import RemoteClient
from .config import USER_NAME, REPO_NAME, IGNORED_ITEMS