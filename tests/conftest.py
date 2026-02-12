"""Pytest configuration.

The repository root is not automatically added to ``sys.path`` when
running tests from the ``tests`` directory.  Adding an explicit
``conftest.py`` ensures that the :mod:`app` package can be imported
directly by test modules.
"""

import sys
from pathlib import Path

# Insert the repository root at the start of ``sys.path`` so that
# imports such as ``import app.agent`` resolve correctly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))