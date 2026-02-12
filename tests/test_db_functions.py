"""Tests for the persistence layer in :mod:`app.db`.

The tests use a temporary SQLite database to avoid polluting the
repository level ``chat_history.db`` file.  The :mod:`app.db` module
exports a mutable ``DB_PATH`` variable which can be patched at import
time, allowing the tests to redirect all file accesses.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List

import pytest

from app import db


@pytest.fixture
def temp_db(tmp_path) -> Path:
    """Return a path to a fresh temporary database file.

    The fixture also patches :data:`app.db.DB_PATH` so that the module
    uses the temporary file for all subsequent operations.
    """

    path = tmp_path / "temp_chat.db"
    db.DB_PATH = path
    return path


def test_init_db_creates_tables_and_index(temp_db: Path) -> None:
    """Calling :func:`app.db.init_db` should create the tables and index."""
    db.init_db()
    # Inspect the schema.
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "chat_log" in tables
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_session" in indexes
    conn.close()


def test_log_and_load_history(temp_db: Path) -> None:
    """Persist two messages and retrieve them back."""
    db.init_db()
    db.log_message("sess1", "user", "Hello")
    db.log_message("sess1", "assistant", "Hi there")
    history = db.load_history("sess1")
    assert history == [("user", "Hello"), ("assistant", "Hi there")]
    # Check limit
    limited = db.load_history("sess1", limit=1)
    assert limited == [("user", "Hello")]


def test_get_session_ids(temp_db: Path) -> None:
    """Return distinct session identifiers."""
    db.init_db()
    db.log_message("s1", "user", "A")
    db.log_message("s2", "assistant", "B")
    db.log_message("s1", "assistant", "C")
    ids = db.get_session_ids()
    assert set(ids) == {"s1", "s2"}