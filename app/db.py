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
