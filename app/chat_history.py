"""Thin wrapper around :mod:`app.db`.

The original repository exposed a :class:`ChatHistory` class used by
the agent process and the FastAPI proxy.  The tests expect this class to
provide :meth:`insert` and :meth:`load_history` methods.  The
implementation simply forwards calls to the module‑level helper
functions defined in :mod:`app.db`.
"""

from . import db


class ChatHistory:
    """Convenience wrapper that mimics the legacy API.

    The wrapper stores no state; each method opens a new database
    connection via :func:`app.db.init_db`.
    """

    def insert(self, session_id: str, role: str, content: str) -> None:
        db.log_message(session_id, role, content)

    def load_history(self, session_id: str, limit: int | None = None):
        return db.load_history(session_id, limit)


# ---------------------------------------------------------------------------
#  Legacy module level API
# ---------------------------------------------------------------------------
# The original repository exposed free‑standing functions.  The FastAPI
# proxy imports ``app.chat_history`` and calls ``insert`` directly.  To
# maintain backward compatibility we provide thin wrappers that simply
# delegate to a singleton ``ChatHistory`` instance.

_singleton = ChatHistory()

def insert(session_id: str, role: str, content: str) -> None:  # pragma: no cover - trivial wrapper
    """Insert a chat line using the singleton wrapper.

    The function is intentionally tiny and mirrors the old public API.
    """
    _singleton.insert(session_id, role, content)

def load_history(session_id: str, limit: int | None = None):  # pragma: no cover - trivial wrapper
    """Return chat history via the singleton wrapper."""
    return _singleton.load_history(session_id, limit)