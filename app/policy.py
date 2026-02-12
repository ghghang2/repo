"""Simple policy hook for the supervisor.

The policy function decides whether an interjection should be sent to the
agent based on the content of an :class:`AgentEvent`.  For the MVP we
implement a very small heuristic:

* If the event's content contains the word ``error`` or ``mistake`` the
  policy returns ``True``.
* Otherwise ``False``.

The function is deliberately pure and has no side‑effects so that it can
be unit‑tested in isolation.
"""

from __future__ import annotations

import re
from typing import Iterable, Dict, Any

from .agent import AgentEvent

_INTERJECTION_PATTERNS = [re.compile(r"\berror\b", re.IGNORECASE),
                          re.compile(r"\bmistake\b", re.IGNORECASE)]


def should_interject(event: AgentEvent | Dict[str, Any]) -> bool:
    """Return ``True`` if the policy decides an interjection is needed.

    The policy inspects the *token* field of a ``token`` event and the
    *content* field of any event.  The implementation accepts either
    an :class:`AgentEvent` instance or a plain ``dict`` (the format used
    by :class:`AgentProcess`).
    """
    # Normalise to dict
    if isinstance(event, dict):
        token = event.get("token")
        content = event.get("content", "")
    else:
        token = getattr(event, "token", None)
        content = getattr(event, "content", "")

    if token:
        for pat in _INTERJECTION_PATTERNS:
            if pat.search(token):
                return True
    for pat in _INTERJECTION_PATTERNS:
        if pat.search(content):
            return True
    # Also check for prompt field in done events
    prompt = event.get("prompt", "") if isinstance(event, dict) else getattr(event, "prompt", "")
    for pat in _INTERJECTION_PATTERNS:
        if pat.search(prompt):
            return True
    return False


__all__ = ["should_interject"]
