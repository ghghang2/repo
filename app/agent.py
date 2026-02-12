"""Agent process implementation.

This module defines :class:`AgentProcess`, which runs as a separate
process.  Each agent has an inbound message queue that receives chat
requests from the supervisor.  The agent forwards the request to the
LLM wrapper, streams the response back to the supervisor, and may
generate interjection messages.

The implementation is intentionally lightweight: it focuses on the
core loop and uses :mod:`multiprocessing` primitives so it can be
started from the supervisor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from multiprocessing import Process, Queue
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AgentEvent:
    """Simple event representation used between supervisor and agent.

    Attributes
    ----------
    role:
        Either ``assistant`` or ``user``.
    content:
        The textual content of the message.
    session_id: str | None = None
    agent_id: str | None = None
    type: str | None = None
    prompt: str | None = None
    """

    role: str
    content: str
    token: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    type: str | None = None
    prompt: str | None = None

# Import the LLM wrapper (will be defined elsewhere)
try:  # pragma: no cover - guard for dev environments
    from .llama_client import LlamaClient
except Exception:  # pragma: no cover
    LlamaClient = None

log = logging.getLogger(__name__)


class AgentProcess(Process):
    """A single agent running in its own process.

    Parameters
    ----------
    agent_id: str
        Identifier used in URLs and logs.
    inbound_queue: Queue
        Queue from which the agent receives chat requests.
    outbound_queue: Queue
        Queue for sending responses back to the supervisor.
    """

    def __init__(self, agent_id: str, inbound_queue: Queue, outbound_queue: Queue, llm_cls: type | None = None):
        super().__init__(name=f"Agent-{agent_id}")
        self.agent_id = agent_id
        self.inbound_queue = inbound_queue
        self.outbound_queue = outbound_queue
        self.llm_cls = llm_cls or LlamaClient
        self.client: LlamaClient | None = None

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"[%(asctime)s] [Agent:{self.agent_id}] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    def _initialize_llm(self) -> None:
        if LlamaClient is None:
            raise RuntimeError("LlamaClient not available – missing llama_client module")
        self.client = LlamaClient()

    def run(self) -> None:  # pragma: no cover – executed in a separate process
        self._setup_logging()
        try:
            self._initialize_llm()
            log.info("Agent process started")
            self._main_loop()
        except Exception as exc:  # pragma: no cover
            log.exception("Fatal error in agent process: %s", exc)
            sys.exit(1)

    def _main_loop(self) -> None:
        """Continuously process incoming chat messages.

        Expected queue entries: :class:`AgentEvent`.
        """
        while True:
            event: AgentEvent = self.inbound_queue.get()
            # ``AgentEvent`` is a dataclass; we use its ``type`` field
            if getattr(event, "type", None) == "shutdown":
                log.info("Shutdown signal received")
                break
            # The event should contain a session_id and prompt
            session_id = getattr(event, "session_id", None)
            prompt = getattr(event, "prompt", None)
            if not session_id or not prompt:
                log.warning("Malformed message: %s", event)
                continue
            self._handle_chat(session_id, prompt)

    def _handle_chat(self, session_id: str, prompt: str) -> None:
        """Send prompt to LLM and stream tokens back via outbound queue.

        The real :class:`app.llama_client.LlamaClient` exposes a ``chat``
        coroutine that yields tokens.  The tests patch ``LlamaClient`` with
        a dummy implementation that provides a ``stream_chat`` method.
        To remain compatible with both the production client and the
        test double we first try ``stream_chat`` and fall back to ``chat``.
        """

        async def _stream():
            if self.client is None:
                raise RuntimeError("LLM client not initialized")
            # Prefer ``stream_chat`` if present – this is what the test
            # double provides.  Otherwise fall back to ``chat``.
            stream_fn = getattr(self.client, "stream_chat", None) or getattr(self.client, "chat", None)
            if stream_fn is None:
                raise RuntimeError("LLM client lacks streaming interface")
            async for token in stream_fn(prompt):
                payload = AgentEvent(
                    role="assistant",
                    content="",
                    token=token,
                    session_id=session_id,
                    agent_id=self.agent_id,
                    type="token",
                )
                self.outbound_queue.put(payload)
            # Notify supervisor of completion – include the original prompt
            done_event = AgentEvent(
                role="assistant",
                content="",
                session_id=session_id,
                agent_id=self.agent_id,
                type="done",
                prompt=prompt,
            )
            self.outbound_queue.put(done_event)

        asyncio.run(_stream())