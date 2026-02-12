"""Tests for :class:`app.agent.AgentProcess`.

The tests exercise the agent process in isolation.  The real LLM client
is replaced with a lightweight dummy implementation that immediately
yields a couple of tokens.  The tests cover:

* normal chat handling
* interjection handling (the agent treats an ``interjection`` event
  as a regular chat request)
* graceful shutdown via a ``shutdown`` event.
"""

import multiprocessing as mp
import time
import pytest
import asyncio
from app.agent import AgentProcess, AgentEvent
from dummy_client import DummyClient


class DummyClient:
    """Simple LLM client that yields two tokens."""

    async def chat(self, prompt: str):  # pragma: no cover - trivial async generator
        for token in ["A", "B"]:
            await asyncio.sleep(0.01)
            yield token


def start_agent(agent_id: str, inbound: mp.Queue, outbound: mp.Queue) -> mp.Process:
    """Return a started :class:`AgentProcess` that uses :class:`DummyClient`."""
    agent = AgentProcess(agent_id, inbound, outbound, llm_cls=DummyClient)
    agent.start()
    return agent


def test_agent_handles_chat_and_interjection_and_shutdown():
    """Send a normal chat, an interjection, and then shutdown.

    The test verifies that the agent emits the expected token events for
    both requests and that the process terminates after receiving the
    shutdown event.
    """

    inbound = mp.Queue()
    outbound = mp.Queue()
    agent = start_agent("test", inbound, outbound)

    # Normal chat
    inbound.put(AgentEvent(role="user", content="Hi", session_id="s1", prompt="Hi"))
    # Interjection
    inbound.put(
        AgentEvent(
            role="assistant",
            content="",
            session_id="s1",
            type="interjection",
            prompt="Apology",
        )
    )
    # Shutdown
    inbound.put(AgentEvent(role="assistant", content="", type="shutdown"))

    # Collect results until we get four token events or timeout.
    tokens: list[str] = []
    while len(tokens) < 4:
        event: AgentEvent = outbound.get(timeout=2)
        if event.type == "token":
            tokens.append(event.token or "")
    assert tokens == ["A", "B", "A", "B"]
    agent.join(timeout=2)
    assert not agent.is_alive()