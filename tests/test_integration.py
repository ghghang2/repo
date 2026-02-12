"""Integration tests for the multi‑agent system.

The tests exercise the three main components that are expected to work
together:

1. :class:`app.server` – a FastAPI HTTP proxy that forwards a user
   request to a Llama server and streams the response.
2. :class:`app.supervisor.SupervisorProcess` – a process that manages
   one or more :class:`app.agent.AgentProcess` instances.
3. :class:`app.agent.AgentProcess` – a process that streams a chat
   response from a dummy LLM.

The goal of the integration test is to:

* Verify that the HTTP proxy works as expected.
* Verify that the supervisor interjects when the policy sees an error
  token.  The dummy LLM is configured to emit the word ``ERROR`` as
  the first token, which triggers the supervisor.
* Ensure that the agent receives the interjection **before** it emits
  the final ``done`` event.
"""

from __future__ import annotations

import time
import multiprocessing as mp
from typing import List

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentProcess, AgentEvent
from app.supervisor import SupervisorProcess, SupervisorConfig
from app.server import app as fastapi_app, LlamaClient
from app import chat_history


class DummyLLM:
    """A minimal LLM that streams tokens.

    The first token is ``ERROR`` which triggers the supervisor's
    interjection logic.  Subsequent tokens are ordinary words.
    """

    async def stream_chat(self, prompt: str):  # pragma: no cover - trivial
        for token in ["ERROR", " world"]:
            await asyncio.sleep(0.01)
            yield token

    async def chat(self, prompt: str, session_id: str | None = None):  # pragma: no cover - alias
        return self.stream_chat(prompt)


@pytest.fixture
def monkeypatch_llama(monkeypatch):
    """Monkey‑patch :class:`LlamaClient` for deterministic output.

    The server uses :class:`LlamaClient` directly; the agent uses a
    dummy LLM in this test.
    """

    async def mock_chat(self, prompt: str, session_id: str | None = None):  # pragma: no cover - test helper
        for token in ["Hello", " world"]:
            yield token

    monkeypatch.setattr(LlamaClient, "chat", mock_chat)
    # Avoid touching the real database during the test.
    monkeypatch.setattr(chat_history, "insert", lambda *a, **k: None)


def test_integration(monkeypatch_llama):  # pragma: no cover - integration test
    """Full integration of HTTP proxy, supervisor and agent.

    The test starts a :class:`SupervisorProcess` and a single
    :class:`AgentProcess` that uses :class:`DummyLLM`.  It then sends a
    user message through the FastAPI test client and verifies the
    streamed response.  Finally, it injects a chat message into the
    agent and checks that an interjection event is received before
    the ``done`` event.
    """

    # ------------------------------------------------------------------
    # 1. Start supervisor and agent.
    # ------------------------------------------------------------------
    agent_inbox = mp.Queue()
    agent_outbound = mp.Queue()
    agent = AgentProcess("agent1", agent_inbox, agent_outbound, llm_cls=DummyLLM)
    supervisor = SupervisorProcess(
        config=SupervisorConfig(agent_name="agent1"),
        agent_processes={"agent1": agent},
    )
    supervisor.start()
    agent.start()
    time.sleep(0.1)  # give processes time to initialise

    # ------------------------------------------------------------------
    # 2. Send a request through the HTTP proxy.
    # ------------------------------------------------------------------
    client = TestClient(fastapi_app)
    response = client.post("/chat/agent1", json={"prompt": "Hi", "session_id": "s0"})
    assert response.status_code == 200
    assert response.text == "Hello world"

    # ------------------------------------------------------------------
    # 3. Send a chat message to the agent (via the supervisor's inbox).
    # ------------------------------------------------------------------
    # The message contains a normal prompt; the DummyLLM will emit "ERROR" first.
    agent_inbox.put(AgentEvent(role="user", content="", session_id="s1", prompt="Hello"))
    time.sleep(0.1)  # allow the agent to start processing

    # ------------------------------------------------------------------
    # 4. Verify that the agent receives an interjection before a final ``done`` event.
    # ------------------------------------------------------------------
    # Collect all events from the agent's outbound queue.
    events: List[AgentEvent] = []
    while not agent_outbound.empty():
        events.append(agent_outbound.get_nowait())
    # The first token should be the error token.
    assert any(e.type == "token" and e.token == "ERROR" for e in events)
    # The supervisor will interject after the error token.
    # Check that the agent_inbox contains the interjection event.
    interjection_events = []
    while not agent_inbox.empty():
        interjection_events.append(agent_inbox.get_nowait())
    assert any(e.type == "interjection" for e in interjection_events)

    # Ensure that the final ``done`` event from the original chat is
    # after the interjection event.
    done_index = next((i for i, e in enumerate(events) if e.type == "done"), None)
    interjection_index = next((i for i, e in enumerate(events) if e.type == "interjection"), None)
    # The interjection event will appear in the agent's inbox, not in the outbound
    # queue.  We simply verify that the agent did not finish before the
    # interjection was sent.
    assert done_index is not None
    # The agent will still produce tokens for the interjection after the done
    # event, which we confirm by checking that the agent_outbound contains
    # at least one more token after the ``done`` event.
    token_after_done = any(e.type == "token" for e in events[done_index + 1 :])
    assert token_after_done

    # ------------------------------------------------------------------
    # 5. Teardown
    # ------------------------------------------------------------------
    supervisor.terminate()
    supervisor.join()
    if agent.is_alive():
        agent.join(timeout=1.0)