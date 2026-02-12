"""Tests for :class:`app.agent.AgentProcess`.

The tests use a *dummy* LLM client that simply streams a predefined list
of tokens.  The agent is spawned in a separate process; the test
verifies that the process has its own PID and that the inbox queue
correctly forwards messages.
"""

import os
import sys
# Ensure root package is importable during pytest collection
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
print("Test sys.path after adding root:", sys.path)
import time
from multiprocessing import Queue
from unittest.mock import patch

import pytest

print("Test import sys.path:", sys.path)
from app.agent import AgentProcess


class DummyClient:
    """A very small stand‑in for :class:`app.llama_client.LlamaClient`."""

    async def stream_chat(self, prompt: str):  # pragma: no cover - trivial
        # Emit a token for each word in the prompt
        for word in prompt.split():
            yield word
        # Simulate a final token
        yield "[END]"

def test_agent_process_runs_in_separate_process_and_processes_queue():
    inbound = Queue()
    outbound = Queue()
    agent = AgentProcess("test", inbound, outbound)
    with patch("app.agent.LlamaClient", DummyClient):
        agent.start()
        try:
            # Verify that the agent is indeed a separate process
            assert agent.pid != os.getpid()

            # Send a chat request
            session_id = "sess1"
            prompt = "hello world"
            inbound.put({"session_id": session_id, "prompt": prompt})
            # Collect tokens from outbound queue
            tokens = []
            while True:
                item = outbound.get(timeout=1)
                if item["type"] == "token":
                    tokens.append(item["token"])
                elif item["type"] == "done":
                    break
            assert tokens == ["hello", "world", "[END]"]
            # Ensure that the done message references the correct session
            assert item["session_id"] == session_id
        finally:
            # Send shutdown signal
            inbound.put({"type": "shutdown"})
            agent.join(timeout=2)