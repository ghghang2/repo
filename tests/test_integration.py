"""Integration test for Supervisor and Agent.

The test verifies that the policy hook triggers an interjection when the
agent returns a response containing the word ``error``.
"""

import time
import sys
import os
from multiprocessing import Queue

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.supervisor import SupervisorProcess, SupervisorConfig


class DummyLLM:
    """Simple LLM that yields tokens.

    The agent expects a ``stream_chat`` coroutine that yields token strings.
    The dummy splits the prompt into words and yields each as a token.
    """

    async def stream_chat(self, prompt: str):
        for token in prompt.split():
            yield token


def test_supervisor_interjection(monkeypatch):
    # Patch the LlamaClient used by AgentProcess
    monkeypatch.setattr("app.agent.LlamaClient", DummyLLM)

    # Start Supervisor (it will start its own AgentProcess)
    supervisor = SupervisorProcess(SupervisorConfig(agent_name="test"))
    supervisor.start()
    time.sleep(0.1)  # give supervisor and agent time to start

    # Send a message that should trigger policy interjection
    session_id = "sess1"
    prompt = "This will error"
    supervisor.agent_inbox.put({"session_id": session_id, "prompt": prompt})

    # Wait until the agent finishes the original request
    done_received = False
    start = time.time()
    while time.time() - start < 5:
        try:
            ev = supervisor.agent_outbound.get(timeout=0.1)
        except Exception:
            continue
        if ev.get("type") == "done" and ev.get("session_id") == session_id:
            done_received = True
            break
    assert done_received, "Agent did not finish the request"

    # After the agent is done, supervisor should have queued an interjection
    # which will be streamed as tokens on agent_outbound.
    interjection_tokens = []
    start = time.time()
    while time.time() - start < 2:
        try:
            ev = supervisor.agent_outbound.get(timeout=0.1)
        except Exception:
            continue
        if ev.get("type") == "token":
            interjection_tokens.append(ev.get("token"))
            if "Apology" in ev.get("token", ""):
                break
    assert interjection_tokens, "Supervisor did not interject"

    # Cleanup: send shutdown to agent and supervisor
    supervisor.agent_inbox.put({"type": "shutdown"})
    supervisor.terminate()
    supervisor.join()
