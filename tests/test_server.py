"""Unit tests for :mod:`app.server`.

The tests focus on the FastAPI endpoint that proxies to the Llama
language model.  The real Llama server is not available during unit
testing, so the :class:`app.llama_client.LlamaClient` is monkey‑patched
to return a deterministic token stream.

Two tests are provided:

``test_chat_endpoint``
    Starts the FastAPI test client, sends a ``POST`` request to the
    ``/chat/{agent_id}`` endpoint and asserts that the response is
    streamed correctly.

``test_llama_server_reachable``
    Verifies that the :class:`LlamaClient` can parse a streamed
    ``/v1/chat/completions`` response.  The network call is mocked so
    the test does not depend on an external server.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import pytest
from fastapi.testclient import TestClient
from requests.models import Response

from app.server import app as fastapi_app, LlamaClient


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client.

    The fixture is intentionally tiny – the FastAPI application is
    already defined in :mod:`app.server`.
    """

    return TestClient(fastapi_app)


async def _mock_chat_generator(prompt: str, session_id: str | None = None) -> List[str]:
    """Return a deterministic list of tokens.

    The real :meth:`LlamaClient.chat` is an async generator.  In tests
    we simply return a list that will be yielded by the generator
    wrapper in :func:`app.server._token_generator`.
    """

    # For the purpose of the test we ignore the prompt and session id.
    return ["Hello", " ", "world", "!"]


def test_chat_endpoint(client, monkeypatch):
    """Test that the chat endpoint streams tokens correctly.

    The :class:`LlamaClient` is monkey‑patched to return a mocked
    generator yielding a small token sequence.  The test then
    validates the HTTP status code and the concatenated token stream.
    """

    async def mock_chat(self, prompt: str, session_id: str | None = None):  # pragma: no cover - used by test
        for token in await _mock_chat_generator(prompt, session_id):
            yield token

    # Replace the real chat method with the mock.
    monkeypatch.setattr(LlamaClient, "chat", mock_chat)
    # Avoid touching the real database during the test.
    from app import chat_history
    monkeypatch.setattr(chat_history, "insert", lambda *a, **k: None)

    agent_id = "test-agent"
    response = client.post(
        f"/chat/{agent_id}", json={"prompt": "Hi", "session_id": "abc"}
    )
    assert response.status_code == 200
    # The streaming response concatenates all tokens.
    assert response.text == "Hello world!"


def test_llama_server_reachable(monkeypatch):
    """Ensure :class:`LlamaClient` can consume a streamed response.

    The network call is mocked to return a minimal successful response
    containing two token lines and a ``[DONE]`` marker.  The test
    exercises :meth:`LlamaClient.chat` directly.
    """

    # Create a fake response that mimics the llama‑server streaming
    # behaviour.
    class DummyResponse(Response):
        def __init__(self):
            super().__init__()
            self.status_code = 200
            # Provide a list of bytes objects; the client expects
            # iter_lines to yield each line.
            self._lines = [
                b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n",
                b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n",
                b"data: [DONE]\n",
            ]

        def iter_lines(self, **kwargs):  # pragma: no cover - trivial
            for line in self._lines:
                yield line

    # Replace the internal _post method to return our dummy response.
    def _post_stub(self, path: str, payload: dict, stream: bool = False):  # pragma: no cover - used by test
        return DummyResponse()

    monkeypatch.setattr(LlamaClient, "_post", _post_stub)

    client = LlamaClient()
    tokens: List[str] = []
    # Run the async generator synchronously.
    async def _collect():  # pragma: no cover - helper for async test
        async for token in client.chat("test", None):
            tokens.append(token)
        return tokens

    collected = asyncio.run(_collect())
    tokens = collected
    # Two tokens should have been yielded.
    assert tokens == ["Hello", " world"]