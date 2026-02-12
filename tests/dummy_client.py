"""Simple dummy LLM client used in tests.

The class is defined in a separate module so that it can be
pickled by the multiprocessing module when passed to
``AgentProcess``.
"""

import asyncio

class DummyClient:
    async def chat(self, prompt: str):  # pragma: no cover - trivial async generator
        for token in ["A", "B"]:
            await asyncio.sleep(0.01)
            yield token