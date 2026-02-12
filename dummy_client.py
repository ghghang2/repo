"""Simple dummy LLM client used in tests.

The class is defined in the repository root so that it can be
imported by test modules without being part of the :mod:`tests`
package.  Multiprocessing requires the target to be pickleable, and
placing it in a top‑level module satisfies that constraint.
"""

import asyncio


class DummyClient:
    """Yield two tokens for any prompt."""

    async def chat(self, prompt: str):  # pragma: no cover - trivial async generator
        for token in ["A", "B"]:
            await asyncio.sleep(0.01)
            yield token