"""Supervisor process for the multi‑agent system.

The supervisor owns a *policy* that decides whether the agent should
interject in an ongoing conversation.  It communicates with the
``AgentProcess`` via two :class:`multiprocessing.Queue` objects:

* ``agent_inbox`` – the queue the supervisor uses to send messages to the
  agent.
* ``agent_outbound`` – the queue the supervisor listens to for events
  emitted by the agent.

The supervisor runs as a :class:`multiprocessing.Process` that
continually reads from ``agent_outbound``.  For each event it checks the
policy hook ``should_interject``.  If the policy returns ``True`` the
supervisor generates an *interjection* – a new :class:`AgentEvent` that
is pushed back into ``agent_inbox``.

This file intentionally avoids any extra helper functions so that the
process lifecycle is explicit and straightforward.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Callable

from .agent import AgentProcess, AgentEvent
from .policy import should_interject

log = logging.getLogger(__name__)


@dataclass
class SupervisorConfig:
    """Configuration for :class:`SupervisorProcess`.

    Parameters
    ----------
    agent_name:
        Identifier passed to the underlying :class:`AgentProcess`.
    policy_func:
        Callable that decides whether an interjection should be sent.
    """

    agent_name: str = "Agent-1"
    policy_func: Callable[[AgentEvent], bool] = should_interject


class SupervisorProcess(mp.Process):
    """Supervisor that forwards events between an agent and a policy."""

    def __init__(self, config: SupervisorConfig | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SupervisorConfig()
        # Queues for communication with the agent
        self.agent_inbox: mp.Queue[AgentEvent] = mp.Queue()
        self.agent_outbound: mp.Queue[AgentEvent] = mp.Queue()
        # External consumers queue
        self.supervisor_outbound: mp.Queue[AgentEvent] = mp.Queue()
        self._terminate_flag = mp.Event()
        self.agent_process: AgentProcess | None = None

    def run(self) -> None:  # pragma: no cover
        log.info("Supervisor starting")
        # Start the underlying agent process
        self.agent_process = AgentProcess(
            self.config.agent_name,
            self.agent_inbox,
            self.agent_outbound,
        )
        self.agent_process.start()
        # Main event loop
        while not self._terminate_flag.is_set():
            try:
                event: AgentEvent = self.agent_outbound.get(timeout=0.1)
            except queue.Empty:
                continue
            log.debug("Supervisor received event: %s", event)
            # Forward event to external consumers
            self.supervisor_outbound.put(event)
            if self.config.policy_func(event):
                # Preserve original error context
                original_msg = getattr(event, "content", "") or getattr(event, "prompt", "")
                interjection_content = (
                    f"Apology: I made a mistake. Let's correct it. Original issue: {original_msg}"
                )
                interjection_event = AgentEvent(
                    role="assistant",
                    content="",
                    session_id=getattr(event, "session_id", "supervisor"),
                    prompt=interjection_content,
                    type="interjection",
                )
                log.info("Supervisor interjecting: %s", interjection_event)
                # Send interjection to agent and external consumers
                self.agent_inbox.put(interjection_event)
                self.supervisor_outbound.put(interjection_event)
        log.info("Supervisor terminating")

    def terminate(self) -> None:  # pragma: no cover
        self._terminate_flag.set()
        if self.agent_process and self.agent_process.is_alive():
            self.agent_process.terminate()
            self.agent_process.join()
        super().terminate()


__all__ = ["SupervisorProcess", "SupervisorConfig"]