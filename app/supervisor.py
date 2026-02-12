"""Supervisor process for the multi‑agent system.

The supervisor owns the *policy* that decides whether the agent should
interject in an ongoing conversation.  It communicates with the
``AgentProcess`` via two :class:`multiprocessing.Queue` objects:

* ``agent_inbox`` – the queue the supervisor uses to send messages to the
  agent.
* ``agent_outbound`` – the queue the supervisor listens to for events
  emitted by the agent.

The supervisor runs as a :class:`multiprocessing.Process` that continually
reads from ``agent_outbound``.  For each event it checks the policy hook
``should_interject``.  If the policy returns ``True`` the supervisor
generates an *interjection* – a new :class:`AgentEvent` that is pushed
back into ``agent_inbox``.

The design keeps the logic simple and deterministic so that unit tests can
exercise the interaction flow without involving I/O or network calls.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .agent import AgentProcess, AgentEvent
from .policy import should_interject

log = logging.getLogger(__name__)


@dataclass
class SupervisorConfig:
    """Configuration for the :class:`SupervisorProcess`.

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
    """Run a supervisor that forwards events between an agent and a policy.

    The supervisor starts an :class:`AgentProcess` in a child process and
    then enters a loop that reads from the agent's outbound queue.  The
    policy function is called on each event and may trigger an
    interjection.
    """

    def __init__(self, config: SupervisorConfig | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or SupervisorConfig()
        # Queues for communication with the agent
        self.agent_inbox: mp.Queue[AgentEvent] = mp.Queue()
        self.agent_outbound: mp.Queue[AgentEvent] = mp.Queue()
        # Supervisor's own outbound queue for external consumers
        self.supervisor_outbound: mp.Queue[AgentEvent] = mp.Queue()
        self._terminate_flag = mp.Event()
        self.agent_process: AgentProcess | None = None

    def _agent_target(self, inbox: mp.Queue[AgentEvent], outbound: mp.Queue[AgentEvent]):
        """Target function for the agent process.

        This helper simply creates an :class:`AgentProcess` that consumes
        ``inbox`` and produces events on ``outbound``.
        """

        agent = AgentProcess(
            self.config.agent_name,
            inbox,
            outbound,
            *self._agent_kwargs,
        )
        agent.run()

    def run(self) -> None:  # pragma: no cover – exercised via integration test
        log.info("Supervisor starting")
        # Start the underlying agent process
        self.agent_process = AgentProcess(
            self.config.agent_name,
            self.agent_inbox,
            self.agent_outbound,
        )
        self.agent_process.start()

        # Main loop – process events from the agent
        while not self._terminate_flag.is_set():
            try:
                event: AgentEvent = self.agent_outbound.get(timeout=0.1)
            except queue.Empty:
                continue
            log.debug("Supervisor received event: %s", event)
            # Pass through the event to external consumers
            self.supervisor_outbound.put(event)
            policy_decision = self.config.policy_func(event)
            log.debug("Policy decision for event %s: %s", event, policy_decision)
            if policy_decision:
                # Create an interjection – a simple apology message
                interjection = AgentEvent(
                    role="assistant",
                    content="Apology: I made a mistake. Let's correct it.",
                )
                log.info("Supervisor interjecting: %s", interjection)
                self.agent_inbox.put(interjection)
            # Continue loop
        log.info("Supervisor terminating")

    def terminate(self) -> None:  # pragma: no cover – exercised via integration test
        self._terminate_flag.set()
        if self.agent_process and self.agent_process.is_alive():
            self.agent_process.terminate()
            self.agent_process.join()
        super().terminate()

    # Allow passing arbitrary kwargs to AgentProcess if needed
    def _agent_kwargs(self):  # pragma: no cover
        return {}


__all__ = ["SupervisorProcess", "SupervisorConfig"]
