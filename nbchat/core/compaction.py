"""Compaction Engine
===================

Keeps the token count sent to the model under a configured threshold by
summarising the oldest portion of the conversation.

Key design decisions
--------------------
* History is never mutated with a ``compacted`` row.  The summary lives in
  ``context_summary`` on the engine and is injected into the system prompt by
  ``chat_builder.build_messages``.

* Splitting is turn-aware.  History is grouped into logical *turns* (each
  starting with a ``user`` row).  Compaction drops whole turns from the front
  so we never accidentally split an ``assistant_full`` / ``tool`` / ``analysis``
  triplet.

* Oversized tool results are truncated in-place rather than being silently
  dropped.  A single large tool output (e.g. a full file read) used to cause
  total amnesia because _safe_tail would walk off the end of the list.

* _safe_tail never returns an empty list.  It falls back progressively:
  forward-nudge -> nearest user row -> full history.  Losing some token budget
  is always better than total forgetfulness.

* Successive compactions are cumulative: the previous context_summary is
  fed to the summariser so the new summary folds it in without an extra API
  call.
"""
from __future__ import annotations

import sys
import threading
from typing import List, Tuple, Optional

from nbchat.ui.chat_builder import build_messages
from .client import get_client

_Row = Tuple[str, str, str, str, str]
_DEPENDENT_ROLES = {"tool", "analysis", "assistant_full"}


class CompactionEngine:

    def __init__(
        self,
        threshold: int,
        tail_messages: int = 5,
        summary_prompt: str = "",
        summary_model: str = "",
        system_prompt: str = "",
    ) -> None:
        self.threshold = threshold
        self.tail_messages = tail_messages
        self.summary_prompt = summary_prompt
        self.summary_model = summary_model
        self.system_prompt = system_prompt
        self.context_summary: str = ""
        self._cache: dict = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def total_tokens(self, history: List[_Row]) -> int:
        total = 0
        for role, content, tool_id, tool_name, tool_args in history:
            key = hash((content, tool_args))
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached is not None:
                total += cached
                continue
            tokens = self._estimate_tokens(content) + (
                self._estimate_tokens(tool_args) if tool_args else 0
            )
            with self._cache_lock:
                self._cache[key] = tokens
            total += tokens
        return total

    def should_compact(self, history: List[_Row]) -> bool:
        tokens = self.total_tokens(history)
        trigger = int(self.threshold * 0.75)
        print(
            f"[compaction] token estimate: {tokens} / {self.threshold}"
            f" (trigger={trigger})",
            file=sys.stderr,
        )
        return tokens >= trigger

    # ------------------------------------------------------------------
    # Turn grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_into_turns(history: List[_Row]) -> List[List[_Row]]:
        turns: List[List[_Row]] = []
        current: List[_Row] = []
        for row in history:
            if row[0] == "user" and current:
                turns.append(current)
                current = []
            current.append(row)
        if current:
            turns.append(current)
        return turns

    # ------------------------------------------------------------------
    # Safe tail — never returns empty
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_tail(history: List[_Row], n: int) -> List[_Row]:
        """Return the last n rows starting at a safe boundary.

        Priority:
        1. Walk forward from tail_start past dependent roles.
        2. If that exhausts the list, walk backward to nearest user row.
        3. If no user row found, return full history.

        Never returns an empty list — losing token budget beats total amnesia.
        """
        if not history or n <= 0:
            return history

        tail_start = max(0, len(history) - n)

        # Walk forward past dependent roles.
        probe = tail_start
        while probe < len(history) and history[probe][0] in _DEPENDENT_ROLES:
            probe += 1

        if probe < len(history):
            return history[probe:]

        # Forward walk exhausted — fall back to nearest user boundary.
        probe = len(history) - 1
        while probe > 0 and history[probe][0] != "user":
            probe -= 1

        if history[probe][0] == "user":
            print(
                f"[compaction] _safe_tail: fell back to user boundary at index {probe}",
                file=sys.stderr,
            )
            return history[probe:]

        print(
            "[compaction] _safe_tail: no user boundary found, returning full history",
            file=sys.stderr,
        )
        return history

    # ------------------------------------------------------------------
    # Tool result truncation
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_tool_results(rows: List[_Row], budget: int) -> List[_Row]:
        """Truncate oversized tool result content so rows fit within budget.

        Largest tool results are trimmed first.  A truncation notice is
        appended so the model knows output was cut.  Non-tool rows are never
        modified.
        """
        def est(text: str) -> int:
            return max(1, len(text) // 3)

        result = list(rows)
        total = sum(est(r[1]) + (est(r[4]) if r[4] else 0) for r in result)

        if total <= budget:
            return result

        tool_indices = sorted(
            [i for i, r in enumerate(result) if r[0] == "tool"],
            key=lambda i: len(result[i][1]),
            reverse=True,
        )

        for idx in tool_indices:
            if total <= budget:
                break
            role, content, tid, tname, targs = result[idx]
            excess_chars = (total - budget) * 3
            keep = max(200, len(content) - excess_chars)
            notice = (
                f"\n[...output truncated from {len(content)} to {keep} chars"
                f" to fit context window...]"
            )
            new_content = content[:keep] + notice
            saved = est(content) - est(new_content)
            result[idx] = (role, new_content, tid, tname, targs)
            total -= saved
            print(
                f"[compaction] truncated tool result '{tname}'"
                f" {len(content)} -> {len(new_content)} chars",
                file=sys.stderr,
            )

        return result

    # ------------------------------------------------------------------
    # Intra-turn safe split
    # ------------------------------------------------------------------

    @staticmethod
    def _find_safe_split(group: List[_Row]) -> Optional[int]:
        for i in range(1, len(group)):
            role = group[i][0]
            prev_role = group[i - 1][0]
            if role not in _DEPENDENT_ROLES and prev_role != "assistant_full":
                return i
        return None

    # ------------------------------------------------------------------
    # Core compaction
    # ------------------------------------------------------------------

    def compact_history(self, history: List[_Row]) -> List[_Row]:
        print(
            f"[compaction] compact_history called,"
            f" history len={len(history)},"
            f" tail_messages={self.tail_messages}",
            file=sys.stderr,
        )

        if len(history) <= self.tail_messages:
            print("[compaction] history too short to compact", file=sys.stderr)
            return history

        turns = self._group_into_turns(history)
        threshold_tokens = int(self.threshold * 0.75)

        to_summarise: List[_Row] = []
        remaining_turns: List[List[_Row]] = list(turns)

        while remaining_turns:
            remaining_flat = [row for t in remaining_turns for row in t]
            if self.total_tokens(remaining_flat) <= threshold_tokens:
                break

            candidate_turn = remaining_turns[0]
            after_drop = [row for t in remaining_turns[1:] for row in t]

            if not after_drop:
                # Last remaining turn.
                split_idx = self._find_safe_split(candidate_turn)
                if split_idx is not None:
                    to_summarise.extend(candidate_turn[:split_idx])
                    remaining_turns[0] = candidate_turn[split_idx:]
                    print(
                        f"[compaction] intra-turn split at index {split_idx}"
                        f" within last turn of {len(candidate_turn)} rows",
                        file=sys.stderr,
                    )
                else:
                    # No structural split possible — truncate tool results to
                    # make the turn fit, summarise what we have, return truncated turn.
                    print(
                        "[compaction] cannot split last remaining turn —"
                        " truncating oversized tool results in place",
                        file=sys.stderr,
                    )
                    truncated = self._truncate_tool_results(
                        candidate_turn, threshold_tokens
                    )
                    self.context_summary = self._call_summariser(
                        to_summarise if to_summarise else history
                    )
                    with self._cache_lock:
                        self._cache.clear()
                    return truncated
                break

            turn_tokens = self.total_tokens(candidate_turn)
            if turn_tokens >= threshold_tokens:
                split_idx = self._find_safe_split(candidate_turn)
                if split_idx is not None:
                    to_summarise.extend(candidate_turn[:split_idx])
                    remaining_turns[0] = candidate_turn[split_idx:]
                    print(
                        f"[compaction] oversized turn ({turn_tokens} tokens):"
                        f" intra-turn split at index {split_idx}",
                        file=sys.stderr,
                    )
                    continue
                # No safe split — truncate tool results so it fits.
                print(
                    f"[compaction] oversized turn with no safe split"
                    f" ({turn_tokens} tokens) — truncating tool results",
                    file=sys.stderr,
                )
                remaining_turns[0] = self._truncate_tool_results(
                    candidate_turn, threshold_tokens
                )
                to_summarise.extend(remaining_turns.pop(0))
                continue

            to_summarise.extend(remaining_turns.pop(0))

        if not to_summarise:
            print("[compaction] nothing to summarise", file=sys.stderr)
            return history

        remaining_history = [row for t in remaining_turns for row in t]

        # Guard: never return empty history.
        if not remaining_history:
            print(
                "[compaction] remaining_history empty after loop —"
                " falling back to safe tail",
                file=sys.stderr,
            )
            remaining_history = self._safe_tail(history, self.tail_messages)

        print(
            f"[compaction] summarising {len(to_summarise)} rows,"
            f" keeping {len(remaining_history)} rows",
            file=sys.stderr,
        )

        self.context_summary = self._call_summariser(to_summarise)

        with self._cache_lock:
            self._cache.clear()

        return remaining_history

    # ------------------------------------------------------------------
    # Summariser call
    # ------------------------------------------------------------------

    def _call_summariser(self, older: List[_Row]) -> str:
        # Truncate older before building messages so the summariser call
        # itself doesn't overflow the context window.
        older = self._truncate_tool_results(older, self.threshold * 2)

        messages = build_messages(older, self.system_prompt)

        if self.context_summary:
            messages.insert(1, {
                "role": "system",
                "content": (
                    "Previous conversation summary (incorporate this into your"
                    f" new summary):\n{self.context_summary}"
                ),
            })

        for msg in messages:
            msg.pop("reasoning_content", None)

        # Remove dangling tool_calls from the last assistant message.
        if messages and messages[-1].get("role") == "assistant":
            messages[-1].pop("tool_calls", None)
            if not messages[-1].get("content"):
                messages.pop()

        messages.append({"role": "user", "content": "we are running out of context window"})
        messages.append({"role": "assistant", "content": self.summary_prompt})

        print(
            f"[compaction] sending {len(messages)} messages to summariser",
            file=sys.stderr,
        )

        try:
            response = get_client().chat.completions.create(
                model=self.summary_model,
                messages=messages,
                max_tokens=4096,
            )
        except Exception as exc:
            raise RuntimeError(f"Summarisation failed: {exc}") from exc

        summary = response.choices[0].message.content
        print(
            f"[compaction] summary produced ({len(summary)} chars):"
            f" {summary[:120]}...",
            file=sys.stderr,
        )
        return summary


__all__ = ["CompactionEngine"]