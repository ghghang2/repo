"""Utility for converting chat history into OpenAI API message format."""
from __future__ import annotations

import json
from typing import List, Dict, Tuple


def build_messages(
    history: List[Tuple[str, str, str, str, str]],
    system_prompt: str,
    context_summary: str = "",
) -> List[Dict[str, str]]:
    """Build OpenAI messages from internal chat history.

    Parameters
    ----------
    history:
        List of tuples ``(role, content, tool_id, tool_name, tool_args)``.
    system_prompt:
        The system message to prepend.
    context_summary:
        Rolling summary produced by CompactionEngine.  When non-empty it is
        merged into the single system message so llama.cpp chat templates
        that only honour one system block still receive the summary.
    """
    # Merge summary into the system prompt rather than adding a second system
    # message — most llama.cpp chat templates silently drop additional system
    # blocks or only render the first one.
    if context_summary:
        system_content = (
            system_prompt
            + "\n\n--- SUMMARY OF EARLIER CONVERSATION ---\n"
            + context_summary
            + "\n--- END SUMMARY ---"
        )
    else:
        system_content = system_prompt

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

    for role, content, tool_id, tool_name, tool_args in history:
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if tool_id:
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args},
                    }],
                })
            else:
                messages.append({"role": "assistant", "content": content})
        elif role == "assistant_full":
            try:
                full_msg = json.loads(tool_args)
                messages.append(full_msg)
            except Exception:
                messages.append({"role": "assistant", "content": content})
        elif role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content,
            })
        elif role == "compacted":
            # Legacy rows from old sessions — merge into system content
            # rather than adding another system block.
            messages[0]["content"] += "\n\n" + content

    return messages


__all__ = ["build_messages"]