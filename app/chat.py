# app/chat.py
"""Utilities that handle the chat logic.

The original implementation of the chat handling lived directly in
``app.py``.  Extracting the functions into this dedicated module keeps
the UI entry point small and makes the chat logic easier to unit‑test.

Functions
---------
* :func:`build_messages` – convert a conversation history into the
  list of messages expected by the OpenAI chat completion endpoint.
* :func:`stream_and_collect` – stream the assistant response while
  capturing any tool calls.
* :func:`process_tool_calls` – invoke the tools requested by the model
  and generate subsequent assistant turns.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

from .config import MODEL_NAME
from .tools import TOOLS

# ---------------------------------------------------------------------------
#  Public helper functions
# ---------------------------------------------------------------------------

def build_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str],
    user_input: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the list of messages to send to the chat model.

    Parameters
    ----------
    history
        List of ``(user, assistant)`` pairs that have already happened.
    system_prompt
        The system message that sets the model behaviour.
    repo_docs
        Optional Markdown string that contains the extracted repo source.
    user_input
        The new user message that will trigger the assistant reply.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": str(system_prompt)}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": str(repo_docs)})

    for u, a in history:
        msgs.append({"role": "user", "content": str(u)})
        msgs.append({"role": "assistant", "content": str(a)})

    if user_input is not None:
        msgs.append({"role": "user", "content": str(user_input)})

    return msgs


def stream_and_collect(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    placeholder: st.delta_generator.delta_generator,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Stream the assistant response while capturing tool calls.

    The function writes the incremental assistant content to the supplied
    Streamlit ``placeholder`` and returns a tuple of the complete
    assistant text and a list of tool calls (or ``None`` if no tool call
    was emitted).
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls_buffer: Dict[int, Dict[str, Any]] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Regular text
        if delta.content:
            full_resp += delta.content
            placeholder.markdown(full_resp, unsafe_allow_html=True)

        # Tool calls – accumulate arguments per call id.
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tc_delta.id,
                        "name": tc_delta.function.name,
                        "arguments": "",
                    }
                if tc_delta.function.arguments:
                    tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

    final_tool_calls = list(tool_calls_buffer.values()) if tool_calls_buffer else None
    return full_resp, final_tool_calls


def process_tool_calls(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    placeholder: st.delta_generator.delta_generator,
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Invoke the tools requested by the model.

    Parameters
    ----------
    client
        The OpenAI client used for subsequent assistant replies.
    messages
        The conversation history that will be extended with the tool call
        messages.
    tools
        List of OpenAI‑compatible tool definitions.
    placeholder
        Streamlit placeholder for the assistant output.
    tool_calls
        List of tool call objects returned by :func:`stream_and_collect`.

    Returns
    -------
    tuple
        ``(final_text, remaining_tool_calls)`` where *remaining_tool_calls*
        is ``None`` if the model finished calling tools.
    """
    if not tool_calls:
        return "", None

    full_text = ""
    for tc in tool_calls:
        args = json.loads(tc.get("arguments") or "{}")
        func = next((t.func for t in TOOLS if t.name == tc.get("name")), None)

        if func:
            try:
                result = func(**args)
            except Exception as exc:
                result = f"❌  Tool error: {exc}"
        else:
            result = f"⚠️  Unknown tool '{tc.get('name')}'"

        # Render the tool‑call result before the next assistant turn.
        tool_output_str = (
            f"**Tool call**: `{tc.get('name')}`"
            f"({', '.join(f'{k}={v}' for k, v in args.items())}) → `{result}`"
        )
        placeholder.markdown(tool_output_str, unsafe_allow_html=True)

        # Build messages to send back to the model
        assistant_tool_call_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments") or "{}",
                    },
                }
            ],
        }

        tool_msg = {
            "role": "tool",
            "tool_call_id": str(tc.get("id") or ""),
            "content": str(result or ""),
        }

        messages.append(assistant_tool_call_msg)
        messages.append(tool_msg)

        # Stream the next assistant reply – each iteration gets a fresh placeholder
        placeholder2 = st.empty()
        new_text, new_tool_calls = stream_and_collect(
            client, messages, tools, placeholder2
        )
        full_text += new_text
        tool_calls = new_tool_calls or []

        if not tool_calls:  # no more calls – break early
            break

    return full_text, tool_calls
