"""Utilities that handle the chat logic.

The original implementation of the chat handling lived directly in
``app.py``.  Extracting the functions into this dedicated module keeps
the UI entry point small and makes the chat logic easier to unit‑test.

Functions
---------
* :func:`build_messages` — convert a conversation history into the
  list of messages expected by the OpenAI chat completion endpoint.
* :func:`stream_and_collect` — stream the assistant response while
  capturing any tool calls.
* :func:`process_tool_calls` — invoke the tools requested by the model
  and generate subsequent assistant turns.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st

from .config import MODEL_NAME
from .tools import TOOLS
import time
import concurrent.futures
import logging
from .db import log_tool_msg, log_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # handlers=[logging.StreamHandler()],  # writes to stdout (the Streamlit console)
    filename='chat.log',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Public helper functions
# ---------------------------------------------------------------------------

def build_messages(
    history: Any,
    system_prompt: str,
    user_input: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the list of messages to send to the chat model.

    Parameters
    ----------
    history
        List of ``(user, assistant)`` pairs that have already happened.
    system_prompt
        The system message that sets the model behaviour.
    user_input
        The new user message that will trigger the assistant reply.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": str(system_prompt)}]

    for role, content, tool_id, tool_name, tool_args in history:
        if tool_name:
            msgs.append({"role": role, "tool_id": tool_id, "tool_args": tool_args})
        if role == 'tool':
            msgs.append({"role": role, "content": content, "tool_id": tool_id})
        else:
            msgs.append({"role": role, "content": content})

    if user_input is not None:
        msgs.append({"role": "user", "content": str(user_input)})

    return msgs


def stream_and_collect(
    client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Tuple[str, Optional[List[Dict[str, Any]]], bool, str]:
    """Stream the assistant response while capturing tool calls.

    The function writes the incremental assistant content to a placeholder
     and returns a tuple of the complete
    assistant text and a list of tool calls (or ``None`` if no tool call
    was emitted).
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        tools=tools,
    )

    assistant_text = ""
    reasoning_text = ""
    tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
    finished = False
    reasoning_placeholder = st.empty()
    placeholder = st.empty()

    for chunk in stream:
        choice = chunk.choices[0]
        if choice.finish_reason == "stop":
            finished = True
            break
        delta = choice.delta

        reasoning_part = delta.reasoning_content if hasattr(delta, 'reasoning_content') else None
        if reasoning_part:
            reasoning_text += reasoning_part
            md = f"<details open><summary>Reasoning</summary>\n{reasoning_text}\n</details>"
            reasoning_placeholder.markdown(md, unsafe_allow_html=True)

        # Regular text
        if delta.content:
            assistant_text += delta.content
            placeholder.markdown(assistant_text, unsafe_allow_html=True)

        # Tool calls — accumulate arguments per call id.
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
    return assistant_text, final_tool_calls, finished, reasoning_text


def process_tool_calls(
    client: Any,
    messages: List[Dict[str, Any]],
    session_id: str,
    history: List[Tuple[str, str]],
    tools: List[Dict[str, Any]],
    tool_calls: Optional[List[Dict[str, Any]]],
    finished: bool,
    assistant_text: str = "",
    reasoning_text: str = "",
) -> str:
    """
    Execute each tool that the model requested and keep asking the model
    for further replies until it stops calling tools.

    Parameters
    ----------
    client
        The OpenAI client used to stream assistant replies.
    messages
        The conversation history that will be extended with the tool‑call
        messages and the tool replies.
    tools
        The list of OpenAI‑compatible tool definitions that will be passed
        to the ``chat.completions.create`` call.
    tool_calls
        The list of tool‑call objects produced by
        :func:`stream_and_collect`.  The function may return a new
        list of calls that the model wants to make after the tool
        result is sent back.
    finished
        Boolean indicating whether the assistant already finished a turn.

    Returns
    -------
    str
        The cumulative assistant reply **including** the text produced
        by the tool calls.
    """
    if not tool_calls:
        return ""

    # Accumulate all text that the assistant will eventually produce
    full_text = assistant_text + reasoning_text

    # We keep looping until the model stops asking for tools
    while tool_calls and not finished:
        
        for tc in tool_calls:
            
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except Exception as exc:
                args = {}
                result = f"\u274c  JSON error: {exc}"
                logger.exception("Failed to parse tool arguments", exc_info=True)
            else:

                logger.info(
                    "Calling tool %s with arguments %s",
                    tc.get("name"),
                    json.dumps(args),
                )

                start = time.time()

                func = next(
                    (t.func for t in TOOLS if t.name == tc.get("name")), None
                )

                if func:
                    try:
                        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        try:
                            future = executor.submit(func, **args)
                            result = future.result(timeout=10)  # 10‑second timeout
                        except concurrent.futures.TimeoutError:  # pragma: no cover
                            result = (
                                "\u26d4  Tool call timed out after 10 seconds. "
                                "Try a shorter or more specific request."
                            )
                        except Exception as exc:  # pragma: no cover
                            result = f"\u274c  Tool error: {exc}"
                        finally:
                            # Avoid blocking on thread shutdown.
                            executor.shutdown(wait=False)
                    except Exception as exc:  # pragma: no cover
                        result = f"\u274c  Tool error: {exc}"
                        logger.exception("Tool raised an exception", exc_info=True)
                else:
                    result = f"\u26a0\ufe0f  Unknown tool '{tc.get('name')}'"
                    
            preview = result[:10] + ("\u2026" if len(result) > 10 else "")
            tool_block = (
                f"<details>"
                f"<summary>{tc.get('name')}|`{json.dumps(args)}`|{preview}</summary>"
                f"\n\n`{result}`\n\n"
                f"</details>"
            )
            tool_placeholder = st.empty()
            tool_placeholder.markdown(tool_block, unsafe_allow_html=True)
            
            tool_id = tc.get("id")
            tool_name = tc.get("name")
            tool_args = tc.get("arguments")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args or "{}",
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_id or ""),
                    "content": result,
                }
            )

            # appending tool call to history
            history.append(("assistant", ...))
            history.append(("analysis", ...))

            log_tool_msg(session_id, tool_id, tool_name, tool_args, result)
            
            
        new_assistant_resp, new_tool_calls, finished, reasoning_text = stream_and_collect(
            client, messages, tools
        )

        # appending reasoning to history
        history.append(("analysis", reasoning_text))
        history.append(("assistant", new_assistant_resp))

        log_message(session_id, "analysis", reasoning_text)
        log_message(session_id, "assistant", new_assistant_resp)
        
        tool_calls = new_tool_calls or None
            
    return history