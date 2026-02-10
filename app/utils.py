# app/utils.py  (only the added/modified parts)
from typing import List, Tuple, Dict, Optional, Any
from .config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME
from .client import get_client
from openai import OpenAI
from .tools import get_tools

def build_api_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict]:
    """
    Convert local chat history into the format expected by the OpenAI API,
    optionally adding a tool list.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if repo_docs:
        msgs.append({"role": "assistant", "content": repo_docs})
    # Enforce a maximum context length to avoid exceeding the model's limit.
    # The context length is estimated by word count; if it exceeds
    # ``max_context_tokens`` we trim the oldest messages.
    max_context_tokens = 4000  # default value, can be overridden if needed
    # Build a preliminary list to compute token count.
    preliminary_msgs = ["system", "repo_docs"]
    for user_msg, bot_msg in history:
        preliminary_msgs.append("user")
        preliminary_msgs.append("assistant")
    # Rough token estimate: 1 token ≈ 4 characters.  Here we use word count.
    token_estimate = sum(len(msg.split()) for msg in preliminary_msgs if isinstance(msg, str))
    # Trim from the start until within limit.
    trimmed_history = history[:]
    while token_estimate > max_context_tokens and trimmed_history:
        # Remove the oldest pair
        token_estimate -= len(trimmed_history[0][0].split()) + len(trimmed_history[0][1].split())
        trimmed_history.pop(0)
    for user_msg, bot_msg in trimmed_history:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    # The client will pass `tools=tools` when calling chat.completions.create
    return msgs

def stream_response(
    history: List[Tuple[str, str]],
    user_msg: str,
    client: OpenAI,
    system_prompt: str,
    repo_docs: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
):
    """
    Yield the cumulative assistant reply while streaming.
    Also returns any tool call(s) that the model requested.
    """
    new_hist = history + [(user_msg, "")]
    api_msgs = build_api_messages(new_hist, system_prompt, repo_docs, tools)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_msgs,
        stream=True,
        tools=tools,
    )

    full_resp = ""
    tool_calls = None
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_resp += token
        yield full_resp

        # Capture tool calls once the model finishes sending them
        if chunk.choices[0].delta.tool_calls:
            tool_calls = chunk.choices[0].delta.tool_calls

    return full_resp, tool_calls