# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Anthropic Messages API adapter for ATOM.

Translates Anthropic /v1/messages requests to ATOM's internal format and
converts responses back to Anthropic format. Enables Claude Code and other
Anthropic-compatible tools to use ATOM as a backend.
"""

import json
import logging
from typing import Any, List, Optional

from pydantic import BaseModel

logger = logging.getLogger("atom")


# ── Anthropic Request Schema ───────────────────────────────────────────


class AnthropicContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    # tool_use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None
    # tool_result fields
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Any  # str or list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 4096
    system: Optional[Any] = None  # str or list
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[dict] = None
    thinking: Optional[dict] = None  # {"type":"enabled","budget_tokens":N}


# ── Format Conversion ──────────────────────────────────────────────────


def anthropic_to_openai_messages(
    messages: List[AnthropicMessage],
    system: Optional[Any] = None,
) -> List[dict]:
    """Convert Anthropic messages to OpenAI format."""
    result = []

    # System message
    if system:
        if isinstance(system, str):
            result.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for b in system:
                if b.get("type") == "text":
                    text = b["text"]
                    if text.startswith("x-anthropic-billing-header"):
                        continue
                    text_parts.append(text)
            if text_parts:
                result.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )
                entry = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)

        elif role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_results = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    b.get("text", "")
                                    for b in tool_content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": str(tool_content),
                                }
                            )
                if text_parts:
                    result.append({"role": "user", "content": "\n".join(text_parts)})
                result.extend(tool_results)
        else:
            result.append({"role": role, "content": str(content) if content else ""})

    return result


def anthropic_to_openai_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    """Convert Anthropic tool definitions to OpenAI format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return result


def _text_of(block: Any) -> str:
    """Flatten an Anthropic content block (str or list of {text} blocks) to text."""
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        return "\n".join(
            _text_of(b.get("text")) if isinstance(b, dict) else _text_of(b)
            for b in block
            if (b.get("text") if isinstance(b, dict) else b)
        )
    return ""


def extract_cwd_from_system(system: Any) -> Optional[str]:
    """Pull the working directory from Claude Code's system prompt.

    Claude Code embeds the working directory as a `Working directory: <path>`
    line in the system message it sends (it does not use Codex's
    `<cwd>...</cwd>` environment_context, which is Anthropic-incompatible).
    DSV4 (sparse indexer, index_topk=1024) cannot reliably verbatim-recall a
    path buried in a ~20-40k-token system prompt, so when the Bash tool
    description omits the cwd the model blind-`cd`s to whatever path the user
    named. Returning the cwd here lets the caller inject it into the Bash tool
    description so the model uses the absolute path or `pwd`s instead.
    """
    import re

    blob = _text_of(system)
    if not blob:
        return None
    m = re.search(r"(?:^|\n)\s*Working directory:\s*([^\s<\n]+)", blob)
    return m.group(1) if m else None


def inject_cwd_into_bash_tool(
    tools: Optional[List[dict]], cwd: Optional[str]
) -> Optional[List[dict]]:
    """Append the working directory to the Bash tool's description.

    Empirically (A/B on the same DSV4 server, temp 0, 2 runs each): with the
    cwd in the Bash tool description, DSV4 stops blind-`cd`-ing to whatever
    path the user named and instead `ls`'s the absolute path or `pwd`s first.
    Without it, DSV4 faithfully `cd <user-named-path>` even when that path
    does not exist. This is a prompt-content mitigation for a model-level
    recall/cliff limitation (the ATOM /v1/messages prompt is otherwise
    correctly constructed — cwd is in the system message), not a serving bug.
    """
    if not tools or not cwd:
        return tools
    for tool in tools:
        if tool.get("name") == "Bash":
            base = tool.get("description", "") or ""
            hint = (
                f" The current working directory is {cwd}."
                if "current working directory" not in base.lower()
                else ""
            )
            if hint and cwd not in base:
                tool["description"] = base + hint
    return tools


# ── Response Construction ──────────────────────────────────────────────


def build_anthropic_response(
    request_id: str,
    model: str,
    content_text: str,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[list] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    stop_reason: str = "end_turn",
) -> dict:
    """Build Anthropic Messages API response.

    Args:
        tool_calls: List of ToolCall objects (from tool_parser.parse_tool_calls).
            Each has .name, .arguments (dict), .call_id.
    """
    content = []

    if reasoning_content:
        import base64
        import hashlib
        import os

        sig = base64.b64encode(hashlib.sha256(os.urandom(32)).digest()).decode()
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning_content,
                "signature": sig,
            }
        )

    if content_text:
        content.append(
            {
                "type": "text",
                "text": content_text,
            }
        )

    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            # ToolCall has .id, .function["name"], .function["arguments"]
            func = tc.function if isinstance(tc.function, dict) else {}
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": func.get("name", ""),
                    "input": args,
                }
            )

    # Ensure at least one content block
    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            # Anthropic convention: input_tokens counts only the
            # non-cached (freshly processed) prompt tokens; cached tokens
            # are reported separately in cache_read_input_tokens.
            "input_tokens": max(input_tokens - cache_read_input_tokens, 0),
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }


# ── Streaming ──────────────────────────────────────────────────────────


def format_sse(event: str, data: Any) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_message_start(
    request_id: str,
    model: str,
    input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> str:
    return format_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{request_id}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": max(input_tokens - cache_read_input_tokens, 0),
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cache_read_input_tokens,
                },
            },
        },
    )


def stream_content_block_start(
    index: int,
    block_type: str = "text",
    tool_use_id: str = "",
    tool_name: str = "",
) -> str:
    if block_type == "thinking":
        block = {"type": "thinking", "thinking": "", "signature": ""}
    elif block_type == "tool_use":
        block = {
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": {},
        }
    else:
        block = {"type": "text", "text": ""}
    return format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        },
    )


def stream_content_block_delta(index: int, text: str, block_type: str = "text") -> str:
    if block_type == "thinking":
        delta = {"type": "thinking_delta", "thinking": text}
    elif block_type == "tool_use":
        delta = {"type": "input_json_delta", "partial_json": text}
    else:
        delta = {"type": "text_delta", "text": text}
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )


def stream_signature_delta(index: int) -> str:
    """Emit a signature_delta for thinking blocks (required by Claude Code)."""
    import base64
    import hashlib
    import os

    dummy_sig = base64.b64encode(hashlib.sha256(os.urandom(32)).digest()).decode()
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": dummy_sig},
        },
    )


def stream_content_block_stop(index: int) -> str:
    return format_sse(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def stream_message_delta(stop_reason: str = "end_turn", output_tokens: int = 0) -> str:
    return format_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


def stream_message_stop() -> str:
    return format_sse("message_stop", {"type": "message_stop"})
