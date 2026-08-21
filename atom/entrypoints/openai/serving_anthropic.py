# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Anthropic Messages API adapter for ATOM.

Translates Anthropic /v1/messages requests to ATOM's internal format and
converts responses back to Anthropic format. Enables Claude Code and other
Anthropic-compatible tools to use ATOM as a backend.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel

from .sse import event_frame

logger = logging.getLogger("atom")


# ── Anthropic Request Schema ───────────────────────────────────────────


class AnthropicContentBlock(BaseModel):
    type: str
    text: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: Any | None = None
    # tool_result fields
    tool_use_id: str | None = None
    content: Any | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: Any  # str or list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: Any | None = None  # str or list
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: Any | None = None
    metadata: dict | None = None
    thinking: dict | None = None  # {"type":"enabled","budget_tokens":N}


# ── Format Conversion ──────────────────────────────────────────────────


def anthropic_to_openai_messages(
    messages: list[AnthropicMessage],
    system: Any | None = None,
) -> list[dict]:
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


def anthropic_to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
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


# ── Response Construction ──────────────────────────────────────────────


def build_anthropic_response(
    request_id: str,
    model: str,
    content_text: str,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
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
    return event_frame(event, data)


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


class AnthropicBlocks:
    """One open content block at a time, closed before the next one opens.

    Anthropic frames a response as indexed blocks of a kind -- text, thinking,
    tool_use -- and a change of kind is a close and an open. Those transitions
    used to be written out at each of the four places a segment could arrive,
    each covering the subset its author needed. The one nobody needed,
    text -> thinking, was missing: a reasoning segment arriving after content
    had started matched no branch and was dropped, with no error and no log.
    Measured on a model that answers, opens a `<think>` block and answers
    again, 29 characters of reasoning went nowhere.

    So the transition is asked for rather than written out: `delta` says which
    kind this text belongs to and the switching is this class's problem. It
    cannot silently do nothing, because there is no branch left to fall off.
    """

    def __init__(self) -> None:
        self.index = 0
        self.kind: str | None = None

    def close(self):
        """End the open block, if any. A thinking block signs off first."""
        if self.kind is None:
            return
        if self.kind == "thinking":
            yield stream_signature_delta(self.index)
        yield stream_content_block_stop(self.index)
        self.index += 1
        self.kind = None

    def open(self, kind: str, **start_kwargs):
        """Start a block of `kind`, closing whatever was open."""
        yield from self.close()
        yield stream_content_block_start(self.index, kind, **start_kwargs)
        self.kind = kind

    def delta(self, kind: str, text: str, **start_kwargs):
        """Emit `text` as `kind`, switching blocks if that is not the open one."""
        if self.kind != kind:
            yield from self.open(kind, **start_kwargs)
        yield stream_content_block_delta(self.index, text, kind)


def tool_event_frames(events, blocks: AnthropicBlocks):
    """One batch of tool-parser events as Anthropic frames.

    Written out twice in the streaming endpoint, once for `process` and once
    for `flush`, twenty-two lines each. That is the same hazard
    :class:`AnthropicBlocks` exists to remove one level up -- two copies of a
    dispatch means a fix that lands in one of them, and nothing says so.

    A plain generator and not `yield from` at the call site, because the
    endpoint is an *async* generator and `yield from` is a syntax error inside
    one. Whether a call started is left to the caller to read off `events`;
    returning it from a generator would need the `yield from` that cannot be
    written there.
    """
    for etype, edata in events:
        if etype == "content":
            yield from blocks.delta("text", edata)
        elif etype == "tool_call_start":
            fn = edata.get("function", {})
            yield from blocks.open(
                "tool_use",
                tool_use_id=edata.get("id", ""),
                tool_name=fn.get("name", ""),
            )
        elif etype == "tool_call_args":
            fn = edata.get("function", {})
            yield from blocks.delta("tool_use", fn.get("arguments", ""))
        elif etype == "tool_call_end":
            yield from blocks.close()


def completes_a_tool_call(events) -> bool:
    """Whether this batch produced a *usable* tool call.

    Keyed on the arguments and not the name, which is what makes announcing a
    name early safe. A name can be sent before the call is known to close --
    the point of announcing it -- so a response truncated at `max_tokens`
    mid-call has sent a name and nothing else. Reporting `tool_use` there
    would tell the client to run a tool whose arguments never arrived.

    Every parser emits name and arguments together unless it announced early,
    so this reads the same as the name for every format that does not.
    """
    return any(etype == "tool_call_args" for etype, _ in events)


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
