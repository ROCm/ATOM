# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Anthropic Messages API adapter for ATOM.

Translates Anthropic /v1/messages requests to ATOM's internal format and
converts responses back to Anthropic format. Enables Claude Code and other
Anthropic-compatible tools to use ATOM as a backend.
"""

import base64
import hashlib
import json
import logging
import os
from typing import Any

from pydantic import BaseModel

from .reasoning import ReasoningChannel
from .sse import event_frame
from .tool_parser import ToolCallStreamParser

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
    min_tokens: int | None = 0
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


def stream_failure_frames(exc, blocks, output_tokens: int, *, opening: str | None):
    """Everything a stream that raised still owes the client.

    ``opening`` is the `message_start` frame when the stream never sent one,
    else ``None``. A failure before the first chunk otherwise produced a
    stream whose first frame is `message_delta`, and the Anthropic SDK builds
    its accumulator from `message_start` and raises on anything before it --
    so the client saw an SDK error rather than the `error` frame this delivers.
    """
    if opening is not None:
        yield opening
    # Every block this stream opened has to be closed before the terminator.
    yield from blocks.close()
    yield stream_error(str(exc))
    yield stream_message_delta("end_turn", output_tokens)
    yield stream_message_stop()


def _sig() -> str:
    """The opaque signature Anthropic puts on a thinking block."""
    return base64.b64encode(hashlib.sha256(os.urandom(32)).digest()).decode()


def read_whole_blocks(
    channel: ReasoningChannel,
    parser_cls,
    text: str,
    tools: list | None = None,
    *,
    suppress_calls: bool = False,
) -> list:
    """One complete output as ordered events, reasoning included.

    The streaming branch's own loop over a single chunk: the reasoning filter
    first, the tool parser on the content segments it yields. `split()`
    returns `(reasoning, content)` and loses the interleaving at that line, so
    blocks rebuilt from the pair put a whole answer in one block ahead of a
    `thinking` that belonged in the middle of it -- two blocks where streaming
    sent three, for the same generation.

    Not Anthropic-specific except that Anthropic is the only endpoint whose
    wire format has an order to lose; the chat path's fields are flat.
    """
    reader = channel.stream()
    engine = ToolCallStreamParser(
        tools=tools, parser_cls=parser_cls, suppress_calls=suppress_calls
    )
    out: list = []
    for field, segment in reader.process(text) + reader.flush():
        if not segment:
            continue
        if field == "reasoning_content":
            out.append(("reasoning", segment))
        else:
            out.extend(engine.process(segment))
    out.extend(engine.flush())
    return out


def _blocks_in_order(events: list) -> list[dict]:
    """The engine's events as Anthropic content blocks, in arrival order.

    The streaming path already does this, through `AnthropicBlocks` and
    `tool_event_frames`. The non-streaming path was given `(content_text,
    tool_calls)` instead -- the same events with the order thrown away -- and
    rebuilt one text block ahead of every `tool_use`. So the same generation
    came back as `['text', 'tool_use', 'tool_use']` unstreamed and
    `['tool_use', 'text', 'tool_use']` streamed, and a client rendering blocks
    in order showed the sentence introducing the second call before the first.

    Consecutive content events are joined: they are one run of answer that the
    engine happened to emit in pieces, and Anthropic has no way to say "two
    adjacent text blocks" that means anything different.
    """
    blocks: list[dict] = []
    pending: dict | None = None
    for etype, data in events:
        if not data and etype in ("reasoning", "content"):
            continue
        if etype == "reasoning":
            blocks.append({"type": "thinking", "thinking": data, "signature": _sig()})
        elif etype == "content":
            if blocks and blocks[-1]["type"] == "text":
                blocks[-1]["text"] += data
            else:
                blocks.append({"type": "text", "text": data})
        elif etype == "tool_call_start":
            pending = {"id": data["id"], "name": data["function"]["name"]}
        elif etype == "tool_call_args" and pending is not None:
            raw = data["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # A call cut off mid-arguments. Logged rather than silent, as
                # SGLang does: the client sees a call with no arguments and
                # nothing else says why.
                logger.warning(
                    "tool call %s: arguments are not JSON, sending {}: %r",
                    pending["name"],
                    raw[:120],
                )
                args = {}
            # Whatever decoded, verbatim. Coercing a non-dict to `{}` dropped
            # a Kimi-K2 call's arguments on `stream=false` alone -- that
            # format passes the wire bytes through and the streaming path
            # forwards them for the SDK to accumulate.
            blocks.append(
                {
                    "type": "tool_use",
                    "id": pending["id"],
                    "name": pending["name"],
                    "input": args,
                }
            )
            pending = None
    return blocks


def build_anthropic_response(
    request_id: str,
    model: str,
    events: list,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    *,
    stop_reason: str,
) -> dict:
    """One response, built from the ordered events and nothing else.

    It also took `(content_text, reasoning_content, tool_calls)` and rebuilt
    the blocks from them, so this file held two tool_use builders and two
    orderings -- the two-readers shape the rest of this branch exists to
    delete. It cost what that always costs: reasoning was prepended ahead of
    everything, so a model that answers, thinks and answers again came back as
    `[thinking, text]` with the two answers glued together where streaming
    sent `[text, thinking, text]`.

    `stop_reason` has no default on purpose -- it used to be overwritten here
    whenever `tool_calls` was non-empty, discarding what
    `anthropic_stop_reason_with_calls` had already decided.
    """
    # A response is never empty: Anthropic has no representation for one, and
    # the streaming path forces a final text block for the same reason.
    content = _blocks_in_order(events) or [{"type": "text", "text": ""}]

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
        # Every call the parser has named, by the index it named it at, and
        # kept for the life of the response. A call spans two parser batches
        # -- the name when the region reveals it, the arguments when the
        # region closes -- and anything at all can arrive in between: a
        # reasoning segment lands as a `thinking` block, which closes whatever
        # was open. Held on the open block instead, the id and name were gone
        # by the time the arguments came and the call reached the client as
        # `tool_use` with `input: {}`.
        self.calls: dict[int, dict] = {}
        # Which call's block is open, so two calls in a row are two blocks.
        self.open_call_index: int | None = None

    def close(self):
        """End the open block, if any. A thinking block signs off first."""
        if self.kind is None:
            return
        if self.kind == "thinking":
            yield stream_signature_delta(self.index)
        yield stream_content_block_stop(self.index)
        self.index += 1
        self.kind = None
        self.open_call_index = None

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

    def tool_delta(self, index: int, arguments: str):
        """Arguments for the call named at ``index``, opening its block.

        The block is opened *here*, when the arguments arrive, and not when
        the name did. On this protocol a `content_block_start` of type
        `tool_use` already carries `"input": {}` and is, on its own, a
        complete zero-argument call -- there is no frame for "the name is
        known, the arguments are still coming". So a name sent before its
        arguments cannot be represented, and sending one anyway put a
        syntactically perfect call the model never made in front of the
        client: measured on a DeepSeek-V4 response containing two calls,
        which reached `/v1/messages` as three.

        OpenAI's wire has the same shape -- a `tool_calls` delta with
        `arguments: ""` -- and is safe with it, because a client there
        accumulates by index and waits for `finish_reason`. That is why the
        name still goes out early on the other endpoint and not on this one.
        """
        call = self.calls.get(index)
        if call is None:
            # Arguments for a call whose name never came. There is nothing to
            # open a block with, and opening one with empty fields would be a
            # `tool_use` the client can neither dispatch nor return a result
            # for. A parser bug; dropping the frame is the only honest option.
            return
        if self.kind != "tool_use" or self.open_call_index != index:
            yield from self.open("tool_use", **call)
            self.open_call_index = index
        yield stream_content_block_delta(self.index, arguments, "tool_use")


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

    Which call is open lives on `blocks` and not here, because this runs once
    per parser batch and one call spans two of them. See `AnthropicBlocks`.
    """
    for etype, edata in events:
        if etype == "content":
            yield from blocks.delta("text", edata)
        elif etype == "tool_call_start":
            # Recorded, not sent. See `AnthropicBlocks.tool_delta` for why a
            # name on its own cannot be put on this wire.
            fn = edata.get("function", {})
            blocks.calls[edata.get("index", 0)] = {
                "tool_use_id": edata.get("id", ""),
                "tool_name": fn.get("name", ""),
            }
        elif etype == "tool_call_args":
            yield from blocks.tool_delta(
                edata.get("index", 0), edata.get("function", {}).get("arguments", "")
            )
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
    dummy_sig = _sig()
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


def stream_error(message: str, error_type: str = "api_error") -> str:
    """Anthropic's `error` event, for a stream that cannot finish.

    Without one, an exception raised inside the SSE generator ended the
    response mid-frame: an open content block with no `content_block_stop`, no
    `message_delta`, no `message_stop` and nothing saying why. An Anthropic
    SDK client blocks on the unterminated block until its own read timeout
    rather than surfacing the failure.
    """
    return format_sse(
        "error", {"type": "error", "error": {"type": error_type, "message": message}}
    )


def stream_message_stop() -> str:
    return format_sse("message_stop", {"type": "message_stop"})
