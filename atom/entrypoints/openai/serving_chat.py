# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat completion handler for the OpenAI-compatible API."""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from .protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    STREAM_DONE_MESSAGE,
    ChatCompletionResponse,
)
from .reasoning import ReasoningFilter, separate_reasoning
from .tool_parser import ToolCallStreamParser, parse_tool_calls

logger = logging.getLogger("atom")


def create_chat_chunk(
    request_id: str,
    model: str,
    delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None,
    index: int = 0,
) -> str:
    """Create a chat completion chunk in SSE format.

    ``index`` selects the ``choices[0].index`` field so fan-out siblings
    (SamplingParams.n>1) can be multiplexed over a single stream.
    """
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta if delta else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


async def stream_chat_response(
    request_id: str,
    model: str,
    prompt: str,
    stream_queue: asyncio.Queue,
    seq_id: int,
    tokenizer,
    cleanup_fn,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response with reasoning and tool calls.

    Yields SSE chunks with:
    - reasoning_content deltas during thinking phase
    - content deltas for the answer
    - tool_calls deltas when model invokes tools
    """
    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = 0
    reasoning_filter = ReasoningFilter()
    tool_parser = ToolCallStreamParser()
    has_tool_calls = False

    # Send initial role chunk
    yield create_chat_chunk(request_id, model, delta={"role": "assistant"})

    while True:
        chunk_data = await stream_queue.get()
        new_text = chunk_data["text"]
        num_tokens_output += len(chunk_data.get("token_ids", []))

        # Phase 1: Process through reasoning filter
        segments = reasoning_filter.process(new_text)
        if chunk_data.get("finished", False):
            segments.extend(reasoning_filter.flush())

        # Phase 2: For content segments, check for tool calls
        for field, text in segments:
            if field == "reasoning_content":
                if text:
                    yield create_chat_chunk(
                        request_id, model, delta={"reasoning_content": text}
                    )
            elif field == "content":
                # Run through tool parser
                events = tool_parser.process(text)
                for event_type, data in events:
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}
                        )
                    elif event_type == "tool_call_start":
                        has_tool_calls = True
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                        )
                    elif event_type == "tool_call_args":
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                        )

        if chunk_data.get("finished", False):
            # Flush tool parser
            for event_type, data in tool_parser.flush():
                if event_type == "content":
                    yield create_chat_chunk(request_id, model, delta={"content": data})
                elif event_type == "tool_call_start":
                    has_tool_calls = True
                    yield create_chat_chunk(
                        request_id, model, delta={"tool_calls": [data]}
                    )
                elif event_type == "tool_call_args":
                    yield create_chat_chunk(
                        request_id, model, delta={"tool_calls": [data]}
                    )
            break

    cleanup_fn(request_id, seq_id)

    # Final chunks
    finish_reason = "tool_calls" if has_tool_calls else "stop"
    usage = {
        "prompt_tokens": num_tokens_input,
        "completion_tokens": num_tokens_output,
        "total_tokens": num_tokens_input + num_tokens_output,
    }
    yield create_chat_chunk(request_id, model, finish_reason=finish_reason)
    usage_chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield STREAM_DONE_MESSAGE


def _build_chat_choice(
    raw_text: str,
    finish_reason: Optional[str],
    index: int = 0,
) -> Dict[str, Any]:
    """Build one entry of ``choices[...]`` from a raw output string.

    Factored out of :func:`build_chat_response` so multi-sample responses
    (SamplingParams.n>1) can reuse the reasoning + tool-call separation
    without duplicating the logic.
    """
    reasoning_content, content_with_tools = separate_reasoning(raw_text)
    content, tool_calls = parse_tool_calls(content_with_tools)

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

    effective_finish_reason = "tool_calls" if tool_calls else finish_reason
    return {
        "index": index,
        "message": message,
        "finish_reason": effective_finish_reason,
    }


def build_chat_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: Dict[str, Any],
) -> ChatCompletionResponse:
    """Build a non-streaming chat completion response (single choice)."""
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[_build_chat_choice(raw_text, final_output["finish_reason"], index=0)],
        usage={
            "prompt_tokens": final_output["num_tokens_input"],
            "completion_tokens": final_output["num_tokens_output"],
            "total_tokens": final_output["num_tokens_input"]
            + final_output["num_tokens_output"],
            "ttft_s": round(final_output.get("ttft", 0.0), 4),
            "tpot_s": round(final_output.get("tpot", 0.0), 4),
            "latency_s": round(final_output.get("latency", 0.0), 4),
        },
    )


def build_chat_response_multi(
    request_id: str,
    model: str,
    final_outputs: List[Dict[str, Any]],
) -> ChatCompletionResponse:
    """Build a non-streaming response with one choice per fan-out sibling.

    Assumes all ``final_outputs`` share the same prompt and therefore the
    same ``num_tokens_input``. Completion-token counts are summed across
    siblings for usage; ttft/tpot/latency are reported as the max observed
    across siblings, which approximates wall-clock time to return the full
    multi-sample response to the client.
    """
    assert final_outputs, "build_chat_response_multi requires at least one output"
    choices = [
        _build_chat_choice(out["text"], out["finish_reason"], index=i)
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_s": round(
                max((out.get("ttft", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "tpot_s": round(
                max((out.get("tpot", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "latency_s": round(
                max((out.get("latency", 0.0) for out in final_outputs), default=0.0), 4
            ),
            "num_choices": len(final_outputs),
        },
    )


async def stream_chat_response_fanout(
    request_id: str,
    model: str,
    prompt: str,
    shared_queue: asyncio.Queue,
    seq_ids: List[int],
    tokenizer,
    cleanup_fn,
) -> AsyncGenerator[str, None]:
    """Streaming variant that multiplexes ``len(seq_ids)`` fan-out siblings
    into a single SSE stream, tagging every chunk with ``choices[0].index``.

    The shared queue receives ``(sibling_index, chunk_data)`` tuples from
    the engine callbacks registered in :func:`setup_streaming_request_fanout`.
    Reasoning + tool-call state is kept independently per sibling.
    """
    n = len(seq_ids)
    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = [0] * n
    reasoning_filters = [ReasoningFilter() for _ in range(n)]
    tool_parsers = [ToolCallStreamParser() for _ in range(n)]
    has_tool_calls = [False] * n
    finished = [False] * n

    for i in range(n):
        yield create_chat_chunk(request_id, model, delta={"role": "assistant"}, index=i)

    while not all(finished):
        idx, chunk_data = await shared_queue.get()
        if finished[idx]:
            # Defensive: should not happen, engine emits finished once per seq.
            continue
        new_text = chunk_data["text"]
        num_tokens_output[idx] += len(chunk_data.get("token_ids", []))

        segments = reasoning_filters[idx].process(new_text)
        if chunk_data.get("finished", False):
            segments.extend(reasoning_filters[idx].flush())

        for field, text in segments:
            if field == "reasoning_content":
                if text:
                    yield create_chat_chunk(
                        request_id,
                        model,
                        delta={"reasoning_content": text},
                        index=idx,
                    )
            elif field == "content":
                for event_type, data in tool_parsers[idx].process(text):
                    if event_type == "content":
                        yield create_chat_chunk(
                            request_id, model, delta={"content": data}, index=idx
                        )
                    elif event_type == "tool_call_start":
                        has_tool_calls[idx] = True
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )
                    elif event_type == "tool_call_args":
                        yield create_chat_chunk(
                            request_id,
                            model,
                            delta={"tool_calls": [data]},
                            index=idx,
                        )

        if chunk_data.get("finished", False):
            for event_type, data in tool_parsers[idx].flush():
                if event_type == "content":
                    yield create_chat_chunk(
                        request_id, model, delta={"content": data}, index=idx
                    )
                elif event_type == "tool_call_start":
                    has_tool_calls[idx] = True
                    yield create_chat_chunk(
                        request_id,
                        model,
                        delta={"tool_calls": [data]},
                        index=idx,
                    )
                elif event_type == "tool_call_args":
                    yield create_chat_chunk(
                        request_id,
                        model,
                        delta={"tool_calls": [data]},
                        index=idx,
                    )
            finished[idx] = True

    # Clean up all sibling seq_id entries then the shared request state.
    for sid in seq_ids:
        cleanup_fn(request_id, sid)

    for i in range(n):
        finish_reason = "tool_calls" if has_tool_calls[i] else "stop"
        yield create_chat_chunk(request_id, model, finish_reason=finish_reason, index=i)

    usage = {
        "prompt_tokens": num_tokens_input,
        "completion_tokens": sum(num_tokens_output),
        "total_tokens": num_tokens_input + sum(num_tokens_output),
        "num_choices": n,
    }
    usage_chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield STREAM_DONE_MESSAGE
