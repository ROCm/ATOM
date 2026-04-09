# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat completion handler for the OpenAI-compatible API."""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

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
) -> str:
    """Create a chat completion chunk in SSE format."""
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
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


def build_chat_response(
    request_id: str,
    model: str,
    raw_text: str,
    final_output: Dict[str, Any],
) -> ChatCompletionResponse:
    """Build a non-streaming chat completion response with reasoning and tool calls."""
    reasoning_content, content_with_tools = separate_reasoning(raw_text)

    # Parse tool calls from content
    content, tool_calls = parse_tool_calls(content_with_tools)

    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

    finish_reason = "tool_calls" if tool_calls else final_output["finish_reason"]

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
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
