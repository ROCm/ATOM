# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Text completion handler for the OpenAI-compatible API."""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from .protocol import (
    STREAM_DONE_MESSAGE,
    TEXT_COMPLETION_OBJECT,
    CompletionResponse,
)

logger = logging.getLogger("atom")


def create_completion_chunk(
    request_id: str,
    model: str,
    text: str,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None,
) -> str:
    """Create a text completion chunk in SSE format."""
    chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


async def stream_completion_response(
    request_id: str,
    model: str,
    prompt: str,
    stream_queue: asyncio.Queue,
    seq_id: int,
    tokenizer,
    cleanup_fn,
) -> AsyncGenerator[str, None]:
    """Generate streaming text completion response."""
    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = 0

    while True:
        chunk_data = await stream_queue.get()
        new_text = chunk_data["text"]
        num_tokens_output += len(chunk_data.get("token_ids", []))

        yield create_completion_chunk(
            request_id,
            model,
            new_text,
            finish_reason=chunk_data.get("finish_reason"),
        )

        if chunk_data.get("finished", False):
            break

    cleanup_fn(request_id, seq_id)

    usage = {
        "prompt_tokens": num_tokens_input,
        "completion_tokens": num_tokens_output,
        "total_tokens": num_tokens_input + num_tokens_output,
    }
    yield create_completion_chunk(request_id, model, "", "stop")
    # Usage-only chunk
    usage_chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield STREAM_DONE_MESSAGE


def build_completion_response(
    request_id: str,
    model: str,
    final_output: Dict[str, Any],
) -> CompletionResponse:
    """Build a non-streaming text completion response."""
    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "text": final_output["text"],
                "finish_reason": final_output["finish_reason"],
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
