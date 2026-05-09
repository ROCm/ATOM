# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Text completion handler for the OpenAI-compatible API."""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

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
    index: int = 0,
    **extra_fields: Any,
) -> str:
    """Create a text completion chunk in SSE format.

    ``index`` selects ``choices[0].index``; fan-out siblings share one SSE
    stream and are distinguished by this field.
    """
    chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "text": text,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    chunk.update(extra_fields)
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

        extra_fields: Dict[str, Any] = {}
        if "kv_transfer_params" in chunk_data:
            extra_fields["kv_transfer_params"] = chunk_data["kv_transfer_params"]

        yield create_completion_chunk(
            request_id,
            model,
            new_text,
            finish_reason=chunk_data.get("finish_reason"),
            **extra_fields,
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
    """Build a non-streaming text completion response (single choice)."""
    response = CompletionResponse(
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
    if "kv_transfer_output_meta_info" in final_output:
        response = response.model_copy(
            update={
                "kv_transfer_params": final_output["kv_transfer_output_meta_info"],
            }
        )
    return response


def build_completion_response_multi(
    request_id: str,
    model: str,
    final_outputs: List[Dict[str, Any]],
) -> CompletionResponse:
    """Build a non-streaming response with one choice per fan-out sibling."""
    assert final_outputs, "build_completion_response_multi requires at least one output"
    choices = [
        {
            "index": i,
            "text": out["text"],
            "finish_reason": out["finish_reason"],
        }
        for i, out in enumerate(final_outputs)
    ]
    prompt_tokens = final_outputs[0]["num_tokens_input"]
    completion_tokens = sum(out["num_tokens_output"] for out in final_outputs)
    return CompletionResponse(
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


async def stream_completion_response_fanout(
    request_id: str,
    model: str,
    prompt: str,
    shared_queue: asyncio.Queue,
    seq_ids: List[int],
    tokenizer,
    cleanup_fn,
) -> AsyncGenerator[str, None]:
    """Streaming variant multiplexing ``len(seq_ids)`` siblings into one SSE.

    Each chunk pulled from ``shared_queue`` is a ``(sibling_index, chunk_data)``
    tuple; we re-emit with ``choices[0].index = sibling_index``. Finishes
    only when every sibling has reported ``finished=True``.
    """
    n = len(seq_ids)
    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = [0] * n
    finished = [False] * n

    while not all(finished):
        idx, chunk_data = await shared_queue.get()
        if finished[idx]:
            continue
        new_text = chunk_data["text"]
        num_tokens_output[idx] += len(chunk_data.get("token_ids", []))

        extra_fields: Dict[str, Any] = {}
        if "kv_transfer_params" in chunk_data:
            extra_fields["kv_transfer_params"] = chunk_data["kv_transfer_params"]

        yield create_completion_chunk(
            request_id,
            model,
            new_text,
            finish_reason=chunk_data.get("finish_reason"),
            index=idx,
            **extra_fields,
        )

        if chunk_data.get("finished", False):
            finished[idx] = True

    for sid in seq_ids:
        cleanup_fn(request_id, sid)

    for i in range(n):
        yield create_completion_chunk(request_id, model, "", "stop", index=i)

    usage = {
        "prompt_tokens": num_tokens_input,
        "completion_tokens": sum(num_tokens_output),
        "total_tokens": num_tokens_input + sum(num_tokens_output),
        "num_choices": n,
    }
    usage_chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    yield f"data: {json.dumps(usage_chunk)}\n\n"
    yield STREAM_DONE_MESSAGE
