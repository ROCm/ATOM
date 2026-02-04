# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Atom OpenAI-compatible API Server.

This module provides a FastAPI-based server that implements OpenAI-compatible
endpoints for chat completions and text completions.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import uvicorn
from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.request import RequestOutput
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

# Configure logging
logger = logging.getLogger("atom")

# Constants
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 256
CHAT_COMPLETION_OBJECT = "chat.completion"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
TEXT_COMPLETION_OBJECT = "text_completion"
STREAM_DONE_MESSAGE = "data: [DONE]\n\n"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatMessage(BaseModel):
    """Represents a single chat message."""

    role: str
    content: str

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""

    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[List[ChatMessage]] = None  # Accept 'prompt' as alias
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False
    seed: Optional[int] = None

    def get_messages(self) -> List[ChatMessage]:
        """Get messages from either 'messages' or 'prompt' field."""
        if self.messages is not None:
            return self.messages
        elif self.prompt is not None:
            return self.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' field is required")


class CompletionRequest(BaseModel):
    """Request model for text completions."""

    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_p: Optional[float] = DEFAULT_TOP_P
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str
    object: str = CHAT_COMPLETION_OBJECT
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]

    model_config = ConfigDict(extra="allow")


class CompletionResponse(BaseModel):
    """Response model for text completions."""

    id: str
    object: str = TEXT_COMPLETION_OBJECT
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


# ============================================================================
# Global State
# ============================================================================


engine = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""
_stream_queues: Dict[str, asyncio.Queue] = {}
_seq_id_to_request_id: Dict[int, str] = {}
_stream_loops: Dict[str, AbstractEventLoop] = {}
_request_start_times: Dict[str, float] = {}


# ============================================================================
# Utility Functions
# ============================================================================


def create_chat_completion_chunk(
    request_id: str,
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None,
) -> str:
    """Create a chat completion chunk in SSE format.

    Args:
        request_id: Unique request identifier
        model: Model name
        content: Text content for this chunk
        finish_reason: Reason for completion (if finished)
        usage: Token usage statistics (optional)

    Returns:
        Formatted SSE chunk string
    """
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def create_chat_usage_chunk(request_id: str, model: str, usage: Dict) -> str:
    """Create a usage-only chunk for chat completions.

    Args:
        request_id: Unique request identifier
        model: Model name
        usage: Token usage statistics

    Returns:
        Formatted SSE chunk string
    """
    chunk = {
        "id": request_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def create_completion_chunk(
    request_id: str,
    model: str,
    text: str,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None,
) -> str:
    """Create a text completion chunk in SSE format.

    Args:
        request_id: Unique request identifier
        model: Model name
        text: Generated text for this chunk
        finish_reason: Reason for completion (if finished)
        usage: Token usage statistics (optional)

    Returns:
        Formatted SSE chunk string
    """
    chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def _stream_finish_log(
    request_id: str,
    started_at: Optional[float],
    finished_at: Optional[float],
    label: str,
) -> str:
    """Format a concise finish log message for streaming."""
    gen_latency = (
        finished_at - started_at
        if finished_at is not None and started_at is not None
        else None
    )
    queue_latency = time.time() - finished_at if finished_at is not None else None
    gen_str = f"{gen_latency:.3f}s gen" if gen_latency is not None else "gen=n/a"
    queue_str = (
        f"{queue_latency:.3f}s queue" if queue_latency is not None else "queue=n/a"
    )
    return f"{label}: Finished streaming for request_id {request_id} {gen_str}, {queue_str}"


def _build_sampling_params(
    temperature: float,
    max_tokens: int,
    stop_strings: Optional[List[str]],
    ignore_eos: bool,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        ignore_eos=ignore_eos,
    )


def _build_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_usage_chunk(request_id: str, model: str, usage: Dict) -> str:
    """Create a usage-only chunk for text completions.

    Args:
        request_id: Unique request identifier
        model: Model name
        usage: Token usage statistics

    Returns:
        Formatted SSE chunk string
    """
    chunk = {
        "id": request_id,
        "object": TEXT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _send_stream_chunk_direct(
    request_output: RequestOutput,
    request_id: str,
    stream_queue: asyncio.Queue,
    loop: AbstractEventLoop,
) -> None:
    """Send stream chunk directly to the queue.

    This function avoids race conditions by directly using the provided queue
    instead of looking it up in the global mapping.

    Args:
        request_output: Output from the engine
        request_id: Request identifier for logging
        stream_queue: Queue to send the chunk to
    """
    global tokenizer

    # Decode the new tokens
    new_text = tokenizer.decode(request_output.output_tokens, skip_special_tokens=True)
    logger.debug(
        f"send_stream_chunk_direct: seq_id={request_output.request_id}, "
        f"request_id={request_id}, tokens={request_output.output_tokens}, "
        f"text='{new_text[:50]}...'"
    )

    started_at = _request_start_times.get(request_id)
    finished_at = time.time()
    # Prepare chunk data
    chunk_data = {
        "text": new_text,
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
        "finished_at": finished_at,
        "started_at": started_at,
    }

    loop.call_soon_threadsafe(stream_queue.put_nowait, chunk_data)


def send_stream_chunk(request_output: RequestOutput) -> None:
    """Send stream chunk using global queue mapping.

    Args:
        request_output: Output from the engine
    """
    global tokenizer, _stream_queues, _seq_id_to_request_id, _stream_loops

    request_id = _seq_id_to_request_id.get(request_output.request_id)
    if request_id is None:
        logger.warning(
            f"send_stream_chunk: No request_id found for sequence "
            f"{request_output.request_id}"
        )
        return

    stream_queue = _stream_queues.get(request_id)
    if stream_queue is None:
        logger.warning(f"send_stream_chunk: No queue found for request_id {request_id}")
        return

    stream_loop = _stream_loops.get(request_id)
    if stream_loop is None:
        try:
            stream_loop = asyncio.get_running_loop()
        except RuntimeError:
            stream_loop = asyncio.get_event_loop()

    _send_stream_chunk_direct(request_output, request_id, stream_queue, stream_loop)


async def generate_async(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate text asynchronously for non-streaming requests.

    Args:
        prompt: Input prompt
        sampling_params: Sampling parameters
        request_id: Request identifier

    Yields:
        Dictionary containing generated text and metadata
    """
    global engine, tokenizer

    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    started_at = time.time()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    all_token_ids: List[int] = []
    finish_reason: Optional[str] = None
    seq = None

    def completion_callback(request_output: RequestOutput):
        """Callback that receives incremental tokens and completion signal."""
        now = time.time()
        loop.call_soon_threadsafe(
            token_queue.put_nowait,
            {
                "token_ids": request_output.output_tokens,
                "finished": request_output.finished,
                "finish_reason": request_output.finish_reason,
                "ts": now,
            },
        )

    def do_preprocess():
        return engine.io_processor.preprocess(
            prompt, sampling_params, stream_callback=completion_callback
        )

    seq = await loop.run_in_executor(None, do_preprocess)

    engine.core_mgr.add_request([seq])

    # Consume tokens until finished
    while True:
        item = await token_queue.get()
        token_ids = item.get("token_ids") or []
        if token_ids:
            if first_token_at is None:
                first_token_at = item.get("ts", time.time())
            last_token_at = item.get("ts", time.time())
            all_token_ids.extend(token_ids)
        if item.get("finished", False):
            finish_reason = item.get("finish_reason")
            break

    text = tokenizer.decode(all_token_ids, skip_special_tokens=True)
    num_tokens_input = (
        seq.num_prompt_tokens if seq is not None else len(tokenizer.encode(prompt))
    )
    num_tokens_output = len(all_token_ids)
    finished_at = time.time()
    latency = finished_at - started_at
    ttft = (first_token_at - started_at) if first_token_at is not None else 0.0
    tpot = (
        (last_token_at - first_token_at) / (num_tokens_output - 1)
        if first_token_at is not None
        and last_token_at is not None
        and num_tokens_output > 1
        else 0.0
    )

    yield {
        "text": text,
        "token_ids": all_token_ids,
        "finish_reason": finish_reason,
        "num_tokens_input": num_tokens_input,
        "num_tokens_output": num_tokens_output,
        "ttft": ttft,
        "tpot": tpot,
        "latency": latency,
    }


def validate_model(requested_model: Optional[str]) -> None:
    """Validate that the requested model matches the server's model.

    Args:
        requested_model: Model name from the request

    Raises:
        HTTPException: If model mismatch detected
    """
    if requested_model is not None and requested_model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{requested_model}' does not match "
            f"server model '{model_name}'",
        )


async def setup_streaming_request(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> Tuple[int, asyncio.Queue]:
    """Set up a streaming request with the engine.

    Args:
        prompt: Input prompt
        sampling_params: Sampling parameters
        request_id: Request identifier

    Returns:
        Tuple of (sequence_id, stream_queue)
    """
    global engine, _stream_queues, _seq_id_to_request_id, _stream_loops, _request_start_times

    stream_queue: asyncio.Queue = asyncio.Queue()
    stream_loop = asyncio.get_running_loop()
    _stream_queues[request_id] = stream_queue
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    # Create callback closure
    def stream_callback(request_output: RequestOutput) -> None:
        _send_stream_chunk_direct(request_output, request_id, stream_queue, stream_loop)

    # Preprocess in executor to avoid blocking
    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seq = engine.io_processor.preprocess(
            prompt, sampling_params, stream_callback=stream_callback
        )
        _seq_id_to_request_id[seq.id] = request_id
        return seq

    seq = await executor_loop.run_in_executor(None, do_preprocess)
    seq_id = seq.id

    logger.info(
        f"API: Created request_id={request_id}, seq_id={seq_id}, "
        f"queue={stream_queue is not None}"
    )

    # Add request to engine
    engine.core_mgr.add_request([seq])

    return seq_id, stream_queue


def cleanup_streaming_request(request_id: str, seq_id: int) -> None:
    """Clean up resources for a streaming request.

    Args:
        request_id: Request identifier
        seq_id: Sequence identifier
    """
    global engine, _stream_queues, _seq_id_to_request_id, _stream_loops, _request_start_times

    _stream_queues.pop(request_id, None)
    _seq_id_to_request_id.pop(seq_id, None)
    _stream_loops.pop(request_id, None)
    _request_start_times.pop(request_id, None)
    engine.io_processor.requests.pop(seq_id, None)


async def stream_chat_response(
    request_id: str, model: str, prompt: str, stream_queue: asyncio.Queue, seq_id: int
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    global tokenizer

    num_tokens_input = len(tokenizer.encode(prompt))
    num_tokens_output = 0

    # Send initial empty chunk
    yield create_chat_completion_chunk(request_id, model, "")

    while True:
        chunk_data = await stream_queue.get()
        new_text = chunk_data["text"]

        num_tokens_output += len(chunk_data.get("token_ids", []))

        yield create_chat_completion_chunk(
            request_id,
            model,
            new_text,
            finish_reason=chunk_data.get("finish_reason"),
        )

        if chunk_data.get("finished", False):
            logger.info(
                _stream_finish_log(
                    request_id,
                    chunk_data.get("started_at"),
                    chunk_data.get("finished_at"),
                    "stream_chat_response",
                )
            )
            break

    cleanup_streaming_request(request_id, seq_id)

    usage = _build_usage(num_tokens_input, num_tokens_output)
    yield create_chat_completion_chunk(request_id, model, "", "stop")
    yield create_chat_usage_chunk(request_id, model, usage)
    yield STREAM_DONE_MESSAGE


async def stream_completion_response(
    request_id: str, model: str, prompt: str, stream_queue: asyncio.Queue, seq_id: int
) -> AsyncGenerator[str, None]:
    """Generate streaming text completion response."""
    global tokenizer

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
            logger.debug(
                _stream_finish_log(
                    request_id,
                    chunk_data.get("started_at"),
                    chunk_data.get("finished_at"),
                    "stream_completion_response",
                )
            )
            break

    cleanup_streaming_request(request_id, seq_id)

    usage = _build_usage(num_tokens_input, num_tokens_output)
    yield create_completion_chunk(request_id, model, "", "stop")
    yield create_usage_chunk(request_id, model, usage)
    yield STREAM_DONE_MESSAGE


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Server started successfully and ready to accept requests")
    yield
    # Shutdown (if needed in the future)


app = FastAPI(title="Atom OpenAI API Server", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible).

    Args:
        request: Chat completion request

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: On validation or processing errors
    """
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        # Get messages and format prompt
        messages = request.get_messages()
        prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Create sampling parameters
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # Handle streaming requests
        if request.stream:
            seq_id, stream_queue = await setup_streaming_request(
                prompt, sampling_params, request_id
            )

            return StreamingResponse(
                stream_chat_response(
                    request_id, model_name, prompt, stream_queue, seq_id
                ),
                media_type="text/event-stream",
            )

        # Handle non-streaming requests
        final_output = None
        async for output in generate_async(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        response_data = ChatCompletionResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_output["text"]},
                    "finish_reason": final_output["finish_reason"],
                    "text": final_output["text"],
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
        return response_data

    except ValueError as e:
        logger.error(f"Validation error in chat_completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle text completion requests (OpenAI-compatible).

    Args:
        request: Text completion request

    Returns:
        Text completion response or streaming response

    Raises:
        HTTPException: On validation or processing errors
    """
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        # Create sampling parameters
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )

        request_id = f"cmpl-{uuid.uuid4().hex}"

        # Handle streaming requests
        if request.stream:
            seq_id, stream_queue = await setup_streaming_request(
                request.prompt, sampling_params, request_id
            )

            return StreamingResponse(
                stream_completion_response(
                    request_id, model_name, request.prompt, stream_queue, seq_id
                ),
                media_type="text/event-stream",
            )

        # Handle non-streaming requests
        final_output = None
        async for output in generate_async(request.prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
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

    except ValueError as e:
        logger.error(f"Validation error in completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models.

    Returns:
        Dictionary containing model list
    """
    global model_name
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "atom",
            }
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        Status dictionary
    """
    return {"status": "ok"}


@app.post("/start_profile")
async def start_profile():
    """Start profiling the engine.

    Returns:
        Success status message

    Raises:
        HTTPException: If profiling fails to start
    """
    global engine
    try:
        engine.start_profile()
        return {"status": "success", "message": "Profiling started"}
    except Exception as e:
        logger.error(f"Failed to start profiling: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to start profiling: {str(e)}"
        )


@app.post("/stop_profile")
async def stop_profile():
    """Stop profiling the engine.

    Returns:
        Success status message

    Raises:
        HTTPException: If profiling fails to stop
    """
    global engine
    try:
        engine.stop_profile()
        return {
            "status": "success",
            "message": "Profiling stopped. Trace files generated.",
        }
    except Exception as e:
        logger.error(f"Failed to stop profiling: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to stop profiling: {str(e)}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the server."""
    global engine, tokenizer, model_name

    parser = argparse.ArgumentParser(description="Atom OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (note: --port is used for internal engine communication)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model_name = args.model

    print(f"Initializing engine with model {args.model}...")
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine()

    print(f"Starting server on {args.host}:{args.server_port}...")
    uvicorn.run(app, host=args.host, port=args.server_port)


if __name__ == "__main__":
    main()
