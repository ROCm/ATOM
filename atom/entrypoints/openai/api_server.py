# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
ATOM OpenAI-compatible API Server.

FastAPI-based server implementing OpenAI-compatible endpoints for chat
completions and text completions, with reasoning content separation for
thinking models (Kimi-K2, DeepSeek-R1, Qwen3, etc.).

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
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
from atom.model_engine.llm_engine import _load_tokenizer
from atom.model_engine.request import RequestOutput
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from .protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from .serving_chat import build_chat_response, stream_chat_response
from .serving_completion import build_completion_response, stream_completion_response

# Configure logging
logger = logging.getLogger("atom")

# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


# ============================================================================
# Global State
# ============================================================================

engine = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""
default_chat_template_kwargs: Dict[str, Any] = {}
_stream_queues: Dict[str, asyncio.Queue] = {}
_seq_id_to_request_id: Dict[int, str] = {}
_stream_loops: Dict[str, AbstractEventLoop] = {}
_request_start_times: Dict[str, float] = {}
_request_logger: Optional[logging.Logger] = None


# ============================================================================
# Request/Response Logging
# ============================================================================


def _log_request_event(event_type: str, request_id: str, data: Any) -> None:
    """Write a JSONL entry to the request log file (if enabled)."""
    if _request_logger is None:
        return
    entry = {
        "timestamp": time.time(),
        "request_id": request_id,
        "type": event_type,
        "data": data,
    }
    _request_logger.info(json.dumps(entry, default=str))


async def _logged_stream(
    gen: AsyncGenerator[str, None], request_id: str
) -> AsyncGenerator[str, None]:
    """Wrap a streaming generator to log each SSE chunk."""
    async for chunk in gen:
        if _request_logger is not None and chunk.startswith("data: "):
            payload = chunk[6:].strip()
            if payload != "[DONE]":
                _log_request_event("stream_chunk", request_id, json.loads(payload))
            else:
                _log_request_event("stream_done", request_id, None)
        yield chunk


# ============================================================================
# Engine Interface
# ============================================================================


def _build_sampling_params(
    temperature: float,
    max_tokens: int,
    stop_strings: Optional[List[str]],
    ignore_eos: bool,
    top_k: int = -1,
    top_p: float = 1.0,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_strings=stop_strings,
        ignore_eos=ignore_eos,
    )


def _send_stream_chunk_direct(
    request_output: RequestOutput,
    request_id: str,
    stream_queue: asyncio.Queue,
    loop: AbstractEventLoop,
) -> None:
    """Send stream chunk directly to the queue."""
    global tokenizer

    new_text = tokenizer.decode(request_output.output_tokens, skip_special_tokens=True)
    started_at = _request_start_times.get(request_id)
    chunk_data = {
        "text": new_text,
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
        "finished_at": time.time(),
        "started_at": started_at,
    }
    loop.call_soon_threadsafe(stream_queue.put_nowait, chunk_data)


async def generate_async(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate text asynchronously for non-streaming requests."""
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
    """Validate that the requested model matches the server's model."""
    if requested_model is not None and requested_model != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{requested_model}' does not match "
            f"server model '{model_name}'",
        )


async def setup_streaming_request(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> Tuple[int, asyncio.Queue]:
    """Set up a streaming request with the engine."""
    global engine, _stream_queues, _seq_id_to_request_id
    global _stream_loops, _request_start_times

    stream_queue: asyncio.Queue = asyncio.Queue()
    stream_loop = asyncio.get_running_loop()
    _stream_queues[request_id] = stream_queue
    _stream_loops[request_id] = stream_loop
    _request_start_times[request_id] = time.time()

    def stream_callback(request_output: RequestOutput) -> None:
        _send_stream_chunk_direct(request_output, request_id, stream_queue, stream_loop)

    executor_loop = asyncio.get_event_loop()

    def do_preprocess():
        seq = engine.io_processor.preprocess(
            prompt, sampling_params, stream_callback=stream_callback
        )
        _seq_id_to_request_id[seq.id] = request_id
        return seq

    seq = await executor_loop.run_in_executor(None, do_preprocess)
    seq_id = seq.id

    logger.info(f"API: Created request_id={request_id}, seq_id={seq_id}")
    engine.core_mgr.add_request([seq])

    return seq_id, stream_queue


def cleanup_streaming_request(request_id: str, seq_id: int) -> None:
    """Clean up resources for a streaming request."""
    global engine, _stream_queues, _seq_id_to_request_id
    global _stream_loops, _request_start_times

    _stream_queues.pop(request_id, None)
    _seq_id_to_request_id.pop(seq_id, None)
    _stream_loops.pop(request_id, None)
    _request_start_times.pop(request_id, None)
    engine.io_processor.requests.pop(seq_id, None)


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Server started successfully and ready to accept requests")
    yield
    logger.info("Server shutting down, releasing resources...")
    if engine is not None:
        engine.close()


app = FastAPI(title="ATOM OpenAI API Server", lifespan=lifespan)


# ---- Error handlers ----


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": 400,
            }
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_server_error",
                "code": 500,
            }
        },
    )


# ---- Endpoints ----


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        messages = request.get_messages()

        merged_kwargs = dict(default_chat_template_kwargs)
        if request.chat_template_kwargs:
            merged_kwargs.update(request.chat_template_kwargs)
        merged_kwargs["tokenize"] = False
        merged_kwargs["add_generation_prompt"] = True
        # Pass tools so the chat template can inject tool declarations
        if request.tools:
            merged_kwargs["tools"] = request.tools

        prompt = tokenizer.apply_chat_template(
            [msg.to_template_dict() for msg in messages],
            **merged_kwargs,
        )

        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        _log_request_event("request", request_id, request.model_dump())

        # Streaming
        if request.stream:
            seq_id, stream_queue = await setup_streaming_request(
                prompt, sampling_params, request_id
            )
            gen = stream_chat_response(
                request_id,
                model_name,
                prompt,
                stream_queue,
                seq_id,
                tokenizer,
                cleanup_streaming_request,
            )
            return StreamingResponse(
                _logged_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        final_output = None
        async for output in generate_async(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        resp = build_chat_response(
            request_id, model_name, final_output["text"], final_output
        )
        _log_request_event("response", request_id, resp.model_dump())
        return resp

    except ValueError as e:
        logger.error(f"Validation error in chat_completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle text completion requests (OpenAI-compatible)."""
    global engine, tokenizer, model_name

    validate_model(request.model)

    try:
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        request_id = f"cmpl-{uuid.uuid4().hex}"

        _log_request_event("request", request_id, request.model_dump())

        # Streaming
        if request.stream:
            seq_id, stream_queue = await setup_streaming_request(
                request.prompt, sampling_params, request_id
            )
            gen = stream_completion_response(
                request_id,
                model_name,
                request.prompt,
                stream_queue,
                seq_id,
                tokenizer,
                cleanup_streaming_request,
            )
            return StreamingResponse(
                _logged_stream(gen, request_id),
                media_type="text/event-stream",
            )

        # Non-streaming
        final_output = None
        async for output in generate_async(request.prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        resp = build_completion_response(request_id, model_name, final_output)
        _log_request_event("response", request_id, resp.model_dump())
        return resp

    except ValueError as e:
        logger.error(f"Validation error in completions: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    global model_name
    return ModelList(data=[ModelCard(id=model_name)])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/start_profile")
async def start_profile():
    """Start profiling the engine."""
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
    """Stop profiling the engine."""
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
    global engine, tokenizer, model_name, default_chat_template_kwargs, _request_logger

    parser = argparse.ArgumentParser(description="ATOM OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port (note: --port is used for internal engine communication)",
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=str,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
            "Example: '{\"enable_thinking\": false}'"
        ),
    )
    parser.add_argument(
        "--request-log",
        type=str,
        default=None,
        help="Path to JSONL file for logging all API requests and responses (debug)",
    )
    args = parser.parse_args()

    if args.request_log:
        _request_logger = logging.getLogger("atom.request_log")
        _request_logger.setLevel(logging.INFO)
        _request_logger.propagate = False
        fh = logging.FileHandler(args.request_log, mode="a")
        fh.setFormatter(logging.Formatter("%(message)s"))
        _request_logger.addHandler(fh)
        logger.info(f"Request logging enabled: {args.request_log}")

    if args.default_chat_template_kwargs:
        default_chat_template_kwargs = json.loads(args.default_chat_template_kwargs)
        logger.info(f"Default chat template kwargs: {default_chat_template_kwargs}")

    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = _load_tokenizer(args.model, args.trust_remote_code)
    model_name = args.model

    logger.info(f"Initializing engine with model {args.model}...")
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(tokenizer=tokenizer)

    import signal

    def _sigint_handler(signum, frame):
        logger.info("Received SIGINT, shutting down engine...")
        engine.close()
        import psutil

        try:
            current = psutil.Process()
            children = current.children(recursive=True)
            psutil.wait_procs(children, timeout=2)
            alive = [c for c in children if c.is_running()]
            for c in alive:
                c.kill()
        except psutil.NoSuchProcess:
            pass
        logger.info("Engine shutdown complete.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    logger.info(f"Starting server on {args.host}:{args.server_port}...")
    uvicorn.run(app, host=args.host, port=args.server_port)


if __name__ == "__main__":
    main()
