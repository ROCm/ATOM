import argparse
import asyncio
import json
import time
import queue
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import uuid
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.request import RequestOutput
import logging

logger = logging.getLogger("atom")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


engine = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""
_stream_queues: Dict[str, queue.Queue] = {}
_seq_id_to_request_id: Dict[int, str] = {}


def create_chat_completion_chunk(
    request_id: str,
    model: str,
    content: str = "",
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None
) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason, "logprobs": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def create_chat_usage_chunk(
    request_id: str,
    model: str,
    usage: Dict
) -> str:
    """Create a chunk containing only usage information for chat completions (no choices)."""
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
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
    usage: Optional[Dict] = None
) -> str:
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def create_usage_chunk(
    request_id: str,
    model: str,
    usage: Dict
) -> str:
    """Create a chunk containing only usage information (no choices)."""
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "usage": usage,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def send_stream_chunk(request_output: RequestOutput):
    global tokenizer, _stream_queues, _seq_id_to_request_id
    
    # Get request_id from sequence ID
    request_id = _seq_id_to_request_id.get(request_output.request_id)
    if request_id is None:
        logger.warning(f"send_stream_chunk: No request_id found for sequence {request_output.request_id}")
        return
    
    # Decode the new tokens
    new_text = tokenizer.decode(request_output.output_tokens, skip_special_tokens=True)
    logger.debug(f"send_stream_chunk: seq_id={request_output.request_id}, request_id={request_id}, tokens={request_output.output_tokens}, text='{new_text}'")
    
    # Get the queue for this request
    stream_queue = _stream_queues.get(request_id)
    if stream_queue is None:
        logger.warning(f"send_stream_chunk: No queue found for request_id {request_id}")
        return
    
    # Prepare chunk data
    chunk_data = {
        "text": new_text,
        "token_ids": request_output.output_tokens,
        "finished": request_output.finished,
        "finish_reason": request_output.finish_reason,
    }

    try:
        stream_queue.put_nowait(chunk_data)
        logger.debug(f"send_stream_chunk: Successfully put chunk data into queue for request_id {request_id}")
    except queue.Full:
        logger.warning(f"send_stream_chunk: Queue full for request_id {request_id}, skipping chunk")
        pass


async def generate_async(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    global engine, tokenizer
    async for output in engine.generate_async(prompt, sampling_params, request_id):
        yield {
            "text": output["text"],
            "token_ids": output["token_ids"],
            "finish_reason": output.get("finish_reason"),
            "num_tokens_input": len(tokenizer.encode(prompt)),
            "num_tokens_output": len(output["token_ids"]),
            "ttft": output.get("ttft", 0.0),
            "tpot": output.get("tpot", 0.0),
            "latency": output.get("latency", 0.0),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Server started successfully and ready to accept requests")
    print("Server started successfully and ready to accept requests!")
    yield
    # Shutdown (if needed in the future)


app = FastAPI(title="Atom OpenAI API Server", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global engine, tokenizer, model_name
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if request.stream:
            async def generate_stream():
                stream_queue = queue.Queue()
                _stream_queues[request_id] = stream_queue

                def stream_callback(request_output: RequestOutput):
                    send_stream_chunk(request_output)

                seq = engine.io_processor.preprocess(prompt, sampling_params, stream_callback=stream_callback)
                seq_id = seq.id
                _seq_id_to_request_id[seq_id] = request_id
                logger.info(f"API: Created request_id={request_id}, seq_id={seq_id}, queue={stream_queue is not None}")
                engine.core_mgr.add_request([seq])
                logger.info(f"API: Added request to engine, callback registered: {seq.stream_callback is not None}")
                
                prev_text = ""
                num_tokens_input = len(tokenizer.encode(prompt))
                num_tokens_output = 0
                yield create_chat_completion_chunk(request_id, request.model, "")
                
                # Consume chunks from queue using executor to avoid blocking
                finished = False
                loop = asyncio.get_event_loop()
                while not finished:
                    try:
                        chunk_data = await asyncio.wait_for(
                            loop.run_in_executor(None, stream_queue.get),
                            timeout=30.0
                        )
                        new_text = chunk_data["text"]
                        current_text = prev_text + new_text
                        new_content = new_text
                        prev_text = current_text

                        chunk_token_ids = chunk_data.get("token_ids", [])
                        num_tokens_output += len(chunk_token_ids)
                        
                        yield create_chat_completion_chunk(
                            request_id, 
                            request.model, 
                            new_content,
                            finish_reason=chunk_data.get("finish_reason")
                        )
                        
                        if chunk_data.get("finished", False):
                            finished = True
                    except asyncio.TimeoutError:
                        break
                
                _stream_queues.pop(request_id, None)
                _seq_id_to_request_id.pop(seq_id, None)
                
                yield create_chat_completion_chunk(request_id, request.model, "", "stop")
                usage = {
                    "prompt_tokens": num_tokens_input,
                    "completion_tokens": num_tokens_output,
                    "total_tokens": num_tokens_input + num_tokens_output,
                }
                yield create_chat_usage_chunk(request_id, request.model, usage)
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # non-streaming
        final_output = None
        async for output in generate_async(prompt, sampling_params, request_id):
            final_output = output
        if final_output is None:
            raise RuntimeError("No output generated")

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_output["text"]},
                    "finish_reason": final_output["finish_reason"],
                }
            ],
            usage={
                "prompt_tokens": final_output["num_tokens_input"],
                "completion_tokens": final_output["num_tokens_output"],
                "total_tokens": final_output["num_tokens_input"] + final_output["num_tokens_output"],
                "ttft_s": round(final_output.get("ttft", 0.0), 4),
                "tpot_s": round(final_output.get("tpot", 0.0), 4),
                "latency_s": round(final_output.get("latency", 0.0), 4),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global engine, tokenizer, model_name
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )

        if request.stream:
            request_id = f"cmpl-{uuid.uuid4().hex}"

            async def generate_stream():
                stream_queue = queue.Queue()
                _stream_queues[request_id] = stream_queue
                
                def stream_callback(request_output: RequestOutput):
                    send_stream_chunk(request_output)
                
                seq = engine.io_processor.preprocess(request.prompt, sampling_params, stream_callback=stream_callback)
                seq_id = seq.id
                _seq_id_to_request_id[seq_id] = request_id
                engine.core_mgr.add_request([seq])
                
                prev_text = ""
                num_tokens_input = len(tokenizer.encode(request.prompt))
                num_tokens_output = 0
                
                finished = False
                loop = asyncio.get_event_loop()
                while not finished:
                    try:
                        chunk_data = await asyncio.wait_for(
                            loop.run_in_executor(None, stream_queue.get),
                            timeout=30.0
                        )
                        new_text = chunk_data["text"]
                        current_text = prev_text + new_text
                        new_content = new_text
                        prev_text = current_text
                        
                        chunk_token_ids = chunk_data.get("token_ids", [])
                        num_tokens_output += len(chunk_token_ids)
                        
                        yield create_completion_chunk(
                            request_id,
                            request.model,
                            new_content,
                            finish_reason=chunk_data.get("finish_reason")
                        )
                        
                        if chunk_data.get("finished", False):
                            finished = True
                    except asyncio.TimeoutError:
                        break
                
                # Cleanup
                _stream_queues.pop(request_id, None)
                _seq_id_to_request_id.pop(seq_id, None)
                
                yield create_completion_chunk(request_id, request.model, "", "stop")
                usage = {
                    "prompt_tokens": num_tokens_input,
                    "completion_tokens": num_tokens_output,
                    "total_tokens": num_tokens_input + num_tokens_output,
                }
                yield create_usage_chunk(request_id, request.model, usage)
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # non-streaming
        request_id = f"cmpl-{uuid.uuid4().hex}"
        final_output = None
        async for output in generate_async(request.prompt, sampling_params, request_id):
            final_output = output
        if final_output is None:
            raise RuntimeError("No output generated")

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
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
                "total_tokens": final_output["num_tokens_input"] + final_output["num_tokens_output"],
                "ttft_s": round(final_output.get("ttft", 0.0), 4),
                "tpot_s": round(final_output.get("tpot", 0.0), 4),
                "latency_s": round(final_output.get("latency", 0.0), 4),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
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
    return {"status": "ok"}


@app.post("/start_profile")
async def start_profile():
    global engine
    try:
        engine.start_profile()
        return {"status": "success", "message": "Profiling started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start profiling: {str(e)}")


@app.post("/stop_profile")
async def stop_profile():
    global engine
    try:
        engine.stop_profile()
        return {"status": "success", "message": "Profiling stopped. Trace files generated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop profiling: {str(e)}")


def main():
    global engine, tokenizer, model_name
    parser = argparse.ArgumentParser(description="Atom OpenAI API Server")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--server-port", type=int, default=8000,
        help="Server port (note: --port is used for internal engine communication)"
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_name = args.model

    print(f"Initializing async engine with model {args.model}...")
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_async_engine()

    print(f"Starting server on {args.host}:{args.server_port}...")
    uvicorn.run(app, host=args.host, port=args.server_port)


if __name__ == "__main__":
    main()