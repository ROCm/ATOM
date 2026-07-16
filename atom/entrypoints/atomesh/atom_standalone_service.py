"""Python-owned ATOM standalone serving logic.

This adapter keeps OpenAI-compatible request semantics in Python so the Rust
standalone router only needs to bridge requests and responses.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import numbers
import queue
import threading
import time
import uuid
from typing import Any

from atom import SamplingParams
from atom.entrypoints.openai.api_server import _build_sampling_params, _coerce_n
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.entrypoints.openai.protocol import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    CHAT_COMPLETION_CHUNK_OBJECT,
    CompletionRequest,
    STREAM_DONE_MESSAGE,
    TEXT_COMPLETION_OBJECT,
)
from atom.entrypoints.openai.reasoning import ReasoningFilter
from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    build_chat_response_multi,
    create_chat_chunk,
)
from atom.entrypoints.openai.serving_completion import (
    build_completion_response,
    build_completion_response_multi,
    create_completion_chunk,
)
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser

logger = logging.getLogger("atom")


@dataclasses.dataclass
class EngineRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    future: concurrent.futures.Future[list[dict[str, Any]]]
    kv_transfer_params: dict[str, Any] | None = None


@dataclasses.dataclass
class EngineStreamRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    stream_queue: queue.Queue[dict[str, Any]]
    kv_transfer_params: dict[str, Any] | None = None


class AtomEngineService:
    def __init__(self, engine: Any, tokenizer: Any) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._queue: queue.Queue[EngineRequest | EngineStreamRequest | None] = (
            queue.Queue()
        )
        self._closed = threading.Event()
        self._active_futures: set[concurrent.futures.Future[list[dict[str, Any]]]] = (
            set()
        )
        self._active_futures_lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AtomStandaloneEngineService",
            daemon=True,
        )
        self._worker.start()

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        effective_n: int,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._closed.is_set():
            raise RuntimeError("ATOM standalone engine service is closed")

        future: concurrent.futures.Future[list[dict[str, Any]]] = (
            concurrent.futures.Future()
        )
        with self._active_futures_lock:
            self._active_futures.add(future)

        self._queue.put(
            EngineRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                effective_n=effective_n,
                future=future,
                kv_transfer_params=kv_transfer_params,
            )
        )
        try:
            return future.result()
        finally:
            with self._active_futures_lock:
                self._active_futures.discard(future)

    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
        with self._active_futures_lock:
            futures = list(self._active_futures)
        for future in futures:
            if not future.done():
                future.set_exception(
                    RuntimeError("ATOM standalone engine service is closed")
                )
        if self._worker.is_alive():
            self._worker.join(timeout=1)

    def _worker_loop(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                break
            if self._closed.is_set():
                if isinstance(request, EngineStreamRequest):
                    request.stream_queue.put(
                        {
                            "error": "ATOM standalone engine service is closed",
                        }
                    )
                    request.stream_queue.put({"done": True})
                else:
                    self._set_future_exception(
                        request.future,
                        RuntimeError("ATOM standalone engine service is closed"),
                    )
                continue

            try:
                if isinstance(request, EngineStreamRequest):
                    self._submit_stream_request(request)
                else:
                    self._submit_request(request)
            except Exception as error:
                if isinstance(request, EngineStreamRequest):
                    request.stream_queue.put(
                        {
                            "error": str(error),
                        }
                    )
                    request.stream_queue.put({"done": True})
                else:
                    self._set_future_exception(request.future, error)

    def _submit_request(self, request: EngineRequest) -> None:
        if request.effective_n > 1:
            seqs = self._preprocess_fanout_request(request)
        else:
            seqs = [self._preprocess_single_request(request)]
        self.engine.core_mgr.add_request(seqs)

    def start_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        effective_n: int,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> queue.Queue[dict[str, Any]]:
        if self._closed.is_set():
            raise RuntimeError("ATOM standalone engine service is closed")

        stream_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._queue.put(
            EngineStreamRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                effective_n=effective_n,
                stream_queue=stream_queue,
                kv_transfer_params=kv_transfer_params,
            )
        )
        return stream_queue

    def _submit_stream_request(self, request: EngineStreamRequest) -> None:
        if request.effective_n > 1:
            seqs = self._preprocess_fanout_stream_request(request)
        else:
            seqs = [self._preprocess_single_stream_request(request)]
        self.engine.core_mgr.add_request(seqs)

    def _preprocess_single_stream_request(self, request: EngineStreamRequest) -> Any:
        state = StreamRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            stream_queue=request.stream_queue,
            n=1,
        )

        def completion_callback(request_output: Any) -> None:
            state.record(0, request_output)

        return self.engine.io_processor.preprocess(
            request.prompt,
            request.sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=request.kv_transfer_params,
        )

    def _preprocess_fanout_stream_request(
        self, request: EngineStreamRequest
    ) -> list[Any]:
        state = StreamRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            stream_queue=request.stream_queue,
            n=request.effective_n,
        )

        def make_callback(index: int):
            def completion_callback(request_output: Any) -> None:
                state.record(index, request_output)

            return completion_callback

        return self.engine.io_processor.preprocess_fanout(
            request.prompt,
            request.sampling_params,
            stream_callbacks=[
                make_callback(index) for index in range(request.effective_n)
            ],
            kv_transfer_params=request.kv_transfer_params,
            parent_request_id=request.request_id,
        )

    def _preprocess_single_request(self, request: EngineRequest) -> Any:
        state = SingleRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            future=request.future,
        )

        def completion_callback(request_output: Any) -> None:
            state.record(request_output)

        seq = self.engine.io_processor.preprocess(
            request.prompt,
            request.sampling_params,
            stream_callback=completion_callback,
            kv_transfer_params=request.kv_transfer_params,
        )
        state.set_num_tokens_input(seq.num_prompt_tokens)
        return seq

    def _preprocess_fanout_request(self, request: EngineRequest) -> list[Any]:
        state = FanoutRequestState(
            request_id=request.request_id,
            tokenizer=self.tokenizer,
            future=request.future,
            n=request.effective_n,
        )

        def make_callback(index: int):
            def completion_callback(request_output: Any) -> None:
                state.record(index, request_output)

            return completion_callback

        seqs = self.engine.io_processor.preprocess_fanout(
            request.prompt,
            request.sampling_params,
            stream_callbacks=[
                make_callback(index) for index in range(request.effective_n)
            ],
            kv_transfer_params=request.kv_transfer_params,
            parent_request_id=request.request_id,
        )
        if seqs:
            state.set_num_tokens_input(seqs[0].num_prompt_tokens)
        return seqs

    @staticmethod
    def _set_future_exception(
        future: concurrent.futures.Future[list[dict[str, Any]]],
        error: BaseException,
    ) -> None:
        if not future.done():
            future.set_exception(error)


class SingleRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        future: concurrent.futures.Future[list[dict[str, Any]]],
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.future = future
        self.started_at = time.time()
        self.first_token_at: float | None = None
        self.last_token_at: float | None = None
        self.token_ids: list[int] = []
        self.finish_reason: str | None = None
        self.num_tokens_input = 0
        self.kv_transfer_output_meta_info: Any = None
        self._lock = threading.Lock()

    def set_num_tokens_input(self, num_tokens_input: int) -> None:
        self.num_tokens_input = num_tokens_input

    def record(self, request_output: Any) -> None:
        with self._lock:
            if self.future.done():
                return
            self.kv_transfer_output_meta_info = getattr(
                request_output, "kv_transfer_params_output", None
            )
            now = time.time()
            output_tokens = request_output.output_tokens or []
            if output_tokens:
                if self.first_token_at is None:
                    self.first_token_at = now
                self.last_token_at = now
                self.token_ids.extend(output_tokens)
            if request_output.finished:
                self.finish_reason = request_output.finish_reason
                self.future.set_result([self._build_output(time.time())])

    def _build_output(self, finished_at: float) -> dict[str, Any]:
        num_tokens_output = len(self.token_ids)
        ttft = (
            self.first_token_at - self.started_at
            if self.first_token_at is not None
            else 0.0
        )
        tpot = (
            (self.last_token_at - self.first_token_at) / (num_tokens_output - 1)
            if self.first_token_at is not None
            and self.last_token_at is not None
            and num_tokens_output > 1
            else 0.0
        )
        output = {
            "text": self.tokenizer.decode(self.token_ids, skip_special_tokens=True),
            "token_ids": self.token_ids,
            "finish_reason": self.finish_reason,
            "num_tokens_input": self.num_tokens_input,
            "num_tokens_output": num_tokens_output,
            "ttft": ttft,
            "tpot": tpot,
            "latency": finished_at - self.started_at,
        }
        if self.kv_transfer_output_meta_info is not None:
            output["kv_transfer_output_meta_info"] = self.kv_transfer_output_meta_info
        return output


class FanoutRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        future: concurrent.futures.Future[list[dict[str, Any]]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.future = future
        self.n = n
        self.started_at = time.time()
        self.per_tokens: list[list[int]] = [[] for _ in range(n)]
        self.per_first_token_at: list[float | None] = [None] * n
        self.per_last_token_at: list[float | None] = [None] * n
        self.per_finish_reason: list[str | None] = [None] * n
        self.finished = [False] * n
        self.num_tokens_input = 0
        self._lock = threading.Lock()

    def set_num_tokens_input(self, num_tokens_input: int) -> None:
        self.num_tokens_input = num_tokens_input

    def record(self, index: int, request_output: Any) -> None:
        with self._lock:
            if self.future.done() or self.finished[index]:
                return
            now = time.time()
            output_tokens = request_output.output_tokens or []
            if output_tokens:
                if self.per_first_token_at[index] is None:
                    self.per_first_token_at[index] = now
                self.per_last_token_at[index] = now
                self.per_tokens[index].extend(output_tokens)
            if request_output.finished:
                self.per_finish_reason[index] = request_output.finish_reason
                self.finished[index] = True
                if all(self.finished):
                    self.future.set_result(self._build_outputs(time.time()))

    def _build_outputs(self, finished_at: float) -> list[dict[str, Any]]:
        outputs = []
        for index in range(self.n):
            num_tokens_output = len(self.per_tokens[index])
            first_token_at = self.per_first_token_at[index]
            last_token_at = self.per_last_token_at[index]
            ttft = (
                first_token_at - self.started_at if first_token_at is not None else 0.0
            )
            tpot = (
                (last_token_at - first_token_at) / (num_tokens_output - 1)
                if first_token_at is not None
                and last_token_at is not None
                and num_tokens_output > 1
                else 0.0
            )
            outputs.append(
                {
                    "text": self.tokenizer.decode(
                        self.per_tokens[index], skip_special_tokens=True
                    ),
                    "token_ids": self.per_tokens[index],
                    "finish_reason": self.per_finish_reason[index],
                    "num_tokens_input": self.num_tokens_input,
                    "num_tokens_output": num_tokens_output,
                    "ttft": ttft,
                    "tpot": tpot,
                    "latency": finished_at - self.started_at,
                }
            )
        return outputs


class StreamRequestState:
    def __init__(
        self,
        request_id: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.stream_queue = stream_queue
        self.finished = [False] * n
        self._lock = threading.Lock()

    def record(self, index: int, request_output: Any) -> None:
        with self._lock:
            if self.finished[index]:
                return

            output_tokens = request_output.output_tokens or []
            text = (
                self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if output_tokens
                else ""
            )
            if output_tokens or request_output.finished:
                event = {
                    "index": index,
                    "text": text,
                    "token_ids": output_tokens,
                    "finished": request_output.finished,
                    "finish_reason": request_output.finish_reason,
                }
                if getattr(request_output, "kv_transfer_params_output", None):
                    event["kv_transfer_params"] = (
                        request_output.kv_transfer_params_output
                    )
                self.stream_queue.put(event)

            if request_output.finished:
                self.finished[index] = True
                if all(self.finished):
                    self.stream_queue.put({"done": True})


class ChatCompletionStreamState:
    def __init__(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.stream_queue = stream_queue
        self.num_tokens_input = len(tokenizer.encode(prompt))
        self.num_tokens_output = [0] * n
        self.reasoning_filters = [ReasoningFilter() for _ in range(n)]
        self.tool_parsers = [ToolCallStreamParser() for _ in range(n)]
        self.has_tool_calls = [False] * n
        self.finished = [False] * n
        self.role_sent = [False] * n
        self.completed = False
        self.closed = False
        self._pending_final_chunks: list[str] | None = None
        self._lock = threading.Lock()

    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            self._append_initial_role_chunks(chunks, max_items)
            if not self.completed and all(self.finished):
                chunks.extend(self._final_chunks(max_items - len(chunks)))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed and all(self.finished):
                        chunks.extend(self._final_chunks(max_items - len(chunks)))
                break

            with self._lock:
                chunks.extend(self._event_to_chunks(event, max_items - len(chunks)))
                if self.completed or self.closed:
                    break
        return chunks

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def _append_initial_role_chunks(self, chunks: list[str], max_items: int) -> None:
        for index, sent in enumerate(self.role_sent):
            if len(chunks) >= max_items:
                break
            if not sent:
                chunks.append(
                    create_chat_chunk(
                        self.request_id,
                        self.model_name,
                        delta={"role": "assistant"},
                        index=index,
                    )
                )
                self.role_sent[index] = True

    def _event_to_chunks(
        self, event: dict[str, Any], remaining_capacity: int
    ) -> list[str]:
        if remaining_capacity <= 0:
            return []
        if event.get("error"):
            if self._pending_final_chunks is None:
                self._pending_final_chunks = [
                    self._error_chunk(str(event["error"])),
                    STREAM_DONE_MESSAGE,
                ]
            return self._drain_pending_final_chunks(remaining_capacity)
        if event.get("done"):
            if not self.completed and all(self.finished):
                return self._final_chunks(remaining_capacity)
            return []

        index = int(event["index"])
        if self.finished[index]:
            return []

        chunks: list[str] = []
        text = event.get("text") or ""
        self.num_tokens_output[index] += len(event.get("token_ids", []))

        segments = self.reasoning_filters[index].process(text)
        if event.get("finished", False):
            segments.extend(self.reasoning_filters[index].flush())

        for field, segment_text in segments:
            if len(chunks) >= remaining_capacity:
                break
            if field == "reasoning_content":
                if segment_text:
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"reasoning_content": segment_text},
                            index=index,
                        )
                    )
            elif field == "content":
                for event_type, data in self.tool_parsers[index].process(segment_text):
                    if len(chunks) >= remaining_capacity:
                        break
                    if event_type == "content":
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"content": data},
                                index=index,
                            )
                        )
                    elif event_type == "tool_call_start":
                        self.has_tool_calls[index] = True
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"tool_calls": [data]},
                                index=index,
                            )
                        )
                    elif event_type == "tool_call_args":
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"tool_calls": [data]},
                                index=index,
                            )
                        )

        if event.get("finished", False):
            for event_type, data in self.tool_parsers[index].flush():
                if len(chunks) >= remaining_capacity:
                    break
                if event_type == "content":
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"content": data},
                            index=index,
                        )
                    )
                elif event_type == "tool_call_start":
                    self.has_tool_calls[index] = True
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
                elif event_type == "tool_call_args":
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
            self.finished[index] = True

        if all(self.finished) and len(chunks) < remaining_capacity:
            chunks.extend(self._final_chunks(remaining_capacity - len(chunks)))
        return chunks

    def _final_chunks(self, remaining_capacity: int) -> list[str]:
        if self._pending_final_chunks is None:
            chunks: list[str] = []
            for index, has_tool_calls in enumerate(self.has_tool_calls):
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                chunks.append(
                    create_chat_chunk(
                        self.request_id,
                        self.model_name,
                        finish_reason=finish_reason,
                        index=index,
                    )
                )
            completion_tokens = sum(self.num_tokens_output)
            usage = {
                "prompt_tokens": self.num_tokens_input,
                "completion_tokens": completion_tokens,
                "total_tokens": self.num_tokens_input + completion_tokens,
            }
            if len(self.num_tokens_output) > 1:
                usage["num_choices"] = len(self.num_tokens_output)
            usage_chunk = {
                "id": self.request_id,
                "object": CHAT_COMPLETION_CHUNK_OBJECT,
                "created": int(time.time()),
                "model": self.model_name,
                "usage": usage,
            }
            chunks.append(f"data: {json.dumps(usage_chunk)}\n\n")
            chunks.append(STREAM_DONE_MESSAGE)
            self._pending_final_chunks = chunks

        return self._drain_pending_final_chunks(remaining_capacity)

    def _drain_pending_final_chunks(self, remaining_capacity: int) -> list[str]:
        if not self._pending_final_chunks or remaining_capacity <= 0:
            return []
        chunks = self._pending_final_chunks[:remaining_capacity]
        del self._pending_final_chunks[:remaining_capacity]
        if not self._pending_final_chunks:
            self.completed = True
        return chunks

    @staticmethod
    def _error_chunk(message: str) -> str:
        return f"data: {json.dumps({'error': {'message': message}})}\n\n"


class CompletionStreamState:
    def __init__(
        self,
        request_id: str,
        model_name: str,
        prompt: str,
        tokenizer: Any,
        stream_queue: queue.Queue[dict[str, Any]],
        n: int,
    ) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.stream_queue = stream_queue
        self.num_tokens_input = len(tokenizer.encode(prompt))
        self.num_tokens_output = [0] * n
        self.finished = [False] * n
        self.completed = False
        self.closed = False
        self._pending_final_chunks: list[str] | None = None
        self._lock = threading.Lock()

    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            if not self.completed and all(self.finished):
                chunks.extend(self._final_chunks(max_items))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed and all(self.finished):
                        chunks.extend(self._final_chunks(max_items - len(chunks)))
                break

            with self._lock:
                chunks.extend(self._event_to_chunks(event, max_items - len(chunks)))
                if self.completed or self.closed:
                    break
        return chunks

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def _event_to_chunks(
        self, event: dict[str, Any], remaining_capacity: int
    ) -> list[str]:
        if remaining_capacity <= 0:
            return []
        if event.get("error"):
            if self._pending_final_chunks is None:
                self._pending_final_chunks = [
                    self._error_chunk(str(event["error"])),
                    STREAM_DONE_MESSAGE,
                ]
            return self._drain_pending_final_chunks(remaining_capacity)
        if event.get("done"):
            if not self.completed and all(self.finished):
                return self._final_chunks(remaining_capacity)
            return []

        index = int(event["index"])
        if self.finished[index]:
            return []

        extra_fields: dict[str, Any] = {}
        if "kv_transfer_params" in event:
            extra_fields["kv_transfer_params"] = event["kv_transfer_params"]

        self.num_tokens_output[index] += len(event.get("token_ids", []))
        chunks = [
            create_completion_chunk(
                self.request_id,
                self.model_name,
                event.get("text") or "",
                finish_reason=event.get("finish_reason"),
                index=index,
                **extra_fields,
            )
        ]

        if event.get("finished", False):
            self.finished[index] = True

        if all(self.finished) and len(chunks) < remaining_capacity:
            chunks.extend(self._final_chunks(remaining_capacity - len(chunks)))
        return chunks

    def _final_chunks(self, remaining_capacity: int) -> list[str]:
        if self._pending_final_chunks is None:
            chunks: list[str] = []
            for index in range(len(self.num_tokens_output)):
                chunks.append(
                    create_completion_chunk(
                        self.request_id,
                        self.model_name,
                        "",
                        finish_reason="stop",
                        index=index,
                    )
                )
            completion_tokens = sum(self.num_tokens_output)
            usage = {
                "prompt_tokens": self.num_tokens_input,
                "completion_tokens": completion_tokens,
                "total_tokens": self.num_tokens_input + completion_tokens,
            }
            if len(self.num_tokens_output) > 1:
                usage["num_choices"] = len(self.num_tokens_output)
            usage_chunk = {
                "id": self.request_id,
                "object": TEXT_COMPLETION_OBJECT,
                "created": int(time.time()),
                "model": self.model_name,
                "usage": usage,
            }
            chunks.append(f"data: {json.dumps(usage_chunk)}\n\n")
            chunks.append(STREAM_DONE_MESSAGE)
            self._pending_final_chunks = chunks

        return self._drain_pending_final_chunks(remaining_capacity)

    def _drain_pending_final_chunks(self, remaining_capacity: int) -> list[str]:
        if not self._pending_final_chunks or remaining_capacity <= 0:
            return []
        chunks = self._pending_final_chunks[:remaining_capacity]
        del self._pending_final_chunks[:remaining_capacity]
        if not self._pending_final_chunks:
            self.completed = True
        return chunks

    @staticmethod
    def _error_chunk(message: str) -> str:
        return f"data: {json.dumps({'error': {'message': message}})}\n\n"


class AtomStandaloneService:
    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        model_name: str,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.custom_message_encoder = load_custom_message_encoder(model_name)
        self.engine_service = AtomEngineService(engine, tokenizer)
        self._streams: dict[str, ChatCompletionStreamState] = {}
        self._completion_streams: dict[str, CompletionStreamState] = {}
        self._streams_lock = threading.Lock()

    def chat_completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        try:
            request_data = self._normalize_chat_request(request_data)
            self._validate_model_name(request_data.get("model"))

            if request_data.get("stream", False):
                raise NotImplementedError(
                    "Streaming chat completions are not implemented for ATOM standalone yet"
                )

            template_kwargs = dict(self.default_chat_template_kwargs)
            if request_data.get("chat_template_kwargs"):
                template_kwargs.update(request_data["chat_template_kwargs"])

            prompt = apply_chat_template(
                self.tokenizer,
                self.custom_message_encoder,
                [
                    self._chat_message_to_template_dict(msg)
                    for msg in self._get_chat_messages(request_data)
                ],
                tools=request_data.get("tools"),
                **template_kwargs,
            )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            if effective_n > 1:
                outputs = self.engine_service.generate(
                    prompt, sampling_params, request_id, effective_n
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                response = build_chat_response_multi(
                    request_id, self.model_name, outputs
                )
            else:
                outputs = self.engine_service.generate(
                    prompt, sampling_params, request_id, effective_n
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                final_output = outputs[0]
                response = build_chat_response(
                    request_id, self.model_name, final_output["text"], final_output
                )
            return self._json_safe(response.model_dump(exclude_none=True))
        except Exception:
            logger.exception("ATOM standalone chat_completions failed")
            raise

    def completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        try:
            request_data = dict(request_data)
            self._validate_model_name(request_data.get("model"))

            if request_data.get("stream", False):
                raise ValueError(
                    "Use start_completions_stream for streaming completions"
                )

            prompts = self._get_completion_prompts(request_data)
            if len(prompts) != 1:
                raise ValueError(
                    "ATOM standalone /v1/completions currently supports exactly one prompt per request"
                )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"cmpl-{uuid.uuid4().hex}"
            outputs = self.engine_service.generate(
                prompts[0],
                sampling_params,
                request_id,
                effective_n,
                kv_transfer_params=request_data.get("kv_transfer_params"),
            )
            if not outputs:
                raise RuntimeError("No output generated")
            if effective_n > 1:
                response = build_completion_response_multi(
                    request_id, self.model_name, outputs
                )
            else:
                response = build_completion_response(
                    request_id, self.model_name, outputs[0]
                )
            return self._json_safe(response.model_dump(exclude_none=True))
        except Exception:
            logger.exception("ATOM standalone completions failed")
            raise

    def start_completions_stream(self, request_data: dict[str, Any]) -> str:
        try:
            request_data = dict(request_data)
            self._validate_model_name(request_data.get("model"))

            prompts = self._get_completion_prompts(request_data)
            if len(prompts) != 1:
                raise ValueError(
                    "ATOM standalone /v1/completions currently supports exactly one prompt per request"
                )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"cmpl-{uuid.uuid4().hex}"
            prompt = prompts[0]
            stream_queue = self.engine_service.start_stream(
                prompt,
                sampling_params,
                request_id,
                effective_n,
                kv_transfer_params=request_data.get("kv_transfer_params"),
            )
            stream_state = CompletionStreamState(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                tokenizer=self.tokenizer,
                stream_queue=stream_queue,
                n=effective_n,
            )
            with self._streams_lock:
                self._completion_streams[request_id] = stream_state
            return request_id
        except Exception:
            logger.exception("ATOM standalone start_completions_stream failed")
            raise

    def drain_completions_stream(
        self,
        stream_id: str,
        max_items: int = 16,
        timeout: float = 0.05,
    ) -> list[str]:
        with self._streams_lock:
            stream_state = self._completion_streams.get(stream_id)
        if stream_state is None:
            return [STREAM_DONE_MESSAGE]

        chunks = stream_state.drain(max_items=max_items, timeout=timeout)
        if stream_state.completed or stream_state.closed:
            with self._streams_lock:
                self._completion_streams.pop(stream_id, None)
        return chunks

    def poll_completions_stream(
        self, stream_id: str, timeout: float = 1.0
    ) -> str | None:
        chunks = self.drain_completions_stream(
            stream_id,
            max_items=1,
            timeout=timeout,
        )
        if not chunks:
            return None
        return chunks[0]

    def close_completions_stream(self, stream_id: str) -> None:
        with self._streams_lock:
            stream_state = self._completion_streams.pop(stream_id, None)
        if stream_state is not None:
            stream_state.close()

    def start_chat_completions_stream(self, request_data: dict[str, Any]) -> str:
        try:
            request_data = self._normalize_chat_request(request_data)
            self._validate_model_name(request_data.get("model"))

            template_kwargs = dict(self.default_chat_template_kwargs)
            if request_data.get("chat_template_kwargs"):
                template_kwargs.update(request_data["chat_template_kwargs"])

            prompt = apply_chat_template(
                self.tokenizer,
                self.custom_message_encoder,
                [
                    self._chat_message_to_template_dict(msg)
                    for msg in self._get_chat_messages(request_data)
                ],
                tools=request_data.get("tools"),
                **template_kwargs,
            )

            effective_n = _coerce_n(
                request_data.get("n"),
                request_data.get("temperature", DEFAULT_TEMPERATURE),
            )
            sampling_params = self._build_sampling_params(request_data, effective_n)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            stream_queue = self.engine_service.start_stream(
                prompt,
                sampling_params,
                request_id,
                effective_n,
            )
            stream_state = ChatCompletionStreamState(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                tokenizer=self.tokenizer,
                stream_queue=stream_queue,
                n=effective_n,
            )
            with self._streams_lock:
                self._streams[request_id] = stream_state
            return request_id
        except Exception:
            logger.exception("ATOM standalone start_chat_completions_stream failed")
            raise

    def drain_chat_completions_stream(
        self,
        stream_id: str,
        max_items: int = 16,
        timeout: float = 0.05,
    ) -> list[str]:
        with self._streams_lock:
            stream_state = self._streams.get(stream_id)
        if stream_state is None:
            return [STREAM_DONE_MESSAGE]

        chunks = stream_state.drain(max_items=max_items, timeout=timeout)
        if stream_state.completed or stream_state.closed:
            with self._streams_lock:
                self._streams.pop(stream_id, None)
        return chunks

    def poll_chat_completions_stream(
        self, stream_id: str, timeout: float = 1.0
    ) -> str | None:
        chunks = self.drain_chat_completions_stream(
            stream_id,
            max_items=1,
            timeout=timeout,
        )
        if not chunks:
            return None
        return chunks[0]

    def close_chat_completions_stream(self, stream_id: str) -> None:
        with self._streams_lock:
            stream_state = self._streams.pop(stream_id, None)
        if stream_state is not None:
            stream_state.close()

    def close(self) -> None:
        with self._streams_lock:
            self._streams.clear()
            self._completion_streams.clear()
        if hasattr(self, "engine_service"):
            self.engine_service.close()
        if hasattr(self.engine, "close"):
            self.engine.close()

    @staticmethod
    def _normalize_chat_request(request_data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(request_data)
        if (
            normalized.get("max_tokens") is None
            and normalized.get("max_completion_tokens") is not None
        ):
            normalized["max_tokens"] = normalized["max_completion_tokens"]
        return normalized

    def _validate_model_name(self, request_model: str | None) -> None:
        if (
            request_model is not None
            and request_model != "unknown"
            and request_model != self.model_name
        ):
            raise ValueError(
                f"requested model `{request_model}` does not match loaded model `{self.model_name}`"
            )

    @staticmethod
    def _get_chat_messages(request_data: dict[str, Any]) -> list[dict[str, Any]]:
        messages = request_data.get("messages")
        if messages is None:
            messages = request_data.get("prompt")
        if messages is None:
            raise ValueError("Either 'messages' or 'prompt' field is required")
        return messages

    @staticmethod
    def _get_completion_prompts(request_data: dict[str, Any]) -> list[str]:
        prompt = request_data.get("prompt")
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            return prompt
        raise ValueError(
            "Completion request field 'prompt' must be a string or list of strings"
        )

    @staticmethod
    def _chat_message_to_template_dict(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif content is None:
            content = ""

        template_message = {
            "role": message.get("role"),
            "content": content,
        }
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
            if key in message:
                template_message[key] = message[key]
        return template_message

    @staticmethod
    def _request_field(request: Any, field: str, default: Any = None) -> Any:
        if isinstance(request, dict):
            value = request.get(field, default)
        else:
            value = getattr(request, field, default)
        return default if value is None else value

    def _build_sampling_params(
        self,
        request: dict[str, Any] | CompletionRequest,
        effective_n: int,
    ) -> SamplingParams:
        return _build_sampling_params(
            temperature=self._request_field(
                request, "temperature", DEFAULT_TEMPERATURE
            ),
            max_tokens=self._request_field(request, "max_tokens", DEFAULT_MAX_TOKENS),
            stop_strings=self._normalize_stop_strings(
                self._request_field(request, "stop")
            ),
            ignore_eos=self._request_field(request, "ignore_eos", False),
            top_k=self._request_field(request, "top_k", DEFAULT_TOP_K),
            top_p=self._request_field(request, "top_p", DEFAULT_TOP_P),
            n=effective_n,
        )

    @staticmethod
    def _normalize_stop_strings(stop: Any) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
            return stop
        raise ValueError("Request field 'stop' must be a string or list of strings")

    @staticmethod
    def _normalize_output(output: Any) -> dict[str, Any]:
        if isinstance(output, str):
            return {
                "text": output,
                "finish_reason": None,
                "num_tokens_input": 0,
                "num_tokens_output": 0,
                "ttft": 0.0,
                "tpot": 0.0,
                "latency": 0.0,
            }
        return AtomStandaloneService._json_safe(dict(output))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): AtomStandaloneService._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [AtomStandaloneService._json_safe(item) for item in value]
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            return AtomStandaloneService._json_safe(value.item())
        return value
