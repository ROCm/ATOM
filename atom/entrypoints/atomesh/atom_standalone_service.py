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
    chat_template_source,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.protocol import (
    CHAT_COMPLETION_CHUNK_OBJECT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    STREAM_DONE_MESSAGE,
    TEXT_COMPLETION_OBJECT,
    CompletionRequest,
    openai_stop_reason,
    openai_stop_reason_with_calls,
)
from atom.entrypoints.openai.reasoning import (
    NO_REASONING,
    ReasoningChannel,
    prompt_starts_in_reasoning,
    template_opens_reasoning_implicitly,
    thinking_switched_off,
)
from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
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
from atom.entrypoints.openai.sse import data_frame
from atom.entrypoints.openai.tool_parser import ToolCallStreamParser
from atom.entrypoints.openai.tool_parser.registry import (
    forbids_tool_calls,
    resolve_tool_call_parser,
)
from atom.model_engine.sequence import new_token_ids

logger = logging.getLogger("atom")


@dataclasses.dataclass
class EngineRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    future: concurrent.futures.Future[list[dict[str, Any]]]
    kv_transfer_params: dict[str, Any] | None = None
    data_parallel_rank: int | None = None


@dataclasses.dataclass
class EngineStreamRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    effective_n: int
    stream_queue: queue.Queue[dict[str, Any]]
    kv_transfer_params: dict[str, Any] | None = None
    data_parallel_rank: int | None = None


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
        # The engine sequences behind each open stream, so closing one can
        # stop it. Nothing here used to: `close_*_stream` popped the Python
        # state and the sequence kept decoding to `max_tokens` on the GPU,
        # putting into a queue no one would ever read again and holding its
        # KV blocks for the life of the process. The OpenAI server does this
        # from a generator's `finally`; this service is polled, so it has to
        # be done where the caller says it is done.
        self._stream_seqs: dict[str, list[Any]] = {}
        # Streams closed before the worker thread got round to submitting
        # them. `start_stream` only enqueues; the sequences do not exist until
        # `_submit_stream_request` runs, so a close in between had nothing to
        # abort and the sequence was added to the engine *afterwards* -- the
        # exact case the abort exists for, and the one it missed.
        self._abandoned: set[str] = set()
        # Stream ids handed to the worker queue and not yet published or
        # forgotten -- the only ids for which "no sequence list" can still
        # mean "not submitted yet" rather than "already finished". Bounded by
        # requests actually in flight.
        self._awaiting_submit: set[str] = set()
        self._stream_seqs_lock = threading.Lock()
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
        data_parallel_rank: int | None = None,
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
                data_parallel_rank=data_parallel_rank,
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
                    self._close_submit_window(request.request_id)
                else:
                    self._set_future_exception(
                        request.future,
                        RuntimeError("ATOM standalone engine service is closed"),
                    )
                continue

            try:
                if isinstance(request, EngineStreamRequest):
                    try:
                        self._submit_stream_request(request)
                    finally:
                        self._close_submit_window(request.request_id)
                else:
                    self._submit_request(request)
            except Exception as error:
                # Broad on purpose: this is the worker thread's outer loop, and
                # anything that escapes here kills it and hangs every request
                # after this one. The client learns what happened either way;
                # the server did not, so a failed request left no trace on the
                # side that could act on it.
                logger.exception(
                    "standalone engine request %s failed", request.request_id
                )
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
        data_parallel_rank: int | None = None,
    ) -> queue.Queue[dict[str, Any]]:
        if self._closed.is_set():
            raise RuntimeError("ATOM standalone engine service is closed")

        stream_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._expect_stream(request_id)
        self._queue.put(
            EngineStreamRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                effective_n=effective_n,
                stream_queue=stream_queue,
                kv_transfer_params=kv_transfer_params,
                data_parallel_rank=data_parallel_rank,
            )
        )
        return stream_queue

    def _expect_stream(self, request_id: str) -> None:
        """This id is on the worker queue and its sequences do not exist yet.

        The one registrar, so that "could still be submitted" has a single
        definition; the window closes in `_submit_stream_request`.
        """
        with self._stream_seqs_lock:
            self._awaiting_submit.add(request_id)

    def _close_submit_window(self, request_id: str) -> None:
        """This id can no longer be submitted, however that came about.

        The counterpart to `_expect_stream`, and called from the worker loop
        rather than from inside `_submit_stream_request` because that function
        has four ways out -- abandoned at the top, abandoned mid-preprocess,
        published, or raising -- and bookkeeping spread over them discarded on
        two. A stream closed while it was preprocessing, and one whose
        preprocess raised, each left an id behind for the life of the process:
        the same leak `_awaiting_submit` was added to remove, one level down.

        It runs *after* the submit, never before, because the window has to
        stay open for an abort arriving while the sequences are still being
        built -- that abort has nothing to stop yet either.
        """
        with self._stream_seqs_lock:
            self._awaiting_submit.discard(request_id)

    def _submit_stream_request(self, request: EngineStreamRequest) -> None:
        if self._take_abandoned(request.request_id):
            return
        if request.effective_n > 1:
            seqs = self._preprocess_fanout_stream_request(request)
        else:
            seqs = [self._preprocess_single_stream_request(request)]
        with self._stream_seqs_lock:
            # Checked again: the close may have arrived while this was
            # preprocessing. Sequences built and never added to the engine
            # cost nothing to drop -- there is no abort to do.
            if request.request_id in self._abandoned:
                self._abandoned.discard(request.request_id)
                return
        self.engine.core_mgr.add_request(seqs)
        # Published *after* the engine has them, and the close re-checked
        # once more. Publishing first left a window where `abort_stream`
        # popped the list and aborted sequences the engine had not been told
        # about -- the abort raised into a swallowed `except`, the worker then
        # added them, and they decoded to `max_tokens` into a queue nobody
        # would read. Which side of `add_request` the publish sits on decides
        # whether a close in that window aborts or leaks; this way one of the
        # two branches below always fires.
        with self._stream_seqs_lock:
            if request.request_id in self._abandoned:
                self._abandoned.discard(request.request_id)
                abandoned = seqs
            else:
                self._stream_seqs[request.request_id] = list(seqs)
                abandoned = ()
        for seq in abandoned:
            self._abort_seq(seq)

    def _take_abandoned(self, request_id: str) -> bool:
        with self._stream_seqs_lock:
            if request_id not in self._abandoned:
                # Still to be submitted, so the window `_awaiting_submit`
                # marks stays open: an abort arriving while this request is
                # being preprocessed has no sequences to stop yet either, and
                # closing the window here let it through to the engine.
                return False
            self._abandoned.discard(request_id)
            return True

    def forget_stream(self, request_id: str) -> None:
        """Drop what is remembered about a stream that ended on its own.

        Without this the sequence list for every completed stream stays for
        the life of the process; `abort_stream` only removes the ones a caller
        closes, and the engine has already forgotten those sequences by then
        anyway.
        """
        with self._stream_seqs_lock:
            self._stream_seqs.pop(request_id, None)
            self._abandoned.discard(request_id)
            self._awaiting_submit.discard(request_id)

    def abort_stream(self, request_id: str) -> None:
        """Stop the sequences behind a stream that is still live.

        Call this only for a stream the caller has just taken ownership of --
        `close_*_stream` does it under the same lock that removes the stream
        from the registry. It cannot be used as a blanket "make sure this is
        gone", because a missing sequence list is ambiguous: it means either
        "not submitted yet" or "already finished and forgotten", and this has
        to assume the first or the pre-submit abort window reopens. Calling it
        on a stream that already ended therefore remembers that id forever.
        """
        with self._stream_seqs_lock:
            seqs = self._stream_seqs.pop(request_id, None)
            if seqs is None:
                # The ambiguity the docstring names: `_awaiting_submit` is the
                # only thing in here that can say "not submitted yet" rather
                # than "already finished", and remembering a finished id leaks.
                if request_id in self._awaiting_submit:
                    self._abandoned.add(request_id)
                return
        for seq in seqs:
            self._abort_seq(seq)

    def _abort_seq(self, seq: Any) -> None:
        seq_id = getattr(seq, "id", None)
        if seq_id is None:
            return
        try:
            self.engine.core_mgr.abort_request(seq_id)
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            logger.debug("abort_request(%s): %s", seq_id, exc)
        requests = getattr(self.engine.io_processor, "requests", None)
        if requests is not None:
            requests.pop(seq_id, None)

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
            data_parallel_rank=request.data_parallel_rank,
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
            data_parallel_rank=request.data_parallel_rank,
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
            data_parallel_rank=request.data_parallel_rank,
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
            data_parallel_rank=request.data_parallel_rank,
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
        # An array for the same reason the engine's `token_ids` is one. It
        # stays one: this dict goes to `build_*_response`, which reads
        # `text` and the counters and never `token_ids`.
        self.token_ids = new_token_ids()
        self.finish_reason: str | None = None
        # Set with `finish_reason` when a stop string ended the request: which
        # one, and where the text has to be cut. -1 leaves the text alone.
        self.stop_reason: str | None = None
        self.stop_truncate_to: int = -1
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
                self.stop_reason = request_output.stop_reason
                self.stop_truncate_to = request_output.stop_truncate_to
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
        text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        # A stop string is defined over the text, so the cut is applied here
        # and not to the token ids -- the token carrying the match can hold
        # characters on both sides of it.
        if self.stop_truncate_to >= 0:
            text = text[: self.stop_truncate_to]
        output = {
            "text": text,
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
        self.per_tokens = [new_token_ids() for _ in range(n)]
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
        tool_parser_cls: type | None = None,
        reasoning: ReasoningChannel = NO_REASONING,
        tools: list | None = None,
        tool_choice: Any = None,
    ) -> None:
        self.request_id = request_id
        self.model_name = model_name
        self.stream_queue = stream_queue
        self.num_tokens_input = len(tokenizer.encode(prompt))
        self.num_tokens_output = [0] * n
        # Same prompt for every sibling, so the same starting state. Without
        # this a template that opens the reasoning channel in the prompt has
        # its whole trace delivered as the answer: state 0 no longer infers
        # reasoning from a bare end marker, because inferring it means waiting
        # for one.
        self.reasoning_filters = [reasoning.stream() for _ in range(n)]
        # `tool_choice="none"` forbids tool calls on this path too, and
        # `tools` is what type-coerces the arguments. The OpenAI server gated
        # and passed both; this one had neither, so the same request answered
        # differently depending on the entrypoint.
        self.tool_choice = tool_choice
        self.tool_parsers = [
            ToolCallStreamParser(
                tools=tools,
                parser_cls=tool_parser_cls,
                suppress_calls=forbids_tool_calls(tool_choice),
            )
            for _ in range(n)
        ]
        self.has_tool_calls = [False] * n
        self.finished = [False] * n
        self.role_sent = [False] * n
        self.completed = False
        self.closed = False
        self._pending_final_chunks: list[str] | None = None
        # Chunks built but not yet handed out. The drain takes `max_items` at
        # a time and this used to `break` out of the build loop when it hit
        # that -- discarding every event past the first, permanently, since
        # the parser had already yielded them. At `max_items=1` a tool call
        # lost its arguments and reported `finish_reason: stop`.
        self._pending_chunks: list[str] = []
        # The engine's own reason per sibling; see serving_chat.
        self.engine_finish_reasons: list[str | None] = [None] * n
        self._lock = threading.Lock()

    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            self._append_initial_role_chunks(chunks, max_items)
            if not self.completed:
                chunks.extend(self._hand_out(max_items - len(chunks)))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed:
                        chunks.extend(self._hand_out(max_items - len(chunks)))
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
                        delta={"role": "assistant", "content": ""},
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
            return self._hand_out(remaining_capacity)
        if event.get("done"):
            return [] if self.completed else self._hand_out(remaining_capacity)

        index = int(event["index"])
        if self.finished[index]:
            # Still hand out whatever is queued, and the final chunks once it
            # is empty -- returning early here left a fully drained stream
            # with no `finish_reason`, no usage and no `[DONE]`.
            return self._hand_out(remaining_capacity)

        chunks: list[str] = []
        text = event.get("text") or ""
        self.num_tokens_output[index] += len(event.get("token_ids", []))
        if event.get("finish_reason"):
            self.engine_finish_reasons[index] = event["finish_reason"]

        segments = self.reasoning_filters[index].process(text)
        if event.get("finished", False):
            segments.extend(self.reasoning_filters[index].flush())

        for field, segment_text in segments:
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
                        chunks.append(
                            create_chat_chunk(
                                self.request_id,
                                self.model_name,
                                delta={"tool_calls": [data]},
                                index=index,
                            )
                        )
                    elif event_type == "tool_call_args":
                        self.has_tool_calls[index] = True
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
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
                elif event_type == "tool_call_args":
                    self.has_tool_calls[index] = True
                    chunks.append(
                        create_chat_chunk(
                            self.request_id,
                            self.model_name,
                            delta={"tool_calls": [data]},
                            index=index,
                        )
                    )
            self.finished[index] = True

        self._pending_chunks.extend(chunks)
        return self._hand_out(remaining_capacity)

    def _build_final_chunks(self) -> list[str]:
        """The finish/usage/[DONE] tail, built once. Only `_hand_out` calls
        this, and only once nothing built is still queued."""
        chunks: list[str] = []
        for index, has_tool_calls in enumerate(self.has_tool_calls):
            finish_reason = openai_stop_reason_with_calls(
                self.engine_finish_reasons[index], has_tool_calls
            )
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
        chunks.append(data_frame(usage_chunk))
        chunks.append(STREAM_DONE_MESSAGE)
        return chunks

    def _take_pending(self, remaining_capacity: int) -> list[str]:
        """As many built chunks as the caller has room for; the rest wait."""
        if remaining_capacity <= 0:
            return []
        out = self._pending_chunks[:remaining_capacity]
        del self._pending_chunks[:remaining_capacity]
        return out

    def _hand_out(self, remaining_capacity: int) -> list[str]:
        """The one exit. Queued chunks first; terminal ones only after them.

        Every producer returns through here, and that is the whole of the
        rule: "every sibling finished" is not the same fact as "everything
        built has been sent", and three places used the first as a proxy for
        the second. The terminal chunks were then handed out while built
        chunks were still queued -- and once the finals go, `completed` is set
        and the caller pops the state, so the queued ones are gone. Measured
        at the production `max_items`: a response with nine tool calls
        delivered eight and reported `finish_reason: tool_calls`; at
        `max_items=1` a single call reported `tool_calls` having sent no
        `tool_calls` delta at all.
        """
        out = self._take_pending(remaining_capacity)
        # `_take_pending` never returns more than the caller asked for, so no
        # room left means chunks are still queued -- there is nothing to test
        # for beyond that, and a second condition here would be a guard that
        # cannot fail.
        if len(out) >= remaining_capacity:
            return out
        if self._pending_final_chunks is None:
            if not all(self.finished):
                return out
            self._pending_final_chunks = self._build_final_chunks()
        return out + self._drain_pending_final_chunks(remaining_capacity - len(out))

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
        # Chunks built but not yet handed out. The drain takes `max_items` at
        # a time and this used to `break` out of the build loop when it hit
        # that -- discarding every event past the first, permanently, since
        # the parser had already yielded them. At `max_items=1` a tool call
        # lost its arguments and reported `finish_reason: stop`.
        self._pending_chunks: list[str] = []
        # The engine's own reason per sibling; see serving_chat.
        self.engine_finish_reasons: list[str | None] = [None] * n
        self._lock = threading.Lock()

    def drain(self, max_items: int = 16, timeout: float = 0.05) -> list[str]:
        max_items = max(1, int(max_items))
        chunks: list[str] = []
        with self._lock:
            if not self.completed:
                chunks.extend(self._hand_out(max_items - len(chunks)))
            if chunks or self.completed or self.closed:
                return chunks

        while len(chunks) < max_items:
            try:
                event = self.stream_queue.get(timeout=timeout if not chunks else 0.0)
            except queue.Empty:
                with self._lock:
                    if not self.completed:
                        chunks.extend(self._hand_out(max_items - len(chunks)))
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
            return self._hand_out(remaining_capacity)
        if event.get("done"):
            return [] if self.completed else self._hand_out(remaining_capacity)

        index = int(event["index"])
        if self.finished[index]:
            # Still hand out whatever is queued, and the final chunks once it
            # is empty -- returning early here left a fully drained stream
            # with no `finish_reason`, no usage and no `[DONE]`.
            return self._hand_out(remaining_capacity)

        extra_fields: dict[str, Any] = {}
        if "kv_transfer_params" in event:
            extra_fields["kv_transfer_params"] = event["kv_transfer_params"]

        self.num_tokens_output[index] += len(event.get("token_ids", []))
        if event.get("finish_reason"):
            self.engine_finish_reasons[index] = event["finish_reason"]
        chunks = [
            create_completion_chunk(
                self.request_id,
                self.model_name,
                event.get("text") or "",
                # The reason goes on the final chunk, as on every other
                # streaming path here and in the OpenAI server. Put here too
                # it was never translated, and `engine_finish_reasons` --
                # which exists to carry it to the terminal chunk -- was never
                # written at all.
                finish_reason=None,
                index=index,
                **extra_fields,
            )
        ]

        if event.get("finished", False):
            self.finished[index] = True

        self._pending_chunks.extend(chunks)
        return self._hand_out(remaining_capacity)

    def _build_final_chunks(self) -> list[str]:
        """The finish/usage/[DONE] tail, built once. Only `_hand_out` calls
        this, and only once nothing built is still queued."""
        chunks: list[str] = []
        for index in range(len(self.num_tokens_output)):
            # The engine's own reason, as the chat path reports it. This
            # was recorded per sibling and then never read: every
            # completion said `stop`, including one the engine cut off at
            # `max_tokens`, which `stream=false` reported as `length`.
            chunks.append(
                create_completion_chunk(
                    self.request_id,
                    self.model_name,
                    "",
                    finish_reason=(
                        openai_stop_reason(self.engine_finish_reasons[index]) or "stop"
                    ),
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
        chunks.append(data_frame(usage_chunk))
        chunks.append(STREAM_DONE_MESSAGE)
        return chunks

    def _take_pending(self, remaining_capacity: int) -> list[str]:
        """As many built chunks as the caller has room for; the rest wait."""
        if remaining_capacity <= 0:
            return []
        out = self._pending_chunks[:remaining_capacity]
        del self._pending_chunks[:remaining_capacity]
        return out

    def _hand_out(self, remaining_capacity: int) -> list[str]:
        """The one exit. Queued chunks first; terminal ones only after them.

        Every producer returns through here, and that is the whole of the
        rule: "every sibling finished" is not the same fact as "everything
        built has been sent", and three places used the first as a proxy for
        the second. The terminal chunks were then handed out while built
        chunks were still queued -- and once the finals go, `completed` is set
        and the caller pops the state, so the queued ones are gone. Measured
        at the production `max_items`: a response with nine tool calls
        delivered eight and reported `finish_reason: tool_calls`; at
        `max_items=1` a single call reported `tool_calls` having sent no
        `tool_calls` delta at all.
        """
        out = self._take_pending(remaining_capacity)
        # `_take_pending` never returns more than the caller asked for, so no
        # room left means chunks are still queued -- there is nothing to test
        # for beyond that, and a second condition here would be a guard that
        # cannot fail.
        if len(out) >= remaining_capacity:
            return out
        if self._pending_final_chunks is None:
            if not all(self.finished):
                return out
            self._pending_final_chunks = self._build_final_chunks()
        return out + self._drain_pending_final_chunks(remaining_capacity - len(out))

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
        tool_call_parser: str | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.custom_message_encoder = load_custom_message_encoder(model_name)
        # Resolved once here, for the same reason the OpenAI server resolves it
        # once: the chat template says which format this model emits, and
        # deciding from the output instead means deciding from a prefix.
        _template_source = chat_template_source(tokenizer, self.custom_message_encoder)
        # Which dialect this model's reasoning is written in, once, from the
        # same evidence and by the same function the OpenAI server uses. Both
        # paths through this service then read it the same way; they used to
        # share only a boolean, and the streaming half closed the channel on
        # any registered dialect's marker.
        self.reasoning_dialect, _stated = resolve_dialect(
            _template_source,
            render_probe_prompt(tokenizer, self.custom_message_encoder, tools=False)
            or "",
        )
        self.model_starts_in_reasoning = template_opens_reasoning_implicitly(
            _template_source
        )
        # The kwarg that turns this model's reasoning off, so
        # `thinking_switched_off` can tell a switch the template reads from
        # one it ignores.
        self.reasoning_toggle = resolve_reasoning_toggle(
            tokenizer, self.custom_message_encoder
        )
        self.tool_parser_cls = resolve_tool_call_parser(
            tool_call_parser, tokenizer, self.custom_message_encoder, model=model_name
        )

        self.engine_service = AtomEngineService(engine, tokenizer)

        self._streams: dict[str, ChatCompletionStreamState] = {}
        self._completion_streams: dict[str, CompletionStreamState] = {}
        self._streams_lock = threading.Lock()

    def _reasoning_channel(
        self, prompt: str, template_kwargs: dict[str, Any] | None = None
    ) -> ReasoningChannel:
        """How to read this response's reasoning channel.

        One function, read by both delivery modes, so they cannot be given
        different dialects -- which is what they were: the non-streaming split
        tried each registered dialect in turn and the streaming filter closed
        on the union of all their end markers.

        `model_starts_in_reasoning` describes the template with reasoning
        *on*, so it only holds while reasoning is on. This service takes no
        `thinking` field, but it does merge `--default-chat-template-kwargs`
        and the request's own `chat_template_kwargs` into the render -- so an
        operator or a client can set the template's switch directly, and then
        the rendered prompt does not open the channel. OR-ing the model-level
        fact in regardless made a DeepSeek-R1-shaped model return its whole
        answer as `reasoning_content` with `content` empty, on both delivery
        modes.
        """
        return ReasoningChannel(
            dialect=self.reasoning_dialect,
            starts_open=prompt_starts_in_reasoning(prompt)
            or (
                self.model_starts_in_reasoning
                and not thinking_switched_off(template_kwargs, self.reasoning_toggle)
            ),
        )

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
            data_parallel_rank = self._get_data_parallel_rank(request_data)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            if effective_n > 1:
                outputs = self.engine_service.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    effective_n,
                    data_parallel_rank=data_parallel_rank,
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                response = build_chat_response_multi(
                    request_id,
                    self.model_name,
                    outputs,
                    reasoning=self._reasoning_channel(prompt, template_kwargs),
                    tools=request_data.get("tools"),
                    tool_choice=request_data.get("tool_choice"),
                    tool_parser_cls=self.tool_parser_cls,
                )
            else:
                outputs = self.engine_service.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    effective_n,
                    data_parallel_rank=data_parallel_rank,
                )
                if not outputs:
                    raise RuntimeError("No output generated")
                final_output = outputs[0]
                response = build_chat_response(
                    request_id,
                    self.model_name,
                    final_output["text"],
                    final_output,
                    reasoning=self._reasoning_channel(prompt, template_kwargs),
                    tools=request_data.get("tools"),
                    tool_choice=request_data.get("tool_choice"),
                    tool_parser_cls=self.tool_parser_cls,
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
                data_parallel_rank=self._get_data_parallel_rank(request_data),
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
                data_parallel_rank=self._get_data_parallel_rank(request_data),
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
            # A stream that ends on its own is never closed by the caller, so
            # this is the only place its sequence list is dropped. Without it
            # the engine service remembers every completed stream for the life
            # of the process.
            self.engine_service.forget_stream(stream_id)
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
        """Close a stream the caller is finished with.

        The abort is conditional on this call being the one that removed the
        stream. A stream that ended on its own was already removed by
        `drain_*`, which called `forget_stream`; the router then closes it
        anyway, as it is documented to. Aborting unconditionally at that point
        reached `abort_stream` with nothing left to pop, which reads as "not
        submitted yet" and permanently remembered the id -- one leaked entry
        per *successful* request, which is the same unbounded growth
        `forget_stream` exists to prevent.

        `_completion_streams` is the liveness answer and it is exact: the entry is
        added before the engine request is submitted and removed either here
        or by the drain that saw the stream finish. So a stream closed during
        the pre-submit window still aborts, which is the case `_abandoned` was
        built for.
        """
        with self._streams_lock:
            stream_state = self._completion_streams.pop(stream_id, None)
        if stream_state is None:
            return
        stream_state.close()
        self.engine_service.abort_stream(stream_id)

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
                data_parallel_rank=self._get_data_parallel_rank(request_data),
            )
            stream_state = ChatCompletionStreamState(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                tokenizer=self.tokenizer,
                stream_queue=stream_queue,
                n=effective_n,
                tool_parser_cls=self.tool_parser_cls,
                reasoning=self._reasoning_channel(prompt, template_kwargs),
                tools=request_data.get("tools"),
                tool_choice=request_data.get("tool_choice"),
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
            # A stream that ends on its own is never closed by the caller, so
            # this is the only place its sequence list is dropped. Without it
            # the engine service remembers every completed stream for the life
            # of the process.
            self.engine_service.forget_stream(stream_id)
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
        """Close a stream the caller is finished with.

        Identical to `close_completions_stream` over `_streams` instead of
        `_completion_streams` -- see there for why the abort is conditional on
        this call being the one that removed the stream.
        """
        with self._streams_lock:
            stream_state = self._streams.pop(stream_id, None)
        if stream_state is None:
            return
        stream_state.close()
        self.engine_service.abort_stream(stream_id)

    def close(self) -> None:
        """Shut down, in the order `close_*_stream` uses.

        `stream_state.close()` is the only thing that sets `closed`, and a
        router thread already inside `drain()` does not hold `_streams_lock`:
        it wakes from its queue, reads `closed` as False and hands out chunks
        for a sequence this method just aborted -- and if `all(finished)`
        happens to hold, a finish/usage/`[DONE]` tail as well, so the caller
        records `finish_reason: stop` on a truncated answer. Clearing the
        registry is not what stops a drain; closing the state is.
        """
        with self._streams_lock:
            open_streams = [
                *self._streams.items(),
                *self._completion_streams.items(),
            ]
            self._streams.clear()
            self._completion_streams.clear()
        for stream_id, stream_state in open_streams:
            stream_state.close()
            self.engine_service.abort_stream(stream_id)
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

    @staticmethod
    def _get_data_parallel_rank(request_data: dict[str, Any]) -> int | None:
        """Extract the DP-attention rank, if available.

        Routers can inject a ``data_parallel_rank`` field into the request body to
        indicate which DP rank to route to. Falls back to round-robin if not available.
        """
        raw = request_data.get("data_parallel_rank")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"data_parallel_rank must be an integer, got {raw!r}")

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
