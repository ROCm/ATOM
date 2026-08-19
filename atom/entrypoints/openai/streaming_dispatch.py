# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-thread dispatch and per-request delivery for streaming model output.

Two halves of one hand-off. :class:`StreamBatchDispatcher` runs on the engine
output threads: it buffers a whole engine step, detokenizes it, and schedules a
single callback per event loop. :class:`StreamOutputCollector` is the loop-side
landing point each stream's SSE generator reads from.
"""

import itertools
import logging
import multiprocessing
import os
import queue as sync_queue
import threading
import traceback
from asyncio import AbstractEventLoop, Event
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# Fields a later chunk overrides on the one it merges into, when it has a value
# of its own. The SSE consumers keep the newest non-empty value they see, so
# merging this way hands them what reading each chunk separately would have.
_LATEST_WINS = ("finish_reason", "kv_transfer_params", "num_cached_tokens")

logger = logging.getLogger("atom.streaming_dispatch")

_DISABLE_TOKENIZER_BATCH_DECODE = (
    os.environ.get("ATOM_DISABLE_TOKENIZER_BATCH_DECODE", "0").lower()
    in {"1", "true", "yes"}
)
_DETOKENIZER_PROCESS_START_TIMEOUT_S = 120.0
_DETOKENIZER_PROCESS_COUNT = max(
    0, int(os.environ.get("ATOM_DETOKENIZER_PROCESS_COUNT", "0") or 0)
)


@dataclass
class IncrementalStreamDetokenizer:
    """Decode token deltas without emitting incomplete UTF-8 characters."""

    tokenizer: Any
    tokens: list[int] = field(default_factory=list)
    prefix_offset: int = 0
    read_offset: int = 0

    def prepare_update(self, token_ids: list[int]) -> tuple[list[int], list[int]]:
        """Append tokens and return the prefix/full windows to decode."""
        self.tokens.extend(token_ids)
        return (
            self.tokens[self.prefix_offset : self.read_offset],
            self.tokens[self.prefix_offset :],
        )

    def finish_update(
        self, prefix_text: str, new_text: str, finished: bool
    ) -> str:
        """Return the printable delta and advance incremental decode offsets."""
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            return delta
        if finished:
            return new_text[len(prefix_text) :]
        return ""

    def update(self, token_ids: list[int], finished: bool) -> str:
        prefix_ids, new_ids = self.prepare_update(token_ids)
        prefix_text = self.tokenizer.decode(
            prefix_ids,
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )
        return self.finish_update(prefix_text, new_text, finished)


def _can_batch_decode(tokenizer: Any, disable_batch_decode: bool) -> bool:
    return (
        not disable_batch_decode
        and getattr(tokenizer, "is_fast", True)
        and callable(getattr(tokenizer, "batch_decode", None))
    )


def _decode_token_batches(
    tokenizer: Any,
    token_batches: list[list[int]],
    *,
    use_batch_decode: bool,
) -> list[str]:
    """Decode non-empty rows, using one public tokenizer batch call."""
    texts = [""] * len(token_batches)
    non_empty = [
        (index, token_ids)
        for index, token_ids in enumerate(token_batches)
        if token_ids
    ]
    if not non_empty:
        return texts

    if use_batch_decode:
        decoded = tokenizer.batch_decode(
            [token_ids for _, token_ids in non_empty],
            skip_special_tokens=True,
        )
    else:
        decoded = [
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )
            for _, token_ids in non_empty
        ]

    if len(decoded) != len(non_empty):
        raise ValueError(
            "tokenizer returned an unexpected number of decoded strings"
        )
    for (index, _), text in zip(non_empty, decoded):
        texts[index] = text
    return texts


def _decode_state_updates(
    tokenizer: Any,
    updates: list[tuple[IncrementalStreamDetokenizer, list[int], bool]],
    *,
    use_batch_decode: bool,
) -> list[str]:
    prefix_batches: list[list[int]] = []
    new_batches: list[list[int]] = []
    for state, token_ids, _ in updates:
        prefix_ids, new_ids = state.prepare_update(token_ids)
        prefix_batches.append(prefix_ids)
        new_batches.append(new_ids)

    prefix_texts = _decode_token_batches(
        tokenizer,
        prefix_batches,
        use_batch_decode=use_batch_decode,
    )
    new_texts = _decode_token_batches(
        tokenizer,
        new_batches,
        use_batch_decode=use_batch_decode,
    )
    return [
        state.finish_update(prefix_text, new_text, finished)
        for (state, _, finished), prefix_text, new_text in zip(
            updates, prefix_texts, new_texts
        )
    ]


def _restore_decode_states(
    snapshots: list[tuple[IncrementalStreamDetokenizer, int, int, int]],
) -> None:
    for state, token_count, prefix_offset, read_offset in snapshots:
        del state.tokens[token_count:]
        state.prefix_offset = prefix_offset
        state.read_offset = read_offset


def _detokenizer_process_main(
    worker_id: int,
    input_queue: Any,
    output_queue: Any,
    tokenizer_model: str | None,
    trust_remote_code: bool,
    disable_batch_decode: bool,
    inherited_tokenizer: Any | None,
) -> None:
    """Own tokenizer and incremental states in a dedicated process."""
    try:
        if tokenizer_model is not None:
            from atom.model_engine.llm_engine import _load_tokenizer

            tokenizer = _load_tokenizer(tokenizer_model, trust_remote_code)
        elif inherited_tokenizer is not None:
            tokenizer = inherited_tokenizer
        else:
            raise ValueError("detokenizer process needs a tokenizer model")
        use_batch_decode = _can_batch_decode(
            tokenizer, disable_batch_decode
        )
        output_queue.put(
            ("ready", worker_id, os.getpid(), use_batch_decode)
        )
    except Exception:
        output_queue.put(("init_error", traceback.format_exc()))
        return

    states: dict[Hashable, IncrementalStreamDetokenizer] = {}
    while True:
        message = input_queue.get()
        kind = message[0]
        if kind == "shutdown":
            output_queue.put(("stopped", worker_id))
            return
        if kind == "discard":
            request_id = message[1]
            for key in list(states):
                if key == request_id or (
                    isinstance(key, tuple) and key and key[0] == request_id
                ):
                    states.pop(key, None)
            continue
        if kind != "decode":
            output_queue.put(
                ("worker_error", None, f"unknown message type: {kind}")
            )
            continue

        _, batch_id, rows = message
        updates = []
        state_rows = []
        for state_key, token_ids, finished in rows:
            state = states.get(state_key)
            if state is None:
                state = states.setdefault(
                    state_key, IncrementalStreamDetokenizer(tokenizer)
                )
            updates.append((state, token_ids, finished))
            state_rows.append((state_key, state, finished))

        snapshots = [
            (
                state,
                len(state.tokens),
                state.prefix_offset,
                state.read_offset,
            )
            for state, _, _ in updates
        ]
        try:
            texts = _decode_state_updates(
                tokenizer,
                updates,
                use_batch_decode=use_batch_decode,
            )
        except Exception:
            _restore_decode_states(snapshots)
            if use_batch_decode:
                try:
                    texts = _decode_state_updates(
                        tokenizer,
                        updates,
                        use_batch_decode=False,
                    )
                except Exception:
                    _restore_decode_states(snapshots)
                    output_queue.put(
                        ("decode_error", batch_id, traceback.format_exc())
                    )
                    continue
            else:
                output_queue.put(
                    ("decode_error", batch_id, traceback.format_exc())
                )
                continue

        for state_key, state, finished in state_rows:
            if finished and states.get(state_key) is state:
                states.pop(state_key, None)
        output_queue.put(("decoded", batch_id, texts))


def merge_chunk(into: dict, new: dict) -> None:
    """Fold ``new`` into the chunk already waiting. ``into`` is modified.

    ``text`` and ``token_ids`` are deltas, so concatenating them is exact.
    ``token_ids`` is rebuilt rather than extended: the first chunk's list is the
    engine's own ``output_tokens``, which must not be appended to.
    """
    into["token_ids"] = [*into.get("token_ids", ()), *new.get("token_ids", ())]
    into["text"] = into.get("text", "") + new.get("text", "")
    into["finished"] = bool(into.get("finished") or new.get("finished"))
    for key in _LATEST_WINS:
        if new.get(key):
            into[key] = new[key]


class StreamOutputCollector:
    """Per-request delivery point that merges chunks when the consumer lags.

    Replaces the unbounded ``asyncio.Queue`` that used to sit between the engine
    output threads and the SSE response generators. A queue hands over one item
    per ``get()``, so when the frontend cannot keep up with the GPU the backlog
    grows without bound and every queued item still costs its own coroutine
    wakeup, JSON encode and socket write. Here a stream holds at most one chunk:
    anything arriving behind an unread one merges into it.

    Nothing is ever held back. With a consumer that keeps up nothing ever
    merges, and delivery is identical to the queue this replaces. Merging only
    covers chunks that were already waiting, so a token is never delivered later
    than it would have been.

    ``tag`` is the fan-out sibling index (``SamplingParams.n>1``) or ``None`` for
    a plain single-sequence stream. Chunks merge per tag, so siblings never mix.
    """

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id
        self._pending: dict[Any, dict] = {}
        self._ready = Event()

    def put_nowait(self, payload: dict | tuple[int, dict]) -> None:
        """Accept one prepared chunk. Called on the event loop, never off it."""
        if type(payload) is tuple:
            tag, chunk = payload
        else:
            tag, chunk = None, payload
        waiting = self._pending.get(tag)
        if waiting is None:
            self._pending[tag] = chunk
        else:
            merge_chunk(waiting, chunk)
        self._ready.set()

    async def get(self) -> dict | tuple[int, dict]:
        """Await the next chunk, carrying whatever merged into it."""
        while not self._pending:
            await self._ready.wait()
        tag, chunk = next(iter(self._pending.items()))
        del self._pending[tag]
        if not self._pending:
            self._ready.clear()
        return chunk if tag is None else (tag, chunk)


class _BufferedChunk(NamedTuple):
    """One stream's chunk, waiting for the end of the current engine step."""

    loop: AbstractEventLoop
    collector: Any
    state: IncrementalStreamDetokenizer
    chunk: dict
    tag: int | None


class _DecodeWorkItem(NamedTuple):
    state_key: Hashable
    state: IncrementalStreamDetokenizer
    delivery: _BufferedChunk
    token_ids: list[int]
    finished: bool


class StreamBatchDispatcher:
    """Collect one engine step per output thread and dispatch it by event loop."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        disable_batch_decode: bool | None = None,
        tokenizer_model: str | None = None,
        trust_remote_code: bool = False,
        use_process: bool | None = None,
        process_count: int | None = None,
        process_start_method: str = "spawn",
    ):
        self.tokenizer = tokenizer
        if disable_batch_decode is None:
            disable_batch_decode = _DISABLE_TOKENIZER_BATCH_DECODE
        self._disable_batch_decode = disable_batch_decode
        self._use_batch_decode = _can_batch_decode(
            tokenizer, disable_batch_decode
        )
        self._thread_local = threading.local()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, list[_DecodeWorkItem]] = {}
        self._batch_ids = itertools.count()
        self._worker_ids = itertools.count()
        self._worker_assignment_lock = threading.Lock()
        self._processes = []
        self._process_inputs = []
        self._process_output = None
        self._result_thread = None
        self._closed = False
        if process_count is None:
            process_count = _DETOKENIZER_PROCESS_COUNT
        if use_process is None:
            use_process = process_count > 0
        elif use_process and process_count <= 0:
            process_count = 1
        self._use_process = use_process
        if use_process:
            self._process_count = max(1, process_count)
            self._start_detokenizer_process(
                tokenizer_model=tokenizer_model,
                trust_remote_code=trust_remote_code,
                process_start_method=process_start_method,
            )
        else:
            self._process_count = 0

    def new_state(self) -> IncrementalStreamDetokenizer:
        """Make the detokenizer for one stream, for its callback to hold."""
        return IncrementalStreamDetokenizer(self.tokenizer)

    def _state_key(self, item: _BufferedChunk) -> Hashable:
        request_id = getattr(item.collector, "request_id", "")
        return (request_id, item.tag, id(item.state))

    def _start_detokenizer_process(
        self,
        *,
        tokenizer_model: str | None,
        trust_remote_code: bool,
        process_start_method: str,
    ) -> None:
        ctx = multiprocessing.get_context(process_start_method)
        self._process_output = ctx.Queue(maxsize=256)
        inherited_tokenizer = None if tokenizer_model is not None else self.tokenizer
        for worker_id in range(self._process_count):
            process_input = ctx.Queue(maxsize=64)
            process = ctx.Process(
                target=_detokenizer_process_main,
                args=(
                    worker_id,
                    process_input,
                    self._process_output,
                    tokenizer_model,
                    trust_remote_code,
                    self._disable_batch_decode,
                    inherited_tokenizer,
                ),
                name=f"ATOMDetokenizer-{worker_id}",
                daemon=True,
            )
            process.start()
            self._process_inputs.append(process_input)
            self._processes.append(process)

        ready = {}
        try:
            for _ in self._processes:
                status = self._process_output.get(
                    timeout=_DETOKENIZER_PROCESS_START_TIMEOUT_S
                )
                if status[0] != "ready":
                    raise RuntimeError(
                        f"failed to start a detokenizer process: {status[1]}"
                    )
                ready[status[1]] = status
        except (sync_queue.Empty, RuntimeError) as exc:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
            raise RuntimeError(
                "failed to start the detokenizer process pool"
            ) from exc
        for worker_id in sorted(ready):
            status = ready[worker_id]
            logger.info(
                "Started independent detokenizer worker=%s pid=%s batch_decode=%s",
                worker_id,
                status[2],
                status[3],
            )
        self._result_thread = threading.Thread(
            target=self._receive_process_results,
            name="ATOMDetokenizerResults",
            daemon=True,
        )
        self._result_thread.start()

    def enqueue(
        self,
        *,
        loop: AbstractEventLoop,
        collector: Any,
        state: IncrementalStreamDetokenizer,
        chunk: dict,
        tag: int | None = None,
    ) -> None:
        """Buffer a raw chunk until the current engine step is flushed."""
        buf = getattr(self._thread_local, "buf", None)
        if buf is None:
            buf = self._thread_local.buf = []
        buf.append(_BufferedChunk(loop, collector, state, chunk, tag))

    def flush(self) -> None:
        """Detokenize buffered chunks and schedule one delivery per event loop."""
        tl = self._thread_local
        buf = getattr(tl, "buf", None)
        if not buf:
            return

        tl.buf = []

        groups: dict[Hashable, list[_BufferedChunk]] = {}
        for item in buf:
            groups.setdefault(self._state_key(item), []).append(item)

        work_items: list[_DecodeWorkItem] = []
        for state_key, items in groups.items():
            delivery = items[0]
            for item in items[1:]:
                merge_chunk(delivery.chunk, item.chunk)
            token_ids = delivery.chunk.get("token_ids") or []
            work_items.append(
                _DecodeWorkItem(
                    state_key=state_key,
                    state=delivery.state,
                    delivery=delivery,
                    token_ids=token_ids,
                    finished=bool(delivery.chunk.get("finished")),
                )
            )

        if self._use_process:
            self._submit_process_batch(work_items)
            return
        texts = self._decode_work_items(work_items)
        self._deliver_work_items(work_items, texts)

    def _submit_process_batch(
        self, work_items: list[_DecodeWorkItem]
    ) -> None:
        worker_id = getattr(
            self._thread_local, "detokenizer_worker_id", None
        )
        if worker_id is None:
            with self._worker_assignment_lock:
                worker_id = next(self._worker_ids) % self._process_count
            self._thread_local.detokenizer_worker_id = worker_id
        process = self._processes[worker_id]
        if not process.is_alive():
            raise RuntimeError(
                f"detokenizer process {worker_id} is not running"
            )
        batch_id = next(self._batch_ids)
        with self._pending_lock:
            self._pending[batch_id] = work_items
        rows = [
            (
                item.state_key,
                item.token_ids,
                item.finished,
            )
            for item in work_items
        ]
        try:
            self._process_inputs[worker_id].put(("decode", batch_id, rows))
        except Exception:
            with self._pending_lock:
                self._pending.pop(batch_id, None)
            raise

    def _receive_process_results(self) -> None:
        stopped_workers = set()
        while True:
            try:
                message = self._process_output.get(timeout=1)
            except sync_queue.Empty:
                dead_workers = [
                    worker_id
                    for worker_id, process in enumerate(self._processes)
                    if not process.is_alive()
                    and worker_id not in stopped_workers
                ]
                if dead_workers and not self._closed:
                    logger.error(
                        "Detokenizer processes exited unexpectedly: %s",
                        dead_workers,
                    )
                    with self._pending_lock:
                        pending = list(self._pending.values())
                        self._pending.clear()
                    for work_items in pending:
                        self._deliver_work_items(
                            work_items, [""] * len(work_items)
                        )
                    return
                continue
            kind = message[0]
            if kind == "stopped":
                stopped_workers.add(message[1])
                if len(stopped_workers) == self._process_count:
                    return
                continue
            if kind not in {"decoded", "decode_error"}:
                logger.error("Detokenizer process error: %s", message)
                continue
            batch_id = message[1]
            with self._pending_lock:
                work_items = self._pending.pop(batch_id, None)
            if work_items is None:
                continue
            if kind == "decoded":
                texts = message[2]
            else:
                logger.error(
                    "Detokenizer process failed batch %s:\n%s",
                    batch_id,
                    message[2],
                )
                texts = [""] * len(work_items)
            self._deliver_work_items(work_items, texts)

    def _deliver_work_items(
        self, work_items: list[_DecodeWorkItem], texts: list[str]
    ) -> None:
        if len(texts) != len(work_items):
            raise ValueError(
                "detokenizer returned an unexpected number of strings"
            )
        by_loop: dict[AbstractEventLoop, list[tuple[Any, Any]]] = {}
        for work_item, text in zip(work_items, texts):
            delivery = work_item.delivery
            delivery.chunk["text"] = text
            payload = (
                delivery.chunk
                if delivery.tag is None
                else (delivery.tag, delivery.chunk)
            )
            by_loop.setdefault(delivery.loop, []).append(
                (delivery.collector, payload)
            )

        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._deliver, items)

    def _decode_work_items(
        self, work_items: list[_DecodeWorkItem]
    ) -> list[str]:
        """Incrementally detokenize all streams in one tokenizer batch."""
        updates = [
            (item.state, item.token_ids, item.finished)
            for item in work_items
        ]
        return _decode_state_updates(
            self.tokenizer,
            updates,
            use_batch_decode=self._use_batch_decode,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._use_process:
            return
        for process_input, process in zip(
            self._process_inputs, self._processes
        ):
            if not process.is_alive():
                continue
            try:
                process_input.put(("shutdown",))
            except Exception:
                logger.exception("Failed to stop detokenizer process cleanly")
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        if self._result_thread is not None:
            self._result_thread.join(timeout=1)

    def discard_request(self, request_id: str) -> None:
        """Drop process-owned detokenizer state after request cleanup."""
        if not self._use_process:
            return
        for process_input, process in zip(
            self._process_inputs, self._processes
        ):
            if process.is_alive():
                process_input.put(("discard", request_id))

    @staticmethod
    def _deliver(items: list[tuple[Any, Any]]) -> None:
        """Run on the target event loop and hand a whole step to its collectors.

        A step is delivered in one callback, never split across loop iterations.
        Splitting was tried as a fairness measure -- deliver 128, re-arm the rest
        with call_soon -- and it silently corrupts streams: the output thread can
        schedule the next step's delivery in between, so a collector receives
        step N+1's chunk before step N's leftovers. Deltas then merge in the
        wrong order, and an end-of-stream that lands before a straggler is
        overwritten by it, hanging that client for good.
        """
        for collector, payload in items:
            collector.put_nowait(payload)
