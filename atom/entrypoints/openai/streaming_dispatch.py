# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-thread dispatch and per-request delivery for streaming model output.

Two halves of one hand-off. :class:`StreamBatchDispatcher` runs on the engine
output threads: it buffers a whole engine step, detokenizes it, and schedules a
single callback per event loop. :class:`StreamOutputCollector` is the loop-side
landing point each stream's SSE generator reads from.
"""

import threading
from asyncio import AbstractEventLoop, Event
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# Fields a later chunk overrides on the one it merges into, when it has a value
# of its own. The SSE consumers keep the newest non-empty value they see, so
# merging this way hands them what reading each chunk separately would have.
_LATEST_WINS = ("finish_reason", "kv_transfer_params", "num_cached_tokens")


@dataclass
class IncrementalStreamDetokenizer:
    """Decode token deltas without emitting incomplete UTF-8 characters."""

    tokenizer: Any
    tokens: list[int] = field(default_factory=list)
    prefix_offset: int = 0
    read_offset: int = 0

    def update(self, token_ids: list[int], finished: bool) -> str:
        self.tokens.extend(token_ids)
        prefix_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.tokens[self.prefix_offset :],
            skip_special_tokens=True,
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            return delta
        if finished:
            return new_text[len(prefix_text) :]
        return ""


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


class StreamBatchDispatcher:
    """Collect one engine step per output thread and dispatch it by event loop.

    Holds no per-stream state. Each stream's detokenizer belongs to the engine
    callback that feeds it, and rides along on every chunk, so nothing here is
    shared between the output threads and the event loop and nothing has to be
    cleaned up: when the engine drops a finished stream's callback the
    detokenizer goes with it.

    It used to live in a dict here, which cost a lock -- 27% of the API
    server's CPU, since every buffered chunk looked its state up with all
    output threads contending -- and then, without the lock, cost an entry
    that two threads had to keep in agreement and that teardown had to
    remember to remove.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._thread_local = threading.local()

    def new_state(self) -> IncrementalStreamDetokenizer:
        """Make the detokenizer for one stream, for its callback to hold."""
        return IncrementalStreamDetokenizer(self.tokenizer)

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

        by_loop: dict[AbstractEventLoop, list[tuple[Any, Any]]] = {}
        for item in buf:
            item.chunk["text"] = item.state.update(
                item.chunk.get("token_ids") or [],
                bool(item.chunk.get("finished")),
            )
            payload = item.chunk if item.tag is None else (item.tag, item.chunk)
            by_loop.setdefault(item.loop, []).append((item.collector, payload))

        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._deliver, items)

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
