# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Batched cross-thread dispatch for streaming model output."""

import os
import threading
import time
from asyncio import AbstractEventLoop, Queue
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

# SSE coalescing. Two tokenizer.decode calls, a queue put/get, a json.dumps
# and a socket write are paid once per (request, engine step); at high
# concurrency that fixed cost, not the GPU, is what caps throughput. Holding
# the buffer open across steps collapses N tokens into one of each (update()
# takes a list and decodes twice regardless of length). Costs up to this much
# extra inter-token latency, so it is opt-in. A finished chunk always flushes.
_COALESCE_S = max(0.0, float(os.environ.get("ATOM_SSE_COALESCE_MS", "0") or 0)) / 1000.0
_COALESCE_MAX_STEPS = int(os.environ.get("ATOM_SSE_COALESCE_MAX_STEPS", "8") or 8)


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


@dataclass
class _BufferedChunk:
    loop: AbstractEventLoop
    queue: Queue
    state_key: Hashable
    chunk: dict
    tag: int | None


class StreamBatchDispatcher:
    """Collect one engine step per output thread and dispatch it by event loop."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._thread_local = threading.local()
        self._states: dict[Hashable, IncrementalStreamDetokenizer] = {}

    def enqueue(
        self,
        *,
        loop: AbstractEventLoop,
        queue: Queue,
        state_key: Hashable,
        chunk: dict,
        tag: int | None = None,
    ) -> None:
        """Buffer a raw chunk until the current engine step is flushed."""
        buf = getattr(self._thread_local, "buf", None)
        if buf is None:
            buf = self._thread_local.buf = []
        buf.append(
            _BufferedChunk(
                loop=loop,
                queue=queue,
                state_key=state_key,
                chunk=chunk,
                tag=tag,
            )
        )

    def flush(self) -> None:
        """Detokenize buffered chunks and schedule one drain per event loop."""
        tl = self._thread_local
        buf = getattr(tl, "buf", None)
        if not buf:
            return

        now = time.monotonic()
        if _COALESCE_S > 0:
            steps = getattr(tl, "steps", 0) + 1
            tl.steps = steps
            if (
                steps < _COALESCE_MAX_STEPS
                and now - getattr(tl, "last_flush", 0.0) < _COALESCE_S
                and not any(i.chunk.get("finished") for i in buf)
            ):
                return  # keep accumulating; nothing here is final yet
        tl.buf = []
        tl.steps = 0
        tl.last_flush = now

        # One group per stream. state_key already distinguishes fan-out
        # siblings, and loop/queue are constant per stream, so the last item
        # of a group carries the right destination plus the terminal
        # finish_reason/finished flags.
        groups: dict[Hashable, list[_BufferedChunk]] = {}
        for item in buf:
            groups.setdefault(item.state_key, []).append(item)

        by_loop: dict[AbstractEventLoop, list[tuple[Queue, Any]]] = {}
        for state_key, items in groups.items():
            last = items[-1]
            token_ids = last.chunk.get("token_ids") or []
            if len(items) > 1:
                token_ids = []
                for i in items:
                    token_ids.extend(i.chunk.get("token_ids") or [])
                last.chunk["token_ids"] = token_ids
                for i in items[:-1]:
                    if "kv_transfer_params" in i.chunk:
                        last.chunk.setdefault(
                            "kv_transfer_params", i.chunk["kv_transfer_params"]
                        )

            state = self._get_state(state_key)
            last.chunk["text"] = state.update(
                token_ids, bool(last.chunk.get("finished"))
            )
            if last.chunk.get("finished"):
                self._drop_state(state_key, state)

            payload = last.chunk if last.tag is None else (last.tag, last.chunk)
            by_loop.setdefault(last.loop, []).append((last.queue, payload))

        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._drain_into_queues, items)

    # These three ran under a shared lock, which cost 27% of the API server's
    # CPU -- _get_state is called once per stream per flush, with all output
    # threads contending. No lock is needed: each operation below is a single
    # C-level dict method that no other Python thread can interrupt, and
    # list() snapshots atomically so discard_request cannot hit "dict changed
    # size". GIL-dependent; a free-threaded build would need real locks.

    def discard_request(self, request_id: str) -> None:
        """Drop direct and fan-out detokenizer state after request cleanup."""
        for key in list(self._states):
            if key == request_id or (
                isinstance(key, tuple) and key and key[0] == request_id
            ):
                self._states.pop(key, None)

    def _get_state(self, state_key: Hashable) -> IncrementalStreamDetokenizer:
        state = self._states.get(state_key)
        if state is None:
            state = self._states.setdefault(
                state_key, IncrementalStreamDetokenizer(self.tokenizer)
            )
        return state

    def _drop_state(
        self, state_key: Hashable, state: IncrementalStreamDetokenizer
    ) -> None:
        if self._states.get(state_key) is state:
            self._states.pop(state_key, None)

    @staticmethod
    def _drain_into_queues(items: list[tuple[Queue, Any]]) -> None:
        """Run on the target event loop and deliver each prepared payload."""
        for queue, payload in items:
            queue.put_nowait(payload)
