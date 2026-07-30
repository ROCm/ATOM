# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Batched cross-thread dispatch for streaming model output."""

import threading
from asyncio import AbstractEventLoop, Queue
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any


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
        self._states_lock = threading.Lock()

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
        buf = getattr(self._thread_local, "buf", None)
        if not buf:
            return
        self._thread_local.buf = []

        by_loop: dict[AbstractEventLoop, list[tuple[Queue, Any]]] = {}
        for item in buf:
            state = self._get_state(item.state_key)
            item.chunk["text"] = state.update(
                item.chunk.get("token_ids") or [],
                bool(item.chunk.get("finished")),
            )
            if item.chunk.get("finished"):
                self._drop_state(item.state_key, state)

            payload = item.chunk if item.tag is None else (item.tag, item.chunk)
            by_loop.setdefault(item.loop, []).append((item.queue, payload))

        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(self._drain_into_queues, items)

    def discard_request(self, request_id: str) -> None:
        """Drop direct and fan-out detokenizer state after request cleanup."""
        with self._states_lock:
            keys = [
                key
                for key in self._states
                if key == request_id
                or (isinstance(key, tuple) and key and key[0] == request_id)
            ]
            for key in keys:
                self._states.pop(key, None)

    def _get_state(self, state_key: Hashable) -> IncrementalStreamDetokenizer:
        with self._states_lock:
            state = self._states.get(state_key)
            if state is None:
                state = self._states[state_key] = IncrementalStreamDetokenizer(
                    self.tokenizer
                )
            return state

    def _drop_state(
        self, state_key: Hashable, state: IncrementalStreamDetokenizer
    ) -> None:
        with self._states_lock:
            if self._states.get(state_key) is state:
                self._states.pop(state_key)

    @staticmethod
    def _drain_into_queues(items: list[tuple[Queue, Any]]) -> None:
        """Run on the target event loop and deliver each prepared payload."""
        for queue, payload in items:
            queue.put_nowait(payload)
