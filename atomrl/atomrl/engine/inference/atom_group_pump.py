# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Concurrent prompt groups on one ATOM engine."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _set_result(future: asyncio.Future, value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(future: asyncio.Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


class _PendingGroup:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future,
        seq_ids: list[Any],
    ) -> None:
        self.loop = loop
        self.future = future
        self.seq_ids = seq_ids
        self.outputs: dict[Any, dict[str, Any]] = {}

    def add(self, seq_id: Any, output: dict[str, Any]) -> bool:
        """Record one finished sequence; True once the whole group has finished."""
        self.outputs[seq_id] = output
        return len(self.outputs) == len(self.seq_ids)

    def resolve(self) -> None:
        results = [self.outputs[seq_id] for seq_id in self.seq_ids]
        self.loop.call_soon_threadsafe(_set_result, self.future, results)

    def fail(self, exc: BaseException) -> None:
        self.loop.call_soon_threadsafe(_set_exception, self.future, exc)


class AtomGroupPump:
    """Generate prompt groups concurrently on one ATOM ``LLMEngine``.

    ``LLMEngine.generate`` adds its requests, then reads outputs until the
    engine has nothing pending, keeping every output it reads: concurrent calls
    wait on each other's requests and take each other's outputs. Here requests
    are added under a lock, since the engine sends them over one ZMQ socket,
    and a single thread reads the finished sequences and hands each to the
    group that submitted it. Once a pump runs, every request on its engine has
    to go through it.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        # CoreManager queues finished sequences only for a reader that asks
        # (ROCm/ATOM#2912), as LLMEngine.add_request does; this pump is one. An
        # older ATOM queues them anyway and never reads the attribute.
        engine.core_mgr.keep_finished_outputs = True
        self._lock = threading.Lock()
        self._pending: dict[Any, _PendingGroup] = {}
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(
        self, prompt_ids: list[int], sampling_params: Any, request_id: str
    ) -> asyncio.Future:
        """Start ``sampling_params.n`` completions of one prompt.

        The future resolves to their output dicts, as ``io_processor.postprocess``
        builds them, in fan-out order.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    "the ATOM output reader has stopped"
                ) from self._error
            seqs = self._engine.io_processor.preprocess_fanout(
                list(prompt_ids),
                sampling_params,
                parent_request_id=request_id,
            )
            group = _PendingGroup(loop, future, [seq.id for seq in seqs])
            for seq in seqs:
                self._pending[seq.id] = group
            self._engine.core_mgr.add_request(seqs)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._read_outputs,
                    name="atom-group-pump",
                    daemon=True,
                )
                self._thread.start()
        return future

    def _read_outputs(self) -> None:
        while True:
            try:
                seqs = self._engine.step()
                with self._lock:
                    outputs = self._engine.io_processor.postprocess(seqs)
                    done = []
                    for seq_id, output in outputs.items():
                        group = self._pending.pop(seq_id, None)
                        if group is None:
                            logger.warning(
                                "ATOM finished sequence %s, which no group awaits",
                                seq_id,
                            )
                        elif group.add(seq_id, output):
                            done.append(group)
            # Pump failure must be delivered to every waiting request group.
            except BaseException as exc:  # noqa: BLE001 -- boundary
                with self._lock:
                    self._error = exc
                    groups = set(self._pending.values())
                    self._pending.clear()
                for group in groups:
                    group.fail(exc)
                return
            for group in done:
                group.resolve()
