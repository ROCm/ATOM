# SPDX-License-Identifier: MIT
"""Bounded async device-to-host copies of per-request rows, read back later.

The DSpark planner and the online STS collector both copy a step's draft
statistics off the GPU and read them two forwards later, when the copy has
long landed. A copy owns its pinned destination and holds its device source
until its event completes: dropping a copy's bookkeeping never frees either
under an in-flight DMA.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DelayedCopy:
    tag: int
    event: object
    cpu: torch.Tensor
    ids: tuple
    # Read by the copy stream until `event` completes.
    source: torch.Tensor | None


class DelayedRows:
    def __init__(
        self,
        stream: Callable[[], object],
        width: int,
        *,
        min_rows: int = 8,
        max_pending: int = 4,
    ):
        # `stream` returns the copy stream; resolved per copy, since the
        # runner builds it after the consumers of this class.
        self._stream = stream
        self.width = width
        self.min_rows = min_rows
        self.max_pending = max_pending
        self.pending: list[DelayedCopy] = []
        self.free: list[torch.Tensor] = []

    def busy(self) -> bool:
        return len(self.pending) >= self.max_pending

    def _buffer(self, n: int, rows: int) -> torch.Tensor:
        # By position: comparing tensors would compare their elements.
        index = next((i for i, b in enumerate(self.free) if b.shape[0] >= n), None)
        if index is not None:
            return self.free.pop(index)
        # device="cpu": warmup runs with a GPU default device, where pinning an
        # unqualified allocation fails.
        return torch.empty(
            (max(n, rows), self.width),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

    def push(
        self, tag: int, ids, source: torch.Tensor, *, rows: int | None = None
    ) -> bool:
        """Start copying `source` ([len(ids), width]) under `tag`; False when
        `max_pending` copies are already outstanding."""
        if self.busy():
            return False
        n = source.shape[0]
        cpu = self._buffer(n, self.min_rows if rows is None else rows)
        stream = self._stream()
        producer = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            stream.wait_stream(producer)
            cpu[:n].copy_(source, non_blocking=True)
            event.record(stream)
        self.pending.append(DelayedCopy(tag, event, cpu, tuple(ids), source))
        return True

    def take(
        self, tag: int, ids=None, *, wait: bool, release: bool = False
    ) -> tuple[tuple, np.ndarray] | None:
        """The rows copied under `tag` (and `ids`, when given) as
        `(ids, [len(ids), width] array)`, or None.

        `wait`: synchronize on the copy; otherwise a copy still in flight is
        not ready. `release`: hand the destination back once read.
        """
        for i, copy in enumerate(self.pending):
            if copy.tag != tag or (ids is not None and copy.ids != tuple(ids)):
                continue
            if wait:
                copy.event.synchronize()
            elif not copy.event.query():
                return None
            rows = copy.cpu[: len(copy.ids)].numpy().copy()
            if release:
                del self.pending[i]
                self.free.append(copy.cpu)
            return copy.ids, rows
        return None

    def reap(self, before: int) -> None:
        """Release every completed copy tagged before `before`."""
        keep = []
        for copy in self.pending:
            if copy.tag < before and copy.event.query():
                self.free.append(copy.cpu)
            else:
                keep.append(copy)
        self.pending = keep

    def invalidate(self) -> None:
        """No outstanding copy may be read any more; each still owns its
        buffers until `reap` sees its event complete."""
        for copy in self.pending:
            copy.tag = -1
