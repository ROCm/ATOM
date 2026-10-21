# SPDX-License-Identifier: MIT
"""Reusable pinned D2H storage with completion-bound output ownership.

This module knows tensor shapes and transfer lifetimes, not model outputs or
speculative-decode fields. Only the owning runner thread submits and consumes.
"""

import math

import numpy as np
import torch


class _PinnedSlot:
    def __init__(self, capacity: int, dtype: torch.dtype):
        self.tensor = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
        self.array = self.tensor.numpy()
        self.done = torch.cuda.Event()
        self.shape = None

    def set_shape(self, shape: tuple[int, ...]) -> None:
        # Cache just the current views. Batch changes do not replace storage
        # or retain an unbounded collection of per-shape tensor wrappers.
        if shape != self.shape:
            count = math.prod(shape)
            self.view = self.tensor[:count].view(shape)
            self.array_view = self.array[:count].reshape(shape)
            self.shape = shape


class PendingD2H:
    """One submitted transfer; consuming it once returns independent arrays."""

    __slots__ = ("_pool", "_slot")

    def __init__(self, pool: "D2HBufferPool", slot: _PinnedSlot):
        self._pool = pool
        self._slot: _PinnedSlot | None = slot

    def result(self) -> np.ndarray:
        return self._consume(rows=False)

    def result_rows(self) -> tuple[np.ndarray, ...]:
        """Take independently owned rows, without copying the whole array first."""
        return self._consume(rows=True)

    def _consume(self, *, rows):
        slot = self._slot
        if slot is None:
            raise RuntimeError("D2H output has already been consumed")
        slot.done.synchronize()
        result = (
            tuple(row.copy() for row in slot.array_view)
            if rows
            else slot.array_view.copy()
        )
        self._slot = None
        self._pool._free.append(slot)
        return result


class D2HBufferPool:
    """Fixed element capacity per slot, with one slot per outstanding output.

    Slots are allocated on demand and reused only after their PendingD2H was
    consumed. A dtype change replaces a free slot; a shape change only changes
    its active view. No GPU staging tensor is allocated.
    """

    def __init__(self, capacity: int, stream: torch.cuda.Stream):
        if capacity < 0:
            raise ValueError("D2H capacity must be nonnegative")
        self.capacity = capacity
        self.stream = stream
        self._free: list[_PinnedSlot] = []

    def copy(
        self, source: torch.Tensor, *, ready: torch.cuda.Event | None = None
    ) -> PendingD2H:
        """Copy one tensor, after ready or the caller's current stream."""
        return self._submit((source,), source.shape, rows=False, ready=ready)

    def copy_rows(
        self, sources: tuple[torch.Tensor, ...], *, ready: torch.cuda.Event
    ) -> PendingD2H:
        """Copy separate same-shaped tensors into consecutive host rows."""
        if not sources:
            raise ValueError("D2H rows must not be empty")
        first = sources[0]
        if any(s.shape != first.shape or s.dtype != first.dtype for s in sources):
            raise ValueError("D2H rows must have matching shapes and dtypes")
        return self._submit(
            sources, (len(sources), *first.shape), rows=True, ready=ready
        )

    def _submit(self, sources, shape, *, rows, ready):
        if math.prod(shape) > self.capacity:
            raise ValueError(
                f"D2H shape {tuple(shape)} exceeds capacity {self.capacity}"
            )
        slot = self._free.pop() if self._free else None
        if slot is None or slot.tensor.dtype != sources[0].dtype:
            slot = _PinnedSlot(self.capacity, sources[0].dtype)
        slot.set_shape(shape)
        producer = torch.cuda.current_stream() if ready is None else None
        with torch.cuda.stream(self.stream):
            if ready is None:
                self.stream.wait_stream(producer)
            else:
                ready.wait(stream=self.stream)
            if rows:
                for destination, source in zip(slot.view.unbind(0), sources):
                    destination.copy_(source, non_blocking=True)
            else:
                slot.view.copy_(sources[0], non_blocking=True)
            slot.done.record(self.stream)
        return PendingD2H(self, slot)
