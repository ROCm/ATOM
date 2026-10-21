# SPDX-License-Identifier: MIT
"""Runner-owned FIFO transfers with reusable pinned storage."""

from collections import deque

import torch


class _D2HSlot:
    def __init__(self, capacity: int, dtype: torch.dtype):
        self.storage = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
        self.array = self.storage.numpy()
        self.done = torch.cuda.Event()
        self.shape = None

    def reshape(self, source: torch.Tensor) -> None:
        # Cache only the current views; changing batch size keeps the allocation.
        if self.shape != source.shape:
            self.shape = source.shape
            count = source.numel()
            self.tensor = self.storage[:count].view(self.shape)
            self.output = self.array[:count].reshape(self.shape)


class D2HQueue:
    """Fixed-capacity slots, reused after their outputs have been received.

    Only the owning runner thread sends and receives. Subclasses can specialize
    the NumPy snapshot, which must detach all returned arrays from pinned storage.
    """

    def __init__(self, capacity: int, stream: torch.cuda.Stream):
        self.capacity = capacity
        self.stream = stream
        self._free: list[_D2HSlot] = []
        self._pending: deque[_D2HSlot] = deque()

    def send(
        self, source: torch.Tensor, *, ready: torch.cuda.Event | None = None
    ) -> None:
        if source.numel() > self.capacity:
            raise ValueError(
                f"D2H shape {tuple(source.shape)} exceeds capacity {self.capacity}"
            )
        slot = self._free.pop() if self._free else None
        if slot is None or slot.storage.dtype != source.dtype:
            slot = _D2HSlot(self.capacity, source.dtype)
        slot.reshape(source)
        producer = torch.cuda.current_stream() if ready is None else None
        with torch.cuda.stream(self.stream):
            if ready is None:
                self.stream.wait_stream(producer)
            else:
                ready.wait(stream=self.stream)
            slot.tensor.copy_(source, non_blocking=True)
            slot.done.record(self.stream)
        self._pending.append(slot)

    def receive(self):
        if not self._pending:
            return None
        slot = self._pending[0]
        slot.done.synchronize()
        output = self._snapshot(slot.output)
        self._pending.popleft()
        self._free.append(slot)
        return output

    @staticmethod
    def _snapshot(array):
        return array.copy()
