# SPDX-License-Identifier: MIT
"""Runner-owned FIFO transfers with reusable pinned storage."""

from collections import deque

import torch


class _D2HSlot:
    def __init__(self, capacity: int, dtype: torch.dtype):
        self.buffer = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
        self.buffer_array = self.buffer.numpy()
        self.copy_done = torch.cuda.Event()
        self.shape = None
        # The tensor being copied, held until copy_done has been waited on.
        self.source: torch.Tensor | None = None

    def set_shape_from(self, source: torch.Tensor) -> None:
        # Cache only the current views; changing batch size keeps the allocation.
        if self.shape != source.shape:
            self.shape = source.shape
            count = source.numel()
            self.tensor_view = self.buffer[:count].view(self.shape)
            self.array_view = self.buffer_array[:count].reshape(self.shape)


class D2HQueue:
    """Fixed-capacity slots, reused after their outputs have been received.

    Capacity counts tensor elements per slot. Only the owning runner thread
    sends and receives. Subclasses customize _copy_output to detach their NumPy
    outputs from pinned storage.

    Each slot holds its source until the copy is known to have completed. The
    caching allocator does not see the side-stream read, so a source the caller
    drops (e.g. the sampler's fresh verdict) would otherwise go to the next
    default-stream allocation before a late copy runs, and the host would
    receive the next step's data. Under MTP a stale bonus count becomes
    num_accepted_tokens, which the GDN conv kernel adds to the state offset
    unchecked.
    """

    def __init__(self, capacity: int, stream: torch.cuda.Stream):
        self.capacity = capacity
        self.stream = stream
        self._free: list[_D2HSlot] = []
        self._pending: deque[_D2HSlot] = deque()

    def send(
        self, source: torch.Tensor, *, ready: torch.cuda.Event | None = None
    ) -> None:
        """Enqueue a copy after ready, or after the caller's current stream."""
        if source.numel() > self.capacity:
            raise ValueError(
                f"D2H shape {tuple(source.shape)} exceeds capacity {self.capacity}"
            )
        slot = self._free.pop() if self._free else None
        if slot is None or slot.buffer.dtype != source.dtype:
            slot = _D2HSlot(self.capacity, source.dtype)
        slot.set_shape_from(source)
        producer = torch.cuda.current_stream() if ready is None else None
        with torch.cuda.stream(self.stream):
            if ready is None:
                self.stream.wait_stream(producer)
            else:
                ready.wait(stream=self.stream)
            slot.tensor_view.copy_(source, non_blocking=True)
            slot.copy_done.record(self.stream)
        slot.source = source
        self._pending.append(slot)

    def receive(self):
        """Wait for the oldest transfer and return an independently owned output."""
        if not self._pending:
            return None
        slot = self._pending[0]
        slot.copy_done.synchronize()
        output = self._copy_output(slot.array_view)
        # The copy has completed; only now may the allocator reuse the source.
        slot.source = None
        self._pending.popleft()
        self._free.append(slot)
        return output

    def drain(self) -> None:
        """Wait for every pending transfer, then drop its output and source."""
        while self._pending:
            slot = self._pending[0]
            slot.copy_done.synchronize()
            slot.source = None
            self._pending.popleft()
            self._free.append(slot)

    @staticmethod
    def _copy_output(array):
        return array.copy()
