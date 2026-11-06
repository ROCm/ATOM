# SPDX-License-Identifier: MIT
"""Runner-owned FIFO transfers with reusable pinned storage."""

import time
from collections import deque

import numpy as np
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

    def send_direct(self, slot: "HostSlot") -> None:
        """Enqueue a transfer a kernel makes itself into ``slot`` (a
        `HostHandoffRing` slot): received in order with the copies."""
        self._pending.append(slot)

    def _release(self, slot) -> None:
        slot.source = None
        self._pending.popleft()
        if isinstance(slot, HostSlot):
            slot.release()
        else:
            self._free.append(slot)

    def receive(self):
        """Wait for the oldest transfer and return an independently owned output."""
        if not self._pending:
            return None
        slot = self._pending[0]
        slot.copy_done.synchronize()
        output = self._copy_output(slot.array_view)
        # The copy has completed; only now may the allocator reuse the source.
        self._release(slot)
        return output

    def drain(self) -> None:
        """Wait for every pending transfer, then drop its output and source."""
        while self._pending:
            slot = self._pending[0]
            slot.copy_done.synchronize()
            self._release(slot)

    @staticmethod
    def _copy_output(array):
        return array.copy()


class HostSlot:
    """One `HostHandoffRing` slot claimed for a transfer: ``view`` (pinned, the
    kernel's destination), ``flag`` (pinned int32, set to ``seq`` last) and
    ``array_view`` (the host's numpy view). Stands in for a `_D2HSlot`: its
    ``copy_done.synchronize()`` polls the flag."""

    def __init__(self, ring: "HostHandoffRing", index: int, seq: int, shape):
        self.ring = ring
        self.index = index
        self.seq = seq
        count = int(np.prod(shape))
        self.view = ring.data[index, :count].view(shape)
        self.array_view = ring.data_np[index, :count].reshape(shape)
        self.flag = ring.flags[index, :1]
        self.copy_done = self
        self.source = None

    def synchronize(self) -> None:
        """Wait (host side, bounded) for the kernel's flag."""
        flags = self.ring.flags_np
        index, seq = self.index, self.seq
        if flags[index, 0] == seq:
            return
        deadline = time.monotonic() + self.ring.timeout_s
        while flags[index, 0] != seq:
            # Hand the GIL over between polls; other runner threads keep going.
            time.sleep(0)
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"host hand-off: slot {index} not flagged {seq} within "
                    f"{self.ring.timeout_s} s (flag {flags[index, 0]})"
                )

    def release(self) -> None:
        self.ring._release(self)


class HostHandoffRing:
    """Pinned slots a kernel writes its results straight into
    (ATOM_MTP_VERIFY_HOST_HANDOFF), each with a flag the kernel sets to the
    slot's sequence number last, with a system-scope release: the host polls
    the flag instead of waiting for an event, a side-stream copy and the
    copy's completion event. Slots are claimed in order and must be released
    (received) before the ring comes back to them."""

    def __init__(self, num_slots: int, capacity: int, timeout_s: float = 600.0):
        self.num_slots = num_slots
        # Whole 128-byte lines a slot, so every slot is aligned alike.
        capacity = -(-int(capacity) // 32) * 32
        self.capacity = capacity
        self.timeout_s = timeout_s
        self.data = torch.zeros(num_slots, capacity, dtype=torch.int32, pin_memory=True)
        self.data_np = self.data.numpy()
        # A flag a 128-byte line.
        self.flags = torch.zeros(num_slots, 32, dtype=torch.int32, pin_memory=True)
        self.flags_np = self.flags.numpy()
        self._seq = 0
        self._busy = [False] * num_slots

    def claim(self, shape) -> HostSlot:
        if int(np.prod(shape)) > self.capacity:
            raise ValueError(
                f"host hand-off shape {tuple(shape)} exceeds {self.capacity}"
            )
        # Positive int32 sequence numbers; a slot's flag never repeats one it held.
        seq = self._seq % 0x7FFFFFFE + 1
        index = seq % self.num_slots
        if self._busy[index]:
            raise RuntimeError(f"host hand-off slot {index} claimed before its receive")
        self._seq = seq
        self._busy[index] = True
        return HostSlot(self, index, seq, shape)

    def _release(self, slot: HostSlot) -> None:
        self._busy[slot.index] = False
