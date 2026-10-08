# SPDX-License-Identifier: MIT
"""Pinned/device metadata ownership for overlapping prefill microbatches."""

from contextlib import contextmanager

import torch


class PrefillStorage:
    def __init__(self, specs, ratios, device):
        self.buffers = {name: spec.allocate(device) for name, spec in specs.items()}
        capacity = self.buffers["positions"].gpu.numel()
        self.indptrs = {
            ratio: tuple(
                torch.empty(capacity + 1, dtype=torch.int32, device=device)
                for _ in range(2)
            )
            for ratio in ratios
        }
        self.device = device
        on_gpu = torch.device(device).type != "cpu"
        self.h2d_done = torch.cuda.Event() if on_gpu else None
        self.done = torch.cuda.Event() if on_gpu else None
        self.consumer_stream = None

    def ready(self):
        return self.done is None or self.done.query()

    def acquire(self):
        if self.ready():
            return
        # Device consumers may still be running after H2D has released the
        # pinned source. Only the latter can require a CPU wait.
        if not self.h2d_done.query():
            self.h2d_done.synchronize()
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self.consumer_stream:
            stream.wait_event(self.done)

    @contextmanager
    def upload(self):
        try:
            yield
        finally:
            if self.h2d_done is not None:
                self.h2d_done.record(torch.cuda.current_stream(self.device))

    def finish(self):
        if self.done is not None:
            stream = torch.cuda.current_stream(self.device)
            self.done.record(stream)
            self.consumer_stream = stream.cuda_stream

    def close(self):
        if self.done is not None:
            self.done.synchronize()


class PrefillStoragePool:
    """Two bounded slots per microbatch; fence only the slots actually used."""

    def __init__(self):
        self.slots = {}
        self.active = {}
        self.last = {}

    def acquire(self, index, specs, ratios, device):
        if index not in self.slots:
            # Allocate together before the first forward, avoiding allocations
            # when switching slots while the previous model call is in flight.
            self.slots[index] = [
                PrefillStorage(specs, ratios, device) for _ in range(2)
            ]
            self.last[index] = 0
        slots = self.slots[index]
        current = self.last[index]
        if not slots[current].ready():
            current = 1 - current
        slot = slots[current]
        slot.acquire()
        self.last[index] = current
        self.active[index] = slot
        return slot

    @contextmanager
    def forward(self):
        active = tuple(self.active.values())
        self.active.clear()
        try:
            yield
        finally:
            for slot in active:
                slot.finish()

    def close(self):
        for slots in self.slots.values():
            for slot in slots:
                slot.close()
        self.slots.clear()
        self.active.clear()
        self.last.clear()
