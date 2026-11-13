# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Owned token transfers and packed mask buffers for structured decoding."""

import torch


class StructuredOutputMask:
    def __init__(self, vocab_size):
        self.words = (vocab_size + 31) // 32
        self.capacity = 0
        self.device = None
        self.copy_done = None
        self.use_done = None

    def prepare(self, count, device):
        # CPU writes must not race an earlier asynchronous H2D transfer.
        if self.copy_done is not None and not self.copy_done.query():
            self.copy_done.synchronize()
        if count > self.capacity or device != self.device:
            # Growth/device changes are rare; keep the old device storage alive
            # until its last kernel finishes, including on a different stream.
            if self.use_done is not None:
                self.use_done.synchronize()
            self.capacity = max(count, self.capacity * 2)
            self.device = device
            gpu = device.type == "cuda"
            self.host_masks = torch.empty(
                (self.capacity, self.words),
                dtype=torch.int32,
                device="cpu",
                pin_memory=gpu,
            )
            self.host_rows = torch.empty(
                self.capacity, dtype=torch.int32, device="cpu", pin_memory=gpu
            )
            self.row_array = self.host_rows.numpy()
            if gpu:
                self.device_masks = torch.empty_like(self.host_masks, device=device)
                self.device_rows = torch.empty_like(self.host_rows, device=device)
                self.copy_done = torch.cuda.Event()
                self.use_done = torch.cuda.Event()
            else:
                self.copy_done = self.use_done = None
        return self.host_masks[:count]

    def apply(self, logits, rows):
        count = len(rows)
        if not count:
            return
        if logits.device.type != "cuda":
            apply_bitmask_reference(logits, self.host_masks, rows)
            return

        from atom.model_ops.triton_structured_output import apply_packed_bitmask

        stream = torch.cuda.current_stream(logits.device)
        # A caller may change streams between batches. Order device-buffer reuse
        # after the prior kernel without making the host wait for that kernel.
        stream.wait_event(self.use_done)
        self.row_array[:count] = rows
        masks = self.device_masks[:count]
        indices = self.device_rows[:count]
        masks.copy_(self.host_masks[:count], non_blocking=True)
        indices.copy_(self.host_rows[:count], non_blocking=True)
        self.copy_done.record(stream)
        apply_packed_bitmask(logits, masks, indices)
        self.use_done.record(stream)


class AsyncTokenCopy:
    def __init__(self, device):
        self.stream = torch.cuda.Stream(device=device)
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.host = None
        self.value = None

    def start(self, snapshot):
        # The caller gathers/clones on the producing stream. Never copy from a
        # reusable input-id buffer while the next forward can overwrite it.
        if (
            self.host is None
            or self.host.numel() < snapshot.numel()
            or self.host.dtype != snapshot.dtype
        ):
            self.host = torch.empty(
                snapshot.numel(), dtype=snapshot.dtype, device="cpu", pin_memory=True
            )
        self.value = self.host[: snapshot.numel()].view(snapshot.shape)
        self.ready.record(torch.cuda.current_stream(snapshot.device))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ready)
            self.value.copy_(snapshot, non_blocking=True)
            # Keep the GPU allocation safe even if a cancelled batch drops its
            # last Python reference before the side-stream copy finishes.
            snapshot.record_stream(self.stream)
            self.done.record(self.stream)

    def wait(self):
        # Only the copy event, recorded before the overlapping model work.
        # Callers must consume the view before start() reuses the pinned buffer.
        self.done.synchronize()
        return self.value


def apply_bitmask_reference(logits, masks, rows):
    """Pure CPU oracle for the packed-mask GPU kernel."""
    tokens = torch.arange(logits.shape[1])
    for i, row in enumerate(rows):
        allowed = ((masks[i, tokens // 32] >> (tokens % 32)) & 1).bool()
        logits[row].masked_fill_(~allowed, float("-inf"))
