# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runner-owned packed mask buffers; allocated only for constrained requests."""

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
            # CPU reference path; GPU execution never expands the packed mask.
            tokens = torch.arange(logits.shape[1])
            for i, row in enumerate(rows):
                allowed = (
                    (self.host_masks[i, tokens // 32] >> (tokens % 32)) & 1
                ).bool()
                logits[row].masked_fill_(~allowed, float("-inf"))
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
