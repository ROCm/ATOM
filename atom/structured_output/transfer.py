# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small, owned token snapshots copied independently of the forward stream."""

import torch


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
