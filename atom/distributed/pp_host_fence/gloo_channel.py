# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Per-edge CPU control channel. No module-global send-work queue."""

import pickle
import threading
from collections import deque
from datetime import timedelta


class GlooChannel:
    def __init__(self, torch, group, peer_global_rank, timeout=60, max_bytes=1048576):
        self.torch, self.group, self.peer = torch, group, peer_global_rank
        self.timeout, self.max_bytes = timeout, max_bytes
        self.pending = deque()
        self.failed_receives = []
        self.failed_sends = []
        self.owner = None

    def _claim(self):
        current = threading.get_ident()
        if self.owner is None:
            self.owner = current
        if current != self.owner:
            raise RuntimeError("each IPC edge channel has one Python owner thread")

    def _wait(self, work, timeout=None):
        result = work.wait(
            timeout=timedelta(seconds=self.timeout if timeout is None else timeout)
        )
        if result is False:
            raise TimeoutError("IPC Gloo metadata work did not complete")

    def _drain_first(self):
        work, _tensor = self.pending[0]
        self._wait(work)
        self.pending.popleft()

    def _receive(self, tensor, timeout):
        work = None
        try:
            work = self.torch.distributed.irecv(tensor, src=self.peer, group=self.group)
            self._wait(work, timeout)
        except Exception:
            self.failed_receives.append((work, tensor))
            raise

    def send(self, obj):
        self._claim()
        torch = self.torch
        blob = pickle.dumps(obj, protocol=5)
        if not 0 < len(blob) <= self.max_bytes:
            raise ValueError("IPC control message length exceeds its bound")
        raw = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        if raw.device.type != "cpu":
            raise RuntimeError("Gloo metadata must remain on CPU")
        size = torch.tensor([len(blob)], dtype=torch.long, device="cpu")
        for tensor in (size, raw):
            try:
                work = torch.distributed.isend(tensor, dst=self.peer, group=self.group)
            except Exception:
                self.failed_sends.append(tensor)
                raise
            self.pending.append((work, tensor))
        while self.pending and self.pending[0][0].is_completed():
            self._drain_first()
        while len(self.pending) > 64:
            self._drain_first()

    def recv(self, timeout):
        self._claim()
        torch = self.torch
        size = torch.empty(1, dtype=torch.long, device="cpu")
        self._receive(size, timeout)
        count = int(size.item())
        if not 0 < count <= self.max_bytes:
            raise ValueError("peer IPC metadata length exceeds its bound")
        raw = torch.empty(count, dtype=torch.uint8, device="cpu")
        self._receive(raw, timeout)
        message = pickle.loads(raw.numpy().tobytes())
        if not isinstance(message, dict):
            raise TypeError("IPC control message must be a dictionary")
        return message

    def flush(self):
        self._claim()
        while self.pending:
            self._drain_first()
