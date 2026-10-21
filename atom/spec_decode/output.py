# SPDX-License-Identifier: MIT
"""Deferred MTP counts: normalize producer layouts and preserve FIFO order."""

from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from atom.model_engine.output_transfer import D2HBufferPool, PendingD2H


class MTPStatus(NamedTuple):
    rejected: np.ndarray | None
    bonus: np.ndarray | None


class MTPStatusQueue:
    """Both submission paths produce host rows [bonus, rejected]."""

    def __init__(self, max_requests: int, stream: torch.cuda.Stream):
        self._buffers = D2HBufferPool(2 * max_requests, stream)
        self._pending: deque[PendingD2H] = deque()

    def send_verdict(self, verdict: torch.Tensor, ready: torch.cuda.Event) -> None:
        # The sampler's post-broadcast [bonus, rejected, anchor] is contiguous.
        # Exclude the GPU-only anchor, without packing or copying on the GPU.
        self._pending.append(self._buffers.copy(verdict[:2], ready=ready))

    def send_counts(
        self, rejected: torch.Tensor, bonus: torch.Tensor, ready: torch.cuda.Event
    ) -> None:
        # Prefill has no verdict. Both counts still share one host slot and
        # completion event; copying separate GPU rows requires two transfers.
        self._pending.append(self._buffers.copy_rows((bonus, rejected), ready=ready))

    def receive(self) -> MTPStatus:
        if not self._pending:
            return MTPStatus(None, None)
        bonus, rejected = self._pending.popleft().result_rows()
        return MTPStatus(rejected, bonus)
