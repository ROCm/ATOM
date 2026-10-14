# SPDX-License-Identifier: MIT
"""Model-owned boundary buffers, independent of each bucket's graph pool."""

import torch


class TokenBuffers:
    """Keep one maximum-capacity allocation per role and return token views.

    A model forward completes each boundary before another forward can reuse
    it. Buckets therefore share this explicit storage while their compiled
    graph intermediates remain in separate CUDA graph pools. Capacity is fixed
    before capture; growing it later would invalidate recorded addresses.
    """

    def __init__(self, capacity: int, token_dim: int):
        self.capacity = capacity
        self.token_dim = token_dim
        self._slots: dict = {}

    def slot(self, role, sample: torch.Tensor) -> torch.Tensor:
        shape = list(sample.shape)
        tokens = shape[self.token_dim]
        if tokens > self.capacity:
            raise ValueError(
                f"Piecewise buffer needs {tokens} tokens, capacity {self.capacity}"
            )
        shape[self.token_dim] = self.capacity
        key = (role, tuple(shape), sample.dtype, sample.device)
        if key not in self._slots:
            self._slots[key] = sample.new_empty(shape)
        return self._slots[key].narrow(self.token_dim, 0, tokens)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self._slots.values())
