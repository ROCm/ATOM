# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The indexer's logits plane, held for the server's life.

A plane is as wide as a request may be long (`max_model_len // ratio`), not as
its live context. Allocated per forward it is multi-GiB at a height that
varies with the batch: the memory profile (a prefill) never sees it, a token
graph's pool keeps the largest bucket's, and the caching allocator is left
holding segments too small for the next long prefill. Sized here from the
configuration, before the memory profile, it is counted in it and never
reallocated; scoring runs in bands of it.
"""

import torch

from atom.model_ops.sparse_indexer_chunk import (
    decode_band_rows,
    flat_view,
    logits_rows,
)


class LogitsWorkspace:
    """`capacity` fp32 elements every indexer of a model scores its bands
    into, one layer after another."""

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self._logits = torch.empty(capacity, dtype=torch.float32, device=device)

    @classmethod
    def for_rows(cls, rows: int, width: int, device) -> "LogitsWorkspace":
        """A plane for `rows` query rows of `width` columns, in as many bands
        as the logits budget asks."""
        return cls(logits_rows(rows, width) * width, device)

    @property
    def nbytes(self) -> int:
        return self._logits.numel() * self._logits.element_size()

    def logits_rows(self, rows: int, width: int, group: int = 1) -> int:
        """Rows of a `width`-column band: within the byte budget and this
        allocation, whichever is smaller, in whole groups of `group` rows."""
        return decode_band_rows(rows, width, group, self._logits.numel() // width)

    def logits(self, rows: int, width: int) -> torch.Tensor:
        """`[rows, width]` fp32, one band of the plane; fails past it."""
        return flat_view(self._logits, rows, width)
