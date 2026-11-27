# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""TP row sharding shared by residual and model-tail paths."""

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn


def _pad_rows(x: torch.Tensor, missing: int) -> torch.Tensor:
    """`x` with `missing` zero rows appended on dim 0, for any rank."""
    if not missing:
        return x
    return nn.functional.pad(x, (0, 0) * (x.dim() - 1) + (0, missing))


class SequenceParallel:
    """Shard rows across TP ranks, gather rows, or scatter partial sums.

    Pad to a multiple of the TP size and drop padded rows on gather.
    Per-token work runs locally; consumers needing every row gather first.
    """

    def __init__(self, num_tokens: int, tp_size: int, tp_rank: int) -> None:
        self.num_tokens = num_tokens
        self.chunk = -(-num_tokens // tp_size)
        self.pad = self.chunk * tp_size - num_tokens
        self.start = tp_rank * self.chunk
        self.group = get_tp_group()

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated `[T, ...]` -> this rank's rows (a view unless padded).

        Only the local rows are padded, never the whole replicated tensor."""
        rows = x[self.start : self.start + self.chunk]
        return _pad_rows(rows, self.chunk - rows.shape[0])

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows -> all `[T, ...]` rows."""
        out = self.group.all_gather(x.contiguous(), use_custom=False, dim=0)
        return out[: self.num_tokens] if self.pad else out

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Per-rank partial sums over all `[T, ...]` rows -> summed local rows."""
        return self.group.reduce_scatter_tensor(
            _pad_rows(x.contiguous(), self.pad), use_custom=False, dim=0
        )
