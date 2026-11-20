# SPDX-License-Identifier: MIT
"""TP row sharding shared by residual and model-tail paths."""

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from torch import nn

from atom.utils import envs


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

    @classmethod
    def maybe(cls, num_tokens: int) -> "SequenceParallel | None":
        """Sharding for a forward of `num_tokens`, or None to stay replicated:
        no TP, below `ATOM_QWEN4_EXP_SP_MIN_TOKENS`, or under graph capture.
        The target model adds its own fused-quant floor
        (`Qwen4ExpModel._sequence_parallel`); the MTP drafter, whose unfused
        hyper-connections never quantize, uses this gate as is."""
        min_tokens = envs.ATOM_QWEN4_EXP_SP_MIN_TOKENS
        if (
            min_tokens <= 0
            or num_tokens < min_tokens
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return None
        return cls(num_tokens, tp_size, get_tensor_model_parallel_rank())
