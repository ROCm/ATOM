# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4 embedding schema; compiled prefill must leave weight versions unchanged."""

import aiter
import torch
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.embed_head import (
    VocabParallelEmbedding,
    _masked_embedding_fake,
    _masked_embedding_launcher,
)
from atom.utils.decorators import mark_trace


@torch_compile_guard(mutates_args=[], gen_fake=_masked_embedding_fake)
def _v4_embedding_read_only(
    x: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
) -> torch.Tensor:
    # Reuse the native lookup kernel; only its mutation schema is V4-specific.
    return _masked_embedding_launcher(x, weight, vocab_start_idx, vocab_end_idx)


class V4VocabParallelEmbedding(VocabParallelEmbedding):
    @mark_trace
    def forward(self, x: torch.Tensor):
        if self.tp_size == 1:
            return super().forward(x)
        output = _v4_embedding_read_only(
            x, self.weight, self.vocab_start_idx, self.vocab_end_idx
        )
        return get_tp_group().all_reduce(output, ca_fp8_quant=False)


def mhc_input(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Prepare Mono's unshuffled mHC residual with the native embedding op."""
    h = model.embed(input_ids)
    if hasattr(aiter, "mhc_res_repeat"):
        return aiter.mhc_res_repeat(h, model.hc_mult, False)
    return h.unsqueeze(-2).repeat(1, model.hc_mult, 1)
