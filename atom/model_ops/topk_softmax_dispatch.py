# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs renormalized softmax top-k routing: the Triton one in
`triton_topk_softmax.py` or AITER's `topk_softmax`.

Shape and row-count checks only, with no Triton or AITER import at module
level, so the GPU-less CI runs their tests too.
"""

import functools

import torch

from atom.utils import envs

# Launches of up to this many rows per compute unit take the Triton kernel,
# longer ones AITER's. Both kernels' times step with rows per CU: Triton runs
# one single-wave program per row, AITER packs 8-64 rows into each block.
# Measured on MI355X (256 CUs), Triton is faster up to 20 rows per CU (256
# experts / top-8 with or without a shared expert, 32 / top-4), 22 (128 /
# top-4), 24 (128 / top-8, 512 / top-10) or 28 (512 / top-10 + a shared
# expert), and AITER past it; 20 keeps Triton only where it is faster for all.
TRITON_TOPK_SOFTMAX_MAX_ROWS_PER_CU = 20


@functools.cache
def _cu_num() -> int:
    from aiter.jit.utils.chip_info import get_cu_num

    return get_cu_num()


def triton_topk_softmax_max_rows() -> int:
    """The most rows a launch can have and still take the Triton kernel."""
    return TRITON_TOPK_SOFTMAX_MAX_ROWS_PER_CU * _cu_num()


def triton_topk_softmax_supported(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_shared_experts: int,
) -> bool:
    """Whether `triton_topk_softmax` takes this launch: the shapes AITER's
    fused `topkGatingSoftmax` handles (power-of-two expert counts up to 512;
    past 512 AITER takes `moeSoftmax` + `moeTopK`, which round the weights
    differently), with a unit last stride on the logits and both outputs."""
    num_experts = gating_output.shape[-1] - num_shared_experts
    topk = topk_ids.shape[-1]
    return (
        renormalize
        and gating_output.dim() == topk_weights.dim() == topk_ids.dim() == 2
        and gating_output.stride(-1) == 1
        and topk_weights.stride(-1) == 1
        and topk_ids.stride(-1) == 1
        and 32 <= num_experts <= 512
        and num_experts & (num_experts - 1) == 0
        and 0 < topk <= 32
        and num_shared_experts in (0, 1, 2, 4, 8)
    )


def use_triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_shared_experts: int,
    shared_experts_scoring_func: str,
) -> bool:
    """Whether `rocm_aiter_topk_softmax_impl` runs `triton_topk_softmax` for
    this launch rather than AITER's `topk_softmax`: the shapes the Triton
    kernel supports, up to `triton_topk_softmax_max_rows()` rows. Only the
    launch decides, never whether the stream is capturing, so a pass of a
    given size runs the same kernel eager or replayed. On MI355X the cutoff
    (5120 rows) is above every default CUDA graph (decode: 512 sequences x 4
    tokens with MTP 3; prefill token graphs: 3072 tokens), so replayed passes
    all run Triton; a long eager prefill runs AITER, which the Triton kernel
    matches bit for bit.
    `ATOM_TRITON_TOPK_SOFTMAX=0` puts every pass on AITER."""
    return (
        envs.ATOM_TRITON_TOPK_SOFTMAX
        and gating_output.shape[0] <= triton_topk_softmax_max_rows()
        and shared_experts_scoring_func in ("", "sigmoid")
        and triton_topk_softmax_supported(
            topk_weights, topk_ids, gating_output, renormalize, num_shared_experts
        )
    )
