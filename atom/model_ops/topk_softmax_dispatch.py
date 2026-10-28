# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs renormalized softmax top-k routing: the Triton one in
`triton_topk_softmax.py` or AITER's `topk_softmax`.

Shape checks only, with no Triton or AITER import, so the GPU-less CI runs
their tests too.
"""

import torch

from atom.utils import envs


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
    this launch rather than AITER's `topk_softmax`. Eager and captured passes
    choose alike, so a token routes through the same kernel whichever batch
    it lands in, without relying on the two kernels matching bit for bit.
    `ATOM_TRITON_TOPK_SOFTMAX=0` puts every pass on AITER."""
    return (
        envs.ATOM_TRITON_TOPK_SOFTMAX
        and shared_experts_scoring_func in ("", "sigmoid")
        and triton_topk_softmax_supported(
            topk_weights, topk_ids, gating_output, renormalize, num_shared_experts
        )
    )
