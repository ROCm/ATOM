# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-PyTorch reference for the Kimi-K3 fused attention-residual op.

Mirrors ``attention_residual._apply_attn_res_impl`` semantics exactly, in plain
eager PyTorch: no Triton, no custom op, no dispatch-by-token-count. Use it to
read what the kernel does, to check the kernel against, or as a fallback where
Triton is unavailable.

It is NOT a drop-in for production: it materializes the [T, Bp, H] candidate
stack that the kernel deliberately avoids, so it costs ~Bp x the memory traffic
and an extra HBM round-trip. That is the whole point of the fused version.
"""

from __future__ import annotations

import torch


def apply_attn_res_ref(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]
    score_weight: torch.Tensor,  # [H] = norm.weight * proj.weight, precomputed
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H]
    out_norm_weight: torch.Tensor | None = None,  # [H]
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,  # [T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-residual soft-attention mix over B+1 candidates.

    The B rows of ``block_residual`` plus ``prefix_sum`` form Bp = B+1 candidate
    vectors per token. Each is rmsnormed, scored against ``score_weight``,
    softmaxed over the candidate axis, and mixed::

        v_b   = candidate b                                   [T, Bp, H]
        s_b   = <v_b * rsqrt(mean(v_b^2) + eps), score_weight>
        p     = softmax_b(s)                                  [T, Bp]
        y     = sum_b p_b * v_b                               [T, H]

    ``score_weight`` is already the product of the rmsnorm gain and the scoring
    projection (folded at load time), which is why the norm applies only the
    rstd here and not a separate weight.

    Returns ``(y, prefix_out)``. When ``add_hidden`` (and optionally
    ``add_hidden2``) is given, the caller's ``prefix_sum = prefix_sum + ...`` is
    folded in before the candidate stack is built, and ``prefix_out`` is that
    sum -- downstream layers reuse it. Otherwise ``prefix_out`` is ``prefix_sum``
    unchanged.

    When ``out_norm_weight`` is given, the caller's rmsnorm OF THE RESULT is
    applied to ``y`` before returning.

    Reductions run in fp32 (the kernel does the same) and the result is cast
    back to ``prefix_sum.dtype``.
    """
    T, B, H = block_residual.shape

    prefix = prefix_sum.float()
    if add_hidden is not None:
        prefix = prefix + add_hidden.float()
    if add_hidden2 is not None:
        prefix = prefix + add_hidden2.float()

    # [T, Bp, H] -- candidate B is the prefix; the kernel reads the two source
    # tensors in place instead of materializing this.
    v = torch.cat([block_residual.float(), prefix.unsqueeze(1)], dim=1)

    rstd = torch.rsqrt(v.square().mean(-1) + eps)  # [T, Bp]
    scores = rstd * (v * score_weight.float()).sum(-1)  # [T, Bp]
    probs = torch.softmax(scores, dim=-1)  # [T, Bp]
    y = (probs.unsqueeze(-1) * v).sum(1)  # [T, H]

    if out_norm_weight is not None:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + out_eps)
        y = y * out_norm_weight.float()

    added = add_hidden is not None or add_hidden2 is not None
    prefix_out = prefix.to(prefix_sum.dtype) if added else prefix_sum
    return y.to(prefix_sum.dtype), prefix_out
