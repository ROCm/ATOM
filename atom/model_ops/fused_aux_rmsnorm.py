# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Fused per-group RMSNorm for EAGLE3 aux hidden-state fusion.

EAGLE3's ``combine_hidden_states`` normalizes ``num_aux`` aux chunks (each with
its own ``fc_norm`` weight) and concatenates them into the ``[N, num_aux*H]``
input of the ``fc`` projection.  The naive path launches one RMSNorm per chunk
plus a concat; this kernel does all chunks in a single launch, writing straight
into the contiguous ``fc`` input buffer.

Input layout: ``x`` is the concatenated aux ``[N, num_aux*H]`` (view as groups
of ``H`` along the last dim).  ``weight`` is the per-group RMSNorm weights
stacked to ``[num_aux, H]``.  Plain RMSNorm (``x * rstd * w``, fp32 reduction) —
matches ``atom.model_ops.layernorm.RMSNorm`` (NOT the Gemma ``1+w`` variant).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_group_rmsnorm_kernel(
    x_ptr,  # [N, G*H] contiguous
    w_ptr,  # [G, H] contiguous
    out_ptr,  # [N, G*H] contiguous
    n_rows,
    G: tl.constexpr,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)
    col = tl.arange(0, BLOCK_H)
    mask = col < H

    row_base = row * (G * H) + g * H
    x = tl.load(x_ptr + row_base + col, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + g * H + col, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row_base + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_group_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    """Per-group RMSNorm over a concatenated ``[N, num_groups*H]`` tensor.

    Args:
        x: contiguous ``[N, num_groups*H]`` (groups of ``H`` along dim -1).
        weight: per-group weights stacked to ``[num_groups, H]`` (contiguous).
        eps: RMSNorm epsilon.
        num_groups: number of aux groups (``G``).

    Returns:
        ``[N, num_groups*H]`` with each group RMS-normalized by its own weight.
    """
    assert x.is_cuda, "fused_group_rmsnorm requires a CUDA tensor."
    assert x.dim() == 2 and x.is_contiguous()
    n_rows, total = x.shape
    assert total % num_groups == 0
    H = total // num_groups
    assert weight.shape == (
        num_groups,
        H,
    ), f"weight must be [{num_groups}, {H}], got {tuple(weight.shape)}"

    out = torch.empty_like(x)
    BLOCK_H = triton.next_power_of_2(H)
    num_warps = 8 if BLOCK_H >= 4096 else (4 if BLOCK_H >= 1024 else 2)
    grid = (n_rows, num_groups)
    _fused_group_rmsnorm_kernel[grid](
        x,
        weight.contiguous(),
        out,
        n_rows,
        num_groups,
        H,
        float(eps),
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return out
