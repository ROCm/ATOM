# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused grouped Gemma RMSNorm.

The eager form is eight kernels over a `[tokens, hc_count * hidden]` tensor --
cast, square, reduce, rsqrt, multiply, flatten, add-one, multiply, cast -- and
Qwen3.8-Flash-Next runs it 100 times per forward (two hyper-connections on each of 48
layers, the final mixer, and PLE's three norms). At 2560-wide groups that was
~10% of all GPU time, spread over four separate kernel families, none of which
is doing enough work to cover its own launch.

One program per (row, group) collapses the whole thing to a single launch that
reads `x` once and writes `y` once.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < GROUP_SIZE
    column = group * GROUP_SIZE + offsets

    x = tl.load(x_ptr + row * stride_row + column, mask=mask, other=0.0).to(tl.float32)
    # Same order as the eager form: variance over the group, then Gemma's
    # `x * (1 + w)` -- accumulated in fp32 whatever the tensor dtype is.
    variance = tl.sum(x * x, axis=0) / GROUP_SIZE
    normalized = x * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + column, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row * stride_row + column,
        (normalized * (1.0 + weight)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """RMSNorm with per-`group_size` variance and a full-width Gemma weight.

    `hidden_states` is `[..., num_groups * group_size]`; every trailing
    dimension but the last is flattened into the row axis.
    """
    shape = hidden_states.shape
    width = shape[-1]
    if width % group_size:
        raise ValueError(f"width {width} is not a multiple of group_size {group_size}")
    if weight.numel() != width:
        raise ValueError("weight must be one element per column")

    x = hidden_states.reshape(-1, width)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.shape[0]
    if rows == 0:
        return out.reshape(shape)

    _grouped_gemma_rmsnorm_kernel[(rows, width // group_size)](
        x,
        weight,
        out,
        x.stride(0),
        float(eps),
        GROUP_SIZE=group_size,
        BLOCK=triton.next_power_of_2(group_size),
        num_warps=8,
    )
    return out.reshape(shape)
