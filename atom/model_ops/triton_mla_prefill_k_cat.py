# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared MLA prefill K concatenation; imports only Torch and Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_k_cat_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    k_stride_t,
    k_stride_h,
    nope_stride_t,
    nope_stride_h,
    rope_stride_t,
    num_heads,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    mask = h < num_heads
    d_nope = tl.arange(0, NOPE_DIM)[None, :]
    d_rope = tl.arange(0, ROPE_DIM)[None, :]
    nope = tl.load(
        k_nope_ptr + t * nope_stride_t + h * nope_stride_h + d_nope, mask=mask
    )
    rope = tl.load(k_rope_ptr + t * rope_stride_t + d_rope)
    out = k_ptr + t * k_stride_t + h * k_stride_h
    tl.store(out + d_nope, nope, mask=mask)
    tl.store(
        out + NOPE_DIM + d_rope,
        tl.broadcast_to(rope, (HEAD_BLOCK, ROPE_DIM)),
        mask=mask,
    )


# Initial settings from the MI455X (gfx1250) DSR1 MLPerf BF16 K benchmark:
# 128 local heads, nope=128, rope=64, strided kv_b_proj views. These are
# provisional defaults; compare with the existing copy helper before choosing
# the final implementation, and retune for other architectures/head layouts.
_PREFILL_K_CAT_MIN_TOKENS = 1024
_PREFILL_K_CAT_HEAD_BLOCK = 16
_PREFILL_K_CAT_NUM_WARPS = 1


def _prefill_k_cat_supported(k_nope: torch.Tensor, k_rope: torch.Tensor) -> bool:
    return (
        k_nope.is_cuda
        and k_nope.dim() == 3
        and k_rope.dim() == 3
        and k_nope.shape[0] >= _PREFILL_K_CAT_MIN_TOKENS
        and k_nope.shape[1] > 0
        and k_nope.shape[-1] > 0
        and k_rope.shape[-1] > 0
        and k_rope.shape[:2] == (k_nope.shape[0], 1)
        and k_rope.dtype == k_nope.dtype
        and k_rope.device == k_nope.device
        and k_nope.stride(-1) == 1
        and k_rope.stride(-1) == 1
        and triton.next_power_of_2(k_nope.shape[-1]) == k_nope.shape[-1]
        and triton.next_power_of_2(k_rope.shape[-1]) == k_rope.shape[-1]
    )


def _cat_prefill_k(k_nope: torch.Tensor, k_rope: torch.Tensor) -> torch.Tensor:
    """``torch.cat((k_nope, k_rope.expand(...)), -1)`` as one vectorized copy.

    The kernel takes k_nope [T, H, nope] and k_rope [T, 1, rope], with unit
    last strides, matching devices/dtypes and power-of-two feature widths.
    It returns the same bytes in one contiguous [T, H, nope + rope] allocation.
    Small batches and unsupported layouts use torch.cat with expanded RoPE.
    """
    if not _prefill_k_cat_supported(k_nope, k_rope):
        return torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)
    tokens, heads, nope_dim = k_nope.shape
    rope_dim = k_rope.shape[-1]
    k = torch.empty(
        (tokens, heads, nope_dim + rope_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    head_block = min(_PREFILL_K_CAT_HEAD_BLOCK, triton.next_power_of_2(heads))
    _prefill_k_cat_kernel[(tokens, triton.cdiv(heads, head_block))](
        k,
        k_nope,
        k_rope,
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        heads,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        HEAD_BLOCK=head_block,
        num_warps=_PREFILL_K_CAT_NUM_WARPS,
    )
    return k
