# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Shared MLA prefill K concatenation; imports only Torch and Triton."""

import functools

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


# Measured on MI455X (gfx1250), BF16 DSR1 K (nope=128 from the strided
# kv_b_proj view, rope=64), 16 and 128 heads, 1 to 10240 tokens. On the GPU
# the kernel beats torch.cat at every size (1.3 vs 3.4 us at one token, 65 vs
# 603 us at 10240 tokens x 128 heads), but a call costs about 9 us more host
# time (13.3 vs 4.3 us). It saves more than that from 32768 token-heads
# (16.6 us; 8.9 us at 16384), so it runs from there; below that, and on other
# arches (not measured; on gfx9 one warp is a wave64), each call site keeps
# what it did on main: torch.cat, or two copies with `copies=True`. Of head
# blocks 4..64 and 1..4 warps, head block 16 with one warp was the fastest at
# 10240 tokens.
_PREFILL_K_CAT_ARCH = "gfx1250"
_PREFILL_K_CAT_MIN_TOKEN_HEADS = 32768
_PREFILL_K_CAT_HEAD_BLOCK = 16
_PREFILL_K_CAT_NUM_WARPS = 1


@functools.cache
def _kernel_arch(device_index: int) -> bool:
    arch = torch.cuda.get_device_properties(device_index).gcnArchName
    return arch.split(":")[0] == _PREFILL_K_CAT_ARCH


def kernel_arch() -> bool:
    """Whether the current device is gfx1250, the arch this file was measured
    on. Call sites decide it once per layer and call `cat_prefill_k` only there;
    elsewhere they keep main's expression, with no per-call check."""
    return torch.cuda.is_available() and _kernel_arch(torch.cuda.current_device())


def _prefill_k_cat_supported(k_nope: torch.Tensor, k_rope: torch.Tensor) -> bool:
    nope_dim, rope_dim = k_nope.shape[-1], k_rope.shape[-1]
    return (
        k_nope.is_cuda
        and k_nope.dim() == 3
        and k_rope.dim() == 3
        and k_nope.shape[0] * k_nope.shape[1] >= _PREFILL_K_CAT_MIN_TOKEN_HEADS
        and _kernel_arch(k_nope.device.index)
        and nope_dim > 0
        and rope_dim > 0
        and k_rope.shape[:2] == (k_nope.shape[0], 1)
        and k_rope.dtype == k_nope.dtype
        and k_rope.device == k_nope.device
        and k_nope.stride(-1) == 1
        and k_rope.stride(-1) == 1
        and nope_dim & (nope_dim - 1) == 0
        and rope_dim & (rope_dim - 1) == 0
    )


def cat_prefill_k(
    k_nope: torch.Tensor, k_rope: torch.Tensor, copies: bool = False
) -> torch.Tensor:
    """K for MLA prefill attention: ``k_nope`` [T, H, nope] followed by
    ``k_rope`` [T, 1, rope] broadcast over the heads, in one new contiguous
    [T, H, nope + rope] tensor. The bytes are those of ``torch.cat``.

    One Triton copy on gfx1250 from ``_PREFILL_K_CAT_MIN_TOKEN_HEADS``
    token-heads, with unit last strides, one dtype and power-of-two widths.
    Everything else (fewer token-heads, other arches, CPU tensors, other
    layouts) is what the call site did before: ``torch.cat`` of the expanded
    ``k_rope``, or with ``copies`` the two slices of a new tensor written by
    two copies.
    """
    if not _prefill_k_cat_supported(k_nope, k_rope):
        if not copies:
            return torch.cat((k_nope, k_rope.expand(*k_nope.shape[:-1], -1)), dim=-1)
        nope_dim = k_nope.shape[-1]
        k = torch.empty(
            (*k_nope.shape[:-1], nope_dim + k_rope.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )
        k[..., :nope_dim] = k_nope
        k[..., nope_dim:] = k_rope
        return k
    tokens, heads, nope_dim = k_nope.shape
    rope_dim = k_rope.shape[-1]
    k = torch.empty(
        (tokens, heads, nope_dim + rope_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    head_block = min(_PREFILL_K_CAT_HEAD_BLOCK, 1 << (heads - 1).bit_length())
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
