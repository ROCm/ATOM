# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton oracle of the NVFP4 page write (tests only): quantize rows of K, V and the index
key into the pages of ``nvfp4_pages`` from bf16 rows.

Production writes the pages inside aiter's fused ``fused_qknorm_idxrqknorm``
(``kv_cache_dtype`` / ``index_cache_dtype`` ``"nvfp4"``); this file is what its bytes are
checked against (``test_minimax_m3_nvfp4_fused_write``), and it writes the pages the
read-path tests need. Before that op had an NVFP4 mode it was the production writer, after
aiter's norm/RoPE wrote bf16 rows into a scratch.

Quantization is the vLLM/aiter NVFP4 one, in fp32 throughout, per 16-element group
of a 128-wide row::

    scale  = e4m3_rne(min(amax * (1/6) * (1/gs), 448))        # the byte stored
    quant  = scale != 0 ? 1 / (scale * gs) : 0
    code   = e2m1 nearest-grid, ties to the even code, of x * quant  (sign = sign bit)

``gs`` is the layer's fp32 global scale (K, V; 1.0 for the index key, which has none).
The e2m1 rounding is written as seven comparisons instead of the hardware convert so
that it is exact against the torch reference (``minimax_m3_nvfp4_ref.quantize_nvfp4``)
and independent of the arch; tests pin it byte for byte.

A token is one (slot, head) row per side. Tokens with a negative slot are skipped.
Two tokens of one 8-token V octet share a 32-bit word of the page (a byte holds
two tokens' nibbles), so V data is written with atomic and/or on the word: the
write needs no ordering between tokens.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.minimax_m3.nvfp4_pages import (
    GROUP,
    GROUPS,
    HEAD_DIM,
    HEAD_SCALE_BYTES,
    HEAD_SIDE_BYTES,
    INDEX_DATA_PAGE_BYTES,
    INDEX_PAGE_BYTES,
    PAGE_TOKENS,
)

# Constants the kernels read (Triton takes only constexpr globals).
_ONE_SIXTH = tl.constexpr(0.1666666716337204)  # float32(1 / 6), as the torch reference
_E4M3_MAX = tl.constexpr(448.0)
_C_PAGE = tl.constexpr(PAGE_TOKENS)
_C_GROUP = tl.constexpr(GROUP)
_C_GROUPS = tl.constexpr(GROUPS)
_C_SIDE_BYTES = tl.constexpr(HEAD_SIDE_BYTES)
_C_DATA_BYTES = tl.constexpr(PAGE_TOKENS * HEAD_DIM // 2)  # a head's data region
_C_SCALE_BYTES = tl.constexpr(HEAD_SCALE_BYTES)
_C_INDEX_DATA = tl.constexpr(INDEX_DATA_PAGE_BYTES)


@triton.jit
def _e2m1_code(y):
    """e2m1 code (sign in bit 3) of an fp32 tile; nearest grid value, ties to the even
    code: 0, .5, 1, 1.5, 2, 3, 4, 6 are codes 0..7, the midpoints .25 .75 1.25 1.75 2.5
    3.5 5 go to 0 2 2 4 4 6 6."""
    a = tl.abs(y)
    mag = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    sign = (y.to(tl.int32, bitcast=True) >> 31) & 1
    return mag | (sign << 3)


@triton.jit
def _e4m3_byte(x):
    """e4m3fn byte of a non-negative fp32 tile in [0, 448], round to nearest even, in
    integer arithmetic: the same bytes on any arch and in the interpreter, which the
    hardware / library convert is not."""
    bits = x.to(tl.int32, bitcast=True)
    # normal range: drop 20 mantissa bits with RNE, rebias fp32 -> e4m3 (127 -> 7)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    # subnormal range (x < 2^-6): m * 2^-9, m = RNE(x * 512) by the 1.5 * 2^23 trick
    sub = ((x * 512.0 + 12582912.0) - 12582912.0).to(tl.int32)
    return tl.where(x >= 0.015625, normal, sub)


@triton.jit
def _e4m3_value(byte):
    """fp32 value of a positive e4m3fn byte (< 0x7F)."""
    e = byte >> 3
    m = byte & 7
    normal = ((e + 120) << 23) | (m << 20)
    return tl.where(
        e > 0, normal.to(tl.float32, bitcast=True), m.to(tl.float32) * 0.001953125
    )


@triton.jit
def _quant_row(row_ptr, gs):
    """Quantize one 128-wide bf16/fp32 row: the e2m1 codes of the even and the odd
    element of every pair, ``[8 groups, 8 pairs]`` each, and the 8 e4m3 scale bytes."""
    g = tl.arange(0, _C_GROUPS)
    p = tl.arange(0, 8)
    offs = g[:, None] * _C_GROUP + p[None, :] * 2
    xe = tl.load(row_ptr + offs).to(tl.float32)
    xo = tl.load(row_ptr + offs + 1).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
    inv_gs = 1.0 / gs
    sf = tl.minimum(amax * _ONE_SIXTH * inv_gs, _E4M3_MAX)
    sbyte = _e4m3_byte(sf)
    sv = _e4m3_value(sbyte)
    quant = tl.where(sv != 0.0, 1.0 / (sv * gs), 0.0)
    ce = _e2m1_code(xe * quant[:, None])
    co = _e2m1_code(xo * quant[:, None])
    return ce, co, sbyte.to(tl.uint8)


@triton.jit
def _nvfp4_kv_insert_kernel(
    x_ptr,  # bf16 [tokens, Hkv, 128] rows of K (side 0) / V (side 1)
    x_stride_t,
    x_stride_h,
    slot_ptr,  # int64 [tokens]
    page_ptr,  # uint8 pages
    page_stride,  # bytes between pages
    gs_ptr,  # fp32 [1] global scale of this side
    NUM_KV_HEADS: tl.constexpr,
    SIDE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    blk = slot // _C_PAGE
    pos = slot % _C_PAGE
    gs = tl.load(gs_ptr)
    ce, co, sbyte = _quant_row(x_ptr + t * x_stride_t + h * x_stride_h, gs)
    page = page_ptr + blk.to(tl.int64) * page_stride
    side_base = SIDE * NUM_KV_HEADS * _C_SIDE_BYTES
    data_base = side_base + h * _C_DATA_BYTES
    scale_base = side_base + NUM_KV_HEADS * _C_DATA_BYTES + h * _C_SCALE_BYTES
    g = tl.arange(0, _C_GROUPS)
    p = tl.arange(0, 8)
    if SIDE == 0:
        # K: data token-major (low nibble = even element), scales in MFMA lane order
        byte = (ce | (co << 4)).to(tl.uint8)
        tl.store(page + data_base + pos * 64 + g[:, None] * 8 + p[None, :], byte)
        r = pos % 32
        n = 4 * (r // 8) + r % 4
        soff = ((((pos // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2
        tl.store(page + scale_base + soff, sbyte)
    else:
        # V: one 32-bit word per (8-token octet, element) holds that octet's 8 nibbles
        words = page.to(tl.pointer_type(tl.int32))
        word0 = (data_base + (pos // 8) * 512) // 4
        shift = ((pos % 8) // 2) * 8 + (pos % 2) * 4
        keep = ~(0xF << shift)
        d_even = g[:, None] * _C_GROUP + p[None, :] * 2
        tl.atomic_and(words + word0 + d_even, keep)
        tl.atomic_and(words + word0 + d_even + 1, keep)
        tl.atomic_or(words + word0 + d_even, ce << shift)
        tl.atomic_or(words + word0 + d_even + 1, co << shift)
        tl.store(page + scale_base + g * _C_PAGE + pos, sbyte)


@triton.jit
def _nvfp4_index_insert_kernel(
    x_ptr,  # bf16 [tokens, 128]
    x_stride_t,
    slot_ptr,  # int64 [tokens]
    page_ptr,  # uint8 index pages
    page_stride,  # bytes between pages
):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    blk = slot // _C_PAGE
    pos = slot % _C_PAGE
    ce, co, sbyte = _quant_row(x_ptr + t * x_stride_t, 1.0)
    page = page_ptr + blk.to(tl.int64) * page_stride
    g = tl.arange(0, _C_GROUPS)
    p = tl.arange(0, 8)
    byte = (ce | (co << 4)).to(tl.uint8)
    tl.store(page + pos * 64 + g[:, None] * 8 + p[None, :], byte)
    tl.store(page + _C_INDEX_DATA + pos * _C_GROUPS + g, sbyte)


def nvfp4_insert_kv(
    k: torch.Tensor,  # bf16 [tokens, Hkv, 128]
    v: torch.Tensor,
    kv_pages: torch.Tensor,  # uint8 [pages, 2 * Hkv * 9216] (page stride may be wider)
    slot_mapping: torch.Tensor,  # int64 [>= tokens]; < 0 skips
    k_scale: torch.Tensor,  # fp32 [1] global scales
    v_scale: torch.Tensor,
) -> None:
    """Quantize K and V rows into their pages (every kv head of the token)."""
    tokens, heads, dim = k.shape
    if dim != HEAD_DIM or v.shape != k.shape:
        raise ValueError(f"K / V rows are [tokens, Hkv, {HEAD_DIM}], got {k.shape}")
    if kv_pages.dtype != torch.uint8 or kv_pages.stride(-1) != 1:
        raise ValueError("NVFP4 KV pages are uint8 with a contiguous byte axis")
    if kv_pages.shape[-1] != 2 * heads * HEAD_SIDE_BYTES or kv_pages.stride(0) % 4:
        raise ValueError(
            f"NVFP4 KV pages are {2 * heads * HEAD_SIDE_BYTES} B for {heads} kv heads"
            f" (4 B aligned stride), got {tuple(kv_pages.shape)} {kv_pages.stride()}"
        )
    if tokens == 0:
        return
    for side, (x, gs) in enumerate(((k, k_scale), (v, v_scale))):
        _nvfp4_kv_insert_kernel[(tokens, heads)](
            x,
            x.stride(0),
            x.stride(1),
            slot_mapping,
            kv_pages,
            kv_pages.stride(0),
            gs,
            NUM_KV_HEADS=heads,
            SIDE=side,
        )


def nvfp4_insert_index(
    keys: torch.Tensor,  # bf16 [tokens, 128]
    index_pages: torch.Tensor,  # uint8 [pages, 9216]
    slot_mapping: torch.Tensor,  # int64 [>= tokens]; < 0 skips
) -> None:
    """Quantize index-key rows into their pages."""
    if keys.shape[-1] != HEAD_DIM or keys.dim() != 2:
        raise ValueError(f"index keys are [tokens, {HEAD_DIM}], got {keys.shape}")
    if index_pages.dtype != torch.uint8 or index_pages.shape[-1] != INDEX_PAGE_BYTES:
        raise ValueError(f"NVFP4 index pages are uint8 [pages, {INDEX_PAGE_BYTES}]")
    if keys.shape[0] == 0:
        return
    _nvfp4_index_insert_kernel[(keys.shape[0],)](
        keys, keys.stride(0), slot_mapping, index_pages, index_pages.stride(0)
    )
