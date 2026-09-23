# SPDX-License-Identifier: MIT
"""Native Triton sparse prefill for V4's two-buffer FP8 KV layout.

The NoPE plane stores 448 FP8 values followed by fourteen duplicated E8M0
scale bytes and padding in a 512-byte row.  The RoPE plane stores the remaining
64 values in BF16.  This kernel consumes that representation directly for both
the paged prefix and the flat extend source; it never materialises a BF16 KV
tensor.

This is deliberately exposed as a separately tunable candidate.  Production
dispatch is gated in :mod:`paged_prefill` until the long-context matrix has
beaten the AITER OPUS implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter import dtypes

from atom.model_ops.v4_kernels.paged_decode_fp8_triton import (
    _bf16_v_group128_dot_asm,
    _e8m0_scale,
    _fp8_v_group_dot,
    _load_bf16_v_group_asm,
    _scaled_e4m3x4_to_bf16_asm,
)
from atom.model_ops.v4_kernels.pool_index import row_offset
from atom.model_ops.v4_kernels.v4_quant import (
    V4_DIM_NOPE,
    V4_DIM_QK,
    V4_DIM_QK_PACKED,
    V4_DIM_ROPE,
    V4_PACK_OFF_SCALE,
    V4_TILE,
)

_LOG2E = 1.4426950408889634
_SCALE_BYTES = V4_DIM_NOPE // 32
_COMPILED_PREFILL_KERNELS: dict[tuple[object, ...], object] = {}


@triton.jit
def _unpack_e4m3x4(raw_word, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Bitcast packed uint32 words to four row-major E4M3 values."""
    value0 = (raw_word & 0xFF).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    value1 = ((raw_word >> 8) & 0xFF).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    value2 = ((raw_word >> 16) & 0xFF).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    value3 = (raw_word >> 24).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    return tl.reshape(
        tl.join(tl.join(value0, value2), tl.join(value1, value3)),
        (ROWS, COLS),
    )


@triton.jit
def _split_word_quarters(raw_word, ROWS: tl.constexpr):
    """Split a [ROWS,128] packed-word tile into four contiguous quarters."""
    halves = tl.permute(tl.reshape(raw_word, (ROWS, 2, 64)), (0, 2, 1))
    words01, words23 = tl.split(halves)
    pair01 = tl.permute(tl.reshape(words01, (ROWS, 2, 32)), (0, 2, 1))
    pair23 = tl.permute(tl.reshape(words23, (ROWS, 2, 32)), (0, 2, 1))
    words0, words1 = tl.split(pair01)
    words2, words3 = tl.split(pair23)
    return words0, words1, words2, words3


@triton.jit
def _split_fp8_quarters(raw, ROWS: tl.constexpr):
    """Split a [ROWS,512] FP8 tile into four contiguous 128-wide tiles."""
    halves = tl.permute(tl.reshape(raw, (ROWS, 2, 256)), (0, 2, 1))
    raw01, raw23 = tl.split(halves)
    pair01 = tl.permute(tl.reshape(raw01, (ROWS, 2, 128)), (0, 2, 1))
    pair23 = tl.permute(tl.reshape(raw23, (ROWS, 2, 128)), (0, 2, 1))
    raw0, raw1 = tl.split(pair01)
    raw2, raw3 = tl.split(pair23)
    return raw0, raw1, raw2, raw3


@triton.jit
def _split_fp8_halves(raw, ROWS: tl.constexpr):
    """Split a [ROWS,128] FP8 tile into two contiguous 64-wide tiles."""
    pair = tl.permute(tl.reshape(raw, (ROWS, 2, 64)), (0, 2, 1))
    return tl.split(pair)


@triton.jit
def _fp8_v_group_dot_from_tile(
    p,
    raw,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
):
    scale_byte = tl.load(
        kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 2 * GROUP,
        mask=valid,
        other=0,
    )
    p_scaled = (p * _e8m0_scale(scale_byte)[None, :]).to(tl.bfloat16)
    return tl.dot_scaled(
        p_scaled,
        None,
        "bf16",
        raw,
        None,
        "e4m3",
        out_dtype=tl.float32,
    )


@triton.jit
def _bf16_v_group128_dot_from_raw(
    p,
    raw_word,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    scale_offs = tl.arange(0, 2)
    scale_pair = tl.load(
        kv_packed_u8_ptr
        + slot[:, None] * kv_stride_n
        + PACK_OFF_SCALE
        + 2 * (GROUP * 2 + scale_offs[None, :]),
        mask=valid[:, None] & (scale_offs < _SCALE_BYTES)[None, :],
        other=0,
    )
    scale_byte = tl.reshape(
        tl.broadcast_to(
            scale_pair[:, :, None],
            (slot.shape[0], 2, TILE // 4),
        ),
        (slot.shape[0], 2 * TILE // 4),
    )
    value = _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_byte,
        ROWS=slot.shape[0],
        COLS=2 * TILE,
    )
    return tl.dot(p.to(tl.bfloat16), value)


@triton.jit
def _bf16_v_group128_acc_asm(
    p,
    acc,
    alpha,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Native-dequantize two V groups and accumulate without a dot temporary."""
    word_offs = tl.arange(0, 2 * TILE // 4)
    raw_word_ptr = (
        kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * 2 * TILE
    ).to(tl.pointer_type(tl.uint32))
    raw_word = tl.load(
        raw_word_ptr + word_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_offs = tl.arange(0, 2)
    scale_pair = tl.load(
        kv_packed_u8_ptr
        + slot[:, None] * kv_stride_n
        + PACK_OFF_SCALE
        + 2 * (GROUP * 2 + scale_offs[None, :]),
        mask=valid[:, None],
        other=0,
    )
    scale_byte = tl.reshape(
        tl.broadcast_to(
            scale_pair[:, :, None],
            (slot.shape[0], 2, TILE // 4),
        ),
        (slot.shape[0], 2 * TILE // 4),
    )
    value = _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_byte,
        ROWS=slot.shape[0],
        COLS=2 * TILE,
    )
    acc = acc * alpha[:, None]
    return tl.dot(p.to(tl.bfloat16), value, acc)


@triton.jit
def _bf16_v_group128_dot_cached_asm(
    p,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """Native-dequantize two V groups with an explicit cache policy."""
    word_offs = tl.arange(0, 2 * TILE // 4)
    raw_word_ptr = (
        kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * 2 * TILE
    ).to(tl.pointer_type(tl.uint32))
    raw_word = tl.load(
        raw_word_ptr + word_offs[None, :],
        mask=valid[:, None],
        other=0,
        cache_modifier=CACHE_MODIFIER,
    )
    scale_offs = tl.arange(0, 2)
    scale_pair = tl.load(
        kv_packed_u8_ptr
        + slot[:, None] * kv_stride_n
        + PACK_OFF_SCALE
        + 2 * (GROUP * 2 + scale_offs[None, :]),
        mask=valid[:, None],
        other=0,
        cache_modifier=CACHE_MODIFIER,
    )
    scale_byte = tl.reshape(
        tl.broadcast_to(
            scale_pair[:, :, None],
            (slot.shape[0], 2, TILE // 4),
        ),
        (slot.shape[0], 2 * TILE // 4),
    )
    value = _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_byte,
        ROWS=slot.shape[0],
        COLS=2 * TILE,
    )
    return tl.dot(p.to(tl.bfloat16), value)


@triton.jit
def _load_bf16_v_group_cached_asm(
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    word_offs = tl.arange(0, TILE // 4)
    scale_byte = tl.load(
        kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 2 * GROUP,
        mask=valid,
        other=0,
        cache_modifier=CACHE_MODIFIER,
    )
    raw_word_ptr = (kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * TILE).to(
        tl.pointer_type(tl.uint32)
    )
    raw_word = tl.load(
        raw_word_ptr + word_offs[None, :],
        mask=valid[:, None],
        other=0,
        cache_modifier=CACHE_MODIFIER,
    )
    scale_full = tl.broadcast_to(
        scale_byte[:, None],
        (slot.shape[0], TILE // 4),
    )
    return _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_full,
        ROWS=slot.shape[0],
        COLS=TILE,
    )


@triton.jit
def _bf16_v_group128_dot_with_scale_asm(
    p,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    scale0,
    scale1,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
):
    """Native-dequantize two V groups using scales retained from QK."""
    word_offs = tl.arange(0, 2 * TILE // 4)
    raw_word_ptr = (
        kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * 2 * TILE
    ).to(tl.pointer_type(tl.uint32))
    raw_word = tl.load(
        raw_word_ptr + word_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_pair = tl.join(scale0, scale1)
    scale_byte = tl.reshape(
        tl.broadcast_to(
            scale_pair[:, :, None],
            (slot.shape[0], 2, TILE // 4),
        ),
        (slot.shape[0], 2 * TILE // 4),
    )
    value = _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_byte,
        ROWS=slot.shape[0],
        COLS=2 * TILE,
    )
    return tl.dot(p.to(tl.bfloat16), value)


@triton.jit
def _load_bf16_v_group_with_scale_asm(
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    scale_byte,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
):
    word_offs = tl.arange(0, TILE // 4)
    raw_word_ptr = (kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * TILE).to(
        tl.pointer_type(tl.uint32)
    )
    raw_word = tl.load(
        raw_word_ptr + word_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_full = tl.broadcast_to(
        scale_byte[:, None],
        (slot.shape[0], TILE // 4),
    )
    return _scaled_e4m3x4_to_bf16_asm(
        raw_word,
        scale_full,
        ROWS=slot.shape[0],
        COLS=TILE,
    )


@triton.jit
def _scaled_e4m3x2_to_bf16_asm(
    raw_half,
    scale_byte,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """Convert one packed FP8 pair with one native gfx950 instruction."""
    scale = _e8m0_scale(scale_byte)
    converted = tl.inline_asm_elementwise(
        asm="""
        v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[0,0];
        """,
        constraints="=&v,v,v",
        args=[raw_half, scale],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    value0 = (converted & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    value1 = (converted >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.reshape(tl.join(value0, value1), (ROWS, COLS))


@triton.jit
def _bf16_v_group128_dot_pair_asm(
    p,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    half_offs = tl.arange(0, 2 * TILE // 2)
    raw_half_ptr = (
        kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * 2 * TILE
    ).to(tl.pointer_type(tl.uint16))
    raw_half = tl.load(
        raw_half_ptr + half_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_offs = tl.arange(0, 2)
    scale_pair = tl.load(
        kv_packed_u8_ptr
        + slot[:, None] * kv_stride_n
        + PACK_OFF_SCALE
        + 2 * (GROUP * 2 + scale_offs[None, :]),
        mask=valid[:, None],
        other=0,
    )
    scale_byte = tl.reshape(
        tl.broadcast_to(
            scale_pair[:, :, None],
            (slot.shape[0], 2, TILE // 2),
        ),
        (slot.shape[0], 2 * TILE // 2),
    )
    value = _scaled_e4m3x2_to_bf16_asm(
        raw_half,
        scale_byte,
        ROWS=slot.shape[0],
        COLS=2 * TILE,
    )
    return tl.dot(p.to(tl.bfloat16), value)


@triton.jit
def _load_bf16_v_group_pair_asm(
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    half_offs = tl.arange(0, TILE // 2)
    scale_byte = tl.load(
        kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 2 * GROUP,
        mask=valid,
        other=0,
    )
    raw_half_ptr = (kv_packed_u8_ptr + slot[:, None] * kv_stride_n + GROUP * TILE).to(
        tl.pointer_type(tl.uint16)
    )
    raw_half = tl.load(
        raw_half_ptr + half_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale_full = tl.broadcast_to(
        scale_byte[:, None],
        (slot.shape[0], TILE // 2),
    )
    return _scaled_e4m3x2_to_bf16_asm(
        raw_half,
        scale_full,
        ROWS=slot.shape[0],
        COLS=TILE,
    )


@triton.jit
def _bf16_v_full_dot_from_groups(
    p,
    kv_rope,
    value0,
    value1,
    value2,
    value3,
    value4,
    value5,
    value6,
):
    value01 = tl.cat(value0, value1, dim=1)
    value23 = tl.cat(value2, value3, dim=1)
    value45 = tl.cat(value4, value5, dim=1)
    value67 = tl.cat(value6, kv_rope, dim=1)
    value03 = tl.cat(value01, value23, dim=1)
    value47 = tl.cat(value45, value67, dim=1)
    value = tl.cat(value03, value47, dim=1)
    return tl.dot(p.to(tl.bfloat16), value)


@triton.jit
def _bf16_v_full_dot_asm(
    p,
    kv_rope,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    TILE: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """Stage the complete 512-wide BF16 V operand for one P@V dot."""
    value0 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=0,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value1 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=1,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value2 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=2,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value3 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=3,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value4 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=4,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value5 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=5,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    value6 = _load_bf16_v_group_cached_asm(
        kv_packed_u8_ptr,
        slot,
        valid,
        kv_stride_n,
        GROUP=6,
        TILE=TILE,
        PACK_OFF_SCALE=PACK_OFF_SCALE,
        CACHE_MODIFIER=CACHE_MODIFIER,
    )
    return _bf16_v_full_dot_from_groups(
        p,
        kv_rope,
        value0,
        value1,
        value2,
        value3,
        value4,
        value5,
        value6,
    )


@triton.jit
def _fp8_prefill_update(
    q_nope_raw,
    q_scale,
    q_rope,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    kv_rope_ptr,
    slot,
    valid,
    h_mask,
    m_i,
    l_i,
    acc0,
    acc1,
    acc2,
    acc3,
    acc4,
    acc5,
    acc6,
    acc7,
    kv_stride_n: tl.constexpr,
    kv_rope_stride_n: tl.constexpr,
    qk_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    USE_MXFP8_V: tl.constexpr,
    USE_BF16_ACC: tl.constexpr,
    PACKED_QK_LOAD: tl.constexpr,
    KEEP_KV_CACHE: tl.constexpr,
    REUSE_KV_RAW: tl.constexpr,
    REUSE_KV_FP8: tl.constexpr,
    COALESCED_V_LOAD: tl.constexpr,
    FUSED_PV_ACC: tl.constexpr,
    V_CACHE_MODIFIER: tl.constexpr,
    REUSE_V_SCALE: tl.constexpr,
    PAIRWISE_BF16_V: tl.constexpr,
    FULL_BF16_V: tl.constexpr,
):
    """Fold one KV tile into the shared online-softmax state."""
    d_offs = tl.arange(0, BLOCK_D)
    nope_mask = d_offs < NOPE
    rope_offs = tl.arange(0, ROPE)

    if PACKED_QK_LOAD or REUSE_KV_RAW:
        word_offs = tl.arange(0, BLOCK_D // 4)
        raw_word_ptr = (kv_packed_u8_ptr + row_offset(slot, kv_stride_n)[:, None]).to(
            tl.pointer_type(tl.uint32)
        )
        kv_raw_word = tl.load(
            raw_word_ptr + word_offs[None, :],
            mask=valid[:, None] & (word_offs < NOPE // 4)[None, :],
            other=0,
        )
        kv_nope_raw = _unpack_e4m3x4(
            kv_raw_word,
            ROWS=BLOCK_K,
            COLS=BLOCK_D,
        )
    else:
        if KEEP_KV_CACHE:
            kv_nope_raw = tl.load(
                kv_packed_ptr
                + row_offset(slot, kv_stride_n)[:, None]
                + d_offs[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
        else:
            kv_nope_raw = tl.load(
                kv_packed_ptr
                + row_offset(slot, kv_stride_n)[:, None]
                + d_offs[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0.0,
            )
    scale_offs = tl.arange(0, 16)
    kv_scale = tl.load(
        kv_packed_u8_ptr
        + row_offset(slot, kv_stride_n)[:, None]
        + PACK_OFF_SCALE
        + scale_offs[None, :],
        mask=valid[:, None],
        other=0,
    )
    if REUSE_V_SCALE:
        scale_word_ptr = (kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE).to(
            tl.pointer_type(tl.uint32)
        )
        scale_word0 = tl.load(scale_word_ptr, mask=valid, other=0)
        scale_word1 = tl.load(scale_word_ptr + 1, mask=valid, other=0)
        scale_word2 = tl.load(scale_word_ptr + 2, mask=valid, other=0)
        scale_word3 = tl.load(scale_word_ptr + 3, mask=valid, other=0)
        scale0 = (scale_word0 & 0xFF).to(tl.uint8)
        scale1 = ((scale_word0 >> 16) & 0xFF).to(tl.uint8)
        scale2 = (scale_word1 & 0xFF).to(tl.uint8)
        scale3 = ((scale_word1 >> 16) & 0xFF).to(tl.uint8)
        scale4 = (scale_word2 & 0xFF).to(tl.uint8)
        scale5 = ((scale_word2 >> 16) & 0xFF).to(tl.uint8)
        scale6 = (scale_word3 & 0xFF).to(tl.uint8)
    scores = tl.dot_scaled(
        q_nope_raw,
        q_scale,
        "e4m3",
        tl.trans(kv_nope_raw),
        kv_scale,
        "e4m3",
        out_dtype=tl.float32,
    )
    kv_rope = tl.load(
        kv_rope_ptr + row_offset(slot, kv_rope_stride_n)[:, None] + rope_offs[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    scores += tl.dot(q_rope, tl.trans(kv_rope))
    scores *= qk_scale

    neg_large = -3.4028234663852886e38
    score_mask = h_mask[:, None] & valid[None, :]
    scores = tl.where(score_mask, scores, neg_large)
    m_block = tl.max(scores, axis=1)
    m_new = tl.maximum(m_i, m_block)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(scores - m_new[:, None])
    p = tl.where(score_mask, p, 0.0)
    l_new = l_i * alpha + tl.sum(p, axis=1)

    if USE_MXFP8_V:
        # Keep the packed V operand native.  Its E8M0 factor is folded into P,
        # so the matrix instruction consumes BF16 probabilities x FP8 values.
        acc0 = acc0 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=0,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc1 = acc1 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=1,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc2 = acc2 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=2,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc3 = acc3 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=3,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc4 = acc4 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=4,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc5 = acc5 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=5,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc6 = acc6 * alpha[:, None] + _fp8_v_group_dot(
            p,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=6,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc7 = acc7 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_rope)
    elif REUSE_KV_FP8:
        quarter0, quarter1, quarter2, quarter3 = _split_fp8_quarters(
            kv_nope_raw,
            ROWS=BLOCK_K,
        )
        raw0, raw1 = _split_fp8_halves(quarter0, ROWS=BLOCK_K)
        raw2, raw3 = _split_fp8_halves(quarter1, ROWS=BLOCK_K)
        raw4, raw5 = _split_fp8_halves(quarter2, ROWS=BLOCK_K)
        raw6, _ = _split_fp8_halves(quarter3, ROWS=BLOCK_K)
        acc0 = acc0 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw0,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=0,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc1 = acc1 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw1,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=1,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc2 = acc2 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw2,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=2,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc3 = acc3 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw3,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=3,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc4 = acc4 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw4,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=4,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc5 = acc5 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw5,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=5,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc6 = acc6 * alpha[:, None] + _fp8_v_group_dot_from_tile(
            p,
            raw6,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=6,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
        )
        acc7 = acc7 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_rope)
    elif REUSE_KV_RAW or COALESCED_V_LOAD:
        if COALESCED_V_LOAD:
            word_offs = tl.arange(0, BLOCK_D // 4)
            raw_word_ptr = (
                kv_packed_u8_ptr + row_offset(slot, kv_stride_n)[:, None]
            ).to(tl.pointer_type(tl.uint32))
            kv_raw_word = tl.load(
                raw_word_ptr + word_offs[None, :],
                mask=valid[:, None] & (word_offs < NOPE // 4)[None, :],
                other=0,
            )
        words0, words1, words2, words3 = _split_word_quarters(
            kv_raw_word,
            ROWS=BLOCK_K,
        )
        acc0 = acc0 * alpha[:, None] + _bf16_v_group128_dot_from_raw(
            p,
            words0,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=0,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc1 = acc1 * alpha[:, None] + _bf16_v_group128_dot_from_raw(
            p,
            words1,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=1,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        acc2 = acc2 * alpha[:, None] + _bf16_v_group128_dot_from_raw(
            p,
            words2,
            kv_packed_u8_ptr,
            slot,
            valid,
            kv_stride_n,
            GROUP=2,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            TILE=TILE,
        )
        scale_tail = tl.load(
            kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 12,
            mask=valid,
            other=0,
        )
        scale_tail_full = tl.broadcast_to(
            scale_tail[:, None],
            (slot.shape[0], TILE // 4),
        )
        tail_pair = tl.permute(
            tl.reshape(words3, (BLOCK_K, 2, TILE // 4)),
            (0, 2, 1),
        )
        tail_words, _ = tl.split(tail_pair)
        kv_nope_tail = _scaled_e4m3x4_to_bf16_asm(
            tail_words,
            scale_tail_full,
            ROWS=BLOCK_K,
            COLS=TILE,
        )
        kv_tail = tl.cat(kv_nope_tail, kv_rope, dim=1)
        acc3 = acc3 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_tail)
    else:
        # Four 128-wide accumulators keep the 512-wide value result out of one
        # monolithic live range. The final group is NoPE[384:448] + RoPE[0:64].
        if FULL_BF16_V:
            acc01 = tl.cat(acc0, acc1, dim=1)
            acc23 = tl.cat(acc2, acc3, dim=1)
            acc_full = tl.cat(acc01, acc23, dim=1)
            acc_full = acc_full * alpha[:, None] + _bf16_v_full_dot_asm(
                p,
                kv_rope,
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                TILE=TILE,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                CACHE_MODIFIER=V_CACHE_MODIFIER,
            )
            acc0, acc1, acc2, acc3 = _split_fp8_quarters(
                acc_full,
                ROWS=BLOCK_H,
            )
        elif FUSED_PV_ACC:
            acc0 = _bf16_v_group128_acc_asm(
                p,
                acc0,
                alpha,
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=0,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
            )
            acc1 = _bf16_v_group128_acc_asm(
                p,
                acc1,
                alpha,
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=1,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
            )
            acc2 = _bf16_v_group128_acc_asm(
                p,
                acc2,
                alpha,
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=2,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
            )
        else:
            if PAIRWISE_BF16_V:
                acc0 = acc0 * alpha[:, None] + _bf16_v_group128_dot_pair_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=0,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
                acc1 = acc1 * alpha[:, None] + _bf16_v_group128_dot_pair_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=1,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
                acc2 = acc2 * alpha[:, None] + _bf16_v_group128_dot_pair_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=2,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
            elif REUSE_V_SCALE:
                acc0 = acc0 * alpha[:, None] + _bf16_v_group128_dot_with_scale_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    scale0,
                    scale1,
                    GROUP=0,
                    TILE=TILE,
                )
                acc1 = acc1 * alpha[:, None] + _bf16_v_group128_dot_with_scale_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    scale2,
                    scale3,
                    GROUP=1,
                    TILE=TILE,
                )
                acc2 = acc2 * alpha[:, None] + _bf16_v_group128_dot_with_scale_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    scale4,
                    scale5,
                    GROUP=2,
                    TILE=TILE,
                )
            elif V_CACHE_MODIFIER == "":
                acc0 = acc0 * alpha[:, None] + _bf16_v_group128_dot_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=0,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
                acc1 = acc1 * alpha[:, None] + _bf16_v_group128_dot_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=1,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
                acc2 = acc2 * alpha[:, None] + _bf16_v_group128_dot_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=2,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                )
            else:
                acc0 = acc0 * alpha[:, None] + _bf16_v_group128_dot_cached_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=0,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                    CACHE_MODIFIER=V_CACHE_MODIFIER,
                )
                acc1 = acc1 * alpha[:, None] + _bf16_v_group128_dot_cached_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=1,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                    CACHE_MODIFIER=V_CACHE_MODIFIER,
                )
                acc2 = acc2 * alpha[:, None] + _bf16_v_group128_dot_cached_asm(
                    p,
                    kv_packed_u8_ptr,
                    slot,
                    valid,
                    kv_stride_n,
                    GROUP=2,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    TILE=TILE,
                    CACHE_MODIFIER=V_CACHE_MODIFIER,
                )
        if FULL_BF16_V:
            kv_nope_tail = tl.zeros((BLOCK_K, TILE), dtype=tl.bfloat16)
        elif PAIRWISE_BF16_V:
            kv_nope_tail = _load_bf16_v_group_pair_asm(
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=6,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
            )
        elif REUSE_V_SCALE:
            kv_nope_tail = _load_bf16_v_group_with_scale_asm(
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                scale6,
                GROUP=6,
                TILE=TILE,
            )
        elif V_CACHE_MODIFIER == "":
            kv_nope_tail = _load_bf16_v_group_asm(
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=6,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
            )
        else:
            kv_nope_tail = _load_bf16_v_group_cached_asm(
                kv_packed_u8_ptr,
                slot,
                valid,
                kv_stride_n,
                GROUP=6,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                TILE=TILE,
                CACHE_MODIFIER=V_CACHE_MODIFIER,
            )
        if not FULL_BF16_V:
            kv_tail = tl.cat(kv_nope_tail, kv_rope, dim=1)
            if FUSED_PV_ACC:
                acc3 = acc3 * alpha[:, None]
                acc3 = tl.dot(p.to(tl.bfloat16), kv_tail, acc3)
            else:
                acc3 = acc3 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_tail)
    if USE_BF16_ACC:
        acc0 = acc0.to(tl.bfloat16)
        acc1 = acc1.to(tl.bfloat16)
        acc2 = acc2.to(tl.bfloat16)
        acc3 = acc3.to(tl.bfloat16)
        acc4 = acc4.to(tl.bfloat16)
        acc5 = acc5.to(tl.bfloat16)
        acc6 = acc6.to(tl.bfloat16)
        acc7 = acc7.to(tl.bfloat16)
    return m_new, l_new, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7


@triton.jit
def _fp8_prefill_no_sentinel_span(
    q_nope_raw,
    q_scale,
    q_rope,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    kv_rope_ptr,
    kv_indices_ptr,
    span_start,
    span_len,
    h_mask,
    m_i,
    l_i,
    acc0,
    acc1,
    acc2,
    acc3,
    acc4,
    acc5,
    acc6,
    acc7,
    kv_stride_n: tl.constexpr,
    kv_rope_stride_n: tl.constexpr,
    qk_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    USE_MXFP8_V: tl.constexpr,
    USE_BF16_ACC: tl.constexpr,
    PACKED_QK_LOAD: tl.constexpr,
    KEEP_KV_CACHE: tl.constexpr,
    REUSE_KV_RAW: tl.constexpr,
    REUSE_KV_FP8: tl.constexpr,
    COALESCED_V_LOAD: tl.constexpr,
    FUSED_PV_ACC: tl.constexpr,
    V_CACHE_MODIFIER: tl.constexpr,
    REUSE_V_SCALE: tl.constexpr,
    PAIRWISE_BF16_V: tl.constexpr,
    FULL_BF16_V: tl.constexpr,
):
    """Process one sentinel-free CSR row with an unmasked main loop.

    ``span_len // BLOCK_K`` tiles take the same predicate-free path as the
    aligned benchmark specialization.  Only the final partial tile uses a
    mask, and it uses a smaller compile-time K tile so that the tail does not
    restore the main loop's previous register pressure.
    """
    main_offs = tl.arange(0, BLOCK_K)
    main_tiles = span_len // BLOCK_K
    for j in tl.range(0, main_tiles, num_stages=PIPE_STAGES):
        slot = tl.load(kv_indices_ptr + span_start + j * BLOCK_K + main_offs).to(
            tl.int64
        )
        valid = tl.full((BLOCK_K,), True, dtype=tl.int1)
        m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _fp8_prefill_update(
            q_nope_raw,
            q_scale,
            q_rope,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            kv_rope_ptr,
            slot,
            valid,
            h_mask,
            m_i,
            l_i,
            acc0,
            acc1,
            acc2,
            acc3,
            acc4,
            acc5,
            acc6,
            acc7,
            kv_stride_n=kv_stride_n,
            kv_rope_stride_n=kv_rope_stride_n,
            qk_scale=qk_scale,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            NOPE=NOPE,
            ROPE=ROPE,
            TILE=TILE,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            SCALE_BYTES=SCALE_BYTES,
            USE_MXFP8_V=USE_MXFP8_V,
            USE_BF16_ACC=USE_BF16_ACC,
            PACKED_QK_LOAD=PACKED_QK_LOAD,
            KEEP_KV_CACHE=KEEP_KV_CACHE,
            REUSE_KV_RAW=REUSE_KV_RAW,
            REUSE_KV_FP8=REUSE_KV_FP8,
            COALESCED_V_LOAD=COALESCED_V_LOAD,
            FUSED_PV_ACC=FUSED_PV_ACC,
            V_CACHE_MODIFIER=V_CACHE_MODIFIER,
            REUSE_V_SCALE=REUSE_V_SCALE,
            PAIRWISE_BF16_V=PAIRWISE_BF16_V,
            FULL_BF16_V=FULL_BF16_V,
        )

    tail_start = main_tiles * BLOCK_K
    tail_len = span_len - tail_start
    tail_offs = tl.arange(0, TAIL_BLOCK_K)
    for j in tl.range(0, tl.cdiv(tail_len, TAIL_BLOCK_K), num_stages=1):
        tail_pos = j * TAIL_BLOCK_K + tail_offs
        valid = tail_pos < tail_len
        slot = tl.load(
            kv_indices_ptr + span_start + tail_start + tail_pos,
            mask=valid,
            other=0,
        ).to(tl.int64)
        m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _fp8_prefill_update(
            q_nope_raw,
            q_scale,
            q_rope,
            kv_packed_ptr,
            kv_packed_u8_ptr,
            kv_rope_ptr,
            slot,
            valid,
            h_mask,
            m_i,
            l_i,
            acc0,
            acc1,
            acc2,
            acc3,
            acc4,
            acc5,
            acc6,
            acc7,
            kv_stride_n=kv_stride_n,
            kv_rope_stride_n=kv_rope_stride_n,
            qk_scale=qk_scale,
            BLOCK_H=BLOCK_H,
            BLOCK_K=TAIL_BLOCK_K,
            BLOCK_D=BLOCK_D,
            NOPE=NOPE,
            ROPE=ROPE,
            TILE=TILE,
            PACK_OFF_SCALE=PACK_OFF_SCALE,
            SCALE_BYTES=SCALE_BYTES,
            USE_MXFP8_V=USE_MXFP8_V,
            USE_BF16_ACC=USE_BF16_ACC,
            PACKED_QK_LOAD=PACKED_QK_LOAD,
            KEEP_KV_CACHE=KEEP_KV_CACHE,
            REUSE_KV_RAW=REUSE_KV_RAW,
            REUSE_KV_FP8=REUSE_KV_FP8,
            COALESCED_V_LOAD=COALESCED_V_LOAD,
            FUSED_PV_ACC=FUSED_PV_ACC,
            V_CACHE_MODIFIER=V_CACHE_MODIFIER,
            REUSE_V_SCALE=REUSE_V_SCALE,
            PAIRWISE_BF16_V=PAIRWISE_BF16_V,
            FULL_BF16_V=FULL_BF16_V,
        )
    return m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7


@triton.jit
def _sparse_attn_v4_paged_prefill_fp8_kernel(
    q_packed_ptr,
    q_packed_u8_ptr,
    q_rope_ptr,
    prefix_packed_ptr,
    prefix_packed_u8_ptr,
    prefix_rope_ptr,
    kv_indices_prefix_ptr,
    kv_indptr_prefix_ptr,
    extend_packed_ptr,
    extend_packed_u8_ptr,
    extend_rope_ptr,
    kv_indices_extend_ptr,
    kv_indptr_extend_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_rope_stride_t: tl.constexpr,
    q_rope_stride_h: tl.constexpr,
    prefix_stride_n: tl.constexpr,
    prefix_rope_stride_n: tl.constexpr,
    extend_stride_n: tl.constexpr,
    extend_rope_stride_n: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    qk_scale,
    log2e,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    EXTEND_PIPE_STAGES: tl.constexpr,
    ASSUME_FULL_TILES: tl.constexpr,
    NO_SENTINEL_HOT_LOOP: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    USE_MXFP8_V: tl.constexpr,
    USE_BF16_ACC: tl.constexpr,
    PACKED_QK_LOAD: tl.constexpr,
    KEEP_KV_CACHE: tl.constexpr,
    REUSE_KV_RAW: tl.constexpr,
    REUSE_KV_FP8: tl.constexpr,
    COALESCED_V_LOAD: tl.constexpr,
    FUSED_PV_ACC: tl.constexpr,
    V_CACHE_MODIFIER: tl.constexpr,
    REUSE_V_SCALE: tl.constexpr,
    PAIRWISE_BF16_V: tl.constexpr,
    FULL_BF16_V: tl.constexpr,
    HEAD_FIRST_GRID: tl.constexpr,
    GRID_GROUP_T: tl.constexpr,
):
    if GRID_GROUP_T > 0:
        linear_pid = tl.program_id(0)
        num_head_blocks: tl.constexpr = tl.cdiv(H, BLOCK_H)
        group_span: tl.constexpr = GRID_GROUP_T * num_head_blocks
        group_id = linear_pid // group_span
        in_group = linear_pid % group_span
        pid_h = in_group // GRID_GROUP_T
        t = (group_id * GRID_GROUP_T + in_group % GRID_GROUP_T).to(tl.int64)
    elif HEAD_FIRST_GRID:
        pid_h = tl.program_id(0)
        t = tl.program_id(1).to(tl.int64)
    else:
        t = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_offs = tl.arange(0, BLOCK_D)
    nope_mask = d_offs < NOPE
    rope_offs = tl.arange(0, ROPE)

    q_nope_raw = tl.load(
        q_packed_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :],
        mask=h_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    scale_offs = tl.arange(0, 16)
    q_scale = tl.load(
        q_packed_u8_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + PACK_OFF_SCALE
        + scale_offs[None, :],
        mask=h_mask[:, None] & (scale_offs < SCALE_BYTES)[None, :],
        other=0,
    )
    q_rope = tl.load(
        q_rope_ptr
        + t * q_rope_stride_t
        + h_offs[:, None] * q_rope_stride_h
        + rope_offs[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if USE_BF16_ACC:
        acc_width: tl.constexpr = TILE if (USE_MXFP8_V or REUSE_KV_FP8) else 2 * TILE
        acc0 = tl.zeros((BLOCK_H, acc_width), dtype=tl.bfloat16)
        acc1 = tl.zeros((BLOCK_H, acc_width), dtype=tl.bfloat16)
        acc2 = tl.zeros((BLOCK_H, acc_width), dtype=tl.bfloat16)
        acc3 = tl.zeros((BLOCK_H, acc_width), dtype=tl.bfloat16)
    elif USE_MXFP8_V or REUSE_KV_FP8:
        acc0 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_H, 2 * TILE), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_H, 2 * TILE), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_H, 2 * TILE), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_H, 2 * TILE), dtype=tl.float32)
    acc_dtype: tl.constexpr = tl.bfloat16 if USE_BF16_ACC else tl.float32
    acc4 = tl.zeros((BLOCK_H, TILE), dtype=acc_dtype)
    acc5 = tl.zeros((BLOCK_H, TILE), dtype=acc_dtype)
    acc6 = tl.zeros((BLOCK_H, TILE), dtype=acc_dtype)
    acc7 = tl.zeros((BLOCK_H, TILE), dtype=acc_dtype)
    k_offs = tl.arange(0, BLOCK_K)

    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start
    if NO_SENTINEL_HOT_LOOP:
        m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = (
            _fp8_prefill_no_sentinel_span(
                q_nope_raw,
                q_scale,
                q_rope,
                prefix_packed_ptr,
                prefix_packed_u8_ptr,
                prefix_rope_ptr,
                kv_indices_prefix_ptr,
                p_start,
                p_len,
                h_mask,
                m_i,
                l_i,
                acc0,
                acc1,
                acc2,
                acc3,
                acc4,
                acc5,
                acc6,
                acc7,
                kv_stride_n=prefix_stride_n,
                kv_rope_stride_n=prefix_rope_stride_n,
                qk_scale=qk_scale,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K,
                TAIL_BLOCK_K=TAIL_BLOCK_K,
                BLOCK_D=BLOCK_D,
                NOPE=NOPE,
                ROPE=ROPE,
                TILE=TILE,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                SCALE_BYTES=SCALE_BYTES,
                PIPE_STAGES=PIPE_STAGES,
                USE_MXFP8_V=USE_MXFP8_V,
                USE_BF16_ACC=USE_BF16_ACC,
                PACKED_QK_LOAD=PACKED_QK_LOAD,
                KEEP_KV_CACHE=KEEP_KV_CACHE,
                REUSE_KV_RAW=REUSE_KV_RAW,
                REUSE_KV_FP8=REUSE_KV_FP8,
                COALESCED_V_LOAD=COALESCED_V_LOAD,
                FUSED_PV_ACC=FUSED_PV_ACC,
                V_CACHE_MODIFIER=V_CACHE_MODIFIER,
                REUSE_V_SCALE=REUSE_V_SCALE,
                PAIRWISE_BF16_V=PAIRWISE_BF16_V,
                FULL_BF16_V=FULL_BF16_V,
            )
        )
    else:
        for j in tl.range(
            0,
            tl.cdiv(p_len, BLOCK_K),
            num_stages=PIPE_STAGES,
        ):
            k_pos = j * BLOCK_K + k_offs
            if ASSUME_FULL_TILES:
                slot = tl.load(kv_indices_prefix_ptr + p_start + k_pos).to(tl.int64)
                valid = tl.full((BLOCK_K,), True, dtype=tl.int1)
            else:
                in_range = k_pos < p_len
                slot = tl.load(
                    kv_indices_prefix_ptr + p_start + k_pos,
                    mask=in_range,
                    other=0,
                ).to(tl.int64)
                valid = in_range & (slot >= 0)
                slot = tl.where(valid, slot, 0)
            m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = (
                _fp8_prefill_update(
                    q_nope_raw,
                    q_scale,
                    q_rope,
                    prefix_packed_ptr,
                    prefix_packed_u8_ptr,
                    prefix_rope_ptr,
                    slot,
                    valid,
                    h_mask,
                    m_i,
                    l_i,
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    acc4,
                    acc5,
                    acc6,
                    acc7,
                    kv_stride_n=prefix_stride_n,
                    kv_rope_stride_n=prefix_rope_stride_n,
                    qk_scale=qk_scale,
                    BLOCK_H=BLOCK_H,
                    BLOCK_K=BLOCK_K,
                    BLOCK_D=BLOCK_D,
                    NOPE=NOPE,
                    ROPE=ROPE,
                    TILE=TILE,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    SCALE_BYTES=SCALE_BYTES,
                    USE_MXFP8_V=USE_MXFP8_V,
                    USE_BF16_ACC=USE_BF16_ACC,
                    PACKED_QK_LOAD=PACKED_QK_LOAD,
                    KEEP_KV_CACHE=KEEP_KV_CACHE,
                    REUSE_KV_RAW=REUSE_KV_RAW,
                    REUSE_KV_FP8=REUSE_KV_FP8,
                    COALESCED_V_LOAD=COALESCED_V_LOAD,
                    FUSED_PV_ACC=FUSED_PV_ACC,
                    V_CACHE_MODIFIER=V_CACHE_MODIFIER,
                    REUSE_V_SCALE=REUSE_V_SCALE,
                    PAIRWISE_BF16_V=PAIRWISE_BF16_V,
                    FULL_BF16_V=FULL_BF16_V,
                )
            )

    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start
    if NO_SENTINEL_HOT_LOOP:
        m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = (
            _fp8_prefill_no_sentinel_span(
                q_nope_raw,
                q_scale,
                q_rope,
                extend_packed_ptr,
                extend_packed_u8_ptr,
                extend_rope_ptr,
                kv_indices_extend_ptr,
                e_start,
                e_len,
                h_mask,
                m_i,
                l_i,
                acc0,
                acc1,
                acc2,
                acc3,
                acc4,
                acc5,
                acc6,
                acc7,
                kv_stride_n=extend_stride_n,
                kv_rope_stride_n=extend_rope_stride_n,
                qk_scale=qk_scale,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K,
                TAIL_BLOCK_K=TAIL_BLOCK_K,
                BLOCK_D=BLOCK_D,
                NOPE=NOPE,
                ROPE=ROPE,
                TILE=TILE,
                PACK_OFF_SCALE=PACK_OFF_SCALE,
                SCALE_BYTES=SCALE_BYTES,
                PIPE_STAGES=EXTEND_PIPE_STAGES,
                USE_MXFP8_V=USE_MXFP8_V,
                USE_BF16_ACC=USE_BF16_ACC,
                PACKED_QK_LOAD=PACKED_QK_LOAD,
                KEEP_KV_CACHE=KEEP_KV_CACHE,
                REUSE_KV_RAW=REUSE_KV_RAW,
                REUSE_KV_FP8=REUSE_KV_FP8,
                COALESCED_V_LOAD=COALESCED_V_LOAD,
                FUSED_PV_ACC=FUSED_PV_ACC,
                V_CACHE_MODIFIER=V_CACHE_MODIFIER,
                REUSE_V_SCALE=REUSE_V_SCALE,
                PAIRWISE_BF16_V=PAIRWISE_BF16_V,
                FULL_BF16_V=FULL_BF16_V,
            )
        )
    else:
        for j in tl.range(
            0,
            tl.cdiv(e_len, BLOCK_K),
            num_stages=EXTEND_PIPE_STAGES,
        ):
            k_pos = j * BLOCK_K + k_offs
            if ASSUME_FULL_TILES:
                slot = tl.load(kv_indices_extend_ptr + e_start + k_pos).to(tl.int64)
                valid = tl.full((BLOCK_K,), True, dtype=tl.int1)
            else:
                in_range = k_pos < e_len
                slot = tl.load(
                    kv_indices_extend_ptr + e_start + k_pos,
                    mask=in_range,
                    other=0,
                ).to(tl.int64)
                valid = in_range & (slot >= 0)
                slot = tl.where(valid, slot, 0)
            m_i, l_i, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = (
                _fp8_prefill_update(
                    q_nope_raw,
                    q_scale,
                    q_rope,
                    extend_packed_ptr,
                    extend_packed_u8_ptr,
                    extend_rope_ptr,
                    slot,
                    valid,
                    h_mask,
                    m_i,
                    l_i,
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    acc4,
                    acc5,
                    acc6,
                    acc7,
                    kv_stride_n=extend_stride_n,
                    kv_rope_stride_n=extend_rope_stride_n,
                    qk_scale=qk_scale,
                    BLOCK_H=BLOCK_H,
                    BLOCK_K=BLOCK_K,
                    BLOCK_D=BLOCK_D,
                    NOPE=NOPE,
                    ROPE=ROPE,
                    TILE=TILE,
                    PACK_OFF_SCALE=PACK_OFF_SCALE,
                    SCALE_BYTES=SCALE_BYTES,
                    USE_MXFP8_V=USE_MXFP8_V,
                    USE_BF16_ACC=USE_BF16_ACC,
                    PACKED_QK_LOAD=PACKED_QK_LOAD,
                    KEEP_KV_CACHE=KEEP_KV_CACHE,
                    REUSE_KV_RAW=REUSE_KV_RAW,
                    REUSE_KV_FP8=REUSE_KV_FP8,
                    COALESCED_V_LOAD=COALESCED_V_LOAD,
                    FUSED_PV_ACC=FUSED_PV_ACC,
                    V_CACHE_MODIFIER=V_CACHE_MODIFIER,
                    REUSE_V_SCALE=REUSE_V_SCALE,
                    PAIRWISE_BF16_V=PAIRWISE_BF16_V,
                    FULL_BF16_V=FULL_BF16_V,
                )
            )

    acc01 = tl.cat(acc0, acc1, dim=1)
    acc23 = tl.cat(acc2, acc3, dim=1)
    if USE_MXFP8_V or REUSE_KV_FP8:
        acc45 = tl.cat(acc4, acc5, dim=1)
        acc67 = tl.cat(acc6, acc7, dim=1)
        acc03 = tl.cat(acc01, acc23, dim=1)
        acc47 = tl.cat(acc45, acc67, dim=1)
        acc = tl.cat(acc03, acc47, dim=1)
    else:
        acc = tl.cat(acc01, acc23, dim=1)
    sink = (
        tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
        * log2e
    )
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    normalize = alpha_kv / denom
    out = tl.where(
        l_final[:, None] > 0.0,
        acc * normalize[:, None],
        0.0,
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out.to(tl.bfloat16),
        mask=h_mask[:, None] & (d_offs < NOPE + ROPE)[None, :],
    )


def sparse_attn_v4_paged_prefill_fp8_triton(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_packed: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    *,
    block_h: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    extend_num_stages: int | None = None,
    waves_per_eu: int | None = None,
    matrix_instr_nonkdim: int | None = None,
    schedule_hint: str | None = None,
    assume_full_tiles: bool = False,
    no_sentinel_hot_loop: bool = False,
    tail_block_k: int = 64,
    use_mxfp8_v: bool = False,
    use_bf16_acc: bool = False,
    packed_qk_load: bool = False,
    keep_kv_cache: bool = False,
    reuse_kv_raw: bool = False,
    reuse_kv_fp8: bool = False,
    coalesced_v_load: bool = False,
    fused_pv_acc: bool = False,
    v_cache_modifier: str = "",
    reuse_v_scale: bool = False,
    pairwise_bf16_v: bool = False,
    full_bf16_v: bool = False,
    head_first_grid: bool = False,
    grid_group_tokens: int = 0,
    compiled_launch: bool = False,
) -> torch.Tensor:
    """Run the gfx950 native two-buffer FP8 sparse-prefill candidate."""
    if not q_packed.is_cuda:
        raise RuntimeError("Triton FP8 sparse prefill requires CUDA/HIP tensors")
    if q_packed.dtype != dtypes.fp8:
        raise RuntimeError(f"q_packed must be {dtypes.fp8}, got {q_packed.dtype}")
    if unified_kv_packed.dtype != q_packed.dtype or k_packed.dtype != q_packed.dtype:
        raise RuntimeError(
            "packed NoPE dtype mismatch: "
            f"q={q_packed.dtype}, prefix={unified_kv_packed.dtype}, "
            f"extend={k_packed.dtype}"
        )
    if q_packed.dim() != 3 or q_packed.shape[-1] != V4_DIM_QK_PACKED:
        raise RuntimeError(
            f"q_packed must have shape [T,H,{V4_DIM_QK_PACKED}], "
            f"got {tuple(q_packed.shape)}"
        )

    k_packed = k_packed.reshape(k_packed.shape[0], -1)
    k_rope = k_rope.reshape(k_rope.shape[0], -1)
    T, H, _ = q_packed.shape
    # Best-known gfx950 schedules from the long-context ABBA matrix.  Keep
    # explicit arguments available to the tuning harness, but do not let the
    # opt-in production dispatch fall back to the original slow prototype
    # defaults (BH16/stage1/waves_per_eu=2).
    small_h = H <= 32
    block_h = block_h if block_h is not None else (16 if small_h else 64)
    block_k = block_k if block_k is not None else 32
    num_warps = num_warps if num_warps is not None else 4
    num_stages = num_stages if num_stages is not None else 2
    extend_num_stages = (
        extend_num_stages if extend_num_stages is not None else num_stages
    )
    waves_per_eu = waves_per_eu if waves_per_eu is not None else (0 if small_h else 1)
    matrix_instr_nonkdim = (
        matrix_instr_nonkdim if matrix_instr_nonkdim is not None else 16
    )
    schedule_hint = schedule_hint if schedule_hint is not None else "attention"
    expected_q_rope = (T, H, V4_DIM_ROPE)
    if q_rope.shape != expected_q_rope or q_rope.dtype != torch.bfloat16:
        raise RuntimeError(
            f"q_rope must be bf16 {expected_q_rope}, got "
            f"shape={tuple(q_rope.shape)} dtype={q_rope.dtype}"
        )
    for name, packed, rope in (
        ("prefix", unified_kv_packed, unified_kv_rope),
        ("extend", k_packed, k_rope),
    ):
        if packed.dim() != 2 or packed.shape[-1] != V4_DIM_QK_PACKED:
            raise RuntimeError(
                f"{name} packed tensor must have shape [N,{V4_DIM_QK_PACKED}], "
                f"got {tuple(packed.shape)}"
            )
        if rope.shape != (packed.shape[0], V4_DIM_ROPE):
            raise RuntimeError(
                f"{name} rope tensor must have shape "
                f"{(packed.shape[0], V4_DIM_ROPE)}, got {tuple(rope.shape)}"
            )
        if rope.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} rope tensor must be bf16, got {rope.dtype}")
    if attn_sink.shape != (H,) or attn_sink.dtype != torch.float32:
        raise RuntimeError(
            f"attn_sink must be fp32 shape {(H,)}, got "
            f"shape={tuple(attn_sink.shape)} dtype={attn_sink.dtype}"
        )
    if block_h not in (8, 16, 32, 64, 128):
        raise ValueError(f"block_h must be one of 8/16/32/64/128, got {block_h}")
    if block_k not in (16, 32, 64):
        raise ValueError(f"block_k must be one of 16/32/64, got {block_k}")
    if tail_block_k not in (16, 32, 64):
        raise ValueError(f"tail_block_k must be 16, 32, or 64, got {tail_block_k}")
    if tail_block_k > block_k:
        raise ValueError(
            f"tail_block_k must not exceed block_k, got {tail_block_k} > {block_k}"
        )
    if assume_full_tiles and no_sentinel_hot_loop:
        raise ValueError(
            "assume_full_tiles and no_sentinel_hot_loop are mutually exclusive"
        )
    if grid_group_tokens < 0 or (grid_group_tokens and T % grid_group_tokens != 0):
        raise ValueError(
            "grid_group_tokens must be zero or exactly divide T, got "
            f"grid_group_tokens={grid_group_tokens}, T={T}"
        )
    if v_cache_modifier not in ("", ".ca", ".cg"):
        raise ValueError(
            f"v_cache_modifier must be empty, '.ca', or '.cg', got {v_cache_modifier!r}"
        )
    if out is None:
        out = torch.empty(
            (T, H, V4_DIM_QK), dtype=torch.bfloat16, device=q_packed.device
        )
    elif out.shape != (T, H, V4_DIM_QK) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out must be bf16 shape {(T, H, V4_DIM_QK)}, got "
            f"shape={tuple(out.shape)} dtype={out.dtype}"
        )

    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()
    if kv_indptr_prefix.numel() != T + 1 or kv_indptr_extend.numel() != T + 1:
        raise RuntimeError(
            "prefill indptr tensors must each contain T+1 entries: "
            f"T={T}, prefix={kv_indptr_prefix.numel()}, "
            f"extend={kv_indptr_extend.numel()}"
        )

    runtime_args = (
        q_packed,
        q_packed.view(torch.uint8),
        q_rope,
        unified_kv_packed,
        unified_kv_packed.view(torch.uint8),
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        k_packed,
        k_packed.view(torch.uint8),
        k_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q_packed.stride(0),
        q_packed.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        unified_kv_packed.stride(0),
        unified_kv_rope.stride(0),
        k_packed.stride(0),
        k_rope.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        float(softmax_scale) * _LOG2E,
        _LOG2E,
    )
    compile_options = {
        "H": H,
        "BLOCK_H": block_h,
        "BLOCK_K": block_k,
        "BLOCK_D": V4_DIM_QK_PACKED,
        "NOPE": V4_DIM_NOPE,
        "ROPE": V4_DIM_ROPE,
        "TILE": V4_TILE,
        "PACK_OFF_SCALE": V4_PACK_OFF_SCALE,
        "SCALE_BYTES": _SCALE_BYTES,
        "PIPE_STAGES": num_stages,
        "EXTEND_PIPE_STAGES": extend_num_stages,
        "ASSUME_FULL_TILES": assume_full_tiles,
        "NO_SENTINEL_HOT_LOOP": no_sentinel_hot_loop,
        "TAIL_BLOCK_K": tail_block_k,
        "USE_MXFP8_V": use_mxfp8_v,
        "USE_BF16_ACC": use_bf16_acc,
        "PACKED_QK_LOAD": packed_qk_load,
        "KEEP_KV_CACHE": keep_kv_cache,
        "REUSE_KV_RAW": reuse_kv_raw,
        "REUSE_KV_FP8": reuse_kv_fp8,
        "COALESCED_V_LOAD": coalesced_v_load,
        "FUSED_PV_ACC": fused_pv_acc,
        "V_CACHE_MODIFIER": v_cache_modifier,
        "REUSE_V_SCALE": reuse_v_scale,
        "PAIRWISE_BF16_V": pairwise_bf16_v,
        "FULL_BF16_V": full_bf16_v,
        "HEAD_FIRST_GRID": head_first_grid,
        "GRID_GROUP_T": grid_group_tokens,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": matrix_instr_nonkdim,
        "schedule_hint": schedule_hint,
    }

    num_head_blocks = triton.cdiv(H, block_h)
    if grid_group_tokens:
        grid = (T * num_head_blocks, 1, 1)
    else:
        grid = (num_head_blocks, T, 1) if head_first_grid else (T, num_head_blocks, 1)
    if compiled_launch:
        stride_key = runtime_args[15:26]
        cache_key = (
            torch.cuda.current_device(),
            *stride_key,
            *compile_options.values(),
        )
        compiled_kernel = _COMPILED_PREFILL_KERNELS.get(cache_key)
        if compiled_kernel is None:
            compiled_kernel = _sparse_attn_v4_paged_prefill_fp8_kernel.warmup(
                *runtime_args,
                grid=grid,
                **compile_options,
            )
            _COMPILED_PREFILL_KERNELS[cache_key] = compiled_kernel
        compiled_kernel[grid](*runtime_args)
    else:
        _sparse_attn_v4_paged_prefill_fp8_kernel[grid](
            *runtime_args,
            **compile_options,
        )
    return out


__all__ = ["sparse_attn_v4_paged_prefill_fp8_triton"]
