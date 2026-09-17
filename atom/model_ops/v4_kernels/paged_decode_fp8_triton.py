# SPDX-License-Identifier: MIT
"""Experimental Triton decode for V4's native FP8 two-buffer cache.

This consumes the exact layout produced by ``qk_norm_rope_maybe_quant`` when
``fp8_2buff=True``:

* packed NoPE + duplicated e8m0 scales: ``[..., 512]`` FP8 bytes
* RoPE tail: ``[..., 64]`` BF16

The dispatcher covers regular decode and rectangular MTP4/DSpark-K6
verification, with split-K specializations for small batches and long KV
ranges. DSpark's seven rows use either individual queries or two four-row
stripes, depending on active concurrency.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter import dtypes

from atom.model_ops.v4_kernels.paged_decode import LOG2E, _paged_decode_reduce_kernel
from atom.model_ops.v4_kernels.v4_quant import (
    V4_DIM_NOPE,
    V4_DIM_QK,
    V4_DIM_QK_PACKED,
    V4_DIM_ROPE,
    V4_NUM_TILES,
    V4_PACK_OFF_SCALE,
    V4_TILE,
)


@triton.jit
def _e8m0_scale(byte):
    """Decode E8M0 byte B as 2**(B-127), with B=0 as the zero sentinel."""
    # E8M0 uses the same 8-bit biased exponent as IEEE fp32.  Placing the
    # byte directly in fp32's exponent field is exact and maps the zero-tile
    # sentinel B=0 to +0.0 without an exp2 instruction.
    bits = byte.to(tl.uint32) << 23
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _fp8_v_group_dot(
    p,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    slot,
    valid,
    kv_stride_n,
    GROUP: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Compute one 64-wide P@V group without expanding V to BF16."""
    v_offs = tl.arange(0, TILE)
    scale_byte = tl.load(
        kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 2 * GROUP,
        mask=valid,
        other=0,
    )
    p_scaled = (p * _e8m0_scale(scale_byte)[None, :]).to(tl.bfloat16)
    v_raw = tl.load(
        kv_packed_ptr + slot[:, None] * kv_stride_n + GROUP * TILE + v_offs[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    return tl.dot_scaled(
        p_scaled,
        None,
        "bf16",
        v_raw,
        None,
        "e4m3",
        out_dtype=tl.float32,
    )


@triton.jit
def _paged_decode_fp8_2buff_fused_kernel(
    q_packed_ptr,  # [T,H,512] fp8
    q_packed_u8_ptr,  # same storage as uint8, for e8m0 scale bytes
    q_rope_ptr,  # [T,H,64] bf16
    kv_packed_ptr,  # [P,512] fp8
    kv_packed_u8_ptr,  # same storage as uint8
    kv_rope_ptr,  # [P,64] bf16
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [T+1] int32
    attn_sink_ptr,  # [H] fp32
    m_partial_ptr,  # [T,KV_SPLITS,H] fp32 when KV_SPLITS>1
    l_partial_ptr,  # [T,KV_SPLITS,H] fp32 when KV_SPLITS>1
    acc_partial_ptr,  # [T,KV_SPLITS,H,512], fp32/bf16/fp16 when split
    out_ptr,  # [T,H,512] bf16
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_rope_stride_t: tl.constexpr,
    q_rope_stride_h: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_rope_stride_n: tl.constexpr,
    mp_stride_t: tl.constexpr,
    mp_stride_k: tl.constexpr,
    mp_stride_h: tl.constexpr,
    lp_stride_t: tl.constexpr,
    lp_stride_k: tl.constexpr,
    lp_stride_h: tl.constexpr,
    ap_stride_t: tl.constexpr,
    ap_stride_k: tl.constexpr,
    ap_stride_h: tl.constexpr,
    ap_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    qk_scale,
    log2e,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    USE_MXFP8_QK: tl.constexpr,
    USE_MXFP8_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Native 2buff attention stage-1; direct output or split-K partials."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    nope_mask = d_offs < NOPE
    rope_mask = (d_offs >= NOPE) & (d_offs < NOPE + ROPE)
    d_mask = d_offs < NOPE + ROPE

    # The 14 duplicated bytes are exactly the 1x32 e8m0 scale layout expected
    # by tl.dot_scaled: each original 1x64 V4 scale is present twice. Pad the
    # NoPE K dimension from 448 to 512 with zero data/scale so arange and MFMA
    # tiles stay power-of-two shaped.
    q_nope_raw = tl.load(
        q_packed_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :],
        mask=h_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_rope_ptr
        + t * q_rope_stride_t
        + h_offs[:, None] * q_rope_stride_h
        + (d_offs - NOPE)[None, :],
        mask=h_mask[:, None] & rope_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if USE_MXFP8_QK:
        mx_scale_offs = tl.arange(0, 16)
        q_scale_mx = tl.load(
            q_packed_u8_ptr
            + t * q_stride_t
            + h_offs[:, None] * q_stride_h
            + PACK_OFF_SCALE
            + mx_scale_offs[None, :],
            mask=h_mask[:, None] & (mx_scale_offs < 14)[None, :],
            other=0,
        )
        r_offs = tl.arange(0, ROPE)
        q_rope_mx = tl.load(
            q_rope_ptr
            + t * q_rope_stride_t
            + h_offs[:, None] * q_rope_stride_h
            + r_offs[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
    if not USE_MXFP8_QK:
        scale64_offs = tl.arange(0, 8)
        q_scale_64 = tl.load(
            q_packed_u8_ptr
            + t * q_stride_t
            + h_offs[:, None] * q_stride_h
            + PACK_OFF_SCALE
            + 2 * scale64_offs[None, :],
            mask=h_mask[:, None] & (scale64_offs < NUM_TILES)[None, :],
            other=0,
        )
        q_scale_full = tl.reshape(
            tl.broadcast_to(
                _e8m0_scale(q_scale_64)[:, :, None],
                (BLOCK_H, 8, TILE),
            ),
            (BLOCK_H, BLOCK_D),
        )
        q_nope = q_nope_raw.to(tl.float32) * q_scale_full
        q = tl.where(nope_mask[None, :], q_nope, q_rope).to(tl.bfloat16)

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    if KV_SPLITS > 1:
        tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
        tile_start = pid_k * tiles_per_segment
        tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)
    else:
        tile_start = 0
        tile_end = num_tiles

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if USE_MXFP8_V:
        # Keep eight 64-wide accumulators so each NoPE group can absorb its
        # per-token e8m0 V scale into P before a native BF16 x FP8 dot.
        acc0 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc4 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc5 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc6 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
        acc7 = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    scale64_offs = tl.arange(0, 8)
    for j in tl.range(tile_start, tile_end, num_stages=PIPE_STAGES):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,
        ).to(tl.int64)

        kv_nope_raw = tl.load(
            kv_packed_ptr + slot[:, None] * kv_stride_n + d_offs[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0.0,
        )
        if USE_MXFP8_QK:
            kv_scale_mx = tl.load(
                kv_packed_u8_ptr
                + slot[:, None] * kv_stride_n
                + PACK_OFF_SCALE
                + mx_scale_offs[None, :],
                mask=valid[:, None] & (mx_scale_offs < 14)[None, :],
                other=0,
            )
            scores = tl.dot_scaled(
                q_nope_raw,
                q_scale_mx,
                "e4m3",
                tl.trans(kv_nope_raw),
                kv_scale_mx,
                "e4m3",
                out_dtype=tl.float32,
            )
            # RoPE remains BF16 and occupies the logical tail [448:512].
            kv_rope_mx = tl.load(
                kv_rope_ptr + slot[:, None] * kv_rope_stride_n + r_offs[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            scores += tl.dot(
                q_rope_mx,
                tl.trans(kv_rope_mx),
            )
            scores *= qk_scale
        else:
            kv_scale_64 = tl.load(
                kv_packed_u8_ptr
                + slot[:, None] * kv_stride_n
                + PACK_OFF_SCALE
                + 2 * scale64_offs[None, :],
                mask=valid[:, None] & (scale64_offs < NUM_TILES)[None, :],
                other=0,
            )
            kv_scale_full = tl.reshape(
                tl.broadcast_to(
                    _e8m0_scale(kv_scale_64)[:, :, None],
                    (BLOCK_K, 8, TILE),
                ),
                (BLOCK_K, BLOCK_D),
            )
            kv_nope = kv_nope_raw.to(tl.float32) * kv_scale_full
            kv_rope = tl.load(
                kv_rope_ptr
                + slot[:, None] * kv_rope_stride_n
                + (d_offs - NOPE)[None, :],
                mask=valid[:, None] & rope_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            kv = tl.where(nope_mask[None, :], kv_nope, kv_rope).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        if USE_MXFP8_V:
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
            acc7 = acc7 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_rope_mx)
        else:
            if USE_MXFP8_QK:
                # Reload V after softmax instead of keeping a full BF16
                # [BLOCK_K, 512] tile live across the score reduction.  The
                # extra read is cheaper than the occupancy loss on gfx950.
                kv_nope_raw = tl.load(
                    kv_packed_ptr + slot[:, None] * kv_stride_n + d_offs[None, :],
                    mask=valid[:, None] & nope_mask[None, :],
                    other=0.0,
                    volatile=True,
                )
                kv_scale_64 = tl.load(
                    kv_packed_u8_ptr
                    + slot[:, None] * kv_stride_n
                    + PACK_OFF_SCALE
                    + 2 * scale64_offs[None, :],
                    mask=valid[:, None] & (scale64_offs < NUM_TILES)[None, :],
                    other=0,
                    volatile=True,
                )
                kv_scale_full = tl.reshape(
                    tl.broadcast_to(
                        _e8m0_scale(kv_scale_64)[:, :, None],
                        (BLOCK_K, 8, TILE),
                    ),
                    (BLOCK_K, BLOCK_D),
                )
                kv_nope = kv_nope_raw.to(tl.float32) * kv_scale_full
                kv_rope = tl.load(
                    kv_rope_ptr
                    + slot[:, None] * kv_rope_stride_n
                    + (d_offs - NOPE)[None, :],
                    mask=valid[:, None] & rope_mask[None, :],
                    other=0.0,
                    volatile=True,
                ).to(tl.bfloat16)
                kv = tl.where(nope_mask[None, :], kv_nope, kv_rope).to(tl.bfloat16)
            # Feed the rescaled online-softmax accumulator directly to the
            # MFMA.  Keeping ``acc * alpha`` and a standalone dot result live
            # at the same time nearly doubles AGPR pressure for the 64x512
            # HCA tile and collapses occupancy on gfx950.
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), kv, acc)
        m_i = m_new
        l_i = l_new

    if USE_MXFP8_V:
        acc01 = tl.cat(acc0, acc1, dim=1)
        acc23 = tl.cat(acc2, acc3, dim=1)
        acc45 = tl.cat(acc4, acc5, dim=1)
        acc67 = tl.cat(acc6, acc7, dim=1)
        acc03 = tl.cat(acc01, acc23, dim=1)
        acc47 = tl.cat(acc45, acc67, dim=1)
        acc = tl.cat(acc03, acc47, dim=1)

    if KV_SPLITS > 1:
        m_base = t * mp_stride_t + pid_k * mp_stride_k
        l_base = t * lp_stride_t + pid_k * lp_stride_k
        a_base = t * ap_stride_t + pid_k * ap_stride_k
        tl.store(
            m_partial_ptr + m_base + h_offs * mp_stride_h,
            m_i,
            mask=h_mask,
        )
        tl.store(
            l_partial_ptr + l_base + h_offs * lp_stride_h,
            l_i,
            mask=h_mask,
        )
        tl.store(
            acc_partial_ptr
            + a_base
            + h_offs[:, None] * ap_stride_h
            + d_offs[None, :] * ap_stride_d,
            acc,
            mask=h_mask[:, None] & d_mask[None, :],
        )
        return

    sink = (
        tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
        * log2e
    )
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0,
        (acc * alpha_kv[:, None]) / denom[:, None],
        0.0,
    )
    tl.store(
        out_ptr + t * out_stride_t + h_offs[:, None] * out_stride_h + d_offs[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _paged_decode_fp8_qk_store_kernel(
    q_packed_ptr,
    q_packed_u8_ptr,
    q_rope_ptr,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    kv_rope_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    scores_ptr,
    q_stride_t,
    q_stride_h,
    q_rope_stride_t,
    q_rope_stride_h,
    kv_stride_n,
    kv_rope_stride_n,
    scores_stride_k,
    scores_stride_h,
    qk_scale,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Two-pass stage 1: native MXFP8 QK and BF16 score materialization."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_offs = tl.arange(0, BLOCK_D)
    nope_mask = d_offs < NOPE
    mx_scale_offs = tl.arange(0, 16)

    q_raw = tl.load(
        q_packed_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :],
        mask=h_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_scale = tl.load(
        q_packed_u8_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + PACK_OFF_SCALE
        + mx_scale_offs[None, :],
        mask=h_mask[:, None] & (mx_scale_offs < 14)[None, :],
        other=0,
    )
    r_offs = tl.arange(0, ROPE)
    q_rope = tl.load(
        q_rope_ptr
        + t * q_rope_stride_t
        + h_offs[:, None] * q_rope_stride_h
        + r_offs[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(0, tl.cdiv(kv_len, BLOCK_K), num_stages=PIPE_STAGES):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < kv_len
        index_pos = kv_start + k_pos
        slot = tl.load(kv_indices_ptr + index_pos, mask=valid, other=0).to(tl.int64)
        kv_raw = tl.load(
            kv_packed_ptr + slot[:, None] * kv_stride_n + d_offs[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0.0,
        )
        kv_scale = tl.load(
            kv_packed_u8_ptr
            + slot[:, None] * kv_stride_n
            + PACK_OFF_SCALE
            + mx_scale_offs[None, :],
            mask=valid[:, None] & (mx_scale_offs < 14)[None, :],
            other=0,
        )
        scores = tl.dot_scaled(
            q_raw,
            q_scale,
            "e4m3",
            tl.trans(kv_raw),
            kv_scale,
            "e4m3",
            out_dtype=tl.float32,
        )
        kv_rope = tl.load(
            kv_rope_ptr + slot[:, None] * kv_rope_stride_n + r_offs[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        scores += tl.dot(q_rope, tl.trans(kv_rope))
        scores *= qk_scale
        tl.store(
            scores_ptr
            + index_pos[None, :] * scores_stride_k
            + h_offs[:, None] * scores_stride_h,
            scores.to(tl.bfloat16),
            mask=h_mask[:, None] & valid[None, :],
        )


@triton.jit
def _paged_decode_fp8_grouped_pv_kernel(
    scores_ptr,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    kv_rope_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    scores_stride_k,
    scores_stride_h,
    kv_stride_n,
    kv_rope_stride_n,
    out_stride_t,
    out_stride_h,
    log2e,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Two-pass stage 2: online softmax plus one spill-free 64-wide PV."""
    t = tl.program_id(0)
    group = tl.program_id(1)
    h_offs = tl.arange(0, BLOCK_H)
    v_offs = tl.arange(0, TILE)
    h_mask = h_offs < H

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, TILE), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(0, tl.cdiv(kv_len, BLOCK_K), num_stages=PIPE_STAGES):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < kv_len
        index_pos = kv_start + k_pos
        scores = tl.load(
            scores_ptr
            + index_pos[None, :] * scores_stride_k
            + h_offs[:, None] * scores_stride_h,
            mask=h_mask[:, None] & valid[None, :],
            other=neg_large,
        ).to(tl.float32)
        slot = tl.load(kv_indices_ptr + index_pos, mask=valid, other=0).to(tl.int64)

        if group < NUM_TILES:
            v_raw = tl.load(
                kv_packed_ptr
                + slot[:, None] * kv_stride_n
                + group * TILE
                + v_offs[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scale_byte = tl.load(
                kv_packed_u8_ptr + slot * kv_stride_n + PACK_OFF_SCALE + 2 * group,
                mask=valid,
                other=0,
            )
            scale = _e8m0_scale(scale_byte)
            v = (v_raw.to(tl.float32) * scale[:, None]).to(tl.bfloat16)
        else:
            v = tl.load(
                kv_rope_ptr + slot[:, None] * kv_rope_stride_n + v_offs[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    sink = (
        tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
        * log2e
    )
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0,
        (acc * alpha_kv[:, None]) / denom[:, None],
        0.0,
    )
    d_offs = group * TILE + v_offs
    tl.store(
        out_ptr + t * out_stride_t + h_offs[:, None] * out_stride_h + d_offs[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None],
    )


@triton.jit
def _paged_decode_reduce2_kernel(
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    attn_sink_ptr,
    kv_indptr_ptr,
    out_ptr,
    mp_stride_t: tl.constexpr,
    mp_stride_k: tl.constexpr,
    mp_stride_h: tl.constexpr,
    lp_stride_t: tl.constexpr,
    lp_stride_k: tl.constexpr,
    lp_stride_h: tl.constexpr,
    ap_stride_t: tl.constexpr,
    ap_stride_k: tl.constexpr,
    ap_stride_h: tl.constexpr,
    ap_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    log2e,
    D: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fully unrolled two-way split reduction for the T=256 long-K path."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)
    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    d_mask = d_offs < D
    neg_large = -3.4028234663852886e38

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    if kv_len == 0:
        tl.store(
            out_ptr + t * out_stride_t + h * out_stride_h + d_offs * out_stride_d,
            tl.zeros((D_CHUNK,), dtype=out_ptr.dtype.element_ty),
            mask=d_mask,
        )
        return

    tiles_per_segment = tl.cdiv(kv_len, 2 * BLOCK_K)
    has_second = kv_len > tiles_per_segment * BLOCK_K
    m0 = tl.load(m_partial_ptr + t * mp_stride_t + h * mp_stride_h)
    l0 = tl.load(l_partial_ptr + t * lp_stride_t + h * lp_stride_h)
    m1 = tl.load(
        m_partial_ptr + t * mp_stride_t + mp_stride_k + h * mp_stride_h,
        mask=has_second,
        other=neg_large,
    )
    l1 = tl.load(
        l_partial_ptr + t * lp_stride_t + lp_stride_k + h * lp_stride_h,
        mask=has_second,
        other=0.0,
    )
    a0 = tl.load(
        acc_partial_ptr + t * ap_stride_t + h * ap_stride_h + d_offs * ap_stride_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    a1 = tl.load(
        acc_partial_ptr
        + t * ap_stride_t
        + ap_stride_k
        + h * ap_stride_h
        + d_offs * ap_stride_d,
        mask=has_second & d_mask,
        other=0.0,
    ).to(tl.float32)

    m_max = tl.maximum(m0, m1)
    alpha0 = tl.exp2(m0 - m_max)
    alpha1 = tl.exp2(m1 - m_max)
    l_combined = l0 * alpha0 + l1 * alpha1
    acc_combined = a0 * alpha0 + a1 * alpha1

    sink = tl.load(attn_sink_ptr + h).to(tl.float32) * log2e
    m_final = tl.maximum(m_max, sink)
    alpha_kv = tl.exp2(m_max - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_combined * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final > 0.0, acc_combined * alpha_kv / denom, 0.0)
    tl.store(
        out_ptr + t * out_stride_t + h * out_stride_h + d_offs * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _paged_decode_fp8_query_group_kernel(
    q_packed_ptr,
    q_packed_u8_ptr,
    q_rope_ptr,
    kv_packed_ptr,
    kv_packed_u8_ptr,
    kv_rope_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    m_partial_ptr,
    l_partial_ptr,
    acc_partial_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_rope_stride_t: tl.constexpr,
    q_rope_stride_h: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_rope_stride_n: tl.constexpr,
    mp_stride_t: tl.constexpr,
    mp_stride_k: tl.constexpr,
    mp_stride_h: tl.constexpr,
    lp_stride_t: tl.constexpr,
    lp_stride_k: tl.constexpr,
    lp_stride_h: tl.constexpr,
    ap_stride_t: tl.constexpr,
    ap_stride_k: tl.constexpr,
    ap_stride_h: tl.constexpr,
    ap_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    qk_scale,
    log2e,
    T,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REQUEST_QUERY_GROUP: tl.constexpr,
    FUSED_QUERY_GROUP: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    USE_MXFP8_QK: tl.constexpr,
    KV_SPLITS: tl.constexpr,
):
    """Fuse adjacent speculative queries that share one request's KV indices."""
    request_query_block = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)
    request_group = request_query_block // QUERY_BLOCKS
    query_block = request_query_block % QUERY_BLOCKS
    m_offs = tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    q_rel = m_offs // BLOCK_H
    h_offs = pid_h * BLOCK_H + m_offs % BLOCK_H
    request_q_rel = query_block * FUSED_QUERY_GROUP + q_rel
    t_offs = request_group * REQUEST_QUERY_GROUP + request_q_rel
    q_mask = (request_q_rel < REQUEST_QUERY_GROUP) & (t_offs < T) & (h_offs < H)
    nope_mask = d_offs < NOPE
    rope_mask = (d_offs >= NOPE) & (d_offs < NOPE + ROPE)
    scale64_offs = tl.arange(0, 8)
    q_raw = tl.load(
        q_packed_ptr
        + t_offs[:, None] * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :],
        mask=q_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    if USE_MXFP8_QK:
        r_offs = tl.arange(0, ROPE)
        mx_scale_offs = tl.arange(0, 16)
        q_scale_mx = tl.load(
            q_packed_u8_ptr
            + t_offs[:, None] * q_stride_t
            + h_offs[:, None] * q_stride_h
            + PACK_OFF_SCALE
            + mx_scale_offs[None, :],
            mask=q_mask[:, None] & (mx_scale_offs < 14)[None, :],
            other=0,
        )
        q_rope_mx = tl.load(
            q_rope_ptr
            + t_offs[:, None] * q_rope_stride_t
            + h_offs[:, None] * q_rope_stride_h
            + r_offs[None, :],
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
    else:
        q_scale_64 = tl.load(
            q_packed_u8_ptr
            + t_offs[:, None] * q_stride_t
            + h_offs[:, None] * q_stride_h
            + PACK_OFF_SCALE
            + 2 * scale64_offs[None, :],
            mask=q_mask[:, None] & (scale64_offs < NUM_TILES)[None, :],
            other=0,
        )
        q_scale_full = tl.reshape(
            tl.broadcast_to(
                _e8m0_scale(q_scale_64)[:, :, None],
                (BLOCK_M, 8, TILE),
            ),
            (BLOCK_M, BLOCK_D),
        )
        q_nope = q_raw.to(tl.float32) * q_scale_full
        q_rope = tl.load(
            q_rope_ptr
            + t_offs[:, None] * q_rope_stride_t
            + h_offs[:, None] * q_rope_stride_h
            + (d_offs - NOPE)[None, :],
            mask=q_mask[:, None] & rope_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q = tl.where(nope_mask[None, :], q_nope, q_rope).to(tl.bfloat16)

    t0 = request_group * REQUEST_QUERY_GROUP
    kv_start = tl.load(kv_indptr_ptr + t0)
    kv_end = tl.load(kv_indptr_ptr + t0 + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    if KV_SPLITS > 1:
        tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
        tile_start = pid_k * tiles_per_segment
        tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)
    else:
        tile_start = 0
        tile_end = num_tiles

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_M,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=PIPE_STAGES):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=valid,
            other=0,
        ).to(tl.int64)
        kv_raw = tl.load(
            kv_packed_ptr + slot[:, None] * kv_stride_n + d_offs[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0.0,
        )
        if USE_MXFP8_QK:
            kv_scale_mx = tl.load(
                kv_packed_u8_ptr
                + slot[:, None] * kv_stride_n
                + PACK_OFF_SCALE
                + mx_scale_offs[None, :],
                mask=valid[:, None] & (mx_scale_offs < 14)[None, :],
                other=0,
            )
            scores = tl.dot_scaled(
                q_raw,
                q_scale_mx,
                "e4m3",
                tl.trans(kv_raw),
                kv_scale_mx,
                "e4m3",
                out_dtype=tl.float32,
            )
            kv_rope_mx = tl.load(
                kv_rope_ptr + slot[:, None] * kv_rope_stride_n + r_offs[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            scores += tl.dot(q_rope_mx, tl.trans(kv_rope_mx))
            scores *= qk_scale
        else:
            kv_scale_64 = tl.load(
                kv_packed_u8_ptr
                + slot[:, None] * kv_stride_n
                + PACK_OFF_SCALE
                + 2 * scale64_offs[None, :],
                mask=valid[:, None] & (scale64_offs < NUM_TILES)[None, :],
                other=0,
            )
            kv_scale_full = tl.reshape(
                tl.broadcast_to(
                    _e8m0_scale(kv_scale_64)[:, :, None],
                    (BLOCK_K, 8, TILE),
                ),
                (BLOCK_K, BLOCK_D),
            )
            kv_nope = kv_raw.to(tl.float32) * kv_scale_full
            kv_rope = tl.load(
                kv_rope_ptr
                + slot[:, None] * kv_rope_stride_n
                + (d_offs - NOPE)[None, :],
                mask=valid[:, None] & rope_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            kv = tl.where(nope_mask[None, :], kv_nope, kv_rope).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if USE_MXFP8_QK:
            # Keep the gathered FP8 tile out of the score/softmax live range.
            # Re-reading V costs bandwidth, but materially lowers register
            # pressure for the 64x512 q4/head tile on gfx950.
            kv_scale_64 = tl.load(
                kv_packed_u8_ptr
                + slot[:, None] * kv_stride_n
                + PACK_OFF_SCALE
                + 2 * scale64_offs[None, :],
                mask=valid[:, None] & (scale64_offs < NUM_TILES)[None, :],
                other=0,
                volatile=True,
            )
            kv_raw = tl.load(
                kv_packed_ptr + slot[:, None] * kv_stride_n + d_offs[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0.0,
                volatile=True,
            )
            kv_scale_full = tl.reshape(
                tl.broadcast_to(
                    _e8m0_scale(kv_scale_64)[:, :, None],
                    (BLOCK_K, 8, TILE),
                ),
                (BLOCK_K, BLOCK_D),
            )
            kv_nope = kv_raw.to(tl.float32) * kv_scale_full
            kv_rope = tl.load(
                kv_rope_ptr
                + slot[:, None] * kv_rope_stride_n
                + (d_offs - NOPE)[None, :],
                mask=valid[:, None] & rope_mask[None, :],
                other=0.0,
                volatile=True,
            ).to(tl.bfloat16)
            kv = tl.where(nope_mask[None, :], kv_nope, kv_rope).to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), kv, acc)
        m_i = m_new

    if KV_SPLITS > 1:
        tl.store(
            m_partial_ptr
            + t_offs * mp_stride_t
            + pid_k * mp_stride_k
            + h_offs * mp_stride_h,
            m_i,
            mask=q_mask,
        )
        tl.store(
            l_partial_ptr
            + t_offs * lp_stride_t
            + pid_k * lp_stride_k
            + h_offs * lp_stride_h,
            l_i,
            mask=q_mask,
        )
        tl.store(
            acc_partial_ptr
            + t_offs[:, None] * ap_stride_t
            + pid_k * ap_stride_k
            + h_offs[:, None] * ap_stride_h
            + d_offs[None, :] * ap_stride_d,
            acc,
            mask=q_mask[:, None],
        )
        return

    sink = (
        tl.load(attn_sink_ptr + h_offs, mask=q_mask, other=neg_large).to(tl.float32)
        * log2e
    )
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0,
        acc * alpha_kv[:, None] / denom[:, None],
        0.0,
    )
    tl.store(
        out_ptr
        + t_offs[:, None] * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :],
        out.to(tl.bfloat16),
        mask=q_mask[:, None],
    )


def sparse_attn_v4_paged_decode_fp8_triton(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    block_h: int = 16,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    waves_per_eu: int = 1,
    matrix_instr_nonkdim: int = 0,
    reduce_d_chunk: int = 512,
    reduce_num_warps: int = 4,
    bf16_partials: bool = False,
    fp16_partials: bool = False,
    use_mxfp8_qk: bool = False,
    use_mxfp8_v: bool = False,
    kv_splits: int = 1,
) -> torch.Tensor:
    """Run the experimental single-pass native-2buff FP8 Triton kernel."""
    if q_packed.dtype != dtypes.fp8 or kv_packed.dtype != dtypes.fp8:
        raise TypeError("q_packed and kv_packed must use aiter.dtypes.fp8")
    if q_rope.dtype != torch.bfloat16 or kv_rope.dtype != torch.bfloat16:
        raise TypeError("q_rope and kv_rope must be BF16")
    if q_packed.dim() != 3 or q_packed.shape[-1] != V4_DIM_QK_PACKED:
        raise ValueError(f"q_packed must be [T,H,{V4_DIM_QK_PACKED}]")
    if q_rope.shape != (*q_packed.shape[:-1], V4_DIM_ROPE):
        raise ValueError("q_rope shape must match q_packed leading dimensions")
    if kv_packed.dim() != 2 or kv_packed.shape[-1] != V4_DIM_QK_PACKED:
        raise ValueError(f"kv_packed must be [P,{V4_DIM_QK_PACKED}]")
    if kv_rope.shape != (kv_packed.shape[0], V4_DIM_ROPE):
        raise ValueError("kv_rope must be [P,64]")
    if block_h not in (8, 16, 32, 64):
        raise ValueError("block_h must be 8, 16, 32, or 64")
    if block_k not in (8, 16, 32, 64):
        raise ValueError("block_k must be one of 8, 16, 32, 64")
    if num_warps not in (2, 4, 8):
        raise ValueError("num_warps must be 2, 4, or 8")
    if num_stages not in (1, 2, 3, 4):
        raise ValueError("num_stages must be one of 1, 2, 3, 4")
    if waves_per_eu not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu must be in [1, 4]")
    if matrix_instr_nonkdim not in (0, 16, 32):
        raise ValueError("matrix_instr_nonkdim must be 0, 16, or 32")
    if reduce_d_chunk not in (64, 128, 256, 512):
        raise ValueError("reduce_d_chunk must be 64, 128, 256, or 512")
    if reduce_num_warps not in (1, 2, 4, 8):
        raise ValueError("reduce_num_warps must be 1, 2, 4, or 8")
    if not 1 <= kv_splits <= 16:
        raise ValueError("kv_splits must be in [1, 16]")
    if bf16_partials and fp16_partials:
        raise ValueError("at most one reduced-precision partial dtype may be selected")
    T, H, _ = q_packed.shape
    if kv_indptr.numel() < T + 1:
        raise ValueError("kv_indptr must contain at least T+1 entries")
    out = torch.empty((T, H, V4_DIM_QK), dtype=torch.bfloat16, device=q_packed.device)
    if kv_splits > 1:
        m_partial = torch.empty(
            (T, kv_splits, H), dtype=torch.float32, device=q_packed.device
        )
        l_partial = torch.empty_like(m_partial)
        acc_partial = torch.empty(
            (T, kv_splits, H, V4_DIM_QK),
            dtype=(
                torch.float16
                if fp16_partials
                else torch.bfloat16 if bf16_partials else torch.float32
            ),
            device=q_packed.device,
        )
    else:
        # Concrete fp32 pointers for the compile-time-dead partial branch.
        m_partial = torch.empty(1, dtype=torch.float32, device=q_packed.device)
        l_partial = m_partial
        acc_partial = m_partial
    grid = (T, triton.cdiv(H, block_h), kv_splits)
    _paged_decode_fp8_2buff_fused_kernel[grid](
        q_packed,
        q_packed.view(torch.uint8),
        q_rope,
        kv_packed,
        kv_packed.view(torch.uint8),
        kv_rope,
        kv_indices,
        kv_indptr,
        attn_sink,
        m_partial,
        l_partial,
        acc_partial,
        out,
        q_packed.stride(0),
        q_packed.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        kv_packed.stride(0),
        kv_rope.stride(0),
        m_partial.stride(0) if kv_splits > 1 else 1,
        m_partial.stride(1) if kv_splits > 1 else 1,
        m_partial.stride(2) if kv_splits > 1 else 1,
        l_partial.stride(0) if kv_splits > 1 else 1,
        l_partial.stride(1) if kv_splits > 1 else 1,
        l_partial.stride(2) if kv_splits > 1 else 1,
        acc_partial.stride(0) if kv_splits > 1 else 1,
        acc_partial.stride(1) if kv_splits > 1 else 1,
        acc_partial.stride(2) if kv_splits > 1 else 1,
        acc_partial.stride(3) if kv_splits > 1 else 1,
        out.stride(0),
        out.stride(1),
        float(softmax_scale) * LOG2E,
        LOG2E,
        H=H,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        BLOCK_D=V4_DIM_QK,
        NOPE=V4_DIM_NOPE,
        ROPE=V4_DIM_ROPE,
        TILE=V4_TILE,
        NUM_TILES=V4_NUM_TILES,
        PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
        PIPE_STAGES=num_stages,
        USE_MXFP8_QK=use_mxfp8_qk,
        USE_MXFP8_V=use_mxfp8_v,
        KV_SPLITS=kv_splits,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )
    if kv_splits > 1:
        reduce_grid = (T, H, triton.cdiv(V4_DIM_QK, reduce_d_chunk))
        reduce_kernel = (
            _paged_decode_reduce2_kernel
            if kv_splits == 2
            else _paged_decode_reduce_kernel
        )
        reduce_args = (
            m_partial,
            l_partial,
            acc_partial,
            attn_sink,
            kv_indptr,
            out,
            m_partial.stride(0),
            m_partial.stride(1),
            m_partial.stride(2),
            l_partial.stride(0),
            l_partial.stride(1),
            l_partial.stride(2),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            acc_partial.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            LOG2E,
        )
        if kv_splits == 2:
            reduce_kernel[reduce_grid](
                *reduce_args,
                V4_DIM_QK,
                D_CHUNK=reduce_d_chunk,
                BLOCK_K=block_k,
                num_warps=reduce_num_warps,
            )
        else:
            reduce_kernel[reduce_grid](
                *reduce_args,
                H,
                V4_DIM_QK,
                kv_splits,
                BLOCK_D=V4_DIM_QK,
                D_CHUNK=reduce_d_chunk,
                BLOCK_K=block_k,
                num_warps=reduce_num_warps,
            )
    return out


def sparse_attn_v4_paged_decode_fp8_triton_query_group(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    query_group: int = 4,
    fused_query_group: int | None = None,
    block_h: int = 16,
    block_k: int = 16,
    kv_splits: int = 1,
    num_stages: int = 3,
    num_warps: int = 8,
    waves_per_eu: int = 1,
    matrix_instr_nonkdim: int = 0,
    use_mxfp8_qk: bool = False,
    reduce_d_chunk: int = 512,
    reduce_num_warps: int = 1,
    bf16_partials: bool = False,
    fp16_partials: bool = False,
) -> torch.Tensor:
    """Fuse adjacent queries and head tiles that share one request's KV."""
    if fused_query_group is None:
        fused_query_group = 4 if query_group == 7 else query_group
    if query_group not in (2, 4, 7):
        raise ValueError("query_group must be 2, 4, or 7")
    if fused_query_group not in (2, 4) or fused_query_group > query_group:
        raise ValueError("fused_query_group must be 2 or 4 and <= query_group")
    if block_h not in (8, 16):
        raise ValueError("block_h must be 8 or 16")
    if block_k not in (16, 32, 64):
        raise ValueError("block_k must be 16, 32, or 64")
    if kv_splits not in (1, 2, 4, 8, 16):
        raise ValueError("kv_splits must be 1, 2, 4, 8, or 16")
    if matrix_instr_nonkdim not in (0, 16, 32):
        raise ValueError("matrix_instr_nonkdim must be 0, 16, or 32")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if bf16_partials and fp16_partials:
        raise ValueError("at most one reduced-precision partial dtype may be selected")
    T, H, _ = q_packed.shape
    if T % query_group:
        raise ValueError("T must be divisible by query_group")
    out = torch.empty((T, H, V4_DIM_QK), dtype=torch.bfloat16, device=q_packed.device)
    if kv_splits > 1:
        m_partial = torch.empty(
            (T, kv_splits, H), dtype=torch.float32, device=q_packed.device
        )
        l_partial = torch.empty_like(m_partial)
        acc_partial = torch.empty(
            (T, kv_splits, H, V4_DIM_QK),
            dtype=(
                torch.float16
                if fp16_partials
                else torch.bfloat16 if bf16_partials else torch.float32
            ),
            device=q_packed.device,
        )
    else:
        m_partial = torch.empty(1, dtype=torch.float32, device=q_packed.device)
        l_partial = m_partial
        acc_partial = m_partial

    query_blocks = triton.cdiv(query_group, fused_query_group)
    grid = (
        (T // query_group) * query_blocks,
        triton.cdiv(H, block_h),
        kv_splits,
    )
    _paged_decode_fp8_query_group_kernel[grid](
        q_packed,
        q_packed.view(torch.uint8),
        q_rope,
        kv_packed,
        kv_packed.view(torch.uint8),
        kv_rope,
        kv_indices,
        kv_indptr,
        attn_sink,
        m_partial,
        l_partial,
        acc_partial,
        out,
        q_packed.stride(0),
        q_packed.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        kv_packed.stride(0),
        kv_rope.stride(0),
        m_partial.stride(0) if kv_splits > 1 else 1,
        m_partial.stride(1) if kv_splits > 1 else 1,
        m_partial.stride(2) if kv_splits > 1 else 1,
        l_partial.stride(0) if kv_splits > 1 else 1,
        l_partial.stride(1) if kv_splits > 1 else 1,
        l_partial.stride(2) if kv_splits > 1 else 1,
        acc_partial.stride(0) if kv_splits > 1 else 1,
        acc_partial.stride(1) if kv_splits > 1 else 1,
        acc_partial.stride(2) if kv_splits > 1 else 1,
        acc_partial.stride(3) if kv_splits > 1 else 1,
        out.stride(0),
        out.stride(1),
        float(softmax_scale) * LOG2E,
        LOG2E,
        T,
        H=H,
        BLOCK_H=block_h,
        REQUEST_QUERY_GROUP=query_group,
        FUSED_QUERY_GROUP=fused_query_group,
        QUERY_BLOCKS=query_blocks,
        BLOCK_M=fused_query_group * block_h,
        BLOCK_K=block_k,
        BLOCK_D=V4_DIM_QK,
        NOPE=V4_DIM_NOPE,
        ROPE=V4_DIM_ROPE,
        TILE=V4_TILE,
        NUM_TILES=V4_NUM_TILES,
        PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
        PIPE_STAGES=num_stages,
        USE_MXFP8_QK=use_mxfp8_qk,
        KV_SPLITS=kv_splits,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )
    if kv_splits > 1:
        reduce_grid = (T, H, triton.cdiv(V4_DIM_QK, reduce_d_chunk))
        reduce_kernel = (
            _paged_decode_reduce2_kernel
            if kv_splits == 2
            else _paged_decode_reduce_kernel
        )
        reduce_args = (
            m_partial,
            l_partial,
            acc_partial,
            attn_sink,
            kv_indptr,
            out,
            m_partial.stride(0),
            m_partial.stride(1),
            m_partial.stride(2),
            l_partial.stride(0),
            l_partial.stride(1),
            l_partial.stride(2),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            acc_partial.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            LOG2E,
        )
        if kv_splits == 2:
            reduce_kernel[reduce_grid](
                *reduce_args,
                V4_DIM_QK,
                D_CHUNK=reduce_d_chunk,
                BLOCK_K=block_k,
                num_warps=reduce_num_warps,
            )
        else:
            reduce_kernel[reduce_grid](
                *reduce_args,
                H,
                V4_DIM_QK,
                kv_splits,
                BLOCK_D=V4_DIM_QK,
                D_CHUNK=reduce_d_chunk,
                BLOCK_K=block_k,
                num_warps=reduce_num_warps,
            )
    return out


def _q7_auto_config(T: int) -> tuple[int, int, int, int, int]:
    """Return (fused_q, block_k, splits, stages, reduce_warps) for q7."""
    requests = T // 7
    if requests <= 2:
        return 1, 16, 16, 2, 4
    if requests <= 8:
        return 4, 16, 16, 2, 1
    return 4, 16, 8, 2, 1


def _q7_dp_auto_config(T: int, kv_kind: str) -> tuple[int, int, int]:
    """Return (block_k, splits, stages) for H=128 DP-attention q7."""
    requests = T // 7
    if requests <= 1:
        return (16, 16, 2) if kv_kind == "csa" else (64, 16, 1)
    if requests <= 2:
        return (16, 8, 2) if kv_kind == "csa" else (32, 8, 1)
    if requests <= 4:
        return 64, 4, 1
    if requests <= 8:
        return 64, 2, 1
    return 64, 4, 1


def _q7_dp_csa_auto_config(T: int) -> tuple[int, int, int, int]:
    """Return (block_k, splits, stages, matrix_nonkdim) for DP q7 CSA.

    CSA has a fixed 1152-row gathered working set in the AgentX workload.  A
    single query over 64 heads matches the hardware's qh64 decomposition and
    avoids the occupancy cliffs of the 4-query x 16-head tile.  The split-K
    choices below cover the exact per-rank request counts observed by the DP
    scheduler; counts above the measured range use the stable no-split tile.
    """
    requests = T // 7
    configs = {
        1: (16, 16, 2, 16),
        2: (16, 8, 2, 16),
        3: (32, 4, 3, 16),
        4: (32, 4, 3, 16),
        5: (64, 4, 1, 0),
        6: (32, 3, 3, 16),
        7: (32, 2, 3, 16),
        8: (32, 2, 3, 16),
        9: (32, 2, 3, 16),
        10: (32, 3, 2, 16),
        11: (32, 3, 2, 16),
        12: (32, 3, 2, 16),
        13: (32, 4, 2, 16),
    }
    return configs.get(requests, (32, 1, 2, 16))


def _q7_dp_hca_regular_config(T: int) -> tuple[int, int, int, int] | None:
    """Return the regular qh64 config for the hot DP q7 HCA batches.

    Once a rank has five or more requests, the extra parallelism from issuing
    one program per query outweighs the KV reuse of the q4/h16 kernel.  The
    bk32 variants are particularly important at B7--B9 and B14+, where the
    old grouped path is 3--22% slower across the measured 384--4224 row HCA
    windows.  B1--B4 keep query fusion because they do not expose enough
    independent regular programs.
    """
    requests = T // 7
    if requests < 5:
        return None
    if requests == 5:
        return 64, 8, 1, 0
    if requests == 6:
        return 32, 8, 2, 16
    if requests <= 9:
        return 32, 2, 2, 16
    if requests == 10:
        return 32, 7, 2, 16
    if requests <= 12:
        return 32, 3, 2, 16
    if requests == 13:
        return 32, 4, 2, 16
    return 32, 2, 2, 16


def _dspark_auto_config(T: int) -> tuple[int, int, int]:
    """Return (block_k, splits, stages) for the six-row DSpark draft block."""
    requests = triton.cdiv(T, 6)
    if requests <= 2:
        return 16, 16, 2
    if requests <= 5:
        return 32, 8, 2
    if requests <= 12:
        return 16, 4, 2
    if requests <= 24:
        return 16, 2, 2
    return 16, 1, 2


def sparse_attn_v4_paged_decode_fp8_triton_auto(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    query_group: int = 1,
    kv_kind: str = "",
) -> torch.Tensor:
    """Dispatch tuned Triton specializations without falling back to ASM."""
    T, H, _ = q_packed.shape
    if kv_kind == "dspark" and H == 16:
        block_k, kv_splits, stages = _dspark_auto_config(T)
        return sparse_attn_v4_paged_decode_fp8_triton(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            block_h=16,
            block_k=block_k,
            kv_splits=kv_splits,
            num_stages=stages,
            num_warps=4,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            use_mxfp8_qk=True,
            reduce_d_chunk=512,
            reduce_num_warps=1,
            fp16_partials=True,
        )
    if query_group == 7:
        if H == 128:
            if kv_kind == "csa":
                block_k, kv_splits, stages, matrix_nonkdim = _q7_dp_csa_auto_config(T)
                return sparse_attn_v4_paged_decode_fp8_triton(
                    q_packed,
                    q_rope,
                    kv_packed,
                    kv_rope,
                    kv_indices,
                    kv_indptr,
                    attn_sink,
                    softmax_scale,
                    block_h=64,
                    block_k=block_k,
                    kv_splits=kv_splits,
                    num_stages=stages,
                    num_warps=4,
                    waves_per_eu=1,
                    matrix_instr_nonkdim=matrix_nonkdim,
                    use_mxfp8_qk=True,
                    reduce_d_chunk=512,
                    reduce_num_warps=1,
                    fp16_partials=True,
                )
            hca_config = _q7_dp_hca_regular_config(T) if kv_kind == "hca" else None
            if hca_config is not None:
                block_k, kv_splits, stages, matrix_nonkdim = hca_config
                return sparse_attn_v4_paged_decode_fp8_triton(
                    q_packed,
                    q_rope,
                    kv_packed,
                    kv_rope,
                    kv_indices,
                    kv_indptr,
                    attn_sink,
                    softmax_scale,
                    block_h=64,
                    block_k=block_k,
                    kv_splits=kv_splits,
                    num_stages=stages,
                    num_warps=4,
                    waves_per_eu=1,
                    matrix_instr_nonkdim=matrix_nonkdim,
                    use_mxfp8_qk=True,
                    reduce_d_chunk=512,
                    reduce_num_warps=1,
                    fp16_partials=True,
                )
            # Low-batch DP attention still benefits from sharing each KV
            # traversal across four adjacent verification queries.
            block_k, kv_splits, stages = _q7_dp_auto_config(T, kv_kind)
            return sparse_attn_v4_paged_decode_fp8_triton_query_group(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                kv_indices,
                kv_indptr,
                attn_sink,
                softmax_scale,
                query_group=7,
                fused_query_group=4,
                block_h=16,
                block_k=block_k,
                kv_splits=kv_splits,
                num_stages=stages,
                num_warps=4,
                waves_per_eu=1,
                matrix_instr_nonkdim=0,
                use_mxfp8_qk=True,
                reduce_d_chunk=512,
                reduce_num_warps=1,
                fp16_partials=True,
            )
        # DSpark K6 verifies seven target rows per request.  C1/C2 use the
        # regular head-tiled kernel with split-K to maximize occupancy.  At
        # C4+, two four-query stripes per request reuse each KV row across up
        # to four queries without the prohibitive register footprint of one
        # monolithic q7 program.  This wins across the measured 8K--512K
        # range, especially for long-context HCA.  The policy depends only on
        # the captured q shape and remains CUDA-Graph safe.
        fused_q, block_k, kv_splits, stages, reduce_warps = _q7_auto_config(T)
        common_args = (
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
        )
        common_kwargs = {
            "block_h": 16,
            "block_k": block_k,
            "kv_splits": kv_splits,
            "num_stages": stages,
            "num_warps": 4,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "use_mxfp8_qk": True,
            "reduce_d_chunk": 512,
            "reduce_num_warps": reduce_warps,
            "fp16_partials": True,
        }
        if fused_q == 1 or H != 16:
            return sparse_attn_v4_paged_decode_fp8_triton(*common_args, **common_kwargs)
        common_kwargs.pop("block_h")
        return sparse_attn_v4_paged_decode_fp8_triton_query_group(
            *common_args,
            query_group=7,
            fused_query_group=fused_q,
            **common_kwargs,
        )
    if query_group == 4 and H == 16 and T >= 1024:
        return sparse_attn_v4_paged_decode_fp8_triton_query_group(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            query_group=4,
            block_k=16,
            kv_splits=1,
            num_stages=3,
            num_warps=8,
            waves_per_eu=2 if T == 1024 and kv_kind == "csa" else 1,
            use_mxfp8_qk=False,
        )
    if query_group == 4 and T == 64 and kv_kind == "hca":
        # At C16 the smaller per-program register footprint of the regular
        # head tile wins even though each of the four verify rows reloads KV.
        # Four splits keep the launch at 256 CTAs and cut partial traffic in
        # half versus the best MTP2 configuration.
        return sparse_attn_v4_paged_decode_fp8_triton(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            block_h=16,
            block_k=16,
            kv_splits=4,
            num_stages=3,
            num_warps=8,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            use_mxfp8_qk=False,
            reduce_d_chunk=512,
            reduce_num_warps=1,
            fp16_partials=True,
        )
    if query_group == 4 and H == 16 and T == 64 and kv_kind == "csa":
        return sparse_attn_v4_paged_decode_fp8_triton_query_group(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            query_group=4,
            block_k=16,
            kv_splits=16,
            num_stages=3,
            num_warps=4,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            use_mxfp8_qk=False,
            reduce_num_warps=1,
            fp16_partials=True,
        )
    if query_group == 4 and H == 16 and kv_kind == "csa":
        return sparse_attn_v4_paged_decode_fp8_triton_query_group(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            query_group=4,
            block_k=16,
            kv_splits=4 if T >= 256 else 8,
            num_stages=3,
            num_warps=4,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            use_mxfp8_qk=False,
            reduce_num_warps=1,
            fp16_partials=True,
        )

    if query_group == 4 and H == 16 and T == 256 and kv_kind == "hca":
        return sparse_attn_v4_paged_decode_fp8_triton_query_group(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            query_group=4,
            block_k=16,
            kv_splits=4,
            num_stages=3,
            num_warps=4,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            use_mxfp8_qk=False,
            reduce_d_chunk=512,
            reduce_num_warps=1,
            fp16_partials=True,
        )

    if T <= 64:
        block_k = 32
        kv_splits = 1
        stages = 3
        use_mx = True
        waves = 1
    elif T <= 256:
        block_k = 32
        kv_splits = 1
        stages = 3
        use_mx = False
        waves = 1
    else:
        block_k = 16
        kv_splits = 1
        stages = 3
        use_mx = False
        waves = 2
    return sparse_attn_v4_paged_decode_fp8_triton(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        block_k=block_k,
        kv_splits=kv_splits,
        num_stages=stages,
        num_warps=4,
        waves_per_eu=waves,
        matrix_instr_nonkdim=16 if T <= 256 else 0,
        use_mxfp8_qk=use_mx,
    )


def sparse_attn_v4_paged_decode_fp8_triton_twopass(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    kv_packed: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    qk_block_k: int = 32,
    qk_num_stages: int = 3,
    pv_block_k: int = 32,
    pv_num_stages: int = 3,
    num_warps: int = 4,
    waves_per_eu: int = 1,
) -> torch.Tensor:
    """Materialize BF16 scores, then run eight spill-free 64-wide PV CTAs."""
    if q_packed.dtype != dtypes.fp8 or kv_packed.dtype != dtypes.fp8:
        raise TypeError("q_packed and kv_packed must use aiter.dtypes.fp8")
    if q_rope.dtype != torch.bfloat16 or kv_rope.dtype != torch.bfloat16:
        raise TypeError("q_rope and kv_rope must be BF16")
    if qk_block_k not in (16, 32, 64) or pv_block_k not in (16, 32, 64):
        raise ValueError("two-pass block_k values must be 16, 32, or 64")
    if qk_num_stages not in (1, 2, 3, 4) or pv_num_stages not in (1, 2, 3, 4):
        raise ValueError("two-pass num_stages values must be in [1, 4]")

    T, H, _ = q_packed.shape
    scores = torch.empty(
        (kv_indices.numel(), H), dtype=torch.bfloat16, device=q_packed.device
    )
    out = torch.empty((T, H, V4_DIM_QK), dtype=torch.bfloat16, device=q_packed.device)
    block_h = 16
    _paged_decode_fp8_qk_store_kernel[(T, triton.cdiv(H, block_h))](
        q_packed,
        q_packed.view(torch.uint8),
        q_rope,
        kv_packed,
        kv_packed.view(torch.uint8),
        kv_rope,
        kv_indices,
        kv_indptr,
        scores,
        q_packed.stride(0),
        q_packed.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        kv_packed.stride(0),
        kv_rope.stride(0),
        scores.stride(0),
        scores.stride(1),
        float(softmax_scale) * LOG2E,
        H=H,
        BLOCK_H=block_h,
        BLOCK_K=qk_block_k,
        BLOCK_D=V4_DIM_QK,
        NOPE=V4_DIM_NOPE,
        ROPE=V4_DIM_ROPE,
        PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
        PIPE_STAGES=qk_num_stages,
        num_warps=num_warps,
        num_stages=qk_num_stages,
        waves_per_eu=waves_per_eu,
    )
    _paged_decode_fp8_grouped_pv_kernel[(T, V4_NUM_TILES + 1)](
        scores,
        kv_packed,
        kv_packed.view(torch.uint8),
        kv_rope,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        scores.stride(0),
        scores.stride(1),
        kv_packed.stride(0),
        kv_rope.stride(0),
        out.stride(0),
        out.stride(1),
        LOG2E,
        H=H,
        BLOCK_H=block_h,
        BLOCK_K=pv_block_k,
        TILE=V4_TILE,
        NUM_TILES=V4_NUM_TILES,
        PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
        PIPE_STAGES=pv_num_stages,
        num_warps=num_warps,
        num_stages=pv_num_stages,
        waves_per_eu=waves_per_eu,
    )
    return out
