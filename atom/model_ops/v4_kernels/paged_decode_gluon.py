# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-pass sparse decode with shared KV panels on CDNA3/4."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _load_kv_panel(kv_ptr, slots, positions, channels, length, row_stride, dim_stride):
    # A real KV pool can exceed int32 element offsets.
    offsets = slots.to(gl.int64)[None, :] * row_stride + channels[:, None] * dim_stride
    return gl.load(kv_ptr + offsets, mask=positions[None, :] < length, other=0)


@gluon.jit
def _paged_decode_fused_gluon_kernel(
    q_ptr,
    unified_kv_ptr,
    _kv_scales_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    _ks_stride_n,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    qk_scale,
    log2e,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    QUANT_KV: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    NUM_GROUPS: gl.constexpr,
    CDNA_VERSION: gl.constexpr,
):
    # Share the existing fused launch signature with the quantized fallback.
    gl.static_assert(BLOCK_D == 512)
    gl.static_assert(D == 512 and (BLOCK_H == 16 or BLOCK_H == 32) and not QUANT_KV)
    gl.static_assert(CDNA_VERSION == 3 or CDNA_VERSION == 4)
    gl.static_assert(BLOCK_K == 32 or (CDNA_VERSION == 4 and BLOCK_K == 64))
    # H16 uses its register budget for a wider tile; H32 prefetches a half panel.
    PREFETCH: gl.constexpr = BLOCK_H == 32
    t = gl.program_id(0)
    pid_h = gl.program_id(1)
    HEAD_WAVES: gl.constexpr = 1 if BLOCK_H <= 16 else 2
    KV_WAVES: gl.constexpr = 4 // HEAD_WAVES
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16, 32 if CDNA_VERSION == 4 else 16],
        transposed=True,
        warps_per_cta=[HEAD_WAVES, KV_WAVES],
    )
    # Match operand packing to the PV reduction width.
    K_WIDTH: gl.constexpr = 8 if BLOCK_K == 32 else 16
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, K_WIDTH)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, K_WIDTH)
    q_layout: gl.constexpr = gl.BlockedLayout(
        [1, K_WIDTH], [16, 4], [4, 1], [0, 1] if BLOCK_H == 16 else [1, 0]
    )
    kv_layout: gl.constexpr = gl.BlockedLayout(
        [8, 1], [4, 16], [HEAD_WAVES, KV_WAVES], [0, 1]
    )
    h = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    d = gl.arange(0, BLOCK_D // 2, layout=gl.SliceLayout(0, q_layout))
    q0 = gl.load(
        q_ptr + t * q_stride_t + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        mask=(h[:, None] < H) & (d[None, :] < D),
        other=0,
    )
    q0 = gl.convert_layout(q0, lhs)
    q1 = gl.load(
        q_ptr
        + t * q_stride_t
        + h[:, None] * q_stride_h
        + (d[None, :] + BLOCK_D // 2) * q_stride_d,
        mask=(h[:, None] < H) & (d[None, :] < D),
        other=0,
    )
    q1 = gl.convert_layout(q1, lhs)
    channels = gl.arange(0, BLOCK_D // 2, layout=gl.SliceLayout(1, kv_layout))
    token = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, kv_layout))
    score_token = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, matrix))
    start = gl.load(kv_indptr_ptr + t)
    length = gl.load(kv_indptr_ptr + t + 1) - start
    neg: gl.constexpr = -3.4028234663852886e38
    maximum = gl.full((BLOCK_H,), neg, gl.float32, gl.SliceLayout(1, matrix))
    denominator = gl.full((BLOCK_H,), 0, gl.float32, gl.SliceLayout(1, matrix))
    acc0 = gl.full((BLOCK_H, BLOCK_D // 2), 0, gl.float32, matrix)
    acc1 = gl.full((BLOCK_H, BLOCK_D // 2), 0, gl.float32, matrix)
    # QK consumes register panels; PV reuses those KV rows through one LDS tile.
    if CDNA_VERSION == 4:
        kv_shared_layout: gl.constexpr = (
            gl.amd.cdna4.compute_efficient_padded_shared_layout(
                rhs,
                [BLOCK_K, BLOCK_D],
                unified_kv_ptr.dtype.element_ty,
                is_k_contig=False,
            )
        )
    else:
        # CDNA3 has 32 LDS banks and 64 KiB per CTA; use an unpadded K32 tile.
        kv_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    kv_shared = gl.allocate_shared_memory(
        unified_kv_ptr.dtype.element_ty, (BLOCK_K, BLOCK_D), kv_shared_layout
    )
    kv_shared0 = kv_shared.slice(0, BLOCK_D // 2, 1)
    kv_shared1 = kv_shared.slice(BLOCK_D // 2, BLOCK_D // 2, 1)
    prob_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    prob_shared = gl.allocate_shared_memory(
        q_ptr.dtype.element_ty, (BLOCK_H, BLOCK_K), prob_shared_layout
    )
    if PREFETCH:
        slots = gl.load(kv_indices_ptr + start + token, mask=token < length, other=0)
        raw0 = _load_kv_panel(
            unified_kv_ptr, slots, token, channels, length, kv_stride_n, kv_stride_d
        )
    for offset in range(0, length, BLOCK_K):
        if not PREFETCH:
            slots = gl.load(
                kv_indices_ptr + start + offset + token,
                mask=offset + token < length,
                other=0,
            )
            raw0 = _load_kv_panel(
                unified_kv_ptr,
                slots,
                offset + token,
                channels,
                length,
                kv_stride_n,
                kv_stride_d,
            )
        raw1 = _load_kv_panel(
            unified_kv_ptr,
            slots,
            offset + token,
            channels + BLOCK_D // 2,
            length,
            kv_stride_n,
            kv_stride_d,
        )
        kv_shared0.store(raw0.permute((1, 0)))
        kv_shared1.store(raw1.permute((1, 0)))
        key0 = gl.convert_layout(raw0, rhs)
        key1 = gl.convert_layout(raw1, rhs)
        scores = gl.amd.cdna3.mfma(
            q0, key0, gl.full((BLOCK_H, BLOCK_K), 0, gl.float32, matrix)
        )
        scores = gl.amd.cdna3.mfma(q1, key1, scores) * qk_scale
        scores = gl.where(offset + score_token[None, :] < length, scores, neg)
        if PREFETCH:
            next_slots = gl.load(
                kv_indices_ptr + start + offset + BLOCK_K + token,
                mask=offset + BLOCK_K + token < length,
                other=0,
            )
            next_raw0 = _load_kv_panel(
                unified_kv_ptr,
                next_slots,
                offset + BLOCK_K + token,
                channels,
                length,
                kv_stride_n,
                kv_stride_d,
            )
        m_new = gl.maximum(maximum, gl.max(scores, 1))
        correction = gl.exp2(maximum - m_new)
        probability = gl.exp2(scores - m_new[:, None])
        denominator = denominator * correction + gl.sum(probability, 1)
        prob_shared.store(probability.to(q_ptr.dtype.element_ty))
        value0 = kv_shared0.load(rhs)
        value1 = kv_shared1.load(rhs)
        p = prob_shared.load(lhs)
        acc0 = gl.amd.cdna3.mfma(p, value0, acc0 * correction[:, None])
        acc1 = gl.amd.cdna3.mfma(p, value1, acc1 * correction[:, None])
        maximum = m_new
        if PREFETCH:
            slots = next_slots
            raw0 = next_raw0
    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [8, 8], [HEAD_WAVES, KV_WAVES], [1, 0]
    )
    # Preserve V4 sink normalization and zero-length CSR behavior.
    sink_heads = pid_h * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, matrix)
    )
    sink = gl.load(attn_sink_ptr + sink_heads, mask=sink_heads < H, other=neg) * log2e
    final = gl.maximum(maximum, sink)
    alpha = gl.exp2(maximum - final)
    total = denominator * alpha + gl.exp2(sink - final)
    scale = alpha / gl.maximum(total, 1e-30)
    out_heads = pid_h * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, out_layout)
    )
    out_dims = gl.arange(0, BLOCK_D // 2, layout=gl.SliceLayout(0, out_layout))
    for panel in gl.static_range(2):
        acc = acc0 if panel == 0 else acc1
        result = gl.where(total[:, None] > 0, acc * scale[:, None], 0)
        result = gl.convert_layout(result.to(out_ptr.dtype.element_ty), out_layout)
        gl.store(
            out_ptr
            + t * out_stride_t
            + out_heads[:, None] * out_stride_h
            + (out_dims[None, :] + panel * (BLOCK_D // 2)) * out_stride_d,
            result,
            mask=out_heads[:, None] < H,
        )
