# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MLA decode K-up (BF16 absorb) + q_pe RoPE + shuffled KV-cache write in one
persistent Gluon kernel for gfx1250.

Targets the arithmetic and output layout of the pair
  batched_gemm_bf16(q_nope, W_K, YQ=q_out[..., :L].transpose(0, 1))
  fused_qk_rope_cat_and_cache_mla(..., q_out=q_out, shuffled_kv_cache=True,
                                  q_nope_prestored=True)
on pure decode. One CTA owns (head, 256 columns of L, slice of M) and streams M tiles:
W_K's slice stays in LDS, A / q_pe tiles are double-buffered TDM loads, the fp8
q_nope tile and the roped q_pe tile leave through TDM stores, and each CTA writes
one KV token per tile. Same arithmetic as the production kernels: one WMMA K
tile, fp32 -> bf16 -> fp8, bf16 RoPE, (x.f32 * (1 / k_scale)).to(fp8).

TDM ops per CTA, one in-order counter: W, then (A, P, D, DQ) x 2 as prologue
(D / DQ are out-of-bounds loads into the store slots, i.e. "slot free" tokens),
then per tile i: A(i+2), P(i+2), S(i), SQ(i). wait(4) at the top of tile i
leaves A(i), P(i) landed and S(i-2), SQ(i-2) drained.
"""

import functools

import torch
import triton.experimental.gluon.language as gl
from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_kv_cache import (
    _freq_from_shared_1d,
    _freq_from_shared_2d,
    _rope_pe,
    _rope_pe_2d,
    _store_mla_kv_cache,
)
from triton.experimental import gluon

from atom.model_ops.mla_fused import _KUP_QPE_LDS_BUDGET, _kup_qpe_lds_bytes


@gluon.jit
def _kv_idx_load(
    t,
    M,
    pos_ptr,
    slot_mapping_ptr,
    pos_stride_b,
    R: gl.constexpr,
    num_warps: gl.constexpr,
):
    # The aiter cache-store helper takes one uniform slot per CTA.
    LH1: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    tc = gl.minimum(t, M - 1).to(gl.int64)
    zh = gl.zeros([R // 2], gl.int64, layout=LH1)
    valid = t < M
    slot = gl.load(slot_mapping_ptr + tc).to(gl.int64)
    slot = gl.where(valid, slot, -1)
    pos = gl.load(pos_ptr + tc * pos_stride_b + zh).to(gl.int64)
    return slot, pos


@gluon.jit
def _kv_data_load(
    t,
    M,
    pos,
    k_nope_ptr,
    k_pe_ptr,
    cos_ptr,
    sin_ptr,
    k_nope_stride_b,
    k_pe_stride_b,
    cos_stride_b,
    L: gl.constexpr,
    R: gl.constexpr,
    num_warps: gl.constexpr,
):
    LN1: gl.constexpr = gl.BlockedLayout(
        [L // (32 * num_warps)], [32], [num_warps], [0]
    )
    LH1: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    LP1: gl.constexpr = gl.BlockedLayout([2], [32], [num_warps], [0])
    tc = gl.minimum(t, M - 1).to(gl.int64)
    dn = gl.arange(0, L, layout=LN1)
    dp = gl.arange(0, R, layout=LP1)
    dh = gl.arange(0, R // 2, layout=LH1)
    k_nope = gl.load(k_nope_ptr + tc * k_nope_stride_b + dn)
    k_pe = gl.load(k_pe_ptr + tc * k_pe_stride_b + dp)
    c = gl.load(cos_ptr + pos * cos_stride_b + dh)
    s = gl.load(sin_ptr + pos * cos_stride_b + dh)
    return k_nope, k_pe, c, s


@gluon.jit
def _kv_values(
    k_nope,
    k_pe,
    c,
    s,
    k_scale_rcprl,
    kv_cache_ptr,
    R: gl.constexpr,
    IS_NEOX: gl.constexpr,
    num_warps: gl.constexpr,
):
    LP1: gl.constexpr = gl.BlockedLayout([2], [32], [num_warps], [0])
    LH1: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    cf_buf = gl.allocate_shared_memory(c.dtype, [R // 2], SH)
    sf_buf = gl.allocate_shared_memory(s.dtype, [R // 2], SH)
    cf_buf.store(c)
    sf_buf.store(s)
    gl.barrier()
    dp = gl.arange(0, R, layout=LP1)
    cf = _freq_from_shared_1d(cf_buf, True, IS_NEOX, R, LP1, LH1)
    sf = _freq_from_shared_1d(sf_buf, True, IS_NEOX, R, LP1, LH1)
    k_pe = _rope_pe(k_pe, cf, sf, dp, IS_NEOX, R, R // 2)
    kn8 = (k_nope.to(gl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
    kp8 = (k_pe.to(gl.float32) * k_scale_rcprl).to(kv_cache_ptr.dtype.element_ty)
    return kn8, kp8


@gluon.jit
def _kv_put(
    kn8,
    kp8,
    slot,
    kv_cache_ptr,
    kv_cache_stride_b,
    L: gl.constexpr,
    R: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
):
    # The shared aiter helper owns the shuffled layout and fp8 K_WIDTH. Never
    # form a pointer for sentinel slots, including padded graph-replay rows.
    LN1: gl.constexpr = gl.BlockedLayout(
        [L // (32 * num_warps)], [32], [num_warps], [0]
    )
    LP1: gl.constexpr = gl.BlockedLayout([2], [32], [num_warps], [0])
    dn = gl.arange(0, L, layout=LN1)
    dp = gl.arange(0, R, layout=LP1)
    if slot >= 0:
        _store_mla_kv_cache(
            kv_cache_ptr,
            slot // BLOCK_SIZE,
            0,
            slot % BLOCK_SIZE,
            dn,
            dp,
            kv_cache_stride_b,
            0,
            1,
            kn8,
            kp8,
            L,
            R,
            BLOCK_SIZE,
            True,
            0,
            0,
            LN1,
            LP1,
            0,
            num_warps,
        )


@gluon.jit
def _qpe_pos_load(
    m_row0,
    M,
    pos_ptr,
    pos_stride_b,
    HALF_M: gl.constexpr,
    R: gl.constexpr,
    num_warps: gl.constexpr,
):
    CL: gl.constexpr = gl.BlockedLayout(
        [1, 4], [32 // (R // 8), R // 8], [num_warps, 1], [1, 0]
    )
    r = m_row0 + gl.arange(0, HALF_M, layout=gl.SliceLayout(1, CL))
    rc = gl.minimum(r, M - 1)
    return gl.load(pos_ptr + rc * pos_stride_b).to(gl.int64)


@gluon.jit
def _qpe_freq_load(
    pos, cos_ptr, sin_ptr, cos_stride_b, R: gl.constexpr, num_warps: gl.constexpr
):
    CL: gl.constexpr = gl.BlockedLayout(
        [1, 4], [32 // (R // 8), R // 8], [num_warps, 1], [1, 0]
    )
    dh = gl.arange(0, R // 2, layout=gl.SliceLayout(0, CL))
    fo = pos[:, None] * cos_stride_b + dh[None, :]
    return gl.load(cos_ptr + fo), gl.load(sin_ptr + fo)


@gluon.jit
def _kup_qpe_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    qpe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    M,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bn,
    stride_qo_m,
    stride_qo_h,
    stride_qpe_m,
    stride_qpe_h,
    k_nope_stride_b,
    k_pe_stride_b,
    pos_stride_b,
    cos_stride_b,
    kv_cache_stride_b,
    H: gl.constexpr,
    L: gl.constexpr,
    P: gl.constexpr,
    R: gl.constexpr,
    M_SPLIT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    IS_NEOX: gl.constexpr,
    HAVE_K_SCALE: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    num_warps: gl.constexpr,
):
    NSPLIT_N: gl.constexpr = L // BLOCK_N
    BLOCK_K: gl.constexpr = P
    HALF_M: gl.constexpr = BLOCK_M // NSPLIT_N
    NUM_CTAS: gl.constexpr = H * NSPLIT_N * M_SPLIT
    pid = gl.program_id(0)
    if NUM_XCDS > 1:
        # Workgroups go round-robin over XCDs: give each XCD a run of adjacent
        # heads (their q_out rows share 128B lines) and all N slices of a head.
        xcd = pid % NUM_XCDS
        j = pid // NUM_XCDS
        nh = j % NSPLIT_N
        hm = xcd * (H * M_SPLIT // NUM_XCDS) + j // NSPLIT_N
        head = hm // M_SPLIT
        ms = hm % M_SPLIT
    else:
        head = pid // (NSPLIT_N * M_SPLIT)
        rem = pid % (NSPLIT_N * M_SPLIT)
        nh = rem % NSPLIT_N
        ms = rem // NSPLIT_N
    n_off = nh * BLOCK_N

    num_m_tiles = gl.cdiv(M, BLOCK_M)
    chunk = gl.cdiv(num_m_tiles, M_SPLIT)
    t0 = ms * chunk
    t1 = gl.minimum(t0 + chunk, num_m_tiles)
    n_local = gl.maximum(t1 - t0, 0)
    r0 = t0 * BLOCK_M
    r_end = gl.minimum(M, t1 * BLOCK_M)
    # Empty M partitions still own KV tokens. Keep their descriptors valid
    # while leaving the GEMM loop empty: no 0-row shape or one-past-end base.
    r_base = gl.minimum(r0, M - 1)
    desc_rows = gl.maximum(r_end - r0, 1)

    SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_M, BLOCK_N], [1, 0]
    )
    SHARED_LAYOUT_P: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[R, 8]], [HALF_M, R], [1, 0]
    )
    SHARED_LAYOUT_Q: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[R, 16]], [HALF_M, R], [1, 0]
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A
    )
    w_buf = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B
    )
    c_buf = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [2, BLOCK_M, BLOCK_N], SHARED_LAYOUT_C
    )
    p_buf = gl.allocate_shared_memory(
        qpe_ptr.type.element_ty, [2, HALF_M, R], SHARED_LAYOUT_P
    )
    q_buf = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [2, HALF_M, R], SHARED_LAYOUT_Q
    )

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + head * stride_ab + r_base.to(gl.int64) * stride_am,
        shape=(desc_rows, BLOCK_K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + head * stride_bb + n_off * stride_bn,
        shape=(BLOCK_N, BLOCK_K),
        strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + head * stride_qo_h,
        shape=(M, L),
        strides=(stride_qo_m, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_LAYOUT_C,
    )
    p_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=qpe_ptr + head * stride_qpe_h + r_base.to(gl.int64) * stride_qpe_m,
        shape=(desc_rows, R),
        strides=(stride_qpe_m, 1),
        block_shape=(HALF_M, R),
        layout=SHARED_LAYOUT_P,
    )
    q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + head * stride_qo_h + L,
        shape=(M, R),
        strides=(stride_qo_m, 1),
        block_shape=(HALF_M, R),
        layout=SHARED_LAYOUT_Q,
    )

    gl.amd.gfx1250.tdm.async_load(w_desc, [0, 0], w_buf)
    for jj in gl.static_range(2):
        gl.amd.gfx1250.tdm.async_load(a_desc, [jj * BLOCK_M, 0], a_buf.index(jj))
        gl.amd.gfx1250.tdm.async_load(
            p_desc, [jj * BLOCK_M + nh * HALF_M, 0], p_buf.index(jj)
        )
        gl.amd.gfx1250.tdm.async_load(c_desc, [M, 0], c_buf.index(jj))
        gl.amd.gfx1250.tdm.async_load(q_desc, [M, 0], q_buf.index(jj))

    if HAVE_K_SCALE:
        k_scale = gl.load(k_scale_ptr)
    else:
        k_scale = 1.0
    k_scale_rcprl = (1 / k_scale).to(gl.float32)

    LQ: gl.constexpr = gl.BlockedLayout(
        [1, 8], [32 // (R // 8), R // 8], [num_warps, 1], [1, 0]
    )
    dq = gl.arange(0, R, layout=gl.SliceLayout(0, LQ))
    LF: gl.constexpr = gl.BlockedLayout(
        [1, 4], [32 // (R // 8), R // 8], [num_warps, 1], [1, 0]
    )
    SH_F: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    cf_buf = gl.allocate_shared_memory(cos_ptr.dtype.element_ty, [HALF_M, R // 2], SH_F)
    sf_buf = gl.allocate_shared_memory(sin_ptr.dtype.element_ty, [HALF_M, R // 2], SH_F)

    # q_pe: positions two tiles ahead, cos/sin one tile ahead.
    pq0 = _qpe_pos_load(
        r0 + nh * HALF_M, M, pos_ptr, pos_stride_b, HALF_M, R, num_warps
    )
    pq1 = _qpe_pos_load(
        r0 + BLOCK_M + nh * HALF_M, M, pos_ptr, pos_stride_b, HALF_M, R, num_warps
    )
    cq, sq = _qpe_freq_load(pq0, cos_ptr, sin_ptr, cos_stride_b, R, num_warps)
    # KV: token pid + i * NUM_CTAS at tile i; index two ahead, data one ahead.
    ks0, kpos0 = _kv_idx_load(
        pid, M, pos_ptr, slot_mapping_ptr, pos_stride_b, R, num_warps
    )
    ks1, kpos1 = _kv_idx_load(
        pid + NUM_CTAS, M, pos_ptr, slot_mapping_ptr, pos_stride_b, R, num_warps
    )
    kn, kp, kc, ks = _kv_data_load(
        pid, M, kpos0, k_nope_ptr, k_pe_ptr, cos_ptr, sin_ptr,
        k_nope_stride_b, k_pe_stride_b, cos_stride_b, L, R, num_warps,
    )  # fmt: skip

    for i in range(n_local):
        slot = i % 2
        m0 = (t0 + i) * BLOCK_M
        gl.amd.gfx1250.tdm.async_wait(4)
        gl.barrier()
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buf.index(slot), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            w_buf.permute([1, 0]), OPERAND_LAYOUT_B
        )
        qpe = p_buf.index(slot).load(LQ)
        cf_buf.store(cq)
        sf_buf.store(sq)
        gl.barrier()
        cf = _freq_from_shared_2d(cf_buf, True, IS_NEOX, HALF_M, R, LQ, LF)
        sf = _freq_from_shared_2d(sf_buf, True, IS_NEOX, HALF_M, R, LQ, LF)
        qr = _rope_pe_2d(qpe, cf, sf, dq, IS_NEOX, HALF_M, R, R // 2)
        q_buf.index(slot).store(qr.to(c_ptr.type.element_ty))
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)

        pq2 = _qpe_pos_load(
            m0 + 2 * BLOCK_M + nh * HALF_M,
            M,
            pos_ptr,
            pos_stride_b,
            HALF_M,
            R,
            num_warps,
        )
        cq, sq = _qpe_freq_load(pq1, cos_ptr, sin_ptr, cos_stride_b, R, num_warps)
        pq1 = pq2
        kv_n, kv_p = _kv_values(
            kn, kp, kc, ks, k_scale_rcprl, kv_cache_ptr, R, IS_NEOX, num_warps
        )
        ks_s = ks0
        ks2, kpos2 = _kv_idx_load(
            pid + (i + 2) * NUM_CTAS, M, pos_ptr, slot_mapping_ptr, pos_stride_b,
            R, num_warps,
        )  # fmt: skip
        kn, kp, kc, ks = _kv_data_load(
            pid + (i + 1) * NUM_CTAS, M, kpos1, k_nope_ptr, k_pe_ptr, cos_ptr, sin_ptr,
            k_nope_stride_b, k_pe_stride_b, cos_stride_b, L, R, num_warps,
        )  # fmt: skip
        ks0 = ks1
        ks1 = ks2
        kpos1 = kpos2

        c_buf.index(slot).store(acc.to(gl.bfloat16).to(c_ptr.type.element_ty))
        gl.barrier()
        nxt = (i + 2) * BLOCK_M
        gl.amd.gfx1250.tdm.async_load(a_desc, [nxt, 0], a_buf.index(slot))
        gl.amd.gfx1250.tdm.async_load(p_desc, [nxt + nh * HALF_M, 0], p_buf.index(slot))
        gl.amd.gfx1250.tdm.async_store(c_desc, [m0, n_off], c_buf.index(slot))
        gl.amd.gfx1250.tdm.async_store(q_desc, [m0 + nh * HALF_M, 0], q_buf.index(slot))
        _kv_put(kv_n, kv_p, ks_s, kv_cache_ptr, kv_cache_stride_b, L, R,
                BLOCK_SIZE, num_warps)  # fmt: skip

    # Tokens past n_local * NUM_CTAS (only when M > #CTAs * tiles per CTA).
    for k in range(n_local, gl.cdiv(M - pid, NUM_CTAS)):
        t = pid + k * NUM_CTAS
        ks, kpos = _kv_idx_load(
            t, M, pos_ptr, slot_mapping_ptr, pos_stride_b, R, num_warps
        )
        k_nope, k_pe, kc2, ks2 = _kv_data_load(
            t, M, kpos, k_nope_ptr, k_pe_ptr, cos_ptr, sin_ptr,
            k_nope_stride_b, k_pe_stride_b, cos_stride_b, L, R, num_warps,
        )  # fmt: skip
        kv_n, kv_p = _kv_values(
            k_nope, k_pe, kc2, ks2, k_scale_rcprl, kv_cache_ptr, R, IS_NEOX, num_warps
        )
        _kv_put(kv_n, kv_p, ks, kv_cache_ptr, kv_cache_stride_b, L, R,
                BLOCK_SIZE, num_warps)  # fmt: skip

    gl.amd.gfx1250.tdm.async_wait(0)


@functools.cache
def _layouts(block_m, block_n, block_k, num_warps):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    wmma, opa, opb = create_wmma_layouts(num_warps)
    sa, sb = create_shared_layouts(block_m, block_n, block_k, "TN", 16)
    return wmma, opa, opb, sa, sb


def kup_qpe_fused(
    q_nope: torch.Tensor,  # (H, M, P) bf16 view of q_proj output
    w_k: torch.Tensor,  # (H, L, P) bf16
    q_pe: torch.Tensor,  # (M, H, R) bf16 view of q_proj output
    k_nope: torch.Tensor,  # (M, 1, L) bf16
    k_pe: torch.Tensor,  # (M, 1, R) bf16
    kv_cache: torch.Tensor,  # (num_blocks, 1, block_size, L + R) fp8, shuffled
    slot_mapping: torch.Tensor,  # (M,)
    positions: torch.Tensor,  # (M,)
    cos: torch.Tensor,  # (max_pos, R / 2)
    sin: torch.Tensor,
    k_scale: torch.Tensor | None,
    is_neox: bool,
    q_out: torch.Tensor,  # (M, H, L + R) fp8, written fully
    m_split: int = 2,
    block_m: int = 64,
    block_n: int = 256,
    num_warps: int = 4,
    num_xcds: int = 8,
) -> torch.Tensor:
    # The shared estimate/budget includes W, double A/C/P/Q and frequency
    # workspace. 224 KiB is ATOM's conservative limit before layout padding;
    # gfx1250 has 320 KiB. Retain this budget pending GPU shape validation.
    # The gate restricts P to {32,64,128}, L to {256,512,1024} and R<=256;
    # do not relax those or the fixed launch geometry without compiler/GPU QA.
    assert (m_split, block_m, block_n, num_warps) == (2, 64, 256, 4)
    H, M, P = q_nope.shape
    L = w_k.shape[1]
    R = q_pe.shape[-1]
    assert M > 0 and L in (256, 512, 1024) and P in (32, 64, 128)
    assert R in (32, 64, 128, 256)
    assert _kup_qpe_lds_bytes(L, P, R) <= _KUP_QPE_LDS_BUDGET
    if cos.dim() == 4:
        cos = cos.squeeze(1).squeeze(1)
        sin = sin.squeeze(1).squeeze(1)
    assert q_nope.stride(2) == 1 and q_pe.stride(2) == 1
    assert q_out.shape == (M, H, L + R) and q_out.stride(2) == 1
    assert cos.shape[-1] == R // 2 and cos.stride(-1) == 1
    assert k_nope.stride(-1) == 1 and k_pe.stride(-1) == 1
    _, kh, block_size, d_cache = kv_cache.shape
    assert kh == 1 and d_cache == L + R and kv_cache.stride(3) == 1
    assert block_size % 16 == 0 and L % block_n == 0
    assert M * q_out.stride(0) < 2**31
    if (H * m_split) % num_xcds:
        num_xcds = 1
    wt = w_k.transpose(1, 2)
    wmma, opa, opb, sa, sb = _layouts(block_m, block_n, P, num_warps)
    grid = (H * (L // block_n) * m_split,)
    _kup_qpe_persistent_kernel[grid](
        q_nope, wt, q_out, q_pe, k_nope, k_pe, positions, cos, sin, kv_cache,
        slot_mapping, k_scale, M,
        q_nope.stride(0), q_nope.stride(1),
        wt.stride(0), wt.stride(2),
        q_out.stride(0), q_out.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        k_nope.stride(0), k_pe.stride(0), positions.stride(0),
        cos.stride(0), kv_cache.stride(0),
        H=H, L=L, P=P, R=R, M_SPLIT=m_split, BLOCK_M=block_m, BLOCK_N=block_n,
        SHARED_LAYOUT_A=sa, SHARED_LAYOUT_B=sb, WMMA_LAYOUT=wmma,
        OPERAND_LAYOUT_A=opa, OPERAND_LAYOUT_B=opb,
        BLOCK_SIZE=block_size, IS_NEOX=is_neox, HAVE_K_SCALE=k_scale is not None,
        NUM_XCDS=num_xcds, num_warps=num_warps,
    )  # fmt: skip
    return q_out
