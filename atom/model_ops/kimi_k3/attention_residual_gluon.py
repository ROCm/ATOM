# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Gluon attn-res kernel: block_residual rows streamed through LDS by TDM.

Same contract as ``_attn_res_fused_kernel`` in ``attention_residual.py`` (B
block_residual rows plus the prefix as candidates, DO_ADD/DO_ADD2 folded into the
prefix, optional output rmsnorm and per-token FP8 quant). Two differences:

* The candidate rows go HBM -> LDS with TENSOR_LOAD_TO_LDS, ``_PREFETCH`` rows
  ahead of the one being reduced, in ``_PREFETCH + 1`` LDS buffers of one
  ``[1, BD]`` BF16 row each (48 KiB at H=7168). The Triton kernel loads each
  tile at the top of its iteration and waits on it; at four waves per SIMD that
  exposed load wait was most of a program's life. Holding prefetched rows in
  VGPRs instead would cost occupancy (the kernel needs <= 256 VGPRs for four
  waves); LDS has room.
* The prefix seeds the online softmax instead of entering it as the last
  candidate, so the loop runs over block_residual rows only. The mix is the
  same softmax; the FP32 accumulation order differs from the Triton kernel.

Every T uses this kernel with one fixed num_warps, so a row's result never
depends on its batch peers (see ``_ATTN_RES_BF16_CONFIG``).
"""

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_NUM_WARPS = 4
# Rows in flight ahead of the one being reduced. 1 leaves the TDM wait exposed
# (one row of math does not cover a row's latency); 3 measured no better than 2.
_PREFETCH = 2


@gluon.jit
def _attn_res_tdm_kernel(
    br_ptr,
    ps_ptr,
    sw_ptr,
    y_ptr,
    ys_ptr,
    hs_ptr,
    hs2_ptr,
    pref_ptr,
    ow_ptr,
    B,
    H: gl.constexpr,
    eps,
    out_eps,
    fp8_max,
    inv_fp8_max,
    stride_br_t,
    stride_br_b,
    stride_ps_t,
    stride_yt,
    stride_hs_t,
    stride_hs2_t,
    stride_pref_t,
    BD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFETCH: gl.constexpr,
    DO_ADD: gl.constexpr,
    DO_ADD2: gl.constexpr,
    WRITE_PREF: gl.constexpr,
    OUT_NORM: gl.constexpr,
    QUANT: gl.constexpr,
):
    gl.static_assert(PREFETCH >= 1 and PREFETCH <= 3)
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    t = gl.program_id(0).to(gl.int64)
    o_d = gl.arange(0, BD, layout=layout)
    m_d = o_d < H
    sw = gl.amd.cdna5.buffer_load(sw_ptr, o_d, mask=m_d, other=0.0).to(gl.float32)

    # One [1, BD] tile per block_residual row; columns H..BD read as zero.
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=br_ptr + t * stride_br_t,
        shape=(B, H),
        strides=(stride_br_b, 1),
        block_shape=(1, BD),
        layout=shared,
    )
    rows = gl.allocate_shared_memory(
        br_ptr.dtype.element_ty, [PREFETCH + 1, 1, BD], shared
    )
    # Issued before the prefix loads so their latency overlaps the prefix work.
    for j in gl.static_range(PREFETCH):
        gl.amd.cdna5.tdm.async_load(desc, [j, 0], rows.index(j), pred=j < B)

    ps = gl.amd.cdna5.buffer_load(
        ps_ptr + t * stride_ps_t, o_d, mask=m_d, other=0.0
    ).to(gl.float32)
    if DO_ADD:
        ps += gl.amd.cdna5.buffer_load(
            hs_ptr + t * stride_hs_t, o_d, mask=m_d, other=0.0
        ).to(gl.float32)
    if DO_ADD2:
        ps += gl.amd.cdna5.buffer_load(
            hs2_ptr + t * stride_hs2_t, o_d, mask=m_d, other=0.0
        ).to(gl.float32)
    if WRITE_PREF:
        gl.amd.cdna5.buffer_store(
            ps.to(pref_ptr.dtype.element_ty),
            pref_ptr + t * stride_pref_t,
            o_d,
            mask=m_d,
        )

    # The prefix is the first softmax term, so the running output starts as it.
    rstd = gl.rsqrt(gl.sum(ps * ps, axis=0) * (1.0 / H) + eps)
    b_m = gl.sum(ps * sw, axis=0) * rstd
    b_acc = gl.full([], 1.0, dtype=gl.float32)
    b_o = ps
    for i_l in range(B):
        # Row i_l has landed once at most min(PREFETCH - 1, rows after it)
        # transfers are still outstanding. Each wave counts only its own slice.
        if PREFETCH == 1:
            gl.amd.cdna5.tdm.async_wait(0)
        elif PREFETCH == 2:
            if i_l + 1 < B:
                gl.amd.cdna5.tdm.async_wait(1)
            else:
                gl.amd.cdna5.tdm.async_wait(0)
        else:
            if i_l + 2 < B:
                gl.amd.cdna5.tdm.async_wait(2)
            elif i_l + 1 < B:
                gl.amd.cdna5.tdm.async_wait(1)
            else:
                gl.amd.cdna5.tdm.async_wait(0)
        # Every wave issues its slice of each transfer (TDM ignores EXEC, so a
        # single-lane producer is not an option). After this barrier all slices
        # of row i_l are in LDS, and every wave is done reading row i_l - 1,
        # whose buffer the next transfer refills.
        gl.barrier()
        gl.amd.cdna5.tdm.async_load(
            desc,
            [i_l + PREFETCH, 0],
            rows.index((i_l + PREFETCH) % (PREFETCH + 1)),
            pred=i_l + PREFETCH < B,
        )
        v = rows.index(i_l % (PREFETCH + 1)).reshape([BD]).load(layout).to(gl.float32)
        rstd = gl.rsqrt(gl.sum(v * v, axis=0) * (1.0 / H) + eps)
        s = gl.sum(v * sw, axis=0) * rstd
        b_mp = b_m
        b_m = gl.maximum(b_m, s)
        r = gl.exp(b_mp - b_m)
        p = gl.exp(s - b_m)
        b_acc = b_acc * r + p
        b_o = b_o * r + p * v

    b_o = b_o * (1.0 / b_acc)
    if OUT_NORM:
        rs = gl.rsqrt(
            gl.sum(gl.where(m_d, b_o * b_o, 0.0), axis=0) * (1.0 / H) + out_eps
        )
        b_o = (
            b_o
            * rs
            * gl.amd.cdna5.buffer_load(ow_ptr, o_d, mask=m_d, other=0.0).to(gl.float32)
        )
    if QUANT:
        # Same scale derivation as the Triton kernel (amax * (1/fp8_max)).
        amax = gl.max(gl.where(m_d, gl.abs(b_o), 0.0), axis=0)
        scale = amax * inv_fp8_max
        inv = gl.where(scale > 0.0, 1.0 / scale, 0.0)
        b_o = gl.minimum(gl.maximum(b_o * inv, -fp8_max), fp8_max)
        gl.store(ys_ptr + t, scale)
    gl.amd.cdna5.buffer_store(
        b_o.to(y_ptr.dtype.element_ty), y_ptr + t * stride_yt, o_d, mask=m_d
    )


def attn_res_tdm_supported(block_residual: torch.Tensor) -> bool:
    """Shapes the kernel handles: a BF16 candidate row of at most 8192 elements
    (one LDS buffer per row; BD must cover the 4-wave layout's 1024 columns) and a
    row stride TDM can address (16-byte aligned)."""
    _, B, H = block_residual.shape
    return (
        block_residual.dtype == torch.bfloat16
        and B >= 1
        and 1 <= H <= 8192
        and block_residual.stride(2) == 1
        and block_residual.stride(1) % 8 == 0
        and block_residual.data_ptr() % 16 == 0
    )


def launch_attn_res_tdm(
    br, ps, sw, y, y_scale, hs, hs2, pref, ow, eps, out_eps, fp8_max, flags
):
    T, B, H = br.shape
    _attn_res_tdm_kernel[(T,)](
        br,
        ps,
        sw,
        y,
        y_scale,
        hs,
        hs2,
        pref,
        ow,
        B,
        H,
        float(eps),
        float(out_eps),
        fp8_max,
        1.0 / fp8_max,
        br.stride(0),
        br.stride(1),
        ps.stride(0),
        y.stride(0),
        hs.stride(0),
        hs2.stride(0),
        pref.stride(0),
        max(1024, 1 << (H - 1).bit_length()),
        _NUM_WARPS,
        _PREFETCH,
        num_warps=_NUM_WARPS,
        **flags,
    )
