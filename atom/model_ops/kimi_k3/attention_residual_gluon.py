# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 Gluon attn-res kernels: block_residual rows streamed through LDS by TDM.

Same contract as ``_attn_res_fused_kernel`` in ``attention_residual.py`` (B
block_residual rows plus the prefix as candidates, DO_ADD/DO_ADD2 folded into the
prefix, optional output rmsnorm and per-token FP8 quant). Differences:

* The candidate rows go HBM -> LDS with TENSOR_LOAD_TO_LDS ahead of the row
  being reduced. The Triton kernel loads each tile at the top of its iteration
  and waits on it; at four waves per SIMD that exposed load wait was most of a
  program's life. Holding prefetched rows in VGPRs instead would cost occupancy;
  LDS has room.
* The prefix seeds the online softmax instead of entering it as the last
  candidate, so the loop runs over block_residual rows only. The mix is the
  same softmax; the FP32 accumulation order differs from the Triton kernel.

Two kernels share one per-token arithmetic (``_chunks_*``, ``_seed``, ``_step``,
``_finish``; same layout, same num_warps, same order), so a row's result never
depends on which of them ran or on its batch peers (see
``_ATTN_RES_BF16_CONFIG``):

* ``_attn_res_chunk_kernel``: a program a token; the prefix and addends by
  buffer_load, all block rows in flight at once through a TDM ring. Small T.
* ``_attn_res_stream_kernel``: a persistent grid (as many programs as the CUs
  hold); every tile of a token -- prefix, addends, block rows, out_norm weight
  -- goes through one TDM ring that runs across token boundaries, so the next
  token's rows land while this one is reduced and its epilogue runs. Each XCD
  takes a contiguous run of tokens, so a column-major E8M0 scale line is
  written from one L2. Large T.

The row math works on ``H // 1024`` chunk tensors of 1024 columns (8 contiguous
elements a lane): no work on padding columns (BD = 8192 for H = 7168 wasted an
eighth), the two per-row sums as FMA chains over the chunks reduced in one
cross-wave exchange, and the running output updated as ``fma(p, v, b_o * r)``
(``b_o * r`` skipped when the running max did not move: r is then exactly 1).
Rows whose H is not a multiple of 1024 (or above 8192) keep the single-tile
``_attn_res_tdm_kernel``.
"""

from __future__ import annotations

import ctypes
import functools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import PropagateNan

_PROPAGATE_NAN = gl.constexpr(PropagateNan.ALL)  # keep NaN through the FP8 clamp
_NUM_WARPS = 4
# Rows in flight ahead of the one being reduced. 1 leaves the TDM wait exposed
# (one row of math does not cover a row's latency); 3 measured no better than 2.
_PREFETCH = 2
# chunk / stream kernels
_STREAM_MIN_T = 4096  # from this many tokens always the persistent kernel
_SPECIALIZE_MAX_T = (
    512  # chunk kernel: B == 1 specialized up to this T (<= 2 programs a CU)
)
_ONE_LANE_MAX_T = 128  # chunk kernel: E8M0 codes stored from one lane up to this T
_STREAM_NBUF = 4  # its ring slots (one [1, BD] BF16 row each: 16 KiB at H = 7168)
_STREAM_PPC = 4  # its programs per CU (lowered to what the compiled kernel fits)


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
    E8M0_SCALE: gl.constexpr = False,
    FP8_MAX: gl.constexpr = 0.0,
    stride_ys_t=0,
    stride_ys_g=0,
    NUM_XCDS: gl.constexpr = 1,
):
    gl.static_assert(PREFETCH >= 1 and PREFETCH <= 3)
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    t = gl.program_id(0)
    if NUM_XCDS > 1:
        # Programs go to the XCDs round-robin. Give XCD x the x-th contiguous run of
        # tokens instead, so a column-major scale line, which holds neighbouring
        # tokens, is written from one XCD's L2 rather than from all of them.
        G = gl.num_programs(0)
        per = (G + NUM_XCDS - 1) // NUM_XCDS
        tall = G % NUM_XCDS
        tall += (tall == 0).to(gl.int32) * NUM_XCDS
        xcd = t % NUM_XCDS
        t = xcd * per + t // NUM_XCDS - (xcd - tall) * (xcd > tall).to(gl.int32)
    t = t.to(gl.int64)
    o_d = gl.arange(0, BD, layout=layout)
    m_d = o_d < H
    sw = gl.amd.cdna5.buffer_load(sw_ptr, o_d, mask=m_d, other=0.0).to(gl.float32)
    # The prefix and addend loads go out before the TDM transfers and are only
    # consumed after them, so all of the token's independent loads are in flight
    # together.
    ps_raw = gl.amd.cdna5.buffer_load(
        ps_ptr + t * stride_ps_t, o_d, mask=m_d, other=0.0
    )
    if DO_ADD:
        hs_raw = gl.amd.cdna5.buffer_load(
            hs_ptr + t * stride_hs_t, o_d, mask=m_d, other=0.0
        )
    if DO_ADD2:
        hs2_raw = gl.amd.cdna5.buffer_load(
            hs2_ptr + t * stride_hs2_t, o_d, mask=m_d, other=0.0
        )

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
    # Their latency overlaps the prefix work.
    for j in gl.static_range(PREFETCH):
        gl.amd.cdna5.tdm.async_load(desc, [j, 0], rows.index(j), pred=j < B)

    ps = ps_raw.to(gl.float32)
    if DO_ADD:
        ps += hs_raw.to(gl.float32)
    if DO_ADD2:
        ps += hs2_raw.to(gl.float32)
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
        # single-lane producer is not an option). The compiler's membar pass puts
        # an s_barrier right after this wait, ahead of the refill and the reads
        # of row i_l below. After it all slices of row i_l are in LDS, and every
        # wave is done reading row i_l - 1, whose buffer the refill overwrites.
        # An explicit gl.barrier() here would add a second barrier per row and,
        # being a full fence, an s_wait_storecnt that drains the prefix_out store.
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
    if E8M0_SCALE:
        # triton_e8m0_quant.fp8_e8m0_group_scale (Gluon cannot call it) on the bf16
        # the unfused path stores
        q = gl.reshape(b_o.to(ps_ptr.dtype.element_ty).to(gl.float32), [BD // 128, 128])
        amax = gl.minimum(
            gl.maximum(gl.max(gl.abs(q), axis=1), 1e-10), 448.0 * 2.0**119
        )
        bits = (amax * (1.0 / FP8_MAX)).to(gl.uint32, bitcast=True)
        code = (bits >> 23) & 0xFF
        code += ((code < 0xFF) & ((bits & 0x7FFFFF) != 0)).to(gl.uint32)
        q = q * gl.expand_dims(((254 - code) << 23).to(gl.float32, bitcast=True), 1)
        q = gl.maximum(q, -FP8_MAX, propagate_nan=_PROPAGATE_NAN)
        q = gl.minimum(q, FP8_MAX, propagate_nan=_PROPAGATE_NAN)
        b_o = gl.convert_layout(gl.reshape(q, [BD]), layout)
        # Store each scale byte from one lane. In the reduction's slice layout a
        # code is replicated over the 16 lanes that reduced its group, and the
        # store goes out as eight masked byte stores per wave; from a one-per-lane
        # layout it is two store instructions per program. Column-major scales put
        # every byte of a token on a different line, so each of those stores is
        # its own request, all at the program's tail (MI455, T=30 B=1: 4.45 -> 4.0 us).
        one: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
        g = gl.arange(0, BD // 128, layout=one)
        gl.store(
            ys_ptr + t * stride_ys_t + g * stride_ys_g,
            gl.convert_layout(code, one).to(gl.uint8),
            mask=g < H // 128,
        )
    gl.amd.cdna5.buffer_store(
        b_o.to(y_ptr.dtype.element_ty), y_ptr + t * stride_yt, o_d, mask=m_d
    )


# ---------------------------------------------------------------------------
# Chunked row math shared by the chunk and stream kernels. A row is a tuple of
# NCH = H // 1024 tensors of [1024] (8 contiguous columns a lane, the same
# column-to-lane map as the BD layout above, without its padding chunk).


@gluon.jit
def _chunks_global(ptr, o, NCH: gl.constexpr):
    """The row at ``ptr`` as NCH chunks, in its own dtype."""
    x = (gl.amd.cdna5.buffer_load(ptr, o),)
    for c in gl.static_range(1, NCH):
        x = x + (gl.amd.cdna5.buffer_load(ptr, o + c * 1024),)
    return x


@gluon.jit
def _chunks_lds(row, layout: gl.constexpr, NCH: gl.constexpr, BD: gl.constexpr):
    """The [.., BD] LDS row ``row`` as NCH FP32 chunks."""
    v = row.reshape([BD])
    x = (v.slice(0, 1024).load(layout).to(gl.float32),)
    for c in gl.static_range(1, NCH):
        x = x + (v.slice(c * 1024, 1024).load(layout).to(gl.float32),)
    return x


@gluon.jit
def _chunks_f32(x, NCH: gl.constexpr):
    y = (x[0].to(gl.float32),)
    for c in gl.static_range(1, NCH):
        y = y + (x[c].to(gl.float32),)
    return y


@gluon.jit
def _chunks_add(x, y, NCH: gl.constexpr):
    z = (x[0] + y[0],)
    for c in gl.static_range(1, NCH):
        z = z + (x[c] + y[c],)
    return z


@gluon.jit
def _row_sums(x, sw, NCH: gl.constexpr):
    """(sum x * x, sum x * sw) over the row: FMA chains over the chunks, then
    both reduced across the program in one exchange."""
    sq = x[0] * x[0]
    dt = x[0] * sw[0]
    for c in gl.static_range(1, NCH):
        sq = gl.fma(x[c], x[c], sq)
        dt = gl.fma(x[c], sw[c], dt)
    return gl.split(gl.sum(gl.join(sq, dt), axis=0))


@gluon.jit
def _seed(ps, sw, H: gl.constexpr, eps, NCH: gl.constexpr):
    """The prefix's score: it seeds the online softmax."""
    sq, dt = _row_sums(ps, sw, NCH)
    return dt * gl.rsqrt(sq * (1.0 / H) + eps)


@gluon.jit
def _step(b_m, b_acc, b_o, v, sw, H: gl.constexpr, eps, NCH: gl.constexpr):
    """Fold block row ``v`` into the online softmax."""
    sq, dt = _row_sums(v, sw, NCH)
    s = dt * gl.rsqrt(sq * (1.0 / H) + eps)
    b_mp = b_m
    b_m = gl.maximum(b_m, s)
    r = gl.exp(b_mp - b_m)
    p = gl.exp(s - b_m)
    b_acc = b_acc * r + p
    if b_mp == b_m:
        # r is exactly 1 and b_o * 1 is b_o: skip the rescale
        z = (gl.fma(v[0], p, b_o[0]),)
        for c in gl.static_range(1, NCH):
            z = z + (gl.fma(v[c], p, b_o[c]),)
    else:
        z = (gl.fma(v[0], p, b_o[0] * r),)
        for c in gl.static_range(1, NCH):
            z = z + (gl.fma(v[c], p, b_o[c] * r),)
    return b_m, b_acc, z


@gluon.jit
def _finish(
    b_o,
    b_acc,
    ow,
    t,
    act_ptr,
    y_ptr,
    ys_ptr,
    o,
    layout: gl.constexpr,
    H: gl.constexpr,
    out_eps,
    fp8_max,
    inv_fp8_max,
    stride_yt,
    stride_ys_t,
    stride_ys_g,
    OUT_NORM: gl.constexpr,
    QUANT: gl.constexpr,
    E8M0_SCALE: gl.constexpr,
    FP8_MAX: gl.constexpr,
    NCH: gl.constexpr,
    ONE_LANE: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
):
    """Normalize the mix, fold out_norm (``ow``: FP32 chunks) and the quant, store."""
    inv_acc = 1.0 / b_acc
    z = (b_o[0] * inv_acc,)
    for c in gl.static_range(1, NCH):
        z = z + (b_o[c] * inv_acc,)
    b_o = z
    if OUT_NORM:
        acc = b_o[0] * b_o[0]
        for c in gl.static_range(1, NCH):
            acc = gl.fma(b_o[c], b_o[c], acc)
        rs = gl.rsqrt(gl.sum(acc, axis=0) * (1.0 / H) + out_eps)
        z = (b_o[0] * rs * ow[0],)
        for c in gl.static_range(1, NCH):
            z = z + (b_o[c] * rs * ow[c],)
        b_o = z
    if QUANT:
        # Same scale derivation as the Triton kernel (amax * (1/fp8_max)).
        am = gl.abs(b_o[0])
        for c in gl.static_range(1, NCH):
            am = gl.maximum(am, gl.abs(b_o[c]))
        scale = gl.max(am, axis=0) * inv_fp8_max
        inv = gl.where(scale > 0.0, 1.0 / scale, 0.0)
        z = (gl.minimum(gl.maximum(b_o[0] * inv, -fp8_max), fp8_max),)
        for c in gl.static_range(1, NCH):
            z = z + (gl.minimum(gl.maximum(b_o[c] * inv, -fp8_max), fp8_max),)
        b_o = z
        gl.store(ys_ptr + t, scale)
    if E8M0_SCALE:
        # triton_e8m0_quant.fp8_e8m0_group_scale on the bf16 the unfused path
        # stores; a chunk is eight 128-column groups, each within 16 lanes
        z = ()
        codes = ()
        for c in gl.static_range(NCH):
            q = gl.reshape(b_o[c].to(act_ptr.dtype.element_ty).to(gl.float32), [8, 128])
            amax = gl.minimum(
                gl.maximum(gl.max(gl.abs(q), axis=1), 1e-10), 448.0 * 2.0**119
            )
            bits = (amax * (1.0 / FP8_MAX)).to(gl.uint32, bitcast=True)
            code = (bits >> 23) & 0xFF
            code += ((code < 0xFF) & ((bits & 0x7FFFFF) != 0)).to(gl.uint32)
            q = q * gl.expand_dims(((254 - code) << 23).to(gl.float32, bitcast=True), 1)
            q = gl.maximum(q, -FP8_MAX, propagate_nan=_PROPAGATE_NAN)
            q = gl.minimum(q, FP8_MAX, propagate_nan=_PROPAGATE_NAN)
            z = z + (gl.convert_layout(gl.reshape(q, [1024]), layout),)
            if ONE_LANE:
                codes = codes + (code,)
            else:
                # straight from the reduction's layout: the store keeps one lane per
                # code, and a one-lane layout would cost an LDS round trip per chunk
                g = gl.arange(0, 8, layout=code.type.layout)
                gl.store(
                    ys_ptr + t * stride_ys_t + (c * 8 + g) * stride_ys_g,
                    code.to(gl.uint8),
                )
        if ONE_LANE:
            # all codes through one layout conversion, each stored from one lane
            zero = gl.zeros_like(codes[0])
            for c in gl.static_range(NCH, 8):
                codes = codes + (zero,)
            lo = gl.join(gl.join(codes[0], codes[1]), gl.join(codes[2], codes[3]))
            hi = gl.join(gl.join(codes[4], codes[5]), gl.join(codes[6], codes[7]))
            allc = gl.reshape(gl.permute(gl.join(lo, hi), [3, 2, 1, 0]), [64])
            one: gl.constexpr = gl.BlockedLayout([1], [32], [NUM_WARPS], [0])
            g = gl.arange(0, 64, layout=one)
            gl.store(
                ys_ptr + t * stride_ys_t + g * stride_ys_g,
                gl.convert_layout(allc, one).to(gl.uint8),
                mask=g < NCH * 8,
            )
        b_o = z
    for c in gl.static_range(NCH):
        gl.amd.cdna5.buffer_store(
            b_o[c].to(y_ptr.dtype.element_ty), y_ptr + t * stride_yt, o + c * 1024
        )


def _attn_res_chunk_fn(
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
    E8M0_SCALE: gl.constexpr = False,
    FP8_MAX: gl.constexpr = 0.0,
    stride_ys_t=0,
    stride_ys_g=0,
    NUM_XCDS: gl.constexpr = 1,
    CODES_ONE_LANE: gl.constexpr = False,
):
    """A program a token: the prefix, addends, out_norm weight and up to
    PREFETCH block rows all in flight at once (the launcher picks PREFETCH as
    deep as the LDS share of the programs a CU gets allows)."""
    NCH: gl.constexpr = H // 1024
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    t = gl.program_id(0)
    if NUM_XCDS > 1:
        # see _attn_res_tdm_kernel
        G = gl.num_programs(0)
        per = (G + NUM_XCDS - 1) // NUM_XCDS
        tall = G % NUM_XCDS
        tall += (tall == 0).to(gl.int32) * NUM_XCDS
        xcd = t % NUM_XCDS
        t = xcd * per + t // NUM_XCDS - (xcd - tall) * (xcd > tall).to(gl.int32)
    t = t.to(gl.int64)
    o = gl.arange(0, 1024, layout=layout)
    sw = _chunks_f32(_chunks_global(sw_ptr, o, NCH), NCH)
    if OUT_NORM:
        ow_raw = _chunks_global(ow_ptr, o, NCH)
    ps_raw = _chunks_global(ps_ptr + t * stride_ps_t, o, NCH)
    if DO_ADD:
        hs_raw = _chunks_global(hs_ptr + t * stride_hs_t, o, NCH)
    if DO_ADD2:
        hs2_raw = _chunks_global(hs2_ptr + t * stride_hs2_t, o, NCH)

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
    for j in gl.static_range(PREFETCH):
        gl.amd.cdna5.tdm.async_load(desc, [j, 0], rows.index(j), pred=j < B)

    ps = _chunks_f32(ps_raw, NCH)
    if DO_ADD:
        ps = _chunks_add(ps, _chunks_f32(hs_raw, NCH), NCH)
    if DO_ADD2:
        ps = _chunks_add(ps, _chunks_f32(hs2_raw, NCH), NCH)
    if WRITE_PREF:
        for c in gl.static_range(NCH):
            gl.amd.cdna5.buffer_store(
                ps[c].to(pref_ptr.dtype.element_ty),
                pref_ptr + t * stride_pref_t,
                o + c * 1024,
            )
    b_m = _seed(ps, sw, H, eps, NCH)
    b_acc = gl.full([], 1.0, dtype=gl.float32)
    b_o = ps
    for i_l in range(B):
        # Every step issues one transfer (predicated off past B). Predicated-off
        # TDM ops still count in TENSORcnt and retire in order, so PREFETCH - 1
        # younger ops outstanding always means row i_l has landed. The membar
        # pass's s_barrier after the wait makes all waves' slices visible and
        # frees row i_l - 1's buffer for the refill.
        gl.amd.cdna5.tdm.async_wait(PREFETCH - 1)
        gl.amd.cdna5.tdm.async_load(
            desc,
            [i_l + PREFETCH, 0],
            rows.index((i_l + PREFETCH) % (PREFETCH + 1)),
            pred=i_l + PREFETCH < B,
        )
        v = _chunks_lds(rows.index(i_l % (PREFETCH + 1)), layout, NCH, BD)
        b_m, b_acc, b_o = _step(b_m, b_acc, b_o, v, sw, H, eps, NCH)
    if OUT_NORM:
        ow = _chunks_f32(ow_raw, NCH)
    else:
        ow = b_o
    _finish(
        b_o,
        b_acc,
        ow,
        t,
        ps_ptr,
        y_ptr,
        ys_ptr,
        o,
        layout,
        H,
        out_eps,
        fp8_max,
        inv_fp8_max,
        stride_yt,
        stride_ys_t,
        stride_ys_g,
        OUT_NORM,
        QUANT,
        E8M0_SCALE,
        FP8_MAX,
        NCH,
        CODES_ONE_LANE,
        NUM_WARPS,
    )


# B == 1 specialized for decode T (a shorter critical path); from a few hundred
# tokens a runtime B, whose kernel needs fewer VGPRs and keeps four programs a CU.
_attn_res_chunk_kernel = gluon.jit(_attn_res_chunk_fn)
_attn_res_chunk_kernel_rt = gluon.jit(do_not_specialize=["B"])(_attn_res_chunk_fn)


@gluon.jit
def _issue_tile(
    q,
    tfirst,
    GP,
    tend,
    N,
    B,
    dst,
    ps_desc,
    hs_desc,
    hs2_desc,
    br_desc,
    ow_desc,
    NP: gl.constexpr,
    DO_ADD2: gl.constexpr,
):
    """Issue the program's q-th tile. Its tokens run tfirst, tfirst + GP, ..
    (< tend), each N tiles: [prefix, add?, add2?, block rows, out_norm weight?].
    Past the last token the op is predicated off (it still counts)."""
    kk = q // N
    jj = q - kk * N
    tok = tfirst + kk * GP
    pred = tok < tend
    if jj < NP:
        if jj == 0:
            gl.amd.cdna5.tdm.async_load(ps_desc, [tok, 0, 0], dst, pred=pred)
        elif DO_ADD2:
            if jj == 1:
                gl.amd.cdna5.tdm.async_load(hs_desc, [tok, 0, 0], dst, pred=pred)
            else:
                gl.amd.cdna5.tdm.async_load(hs2_desc, [tok, 0, 0], dst, pred=pred)
        else:
            gl.amd.cdna5.tdm.async_load(hs_desc, [tok, 0, 0], dst, pred=pred)
    elif jj < NP + B:
        gl.amd.cdna5.tdm.async_load(br_desc, [tok, jj - NP, 0], dst, pred=pred)
    else:
        gl.amd.cdna5.tdm.async_load(ow_desc, [0, 0, 0], dst, pred=pred)


@gluon.jit(do_not_specialize=["B"])
def _attn_res_stream_kernel(
    br_ptr,
    ps_ptr,
    sw_ptr,
    y_ptr,
    ys_ptr,
    hs_ptr,
    hs2_ptr,
    pref_ptr,
    ow_ptr,
    T,
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
    D: gl.constexpr,
    NX: gl.constexpr,
    NP: gl.constexpr,
    NW: gl.constexpr,
    DO_ADD: gl.constexpr,
    DO_ADD2: gl.constexpr,
    WRITE_PREF: gl.constexpr,
    OUT_NORM: gl.constexpr,
    QUANT: gl.constexpr,
    E8M0_SCALE: gl.constexpr = False,
    FP8_MAX: gl.constexpr = 0.0,
    stride_ys_t=0,
    stride_ys_g=0,
):
    """A persistent program walks its tokens' tiles through one ring of D + 1
    LDS rows with D transfers in flight, across token boundaries."""
    NCH: gl.constexpr = H // 1024
    NB: gl.constexpr = D + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [NUM_WARPS], [0])
    pid = gl.program_id(0)
    G = gl.num_programs(0)
    N = B + (
        NP + NW
    )  # tiles a token: NP prefix/addend rows, B block rows, NW out_norm weight rows
    # XCD x (pid % NX under round-robin dispatch) owns the x-th contiguous run of
    # TP tokens; its GP programs take them in turn.
    xcd = pid % NX
    GP = G // NX
    TP = (T + NX - 1) // NX
    t0 = xcd * TP
    TPx = gl.minimum(gl.maximum(T - t0, 0), TP)
    tend = t0 + TPx
    tfirst = t0 + pid // NX
    K = gl.maximum(tend - tfirst + GP - 1, 0) // GP

    o = gl.arange(0, 1024, layout=layout)
    sw = _chunks_f32(_chunks_global(sw_ptr, o, NCH), NCH)

    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    ps_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=ps_ptr,
        shape=(T, 1, H),
        strides=(stride_ps_t, H, 1),
        block_shape=(1, 1, BD),
        layout=shared,
    )
    hs_desc = ps_desc
    hs2_desc = ps_desc
    ow_desc = ps_desc
    if DO_ADD:
        hs_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=hs_ptr,
            shape=(T, 1, H),
            strides=(stride_hs_t, H, 1),
            block_shape=(1, 1, BD),
            layout=shared,
        )
    if DO_ADD2:
        hs2_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=hs2_ptr,
            shape=(T, 1, H),
            strides=(stride_hs2_t, H, 1),
            block_shape=(1, 1, BD),
            layout=shared,
        )
    if OUT_NORM:
        ow_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=ow_ptr,
            shape=(1, 1, H),
            strides=(H, H, 1),
            block_shape=(1, 1, BD),
            layout=shared,
        )
    br_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=br_ptr,
        shape=(T, B, H),
        strides=(stride_br_t, stride_br_b, 1),
        block_shape=(1, 1, BD),
        layout=shared,
    )
    rows = gl.allocate_shared_memory(ps_ptr.dtype.element_ty, [NB, 1, 1, BD], shared)

    for q0 in gl.static_range(D):
        _issue_tile(
            q0,
            tfirst,
            GP,
            tend,
            N,
            B,
            rows.index(q0),
            ps_desc,
            hs_desc,
            hs2_desc,
            br_desc,
            ow_desc,
            NP,
            DO_ADD2,
        )

    for k in range(K):
        t = tfirst + k * GP
        t64 = t.to(gl.int64)
        base = k * N
        # Each step: tile q has landed once D - 1 younger ops are outstanding
        # (one op is issued every step, predicated off past the last token: those
        # count too, in order); the membar s_barrier after the wait makes every
        # wave's slice visible and frees tile q - 1's slot, which tile q + D takes.
        for j in gl.static_range(NP):
            gl.amd.cdna5.tdm.async_wait(D - 1)
            _issue_tile(
                base + j + D,
                tfirst,
                GP,
                tend,
                N,
                B,
                rows.index((base + j + D) % NB),
                ps_desc,
                hs_desc,
                hs2_desc,
                br_desc,
                ow_desc,
                NP,
                DO_ADD2,
            )
            xv = _chunks_lds(rows.index((base + j) % NB), layout, NCH, BD)
            if j == 0:
                ps = xv
            else:
                ps = _chunks_add(ps, xv, NCH)
        if WRITE_PREF:
            for c in gl.static_range(NCH):
                gl.amd.cdna5.buffer_store(
                    ps[c].to(pref_ptr.dtype.element_ty),
                    pref_ptr + t64 * stride_pref_t,
                    o + c * 1024,
                )
        b_m = _seed(ps, sw, H, eps, NCH)
        b_acc = gl.full([], 1.0, dtype=gl.float32)
        b_o = ps
        for i in range(B):
            q = base + NP + i
            gl.amd.cdna5.tdm.async_wait(D - 1)
            _issue_tile(
                q + D,
                tfirst,
                GP,
                tend,
                N,
                B,
                rows.index((q + D) % NB),
                ps_desc,
                hs_desc,
                hs2_desc,
                br_desc,
                ow_desc,
                NP,
                DO_ADD2,
            )
            v = _chunks_lds(rows.index(q % NB), layout, NCH, BD)
            b_m, b_acc, b_o = _step(b_m, b_acc, b_o, v, sw, H, eps, NCH)
        if OUT_NORM:
            q = base + NP + B
            gl.amd.cdna5.tdm.async_wait(D - 1)
            _issue_tile(
                q + D,
                tfirst,
                GP,
                tend,
                N,
                B,
                rows.index((q + D) % NB),
                ps_desc,
                hs_desc,
                hs2_desc,
                br_desc,
                ow_desc,
                NP,
                DO_ADD2,
            )
            ow = _chunks_lds(rows.index(q % NB), layout, NCH, BD)
        else:
            ow = b_o
        _finish(
            b_o,
            b_acc,
            ow,
            t64,
            ps_ptr,
            y_ptr,
            ys_ptr,
            o,
            layout,
            H,
            out_eps,
            fp8_max,
            inv_fp8_max,
            stride_yt,
            stride_ys_t,
            stride_ys_g,
            OUT_NORM,
            QUANT,
            E8M0_SCALE,
            FP8_MAX,
            NCH,
        )
    gl.amd.cdna5.tdm.async_wait(0)


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


@functools.cache
def _num_xcds() -> int:
    from aiter.ops.triton.utils.device_info import get_num_xcds

    return get_num_xcds()


@functools.cache
def _num_cus(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _hip():
    return ctypes.CDLL("libamdhip64.so")


# compiled stream kernel -> programs a CU holds (registers / LDS)
_STREAM_OCCUPANCY: dict = {}
_CONSTEXPR_FLAGS = (
    "DO_ADD",
    "DO_ADD2",
    "WRITE_PREF",
    "OUT_NORM",
    "QUANT",
    "E8M0_SCALE",
    "FP8_MAX",
)


def _occupancy(kernel) -> int | None:
    """Programs of ``kernel`` (a compiled Triton kernel) one CU holds at once."""
    try:
        kernel._init_handles()
        n = ctypes.c_int(0)
        err = _hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(n),
            ctypes.c_void_p(kernel.function),
            ctypes.c_int(_NUM_WARPS * 32),
            ctypes.c_size_t(kernel.metadata.shared),
        )
    except Exception:  # noqa: BLE001
        return None
    return n.value if err == 0 and n.value > 0 else None


def _launch_tdm(
    br, ps, sw, y, y_scale, hs, hs2, pref, ow, eps, out_eps, fp8_max, flags
):
    T, B, H = br.shape
    e8m0 = flags.get("E8M0_SCALE", False)
    # Only column-major E8M0 scales (stride_ys_t == 1) share lines across tokens.
    col_scales = e8m0 and flags.get("stride_ys_t") == 1
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
        NUM_XCDS=_num_xcds() if col_scales else 1,
        # The E8M0 epilogue lands at 261 VGPRs, past the 256 that four waves per
        # SIMD allow; the hint fits it in 254 without spills. Register allocation
        # only, the result is bit-identical (MI455, T=8192 B=8: 116 -> 109 us). From T=1024 only: at decode T
        # (T=550, B=8) it costs ~0.26 us, while the one-lane scale store alone already helps there.
        **({"waves_per_eu": 4} if e8m0 and T >= 1024 else {}),
        **flags,
    )


def _use_stream(T, B, col_scales) -> bool:
    """The persistent kernel, or a program a token? Both compute every row the same
    way, so this is a speed choice only (MI455X UT, H = 7168)."""
    if T >= _STREAM_MIN_T:
        return True
    if col_scales:
        # Column-major E8M0 scales: past decode T the stream kernel's XCD-contiguous
        # token runs win, except at a couple of thousand tokens with one or two rows.
        return T > _ONE_LANE_MAX_T and (T < 2048 or B >= 3)
    return T >= 2048


def launch_attn_res_tdm(
    br, ps, sw, y, y_scale, hs, hs2, pref, ow, eps, out_eps, fp8_max, flags
):
    T, B, H = br.shape
    if H % 1024 or ps.dtype != br.dtype:
        _launch_tdm(
            br, ps, sw, y, y_scale, hs, hs2, pref, ow, eps, out_eps, fp8_max, flags
        )
        return
    BD = 1 << (H - 1).bit_length()
    ncu = _num_cus(
        br.device.index if br.device.index is not None else torch.cuda.current_device()
    )
    args = (
        br,
        ps,
        sw,
        y,
        y_scale,
        hs,
        hs2,
        pref,
        ow,
    )
    scalars = (
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
    )
    e8m0 = flags.get("E8M0_SCALE", False)
    col_scales = e8m0 and flags.get("stride_ys_t") == 1
    if not _use_stream(T, B, col_scales):
        # Block rows in flight, as deep as the LDS share of the programs a CU gets
        # allows (a [1, BD] BF16 row is 16 KiB at H = 7168; a CU has 320 KiB).
        # Three depths only, so the kernel compiles at most three ways a flag set.
        per_cu = -(-T // ncu)
        prefetch = 8 if per_cu <= 1 else (4 if per_cu <= 2 else 2)
        kernel = (
            _attn_res_chunk_kernel
            if T <= _SPECIALIZE_MAX_T
            else _attn_res_chunk_kernel_rt
        )
        kernel[(T,)](
            *args,
            B,
            H,
            *scalars,
            BD,
            _NUM_WARPS,
            prefetch,
            num_warps=_NUM_WARPS,
            NUM_XCDS=_num_xcds() if col_scales else 1,
            # E8M0 codes: one layout conversion and a one-lane store per token pays
            # at decode T; with many programs finishing together the per-chunk
            # stores from the reduction layout are cheaper (T=256: 9.1 vs 9.8 us)
            CODES_ONE_LANE=T <= _ONE_LANE_MAX_T,
            **flags,
        )
        return
    # Persistent grid: never more programs than the CUs hold at once, or the late
    # ones would start their fixed token lists only after others finish.
    D = _STREAM_NBUF - 1
    NP = 1 + int(flags["DO_ADD"]) + int(flags["DO_ADD2"])
    NW = int(flags["OUT_NORM"])
    # the occupancy cache key: what selects the compiled kernel (B is not specialized)
    key = (H, D, y.dtype, *(flags.get(f, False) for f in _CONSTEXPR_FLAGS))
    ppc = min(_STREAM_PPC, _STREAM_OCCUPANCY.get(key, _STREAM_PPC))
    G = min(T, ncu * ppc)
    nx = _num_xcds()
    NX = nx if G % nx == 0 else 1
    kernel = _attn_res_stream_kernel[(G,)](
        *args,
        T,
        B,
        H,
        *scalars,
        BD,
        _NUM_WARPS,
        D,
        NX,
        NP,
        NW,
        num_warps=_NUM_WARPS,
        **flags,
    )
    if key not in _STREAM_OCCUPANCY:
        occ = _occupancy(kernel) if kernel is not None else None
        _STREAM_OCCUPANCY[key] = occ or _STREAM_PPC
