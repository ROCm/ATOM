# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stage: the post-attention residual and norm of a front-half build (``front``): the
hand-off to the MoE half, one task per token."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from atom.models.minimax_m3.mono.config import HIDDEN
from atom.models.minimax_m3.mono.kernels.common import ld_bf16x12
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from atom.mono.device.sync import preg, publish
from atom.mono.plan.execution import THREADS, WAVES


@traced
def post_norm_defs(k4ctx):
    """``stage_post_norm``: token k's task (every thread 12 of the row's 6144 elements) reads the
    all-reduced attention row the o tasks gathered into ``a_ag`` and the layer's input
    residual, writes ``h_mid`` = bf16(bf16(sum) + residual) (the hidden state after attention)
    and x_moe = the post-attention gemma RMSNorm of the unrounded f32 sum (the MoE's input,
    in the fused all-reduce's reduction order: the router stage's) into scratch ``xmoe`` (and
    ``ar_out`` in a front-only build), all device scope,
    then publishes token k's done flag ``xmoe_done`` (the router stage waits on it)."""
    G = k4ctx["G"]
    SY = k4ctx["SY"]
    acquire = k4ctx["acquire"]
    ar_out = k4ctx["ar_out"]
    eps = k4ctx["eps"]
    fuse_k1 = k4ctx["fuse_k1"]
    # the post-attention norm's weight comes in the ``w_gate`` argument slot: a front build has no
    # router, and reading the ``g_post`` argument late in the kernel gives it a private segment
    # (68 B, measured at 32 rows; any other unused argument does not): a kernel with scratch is
    # refused (``WidthBuilds``)
    # (a whole-layer ``moe_back`` build has the router, so its ``w_gate`` is the router's: it reads ``g_post``)
    g_post = k4ctx["g_post"] if k4ctx["moe_back"] else k4ctx["w_gate"]
    h_in = k4ctx["h_in"]
    h_mid = k4ctx["h_mid"]
    k1_poll = k4ctx["k1_poll"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    mr = k4ctx["mr"]
    rdone_mb = k4ctx["rdone_mb"]
    red = k4ctx["red"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    wave = k4ctx["wave"]
    front = k4ctx["front"]
    assert k4ctx["AG_RS"]

    def stage_post_norm():
        r_g = rsrc(g_post)
        RE = (
            HIDDEN // THREADS
        )  # elements a thread: 12, every thread the same word offsets
        rw0 = tid * (RE // 2)
        for k in range(start("post"), tokens, G):
            k = fx.Int32(k)
            gp = ld_bf16x12(r_g, rw0)
            if const_expr(fuse_k1):
                # the residual (K1's norm task)
                if wave == 0:
                    k1_poll([(rdone_mb, k, 1)])
                gpu.barrier()
                acquire()
            hres = ld_bf16x12(rsrc(h_in), k * (HIDDEN // 2) + rw0, 0)
            arow = mb_poll(
                [
                    (
                        preg(sym, SY["a_ag"], "a_ag"),
                        (k * HIDDEN + tid * RE) // 2 + 2 * h,
                        2,
                    )
                    for h in range(RE // 4)
                ]
            )
            v = [
                f + fx.Float32(hres[4 * h + 2 * qq + i])
                for h in range(RE // 4)
                for qq in range(2)
                for i, f in enumerate(bf2_f32(arow[h][qq]))
            ]
            hw = (k * HIDDEN + tid * RE) // 2
            hb = (
                fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            )
            r_h = rsrc(h_mid)
            bo.buffer_store(
                fx.Vector.from_elements([hb[e] for e in range(4)], fx.Int32),
                r_h,
                hw,
                cache_modifier=CM_DEV,
            )
            bo.buffer_store(
                fx.Vector.from_elements([hb[4], hb[5]], fx.Int32),
                r_h,
                hw + 4,
                cache_modifier=CM_DEV,
            )
            ssw = fx.Float32(0.0)
            for e in range_constexpr(RE):
                ssw = fmath.fma(v[e], v[e], ssw)
            ssw = wave_sum(ssw)
            if lane == 0:
                fx.ptr_store(ssw, red + wave)
            gpu.barrier()
            ss = fx.ptr_load(red)
            for w in range_constexpr(1, WAVES):
                ss = ss + fx.ptr_load(red + w)
            rstd = hw_rsq(ss / float(HIDDEN) + eps)
            xn = [
                bf16_pair(
                    v[2 * j] * rstd * (fx.Float32(gp[2 * j]) + 1.0),
                    v[2 * j + 1] * rstd * (fx.Float32(gp[2 * j + 1]) + 1.0),
                ).bitcast(fx.Int32)
                for j in range(RE // 2)
            ]
            # x_moe: the scratch region ``xmoe`` (the merged kernel's router stage reads it there;
            # ``ar_out`` stays the layer's FFN output) and, in a front-only build, ``ar_out`` too
            for r_x in (rsrc(mb("xmoe")),) + ((rsrc(ar_out),) if front else ()):
                bo.buffer_store(
                    fx.Vector.from_elements(xn[0:4], fx.Int32),
                    r_x,
                    hw,
                    cache_modifier=CM_DEV,
                )
                bo.buffer_store(
                    fx.Vector.from_elements(xn[4:6], fx.Int32),
                    r_x,
                    hw + 4,
                    cache_modifier=CM_DEV,
                )
            stamp(26)
            publish(mb_put, mr("xmoe_done"), k, fx.Int32(1), tid == 0)
            gpu.barrier()  # `red` is the next token's reduction

    return {"stage_post_norm": stage_post_norm}
