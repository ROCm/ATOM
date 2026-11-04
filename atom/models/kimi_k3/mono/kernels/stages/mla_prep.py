# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2 MLA prep (``ATOM_MONO_MLA_PREP``), P0: an MLA layer M's attention seam as
a stage at the tail of the PRECEDING layer's K2 launch -- no new launch, no
grid barrier.

Without it, between K2(M-1) and layer M's ``fused_qkv_a_proj`` the production
path runs a copy of ``blocks[:, :before]`` and the triton
``_attn_res_fused_kernel`` (prefix + add, the AttnRes mix, input_layernorm, the
per-token FP8 activation ``x8`` / ``xs``). P0 makes the same three outputs --
``pref`` (prefix(M)), ``x8``, ``xs`` -- from what K2(M-1) already holds:

    CTA j < 56   after K2's final reduce: layer M-1's output, column slice j,
                 from FING (every owner pushes its bf16 sums to every rank's
                 FING, its own included, so a poll there is ``out``'s value);
                 the slice's sources and the prefix (``kda.slice_loads`` /
                 ``slice_compute`` with ``delta_lds`` and ``prefix_out``):
                 prefix(M) = prefix(M-1) + out into ``pref``, PPART; the gate
                 of token j (j < S, ``kda.stage_gate``) -> PWGT; the mix and
                 RMSNorm (``p0_mix_q``) -> PMSQ, the per-token amax -> PAMX,
                 x8 / xs at device scope, then PXRDY.

The AttnRes CTAs of a one-token-group step (S <= 16) are exactly the CTAs whose
MLP seam wrote prefix(M-1)'s slice j, so they read it with plain loads.

Numerics: the mono AttnRes seam (as every KDA layer's K1 and every MLP seam's
K2: the triton kernel's per-element arithmetic, the row reductions split
differently), then the triton kernel's per-token quant (amax * (1/448), IEEE
1 / scale, clamp, RNE E4M3).

Ported from the pre-#2871 fork (``up/mono-mla-prep``), P0 only: P1 (q_a / kv_a
GEMV), P2 (their RMSNorms + quant) and P3..P5 (q_b, absorb, rope + KV write)
follow.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr

from atom.models.kimi_k3.mono.kernels.attention.kda import (
    COLS,
    GSL,
    HIDDEN,
    NA,
    QC,
    f32_of,
    ld_bf4,
    slice_compute,
    slice_loads,
    stage_gate,
    sum_polled,
)
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    hw_rsq,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import THREADS, WAVES

FP8_MAX = 448.0
INV_FP8_MAX = 1.0 / FP8_MAX  # f32: 0x3b124925, as triton's constant
MSQ0 = 128  # P0's MSQ sums in ``red``, past the gate weights (``kda.stage_mix``)
AMX0 = 1024  # P0's AMX sums in ``red``, past the MSQ sums
MAX_ROWS = 16  # one token group: the AttnRes CTAs wrote prefix(M-1) themselves

REGIONS = ("ppart", "pwgt", "pmsq", "pamx", "pxrdy")

# the prep build's arguments past K2's (and the MLA post stage's)
ARGS = (
    "pref",  # bf16 [S, 7168]: prefix(M) (the triton kernel's ``pref`` output)
    "x8",  # fp8 [S, 7168]: input_layernorm's per-token FP8 activation
    "xs",  # fp32 [S]: its scale
    "p_nw",  # layer M's attention seam: norm weight
    "p_qk",  # ... its query projection
    "p_onw",  # ... input_layernorm weight
)


def scratch_pairs(s: int, ns: int) -> dict:
    """P0's mailbox pairs: layer M's seam (``ns`` sources) on the 56 slices."""
    return {
        "ppart": NA * s * ns * 2,
        "pwgt": s * (ns + 1),
        "pmsq": NA * s,
        "pamx": NA * s,
        "pxrdy": NA,
    }


def red_fits(s: int, ns: int) -> bool:
    """P0's sums fit ``red`` (WAVES x 64 x 4 floats) past each other."""
    return (
        s * (ns + 1) <= MSQ0
        and MSQ0 + 8 * s <= AMX0
        and AMX0 + 8 * s <= WAVES * 64 * 4
        and 8 * s <= THREADS
    )


def _f32(v):
    return fx.Float32(v)


def prep_context(c, ns, eps, out_eps):
    """K2's context with layer M's seam (``ns`` sources): its weights and eps,
    its own mailbox regions (this launch's tag already marks K2's MLP seam
    pairs), the prefix out to ``pref``."""
    a = c["args"]
    cp = dict(c)
    cp.update(
        NS=ns,
        NB=ns - 1,
        delta=True,
        reset=None,
        eps=eps,
        out_eps=out_eps,
        part=c["ppart"],
        wgt=c["pwgt"],
        msq=c["pmsq"],
        xrdy=c["pxrdy"],
        args={
            **a,
            "prefix_out": a["pref"],
            "ares_nw": a["p_nw"],
            "ares_qk": a["p_qk"],
            "in_nw": a["p_onw"],
        },
    )
    return cp


@traced
def p0_delta(c, j):
    """Columns 128 j ..: layer M-1's output from this rank's FING (the bf16
    words ``out`` holds) -> LDS ``delta_lds`` [t][128] as fp32."""
    s, tid = c["S"], c["tid"]
    n = s * 64
    dl = c["delta_lds"]
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        live = (tid + THREADS * i) < n
        t = e // 64
        cp2 = e % 64
        at = dl + (t * COLS + 2 * cp2)
        if live:
            w = c["poll"](
                [(c["peer"]["fing_own"], (t * HIDDEN + j * 128 + 2 * cp2) // 2, 1)]
            )[0][0]
            fx.ptr_store(bf_lo(w), at)
            fx.ptr_store(bf_hi(w), at + 1)
    gpu.barrier()


@traced
def p0_mix_q(c, task):
    """``kda.stage_mix``'s mix and output RMSNorm for any row count (a thread 4
    columns of one token a round), then the triton kernel's per-token FP8 quant
    of the normed fp32 row: per-slice amax -> PAMX, every slice's -> scale =
    amax * (1/448), inv = 1 / scale (0 at scale 0), clamp(y inv, +-448) -> RNE
    E4M3 x8 (device scope); slice 0 stores xs. Then PXRDY."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    c0 = task * COLS
    nw = s * (ns + 1)
    assert red_fits(s, ns)
    for i in range_constexpr((nw + THREADS - 1) // THREADS):
        w = tid + THREADS * i
        if w < nw:
            v = c["poll"]([(c["wgt"], w, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + w)
    gpu.barrier()
    tpr = THREADS // QC  # tokens a round
    cc = (tid % QC) * 4
    rounds = []
    for rr in range_constexpr((s + tpr - 1) // tpr):
        tr = rr * tpr + tid // QC
        mine = tr < s
        t = fx.min(tr, s - 1)
        mixed = [_f32(0.0) for _ in range(4)]
        for src in range_constexpr(ns):
            wt = fx.ptr_load(red + (t * (ns + 1) + src))
            v = [
                fx.ptr_load(c["vals"] + ((t * ns + src) * COLS + cc + j))
                for j in range(4)
            ]
            mixed = [mixed[j] + wt * v[j] for j in range(4)]
        den = fx.ptr_load(red + (t * (ns + 1) + ns))
        mixed = [mixed[j] / den for j in range(4)]
        sq = (
            mixed[0] * mixed[0]
            + mixed[1] * mixed[1]
            + mixed[2] * mixed[2]
            + mixed[3] * mixed[3]
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        if mine & (lane % 32 == 0):
            c["put"](c["msq"], task * s + t, sq)
        rounds.append((mine, t, mixed))
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (MSQ0 + g * s + tt))
    gpu.barrier()
    wn = ld_bf4(a["in_nw"], c0 + cc)
    ys = []
    for rr in range_constexpr(len(rounds)):
        mine, t, mixed = rounds[rr]
        tot = fx.ptr_load(red + (MSQ0 + t))
        for g in range_constexpr(1, 8):
            tot = tot + fx.ptr_load(red + (MSQ0 + g * s + t))
        r = hw_rsq(tot * _f32(1.0 / HIDDEN) + _f32(c["out_eps"]))
        y = [mixed[j] * r * wn[j] for j in range(4)]
        am = fx.max(fx.max(abs(y[0]), abs(y[1])), fx.max(abs(y[2]), abs(y[3])))
        am = butterfly(am, (16, 8, 4, 2, 1), fx.max)
        if mine & (lane % 32 == 0):
            c["put"](c["pamx"], task * s + t, am)
        ys.append(y)
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["pamx"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        m = f32_of(vals[0][0])
        for k in range_constexpr(1, GSL):
            m = fx.max(m, f32_of(vals[k][0]))
        fx.ptr_store(m, red + (AMX0 + g * s + tt))
    gpu.barrier()
    r_x8 = rsrc(a["x8"])
    for rr in range_constexpr(len(rounds)):
        mine, t, _ = rounds[rr]
        y = ys[rr]
        amax = fx.ptr_load(red + (AMX0 + t))
        for g in range_constexpr(1, 8):
            amax = fx.max(amax, fx.ptr_load(red + (AMX0 + g * s + t)))
        scale = amax * _f32(INV_FP8_MAX)
        inv = (scale > _f32(0.0)).select(_f32(1.0) / scale, _f32(0.0))
        qv = [
            fx.min(fx.max(y[j] * inv, _f32(-FP8_MAX)), _f32(FP8_MAX)) for j in range(4)
        ]
        if mine:
            bo.buffer_store(
                fp8_pack4(qv[0], qv[1], qv[2], qv[3]),
                r_x8,
                (t * HIDDEN + c0 + cc) // 4,
                cache_modifier=CM_DEV,
            )
            if (task == 0) & (cc == 0):
                bo.buffer_store(scale, rsrc(a["xs"]), t, cache_modifier=CM_DEV)
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def p0_seam(cp, j):
    """CTA j (< 56): layer M's attention seam, column slice j, then (CTA j < S)
    the gate of token j, then the mix and quant of slice j. ``cp``:
    ``prep_context``."""
    # the final stages' last LDS reads behind every wave before the slice's
    gpu.barrier()
    ld = slice_loads(cp, j)
    p0_delta(cp, j)
    slice_compute(cp, j, ld)
    if j < cp["S"]:
        stage_gate(cp, j)
    p0_mix_q(cp, j)
