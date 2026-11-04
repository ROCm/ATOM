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

Sub-mode 2 adds, after P0 (``ATOM_MONO_MLA_PREP=2``):

    P1  CTAs 84 .. 215 (132 tasks of 16 of fused_qkv_a_proj's 2112 rows), past
        the lnorm CTAs, after their own K2 work: the FP8 weight rows, then
        PXRDY, x8 at device scope (this launch's P0 wrote it), bf16 MFMAs on
        the exactly widened operands, ``(sum * xs) * ws`` -> bf16 ``qkv``,
        P1DONE (``mla_post.post_task``'s g_proj half).
    P2  CTAs 0 .. 2S-1 after P0: row t of q (CTA t) or kv (CTA S + t): aiter's
        ``fused_qk_rmsnorm_group_quant_kernel<bf16, fp8, 256, 8, 1, ..., per
        token>`` operation for operation on 256 of the 512 threads (thread v
        holds elements 8 v .. 8 v + 7, the same reduction tree), -> q8 + qs or
        kv bf16, P2DONE. Bit-exact given equal inputs: the same operations in
        the same order and tree, the same hardware approximations
        (``v_rcp_f32`` / ``v_rsq_f32``), and aiter's FP32 flush-to-zero
        denormal mode emulated (``ftz``: K2 itself keeps denormals).

P1 sums the same exact FP8 products as the FlyDSL GEMM in a different order.

Ported from the pre-#2871 fork (``up/mono-mla-prep``); P3..P5 (q_b, absorb,
rope + the FP8 KV-cache write) follow.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

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
from atom.models.kimi_k3.mono.kernels.stages import mla_post as mp
from atom.mono.device import gemv
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fp8x8_bf16_pk,
    fresh,
    hw_rcp,
    hw_rsq,
    mfma_bf16,
    row_sum,
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

ROWS = gemv.ROWS  # 16
QL = 1536  # q_lora_rank
KVL = 512  # kv_lora_rank
ROPE = 64  # qk_rope_head_dim
QKV_A = QL + KVL + ROPE  # 2112: fused_qkv_a_proj's rows
P1_TASKS = QKV_A // ROWS  # 132
P1_CTA0 = 84  # past the lnorm CTAs (56 .. 83); 84 .. 215
P1_Q = QL // ROWS  # tasks 0 .. 95 make q_c
P1_KV = KVL // ROWS  # tasks 96 .. 127 make kv_c (128 .. 131: k_rope)
GROUP_COLS = gemv.KGROUP  # 1024
AMAX_FLOOR = 1e-10  # aiter's thread_max start (0x2edbe6ff)
# P2: aiter's per-token kernel for max(n1, n2) = 1536: BlockSize 256, 8 a thread
P2_THREADS = 256
P2_TDS = 8
FLT_MIN = 1.1754943508222875e-38

REGIONS = ("ppart", "pwgt", "pmsq", "pamx", "pxrdy", "p1done", "p2done")

# the prep build's arguments past K2's (and the MLA post stage's): one ABI for
# every sub-mode, a later mode's 0 below it
ARGS = (
    "pref",  # bf16 [S, 7168]: prefix(M) (the triton kernel's ``pref`` output)
    "x8",  # fp8 [S, 7168]: input_layernorm's per-token FP8 activation
    "xs",  # fp32 [S]: its scale
    "p_nw",  # layer M's attention seam: norm weight
    "p_qk",  # ... its query projection
    "p_onw",  # ... input_layernorm weight
    "w_qa",  # mode 2: fused_qkv_a_proj FP8 plane (AITER (16, 16) preshuffle)
    "w_qa_s",  # ... its fp32 row scales
    "qkv",  # bf16 [S, 2112]: q_c | kv_c | k_rope
    "q8",  # fp8 [S, 1536]: q_a_layernorm(q_c), per-token quant
    "qs",  # fp32 [S]: its scale
    "kvn",  # bf16 [S, 512]: kv_a_layernorm(kv_c)
    "qa_nw",  # q_a_layernorm weight
    "kva_nw",  # kv_a_layernorm weight
)


def scratch_pairs(s: int, ns: int) -> dict:
    """P0's mailbox pairs: layer M's seam (``ns`` sources) on the 56 slices."""
    return {
        "ppart": NA * s * ns * 2,
        "pwgt": s * (ns + 1),
        "pmsq": NA * s,
        "pamx": NA * s,
        "pxrdy": NA,
        "p1done": P1_TASKS,
        "p2done": 2 * s,
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


# ------------------------------------------------------- P1 q_a / kv_a GEMV
@traced
def p1_task(c, j):
    """Task j: fused_qkv_a_proj rows 16 j .. for every token -> ``qkv`` (bf16,
    device scope), then its P1DONE pair. ``mla_post.post_task``'s g_proj half
    with x read past the L2 (this launch's P0 wrote it)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    xl = c["prep_xl"]
    plan = c["prep_slabs"]
    g = lane // ROWS
    row0 = j * ROWS
    # the weights in flight across the wait for P0
    gops = mp.w8_loads(lane, wave, a["w_qa"], row0, HIDDEN)
    if tid < NA:
        c["poll"]([(c["pxrdy"], tid, 1)])
    gpu.barrier()
    tiles = mp.token_tiles(s)
    ts = [t0 + fx.min(lane % ROWS, nt - 1) for t0, nt in tiles]
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in tiles]
    xrow = max(g1 - g0 for g0, g1 in plan) * GROUP_COLS + mp.XPAD
    for sl in range_constexpr(len(plan)):
        g0 = plan[sl][0]
        g1 = plan[sl][1]
        if const_expr(sl):
            gpu.barrier()
        gemv.rows_to_lds(
            tid,
            a["x8"],
            (g1 - g0) * GROUP_COLS // 2,
            s,
            xl,
            xrow // 2,
            CM_DEV,
            ldk=HIDDEN // 2,
            k0=g0 * GROUP_COLS // 2,
        )
        for i in range_constexpr(g0, g1):
            kk = ((i - g0) * WAVES + wave) * gemv.KCH + g * 8
            for q in range_constexpr(4):
                wa = mp.w8_operand(gops[i][q])
                for jt in range_constexpr(len(tiles)):
                    d = fx.Vector(
                        fx.ptr_load(
                            xl + (ts[jt] * xrow + kk + 32 * q) // 4,
                            result_type=fx.Vector.make_type(2, fx.Int32),
                        )
                    )
                    accs[jt] = mfma_bf16(wa, fp8x8_bf16_pk([d[0], d[1]]), accs[jt])
    xs_r = rsrc(a["xs"])
    r_out = rsrc(a["qkv"])
    for jt in range_constexpr(len(tiles)):
        u0 = tiles[jt][0]
        n = tiles[jt][1]
        fx.ptr_store(accs[jt], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < n * (ROWS // 2):
            tt = tid // (ROWS // 2)
            rp = tid % (ROWS // 2)
            t = u0 + tt
            xs = fx.Float32(
                bo.buffer_load(xs_r, t, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV)
            )
            outs = []
            for e in range_constexpr(2):
                r = 2 * rp + e
                outs.append(
                    (row_sum(red, r, tt) * xs) * mp.w8_scale(a["w_qa_s"], row0 + r)
                )
            bo.buffer_store(
                bf16_pair(outs[0], outs[1]).bitcast(fx.Int32),
                r_out,
                (t * QKV_A + row0 + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        gpu.barrier()
    publish(c["put"], c["p1done"], j, fx.Int32(1), tid == 0)


# ----------------------------------------- P2 q_a / kv_a RMSNorm (+ q quant)
def ftz(x):
    """gfx9 FP32 denormal mode 0, which aiter's kernels run in: a denormal
    operand or result is a zero of its sign. K2 keeps denormals, so P2 applies
    this to every operand it loads and to every result, as the hardware does
    to each of aiter's instructions."""
    zero = (x.bitcast(fx.Int32) & fx.Int32(-2147483648)).bitcast(fx.Float32)
    return (abs(x) < _f32(FLT_MIN)).select(zero, x)


def _fadd(p, q):
    return ftz(p + q)


def _p2_wave_reduce(v, op):
    """aiter ``wave_reduce`` over 64 lanes: quad_perm [1,0,3,2], [2,3,0,1],
    row_half_mirror, row_mirror, row_bcast 15 / 31 (lane 63) -- the binary
    tree pairing lanes (0,1), then pairs, ... -- which the xor butterfly
    1, 2, 4, 8, 16, 32 forms node for node (op is commutative)."""
    return butterfly(v, (1, 2, 4, 8, 16, 32), op)


@traced
def p2_norm(c, k):
    """CTA k < 2S: row t = k % S of q_c (k < S) or kv_c (k >= S), aiter's
    per-token kernel operation for operation on threads 0 .. 255 (its block):
    thread v holds elements 8 v .. 8 v + 7 (zeros past n), sum of squares
    ``x0 x0`` then fma chain, the wave tree, waves (0+1)+(2+3);
    r = rsq(fma(sum, rcp(n), eps)); y = (x r) w. q: amax over the chain from
    1e-10, scale = amax * (1/448), inv = rcp(scale), RNE E4M3 of clamp(y inv)
    -> q8 / qs; kv: bf16 RNE -> kvn. Then P2DONE."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    is_kv = k >= s
    t = k % s
    # q needs tasks 0 .. 95, kv 96 .. 127: a thread a task, the other kind's
    # threads idle
    if tid < P1_Q + P1_KV:
        want = is_kv.select(tid >= P1_Q, tid < P1_Q)
        if want:
            c["poll"]([(c["p1done"], tid, 1)])
    gpu.barrier()
    n = is_kv.select(fx.Int32(KVL), fx.Int32(QL))
    base = is_kv.select(fx.Int32(QL), fx.Int32(0))
    v_ = fx.min(tid, P2_THREADS - 1)
    active = (tid < P2_THREADS) & (v_ * P2_TDS < n)
    off = fx.min(v_ * P2_TDS, n - P2_TDS)
    w4 = fx.Vector(
        bo.buffer_load(
            rsrc(a["qkv"]),
            (t * QKV_A + base + off) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    wptr = is_kv.select(a["kva_nw"], a["qa_nw"])
    ww = fx.Vector(bo.buffer_load(rsrc(wptr), off // 2, vec_width=4, dtype=T.i32))
    x = []
    wv = []
    for d in range_constexpr(4):
        x += [bf_lo(w4[d]), bf_hi(w4[d])]
        wv += [bf_lo(ww[d]), bf_hi(ww[d])]
    x = [ftz(active.select(x[i], _f32(0.0))) for i in range(P2_TDS)]
    wv = [ftz(wv[i]) for i in range(P2_TDS)]
    ss = ftz(x[0] * x[0])
    for i in range_constexpr(1, P2_TDS):
        ss = ftz(_f32(fmath.fma(x[i], x[i], ss)))
    ws = _p2_wave_reduce(ss, _fadd)
    if (lane == 0) & (wave < P2_THREADS // 64):
        fx.ptr_store(ws, red + wave)
    gpu.barrier()
    tot = _fadd(
        _fadd(fx.ptr_load(red + 0), fx.ptr_load(red + 1)),
        _fadd(fx.ptr_load(red + 2), fx.ptr_load(red + 3)),
    )
    # the n of aiter's ``rcp[0] / n``: v_cvt_f32 + v_rcp_f32 at run time
    # (fast-math division), so not folded to the correctly rounded 1/n here
    rn = ftz(hw_rcp(fresh(n).to(fx.Float32)))
    eps = is_kv.select(_f32(c["kva_eps"]), _f32(c["qa_eps"]))
    r = ftz(hw_rsq(ftz(_f32(fmath.fma(tot, rn, eps)))))
    y = [ftz(ftz(x[i] * r) * wv[i]) for i in range(P2_TDS)]
    gpu.barrier()
    if is_kv:
        if active:
            words = [
                bf16_pair(y[2 * d], y[2 * d + 1]).bitcast(fx.Int32) for d in range(4)
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(a["kvn"]),
                (t * KVL + off) // 2,
                cache_modifier=CM_DEV,
            )
    else:
        tm = active.select(_f32(AMAX_FLOOR), _f32(0.0))
        for i in range_constexpr(P2_TDS):
            tm = fx.max(tm, abs(y[i]))
        wm = _p2_wave_reduce(tm, fx.max)
        if (lane == 0) & (wave < P2_THREADS // 64):
            fx.ptr_store(wm, red + wave)
        gpu.barrier()
        amax = fx.max(
            fx.max(fx.ptr_load(red + 0), fx.ptr_load(red + 1)),
            fx.max(fx.ptr_load(red + 2), fx.ptr_load(red + 3)),
        )
        scale = ftz(amax * _f32(INV_FP8_MAX))
        if tid == 0:
            bo.buffer_store(scale, rsrc(a["qs"]), t, cache_modifier=CM_DEV)
        inv = ftz(hw_rcp(scale))
        qv = [
            fx.min(fx.max(ftz(y[i] * inv), _f32(-FP8_MAX)), _f32(FP8_MAX))
            for i in range(P2_TDS)
        ]
        if active:
            bo.buffer_store(
                fx.Vector.from_elements(
                    [
                        fp8_pack4(qv[0], qv[1], qv[2], qv[3]),
                        fp8_pack4(qv[4], qv[5], qv[6], qv[7]),
                    ],
                    fx.Int32,
                ),
                rsrc(a["q8"]),
                (t * QL + off) // 4,
                cache_modifier=CM_DEV,
            )
    publish(c["put"], c["p2done"], k, fx.Int32(1), tid == 0)
    gpu.barrier()
