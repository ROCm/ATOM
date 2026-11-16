# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2 MLA prep (``ATOM_MONO_MLA_PREP``): an MLA layer M's prep -- its attention
seam (P0), fused_qkv_a_proj and the q_a / kv_a RMSNorms (P1 / P2), q_b, the
absorb, rope and the FP8 KV-cache write (P3 .. P5) -- as stages at the tail of
the PRECEDING layer's K2 launch: no new launch, no grid barrier. P0 and P1 are
here, with the stages' constants, arguments and mailbox regions; P2 .. P5 are
in ``mla_prep_qkv``.

Without it, between K2(M-1) and layer M's ``fused_qkv_a_proj`` the production
path runs the triton ``_attn_res_fused_kernel`` (prefix + add, the AttnRes mix,
input_layernorm, the per-token FP8 activation ``x8`` / ``xs``) on
``blocks[:, :before]``, read in place (copied first only if the bank is strided
along H). P0 makes the same three outputs --
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

After P0:

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

Then:

    P3  CTAs 56 .. 199 (144 tasks of 16 of q_b_proj's 2304 rows), after their
        P1: the FP8 weight rows, then P2DONE of the q rows, q8 at device scope,
        ``(sum * qs) * ws`` -> bf16 ``qb``, P3DONE.
    P4  CTAs 208 .. 255 (48 tasks: head h, 128 absorbed columns): P3DONE of the
        head's 12 tasks, the triton FP8 BMM's per-token 128-group prequant of
        q_nope against W_K (``mla_post``'s v-up code), then straight on to
        q_out: the rope kernel's nope part, RNE E4M3 of clamp(bf16 * inv_q);
        a head's first task also ropes its pe part into q_out.
    P5  on P2's kv CTAs: the FP8 KV-cache row of token t at its slot -- the
        nope part from the bf16 kv P2 made, the rope part of k_rope -- as
        aiter's ``fuse_qk_rope_concat_and_cache_mla_per_head_kernel``.

P3 / P4 sum the same exact FP8 products as the FlyDSL GEMM / the triton BMM
in another order; P5's KV row and q_out's rope part are bit-exact given equal
inputs (aiter's contraction, ``ftz``). A token with a negative slot (a padded
row) writes no KV row and no q_out row, as the rope kernel's early return.

Ported from the pre-#2871 fork (``up/mono-mla-prep``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
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
from atom.models.kimi_k3.mono.kernels.stages.mla_post_plan import (  # noqa: F401
    FP8_MAX,
    HEADS,
    INV_FP8_MAX,
    QH,
    QK_NOPE,
    QKV_A,
    QL,
    ROPE,
)
from atom.models.kimi_k3.mono.kernels.stages.mla_post_plan import LORA as KVL
from atom.mono.device import gemv
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fp8x8_bf16_pk,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import THREADS, WAVES

MSQ0 = 128  # P0's MSQ sums in ``red``, past the gate weights (``kda.stage_mix``)
AMX0 = 1024  # P0's AMX sums in ``red``, past the MSQ sums
MAX_ROWS = 16  # one token group: the AttnRes CTAs wrote prefix(M-1) themselves

ROWS = gemv.ROWS  # 16
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

# P3 .. P5
QB = HEADS * QH  # 2304
P3_TASKS = QB // ROWS  # 144
P3_CTA0 = 56  # 56 .. 199 (the lnorm CTAs, then P1's)
P3_PER_HEAD = QH // ROWS  # 12 tasks a head: 8 nope, then 4 rope
QB_K = QL  # 1536
ENTRY = KVL + ROPE  # 576: a KV-cache row, a q_out head
L_BLK = 128  # P4: absorbed columns a CTA (8 waves x 16)
P4_TASKS = HEADS * (KVL // L_BLK)  # 48
P4_CTA0 = 208  # 208 .. 255 (208 .. 215 after their P1, the rest after reduce)

REGIONS = ("ppart", "pwgt", "pmsq", "pamx", "pxrdy", "p1done", "p2done", "p3done")

# the prep build's arguments past K2's (and the MLA post stage's)
ARGS = (
    "pref",  # bf16 [S, 7168]: prefix(M) (the triton kernel's ``pref`` output)
    "x8",  # fp8 [S, 7168]: input_layernorm's per-token FP8 activation
    "xs",  # fp32 [S]: its scale
    "p_nw",  # layer M's attention seam: norm weight
    "p_qk",  # ... its query projection
    "p_onw",  # ... input_layernorm weight
    "w_qa",  # P1: fused_qkv_a_proj FP8 plane (AITER (16, 16) preshuffle)
    "w_qa_s",  # ... its fp32 row scales
    "qkv",  # bf16 [S, 2112]: q_c | kv_c | k_rope
    "q8",  # fp8 [S, 1536]: q_a_layernorm(q_c), per-token quant
    "qs",  # fp32 [S]: its scale
    "kvn",  # bf16 [S, 512]: kv_a_layernorm(kv_c)
    "qa_nw",  # q_a_layernorm weight
    "kva_nw",  # kv_a_layernorm weight
    "w_qb",  # P3: q_b_proj FP8 plane (AITER (16, 16) preshuffle)
    "w_qb_s",  # ... its fp32 row scales
    "w_k",  # W_K [12, 512, 128] E4M3
    "w_k_s",  # ... its fp32 scale
    "qb",  # bf16 [S, 2304]: q_b_proj's output
    "qout",  # fp8 [S, heads, 576]: the decode's q
    "qout_sr",  # ... its row stride (elements)
    "kvc",  # the layer's FP8 KV cache, [slots, 576] rows
    "kvc_sr",  # ... its row stride (bytes)
    "slots",  # int64 [S]: the step's slot mapping (negative: no row)
    "pos",  # int64 [S]: the positions
    "cosc",  # bf16 [max_pos, 32]: the rope tables
    "sinc",
    "k_sc",  # fp32 [1]: the KV cache's scale
    "q_sc",  # fp32 [1]: q_out's
)
INT32_ARGS = ("qout_sr", "kvc_sr")


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
        "p3done": P3_TASKS,
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
