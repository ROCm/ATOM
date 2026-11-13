# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023 -- the stage's closures are built and called inside the loop that binds their names (FlyDSL traces them once per unrolled step)
"""The back half of a 32 / 64-row MoE layer as stages (``moe_back_defs``): router + xn quantization, routing units, up / gate (routed
a4w4 + MXFP8 shared pairs), down + FFN all-reduce. Used by the standalone ``kernels/moe_back.py`` and by K4's ``moe_back`` build.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    N_ROUTED,
    SHARED_EXPERT,
    TOP_K,
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S2S_BYTES,
    S13_BYTES,
    S13S_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W2S_BYTES,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
    W13S_BYTES,
)
from atom.models.minimax_m3.mono.layout import (
    SCORE_COPIES,
    UG_PER_SLOT,
    mxfp8_tile_dword,
)
from atom.mono.device.mx import (
    FP4,
    FP8,
    FP8_MAX,
    clamp_fp8,
    code_ceil,
    fp4_pack2,
    fp4_scale,
    mfma_scaled,
    pow2,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    batched_rounds,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    ld_dev,
    load_ptr64,
    popcount,
    row_live,
    rsrc,
    traced,
    uniform,
    wave_incl_scan,
    wave_sum,
    xshfl,
)
from atom.mono.device.queue import take, word
from atom.mono.device.ranks import sum_partials
from atom.mono.device.sync import preg, publish, shift
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

DN_ROWS = 32  # output rows of one down CTA
N_DN = HIDDEN // DN_ROWS  # 192 down CTAs
UG_PAIRS = UG_PER_SLOT // 2  # 24 pairs (32 mid rows each) per unit
UG_CPW = HIDDEN // 128 // WAVES  # 6 128-k chunks per wave
DN_KC = INTER // 128  # 6
XN_SLICE = 384  # xn elements one router task publishes per token (HIDDEN / 16)
N_RT = 16  # router tasks (expert groups of 8) per token
MID4_WORDS = INTER // 8  # 96
MID8_WORDS = INTER // 4  # 192
MIDSC_WORDS = INTER // 32  # 24
XN8_WORDS = HIDDEN // 4  # 1536
XN4_WORDS = HIDDEN // 8  # 768
XSC_WORDS = HIDDEN // 32  # 192
SH_HOST0 = BLOCKS - UG_PAIRS  # the shared pairs' host CTAs: SH_HOST0 .. BLOCKS - 1
# timeline points (``atom.mono.device.stamps``): router done, routing done, up / gate: xn in, routed loop done, shared pair done,
# down: flags in, down loop done, FFN reduce done
BK_STAMPS = {
    "r_x": 20,
    "r_dot": 21,
    "r_quant": 9,
    "router": 12,
    "routing": 13,
    "ug_xn": 14,
    "ug_loop": 15,
    "shared": 16,
    "dn_flags": 17,
    "dn_loop": 18,
    "fin": 19,
}


def umax_of(tokens: int) -> int:
    """Most (expert, 16-column tile) units a step has: every expert once, plus a tile per 16 rows past it (and never
    more than the TOP_K * tokens (expert, token) rows: a unit holds at least one)."""
    return min(N_ROUTED + TOP_K * tokens // 16, TOP_K * tokens)


def back_lds_layout(n: int) -> dict:
    """Word offsets (Int32 words from the pool base, 4-word aligned) of the back half's LDS arrays and their total ``end``: ``pool`` (the router's
    x rows, then the routing's wave masks, then the down stage's per-wave token partials and shared partials), ``red`` / ``upp`` (the waves'
    MFMA partials), ``misc`` (a down task's output rows), the unit tables."""
    um = umax_of(n)
    items = [
        ("pool", max(n // 16 * (HIDDEN // 2), 2 * N_ROUTED * n)),
        ("red", WAVES * 64 * 4),
        ("upp", WAVES * 64 * 4),
        ("misc", DN_ROWS * n),
        ("uexp", um + 2),
        ("ubase", N_ROUTED + 2),
        ("emask", 2 * N_ROUTED),
        ("urow", um * 16),
        ("uwt", um * 16),
        ("route", 8 * n),
        ("rwt", 8 * n),
        ("sig", WAVES * N_ROUTED),
        ("wscan", 8),
        ("claim", 4),
    ]
    out, off = {}, 0
    for name, words in items:
        out[name] = off
        off += (words + 3) // 4 * 4
    out["end"] = off
    return out


@traced
def moe_back_defs(ctx):
    """The back half's stage functions over the kernel's names (``ctx``): ``stage_router`` .. ``stage_down``."""
    n = ctx["n"]
    TPT = n // 16  # tokens per router task = column tiles of the shared expert
    NT = n // 16
    UMAX = umax_of(n)
    SH_FLAG0 = (
        UMAX * UG_PAIRS
    )  # the shared pairs' flags (fixed: known before the routing)
    W = ctx["W"]
    G = BLOCKS
    SY = ctx["SY"]
    DEP = ctx["DEP"]
    SU = ctx["SU"]
    stop = ctx["stop"]
    MERGED = ctx["MERGED"]
    stamp = ctx["stamp"]
    AR_FLAG = ctx["AR_FLAG"]
    A_NT = ctx.get("A_NT", 1)  # nontemporal weight loads
    NOCOMP = ctx.get(
        "NOCOMP", 0
    )  # a probe: the weight tiles are loaded and summed, no MFMA (what the loads alone cost)
    route_scale = ctx["route_scale"]
    shared_weight = ctx["shared_weight"]
    swiglu_limit = ctx["swiglu_limit"]
    tid, bid, lane, wave, g4, l16 = (
        ctx[k] for k in ("tid", "bid", "lane", "wave", "g4", "l16")
    )
    xs, v4f = ctx["xs"], ctx["v4f"]
    MID4_BYTES = TOP_K * n * MID4_WORDS * 4
    MIDSC4_BYTES = TOP_K * n * MIDSC_WORDS * 4
    mb, mr, acquire = ctx["mb"], ctx["mr"], ctx["acquire"]
    mb_put, mb_put_bf, mb_put_words, mb_poll = (
        ctx["mb_put"],
        ctx["mb_put_bf"],
        ctx["mb_put_words"],
        ctx["mb_poll"],
    )
    x_base = ctx["x_base"]
    w_gate, bias, w13, s13, w2, s2 = (
        ctx[k] for k in ("w_gate", "bias", "w13", "s13", "w2", "s2")
    )
    w13s, s13s, w2s, s2s = (ctx[k] for k in ("w13s", "s13s", "w2s", "s2s"))
    ar_out, sym, peers, rank, batch_ids = (
        ctx[k] for k in ("ar_out", "sym", "peers", "rank", "batch_ids")
    )

    # the back half's LDS arrays, carved out of the kernel's pool ``xs``
    LDS = back_lds_layout(n)
    fpool = fx.recast_iter(fx.Float32, xs)
    red = fx.recast_iter(fx.Float32, xs + LDS["red"])
    upp = fx.recast_iter(fx.Float32, xs + LDS["upp"])
    misc = fx.recast_iter(fx.Float32, xs + LDS["misc"])
    uexp = xs + LDS["uexp"]
    ubase = xs + LDS["ubase"]
    emask = xs + LDS["emask"]
    urow = xs + LDS["urow"]
    uwt = fx.recast_iter(fx.Float32, xs + LDS["uwt"])
    route = xs + LDS["route"]
    rwt = fx.recast_iter(fx.Float32, xs + LDS["rwt"])
    sig = fx.recast_iter(fx.Float32, xs + LDS["sig"])
    wscan = xs + LDS["wscan"]
    claim_slot = xs + LDS["claim"]
    STEAL = ctx.get("STEAL", 0)
    DN_SPLIT = ctx.get("DN_SPLIT", 0)
    DEPB = ctx.get("DEPB", 1)  # down units whose mid rows (B) are in flight

    def route_key(v, e):
        """Signed-orderable i32 of score v with its low 7 bits := 127 - expert e: a pick is one integer max."""
        b = v.bitcast(fx.Int32)
        b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
        return (b & fx.Int32(-128)) | (127 - e)

    def a_tile(r_w, rg, k, st):
        """A lane's 16 B of an FP4 weight's (16, 16)-shuffled 16-row group rg (K = k) at 128-k step st (``fp4_tile_load``; ``A_NT``: nontemporal)."""
        return fx.Vector(
            bo.buffer_load(
                r_w,
                rg * (2 * k) + st * 256 + lane * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT if A_NT else 0,
            )
        )

    def e8m0_load(r_s, byte_idx):
        return fx.Int32(bo.buffer_load(r_s, byte_idx, vec_width=1, dtype=T.i8))

    def group_scale_byte(expert, rows, rg, kc, cols):
        """scale_index(expert rows + 16 rg + l16, 4 kc + g4, cols) folded to wave-uniform terms plus the lane."""
        base = (expert * (rows // 32) + rg // 2) * (cols // 8) + kc // 2
        return (base * 64 + lane) * 4 + (kc % 2) * 2 + rg % 2

    def mx8_scale(vals, offsets):
        amax = fx.max(vals[0], -vals[0])
        for v in vals[1:]:
            amax = fx.max(amax, fx.max(v, -v))
        amax = butterfly(amax, offsets, fx.max)
        e = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        return e, hw_rcp(pow2(e))

    def mx8_q(v, inv):
        return clamp_fp8(v * inv)

    def fp8_lanes4(q):
        """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False))
        return lo | (xshfl(lo, 2) << 16)

    def fp8_tile_load(r_w, row, cols, kc):
        """A lane's 32 B of an MXFP8 weight in aiter's (16, 16) preshuffle, row ``row`` at 128-k chunk ``kc``."""
        dw = mxfp8_tile_dword(row, cols, kc, g4)
        halves = [
            fx.Vector(
                bo.buffer_load(
                    r_w, dw + 256 * h, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
                )
            )
            for h in range(2)
        ]
        return fx.Vector.from_elements(
            [halves[h][e] for h in range(2) for e in range(4)], fx.Int32
        )

    def fp8_scale_load(r_s, row, cols, kc):
        """The E8M0 scale dword of row ``row``'s 128-k chunk ``kc`` (four 32-block bytes), RAW: ``fp8_scale_byte`` picks this lane's block
        where the MFMA uses it (shifted at load, every load waits every load before it).
        """
        return fx.Int32(
            bo.buffer_load(r_s, row * (cols // 128) + kc, vec_width=1, dtype=T.i32)
        )

    def fp8_scale_byte(w):
        return fresh(w) >> (g4 * 8)

    # =========================================================== R. router + xn quantization
    def stage_router():
        tg = bid // N_RT
        t = bid % N_RT
        tok0 = tg * TPT
        e_r = t * WAVES + wave
        bias_r = fx.Float32(bo.buffer_load(rsrc(bias), e_r, vec_width=1, dtype=T.f32))
        r_gate = rsrc(w_gate)
        gw = [
            fx.Vector(
                bo.buffer_load(
                    r_gate,
                    (e_r * HIDDEN + (i * 64 + lane) * 8) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for i in range(HIDDEN // 512)
        ]
        if const_expr(MERGED):  # the front half's post stage flags each token's row
            if wave == 0:
                mb_poll([(mr("xmoe_done"), tok0 + tk, 1) for tk in range(TPT)])
            gpu.barrier()
            acquire()
        x_src = x_base + fx.Int64(tok0 * HIDDEN * 2)
        batched_rounds(
            tid, TPT * (HIDDEN // 2) // 4, lambda v: fx.Vector(ld_dev(x_src, v * 4, 4)),
            lambda v, w: fx.ptr_store(w, xs + v * 4), batch=8,
        )  # fmt: skip
        gpu.barrier()
        stamp(BK_STAMPS["r_x"])
        for tk in range_constexpr(TPT):
            accs = [fx.Float32(0.0) for _ in range(4)]
            for i in range_constexpr(HIDDEN // 512):
                xv = fx.Vector(
                    fx.ptr_load(
                        xs + (tk * (HIDDEN // 2) + (i * 64 + lane) * 4), result_type=v4f
                    )
                ).bitcast(fx.Int32)
                for j in range_constexpr(4):
                    accs[j] = fx.Float32(
                        rocdl.fdot2_f32_bf16(
                            T.f32,
                            as_ir_value(
                                fx.Vector.from_elements([gw[i][j]], fx.Int32).bitcast(
                                    fx.BFloat16
                                )
                            ),
                            as_ir_value(
                                fx.Vector.from_elements([xv[j]], fx.Int32).bitcast(
                                    fx.BFloat16
                                )
                            ),
                            as_ir_value(accs[j]),
                            clamp=False,
                        ).result
                    )
            logit = wave_sum((accs[0] + accs[1]) + (accs[2] + accs[3]))
            sc = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
            rk = route_key(sc + bias_r, e_r)
            if lane < SCORE_COPIES:
                mb_put_words(
                    shift(
                        mr("scores"),
                        (fx.Int64(lane) * n + fx.Int64(tok0 + tk)) * (N_ROUTED * 16),
                    ),
                    e_r * 2,
                    [rk, sc.bitcast(fx.Int32)],
                )
        stamp(BK_STAMPS["r_dot"])
        # the slice's quantization: a lane 4 elements, 8 lanes a 32-block, thread tid -> (token tid / 96, lane tid % 96)
        if tid < TPT * 96:
            tkq = tid // 96
            q = tid % 96
            tk_g = tok0 + tkq
            w0 = tkq * (HIDDEN // 2) + t * (XN_SLICE // 2) + q * 2
            f = [
                v
                for j in range(2)
                for v in bf2_f32(fx.ptr_load(xs + (w0 + j)).bitcast(fx.Int32))
            ]
            e, inv = mx8_scale(f, (1, 2, 4))
            word = fp8_pack4(*[mx8_q(v, inv) for v in f])
            bo.buffer_store(
                word,
                rsrc(mb("xn8p")),
                tk_g * XN8_WORDS + t * (XN_SLICE // 4) + q,
                cache_modifier=CM_DEV,
            )
            if q % 8 == 0:
                bo.buffer_store(
                    e,
                    rsrc(mb("xscp")),
                    tk_g * XSC_WORDS + t * (XN_SLICE // 32) + q // 8,
                    cache_modifier=CM_DEV,
                )
            amax4 = fx.max(fx.max(f[0], -f[0]), fx.max(f[1], -f[1]))
            amax4 = fx.max(amax4, fx.max(fx.max(f[2], -f[2]), fx.max(f[3], -f[3])))
            e4, sc4 = fp4_scale(butterfly(amax4, (1, 2, 4), fx.max))
            w16 = fp4_pack2(fx.Int32(0), f[0], f[1], sc4, 0)
            w16 = fp4_pack2(w16, f[2], f[3], sc4, 1) & 0xFFFF
            word4 = w16 | (xshfl(w16, 1) << 16)
            if q % 2 == 0:
                bo.buffer_store(
                    word4,
                    rsrc(mb("xn4p")),
                    tk_g * XN4_WORDS + t * (XN_SLICE // 8) + q // 2,
                    cache_modifier=CM_DEV,
                )
            if q % 8 == 0:
                bo.buffer_store(
                    e4,
                    rsrc(mb("xs4p")),
                    tk_g * XSC_WORDS + t * (XN_SLICE // 32) + q // 8,
                    cache_modifier=CM_DEV,
                )
        stamp(BK_STAMPS["r_quant"])
        publish(mb_put, mr("xndone"), bid, fx.Int32(1), tid == 0)
        stamp(BK_STAMPS["router"])

    # ====================================================== T. routing -> units
    def route_top4(k, k0, sc0, k1, sc1, wmask, mask_word, mask_bit):
        """route[8k : 8k+4] / rwt := token k's sigmoid top-4 from its routing keys k0 / k1 (experts lane / lane +
        64) and sigmoids; the picks' bit set in this wave's expert masks (``wmask``: expert e at words 2e, 2e + 1).
        """
        fx.ptr_store(sc0, sig + (wave * N_ROUTED + lane))
        fx.ptr_store(sc1, sig + (wave * N_ROUTED + 64 + lane))
        pid = fx.Int32(0)
        for p in range_constexpr(TOP_K):
            m = butterfly(fx.max(k0, k1), (32, 16, 8, 4, 2, 1), fx.max)
            hit0 = k0 == m
            k0 = hit0.select(fx.Int32(-(2**31)), k0)
            k1 = ((k1 == m) & (k0 != m)).select(fx.Int32(-(2**31)), k1)
            pid = (lane == p).select(127 - (m & 127), pid)
        if lane < TOP_K:
            w = fx.ptr_load(sig + (wave * N_ROUTED + pid))
            tot = butterfly(w, (1, 2))
            fx.ptr_store(pid, route + (8 * k + lane))
            fx.ptr_store(w * (route_scale / fx.max(tot, 1e-20)), rwt + (8 * k + lane))
            was = fx.Int32(fx.ptr_load(wmask + (pid * 2 + mask_word)))
            fx.ptr_store(was | mask_bit, wmask + (pid * 2 + mask_word))
        if lane == 0:
            fx.ptr_store(fx.Int32(SHARED_EXPERT), route + (8 * k + TOP_K))
            fx.ptr_store(fx.Float32(shared_weight), rwt + (8 * k + TOP_K))

    def stage_routing():
        """-> uexp[0 .. nu), urow / uwt (per unit, column), wscan[1] = nu. Ends with a barrier."""
        mine = bid % SCORE_COPIES
        wmask = xs + wave * (2 * N_ROUTED)
        for h in range_constexpr(4):
            fx.ptr_store(fx.Int32(0), wmask + (lane + 64 * h))
        wave_toks = [wave + WAVES * j for j in range(n // WAVES)]
        sg = mb_poll(
            [
                (
                    shift(
                        mr("scores"),
                        (fx.Int64(mine) * n + fx.Int64(k)) * (N_ROUTED * 16),
                    ),
                    (lane + 64 * h) * 2,
                    2,
                )
                for k in wave_toks
                for h in range(2)
            ]
            + [
                (mr("xndone"), lane + 64 * c, 1) for c in range(BLOCKS // 64)
            ],  # every router task's xn rows are in
            batch=24,
        )
        for j in range_constexpr(n // WAVES):
            k = wave_toks[j]
            live = row_live(batch_ids, k)
            route_top4(
                k,
                sg[2 * j][0],
                sg[2 * j][1].bitcast(fx.Float32),
                sg[2 * j + 1][0],
                sg[2 * j + 1][1].bitcast(fx.Float32),
                wmask,
                k // 32,
                live.select(fx.Int32(1) << (k % 32), fx.Int32(0)),
            )
        gpu.barrier()
        acquire()
        # per expert: the token mask (the waves' or), its tiles, the units' prefix
        ntile = fx.Int32(0)
        m0 = fx.Int32(0)
        m1 = fx.Int32(0)
        if tid < N_ROUTED:
            for w in range_constexpr(WAVES):
                m0 = m0 | fx.Int32(fx.ptr_load(xs + (w * 2 * N_ROUTED + tid * 2)))
                m1 = m1 | fx.Int32(fx.ptr_load(xs + (w * 2 * N_ROUTED + tid * 2 + 1)))
            cnt = popcount(m0) + popcount(m1)
            ntile = (cnt + 15) // 16
        incl = wave_incl_scan(ntile, lane)
        if (tid < N_ROUTED) & (lane == 63):
            fx.ptr_store(incl, wscan + wave)
        gpu.barrier()
        if tid < N_ROUTED:
            off0 = fx.Int32(fx.ptr_load(wscan))
            base = incl - ntile + (wave == 1).select(off0, fx.Int32(0))
            fx.ptr_store(base, ubase + tid)
            fx.ptr_store(m0, emask + (tid * 2))
            fx.ptr_store(m1, emask + (tid * 2 + 1))
            for tl in range_constexpr(n // 16):
                if tl < ntile:
                    fx.ptr_store(fx.Int32(tid), uexp + (base + tl))
            if tid == N_ROUTED - 1:
                fx.ptr_store(base + ntile, wscan + 2)  # nu
        for i in range_constexpr((UMAX * 16 + THREADS - 1) // THREADS):
            ix = tid + THREADS * i
            if ix < UMAX * 16:
                fx.ptr_store(fx.Int32(-1), urow + ix)
                fx.ptr_store(fx.Float32(0.0), uwt + ix)
        gpu.barrier()
        # token k's pick j -> its unit column
        if tid < TOP_K * n:
            k = tid // TOP_K
            j = tid % TOP_K
            e = fx.Int32(fx.ptr_load(route + (8 * k + j)))
            mm0 = fx.Int32(fx.ptr_load(emask + (e * 2)))
            mm1 = fx.Int32(fx.ptr_load(emask + (e * 2 + 1)))
            bit = fx.Int32(1) << (k % 32)
            below = bit - 1
            rk_ = (k < 32).select(
                popcount(mm0 & below), popcount(mm0) + popcount(mm1 & below)
            )
            ub = fx.Int32(fx.ptr_load(ubase + e))
            ix = (ub + rk_ // 16) * 16 + rk_ % 16
            if (
                fx.Int32(bo.buffer_load(rsrc(batch_ids), k, vec_width=1, dtype=T.i32))
                >= 0
            ):
                fx.ptr_store(k * TOP_K + j, urow + ix)
                fx.ptr_store(fx.ptr_load(rwt + (8 * k + j)), uwt + ix)
        # the routing record (tests)
        if (bid == 0) & (tid < MOE_SLOTS * n):
            sk = tid // MOE_SLOTS
            sj = tid % MOE_SLOTS
            mb_put(
                mr("sel"), sk * 2 * MOE_SLOTS + sj, fx.ptr_load(route + (8 * sk + sj))
            )
            mb_put(
                mr("sel"),
                sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                fx.ptr_load(rwt + (8 * sk + sj)),
            )
        gpu.barrier()
        stamp(BK_STAMPS["routing"])

    # ============================================================ U. up / gate
    def ug_loadA(uu, blk, half, has, scales):
        """The routed pair (unit ``uu``, block ``blk``)'s half: gate / up A tiles of this wave's 6 chunks (+ the
        scale dwords of both halves)."""
        e_u = fx.min(uniform(fx.ptr_load(uexp + fx.min(uu, UMAX - 1))), N_ROUTED - 1)
        r_w = rsrc(
            w13 + fx.Int64(e_u) * W13_BYTES,
            has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
        )
        r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        cg = blk * 2 + half
        wts = []
        scs = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(a_tile(r_w, rg, HIDDEN, kc))
                if const_expr(scales and i % 2 == 0):
                    scs[gu].append(
                        fx.Int32(
                            bo.buffer_load(
                                r_s,
                                group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS)
                                // 4,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                    )
        return wts, scs

    def ug_loadB(uu, has):
        """This lane's column of unit ``uu``: its MXFP4 xn rows (6 chunks of this wave) and their codes; a column
        without a token reads out of range (zeros)."""
        row = fx.Int32(fx.ptr_load(urow + (fx.min(uu, UMAX - 1) * 16 + l16)))
        ok = has & (row >= 0)
        tok = fx.max(row, 0) // TOP_K
        bq = []
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xv = fx.Vector(
                bo.buffer_load(
                    rsrc(mb("xn4p"), n * XN4_WORDS * 4),
                    tok * XN4_WORDS + kc * 16 + g4 * 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                    mask=ok,
                )
            )
            xc = fx.Int32(
                bo.buffer_load(
                    rsrc(mb("xs4p"), n * XSC_WORDS * 4),
                    tok * XSC_WORDS + kc * 4 + g4,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                    mask=ok,
                )
            )
            bq.append((xv, xc))
        return bq

    def ug_reduce(cg_, cu_):
        """The wave's gate / up accumulators -> swiglu: thread (row r, column c)'s mid value."""
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, upp + (wave * 64 + lane) * 4)
        gpu.barrier()
        r = tid % 16
        c = fx.min(tid // 16, 15)
        gsum = fx.Float32(0.0)
        usum = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            gsum = gsum + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + c) * 4 + r % 4))
            usum = usum + fx.ptr_load(upp + ((w * 64 + 16 * (r // 4) + c) * 4 + r % 4))
        gpu.barrier()
        return gsum, usum

    def ug_y(wts, scs, half, bq):
        cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
        cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(UG_CPW):
            sel = (i % 2) * 2 + half
            if const_expr(NOCOMP):
                cg_ = fx.Vector.from_elements(
                    [
                        fx.Float32(cg_[e])
                        + fx.Float32(wts[0][i][e].bitcast(fx.Float32))
                        for e in range(4)
                    ],
                    fx.Float32,
                )
                cu_ = fx.Vector.from_elements(
                    [
                        fx.Float32(cu_[e])
                        + fx.Float32(wts[1][i][e].bitcast(fx.Float32))
                        for e in range(4)
                    ],
                    fx.Float32,
                )
                continue
            cg_ = mfma_scaled(
                wts[0][i],
                bq[i][0],
                cg_,
                scs[0][i // 2],
                bq[i][1],
                a_fmt=FP4,
                b_fmt=FP4,
                sa_byte=sel,
            )
            cu_ = mfma_scaled(
                wts[1][i],
                bq[i][0],
                cu_,
                scs[1][i // 2],
                bq[i][1],
                a_fmt=FP4,
                b_fmt=FP4,
                sa_byte=sel,
            )
        gsum, usum = ug_reduce(cg_, cu_)
        return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

    def ug_publish(pr, y_lo, y_hi):
        """Pair pr = (unit pr / 24, block pr % 24)'s mid rows as MXFP4 words + E8M0 code per 32-block, then its flag."""
        uu = pr // UG_PAIRS
        blk = pr % UG_PAIRS
        r = tid % 16
        c = fx.min(tid // 16, 15)
        lo = bf16_round(y_lo)
        hi = bf16_round(y_hi)
        amax = butterfly(fx.max(fx.max(lo, -lo), fx.max(hi, -hi)), (1, 2, 4, 8), fx.max)
        e4, sc4 = fp4_scale(amax)
        words = []
        for v in (lo, hi):
            b = fp4_pack2(fx.Int32(0), v, xshfl(v, 1), sc4, 0) & 0xFF
            b = b | (xshfl(b, 2) << 8)
            words.append(b | (xshfl(b, 4) << 16))
        row = fx.Int32(fx.ptr_load(urow + (fx.min(uu, UMAX - 1) * 16 + c)))
        if (tid < 256) & (row >= 0):
            if r % 8 == 0:
                for h in range_constexpr(2):
                    bo.buffer_store(
                        words[h],
                        rsrc(mb("mid4p")),
                        row * MID4_WORDS + blk * 4 + h * 2 + r // 8,
                        cache_modifier=CM_DEV,
                    )
            if r == 0:
                bo.buffer_store(
                    e4,
                    rsrc(mb("mids4p")),
                    row * MIDSC_WORDS + blk,
                    cache_modifier=CM_DEV,
                )
        if const_expr(
            ctx.get("NOPUBWAIT", 0)
        ):  # a timing probe (wrong: the flag may overtake its data): no drain before the flag
            gpu.barrier()
            if tid == 0:
                mb_put(mr("ugdone"), pr, fx.Int32(1))
        else:
            publish(mb_put, mr("ugdone"), pr, fx.Int32(1), tid == 0)

    # the shared expert (MXFP8 weights, every token): pair j = 32 mid rows
    def sh_loadA(cg):
        r_w = rsrc(w13s, W13S_BYTES)
        r_s = rsrc(s13s, S13S_BYTES)
        wts, scs = [], []
        for gu in range_constexpr(2):
            row = (gu * UG_PER_SLOT + cg) * 16 + l16
            kcs = [wave * UG_CPW + i for i in range(UG_CPW)]
            wts.append([fp8_tile_load(r_w, row, HIDDEN, kc) for kc in kcs])
            scs.append([fp8_scale_load(r_s, row, HIDDEN, kc) for kc in kcs])
        return wts, scs

    def sh_tile(wts, scs, t):
        """Column tile t's gate / up accumulators over this wave's 6 chunks; B = xn8 rows read from scratch (all 6 chunks' loads
        first, then the MFMAs: a load behind each MFMA is a round trip a chunk)."""
        accg = fx.Vector.filled(4, 0.0, fx.Float32)
        accu = fx.Vector.filled(4, 0.0, fx.Float32)
        col = fresh(
            t * 16 + l16
        )  # keeps tile t's B loads behind tile t - 1's reduce (live ranges)
        SHB = 3  # chunks whose B loads are in flight together (all 6: 54 more VGPRs)
        for g0 in range_constexpr(0, UG_CPW, SHB):
            bs = []
            for i in range_constexpr(g0, g0 + SHB):
                kc = wave * UG_CPW + i
                parts = [
                    fx.Vector(
                        bo.buffer_load(
                            rsrc(mb("xn8p"), n * XN8_WORDS * 4),
                            col * XN8_WORDS + kc * 32 + 16 * j + 4 * g4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                    for j in range(2)
                ]
                sb = fx.Int32(
                    bo.buffer_load(
                        rsrc(mb("xscp"), n * XSC_WORDS * 4),
                        col * XSC_WORDS + kc * 4 + g4,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                bs.append((parts, sb))
            for k, i in enumerate(range_constexpr(g0, g0 + SHB)):
                parts, sb = bs[k]
                xb = fx.Vector.from_elements(
                    [parts[j][e] for j in range(2) for e in range(4)], fx.Int32
                )
                accg = mfma_scaled(
                    wts[0][i], xb, accg, fp8_scale_byte(scs[0][i]), sb, a_fmt=FP8
                )
                accu = mfma_scaled(
                    wts[1][i], xb, accu, fp8_scale_byte(scs[1][i]), sb, a_fmt=FP8
                )
        return accg, accu

    def sh_y(wts, scs, t):
        accg, accu = sh_tile(wts, scs, t)
        gsum, usum = ug_reduce(accg, accu)
        return swiglu_mul_batch(
            [bf16_round(gsum)], [bf16_round(usum)], fx.Float32(-swiglu_limit)
        )[0]

    def sh_pair(jsh):
        """The host's shared pair: per half and column tile reduce -> swiglu; MXFP8 mid rows, then the flag."""
        wts, scs = sh_loadA(2 * jsh)
        y_lo = [sh_y(wts, scs, t) for t in range_constexpr(NT)]
        wts, scs = sh_loadA(2 * jsh + 1)
        for t in range_constexpr(NT):
            y_hi = sh_y(wts, scs, t)
            r = tid % 16
            c = fx.min(tid // 16, 15)
            lo = bf16_round(y_lo[t])
            hi = bf16_round(y_hi)
            sc, inv = mx8_scale([lo, hi], (1, 2, 4, 8))
            words = [fp8_lanes4(mx8_q(v, inv)) for v in (lo, hi)]
            tokr = t * 16 + c
            if tid < 256:
                if r % 4 == 0:
                    for h in range_constexpr(2):
                        bo.buffer_store(
                            words[h],
                            rsrc(mb("mid8p")),
                            tokr * MID8_WORDS + jsh * 8 + h * 4 + r // 4,
                            cache_modifier=CM_DEV,
                        )
                if r == 0:
                    bo.buffer_store(
                        sc,
                        rsrc(mb("midsp")),
                        tokr * MIDSC_WORDS + jsh,
                        cache_modifier=CM_DEV,
                    )
        publish(mb_put, mr("ugdone"), SH_FLAG0 + jsh, fx.Int32(1), tid == 0)

    def stage_ug(nu):
        n_pairs = nu * UG_PAIRS
        if const_expr(STEAL):
            stage_ug_steal(n_pairs)
            return
        rounds = (n_pairs + SU * UG_PAIRS + (G - 1)) // G
        full = fx.max(rounds - SU, 0)
        cnt = (bid < SH_HOST0).select(rounds, full)

        def pair_of(rd):
            tail = full * G + (rd - full) * SH_HOST0 + bid
            pr = (rd < full).select(
                rd * G + bid, (bid < SH_HOST0).select(tail, n_pairs)
            )
            return fx.min(pr, fx.max(n_pairs - 1, 0)), pr < n_pairs

        pr0, has0 = pair_of(fx.Int32(0))
        A_lo, scs = ug_loadA(pr0 // UG_PAIRS, pr0 % UG_PAIRS, 0, has0, True)
        stamp(BK_STAMPS["ug_xn"])
        B = ug_loadB(pr0 // UG_PAIRS, has0)
        y_lo = fx.Float32(0.0)
        for rd in range(cnt):
            pr, has = pair_of(rd)
            A_hi, _ = ug_loadA(pr // UG_PAIRS, pr % UG_PAIRS, 1, has, False)
            if has:
                y_lo = ug_y(A_lo, scs, 0, B)
            prn, hasn = pair_of(rd + 1)
            nA_lo, nscs = ug_loadA(prn // UG_PAIRS, prn % UG_PAIRS, 0, hasn, True)
            nB = ug_loadB(prn // UG_PAIRS, hasn)
            if has:
                y_hi = ug_y(A_hi, scs, 1, B)
                ug_publish(pr, y_lo, y_hi)
            A_lo = nA_lo
            scs = nscs
            B = nB
        stamp(BK_STAMPS["ug_loop"])

    def stage_ug_steal(n_pairs):
        """The up / gate pairs by work stealing: a CTA below SH_HOST0 starts on pair ``bid``, every other pair (SH_HOST0 + k) is the k-th
        atomic claim of the layer's counter, taken two iterations before it is worked (its latency under a whole pair's loads); a CTA
        the shared pair kept busy (a host) simply claims later, so ``sh_units`` is not needed. A claim is made only if this CTA's loop
        will still work it, so no claimed pair is dropped."""
        cq = word(mb("claim") + fx.Int64((fx.Int32(ctx["layer"]) & 127) * 4))
        cnt = (n_pairs + (G - 1)) // G + 2
        last = fx.max(n_pairs - 1, 0)

        def claim_issue(ok):
            tk = fx.Int32(0)
            if (tid == 0) & ok:
                tk = fx.Int32(take(cq))
            return tk

        def claim_land(tk, ok):
            if tid == 0:
                fx.ptr_store(tk, claim_slot)
            gpu.barrier()
            return ok.select(
                fx.Int32(SH_HOST0) + uniform(fx.ptr_load(claim_slot)), n_pairs
            )

        pr = (bid < SH_HOST0).select(fx.Int32(bid), n_pairs)
        prc = fx.min(pr, last)
        has0 = pr < n_pairs
        A_lo, scs = ug_loadA(prc // UG_PAIRS, prc % UG_PAIRS, 0, has0, True)
        stamp(BK_STAMPS["ug_xn"])
        B = ug_loadB(prc // UG_PAIRS, has0)
        true_ = fx.Int32(0) == 0
        prn = claim_land(claim_issue(true_), true_)
        y_lo = fx.Float32(0.0)
        for rd in range(cnt):
            prc = fx.min(pr, last)
            has = pr < n_pairs
            A_hi, _ = ug_loadA(prc // UG_PAIRS, prc % UG_PAIRS, 1, has, False)
            okc = rd + 2 < cnt
            tk = claim_issue(okc)
            if has:
                y_lo = ug_y(A_lo, scs, 0, B)
            prnc = fx.min(prn, last)
            hasn = prn < n_pairs
            nA_lo, nscs = ug_loadA(prnc // UG_PAIRS, prnc % UG_PAIRS, 0, hasn, True)
            nB = ug_loadB(prnc // UG_PAIRS, hasn)
            if has:
                y_hi = ug_y(A_hi, scs, 1, B)
                ug_publish(prc, y_lo, y_hi)
            pn2 = claim_land(tk, okc)
            A_lo = nA_lo
            scs = nscs
            B = nB
            pr = prn
            prn = pn2
        stamp(BK_STAMPS["ug_loop"])

    def stage_shared_host():
        """The shared pair of host CTA SH_HOST0 + j, before the routing (it needs the xn rows, not the routing): both its halves'
        tiles streaming while the others' routing is still on its way."""
        jsh = fx.Int32(bid) - SH_HOST0
        if jsh >= 0:
            if wave == 0:
                mb_poll([(mr("xndone"), lane + 64 * c, 1) for c in range(BLOCKS // 64)])
            gpu.barrier()
            acquire()
            sh_pair(jsh)
        stamp(BK_STAMPS["shared"])

    # =============================================================== D. down + AR
    def ffn_finish_flag(t):
        """Tag-free FFN all-reduce (``ar_flag``): misc (this CTA's bf16 partial rows for the 32 columns of row group t) -> the owner rank
        t % W as plain bf16 (8 B a thread: 4 columns of one token), a drain, then one flag pair per (source, t); the owner sums the W partials
        in rank order in fp32, writes bf16 sums to every rank (plain payload, drain, flag pair per t); every rank stores ar_out.
        """
        owner = t % W
        dst = load_ptr64(peers, owner)
        tk = tid // 8
        q4 = (tid % 8) * 4
        if tid < n * 8:
            v = [fx.ptr_load(misc + (tk * DN_ROWS + q4 + e)) for e in range(4)]
            bo.buffer_store(
                fx.Vector.from_elements(
                    [
                        bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                        bf16_pair(v[2], v[3]).bitcast(fx.Int32),
                    ],
                    fx.Int32,
                ),
                rsrc(dst + fx.Int64(SY["ffnb"])),
                ((rank * n + tk) * HIDDEN + t * DN_ROWS + q4) // 2,
                cache_modifier=CM_SYS,
            )
        publish_peer(preg(dst, SY["ffnb_f"], "ffnb_f"), rank * N_DN + t)
        if t % W == rank:  # the owner
            if wave == 0:
                mb_poll(
                    [
                        (preg(sym, SY["ffnb_f"], "ffnb_f"), src * N_DN + t, 1)
                        for src in range(W)
                    ]
                )
            gpu.barrier()
            if tid < n * 8:
                acc = [fx.Float32(0.0) for _ in range(4)]
                for src in range_constexpr(W):
                    pw_ = fx.Vector(
                        bo.buffer_load(
                            rsrc(sym + fx.Int64(SY["ffnb"])),
                            ((src * n + tk) * HIDDEN + t * DN_ROWS + q4) // 2,
                            vec_width=2,
                            dtype=T.i32,
                            cache_modifier=CM_SYS,
                        )
                    )
                    for h in range_constexpr(2):
                        lo, hi = bf2_f32(pw_[h])
                        acc[2 * h] = acc[2 * h] + lo
                        acc[2 * h + 1] = acc[2 * h + 1] + hi
                ow = fx.Vector.from_elements(
                    [
                        bf16_pair(acc[0], acc[1]).bitcast(fx.Int32),
                        bf16_pair(acc[2], acc[3]).bitcast(fx.Int32),
                    ],
                    fx.Int32,
                )
                for w in range_constexpr(W):
                    bo.buffer_store(
                        ow,
                        rsrc(load_ptr64(peers, w) + fx.Int64(SY["ffnb_ag"])),
                        (tk * HIDDEN + t * DN_ROWS + q4) // 2,
                        cache_modifier=CM_SYS,
                    )
            for w in range_constexpr(W):
                publish_peer(preg(load_ptr64(peers, w), SY["ffnb_agf"], "ffnb_agf"), t)
        if wave == 0:
            mb_poll([(preg(sym, SY["ffnb_agf"], "ffnb_agf"), t, 1)])
        gpu.barrier()
        if tid < n * 8:
            got = fx.Vector(
                bo.buffer_load(
                    rsrc(sym + fx.Int64(SY["ffnb_ag"])),
                    (tk * HIDDEN + t * DN_ROWS + q4) // 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
            )
            bo.buffer_store(got, rsrc(ar_out), (tk * HIDDEN + t * DN_ROWS + q4) // 2)

    def publish_peer(a, i):
        """The CTA's plain peer stores drained, then flag pair ``i`` of region ``a`` (tid 0)."""
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            mb_put(a, i, fx.Int32(1))
        gpu.barrier()

    def ffn_finish(t):
        """misc (row group t's bf16 partial rows, token k's DN_ROWS from k DN_ROWS) -> every peer ->
        rank-ordered sum -> ar_out."""
        if const_expr(W > 1 and AR_FLAG):
            ffn_finish_flag(t)
            return
        peer_dst = load_ptr64(peers, fx.min(wave, W - 1))
        if const_expr(W > 1):
            # push bf16 partial rows to the row group's owner rank (wave w -> peer w; only the owner's wave pushes)
            owner = t % W
            half = DN_ROWS // 2
            for c in range_constexpr((half * n + 63) // 64):
                i = lane + 64 * c
                if (wave == owner) & (i < half * n):
                    mb_put_bf(
                        preg(peer_dst, SY["ffn"], "ffn"),
                        (rank * n + i // half) * HIDDEN + t * DN_ROWS + (i % half) * 2,
                        [fx.ptr_load(misc + i * 2), fx.ptr_load(misc + (i * 2 + 1))],
                    )
            gpu.barrier()
        for c in range_constexpr((DN_ROWS // 2 * n + THREADS - 1) // THREADS):
            ix = tid + THREADS * c
            if ix < DN_ROWS // 2 * n:
                row_tok = ix // (DN_ROWS // 2)
                row = t * DN_ROWS + (ix % (DN_ROWS // 2)) * 2
                if const_expr(W == 1):
                    t0 = fx.ptr_load(misc + ix * 2)
                    t1 = fx.ptr_load(misc + (ix * 2 + 1))
                else:
                    own = preg(sym, SY["ffn"], "ffn")
                    t0 = fx.Float32(0.0)
                    t1 = fx.Float32(0.0)
                    if t % W == rank:  # the owner: sum, gather to every rank
                        t0, t1 = sum_partials(
                            mb_poll,
                            own,
                            lambda src: ((src * n + row_tok) * HIDDEN + row) // 2,
                            W,
                        )
                        for w in range_constexpr(W):
                            mb_put_bf(
                                preg(load_ptr64(peers, w), SY["ffn_ag"], "ffn_ag"),
                                row_tok * HIDDEN + row,
                                [t0, t1],
                            )
                    else:
                        got = mb_poll(
                            [
                                (
                                    preg(sym, SY["ffn_ag"], "ffn_ag"),
                                    (row_tok * HIDDEN + row) // 2,
                                    1,
                                )
                            ]
                        )
                        t0, t1 = bf2_f32(got[0][0])
                bo.buffer_store(
                    fx.Vector.from_elements([t0, t1], fx.Float32).to(fx.BFloat16),
                    rsrc(ar_out),
                    row_tok * HIDDEN + row,
                )

    def stage_down(nu):
        td = bid
        rg_d = td * 2 + wave // 4
        q = uniform(wave % 4)
        sh8 = (wave // 4) * 8  # the scale byte of this wave's row group within a dword
        nw = fx.max((nu - q + 3) // 4, 0)
        pw = fpool + (wave * n * 16)  # this wave's per-token partials, [token][16 rows]
        shp = fpool + (WAVES * n * 16)  # the shared partials, [wave][tile][lane][4]

        def dn_loadA(j):
            uu = q + 4 * j
            ok = uu < nu
            de = fx.min(uniform(fx.ptr_load(uexp + fx.min(uu, UMAX - 1))), N_ROUTED - 1)
            r_w = rsrc(
                w2 + fx.Int64(de) * W2_BYTES, ok.select(fx.Int32(W2_BYTES), fx.Int32(0))
            )
            r_s = rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0)))
            wts = [a_tile(r_w, rg_d, INTER, kc) for kc in range(DN_KC)]
            scd = [
                fx.Int32(
                    bo.buffer_load(
                        r_s,
                        group_scale_byte(de, HIDDEN, rg_d, 2 * d, W2_SCALE_COLS) // 4,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                for d in range(DN_KC // 2)
            ]
            return wts, scd

        def dn_loadB(j):
            uu = q + 4 * j
            ok = j < nw
            row = fx.Int32(fx.ptr_load(urow + (fx.min(uu, UMAX - 1) * 16 + l16)))
            w = fx.ptr_load(uwt + (fx.min(uu, UMAX - 1) * 16 + l16))
            vok = ok & (row >= 0)
            rr_ = fx.max(row, 0)
            bt = [
                fx.Vector(
                    bo.buffer_load(
                        rsrc(mb("mid4p"), MID4_BYTES),
                        rr_ * MID4_WORDS + kc * 16 + g4 * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                        mask=vok,
                    )
                )
                for kc in range(DN_KC)
            ]
            bs = [
                fx.Int32(
                    bo.buffer_load(
                        rsrc(mb("mids4p"), MIDSC4_BYTES),
                        rr_ * MIDSC_WORDS + kc * 4 + g4,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                        mask=vok,
                    )
                )
                for kc in range(DN_KC)
            ]
            return bt, bs, row, w

        Apipe = [dn_loadA(i) for i in range(DEP)]
        # the shared expert's units (mid8p / midsp rows of every token): this wave's chunks q and q + 4, tiles in flight now
        sh_units = []
        for h in range_constexpr(2):
            skc = q + 4 * h
            sok = skc < DN_KC
            skc = fx.min(skc, DN_KC - 1)
            srow = rg_d * 16 + l16
            sr_w = rsrc(w2s, sok.select(fx.Int32(W2S_BYTES), fx.Int32(0)))
            sr_s = rsrc(s2s, sok.select(fx.Int32(S2S_BYTES), fx.Int32(0)))
            sh_units.append(
                (
                    fp8_tile_load(sr_w, srow, INTER, skc),
                    fp8_scale_load(sr_s, srow, INTER, skc),
                    skc,
                    sok,
                )
            )
        # zero this wave's partials (its own, no barrier)
        for i in range_constexpr(n * 16 // 256):
            fx.ptr_store(
                fx.Vector.filled(4, 0.0, fx.Float32), pw + (lane * 4 + 256 * i)
            )
        # the shared pairs' flags (the hosts ran them first), then the shared expert's down MFMAs while the routed pairs finish
        mb_poll([(mr("ugdone"), SH_FLAG0 + fx.min(tid, UG_PAIRS - 1), 1)])
        gpu.barrier()
        acquire()
        shacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NT)]
        for swv, ssc, skc, sok in sh_units:
            for t in range_constexpr(NT):
                col = t * 16 + l16
                parts = [
                    fx.Vector(
                        bo.buffer_load(
                            rsrc(mb("mid8p"), n * MID8_WORDS * 4),
                            col * MID8_WORDS + skc * 32 + 16 * jh + 4 * g4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                    for jh in range(2)
                ]
                xb = fx.Vector.from_elements(
                    [parts[jh][e] for jh in range(2) for e in range(4)], fx.Int32
                )
                sb = fx.Int32(
                    bo.buffer_load(
                        rsrc(mb("midsp"), n * MIDSC_WORDS * 4),
                        col * MIDSC_WORDS + skc * 4 + g4,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                scu = mfma_scaled(
                    swv,
                    xb,
                    fx.Vector.filled(4, 0.0, fx.Float32),
                    fp8_scale_byte(ssc),
                    sb,
                    a_fmt=FP8,
                )
                shacc[t] = fx.Vector.from_elements(
                    [
                        shacc[t][e]
                        + sok.select(scu[e] * shared_weight, fx.Float32(0.0))
                        for e in range(4)
                    ],
                    fx.Float32,
                )
        for t in range_constexpr(NT):
            fx.ptr_store(shacc[t], shp + (((wave * NT + t) * 64 + lane) * 4))
        # every routed up / gate pair's flag -- in two polls: a wave takes its units in ascending order and the pairs are worked unit-major,
        # so the units of the first half (and the one past it: its B rows are loaded one unit ahead) are in long before the last pair
        # is; the rest are polled at the half. ``DN_SPLIT``: the whole poll first, as before (the last pair's latency is then exposed)
        n_ug = nu * UG_PAIRS
        jm = fx.min(nw, (nu // 2 + 3) // 4) if const_expr(DN_SPLIT) else nw
        cut = fx.min(
            n_ug, (jm + DEPB) * 4 * UG_PAIRS
        )  # pairs [0, cut) first: the B rows are loaded DEPB units ahead

        def poll_pairs(lo, hi):
            mb_poll(
                [
                    (mr("ugdone"), fx.min(lo + tid + THREADS * i, fx.max(hi - 1, 0)), 1)
                    for i in range((UMAX * UG_PAIRS + THREADS - 1) // THREADS)
                ]
            )
            gpu.barrier()
            acquire()

        if nu > 0:
            poll_pairs(fx.Int32(0), cut if const_expr(DN_SPLIT) else n_ug)
        else:
            gpu.barrier()
            acquire()
        stamp(BK_STAMPS["dn_flags"])
        if const_expr(stop == "dnpre"):
            fx.ptr_store(fx.Float32(Apipe[0][1][0]), misc + tid)
            return

        def run_units(lo, hi, Apipe, Bpipe):
            for j in range(lo, hi):
                nA = dn_loadA(j + DEP)
                nB = dn_loadB(j + DEPB)
                wts, scd = Apipe[0]
                bt, bs, row, w = Bpipe[0]
                cacc = fx.Vector.filled(4, 0.0, fx.Float32)
                for kc in range_constexpr(DN_KC):
                    if const_expr(NOCOMP):
                        cacc = fx.Vector.from_elements(
                            [
                                fx.Float32(cacc[e])
                                + fx.Float32(wts[kc][e].bitcast(fx.Float32))
                                for e in range(4)
                            ],
                            fx.Float32,
                        )
                        continue
                    cacc = mfma_scaled(
                        wts[kc],
                        bt[kc],
                        cacc,
                        fresh(scd[kc // 2])
                        >> sh8,  # shifted where used: shifted at load it waits every prefetch in flight
                        bs[kc],
                        a_fmt=FP4,
                        b_fmt=FP4,
                        sa_byte=(kc % 2) * 2,
                    )
                if row >= 0:
                    tok = row // TOP_K
                    pv = fx.ptr_load(pw + (tok * 16 + g4 * 4), result_type=v4f)
                    fx.ptr_store(
                        fx.Vector.from_elements(
                            [
                                fx.Float32(pv[e]) + bf16_round(cacc[e] * w)
                                for e in range(4)
                            ],
                            fx.Float32,
                        ),
                        pw + (tok * 16 + g4 * 4),
                    )
                Apipe = Apipe[1:] + [nA]
                Bpipe = Bpipe[1:] + [nB]
            return Apipe, Bpipe

        Bpipe = [dn_loadB(fx.Int32(i)) for i in range(DEPB)]
        Apipe, Bpipe = run_units(fx.Int32(0), jm, Apipe, Bpipe)
        if const_expr(DN_SPLIT):
            if nu > 0:
                poll_pairs(cut, n_ug)
            Apipe, Bpipe = run_units(jm, nw, Apipe, Bpipe)
        stamp(BK_STAMPS["dn_loop"])
        if const_expr(stop == "dnloop"):
            fx.ptr_store(Bpipe[0][3], misc + tid)
            return
        gpu.barrier()
        for c in range_constexpr((DN_ROWS * n + THREADS - 1) // THREADS):
            ix = tid + THREADS * c
            if ix < DN_ROWS * n:
                k = ix // DN_ROWS
                grp = (ix % DN_ROWS) // 16
                rr = ix % 16
                routed = fx.Float32(0.0)
                shared = fx.Float32(0.0)
                for qq in range_constexpr(4):
                    w_ = grp * 4 + qq
                    routed = routed + fx.ptr_load(fpool + (w_ * n * 16 + k * 16 + rr))
                    shared = shared + fx.ptr_load(
                        fpool
                        + (
                            WAVES * n * 16
                            + ((w_ * NT + k // 16) * 64 + 16 * (rr // 4) + k % 16) * 4
                            + rr % 4
                        )
                    )
                fx.ptr_store(
                    bf16_round(bf16_round(routed) + bf16_round(shared)), misc + ix
                )
        gpu.barrier()
        ffn_finish(td)
        stamp(BK_STAMPS["fin"])

    return {
        "stage_router": stage_router,
        "stage_routing": stage_routing,
        "stage_ug": stage_ug,
        "stage_shared_host": stage_shared_host,
        "stage_down": stage_down,
        "wscan": wscan,
    }
