# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.
# FlyDSL traces closures and constexpr guards during kernel construction.
# ruff: noqa: B023

"""The MoE FFN of the DeepSeek-V4 MonoKernel: router, experts and the TP reduce."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.deepseek_v4.mono.config import SCALE_BM
from atom.models.deepseek_v4.mono.kernels import buffer_ops as bo
from atom.models.deepseek_v4.mono.kernels.common import sqrt_softplus, swiglu
from atom.models.deepseek_v4.mono.kernels.helpers import traced
from atom.models.deepseek_v4.mono.kernels.plan import (
    ROUTER_TILE,
    UG8,
    UG_TILE,
    WAVES,
    down_route_samples,
    ffn_hcc,
    router_spt,
)
from atom.models.deepseek_v4.mono.kernels.primitives import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_roundtrip,
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    rsrc,
    uniform,
    uniform_f32,
    xshfl,
)


@traced
def ffn_stages(ctx):
    """8-10. FFN-side mixers, router, expert up/gate and down, and the MoE TP peer reduce."""
    DN_TILE = ctx["DN_TILE"]
    DOWN_ROUTE_ROUNDS = ctx["DOWN_ROUTE_ROUNDS"]
    G = ctx["G"]
    HC = ctx["HC"]
    HC_COEF = ctx["HC_COEF"]
    HC_MISC = ctx["HC_MISC"]
    HIDDEN = ctx["HIDDEN"]
    INTER = ctx["INTER"]
    MOE_SLOTS = ctx["MOE_SLOTS"]
    N_DN_TILES = ctx["N_DN_TILES"]
    N_EXPERTS = ctx["N_EXPERTS"]
    N_ROUTER = ctx["N_ROUTER"]
    N_UG = ctx["N_UG"]
    N_UG_PER_SLOT = ctx["N_UG_PER_SLOT"]
    N_UG_TASKS = ctx["N_UG_TASKS"]
    PUBLISH_BLOCKS = ctx["PUBLISH_BLOCKS"]
    S = ctx["S"]
    SHARED_EXPERT = ctx["SHARED_EXPERT"]
    SHARED_FP8 = ctx["SHARED_FP8"]
    shared_bf16_block128 = ctx["shared_bf16_block128"]
    routed_down_bf16 = ctx["routed_down_bf16"]
    TOP_K = ctx["TOP_K"]
    UG_PER_SLOT = ctx["UG_PER_SLOT"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    XQ_WAVES = ctx["XQ_WAVES"]
    dnw = ctx["dnw"]
    emit_out = ctx["emit_out"]
    g_post = ctx["g_post"]
    get = ctx["get"]
    get2_many = ctx["get2_many"]
    hc_ffn_fn = ctx["hc_ffn_fn"]
    hc_ffn_sb = ctx["hc_ffn_sb"]
    hc_post = ctx["hc_post"]
    hc_pre_stages = ctx["hc_pre_stages"]
    hc_stage_coef = ctx["hc_stage_coef"]
    hint_wait = ctx["hint_wait"]
    keys = ctx["keys"]
    lane = ctx["lane"]
    load_bias = ctx["load_bias"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    mma_units = ctx["mma_units"]
    mx_rg = ctx["mx_rg"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    peer_reduce = ctx["peer_reduce"]
    poll = ctx["poll"]
    put = ctx["put"]
    put2 = ctx["put2"]
    quant_mxfp8 = ctx["quant_mxfp8"]
    quant_scaled = ctx["quant_scaled"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    route_topk = ctx["route_topk"]
    run_units = ctx["run_units"]
    s_dn = ctx["s_dn"]
    s_sdn = ctx["s_sdn"]
    s_sug = ctx["s_sug"]
    s_ug = ctx["s_ug"]
    st_f8 = ctx["st_f8"]
    stage_moe_input = ctx["stage_moe_input"]
    stage_x_rmsnorm = ctx["stage_x_rmsnorm"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    swiglu_limit = ctx["swiglu_limit"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_f8f8 = ctx["unit_f8f8"]
    unit_fp8 = ctx["unit_fp8"]
    unit_fp8mx = ctx["unit_fp8mx"]
    unit_mxfp4 = ctx["unit_mxfp4"]
    use_fp8_block128 = ctx["use_fp8_block128"]
    use_mxfp4_weight = ctx["use_mxfp4_weight"]
    use_mxfp8_block32 = ctx["use_mxfp8_block32"]
    w_dn = ctx["w_dn"]
    w_r = ctx["w_r"]
    w_sdn = ctx["w_sdn"]
    w_sug = ctx["w_sug"]
    w_ug = ctx["w_ug"]
    wave = ctx["wave"]
    x_out = ctx["x_out"]
    xs = ctx["xs"]
    attention_fp8_bf16 = ctx["attention_fp8_bf16"]
    hc_contract = ctx["hc_contract"]
    load_x_rmsnorm = ctx["load_x_rmsnorm"]
    rmsnorm_tail_ks = ctx["_rmsnorm_tail_ks"]
    # FFN side: the router contracts the streams itself unless ffn_hcc
    hc_coef_f = None
    FFN_HCC = ffn_hcc(S, HC, N_EXPERTS)
    if const_expr(FFN_HCC):
        hc_pre_stages(
            "f",
            1,
            hc_ffn_fn,
            hc_ffn_sb,
            lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
            "ain",
            src_pair=lambda s, k: (mb("a"), (s * HC * HIDDEN + k) // 2),
        )
    elif const_expr(HC > 1):
        hc_coef_f = hc_pre_stages(
            "f",
            1,
            hc_ffn_fn,
            hc_ffn_sb,
            lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
            None,
            contract=False,
        )

    # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
    r_wr = rsrc(w_r)
    R_NKC = HIDDEN // 64
    SPT = router_spt(S, N_EXPERTS)
    assert SPT <= ROUTER_TILE
    assert S % SPT == 0, "router tasks must publish every live sample's scores"
    for tt in range(start("router"), S * N_ROUTER // SPT, G):
        tt = fx.Int32(tt)
        t = tt % N_ROUTER
        rs0 = (tt // N_ROUTER) * SPT
        stamp("router", tt, 0)

        # K-fold: rows / columns 0..7 take the first K half, 8..15 the second
        r_sub = t * ROUTER_TILE % 16
        r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
        R_CPW = R_NKC // WAVES // 2
        r_fold = (lane % 16) // ROUTER_TILE
        r_ns = fx.min(lane % 8, fx.Int32(SPT - 1))

        def u_r(c):
            kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
            return unit_bf16(
                r_wr,
                t * ROUTER_TILE // 16,
                kc,
                R_NKC,
                (r_ns * HIDDEN + kc * 64) // 2,
                r_ln,
            )

        pre = [u_r(c) for c in range(R_CPW)]
        hint_wait(0, None, mark=("router", tt))
        # this task's expert-activation blocks ride along (MXFP8: 4 16-lane groups a wave)
        r_gp = rsrc(g_post)
        if const_expr(use_mxfp8_block32):
            x_blk = (wave * N_ROUTER + t) * 4 + lane // 16
            xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 32 + lane % 16 * 2
        else:
            x_blk = wave * N_ROUTER + t
            xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
        x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
        xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
        xa = []
        raw_collapse = []

        def ld_a(sks):
            # sks = (local sample s, k) pairs; local sample s is sample rs0 + s
            n4 = len(sks)
            if const_expr(
                HC == 1 or FFN_HCC
            ):  # one stream: `a` itself, or hcc_f's `ain`
                x_src = "ain" if FFN_HCC else "a"
                specs = [(mb(x_src), ((rs0 + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                specs += [
                    (mb(x_src), ((rs0 + s) * HIDDEN + xk) // 2, 1) for s in range(SPT)
                ]
                v = poll(specs, batch=len(specs))
                for s in range_constexpr(SPT):
                    xa.append(bf2_f32(v[n4 + s][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:n4]]
            # x = sum_j pre[j] * stream j, contracted here in hcc's order and rounding
            specs = [
                (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + k) // 2, 2)
                for s, k in sks
                for j in range(HC)
            ]
            specs += [
                (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + xk) // 2, 1)
                for s in range(SPT)
                for j in range(HC)
            ]
            v = poll(specs)
            hc_coef_f(1, tt == 0)  # task 0 also publishes post / comb for down
            pjs = [
                [lds_ld(misc, HC_MISC + (rs0 + s) * HC_COEF + j) for j in range(HC)]
                for s in range(SPT)
            ]

            def mix(words, pj):
                """bf16(sum_j pre[j] * x_j) for each element of the words' streams."""
                return hc_contract(words, pj)

            for s in range_constexpr(SPT):
                xa.append(
                    tuple(
                        bf16_round(a)
                        for a in mix(v[(n4 + s) * HC : (n4 + s + 1) * HC], pjs[s])
                    )
                )
            raw = [mix(v[i * HC : (i + 1) * HC], pjs[sks[i][0]]) for i in range(n4)]
            raw_collapse.extend(raw)
            return [[bf16_round(a) for a in row] for row in raw]

        loaded = load_x_rmsnorm(ld_a, HIDDEN, g_post, count=SPT)
        if const_expr(attention_fp8_bf16 and HC > 1 and FFN_HCC):
            # The handoff rounds the numerator to BF16, not the RMS sum.
            # Read the producer's original FP32 collapse at the same tail
            # positions as the inline path, preserving its reduction order.
            ks, _ = rmsnorm_tail_ks(HIDDEN)
            raw = get2_many(
                [
                    (mb("ain_raw"), (rs0 + s) * HIDDEN + k + j)
                    for s in range(SPT)
                    for k in ks
                    for j in (0, 2)
                ]
            )
            raw_collapse = [
                list(raw[2 * i]) + list(raw[2 * i + 1]) for i in range(SPT * len(ks))
            ]
        rstds = stage_x_rmsnorm(
            ld_a,
            HIDDEN,
            g_post,
            count=SPT,
            loaded=loaded,
            norm_loaded=(raw_collapse if attention_fp8_bf16 and HC > 1 else None),
        )
        stamp("router", tt, 2)
        for s_l in range_constexpr(SPT):
            if x_ok:
                x_s = rs0 + s_l
                x_rstd = rstds[s_l]
                a0, a1 = xa[s_l]
                v0, v1 = a0 * x_rstd * xg[0], a1 * x_rstd * xg[1]
                if const_expr(attention_fp8_bf16):
                    v0, v1 = (
                        bf16_round(a0 * xg[0] * x_rstd),
                        bf16_round(a1 * xg[1] * x_rstd),
                    )
                if const_expr(shared_bf16_block128):
                    put(mb("sx"), (x_s * HIDDEN + xk) // 2, bf16_pair(v0, v1))
                if const_expr(use_fp8_block128):
                    q0, q1, qs = quant_scaled(v0, v1)
                    w8 = (
                        fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
                        )
                        & 0xFFFF
                    )
                    w8n = xshfl(w8, 1)
                    if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    d0, d1 = fp8_roundtrip(q0, q1)
                    d0, d1 = d0 * qs, d1 * qs
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                elif const_expr(use_mxfp8_block32):
                    d0, d1, qs = quant_mxfp8(v0, v1)
                    w8 = (
                        fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, d0, d1, fx.Int32(0), False)
                        )
                        & 0xFFFF
                    )
                    w8n = xshfl(w8, 1)
                    if lane % 2 == 0:
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    if lane % 16 == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    d0, d1 = d0 * qs, d1 * qs
                else:
                    d0, d1 = bf16_round(v0), bf16_round(v1)
                    put(mb("xq"), (x_s * HIDDEN + xk) // 2, bf16_pair(d0, d1))
                bo.buffer_store(
                    fx.Vector.from_elements([d0, d1], fx.Float32),
                    rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
        gpu.barrier()
        acc = run_units(u_r, R_CPW, R_CPW, pre)
        fx.ptr_store(
            fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
        )
        gpu.barrier()
        stamp("router", tt, 3)
        if tid < ROUTER_TILE * SPT:
            r = tid % ROUTER_TILE
            n = tid // ROUTER_TILE
            logit = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                for f in range_constexpr(2):
                    m = f * ROUTER_TILE + r
                    logit = logit + lds_ld(
                        red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4
                    )
            score = sqrt_softplus(bf16_round(logit))
            put(mb("scores"), (rs0 + n) * N_EXPERTS + t * ROUTER_TILE + r, score)
            # bf16 gate logits, as ATOM's
        stamp("router", tt, 4)

    def dn_route(bs):
        """One wave selects one sample per round; inactive waves never publish."""
        for round_id in range_constexpr(DOWN_ROUTE_ROUNDS):
            sample = wave + round_id * WAVES
            if sample < S:
                e, w = route_topk(sample, bs=bs)
                if lane < MOE_SLOTS:
                    q = sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

    # ================================ 9. expert up/gate + SiLU
    # one 16-row group (8 gate + 8 up rows) per tile, all eight waves splitting K
    UG_NKC = HIDDEN // 64
    UG_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
    UG_W_BYTES = 2 * INTER * HIDDEN // (2 if use_mxfp4_weight else 1)
    UG_S_BYTES = (  # an MXFP4 bank's scales pad K / 32 to a multiple of 8 (pack_mxfp4_scales)
        2 * INTER * (-(-(HIDDEN // 32) // 8) * 8)
        if use_mxfp4_weight
        else 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4
    )
    SUG_S_BYTES = (
        2 * INTER // SCALE_BM * (HIDDEN // 128) * 4
    )  # the FP8 shared expert's scales

    def up_gate():
        """In its own scope: its unit lists must not join `down`'s scf.for-carried state."""
        # (tile, sample) items, unrolled at build time (the unit lists cannot be
        # scf.for-carried). An item past the end is masked by `live` (zero
        # num_records, no publishes), not skipped, so every barrier stays uniform;
        # every item must be covered or `down` polls its `mid` slots forever.
        UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
        XW = HIDDEN // (4 if use_fp8_block128 else 2)
        u0 = fx.Int32(start("ug"))
        grouped_input = S > WAVES
        shared_count = min(S, WAVES)

        def ug8_units(c, w_rg, w_ln, s_rg, e, sample, live=None, count=shared_count):
            nw = (
                None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
            )
            ns = (
                None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
            )
            rw = bo.create_buffer_resource_from_addr(
                w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
            )
            rs = bo.create_buffer_resource_from_addr(
                s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
            )
            sn = n_sel(count) if sample is None else fx.Int32(sample)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                unit = wave * UG8_UNITS + cc
                if const_expr(use_mxfp4_weight):
                    coefficients = None
                    units.append(
                        unit_mxfp4(
                            rw,
                            rs,
                            mx_rg(c),
                            unit,
                            HIDDEN,
                            sn * XW + unit * 64,
                            coefficients,
                            w_ln,
                        )
                    )
                    continue
                kc = unit * (2 if use_fp8_block128 else 1)
                nwc = 2 if use_fp8_block128 else 1
                wv = [
                    fx.Vector(
                        bo.buffer_load(
                            rw,
                            ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for j in range(nwc)
                ]
                sc = ld_f32(rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)

                if const_expr(use_fp8_block128):

                    def coefficient(sc=sc, kb=kc // 2, sn=sn):
                        return sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kb)

                    units.append(
                        ("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4)
                    )
                else:
                    units.append(("fp8", wv, sc, sn * XW + kc * 32 + (lane // 16) * 4))
            return units

        def ug8_units_sh(c, w_rg, w_ln, s_rg, live, count=shared_count):
            """The FP8 shared expert's units: every sample at once (lane column = sample)."""
            rw = bo.create_buffer_resource_from_addr(
                w_sug,
                num_records_bytes=live.select(
                    fx.Int32(2 * INTER * HIDDEN), fx.Int32(0)
                ),
            )
            rs = bo.create_buffer_resource_from_addr(
                s_sug, num_records_bytes=live.select(fx.Int32(SUG_S_BYTES), fx.Int32(0))
            )
            sn = n_sel(count)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                unit = wave * UG8_UNITS + cc
                coefficients = None
                units.append(
                    unit_fp8mx(
                        rw,
                        rs,
                        w_rg,
                        s_rg,
                        unit,
                        HIDDEN,
                        sn * XW + unit * 64,
                        coefficients,
                        w_ln,
                    )
                )
            return units

        def shared_units(c, w_rg, w_ln, s_rg, live, count=shared_count):
            if const_expr(SHARED_FP8):
                return ug8_units_sh(c, w_rg, w_ln, s_rg, live, count=count)
            return ug8_units(
                c, w_rg, w_ln, s_rg, fx.Int32(SHARED_EXPERT), None, live, count=count
            )

        def ug8_emit(c, slot, sample, shared, live, count=S):
            if tid < (count if shared else 1) * UG8 // 2:
                n = tid // (UG8 // 2)
                r = (tid % (UG8 // 2)) * 2
                g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                v0, v1 = (
                    lds_ld(outs, n * 16 + UG8 + r),
                    lds_ld(outs, n * 16 + UG8 + r + 1),
                )
                sn = n + sample if shared else fx.Int32(sample)
                sl = fx.Int32(0) if shared else slot
                if live:
                    if const_expr(shared and shared_bf16_block128):
                        g0, g1 = bf16_round(g0), bf16_round(g1)
                        v0, v1 = bf16_round(v0), bf16_round(v1)
                    m0, m1 = swiglu(g0, v0, swiglu_limit), swiglu(g1, v1, swiglu_limit)
                    if const_expr(shared and shared_bf16_block128):
                        m0, m1 = bf16_round(m0), bf16_round(m1)
                    put2(
                        mb("mid"),
                        (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r,
                        m0,
                        m1,
                    )
            if (c == 0) & (tid < count if shared else tid == 0):
                sn = tid + sample if shared else fx.Int32(sample)
                sl = fx.Int32(0) if shared else slot
                if live:
                    put(
                        mb("sel"),
                        sn * MOE_SLOTS + sl,
                        lds_ld(keys, sn * MOE_SLOTS + sl),
                    )
                    put(
                        mb("prob"),
                        sn * MOE_SLOTS + sl,
                        lds_ld(dnw, sn * MOE_SLOTS + sl),
                    )

        def ug8_tile(u):
            """Tile ``u`` (clamped; ``live`` masks a dead one): (live, uu, c, slot, has_sh, w_rg, w_ln, s_rg)."""
            live = u < N_UG_TASKS
            uu = fx.min(u, fx.Int32(N_UG_TASKS - 1))
            c = uu % UG_PER_SLOT
            has_sh = uu < UG_PER_SLOT
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), uu // UG_PER_SLOT)
            w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (INTER // 16) + c // 2
            return live, uu, c, slot, has_sh, w_rg, w_ln, s_rg

        # a shared-expert tile serves every sample at once (B columns); routed work is items
        N_UG_ITEMS = S * N_UG_TASKS
        UG_ITEMS = (N_UG_ITEMS + G - 1) // G

        def ug8_item(k):
            """This CTA's k-th routed item, clamped as ug8_tile: (live, uu, sample, c, ...)."""
            w = u0 + k * G
            live = w < N_UG_ITEMS
            ww = fx.min(w, fx.Int32(N_UG_ITEMS - 1))
            uu = ww % N_UG_TASKS
            sample = ww // N_UG_TASKS
            _l, _u, c, slot, _h, w_rg, w_ln, s_rg = ug8_tile(uu)
            return live, uu, sample, c, slot, w_rg, w_ln, s_rg

        live0, _uu0, c0, slot0, has_sh0, wr0, wl0, sr0 = ug8_tile(u0)
        if const_expr(not grouped_input):
            shared_pre = shared_units(c0, wr0, wl0, sr0, has_sh0 & live0)
        dn_route(load_bias())
        gpu.barrier()
        items = [ug8_item(0)]
        lv, _, sm, c_, sl, wr, wl, sr = items[0]
        if const_expr(not grouped_input):
            cur = ug8_units(
                c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv
            )
        if const_expr(grouped_input):
            # Local MFMA columns address the compact staged group; publications
            # retain global sample IDs. All CTAs cross the same barriers.
            for first_sample in range_constexpr(0, S, shared_count):
                count = min(shared_count, S - first_sample)
                stage_moe_input(
                    list(range(first_sample, first_sample + count)),
                    shared=shared_bf16_block128,
                )
                gpu.barrier()
                group_pre = shared_units(
                    c0, wr0, wl0, sr0, has_sh0 & live0, count=count
                )
                if has_sh0 & live0:
                    reduce_rows(
                        1,
                        mma_units([fx.Float32(0.0) for _ in range(4)], group_pre),
                        emit_out(16),
                        count=count,
                    )
                    gpu.barrier()
                    ug8_emit(c0, slot0, first_sample, True, live0, count=count)
                gpu.barrier()
        else:
            stage_moe_input(list(range(S)), shared=shared_bf16_block128)
            gpu.barrier()
            if has_sh0 & live0:
                reduce_rows(
                    1,
                    mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                    emit_out(16),
                )
                gpu.barrier()
                ug8_emit(c0, slot0, 0, True, live0)
        if const_expr(shared_bf16_block128 and not grouped_input):
            # Shared has consumed its own operand. Reuse the SAME LDS arena
            # for routed MXFP8/32; do not add another scheduler or launch.
            gpu.barrier()
            stage_moe_input(list(range(S)))
            gpu.barrier()
        if const_expr(grouped_input):
            # Do not keep a full hidden-K routed weight prefetch alive across
            # the shared groups' staging and MFMA work.
            cur = ug8_units(
                c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), 0, lv
            )
        for k in range_constexpr(UG_ITEMS):
            live, uu, sample, c, slot, _w_rg, _w_ln, _s_rg = items[k]
            stamp("ug", sample * N_UG_TASKS + uu, 0, pred=live)
            if const_expr(grouped_input):
                gpu.barrier()
                stage_moe_input([uniform(sample)])
                gpu.barrier()

            pre = cur
            if const_expr(k + 1 < UG_ITEMS):
                items.append(ug8_item(k + 1))
                lv, _, sm, c_, sl, wr, wl, sr = items[k + 1]
                cur = ug8_units(
                    c_,
                    wr,
                    wl,
                    sr,
                    uniform(lds_ld(keys, sm * MOE_SLOTS + sl)),
                    0 if grouped_input else sm,
                    lv,
                )
            reduce_rows(
                1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16)
            )
            gpu.barrier()
            ug8_emit(c, slot, sample, False, live)
            stamp("ug", sample * N_UG_TASKS + uu, 4, pred=live)

    up_gate()

    # =============== 10. expert down + route weighting + MoE TP reduce
    DN_NKC = INTER // 64
    # see dn_tile: every tile starts at row 0 of a group
    assert (
        DN_TILE % 16 == 0 and HIDDEN % DN_TILE == 0
    ), f"down tile {DN_TILE} must divide {HIDDEN} by 16s"
    DN_R = DN_TILE // 16
    DN_WPR = WAVES // DN_R
    DN_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
    DN_UNITS_PER_SLOT = INTER // DN_UNIT_K
    # q6 crosses a different K/wave boundary when two whole requests are
    # batched: preserve the qualified single-request partition and rounding.
    # Other shapes retain their existing distribution; no token is split q1.
    ROUTE_SAMPLES = down_route_samples(S, ctx["TOK"], routed_down_bf16)
    ROUTE_GROUPS = S // ROUTE_SAMPLES
    # routed units over (sample, slot, K chunk), then any FP8 shared-expert units
    DN_SLOTS = TOP_K if SHARED_FP8 else MOE_SLOTS
    DN_NU = S * DN_SLOTS * DN_UNITS_PER_SLOT
    DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
    # the shared expert's units: one per K chunk, every sample in its own B column
    DN_SH_NU = DN_UNITS_PER_SLOT if SHARED_FP8 else 0
    DN_SH_UPW = (DN_SH_NU + DN_WPR - 1) // DN_WPR
    DN_CPW = DN_UPW + DN_SH_UPW
    DN_BLK = S * MOE_SLOTS * INTER // 128
    DN_W_BYTES = HIDDEN * INTER // (2 if use_mxfp4_weight else 1)
    DN_S_BYTES = (
        HIDDEN * (-(-(INTER // 32) // 8) * 8)
        if use_mxfp4_weight
        else HIDDEN // SCALE_BM * (INTER // 128) * 4
    )
    SDN_S_BYTES = (
        HIDDEN // SCALE_BM * (INTER // 128) * 4
    )  # the FP8 shared expert's scales
    DN_BATCH = 9  # 128-k chunks per wave in flight / prefetched before the mid wait
    for t in range(start("down"), N_DN_TILES, G):
        t = fx.Int32(t)
        stamp("down", t, 0)
        gpu.barrier()
        gu = wave // DN_WPR
        dn_rg = t * DN_TILE // 16
        dn_off = t * DN_TILE % 16
        # rows outside the tile load their lane ^ 8 twin (same lines) and are dropped
        dn_lr = gu * 16 + lane % 16 - dn_off
        dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

        def dn_coefficients(q, s_q, slot_q):
            """A unit's factor: its route weight, in its sample's column only. A VGPR LDS
            broadcast, not a readfirstlane (SGPR copies per unit cause spills)."""

            def coefficient():
                return (lane % 16 == s_q).select(
                    lds_ld(dnw, s_q * MOE_SLOTS + slot_q), fx.Float32(0.0)
                )

            return coefficient

        def u_dn_sh(cc):
            qs = (wave % DN_WPR) * DN_SH_UPW + cc
            live = qs < DN_SH_NU
            kc = fx.min(qs, DN_SH_NU - 1)
            # B column n_sel() reads its sample's slot-0 mid; the route weight is 1
            q = n_sel() * MOE_SLOTS * DN_UNITS_PER_SLOT + kc
            masked = DN_SH_NU % DN_WPR != 0
            wb = bo.create_buffer_resource_from_addr(
                w_sdn,
                num_records_bytes=(
                    live.select(fx.Int32(HIDDEN * INTER), fx.Int32(0))
                    if masked
                    else None
                ),
            )
            sb = bo.create_buffer_resource_from_addr(
                s_sdn,
                num_records_bytes=(
                    live.select(fx.Int32(SDN_S_BYTES), fx.Int32(0)) if masked else None
                ),
            )
            rg = dn_rg + gu
            return unit_fp8mx(wb, sb, rg, rg, kc, INTER, q * 64, None, dn_ln)

        def u_dn(cc):
            if const_expr(cc >= DN_UPW):
                return u_dn_sh(cc - DN_UPW)
            qu = (wave % DN_WPR) * DN_UPW + cc
            live = qu < DN_NU
            r = fx.min(qu, DN_NU - 1)
            s_q = r // (DN_SLOTS * DN_UNITS_PER_SLOT)
            slot_q = (r // DN_UNITS_PER_SLOT) % DN_SLOTS + (MOE_SLOTS - DN_SLOTS)
            kc = r % DN_UNITS_PER_SLOT
            q = (s_q * MOE_SLOTS + slot_q) * DN_UNITS_PER_SLOT + kc
            e = uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
            wb = bo.create_buffer_resource_from_addr(
                w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                num_records_bytes=(
                    None
                    if DN_NU % DN_WPR == 0
                    else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                ),
            )
            sb = bo.create_buffer_resource_from_addr(
                s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                num_records_bytes=(
                    None
                    if DN_NU % DN_WPR == 0
                    else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                ),
            )

            if const_expr(use_mxfp4_weight):
                coefficients = dn_coefficients(q, s_q, slot_q)
                return unit_mxfp4(
                    wb,
                    sb,
                    dn_rg + gu,
                    kc,
                    INTER,
                    q * 64,
                    coefficients,
                    dn_ln,
                )

            kc64 = kc * (2 if use_fp8_block128 else 1)
            if const_expr(use_fp8_block128):

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(
                        uniform_f32(lds_ld(misc, q)), fx.Float32(0.0)
                    )

                return unit_f8f8(
                    wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, q * 32, coef, dn_ln
                )

            def coef():
                return (lane % 16 == s_q).select(
                    uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), 0.0
                )

            return unit_fp8(
                wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, 128, q * 32, coef, dn_ln
            )

        def u_dn_route(cc, route, first_sample=0):
            # Keep an entire route's K sum separate until reduce_rows. Rounding
            # each K chunk, or a mixed-route accumulator, is NOT per-route BF16.
            cpw = -(-(ROUTE_SAMPLES * DN_UNITS_PER_SLOT) // DN_WPR)
            qu = (wave % DN_WPR) * cpw + cc
            live = qu < ROUTE_SAMPLES * DN_UNITS_PER_SLOT
            r = fx.min(qu, ROUTE_SAMPLES * DN_UNITS_PER_SLOT - 1)
            sample = first_sample + r // DN_UNITS_PER_SLOT
            kc = r % DN_UNITS_PER_SLOT
            q = (sample * MOE_SLOTS + route) * DN_UNITS_PER_SLOT + kc
            e = uniform(lds_ld(keys, sample * MOE_SLOTS + route))
            wb = bo.create_buffer_resource_from_addr(
                w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                num_records_bytes=live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)),
            )
            sb = bo.create_buffer_resource_from_addr(
                s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                num_records_bytes=live.select(fx.Int32(DN_S_BYTES), fx.Int32(0)),
            )

            # Route weight is applied ONCE after the cross-wave full-K sum.
            # The B column mask prevents another sample's LDS mid being read.
            def mask():
                return (lane % 16 == sample).select(fx.Float32(1.0), fx.Float32(0.0))

            return unit_mxfp4(wb, sb, dn_rg + gu, kc, INTER, q * 64, mask, dn_ln)

        # The experts are known: stream the first down weights while UG finishes.
        if const_expr(routed_down_bf16):
            route_cpw = -(-(ROUTE_SAMPLES * DN_UNITS_PER_SLOT) // DN_WPR)
            pre = [u_dn_route(cc, 1) for cc in range(min(DN_BATCH, route_cpw))]
        else:
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_CPW))]
        hint_wait(
            N_UG,
            lambda k: (
                mb("mid"),
                (
                    k // (MOE_SLOTS * N_UG_PER_SLOT) * MOE_SLOTS
                    + (k // N_UG_PER_SLOT) % MOE_SLOTS
                )
                * INTER
                + (k % N_UG_PER_SLOT) * UG_TILE
                + UG_TILE
                - 1,
            ),
            mark=("down", t),
        )
        if const_expr(S <= WAVES):
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
        stamp("down", t, 2)
        for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
            blk = wave + b * WAVES
            if const_expr(S > WAVES):
                # Bound the live payload list, not the down arena or K sum.
                # The same whole 128-value group reaches the same LDS offset.
                mid0, mid1 = get2_many(
                    [(mb("mid"), fx.min(blk, DN_BLK - 1) * 128 + lane * 2)]
                )[0]
            else:
                mid0, mid1 = mids[b]
            if blk < DN_BLK:
                if const_expr(use_fp8_block128):
                    q0, q1, qs = quant_scaled(mid0, mid1)
                    st_f8(blk * 128 + lane * 2, q0, q1)
                    if lane == 0:
                        lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
                elif const_expr(use_mxfp8_block32):
                    if const_expr(shared_bf16_block128):
                        # blk is wave-uniform, and each expert has INTER/128
                        # whole groups. Only shared slot zero changes format.
                        if (blk // (INTER // 128)) % MOE_SLOTS == 0:
                            sh0, sh1, sh_scale = quant_mxfp8(mid0, mid1, 128)
                            sh_payload = bf16_pair(sh0 * sh_scale, sh1 * sh_scale)
                            lds_st(xs, blk * 64 + lane, sh_payload)
                            sh_debug0, sh_debug1 = bf2_f32(sh_payload.bitcast(fx.Int32))
                            sh_sample = blk // ((INTER // 128) * MOE_SLOTS)
                            sh_channel = (blk % (INTER // 128)) * 128 + lane * 2
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [sh_debug0, sh_debug1], fx.Float32
                                ),
                                rsrc(mb("xqd")),
                                2 * S * HIDDEN + sh_sample * INTER + sh_channel,
                            )
                        else:
                            rt0, rt1, rt_scale = quant_mxfp8(mid0, mid1)
                            lds_st(
                                xs,
                                blk * 64 + lane,
                                bf16_pair(rt0 * rt_scale, rt1 * rt_scale),
                            )
                    else:
                        d0, d1, qs = quant_mxfp8(mid0, mid1)
                        lds_st(xs, blk * 64 + lane, bf16_pair(d0 * qs, d1 * qs))
                else:
                    lds_st(xs, blk * 64 + lane, bf16_pair(mid0, mid1))
        gpu.barrier()

        def emit_dn(rl, n, v):
            if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                lds_st(outs, n * DN_TILE + rl - dn_off, v)

        if const_expr(routed_down_bf16):
            for group in range_constexpr(ROUTE_GROUPS):
                first_sample = group * ROUTE_SAMPLES
                for route in range_constexpr(1, TOP_K + 1):

                    def make_route(cc, route=route):
                        return u_dn_route(cc, route, first_sample)

                    acc = run_units(
                        make_route,
                        route_cpw,
                        DN_BATCH,
                        pre if const_expr(route == 1 and group == 0) else None,
                    )

                    def emit_route(rl, n, v, route=route):
                        if (
                            (rl >= dn_off)
                            & (rl < dn_off + DN_TILE)
                            & (n >= first_sample)
                            & (n < first_sample + ROUTE_SAMPLES)
                        ):
                            offset = n * DN_TILE + rl - dn_off
                            weighted = bf16_round(
                                v * lds_ld(dnw, n * MOE_SLOTS + route)
                            )
                            total = (
                                weighted
                                if const_expr(route == 1)
                                else lds_ld(outs, offset) + weighted
                            )
                            # Ordinary controlled route-reduce returns BF16 before
                            # shared add. Natural atomic order remains a separate gate.
                            lds_st(
                                outs,
                                offset,
                                (
                                    bf16_round(total)
                                    if const_expr(route == TOP_K)
                                    else total
                                ),
                            )

                    reduce_rows(DN_R, acc, emit_route)
                    gpu.barrier()
            shared_acc = run_units(u_dn_sh, DN_SH_UPW, DN_BATCH)

            def emit_shared(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    offset = n * DN_TILE + rl - dn_off
                    local = bf16_round(lds_ld(outs, offset) + bf16_round(v))
                    lds_st(outs, offset, local)

            reduce_rows(DN_R, shared_acc, emit_shared)
        else:
            acc = run_units(u_dn, DN_CPW, DN_BATCH, pre)
            reduce_rows(DN_R, acc, emit_dn)
        stamp("down", t, 3)
        gpu.barrier()

        def store_x(s, row, v0, v1):
            bo.buffer_store(
                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                rsrc(x_out),
                s * HIDDEN + row,
            )

        if const_expr(HC > 1):
            hc_stage_coef(1)

            def store_x_hc(s_, k, r_, o0, o1):
                bo.buffer_store(
                    fx.Vector.from_elements([o0, o1], fx.Float32).to(fx.BFloat16),
                    rsrc(x_out),
                    (s_ * HC + k) * HIDDEN + r_,
                )

            def reduce_then_hc(s, row, v0, v1):
                return hc_post(
                    s,
                    row,
                    v0,
                    v1,
                    lambda s_, j, r_: get(mb("a"), ((s_ * HC + j) * HIDDEN + r_) // 2),
                    store_x_hc,
                )

            peer_reduce(
                "ffn",
                t,
                None,
                reduce_then_hc,
                tile=DN_TILE,
            )
        else:
            peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
        gpu.barrier()
        stamp("down", t, 4)
    return {}
