# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse prefill attention with fp16 compute (fp16 K / V, P and MFMA), the layout and epilogues of
``bf16/kernel.py``."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..flydsl_utils import (
    _dequant_f16,
    _exp2,
    _global_i32_ptr,
    _load_page,
    _maxf_nn,
    _mfma_f16,
    _page_offsets,
    _pk_bf16,
    _pk_f16,
    _q_f16,
    _store_out,
    _xlane4,
)

BLK = 128  # tokens per page (one sparse block)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every page
KPW = BLK // NW
TOK_PER_WG__4 = 4  # query tokens per workgroup
TOPK__16 = 16  # selected pages per token
MAXU__64 = TOK_PER_WG__4 * TOPK__16  # union capacity (one lane per (token, slot))
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per page
Q_ROW = D * 2 // 16 + 1  # 16 B LDS tiles per (token, head) row of Q^T, padded by one


def _q_stage(v, q_f16):
    """A Q dword for the MFMA operand: bf16 -> fp16, or as given when Q already is fp16 (``q_f16``)."""
    return Vec(v) if q_f16 else _q_f16(v)


def compile_nvfp4_prefill_fp16(
    pack: bool = False,
    rounds: int = 2,
    quant: bool = False,
    q_f16: bool = False,
    sched: bool = True,
):
    """The four-query launcher.

    Args:
        pack: write the bf16 output's MXFP8 quant as sequence-parallel all-to-all rows (``reference.sp_pack_rows``)
            to ``arg_pack`` instead of the bf16 output.
        rounds: the combine's rounds; 2 halves its LDS so two workgroups share a CU, 1 finalizes all tokens at once
            (grids of at most one workgroup per CU).
        quant: row-major MXFP8 codes to ``arg_out``, AITER-preshuffled scales to ``arg_pack``.
        q_f16: Q is fp16 already; the bf16 -> fp16 staging conversion is skipped.
        sched: the scheduling ports, bitwise equal to ``sched=False``: the current page id in the loop state (one
            union walk per page instead of two), the independent prologue loads before the Q^T staging writes, and
            the running max clamped with max(m, -8e37) instead of an == -inf select.
    """
    assert rounds in (1, 2)
    assert not (pack and quant)
    tpr = TOK_PER_WG__4 // rounds  # tokens per combine round
    wpe = rounds  # waves per SIMD
    name = f"nvfp4_f16mfma_sparse_prefill_gqa16_d128_p128_top16_tq{TOK_PER_WG__4}_ksplit"
    if sched:
        name += "_sched"
    if q_f16:
        name += "_qf16"
    if pack:
        name += "_pack"
    if quant:
        name += "_quant"
    if rounds == 1:
        name += "_r1"

    @fx.struct
    class Shared:
        # 16 B tiles: Q^T during the page loop, then the partial O per (wave, token of the round, d-tile, lane);
        # (max, sum) per (wave, token, lane)
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG__4 * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] bf16
        arg_cache: fx.Int64,  # [pages, 2 Hkv, 128, 72] uint8
        arg_topk: fx.Int64,  # [Hkv, T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_pos: fx.Int64,  # [T] int32
        arg_out: fx.Int64,  # [T, H, 128] bf16 (quant: the codes)
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_pack: fx.Int64,  # pack: [T, H * 132] uint8 packed rows (else unused)
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,  # topk stride over KV heads (elements)
        i32_bt_s: fx.Int32,
        i32_page_byte: fx.Int32,  # cache page stride (bytes)
        f32_c: fx.Float32,  # sm_scale * log2(e)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        bidx, kvh = fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk = _global_i32_ptr(arg_topk)
        bt = _global_i32_ptr(arg_bt)
        req_p = _global_i32_ptr(arg_req)
        pos_p = _global_i32_ptr(arg_pos)
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        c = f32_c * fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_ks))[0])
        v_scale = fx.Float32(fx.inttoptr(f32p, fx.Int64(arg_vs))[0])
        neg_inf = fx.Float32(float("-inf"))
        topk_base__0 = kvh * i32_topk_hs

        lds = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.o.ptr),
                fx.make_layout(NW * tpr * NDT * 64 * 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2)
        )
        # sched: every independent prologue load goes before the Q staging writes; the values are used below in serial
        # order
        if const_expr(sched):
            pre_tc, pre_pos, pre_req = [], [], []
            for t in range_constexpr(TOK_PER_WG__4):
                tok_t = bidx * TOK_PER_WG__4 + t
                tc_t = (tok_t < i32_T).select(tok_t, fx.Int32(0))
                pre_tc.append(tc_t)
                pre_pos.append(fx.Int32(pos_p[tc_t]))
                pre_req.append(fx.Int32(req_p[tc_t]))
            lane_tok = bidx * TOK_PER_WG__4 + lane // TOPK__16
            lane_tok_ok = lane_tok < i32_T
            lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
            pre_blk = fx.Int32(
                topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
            )
            pre_lane_req = fx.Int32(req_p[lane_tok_clamped])
        # rounds 2: Q^T in LDS (not 64 registers), thread (token, head, quarter) stages 64 B
        if const_expr(rounds == 2):
            tq, hq, qq = tx // 64, (tx // 4) % HG, tx % 4
            tok_q = bidx * TOK_PER_WG__4 + tq
            tok_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
            q_src = (tok_q * i32_H + kvh * HG + hq) * (D // 2) + qq * 16
            if const_expr(sched):
                pre_q4 = [
                    buffer_ops.buffer_load(
                        qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                    )
                    for i in range_constexpr(4)
                ]
        if const_expr(sched):
            # dependent page-table gathers (need the pos / req / topk values)
            pre_diag = [
                fx.Int32(bt[pre_req[t] * i32_bt_s + pre_pos[t] // BLK])
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            lane_blk_ok = lane_tok_ok & (pre_blk >= 0)
            pre_lane_page = fx.Int32(
                bt[pre_lane_req * i32_bt_s + lane_blk_ok.select(pre_blk, fx.Int32(0))]
            )
        if const_expr(rounds == 2):
            for i in range_constexpr(4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                if const_expr(sched):
                    q4 = pre_q4[i]
                else:
                    q4 = buffer_ops.buffer_load(
                        qr, q_src + 4 * i, vec_width=4, dtype=fx.Int32
                    )
                r.store(_q_stage(q4, q_f16))
                q_tile = (tq * HG + hq) * Q_ROW + qq * 4 + i
                fx.copy(lds_atom, r, fx.slice(lds, (None, q_tile)))

        def q_frags(t):
            """Token t's Q^T: k-step ks's B operand (dims 32 g + 8 ks.. of head n)."""
            if const_expr(rounds == 1):
                return q_regs[t]
            out = []
            for ks in range_constexpr(NKS):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                q_tile = (t * HG + n) * Q_ROW + g * 4 + ks
                fx.copy(lds_atom, fx.slice(lds, (None, q_tile)), r)
                out.append(Vec(r.load()))
            return out

        # per token: its own block's page, the causal limit on this wave's key offsets (rounds 1: Q^T in registers)
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG__4):
            if const_expr(sched):
                tc = pre_tc[t]
                pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_pos[t]))
                diag.append(fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_diag[t])))
            else:
                tok = bidx * TOK_PER_WG__4 + t
                ok = tok < i32_T
                tc = ok.select(tok, fx.Int32(0))
                pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(pos_p[tc])))
                req = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(req_p[tc])))
                diag.append(
                    fx.Int32(
                        fx.rocdl.readfirstlane(
                            T.i32, fx.Int32(bt[req * i32_bt_s + pos // BLK])
                        )
                    )
                )
            lim.append(pos % BLK - w * KPW)
            if const_expr(rounds == 1):
                q_row = (tc * i32_H + kvh * HG + n) * (D // 2) + g * 16
                q_regs.append(
                    [
                        _q_stage(
                            buffer_ops.buffer_load(
                                qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32
                            ),
                            q_f16,
                        )
                        for ks in range_constexpr(NKS)
                    ]
                )

        offs = _page_offsets(i32_hkv, kvh, w, n, g)

        def load_page(page):
            return _load_page(page, arg_cache, i32_page_byte, *offs)

        # lane = (token lane // 16, slot lane % 16): the page it selected (-1: none)
        if const_expr(sched):
            lane_page = lane_blk_ok.select(pre_lane_page, fx.Int32(-1))
        else:
            lane_tok = bidx * TOK_PER_WG__4 + lane // TOPK__16
            lane_tok_ok = lane_tok < i32_T
            lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
            blk = fx.Int32(
                topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
            )
            lane_blk_ok = lane_tok_ok & (blk >= 0)
            lane_page = fx.Int32(
                bt[
                    fx.Int32(req_p[lane_tok_clamped]) * i32_bt_s
                    + lane_blk_ok.select(blk, fx.Int32(0))
                ]
            )
            lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))
        # the union's first page is the lowest valid lane's: load it before the dedup, which runs under its latency
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_page >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest union lane left in ``mask`` (the first page again when it is empty)."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        if const_expr(sched):
            # the lowest valid lane is always a first occurrence; its page is carried in the loop state
            page0 = page_at(valid)
            raw0 = load_page(page0)
        else:
            raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        # union: the first lane of each page, in lane order
        n_dup = fx.Int32(0)  # 0: a first occurrence
        for j in range_constexpr(MAXU__64 - 1):
            pj = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, j))
            n_dup = n_dup + ((lane_page == pj) & (lane > j)).select(
                fx.Int32(1), fx.Int32(0)
            )
        umask = fx.Int64(
            fx.rocdl.ballot(T.i64, (lane_page >= 0) & (n_dup == 0))
        )
        n_u = fx.Int32(fx.math.ctpop(umask).to(fx.Int32))  # distinct pages

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # page offset of S^T tile st, row 4 g + e within this wave's 32 keys
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        if const_expr(rounds == 2):
            fx.gpu.barrier()  # Q^T staged
        # iteration u loads page u + 1 and dequantizes page u; sched carries the current page id in the loop state
        base = 2 if sched else 1  # loop-state prefix: [rem] (+ [page] if sched)
        init = [umask & (umask - fx.Int64(1)), page0] if sched else [umask]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG__4)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG__4)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG__4 * NDT)]
        init += raw0
        n_st = base + 2 * TOK_PER_WG__4 + TOK_PER_WG__4 * NDT
        results = init
        for u, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index(n_u),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            rem = fx.Int64(st_[0])
            if const_expr(sched):
                page = fx.Int32(st_[1])
            m = [fx.Float32(st_[base + t]) for t in range_constexpr(TOK_PER_WG__4)]
            l_run = [
                fx.Float32(st_[base + TOK_PER_WG__4 + t])
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            o = [
                [
                    Vec(st_[base + 2 * TOK_PER_WG__4 + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG__4)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            if const_expr(sched):
                page_next = page_at(rem)
                rem = rem & (rem - fx.Int64(1))
                raw_next = load_page(page_next)
            else:
                page = page_at(rem)
                rem = rem & (rem - fx.Int64(1))
                raw_next = load_page(page_at(rem))
            kf, vf = _dequant_f16(raw)
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            for t in range_constexpr(TOK_PER_WG__4):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if (hits.shrui(fx.Int64(TOPK__16 * t)) & fx.Int64(0xFFFF)) != fx.Int64(
                    0
                ):
                    qf = q_frags(t)
                    a = []
                    for st in range_constexpr(2):
                        acc = zero4
                        for ks in range_constexpr(NKS):
                            acc = _mfma_f16(kf[st][ks], qf[ks], acc)
                        a += [acc[e] for e in range_constexpr(4)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
                            for i in range_constexpr(8)
                        ]
                    mx = a[0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf_nn(mx, a[i])
                    # the running max is kept in scaled units: max(a) * c == max(a * c) (c > 0)
                    m_new = _maxf_nn(m_t, _xlane4(mx, _maxf_nn) * c)
                    if const_expr(sched):
                        # the same results as the -inf select: an all-masked page gives p = alpha = 0 either way,
                        # and a finite scaled score is > -8e37
                        m_use = _maxf_nn(m_new, fx.Float32(-8.0e37))
                    else:
                        m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
                    alpha = _exp2(m_t - m_use)
                    p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t * alpha + ps
                    pf = Vec.from_elements(
                        [_pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)],
                        fx.Int32,
                    )
                    # alpha is exactly 1 unless some row's max grew: rescale only then
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [_mfma_f16(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = (
                ([rem, page_next] if sched else [rem])
                + m
                + l_run
                + [x for token_out in o for x in token_out]
                + raw_next
            )
            results = yield [fx.as_ir_value(x) for x in state]

        # combine the waves' partials through LDS, tpr tokens per round; wave w finalizes token w in round w // tpr
        ml = smem.ml.ptr
        if const_expr(rounds == 2):
            fx.gpu.barrier()
        for t in range_constexpr(TOK_PER_WG__4):
            ml[((w * TOK_PER_WG__4 + t) * 2) * 64 + lane] = fx.Float32(
                results[base + t]
            )
            ml[((w * TOK_PER_WG__4 + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[base + TOK_PER_WG__4 + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = base + 2 * TOK_PER_WG__4 + (tpr * rnd + tt) * NDT + dt
                    r.store(Vec(results[ri]).bitcast(fx.Int32))
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(lds, (None, ((w * tpr + tt) * NDT + dt) * 64 + lane)),
                    )
            fx.gpu.barrier()
            if w // tpr == rnd:
                t = w
                ms = [
                    fx.Float32(ml[((ww * TOK_PER_WG__4 + t) * 2) * 64 + lane])
                    for ww in range_constexpr(NW)
                ]
                big = ms[0]
                for ww in range_constexpr(1, NW):
                    big = _maxf_nn(big, ms[ww])
                big = (big == neg_inf).select(fx.Float32(0.0), big)
                wt = [
                    fx.Float32(fx.math.exp2(ms[ww] - big)) for ww in range_constexpr(NW)
                ]
                l_tot = fx.Float32(0.0)
                for ww in range_constexpr(NW):
                    l_w = ml[((ww * TOK_PER_WG__4 + t) * 2 + 1) * 64 + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = _xlane4(l_tot, lambda a, b: a + b)
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                acc = [zero4 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        o_tile = ((ww * tpr + t - tpr * rnd) * NDT + dt) * 64 + lane
                        fx.copy(lds_atom, fx.slice(lds, (None, o_tile)), r)
                        acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
                # lane (n, g) holds dims 8 (4 g + e) + dt of head n: dims [32 g, 32 g + 32) in order
                tok = bidx * TOK_PER_WG__4 + t
                words = [
                    [
                        _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                        for j in range_constexpr(4)
                    ]
                    for e in range_constexpr(4)
                ]
                _store_out(words, tok, kvh, n, g, i32_T, i32_H, arg_out, arg_pack, pack, quant)
            fx.gpu.barrier()

    @flyc.jit
    def launch_fp16(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_pos: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_pack: fx.Int64,
        i32_T: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_cache,
            arg_topk,
            arg_bt,
            arg_req,
            arg_pos,
            arg_out,
            arg_ks,
            arg_vs,
            arg_pack,
            i32_T,
            i32_H,
            i32_hkv,
            i32_topk_hs,
            i32_bt_s,
            i32_page_b,
            f32_c,
            # rounds 2: two workgroups per CU (256 registers, 72 KB of LDS)
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=((i32_T + TOK_PER_WG__4 - 1) // TOK_PER_WG__4, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_fp16.kernel_name = name
    return launch_fp16
