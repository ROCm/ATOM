# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-query fp16 attention with overlapped query and page-table loads, fused MXFP8 output."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .flydsl_utils import (
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
    _v_e4m3x2_f16,
    _xlane4,
)

BLK = 128  # tokens per page (one sparse block)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every page
KPW = BLK // NW
TOK_PER_WG = 1  # query tokens per workgroup
TOPK__16 = 16  # selected pages per token
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per page


def _q_stage(v, q_f16):
    """A Q dword for the MFMA operand: bf16 -> fp16, or as given when Q already is fp16 (``q_f16``)."""
    return Vec(v) if q_f16 else _q_f16(v)


def compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_qpre(
    q_f16: bool = False
):
    """One-query fp16 attention with fused row-major MXFP8 output."""
    pack, quant, rounds = False, True, 1
    tpr = TOK_PER_WG // rounds  # tokens per combine round
    wpe = 2  # waves per SIMD
    name = "nvfp4_f16mfma_sparse_gqa16_d128_p128_top16_tq1_qpre"
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
        # Each query combines its wave partials in wave order.
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG * 2 * 64]

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
        tc = (bidx < i32_T).select(bidx, fx.Int32(0))
        pre_pos = fx.Int32(pos_p[tc])
        pre_req = fx.Int32(req_p[tc])
        lane_tok = bidx + lane // TOPK__16
        lane_tok_ok = (lane_tok < i32_T) & (lane < 16)
        lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
        pre_blk = fx.Int32(topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16])
        pre_lane_req = fx.Int32(req_p[lane_tok_clamped])
        pre_lane_pos = fx.Int32(pos_p[lane_tok_clamped])
        q_row = (tc * i32_H + kvh * HG + n) * (D // 2) + g * 16
        pre_q4 = [buffer_ops.buffer_load(qr, q_row + ks * 4, vec_width=4, dtype=fx.Int32)
                  for ks in range_constexpr(NKS)]
        pre_diag = fx.Int32(bt[pre_req * i32_bt_s + pre_pos // BLK])
        lane_blk_ok = lane_tok_ok & (pre_blk >= 0) & (pre_blk <= pre_lane_pos // BLK)
        pre_lane_page = fx.Int32(bt[pre_lane_req * i32_bt_s + lane_blk_ok.select(pre_blk, fx.Int32(0))])
        pos = fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_pos))
        diag = [fx.Int32(fx.rocdl.readfirstlane(T.i32, pre_diag))]
        lim = [pos % BLK - w * KPW]
        offs = _page_offsets(i32_hkv, kvh, w, n, g)

        def load_page(page):
            return _load_page(page, arg_cache, i32_page_byte, *offs)

        lane_page = lane_blk_ok.select(pre_lane_page, fx.Int32(-1))
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

        page0 = page_at(valid)
        raw0 = load_page(page0)
        q_regs = [[_q_stage(v, q_f16) for v in pre_q4]]
        fx.rocdl.sched_barrier(0)
        umask = valid

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # page offset of S^T tile st, row 4 g + e within this wave's 32 keys
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        # Carry the prefetched page and drop all its duplicate picks in the loop.
        init = [umask, page0]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG * NDT)]
        init += raw0
        n_st = 2 + 2 * TOK_PER_WG + TOK_PER_WG * NDT
        results = init
        rem_test = umask
        while rem_test != fx.Int64(0):
            st_ = results
            rem = fx.Int64(st_[0])
            m = [fx.Float32(st_[2 + t]) for t in range_constexpr(TOK_PER_WG)]
            l_run = [
                fx.Float32(st_[2 + TOK_PER_WG + t])
                for t in range_constexpr(TOK_PER_WG)
            ]
            o = [
                [
                    Vec(st_[2 + 2 * TOK_PER_WG + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            page = fx.Int32(st_[1])
            hits = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            rem = rem & ~hits
            kf, vf = _dequant_f16(raw, _v_e4m3x2_f16)
            fx.rocdl.sched_barrier(0)
            page_next = page_at(rem)
            raw_next = load_page(page_next)
            # keep the prefetch here: LLVM otherwise sinks these loads past the token's branches to the latch, so every
            # page waits a full load latency
            _llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
            for t in range_constexpr(TOK_PER_WG):
                m_t, l_t, o_t = m[t], l_run[t], o[t]
                if (hits.shrui(fx.Int64(TOPK__16 * t)) & fx.Int64(0xFFFF)) != fx.Int64(
                    0
                ):
                    qf = q_regs[t]
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
                    m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
                    alpha = _exp2(m_t - m_use)
                    p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t * alpha + ps
                    pf = Vec.from_elements(
                        [_pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)], fx.Int32,
                    )
                    # alpha is exactly 1 unless some row's max grew: rescale only then
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [
                        _mfma_f16(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)
                    ]
                    m_t = m_new
                m[t], l_run[t], o[t] = m_t, l_t, o_t
            state = [rem, page_next] + m + l_run + [x for ot in o for x in ot] + raw_next
            results = [fx.as_ir_value(x) for x in state]
            rem_test = rem

        # combine the waves' partials through LDS, tpr tokens per round; wave w finalizes token w in round w // tpr
        ml = smem.ml.ptr
        for t in range_constexpr(TOK_PER_WG):
            ml[((w * TOK_PER_WG + t) * 2) * 64 + lane] = fx.Float32(results[2 + t])
            ml[((w * TOK_PER_WG + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[2 + TOK_PER_WG + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = 2 + 2 * TOK_PER_WG + (tpr * rnd + tt) * NDT + dt
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
                    fx.Float32(ml[((ww * TOK_PER_WG + t) * 2) * 64 + lane])
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
                    l_w = ml[((ww * TOK_PER_WG + t) * 2 + 1) * 64 + lane]
                    l_tot = l_tot + fx.Float32(l_w) * wt[ww]
                l_tot = _xlane4(l_tot, lambda a, b: a + b)
                inv = (l_tot > fx.Float32(0.0)).select(
                    v_scale / l_tot, fx.Float32(0.0)
                )
                acc = [zero4 for _ in range_constexpr(NDT)]
                for ww in range_constexpr(NW):
                    wv = Vec.filled(4, wt[ww], fx.Float32)
                    for dt in range_constexpr(NDT):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        o_tile = ((ww * tpr + t - tpr * rnd) * NDT + dt) * 64 + lane
                        fx.copy(lds_atom, fx.slice(lds, (None, o_tile)), r)
                        acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
                # lane (n, g) holds dims 8 (4 g + e) + dt of head n: dims [32 g, 32 g + 32) in order
                tok = bidx * TOK_PER_WG + t
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
    def launch_f16mfma_tq1_qpre(
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
            value_attrs={
                "rocdl.waves_per_eu": wpe,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=((i32_T + TOK_PER_WG - 1) // TOK_PER_WG, i32_hkv, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_f16mfma_tq1_qpre.kernel_name = name
    return launch_f16mfma_tq1_qpre
