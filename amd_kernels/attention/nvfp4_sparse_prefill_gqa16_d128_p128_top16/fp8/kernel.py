# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse prefill attention with e4m3 compute and the chunk-P rule (``reference.fp8_chunk_prefill_reference``),
two queries per workgroup, Q in registers (gfx950; opt-in)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..flydsl_utils import (
    _dequant_fp8,
    _exp2,
    _global_i32_ptr,
    _join4,
    _load_page,
    _maxf_nn,
    _mfma128,
    _mfma_fp8,
    _pack_fp8,
    _page_offsets,
    _pk_bf16,
    _store_out,
    _xlane4,
)

BLK = 128  # tokens per page (one sparse block)
D = 128
HG = 16  # query heads per KV head = the MFMA N dimension
NW = 4  # waves per workgroup; wave w owns keys [KPW w, KPW w + KPW) of every page
KPW = BLK // NW
TOK_PER_WG = 2  # query tokens per workgroup
assert TOK_PER_WG == 2
TOPK__16 = 16  # selected pages per token
NDT = D // 16  # d-tiles of O^T
NKS = D // 32  # k-steps of S^T
RAW = 2 * 4 + 1 + 8 + 2  # NVFP4 dwords a lane loads per page
M_FLOOR = -8.0e37  # Finite initial references remove the per-chunk -inf-to-zero select.
REF_SPAN = 32.0  # a P chunk's reference is >= the running max - REF_SPAN (decode's)


def compile_nvfp4_prefill_fp8(pack=False, rounds=2, quant=False, q_bf16=False):
    """Two-query union walk with register Q and the chunk-P rule."""
    tpr, wpe = TOK_PER_WG // rounds, 2
    assert rounds in (1, 2)
    name = "nvfp4_fp8mfma_sparse_gqa16_d128_p128_top16_tq2_walk"
    name += ("_r1" if rounds == 1 else "")
    name += ("_pack" if pack else "_quant" if quant else "") + ("_qbf16" if q_bf16 else "")

    @fx.struct
    class Shared:
        # the NVFP4 -> E4M3 table at LDS offset 0, so a lookup address is the scale byte x 8 (one SDWA shift); LLVM
        # places each field by alignment, then size (both descending), so the table's larger alignment puts it first
        lut: fx.Array[fx.Int64, 256, 32]
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] packed E4M3 bytes
        arg_cache: fx.Int64,  # [pages, 2 Hkv, 128, 72] uint8
        arg_topk: fx.Int64,  # [Hkv, T, 16] int32 logical blocks, -1 = none
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_req: fx.Int64,  # [T] int32
        arg_pos: fx.Int64,  # [T] int32
        arg_out: fx.Int64,  # [T, H, 128] bf16 (quant: the codes)
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_lut: fx.Int64,
        arg_pack: fx.Int64,
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
        lp = _global_i32_ptr(arg_lut)
        lut_word = Vec.from_elements(
            [fx.Int32(lp[tx * 2 + i]) for i in range_constexpr(2)], fx.Int32
        )
        smem.lut.ptr[tx] = lut_word.bitcast(fx.Int64)[0]
        fx.gpu.barrier()
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
            arg_q, num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * D * (2 if q_bf16 else 1)
        )

        # per token: its own block's page, the causal limit on this wave's key offsets
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG):
            tok = bidx * TOK_PER_WG + t
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
            q_row = (tc * i32_H + kvh * HG + n) * (D // 4) + g * 8
            if const_expr(q_bf16):
                row = []
                for ks in range_constexpr(NKS):
                    vals = Vec(buffer_ops.buffer_load(
                        qr, q_row * 2 + ks * 4, vec_width=4, dtype=fx.Int32
                    )).bitcast(fx.BFloat16).to(fx.Float32)
                    row.append(_pack_fp8([
                        vals[j].maximumf(fx.Float32(-448.0)).minimumf(fx.Float32(448.0))
                        for j in (0, 2, 4, 6, 1, 3, 5, 7)
                    ]))
                q_regs.append(row)
            else:
                q_regs.append([
                    Vec(buffer_ops.buffer_load(qr, q_row + ks * 2, vec_width=2, dtype=fx.Int32)).bitcast(fx.Int64)[0]
                    for ks in range_constexpr(NKS)
                ])


        offs = _page_offsets(i32_hkv, kvh, w, n, g)

        def load_page(page):
            return _load_page(page, arg_cache, i32_page_byte, *offs)

        # lane = (token lane // 16, slot lane % 16): the page it selected (-1: none)
        lane_tok = bidx * 2 + lane // TOPK__16
        lane_tok_ok = (lane_tok < i32_T) & (lane < 32)
        lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
        blk = fx.Int32(
            topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
        )
        # a pick past the token's own block selects nothing and reads no table entry
        lane_blk_ok = lane_tok_ok & (blk >= 0) & (blk <= fx.Int32(pos_p[lane_tok_clamped]) // BLK)
        lane_page = fx.Int32(
            bt[
                fx.Int32(req_p[lane_tok_clamped]) * i32_bt_s
                + lane_blk_ok.select(blk, fx.Int32(0))
            ]
        )
        lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))
        # the union of this pair only (the chunk-P rule allows its own page order); the walk drops every lane of the
        # page it visits
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

        raw0 = load_page(page_at(valid))
        fx.rocdl.sched_barrier(0)
        umask = valid

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # page offset of S^T tile st, row 4 g + e within this wave's 32 keys
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        # iteration u loads page u + 1 and dequantizes page u
        init = [umask]
        init += [fx.Float32(M_FLOOR) for _ in range_constexpr(TOK_PER_WG)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG * NDT)]
        init += raw0
        init += [fx.Float32(M_FLOOR) for _ in range_constexpr(TOK_PER_WG)]  # running maxima
        n_st = 1 + 2 * TOK_PER_WG + TOK_PER_WG * NDT
        results = init
        rem_test = umask
        while rem_test != fx.Int64(0):
            st_ = results
            rem = fx.Int64(st_[0])
            m = [fx.Float32(st_[1 + t]) for t in range_constexpr(TOK_PER_WG)]
            l_run = [
                fx.Float32(st_[1 + TOK_PER_WG + t]) for t in range_constexpr(TOK_PER_WG)
            ]
            o = [
                [
                    Vec(st_[1 + 2 * TOK_PER_WG + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            r_run = [fx.Float32(st_[n_st + RAW + t]) for t in range_constexpr(TOK_PER_WG)]
            page = page_at(rem)
            duplicates = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            rem = rem & ~duplicates
            raw_next = load_page(page_at(rem))
            kf, vf = _dequant_fp8(raw, smem)
            hits = duplicates
            for t in range_constexpr(TOK_PER_WG):
                m_t, r_t, l_t, o_t = m[t], r_run[t], l_run[t], o[t]
                if (
                    hits.shrui(fx.Int64(TOPK__16 * t))
                    & fx.Int64(0xFFFF)
                ) != fx.Int64(0):
                    qf = q_regs[t]
                    a = []
                    for st in range_constexpr(2):
                        acc = _mfma128(_join4(kf[st]), _join4(qf), zero4)
                        a += [acc[e] for e in range_constexpr(4)]
                    if page == diag[t]:
                        a = [
                            (key_off[i] <= lim[t]).select(a[i], neg_inf)
                            for i in range_constexpr(8)
                        ]
                    mx = a[0]
                    for i in range_constexpr(1, 8):
                        mx = _maxf_nn(mx, a[i])
                    # maxima in scaled units: max(a) * c == max(a * c) (c > 0); P against its chunk's own max (this
                    # wave's 32 keys of the page, ``reference.fp8_chunk_attend``), O and the sum against the last
                    # chunk's, at most REF_SPAN below the running max
                    mc = _xlane4(mx, _maxf_nn) * c
                    r_t = _maxf_nn(r_t, mc)
                    m_new = (mc == neg_inf).select(
                        m_t, _maxf_nn(mc, r_t - fx.Float32(REF_SPAN))
                    )
                    alpha = _exp2(m_t - m_new)
                    p = [_exp2(fx.math.fma(x, c, -m_new)) for x in a]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t * alpha + ps
                    pf = _pack_fp8([x * fx.Float32(448.0) for x in p])
                    # alpha is exactly 1 unless some row's reference moved: rescale only then
                    if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                        av = Vec.filled(4, alpha, fx.Float32)
                        o_t = [x * av for x in o_t]
                    o_t = [_mfma_fp8(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    m_t = m_new
                m[t], r_run[t], l_run[t], o[t] = m_t, r_t, l_t, o_t
            state = (
                [rem] + m + l_run + [x for token_out in o for x in token_out] + raw_next
                + r_run
            )
            results = state
            rem_test = rem

        # combine the waves' partials through LDS, tpr tokens per round; wave w finalizes token w in round w // tpr
        ml = smem.ml.ptr
        for t in range_constexpr(TOK_PER_WG):
            ml[((w * TOK_PER_WG + t) * 2) * 64 + lane] = fx.Float32(results[1 + t])
            ml[((w * TOK_PER_WG + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[1 + TOK_PER_WG + t]
            )
        for rnd in range_constexpr(rounds):
            for tt in range_constexpr(tpr):
                for dt in range_constexpr(NDT):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = 1 + 2 * TOK_PER_WG + (tpr * rnd + tt) * NDT + dt
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
                    v_scale / (l_tot * fx.Float32(448.0)), fx.Float32(0.0)
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
    def launch_fp8(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_req: fx.Int64,
        arg_pos: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
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
            arg_lut,
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

    launch_fp8.kernel_name = name
    return launch_fp8
