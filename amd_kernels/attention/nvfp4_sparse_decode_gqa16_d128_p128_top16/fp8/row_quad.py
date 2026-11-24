# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""fp8 decode (``fp8/kernel.py``'s numerics, P per 32-key chunk) of four rows per workgroup over their page union.

At 8 rows per request, 1 KV head, top-16: lane (t, s) holds row t's slot s; the workgroup walks the union of the four
rows' pages in lane order, dequantizes each page once and attends every row that selected it, Q in registers. Each
wave finalizes one row; the output tile is shared twice (one half of D per round). The host takes it from 769 rows,
where the shared dequant beats the per-row kernel's next grid round.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..kernel import (
    BLK, D, DATA_B, HG, KPW, NDT, NKS, NW, RAW, SCALE_B, _exp2, _global_i32_ptr, _mxfp8_group, _permlane_swap, _pk_bf16,
    _pk_fp8, _xlane4,
)
from .kernel import REF_SPAN

TOK_PER_WG = 4  # query tokens per workgroup
TOPK__16 = 16  # selected pages per token
PACK_ROW_PER_HEAD = 132


def _maxf_nn(a, b):
    """v_max_f32 with nnan (no canonicalizing v_max x, x, x on the inputs)."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _mfma(a, b, acc):
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
            Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]
        )
    )


def _join4(vs):
    words = []
    for i in range_constexpr(4):
        v = Vec.from_elements([vs[i]], fx.Int64).bitcast(fx.Int32)
        words += [v[j] for j in range_constexpr(2)]
    return Vec.from_elements(words, fx.Int32)


def _mfma128(a, b, acc):
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    sel = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ops.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(acc),
            e4m3,
            e4m3,
            sel,
            zero,
            sel,
            zero,
        ).result
    )


def _pack_fp8(x):
    words = []
    for i in range_constexpr(2):
        lo = _pk_fp8(x[4 * i], x[4 * i + 1], fx.Int32(_llvm.mlir_undef(T.i32)), False)
        words.append(_pk_fp8(x[4 * i + 2], x[4 * i + 3], lo, True))
    return Vec.from_elements(words, fx.Int32).bitcast(fx.Int64)[0]


def _xlane4_or(x):
    """Bitwise or over lanes L, L^16, L^32, L^48 (Int32), broadcast to all four."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, fx.Int32(x), fx.Int32(x))
    m = a | b
    a, b = _permlane_swap(_rocdl.permlane16_swap, m, m)
    return a | b


def compile_nvfp4_sparse_gqa16_d128_p128_top16_q8_tq4(quant=False):
    """The launcher (``quant``: the MXFP8 output)."""
    pack, rounds, q_bf16 = False, 2, True
    tpr, wpe = 2, 2
    assert rounds in (1, 2)
    name = "nvfp4_fp8mfma_sparse_gqa16_d128_p128_top16_q8_tq4_dsplit"

    name += ("_r1" if rounds == 1 else "")
    name += ("_pack" if pack else "_quant" if quant else "") + ("_qbf16" if q_bf16 else "")

    name += "_walk"

    @fx.struct
    class Shared:
        # The NVFP4 -> E4M3 table at LDS offset 0, so a lookup address is the scale byte times 8 with no base to add
        # (one SDWA shift). LLVM places each field by alignment, then size (both descending): the table's larger
        # alignment puts it first.
        lut: fx.Array[fx.Int64, 256, 32]
        o: fx.Array[fx.Int32, NW * tpr * NDT * 64 * 4, 16]
        ml: fx.Array[fx.Float32, NW * TOK_PER_WG * 2 * 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [T, H, 128] packed E4M3 bytes
        arg_cache: fx.Int64,  # [pages, 2 * 1, 128, 72] uint8
        arg_topk: fx.Int64,  # [1, T, 16] int32 logical blocks, -1 = none
        arg_seq: fx.Int64,
        arg_bt: fx.Int64,  # [reqs, stride] int32
        arg_out: fx.Int64,  # [T, 16, 128] bf16
        arg_ks: fx.Int64,  # fp32 global K scale
        arg_vs: fx.Int64,  # fp32 global V scale
        arg_lut: fx.Int64,
        arg_pack: fx.Int64,
        i32_ql: fx.Int32,
        i32_tile: fx.Int32,
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
        seq_p = _global_i32_ptr(arg_seq)
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

        # Q occupies the future output combine space during the page loop.
        qp = fx.recast_iter(fx.Int64, smem.o.ptr)
        # per token: its own block's page, the causal limit on this wave's key offsets
        diag, lim, q_regs = [], [], []
        for t in range_constexpr(TOK_PER_WG):
            tok = bidx * TOK_PER_WG + t
            ok = tok < i32_T
            tc = ok.select(tok, fx.Int32(0))
            req = tc // i32_ql
            seq = fx.Int32(seq_p[req])
            pos = seq - i32_ql + tc % i32_ql
            diag.append(
                fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, fx.Int32(bt[req * i32_bt_s + (pos >= 0).select(pos, fx.Int32(0)) // BLK])
                    )
                )
            )
            lim.append(pos % BLK - w * KPW)
        # every wave stages one query in LDS; each then holds all four in registers (q_all)
        tq = w
        tok_q = bidx * TOK_PER_WG + tq
        tc_q = (tok_q < i32_T).select(tok_q, fx.Int32(0))
        q_row = (tc_q * i32_H + kvh * HG + n) * (D // 4) + g * 8
        for ks in range_constexpr(NKS):
            if const_expr(q_bf16):
                vals = Vec(buffer_ops.buffer_load(qr, q_row * 2 + ks * 4,
                    vec_width=4, dtype=fx.Int32)).bitcast(fx.BFloat16).to(fx.Float32)
                packed = _pack_fp8([vals[j].maximumf(fx.Float32(-448.0)).minimumf(fx.Float32(448.0))
                    for j in (0,2,4,6,1,3,5,7)])
            else:
                packed = Vec(buffer_ops.buffer_load(qr, q_row + ks * 2,
                    vec_width=2, dtype=fx.Int32)).bitcast(fx.Int64)[0]
            qp[(tq * NKS + ks) * 64 + lane] = packed
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

        # page offsets (bytes) of this KV head's K data / scales, V data / scales
        side_b = i32_hkv * (DATA_B + SCALE_B)
        k_data = kvh * DATA_B
        k_scl = i32_hkv * DATA_B + kvh * SCALE_B
        v_data = side_b + kvh * DATA_B
        v_scl = side_b + i32_hkv * DATA_B + kvh * SCALE_B
        # Keys in page order: S^T tile st row r is key 32 w + 8 (r // 4) + 4 st + r % 4, so lane (n, g)'s eight
        # P^T / V^T key slots are keys 32 w + 8 g + j. K: rows 32 w + 8 (n // 4) + 4 st + n % 4, bytes
        # [16 g, 16 g + 16) (dims 32 g..); the scales of both rows (groups 2 g, 2 g + 1) are one dword.
        k_rows = [w * KPW + 8 * (n // 4) + 4 * st + n % 4 for st in range_constexpr(2)]
        k_load = [(k_data + r * (D // 2) + g * 16) // 4 for r in k_rows]
        k_scale_load = (k_scl + ((w * 4 + g) * 16 + n) * 4) // 4
        # V: keys 32 w + 8 g + j of dims 8 n.. are one 32-byte (token octet, dim octet) tile, dword dt = dim 8 n + dt (a
        # byte = two consecutive keys); their scales (group n // 2) are 8 contiguous bytes (group-major scales).
        v_load = (v_data + ((w * KPW + 8 * g) // 8 * 16 + n) * 32) // 4
        v_scale_load = (v_scl + (n // 2) * BLK + w * KPW + 8 * g) // 4

        def load_page(page):
            """This lane's NVFP4 dwords of ``page`` (RAW): K rows st = 0, 1 (4 + 4), their scales (1), V dims 8 n + dt
            (8), the scales of their keys (2)."""
            # TODO: check whether one resource buffer can cover all pages when the total size is below 4 GB
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_cache + fx.Int64(page) * fx.Int64(i32_page_byte),
                num_records_bytes=i32_page_byte,
            )
            raw = []
            for st in range_constexpr(2):
                kd = fx.Vector(
                    buffer_ops.buffer_load(pr, k_load[st], vec_width=4, dtype=fx.Int32)
                )
                raw += [kd[i] for i in range_constexpr(4)]
            raw.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        pr, k_scale_load, vec_width=1, dtype=fx.Int32
                    )
                )
            )
            for h in range_constexpr(2):
                vd = fx.Vector(
                    buffer_ops.buffer_load(
                        pr, v_load + 4 * h, vec_width=4, dtype=fx.Int32
                    )
                )
                raw += [vd[i] for i in range_constexpr(4)]
            vs = fx.Vector(
                buffer_ops.buffer_load(pr, v_scale_load, vec_width=2, dtype=fx.Int32)
            )
            raw += [vs[i] for i in range_constexpr(2)]
            return raw

        # lane = (token lane // 16, slot lane % 16): the page it selected (-1: none)
        lane_tok = bidx * 4 + lane // TOPK__16
        lane_tok_ok = lane_tok < i32_T
        lane_tok_clamped = lane_tok_ok.select(lane_tok, fx.Int32(0))
        blk = fx.Int32(
            topk[topk_base__0 + lane_tok_clamped * TOPK__16 + lane % TOPK__16]
        )  # topk[0, lane_tok, slot]; the block resolves through the block table
        lane_pos = fx.Int32(seq_p[lane_tok_clamped // i32_ql]) - i32_ql + lane_tok_clamped % i32_ql
        lane_blk_ok = lane_tok_ok & (lane_pos >= 0) & (blk >= 0) & (blk <= lane_pos // BLK)
        lane_page = fx.Int32(
            bt[
                (lane_tok_clamped // i32_ql) * i32_bt_s
                + lane_blk_ok.select(blk, fx.Int32(0))
            ]
        )
        lane_page = lane_blk_ok.select(lane_page, fx.Int32(-1))
        # The union's first page is the lowest valid lane's (lane order): load it before the dedup so the dedup runs
        # under its latency. A page's representative is its first lane over all four rows.
        valid = fx.Int64(fx.rocdl.ballot(T.i64, lane_page >= 0))
        first = (valid != fx.Int64(0)).select(
            fx.Int32(fx.math.cttz(valid).to(fx.Int32)), fx.Int32(0)
        )

        def page_at(mask):
            """Page of the lowest union lane left in ``mask`` (the first page again
            when it is empty)."""
            idx = (mask != fx.Int64(0)).select(
                fx.Int32(fx.math.cttz(mask).to(fx.Int32)), first
            )
            pg = fx.Int32(fx.rocdl.readlane(T.i32, lane_page, idx))
            return (pg >= 0).select(pg, fx.Int32(0))

        page0 = page_at(valid)
        raw0 = load_page(page0)
        fx.rocdl.sched_barrier(0)
        # union: the first lane of each page, in lane order
        umask = valid

        def tab(sc):
            # Shared.lut sits at LDS offset 0: the address is sc * 8 (one SDWA shift)
            value = fx.Int64(smem.lut.ptr[sc])
            pair = Vec.from_elements([value], fx.Int64).bitcast(fx.Int32)
            return [pair[0], pair[1]]

        def perm(hi, lo, sel):
            return fx.Int32(_rocdl.perm_b32(hi, lo, sel))

        def dq8(c, t):
            even = perm(t[1], t[0], c & 0x07070707) ^ ((c << 4) & -2139062144)
            odd = perm(t[1], t[0], (c >> 4) & 0x07070707) ^ (c & -2139062144)
            return Vec.from_elements([even, odd], fx.Int32).bitcast(fx.Int64)[0]

        def dequant(raw):
            kt = [tab((raw[8] >> (8 * i)) & 255) for i in range_constexpr(4)]
            kf = [
                [
                    dq8(raw[st * 4 + ks], kt[st * 2 + ks // 2])
                    for ks in range_constexpr(NKS)
                ]
                for st in range_constexpr(2)
            ]
            x = [raw[9 + dt] for dt in range_constexpr(8)]
            for dt in range_constexpr(4):
                a, b = x[dt], x[dt + 4]
                x[dt] = perm(b, a, fx.Int32(0x05040100))
                x[dt + 4] = perm(b, a, fx.Int32(0x07060302))
            for r in (0, 1, 4, 5):
                a, b = x[r], x[r + 2]
                x[r] = perm(b, a, fx.Int32(0x06020400))
                x[r + 2] = perm(b, a, fx.Int32(0x07030501))
            vt = [
                tab((raw[17 + k // 4] >> (8 * (k % 4))) & 255)
                for k in range_constexpr(8)
            ]
            z = []
            for kp in range_constexpr(4):
                even, odd = [], []
                for par in range_constexpr(2):
                    c = x[2 * kp + par]
                    te, to = vt[2 * kp], vt[2 * kp + 1]
                    even.append(
                        perm(te[1], te[0], c & 0x07070707) ^ ((c << 4) & -2139062144)
                    )
                    odd.append(
                        perm(to[1], to[0], (c >> 4) & 0x07070707) ^ (c & -2139062144)
                    )
                z.extend([even, odd])
            words = [[fx.Int32(0), fx.Int32(0)] for dt in range_constexpr(8)]
            for par in range_constexpr(2):
                for half in range_constexpr(2):
                    a, b, c, d = [z[4 * half + k][par] for k in range_constexpr(4)]
                    u0 = perm(b, a, fx.Int32(0x05010400))
                    u1 = perm(b, a, fx.Int32(0x07030602))
                    v0 = perm(d, c, fx.Int32(0x05010400))
                    v1 = perm(d, c, fx.Int32(0x07030602))
                    vals = [
                        perm(v0, u0, fx.Int32(0x05040100)),
                        perm(v0, u0, fx.Int32(0x07060302)),
                        perm(v1, u1, fx.Int32(0x05040100)),
                        perm(v1, u1, fx.Int32(0x07060302)),
                    ]
                    for i in range_constexpr(4):
                        words[par + 2 * i][half] = vals[i]
            vf = [Vec.from_elements(w, fx.Int32).bitcast(fx.Int64)[0] for w in words]
            return kf, vf

        zero4 = Vec.filled(4, 0.0, fx.Float32)
        # page offset of S^T tile st, row 4 g + e within this wave's 32 keys
        key_off = [
            8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)
        ]
        # The four packed queries in registers after the Q barrier: eight registers per query remove the LDS waits
        # before every score MFMA; both output modes still fit two workgroups per CU without scratch.
        q_all = [
            [fx.Int64(qp[(t * NKS + ks) * 64 + lane]) for ks in range_constexpr(NKS)]
            for t in range_constexpr(TOK_PER_WG)
        ]
        # iteration u loads page u + 1 and dequantizes page u (loaded one iteration earlier)
        init = [umask, page0]
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG)] + [
            fx.Float32(0.0) for _ in range_constexpr(TOK_PER_WG)
        ]
        init += [zero4 for _ in range_constexpr(TOK_PER_WG * NDT)]
        init += raw0
        init += [neg_inf for _ in range_constexpr(TOK_PER_WG)]  # running maxima
        n_st = 2 + 2 * TOK_PER_WG + TOK_PER_WG * NDT
        results = init
        # FlyDSL rewrites this loop with loop-carried state.
        rem_test = umask
        while rem_test != fx.Int64(0):
            st_ = results
            rem = fx.Int64(st_[0])
            page = fx.Int32(st_[1])
            m = [fx.Float32(st_[2 + t]) for t in range_constexpr(TOK_PER_WG)]
            l_run = [
                fx.Float32(st_[2 + TOK_PER_WG + t]) for t in range_constexpr(TOK_PER_WG)
            ]
            o = [
                [
                    Vec(st_[2 + 2 * TOK_PER_WG + t * NDT + dt])
                    for dt in range_constexpr(NDT)
                ]
                for t in range_constexpr(TOK_PER_WG)
            ]
            raw = [fx.Int32(st_[n_st + i]) for i in range_constexpr(RAW)]
            r_run = [fx.Float32(st_[n_st + RAW + t]) for t in range_constexpr(TOK_PER_WG)]
            duplicates = fx.Int64(fx.rocdl.ballot(T.i64, lane_page == page))
            rem = rem & ~duplicates
            page_next = page_at(rem)
            raw_next = load_page(page_next)
            kf, vf = dequant(raw)
            hits = duplicates
            for t in range_constexpr(TOK_PER_WG):
                m_t, r_t, l_t, o_t = m[t], r_run[t], l_run[t], o[t]
                if (
                    hits.shrui(fx.Int64(TOPK__16 * t))
                    & fx.Int64(0xFFFF)
                ) != fx.Int64(0):
                    qf = q_all[t]
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
                    # maxima in scaled units: max(a) * c == max(a * c) (c > 0). P against its chunk's own max (this
                    # wave's 32 keys of the page; fp8/kernel.py's rule), O and the sum against the last chunk's, at
                    # most REF_SPAN below the running max
                    mc = _xlane4(mx, _maxf_nn) * c
                    r_t = _maxf_nn(r_t, mc)
                    m_new = (mc == neg_inf).select(
                        m_t, _maxf_nn(mc, r_t - fx.Float32(REF_SPAN))
                    )
                    m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
                    alpha = _exp2(m_t - m_use)
                    p = [_exp2(fx.math.fma(x, c, -m_use)) for x in a]
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t * alpha + ps
                    pf = _pack_fp8([x * fx.Float32(448.0) for x in p])
                    # the reference moves with nearly every chunk: rescale always
                    av = Vec.filled(4, alpha, fx.Float32)
                    o_t = [x * av for x in o_t]
                    o_t = [_mfma(vf[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)]
                    m_t = m_new
                m[t], r_run[t], l_run[t], o[t] = m_t, r_t, l_t, o_t
            state = (
                [rem, page_next] + m + l_run + [x for token_out in o for x in token_out] + raw_next
                + r_run
            )
            results = state
            rem_test = rem

        # one 64 KiB partial tile, used twice (one half of D per round); each wave finalizes its own row, every dim
        # in the per-row kernel's sum order
        ml = smem.ml.ptr
        for t in range_constexpr(TOK_PER_WG):
            ml[((w * TOK_PER_WG + t) * 2) * 64 + lane] = fx.Float32(results[2 + t])
            ml[((w * TOK_PER_WG + t) * 2 + 1) * 64 + lane] = fx.Float32(
                results[2 + TOK_PER_WG + t]
            )
        fx.gpu.barrier()  # ML ready and every wave past its Q reads
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
        for rnd in range_constexpr(2):
            for tt in range_constexpr(TOK_PER_WG):
                for dd in range_constexpr(NDT // 2):
                    dt = rnd * (NDT // 2) + dd
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ri = 2 + 2 * TOK_PER_WG + tt * NDT + dt
                    r.store(Vec(results[ri]).bitcast(fx.Int32))
                    fx.copy(
                        lds_atom,
                        r,
                        fx.slice(
                            lds,
                            (None, ((w * TOK_PER_WG + tt) * (NDT // 2) + dd) * 64 + lane),
                        ),
                    )
            fx.gpu.barrier()
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for dd in range_constexpr(NDT // 2):
                    dt = rnd * (NDT // 2) + dd
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    o_tile = ((ww * TOK_PER_WG + t) * (NDT // 2) + dd) * 64 + lane
                    fx.copy(lds_atom, fx.slice(lds, (None, o_tile)), r)
                    acc[dt] = acc[dt] + Vec(r.load()).bitcast(fx.Float32) * wv
            if const_expr(rnd == 0):
                fx.gpu.barrier()  # all reads complete before overwriting the tile
        # lane (n, g) holds dims 8 (4 g + e) + dt of head n: per e, 8 consecutive dims, i.e. dims [32 g, 32 g + 32)
        # of head n in order
        tok = bidx * TOK_PER_WG + t
        words = [
            [
                _pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv)
                for j in range_constexpr(4)
            ]
            for e in range_constexpr(4)
        ]
        if const_expr(pack or quant):
            # the lane's 32 dims are one MXFP8 group (``_mxfp8_group``)
            dws, sb = _mxfp8_group([words[e][j] for e in range_constexpr(4) for j in range_constexpr(4)])
            if const_expr(quant):
                k_row = i32_H * D
                head = kvh * HG + n
                qcr = buffer_ops.create_buffer_resource_from_addr(
                    arg_out, num_records_bytes=fx.Int64((i32_T+15)//16*16) * fx.Int64(k_row)
                )
                k0 = head * D + g * 32
                off_t = ((tok // 16) * (16 * k_row) + (k0 // 64) * 1024
                         + ((k0 % 64) // 16) * 256 + (tok % 16) * 16) // 4
                off_l = (tok * k_row + k0) // 4
                q_off = (i32_tile > 0).select(off_t, off_l)
                step = (i32_tile > 0).select(fx.Int32(64), fx.Int32(4))
                for h in range_constexpr(2):
                    val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                    buffer_ops.buffer_store(
                        val, qcr, q_off + h * step, mask=tok < i32_T
                    )
                # AITER scales: [M/32,K/256,4,16,2,2].
                gi = head * 4 + g
                s_off = (
                    (
                        (((tok // 32) * (k_row // 256) + gi // 8) * 4 + gi % 4)
                        * 16
                        + tok % 16
                    )
                    * 2
                    + (gi % 8) // 4
                ) * 2 + (tok % 32) // 16
                rows32 = (i32_T + 31) // 32 * 32
                qsr = buffer_ops.create_buffer_resource_from_addr(
                    arg_pack,
                    num_records_bytes=fx.Int64(rows32) * fx.Int64(k_row // 32),
                )
                buffer_ops.buffer_store(
                    sb.to(fx.Int32).to(fx.Int8), qsr, s_off, mask=tok < i32_T
                )
            else:
                row_dw = i32_H * (PACK_ROW_PER_HEAD // 4)
                pack_bytes = (
                    fx.Int64(i32_T) * fx.Int64(i32_H) * PACK_ROW_PER_HEAD
                )
                packr = buffer_ops.create_buffer_resource_from_addr(
                    arg_pack, num_records_bytes=pack_bytes
                )
                p_row = tok * row_dw + (kvh * HG + n) * (D // 4) + g * 8
                for h in range_constexpr(2):
                    val = Vec.from_elements(dws[4 * h : 4 * h + 4], fx.Int32)
                    buffer_ops.buffer_store(
                        val, packr, p_row + h * 4, mask=tok < i32_T
                    )
                # The head's four scale bytes: one dword after the codes.
                sdw = _xlane4_or(sb.to(fx.Int32) << (g * 8))
                buffer_ops.buffer_store(
                    sdw,
                    packr,
                    tok * row_dw + i32_H * (D // 4) + kvh * HG + n,
                    mask=(tok < i32_T) & (g == 0),
                )
        else:
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out,
                num_records_bytes=fx.Int64(i32_T) * fx.Int64(i32_H) * (D * 2),
            )
            o_row = (tok * i32_H + kvh * HG + n) * (D // 2) + g * 16
            for e in range_constexpr(4):
                val = Vec.from_elements(words[e], fx.Int32)
                buffer_ops.buffer_store(
                    val, outr, o_row + e * 4, mask=tok < i32_T
                )

    @flyc.jit
    def launch_fp8_q8_tq4(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_seq: fx.Int64,
        arg_bt: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_lut: fx.Int64,
        arg_pack: fx.Int64,
        i32_ql: fx.Int32,
        i32_tile: fx.Int32,
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
            arg_seq,
            arg_bt,
            arg_out,
            arg_ks,
            arg_vs,
            arg_lut,
            arg_pack,
            i32_ql,
            i32_tile,
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

    launch_fp8_q8_tq4.kernel_name = name
    return launch_fp8_q8_tq4
