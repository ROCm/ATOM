# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 NVFP4 sparse paged attention for decode rows (DSpark verify: ``query_len`` rows
a request), with the top-16 selection inside. FlyDSL, gfx950.

The decode chain was index scorer -> aiter ``topk_per_row_small_k`` -> aiter
``sparse_pa_nvfp4_decode``. Given the scorer's fp32 block scores this kernel does the last
two: each workgroup selects its rows' top-16 itself (bit for bit the aiter selector: same
64 lane chunks, MSB-first cut search, tie order) and attends.

A workgroup is (unit, split), unit = (request, group of R rows, kv head). The rows' selected
blocks are deduplicated into one list (per block, the mask of rows that picked it); every
listed page is loaded and dequantized once per wave and each picking row runs its QK /
softmax / PV against the same registers. ``S`` splits partition the list and merge through
fp32 partials (the last workgroup of a unit to arrive merges). A row attends exactly its own
blocks, causally capped at its position; only the order blocks are accumulated in differs
from the reference kernel (fp32 accumulation, bf16 output).

What decides the speed (measured, MI355X, notes/attn-decode.md): at large batch the page
loads (one 18 KB page per row and block), so the units of a request sit on one XCD and are
dispatched together and its rows find the pages they share in that L2; per (row, block) the
VALU softmax, trimmed by the row sum on the MFMA and the lazy rescale. One row per workgroup
(R = 1) wins at every measured batch on real selection overlap (8 rows of a request pick ~45
distinct blocks between them): a larger R shares the dequantization but drops to one wave
per SIMD.

Page layout, MFMA mapping and dequantization are aiter's (``sparse_pa_nvfp4_common``):
wave w owns keys 32 w .. 32 w + 31 of every block; S^T = K Q^T and O^T += V^T P^T with
v_mfma_f32_16x16x32_f16 over the exact e2m1 x e4m3 products.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    atomic_or_i32,
    ord_signed_f32,
    uint32_to_int32,
)
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_common import (
    BLK,
    D,
    KPW,
    NDT,
    NKS,
    NW,
    OOB_DWORD,
    PRELOAD_HINTS,
    RAW,
    SC1,
    bf16x8_to_f16,
    bits,
    as_f32,
    check_operands,
    cvt_fp8x2_f16,
    exp2,
    global_f32_ptr,
    global_i32_ptr,
    host_scales,
    maxf_nn,
    mfma_f16,
    pk_bf16,
    pk_f16,
    refuse_alloc_in_capture,
    require,
    scales_up_to,
    unit_scale,
    xlane4,
)
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import (
    _atomic_add,
    _dq4,
    _page_loader,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    _to_raw,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

XCDS = 8
CAP = 8192  # block ids the union table covers (1M tokens)
MAX_UNITS = 4096
MAX_S = 16
PART_CAP = 16384  # (unit, split, row, merge part) slots of the split scratch; one spare past it
PART_LANE_F = 20  # fp32 per slot lane (>= PFL_PAD)


def compile_q8(
    ql: int,
    rows: int,
    splits: int,
    topk: int = 16,
    prefetch: int = 1,
    lazy_t: float = 0.0,
    timing: bool = False,
    select_cols: int = 0,
    local_blocks: int = 1,
    init_blocks: int = 0,
    write_topk: bool = False,
    l_mfma: bool = False,
    ablate: str = "",
    req_xcd: bool = False,
):
    """``select_cols`` > 0: the workgroup selects its rows' top-``topk`` itself from the index
    scorer's fp32 rows (row width bound ``select_cols``), exactly as aiter's
    ``topk_per_row_small_k`` (same 64 lane chunks, cut search and tie order), instead of reading
    ``topk_idx``; ``write_topk`` also stores the selection (split 0) to ``topk_idx``.

    ``ql`` rows per request, ``rows`` (R) of them per workgroup, ``splits`` (S) workgroups
    per unit, ``prefetch`` pages in flight ahead of the one attended. ``lazy_t`` > 0: a row's
    reference max moves only when a block's max exceeds it by more than ``lazy_t`` (log2
    units; P stays <= 2^lazy_t), 0: whenever it grows (the reference kernel's per-block
    rescale). ``l_mfma``: the softmax row sum is one more MFMA (ones x P^T) instead of VALU
    adds. ``req_xcd``: all units of a request on one XCD, dispatched together, so its rows
    share that L2 for the pages their selections share. ``timing`` / ``ablate``: diagnosis
    only (per-workgroup phase stamps; dropping loop parts gives wrong results)."""
    QL, R, S, TOPK, PF = ql, rows, splits, topk, prefetch
    LT = float(lazy_t)
    require(LT < 15, "lazy_t must keep P = 2^lazy_t inside f16")
    require(QL % R == 0, "rows must divide ql")
    require(R in (1, 2, 4, 8), "rows per workgroup must be 1, 2, 4 or 8")
    require(1 <= S <= MAX_S, "splits")
    G = QL // R
    NE = R * TOPK  # selection entries of a workgroup
    require(NE <= 64 * NW, "R * topk must fit the workgroup's threads")
    RR = 1  # rows per merge round
    WPR = NW // RR  # waves per row in a merge round
    DPW = NDT // WPR  # d-tiles a merging wave owns
    PFL = DPW * 4 + 2  # fp32 per partial lane: O part, max, sum
    require(PFL <= PART_LANE_F, "partial lane")
    U2 = PF + 1
    name = f"m3_nvfp4_spa_q8_ql{QL}_r{R}_s{S}_k{TOPK}_p{PF}"
    TIMING = timing
    if TIMING:
        name += "_tm"
    LM = l_mfma
    LAZY = LT > 0
    if LAZY:
        name += f"_lz{int(LT)}"
    # diagnosis only (wrong results): drop the page loads / dequant / softmax / PV of the loop
    ABL = set(x for x in ablate.split("+") if x)
    if ABL:
        name += "_abl_" + "_".join(sorted(ABL))
    # all units of one request (its row groups, splits, kv heads) on one XCD, dispatched
    # together: the request's rows share that L2 for the pages their selections share
    RX = req_xcd
    if RX:
        name += "_rx"
    QREG = R == 1  # one row: Q stays in registers (no LDS round trip per block)
    ENT_REG = NE <= 64  # the union list fits one lane per entry
    NO = NDT + (1 if LM else 0)  # accumulator tiles per row (O^T d-tiles [+ the row sum])
    if LM:
        name += "_lm"

    def stamp():
        from flydsl._mlir.dialects import llvm as _llvm_d

        return fx.Int64(_llvm_d.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
    # LDS: [table | Q | list] while attending, then [O | max, sum] for the merge (overlaid)
    TBL_OFF, TBL_W = 0, CAP // 4
    Q_OFF, Q_W = TBL_W, R * NKS * 64 * 4
    LST_OFF, LST_W = Q_OFF + Q_W, 2 * NE
    O_OFF, O_W = 0, RR * NW * NDT * 64 * 4
    ML_OFF, ML_W = O_W, RR * NW * 2 * 64
    CNT_OFF = max(LST_OFF + LST_W, ML_OFF + ML_W)
    BUF_W = CNT_OFF + 8
    # in-kernel selection: [chunk maxima | survivor keys | survivor columns | control] first,
    # the selected ids at the very end (read by the union build after the scratch is reused)
    SEL = select_cols > 0
    MAXCOL = -(-max(select_cols, 1) // 1024) * 1024
    VPT = MAXCOL // 1024  # 4-float vectors per thread and row
    SURV = TOPK * (MAXCOL // 256) * 4  # hard survivor bound: k chunks x elements per chunk
    LOCAL, INIT, WTOPK = local_blocks, init_blocks, write_topk
    if SEL:
        require(R <= NW, "in-kernel selection: one wave per row")
        require(TOPK <= 64, "topk")
        CM_OFF = 0
        SO_OFF = CM_OFF + R * 256
        SC_OFF = SO_OFF + R * SURV
        CTL_OFF = SC_OFF + R * SURV
        A_END = CTL_OFF + 4 * R
        SEL_OFF = max(BUF_W, A_END)
        BUF_W = SEL_OFF + R * TOPK
        name += f"_sel{MAXCOL}_l{LOCAL}i{INIT}" + ("w" if WTOPK else "")
    INT32_MIN = -2147483648

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Int32, BUF_W, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_units: fx.Int32,
        i32_hkv: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_H: fx.Int32,
        i32_rows: fx.Int32,
        arg_seq: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_out: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        arg_dbg: fx.Int64,
        arg_score: fx.Int64,  # [Hkv, >= rows, width] fp32 index block scores (SEL)
        i32_score_hs: fx.Int32,
        i32_score_rs: fx.Int32,
    ):
        storage = fx.SharedAllocator().allocate(Shared)
        smem = storage.peek()
        tx = fx.thread_idx.x
        ts = [stamp()] if TIMING else None
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n, g = lane % 16, lane // 16
        topk_p = global_i32_ptr(arg_topk)
        seq_p = global_i32_ptr(arg_seq)
        c = f32_c * fx.Float32(global_f32_ptr(arg_ks)[0])
        v_scale = fx.Float32(global_f32_ptr(arg_vs)[0])
        neg_inf = fx.Float32(float("-inf"))
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        ones4 = Vec.from_elements([fx.Int32(0x3C003C00)] * 4, fx.Int32)  # f16 1.0 pairs

        # 1-D grid; workgroup b runs on XCD b % 8; all splits of a unit share an XCD.
        bid = fx.block_idx.x
        xcd = bid % XCDS
        slot = bid // XCDS
        zz = slot % S
        if const_expr(RX):
            # slot = ((req // 8) * (units per request) + unit in request) * S + split
            upr = i32_hkv * G
            t = slot // S
            req = (t // upr) * XCDS + xcd
            uir = t % upr
            unit = req * upr + uir
        else:
            unit = (slot // S) * XCDS + xcd
        unit_ok = unit < i32_units
        unit = unit_ok.select(unit, fx.Int32(0))
        kvh = unit % i32_hkv
        rest = unit // i32_hkv
        rg = rest % G
        req = rest // G
        seq = fx.Int32(seq_p[req])
        seq = unit_ok.select(seq, fx.Int32(0))
        seq = fx.Int32(fx.rocdl.readfirstlane(T.i32, seq))
        row0 = req * QL + rg * R  # first row of the workgroup
        pos0 = seq - QL + rg * R  # its position; row r sits at pos0 + r
        tail_blk = (seq > 0).select((seq - 1) // BLK, fx.Int32(-1))
        lim_max = (seq - 1) % BLK - w * KPW  # keys of the tail block past it are unwritten

        # ---- Q of the R rows -> LDS (f16), unit (r, ks, lane) = the B operand of k-step ks.
        qr = ptr_buf_tensor(
            arg_q,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=i32_rows * i32_H * (D * 2),
        )
        base_it = fx.recast_iter(fx.Int32, smem.buf.ptr)
        q_lds = fx.logical_divide(
            fx.make_view(base_it + Q_OFF, fx.make_layout(Q_W, 1)),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        # wave w loads k-step w of every row (NKS == NW); the loads are issued before the
        # selection, the LDS stores after it (the selection scratch overlays Q)
        q_regs = []
        if const_expr(QREG):
            # one row: every lane keeps its 4 k-step B operands in registers
            qd = ((row0 * i32_H + kvh * 16 + n) * (D // 2)) + g * 16
            q_regs = [buf_copy_load(qr, qd + ks * 4, unit_elems=4) for ks in range_constexpr(NKS)]
        else:
            for r in range_constexpr(R):
                qd = (((row0 + r) * i32_H + kvh * 16 + n) * (D // 2)) + g * 16 + w * 4
                q_regs.append(buf_copy_load(qr, qd, unit_elems=4))

        def uge(a, b):
            return fx.arith.cmpi(fx.arith.CmpIPredicate.uge, fx.Int32(a), fx.Int32(b))

        def ugt(a, b):
            return fx.arith.cmpi(fx.arith.CmpIPredicate.ugt, fx.Int32(a), fx.Int32(b))

        def select_rows():
            """Top-TOPK of each of the R rows into LDS (aiter topk_per_row_small_k, tie_low off):
            chunk = lane (vector index mod 64), the k-th largest chunk maximum by an MSB-first
            ballot search, survivors of the chosen chunks at or above it, ranked by (score,
            column) descending. Returns this thread's entry (row tx // TOPK, slot tx % TOPK)."""
            cm = fx.make_view(base_it + CM_OFF, fx.make_layout(R * 256, 1))
            so = fx.make_view(base_it + SO_OFF, fx.make_layout(R * SURV, 1))
            scv = fx.make_view(base_it + SC_OFF, fx.make_layout(R * SURV, 1))
            ctl = fx.make_view(base_it + CTL_OFF, fx.make_layout(4 * R, 1))
            selv = fx.make_view(base_it + SEL_OFF, fx.make_layout(R * TOPK, 1))
            neg = fx.Int32(INT32_MIN)
            rls, lss, srows = [], [], []
            for r in range_constexpr(R):
                pos_r = pos0 + r
                rl = (unit_ok & (pos_r >= 0)).select(pos_r // BLK + 1, fx.Int32(0))
                rls.append(rl)
                lss.append(fx.max(fx.Int32(0), rl - LOCAL))
                base = arg_score + fx.Int64(kvh * i32_score_hs + (row0 + r) * i32_score_rs) * 4
                srows.append(
                    ptr_buf_tensor(base, fx.Float32, unit_elems=4, num_records_bytes=rl * 4)
                )

            def ord_at(value, col, r):
                live = col < rls[r]
                if const_expr(INIT > 0):
                    value = (live & (col < INIT)).select(fx.Float32(1e30), value)
                if const_expr(LOCAL > 0):
                    value = (live & (col >= lss[r])).select(fx.Float32(1e29), value)
                return live.select(ord_signed_f32(fx.Float32(value)), neg)

            # 1. thread maxima (thread t reads vectors t, t + 256, ...). Every load is issued
            # at once: past the row the buffer's record count turns it into a zero with no
            # memory access, and the keys stay in registers for the survivor pass.
            keys = []
            for r in range_constexpr(R):
                lvs = [
                    Vec(buf_copy_load(srows[r], tx + 256 * j, elem=fx.Float32, unit_elems=4))
                    for j in range_constexpr(VPT)
                ]
                kr = []
                mx = neg
                for j in range_constexpr(VPT):
                    kj = [neg, neg, neg, neg]
                    mj = neg
                    # vectors past the row (a 175K row is 2 of 8 at 1M width) cost nothing
                    if rls[r] > 1024 * j:
                        kj = [ord_at(lvs[j][e], (tx + 256 * j) * 4 + e, r) for e in range_constexpr(4)]
                        mj = fx.max(fx.max(kj[0], kj[1]), fx.max(kj[2], kj[3]))
                    kr += kj
                    mx = fx.max(mx, mj)
                keys.append(kr)
                cm[r * 256 + tx] = mx
            if tx < R:
                ctl[tx * 4 + 3] = fx.Int32(0)
            fx.gpu.barrier()
            # 2. wave r: the k chunks and the cut of row r
            for r in range_constexpr(R):
                if w == r:
                    sup = neg
                    for s in range_constexpr(NW):
                        sup = fx.max(sup, fx.Int32(cm[r * 256 + s * 64 + lane]))
                    key = sup ^ neg
                    cut = fx.Int32(0)
                    for bit in range_constexpr(32):
                        probe = cut | fx.Int32(uint32_to_int32(1 << (31 - bit)))
                        hits = fx.Int64(fx.rocdl.ballot(T.i64, uge(key, probe)))
                        cntb = fx.Int32(fx.math.ctpop(fx.Int32(hits))) + fx.Int32(
                            fx.math.ctpop(fx.Int32(hits >> fx.Int64(32)))
                        )
                        cut = (cntb >= TOPK).select(probe, cut)
                    above = ugt(key, cut)
                    tied = key == cut
                    blw = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
                    tie_rank = fx.Int32(fx.math.ctpop(fx.Int64(fx.rocdl.ballot(T.i64, tied)) & blw))
                    quota = TOPK - fx.Int32(fx.math.ctpop(fx.Int64(fx.rocdl.ballot(T.i64, above))))
                    chosen = fx.Int64(fx.rocdl.ballot(T.i64, above | (tied & (tie_rank < quota))))
                    if lane == 0:
                        ctl[r * 4 + 0] = cut
                        ctl[r * 4 + 1] = fx.Int32(chosen)
                        ctl[r * 4 + 2] = fx.Int32(chosen >> fx.Int64(32))
            fx.gpu.barrier()
            # 3. survivors of the chosen chunks at or above the cut
            for r in range_constexpr(R):
                cut = fx.Int32(ctl[r * 4 + 0])
                mask = (fx.Int64(fx.Int32(ctl[r * 4 + 2])) << fx.Int64(32)) | (
                    fx.Int64(fx.Int32(ctl[r * 4 + 1])) & fx.Int64(0xFFFFFFFF)
                )
                in_ch = ((mask >> fx.Int64(lane)) & fx.Int64(1)) == fx.Int64(1)
                if in_ch:
                    for j in range_constexpr(VPT):
                        if rls[r] > 1024 * j:
                            for e in range_constexpr(4):
                                col = (tx + 256 * j) * 4 + e
                                o = keys[r][j * 4 + e]
                                if uge(o ^ neg, cut) & (col < rls[r]):
                                    at = atomic_add_i32(ctl, fx.Int32(1), r * 4 + 3, "workgroup")
                                    so[r * SURV + at] = o
                                    scv[r * SURV + at] = col
            fx.gpu.barrier()
            # 4. wave r ranks row r's survivors; the rank is the slot
            for r in range_constexpr(R):
                if w == r:
                    found = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(ctl[r * 4 + 3])))
                    if found <= 64:
                        li = (lane < found).select(lane, fx.Int32(0))
                        my_o = fx.Int32(so[r * SURV + li])
                        my_c = fx.Int32(scv[r * SURV + li])
                        my_key = (fx.Int64(my_o) << fx.Int64(32)) | fx.Int64(my_c)
                        place = fx.Int32(0)
                        for other in range(fx.Int32(0), found, fx.Int32(1)):
                            peer = fx.Int64(fx.rocdl.readlane(T.i64, _to_raw(my_key), _to_raw(other)))
                            place = place + (peer > my_key).select(fx.Int32(1), fx.Int32(0))
                        if (lane < found) & (place < TOPK):
                            selv[r * TOPK + place] = my_c
                    else:
                        for mine in range(lane, found, fx.Int32(64)):
                            mo = fx.Int32(so[r * SURV + mine])
                            mc = fx.Int32(scv[r * SURV + mine])
                            place = fx.Int32(0)
                            for other in range(fx.Int32(0), found, fx.Int32(1)):
                                po = fx.Int32(so[r * SURV + other])
                                pc = fx.Int32(scv[r * SURV + other])
                                ahead = (po > mo) | ((po == mo) & (pc > mc))
                                place = place + ahead.select(fx.Int32(1), fx.Int32(0))
                            if place < TOPK:
                                selv[r * TOPK + place] = mc
                    real = fx.min(fx.Int32(TOPK), rls[r])
                    if (lane >= real) & (lane < TOPK):
                        selv[r * TOPK + lane] = fx.Int32(-1)
            fx.gpu.barrier()
            mine = fx.Int32(selv[(tx < NE).select(tx, fx.Int32(0))])
            if const_expr(WTOPK):
                tko = global_i32_ptr(
                    arg_topk + fx.Int64(kvh * i32_topk_hs + (row0 + tx // TOPK) * i32_topk_rs + tx % TOPK) * 4
                )
                if (zz == 0) & unit_ok & (tx < NE):
                    tko[0] = mine
            return mine

        e_in = tx < NE
        r_e = tx // TOPK
        pos_e = pos0 + r_e
        if const_expr(SEL):
            sel_ids = select_rows()
            blk_e = sel_ids
        else:
            blk_e = fx.Int32(
                topk_p[
                    kvh * i32_topk_hs
                    + e_in.select(row0 + r_e, fx.Int32(0)) * i32_topk_rs
                    + e_in.select(tx % TOPK, fx.Int32(0))
                ]
            )

        if const_expr(QREG):
            q_regs = [Vec(bf16x8_to_f16(x)) for x in q_regs]
        else:
            for r in range_constexpr(R):
                rq = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                rq.store(Vec(bf16x8_to_f16(q_regs[r])))
                fx.copy(lds_atom, rq, fx.slice(q_lds, (None, (r * NKS + w) * 64 + lane)))

        # ---- union of the rows' selections: entry e = (row e // TOPK, slot e % TOPK).
        tbl = fx.make_view(base_it + TBL_OFF, fx.make_layout(TBL_W, 1))
        lst = fx.make_view(base_it + LST_OFF, fx.make_layout(LST_W, 1))
        cnt = fx.make_view(base_it + CNT_OFF, fx.make_layout(8, 1))
        valid = (
            e_in
            & (pos_e >= 0)
            & (blk_e >= 0)
            & (blk_e <= pos_e // BLK)
            & (blk_e < CAP)
        )
        if const_expr(R == 1):
            # one row: its ids are distinct already, no table
            msk = valid.select(fx.Int32(1), fx.Int32(0))
            rep = valid
        else:
            word = valid.select(blk_e >> 2, fx.Int32(0))
            sh = (blk_e & 3) * 8
            if valid:
                tbl[word] = fx.Int32(0)
            fx.gpu.barrier()
            if valid:
                atomic_or_i32(tbl, (fx.Int32(1) << r_e) << sh, word, "workgroup")
            fx.gpu.barrier()
            msk = (fx.Int32(tbl[word]) >> sh) & 0xFF
            msk = valid.select(msk, fx.Int32(0))
            low = fx.Int32(fx.math.cttz((msk | 0x100)).to(fx.Int32))
            rep = valid & (low == r_e)
        bal = fx.Int64(fx.rocdl.ballot(T.i64, rep))
        if lane == 0:
            cnt[w] = fx.Int32(fx.math.ctpop(bal).to(fx.Int32))
        fx.gpu.barrier()
        cw = [fx.Int32(cnt[i]) for i in range_constexpr(NW)]
        off = fx.Int32(0)
        for i in range_constexpr(1, NW):
            off = off + (w >= i).select(cw[i - 1], fx.Int32(0))
        nu = cw[0]
        for i in range_constexpr(1, NW):
            nu = nu + cw[i]
        below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        rank = off + fx.Int32(fx.math.ctpop(bal & below).to(fx.Int32))
        bt_at = arg_bt + fx.Int64(req * i32_bt_s + rep.select(blk_e, fx.Int32(0))) * 4
        page_e = fx.Int32(global_i32_ptr(rep.select(bt_at, arg_seq))[0])
        if rep:
            lst[rank] = blk_e | (msk << 16)
            lst[NE + rank] = page_e
        fx.gpu.barrier()

        nu = fx.Int32(fx.rocdl.readfirstlane(T.i32, nu))
        if const_expr(TIMING):
            ts.append(stamp())
        chunk = (nu + (S - 1)) // S
        lo = zz * chunk
        hi = (lo + chunk < nu).select(lo + chunk, nu)
        n_e = (hi > lo).select(hi - lo, fx.Int32(0))

        open_page, load_k, load_v = _page_loader(arg_cache, i32_page_b, kvh, i32_hkv, w, n, g)
        key_off = [8 * g + 4 * st + e for st in range_constexpr(2) for e in range_constexpr(4)]

        if const_expr(ENT_REG):
            # <= 64 entries: lane j keeps entry j, the loop reads them with readlane
            e_bm = fx.Int32(lst[(lane < nu).select(lane, fx.Int32(0))])
            e_pg = fx.Int32(lst[NE + (lane < nu).select(lane, fx.Int32(0))])

        def ent(i):
            ok = i < n_e
            idx = ok.select(lo + i, fx.Int32(0))
            if const_expr(ENT_REG):
                bm = fx.Int32(fx.rocdl.readlane(T.i32, fx.as_ir_value(e_bm), fx.as_ir_value(idx)))
                pg = fx.Int32(fx.rocdl.readlane(T.i32, fx.as_ir_value(e_pg), fx.as_ir_value(idx)))
            else:
                bm = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(lst[idx])))
                pg = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(lst[NE + idx])))
            return ok.select(bm & 0xFFFF, fx.Int32(-1)), ok.select(bm >> 16, fx.Int32(0)), ok, pg

        def load_entry(i):
            _, _, ok, pg = ent(i)
            if const_expr("noload" in ABL):
                return [fx.Int32(0x38383838) + pg + k for k in range_constexpr(RAW)]
            pr = open_page(ok.select(pg, fx.Int32(0)), ok)
            return load_k(pr) + load_v(pr)

        def q_frag(r):
            if const_expr(QREG):
                return q_regs
            out = []
            for ks in range_constexpr(NKS):
                rq = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(lds_atom, fx.slice(q_lds, (None, (r * NKS + ks) * 64 + lane)), rq)
                out.append(Vec(rq.load()))
            return out

        def row_update(r, kd, vd, eblk, m_t, l_t, o_t):
            qf = q_frag(r)
            a = []
            for st in range_constexpr(2):
                acc = zero4
                for ks in range_constexpr(NKS):
                    acc = mfma_f16(kd[st * NKS + ks], qf[ks], acc)
                a += [acc[e] for e in range_constexpr(4)]
            pos_r = pos0 + r
            lim_r = pos_r % BLK - w * KPW
            if eblk == pos_r // BLK:
                a = [(key_off[i] > lim_r).select(neg_inf, a[i]) for i in range_constexpr(8)]
            if const_expr("nosm" in ABL):
                pf = Vec.from_elements([pk_f16(a[2 * j], a[2 * j + 1]) for j in range_constexpr(4)], fx.Int32)
                if const_expr("nopv" in ABL):
                    o_t = [o_t[0] + pf.bitcast(fx.Float32)] + o_t[1:]
                else:
                    o_t = [mfma_f16(vd[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)] + (
                        [mfma_f16(ones4, pf, o_t[NDT])] if LM else []
                    )
                return m_t, l_t, o_t
            mx = a[0]
            for i in range_constexpr(1, 8):
                mx = maxf_nn(mx, a[i])
            if const_expr(LAZY):
                # the reference max moves only when the block's max exceeds it by > LT (log2
                # units), so P stays <= 2^LT and most blocks skip the rescale entirely
                mx_s = xlane4(mx, maxf_nn) * c
                need = mx_s > m_t + fx.Float32(LT)
                if fx.Int64(fx.rocdl.ballot(T.i64, need)) != fx.Int64(0):
                    al = need.select(exp2(m_t - mx_s), fx.Float32(1.0))
                    o_t = [x * Vec.filled(4, al, fx.Float32) for x in o_t]
                    l_t = l_t * al
                    m_t = need.select(mx_s, m_t)
                m_use = (m_t == neg_inf).select(fx.Float32(0.0), m_t)
                p = [exp2(fx.math.fma(x, c, -m_use)) for x in a]
                pf = Vec.from_elements([pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)], fx.Int32)
                if const_expr(not LM):
                    ps = p[0]
                    for i in range_constexpr(1, 8):
                        ps = ps + p[i]
                    l_t = l_t + ps
                if const_expr("nopv" in ABL):
                    o_t = [o_t[0] + pf.bitcast(fx.Float32)] + o_t[1:]
                else:
                    o_t = [mfma_f16(vd[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)] + (
                        [mfma_f16(ones4, pf, o_t[NDT])] if LM else []
                    )
                return m_t, l_t, o_t
            m_new = maxf_nn(m_t, xlane4(mx, maxf_nn) * c)
            m_use = (m_new == neg_inf).select(fx.Float32(0.0), m_new)
            p = [exp2(fx.math.fma(x, c, -m_use)) for x in a]
            ps = p[0]
            for i in range_constexpr(1, 8):
                ps = ps + p[i]
            alpha = exp2(m_t - m_use)
            pf = Vec.from_elements([pk_f16(p[2 * j], p[2 * j + 1]) for j in range_constexpr(4)], fx.Int32)
            if const_expr(not LM):
                l_t = l_t * alpha + ps
            if fx.Int64(fx.rocdl.ballot(T.i64, m_new != m_t)) != fx.Int64(0):
                av = Vec.filled(4, alpha, fx.Float32)
                o_t = [x * av for x in o_t]
            o_t = [mfma_f16(vd[dt], pf, o_t[dt]) for dt in range_constexpr(NDT)] + (
                [mfma_f16(ones4, pf, o_t[NDT])] if LM else []
            )
            return m_new, l_t, o_t

        def attend(raw, eblk, emask, ms, ls, os_):
            if const_expr("nodq" in ABL):
                kd = [Vec.from_elements([raw[(i + k) % RAW] for k in range_constexpr(4)], fx.Int32) for i in range_constexpr(8)]
                vd = [Vec.from_elements([raw[(9 + i + k) % RAW] for k in range_constexpr(4)], fx.Int32) for i in range_constexpr(NDT)]
                ms, ls, os_ = list(ms), list(ls), list(os_)
                for r in range_constexpr(R):
                    m_t, l_t, o_t = ms[r], ls[r], os_[r]
                    if ((emask >> r) & 1) != 0:
                        m_t, l_t, o_t = row_update(r, kd, vd, eblk, m_t, l_t, o_t)
                    ms[r], ls[r], os_[r] = m_t, l_t, o_t
                return ms, ls, os_
            kd = []
            for st in range_constexpr(2):
                ksc = cvt_fp8x2_f16(raw[8], st == 1)
                for ks in range_constexpr(NKS):
                    sv = Vec.from_elements([ksc[ks // 2], ksc[ks // 2]], fx.Float16)
                    kd.append(_dq4(raw[st * 4 + ks], [sv] * 4))
            v_scl = [raw[17], raw[18]]
            if eblk == tail_blk:
                # keys past the request's last position may be unwritten (NaN V scales)
                v_scl = [scales_up_to(raw[17 + h], h, g, lim_max) for h in range_constexpr(2)]
            vs = []
            for h in range_constexpr(2):
                vs += [cvt_fp8x2_f16(v_scl[h], False), cvt_fp8x2_f16(v_scl[h], True)]
            vd = [_dq4(raw[9 + dt], vs) for dt in range_constexpr(NDT)]
            ms, ls, os_ = list(ms), list(ls), list(os_)
            for r in range_constexpr(R):
                m_t, l_t, o_t = ms[r], ls[r], os_[r]
                if ((emask >> r) & 1) != 0:
                    m_t, l_t, o_t = row_update(r, kd, vd, eblk, m_t, l_t, o_t)
                ms[r], ls[r], os_[r] = m_t, l_t, o_t
            return ms, ls, os_

        attend_fn = attend

        ring = []
        for j in range_constexpr(PF):
            ring += load_entry(fx.Int32(j))
        init = [fx.Int32(0)]
        init += [neg_inf] * R + [fx.Float32(0.0)] * R
        init += [zero4] * (R * NO)
        init += ring
        NST = 1 + 2 * R + R * NO
        results = init
        for _it, st_ in range(  # type: ignore[call-overload]
            fx.Index(0),
            fx.Index((n_e + (U2 - 1)) // U2),
            fx.Index(1),
            init=[fx.as_ir_value(x) for x in init],
        ):
            cur_i = fx.Int32(st_[0])
            ms = [fx.Float32(st_[1 + r]) for r in range_constexpr(R)]
            ls = [fx.Float32(st_[1 + R + r]) for r in range_constexpr(R)]
            os_ = [
                [Vec(st_[1 + 2 * R + r * NO + dt]) for dt in range_constexpr(NO)]
                for r in range_constexpr(R)
            ]
            bufs = [
                [fx.Int32(st_[NST + j * RAW + i]) for i in range_constexpr(RAW)]
                for j in range_constexpr(PF)
            ]
            for _u in range_constexpr(U2):
                bufs.append(load_entry(cur_i + PF))
                cur = bufs.pop(0)
                eblk, emask, _, _ = ent(cur_i)
                fx.rocdl.sched_barrier(0)
                ms, ls, os_ = attend_fn(cur, eblk, emask, ms, ls, os_)
                cur_i = cur_i + 1
            state = [cur_i] + ms + ls
            for r in range_constexpr(R):
                state += os_[r]
            for b_ in bufs:
                state += b_
            results = yield [fx.as_ir_value(x) for x in state]

        ms = [fx.Float32(results[1 + r]) for r in range_constexpr(R)]
        ls = [fx.Float32(results[1 + R + r]) for r in range_constexpr(R)]
        os_ = [
            [Vec(results[1 + 2 * R + r * NO + dt]) for dt in range_constexpr(NO)]
            for r in range_constexpr(R)
        ]
        if const_expr(LM):
            # every element of the sum tile is the wave's key sum of head n (no lane reduction left)
            ls = [fx.Float32(os_[r][NDT][0]) for r in range_constexpr(R)]

        # ---- combine the waves' partials, RR rows a round; wave w merges d-tiles
        # (w % WPR) * DPW .. of row w // WPR of the round.
        if const_expr(TIMING):
            ts.append(stamp())
        # the merge buffer overlays the table, Q and the list: every wave must be done with them
        fx.gpu.barrier()
        o_lds = fx.logical_divide(
            fx.make_view(base_it + O_OFF, fx.make_layout(O_W, 1)),
            fx.make_layout(4, 1),
        )
        ml = fx.make_view(fx.recast_iter(fx.Float32, smem.buf.ptr) + ML_OFF, fx.make_layout(ML_W, 1))
        rr_w = w // WPR
        dt0 = (w % WPR) * DPW
        outr = ptr_buf_tensor(
            arg_out,
            fx.Int32,
            unit_elems=DPW // 2,
            unit_stride=1,
            num_records_bytes=i32_rows * i32_H * (D * 2),
        )
        pr4 = ptr_buf_tensor(arg_part, fx.Int32, unit_elems=4, unit_stride=1, num_records_bytes=i64_part_bytes)
        pr2 = ptr_buf_tensor(arg_part, fx.Int32, unit_elems=2, unit_stride=1, num_records_bytes=i64_part_bytes)
        PART_SCOPE = 0

        def store_out(acc, inv, row):
            # lane (n, g): element e of d-tile dt is dim 32 g + 8 e + dt of head n
            o_row = (row * i32_H + kvh * 16 + n) * (D // 2) + g * 16 + dt0 // 2
            for e in range_constexpr(4):
                val = Vec.from_elements(
                    [pk_bf16(acc[2 * j][e] * inv, acc[2 * j + 1][e] * inv) for j in range_constexpr(DPW // 2)],
                    fx.Int32,
                )
                buf_copy_store(
                    outr,
                    unit_ok.select(o_row + e * 4, fx.Int32(OOB_DWORD)),
                    val if DPW // 2 > 1 else val[0],
                    unit_elems=DPW // 2,
                )

        def part_lane(r_rel, sp):
            pidx = ((unit * S + sp) * R + r_rel) * WPR + (w % WPR)
            pidx = unit_ok.select(pidx, fx.Int32(PART_CAP))
            return (pidx * 64 + lane) * PART_LANE_F

        for j in range_constexpr(R // RR):
            for q_ in range_constexpr(RR):
                r = j * RR + q_
                ml[((q_ * NW + w) * 2) * 64 + lane] = ms[r]
                ml[((q_ * NW + w) * 2 + 1) * 64 + lane] = ls[r]
                for dt in range_constexpr(NDT):
                    ro = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    ro.store(os_[r][dt].bitcast(fx.Int32))
                    fx.copy(lds_atom, ro, fx.slice(o_lds, (None, ((q_ * NW + w) * NDT + dt) * 64 + lane)))
            fx.gpu.barrier()
            mw = [fx.Float32(ml[((rr_w * NW + ww) * 2) * 64 + lane]) for ww in range_constexpr(NW)]
            big = mw[0]
            for ww in range_constexpr(1, NW):
                big = maxf_nn(big, mw[ww])
            big_raw = big
            big = (big == neg_inf).select(fx.Float32(0.0), big)
            wt = [fx.Float32(fx.math.exp2(mw[ww] - big)) for ww in range_constexpr(NW)]
            l_tot = fx.Float32(0.0)
            for ww in range_constexpr(NW):
                l_tot = l_tot + fx.Float32(ml[((rr_w * NW + ww) * 2 + 1) * 64 + lane]) * wt[ww]
            if const_expr(not LM):
                l_tot = xlane4(l_tot, lambda x, y: x + y)
            acc = [zero4 for _ in range_constexpr(DPW)]
            for ww in range_constexpr(NW):
                wv = Vec.filled(4, wt[ww], fx.Float32)
                for d in range_constexpr(DPW):
                    ro = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(lds_atom, fx.slice(o_lds, (None, ((rr_w * NW + ww) * NDT + dt0 + d) * 64 + lane)), ro)
                    acc[d] = acc[d] + Vec(ro.load()).bitcast(fx.Float32) * wv
            r_rel = j * RR + rr_w
            if const_expr(S == 1):
                inv = (l_tot > fx.Float32(0.0)).select(v_scale / l_tot, fx.Float32(0.0))
                store_out(acc, inv, row0 + r_rel)
            else:
                pl = part_lane(r_rel, zz)
                for d in range_constexpr(DPW):
                    buf_copy_store(pr4, pl + d * 4, acc[d].bitcast(fx.Int32), unit_elems=4, cache_modifier=PART_SCOPE)
                buf_copy_store(
                    pr2,
                    pl + DPW * 4,
                    Vec.from_elements([bits(big_raw), bits(l_tot)], fx.Int32),
                    unit_elems=2,
                    cache_modifier=PART_SCOPE,
                )
            fx.gpu.barrier()

        if const_expr(TIMING):
            ts.append(stamp())
        if const_expr(S > 1):
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            # padding workgroups take no ticket (unit_ok is workgroup-uniform)
            if (tx == 0) & unit_ok:
                cnt[4] = _atomic_add(arg_cnt + fx.Int64(unit) * 4, 1)
            fx.gpu.barrier()
            last = unit_ok & (fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(cnt[4]))) == S - 1)
            if last:
                for j in range_constexpr(R // RR):
                    r_rel = j * RR + rr_w
                    pls = [part_lane(r_rel, sp) for sp in range_constexpr(S)]
                    mls = [
                        buf_copy_load(pr2, pl + DPW * 4, unit_elems=2, cache_modifier=PART_SCOPE)
                        for pl in pls
                    ]
                    bigs = [as_f32(x[0]) for x in mls]
                    top = bigs[0]
                    for sp in range_constexpr(1, S):
                        top = maxf_nn(top, bigs[sp])
                    top = (top == neg_inf).select(fx.Float32(0.0), top)
                    ws = [fx.Float32(fx.math.exp2(b - top)) for b in bigs]
                    l_all = fx.Float32(0.0)
                    for sp in range_constexpr(S):
                        l_all = l_all + as_f32(mls[sp][1]) * ws[sp]
                    inv_all = (l_all > fx.Float32(0.0)).select(v_scale / l_all, fx.Float32(0.0))
                    acc_all = [zero4 for _ in range_constexpr(DPW)]
                    for sp in range_constexpr(S):
                        wv = Vec.filled(4, ws[sp], fx.Float32)
                        for d in range_constexpr(DPW):
                            part = buf_copy_load(
                                pr4, pls[sp] + d * 4, unit_elems=4, cache_modifier=PART_SCOPE
                            ).bitcast(fx.Float32)
                            acc_all[d] = acc_all[d] + part * wv
                    store_out(acc_all, inv_all, row0 + r_rel)
                if tx == 0:
                    global_i32_ptr(arg_cnt + fx.Int64(unit) * 4)[0] = fx.Int32(0)

        if const_expr(TIMING):
            ts.append(stamp())
            if tx == 0:
                # per workgroup: start, setup done, loop done, merge done, end, n_e, (xcd, unit)
                dbg = arg_dbg + fx.Int64(bid) * 64
                for k in range_constexpr(5):
                    fx.inttoptr(
                        fx.PointerType.get(T.i64, address_space=fx.AddressSpace.Global, alignment=8),
                        dbg + k * 8,
                    )[0] = ts[k]
                global_i32_ptr(dbg + 40)[0] = n_e
                global_i32_ptr(dbg + 44)[0] = unit_ok.select(unit, fx.Int32(-1))

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_cache: fx.Int64,
        arg_topk: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_out: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_part: fx.Int64,
        arg_cnt: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_units: fx.Int32,
        i32_rows: fx.Int32,
        i32_H: fx.Int32,
        i32_hkv: fx.Int32,
        i32_grid_x: fx.Int32,
        i32_topk_hs: fx.Int32,
        i32_topk_rs: fx.Int32,
        i32_bt_s: fx.Int32,
        i32_page_b: fx.Int32,
        f32_c: fx.Float32,
        arg_dbg: fx.Int64,
        arg_score: fx.Int64,
        i32_score_hs: fx.Int32,
        i32_score_rs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_units,
            i32_hkv,
            i32_topk_hs,
            i32_topk_rs,
            i32_bt_s,
            i32_H,
            i32_rows,
            arg_seq,
            arg_topk,
            arg_bt,
            arg_q,
            arg_cache,
            arg_ks,
            arg_vs,
            arg_out,
            arg_part,
            arg_cnt,
            i64_part_bytes,
            i32_page_b,
            f32_c,
            arg_dbg,
            arg_score,
            i32_score_hs,
            i32_score_rs,
        ).launch(grid=(i32_grid_x, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch


@functools.cache
def _get_kernel(ql, rows, splits, topk, prefetch, lazy_t, timing, select_cols, local_blocks, init_blocks,
                write_topk, l_mfma, ablate, req_xcd, device_index):
    return compile_q8(ql, rows, splits, topk, prefetch, lazy_t, timing, select_cols, local_blocks,
                      init_blocks, write_topk, l_mfma, ablate, req_xcd)


def parse_cfg(cfg: str) -> dict:
    """'RxS[-lm][-tN][-rx][-pN][-a<x+y>]' -> launch kwargs (bench / test helper)."""
    parts = cfg.split("-")
    r, s = (int(x) for x in parts[0].split("x"))
    kw = dict(rows_per_wg=r, splits=s)
    for f in parts[1:]:
        if f.startswith("t"):
            kw["lazy_t"] = float(f[1:])
        elif f == "lm":
            kw["l_mfma"] = True
        elif f == "rx":
            kw["req_xcd"] = True
        elif f.startswith("a"):
            kw["ablate"] = f[1:]
        elif f.startswith("p"):
            kw["prefetch"] = int(f[1:])
        else:
            raise ValueError(cfg)
    return kw


PART_FLOATS = (PART_CAP + 1) * 64 * PART_LANE_F


@functools.cache
def _workspace(device_index, slot=0):
    """Split scratch; one ``slot`` per launch that may run concurrently with another."""
    refuse_alloc_in_capture("q8 split scratch")
    dev = torch.device("cuda", device_index)
    unit_scale(device_index)
    return (
        torch.empty(PART_FLOATS, dtype=torch.float32, device=dev),
        torch.zeros(MAX_UNITS + 1, dtype=torch.int32, device=dev),
    )


def default_config(reqs: int, ql: int, cus: int = 256):
    """(rows per workgroup, splits) by request count: one row per workgroup, split so the
    grid reaches ~2 workgroups per CU (measured, MI355X, notes/attn-decode.md)."""
    rows = reqs * ql
    if rows >= cus:
        return 1, 1
    if rows >= cus // 2:
        return 1, 2
    return 1, 4


def tuned_kwargs(reqs: int, ql: int) -> dict:
    """The serving launch options: the measured table plus the row sum on the MFMA, the
    lazy rescale (the reference max moves only when a block's max exceeds it by > 12 in
    log2 units, so P <= 2^12 stays far inside f16) and a request's units on one XCD."""
    r, s = default_config(reqs, ql)
    return dict(rows_per_wg=r, splits=s, l_mfma=True, lazy_t=12.0, req_xcd=True)


def alloc_workspace(device, slots: int = 2) -> None:
    """Allocate the split scratch (and the unit scale) now, outside any graph capture."""
    device = torch.device(device)
    for slot in range(slots):
        _workspace(device.index if device.index is not None else torch.cuda.current_device(), slot)


def sparse_pa_nvfp4_decode_q8(
    q: torch.Tensor,  # [rows, Hq, 128] bf16, rows = reqs * query_len
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [Hkv, >= rows, topk] int32
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    query_len: int = 8,
    rows_per_wg: int | None = None,
    splits: int | None = None,
    prefetch: int = 1,
    lazy_t: float = 0.0,
    timing: torch.Tensor | None = None,  # int64 [grid, 8]: per-workgroup phase stamps (debug)
    index_score: torch.Tensor | None = None,  # [Hkv, >= rows, width] fp32: select in-kernel
    local_blocks: int = 1,
    init_blocks: int = 0,
    write_topk: bool = False,  # with index_score: also store the selection into topk_idx
    l_mfma: bool = False,
    ablate: str = "",
    req_xcd: bool = False,
    workspace_slot: int = 0,  # concurrent launches (two streams) need different slots
) -> torch.Tensor:
    """With ``index_score`` the kernel selects each row's top-``topk_idx.shape[-1]`` itself
    (bit-identical to aiter's ``topk_per_row_small_k`` over the row's ``position // 128 + 1``
    live columns, ``local_blocks`` / ``init_blocks`` forced) and ``topk_idx`` is only written
    (``write_topk``) or not touched."""
    arch = get_gfx_runtime()
    require(arch == "gfx950", f"needs gfx950, got {arch}")
    rows = q.shape[0]
    hkv = topk_idx.shape[0]
    topk = topk_idx.shape[-1]
    if out is None:
        out = torch.empty_like(q)
    if rows == 0:
        return out
    group = check_operands(q, out, kv_cache, topk_idx, block_table)
    select_cols = 0
    if index_score is not None:
        require(index_score.dtype == torch.float32 and index_score.stride(2) == 1, "index_score fp32")
        require(index_score.shape[0] == hkv and index_score.shape[1] >= rows, "index_score shape")
        require(index_score.shape[2] <= CAP, f"index_score wider than {CAP} blocks")
        select_cols = index_score.shape[2]
    require(group == 16, "16 query heads per kv head")
    reqs = seq_lens.shape[0]
    require(rows == reqs * query_len, "rows must be reqs * query_len")
    require(block_table.shape[1] <= CAP, f"block table wider than {CAP} blocks")
    if rows_per_wg is None or splits is None:
        r0, s0 = default_config(reqs, query_len)
        rows_per_wg = rows_per_wg or r0
        splits = splits or s0
    units = reqs * (query_len // rows_per_wg) * hkv
    require(units <= MAX_UNITS, "too many units")
    wpr = NW
    require(
        splits == 1 or units * splits * rows_per_wg * wpr <= PART_CAP,
        "split scratch too small for this batch",
    )
    dev = q.device.index
    ks, vs, c = host_scales(q, k_scale, v_scale, sm_scale)
    part, cnt = _workspace(dev, workspace_slot)
    if req_xcd:
        grid_x = -(-reqs // XCDS) * XCDS * (units // reqs) * splits
    else:
        grid_x = -(-units // XCDS) * XCDS * splits
    with torch.cuda.device(q.device):
        _run_compiled(
            _get_kernel(
                query_len, rows_per_wg, splits, topk, prefetch, float(lazy_t), timing is not None, select_cols,
                local_blocks, init_blocks, bool(write_topk), bool(l_mfma), ablate, bool(req_xcd), dev,
            ),
            q.data_ptr(),
            kv_cache.data_ptr(),
            topk_idx.data_ptr(),
            block_table.data_ptr(),
            seq_lens.data_ptr(),
            out.data_ptr(),
            ks.data_ptr(),
            vs.data_ptr(),
            part.data_ptr(),
            cnt.data_ptr(),
            part.numel() * 4,
            units,
            rows,
            q.shape[1],
            hkv,
            grid_x,
            topk_idx.stride(0),
            topk_idx.stride(1),
            block_table.stride(0),
            kv_cache.stride(0),
            c,
            0 if timing is None else timing.data_ptr(),
            0 if index_score is None else index_score.data_ptr(),
            0 if index_score is None else index_score.stride(0),
            0 if index_score is None else index_score.stride(1),
            torch.cuda.current_stream(q.device),
        )
    return out
