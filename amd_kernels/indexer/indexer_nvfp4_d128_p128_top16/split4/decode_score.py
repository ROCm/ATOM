# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Decode index scorer on the rank-owned index cache, split over four GPUs by context, both exchanges fused (FlyDSL,
gfx950).

Rank ``me`` scores its blocks b % 4 == me (sharded cache: table entry p at slot p // 4; replicated: slot p) for all
four index heads x QL rows (QL = 1 + the draft length, up to 8: column 16 g + c of MFMA group g = head c % 4, row
4 g + c // 4) with ``score_decode.kernel_split4``'s per-tile math. Head h's scores go straight into rank h's receive
buffer [rows, S] in the rank-owned order: per 64-block chunk c, rank r's blocks 64 c + 4 j + r at 64 c + 16 r + j.

Index-q: workgroup g sends this rank's rows of requests g, g + grid, ... to slot ``me`` of every rank's receive
buffer, then a per-request flag; a wave issues its first page's loads before it waits for its requests' flags.

Work: the rank's runs (its 16 blocks of a chunk) of every request, prefix-summed in LDS; workgroup g takes runs g,
g + grid, ..., wave w quarter w (4 blocks) of each, the next page loading tile by tile under the current one. A call
of at most UNIT_MAX / 4 runs (from the device-side lengths) splits each run into four 4-block units instead, wave w
its block w. A wave keeps its request's Q columns in registers and 64 pages' table entries in one VGPR (readlane per
page). The quarters meet in LDS and one wave stores the run: 64 B per (head, row), sc0 sc1 (16 B pieces cost more
over xGMI).

Then every workgroup drains its stores and writes its done word. Scores only (``topk`` False): the last workgroup
collects them, signals every rank and waits for the four, so the receive buffer is complete when the kernel ends.
With the top-k (``topk``, its own receive buffer, all ``MARKER`` between calls): no flags; each row's selector spins on
the row's score words until none is ``MARKER``, selects, and writes ``MARKER`` back; the workgroup before the
selectors collects the done words and bumps the call count. Index-q flags and per-workgroup done words carry the call
count (graph-safe).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .kernel import (
    CNT_SC,
    CNT_WORDS,
    FLAG_BYTES,
    SC_DONE,
    SC_FLAGS,
    SC_IQ,
    SC_MAX_REQS,
    SC_MAX_WG,
    SYS,
    _flag_read,
    _sel4,
)
from ..score_decode.kernel_split4 import (
    BLK,
    CP_CH,
    MT,
    NT,
    PAGE_BYTES,
    PAGE_CODES,
    SENTINEL_INIT,
    SENTINEL_LOCAL,
    WORLD,
    D,
    _xlane_max4,
)
from ..topk.prefill_kernel import SEL_DWORDS
from ..topk.top16_indices_p128_wave4 import row_top16_w4, CAND_DWORDS
from ..score_prefill.kernel import (
    PERM_EVEN,
    PERM_ODD,
    _bits,
    _dequant8,
    _global_i32_ptr,
    _maxf_nn,
    _mfma_fp8_16x16x128,
    _pack8,
    _perm,
    _xlane_max4_pair,
)

# grid by rows per request and request count (graphs fix both): benchmark_rank_owned.py sweeps, 4 GPUs
GRID_SMALL, GRID = 512, 1024
GRID_SMALL_REQS = 32
GRID_BY_QL = {4: ((40, 512), (64, 768))}


def grid_for(num_reqs: int, ql: int) -> int:
    """The scorer's grid for ``num_reqs`` requests of ``ql`` rows."""
    for reqs, grid in GRID_BY_QL.get(ql, ((GRID_SMALL_REQS, GRID_SMALL),)):
        if num_reqs <= reqs:
            return grid
    return GRID


NW = 4  # waves per workgroup: quarter w of every run
QD = CP_CH // NW  # blocks per quarter
WIN = 64  # pages per block-table window (one VGPR)
MAX_REQS = 64 * NW  # one request per thread in the prefix sum
MAX_QL = 8  # rows per request (1 + draft length): two 16-column MFMA groups
HEADS = 4
CHUNK = WORLD * CP_CH  # blocks per chunk: one run of every rank
SC1 = 0x10  # load cache policy: coherent across the XCDs' L2s
# units of 4 blocks (one page per wave) up to this many: more workgroups shorten a wave's page chain; above it their
# 16 B peer stores lose (M 8 at 32K 17.2 -> 13.3 us; 512 units at 512K +6.5%)
UNIT_MAX = 256
# the fused top-k (one waiting workgroup per row, the standalone four-wave selector's body) up to this many rows: the
# waiting workgroups slow the call as their count grows (128K M 40-128 -0.7..-1.0 us vs two launches)
TK_MAX_ROWS = 128
# the fused tail's empty score word: a signaling-NaN payload, which no scorer arithmetic writes (MFMA, max and select
# give quiet NaNs at most)
MARKER = 0x7FA5A5A5


def _uni(x):
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(x)))


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _wave_scan(x, lane, steps):
    """Inclusive prefix sum through the first 2**steps lanes."""
    for d in range_constexpr(steps):
        y = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane - (1 << d)) * 4, x))
        x = x + (lane >= (1 << d)).select(y, fx.Int32(0))
    return x


def compile_score_push(ql: int, topk: bool, small: bool):
    """``ql`` rows per request (1 .. MAX_QL). ``topk``: also this rank's head's top 16 (the indices top-k in
    rank-owned order) in the same launch. ``small``: at most 32 requests; one wave builds their prefixes."""
    assert 1 <= ql <= MAX_QL, ql
    QL = ql
    NR = 32 if small else MAX_REQS
    NR_LOG = NR.bit_length() - 1
    TOPK_FUSED = topk
    NG = (HEADS * QL + 15) // 16  # 16-column MFMA groups
    if topk:
        name = f"nvfp4_fp8mfma_index_score_top16_rank_owned_d128_p128_r{HEADS * QL}h4_wave4_w28w64_e4"
    else:
        name = f"nvfp4_fp8mfma_index_score_decode_rank_owned_push_d128_p128_r{HEADS * QL}h4"
    if small:
        name = name.replace("_decode", "") + "_n32"
    TKS = SEL_DWORDS

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]  # per scale byte: keys of magnitudes 0..3, 4..7
        # [run % 2][group][col][quarter]
        stage: fx.Array[fx.Int32, 2 * NG * 16 * NW * QD, 16]
        incl: fx.Array[fx.Int32, NR]  # inclusive prefix of the run counts
        seq: fx.Array[fx.Int32, NR]
        wsum: fx.Array[fx.Int32, NW]

    if topk:

        @fx.struct
        class Shared:  # noqa: F811 (the same, and the top-k's rows)
            lut: fx.Array[fx.Int32, 512, 16]
            stage: fx.Array[fx.Int32, 2 * NG * 16 * NW * QD, 16]
            incl: fx.Array[fx.Int32, NR]
            seq: fx.Array[fx.Int32, NR]
            wsum: fx.Array[fx.Int32, NW]
            tk_cand: fx.Array[fx.Int32, CAND_DWORDS, 16]
            tk_sel: fx.Array[fx.Int32, TKS, 16]
            tk_tmp: fx.Array[fx.Int32, 260, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_iq: fx.Int64,  # this rank's head: [num_reqs * QL, 128] e4m3
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_d0: fx.Int64,  # head h's scores [rows, S] fp32 (rank h's receive buffer)
        arg_d1: fx.Int64,
        arg_d2: fx.Int64,
        arg_d3: fx.Int64,
        arg_bt: fx.Int64,  # block table [num_reqs, >= max_blocks] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        arg_cnt: fx.Int64,  # device counters (split4.kernel.CNT_*)
        arg_f0: fx.Int64,  # rank p's flag words
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        arg_q0: fx.Int64,  # rank p's index-q receive buffer [4, rmax, 128] e4m3
        arg_q1: fx.Int64,
        arg_q2: fx.Int64,
        arg_q3: fx.Int64,
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,  # bytes between pages
        i32_s: fx.Int32,  # score row stride (elements)
        i64_d_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_me: fx.Int32,
        i32_shift: fx.Int32,  # page slot = table entry >> shift (sharded cache: 2)
        i32_nwg: fx.Int32,  # workgroups (the grid)
        arg_topk: fx.Int64,  # topk: this head's ids [rows, 16] i32 (row stride below)
        i64_t_row: fx.Int64,
        i32_tk_width: fx.Int32,  # topk: blocks per score row it reads
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = _uni(tx // 64)
        col, kl = lane % 16, lane // 16
        st_qoff = col // HEADS  # this lane's column of group g: row 4 g + st_qoff
        st_head = col % HEADS  # of head st_head
        incl_l, seq_l = smem.incl.ptr, smem.seq.ptr

        # this call's count (sc1: coherent across the XCDs' L2s)
        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cnt, num_records_bytes=CNT_WORDS * 4
        )
        calls = (
            _uni(
                buffer_ops.buffer_load(
                    cr, CNT_SC, vec_width=1, dtype=fx.Int32, cache_modifier=SC1
                )
            )
            + 1
        )
        btr = buffer_ops.create_buffer_resource_from_addr(
            arg_bt, num_records_bytes=i64_bt_bytes
        )
        qbytes = fx.Int64(i32_rmax) * (HEADS * D)
        qs = (arg_q0, arg_q1, arg_q2, arg_q3)
        fs = (arg_f0, arg_f1, arg_f2, arg_f3)
        mine = _sel4(i32_me, arg_f0, arg_f1, arg_f2, arg_f3)
        qr = buffer_ops.create_buffer_resource_from_addr(
            _sel4(i32_me, arg_q0, arg_q1, arg_q2, arg_q3), num_records_bytes=qbytes
        )
        drs = [
            buffer_ops.create_buffer_resource_from_addr(
                d, num_records_bytes=i64_d_bytes
            )
            for d in (arg_d0, arg_d1, arg_d2, arg_d3)
        ]

        # e4m3 table; per request its runs of this rank, prefix-summed
        lut_g = _global_i32_ptr(arg_lut)
        lut_l = smem.lut.ptr
        for i in range_constexpr(2):
            lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])
        r_ok = tx < i32_num_reqs
        seq_t = fx.Int32(_global_i32_ptr(arg_seq)[r_ok.select(tx, fx.Int32(0))])
        seq_t = r_ok.select(seq_t, fx.Int32(0))
        # this rank's index-q rows of requests pid, pid + nwg, ... to every rank
        # (slot me); the flags follow once the stores are done (before the pages)
        ir = buffer_ops.create_buffer_resource_from_addr(
            arg_iq, num_records_bytes=fx.Int64(i32_num_reqs) * (QL * D)
        )
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            v = buffer_ops.buffer_load(
                ir,
                (rs * QL * D + lane * 16) // 4,
                vec_width=4,
                dtype=fx.Int32,
                mask=lane < QL * D // 16,
            )
            for p in range_constexpr(WORLD):
                buffer_ops.buffer_store(
                    v,
                    buffer_ops.create_buffer_resource_from_addr(
                        qs[p], num_records_bytes=qbytes
                    ),
                    ((i32_me * i32_rmax + rs * QL) * D + lane * 16) // 4,
                    mask=lane < QL * D // 16,
                    cache_modifier=SYS,
                )
            rs = rs + i32_nwg
        nb_t = (seq_t + (BLK - 1)) // BLK
        runs_t = (nb_t > i32_me).select(
            (nb_t - i32_me + (CHUNK - 1)) // CHUNK, fx.Int32(0)
        )
        x = _wave_scan(runs_t, lane, min(NR_LOG, 6))
        if const_expr(small):
            # One wave holds all request prefixes. Other waves join the single
            # barrier before they read them; no cross-wave prefix stage.
            if tx < NR:
                seq_l[tx] = seq_t
                incl_l[tx] = x
            fx.gpu.barrier()
        else:
            if lane == 63:
                smem.wsum.ptr[wave] = x
            seq_l[tx] = seq_t
            fx.gpu.barrier()
            for w in range_constexpr(NW - 1):
                x = x + (wave > w).select(fx.Int32(smem.wsum.ptr[w]), fx.Int32(0))
            incl_l[tx] = x
            fx.gpu.barrier()

        # this workgroup's items: pid, pid + grid, ... (the busy workgroups stay the first ones, one per CU; a
        # contiguous split spreads them and stacks some CUs: 16 requests 15.0 -> 18.5 us)
        total = _uni(incl_l[NR - 1])
        unit = total * NW <= UNIT_MAX

        lut8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
            ),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        stage16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.stage.ptr),
                fx.make_layout(2 * NG * 16 * NW * QD, 1),
            ),
            fx.make_layout(4, 1),
        )
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        stage4 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.stage.ptr),
                fx.make_layout(2 * NG * 16 * NW * QD, 1),
            ),
            fx.make_layout(1, 1),
        )
        atom4 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        code_off = (col * 64 + kl * 16) // 4  # dwords
        # token col's group scales 2 kl, 2 kl + 1: half kl % 2 of a dword (a u16
        # load gets its sign extension folded through the loop phi, which waits
        # for the whole next page at the loop head)
        scale_off = col * 2 + kl // 2  # dwords
        # v_perm selectors: scale byte 2 (kl % 2) (+ 1) of the dword, zeros above
        scale_sel = (kl % 2) * 2 + 0x0C0C0C00

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 16 B of codes (token 16 t + col, dims
            32 kl ..) and its group scales; ok false: no memory access"""
            c = Vec(
                buffer_ops.buffer_load(
                    kr,
                    code_off,
                    vec_width=4,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=t * 1024,
                    cache_modifier=NT,
                )
            )
            sc = fx.Int32(
                buffer_ops.buffer_load(
                    kr,
                    scale_off,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=PAGE_CODES + t * 128,
                    cache_modifier=NT,
                )
            )
            return c, sc

        def dequant_tile(c, sc):
            """The A fragment of a tile (e4m3 keys, even dims then odd dims per 8)"""
            t0 = lut_load(_perm(sc, sc, scale_sel))
            t1 = lut_load(_perm(sc, sc, scale_sel + 1))
            f = []
            for e in range_constexpr(4):
                lo, hi = _dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        # rows past QL in the last group: no Q load, no store
        row_ok = [4 * g + st_qoff < QL for g in range_constexpr(NG)]

        def q_load(r, ok):
            """Per group this lane's 32 B of Q (row r * QL + 4 g + st_qoff, head
            st_head, dims 32 kl ..), raw"""
            out = []
            for g in range_constexpr(NG):
                off = (
                    (st_head * i32_rmax + r * QL + 4 * g + st_qoff) * D + kl * 32
                ) // 4
                okg = ok
                if const_expr(4 * g + 4 > QL):
                    okg = ok & row_ok[g]
                for hf in range_constexpr(2):
                    out.append(
                        Vec(
                            buffer_ops.buffer_load(
                                qr, off + hf * 4, vec_width=4, dtype=fx.Int32, mask=okg
                            )
                        )
                    )
            return out

        def q_frag(a, b):
            """B operand: the 32 B in the key's dim order (even dims, odd dims per 8)"""
            halves = []
            for v in (a, b):
                d = [fx.Int32(Vec(v)[e]) for e in range_constexpr(4)]
                halves.append(
                    Vec.from_elements(
                        [
                            _perm(d[1], d[0], PERM_EVEN),
                            _perm(d[1], d[0], PERM_ODD),
                            _perm(d[3], d[2], PERM_EVEN),
                            _perm(d[3], d[2], PERM_ODD),
                        ],
                        fx.Int32,
                    )
                )
            return _pack8(halves[0], halves[1])

        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))
        sent_init = fx.Float32(SENTINEL_INIT)
        sent_local = fx.Float32(SENTINEL_LOCAL)

        def reduce(v, t, run, masked, lim):
            for e in range_constexpr(4):
                x = v[e]
                if const_expr(masked):
                    x = (lim >= t * 16 + e).select(x, neg_inf)
                run = _maxf_nn(run, x)
            return run

        def page_pass(cs, ss, qf0, qf1, krn, okn, masked, lim):
            """Per group the max over the page's tokens of this lane's column; tile
            t of the next page loads into tile t's freed registers (the barriers
            keep LLVM from hoisting the loads, which copies the carried registers
            and waits at the latch). Only the reduction depends on the causal
            mask: a branch around the loads lets LLVM hoist every tile's dequant
            out of it."""
            accs = []
            nxt = []
            for t in range_constexpr(MT):
                f = dequant_tile(cs[t], ss[t])
                nxt.append(load_tile(krn, okn, t))
                if const_expr(NG == 2):
                    accs.append(
                        [
                            _mfma_fp8_16x16x128(f, qf0, zero4),
                            _mfma_fp8_16x16x128(f, qf1, zero4),
                        ]
                    )
                else:
                    accs.append([_mfma_fp8_16x16x128(f, qf0, zero4)])
                fx.rocdl.sched_barrier(0)
            mx0 = neg_inf
            mx1 = neg_inf
            if masked:
                run0 = neg_inf
                run1 = neg_inf
                for t in range_constexpr(MT):
                    run0 = reduce(accs[t][0], t, run0, True, lim)
                    if const_expr(NG == 2):
                        run1 = reduce(accs[t][1], t, run1, True, lim + 4)
                if const_expr(NG == 2):
                    mx0, mx1 = _xlane_max4_pair(run0, run1)
                else:
                    mx0 = _xlane_max4(run0)
            else:
                run0 = neg_inf
                run1 = neg_inf
                for t in range_constexpr(MT):
                    run0 = reduce(accs[t][0], t, run0, False, None)
                    if const_expr(NG == 2):
                        run1 = reduce(accs[t][1], t, run1, False, None)
                if const_expr(NG == 2):
                    mx0, mx1 = _xlane_max4_pair(run0, run1)
                else:
                    mx0 = _xlane_max4(run0)
            return mx0, mx1, nxt

        def row_value(mx, seq, q_off, b):
            """AITER's sentinels on block b of the row at q_off (a block both init
            and local is local)"""
            v1 = seq - QL + q_off + 1  # the row's visible tokens
            rb = ((v1 < 0).select(fx.Int32(0), v1) + (BLK - 1)) // BLK
            l0 = rb - i32_local
            local0 = (l0 < 0).select(fx.Int32(0), l0)
            vis = b < rb
            v = (vis & (b < i32_init)).select(sent_init, mx)
            return (vis & (b >= local0)).select(sent_local, v)

        fx.rocdl.s_waitcnt(vmcnt=0)
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            for p in range_constexpr(WORLD):
                buffer_ops.buffer_store(
                    calls,
                    buffer_ops.create_buffer_resource_from_addr(
                        fs[p], num_records_bytes=FLAG_BYTES
                    ),
                    SC_IQ + i32_me * SC_MAX_REQS + rs,
                    mask=lane == 0,
                    cache_modifier=SYS,
                )
            rs = rs + i32_nwg

        def scan(UNIT):
            """The pages of this workgroup's items: units (``UNIT``, a quarter of
            a run, wave w its block w) or runs (wave w its quarter w)."""
            qdv = 1 if UNIT else QD  # pages per item and wave
            items = total * NW if UNIT else total
            nitems = (pid < items).select((items - pid + i32_nwg - 1) // i32_nwg, 0)
            nslots = nitems * qdv  # pages of this wave
            w0 = fx.Int32(0)
            while w0 < nslots:
                # the window's pages, one per lane: page s is item pid + (s / qdv) *
                # grid's: a run's block 4 w + s % 4, or unit u's block 4 u + w (run
                # k = item / 4), of request rj (binary search in the prefix)
                s = w0 + lane
                it = pid + (s if UNIT else s >> 2) * i32_nwg
                k = (it >> 2) if UNIT else it
                rj = fx.Int32(0)
                for bs in range_constexpr(NR_LOG):
                    step = NR >> (bs + 1)
                    rj = (fx.Int32(incl_l[rj + step - 1]) <= k).select(rj + step, rj)
                valid = s < nslots
                rj = valid.select(rj, fx.Int32(0))
                excl = (rj > 0).select(fx.Int32(incl_l[_max(rj - 1, 0)]), fx.Int32(0))
                seqj = fx.Int32(seq_l[rj])
                nblkj = (seqj + (BLK - 1)) // BLK
                in_run = ((it & 3) * NW + wave) if UNIT else (wave * QD + (s & 3))
                bj = (k - excl) * CHUNK + in_run * WORLD + i32_me
                frow = seqj - (QL - 1)
                rbf = ((frow < 0).select(fx.Int32(0), frow) + (BLK - 1)) // BLK
                ll = nblkj - i32_local
                loc_last = (ll < 0).select(fx.Int32(0), ll)
                # blocks every row sees whole: init / local ones only need their sentinel
                skip = (bj < rbf) & ((bj < i32_init) | (bj >= loc_last))
                need = valid & (bj < nblkj) & ~skip
                pgj = fx.Int32(
                    buffer_ops.buffer_load(
                        btr,
                        need.select(rj * i32_bt_stride + bj, fx.Int32(0)),
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                pgj = need.select(pgj >> i32_shift, fx.Int32(-1))
                n = _min(nslots - w0, WIN)

                p_0 = fx.Int32(fx.rocdl.readlane(T.i32, pgj, 0))
                r_0 = fx.Int32(fx.rocdl.readlane(T.i32, rj, 0))
                ok0 = p_0 >= 0
                kr0 = page_rsrc(ok0.select(p_0, fx.Int32(0)))
                fx.rocdl.sched_barrier(0)
                c0, s0 = load_tile(kr0, ok0, 0)
                fx.rocdl.sched_barrier(0)
                c1, s1 = load_tile(kr0, ok0, 1)
                fx.rocdl.sched_barrier(0)
                c2, s2 = load_tile(kr0, ok0, 2)
                fx.rocdl.sched_barrier(0)
                c3, s3 = load_tile(kr0, ok0, 3)
                fx.rocdl.sched_barrier(0)
                c4, s4 = load_tile(kr0, ok0, 4)
                fx.rocdl.sched_barrier(0)
                c5, s5 = load_tile(kr0, ok0, 5)
                fx.rocdl.sched_barrier(0)
                c6, s6 = load_tile(kr0, ok0, 6)
                fx.rocdl.sched_barrier(0)
                c7, s7 = load_tile(kr0, ok0, 7)
                fx.rocdl.sched_barrier(0)
                # the window's requests' index-q rows from every rank: lane i polls
                # source i % 4 of its page's request (lanes 4 m .. 4 m + 3: run m);
                # a unit is one slot, so its lane polls all four sources
                want = lane < n
                word = mine + fx.Int64(
                    (SC_IQ + (0 if UNIT else lane % WORLD) * SC_MAX_REQS + rj) * 4
                )

                def poll():
                    lo = _flag_read(word)
                    if const_expr(UNIT):
                        for p in range_constexpr(1, WORLD):
                            lo = _min(
                                lo, _flag_read(word + fx.Int64(p * SC_MAX_REQS * 4))
                            )
                    return lo

                f = calls
                if want:
                    f = poll()
                while fx.Int64(fx.rocdl.ballot(T.i64, want & (f < calls))) != 0:
                    if want & (f < calls):
                        f = poll()
                qr0 = q_load(r_0, n > 0)
                # the loop head merges this path with the back edge: nothing pending
                fx.rocdl.s_waitcnt(vmcnt=0)
                qa0, qb0 = qr0[0], qr0[1]
                qa1, qb1 = qr0[-2], qr0[-1]
                qf0 = q_frag(qa0, qb0)
                qf1 = qf0
                if const_expr(NG == 2):
                    qf1 = q_frag(qa1, qb1)
                qna0, qnb0, qna1, qnb1 = qa0, qb0, qa1, qb1
                sw = n < 0  # false: Q of the next page already loaded
                v00 = neg_inf
                v01 = neg_inf
                v02 = neg_inf
                v03 = neg_inf
                v10 = neg_inf
                v11 = neg_inf
                v12 = neg_inf
                v13 = neg_inf
                j = fx.Int32(0)
                while j < n:
                    r = fx.Int32(fx.rocdl.readlane(T.i32, rj, j))
                    b = fx.Int32(fx.rocdl.readlane(T.i32, bj, j))
                    seq = fx.Int32(fx.rocdl.readlane(T.i32, seqj, j))
                    if sw:
                        qf0 = q_frag(qna0, qnb0)
                        if const_expr(NG == 2):
                            qf1 = q_frag(qna1, qnb1)
                    jn = j + 1
                    okj = jn < n
                    jc = okj.select(jn, fx.Int32(0))
                    pgn = fx.Int32(fx.rocdl.readlane(T.i32, pgj, jc))
                    rn = fx.Int32(fx.rocdl.readlane(T.i32, rj, jc))
                    okn = okj & (pgn >= 0)
                    krn = page_rsrc(okn.select(pgn, fx.Int32(0)))
                    sw = okj & (rn != r)
                    qrn = q_load(rn, sw)
                    qna0, qnb0 = qrn[0], qrn[1]
                    if const_expr(NG == 2):
                        qna1, qnb1 = qrn[2], qrn[3]
                    fx.rocdl.sched_barrier(0)
                    res = page_pass(
                        (c0, c1, c2, c3, c4, c5, c6, c7),
                        (s0, s1, s2, s3, s4, s5, s6, s7),
                        qf0,
                        qf1,
                        krn,
                        okn,
                        b * BLK + (BLK - 1) > seq - QL,
                        seq - QL + st_qoff - b * BLK - kl * 4,
                    )
                    c0, s0 = res[2][0]
                    c1, s1 = res[2][1]
                    c2, s2 = res[2][2]
                    c3, s3 = res[2][3]
                    c4, s4 = res[2][4]
                    c5, s5 = res[2][5]
                    c6, s6 = res[2][6]
                    c7, s7 = res[2][7]
                    e = fx.Int32(0) if UNIT else (w0 + j) & 3
                    ov0 = row_value(res[0], seq, st_qoff, b)
                    v00 = (e == 0).select(ov0, v00)
                    v01 = (e == 1).select(ov0, v01)
                    v02 = (e == 2).select(ov0, v02)
                    v03 = (e == 3).select(ov0, v03)
                    if const_expr(NG == 2):
                        ov1 = row_value(res[1], seq, st_qoff + 4, b)
                        v10 = (e == 0).select(ov1, v10)
                        v11 = (e == 1).select(ov1, v11)
                        v12 = (e == 2).select(ov1, v12)
                        v13 = (e == 3).select(ov1, v13)
                    done = (j >= 0) if UNIT else (e == QD - 1)
                    if done:
                        # the item's pages meet in LDS ([slot][group][col][wave]: a
                        # unit's 4 B or a run quarter's 16 B per wave); wave m % NW
                        # stores 16 B per lane: a unit's 4 blocks (lanes kl 0) or
                        # quarter kl of the run
                        m = (w0 + j) if UNIT else (w0 + j) >> 2
                        slot = m % 2
                        vsets = [[v00, v01, v02, v03], [v10, v11, v12, v13]]
                        if kl == 0:
                            for g in range_constexpr(NG):
                                base = ((slot * NG + g) * 16 + col) * NW + wave
                                if const_expr(UNIT):
                                    val = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                                    val.store(Vec.from_elements([_bits(vsets[g][0])], fx.Int32))
                                    fx.copy(atom4, val, fx.slice(stage4, (None, base)))
                                else:
                                    val = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                                    val.store(
                                        Vec.from_elements(
                                            [_bits(vv) for vv in vsets[g]], fx.Int32
                                        )
                                    )
                                    fx.copy(atom16, val, fx.slice(stage16, (None, base)))
                        if const_expr(UNIT):
                            fx.rocdl.s_waitcnt(lgkmcnt=0)
                        fx.gpu.barrier()
                        if wave == m % NW:
                            # the 16 B's 4-block group within this rank's 16 of
                            # the chunk
                            qsel = (b % CHUNK) // (NW * WORLD) if UNIT else kl
                            for g in range_constexpr(NG):
                                base = (slot * NG + g) * 16 + col
                                got = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                                fx.copy(
                                    atom16,
                                    fx.slice(
                                        stage16, (None, base if UNIT else base * NW + kl)
                                    ),
                                    got,
                                )
                                off = (
                                    (r * QL + 4 * g + st_qoff) * i32_s
                                    + (b - b % CHUNK)
                                    + i32_me * CP_CH
                                    + qsel * 4
                                )
                                for h in range_constexpr(HEADS):
                                    ok_h = (
                                        (st_head == h) & (kl == 0) if UNIT else st_head == h
                                    )
                                    if const_expr(4 * g + 4 > QL):
                                        ok_h = ok_h & row_ok[g]
                                    buffer_ops.buffer_store(
                                        got.load(),
                                        drs[h],
                                        off,
                                        mask=ok_h,
                                        cache_modifier=SYS,
                                    )
                    j = jn
                w0 = w0 + WIN

        # one code path per split (selects in the page loop cost run calls ~1%)
        if unit:
            scan(True)
        else:
            scan(False)

        # Every global write of the workgroup is an sc0 sc1 store, visible system-wide once drained (a system release
        # per workgroup writes back L2 every time: 512 workgroups +45 us). The last workgroup (dispatched last, so
        # its spinning cannot starve the others) collects the done words, signals every rank and waits for the four.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        own = buffer_ops.create_buffer_resource_from_addr(
            mine, num_records_bytes=FLAG_BYTES
        )
        buffer_ops.buffer_store(
            calls, own, SC_DONE + pid, mask=tx == 0, cache_modifier=SYS
        )
        def collect():
            """Every done word of this rank: thread t polls words t, t + 256, ... in one round trip per poll."""

            def done_min():
                lo = calls
                for i in range_constexpr(SC_MAX_WG // (64 * NW)):
                    w = tx + i * 64 * NW
                    f = _flag_read(
                        mine + fx.Int64((SC_DONE + _min(w, SC_MAX_WG - 1)) * 4)
                    )
                    lo = _min(lo, (w < i32_nwg).select(f, calls))
                return lo

            lo = done_min()
            while fx.Int64(fx.rocdl.ballot(T.i64, lo < calls)) != 0:
                lo = done_min()
            fx.gpu.barrier()

        def collect_and_count():
            """The fused tail's collector: every workgroup of this call has read the count, so bump it."""
            collect()
            buffer_ops.buffer_store(calls, cr, CNT_SC, mask=tx == 0, cache_modifier=SYS)

        # a row's score words: thread tx reads 16 B (positions 4 q .. 4 q + 3) of quad q = tx, tx + 256, ...; the four
        # are rank (tx % 16) // 4's slots of their chunk
        src = (tx % 16) // 4

        def wait_row(row, nb):
            """Spin until every score word the request's pushes write holds no MARKER: rank p writes its slots of the
            chunks c < cdiv(nb - p, 64) (nb: the request's blocks), visible or not, so none lands after the reset."""
            chunks = (nb + 63) // 64
            src_chunks = (nb > src).select((nb - src + 63) // 64, fx.Int32(0))  # chunks this thread's source writes
            rr = buffer_ops.create_buffer_resource_from_addr(row, num_records_bytes=fx.Int64(chunks) * 256)
            wsum = smem.wsum.ptr
            miss = fx.Int32(1)
            while miss != 0:
                m = fx.Int32(0)
                qi = tx
                while qi < chunks * 16:
                    v = Vec(buffer_ops.buffer_load(rr, qi * 4, vec_width=4, dtype=fx.Int32, cache_modifier=SYS))
                    for e in range_constexpr(4):
                        hole = (fx.Int32(v[e]) == MARKER) & (qi // 16 < src_chunks)
                        m = m | hole.select(fx.Int32(1), fx.Int32(0))
                    qi = qi + 64 * NW
                any_w = fx.Int64(fx.rocdl.ballot(T.i64, m != 0)) != 0
                if lane == 0:
                    wsum[wave] = any_w.select(fx.Int32(1), fx.Int32(0))
                # the fence keeps the loads in the loop
                _llvm.FenceOp(_llvm.AtomicOrdering.acq_rel, syncscope="workgroup")
                fx.gpu.barrier()
                miss = fx.Int32(wsum[0]) | fx.Int32(wsum[1]) | fx.Int32(wsum[2]) | fx.Int32(wsum[3])
                fx.gpu.barrier()

        def reset_row(row, nb):
            """MARKER over the row's positions of every chunk its request's pushes can write (chunks < cdiv(nb, 64))."""
            n4 = ((nb + 63) // 64) * 16
            rr = buffer_ops.create_buffer_resource_from_addr(row, num_records_bytes=fx.Int64(n4) * 16)
            mk = Vec.from_elements([fx.Int32(MARKER)] * 4, fx.Int32)
            qi = tx
            while qi < n4:
                buffer_ops.buffer_store(mk, rr, qi * 4, cache_modifier=SYS)
                qi = qi + 64 * NW

        def collect_and_signal():
            """The last workgroup: every done word of this rank, then this rank's
            flag on every rank."""

            collect()
            # workgroup scope: every byte a peer reads (index-q rows, scores, done words) went out as an sc0 sc1
            # store drained before its done word, so a system release (an L2 writeback) guards no data
            _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="workgroup")
            for p in range_constexpr(WORLD):
                fr = buffer_ops.create_buffer_resource_from_addr(
                    fs[p], num_records_bytes=FLAG_BYTES
                )
                buffer_ops.buffer_store(
                    calls, fr, SC_FLAGS + i32_me, mask=tx == 0, cache_modifier=SYS
                )
            buffer_ops.buffer_store(calls, cr, CNT_SC, mask=tx == 0, cache_modifier=SYS)

        def wait_all():
            """Every rank's flag: this rank's receive buffer is complete."""
            if tx < WORLD:
                f = _flag_read(mine + fx.Int64((SC_FLAGS + tx) * 4))
                while f < calls:
                    f = _flag_read(mine + fx.Int64((SC_FLAGS + tx) * 4))
            fx.gpu.barrier()
            # workgroup scope: the receive buffer is uncached memory, no stale L2 lines to invalidate
            _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="workgroup")

        if const_expr(TOPK_FUSED):
            # No flags: the last min(rows, grid - 1) workgroups select, one row each (a smaller grid continues at row
            # + their count); a selector spins until every word its request's pushes write (all four ranks', its own
            # too) holds no MARKER (the receive buffer starts all MARKER), selects, then writes MARKER back over those
            # chunks. No peer pushes the next call's scores before this rank's next index-q rows, so the reset lands
            # first. The workgroup before the selectors collects this rank's done words and bumps the call count.
            rows = i32_num_reqs * QL
            ktk = _min(i32_nwg - 1, rows)
            first = i32_nwg - ktk
            if pid == first - 1:
                collect_and_count()
            if pid >= first:
                t = pid - first
                while t < rows:
                    seq = fx.Int32(seq_l[t // QL])
                    kv0 = seq - QL + t % QL + 1
                    nblk = ((kv0 < 0).select(fx.Int32(0), kv0) + BLK - 1) // BLK
                    row = _sel4(i32_me, arg_d0, arg_d1, arg_d2, arg_d3) + fx.Int64(t) * fx.Int64(i32_s) * 4
                    nb = (seq + BLK - 1) // BLK
                    wait_row(row, nb)
                    row_top16_w4(
                        tx,
                        row,
                        i32_tk_width,
                        nblk,
                        smem.tk_cand.ptr,
                        smem.tk_sel.ptr,
                        smem.tk_tmp.ptr,
                        arg_topk + fx.Int64(t) * i64_t_row * 4,
                    )
                    reset_row(row, nb)
                    t = t + ktk
        else:
            if pid == i32_nwg - 1:
                collect_and_signal()
                wait_all()

    @flyc.jit
    def launch(
        arg_iq: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_d0: fx.Int64,
        arg_d1: fx.Int64,
        arg_d2: fx.Int64,
        arg_d3: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_cnt: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        arg_q0: fx.Int64,
        arg_q1: fx.Int64,
        arg_q2: fx.Int64,
        arg_q3: fx.Int64,
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,
        i32_s: fx.Int32,
        i64_d_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_me: fx.Int32,
        i32_shift: fx.Int32,
        i32_nwg: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_row: fx.Int64,
        i32_tk_width: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_iq,
            arg_kv,
            arg_lut,
            arg_d0,
            arg_d1,
            arg_d2,
            arg_d3,
            arg_bt,
            arg_seq,
            arg_cnt,
            arg_f0,
            arg_f1,
            arg_f2,
            arg_f3,
            arg_q0,
            arg_q1,
            arg_q2,
            arg_q3,
            i32_rmax,
            i64_page_stride,
            i32_s,
            i64_d_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_init,
            i32_local,
            i32_me,
            i32_shift,
            i32_nwg,
            arg_topk,
            i64_t_row,
            i32_tk_width,
        ).launch(grid=(fx.Int64(i32_nwg), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


assert CNT_SC < CNT_WORDS and SC_DONE + SC_MAX_WG <= FLAG_BYTES // 4
