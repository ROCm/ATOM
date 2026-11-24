# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The top-16 block ids of ragged prefill score rows (no page table) for a fixed set of window tiles, row and window
lengths at runtime; ``row_top16`` is one wave's selector, also run by the four-wave and prologue selectors."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .kernel import _ballot, _below, _comp, _fold_nan, _imax, _key, _one, _popc, _s32

BLK = 128  # tokens per block
TOPK = 16
PREFILL_STRIPS = (4, 8, 16, 20, 26)
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # block index bits (slots * 64 <= 8192)
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000
# Bits of the 16th lane maximum that bound the candidates: 16 keep ~18 per row on Gaussian scores.
BOUND_BITS = 16
# The prefill ids' bound (12-16 swept, 4-16K rows, 0-256K context): 14 as fast as 16 on long rows, -1% at 0 prefix.
PREFILL_BOUND_BITS = 14
# Strips of the first window loaded before the row's length is known; the rest
# only up to the row's live end (a 32K row has 4 live strips of the 26).
FIRST_STRIPS = 4


def _rsrc(base, nbytes):
    """A buffer over ``nbytes`` from the 64-bit address ``base``: every tensor is rebased per row (bases in i64, the
    operands widened before they multiply), so offsets stay small at any tensor size."""
    return buffer_ops.create_buffer_resource_from_addr(
        base, num_records_bytes=fx.Int64(nbytes)
    )


def cand_dwords(strips: int) -> int:
    """LDS dwords of one row's candidate pairs (``row_top16``'s ``cand``)."""
    return 2 * (CAP + 8 + 64)


SEL_DWORDS = 4 * TOPK  # LDS dwords of one row's running top 16 (``sel``)


def row_top16(
    strips,
    rank_owned_order,
    lane,
    score_row,
    width,
    nblk_fn,
    cand_ptr,
    cand_total,
    sel_ptr,
    sel_total,
    cbase,
    sbase,
    out_row,
    sync,
    bound_bits=BOUND_BITS,
):
    """One wave's top 16 of one score row (``strips`` x 64 blocks per window): ``score_row`` the row's i64 address,
    ``width`` (i32) its readable blocks, ``nblk_fn()`` its block count (called once the first strips are in flight),
    the row's LDS at pair ``cbase`` of ``cand_ptr`` and quad ``sbase`` of ``sel_ptr`` (dwords ``cand_total`` /
    ``sel_total`` in all), the 16 ids (-1 past the row's count) to ``out_row``; ``sync()`` orders the wave's LDS
    writes before its reads of other lanes' (a workgroup barrier when the wave is the workgroup, else a wave-local
    wait)."""
    STRIPS = strips
    COUNT_BITS = STRIPS.bit_length()  # a lane's count of kept keys is <= STRIPS
    FIRST = min(FIRST_STRIPS, STRIPS)
    BITS = bound_bits
    WIN = STRIPS * 64
    RANK_BLOCK = 8
    # running / window candidates, rank-loop padding, then one discard slot per lane; an overflowing window rebuilds
    # its candidates exactly
    TRASH = CAP + RANK_BLOCK
    assert cand_dwords(strips) == 2 * (TRASH + 64)

    def at(base, i):
        return i if isinstance(base, int) and base == 0 else i + base

    # this row's scores, bounded by its width: a window past it reads 0s
    srs = _rsrc(score_row, width.to(fx.Int64) * 4)
    # rank_owned_order: lane l still holds block 64 j + l of a strip, read from its
    # position 64 j + 16 (l % 4) + l // 4
    pos = (((lane & 3) << 4) | (lane >> 2)) if rank_owned_order else lane

    def load_strips(srs, wbase, j0, j1):
        """Strips j0..j1-1 of a window's score bits (past the row: masked as
        keys, or 0)."""
        return [
            fx.Int32(
                buffer_ops.buffer_load(
                    srs,
                    wbase + (j * 64) + pos,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )
            for j in range_constexpr(j0, j1)
        ]

    def load_window(srs, wbase):
        return Vec.from_elements(load_strips(srs, wbase, 0, STRIPS), fx.Int32)

    # the first window's first strips go out before the row's length is known
    head = load_strips(srs, fx.Int32(0), 0, FIRST)
    nblk = nblk_fn()
    nlive = (nblk < width).select(nblk, width)
    # later windows: the row's live blocks only (no dead-tail reads). In rank_owned_order a live block of the last
    # 64-block chunk can sit past nlive (block 64 c + 4 j + r at 64 c + 16 r + j), so the bound is that chunk's end;
    # the positions there no rank wrote hold dead blocks, masked by their block index
    live_end = ((nlive + 63) // 64) * 64 if rank_owned_order else nlive
    live_end = (live_end < width).select(live_end, width)
    srs_live = _rsrc(score_row, live_end.to(fx.Int64) * 4)
    # the rest of the first window: the row's live strips only
    bits = Vec.from_elements(
        head + load_strips(srs_live, fx.Int32(0), FIRST, STRIPS), fx.Int32
    )

    def view(ptr, n_dw, w):
        return fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
            fx.make_layout(w, 1),
        )

    pairs = view(cand_ptr, cand_total, 2)
    quads = view(cand_ptr, cand_total, 4)
    sels = view(sel_ptr, sel_total, 4)
    qbase = cbase // 2 if isinstance(cbase, int) else cbase >> 1
    atom64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    atom128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def lds_put(atom, v, w, vals):
        r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
        r.store(Vec.from_elements(vals, fx.Int32))
        fx.copy(atom, r, v)

    def lds_get(atom, v, w):
        r = fx.make_rmem_tensor(fx.make_layout(w, 1), fx.Int32)
        fx.copy(atom, v, r)
        x = Vec(r.load())
        return [fx.Int32(x[e]) for e in range_constexpr(w)]

    def put_pair(pos, i, k):
        lds_put(atom64, fx.slice(pairs, (None, at(cbase, pos))), 2, [i, k])

    def wave_sum(c):
        """Sum over the wave of a per-lane count in [0, STRIPS]."""
        t = fx.Int32(0)
        for b in range_constexpr(COUNT_BITS):
            t = t + (_popc(_ballot((c & (1 << b)) != 0)) << b)
        return t

    def compact(keys, wbase, base, pred):
        """The keys ``pred`` keeps, appended at ``base`` in lane-major order
        (only if they fit below CAP); returns the new end either way."""
        oks = [
            _one(pred(keys[j], wbase + (j * 64) + lane))
            for j in range_constexpr(STRIPS)
        ]
        cnt = oks[0]
        for j in range_constexpr(1, STRIPS):
            cnt = cnt + oks[j]
        pos = fx.Int32(base)
        end = fx.Int32(base)
        for b in range_constexpr(6):  # cnt <= 32
            bm = _ballot((cnt & (1 << b)) != 0)
            pos = pos + (_below(bm, 0) << b)
            end = end + (_popc(bm) << b)
        # the wave-uniform count includes every kept key; an overflow skips the writes and the caller rebuilds
        if end <= CAP:
            for j in range_constexpr(STRIPS):
                put_pair(
                    (oks[j] != 0).select(pos, TRASH + lane),
                    wbase + (j * 64) + lane,
                    keys[j],
                )
                pos = pos + oks[j]
        return end

    def exact_cut(keys, wbase):
        """The window's 16th key v and the index cut ie among the keys tied
        with it: (k > v) | (k == v & i >= ie) keeps exactly its top 16."""
        pe = fx.Int32(0)
        b = fx.Int32(31)
        while b >= 0:
            t = (pe | (fx.Int32(1) << b)) ^ INT_MIN
            c = fx.Int32(0)
            for j in range_constexpr(STRIPS):
                c = c + _one(keys[j] >= t)
            pe = (wave_sum(c) >= TOPK).select(t ^ INT_MIN, pe)
            b = b - 1
        v = pe ^ INT_MIN
        c = fx.Int32(0)
        for j in range_constexpr(STRIPS):
            c = c + _one(keys[j] > v)
        need = TOPK - wave_sum(c)
        ie = fx.Int32(0)
        b = fx.Int32(IDX_BITS - 1)
        while b >= 0:
            t = ie | (fx.Int32(1) << b)
            c = fx.Int32(0)
            for j in range_constexpr(STRIPS):
                c = c + _one((keys[j] == v) & ((wbase + (j * 64) + lane) >= t))
            ie = (wave_sum(c) >= need).select(t, ie)
            b = b - 1
        return v, ie

    def lane_max_bound(m):
        p = fx.Int32(0)
        for d in range_constexpr(BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= TOPK).select(t, p)
        return p ^ INT_MIN

    def window(bits, wbase, nrun, tprev):
        """One window of keys merged into the running top 16 (sel)."""
        keys = []
        for j in range_constexpr(STRIPS):
            live = (wbase + (j * 64) + lane) < nlive
            keys.append(live.select(_key(bits[j]), fx.Int32(INT_MIN)))
        m = keys[0]
        for j in range_constexpr(1, STRIPS):
            m = _imax(m, keys[j])
        # the first window is bounded by its lane maxima, a later one by the
        # running 16th
        tw = fx.Int32(INT_MIN)
        if nrun < TOPK:
            tw = lane_max_bound(m)
        ta = _imax(tw, tprev)
        total = compact(keys, wbase, nrun, lambda k, i: k >= ta)
        # NaN keys are not folded yet: exact only while 16 keys above -inf
        # bound the window and no NaN sits above +inf
        nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
        if (total > CAP) | (ta <= KEY_NEG_INF) | nan_hi:
            wl = nlive - wbase
            if wl <= CAP - nrun:
                # every live block of the window is a candidate (strip 0)
                put_pair(
                    (lane < wl).select(nrun + lane, TRASH + lane),
                    wbase + lane,
                    _fold_nan(keys[0]),
                )
                total = nrun + wl
            else:
                # NaNs as -inf, re-bounded by the lane maxima; the exact
                # search only if that still overflows
                fixed = []
                for j in range_constexpr(STRIPS):
                    live = (wbase + (j * 64) + lane) < nlive
                    fixed.append(live.select(_fold_nan(keys[j]), fx.Int32(INT_MIN)))
                mf = fixed[0]
                for j in range_constexpr(1, STRIPS):
                    mf = _imax(mf, fixed[j])
                tf = _imax(lane_max_bound(mf), tprev)
                total = compact(fixed, wbase, nrun, lambda k, i: k >= tf)
                if total > CAP:
                    v, ie = exact_cut(fixed, wbase)
                    total = compact(
                        fixed,
                        wbase,
                        nrun,
                        lambda k, i: (k > v) | ((k == v) & (i >= ie)),
                    )
        if lane < RANK_BLOCK:
            put_pair(total + lane, fx.Int32(0), fx.Int32(INT_MIN))
        sync()
        mi, mk = lds_get(atom64, fx.slice(pairs, (None, at(cbase, lane))), 2)
        mw = _comp(mk, mi)
        rank = fx.Int32(0)
        for g in range_constexpr(CAP // RANK_BLOCK):
            if total > RANK_BLOCK * g:
                es = [
                    lds_get(
                        atom128,
                        fx.slice(quads, (None, at(qbase, RANK_BLOCK // 2 * g + u))),
                        4,
                    )
                    for u in range_constexpr(RANK_BLOCK // 2)
                ]
                # all of a block's reads in flight before the compares
                fx.rocdl.sched_barrier(0)
                for e in es:
                    rank = rank + _one(_comp(e[1], e[0]) > mw)
                    rank = rank + _one(_comp(e[3], e[2]) > mw)
        if (lane < total) & (rank < TOPK):
            lds_put(atom128, fx.slice(sels, (None, at(sbase, rank))), 4, [mi, mk, 0, mi])
        return (total < TOPK).select(total, fx.Int32(TOPK))

    wbase = fx.Int32(0)
    nrun = fx.Int32(0)
    tprev = fx.Int32(INT_MIN + 1)  # dead keys are INT_MIN
    while (wbase == 0) | (wbase < nlive):
        if wbase > 0:
            bits = load_window(srs_live, wbase)
            # the running top 16 back into cand; its 16th key bounds the window
            sync()
            r = lds_get(atom128, fx.slice(sels, (None, at(sbase, lane & (TOPK - 1)))), 4)
            if lane < nrun:
                put_pair(lane, r[0], r[1])
            k15 = fx.Int32(fx.rocdl.readlane(T.i32, r[1], fx.Int32(TOPK - 1)))
            tprev = (nrun >= TOPK).select(k15, fx.Int32(INT_MIN + 1))
            sync()
        nrun = window(
            [fx.Int32(bits[j]) for j in range_constexpr(STRIPS)],
            wbase,
            nrun,
            tprev,
        )
        wbase = wbase + WIN

    # outputs
    sync()
    r = lds_get(atom128, fx.slice(sels, (None, at(sbase, lane & (TOPK - 1)))), 4)
    blk = (lane < nrun).select(r[0], fx.Int32(-1))
    buffer_ops.buffer_store(
        blk,
        _rsrc(out_row, TOPK * 4),
        lane,
        mask=lane < TOPK,
    )


# A module-level helper is not rewritten with the kernel that calls it: its dynamic ifs / whiles become scf ops only
# through the kernel's rewriter, applied here by hand.
row_top16 = ASTRewriter.transform(row_top16)


def wave_sync():
    """``row_top16``'s sync for a wave that shares its workgroup: the wave's LDS writes done, and no code motion
    across (the wave runs in lockstep)."""
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _rocdl_ops.wave_barrier()


def compile_topk_prefill(strips: int, *, decode: bool = False, rank_owned_order: bool = False):
    """One window tile (``strips``; ``decode``: 26 strips over decode rows); shapes are launch arguments.
    ``rank_owned_order`` (decode only): ``score`` rows in the rank-owned scorer's block order, block 64 c + 4 j + r
    at 64 c + 16 r + j."""
    assert strips == 26 if decode else strips in PREFILL_STRIPS
    assert decode or not rank_owned_order
    STRIPS = strips
    CAND = cand_dwords(STRIPS) // 2
    name = f"index_top16_p128_{'decode' if decode else 'prefill'}_indices_w{STRIPS}"
    name += "_rank_owned" if rank_owned_order else ""
    bound_bits = BOUND_BITS if decode else PREFILL_BOUND_BITS

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * CAND, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_query_len: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, h = fx.block_idx.x, fx.block_idx.y
        lane = fx.thread_idx.x
        n64 = fx.Int32(n).to(fx.Int64)
        h64 = fx.Int32(h).to(fx.Int64)

        def scalar_at(base, idx64, nbytes):
            """One i32 at ``base[idx64]`` (a scalar load; 0 if ``nbytes`` is 0)"""
            return fx.Int32(
                buffer_ops.buffer_load(
                    _rsrc(base + idx64 * 4, nbytes),
                    fx.Int32(0),
                    vec_width=1,
                    is_scalar=True,
                )
            )

        def nblk_fn():
            if const_expr(decode):
                req = (fx.Int32(n) // i32_query_len).to(fx.Int64)
                seq = scalar_at(arg_nvp, req, fx.Int64(4))
                kv0 = seq - i32_query_len + fx.Int32(n) % i32_query_len + 1
                kv = (kv0 < 0).select(fx.Int32(0), kv0)
                return (kv + BLK - 1) // BLK
            return scalar_at(arg_nvp, n64, fx.Int64(4))

        row_top16(
            STRIPS,
            rank_owned_order,
            lane,
            arg_score + (h64 * i64_s_head + n64 * i64_s_row) * 4,
            i32_width,
            nblk_fn,
            smem.cand.ptr,
            2 * CAND,
            smem.sel.ptr,
            SEL_DWORDS,
            0,
            0,
            arg_topk + (h64 * i64_t_head + n64 * i64_t_row) * 4,
            fx.gpu.barrier,
            bound_bits,
        )

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        i64_s_head: fx.Int64,
        i64_s_row: fx.Int64,
        i32_width: fx.Int32,
        arg_nvp: fx.Int64,
        i32_query_len: fx.Int32,
        arg_topk: fx.Int64,
        i64_t_head: fx.Int64,
        i64_t_row: fx.Int64,
        i32_rows: fx.Int32,
        i32_heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            i64_s_head,
            i64_s_row,
            i32_width,
            arg_nvp,
            i32_query_len,
            arg_topk,
            i64_t_head,
            i64_t_row,
        ).launch(
            grid=(fx.Int64(i32_rows), fx.Int64(i32_heads), 1),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    # the score / table / length addresses arrive in SGPRs: the first loads do not wait for the kernel-argument load
    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch
