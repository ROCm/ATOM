# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact top-16 block ids of score rows in plain block order, four waves per row: the one-GPU score + top-16 entry's
selector, as a kernel and as a consumer kernel's prologue (``row_top16_prologue_ids``).

Windows of 7 strips per wave up to 1792 blocks (one window, ``row_top16_w4_one``), 16 above (``row_top16_w4_fast``).
The waves share lane maxima and candidate offsets, then split each candidate's rank compares. A row whose candidates
overflow, or whose NaNs / -infs could reach the top 16, restarts ``prefill_kernel.row_top16`` (one wave).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr import rocdl as _rocdl
from ..score_prefill.kernel import _permlane_swap
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from .prefill_kernel import (
    BOUND_BITS,
    INT_MIN,
    KEY_NEG_INF,
    KEY_POS_INF,
    SEL_DWORDS,
    _ballot,
    _below,
    _comp,
    _imax,
    _key,
    _one,
    _popc,
    _rsrc,
    _s32,
    cand_dwords,
    row_top16,
    wave_sync,
)

# strips per wave, chosen from the device-side length: 7 keeps a short row's issue cost low, 16 (64 strips) saves two
# selection passes at 512K
SHORT_STRIPS = 7
LONG_STRIPS = 16
SHORT_BLOCKS = 4 * SHORT_STRIPS * 64
CAND_DWORDS = cand_dwords(28)


def make_row_top16_w4_fast(strips):
    STRIPS = strips

    def row_top16_w4_fast(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre):
        lane, wave = tx % 64, tx // 64
        nlive = (nblk < width).select(nblk, width)
        live_end = ((nlive + 63) // 64) * 64
        live_end = (live_end < width).select(live_end, width)
        srs = _rsrc(score_row, live_end.to(fx.Int64) * 4)
        pos = lane

        def sync():
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()

        def view(ptr, n_dw, w):
            return fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
                fx.make_layout(w, 1),
            )

        pairs, quads = view(cand, CAND_DWORDS, 2), view(cand, CAND_DWORDS, 4)
        atom64, atom128 = (
            fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32),
            fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
        )

        def pair_put(i, blk, key):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            r.store(Vec.from_elements([blk, key], fx.Int32))
            fx.copy(atom64, r, fx.slice(pairs, (None, i)))

        def quad_get(i):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom128, fx.slice(quads, (None, i)), r)
            v = Vec(r.load())
            return [fx.Int32(v[e]) for e in range_constexpr(4)]

        def reduce_waves(v):
            tmp[tx] = v
            sync()
            return _imax(
                _imax(fx.Int32(tmp[lane]), fx.Int32(tmp[64 + lane])),
                _imax(fx.Int32(tmp[128 + lane]), fx.Int32(tmp[192 + lane])),
            )

        nrun, tprev = fx.Int32(0), fx.Int32(INT_MIN + 1)
        wbase, bad = fx.Int32(0), fx.Int32(0)
        while ((wbase == 0) | (wbase < nlive)) & (bad == 0):
            keys = []
            for j in range_constexpr(STRIPS):
                if const_expr(j < len(pre)):
                    # the first window's strips arrived with the kernel's start
                    raw = fx.Int32(buffer_ops.buffer_load(
                        srs, wbase + (j * 4 + wave) * 64 + pos,
                        vec_width=1, dtype=fx.Int32, mask=wbase != 0))
                    raw = (wbase == 0).select(pre[j], raw)
                else:
                    raw = fx.Int32(buffer_ops.buffer_load(
                        srs, wbase + (j * 4 + wave) * 64 + pos,
                        vec_width=1, dtype=fx.Int32))
                keys.append((wbase + (j * 4 + wave) * 64 + lane < nlive).select(
                    _key(raw), fx.Int32(INT_MIN)))
            m = keys[0]
            for j in range_constexpr(1, STRIPS):
                m = _imax(m, keys[j])
            m = reduce_waves(m)
            # Bound: 16 bits of the first window's 16th lane maximum. A later window (rows past 4096 blocks) is
            # bounded by the running 16th key and, when 16 of its lane maxima reach that key (scores that rise
            # toward the row's end), by all 32 bits of its own 16th lane maximum, as in top16_indices_p128_wave4.py.
            tw = fx.Int32(INT_MIN)
            if nrun < 16:
                p = fx.Int32(0)
                for d in range_constexpr(BOUND_BITS):
                    t = p | _s32(1 << (31 - d))
                    count = _popc(_ballot(m >= (t ^ INT_MIN)))
                    p = (count >= 16).select(t, p)
                tw = p ^ INT_MIN
            else:
                if _popc(_ballot(m >= tprev)) >= 16:
                    p = fx.Int32(0)
                    bit = fx.Int32(31)
                    while bit >= 0:
                        t = p | (fx.Int32(1) << bit)
                        p = (_popc(_ballot(m >= (t ^ INT_MIN))) >= 16).select(t, p)
                        bit = bit - 1
                    tw = p ^ INT_MIN
            ta = _imax(tw, tprev)
            oks = [_one(k >= ta) for k in keys]
            cnt = oks[0]
            for j in range_constexpr(1, STRIPS):
                cnt = cnt + oks[j]
            prefix, wtotal = fx.Int32(0), fx.Int32(0)
            for b in range_constexpr(STRIPS.bit_length()):
                mask = _ballot((cnt & (1 << b)) != 0)
                prefix = prefix + (_below(mask, 0) << b)
                wtotal = wtotal + (_popc(mask) << b)
            if lane == 0:
                tmp[256 + wave] = wtotal
            sync()
            offset, total = nrun + prefix, nrun
            for w in range_constexpr(4):
                wt = fx.Int32(tmp[256 + w])
                offset = offset + (wave > w).select(wt, fx.Int32(0))
                total = total + wt
            # every wave sees the same maximum and total; an exceptional window restarts the exact one-wave
            # selector over the whole row
            nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
            bad = ((total > 64) | (ta <= KEY_NEG_INF) | nan_hi).select(
                fx.Int32(1), fx.Int32(0)
            )
            if bad == 0:
                if wave == 0:
                    if lane < nrun:
                        pair_put(lane, fx.Int32(sel[4 * lane]), fx.Int32(sel[4 * lane + 1]))
                    if lane < 8:
                        pair_put(total + lane, fx.Int32(0), fx.Int32(INT_MIN))
                for j in range_constexpr(STRIPS):
                    if oks[j] != 0:
                        pair_put(offset, wbase + (j * 4 + wave) * 64 + lane, keys[j])
                    offset = offset + oks[j]
                sync()
                mi, mk = fx.Int32(cand[2 * lane]), fx.Int32(cand[2 * lane + 1])
                mw = _comp(mk, mi)
                rank = fx.Int32(0)
                # each wave compares two groups of eight; the partial ranks are summed
                for g in range_constexpr(2):
                    start = wave * 8 + g * 32
                    if total > start:
                        es = [quad_get(start // 2 + u) for u in range_constexpr(4)]
                        fx.rocdl.sched_barrier(0)
                        for e in es:
                            rank = rank + _one(_comp(e[1], e[0]) > mw)
                            rank = rank + _one(_comp(e[3], e[2]) > mw)
                tmp[tx] = rank
                sync()
                if wave == 0:
                    r = (
                        fx.Int32(tmp[lane]) + fx.Int32(tmp[64 + lane])
                        + fx.Int32(tmp[128 + lane]) + fx.Int32(tmp[192 + lane])
                    )
                    if (lane < total) & (r < 16):
                        sel[4 * r] = mi
                        sel[4 * r + 1] = mk
                sync()
                nrun = (total < 16).select(total, fx.Int32(16))
                tprev = (nrun >= 16).select(fx.Int32(sel[4 * 15 + 1]), fx.Int32(INT_MIN + 1))
            wbase = wbase + 4 * STRIPS * 64
        if bad == 0:
            if tx < 16:
                blk = (tx < nrun).select(fx.Int32(sel[4 * tx]), fx.Int32(-1))
                buffer_ops.buffer_store(blk, _rsrc(out_row, 64), tx)
        sync()
        return bad

    return ASTRewriter.transform(row_top16_w4_fast)


def _xlane_sum4(x):
    """Sum over lanes L, L^16, L^32, L^48, on every lane."""
    a, b = _permlane_swap(_rocdl.permlane32_swap, x, x)
    s = a + b
    a, b = _permlane_swap(_rocdl.permlane16_swap, s, s)
    return a + b


def row_top16_w4_one(tx, score_row, width, nblk, cand, tmp, out_row, pre):
    """A row of at most SHORT_BLOCKS blocks, its whole window in ``pre``: the window selection of
    ``row_top16_w4_fast`` (16-bit bound of the 16th lane maximum, candidates >= it, exact rank by (key, block)) with
    four barriers instead of six. Candidate c's rank is split over lanes c % 16 + 16 q of wave c // 16 (16 candidates
    each), summed by permlane swaps; id r goes through LDS to one contiguous 64-byte store of wave 0 (per-lane stores
    from four waves cost +0.7-1.1 us per call). Returns ``bad`` as the window loop does (the caller restarts the
    exact one-wave selector)."""
    lane, wave = tx % 64, tx // 64
    nlive = (nblk < width).select(nblk, width)

    def sync():
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()

    def view(ptr, n_dw, w):
        return fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
            fx.make_layout(w, 1),
        )

    pairs, quads = view(cand, CAND_DWORDS, 2), view(cand, CAND_DWORDS, 4)
    atom64, atom128 = (
        fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32),
        fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
    )

    def pair_put(i, blk, key):
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        r.store(Vec.from_elements([blk, key], fx.Int32))
        fx.copy(atom64, r, fx.slice(pairs, (None, i)))

    def quad_get(i):
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy(atom128, fx.slice(quads, (None, i)), r)
        v = Vec(r.load())
        return [fx.Int32(v[e]) for e in range_constexpr(4)]

    # candidate slots past the live count compare below every candidate
    if tx < 64:
        pair_put(tx, fx.Int32(0), fx.Int32(INT_MIN))
    keys = []
    for j in range_constexpr(SHORT_STRIPS):
        keys.append(((j * 4 + wave) * 64 + lane < nlive).select(_key(pre[j]), fx.Int32(INT_MIN)))
    m = keys[0]
    for j in range_constexpr(1, SHORT_STRIPS):
        m = _imax(m, keys[j])
    tmp[tx] = m
    sync()
    m = _imax(
        _imax(fx.Int32(tmp[lane]), fx.Int32(tmp[64 + lane])),
        _imax(fx.Int32(tmp[128 + lane]), fx.Int32(tmp[192 + lane])),
    )
    # both endpoint counts prove this six-bit interval holds the 16th lane maximum; a failed guess runs the whole
    # 16-bit search
    seed = fx.Int32(fx.rocdl.readlane(T.i32, m.ir_value(), fx.Int32(31).ir_value()))
    seed = (seed ^ INT_MIN) & _s32(0xffffffff << 26 & 0xffffffff)
    upper = seed + _s32(1 << 26)
    lo_count = _popc(_ballot(m >= (seed ^ INT_MIN)))
    hi_count = _popc(_ballot(m >= (upper ^ INT_MIN)))
    valid = (lo_count >= 16) & (hi_count < 16) & (upper != 0)
    p = fx.Int32(0)
    if valid:
        p = seed
        for d in range_constexpr(6, BOUND_BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= 16).select(t, p)
    else:
        for d in range_constexpr(BOUND_BITS):
            t = p | _s32(1 << (31 - d))
            count = _popc(_ballot(m >= (t ^ INT_MIN)))
            p = (count >= 16).select(t, p)
    ta = _imax(p ^ INT_MIN, fx.Int32(INT_MIN + 1))
    oks = [_one(k >= ta) for k in keys]
    cnt = oks[0]
    for j in range_constexpr(1, SHORT_STRIPS):
        cnt = cnt + oks[j]
    prefix, wtotal = fx.Int32(0), fx.Int32(0)
    for b in range_constexpr(SHORT_STRIPS.bit_length()):
        mask = _ballot((cnt & (1 << b)) != 0)
        prefix = prefix + (_below(mask, 0) << b)
        wtotal = wtotal + (_popc(mask) << b)
    if lane == 0:
        tmp[256 + wave] = wtotal
    sync()
    offset, total = prefix, fx.Int32(0)
    for w in range_constexpr(4):
        wt = fx.Int32(tmp[256 + w])
        offset = offset + (wave > w).select(wt, fx.Int32(0))
        total = total + wt
    nan_hi = _ballot(m > KEY_POS_INF) != fx.Int64(0)
    bad = ((total > 64) | (ta <= KEY_NEG_INF) | nan_hi).select(fx.Int32(1), fx.Int32(0))
    if bad == 0:
        for j in range_constexpr(SHORT_STRIPS):
            if oks[j] != 0:
                pair_put(offset, (j * 4 + wave) * 64 + lane, keys[j])
            offset = offset + oks[j]
        sync()
        c = wave * 16 + lane % 16
        q = lane // 16
        mi, mk = fx.Int32(cand[2 * c]), fx.Int32(cand[2 * c + 1])
        mw = _comp(mk, mi)
        rank = fx.Int32(0)
        if total <= 32:
            es = [quad_get(q * 4 + u) for u in range_constexpr(4)]
            fx.rocdl.sched_barrier(0)
            for e in es:
                rank = rank + _one(_comp(e[1], e[0]) > mw)
                rank = rank + _one(_comp(e[3], e[2]) > mw)
        else:
            es = [quad_get(q * 8 + u) for u in range_constexpr(8)]
            fx.rocdl.sched_barrier(0)
            for e in es:
                rank = rank + _one(_comp(e[1], e[0]) > mw)
                rank = rank + _one(_comp(e[3], e[2]) > mw)
        rank = _xlane_sum4(rank)
        if (q == 0) & (c < total) & (rank < 16):
            tmp[rank] = mi
        sync()
        if tx < 16:
            buffer_ops.buffer_store(fx.Int32(tmp[tx]), _rsrc(out_row, 64), tx)
    return bad


row_top16_w4_one = ASTRewriter.transform(row_top16_w4_one)


row_top16_w4_short = make_row_top16_w4_fast(SHORT_STRIPS)
row_top16_w4_long = make_row_top16_w4_fast(LONG_STRIPS)


def row_top16_w4(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre=()):
    """``pre``: the first window's raw score words of strips 0..len - 1 (``first_window``), loaded before the row
    length is known; empty: the loop loads them."""
    bad = fx.Int32(0)
    if nblk <= SHORT_BLOCKS:
        if const_expr(len(pre) == SHORT_STRIPS):
            bad = row_top16_w4_one(tx, score_row, width, nblk, cand, tmp, out_row, pre)
        else:
            bad = row_top16_w4_short(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre)
    else:
        bad = row_top16_w4_long(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre)
    if bad != 0:
        if tx < 64:
            # four strips lower the cold exact path's register footprint
            row_top16(
                4, False, tx, score_row, width, lambda: nblk,
                cand, CAND_DWORDS, sel, SEL_DWORDS, 0, 0, out_row, wave_sync,
            )
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()


row_top16_w4 = ASTRewriter.transform(row_top16_w4)


def first_window(tx, score_row, width):
    """The first window's SHORT_STRIPS strips of a row's raw score words, for ``row_top16_w4(..., pre)``: issued
    before the row's length is loaded (it only masks them), so their latency overlaps the length's and the setup."""
    lane, wave = tx % 64, tx // 64
    pos = lane
    span = (width < SHORT_BLOCKS).select(width, fx.Int32(SHORT_BLOCKS))
    prs = _rsrc(score_row, span.to(fx.Int64) * 4)
    return [
        fx.Int32(buffer_ops.buffer_load(
            prs, (j * 4 + wave) * 64 + pos, vec_width=1, dtype=fx.Int32))
        for j in range_constexpr(SHORT_STRIPS)
    ]


def row_top16_w4_ids(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre):
    """``row_top16_w4`` (the same selection and stored bytes) that also returns 1 when the one-window path finished:
    its 16 ids are then in ``tmp[0..15]`` behind the last barrier, so a consumer reads them from LDS instead of
    waiting for its own store and loading it back (else 0: read ``out_row`` after ``vmcnt(0)`` and a barrier)."""
    bad = fx.Int32(0)
    fast = fx.Int32(0)
    if nblk <= SHORT_BLOCKS:
        bad = row_top16_w4_one(tx, score_row, width, nblk, cand, tmp, out_row, pre)
        fast = (bad == 0).select(fx.Int32(1), fx.Int32(0))
    else:
        bad = row_top16_w4_long(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre)
    if bad != 0:
        if tx < 64:
            row_top16(
                4, False, tx, score_row, width, lambda: nblk,
                cand, CAND_DWORDS, sel, SEL_DWORDS, 0, 0, out_row, wave_sync,
            )
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()
    return fast


row_top16_w4_ids = ASTRewriter.transform(row_top16_w4_ids)


def row_top16_prologue_ids(tx, score_row, width, arg_seq, n, ql, cand, sel, tmp, out_row):
    """The top-16 of decode row ``n`` inside a consumer kernel, before its own work, so the chain needs no selector
    launch (``score_decode.nvfp4_index_score_rows`` then the consumer): the row's top-16 stored to ``out_row`` (the
    selector kernel's bytes), returning ``fast`` of ``row_top16_w4_ids`` (1: ids in ``tmp[0..15]``; 0: in
    ``out_row``, visible after ``vmcnt(0)``, a barrier and an L1-bypassing load). Needs a 256-thread workgroup and
    LDS ``cand`` (CAND_DWORDS), ``sel`` (SEL_DWORDS), ``tmp`` (260) dwords, all reusable after; ``score_row`` /
    ``out_row`` are the row's 64-bit addresses, ``arg_seq`` the [num_reqs] int32 lengths. Every workgroup of a row
    may run it: they store equal bytes."""
    pre = first_window(tx, score_row, width)
    seq = fx.Int32(buffer_ops.buffer_load(
        _rsrc(arg_seq + (n // ql).to(fx.Int64) * 4, 4),
        fx.Int32(0), vec_width=1, is_scalar=True,
    ))
    kv0 = seq - ql + n % ql + 1
    nblk = ((kv0 > 0).select(kv0, fx.Int32(0)) + 127) // 128
    return row_top16_w4_ids(tx, score_row, width, nblk, cand, sel, tmp, out_row, pre)


def compile_top16_indices_wave4():
    name = "index_top16_indices_p128_w28w64_plain_wave4_b6r32e4"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, CAND_DWORDS, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]
        tmp: fx.Array[fx.Int32, 260, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_score: fx.Int64,
        s_head: fx.Int64,
        s_row: fx.Int64,
        width: fx.Int32,
        arg_seq: fx.Int64,
        ql: fx.Int32,
        arg_topk: fx.Int64,
        t_head: fx.Int64,
        t_row: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, h, tx = fx.block_idx.x, fx.block_idx.y, fx.thread_idx.x
        row = arg_score + (h.to(fx.Int64) * s_head + n.to(fx.Int64) * s_row) * 4
        pre = first_window(tx, row, width)
        seq = fx.Int32(buffer_ops.buffer_load(
            _rsrc(arg_seq + (n // ql).to(fx.Int64) * 4, 4),
            fx.Int32(0), vec_width=1, is_scalar=True,
        ))
        kv0 = seq - ql + n % ql + 1
        nblk = ((kv0 > 0).select(kv0, fx.Int32(0)) + 127) // 128
        row_top16_w4(
            tx,
            row,
            width, nblk, smem.cand.ptr, smem.sel.ptr, smem.tmp.ptr,
            arg_topk + (h.to(fx.Int64) * t_head + n.to(fx.Int64) * t_row) * 4,
            pre,
        )

    @flyc.jit
    def launch_top16_indices_plain_wave4_bound6_rank32_exact4(
        arg_score: fx.Int64,
        s_head: fx.Int64,
        s_row: fx.Int64,
        width: fx.Int32,
        arg_seq: fx.Int64,
        ql: fx.Int32,
        arg_topk: fx.Int64,
        t_head: fx.Int64,
        t_row: fx.Int64,
        rows: fx.Int32,
        heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_score, s_head, s_row, width, arg_seq, ql, arg_topk, t_head, t_row).launch(
            grid=(fx.Int64(rows), fx.Int64(heads), 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_top16_indices_plain_wave4_bound6_rank32_exact4.kernel_name = name
    launch_top16_indices_plain_wave4_bound6_rank32_exact4.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch_top16_indices_plain_wave4_bound6_rank32_exact4
