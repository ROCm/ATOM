# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-k of the index scores with the attend's page table, FlyDSL (gfx950): AITER's ``pa_sparse_block_topk`` bit for
bit, for ragged prefill rows that carry their request, causal length and block count.

AITER's order: key = the fp32 score's order-preserving bits (NaN as -inf, -0 below +0), ties to the larger block
index; the top min(16, live) of a row in descending order, -1 after; live = block < min(the row's block count, slots *
64). Its page table: the full blocks in that order, the row's own (partial) block last, 8 pages each, zeros after; the
context bound from the same rule.

One wave per (row, head), no workgroup barriers. The row is read in windows of STRIPS 64-wide strips (lane l holds
blocks 64 j + l; up to 32 strips, fewer if max_seq_len needs fewer); the first window's loads go out before the row's
length is known. A window's candidates are the keys at or above the 16th largest lane maximum (its top 16 bits) and the
running 16th of the earlier windows, compacted into LDS and ranked against each other while their pages load. A window
whose candidates overflow one wave, or whose NaNs or -infs could reach the top 16, takes an exact bit search for its
16th key and the index cut among the keys tied with it instead.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

BLK = 128  # tokens per block
TOPK = 16
PPB = 8  # pages per block in the emitted table
MAX_STRIPS = 32  # 64-wide strips per window: 2048 blocks (262K tokens) in one pass
CAP = 64  # candidates one rank pass holds (one per lane)
IDX_BITS = 13  # block index bits (slots * 64 <= 8192)
BOUND_BITS = 16  # exact upper bits; one ballot per bit instead of three per radix-4 digit
INT_MIN = -(1 << 31)
KEY_NEG_INF = (0xFF800000 ^ 0x7FFFFFFF) - (1 << 32)  # signed key of -inf (NaN's)
KEY_POS_INF = 0x7F800000
MAX_RANK_BLOCK = 32  # candidates whose reads one rank step keeps in flight
# LDS: cand = (index, key) pairs (layout in ``compile_topk_ragged``); sel = (index, key, page, -) of the running top 16
# in order


def _s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def _key(bits):
    """Signed key in AITER's order (the float's bits, negatives flipped); NaNs land above +inf / below -inf and are
    folded to -inf only where they could matter (``_fold_nan``)."""
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _fold_nan(k):
    return ((k > KEY_POS_INF) | (k < KEY_NEG_INF)).select(fx.Int32(KEY_NEG_INF), k)


def _ballot(c):
    return fx.Int64(fx.rocdl.ballot(T.i64, c))


def _popc(m):
    """Set bits of a wave mask as i32 (two 32-bit counts keep the compare scalar)."""
    lo = fx.math.ctpop(m.to(fx.Int32)).to(fx.Int32)
    hi = fx.math.ctpop((m >> 32).to(fx.Int32)).to(fx.Int32)
    return fx.Int32(lo) + fx.Int32(hi)


def _below(mask, base):
    """The set bits of ``mask`` below this lane, plus ``base`` (v_mbcnt)."""
    lo = mask.to(fx.Int32)
    hi = (mask >> 32).to(fx.Int32)
    r = _rocdl_ops.mbcnt_lo(T.i32, lo.ir_value(), fx.Int32(base).ir_value())
    return fx.Int32(_rocdl_ops.mbcnt_hi(T.i32, hi.ir_value(), r))


def _imax(a, b):
    return (a > b).select(a, b)


def _one(c):
    return c.select(fx.Int32(1), fx.Int32(0))


def _comp(k, i):
    """(key, index) as one i64, the key in the high word: the signed compare is AITER's order."""
    return (k.to(fx.Int64) << 32) | (i.to(fx.Int64) & 0xFFFFFFFF)


def compile_topk_ragged(heads_per_group: int, strips: int = MAX_STRIPS):
    """Row n is request ``row_req_id[n]`` with ``kv_lens[n]`` causal tokens over ``num_valid_pages[n]`` blocks."""
    HPG = heads_per_group
    # a window of ``strips`` 64-wide strips, any count up to 32 (a narrow window holds fewer keys in registers). A
    # row's last window may load past the score width (another row's scores, or zeros past the buffer): dead keys.
    assert 1 <= strips <= MAX_STRIPS, strips
    STRIPS = strips
    WIN = STRIPS * 64
    # smaller rank steps for narrow windows (fewer VGPRs); a full window already needs those registers for its keys
    RANK_BLOCK = MAX_RANK_BLOCK if STRIPS == MAX_STRIPS else 8
    # running / window candidates, rank-loop padding, then one discard slot per lane; an overflowing window rebuilds
    # its candidates exactly
    TRASH = CAP + RANK_BLOCK
    CAND = TRASH + 64
    name = f"index_top16_p128_ragged_g{HPG}_w{STRIPS}"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * CAND, 16]
        sel: fx.Array[fx.Int32, 4 * TOPK, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        # first: what the first loads need (preloaded into SGPRs)
        arg_score: fx.Int64,  # [H, total_q, >= width] fp32
        arg_seq: fx.Int64,  # unread (the rows carry their lengths), like i32_reqs
        arg_bt: fx.Int64,  # [num_reqs, stride] i32, block -> page / 8
        i32_score_bytes: fx.Int32,
        i32_bt_bytes: fx.Int32,
        i32_s_head: fx.Int32,  # strides, elements
        i32_s_row: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_reqs: fx.Int32,
        i32_width: fx.Int32,  # slots * 64
        arg_topk: fx.Int64,  # [H, total_q, 16] i32 (strides)
        arg_sbt: fx.Int64,  # [total_q * nkv, 128] i32 (row stride)
        arg_sctx: fx.Int64,  # [total_q * nkv] i32
        i32_topk_bytes: fx.Int32,
        i32_sbt_bytes: fx.Int32,
        i32_sctx_bytes: fx.Int32,
        i32_t_head: fx.Int32,
        i32_t_row: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_nkv: fx.Int32,
        # the rows' request, causal length and block count: [total_q] i32 each
        arg_rid: fx.Int64,
        arg_kvl: fx.Int64,
        arg_nvp: fx.Int64,
        i32_meta_bytes: fx.Int32,
        # the launch's first index head: a split launch (one head, block y 0) still writes its KV heads' rows and
        # page ids
        i32_h0: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, h = fx.block_idx.x, fx.block_idx.y
        lane = fx.thread_idx.x

        def rsrc(arg, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(
                arg, num_records_bytes=nbytes
            )

        srs = rsrc(arg_score, i32_score_bytes)
        row_off = h * i32_s_head + n * i32_s_row

        def load_window(wbase):
            """A window's score bits (past the row: masked as keys, or 0)."""
            return Vec.from_elements(
                [
                    fx.Int32(
                        buffer_ops.buffer_load(
                            srs,
                            row_off + wbase + (j * 64) + lane,
                            vec_width=1,
                            dtype=fx.Int32,
                        )
                    )
                    for j in range_constexpr(STRIPS)
                ],
                fx.Int32,
            )

        # the first window goes out before the row's length is known
        bits = load_window(0)

        def meta(arg):
            """Row n's entry of a ragged row array (a scalar load)"""
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc(arg, i32_meta_bytes), n, vec_width=1, is_scalar=True
                )
            )

        req = meta(arg_rid)
        kv = meta(arg_kvl)
        nblk = meta(arg_nvp)
        nlive = (nblk < i32_width).select(nblk, i32_width)
        brs = rsrc(arg_bt, i32_bt_bytes)
        bt_row = req * i32_bt_stride

        def view(ptr, n_dw, width):
            return fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, ptr), fx.make_layout(n_dw, 1)),
                fx.make_layout(width, 1),
            )

        pairs = view(smem.cand.ptr, 2 * CAND, 2)
        quads = view(smem.cand.ptr, 2 * CAND, 4)
        sels = view(smem.sel.ptr, 4 * TOPK, 4)
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
            lds_put(atom64, fx.slice(pairs, (None, pos)), 2, [i, k])

        def wave_sum(c):
            """Sum over the wave of a per-lane count in [0, 63]."""
            t = fx.Int32(0)
            for b in range_constexpr(6):
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
            """The 16th largest lane maximum, its top 16 bits (unsigned order)."""
            p = fx.Int32(0)
            for d in range_constexpr(BOUND_BITS):
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
            fx.gpu.barrier()
            mi, mk = lds_get(atom64, fx.slice(pairs, (None, lane)), 2)
            # the page loads under the rank loop
            page = fx.Int32(
                buffer_ops.buffer_load(
                    brs, bt_row + mi, vec_width=1, dtype=fx.Int32, mask=lane < total
                )
            )
            mw = _comp(mk, mi)
            rank = fx.Int32(0)
            for g in range_constexpr(CAP // RANK_BLOCK):
                if total > RANK_BLOCK * g:
                    es = [
                        lds_get(
                            atom128, fx.slice(quads, (None, RANK_BLOCK // 2 * g + u)), 4
                        )
                        for u in range_constexpr(RANK_BLOCK // 2)
                    ]
                    # all of a block's reads in flight before the compares
                    fx.rocdl.sched_barrier(0)
                    for e in es:
                        rank = rank + _one(_comp(e[1], e[0]) > mw)
                        rank = rank + _one(_comp(e[3], e[2]) > mw)
            if (lane < total) & (rank < TOPK):
                lds_put(atom128, fx.slice(sels, (None, rank)), 4, [mi, mk, page, mi])
            return (total < TOPK).select(total, fx.Int32(TOPK))

        wbase = fx.Int32(0)
        nrun = fx.Int32(0)
        tprev = fx.Int32(INT_MIN + 1)  # dead keys are INT_MIN
        while (wbase == 0) | (wbase < nlive):
            if wbase > 0:
                bits = load_window(wbase)
                # the running top 16 back into cand; its 16th key bounds the window
                fx.gpu.barrier()
                r = lds_get(atom128, fx.slice(sels, (None, lane & (TOPK - 1))), 4)
                if lane < nrun:
                    put_pair(lane, r[0], r[1])
                k15 = fx.Int32(fx.rocdl.readlane(T.i32, r[1], fx.Int32(TOPK - 1)))
                tprev = (nrun >= TOPK).select(k15, fx.Int32(INT_MIN + 1))
                fx.gpu.barrier()
            nrun = window(
                [fx.Int32(bits[j]) for j in range_constexpr(STRIPS)],
                wbase,
                nrun,
                tprev,
            )
            wbase = wbase + WIN

        # outputs
        fx.gpu.barrier()
        r = lds_get(atom128, fx.slice(sels, (None, lane & (TOPK - 1))), 4)
        blk = (lane < nrun).select(r[0], fx.Int32(-1))
        page = r[2]
        buffer_ops.buffer_store(
            blk,
            rsrc(arg_topk, i32_topk_bytes),
            h * i32_t_head + n * i32_t_row + lane,
            mask=lane < TOPK,
        )
        self_blk = (kv > 0).select((kv - 1) // BLK, fx.Int32(0))
        valid = (kv > 0) & (blk >= 0) & (blk <= self_blk) & (lane < TOPK)
        is_full = valid & (blk < self_blk)
        fm = _ballot(is_full)
        n_full = _popc(fm)
        n_valid = _popc(_ballot(valid))
        slot = is_full.select(_below(fm, 0), n_full)
        has_tail = n_valid > n_full
        cap_ctx = n_valid * BLK
        ctx = has_tail.select(
            n_full * BLK + kv - self_blk * BLK, (cap_ctx < kv).select(cap_ctx, kv)
        )
        zq = n_valid * 2 + lane  # 16 B pieces of the zero tail
        zeros = Vec.filled(4, 0, fx.Int32)
        ors = rsrc(arg_sbt, i32_sbt_bytes)
        crs = rsrc(arg_sctx, i32_sctx_bytes)
        for g in range_constexpr(HPG):
            kvh = (h + i32_h0) * HPG + g
            row = n * i32_nkv + kvh
            ro = row * i32_sbt_stride
            base = page * (PPB * i32_nkv) + kvh
            v0 = Vec.from_elements(
                [base + e * i32_nkv for e in range_constexpr(4)], fx.Int32
            )
            v1 = Vec.from_elements(
                [base + (4 + e) * i32_nkv for e in range_constexpr(4)], fx.Int32
            )
            buffer_ops.buffer_store(v0, ors, ro + slot * PPB, mask=valid)
            buffer_ops.buffer_store(v1, ors, ro + slot * PPB + 4, mask=valid)
            buffer_ops.buffer_store(zeros, ors, ro + zq * 4, mask=zq < TOPK * PPB // 4)
            buffer_ops.buffer_store(ctx, crs, row, mask=lane == 0)

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        arg_seq: fx.Int64,
        arg_bt: fx.Int64,
        i32_score_bytes: fx.Int32,
        i32_bt_bytes: fx.Int32,
        i32_s_head: fx.Int32,
        i32_s_row: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_reqs: fx.Int32,
        i32_width: fx.Int32,
        arg_topk: fx.Int64,
        arg_sbt: fx.Int64,
        arg_sctx: fx.Int64,
        i32_topk_bytes: fx.Int32,
        i32_sbt_bytes: fx.Int32,
        i32_sctx_bytes: fx.Int32,
        i32_t_head: fx.Int32,
        i32_t_row: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_nkv: fx.Int32,
        arg_rid: fx.Int64,
        arg_kvl: fx.Int64,
        arg_nvp: fx.Int64,
        i32_meta_bytes: fx.Int32,
        i32_h0: fx.Int32,
        i32_rows: fx.Int32,
        i32_heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            arg_seq,
            arg_bt,
            i32_score_bytes,
            i32_bt_bytes,
            i32_s_head,
            i32_s_row,
            i32_bt_stride,
            i32_reqs,
            i32_width,
            arg_topk,
            arg_sbt,
            arg_sctx,
            i32_topk_bytes,
            i32_sbt_bytes,
            i32_sctx_bytes,
            i32_t_head,
            i32_t_row,
            i32_sbt_stride,
            i32_nkv,
            arg_rid,
            arg_kvl,
            arg_nvp,
            i32_meta_bytes,
            i32_h0,
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
