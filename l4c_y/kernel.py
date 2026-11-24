# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Registered-input, batched two-stage bf16 all-reduce with the 12-wave norm and bf16-row quant rule.

Owners reduce their whole rows in rank order and publish a batch; peers pull it; then the norm and optional quant. The
``owner`` form publishes the 6144 code + 192 scale bytes of a row (bf16 out: the bf16 row) instead of the reduced
bf16 row: about half the gather bytes.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .norm_math import _block_sum
from .quant_math import (
    _code_off,
    _codes,
    _e8m0,
    _scale_off,
    _set_fp16_ovfl,
)
from .norm_math import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    swap_xor1,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)

from .common import (
    SYS, WORLD, arrive, load_ptrs, publish, sel4, sum4, unrotate,
    wait_pullers, wait_reduced,
)
from .common import MAX_ROWS, RED_OFF
from flydsl._mlir.dialects import llvm as _llvm

LLROW = H * 4  # bytes of one row's LL words: 3072 x {bf16x2 data, epoch}
LL_MAX_ROWS = 64
LLREG = LL_MAX_ROWS * LLROW  # one LL region (region 0: stage-2 rows in; regions 1..3: ll2 stage-1 rows from owner+1+s)


def _ll_words(addr, n):
    """n 8-byte {data, epoch} words at addr as monotonic system-scope atomic loads (global_load_dwordx2 sc0 sc1, never
    hoisted, no per-load wait): [(data, epoch)]."""
    out = []
    for i in range(n):
        p = fx.inttoptr(fx.PointerType.get(T.i64, address_space=fx.AddressSpace.Global, alignment=8), addr + fx.Int64(8 * i))
        v = fx.generic_load(p, dtype=fx.Int64, memory_order=fx.AtomicOrdering.Monotonic)
        out.append((fx.Int32(v & fx.Int64(0xFFFFFFFF)), fx.Int32(v >> 32)))
    return out

NW = H // 8 // 64  # 12 waves of 8 columns per lane (the plain norm's layout)


def compile_allreduce_gemma_bf16_mxfp8(rpw: int, shuffled: bool, quantized: bool, owner: bool = False, pf: bool = False,
                                       ll: bool = False, ts: bool = False, ll2: bool = False):
    """``owner``: each chunk's owner runs the add + norm (+ quant) on its rows and
    publishes the result (MXFP8 codes + scale bytes, or bf16); peers copy it. The
    residual output is then written for the owned rows only (reduce-scatter form)."""
    R = rpw
    name = (f"allreduce_gemma_bf16_{'mxfp8' if quantized else 'out'}_{'tiled' if shuffled else 'rowmajor'}_h6144_r{R}"
            + ("_owner" if owner else "") + ("_l4pf" if pf else "") + ("_l4ll" if ll else "") + ("_l4ll2" if ll2 else "") + ("_ts" if ts else ""))
    # pf (L4-comm overlay): R == 1 non-owner form only. The residual + weight loads are issued before the
    # puller's RED wait / with the reducer's peer reads, and the reducer feeds its in-register sum to the tail
    # instead of re-reading its published row. Same arithmetic, same stores: bit-identical outputs.
    assert not pf or (R == 1 and not owner)
    # ll (L4-comm overlay): R == 1 non-owner form; stage 2 (owner -> pullers) as an LL push: the owner stores each
    # reduced dword next to the row's epoch (8-byte {data, epoch} words, 16-byte stores) straight into the pullers'
    # LL buffers; a puller polls its own (local) LL row until every epoch matches. No staging store, no RED flag,
    # no remote read on the puller's path. Stage 1 (ARR + the owner's in-place peer reads) and all arithmetic are
    # unchanged: bit-identical outputs. 2x stage-2 bytes: for small row counts only.
    # ts: lane 0 of each workgroup stores s_memrealtime at 5 points into arg_ts[wg * 8 + k] (int64).
    assert not ll or (R == 1 and not owner and not pf)
    # ll2: both stages LL. Every non-owner workgroup pushes its own partial row as LL words into the owner's region
    # 1 + s (s = (me - c - 1) mod 4); the owner polls its 3 local regions, sums in the same order (own, owner+1, +2, +3),
    # pushes stage 2 as in ll. No ARR flag, no remote read at all. 2x bytes in both stages.
    assert not ll2 or (R == 1 and not owner and not pf and not ll)
    assert not ts or (pf or ll or ll2)

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_me: fx.Int32,
        arg_xs: fx.Int64,
        arg_b0: fx.Int64,
        arg_b1: fx.Int64,
        arg_b2: fx.Int64,
        arg_b3: fx.Int64,
        arg_cnt: fx.Int64,
        i32_sr: fx.Int32,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_res: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_eps: fx.Int32,
        arg_ll: fx.Int64,
        arg_ts: fx.Int64,
    ):
        _set_fp16_ovfl()
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        q = (i32_m + 3) // 4
        qg = (q + R - 1) // R
        qp = (qg + 7) // 8 * 8
        grp = b // qp
        jg, inq = xcd_row(b % qp, qg)
        c = (i32_me + grp) % WORLD
        j0 = jg * R
        row0 = c * q + j0
        live = inq & (row0 < i32_m)
        if live:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            bs = (arg_b0, arg_b1, arg_b2, arg_b3)
            mine = sel4(i32_me, *bs)
            reducer = grp == 0
            puller = grp != 0
            cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=MAX_ROWS * 4)
            e = fx.Int32(buffer_ops.buffer_load(cr, row0, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1
            if const_expr(ts):
                if tx == 0:
                    t0 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                    tr0 = buffer_ops.create_buffer_resource_from_addr(arg_ts, num_records_bytes=1 << 20)
                    buffer_ops.buffer_store(Vec.from_elements([fx.Int32(t0 & fx.Int64(0xFFFFFFFF)), fx.Int32(t0 >> 32)], fx.Int32),
                                            tr0, b * 16)
            xs = load_ptrs(arg_xs)
            nb = H * 2
            off = tx * 4
            if const_expr(not ll2):
                arrive(*bs, c, i32_me, row0, e, puller)
                wait_pullers(mine, i32_me, row0, e, reducer)
            rot = wave % 3

            def read_row(base, cm, ok):
                rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=nb)
                return Vec(buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, mask=ok, cache_modifier=cm))

            def load_rw(row, ok=None):
                """This lane's 8 residual and 8 norm-weight values of ``row``."""
                rr = buffer_ops.create_buffer_resource_from_addr(
                    arg_r + fx.Int64(row) * fx.Int64(i32_sr * 2), num_records_bytes=nb
                )
                wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=nb)
                rv = Vec(buffer_ops.buffer_load(rr, off, vec_width=4, dtype=fx.Int32, mask=ok))
                wv = Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32, mask=ok))
                return rv, wv

            def tail(xv, row, stage, part, rw=None):
                """add + norm (+ quant) of one row from its 8 reduced fp32 per lane; stores
                the local outputs and, if ``stage``, the published copy for the peers.
                ``rw``: the residual / weight packs, if already loaded."""
                resr = buffer_ops.create_buffer_resource_from_addr(
                    arg_res + fx.Int64(row) * nb, num_records_bytes=nb
                )
                rvp, wvp = load_rw(row) if rw is None else rw
                rv, wv = bf16x8(rvp), bf16x8(wvp)
                s = [xv[i] + rv[i] for i in range_constexpr(8)]
                res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(4)]
                buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
                acc = wave_sum_lane63(seq_sum_sq(s))
                if lane == 63:
                    part[wave] = acc
                fx.gpu.barrier()
                rstd = rsqrt_rn(
                    _block_sum([part[i] for i in range_constexpr(NW)]),
                    i32_eps.bitcast(fx.Float32),
                )
                y = [(s[i] * rstd) * (wv[i] + fx.Float32(1.0)) for i in range_constexpr(8)]
                ypack = Vec.from_elements([pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)], fx.Int32)
                if const_expr(quantized):
                    # the standalone quantizer reads the bf16-rounded normed value
                    sb, rescale = _e8m0(bf16x8(ypack))
                    codes = _codes(bf16x8(ypack), rescale)  # 2 dwords, column order
                    sbyte = fx.Int32(sb.to(fx.Int32))
                    store_q(codes, row)
                    if (tx % 4) == 0:
                        store_s(sbyte, row, tx // 4)
                    if const_expr(stage is not None):
                        # published row: 6144 code bytes, then 192 scale bytes
                        buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), stage, tx * 2, cache_modifier=SYS)
                        if (tx % 4) == 0:
                            buffer_ops.buffer_store(sbyte.to(fx.Int8), stage, H + tx // 4,
                                                    offset_is_bytes=True, cache_modifier=SYS)
                else:
                    qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i64_q_bytes)
                    buffer_ops.buffer_store(ypack, qr, row * (H // 2) + off)
                    if const_expr(stage is not None):
                        buffer_ops.buffer_store(ypack, stage, off, cache_modifier=SYS)

            def store_q(codes, row):
                qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i64_q_bytes)
                if const_expr(shuffled):
                    # (16, 16) layout: even lanes store their piece's low 8 B plus
                    # the odd lane's high 8 B as one 16 B store; odd lanes idle
                    recv = swap_xor1(codes)
                    piece = Vec.from_elements([codes[0], codes[1], recv[0], recv[1]], fx.Int32)
                    buffer_ops.buffer_store(piece, qr, _code_off(row, tx // 2) // 4, mask=(tx % 2) == 0)
                else:
                    buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), qr, (row * (H // 4)) + tx * 2)

            def store_s(sbyte, row, g):
                sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i64_s_bytes)
                buffer_ops.buffer_store(sbyte.to(fx.Int8), sr, _scale_off(row, g), offset_is_bytes=True)

            def staged(owner_base, j):
                return buffer_ops.create_buffer_resource_from_addr(
                    owner_base + RED_OFF + fx.Int64(j) * nb, num_records_bytes=nb)

            if const_expr(owner):
                if reducer:
                    for ri in range(fx.Int32(0), fx.Int32(R), fx.Int32(1)):
                        row = row0 + ri
                        ok = (j0 + ri < q) & (row < i32_m)
                        peer = [read_row(sel4((i32_me + 1 + (t + rot) % 3) % 4, *xs) + fx.Int64(row) * nb, SYS, ok)
                                for t in range_constexpr(3)]
                        own = read_row(sel4(i32_me, *xs) + fx.Int64(row) * nb, 0, ok)
                        rw = load_rw(row, ok)  # issued with the peer reads, off the sum's path
                        red = sum4([own] + unrotate(rot, peer))
                        if ok:
                            tail(bf16x8(red), row, staged(mine, j0 + ri), part, rw)
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        fx.gpu.barrier()
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    publish(*bs, i32_me, row0, e, reducer)
                if puller:
                    wait_reduced(mine, row0, e)
                    for ri in range(fx.Int32(0), fx.Int32(R), fx.Int32(1)):
                        row = row0 + ri
                        ok = (j0 + ri < q) & (row < i32_m)
                        if ok:
                            src = staged(sel4(c, *bs), j0 + ri)
                            if const_expr(quantized):
                                cv = Vec(buffer_ops.buffer_load(src, tx * 2, vec_width=2, dtype=fx.Int32,
                                                                cache_modifier=SYS))
                                store_q([fx.Int32(cv[0]), fx.Int32(cv[1])], row)
                                if tx < H // 128:
                                    # lane l copies the scale bytes of groups 4l .. 4l + 3
                                    sw = fx.Int32(buffer_ops.buffer_load(src, H // 4 + tx, vec_width=1,
                                                                         dtype=fx.Int32, cache_modifier=SYS))
                                    for k in range_constexpr(4):
                                        store_s((sw >> (8 * k)) & 0xFF, row, tx * 4 + k)
                            else:
                                v = Vec(buffer_ops.buffer_load(src, off, vec_width=4, dtype=fx.Int32,
                                                               cache_modifier=SYS))
                                qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=i64_q_bytes)
                                buffer_ops.buffer_store(v, qr, row * (H // 2) + off)
            def mark(k):
                if const_expr(ts):
                    if tx == 0:
                        t = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                        tr = buffer_ops.create_buffer_resource_from_addr(arg_ts, num_records_bytes=1 << 20)
                        lo = fx.Int32(t & fx.Int64(0xFFFFFFFF))
                        hi = fx.Int32(t >> 32)
                        buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), tr, (b * 8 + k) * 2)

            if const_expr(ll2):
                rwp2 = load_rw(row0)
                lls2 = load_ptrs(arg_ll)
                llrow2 = fx.Int64(row0) * LLROW
                own2 = read_row(sel4(i32_me, *xs) + fx.Int64(row0) * nb, 0, None)
                mark(1)
                if puller:
                    s = (i32_me - c + 3) % 4
                    dst2 = buffer_ops.create_buffer_resource_from_addr(
                        sel4(c, *lls2) + fx.Int64(1 + s) * LLREG + llrow2, num_records_bytes=LLROW)
                    buffer_ops.buffer_store(Vec.from_elements([fx.Int32(own2[0]), e, fx.Int32(own2[1]), e], fx.Int32),
                                            dst2, tx * 8, cache_modifier=SYS)
                    buffer_ops.buffer_store(Vec.from_elements([fx.Int32(own2[2]), e, fx.Int32(own2[3]), e], fx.Int32),
                                            dst2, tx * 8 + 4, cache_modifier=SYS)
                    wq = _ll_words(sel4(i32_me, *lls2) + llrow2 + fx.Int64(tx * 32), 4)
                    a0q, a1q, a2q, a3q = wq[0][0], wq[0][1], wq[1][0], wq[1][1]
                    b0q, b1q, b2q, b3q = wq[2][0], wq[2][1], wq[3][0], wq[3][1]
                    while ((a1q - e) | (a3q - e) | (b1q - e) | (b3q - e)) != 0:
                        wq = _ll_words(sel4(i32_me, *lls2) + llrow2 + fx.Int64(tx * 32), 4)
                        a0q, a1q, a2q, a3q = wq[0][0], wq[0][1], wq[1][0], wq[1][1]
                        b0q, b1q, b2q, b3q = wq[2][0], wq[2][1], wq[3][0], wq[3][1]
                    mark(2)
                    tail(bf16x8(Vec.from_elements([a0q, a2q, b0q, b2q], fx.Int32)), row0, None, part, rwp2)
                if reducer:
                    base_in = sel4(i32_me, *lls2) + llrow2 + fx.Int64(tx * 32)
                    w1 = _ll_words(base_in + LLREG, 4)
                    w2 = _ll_words(base_in + 2 * LLREG, 4)
                    w3 = _ll_words(base_in + 3 * LLREG, 4)
                    p10, p11, p12, p13 = w1[0][0], w1[1][0], w1[2][0], w1[3][0]
                    p20, p21, p22, p23 = w2[0][0], w2[1][0], w2[2][0], w2[3][0]
                    p30, p31, p32, p33 = w3[0][0], w3[1][0], w3[2][0], w3[3][0]
                    bad = fx.Int32(0)
                    for ww in (w1, w2, w3):
                        for k in range_constexpr(4):
                            bad = bad | (ww[k][1] - e)
                    while bad != 0:
                        w1 = _ll_words(base_in + LLREG, 4)
                        w2 = _ll_words(base_in + 2 * LLREG, 4)
                        w3 = _ll_words(base_in + 3 * LLREG, 4)
                        p10, p11, p12, p13 = w1[0][0], w1[1][0], w1[2][0], w1[3][0]
                        p20, p21, p22, p23 = w2[0][0], w2[1][0], w2[2][0], w2[3][0]
                        p30, p31, p32, p33 = w3[0][0], w3[1][0], w3[2][0], w3[3][0]
                        bad = fx.Int32(0)
                        for ww in (w1, w2, w3):
                            for k in range_constexpr(4):
                                bad = bad | (ww[k][1] - e)
                    peers2 = [Vec.from_elements([p10, p11, p12, p13], fx.Int32), Vec.from_elements([p20, p21, p22, p23], fx.Int32),
                             Vec.from_elements([p30, p31, p32, p33], fx.Int32)]
                    red2 = sum4([own2] + peers2)
                    mark(2)
                    lo_w2 = Vec.from_elements([fx.Int32(red2[0]), e, fx.Int32(red2[1]), e], fx.Int32)
                    hi_w2 = Vec.from_elements([fx.Int32(red2[2]), e, fx.Int32(red2[3]), e], fx.Int32)
                    for t in range_constexpr(3):
                        pq = (i32_me + 1 + (t + rot) % 3) % 4
                        dst2 = buffer_ops.create_buffer_resource_from_addr(sel4(pq, *lls2) + llrow2, num_records_bytes=LLROW)
                        buffer_ops.buffer_store(lo_w2, dst2, tx * 8, cache_modifier=SYS)
                        buffer_ops.buffer_store(hi_w2, dst2, tx * 8 + 4, cache_modifier=SYS)
                    mark(3)
                    tail(bf16x8(red2), row0, None, part, rwp2)
                mark(4)
            elif const_expr(ll):
                rwp = load_rw(row0)
                lls = load_ptrs(arg_ll)
                llrow = fx.Int64(row0) * LLROW
                mark(1)
                if reducer:
                    peer = [read_row(sel4((i32_me + 1 + (t + rot) % 3) % 4, *xs) + fx.Int64(row0) * nb, SYS, None)
                            for t in range_constexpr(3)]
                    own = read_row(sel4(i32_me, *xs) + fx.Int64(row0) * nb, 0, None)
                    red = sum4([own] + unrotate(rot, peer))
                    mark(2)
                    lo_w = Vec.from_elements([fx.Int32(red[0]), e, fx.Int32(red[1]), e], fx.Int32)
                    hi_w = Vec.from_elements([fx.Int32(red[2]), e, fx.Int32(red[3]), e], fx.Int32)
                    for t in range_constexpr(3):
                        p = (i32_me + 1 + (t + rot) % 3) % 4
                        dst = buffer_ops.create_buffer_resource_from_addr(sel4(p, *lls) + llrow, num_records_bytes=LLROW)
                        buffer_ops.buffer_store(lo_w, dst, tx * 8, cache_modifier=SYS)
                        buffer_ops.buffer_store(hi_w, dst, tx * 8 + 4, cache_modifier=SYS)
                    mark(3)
                    tail(bf16x8(red), row0, None, part, rwp)
                if puller:
                    w = _ll_words(sel4(i32_me, *lls) + llrow + fx.Int64(tx * 32), 4)
                    a0, a1, a2, a3 = w[0][0], w[0][1], w[1][0], w[1][1]
                    b0, b1, b2, b3 = w[2][0], w[2][1], w[3][0], w[3][1]
                    while ((a1 - e) | (a3 - e) | (b1 - e) | (b3 - e)) != 0:
                        w = _ll_words(sel4(i32_me, *lls) + llrow + fx.Int64(tx * 32), 4)
                        a0, a1, a2, a3 = w[0][0], w[0][1], w[1][0], w[1][1]
                        b0, b1, b2, b3 = w[2][0], w[2][1], w[3][0], w[3][1]
                    mark(2)
                    xv_pack = Vec.from_elements([a0, a2, b0, b2], fx.Int32)
                    tail(bf16x8(xv_pack), row0, None, part, rwp)
                mark(4)
            elif const_expr(pf):
                mark(1)
                rwp = load_rw(row0)
                if reducer:
                    peer = [read_row(sel4((i32_me + 1 + (t + rot) % 3) % 4, *xs) + fx.Int64(row0) * nb, SYS, None)
                            for t in range_constexpr(3)]
                    own = read_row(sel4(i32_me, *xs) + fx.Int64(row0) * nb, 0, None)
                    red = sum4([own] + unrotate(rot, peer))
                    dest = buffer_ops.create_buffer_resource_from_addr(mine + RED_OFF + fx.Int64(j0) * nb, num_records_bytes=nb)
                    buffer_ops.buffer_store(red, dest, off, cache_modifier=SYS)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    mark(2)
                    publish(*bs, i32_me, row0, e, reducer)
                    mark(3)
                    tail(bf16x8(red), row0, None, part, rwp)
                if puller:
                    wait_reduced(mine, row0, e)
                    mark(2)
                    xv_pack = read_row(sel4(c, *bs) + RED_OFF + fx.Int64(j0) * nb, SYS, None)
                    tail(bf16x8(xv_pack), row0, None, part, rwp)
                mark(4)
            else:
                if reducer:
                    for ri in range(fx.Int32(0), fx.Int32(R), fx.Int32(1)):
                        row = row0 + ri
                        ok = (j0 + ri < q) & (row < i32_m)
                        peer = [read_row(sel4((i32_me + 1 + (t + rot) % 3) % 4, *xs) + fx.Int64(row) * nb, SYS, ok)
                                for t in range_constexpr(3)]
                        own = read_row(sel4(i32_me, *xs) + fx.Int64(row) * nb, 0, ok)
                        red = sum4([own] + unrotate(rot, peer))
                        dest = buffer_ops.create_buffer_resource_from_addr(mine + RED_OFF + fx.Int64(j0 + ri) * nb, num_records_bytes=nb)
                        buffer_ops.buffer_store(red, dest, off, mask=ok, cache_modifier=SYS)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    publish(*bs, i32_me, row0, e, reducer)
                if puller:
                    wait_reduced(mine, row0, e)
                for ri in range(fx.Int32(0), fx.Int32(R), fx.Int32(1)):
                    row = row0 + ri
                    ok = (j0 + ri < q) & (row < i32_m)
                    xv_pack = read_row(sel4(c, *bs) + RED_OFF + fx.Int64(j0 + ri) * nb, SYS, ok)
                    if ok:
                        tail(bf16x8(xv_pack), row, None, part)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()
            buffer_ops.buffer_store(e, cr, row0, mask=tx == 0)

    @flyc.jit
    def launch_allreduce_gemma_bf16_mxfp8(
        i32_m: fx.Int32, i32_me: fx.Int32, arg_xs: fx.Int64,
        arg_b0: fx.Int64, arg_b1: fx.Int64, arg_b2: fx.Int64, arg_b3: fx.Int64,
        arg_cnt: fx.Int64, i32_sr: fx.Int32, arg_r: fx.Int64, arg_w: fx.Int64,
        arg_q: fx.Int64, arg_s: fx.Int64, arg_res: fx.Int64,
        i64_q_bytes: fx.Int64, i64_s_bytes: fx.Int64, i32_eps: fx.Int32,
        arg_ll: fx.Int64, arg_ts: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(i32_m, i32_me, arg_xs, arg_b0, arg_b1, arg_b2, arg_b3, arg_cnt,
               i32_sr, arg_r, arg_w, arg_q, arg_s, arg_res, i64_q_bytes,
               i64_s_bytes, i32_eps, arg_ll, arg_ts).launch(
            grid=(fx.Int64(xcd_grid(((i32_m + 3) // 4 + R - 1) // R) * 4), 1, 1),
            block=(64 * NW, 1, 1), stream=stream,
        )
    launch_allreduce_gemma_bf16_mxfp8.kernel_name = name
    launch_allreduce_gemma_bf16_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_allreduce_gemma_bf16_mxfp8
