# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSpark draft block attention of ONE request as one launch (gfx1250): rows 1..8 (the block's positions) x 16 q
heads over one KV head, fp8 page-16 SHUFFLE cache with per-token scales, each row's keys a window of its request's
keys (production: ``sparse_pa_per_row`` = pa_decode_plan + pa_decode_tile kvfp8 cluster8, one CTA per row and
partition).

The rows of one request read the same keys up to a per-row start (the sliding window, page-aligned): row r's
keys are row 0's keys from ``shift_r = (ctx_0 - ctx_r) / 16`` pages on, so every K / V tile is read once, for
all rows, through row 0's page table.

* CH (72 = 9 clusters of 8): keys 64c .. 64c + 63 of row 0. A wave takes rows w and w + 4. S^T = K Q^T as
  bf16 WMMAs 16x16x32 (K fp8 -> bf16 exactly, per-token K scale and sm_scale on the fp32 scores), masked
  softmax over the chunk, P (x per-token V scale, bf16) V as bf16 WMMAs. Partials (max, sum, o) to memory; the
  cluster merges its 8 chunks (cluster barrier); the merged partials (device scope) are counted.
* MG (32): a wave a (row, head): the 9 cluster partials merged, bf16 output [rows, 16, 128].
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

from .dspark_draft_mlp_gfx1250 import (
    BLOCK,
    CL,
    WAVE,
    _bf_hi,
    _bf_lo,
    _cluster_sync,
    _f2bf2,
    _fmax,
    _lds_barrier,
    _load_agent_i32,
    _now32,
)

HQ = 16
HD = 128
KC = 64  # keys a chunk
N_CH = 72  # chunks: 4608 keys (window 4096 + page offset + the block)
N_CL = N_CH // CL
N_MG = 32
GRID = N_CH + N_MG
MAXROWS = 8
# scratch (fp32 words)
PM_OFF = 0  # [N_CH][8 rows][16 heads] chunk max
PL_OFF = PM_OFF + N_CH * MAXROWS * HQ  # chunk sum
PO_OFF = PL_OFF + N_CH * MAXROWS * HQ  # [N_CH][8][16][128] chunk o
CM_OFF = PO_OFF + N_CH * MAXROWS * HQ * HD  # [N_CL][8][16] cluster max
CLS_OFF = CM_OFF + N_CL * MAXROWS * HQ  # cluster sum
CO_OFF = CLS_OFF + N_CL * MAXROWS * HQ  # [N_CL][8][16][128] cluster o
SCR_WORDS = CO_OFF + N_CL * MAXROWS * HQ * HD
SCR_BYTES = SCR_WORDS * 4
# ctl words (normal device memory, 128-byte lines, never reset)
C_DONE, EXIT, CALLS, ERR = 0, 32, 64, 96
CTL_BYTES = 512
NEG_INF = float("-inf")
LOG2E = 1.4426950408889634


def _bf16x16(words):
    """8 packed-bf16 int32 -> a v16 bf16 WMMA operand."""
    return Vec.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _wmma_bf16(a, b, c):
    return Vec(
        fx.rocdl.wmma_f32_16x16x32_bf16(
            fx.Vector.make_type(8, fx.Float32), fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c),
            reuseA=False, reuseB=False,
        ).result
    )


def _fp8x8_to_bf16(w0, w1):
    """8 e4m3 bytes (two int32) -> 4 packed-bf16 int32 (exact: v_cvt_scale_pk8_bf16_fp8 with unit E8M0 scales)."""
    raw8 = Vec.from_elements([fx.Int32(w0), fx.Int32(w1)], fx.Int32)
    v8 = Vec(fx.rocdl.cvt_scale_pk8_bf16_fp8(T.vec(8, T.bf16), fx.as_ir_value(raw8), fx.as_ir_value(fx.Int32(0x7F7F7F7F)), 0))
    w = v8.bitcast(fx.Int32)
    return [fx.Int32(w[i]) for i in range_constexpr(4)]


@flyc.jit
def launch_draft_pa(
    arg_q: fx.Pointer,
    arg_kc: fx.Pointer,
    arg_vc: fx.Pointer,
    arg_ks: fx.Pointer,
    arg_vs: fx.Pointer,
    arg_bt: fx.Pointer,
    arg_ctx: fx.Pointer,
    arg_out: fx.Pointer,
    arg_scr: fx.Pointer,
    arg_ctl: fx.Pointer,
    arg_tl: fx.Pointer,
    i32_m: fx.Int32,
    i32_bt_stride: fx.Int32,
    f32_scale: fx.Float32,
    i32_maxit: fx.Int32,
    stream: fx.Stream,
    tl: Constexpr[int] = 0,
):
    """``arg_q`` bf16 [m, 16, 128]; ``arg_kc`` / ``arg_vc`` the page-16 SHUFFLE e4m3 caches [P16, 1, 8, 16, 16] /
    [P16, 1, 1, 128, 16], ``arg_ks`` / ``arg_vs`` fp32 per-slot scales; ``arg_bt`` int32 [m, stride] the rows'
    page-16 tables, ``arg_ctx`` int32 [m] their key counts (ONE request: tables are windows of one request table);
    out ``arg_out`` bf16 [m, 16, 128]; ``arg_scr`` SCR_BYTES, ``arg_ctl`` CTL_BYTES zeroed once."""
    kernel_name = f"dspark_draft_pa_gfx1250{'_tl' if tl else ''}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK, 1, 1])
    def kernel_draft_pa(
        arg_q: fx.Pointer,
        arg_kc: fx.Pointer,
        arg_vc: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_vs: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_ctx: fx.Pointer,
        arg_out: fx.Pointer,
        arg_scr: fx.Pointer,
        arg_ctl: fx.Pointer,
        arg_tl: fx.Pointer,
        i32_m: fx.Int32,
        i32_bt_stride: fx.Int32,
        f32_scale: fx.Float32,
        i32_maxit: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        l16 = lane % 16
        g = lane // 16
        gptr = lambda p: fx.recast_iter(  # noqa: E731
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), p
        )
        tlp = gptr(arg_tl)
        ctl = gptr(arg_ctl)

        def stamp(slot):
            if const_expr(tl):
                if tid == 0:
                    fx.ptr_store(_now32(), tlp + (bid * 4 + slot))

        stamp(0)
        arena = fx.SharedAllocator(static=False)
        arena.allocate(64)
        big = 0x7FFFFFF0
        rs = lambda p: create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(p)), num_records_bytes=big)  # noqa: E731
        scr_rs = rs(arg_scr)
        ctx_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_ctx)), num_records_bytes=i32_m * 4)
        ctx0 = fx.Int32(buffer_load(ctx_rs, 0, vec_width=1, dtype=T.i32))
        ninf = fx.Float32(NEG_INF)

        if bid < N_CH:
            # ============================================================ chunk c: keys 64c .. 64c + 63 of row 0
            c = bid
            kc_rs, vc_rs, ks_rs, vs_rs = rs(arg_kc), rs(arg_vc), rs(arg_ks), rs(arg_vs)
            bt_rs = rs(arg_bt)
            q_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_q)), num_records_bytes=i32_m * HQ * HD * 2)
            key0 = c * KC
            # the chunk's 4 pages (row 0's table); past ctx0 / the table: page 0, masked
            pg = []
            for t in range_constexpr(4):
                j = c * 4 + t
                ok = (j < i32_bt_stride) & (j * 16 < ctx0)
                pj = fx.Int32(buffer_load(bt_rs, ok.select(j, fx.Int32(0)), vec_width=1, dtype=T.i32))
                pg.append(ok.select(pj, fx.Int32(0)))
            # K as the A operand of S^T = K Q^T: lane row = key l16 of a tile, d chunks 8g / 16 + 8g of a d step
            kop = []  # [tile][dstep] -> v16 bf16
            for t in range_constexpr(4):
                row = []
                for s in range_constexpr(4):
                    w = []
                    for h2 in range_constexpr(2):
                        off = pg[t] * 2048 + (2 * s + h2) * 256 + l16 * 16 + 8 * g
                        two = Vec(buffer_load(kc_rs, off // 4, vec_width=2, dtype=T.i32))
                        w += _fp8x8_to_bf16(two[0], two[1])
                    row.append(_bf16x16(w))
                kop.append(row)
            # per-token scales of the lane's D rows: keys 8g .. 8g + 7 of each tile
            ksc = []
            vsc = []
            for t in range_constexpr(4):
                a0 = Vec(buffer_load(ks_rs, pg[t] * 16 + 8 * g, vec_width=4, dtype=T.f32))
                a1 = Vec(buffer_load(ks_rs, pg[t] * 16 + 8 * g + 4, vec_width=4, dtype=T.f32))
                b0 = Vec(buffer_load(vs_rs, pg[t] * 16 + 8 * g, vec_width=4, dtype=T.f32))
                b1 = Vec(buffer_load(vs_rs, pg[t] * 16 + 8 * g + 4, vec_width=4, dtype=T.f32))
                ksc.append([fx.Float32(a0[i]) for i in range_constexpr(4)] + [fx.Float32(a1[i]) for i in range_constexpr(4)])
                vv = [fx.Float32(b0[i]) for i in range_constexpr(4)] + [fx.Float32(b1[i]) for i in range_constexpr(4)]
                vsc.append([(key0 + 16 * t + 8 * g + i < ctx0).select(vv[i], fx.Float32(0.0)) for i in range_constexpr(8)])
            # V as the B operand of O = P V: lane col = d 16nb + l16, keys 8g .. (tile 2u) and 16 + 8g .. (2u + 1)
            vop = []  # [u][nb]
            for u in range_constexpr(2):
                row = []
                for nb in range_constexpr(8):
                    w = []
                    for h2 in range_constexpr(2):
                        off = pg[2 * u + h2] * 2048 + (16 * nb + l16) * 16 + 8 * g
                        two = Vec(buffer_load(vc_rs, off // 4, vec_width=2, dtype=T.i32))
                        mw = []
                        for q_ in range_constexpr(2):
                            # keys past the context (a partial last page, a stand-in page): bytes zeroed, so stale
                            # or NaN bytes there cannot reach O through 0 * NaN
                            kb = key0 + 16 * (2 * u + h2) + 8 * g + 4 * q_
                            msk = fx.Int32(0)
                            for b in range_constexpr(4):
                                msk = msk | (kb + b < ctx0).select(fx.Int32(0xFF << (8 * b) if b < 3 else -0x1000000),
                                                                   fx.Int32(0))
                            mw.append(fx.Int32(two[q_]) & msk)
                        w += _fp8x8_to_bf16(mw[0], mw[1])
                    row.append(_bf16x16(w))
                vop.append(row)
            stamp(1)
            for rr_ in range_constexpr(2):
                r = wave + 4 * rr_
                live = r < i32_m
                if live:
                    ctxr = fx.Int32(buffer_load(ctx_rs, r, vec_width=1, dtype=T.i32))
                    start = ((ctx0 - ctxr) // 16) * 16
                    # Q^T as the B operand: lane col = head l16, d chunks 8g / 16 + 8g of a d step
                    qop = []
                    for s in range_constexpr(4):
                        w0 = Vec(buffer_load(q_rs, ((r * HQ + l16) * HD + 32 * s + 8 * g) // 2, vec_width=4, dtype=T.i32))
                        w1 = Vec(buffer_load(q_rs, ((r * HQ + l16) * HD + 32 * s + 16 + 8 * g) // 2, vec_width=4,
                                             dtype=T.i32))
                        qop.append(_bf16x16([fx.Int32(w0[i]) for i in range_constexpr(4)]
                                            + [fx.Int32(w1[i]) for i in range_constexpr(4)]))
                    sv = []  # [tile][i]: score of key 16t + 8g + i, head l16
                    for t in range_constexpr(4):
                        acc = fx.full(8, 0.0, fx.Float32)
                        for s in range_constexpr(4):
                            acc = _wmma_bf16(kop[t][s], qop[s], acc)
                        row = []
                        for i in range_constexpr(8):
                            k = key0 + 16 * t + 8 * g + i
                            v_ = fx.Float32(acc[i]) * ksc[t][i] * f32_scale
                            row.append(((k >= start) & (k < ctx0)).select(v_, ninf))
                        sv.append(row)
                    mx = sv[0][0]
                    for t in range_constexpr(4):
                        for i in range_constexpr(8):
                            mx = _fmax(mx, sv[t][i])
                    mx = _fmax(mx, mx.shuffle_xor(fx.Int32(16), fx.Int32(WAVE)))
                    msafe = (mx == ninf).select(fx.Float32(0.0), mx)
                    lsum = fx.Float32(0.0)
                    pw = []  # [u] P operand words
                    pv = [[None] * 8 for _ in range(4)]
                    for t in range_constexpr(4):
                        for i in range_constexpr(8):
                            p_ = fx.exp(sv[t][i] - msafe)
                            lsum = lsum + p_
                            pv[t][i] = p_ * vsc[t][i]
                    lsum = lsum + lsum.shuffle_xor(fx.Int32(16), fx.Int32(WAVE))
                    for u in range_constexpr(2):
                        w = []
                        for h2 in range_constexpr(2):
                            t = 2 * u + h2
                            for i in range_constexpr(4):
                                w.append(_f2bf2(pv[t][2 * i], pv[t][2 * i + 1]))
                        pw.append(_bf16x16(w))
                    # O = P V: lane col = d 16nb + l16, rows = heads 8g + i
                    base_o = PO_OFF + ((c * MAXROWS + r) * HQ) * HD
                    for nb in range_constexpr(8):
                        acc = fx.full(8, 0.0, fx.Float32)
                        for u in range_constexpr(2):
                            acc = _wmma_bf16(pw[u], vop[u][nb], acc)
                        for i in range_constexpr(8):
                            buffer_store(fx.Float32(acc[i]).ir_value(), scr_rs,
                                         (base_o + (8 * g + i) * HD + 16 * nb + l16) * 4, offset_is_bytes=True)
                    if g == 0:
                        buffer_store(mx.ir_value(), scr_rs, (PM_OFF + (c * MAXROWS + r) * HQ + l16) * 4,
                                     offset_is_bytes=True)
                        buffer_store(lsum.ir_value(), scr_rs, (PL_OFF + (c * MAXROWS + r) * HQ + l16) * 4,
                                     offset_is_bytes=True)
            stamp(2)
            rocdl.s_wait_storecnt(0)
            _cluster_sync()
            # the cluster's 8 chunks merged: CTA cr of the cluster takes heads 2cr, 2cr + 1 of every row;
            # a wave a (row pair, head), a lane 4 d
            cl = c // CL
            cr = c % CL
            for it_ in range_constexpr(4):
                # work item it_ of this wave: rows (it_ * 2 + wave // 2) (0..7), head 2cr + wave % 2
                r = it_ * 2 + wave // 2
                h = 2 * cr + wave % 2
                if r < i32_m:
                    ms = []
                    ls = []
                    for j in range_constexpr(CL):
                        cc = cl * CL + j
                        ms.append(fx.Float32(buffer_load(scr_rs, PM_OFF + (cc * MAXROWS + r) * HQ + h, vec_width=1,
                                                         dtype=T.f32)))
                        ls.append(fx.Float32(buffer_load(scr_rs, PL_OFF + (cc * MAXROWS + r) * HQ + h, vec_width=1,
                                                         dtype=T.f32)))
                    M = ms[0]
                    for j in range_constexpr(1, CL):
                        M = _fmax(M, ms[j])
                    Ms = (M == ninf).select(fx.Float32(0.0), M)
                    L = fx.Float32(0.0)
                    o4 = [fx.Float32(0.0)] * 4
                    for j in range_constexpr(CL):
                        cc = cl * CL + j
                        f_ = fx.exp(ms[j] - Ms)
                        f_ = (ms[j] == ninf).select(fx.Float32(0.0), f_)
                        L = L + ls[j] * f_
                        ov = Vec(buffer_load(scr_rs, PO_OFF + ((cc * MAXROWS + r) * HQ + h) * HD + 4 * lane,
                                             vec_width=4, dtype=T.f32))
                        o4 = [o4[i] + fx.Float32(ov[i]) * f_ for i in range_constexpr(4)]
                    cbase = (cl * MAXROWS + r) * HQ + h
                    buffer_store(Vec.from_elements(o4, fx.Float32).ir_value(), scr_rs, (CO_OFF + cbase * HD + 4 * lane) * 4,
                                 offset_is_bytes=True, cache_modifier=2 << 3)
                    if lane == 0:
                        buffer_store(M.ir_value(), scr_rs, (CM_OFF + cbase) * 4, offset_is_bytes=True,
                                     cache_modifier=2 << 3)
                        buffer_store(L.ir_value(), scr_rs, (CLS_OFF + cbase) * 4, offset_is_bytes=True,
                                     cache_modifier=2 << 3)
            rocdl.s_wait_storecnt(0)
            _lds_barrier()
            if tid == 0:
                fx.atomic_add(ctl + C_DONE, fx.Int32(1), syncscope="agent")
            stamp(3)
        else:
            # ============================================================ final merge: a wave a (row, head)
            item = (bid - N_CH) * 4 + wave
            r = item // HQ
            h = item % HQ
            if tid == 0:
                it = fx.Int32(0)
                target = (_load_agent_i32(ctl + CALLS) + fx.Int32(1)) * fx.Int32(N_CH)
                cur = _load_agent_i32(ctl + C_DONE)
                while ((cur - target) < fx.Int32(0)) & (it < i32_maxit):
                    rocdl.s_sleep(16)
                    cur = _load_agent_i32(ctl + C_DONE)
                    it = it + 1
                if (cur - target) < fx.Int32(0):
                    fx.atomic_add(ctl + ERR, fx.Int32(1), syncscope="agent")
            _lds_barrier()
            stamp(1)
            if r < i32_m:
                ms = []
                ls = []
                for j in range_constexpr(N_CL):
                    cb = (j * MAXROWS + r) * HQ + h
                    ms.append(fx.Float32(buffer_load(scr_rs, CM_OFF + cb, vec_width=1, dtype=T.f32, cache_modifier=2 << 3)))
                    ls.append(fx.Float32(buffer_load(scr_rs, CLS_OFF + cb, vec_width=1, dtype=T.f32, cache_modifier=2 << 3)))
                M = ms[0]
                for j in range_constexpr(1, N_CL):
                    M = _fmax(M, ms[j])
                Ms = (M == ninf).select(fx.Float32(0.0), M)
                L = fx.Float32(0.0)
                o4 = [fx.Float32(0.0)] * 4
                for j in range_constexpr(N_CL):
                    cb = (j * MAXROWS + r) * HQ + h
                    f_ = fx.exp(ms[j] - Ms)
                    f_ = (ms[j] == ninf).select(fx.Float32(0.0), f_)
                    L = L + ls[j] * f_
                    ov = Vec(buffer_load(scr_rs, CO_OFF + cb * HD + 4 * lane, vec_width=4, dtype=T.f32,
                                         cache_modifier=2 << 3))
                    o4 = [o4[i] + fx.Float32(ov[i]) * f_ for i in range_constexpr(4)]
                inv = fx.Float32(1.0) / L
                out_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_out)),
                                                          num_records_bytes=i32_m * HQ * HD * 2)
                pk = Vec.from_elements([_f2bf2(o4[0] * inv, o4[1] * inv), _f2bf2(o4[2] * inv, o4[3] * inv)], fx.Int32)
                buffer_store(pk.ir_value(), out_rs, ((r * HQ + h) * HD + 4 * lane) * 2, offset_is_bytes=True)
            stamp(2)

        # ------------------------------------------------------------------ exit: the last workgroup advances CALLS
        if tid == 0:
            calls = _load_agent_i32(ctl + CALLS)
            old_e = fx.Int32(fx.atomic_add(ctl + EXIT, fx.Int32(1), syncscope="agent"))
            if old_e + fx.Int32(1) == (calls + fx.Int32(1)) * fx.Int32(GRID):
                fx.atomic_add(ctl + CALLS, fx.Int32(1), syncscope="agent")

    kernel_draft_pa(
        arg_q, arg_kc, arg_vc, arg_ks, arg_vs, arg_bt, arg_ctx, arg_out, arg_scr, arg_ctl, arg_tl, i32_m,
        i32_bt_stride, f32_scale, i32_maxit,
        value_attrs={"rocdl.cluster_dims": f"{CL},1,1"},
    ).launch(grid=(GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream, cluster=(CL, 1, 1))


launch_draft_pa.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
