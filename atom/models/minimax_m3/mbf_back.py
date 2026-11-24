# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1 decode MoE back half in ONE launch (``ATOM_P1_MBF_BACK``, integration branches only, default off):
the routed GEMM2 (+ the shared expert's down tiles, first) and the slot / shared sum of ``moe_top4_shared_sum_h6144``,
at 1..128 rows (amd-kernels' inline-sort / id-rows / sorted 16-row tiers, production tiling incl. ``ak_moe_tiles``).

Grid = [shared down tiles | routed down tiles | 3 * m sum workgroups]. The GEMM tile bodies are amd-kernels' (same K
order, same bf16 values); their bf16 slot / shared rows go write-through (SC1) into static per-stream buffers that hold
the sentinel 0xFFFF_FFFF between calls ("data" mode: the payload is its own ready flag, so no producer waits for its
stores and no flag word is written; no bf16 pair the tiles write is 0xFFFF_FFFF, both halves would be NaN with a full
payload). The sum workgroups have the highest ids, so they are dispatched only after every producer was (a poller can
never hold a slot a producer still needs); each polls its row's 2048 columns of the 5 sources with SC1 loads (bounded),
sums slots 0..3 then shared in fp32 exactly as ``moe_top4_shared_sum_h6144`` (bit-identical output) and writes the
sentinel back (it is the only reader). The block table words are read with scalar loads (as production's compiled code
does). Also here: the "flag" mode (per-producer flag words; measured slower), timing-only controls (nosum, xbits,
stamps) and ``compile_joint_nosync`` (the GEMM1 -> GEMM2 one-launch ceiling, timing only, never correct).
Measured (p1/mbf-build-gemm2 NOTES.md): -0.3..-0.5 us per MoE layer vs production, bitwise at 1..128 rows.
"""

import functools

import torch

H, I, E = 6144, 768, 128
SC1 = 16
MAXN = 128
_flags: dict = {}


def flags(dev, tag=0) -> torch.Tensor:
    """Per-device flag words [MAXN, 5, H / 64] (zero between calls: each sum workgroup clears the words it read)."""
    f = _flags.get((dev, tag))
    if f is None:
        f = _flags[(dev, tag)] = torch.zeros(MAXN * 5 * (H // 64), dtype=torch.int32, device=dev)
    return f


def _poll(addr):
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.typing import T
    return fx.Int32(_llvm.inline_asm(T.i32, [fx.as_ir_value(fx.Int64(addr))],
                                     "global_load_dword $0, $1, off sc1\n\ts_waitcnt vmcnt(0)", "=v,v,~{memory}",
                                     has_side_effects=True, is_align_stack=False,
                                     asm_dialect=_llvm.AsmDialect.AD_ATT))


def _now():
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.typing import T
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _stamp(base, bid, k, cond=True):
    """debug: thread 0 of the workgroup stores s_memrealtime (100 MHz) at base[bid * 4 + k] (int64)."""
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    t = _now()
    r = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=1 << 20)
    lo, hi = t.to(fx.Int32), (t >> 32).to(fx.Int32)
    buffer_ops.buffer_store(lo, r, (bid * 4 + k) * 2, mask=(fx.thread_idx.x == 0) & cond)
    buffer_ops.buffer_store(hi, r, (bid * 4 + k) * 2 + 1, mask=(fx.thread_idx.x == 0) & cond)


def _sload2(a0, a1):
    """Two scalar (SMEM) dword loads from uniform global addresses, one wait: the routing-table words every 16-row
    block reads first (the scalar path; a vector load of them queues behind the weight streams)."""
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.typing import T

    def sgpr64(a):
        a = fx.Int64(a)
        lo = fx.Int32(fx.rocdl.readfirstlane(T.i32, a.to(fx.Int32)))
        hi = fx.Int32(fx.rocdl.readfirstlane(T.i32, (a >> 32).to(fx.Int32)))
        return (lo.to(fx.Uint32).to(fx.Uint64) | (hi.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    v = _llvm.inline_asm(ty, [fx.as_ir_value(sgpr64(a0)), fx.as_ir_value(sgpr64(a1))],
                         "s_load_dword $0, $2, 0x0\n\ts_load_dword $1, $3, 0x0\n\ts_waitcnt lgkmcnt(0)",
                         "=&s,=&s,s,s", has_side_effects=True, is_align_stack=False,
                         asm_dialect=_llvm.AsmDialect.AD_ATT)
    return fx.Int32(_llvm.extractvalue(T.i32, v, [0])), fx.Int32(_llvm.extractvalue(T.i32, v, [1]))


@functools.cache
def _routed(tier, tn, bm, pf, gc, sc1=True, wait=True, flag=True):
    """amd-kernels' routed down tile of ``tier`` + SC1 stores + one flag per (token, slot, chunk)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_small import _mfma, _hold, _raw, _lds_ptr
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.a16w4_utils import inline_sort_table
    ni, mr = tn // 64, bm // 16
    nch, cpt = H // gc, tn // gc

    def body(base, mb, nb, expert, A, AS, W, WS, rows, rowinfo, FLP, m, bpre=None, spre=None, prebuild=None):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=rows * (I // 2))
        sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=rows * (I // 32))
        wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=H * I // 2,
                                               base_byte_offset=fx.Int64(expert) * (H * I // 2))
        sw = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * H * (I // 32))
        wd = [((nb * (tn // 16) + wave * ni + j) * 6 * 1024 + q16 * 256 + l16 * 16) // 4 for j in range_constexpr(ni)]
        swd = [((expert * (H // 32) + nb * (tn // 32) + (wave * ni + j) // 2) * 3) * 64 + q16 * 16 + l16
               for j in range_constexpr(ni)]
        if prebuild is not None:
            bpre = [buffer_ops.buffer_load(wr, wd[j], vec_width=4, dtype=fx.Int32, cache_modifier=2)
                    for j in range_constexpr(ni)]
            spre = [buffer_ops.buffer_load(sw, swd[j], vec_width=1, dtype=fx.Int32) for j in range_constexpr(ni)]
            prebuild()
        ad = [((mb * bm + mi * 16 + l16) * (I // 2) + q16 * 16) // 4 for mi in range_constexpr(mr)]
        sd = [(mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)]

        def load(k):
            aa = [buffer_ops.buffer_load(ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64) for mi in range_constexpr(mr)]
            sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32, soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                  for mi in range_constexpr(mr)]
            if k == 0 and prebuild is not None:
                bb, sb = bpre, spre
            else:
                bb = [buffer_ops.buffer_load(wr, wd[j], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 1024, cache_modifier=2) for j in range_constexpr(ni)]
                sb = [buffer_ops.buffer_load(sw, swd[j], vec_width=1, dtype=fx.Int32, soffset_bytes=(k // 2) * 256) for j in range_constexpr(ni)]
            return aa, sa, bb, sb
        acc = [[None] * ni for _ in range(mr)]
        ring = [load(k) for k in range_constexpr(pf)]
        for k in range_constexpr(6):
            if k + pf < 6:
                ring.append(load(k + pf))
            aa, sa, bb, sb = ring.pop(0)
            for mi in range_constexpr(mr):
                for j in range_constexpr(ni):
                    acc[mi][j] = _mfma(aa[mi], bb[j], acc[mi][j], sa[mi], sb[j], (k % 2) * 2 + j % 2)
        flat = _hold([acc[mi][j] for mi in range(mr) for j in range(ni)])
        for mi in range_constexpr(mr):
            for j in range_constexpr(ni):
                v = Vec(flat[mi * ni + j])
                col = wave * (tn // 4) + j * 16 + l16
                for ii in range_constexpr(4):
                    off = ((mi * 16 + q16 * 4 + ii) * tn + col) * 4
                    llvm.StoreOp(_raw(v[ii]), _lds_ptr(base, off))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        return tx

    def epilogue(base, nb, OUT, FLP, m, toks_weights):
        """toks_weights: per rr, (token, slot, weight) of row rr * 8 + tx // 32 (all values of this thread)."""
        tx = fx.thread_idx.x
        out = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=m * 4 * H * 2)
        for rr in range_constexpr(bm // 8):
            row = rr * 8 + tx // 32
            token, slot, weight = toks_weights[rr]
            for c in range_constexpr(tn // 64):
                col = tx % 32 * 2 + c * 64
                v = Vec(llvm.load(T.vec(2, T.f32), _lds_ptr(base, (row * tn + col) * 4), alignment=8))
                pk = (v * weight).to(fx.BFloat16).bitcast(fx.Int32)[0]
                buffer_ops.buffer_store(pk, out, ((token * 4 + slot) * H + nb * tn + col) // 2, mask=token < m,
                                        cache_modifier=SC1 if sc1 else 0)
        # every wave's slot rows are written through before any flag of the tile
        if wait:
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
        if flag:
            fr = buffer_ops.create_buffer_resource_from_addr(FLP, num_records_bytes=m * 5 * nch * 4)
            for rr in range_constexpr(bm // 8):
                token, slot, _ = toks_weights[rr]
                for cc in range_constexpr(cpt):
                    buffer_ops.buffer_store(fx.Int32(1), fr, (token * 5 + slot) * nch + nb * cpt + cc,
                                            mask=(token < m) & (tx % 32 == cc), cache_modifier=SC1)

    if tier == "inline":
        assert bm == 16

        @flyc.jit
        def routed_tile(base, tab, rb, A, AS, W, WS, OUT, IDS, WT, FLP, m, rows):
            tx = fx.thread_idx.x
            lane = tx % 64
            mb, nb = rb // (H // tn), rb % (H // tn)
            expert, owner, count, build_tab = inline_sort_table(
                IDS, m, 4, mb, lane, tab, max_pairs=64, bm=16, multi_block=True)
            if owner:
                body(base, mb, nb, expert, A, AS, W, WS, rows, None, FLP, m, prebuild=build_tab)
                tw = []
                for rr in range_constexpr(bm // 8):
                    row = rr * 8 + tx // 32
                    packed = fx.Int32(tab[row])
                    token, slot = packed & 0xFFFFFF, packed.shrui(fx.Int32(24))
                    pair = token * 4 + slot
                    pair = (token < m).select(pair, fx.Int32(0))
                    tw.append((token, slot, fx.Float32(WT[pair])))
                epilogue(base, nb, OUT, FLP, m, tw)
        return routed_tile

    def rows_of(mb, IDS, WT, m):
        tx = fx.thread_idx.x
        tw = []
        for rr in range_constexpr(bm // 8):
            row = rr * 8 + tx // 32
            packed = fx.Int32(IDS[mb * bm + row])
            token, slot = packed & 0xFFFFFF, packed.shrui(fx.Int32(24))
            if tier == "id_rows":
                pair = (token < m).select(token * 4 + slot, fx.Int32(0))
                weight = fx.Float32(WT[pair])
            else:
                weight = fx.Float32(WT[mb * bm + row])
            tw.append((token, slot, weight))
        return tw

    # the two tiers' prologues exactly as amd-kernels writes them (an `ok = ...; if ok:` indirection made LLVM load the
    # block ids with vector loads: the empty 16-row blocks then waited ~1 us behind the weight streams, ATT A1)
    if tier == "id_rows":
        @flyc.jit
        def routed_tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, FLP, m, rows, EP, NP):
            mb, nb = rb // (H // tn), rb % (H // tn)
            record, blocks = _sload2(EP + fx.Int64(mb) * 4, EP + fx.Int64(rows // 16) * 4)
            fx.rocdl.sched_barrier(0)
            if mb < blocks:
                expert = record & 127
                body(base, mb, nb, expert, A, AS, W, WS, rows, None, FLP, m)
                epilogue(base, nb, OUT, FLP, m, rows_of(mb, IDS, WT, m))
        return routed_tile

    @flyc.jit
    def routed_tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, FLP, m, rows, EP, NP):
        mb, nb = rb // (H // tn), rb % (H // tn)
        nr, expert = _sload2(NP, EP + fx.Int64(mb) * 4)
        if mb * bm < nr:
            body(base, mb, nb, expert, A, AS, W, WS, rows, None, FLP, m)
            epilogue(base, nb, OUT, FLP, m, rows_of(mb, IDS, WT, m))
    return routed_tile


@functools.cache
def _shared(tn, bm, pf, gc, sc1=True, wait=True, flag=True):
    """``shared_gemm2_a8w8.compile_gemm2`` tile + SC1 stores + one flag per (token, 4, chunk)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_gemm1_a8w8 import _mfma, _hold, _raw, _lds_ptr, _load8
    assert tn in (64, 128, 256, 512) and bm in (16, 32, 64)
    ni, mr = tn // 64, bm // 16
    nch, cpt = H // gc, tn // gc

    @flyc.jit
    def tile(base, bid, A, AS, W, WS, OUT, FLP, m):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = bid // (H // tn), bid % (H // tn)
        if mb * bm < m:
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=m * I)
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=m * (I // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=H * I)
            sw = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=H * (I // 32))
            out = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=m * H * 2)
            ad = [((mb * bm + mi * 16 + l16) * I + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [(mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)]
            wd = [((nb * (tn // 16) + wave * ni + j) * 6 * 2048 + q16 * 256 + l16 * 16) // 4 for j in range_constexpr(ni)]
            swd = [((nb * (tn // 32) + (wave * ni + j) // 2) * 3) * 64 + q16 * 16 + l16
                   for j in range_constexpr(ni)]

            def load(k):
                aa = [_load8(ar, ad[mi], k * 128) for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32, soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                bb = [_load8(wr, wd[j], k * 2048, 256) for j in range_constexpr(ni)]
                sb = [buffer_ops.buffer_load(sw, swd[j], vec_width=1, dtype=fx.Int32, soffset_bytes=(k // 2) * 256) for j in range_constexpr(ni)]
                if fx.const_expr(ni == 1):
                    sb = [fx.Int32(sb[0]) >> ((wave % 2) * 8)]
                return aa, sa, bb, sb
            acc = [[None] * ni for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(6):
                if fx.const_expr(k + pf < 6):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for j in range_constexpr(ni):
                        acc[mi][j] = _mfma(aa[mi], bb[j], acc[mi][j], sa[mi], sb[j], (k % 2) * 2 + j % 2)
            flat = _hold([acc[mi][j] for mi in range(mr) for j in range(ni)])
            for mi in range_constexpr(mr):
                for j in range_constexpr(ni):
                    v = Vec(flat[mi * ni + j])
                    col = wave * (tn // 4) + j * 16 + l16
                    for ii in range_constexpr(4):
                        off = ((mi * 16 + q16 * 4 + ii) * tn + col) * 4
                        llvm.StoreOp(_raw(v[ii]), _lds_ptr(base, off))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            for rr in range_constexpr(bm // 8):
                row = rr * 8 + tx // 32
                token = mb * bm + row
                for c in range_constexpr(tn // 64):
                    col = tx % 32 * 2 + c * 64
                    v = Vec(llvm.load(T.vec(2, T.f32), _lds_ptr(base, (row * tn + col) * 4), alignment=8))
                    pk = v.to(fx.BFloat16).bitcast(fx.Int32)[0]
                    buffer_ops.buffer_store(pk, out, (token * H + nb * tn + col) // 2, mask=token < m,
                                            cache_modifier=SC1 if sc1 else 0)
            if fx.const_expr(wait):
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
            if fx.const_expr(flag):
                fr = buffer_ops.create_buffer_resource_from_addr(FLP, num_records_bytes=m * 5 * nch * 4)
                for rr in range_constexpr(bm // 8):
                    token = mb * bm + rr * 8 + tx // 32
                    for cc in range_constexpr(cpt):
                        buffer_ops.buffer_store(fx.Int32(1), fr, (token * 5 + 4) * nch + nb * cpt + cc,
                                                mask=(token < m) & (tx % 32 == cc), cache_modifier=SC1)

    tile.lds_bytes = bm * tn * 4
    return tile


@functools.cache
def _sum(gc, sleep):
    """One (row, 2048 columns) of the reduced output after its 5 sources' flags: ``moe_top4_shared_sum_h6144``'s math."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    from flydsl.expr import range_constexpr, rocdl
    from flydsl.expr.typing import Vector as Vec
    nch, per = H // gc, 2048 // gc
    nw = 5 * per
    assert nw <= 256

    @flyc.jit
    def sum_tile(sid, P, S, O, FLP, m):
        tx = fx.thread_idx.x
        row, chunk = sid // 3, sid % 3
        widx = (row * 5 + tx // per) * nch + chunk * per + tx % per
        if tx < nw:
            got = _poll(FLP + fx.Int64(widx * 4))
            while got == 0:
                rocdl.s_sleep(sleep)
                got = _poll(FLP + fx.Int64(widx * 4))
        fx.gpu.barrier()
        col = chunk * 2048 + tx * 8
        pr = buffer_ops.create_buffer_resource(P, max_size=False, num_records_bytes=m * 4 * 6144 * 2)
        sr = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=m * 6144 * 2)
        out = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=m * 6144 * 2)
        acc = Vec.filled(8, 0.0, fx.Float32)
        for k in range_constexpr(4):
            v = Vec(buffer_ops.buffer_load(pr, ((row * 4 + k) * 6144 + col) // 2, vec_width=4, dtype=fx.Int32,
                                           cache_modifier=SC1)).bitcast(fx.BFloat16).to(fx.Float32)
            acc = acc + v
        sh = Vec(buffer_ops.buffer_load(sr, (row * 6144 + col) // 2, vec_width=4, dtype=fx.Int32,
                                        cache_modifier=SC1)).bitcast(fx.BFloat16).to(fx.Float32)
        buffer_ops.buffer_store((acc + sh).to(fx.BFloat16).bitcast(fx.Int32), out, (row * 6144 + col) // 2)
        fr = buffer_ops.create_buffer_resource_from_addr(FLP, num_records_bytes=m * 5 * nch * 4)
        buffer_ops.buffer_store(fx.Int32(0), fr, widx, mask=tx < nw)
    return sum_tile


@functools.cache
def _sum_data(sleep, carry=True, hint=0, stamps=False, reset=True):
    """``moe_top4_shared_sum_h6144``'s math on the static sentinel-filled slot / shared buffers: the payload is its own
    flag (no bf16 pair the GEMM tiles write is 0xFFFF_FFFF: both halves would be NaN with a full payload); after the
    sum the workgroup writes the sentinel back (it is the only reader)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    from flydsl.expr import range_constexpr, rocdl
    from flydsl.expr.typing import Vector as Vec

    @flyc.jit
    def sum_tile(sid, P, S, O, m, PB):
        tx = fx.thread_idx.x
        row, chunk = sid // 3, sid % 3
        if fx.const_expr(stamps):
            _stamp(PB, fx.Int32(fx.block_idx.x), 1)
        col = chunk * 2048 + tx * 8
        pr = buffer_ops.create_buffer_resource(P, max_size=False, num_records_bytes=MAXN * 4 * 6144 * 2)
        sr = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=MAXN * 6144 * 2)
        out = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=m * 6144 * 2)

        def ld():
            vs = [buffer_ops.buffer_load(pr, ((row * 4 + k) * 6144 + col) // 2, vec_width=4, dtype=fx.Int32,
                                         cache_modifier=SC1) for k in range_constexpr(4)]
            vs.append(buffer_ops.buffer_load(sr, (row * 6144 + col) // 2, vec_width=4, dtype=fx.Int32,
                                             cache_modifier=SC1))
            return vs

        def ready(vs):
            ok = fx.Int32(1)
            for j in range_constexpr(5):
                vv = Vec(vs[j])
                for i in range_constexpr(4):
                    ok = (fx.Int32(vv[i]) == fx.Int32(-1)).select(fx.Int32(0), ok)
            return ok

        if fx.const_expr(hint > 0):
            # cheap phase: one dword per producer tile (its last column pair of this row), one lane each, until all
            # are written; only then the full 20 KB operand check below (normally passes on its first load)
            nrt_h = 32
            nsh_h = 2048 // hint
            k_h = (tx < nrt_h).select(tx // 8, fx.Int32(4))
            c_h = (tx < nrt_h).select((tx % 8) * 256 + 254, (tx - nrt_h) * hint + hint - 2)
            hoff = (tx < nrt_h).select(((row * 4 + k_h) * 6144 + chunk * 2048 + c_h) // 2,
                                       fx.Int32(MAXN * 4 * 6144 // 2) + (row * 6144 + chunk * 2048 + c_h) // 2)
            if tx < nrt_h + nsh_h:
                hv = _poll(PB + fx.Int64(hoff) * 4)
                while hv == -1:
                    rocdl.s_sleep(sleep)
                    hv = _poll(PB + fx.Int64(hoff) * 4)
            fx.gpu.barrier()
        if fx.const_expr(carry):
            # the loads that saw every word written are the operands (loop-carried; no second round trip)
            results = ld()
            got = ready(results)
            it = fx.Int32(0)
            # bounded (~1-2 s): a lost write can never hang the GPU (the output would then be wrong, not stuck);
            # one carried predicate (a compound `while a & b` condition exits at once in FlyDSL, J10)
            while got == 0:
                st_ = results
                rocdl.s_sleep(sleep)
                nxt = ld()
                results = nxt
                it = it + 1
                got = (it >= fx.Int32(1 << 20)).select(fx.Int32(1), ready(nxt))
            vs = [results[k] for k in range_constexpr(5)]
            if fx.const_expr(stamps):
                _stamp(PB, fx.Int32(fx.block_idx.x), 2)
        else:
            got = ready(ld())
            while got == 0:
                rocdl.s_sleep(sleep)
                got = ready(ld())
            vs = ld()
        acc = Vec.filled(8, 0.0, fx.Float32)
        for k in range_constexpr(4):
            acc = acc + Vec(vs[k]).bitcast(fx.BFloat16).to(fx.Float32)
        sh = Vec(vs[4]).bitcast(fx.BFloat16).to(fx.Float32)
        buffer_ops.buffer_store((acc + sh).to(fx.BFloat16).bitcast(fx.Int32), out, (row * 6144 + col) // 2)
        if fx.const_expr(reset):
            neg = Vec.filled(4, -1, fx.Int32)
            for k in range_constexpr(4):
                buffer_ops.buffer_store(neg, pr, ((row * 4 + k) * 6144 + col) // 2, cache_modifier=SC1)
            buffer_ops.buffer_store(neg, sr, (row * 6144 + col) // 2, cache_modifier=SC1)
        if fx.const_expr(stamps):
            _stamp(PB, fx.Int32(fx.block_idx.x), 3)
    return sum_tile


@functools.cache
def compile_back(tier, *, tn=256, bm=16, pf=2, sh_tn=256, sh_bm=16, sleep=1, nosum=False, mode="flag", sc1=True,
                 wait=True, flag=True, carry=True, hint=False, stamps=False, reset=True):
    """GEMM2 (+ shared down tiles first) + the reduced sum, one launch. ``nosum``: timing control without the sum
    workgroups (flags are still set and never cleared: do not mix with real calls)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    gc = sh_tn
    if mode == "data":
        wait = flag = False
    sh = _shared(sh_tn, sh_bm, 2, gc, sc1, wait, flag)
    rt = _routed(tier, tn, bm, pf, gc, sc1, wait, flag)
    sm = _sum(gc, sleep) if mode == "flag" else _sum_data(sleep, carry, sh_tn if hint else 0, stamps, reset)
    assert not (stamps and hint)
    lds = max(bm * tn * 4, sh.lds_bytes)
    name = (f"mbf_back_{tier}_a4w4_bm{bm}_tn{tn}_pf{pf}_shared_a8w8_bm{sh_bm}_tn{sh_tn}_sum_h6144_s{sleep}"
            + ("_nosum" if nosum else "") + ("" if mode == "flag" else "_" + mode)
            + ("" if (sc1, wait, flag) == (True, True, True) else f"_x{int(sc1)}{int(wait)}{int(flag)}")
            + ("" if carry or mode == "flag" else "_reload") + ("_hint" if hint and mode == "data" else "")
            + ("_stamps" if stamps else "") + ("" if reset else "_noreset"))

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]
        tab: fx.Array[fx.Int32, 33]

    if tier == "inline":
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, IDS: fx.Int64,
                   WT: fx.Tensor, A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, OUT8: fx.Tensor,
                   FIN: fx.Tensor, FLP: fx.Int64, m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, nrt: fx.Int32):
            smem = fx.SharedAllocator().allocate(Shared).peek()
            base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
            bid = fx.Int32(fx.block_idx.x)
            # one if / else chain: no role's stores lie on another role's path (keeps the routing-table loads
            # scalar: a store on the path makes LLVM drop noclobber and the 16-row block ids become vector loads)
            if fx.const_expr(stamps):
                _stamp(FLP, bid, 0)
            if bid < nsh:
                sh(base, bid, A8, AS8, W8, WS8, OUT8, FLP, m)
                if fx.const_expr(stamps):
                    _stamp(FLP, bid, 3)
            else:
                if bid < nsh + nrt:
                    rt(base, smem.tab.ptr, bid - nsh, A, AS, W, WS, OUT, IDS, WT, FLP, m, rows)
                    if fx.const_expr(stamps):
                        _stamp(FLP, bid, 3)
                else:
                    if fx.const_expr(not nosum):
                        if fx.const_expr(mode == "flag"):
                            sm(bid - nsh - nrt, OUT, OUT8, FIN, FLP, m)
                        else:
                            sm(bid - nsh - nrt, OUT, OUT8, FIN, m, FLP)

        @flyc.jit
        def launch_back(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, IDS: fx.Int64,
                        WT: fx.Tensor, A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, OUT8: fx.Tensor,
                        FIN: fx.Tensor, FLP: fx.Int64, m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, nrt: fx.Int32,
                        ngrid: fx.Int32, stream: fx.Stream):
            kernel(A, AS, W, WS, OUT, IDS, WT, A8, AS8, W8, WS8, OUT8, FIN, FLP, m, rows, nsh, nrt).launch(
                grid=(ngrid, 1, 1), block=(256, 1, 1), stream=stream)
    else:
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, IDS: fx.Tensor,
                   WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor,
                   WS8: fx.Tensor, OUT8: fx.Tensor, FIN: fx.Tensor, FLP: fx.Int64, m: fx.Int32, rows: fx.Int32,
                   nsh: fx.Int32, nrt: fx.Int32, EP: fx.Int64, NP: fx.Int64):
            smem = fx.SharedAllocator().allocate(Shared).peek()
            base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
            bid = fx.Int32(fx.block_idx.x)
            if fx.const_expr(stamps):
                _stamp(FLP, bid, 0)
            if bid < nsh:
                sh(base, bid, A8, AS8, W8, WS8, OUT8, FLP, m)
                if fx.const_expr(stamps):
                    _stamp(FLP, bid, 3)
            else:
                if bid < nsh + nrt:
                    rt(base, bid - nsh, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, FLP, m, rows, EP, NP)
                    if fx.const_expr(stamps):
                        _stamp(FLP, bid, 3)
                else:
                    if fx.const_expr(not nosum):
                        if fx.const_expr(mode == "flag"):
                            sm(bid - nsh - nrt, OUT, OUT8, FIN, FLP, m)
                        else:
                            sm(bid - nsh - nrt, OUT, OUT8, FIN, m, FLP)

        @flyc.jit
        def launch_back(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, OUT: fx.Tensor, IDS: fx.Tensor,
                        WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor,
                        WS8: fx.Tensor, OUT8: fx.Tensor, FIN: fx.Tensor, FLP: fx.Int64, m: fx.Int32, rows: fx.Int32,
                        nsh: fx.Int32, nrt: fx.Int32, ngrid: fx.Int32, EP: fx.Int64, NP: fx.Int64, stream: fx.Stream):
            kernel(A, AS, W, WS, OUT, IDS, WT, EIDS, NV, A8, AS8, W8, WS8, OUT8, FIN, FLP, m, rows, nsh, nrt, EP, NP).launch(
                grid=(ngrid, 1, 1), block=(256, 1, 1), stream=stream)
    launch_back.kernel_name = name
    return launch_back


def _tiles_mod():
    from atom.models.minimax_m3 import ak_moe_tiles
    return ak_moe_tiles


def tiles_for(n: int, moe_tiles: bool) -> dict:
    """Production tiling: amd-kernels defaults where ``ak_moe_tiles`` does not serve ``n`` (or ATOM_P1_MOE_TILES is off),
    else ``ak_moe_tiles.tiles_for`` (17-128 rows; with ATOM_P1_MOE_IDROWS also 8 / 16 rows and the id-rows host at
    97-128; with ATOM_P1_MBF_ASM the GEMM2 prefetch 3 + GEMM1 prologue priority at 32-128) -- p1/mbf-verify-compose."""
    if not moe_tiles:
        return {}
    T = _tiles_mod()
    return T.tiles_for(n) if T.supported(n) else {}


class Plan:
    def __init__(self, n, moe_tiles=True, sleep=1, nosum=False, mode="flag", x=(True, True, True), carry=True,
                 hint=False, stamps=False, fill=False):
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_small as XS
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4 import max_sorted_rows
        P = "amd_kernels.moe.mxfp4_h6144_i768_e128_top4."
        import importlib
        cfg = tiles_for(n, moe_tiles)
        kw, bm = XS.small_config_for(n)
        assert kw == 4 and bm == 16 and 1 <= n <= MAXN
        self.n, self.bm = n, bm
        self.id_rows = XS.id_rows_for(n) or cfg.get("id_rows", False)
        self.inline = n <= XS.INLINE_SORT_MAX and not self.id_rows
        self.tier = "inline" if self.inline else "id_rows" if self.id_rows else "small"
        m1 = importlib.import_module(P + f"gemm1_a4w4_{'inline_sort' if self.inline else self.tier}")
        self.rows = n * 4 * 16 if self.inline else max_sorted_rows(n, 128, 4, bm)
        sh_bm, sh_kw, sh_pf = XS.shared_fold_tile_for(n)
        sh_bm, sh_pf = cfg.get("sh_bm", sh_bm), cfg.get("sh_pf", sh_pf)
        self.nsh1 = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
        self.sh_tn = cfg.get("sh_tn", XS.shared_g2_columns_for(n))
        g2_bm = XS.shared_fold_down_rows_for(sh_bm, bm)
        self.nsh2 = (n + g2_bm - 1) // g2_bm * (H // self.sh_tn)
        g1_kw = {} if self.inline else {"sh_bm": sh_bm, "sh_kw": sh_kw, "sh_pf": sh_pf}
        if self.id_rows:
            g1_kw["pairs"] = XS.id_pairs_for(n)
        g1c = m1.compile_gemm1_shared_first
        if self.id_rows and "g1_prio" in cfg:  # ATOM_P1_MBF_ASM: the id-rows GEMM1 with the prologue s_setprio pair
            from atom.models.minimax_m3.ak_moe_g1prio import compile_gemm1_shared_first as _g1prio

            def g1c(**kw_):
                return _g1prio(prio=cfg["g1_prio"], **kw_)
        self.k1 = g1c(kw=kw, pf=cfg.get("g1_pf", 3), bm=bm, **g1_kw)
        tn = cfg.get("g2_tn", 256)
        self.nrt = self.rows // bm * (H // tn)
        self.nosum, self.mode = nosum, mode
        self.k2 = compile_back(self.tier, tn=tn, bm=bm, pf=cfg.get("g2_pf", 2), sh_tn=self.sh_tn, sh_bm=g2_bm,
                               sleep=sleep, nosum=nosum, mode=mode, sc1=x[0], wait=x[1], flag=x[2],
                               carry=carry, hint=hint, stamps=stamps)
        self.stamps = stamps
        # the inline tier keeps the sum workgroups' sentinel restore (no GEMM1 fill copy for it)
        self.fill = fill and not self.inline
        if self.fill:
            assert mode == "data" and tn == 256 and "g1_prio" not in cfg and "g1_pf" not in cfg
            self.k2 = compile_back(self.tier, tn=256, bm=bm, pf=cfg.get("g2_pf", 2), sh_tn=self.sh_tn, sh_bm=g2_bm,
                                   sleep=sleep, nosum=nosum, mode=mode, sc1=x[0], wait=x[1], flag=x[2], carry=carry,
                                   hint=hint, stamps=stamps, reset=False)
            self.k1 = compile_gemm1_fill(self.tier, kw=kw, pf=3, bm=bm, **g1_kw)
        self.nrt1 = self.rows // bm * (I // 32)


_plans: dict = {}


_static: dict = {}
_spare: dict = {}
SPARE_SETS = 4


def _new_parts(dev):
    ps = torch.full((MAXN * 5 * H // 2,), -1, dtype=torch.int32, device=dev)
    return ps[:MAXN * 4 * H // 2], ps[MAXN * 4 * H // 2:]


def reserve(dev, sets=SPARE_SETS) -> None:
    """Allocate sentinel-filled buffer sets outside graph capture (warm-up); streams seen first under capture take one."""
    lst = _spare.setdefault((dev, 0), [])
    while len(lst) < sets:
        lst.append(_new_parts(dev))


def static_parts(dev, tag=0):
    """Per (device, stream, variant) sentinel-filled slot [MAXN, 4, H] / shared [MAXN, H] buffers of the data mode.
    Each stream owns its set (two streams never share one); a stream first seen under graph capture takes a set
    reserved at warm-up (no fill kernel is captured); None when none is left (the caller falls back)."""
    key = (dev, torch.cuda.current_stream(dev).cuda_stream, tag)
    b = _static.get(key)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            lst = _spare.get((dev, tag), [])
            if not lst:
                return None
            b = lst.pop()
        else:
            b = _new_parts(dev)
        _static[key] = b
    return b


def mxfp4_moe_back(x: torch.Tensor, w, *, moe_tiles=True, sleep=1, nosum=False, mode="flag",
                   xbits=(True, True, True), carry=True, hint=False, stamps=None, fill=False) -> torch.Tensor:
    """The BF16 reduced MoE output of ``mxfp4_moe`` (<= 16 rows) / ``ak_moe_tiles`` (17-128): same bits, the sum in
    the GEMM2 launch."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import chain
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4 import moe_sort_a16w4
    n, dev = x.shape[0], x.device
    key = (n, moe_tiles, sleep, nosum, mode, xbits, carry, hint, fill, stamps is not None)
    p = _plans.get(key)
    if p is None:
        p = _plans[key] = Plan(n, moe_tiles, sleep, nosum, mode, xbits, carry, hint, stamps is not None, fill)
    sw = w.shared_weights
    if p.mode == "data" and static_parts(dev, int(p.fill)) is None:
        return None
    (xq, xs), (q8, s8), wt, ids = chain._route_quant_small(x, w)
    rows = p.rows
    if p.inline:
        rargs, g2r = (ids.data_ptr(),), (ids.data_ptr(), wt.view(-1))
    elif p.id_rows:
        eids = torch.empty(rows // 16 + 1, device=dev, dtype=torch.int32)
        nv = torch.empty(rows, device=dev, dtype=torch.int32)
        rargs, g2r = (ids.view(-1), eids, nv), (nv, wt.view(-1), eids, nv)
    else:
        sids, wt2, eids, nv = moe_sort_a16w4(ids, wt, 128, p.bm)
        assert sids.numel() == rows
        rargs, g2r = (sids, eids, nv), (sids, wt2, eids, nv)
    q = torch.empty((rows, 384), device=dev, dtype=torch.uint8)
    s = torch.empty((rows, 24), device=dev, dtype=torch.uint8)
    hq = torch.empty((n, 768), device=dev, dtype=torch.uint8)
    hs = torch.empty((n, 24), device=dev, dtype=torch.uint8)
    if p.mode == "data":
        parts = static_parts(dev, int(p.fill))
        if parts is None:
            return None
        out, sh_out = parts
    else:
        out = torch.empty((n, 4, H), device=dev, dtype=torch.bfloat16)
        sh_out = torch.empty((n, H), device=dev, dtype=torch.bfloat16)
    fin = torch.empty((n, H), device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dev)
    if p.fill:
        _run_compiled(p.k1, xq.view(-1), xs.view(torch.uint8).view(-1), w.w13.view(-1), w.a4w4_scales.w13.view(-1),
                      q.view(-1), s.view(-1), *rargs, q8.view(-1), s8.view(-1), sw.w13.view(-1), sw.s13.view(-1),
                      hq.view(-1), hs.view(-1), out.view(-1), sh_out.view(-1), n, rows, p.nsh1, p.nrt1, stream)
    else:
        _run_compiled(p.k1, xq.view(-1), xs.view(torch.uint8).view(-1), w.w13.view(-1), w.a4w4_scales.w13.view(-1),
                      q.view(-1), s.view(-1), *rargs, q8.view(-1), s8.view(-1), sw.w13.view(-1), sw.s13.view(-1),
                      hq.view(-1), hs.view(-1), n, rows, p.nsh1, stream)
    ngrid = p.nsh2 + p.nrt + (0 if p.nosum else 3 * n)
    _run_compiled(p.k2, q.view(-1), s.view(-1), w.w2.view(-1), w.a4w4_scales.w2.view(-1), out.view(-1), *g2r,
                  hq.view(-1), hs.view(-1), sw.w2.view(-1), sw.s2.view(-1), sh_out.view(-1), fin.view(-1),
                  (stamps.data_ptr() if p.stamps else out.data_ptr() if p.mode == "data" else
                   flags(dev, int(p.nosum)).data_ptr()), n, rows, p.nsh2, p.nrt,
                  ngrid, *(() if p.inline else (eids.data_ptr(), nv.data_ptr())), stream)
    return (out, sh_out) if p.nosum else fin


def supported(n: int) -> bool:
    return 1 <= n <= MAXN


def warm_up(w, moe_tiles=True) -> None:
    """Compiles and loads every row count's launches (data mode) before graph capture and reserves the buffer sets."""
    dev = w.router_weight.device
    reserve(dev)
    for n in range(1, MAXN + 1):
        mxfp4_moe_back(torch.zeros((n, H), device=dev, dtype=torch.bfloat16), w, moe_tiles=moe_tiles, mode="data")
    torch.cuda.synchronize()


@functools.cache
def compile_joint_nosync(*, pairs, sh_bm1, sh_kw1, sh_pf1, sh_tn, g2_pf=2):
    """TIMING CEILING ONLY (wrong without the producers' previous results in the buffers): the id-rows GEMM1 launch
    and GEMM2 launch (no sum) as one grid [shared G1 | routed G1 | shared G2 | routed G2] with NO GEMM1 -> GEMM2
    ordering. Under graph replay the stale q / EIDS / NV are the previous replay's (same input), so the bytes moved are
    the real ones; it bounds what a flagged GEMM1 -> GEMM2 launch with W2 prefetch could save."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_id_rows import compile_gemm1
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_gemm1_a8w8 import compile_gemm1_tile
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.routing_ids import WAVE_BYTES
    bm = 16
    sh1 = compile_gemm1_tile(kw=sh_kw1, pf=sh_pf1, bm=sh_bm1)
    rt1 = compile_gemm1(kw=4, pf=3, bm=bm, pairs=pairs)
    sh_ex = 4 * (max(bm, sh_bm1 if sh_kw1 > 1 else 16) // 16) * 4 * 64 * 16
    lds1 = max(sh_ex, 4 * (bm // 16) * 4 * 64 * 16 + 4 * WAVE_BYTES)
    sh2 = _shared(sh_tn, 16, 2, sh_tn, False, False, False)
    rt2 = _routed("id_rows", 256, bm, g2_pf, sh_tn, False, False, False)
    lds = max(lds1, bm * 256 * 4, sh2.lds_bytes)
    name = f"mbf_joint_nosync_id_rows_p{pairs}_sh{sh_bm1}_{sh_kw1}_{sh_pf1}_tn{sh_tn}_pf{g2_pf}"

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(XA: fx.Tensor, XS: fx.Tensor, W1: fx.Tensor, WS1: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
               IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, XA8: fx.Tensor, XS8: fx.Tensor, W18: fx.Tensor,
               WS18: fx.Tensor, HQ: fx.Tensor, HS: fx.Tensor, W2: fx.Tensor, WS2: fx.Tensor, OUT: fx.Tensor,
               WT: fx.Tensor, W28: fx.Tensor, WS28: fx.Tensor, OUT8: fx.Tensor,
               m: fx.Int32, rows: fx.Int32, nsh1: fx.Int32, nrt1: fx.Int32, nsh2: fx.Int32, EP: fx.Int64, NP: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        b2 = nsh1 + nrt1
        if bid < nsh1:
            sh1(base, bid, XA8, XS8, W18, WS18, HQ, HS, m, m)
        if bid >= nsh1:
            if bid < b2:
                rt1(base, bid - nsh1, XA, XS, W1, WS1, Q, S, IDS, EIDS, NV, m, rows)
        if bid >= b2:
            if bid < b2 + nsh2:
                sh2(base, bid - b2, HQ, HS, W28, WS28, OUT8, fx.Int64(0), m)
        if bid >= b2 + nsh2:
            rt2(base, bid - b2 - nsh2, Q, S, W2, WS2, OUT, NV, WT, EIDS, NV, fx.Int64(0), m, rows, EP, NP)

    @flyc.jit
    def launch_joint(XA: fx.Tensor, XS: fx.Tensor, W1: fx.Tensor, WS1: fx.Tensor, Q: fx.Tensor, S: fx.Tensor,
                     IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, XA8: fx.Tensor, XS8: fx.Tensor, W18: fx.Tensor,
                     WS18: fx.Tensor, HQ: fx.Tensor, HS: fx.Tensor, W2: fx.Tensor, WS2: fx.Tensor, OUT: fx.Tensor,
                     WT: fx.Tensor, W28: fx.Tensor, WS28: fx.Tensor, OUT8: fx.Tensor,
                     m: fx.Int32, rows: fx.Int32, nsh1: fx.Int32, nrt1: fx.Int32, nsh2: fx.Int32, ngrid: fx.Int32,
                     EP: fx.Int64, NP: fx.Int64, stream: fx.Stream):
        kernel(XA, XS, W1, WS1, Q, S, IDS, EIDS, NV, XA8, XS8, W18, WS18, HQ, HS, W2, WS2, OUT, WT, W28, WS28,
               OUT8, m, rows, nsh1, nrt1, nsh2, EP, NP).launch(grid=(ngrid, 1, 1), block=(256, 1, 1), stream=stream)
    launch_joint.kernel_name = name
    return launch_joint


_jbufs: dict = {}


def joint_ceiling(x: torch.Tensor, w, *, moe_tiles=True):
    """Timing-only arm: router launches + ONE GEMM1/GEMM2 launch without ordering (id-rows tier 16-96 rows); returns
    (slots, shared) partials. Its buffers persist per (n, layer weights) so graph replays see their own last results."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import chain
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_small as XS
    n, dev = x.shape[0], x.device
    assert XS.id_rows_for(n)
    p = _plans.get((n, moe_tiles, 1, True, "flag", (False, False, False), True))
    if p is None:
        p = _plans[(n, moe_tiles, 1, True, "flag", (False, False, False), True)] = Plan(n, moe_tiles, 1, True, "flag",
                                                                                        (False, False, False))
    cfg = tiles_for(n, moe_tiles)
    sh_bm, sh_kw, sh_pf = XS.shared_fold_tile_for(n)
    sh_bm, sh_pf = cfg.get("sh_bm", sh_bm), cfg.get("sh_pf", sh_pf)
    k = compile_joint_nosync(pairs=XS.id_pairs_for(n), sh_bm1=sh_bm, sh_kw1=sh_kw, sh_pf1=sh_pf, sh_tn=p.sh_tn,
                             g2_pf=cfg.get("g2_pf", 2))
    sw = w.shared_weights
    (xq, xs), (q8, s8), wt, ids = chain._route_quant_small(x, w)
    rows = p.rows
    key = (n, id(w))
    b = _jbufs.get(key)
    if b is None:
        b = _jbufs[key] = dict(
            eids=torch.zeros(rows // 16 + 1, device=dev, dtype=torch.int32),
            nv=torch.zeros(rows, device=dev, dtype=torch.int32),
            q=torch.zeros((rows, 384), device=dev, dtype=torch.uint8), s=torch.zeros((rows, 24), device=dev, dtype=torch.uint8),
            out=torch.zeros((n, 4, H), device=dev, dtype=torch.bfloat16), hq=torch.zeros((n, 768), device=dev, dtype=torch.uint8),
            hs=torch.zeros((n, 24), device=dev, dtype=torch.uint8), sh=torch.zeros((n, H), device=dev, dtype=torch.bfloat16))
    nrt1 = rows // 16 * (I // 32)
    ngrid = p.nsh1 + nrt1 + p.nsh2 + p.nrt
    stream = torch.cuda.current_stream(dev)
    _run_compiled(k, xq.view(-1), xs.view(torch.uint8).view(-1), w.w13.view(-1), w.a4w4_scales.w13.view(-1),
                  b["q"].view(-1), b["s"].view(-1), ids.view(-1), b["eids"], b["nv"], q8.view(-1), s8.view(-1),
                  sw.w13.view(-1), sw.s13.view(-1), b["hq"].view(-1), b["hs"].view(-1), w.w2.view(-1),
                  w.a4w4_scales.w2.view(-1), b["out"].view(-1), wt.view(-1), sw.w2.view(-1), sw.s2.view(-1),
                  b["sh"].view(-1), n, rows, p.nsh1, nrt1, p.nsh2, ngrid, b["eids"].data_ptr(), b["nv"].data_ptr(),
                  stream)
    return b["out"], b["sh"]


@functools.cache
def compile_gemm1_fill(tier, *, kw=4, pf=3, bm=16, sh_bm=16, sh_kw=4, sh_pf=2, pairs=256):
    """amd-kernels' routed GEMM1 launch (shared gate / up tiles first) + 3m trailing workgroups that write the
    0xFFFF_FFFF sentinel over the next GEMM2's slot / shared rows (the static buffers of the data mode), so the GEMM2
    launch's sum workgroups need not restore it at the end of the critical path. The GEMM1 tiles are production's
    bodies; the fill workgroups run in GEMM1's idle tail."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Vector as Vec
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_gemm1_a8w8 import compile_gemm1_tile
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.routing_ids import WAVE_BYTES
    assert tier in ("id_rows", "small") and kw == 4 and bm == 16
    if tier == "id_rows":
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_id_rows import compile_gemm1
        rt = compile_gemm1(kw=kw, pf=pf, bm=bm, pairs=pairs)
        lds = max(4 * (max(bm, sh_bm if sh_kw > 1 else 16) // 16) * 4 * 64 * 16, 4 * (bm // 16) * 4 * 64 * 16 + 4 * WAVE_BYTES)
    else:
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_small import compile_gemm1
        rt = compile_gemm1(kw=kw, pf=pf, bm=bm)
        lds = max(4 * (max(bm, sh_bm if sh_kw > 1 else 16) // 16) * 4 * 64 * 16, 4 * (bm // 16) * 4 * 64 * 16)
    sh = compile_gemm1_tile(kw=sh_kw, pf=sh_pf, bm=sh_bm)
    name = f"mbf_gemm1_{tier}_fill_bm{bm}_kw{kw}_pf{pf}_p{pairs}_sh{sh_bm}_{sh_kw}_{sh_pf}"

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.jit
    def fill(fid, P, S):
        tx = fx.thread_idx.x
        row, chunk = fid // 3, fid % 3
        col = chunk * 2048 + tx * 8
        pr = buffer_ops.create_buffer_resource(P, max_size=False, num_records_bytes=MAXN * 4 * 6144 * 2)
        sr = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=MAXN * 6144 * 2)
        neg = Vec.filled(4, -1, fx.Int32)
        for k in range_constexpr(4):
            buffer_ops.buffer_store(neg, pr, ((row * 4 + k) * 6144 + col) // 2)
        buffer_ops.buffer_store(neg, sr, (row * 6144 + col) // 2)

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
               Q: fx.Tensor, S: fx.Tensor, IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor,
               A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, Q8: fx.Tensor, S8: fx.Tensor,
               PF: fx.Tensor, SF: fx.Tensor, m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, nrt: fx.Int32):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        if bid < nsh:
            sh(base, bid, A8, AS8, W8, WS8, Q8, S8, m, m)
        if bid >= nsh:
            if bid < nsh + nrt:
                rt(base, bid - nsh, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows)
        if bid >= nsh + nrt:
            fill(bid - nsh - nrt, PF, SF)

    @flyc.jit
    def launch_gemm1_fill(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
                          Q: fx.Tensor, S: fx.Tensor, IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor,
                          A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, Q8: fx.Tensor, S8: fx.Tensor,
                          PF: fx.Tensor, SF: fx.Tensor, m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, nrt: fx.Int32,
                          stream: fx.Stream):
        kernel(A, AS, W, WS, Q, S, IDS, EIDS, NV, A8, AS8, W8, WS8, Q8, S8, PF, SF, m, rows, nsh, nrt).launch(
            grid=(nsh + nrt + 3 * m, 1, 1), block=(256, 1, 1), stream=stream)
    launch_gemm1_fill.kernel_name = name
    return launch_gemm1_fill
