# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gemm1_a4w4_small.py`` whose blocks build their rows from ``topk_ids`` (``routing_ids``: no sort launch, no atomic
counters): every wave histograms the M*4 routing pairs in its own LDS slice, scans the 128 counts into the canonical
expert blocks (experts ascending, ceil(count / 16) blocks each) and finds its block's rows by ballot in pair order.
Column group 0 publishes the block's expert and its 16 packed rows for GEMM2.
"""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.warm_up import launcher
from .routing_ids import wave_expert, wave_block_rows, WAVE_BYTES

H, I, E = 6144, 768, 128


def _raw(x):
    """Unwrap only at LLVM intrinsic and inline-assembly boundaries."""
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mfma(a, b, acc, sa, sb, b_byte, transpose=False):
    """Native gfx950 FP4 scaled MFMA into AGPRs; transpose selects GEMM1 C layout."""
    ty = T.vec(4, T.f32)
    a_byte = 0
    if transpose:
        a, b, sa, sb = b, a, sb, sa
        a_byte, b_byte = b_byte, 0
    asm = (f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {'$0' if acc is not None else '0'}, $3, $4 "
           f"op_sel:[{a_byte % 2},{b_byte % 2},0] op_sel_hi:[{a_byte // 2},{b_byte // 2},0] cbsz:4 blgp:4")
    ops = [_raw(a), _raw(b), _raw(sa), _raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def _hold(xs):
    """Keep the required inline-MFMA latency before any accumulator reads."""
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(xs)) + ")>")
    cons = ",".join(["=a"] * len(xs) + [str(i) for i in range(len(xs))])
    v = llvm.inline_asm(ty, [_raw(x) for x in xs], "s_nop 7\ns_nop 7\ns_nop 7", cons, has_side_effects=True)
    return [llvm.extractvalue(T.vec(4, T.f32), v, [i]) for i in range(len(xs))]


def _lds_ptr(base, off):
    """Byte-address the aligned LDS exchange and transpose regions."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(base + off)))


def _fp4(old, a, b, scale, sel):
    """FlyDSL lacks a wrapper for this FP32-to-FP4 intrinsic."""
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
        [_raw(old), _raw(a), _raw(b), _raw(scale), _raw(fx.Int32(sel))], [], []))


def _swiglu(g, u):
    g = (g < fx.Float32(7)).select(g, fx.Float32(7))
    u = (u < fx.Float32(7)).select(u, fx.Float32(7))
    u = (u > fx.Float32(-7)).select(u, fx.Float32(-7))
    t = (g * fx.Float32(1.702)) * fx.Float32(-1.4426950408889634)
    e = fx.Float32(fx.rocdl.exp2(T.f32, _raw(t)))
    r = fx.Float32(fx.rocdl.rcp(T.f32, _raw(fx.Float32(1) + e)))
    return (g * r) * (u + fx.Float32(1))


@functools.cache
def compile_gemm1(*, kw=1, pf=3, bm=16, pairs=256):
    assert kw in (1, 2, 4) and bm in (16, 32) and pairs in (64, 128, 256, 512)
    nw = 4 // kw
    tn, kt, mr = nw * 32, 48 // kw, bm // 16
    assert bm == 16
    ex = 4 * mr * 4 * 64 * 16
    route = ex if kw > 1 else 16  # the waves' routing slices follow the K-wave exchange

    @flyc.jit
    def routed_tile(base, rb, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        mb, nb = rb // (I // tn), rb % (I // tn)
        hb = base + route + wave * WAVE_BYTES
        expert, first, blocks, pair_ids = wave_expert(IDS, m, mb, lane, hb, pairs)
        if (tx == 0) & (nb == 0):
            if mb == 0:
                EIDS[rows // 16] = blocks
            if mb < blocks:
                EIDS[mb] = expert
        if mb < blocks:
            col = nb * tn + nk * 32
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=m * (H // 2))
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=m * (H // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=I * H,
                                                   base_byte_offset=fx.Int64(expert) * (I * H))
            ws = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * I * (H // 16))
            qr = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=rows * (I // 2))
            ss = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=rows * (I // 32))
            wd = [[(((col // 16 + ni) * 2 + gu) * 48 * 1024 + q16 * 256 + l16 * 16) // 4 + kk * kt * 256
                   for gu in range_constexpr(2)] for ni in range_constexpr(2)]
            swd = [((expert * 48 + col // 16 + ni) * 24) * 64 + q16 * 16 + l16 + kk * (kt // 2) * 64
                   for ni in range_constexpr(2)]

            bpre = [[buffer_ops.buffer_load(wr, wd[ni][gu], vec_width=4, dtype=fx.Int32, cache_modifier=2)
                      for gu in range_constexpr(2)] for ni in range_constexpr(2)]
            spre = [buffer_ops.buffer_load(ws, swd[ni], vec_width=1, dtype=fx.Int32) for ni in range_constexpr(2)]
            # the rows after the first weight loads are in flight
            packed = wave_block_rows(pair_ids, expert, first, lane, hb)
            if (tx < 16) & (nb == 0):
                NV[mb * 16 + tx] = packed
            tok = [packed & 0xFFFFFF for mi in range_constexpr(mr)]
            ad = [(tok[mi] * (H // 2) + kk * (kt * 64) + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [tok[mi] * (H // 32) // 4 + kk * kt for mi in range_constexpr(mr)]
            def load(k):
                aa = [buffer_ops.buffer_load(ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64)
                      for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32,
                                                    soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                bb = bpre if k == 0 else [[buffer_ops.buffer_load(wr, wd[ni][gu], vec_width=4, dtype=fx.Int32,
                                             soffset_bytes=k * 1024, cache_modifier=2)
                       for gu in range_constexpr(2)] for ni in range_constexpr(2)]
                sb = spre if k == 0 else [buffer_ops.buffer_load(ws, swd[ni], vec_width=1, dtype=fx.Int32,
                                            soffset_bytes=(k // 2) * 256) for ni in range_constexpr(2)]
                return aa, sa, bb, sb

            acc = [[[None] * 2 for _ in range(2)] for _ in range(mr)]
            ring = [load(k) for k in range_constexpr(pf)]
            for k in range_constexpr(kt):
                if const_expr(k + pf < kt):
                    ring.append(load(k + pf))
                aa, sa, bb, sb = ring.pop(0)
                for mi in range_constexpr(mr):
                    for ni in range_constexpr(2):
                        for gu in range_constexpr(2):
                            acc[mi][ni][gu] = _mfma(aa[mi], bb[ni][gu], acc[mi][ni][gu],
                                                  sa[mi], sb[ni], (k % 2) * 2 + gu, True)
            flat = _hold([acc[mi][ni][gu] for mi in range(mr) for ni in range(2) for gu in range(2)])
            if const_expr(kw > 1):
                for j in range_constexpr(mr * 4):
                    llvm.StoreOp(_raw(flat[j]), _lds_ptr(base, ((wave * mr * 4 + j) * 64 + lane) * 16), alignment=16)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if kk == 0:
                vals = [Vec(v) for v in flat]
                for k in range_constexpr(1, kw):
                    for j in range_constexpr(mr * 4):
                        other = Vec(llvm.load(T.vec(4, T.f32), _lds_ptr(base, (((nk + k * nw) * mr * 4 + j) * 64 + lane) * 16), alignment=16))
                        vals[j] = vals[j] + other
                for mi in range_constexpr(mr):
                    y = [[_swiglu(fx.Float32(vals[mi * 4 + ni * 2][v]), fx.Float32(vals[mi * 4 + ni * 2 + 1][v]))
                          for v in range_constexpr(4)] for ni in range_constexpr(2)]
                    amax = fx.math.absf(y[0][0])
                    for ni in range_constexpr(2):
                        for v in range_constexpr(4):
                            z = fx.math.absf(y[ni][v])
                            amax = (amax > z).select(amax, z)
                    for sh in range_constexpr(2):
                        z = amax.shuffle_xor(16 << sh, 64)
                        amax = (amax > z).select(amax, z)
                    e = (amax.bitcast(fx.Int32) - fx.Int32(0xE00000)) >> 23
                    e = (e > 1).select(e, fx.Int32(1))
                    scale = (e << 23).bitcast(fx.Float32)
                    row = mb * bm + mi * 16 + l16
                    for ni in range_constexpr(2):
                        pk = _fp4(fx.Int32(0), y[ni][0], y[ni][1], scale, 0)
                        pk = _fp4(pk, y[ni][2], y[ni][3], scale, 1)
                        off = row * (I // 2) + col // 2 + ni * 8 + q16 * 2
                        buffer_ops.buffer_store(pk.to(fx.Int16), qr, off, offset_is_bytes=True, mask=tok[mi] < m)
                    buffer_ops.buffer_store(e.to(fx.Int8), ss, row * (I // 32) + col // 32,
                                            offset_is_bytes=True, mask=(tok[mi] < m) & (q16 == 0))
    return routed_tile


@functools.cache
@launcher
def compile_gemm1_shared_first(*, kw=4, pf=3, bm=16, sh_bm=16, sh_kw=4, sh_pf=2, pairs=256):
    """``compile_gemm1`` whose first ``nsh`` workgroups run the shared expert's unsplit MXFP8
    gate/up tiles (``shared_gemm1_a8w8.compile_gemm1_tile``, ``sh_bm`` rows), so the shared
    GEMM1 streams next to the routed tiles in one launch; the routed tiles compute exactly as alone."""
    from .shared_gemm1_a8w8 import compile_gemm1_tile
    assert kw == 4 and bm in (16, 32) and sh_bm in (16, 32, 64) and sh_kw in (1, 2, 4)
    tn = 32
    # shared workgroups use their exchange region; routed ones their exchange + routing slices
    sh_ex = 4 * (max(bm, sh_bm if sh_kw > 1 else 16) // 16) * 4 * 64 * 16
    lds = max(sh_ex, 4 * (bm // 16) * 4 * 64 * 16 + 4 * WAVE_BYTES)
    sh_name = (("" if sh_bm == 16 else f"_bm{sh_bm}") + ("" if sh_kw == 4 else f"_kw{sh_kw}")
               + ("" if sh_pf == 2 else f"_pf{sh_pf}"))
    name = f"moe_gemm1_id_rows_a4w4_swiglu_quant_h6144_i768_bm{bm}_tn{tn}_kw{kw}_pf{pf}_p{pairs}_shared_a8w8{sh_name}_first"
    sh_tile = compile_gemm1_tile(kw=sh_kw, pf=sh_pf, bm=sh_bm)
    routed_tile = compile_gemm1(kw=kw, pf=pf, bm=bm, pairs=pairs)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
               Q: fx.Tensor, S: fx.Tensor, IDS: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor,
               A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, Q8: fx.Tensor, S8: fx.Tensor,
               m: fx.Int32, rows: fx.Int32, nsh: fx.Int32):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        if bid < nsh:
            sh_tile(base, bid, A8, AS8, W8, WS8, Q8, S8, m, m)
        if bid >= nsh:
            routed_tile(base, bid - nsh, A, AS, W, WS, Q, S, IDS, EIDS, NV, m, rows)

    @flyc.jit
    def launch_id_rows_gemm1_a4w4_small_shared_first(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
                                             Q: fx.Tensor, S: fx.Tensor, IDS: fx.Tensor, EIDS: fx.Tensor,
                                             NV: fx.Tensor, A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor,
                                             WS8: fx.Tensor, Q8: fx.Tensor, S8: fx.Tensor,
                                             m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, stream: fx.Stream):
        kernel(A, AS, W, WS, Q, S, IDS, EIDS, NV, A8, AS8, W8, WS8, Q8, S8, m, rows, nsh).launch(
            grid=(rows // bm * (I // tn) + nsh, 1, 1), block=(256, 1, 1), stream=stream)
    launch_id_rows_gemm1_a4w4_small_shared_first.kernel_name = name
    return launch_id_rows_gemm1_a4w4_small_shared_first
