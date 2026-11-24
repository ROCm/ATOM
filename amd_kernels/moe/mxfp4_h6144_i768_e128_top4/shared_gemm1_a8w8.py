# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared-expert MXFP8 gate/up tile bodies with direct operands (``SharedWeights`` layout), FP32 SwiGLU and MXFP8
quant, which the routed small-tile GEMM1s run first in their launch (``compile_gemm1_tile``).

One wave owns 32 intermediate columns with their whole quant group; K-waves sum their FP32 partials through LDS in
rank order.
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

from .shared_quant import _e8m0_fp8, _fp8
from amd_kernels.warm_up import launcher

H, I, E = 6144, 768, 1


def _raw(x):
    """The IR value of ``x`` for LLVM intrinsics and inline asm."""
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mfma(a, b, acc, sa, sb, b_byte, transpose=False):
    """Native gfx950 MXFP8 scaled MFMA into AGPRs; transpose selects GEMM1 C layout."""
    ty = T.vec(4, T.f32)
    a_byte = 0
    if transpose:
        a, b, sa, sb = b, a, sb, sa
        a_byte, b_byte = b_byte, 0
    asm = (f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {'$0' if acc is not None else '0'}, $3, $4 "
           f"op_sel:[{a_byte % 2},{b_byte % 2},0] op_sel_hi:[{a_byte // 2},{b_byte // 2},0] cbsz:0 blgp:0")
    ops = [_raw(a), _raw(b), _raw(sa), _raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(_raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def _hold(xs):
    """Keep the required inline-MFMA latency before any accumulator reads."""
    if len(xs) == 1:  # one output cannot be an LLVM inline-asm struct
        return [llvm.inline_asm(T.vec(4, T.f32), [_raw(xs[0])], "s_nop 7\ns_nop 7\ns_nop 7", "=a,0",
                                has_side_effects=True)]
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(xs)) + ")>")
    cons = ",".join(["=a"] * len(xs) + [str(i) for i in range(len(xs))])
    v = llvm.inline_asm(ty, [_raw(x) for x in xs], "s_nop 7\ns_nop 7\ns_nop 7", cons, has_side_effects=True)
    return [llvm.extractvalue(T.vec(4, T.f32), v, [i]) for i in range(len(xs))]


def _lds_ptr(base, off):
    """Byte-address the aligned LDS exchange and transpose regions."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(base + off)))


def _swiglu(g, u):
    """SwiGLU-OAI in fp32: g <= 7, -7 <= u <= 7, g * sigmoid(1.702 g) * (u + 1), the sigmoid by exp2 + rcp."""
    g = (g < fx.Float32(7)).select(g, fx.Float32(7))
    u = (u < fx.Float32(7)).select(u, fx.Float32(7))
    u = (u > fx.Float32(-7)).select(u, fx.Float32(-7))
    t = (g * fx.Float32(1.702)) * fx.Float32(-1.4426950408889634)
    e = fx.Float32(fx.rocdl.exp2(T.f32, _raw(t)))
    r = fx.Float32(fx.rocdl.rcp(T.f32, _raw(fx.Float32(1) + e)))
    return (g * r) * (u + fx.Float32(1))


def _load8(rsrc, off, soff, plane=16):
    """Two 16-byte loads ``plane`` dwords apart as one 8-dword MFMA operand."""
    lo = Vec(buffer_ops.buffer_load(rsrc, off, vec_width=4, dtype=fx.Int32, soffset_bytes=soff))
    hi = Vec(buffer_ops.buffer_load(rsrc, off + plane, vec_width=4, dtype=fx.Int32, soffset_bytes=soff))
    return Vec.from_elements([lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Int32)


def _swiglu_quant_store(vals, mr, row0, col, l16, q16, tok, m, qr, ss):
    """FP32 gate/up sums of one wave's 32 columns -> SwiGLU -> MXFP8 codes + e8m0."""
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
        e = _e8m0_fp8(amax)
        scale = (e << 23).bitcast(fx.Float32)
        row = row0 + mi * 16 + l16
        for ni in range_constexpr(2):
            pk = _fp8(fx.Int32(0), y[ni][0], y[ni][1], scale, False)
            pk = _fp8(pk, y[ni][2], y[ni][3], scale, True)
            off = row * I + col + ni * 16 + q16 * 4
            buffer_ops.buffer_store(pk, qr, off // 4, mask=tok[mi] < m)
        buffer_ops.buffer_store(e.to(fx.Int8), ss, row * (I // 32) + col // 32,
                                offset_is_bytes=True, mask=(tok[mi] < m) & (q16 == 0))


@functools.cache
def compile_gemm1(*, kw=1, pf=3, bm=16):
    """The body of a gate/up tile of ``bm`` rows x 128 / ``kw`` columns: ``kw`` K-waves, ``pf`` K-steps in flight."""
    # a K range starts on an even 128-K step: one weight-scale word covers two steps
    assert kw in (1, 2, 4) and bm in (16, 32, 64) and 48 % kw == 0 and 48 // kw % 2 == 0
    nw = 4 // kw
    tn, kt, mr = nw * 32, 48 // kw, bm // 16

    @flyc.jit
    def tile(base, bid, A, AS, W, WS, Q, S, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        nk, kk = wave % nw, wave // nw
        tb, kg = bid, kk
        mb, nb = tb // (I // tn), tb % (I // tn)
        if mb * bm < m:
            col = nb * tn + nk * 32
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=m * H)
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=m * (H // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=2 * I * H)
            ws = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=I * (H // 16))
            qr = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=m * I)
            ss = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=m * (I // 32))
            tok = [mb * bm + mi * 16 + l16 for mi in range_constexpr(mr)]
            ad = [(tok[mi] * H + kg * (kt * 128) + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [tok[mi] * (H // 32) // 4 + kg * kt for mi in range_constexpr(mr)]
            wd = [[((col // 16 + ni + gu * 48) * 48 * 2048 + q16 * 256 + l16 * 16) // 4 + kg * kt * 512
                   for gu in range_constexpr(2)] for ni in range_constexpr(2)]
            swd = [((col // 32 + gu * 24) * 24) * 64 + q16 * 16 + l16 + kg * (kt // 2) * 64
                   for gu in range_constexpr(2)]

            def load(k):
                aa = [_load8(ar, ad[mi], k * 128) for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32,
                                                    soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                bb = [[_load8(wr, wd[ni][gu], k * 2048, 256)
                       for gu in range_constexpr(2)] for ni in range_constexpr(2)]
                sb = [buffer_ops.buffer_load(ws, swd[gu], vec_width=1, dtype=fx.Int32,
                                            soffset_bytes=(k // 2) * 256) for gu in range_constexpr(2)]
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
                                                  sa[mi], sb[gu], (k % 2) * 2 + ni, True)
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
                _swiglu_quant_store(vals, mr, mb * bm, col, l16, q16, tok, m, qr, ss)

    return tile


@launcher
def compile_gemm1_tile(*, kw=4, pf=2, bm=16):
    """The gate/up tile of ``compile_gemm1(kw, pf, bm)`` as a body another launch runs on its workgroup ``bid``:
    ``tile(base, bid, A, AS, W, WS, Q, S, m, rows)``, ``base`` = its LDS."""
    return compile_gemm1(kw=kw, pf=pf, bm=bm)
