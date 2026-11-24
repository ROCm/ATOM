# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared MXFP8 down-GEMM tile bodies with direct operands and one BF16 writer per output, which the routed small-tile
GEMM2s run first in their launch."""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .shared_gemm1_a8w8 import _mfma, _hold, _raw, _lds_ptr, _load8

H, I, E = 6144, 768, 128


@functools.cache
def compile_gemm2(*, tn=256, bm=16, pf=2):
    """The body ``tile(base, bid, A, AS, W, WS, OUT, m)`` of a down-GEMM tile of ``bm`` rows x ``tn`` columns with
    ``pf`` K-steps in flight; the output goes through its ``lds_bytes`` of LDS from ``base``."""
    # tn 64: one 16-column subtile per wave, so the wave's half of a 32-column scale word is runtime (wave parity);
    # its bytes are shifted into the even slot
    assert tn in (64, 128, 256, 512) and bm in (16, 32, 64)
    ni, mr = tn // 64, bm // 16

    @flyc.jit
    def tile(base, bid, A, AS, W, WS, OUT, m):
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
                    buffer_ops.buffer_store(pk, out, (token * H + nb * tn + col) // 2, mask=token < m)

    tile.lds_bytes = bm * tn * 4
    return tile
