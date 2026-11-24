# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gemm2_a4w4_small.py`` taking the block's expert and its 16 packed rows from ``gemm1_a4w4_id_rows`` (no sort
launch, no counters)."""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .gemm1_a4w4_small import _mfma, _hold, _raw, _lds_ptr
from amd_kernels.warm_up import launcher

H, I, E = 6144, 768, 128


@functools.cache
def compile_gemm2(*, tn=256, bm=16, pf=2):
    assert tn in (128, 256, 512) and bm == 16
    ni, mr = tn // 64, bm // 16

    @flyc.jit
    def routed_tile(base, rb, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = rb // (H // tn), rb % (H // tn)
        record = fx.Int32(EIDS[mb])
        blocks = fx.Int32(EIDS[rows // 16])
        fx.rocdl.sched_barrier(0)
        if mb < blocks:
            expert = record & 127
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=rows * (I // 2))
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=rows * (I // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=H * I // 2,
                                                   base_byte_offset=fx.Int64(expert) * (H * I // 2))
            sw = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * H * (I // 32))
            out = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=m * 4 * H * 2)
            ad = [((mb * bm + mi * 16 + l16) * (I // 2) + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [(mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)]
            wd = [((nb * (tn // 16) + wave * ni + j) * 6 * 1024 + q16 * 256 + l16 * 16) // 4 for j in range_constexpr(ni)]
            swd = [((expert * (H // 32) + nb * (tn // 32) + (wave * ni + j) // 2) * 3) * 64 + q16 * 16 + l16
                   for j in range_constexpr(ni)]
            def load(k):
                aa = [buffer_ops.buffer_load(ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64) for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32, soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                bb = [buffer_ops.buffer_load(wr, wd[j], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 1024, cache_modifier=2) for j in range_constexpr(ni)]
                sb = [buffer_ops.buffer_load(sw, swd[j], vec_width=1, dtype=fx.Int32, soffset_bytes=(k // 2) * 256) for j in range_constexpr(ni)]
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
                packed = fx.Int32(IDS[mb * 16 + row])
                token, slot = packed & 0xFFFFFF, packed.shrui(fx.Int32(24))
                pair = (token < m).select(token * 4 + slot, fx.Int32(0))
                weight = fx.Float32(WT[pair])
                for c in range_constexpr(tn // 64):
                    col = tx % 32 * 2 + c * 64
                    v = Vec(llvm.load(T.vec(2, T.f32), _lds_ptr(base, (row * tn + col) * 4), alignment=8))
                    pk = (v * weight).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    buffer_ops.buffer_store(pk, out, ((token * 4 + slot) * H + nb * tn + col) // 2, mask=token < m)
    return routed_tile


@functools.cache
@launcher
def compile_gemm2_shared_first(*, tn=256, bm=16, pf=2, sh_tn=256, sh_bm=16):
    """``compile_gemm2`` whose first ``nsh`` workgroups run the shared expert's MXFP8 down
    tiles (``shared_gemm2_a8w8.compile_gemm2(tn=sh_tn, bm=sh_bm)``, its own output); routed as alone."""
    from .shared_gemm2_a8w8 import compile_gemm2 as compile_shared_gemm2
    assert bm in (16, 32) and sh_bm in (16, 32, 64)
    sh_name = "" if sh_bm == 16 else f"_bm{sh_bm}"
    name = f"moe_gemm2_id_rows_a4w4_partials_h6144_i768_bm{bm}_tn{tn}_pf{pf}_shared_a8w8{sh_name}_tn{sh_tn}_first"
    sh = compile_shared_gemm2(tn=sh_tn, bm=sh_bm, pf=2)
    sh_tile, routed_tile = sh, compile_gemm2(tn=tn, bm=bm, pf=pf)
    lds = max(bm * tn * 4, sh.lds_bytes)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
               OUT: fx.Tensor, IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor,
               A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, OUT8: fx.Tensor,
               m: fx.Int32, rows: fx.Int32, nsh: fx.Int32):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        if bid < nsh:
            sh_tile(base, bid, A8, AS8, W8, WS8, OUT8, m)
        if bid >= nsh:
            routed_tile(base, bid - nsh, A, AS, W, WS, OUT, IDS, WT, EIDS, NV, m, rows)

    @flyc.jit
    def launch_id_rows_gemm2_a4w4_small_shared_first(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor,
                OUT: fx.Tensor, IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor,
                A8: fx.Tensor, AS8: fx.Tensor, W8: fx.Tensor, WS8: fx.Tensor, OUT8: fx.Tensor,
                m: fx.Int32, rows: fx.Int32, nsh: fx.Int32, stream: fx.Stream):
        kernel(A, AS, W, WS, OUT, IDS, WT, EIDS, NV, A8, AS8, W8, WS8, OUT8, m, rows, nsh).launch(
            grid=(rows // bm * (H // tn) + nsh, 1, 1), block=(256, 1, 1), stream=stream)
    launch_id_rows_gemm2_a4w4_small_shared_first.kernel_name = name
    return launch_id_rows_gemm2_a4w4_small_shared_first
