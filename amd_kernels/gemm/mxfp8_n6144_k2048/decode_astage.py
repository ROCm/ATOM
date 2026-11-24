# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 N6144 K2048 GEMM on 64x48 tiles and immutable activation LDS.

All four waves stage A and two 32-column/row scale groups. Three waves then
compute 64x16 columns each, with the fourth finished after the only barrier.
A 139264-byte allocation keeps one workgroup per CU. The M97-128 producer's
64-row codes, scale layouts, increasing-K accumulation and BF16 output stay
unchanged. The constants come from a cold-weight screen.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_4wave import (
    G2SLoaderAsm,
    MfmaRowStationary,
    S2RLoader,
    _Buf,
    _global_swizzle,
    _pack8,
    _s2r_thunks,
)
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import _pin_accumulators

# Fixed at weight shape, independent of M. Weight cache 2 is non-temporal.
_N, _K, _BLOCK_M, _BLOCK_N = 6144, 2048, 64, 48
_LOOKAHEAD_W, _WEIGHT_CACHE_MOD = 8, 2


def compile_mxfp8_gemm_n6144_k2048_astage():
    LA, WCM = _LOOKAHEAD_W, _WEIGHT_CACHE_MOD
    N, K, BM, BN = _N, _K, _BLOCK_M, _BLOCK_N
    NA, SW, SCB, AB = BM // 16, K // 128, K // 256, BM * 128
    name = "flydsl_mxfp8_astage_n6144_k2048_64x48_nw3_svec"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Int8, SW * AB, 16]
        scales: fx.Array[fx.Int32, 2048, 16]

    @flyc.kernel(name=name, known_block_size=(256, 1, 1))
    def kernel(
        A: fx.Tensor, W: fx.Tensor, C: fx.Tensor, AS: fx.Tensor, WS: fx.Tensor,
        m: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // 64))
        bx = fx.block_idx.x
        nm = (m + BM - 1) // BM
        nn = N // BN
        nt = nm * nn
        x, q = (bx % 8, bx // 8)
        per, extra = (nt // 8, nt % 8)
        idx = x * per + (x < extra).select(x, extra) + q
        mi, ni = (idx % nm, idx // nm)
        ar, br = (mi * BM, ni * BN)
        sm = fx.SharedAllocator().allocate(Shared).peek()
        a_r = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=nm * BM * K)
        w_r = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=N * K)
        as_r = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=(m + 31) // 32 * 32 * (K // 32))
        ws_r = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=N * (K // 32))
        out_r = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=m * N * 2)
        ag = G2SLoaderAsm(a_r, _global_swizzle(lane, wave, K, 2, True), 2, wave)
        ag.set_wave_base(sm.a.ptr)
        for z in range_constexpr(SW):
            ag.load(_Buf(sm.a.ptr, z * AB), ar * K + z * 2048)
        # Two immutable 4-KB packets: 64 rows of A scales and the 64 columns
        # surrounding this tile's 48 W columns. Each staging wave moves 1 KB.
        lds16 = fx.logical_divide(
            fx.make_view(sm.scales.ptr, fx.make_layout(2048, 1)),
            fx.make_layout(4, 1),
        )
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        for kind in range_constexpr(2):
            src = as_r if kind == 0 else ws_r
            group = ar // 32 if kind == 0 else br // 32
            value = buffer_ops.buffer_load(
                src, group * SCB * 64 + wave * 256 + lane * 4,
                vec_width=4, dtype=fx.Int32,
            )
            rmem = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            rmem.store(value)
            fx.copy(atom, rmem, fx.slice(lds16, (None, kind * 256 + wave * 64 + lane)))
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        rocdl.s_barrier()
        # Every staging write has completed. The arrays never refill, so the
        # fourth wave can finish while the other three read the shared panels.
        if wave < 3:
            al = S2RLoader(fx.Int32(0), NA)
            l16, q16 = (lane % 16, lane // 16)
            lo = q16 * 256 + l16 * 16
            n16 = br // 16 + wave
            woff = n16 * 16 * K + lo
            scales_a = [
                [fx.Int32(sm.scales.ptr[(r * SCB + z) * 64 + lane])
                 for r in range_constexpr(2)]
                for z in range_constexpr(SW // 2)
            ]
            # A 48-column tile can start on the odd half of a 32-column group.
            scales_w = [
                fx.Int32(sm.scales.ptr[
                    1024 + ((n16 // 2 - br // 32) * SCB + z) * 64 + lane
                ]).shrui(n16 % 2 * 8)
                for z in range_constexpr(SW // 2)
            ]

            def load_w(kt):
                return _pack8(
                    Vec(buffer_ops.buffer_load(
                        w_r, woff // 4, vec_width=4, dtype=fx.Int32,
                        soffset_bytes=kt * 2048, cache_modifier=WCM,
                    )),
                    Vec(buffer_ops.buffer_load(
                        w_r, woff // 4 + 256, vec_width=4, dtype=fx.Int32,
                        soffset_bytes=kt * 2048, cache_modifier=WCM,
                    )),
                )
            wq = [load_w(z) for z in range_constexpr(LA)]
            av = al.load(_Buf(sm.a.ptr, 0), True)
            zero = Vec.filled(4, 0.0, fx.Float32)
            acc = [fx.as_ir_value(zero)] * NA
            mf = MfmaRowStationary(NA, 1)
            for kt in range_constexpr(SW):
                if const_expr(kt + LA < SW):
                    wq.append(load_w(kt + LA))
                an = [None] * NA
                rd = _s2r_thunks(
                    al, _Buf(fx.add_offset(sm.a.ptr, (kt + 1) % SW * AB), 0),
                    an, NA, True,
                )
                for z, (i, j) in enumerate(mf._order()):
                    if const_expr(z < len(rd)):
                        rd[z]()
                    acc[i] = mf._mfma(
                        av[i], wq[kt], acc[i], scales_a[kt // 2][i // 2],
                        scales_w[kt // 2], kt % 2, i % 2, 0,
                    )
                # Four MFMAs have only four interleave slots for eight A reads.
                # Finish the other reads before carrying the next panel.
                for z in range_constexpr(NA, len(rd)):
                    rd[z]()
                av = an
            flat = _pin_accumulators(acc)
            for i in range_constexpr(NA):
                row = ar + i * 16 + l16
                col = br + wave * 16 + q16 * 4
                buffer_ops.buffer_store(
                    Vec(flat[i]).to(fx.BFloat16).bitcast(fx.Int32), out_r,
                    (row * N + col) // 2, mask=row < m,
                )

    @flyc.jit
    def launch_gemm_n6144_k2048_astage(
        A: fx.Tensor, W: fx.Tensor, C: fx.Tensor, AS: fx.Tensor, WS: fx.Tensor,
        m: fx.Int32, a_scale_bytes: fx.Int32, n_mb: fx.Int32, grid: fx.Int32,
        stream: fx.Stream,
    ):
        # Keep the direct kernel's launcher ABI, including its shape scalars.
        kernel(A, W, C, AS, WS, m).launch(grid=(fx.Int64(grid), 1, 1), block=(256, 1, 1), stream=stream)
    launch_gemm_n6144_k2048_astage.kernel_name = name
    launch_gemm_n6144_k2048_astage.k_split = 1
    launch_gemm_n6144_k2048_astage.n_blocks = N // BN
    launch_gemm_n6144_k2048_astage.block_m = BM
    launch_gemm_n6144_k2048_astage.fold = 1
    return launch_gemm_n6144_k2048_astage
