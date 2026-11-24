# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""N6144 K2048 MXFP8 GEMM, 128x192 tiles with a terminal K pipeline.

Four waves keep the existing three-A / four-W operand ring and increasing-K
accumulation order. The final four K128 steps omit wraparound A/W prefetches,
unused scale gathers, and the final fragment reads. The first terminal step
retains the steady wait; the second permits NA+NB pending requests, and the
last two drain VMEM. Every step retains its all-wave barrier. Output remains
the same coalesced BF16 transpose with streaming stores. The caller provides
row-major activation codes and preshuffled scales at 769-1024 rows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_4wave import (
    G2SLoaderAsm,
    MfmaRowStationary,
    S2RLoader,
    _asm_void,
    _Buf,
    _global_swizzle,
    _s2r_thunks,
    _uniform_i32,
)

BLOCK_M, BLOCK_N = 128, 192
K_UNROLL = 12


def compile_mxfp8_gemm_n6144_k2048_128x192():
    K = 2048
    BM, BN = BLOCK_M, BLOCK_N
    KSTEPS = K // 128
    NA, NB = BM // 32, BN // 32  # MFMA tiles per wave; also DMA rounds per panel
    AB, BB = BM * 128, BN * 128
    # gfx950 buffer_store bits: 0x2 = nt. The output is written once; NT stores keep the
    # operands in L2 / MALL (QKV -2.9..-3.1%, o_proj K2048 -6.8..-7.1%, K8192 -1.7..-2.1%).
    CPOL = 2
    name = "flydsl_mxfp8_128x192_n6144_k2048_tail_pipe3a4w_xor_sb_nt"

    @fx.struct
    class Shared:
        data: fx.Array[fx.Int8, 3 * AB + 4 * BB, 16]
        scales: fx.Array[fx.Int8, 8192, 16]

    @flyc.kernel(name=name, known_block_size=(256, 1, 1))
    def kernel(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, AS: fx.Tensor, BS: fx.Tensor,
        m: fx.Int32, n: fx.Int32,
    ):
        tid = fx.Int32(fx.gpu.thread_id("x"))
        wid = fx.Int32(_uniform_i32(tid // 64))
        lane = tid % 64
        wi, wj = wid // 2, wid % 2
        wg = fx.Int32(fx.gpu.block_id("x"))
        nm, nn = (m + BM - 1) // BM, n // BN
        # Workgroup w runs on XCD w%8. Deal contiguous column strips to it,
        # so its CUs share W through L2 rather than reading every W column.
        total = nm * nn
        x, q = wg % 8, wg // 8
        base, extra = total // 8, total % 8
        idx = x * base + (x < extra).select(x, extra) + q
        pi, pj = idx % nm, idx // nm
        ar, br = pi * BM, pj * BN
        lds = fx.SharedAllocator().allocate(Shared).peek()
        a0, a1 = _Buf(lds.data.ptr, 0), _Buf(lds.data.ptr, AB)
        b0, b1 = _Buf(lds.data.ptr, 3 * AB), _Buf(lds.data.ptr, 3 * AB + BB)
        ag = G2SLoaderAsm(
            buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=m * K),
            _global_swizzle(lane, wid, K, NA, False), NA, wid,
        )
        bg = G2SLoaderAsm(
            buffer_ops.create_buffer_resource(B, max_size=False, num_records_bytes=n * K),
            _global_swizzle(lane, wid, K, NB, True), NB, wid,
        )
        ag.set_wave_base(lds.data.ptr)
        bg.set_wave_base(lds.data.ptr)
        al, bl = S2RLoader(wi, NA), S2RLoader(wj, NB)
        sar = buffer_ops.create_buffer_resource(
            AS, max_size=False, num_records_bytes=((m + 31) // 32 * 32) * (K // 32),
        )
        sbr = buffer_ops.create_buffer_resource(BS, max_size=False, num_records_bytes=n * (K // 32))
        sp = fx.recast_iter(fx.Int32, lds.scales.ptr)
        is_a = wid == 0
        sr = fx.arith.select(is_a, sar, sbr)
        grp = is_a.select(ar // 32, br // 32 + (wid - 1) * 4) + lane // 16
        svoff = grp * (K // 256) * 256 + (lane % 16) * 16
        swb = _uniform_i32(fx.Int32(fx.ptrtoint(lds.scales.ptr)) + wid * 1024)

        def gather(blk, slot):
            # Async G2S asm avoids LLVM alias waits before unrelated operand LDS reads.
            _asm_void(
                [fx.as_ir_value(fx.Int32(swb) + slot * 4096), fx.as_ir_value(svoff),
                 fx.as_ir_value(sr), _uniform_i32(blk * 256)],
                "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds",
                "s,v,s,s", "~{m0}",
            )

        def scales(kt):
            sa, sb = [], []
            slot = (kt // 2) % 2
            sl = (lane // 16) * 16 + lane % 16
            for i in range_constexpr(NA):
                r = wi * (NA * 16) + i * 16
                off = slot * 1024 + (r // 32) * 64 + sl
                sa.append(fx.Int32(sp[off]).shrui((r % 32) // 16 * 8))
            for j in range_constexpr(NB):
                r = wj * (NB * 16) + j * 16
                off = slot * 1024 + 256 + (r // 32) * 64 + sl
                sb.append(fx.Int32(sp[off]).shrui((r % 32) // 16 * 8))
            return sa, sb

        mf = MfmaRowStationary(NA, NB)
        ag.load(a0, ar * K)
        bg.load(b0, br * K)
        ag.load(a1, ar * K + 128)
        bg.load(b1, br * K + 2048)
        bg.load(_Buf(lds.data.ptr, 3 * AB + 2 * BB), br * K + 4096)
        gather(fx.Int32(0), fx.Int32(0))
        gather(fx.Int32(1), fx.Int32(1))
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        rocdl.s_barrier()
        zero = Vec.filled(4, 0.0, fx.Float32)
        init = (
            [fx.as_ir_value(zero)] * (NA * NB)
            + [fx.as_ir_value(v) for v in al.load(a0)]
            + [fx.as_ir_value(v) for v in bl.load(b0, True)]
        )
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        def one_step(it, u, acc, av, bv, final=False):
            kt = fx.Int32(it) + u
            fut, nr = fx.Int32((u + 3) % 4), fx.Int32((u + 1) % 4)
            an = _Buf(lds.data.ptr, fx.Int32((u + 2) % 3) * AB)
            bn = _Buf(lds.data.ptr, 3 * AB + fut * BB)
            na_buf = _Buf(fx.add_offset(lds.data.ptr, fx.Int32((u + 1) % 3) * AB), 0)
            nb_buf = _Buf(fx.add_offset(lds.data.ptr, 3 * AB + nr * BB), 0)
            sa, sb = scales(kt)
            nxt = (kt + 3) % (K // 128)
            if const_expr(not final or u < 2):
                ag.load(an, ar * K + ((kt + 2) % (K // 128)) * 128)
            if const_expr(not final or u == 0):
                bg.load(bn, br * K + nxt * 2048)
            # One future A panel and two W panels (NA+2*NB DMAs per wave) may stay
            # in flight, and after an odd step also its post-barrier scale gather.
            # The following panel is complete on every wave before LDS reads; at an
            # odd step the gather issued three steps ago (the next packet) is too.
            # Terminal u=1 issues only A15: W15 + A15 may remain. From u=2,
            # no future DMA remains, so drain before the next fragment reads.
            wait_loads = ((NA + 2 * NB + 1) if u == 0 else (NA + NB if u == 1 else 0)) if final else (NA + 2 * NB + (1 if u % 2 == 0 else 0))
            rocdl.s_waitcnt(vmcnt=wait_loads, lgkmcnt=0)
            rocdl.s_barrier()
            if const_expr(u % 2 == 1 and not final):
                # Waves 0-2 gather the packets every wave reads: overwrite the slot of
                # this step's packet only after the barrier, when every wave's reads of
                # it (lgkmcnt(0) above) are complete.
                gather((kt // 2 + 2) % (K // 256), (kt // 2 + 2) % 2)
            anext, bnext = [None] * NA, [None] * NB
            rd = (_s2r_thunks(al, na_buf, anext, NA, False) + _s2r_thunks(bl, nb_buf, bnext, NB, True)) if not final or u < 3 else []
            for z, (i, j) in enumerate(mf._order()):
                if const_expr(z < len(rd)):
                    rd[z]()
                # Inline-asm MFMA pins the accumulator in AGPRs.
                acc[i * NB + j] = mf._mfma(av[i], bv[j], acc[i * NB + j], sa[i], sb[j], u % 2, 0, 0)
            av, bv = (anext, bnext) if not final or u < 3 else (av, bv)
            return acc, av, bv

        # KSTEPS - 4 K128 steps preserve the 3-A / 4-W slot phase; peel the final four.
        for it, state in range(fx.Index(0), fx.Index(KSTEPS - 4), fx.Index(K_UNROLL), init=init):
            acc = list(state[:NA * NB])
            av = list(state[NA * NB:NA * NB + NA])
            bv = list(state[NA * NB + NA:])
            for u in range_constexpr(K_UNROLL):
                acc, av, bv = one_step(it, u, acc, av, bv)
            state = yield acc + [fx.as_ir_value(v) for v in av] + [fx.as_ir_value(v) for v in bv]
        acc = list(state[:NA * NB])
        av = list(state[NA * NB:NA * NB + NA])
        bv = list(state[NA * NB + NA:])
        for u in range_constexpr(4):
            acc, av, bv = one_step(fx.Int32(KSTEPS - 4), u, acc, av, bv, final=True)
        results = acc + [fx.as_ir_value(v) for v in av] + [fx.as_ir_value(v) for v in bv]


        # All DMAs and fragment reads finish before the operand ring becomes output LDS.
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        rocdl.s_barrier()
        p16 = fx.recast_iter(fx.Int16, lds.data.ptr)
        for i in range_constexpr(NA):
            for j in range_constexpr(NB):
                a = Vec(results[i * NB + j]).to(fx.BFloat16).bitcast(fx.Int16)
                for z in range_constexpr(4):
                    # MFMA is W*A^T: lane%16 is row, 4*(lane//16)+z is column.
                    rr = wi * (NA * 16) + i * 16 + lane % 16
                    cc = wj * (NB * 16) + j * 16 + 4 * (lane // 16) + z
                    p16[rr * BN + (cc ^ ((rr % 8) * 8))] = a[z]
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        out = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=m * n * 2)
        p32 = fx.recast_iter(fx.Int32, lds.data.ptr)
        for z in range_constexpr(BM * BN // 2048):
            lin = (tid + z * 256) * 8
            rr, cc = lin // BN, lin % BN
            off = ((ar + rr) * n + br + cc) // 2
            read = (rr * BN + (cc ^ ((rr % 8) * 8))) // 2
            v = Vec.from_elements([p32[read + i] for i in range(4)], fx.Int32)
            buffer_ops.buffer_store(v, out, off, cache_modifier=CPOL)

    @flyc.jit
    def launch_gemm_n6144_k2048_128x192(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, AS: fx.Tensor, BS: fx.Tensor,
        m: fx.Int32, n: fx.Int32, stream: fx.Stream,
    ):
        kernel(A, B, C, AS, BS, m, n, value_attrs={"rocdl.waves_per_eu": 1}).launch(
            grid=(((m + BM - 1) // BM) * (n // BN), 1, 1), block=(256, 1, 1), stream=stream,
        )

    launch_gemm_n6144_k2048_128x192.kernel_name = name
    return launch_gemm_n6144_k2048_128x192
