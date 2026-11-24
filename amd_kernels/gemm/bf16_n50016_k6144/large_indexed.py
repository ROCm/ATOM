# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexed large-M BF16 lm_head GEMM with a wider vocabulary tile and bounded operand liveness, publishing into the
row exchange (193-2048 rows; publish.py).

200 vocabulary tiles cover 15 or 16 fragments each. Eight waves take two M
halves, sharing activation and weight LDS rings. Splitting each LDS operand
prefetch into two M slices removes the wider tile's spills. The epilogue
reuses the dead LDS ring to write 16-byte contiguous output pieces.
K remains ascending in fp32, with one BF16 rounding and no output atomics.
The load-time weight layout is the same as the small-M kernel's.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .publish import SYS_CM, flag_add
from .reference import K, N

G16 = N // 16
NWV = 4
NB = K // 64
# Sweep: 200 vs 256/264/392 vocabulary tiles, ring 2, and split
# operand prefetch. Keep the original load-time weight layout at every M.
VOCAB_TILES = 200
RING = 2
M_HALVES = 2
W_CM = 2  # nt wins the full-shaped sweep; cached reads are slower


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def compile_large_indexed_bf16_gemm_n50016_k6144(mt: int, shuf: int):
    TILES, mh, ring = VOCAB_TILES, M_HALVES, RING
    NRB = G16 // TILES
    NBIG = G16 - NRB * TILES
    NR = NRB + 1
    NRW = (NR + NWV - 1) // NWV
    # Wave w owns vocabulary fragments w % 4 and M half w // 4.
    MT, SHUF, MH = mt, shuf, mh
    NWT = NWV * MH  # waves per workgroup
    MTT = MT * MH  # activation fragments per workgroup
    R = ring
    TM = 16 * MTT
    XD = MTT * 2  # activation DMAs per block (2 per 16-row fragment)
    XPW = (XD + NWT - 1) // NWT
    PER = NRW * 2 // MH + XPW  # DMAs per wave per block (MH 2: each half issues one of a group's two DMAs)
    WF = NR + 1  # weight positions per slot: 16 groups + one junk position
    SLOT = (WF + MTT) * 2048
    LDS = max(R * SLOT, TM * 256 * 2)
    assert 5 <= MT <= 8 and LDS <= 160 * 1024 and (R - 1) * PER <= 63, (LDS, R, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"bf16_gemm_n50016_k6144_large_indexed_tiles{TILES}_tm{TM}_r{R}" + ("_shuf" if SHUF else "_row") + ("_w8" if MH == 2 else "") + "_interleaved"
    name = name + "_publish"

    def body(arg_x, arg_w, arg_y, arg_indices, i32_xrows, i32_m, i32_mtiles, y_cm):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV  # the weight groups
        mhalf = wave_all // NWV  # the M half (0 when MH 1)
        # workgroup b runs on XCD b % 8; the M tiles of one vocabulary tile follow each other on that XCD
        xcd = bid % 8
        j = bid // 8
        t = (j // i32_mtiles) * 8 + xcd
        m0 = (j % i32_mtiles) * TM
        big = t < NBIG
        n0 = big.select(t * (16 * NR), t * (16 * NRB) + NBIG * 16)
        nfrag = big.select(fx.Int32(NR), fx.Int32(NRB))
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_xrows * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        yr = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (N * 2))
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            ok = rn < nfrag
            for jq in range_constexpr(2 // MH):
                jj = mhalf + jq if MH == 2 else jq
                pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
                if const_expr(SHUF):
                    off = (n0 // 16 + rn) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
                else:
                    off = (n0 + rn * 16 + jj * 8 + r8) * (K * 2) + pc * 16
                w_src.append(ok.select(off, fx.Int32(0x70000000)))  # past the tile: zeros, no traffic
                w_dst.append((ok.select(rn, fx.Int32(NR)) * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            row = fx.Int32(arg_indices[row])
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            for q in range_constexpr(NRW * 2 // MH):
                issue(wr, w_dst[q], w_src[q], b, b * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, b * 128, aux_x)

        wpos = [(wave + i * NWV < NR).select(wave + i * NWV, fx.Int32(NR)) for i in range_constexpr(NRW)]
        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)] for _ in range_constexpr(NRW)]

        P = (MT + 1) // 2
        def read_w(b, kk):
            return [rd(b, wpos[i], kk) for i in range_constexpr(NRW)]
        def read_x(b, kk, lo, hi):
            return [rd(b, WF + mhalf * MT + m, kk) for m in range_constexpr(lo, hi)]
        def mfmas(wv, xv, lo, hi, prefetch=None, operand=0):
            for i in range_constexpr(NRW):
                # Spread future DMA requests between four-MFMA groups instead
                # of issuing an eight-request burst before the computation.
                if prefetch is not None:
                    if operand == 0:
                        issue(wr, w_dst[i], w_src[i], prefetch, prefetch * (2048 if SHUF else 128), aux_w)
                    elif i < XPW:
                        issue(xr, x_dst[i], x_src[i], prefetch, prefetch * 128, aux_x)
                _sb()
                for m in range_constexpr(lo, hi):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m - lo], acc[i][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        # half-block software pipeline: the LDS reads of (b, kk 1) overlap the MFMAs of (b, kk 0), those of
        # (b + 1, kk 0) the MFMAs of (b, kk 1); slot b is refilled (block b + R) once every wave has read it
        # (lgkmcnt(0) before the barrier), so R blocks are in flight
        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        barrier()
        _sb()
        w_a = read_w(0, 0)
        x_a0 = read_x(0, 0, 0, P)
        x_a1 = read_x(0, 0, P, MT)
        for b in range_constexpr(NB):
            _sb()
            w_b = read_w(b, 1)
            x_b0 = read_x(b, 1, 0, P)
            _sb()
            mfmas(w_a, x_a0, 0, P)
            _sb()
            x_b1 = read_x(b, 1, P, MT)
            _sb()
            mfmas(w_a, x_a1, P, MT)
            _sb()
            if const_expr(b + 1 < NB):
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NB - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                w_a = read_w(b + 1, 0)
                x_a0 = read_x(b + 1, 0, 0, P)
                _sb()
                _sb()
            mfmas(w_b, x_b0, 0, P, (b + R) if b + R < NB else None, 0)
            _sb()
            if const_expr(b + 1 < NB):
                x_a1 = read_x(b + 1, 0, P, MT)
                _sb()
            mfmas(w_b, x_b1, P, MT, (b + R) if b + R < NB else None, 1)

        fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=0)
        barrier()
        out_view = fx.make_view(fx.recast_iter(fx.Int32, smem.buf.ptr), fx.make_layout(TM * 128, 1))
        out8 = fx.logical_divide(out_view, fx.make_layout(2, 1))
        out16 = fx.logical_divide(out_view, fx.make_layout(4, 1))
        atom8 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            for m in range_constexpr(MT):
                r = (mhalf * MT + m) * 16 + l16
                c = rn * 16 + g * 4
                pk = acc[i][m].to(fx.BFloat16).bitcast(fx.Int32)
                phys = (r * 128 + c // 2) ^ ((r % 8) * 4)
                rmem = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                rmem.store(pk)
                fx.copy(atom8, rmem, fx.slice(out8, (None, phys // 2)))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        for it in range_constexpr((TM * 256 // 8 + 64 * NWT - 1) // (64 * NWT)):
            v = tx + it * (64 * NWT)
            r = v // 32
            c = (v % 32) * 8
            phys = (r * 128 + c // 2) ^ ((r % 8) * 4)
            rmem = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom16, fx.slice(out16, (None, phys // 4)), rmem)
            val = Vec(rmem.load())
            buffer_ops.buffer_store(val, yr, (m0 + r) * (N // 2) + (n0 + c) // 2,
                                    mask=(r < TM) & (m0 + r < i32_m) & (c < nfrag * 16), cache_modifier=y_cm)

    # Publish: the output goes to the exchange's slot (counter[0] & 1) with system-scope stores, and the last
    # workgroup publishes it (all_to_all_bf16_n50016_ranks4's stage_publish protocol: the same ticket and flag words).
    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel_publish(arg_x: fx.Int64, arg_w: fx.Int64, arg_pool: fx.Int64, arg_indices: fx.Tensor,
                       i32_xrows: fx.Int32, i32_m: fx.Int32, i32_mtiles: fx.Int32, counter: fx.Tensor,
                       i64_slot_bytes: fx.Int64, arg_flag: fx.Int64):
        slot = fx.Int32(counter[0]) & 1
        body(arg_x, arg_w, arg_pool + fx.Int64(slot) * i64_slot_bytes, arg_indices, i32_xrows, i32_m, i32_mtiles,
             SYS_CM)
        # every workgroup's stores land before its ticket; the last of them publishes (dynamic ifs stay in the
        # kernel body: FlyDSL rewrites only this function's control flow)
        nwg = TILES * i32_mtiles
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if fx.thread_idx.x == 0:
            ticket = flag_add(arg_flag + 4, 1)
            if ticket == nwg - 1:
                flag_add(arg_flag + 4, -nwg)
                seq = fx.Int32(counter[0]) + 1
                counter[0] = seq
                fr = buffer_ops.create_buffer_resource_from_addr(arg_flag, num_records_bytes=64)
                buffer_ops.buffer_store(seq, fr, 0, cache_modifier=SYS_CM)

    @flyc.jit
    def launch_large_indexed_bf16_gemm_n50016_k6144_publish(arg_x: fx.Int64, arg_w: fx.Int64, arg_pool: fx.Int64, arg_indices: fx.Tensor,
                       i32_xrows: fx.Int32, i32_m: fx.Int32, i32_mtiles: fx.Int32, counter: fx.Tensor,
                       i64_slot_bytes: fx.Int64, arg_flag: fx.Int64, stream: fx.Stream):
        kernel_publish(arg_x, arg_w, arg_pool, arg_indices, i32_xrows, i32_m, i32_mtiles, counter, i64_slot_bytes,
                       arg_flag).launch(grid=(TILES * i32_mtiles, 1, 1), block=(64 * NWT, 1, 1), stream=stream)

    return launch_large_indexed_bf16_gemm_n50016_k6144_publish
