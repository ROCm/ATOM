# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexed activation variant of the BF16 lm_head GEMM, publishing into the row exchange (1-192 rows).

The device's int64 row map selects each activation row before its LDS DMA.
The epilogue writes in map order, so the target hidden gather and the draft
owner-sorting copy disappear. The map is re-read on every graph replay.
Weight layout and fixed K-ascending sums match the contiguous variant.
The output goes into all_to_all_bf16_n50016_ranks4's send slot (publish.py).
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

TILES = 256  # vocabulary tiles per M tile: one round of the 256 CUs
G16 = N // 16  # 3126 16-row groups
NRB = G16 // TILES  # 12 groups per tile
NBIG = G16 - NRB * TILES  # the first 54 tiles take one more
NR = NRB + 1  # 13
NWV = 4  # waves
NRW = (NR + NWV - 1) // NWV  # 4 group slots per wave
NB = K // 64  # 96 K blocks
W_CM = 2  # nt
MT_MAX = 8


def ring_depth(mt: int) -> int:
    """Ring slots that fit the 160 KB LDS (one slot = (14 + mt) x 2 KB)."""
    return 4 if mt <= 6 else 3


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def compile_indexed_bf16_gemm_n50016_k6144(mt: int, shuf: int, mh: int = 1):
    # MH: M halves; MH 2 = 8 waves (2 per SIMD), wave w takes the groups of w % 4 and the M fragments of half w // 4
    MT, SHUF, MH = mt, shuf, mh
    NWT = NWV * MH  # waves per workgroup
    MTT = MT * MH  # activation fragments per workgroup
    R = ring_depth(MTT) if MH == 1 else 2
    TM = 16 * MTT
    XD = MTT * 2  # activation DMAs per block (2 per 16-row fragment)
    XPW = (XD + NWT - 1) // NWT
    PER = NRW * 2 // MH + XPW  # DMAs per wave per block (MH 2: each half issues one of a group's two DMAs)
    WF = NR + 1  # weight positions per slot: 13 groups + the junk position
    SLOT = (WF + MTT) * 2048
    LDS = R * SLOT
    assert 1 <= MT <= MT_MAX and LDS <= 160 * 1024 and (R - 1) * PER <= 63, (LDS, R, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"bf16_gemm_n50016_k6144_indexed_tm{TM}_r{R}" + ("_shuf" if SHUF else "_row") + ("_w8" if MH == 2 else "")
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

        def read_ops(b, kk):
            return ([rd(b, wpos[i], kk) for i in range_constexpr(NRW)],
                    [rd(b, WF + mhalf * MT + m, kk) for m in range_constexpr(MT)])

        def mfmas(ops):
            wv, xv = ops
            for i in range_constexpr(NRW):
                for m in range_constexpr(MT):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m], acc[i][m], 0, 0, 0]))

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
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NB):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NB:
                # blocks issued so far: 0 .. min(b + R - 1, NB - 1); block b + 1 must have landed
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NB - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + R < NB:
                    dma(b + R)
                _sb()
            mfmas(ops_b)

        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            n = n0 + rn * 16 + g * 4
            for m in range_constexpr(MT):
                r = m0 + (mhalf * MT + m) * 16 + l16
                pk = acc[i][m].to(fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(pk, yr, r * (N // 2) + n // 2, mask=(rn < nfrag) & (r < i32_m),
                                        cache_modifier=y_cm)

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
    def launch_indexed_bf16_gemm_n50016_k6144_publish(arg_x: fx.Int64, arg_w: fx.Int64, arg_pool: fx.Int64, arg_indices: fx.Tensor,
                       i32_xrows: fx.Int32, i32_m: fx.Int32, i32_mtiles: fx.Int32, counter: fx.Tensor,
                       i64_slot_bytes: fx.Int64, arg_flag: fx.Int64, stream: fx.Stream):
        kernel_publish(arg_x, arg_w, arg_pool, arg_indices, i32_xrows, i32_m, i32_mtiles, counter, i64_slot_bytes,
                       arg_flag).launch(grid=(TILES * i32_mtiles, 1, 1), block=(64 * NWT, 1, 1), stream=stream)

    return launch_indexed_bf16_gemm_n50016_k6144_publish
