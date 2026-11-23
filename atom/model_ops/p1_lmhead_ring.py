# p1/L5-dense-gemm (integration-branch only, flag ATOM_P1_LMHEAD_RING, default off): the lm_head vocabulary-shard GEMM
# bf16 [M, 6144] x [50016, 6144]^T at 65-128 rows. Kernel = amd-kernels gemm/bf16_n50016_k6144 (fd233ef) with its two
# M halves on 8 waves (2 per SIMD) and a 3-slot LDS ring (amd-kernels runs 4 waves, ring 4 up to 96 rows and 3 above,
# and switches to 8 waves only above 128 rows with ring 2). Same tiles, MFMA, operands and K order: bit-identical to
# amd-kernels' kernel at every row count (measured 65-128); vs aiter hgemm (today above 64 rows) fp32 summation order only.
# Measured (cold, graph replay, p1/L5-dense-gemm/NOTES.md): 70-128 rows -9..-11 us per call vs hgemm.
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

K = 6144
N = 50016
TILES = 256
G16 = N // 16
NRB = G16 // TILES
NBIG = G16 - NRB * TILES
NR = NRB + 1
NWV = 4
NRW = (NR + NWV - 1) // NWV
NB = K // 64
W_CM = 2
MIN_ROWS, MAX_ROWS = 65, 128


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _barrier():
    llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


def compile_ring(mt: int, mh: int, R: int, shuf: int = 1):
    """amd-kernels' kernel (shuf layout) with the ring depth R and M halves MH free; name l5_ring_*."""
    MT, SHUF, MH = mt, shuf, mh
    NWT = NWV * MH
    MTT = MT * MH
    TM = 16 * MTT
    XD = MTT * 2
    XPW = (XD + NWT - 1) // NWT
    PER = NRW * 2 // MH + XPW
    WF = NR + 1
    SLOT = (WF + MTT) * 2048
    LDS = R * SLOT
    assert 1 <= MT <= 8 and LDS <= 160 * 1024 and (R - 1) * PER <= 63, (LDS, R, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"p1_ring_bf16_gemm_n50016_k6144_tm{TM}_r{R}_mh{MH}"

    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV
        mhalf = wave_all // NWV
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

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        yr = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (N * 2))
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

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
                w_src.append(ok.select(off, fx.Int32(0x70000000)))
                w_dst.append((ok.select(rn, fx.Int32(NR)) * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)
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

        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        _barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NB):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NB:
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NB - 1) - (b + 1)) * PER)
                _barrier()
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
                buffer_ops.buffer_store(pk, yr, r * (N // 2) + n // 2, mask=(rn < nfrag) & (r < i32_m))

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, i32_m, i32_mtiles).launch(
            grid=(TILES * i32_mtiles, 1, 1), block=(64 * NWT, 1, 1), stream=stream)

    return launch




@functools.cache
def _launcher(mt: int):
    return compile_ring(mt, 2, 3)


def lmhead_ring(x: torch.Tensor, wshuf: torch.Tensor) -> torch.Tensor:
    """[M, 50016] bf16 logits of 65..128 rows on amd-kernels' shuffled shard (``shuffle_weight`` layout)."""
    m = x.shape[0]
    assert MIN_ROWS <= m <= MAX_ROWS and x.is_contiguous() and wshuf.shape == (N, K)
    out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    mt = (m + 31) // 32
    exe = _launcher(mt)
    args = (x.data_ptr(), wshuf.data_ptr(), out.data_ptr(), m, 1, torch.cuda.current_stream())
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)
    return out


def warmup(device, wshuf: torch.Tensor) -> None:
    """Compile and run both row tiers (65-96: 3 fragments per half, 97-128: 4) before graph capture."""
    for m in (96, 128):
        lmhead_ring(torch.zeros(m, K, dtype=torch.bfloat16, device=device), wshuf)
    torch.cuda.synchronize(device)
