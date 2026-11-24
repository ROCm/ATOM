# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 pack of the attention output for the sequence-parallel all-to-all: bf16 ``[M, 2048]`` row-major ->
uint8 ``[M, 2112]``, each row its 2048 e4m3 codes, then its 64 e8m0 scale bytes (row-major, one per 32 values). The
layout ``pull_collectives_ranks4``'s unpacking all-to-alls read (``unpack_reference``).

A wave per row pair (its whole 2048 values), 4 waves per workgroup: lanes 0-31 row 2j, 32-63 row 2j + 1, a lane one
16-byte piece (8 values) of each 512-byte block, all eight loads issued up front; per block the 32-value maximum on
the packed bf16 patterns, completed over the quad with two DPP shuffles, the quantizer's scale byte from the bf16
pattern of the maximum and 8 codes per lane from ``v_cvt_scalef32_pk_fp8_bf16`` (the arithmetic of
``gemm/mxfp8_common/mxfp8_quant.py``, copied below),
stored as 8 bytes per lane (non-temporal). One lane per group drops its scale byte into the wave's 128 LDS bytes; the
wave then stores both rows' 64 scale bytes in one dword store (no barrier: only the wave reads its LDS bytes). Rows
beyond M read zero and store nowhere (buffer resource bounds).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

D = 2048  # values per row
ROW_BYTES = D + D // 32  # 2048 codes + 64 scale bytes
KB = D // 256  # 256-K blocks per row
WAVES = 4  # one row pair each (its whole 2048 values): 8 rows per workgroup
# Non-temporal code stores (the peers pull the slot from memory); plain loads: the attention output is usually still
# in the caches (one-off timing, MI355X, graph replay, FlyDSL vs vLLM's Triton pack, us: 4096 / 8885 / 15745 /
# 32768 rows 3.2 / 8.6 / 15.4 / 30.0 vs 3.8 / 9.1 / 15.0 / 28.8; non-temporal loads 3.9 / 9.7 / 16.8 / 32.6)
NT_STORE = 2
QUAD_SWAP_ADJ = 0xB1  # dpp quad_perm [1, 0, 3, 2]
QUAD_SWAP_PAIR = 0x4E  # dpp quad_perm [2, 3, 0, 1]


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _i32_max(a, b):
    return (a > b).select(a, b)


def _pk_max_u16(a, b):
    """per-16-bit-half unsigned max (v_pk_max_u16): on sign-cleared bf16 patterns that is the value order."""
    va = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(a))).result
    vb = llvm.BitcastOp(T.i16x2, _raw(fx.Int32(b))).result
    return fx.Int32(llvm.BitcastOp(T.i32, llvm.intr_umax(va, vb)).result)


def _amax_bf16(dwords):
    """max |x| over the 2 * len(dwords) bf16 values, as a 16-bit bf16 pattern in an i32."""
    m = None
    for d in dwords:
        a = fx.Int32(d) & fx.Int32(0x7FFF7FFF)
        m = a if m is None else _pk_max_u16(m, a)
    lo, hi = m & fx.Int32(0xFFFF), m >> 16
    return (lo > hi).select(lo, hi)


def _dpp(v, ctrl):
    return fx.Int32(fx.rocdl.update_dpp(T.i32, _raw(fx.Int32(0)), _raw(fx.Int32(v)), ctrl, 0xF, 0xF, True))


def _quad_max(v):
    """max over the 4 lanes of a quad, in every lane of it"""
    m = _i32_max(v, _dpp(v, QUAD_SWAP_ADJ))
    return _i32_max(m, _dpp(m, QUAD_SWAP_PAIR))


def _e8m0_bf16(amax_bits):
    """(scale byte, conversion scale) of a block from the bf16 pattern of its maximum: clamp(ceil(log2(max(amax,
    tiny) / 448)) + 127, 0, 254), i.e. the exponent field - 8 + [mantissa > 0x60] in one carry; zero, denormal and
    tiny maxima land on 0. Byte 0 would make the conversion scale a denormal, so such a block (maximum below 2^-119)
    is converted with 2^-126 (reference.py)."""
    t = (fx.Int32(amax_bits) + fx.Int32(0x1F)) >> 7
    sb = _i32_max(t, fx.Int32(8)) - fx.Int32(8)
    scale = ((_i32_max(t, fx.Int32(9)) - fx.Int32(8)) << 23).bitcast(fx.Float32)
    return sb, scale


def _quant_bf16(dwords, scale):
    """dwords of bf16 pairs -> half as many dwords of fp8 codes: x / scale, round to nearest even."""
    out = []
    for i in range(0, len(dwords), 2):
        pair0 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i]))).result
        pair1 = llvm.BitcastOp(T.bf16x2, _raw(fx.Int32(dwords[i + 1]))).result
        # the first conversion's other half is overwritten by the second
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(T.i16x2, llvm.mlir_undef(T.i16x2), pair0, _raw(scale), False)
        w = fx.rocdl.cvt_scalef32_pk_fp8_bf16(T.i16x2, w, pair1, _raw(scale), True)
        out.append(fx.Int32(llvm.BitcastOp(T.i32, w).result))
    return Vec.from_elements(out, fx.Int32)


@launcher
def compile_mxfp8_pack_q2048(waves: int = WAVES):
    name = f"seq_parallel_mxfp8_pack_d2048_w{waves}"

    @fx.struct
    class Shared:
        sc: fx.Array[fx.Uint8, 128 * waves, 16]

    @flyc.kernel(name=name, known_block_size=[64 * waves, 1, 1])
    def kernel(X: fx.Tensor, P: fx.Tensor, i32_m: fx.Int32):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        half, piece = lane // 32, lane % 32  # row of the pair, 16-byte piece of a 512-byte block
        group = piece // 4  # 32-value group of the 256-value block (0..7)
        x_rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=i32_m * (2 * D))
        p_rsrc = buffer_ops.create_buffer_resource(P, max_size=False, num_records_bytes=i32_m * ROW_BYTES)
        row = (bx * waves + wave) * 2 + half
        x_dw = (row * (2 * D) + piece * 16) // 4
        raw = [
            buffer_ops.buffer_load(x_rsrc, x_dw, vec_width=4, dtype=fx.Int32, soffset_bytes=kb * 512)  # fmt: skip
            for kb in range_constexpr(KB)
        ]
        out = row * ROW_BYTES
        # this wave's two rows of scale bytes (64 each) in LDS: written and read by this wave only
        lds_wave = fx.Int32(fx.ptrtoint(smem.sc.ptr)) + wave * 128

        def lds_ptr(byte_off):
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(lds_wave + byte_off)))

        for kb in range_constexpr(KB):
            vals = [Vec(raw[kb])[i] for i in range(4)]
            sb, scale = _e8m0_bf16(_quad_max(_amax_bf16(vals)))
            buffer_ops.buffer_store(_quant_bf16(vals, scale), p_rsrc, out + piece * 8, offset_is_bytes=True,
                                    soffset_bytes=kb * 256, cache_modifier=NT_STORE)  # fmt: skip
            if piece % 4 == 0:
                llvm.StoreOp(
                    llvm.TruncOp(T.i8, _raw(sb), llvm.IntegerOverflowFlags.none).result,
                    lds_ptr(half * 64 + kb * 8 + group),
                )
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        if lane < 32:  # lane l: scale dword l % 16 of the pair's row l // 16, both rows in one store
            s_val = fx.Int32(llvm.load(T.i32, lds_ptr(lane * 4), alignment=4))
            s_row = (bx * waves + wave) * 2 + lane // 16
            buffer_ops.buffer_store(s_val, p_rsrc, s_row * ROW_BYTES + D + (lane % 16) * 4, offset_is_bytes=True)

    @flyc.jit
    def launch_mxfp8_pack_q2048(X: fx.Tensor, P: fx.Tensor, i32_m: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel(X, P, i32_m).launch(grid=(fx.Int64(grid), 1, 1), block=(64 * waves, 1, 1), stream=stream)

    launch_mxfp8_pack_q2048.kernel_name = name
    launch_mxfp8_pack_q2048.rows_per_wg = 2 * waves
    return launch_mxfp8_pack_q2048


@functools.cache
def _launcher(waves: int = WAVES):
    return compile_mxfp8_pack_q2048(waves)


def mxfp8_pack_q2048(x: torch.Tensor, out: torch.Tensor | None = None, waves: int = WAVES) -> torch.Tensor:
    """bf16 ``[M, 2048]`` contiguous -> uint8 ``[M, 2112]`` packed rows (module doc; ``reference.pack_reference``).
    ``out``: a contiguous ``[M, 2112]`` uint8 tensor to write (a pool slot's view), else a new one. One launch."""
    from amd_kernels.common.tensor_shim import _run_compiled

    m, d = x.shape
    assert d == D and x.dtype == torch.bfloat16 and x.is_contiguous(), (x.shape, x.dtype)
    if out is None:
        out = torch.empty((m, ROW_BYTES), dtype=torch.uint8, device=x.device)
    assert out.shape == (m, ROW_BYTES) and out.dtype == torch.uint8 and out.is_contiguous(), (out.shape, out.dtype)
    if m:
        exe = _launcher(waves)
        grid = (m + exe.rows_per_wg - 1) // exe.rows_per_wg
        _run_compiled(exe, x.view(torch.uint8).view(-1), out.view(-1), m, grid, torch.cuda.current_stream())
    return out
