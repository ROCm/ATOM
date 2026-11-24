# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""gfx950 MXFP8 activation quantizer: bf16 [M, K] row-major -> fp8 e4m3 codes [M, K]
row-major + e8m0 scales in AITER's shuffled layout (``shuffle_scale_a16w4``: [M/32, K/256,
klane 4, mlane 16, kpack 2, mpack 2]), what the staged / aiter GEMM kernels take.

A workgroup per 32-row x 256-K tile, by default 16 waves of one row pair each (``rpw``): a
load instruction covers 2 rows x 512 contiguous bytes (8 full lines; lanes 0-31 row 2j, 32-63
row 2j + 1, a lane one 16-byte piece = 8 values, a 32-K group = 4 adjacent lanes), then the
per-32 maximum on the packed bf16 patterns, completed over the quad with two DPP shuffles,
the Triton quantizer's scale byte from the bf16 pattern of the maximum (`mxfp8_gemm_decode.py`'s
helpers) and 8 codes per lane from ``v_cvt_scalef32_pk_fp8_bf16``, stored as 2 rows x 256
contiguous bytes per instruction. One lane per group drops the scale byte into the tile's
256-byte chunk of the shuffled layout in LDS; after a barrier wave 0 reads it back a dword
per lane and stores the chunk contiguously. Rows beyond M read zero
and store nowhere (buffer resource bounds); the scale tensor's padded rows get byte 0.
64-byte row pieces per instruction (half lines, four klanes per group joined by permlane
swaps, 32-byte output pieces) run no faster than the Triton kernel: the access pattern, not
the VALU work, bounds such a kernel.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .mxfp8_gemm_decode import (
    W_CACHE_MOD,
    _amax_bf16,
    _e8m0_bf16,
    _i32_max,
    _quant_bf16,
    _raw,
)

WAVES = 4  # waves per workgroup (consecutive K blocks of the same 32 rows)
TPW = 1  # tiles per wave (consecutive K blocks, all loads issued up front)
QUAD_SWAP_ADJ = 0xB1  # dpp quad_perm [1, 0, 3, 2]
QUAD_SWAP_PAIR = 0x4E  # dpp quad_perm [2, 3, 0, 1]


def _dpp(v, ctrl):
    return fx.Int32(fx.rocdl.update_dpp(T.i32, _raw(fx.Int32(0)), _raw(fx.Int32(v)), ctrl, 0xF, 0xF, True))


def _quad_max(v):
    """max over the 4 lanes of a quad, in every lane of it"""
    m = _i32_max(v, _dpp(v, QUAD_SWAP_ADJ))
    return _i32_max(m, _dpp(m, QUAD_SWAP_PAIR))


def compile_mxfp8_quant(
    *, K: int, waves: int = WAVES, tpw: int = TPW, nt_store: bool = False, shuffled: bool = False,
    rpw: int = 32,
):
    """``waves`` per workgroup, ``tpw`` consecutive tiles per wave (their loads all in flight
    together), ``nt_store``: non-temporal code stores; ``shuffled``: the codes in the weight's
    (16, 16) preshuffled layout for the decode kernel (byte (m, k) at (m // 16) * 16K + (k // 64)
    * 1024 + ((k % 64) // 16) * 256 + (m % 16) * 16 + k % 16; the code buffer then holds the
    row count padded to the decode row's tile, and rows beyond M get zero codes).
    ``rpw`` < 32: rows per wave -- a tile is shared by 32 / rpw waves of one workgroup (one
    tile per workgroup), which puts many short chains in flight instead of a few long ones
    (what the decode-row launches with a handful of tiles need, and what the short K 2048
    rows want at prefill M too); the scale chunk is assembled by all of them and stored by
    wave 0. The default is 2."""
    assert K % 256 == 0 and (K // 256) % tpw == 0
    assert rpw in (2, 4, 8, 16, 32) and (rpw == 32 or tpw == 1)
    KB = K // 256  # 256-K blocks per row
    JP = rpw // 2  # row pairs per wave
    WPT = 32 // rpw  # waves per tile
    WAVES, TPW = (waves, tpw) if rpw == 32 else (WPT, 1)
    name = (f"flydsl_mxfp8_quant_k{K}_w{WAVES}_t{TPW}" + ("_nt" if nt_store else "")
            + ("_sh" if shuffled else "") + (f"_r{rpw}" if rpw < 32 else ""))  # fmt: skip

    @fx.struct
    class Shared:
        sc: fx.Array[fx.Uint8, 256 * (WAVES * TPW if rpw == 32 else 1), 16]

    @flyc.kernel(name=name, known_block_size=[64 * WAVES, 1, 1])
    def kernel(
        X: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        i32_m: fx.Int32,
        q_rows: fx.Int32,
        n_tiles: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        half, piece = lane // 32, lane % 32  # row of the pair, 16-byte piece of its 512
        group = piece // 4  # 32-K group of the 256-K block (0..7)
        if const_expr(rpw == 32):
            tile = (bx * WAVES + wave) * TPW  # this wave's first tile
            wrow = 0
        else:
            tile = bx  # the workgroup's tile, this wave's rows wrow .. wrow + rpw
            wrow = wave * rpw
        mb, kb = tile // KB, tile % KB
        x_rsrc = buffer_ops.create_buffer_resource(X, max_size=False, num_records_bytes=i32_m * (2 * K))
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=q_rows * K)
        s_rsrc = buffer_ops.create_buffer_resource(S, max_size=False, num_records_bytes=n_tiles * 256)
        row0 = mb * 32 + wrow + half  # this lane's row in its first row pair (pair j: + 2j)
        x_dw = (row0 * (2 * K) + kb * 512 + piece * 16) // 4
        if const_expr(shuffled):
            # this lane's 8 codes: K-chunk of 64 (kb * 4 + piece // 8), 16-chunk (piece % 8) // 2,
            # byte (piece % 2) * 8; the row part is added per row pair below
            q_off = (kb * 4 + piece // 8) * 1024 + ((piece % 8) // 2) * 256 + (piece % 2) * 8
        else:
            q_off = row0 * K + kb * 256 + piece * 8  # bytes
        lds_base = fx.Int32(fx.ptrtoint(smem.sc.ptr)) + (wave * (256 * TPW) if rpw == 32 else 0)
        # the scale byte of (group, row): shuffled chunk position
        # ((klane * 16 + mlane) * 2 + kpack) * 2 + mpack
        sc_pos = ((group % 4) * 16) * 4 + (group // 4) * 2

        def lds_ptr(byte_off):
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(lds_base + byte_off)))

        store_kw = dict(cache_modifier=W_CACHE_MOD) if nt_store else {}
        raw = [
            [
                buffer_ops.buffer_load(
                    x_rsrc, x_dw, vec_width=4, dtype=fx.Int32,
                    soffset_bytes=t * 512 + j * (2 * 2 * K), cache_modifier=W_CACHE_MOD,
                )  # fmt: skip
                for j in range_constexpr(JP)
            ]
            for t in range_constexpr(TPW)
        ]
        for t in range_constexpr(TPW):
            for j in range_constexpr(JP):
                vals = [Vec(raw[t][j])[i] for i in range(4)]
                sb, scale = _e8m0_bf16(_quad_max(_amax_bf16(vals)))
                if const_expr(shuffled):
                    row = row0 + 2 * j
                    q_addr = q_off + t * 4096 + (row // 16) * (16 * K) + (row % 16) * 16
                else:
                    q_addr = q_off + (t * 256 + j * (2 * K))
                buffer_ops.buffer_store(
                    _quant_bf16(vals, scale), q_rsrc, q_addr, offset_is_bytes=True, **store_kw
                )
                # row of the tile wrow + 2j + half -> mlane (.) % 16, mpack (.) // 16
                m_row = wrow + 2 * j + half
                if piece % 4 == 0:
                    llvm.StoreOp(
                        llvm.TruncOp(T.i8, _raw(sb), llvm.IntegerOverflowFlags.none).result,
                        lds_ptr(t * 256 + sc_pos + (m_row % 16) * 4 + m_row // 16),
                    )
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        if const_expr(rpw < 32):
            fx.gpu.barrier()
            if wave == 0:
                s_val = fx.Int32(llvm.load(T.i32, lds_ptr(lane * 4), alignment=4))
                buffer_ops.buffer_store(s_val, s_rsrc, (tile * 64 + lane) * 4, offset_is_bytes=True)
        else:
            for t in range_constexpr(TPW):
                s_val = fx.Int32(llvm.load(T.i32, lds_ptr(t * 256 + lane * 4), alignment=4))
                buffer_ops.buffer_store(s_val, s_rsrc, ((tile + t) * 64 + lane) * 4, offset_is_bytes=True)

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        i32_m: fx.Int32,
        q_rows: fx.Int32,
        n_tiles: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(X, Q, S, i32_m, q_rows, n_tiles).launch(
            grid=(fx.Int64(grid), 1, 1), block=(64 * WAVES, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.waves = WAVES
    launch.tpw = TPW
    launch.tiles_per_wg = WAVES * TPW if rpw == 32 else 1
    return launch


DEFAULT_RPW = 2  # 16 waves per tile: equal at K 6144, -2..-3% linear at K 2048 (prefill), and
# the only way the decode-row launches (a few tiles) keep enough chains in flight
PREFILL_RPW = 4  # from PREFILL_ROWS rows: the quantizer alone -6..-12% at K 3072 / 6144 / 8192
PREFILL_ROWS = 1024  # (M 16384: 29.2 / 53.2 / 67.1 -> 26.9 / 47.1 / 61.2 us), same bytes out


@functools.cache
def _launcher(k, shuffled, rpw):
    return compile_mxfp8_quant(K=k, shuffled=shuffled, rpw=rpw)


def quantize(x, quant, scale, shuffled=False, rpw=None):
    """``x`` bf16 [M, K] contiguous -> ``quant`` fp8 [M, K] row-major (or [padded rows, K] in
    the (16, 16) layout when ``shuffled``), ``scale`` uint8 [padded M, K/32] (shuffled scale
    layout); flat views, one launch. The scale tensor's row count sets the tiles."""
    from amd_kernels.common.tensor_shim import _run_compiled

    m, k = x.shape
    if rpw is None:
        rpw = PREFILL_RPW if m >= PREFILL_ROWS and not shuffled else DEFAULT_RPW
    exe = _launcher(k, shuffled, rpw)
    n_tiles = (scale.shape[0] // 32) * (k // 256)
    per_wg = exe.tiles_per_wg
    grid = (n_tiles + per_wg - 1) // per_wg
    u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
    _run_compiled(
        exe, u8(x), u8(quant), u8(scale), m, quant.shape[0], n_tiles, grid, torch.cuda.current_stream()
    )

