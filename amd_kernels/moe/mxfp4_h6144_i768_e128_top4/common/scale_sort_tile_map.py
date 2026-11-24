# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""GEMM1's sorted + swizzled e8m0 activation scales (H 6144: 192 bytes per token row, the bytes of aiter's
``e8m0_shuffle`` of the gathered rows) and, with ``with_tile_map``, ``tile_map.py``'s table from the first ``TM_CTAS``
CTAs, in one launch (gfx950).

Per 32-row tile, each row gathers its token's 192 scale bytes (8 threads per row, three 8-B masked loads; invalid and
padding rows stage 0), then the tile's byte (xl, y) lands at tile * 6144 + (y // 8) * 256 + ((y % 8) % 4) * 64 +
(xl % 16) * 4, packed little-endian as [xl][y], [xl + 16][y], [xl][y + 4], [xl + 16][y + 4]. Tiles past
num_valid_ids[0] are skipped (the GEMM never reads them).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .epilogue import _lds_ptr
from .loaders import _min
from amd_kernels.warm_up import launcher

SCN = 192  # H / 32 scale bytes per token row
LDS_STRIDE = SCN + 4  # +4B pad: kills the 32-bank aliasing of the xl/xl+16 reads
TILE_B = 32 * SCN  # 6144 B per swizzled tile

TM_CTAS = 16  # the tile_map duty lives on CTAs [0, TM_CTAS)
_MAX_BLOCKS = 4096  # tile_map.py: >= max_sorted / BM at every covered shape
_SEARCH_STEPS = 13  # 2^13 > _MAX_BLOCKS


def _lds_u8(base, off):
    v = _llvm.LoadOp(
        _ir.IntegerType.get_signless(8), _lds_ptr(base + off), alignment=1
    ).result
    return fx.Int32(_llvm.ZExtOp(T.i32, v).result)


def _lds_store_dw(val, base, off):
    _llvm.StoreOp(fx.Int32(val).ir_value(), _lds_ptr(base + off), alignment=4)


@launcher
def compile_scale_sort_tile_map(*, block_m: int, a_group_bj: int = 0, with_tile_map: bool = True):
    """The scale sort for sort blocks of ``block_m`` rows, with the tile map on the first ``TM_CTAS`` CTAs.

    Args:
        a_group_bj: ``tile_map.py``'s grouped tile order (a trace-time int; 0: n-major).
        with_tile_map: False drops the tile-map CTAs (the routing sort writes the map; their two 13-step binary
            searches over global memory were this kernel's longest path).
    """
    bm_shift = block_m.bit_length() - 1
    name = f"scale_sort_tile_map_h6144_i768_bm{block_m}"
    if not with_tile_map:
        name = f"scale_sort_h6144_i768_bm{block_m}"
    if a_group_bj:
        name += f"_agrp{a_group_bj}"
    TM = TM_CTAS if with_tile_map else 0
    NB_N = 6  # I // 128 with I = 768
    N_BITS = (NB_N - 1).bit_length()
    TAIL_ITERS = (_MAX_BLOCKS * NB_N + 8) // (256 * TM_CTAS) + 1
    assert a_group_bj == 0 or NB_N % a_group_bj == 0

    @fx.struct
    class Shared:
        stage: fx.Array[fx.Int8, 32 * LDS_STRIDE]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        scale: fx.Tensor,  # [n_tok, SCN] u8, token order
        sorted_ids: fx.Tensor,  # [capacity] i32, token | slot<<24
        sorted_expert_ids: fx.Tensor,  # [capacity // block_m] i32
        num_valid: fx.Tensor,  # [2] i32
        out: fx.Tensor,  # [capacity32, SCN] u8, swizzled tiles
        tile_map: fx.Tensor,  # [grid_entries + 1] i32
        n_tok: fx.Int32,
        grid_entries: fx.Int32,
        i32_sid_dw: fx.Int32,  # sorted_ids capacity in dwords (bounds for masks)
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        nv = fx.Int32(num_valid[0])
        if bx < TM:
            # tile_map.py's kernel: thread gtid handles sorted block gtid
            gtid = bx * 256 + tx
            vb = nv >> bm_shift  # valid blocks (num_valid is BM-padded)

            def _eid(i):
                return fx.Int32(sorted_expert_ids[i])

            def _bound(e, upper):
                """First index in [0, vb) whose expert id is > e / >= e."""
                lo = fx.Int32(0)
                n = vb
                for _ in range_constexpr(_SEARCH_STEPS):
                    half = n // 2
                    mid = lo + half
                    probe = _eid(_min(mid, vb - 1))
                    go_right = (n > 0) & ((probe <= e) if upper else (probe < e))
                    lo = go_right.select(mid + 1, lo)
                    n = go_right.select(n - half - 1, half)
                return lo

            m = gtid
            if m < vb:
                e = _eid(m)
                lo = _bound(e, False)
                hi = _bound(e, True)
                cnt = hi - lo
                # tile_map.py's _pos: n-major, or 8-m-block x a_group_bj-slab groups, m fastest
                for n in range_constexpr(NB_N):
                    if const_expr(a_group_bj > 0):
                        mg = (m - lo) // 8
                        mi = (m - lo) - mg * 8
                        gs = _min(cnt - mg * 8, fx.Int32(8))
                        pos = (
                            lo * NB_N
                            + mg * (8 * NB_N)
                            + (n // a_group_bj) * (gs * a_group_bj)
                            + (n % a_group_bj) * gs
                            + mi
                        )
                    else:
                        pos = lo * NB_N + cnt * n + (m - lo)
                    tile_map[pos] = (m << N_BITS) | n
            tail0 = vb * NB_N
            for it in range_constexpr(TAIL_ITERS):
                i = tail0 + gtid + it * (256 * TM_CTAS)
                if i < grid_entries:
                    tile_map[i] = fx.Int32(-1)
            if gtid == 0:
                tile_map[grid_entries] = tail0
        else:
            tile = bx - TM
            if tile * 32 < nv:
                lds = fx.SharedAllocator().allocate(Shared).peek()
                base = fx.Int32(fx.ptrtoint(lds.stage.ptr))
                # load phase: thread handles row i = tx//8, bytes [kk, kk+24)
                i = tx // 8
                kk4 = (tx % 8) * 6  # 24 bytes = 6 dwords; kk4 = kk / 4
                row = tile * 32 + i
                # masked: rows >= num_valid (past capacity too) load 0, which the row < nv term excludes again
                idr = buffer_ops.create_buffer_resource(
                    sorted_ids, max_size=False, num_records_bytes=i32_sid_dw * 4
                )
                in_nv = row < nv
                sid = fx.Int32(
                    buffer_ops.buffer_load(
                        idr, row, vec_width=1, dtype=fx.Int32, mask=in_nv
                    )
                )
                tok = sid & 0x00FFFFFF
                ok = in_nv & (tok < n_tok)
                sr = buffer_ops.create_buffer_resource(
                    scale, max_size=False, num_records_bytes=n_tok * SCN
                )
                row_dw = tok * (SCN // 4) + kk4
                w = [
                    Vec(
                        buffer_ops.buffer_load(
                            sr,
                            row_dw + 2 * j,
                            vec_width=2,
                            dtype=fx.Int32,
                            mask=ok,
                        )
                    )
                    for j in range_constexpr(3)
                ]
                for j in range_constexpr(3):
                    for d in range_constexpr(2):
                        _lds_store_dw(
                            fx.Int32(w[j][d]),
                            base,
                            i * LDS_STRIDE + (kk4 + 2 * j + d) * 4,
                        )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                # store phase: 1536 dwords; thread tx takes d = tx + 256*j
                or_ = buffer_ops.create_buffer_resource(out, max_size=True)
                tile_dw = tile * (TILE_B // 4)
                for j in range_constexpr(6):
                    d = tx + 256 * j
                    y_blk = d >> 6
                    t63 = d & 63
                    xl = t63 & 15
                    c = t63 >> 4
                    col_a = y_blk * 8 + c
                    b0 = _lds_u8(base, xl * LDS_STRIDE + col_a)
                    b1 = _lds_u8(base, (xl + 16) * LDS_STRIDE + col_a)
                    b2 = _lds_u8(base, xl * LDS_STRIDE + col_a + 4)
                    b3 = _lds_u8(base, (xl + 16) * LDS_STRIDE + col_a + 4)
                    packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                    buffer_ops.buffer_store(
                        packed, or_, tile_dw + y_blk * 64 + c * 16 + xl
                    )

    @flyc.jit
    def launch_scale_sort_tile_map_mxfp4_i768(
        scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        out: fx.Tensor,
        tile_map: fx.Tensor,
        n_tok: fx.Int32,
        grid_entries: fx.Int32,
        n_tiles: fx.Int32,
        i32_sid_dw: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid,
            out,
            tile_map,
            n_tok,
            grid_entries,
            i32_sid_dw,
        ).launch(
            grid=(fx.Int64(n_tiles + TM), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_scale_sort_tile_map_mxfp4_i768.kernel_name = name
    return launch_scale_sort_tile_map_mxfp4_i768


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_scale_sort_tile_map(block_m: int, a_group_bj: int = 0, with_tile_map: bool = True):
    return compile_scale_sort_tile_map(block_m=block_m, a_group_bj=a_group_bj, with_tile_map=with_tile_map)


def scale_sort_tile_map(
    scale: torch.Tensor,  # [n_tokens, 192] u8 / fp8_e8m0, token order
    sorted_ids: torch.Tensor,  # [capacity] int32
    sorted_expert_ids: torch.Tensor,  # [capacity // block_m] int32
    num_valid: torch.Tensor,  # [2] int32
    out: torch.Tensor,  # [capacity32, 192] u8 / fp8_e8m0 (swizzled out)
    tile_map: torch.Tensor,  # [grid_entries + 1] int32
    grid_entries: int,
    n_tokens: int,
    block_m: int,
    a_group_bj: int = 0,
    with_tile_map: bool = True,
) -> None:
    """GEMM1's sorted + swizzled activation scales and, with ``with_tile_map``, the tile map, in one launch."""
    n_tiles = (sorted_ids.numel() + 31) // 32
    _run_compiled(
        _get_scale_sort_tile_map(block_m, a_group_bj, with_tile_map),
        scale.view(torch.uint8).view(-1),
        sorted_ids.view(-1),
        sorted_expert_ids.view(-1),
        num_valid.view(-1),
        out.view(torch.uint8).view(-1),
        tile_map.view(-1),
        n_tokens,
        grid_entries,
        n_tiles,
        sorted_ids.numel(),
        torch.cuda.current_stream(),
    )
