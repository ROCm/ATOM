# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-workgroup expert sort for at most 512 tokens (16 / 32-row blocks); no output clearing.

The sorted table is built in LDS: every row is written as padding under the id loads, the pairs land on their rows
after one wave's scan of the 64 expert pairs, and rows [0, num_valid) leave in 16 B stores after all LDS reads (one
CU's memory pipeline carries every store, so scattered dword stores would be most of the kernel). A block's first row
always holds a pair, which writes the block -> expert entry. Each pair owns its GEMM2 output row, so placement order
cannot change its bits.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from .common.launch import _run_compiled
from .common.sort import max_sorted_rows
from .common.utils import _lds_atomic_add_i32, wave_incl_scan_i32
from amd_kernels.warm_up import launcher


threads = 256  # one thread per histogram/scan entry
# rank placement from 17 (inline routing below): chain -0.05..-0.20 us at 64-512, sort alone -0.03 us at 17-32
SORT_RANK_FROM = 17
# compile tiers (max tokens; pairs per thread = 4 * tier / 256), each reached by a WARM_UP_TOKENS size of the entry
SORT_TOKEN_TIERS = (64, 128, 192, 256, 384, 448, 512)


@functools.cache
@launcher
def compile_a16w4_sort(
    *, E: int, topk: int, block_m: int, max_tokens: int, rank: bool = True
):
    """Histogram, padded-count scan and placement, with no cross-CTA state."""
    assert E == 128
    assert (block_m & (block_m - 1)) == 0 and threads >= E
    bm_shift = block_m.bit_length() - 1
    PPT = (max_tokens * topk + threads - 1) // threads
    n_waves = threads // 64
    pad_epw = 64 // block_m  # experts per wave and padding pass
    assert block_m <= 64 and E % (pad_epw * n_waves) == 0
    tag = f"e{E}_top{topk}_bm{block_m}_ppt{PPT}_t{threads}" + ("_rank" if rank else "")
    # rank: invalid pairs count into their own slot past E (distinct addresses, never read)
    COUNT = E + threads * PPT if rank else E
    # cursor: invalid pairs advance their own slot past E too (no branch around the atomic)
    CURSOR = E if rank else E + threads * PPT
    # the sorted table in LDS (ids rows, then weight bits): padding everywhere first, the pairs on top, then
    # [0, num_valid) leaves in 16 B stores
    MS = max_sorted_rows(max_tokens, E, topk, block_m)
    CH = (MS // 4 + threads - 1) // threads  # 16 B tiles per thread and array
    TAB = CH * threads * 4  # rows per array
    assert TAB > MS  # row MS takes the invalid pairs
    NB = MS // block_m  # blocks of the bound: one block -> expert entry per thread
    assert NB <= threads

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, COUNT]  # rows per expert
        cursor: fx.Array[fx.Int32, CURSOR]  # next free sorted row per expert
        scan: fx.Array[fx.Int32, E]  # ping-pong scan arrays
        nvs: fx.Array[fx.Int32, 4]  # num_valid for every wave
        tab: fx.Array[fx.Int32, 2 * TAB, 16]  # sorted ids rows, then weight bits (16 B aligned)
        blk: fx.Array[fx.Int32, threads + threads * PPT]  # block -> expert; past threads: one dummy per pair

    @flyc.kernel(name=f"sort_a16w4_pairs_{tag}", known_block_size=[threads, 1, 1])
    def sort_a16w4(
        topk_ids: fx.Tensor,  # [n_tokens * topk] i32
        topk_w: fx.Tensor,  # [n_tokens * topk] f32
        sorted_ids: fx.Tensor,  # [max_sorted] i32
        sorted_w: fx.Tensor,  # [max_sorted] f32
        sorted_eids: fx.Tensor,  # [max_sorted / block_m] i32
        num_valid: fx.Tensor,  # [2] i32
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        if bx == 0:
            count, cursor, blk = smem.count.ptr, smem.cursor.ptr, smem.blk.ptr
            scan = [smem.scan.ptr]
            n_pairs = n_tok * topk
            last_pair = n_pairs - 1
            t_lt_E = tx < E
            active = (n_pairs < E).select(n_pairs, fx.Int32(E))
            ms_rows = ((n_pairs + active * (block_m - 1) + (block_m - 1)) >> bm_shift) << bm_shift
            ids_r = buffer_ops.create_buffer_resource(sorted_ids, max_size=False, num_records_bytes=ms_rows * 4)
            w_r = buffer_ops.create_buffer_resource(sorted_w, max_size=False, num_records_bytes=ms_rows * 4)
            e_r = buffer_ops.create_buffer_resource(sorted_eids, max_size=False,
                                                    num_records_bytes=(ms_rows >> bm_shift) * 4)

            # 0. this thread's pairs: one batch of loads (clamped index, validity)
            pairs = []
            for k in range_constexpr(PPT):
                p = tx + k * threads
                valid = p < n_pairs
                pc = valid.select(p, last_pair)
                pairs.append((p, valid, fx.Int32(topk_ids[pc]), fx.Float32(topk_w[pc])))

            lds16 = fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, smem.tab.ptr), fx.make_layout(2 * TAB, 1)),
                fx.make_layout(4, 1),
            )
            lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

            def lds_store16(tile, vec4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r.store(vec4)
                fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

            def lds_load16(tile):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
                return Vec(r.load())

            # every table row as padding (token n_tokens, slot 0, weight 0), under the id loads
            pad_ids = Vec.from_elements([n_tok] * 4, fx.Int32)
            pad_w = Vec.filled(4, 0, fx.Int32)
            for j in range_constexpr(CH):
                lds_store16(tx + j * threads, pad_ids)
                lds_store16(tx + (j + CH) * threads, pad_w)
            tab = smem.tab.ptr

            # 1. zero the counters
            if t_lt_E:
                count[tx] = 0
            fx.gpu.barrier()

            # 2. histogram (by scan slot); rank: the returned count is the pair's rank
            ranks = []
            for k, (p, valid, e, _w) in enumerate(pairs):
                if const_expr(rank):
                    slot = valid.select(e, fx.Int32(E + k * threads) + tx)
                    ranks.append(_lds_atomic_add_i32(count + slot, 1))
                else:
                    if valid:
                        _lds_atomic_add_i32(count + e, 1)
            fx.gpu.barrier()

            # One wave scans 64 consecutive expert pairs; starts/ends stay in expert order.
            if wave == 0:
                c0, c1 = fx.Int32(count[lane*2]), fx.Int32(count[lane*2+1])
                p0 = (c0 + (block_m-1)) & ~(block_m-1)
                p1 = (c1 + (block_m-1)) & ~(block_m-1)
                pair_end = wave_incl_scan_i32(p0+p1)
                pair_start = pair_end-(p0+p1)
                for j in range_constexpr(2):
                    e_t = lane*2+j
                    start = pair_start + (p0 if const_expr(j == 1) else fx.Int32(0))
                    end = start + (p0 if const_expr(j == 0) else p1)
                    scan[0][e_t] = start if const_expr(rank) else end
                    if const_expr(not rank):
                        cursor[e_t] = start
                if lane == 63:
                    num_valid[0] = pair_end
                    smem.nvs.ptr[0] = pair_end
            fx.gpu.barrier()
            rows = []
            for k, (p, valid, e, w) in enumerate(pairs):
                if const_expr(rank):
                    rows.append(fx.Int32(scan[0][e]) + ranks[k])
                else:
                    slot = valid.select(e, fx.Int32(E + k * threads) + tx)
                    rows.append(_lds_atomic_add_i32(cursor + slot, 1))
            nv = fx.Int32(smem.nvs.ptr[0])
            # 5. place the pairs into the table (rows of a pair's expert, disjoint from its padding)
            for k, (p, valid, e, w) in enumerate(pairs):
                r = valid.select(rows[k], fx.Int32(MS))  # invalid pairs: a row past MS, never copied
                tab[r] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
                tab[r + TAB] = w.bitcast(fx.Int32)
                # a block's first row is always a pair (padding < block_m rows): its pair names the block
                first = valid & ((r & (block_m - 1)) == 0)
                blk[first.select(r >> bm_shift, fx.Int32(threads + k * threads) + tx)] = e
            fx.gpu.barrier()
            # 6. rows [0, num_valid) in 16 B stores: every LDS read first, then the stores
            tiles = []
            for j in range_constexpr(CH):
                t = tx + j * threads
                tiles.append((t, lds_load16(t), lds_load16(t + CH * threads)))
            blk_e = fx.Int32(blk[tx])
            buffer_ops.buffer_store(blk_e, e_r, tx, mask=tx < (nv >> bm_shift))
            for t, vi, vw in tiles:
                in_rows = t * 4 < nv
                buffer_ops.buffer_store(vi, ids_r, t * 4, mask=in_rows)
                buffer_ops.buffer_store(vw, w_r, t * 4, mask=in_rows)

    @flyc.jit
    def launch_sort_pairs_mxfp4_a16w4(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        sort_a16w4(
            topk_ids,
            topk_w,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            n_tokens,
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch_sort_pairs_mxfp4_a16w4


def moe_sort_a16w4(topk_ids, topk_weights, E, block_m, *, rank=None):
    """Packed rows, sorted weights, block experts and the padded valid row count; ``rank`` (None: from
    ``SORT_RANK_FROM`` tokens) picks ``compile_a16w4_sort``'s placement."""
    n_tokens, topk = topk_ids.shape
    if rank is None:
        rank = n_tokens >= SORT_RANK_FROM
    dev = topk_ids.device
    ms = max_sorted_rows(n_tokens, E, topk, block_m)
    sorted_ids = torch.empty(ms, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(ms, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(ms // block_m, dtype=torch.int32, device=dev)
    num_valid = torch.empty(1, dtype=torch.int32, device=dev)
    assert n_tokens <= 512
    launch = compile_a16w4_sort(
        E=E, topk=topk, block_m=block_m,
        max_tokens=next(t for t in SORT_TOKEN_TIERS if n_tokens <= t), rank=rank,
    )
    # through the cached handle: a direct @flyc.jit call enters the JIT on every call (FlyDSL 0.5: a full trace)
    _run_compiled(
        launch,
        topk_ids.contiguous().int().view(-1),
        topk_weights.contiguous().float().view(-1),
        sorted_ids, sorted_w, sorted_eids, num_valid, int(n_tokens),
        torch.cuda.current_stream(),
    )
    return sorted_ids, sorted_w, sorted_eids, num_valid
