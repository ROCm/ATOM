# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sort.py``'s two-launch routing sort with a CTA-major count workspace, plus GEMM1's n-major tile map.

The count launch stores [SORT_CTAS, E] expert histograms, so placement loads contiguous experts per wave and sums eight
partial sums through LDS. CTA e writes expert e's tile-map run (``tile_map.py``'s layout) from the start rows it
already has, without the tile map's binary searches.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from .utils import _lds_atomic_add_i32, wave_incl_scan_i32

from .sort import SORT_CTAS, THREADS
from amd_kernels.warm_up import launcher

# eight threads per expert: fewer requests than four, coalesced across experts
COUNT_PARTS = 8
NB_N = 6  # I / 128 for I768
N_BITS = 3


@functools.cache
@launcher
def compile_moe_sort_tile_map(*, E: int, topk: int, block_m: int):
    assert E <= SORT_CTAS and topk == 4
    assert (block_m & (block_m - 1)) == 0 and max(E, block_m) <= THREADS
    experts_per_cta = (E + SORT_CTAS - 1) // SORT_CTAS
    bm_shift = block_m.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}"

    SCAN_W = THREADS  # the padded-count scan runs on the whole block (no branch)
    assert E <= SCAN_W and (SCAN_W & (SCAN_W - 1)) == 0
    PART = COUNT_PARTS  # threads per expert summing the per-CTA counts in place_pad
    assert PART * E <= THREADS and SORT_CTAS % PART == 0 and PART * E <= 2 * SCAN_W
    E_WAVES = (E + 63) // 64  # waves holding an expert's padded count

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, E]  # per-CTA count / cursor
        total: fx.Array[fx.Int32, E]  # expert total
        padded: fx.Array[fx.Int32, E]
        prefix: fx.Array[
            fx.Int32, E
        ]  # rows of expert e placed by the CTAs before this one
        starts: fx.Array[fx.Int32, E + 1]  # expert start rows, [E] = padded total
        scan: fx.Array[
            fx.Int32, 2 * SCAN_W
        ]  # ping-pong scan arrays; first the PART sums

    def pair_range(n_tok, bx):
        """This CTA's [start, end) of the n_tok * topk routing pairs."""
        total = n_tok * topk
        per_cta = (total + (SORT_CTAS - 1)) // SORT_CTAS
        start = bx * per_cta
        end = start + per_cta
        return start, (end < total).select(end, total)

    @flyc.kernel(name=f"sort_count_cta_major_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_count(topk_ids: fx.Tensor, block_offsets: fx.Tensor, n_tok: fx.Int32):
        count = fx.SharedAllocator().allocate(Shared).peek().count.ptr
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        start, end = pair_range(n_tok, bx)
        if tx < E:
            count[tx] = 0
        fx.gpu.barrier()
        for i in range(start + tx, end, THREADS):
            _lds_atomic_add_i32(count + fx.Int32(topk_ids[fx.Int32(i)]), 1)
        fx.gpu.barrier()
        if tx < E:
            block_offsets[bx * E + tx] = fx.Int32(count[tx])

    @flyc.kernel(name=f"sort_place_pad_tile_map_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_place_pad(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        block_offsets: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        n_tok: fx.Int32,
        tile_map: fx.Tensor,
        grid_entries: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        cursor, total_c, padded_c = smem.count.ptr, smem.total.ptr, smem.padded.ptr
        prefix_c, starts = smem.prefix.ptr, smem.starts.ptr
        scan = [smem.scan.ptr, smem.scan.ptr + SCAN_W]
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        # 1. per expert: the total over the SORT_CTAS counts and the rows the CTAs
        #    before this one place; PART threads per expert, SORT_CTAS / PART
        #    counts each (independent loads), partial sums through LDS
        e, k = tx % E, tx // E
        if tx < PART * E:
            tot = fx.Int32(0)
            pre = fx.Int32(0)
            for i in range_constexpr(SORT_CTAS // PART):
                c = k * (SORT_CTAS // PART) + i
                v = fx.Int32(block_offsets[c * E + e])
                tot = tot + v
                pre = pre + (c < bx).select(v, 0)
            scan[0][tx] = tot
            scan[1][tx] = pre
        fx.gpu.barrier()
        if tx < E:
            tot = fx.Int32(0)
            pre = fx.Int32(0)
            for i in range_constexpr(PART):
                tot = tot + fx.Int32(scan[0][i * E + tx])
                pre = pre + fx.Int32(scan[1][i * E + tx])
            total_c[tx] = tot
            padded_c[tx] = (tot + (block_m - 1)) & ~(block_m - 1)
            prefix_c[tx] = pre
        fx.gpu.barrier()
        # 2. inclusive scan of the padded counts (0 beyond E) -> expert start rows
        #    (every CTA computes them): per wave by DPP, the totals of the waves
        #    holding experts through LDS, one barrier
        in_e = tx < E
        pd = in_e.select(fx.Int32(padded_c[in_e.select(tx, 0)]), 0)
        incl = wave_incl_scan_i32(pd)
        lane, wave = tx % 64, tx // 64
        scan[0][(lane == 63).select(wave, fx.Int32(E_WAVES))] = incl
        fx.gpu.barrier()
        for w in range_constexpr(E_WAVES - 1):
            incl = incl + (fx.Int32(w) < wave).select(fx.Int32(scan[0][w]), fx.Int32(0))
        if tx < E:
            start = incl - pd
            starts[tx] = start
            cursor[tx] = start + fx.Int32(prefix_c[tx])
        if tx == E - 1:
            starts[E] = incl
        fx.gpu.barrier()
        if bx == 0:
            if tx == 0:
                num_valid[0] = fx.Int32(starts[E])
                num_valid[1] = fx.Int32(starts[E - 1])  # the last expert's first row
        # every CTA knows every expert start: CTA e writes expert e's map
        if bx < E:
            blo = fx.Int32(starts[bx]) >> bm_shift
            cnt = (fx.Int32(starts[bx + 1]) >> bm_shift) - blo
            if tx // 64 < NB_N:
                nn = tx // 64
                for j in range(tx % 64, cnt, 64):
                    local = fx.Int32(j)
                    m = blo + local
                    tile_map[blo * NB_N + cnt * nn + local] = (m << N_BITS) | nn
            for j in range(tx, cnt, THREADS):
                sorted_expert_ids[blo + fx.Int32(j)] = bx
        vb = fx.Int32(starts[E]) >> bm_shift
        tail = vb * NB_N + bx * THREADS + tx
        if tail < grid_entries:
            tile_map[tail] = fx.Int32(-1)
        if bx == 0:
            if tx == 0:
                tile_map[grid_entries] = vb * NB_N
        # 3. place this CTA's pairs
        start, end = pair_range(n_tok, bx)
        for i in range(start + tx, end, THREADS):
            p = fx.Int32(i)
            row = _lds_atomic_add_i32(cursor + fx.Int32(topk_ids[p]), 1)
            sorted_ids[row] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
            sorted_w[row] = fx.Float32(topk_w[p])
        fx.gpu.barrier()
        # 4. padding: every expert pads by < block_m rows, lanes [0, block_m) do it
        if tx < block_m:
            for ee in range_constexpr(experts_per_cta):
                e = bx * experts_per_cta + ee
                if e < E:
                    lo = fx.Int32(starts[e]) + fx.Int32(total_c[e]) + tx
                    for r in range(lo, fx.Int32(starts[e + 1]), THREADS):
                        sorted_ids[fx.Int32(r)] = n_tok  # token n_tokens, slot 0
                        sorted_w[fx.Int32(r)] = fx.Float32(0.0)

    @flyc.jit
    def launch_sort_tile_map_mxfp4_i768(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_w: fx.Tensor,
        block_offsets: fx.Tensor,
        n_tokens: fx.Int32,
        tile_map: fx.Tensor,
        grid_entries: fx.Int32,
        stream: fx.Stream,
    ):
        sort_count(topk_ids, block_offsets, n_tokens).launch(
            grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )
        sort_place_pad(
            topk_ids,
            topk_w,
            block_offsets,
            num_valid,
            sorted_expert_ids,
            sorted_ids,
            sorted_w,
            n_tokens,
            tile_map,
            grid_entries,
        ).launch(grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch_sort_tile_map_mxfp4_i768


def moe_sort_tile_map(bufs, topk_ids, topk_weights, tile_map, grid_entries, n_tokens, *, E):
    """The two-launch sort and the I768 n-major tile map into the caller's buffers."""
    from .launch import _run_compiled

    args = bufs.launch_args(topk_ids, topk_weights, n_tokens)
    # the cached CompiledFunction, not a direct jit call (host cost)
    _run_compiled(
        compile_moe_sort_tile_map(E=E, topk=topk_ids.shape[1], block_m=bufs.block_m),
        *args[:-1], tile_map, int(grid_entries), args[-1],
    )
