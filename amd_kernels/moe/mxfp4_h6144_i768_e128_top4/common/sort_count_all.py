# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sort.py``'s routing sort in one launch, plus GEMM1's tile map: every CTA histograms all (token, slot) pairs in
LDS itself (those before its range and the rest: the totals and per-CTA prefix ``sort_place_pad`` reads from the
workspace), then pads, scans, places and pads as ``sort_place_pad``. Each CTA reads all of ``topk_ids`` (L2 resident),
so the host takes it while that costs less than the second launch.

Block b's expert comes from a search over the start rows (blocks spread over all CTAs, so skewed routing does not
serialize them); the same thread writes block b's ``I // 128`` entries of ``tile_map.py``'s table.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .sort import SORT_CTAS, THREADS
from .tile_map import _MAX_BLOCKS
from .utils import _lds_atomic_add_i32, wave_incl_scan_i32
from amd_kernels.warm_up import launcher


@functools.cache
@launcher
def compile_moe_sort_count_all(*, E: int, topk: int, block_m: int, I: int):  # noqa: E741
    assert (block_m & (block_m - 1)) == 0 and max(E, block_m) <= THREADS
    assert topk == 4 and 2 * E <= THREADS
    NB_N = I // 128
    N_BITS = (NB_N - 1).bit_length()
    assert _MAX_BLOCKS * NB_N + 16 <= SORT_CTAS * THREADS  # the tail in one pass
    experts_per_cta = (E + SORT_CTAS - 1) // SORT_CTAS
    bm_shift = block_m.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}"
    SCAN_W = THREADS
    E_WAVES = (E + 63) // 64  # waves holding an expert's padded total
    search_steps = E.bit_length() + 1
    # SUB counters per histogram bin, picked by lane: lanes of a wave that count the same expert (skewed routing) hit
    # SUB addresses, not one (an LDS atomic to one address serializes the wave's lanes)
    SUB = 8
    assert (2 * E * SUB) % THREADS == 0 and 2 * E <= THREADS

    @fx.struct
    class Shared:
        cursor: fx.Array[fx.Int32, E]
        total: fx.Array[fx.Int32, E]
        starts: fx.Array[fx.Int32, E + 1]  # expert start rows, [E] = padded total
        hist: fx.Array[fx.Int32, 2 * E * SUB]  # pairs before this CTA's range | the rest, SUB counters each
        scan: fx.Array[fx.Int32, 2 * SCAN_W]

    def pair_range(n_tok, bx):
        """This CTA's [start, end) of the n_tok * topk routing pairs (as sort.py)."""
        total = n_tok * topk
        per_cta = (total + (SORT_CTAS - 1)) // SORT_CTAS
        start = bx * per_cta
        end = start + per_cta
        return start, (end < total).select(end, total)

    @flyc.kernel(name=f"sort_all_place_pad_tile_map_{tag}_i{I}", known_block_size=[THREADS, 1, 1])
    def sort_all_place_pad(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        tile_map: fx.Tensor,  # [grid_entries + 1] i32
        n_tok: fx.Int32,
        grid_entries: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        cursor, total_c, starts, hist = smem.cursor.ptr, smem.total.ptr, smem.starts.ptr, smem.hist.ptr
        scan = [smem.scan.ptr, smem.scan.ptr + SCAN_W]
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        start, end = pair_range(n_tok, bx)
        # 1. histogram of all pairs: [0, E) those before this CTA's range, [E, 2E) the rest, SUB counters per bin
        for z in range_constexpr(2 * E * SUB // THREADS):
            hist[tx + z * THREADS] = 0
        fx.gpu.barrier()
        sub = tx % SUB
        ids_r = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=n_tok * (topk * 4)
        )
        for ct in range(tx, n_tok, THREADS):
            cv = Vec(buffer_ops.buffer_load(ids_r, ct * topk, vec_width=4, dtype=fx.Int32))
            for ck in range_constexpr(topk):
                cp = ct * topk + ck
                half = (cp >= start).select(fx.Int32(E), fx.Int32(0))
                _lds_atomic_add_i32(hist + (half + fx.Int32(cv[ck])) * SUB + sub, 1)
        fx.gpu.barrier()
        if tx < 2 * E:
            hs = fx.Int32(0)
            for j in range_constexpr(SUB):
                hs = hs + fx.Int32(hist[tx * SUB + j])
            scan[1][tx] = hs
        fx.gpu.barrier()
        # 2. inclusive scan of the padded totals (0 beyond E) -> start rows
        in_e = tx < E
        ec = in_e.select(tx, fx.Int32(0))
        pre = fx.Int32(scan[1][ec])
        tot = pre + fx.Int32(scan[1][E + ec])
        padded = in_e.select((tot + (block_m - 1)) & ~(block_m - 1), fx.Int32(0))
        if in_e:
            total_c[tx] = tot
        # per wave by DPP, the totals of the waves holding experts through LDS (one barrier, not one per round)
        incl = wave_incl_scan_i32(padded)
        lane, wave = tx % 64, tx // 64
        scan[0][(lane == 63).select(wave, fx.Int32(E_WAVES))] = incl
        fx.gpu.barrier()
        for w in range_constexpr(E_WAVES - 1):
            incl = incl + (fx.Int32(w) < wave).select(fx.Int32(scan[0][w]), fx.Int32(0))
        if in_e:
            st = incl - padded
            starts[tx] = st
            cursor[tx] = st + pre
        if tx == E - 1:
            starts[E] = incl
        fx.gpu.barrier()
        nv = fx.Int32(starts[E])
        if bx == 0:
            if tx == 0:
                num_valid[0] = nv
                num_valid[1] = fx.Int32(starts[E - 1])  # the last expert's first row
        # 3. block b = gtid: its expert = the first e with starts[e + 1] > b * block_m, its tile-map entries; then the
        #    tile map's -1 tail and count
        gtid = bx * THREADS + tx
        vb = nv >> bm_shift
        if gtid < vb:
            r0 = gtid << bm_shift
            lo = fx.Int32(0)
            n = fx.Int32(E)
            for _ in range_constexpr(search_steps):
                hn = n // 2
                mid = lo + hn
                probe = fx.Int32(starts[(mid < E).select(mid + 1, fx.Int32(E))])
                go_right = (n > 0) & (probe <= r0)
                lo = go_right.select(mid + 1, lo)
                n = go_right.select(n - hn - 1, hn)
            sorted_expert_ids[gtid] = lo
            blo = fx.Int32(starts[lo]) >> bm_shift
            cnt = (fx.Int32(starts[lo + 1]) >> bm_shift) - blo
            for nn in range_constexpr(NB_N):
                tile_map[blo * NB_N + cnt * nn + (gtid - blo)] = (gtid << N_BITS) | nn
        tail = vb * NB_N + gtid
        if tail < grid_entries:
            tile_map[tail] = fx.Int32(-1)
        if gtid == 0:
            tile_map[grid_entries] = vb * NB_N
        # 4. place this CTA's pairs
        for pi in range(start + tx, end, THREADS):
            pp = fx.Int32(pi)
            prow = _lds_atomic_add_i32(cursor + fx.Int32(topk_ids[pp]), 1)
            sorted_ids[prow] = ((pp // topk) & 0x00FFFFFF) | ((pp % topk) << 24)
            sorted_w[prow] = fx.Float32(topk_w[pp])
        # 5. padding: every expert pads by < block_m rows, lanes [0, block_m) do it
        if tx < block_m:
            for ee in range_constexpr(experts_per_cta):
                pe = bx * experts_per_cta + ee
                if pe < E:
                    zlo = fx.Int32(starts[pe]) + fx.Int32(total_c[pe]) + tx
                    for zr in range(zlo, fx.Int32(starts[pe + 1]), THREADS):
                        sorted_ids[fx.Int32(zr)] = n_tok  # token n_tokens, slot 0
                        sorted_w[fx.Int32(zr)] = fx.Float32(0.0)

    @flyc.jit
    def launch_sort_all_tile_map_mxfp4_i768(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_w: fx.Tensor,
        tile_map: fx.Tensor,
        n_tokens: fx.Int32,
        grid_entries: fx.Int32,
        stream: fx.Stream,
    ):
        sort_all_place_pad(
            topk_ids, topk_w, num_valid, sorted_expert_ids, sorted_ids, sorted_w, tile_map,
            n_tokens, grid_entries,
        ).launch(grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)  # fmt: skip

    return launch_sort_all_tile_map_mxfp4_i768


def moe_sort_count_all(bufs, topk_ids, topk_weights, tile_map, grid_entries, n_tokens, *, E, I):  # noqa: E741
    """The routing sort into ``bufs`` (a ``SortBuffers``) and GEMM1's tile map, one launch."""
    import torch

    from .launch import _run_compiled

    # the cached CompiledFunction: a direct jit call re-resolves its cache key on every call (~100+ us of host time)
    _run_compiled(
        compile_moe_sort_count_all(E=E, topk=topk_ids.shape[1], block_m=bufs.block_m, I=I),
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)[:6],
        tile_map.view(-1),
        int(n_tokens),
        int(grid_entries),
        torch.cuda.current_stream(),
    )  # fmt: skip
