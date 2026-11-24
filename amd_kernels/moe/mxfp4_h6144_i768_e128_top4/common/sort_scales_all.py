# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sort_count_all.py``'s routing sort and tile map plus ``scale_sort_tile_map.py``'s scale tiles, in one launch.

The scale tiles need every row's token, so placement is owned by row blocks, not pair ranges: CTA c owns the sorted
blocks ``[c * bpc, (c + 1) * bpc)`` (skewed routing spreads a hot expert's blocks over many CTAs), and a block scan of
the threads' match counts of the block's expert ranks its pairs in pair order. Rows inside an expert thus come out in
pair order, not atomic order; a GEMM output element depends on its own row only, so the entry's outputs are bit-equal.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .scale_sort_tile_map import LDS_STRIDE, SCN, TILE_B, _lds_store_dw, _lds_u8
from .sort import SORT_CTAS, THREADS, max_sorted_rows
from .tile_map import _MAX_BLOCKS
from .utils import _lds_atomic_add_i32, wave_incl_scan_i32
from amd_kernels.warm_up import launcher

# tokens per thread of the pair scan, by token count (one compile each)
TPT_TIERS = ((1024, 1), (2048, 2), (4096, 4), (8192, 8))
MAX_TOKENS = TPT_TIERS[-1][0]


def tpt_for(n_tokens: int) -> int:
    for bound, tpt in TPT_TIERS:
        if n_tokens <= bound:
            return tpt
    raise ValueError(f"sort_scales_all covers up to {MAX_TOKENS} tokens, got {n_tokens}")


@functools.cache
@launcher
def compile_moe_sort_scales_all(*, E: int, topk: int, block_m: int, I: int, tpt: int):  # noqa: E741
    assert (block_m & (block_m - 1)) == 0 and 128 <= block_m <= THREADS
    assert topk == 4 and 2 * E <= THREADS
    NB_N = I // 128
    N_BITS = (NB_N - 1).bit_length()
    assert _MAX_BLOCKS * NB_N + 16 <= SORT_CTAS * THREADS  # the tail in one pass
    bm_shift = block_m.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}"
    NWAVE = THREADS // 64
    search_steps = E.bit_length() + 1
    SUB = 8  # sub-counters per histogram bin, picked by lane (skewed routing)
    assert (E * SUB) % THREADS == 0
    TILES = block_m // 32  # 32-row scale tiles per block
    TPP = THREADS // 256  # tiles staged per pass (256 threads per tile)
    PASSES = TILES // TPP
    assert TILES % TPP == 0
    max_blocks = max_sorted_rows(THREADS * tpt, E, topk, block_m) // block_m
    BPC_MAX = (max_blocks + SORT_CTAS - 1) // SORT_CTAS

    @fx.struct
    class Shared:
        total: fx.Array[fx.Int32, E]
        starts: fx.Array[fx.Int32, E + 1]  # expert start rows, [E] = padded total
        hist: fx.Array[fx.Int32, E * SUB]
        wsum: fx.Array[fx.Int32, NWAVE + 1]  # wave totals of a block scan
        blk_e: fx.Array[fx.Int32, 1]  # the current block's expert
        rows: fx.Array[fx.Int32, block_m + 64]  # the block's sorted ids, then a dump slot per lane
        stage: fx.Array[fx.Int8, TPP * 32 * LDS_STRIDE]

    @flyc.kernel(
        name=f"sort_all_place_pad_tile_map_scales_{tag}_i{I}_t{tpt}",
        known_block_size=[THREADS, 1, 1],
    )
    def sort_scales_all(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        tile_map: fx.Tensor,  # [grid_entries + 1] i32
        scale: fx.Tensor,  # [n_tok, SCN] u8, token order
        out: fx.Tensor,  # [capacity32, SCN] u8, swizzled tiles
        n_tok: fx.Int32,
        grid_entries: fx.Int32,
        i32_ctas: fx.Int32,  # grid size (>= SORT_CTAS)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        total_c, starts, hist, rows = smem.total.ptr, smem.starts.ptr, smem.hist.ptr, smem.rows.ptr
        stage_base = fx.Int32(fx.ptrtoint(smem.stage.ptr))
        wsum, blk_e = smem.wsum.ptr, smem.blk_e.ptr
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = tx // 64

        def block_incl_scan(v):
            """Inclusive scan of v over the block: per wave by DPP, then the wave totals through LDS and one barrier;
            the caller barriers before the next scan."""
            x = wave_incl_scan_i32(v)
            wsum[(lane == 63).select(wave, fx.Int32(NWAVE))] = x  # [NWAVE] = dump slot
            fx.gpu.barrier()
            off = fx.Int32(0)
            for w in range_constexpr(NWAVE):
                off = off + (fx.Int32(w) < wave).select(fx.Int32(wsum[w]), fx.Int32(0))
            return x + off

        # 1. this thread's tokens' ids; histogram of all pairs
        for z in range_constexpr(E * SUB // THREADS):
            hist[tx + z * THREADS] = 0
        ids_r = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=n_tok * (topk * 4)
        )
        toks, ok, cid = [], [], []
        for j in range_constexpr(tpt):
            t = tx + j * THREADS  # coalesced; the rank order is (thread, j, slot)
            v = t < n_tok
            toks.append(t)
            ok.append(v)
            cid.append(Vec(buffer_ops.buffer_load(ids_r, t * topk, vec_width=4, dtype=fx.Int32, mask=v)))
        fx.gpu.barrier()
        sub = tx % SUB
        for j in range_constexpr(tpt):
            if ok[j]:
                for k in range_constexpr(topk):
                    _lds_atomic_add_i32(hist + fx.Int32(cid[j][k]) * SUB + sub, 1)
        fx.gpu.barrier()
        # 2. totals, inclusive scan of the padded totals -> start rows
        in_e = tx < E
        ec = in_e.select(tx, fx.Int32(0))
        tot = fx.Int32(0)
        for s in range_constexpr(SUB):
            tot = tot + fx.Int32(hist[ec * SUB + s])
        padded = in_e.select((tot + (block_m - 1)) & ~(block_m - 1), fx.Int32(0))
        if in_e:
            total_c[tx] = tot
        incl = block_incl_scan(padded)
        if in_e:
            starts[tx] = incl - padded
        if tx == E - 1:
            starts[E] = incl
        fx.gpu.barrier()
        nv = fx.Int32(starts[E])
        if bx == 0:
            if tx == 0:
                num_valid[0] = nv
                num_valid[1] = fx.Int32(starts[E - 1])  # the last expert's first row

        def expert_of_row(r0):
            """The first e with starts[e + 1] > r0."""
            lo = fx.Int32(0)
            n = fx.Int32(E)
            for _ in range_constexpr(search_steps):
                hn = n // 2
                mid = lo + hn
                probe = fx.Int32(starts[(mid < E).select(mid + 1, fx.Int32(E))])
                go_right = (n > 0) & (probe <= r0)
                lo = go_right.select(mid + 1, lo)
                n = go_right.select(n - hn - 1, hn)
            return lo

        # 3. block b = gtid: its expert and tile-map entries; the tile map's tail, count
        gtid = bx * THREADS + tx
        vb = nv >> bm_shift
        if gtid < vb:
            lo = expert_of_row(gtid << bm_shift)
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
        # 4. this CTA's blocks: placement in pair order, padding, scale tiles (the host launches a CTA per possible
        #    block, so bpc is 1; the loop keeps any grid valid)
        bpc = (vb + (i32_ctas - 1)) // i32_ctas
        b0 = bx * bpc
        b1 = b0 + bpc
        b1 = (b1 < vb).select(b1, vb)
        sr = buffer_ops.create_buffer_resource(
            scale, max_size=False, num_records_bytes=n_tok * SCN
        )
        or_ = buffer_ops.create_buffer_resource(out, max_size=True)
        w_r = buffer_ops.create_buffer_resource(
            topk_w, max_size=False, num_records_bytes=n_tok * (topk * 4)
        )
        for blk in range(b0, b1):
            row0 = fx.Int32(blk) << bm_shift
            # the block's expert: the one whose padded rows hold row0 (one compare per thread, no dependent search)
            if in_e:
                if (fx.Int32(starts[tx]) <= row0) & (row0 < fx.Int32(starts[tx + 1])):
                    blk_e[0] = tx
            fx.gpu.barrier()
            e = fx.Int32(blk_e[0])
            e_start = fx.Int32(starts[e])
            e_tot = fx.Int32(total_c[e])
            r0 = row0 - e_start  # the block's first rank of e
            # exclusive scan of the threads' match counts of e
            mc = fx.Int32(0)
            for j in range_constexpr(tpt):
                for k in range_constexpr(topk):
                    hit = ok[j] & (fx.Int32(cid[j][k]) == e)
                    mc = mc + hit.select(fx.Int32(1), fx.Int32(0))
            rank = block_incl_scan(mc) - mc - r0
            # the pairs ranked in this block put their id in its LDS row (no branch: the others write their lane's dump
            # slot)
            for j in range_constexpr(tpt):
                for k in range_constexpr(topk):
                    hit = ok[j] & (fx.Int32(cid[j][k]) == e)
                    put = hit & (rank >= 0) & (rank < block_m)
                    rows[put.select(rank, block_m + lane)] = (toks[j] & 0x00FFFFFF) | (k << 24)
                    rank = rank + hit.select(fx.Int32(1), fx.Int32(0))
            fx.gpu.barrier()
            # one thread per row: sorted id and weight (stored after the first scale loads are out); rows past e's total
            # are padding
            in_b = tx < block_m
            pad = r0 + tx >= e_tot
            val = (pad | ~in_b).select(n_tok, fx.Int32(rows[in_b.select(tx, fx.Int32(0))]))
            pair = (val & 0x00FFFFFF) * topk + (val >> 24)
            wgt = fx.Float32(buffer_ops.buffer_load(w_r, pair, vec_width=1, dtype=fx.Float32, mask=~pad & in_b))
            # scale tiles: 256 threads per tile, TPP tiles per pass
            lt = tx % 256
            ti = tx // 256
            i = lt // 8  # row of the tile
            kk4 = (lt % 8) * 6  # 24 bytes = 6 dwords of the row
            st = stage_base + ti * (32 * LDS_STRIDE)
            for p in range_constexpr(PASSES):
                tile_r = (p * TPP + ti) * 32  # first row of the tile in the block
                okr = r0 + tile_r + i < e_tot  # padding rows stage zeros
                tok = fx.Int32(rows[okr.select(tile_r + i, fx.Int32(0))]) & 0x00FFFFFF
                row_dw = tok * (SCN // 4) + kk4
                wv = [
                    Vec(buffer_ops.buffer_load(sr, row_dw + 2 * jj, vec_width=2, dtype=fx.Int32, mask=okr))
                    for jj in range_constexpr(3)
                ]
                if p == 0:
                    if in_b:
                        sorted_ids[row0 + tx] = val  # padding: token n_tokens, slot 0
                        sorted_w[row0 + tx] = wgt  # padding: 0 (masked load)
                for jj in range_constexpr(3):
                    for dd in range_constexpr(2):
                        _lds_store_dw(fx.Int32(wv[jj][dd]), st, i * LDS_STRIDE + (kk4 + 2 * jj + dd) * 4)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                tile_dw = ((row0 + tile_r) >> 5) * (TILE_B // 4)
                for jj in range_constexpr(6):
                    dw = lt + 256 * jj
                    y_blk = dw >> 6
                    t63 = dw & 63
                    xl = t63 & 15
                    c = t63 >> 4
                    col_a = y_blk * 8 + c
                    q0 = _lds_u8(st, xl * LDS_STRIDE + col_a)
                    q1 = _lds_u8(st, (xl + 16) * LDS_STRIDE + col_a)
                    q2 = _lds_u8(st, xl * LDS_STRIDE + col_a + 4)
                    q3 = _lds_u8(st, (xl + 16) * LDS_STRIDE + col_a + 4)
                    packed = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
                    buffer_ops.buffer_store(packed, or_, tile_dw + y_blk * 64 + c * 16 + xl)
                fx.gpu.barrier()

    @flyc.jit
    def launch_sort_scales_all_mxfp4_i768(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_w: fx.Tensor,
        tile_map: fx.Tensor,
        scale: fx.Tensor,
        out: fx.Tensor,
        n_tokens: fx.Int32,
        grid_entries: fx.Int32,
        i32_ctas: fx.Int32,
        stream: fx.Stream,
    ):
        sort_scales_all(
            topk_ids, topk_w, num_valid, sorted_expert_ids, sorted_ids, sorted_w, tile_map,
            scale, out, n_tokens, grid_entries, i32_ctas,
        ).launch(grid=(fx.Int64(i32_ctas), 1, 1), block=(THREADS, 1, 1), stream=stream)  # fmt: skip

    assert BPC_MAX >= 1
    return launch_sort_scales_all_mxfp4_i768


def moe_sort_scales_all(bufs, topk_ids, topk_weights, scale, a_s, tile_map, grid_entries, n_tokens, *, E, I):  # noqa: E741
    """The routing sort into ``bufs`` (a ``SortBuffers``), GEMM1's tile map and the sorted scale tiles ``a_s`` (rows <
    num_valid) in one launch, up to ``MAX_TOKENS``."""
    import torch
    from .launch import _run_compiled

    # the cached CompiledFunction, not a direct jit call (host cost)
    _run_compiled(
        compile_moe_sort_scales_all(
            E=E, topk=topk_ids.shape[1], block_m=bufs.block_m, I=I, tpt=tpt_for(n_tokens)
        ),
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)[:6],
        tile_map.view(-1),
        scale.view(torch.uint8).view(-1),
        a_s.view(torch.uint8).view(-1),
        int(n_tokens),
        int(grid_entries),
        max(SORT_CTAS, bufs.max_sorted // bufs.block_m),  # a CTA per possible block
        torch.cuda.current_stream(),
    )  # fmt: skip
