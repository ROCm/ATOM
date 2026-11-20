# SPDX-License-Identifier: MIT
"""ATOM_HOSTNEXT_EARLY_META: build a speculative decode step's attention metadata
before the previous step's verify status reaches the host.

A decode step that carries sequences over needs the previous verify's rejected
counts: they shorten each carried-over sequence's context, and with it its
positions, KV write slots, CSR indptr and token-split tiles, and they set the
GDN accepted counts (bonus + 1). Without this flag the host waits for that
status (`recv_spec_status`) and only then builds the metadata, which puts the
whole build (~320 us at c64 rank batch sizes) between the wait and the verify
launch, while the GPU runs only the ~350 us of draft passes in between.

With it the host builds everything as if every carried-over row had rejected
nothing (an upper bound: same blocks, same table), and one small kernel, queued
behind the previous step's verify epilogue that produced the status on the
device, subtracts each row's rejections where the host would have: context
lens, positions (and the M-RoPE rows), write slots from the packed block table,
the CSR indptr and the token-split tiles. A second one writes the GDN accepted
counts. The status is then taken on the host only after the verify graph is
launched (`take_deferred_spec_status`), for the scheduler's output.

The integers come out exactly what the host computes after the wait, so the
step runs the same metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class EarlyFix:
    """What the previous step left on the device for this step's fix."""

    src: torch.Tensor  # int32 [bs]: previous row of each row, < 0 for a new one
    prev_bonus: torch.Tensor  # int32 [prev_bs] (verdict row 0)
    prev_rejected: torch.Tensor  # int32 [prev_bs] (verdict row 1)
    bs: int
    applied_attn: bool = False
    applied_gdn: bool = False


@triton.jit(do_not_specialize=["bs", "running_bs", "bt_stride", "mrope_width"])
def _early_meta_fix_kernel(
    src_ptr,
    prev_rej_ptr,
    ctx_ptr,
    pos_ptr,
    slot_ptr,
    bt_ptr,
    indptr_ptr,
    tiles_ptr,
    mrope_ptr,
    bs,
    running_bs,
    bt_stride,
    mrope_width,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDPTR_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    WRITE_POS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    HAS_TILES: tl.constexpr,
    TILE: tl.constexpr,
    HEADS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    TILES_LO: tl.constexpr,
    TILES_HI: tl.constexpr,
):
    # Every load before any store: the row and token views read the same lengths.
    rows = tl.arange(0, BLOCK_ROWS)
    real = rows < bs
    src = tl.load(src_ptr + rows, mask=real, other=-1)
    rej = tl.load(prev_rej_ptr + src, mask=real & (src >= 0), other=0)
    ctx = tl.load(ctx_ptr + rows, mask=real, other=0) - rej
    tok = tl.arange(0, BLOCK)
    row = tok // Q
    j = tok % Q
    treal = row < bs
    trow_src = tl.load(src_ptr + row, mask=treal, other=-1)
    trej = tl.load(prev_rej_ptr + trow_src, mask=treal & (trow_src >= 0), other=0)
    tctx = tl.load(ctx_ptr + row, mask=treal, other=0) - trej
    # token axis: row i forwards positions [ctx_i - Q, ctx_i)
    pos = (tctx - Q + j).to(tl.int64)
    blk = tl.load(
        bt_ptr + row.to(tl.int64) * bt_stride + pos // BLOCK_SIZE, mask=treal, other=0
    )
    slot = blk.to(tl.int64) * BLOCK_SIZE + pos % BLOCK_SIZE
    if HAS_MROPE:
        m0 = tl.load(mrope_ptr + tok, mask=treal, other=0)
        m1 = tl.load(mrope_ptr + mrope_width + tok, mask=treal, other=0)
        m2 = tl.load(mrope_ptr + 2 * mrope_width + tok, mask=treal, other=0)
    tl.debug_barrier()
    tl.store(ctx_ptr + rows, ctx, mask=real)
    # CSR indptr over the scheduled rows; pad rows (to running_bs) repeat the total
    nblk = tl.where(real, (ctx + INDPTR_BLOCK - 1) // INDPTR_BLOCK, 0)
    incl = tl.cumsum(nblk, axis=0)
    total = tl.sum(nblk, axis=0)
    tl.store(indptr_ptr + 1 + rows, incl, mask=real)
    tl.store(
        indptr_ptr + 1 + rows,
        tl.zeros_like(incl) + total,
        mask=(rows >= bs) & (rows < running_bs),
    )
    if HAS_TILES:
        t = tl.where(real, (ctx + TILE - 1) // TILE, 0).to(tl.int64)
        want = (tl.sum(t, axis=0) * HEADS + PROGRAMS - 1) // PROGRAMS
        want = tl.minimum(tl.maximum(want, TILES_LO), TILES_HI)
        tl.store(tiles_ptr, want.to(tl.int32))
    if WRITE_POS:
        tl.store(pos_ptr + tok, pos, mask=treal)
    tl.store(slot_ptr + tok, slot, mask=treal)
    if HAS_MROPE:
        d = trej.to(tl.int64)
        tl.store(mrope_ptr + tok, m0 - d, mask=treal)
        tl.store(mrope_ptr + mrope_width + tok, m1 - d, mask=treal)
        tl.store(mrope_ptr + 2 * mrope_width + tok, m2 - d, mask=treal)


@triton.jit(do_not_specialize=["bs"])
def _early_gdn_accepted_kernel(
    src_ptr, prev_bonus_ptr, nat_ptr, bs, BLOCK_ROWS: tl.constexpr
):
    rows = tl.arange(0, BLOCK_ROWS)
    real = rows < bs
    src = tl.load(src_ptr + rows, mask=real, other=-1)
    carried = real & (src >= 0)
    bonus = tl.load(prev_bonus_ptr + src, mask=carried, other=0)
    tl.store(nat_ptr + rows, bonus + 1, mask=carried)


def early_meta_fix(
    fix: EarlyFix,
    *,
    context_lens: torch.Tensor,
    positions: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    kv_indptr: torch.Tensor,
    tiles: torch.Tensor | None,
    split_constants: tuple | None,
    mrope: torch.Tensor | None,
    mrope_width: int,
    running_bs: int,
    max_seqlen_q: int,
    block_size: int,
    indptr_block: int,
) -> None:
    bs = fix.bs
    block_rows = max(16, triton.next_power_of_2(max(running_bs, bs)))
    block = max(16, triton.next_power_of_2(bs * max_seqlen_q))
    has_tiles = tiles is not None and split_constants is not None
    tile, heads, programs, lo, hi = split_constants if has_tiles else (1, 1, 1, 0, 0)
    dummy = context_lens
    _early_meta_fix_kernel[(1,)](
        fix.src,
        fix.prev_rejected,
        context_lens,
        positions if positions is not None else dummy,
        slot_mapping,
        block_tables,
        kv_indptr,
        tiles if has_tiles else dummy,
        mrope if mrope is not None else dummy,
        bs,
        running_bs,
        block_tables.stride(0),
        mrope_width,
        Q=max_seqlen_q,
        BLOCK=block,
        BLOCK_SIZE=block_size,
        INDPTR_BLOCK=indptr_block,
        BLOCK_ROWS=block_rows,
        WRITE_POS=positions is not None,
        HAS_MROPE=mrope is not None,
        HAS_TILES=has_tiles,
        TILE=tile,
        HEADS=heads,
        PROGRAMS=programs,
        TILES_LO=lo,
        TILES_HI=hi,
        num_warps=4,
    )
    fix.applied_attn = True


def early_gdn_accepted(fix: EarlyFix, num_accepted_tokens: torch.Tensor) -> None:
    block_rows = max(16, triton.next_power_of_2(fix.bs))
    _early_gdn_accepted_kernel[(1,)](
        fix.src,
        fix.prev_bonus,
        num_accepted_tokens,
        fix.bs,
        BLOCK_ROWS=block_rows,
        num_warps=1,
    )
    fix.applied_gdn = True
