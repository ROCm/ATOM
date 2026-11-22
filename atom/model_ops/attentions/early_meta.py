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
    # ATOM_HOSTNEXT_FUSED_PROLOGUE: the deferred launches' arguments
    fused: bool = False
    attn: dict | None = None
    gdn: dict | None = None
    kv: tuple | None = None


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


# --------------------------------------------------------------------------- #
# ATOM_HOSTNEXT_FUSED_PROLOGUE: the step's pre-verify small kernels in one      #
# --------------------------------------------------------------------------- #
# With ATOM_HOSTNEXT_EARLY_META the builders queue, between the last draft pass
# and the verify graph: the early fix, the GDN accepted counts, the spec mask
# fill, the ReplaySSM commit and the GDN spec-decode pad (five launches), then
# kv_indices. They only feed the verify graph, so they can all wait for the end
# of the GDN build (after its state-index uploads) and run as one program, then
# kv_indices. Same values: each part is the kernel it replaces, in its order.


@triton.jit(
    do_not_specialize=[
        "bs", "running_bs", "bt_stride", "mrope_width", "mask_len", "commit_n",
        "T_MAX", "CAP", "ns_len", "sp_rows", "sp_stride", "qsl_len", "acc_len",
    ]
)
def _early_prologue_kernel(
    src_ptr, prev_rej_ptr, prev_bonus_ptr,
    ctx_ptr, pos_ptr, slot_ptr, bt_ptr, indptr_ptr, tiles_ptr, mrope_ptr,
    nat_ptr, mask_ptr, wpos_ptr, sidx_ptr, ns_idx_ptr, sp_idx_ptr, qsl_dst_ptr, qsl_src_ptr,
    bs, running_bs, bt_stride, mrope_width, mask_len, commit_n, T_MAX, CAP,
    ns_len, sp_rows, sp_stride, qsl_len, acc_len,
    Q: tl.constexpr, BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr, INDPTR_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, WRITE_POS: tl.constexpr, HAS_MROPE: tl.constexpr,
    HAS_TILES: tl.constexpr, TILE: tl.constexpr, HEADS: tl.constexpr, PROGRAMS: tl.constexpr,
    TILES_LO: tl.constexpr, TILES_HI: tl.constexpr,
    HAS_COMMIT: tl.constexpr, HAS_NS: tl.constexpr, PAD: tl.constexpr, SP_COLS: tl.constexpr,
    PBLOCK: tl.constexpr,
):
    # ---- loads (every load before any store) ----
    rows = tl.arange(0, BLOCK_ROWS)
    real = rows < bs
    src = tl.load(src_ptr + rows, mask=real, other=-1)
    carried = real & (src >= 0)
    rej = tl.load(prev_rej_ptr + src, mask=carried, other=0)
    bonus = tl.load(prev_bonus_ptr + src, mask=carried, other=0)
    ctx = tl.load(ctx_ptr + rows, mask=real, other=0) - rej
    nat_host = tl.load(nat_ptr + rows, mask=real, other=1)
    nat = tl.where(carried, bonus + 1, nat_host)
    tok = tl.arange(0, BLOCK)
    row = tok // Q
    j = tok % Q
    treal = row < bs
    trow_src = tl.load(src_ptr + row, mask=treal, other=-1)
    trej = tl.load(prev_rej_ptr + trow_src, mask=treal & (trow_src >= 0), other=0)
    tctx = tl.load(ctx_ptr + row, mask=treal, other=0) - trej
    pos = (tctx - Q + j).to(tl.int64)
    blk = tl.load(bt_ptr + row.to(tl.int64) * bt_stride + pos // BLOCK_SIZE, mask=treal, other=0)
    slot = blk.to(tl.int64) * BLOCK_SIZE + pos % BLOCK_SIZE
    if HAS_MROPE:
        m0 = tl.load(mrope_ptr + tok, mask=treal, other=0)
        m1 = tl.load(mrope_ptr + mrope_width + tok, mask=treal, other=0)
        m2 = tl.load(mrope_ptr + 2 * mrope_width + tok, mask=treal, other=0)
    # ReplaySSM commit inputs (rows < commit_n read the accepted counts above)
    if HAS_COMMIT:
        crow = rows < commit_n
        cslot = tl.load(sidx_ptr + rows, mask=crow, other=-1).to(tl.int64)
        clive = crow & (cslot >= 0)
        h = tl.load(wpos_ptr + cslot, mask=clive, other=0)
    tl.debug_barrier()
    # ---- early fix (as _early_meta_fix_kernel) ----
    tl.store(ctx_ptr + rows, ctx, mask=real)
    nblk = tl.where(real, (ctx + INDPTR_BLOCK - 1) // INDPTR_BLOCK, 0)
    incl = tl.cumsum(nblk, axis=0)
    total = tl.sum(nblk, axis=0)
    tl.store(indptr_ptr + 1 + rows, incl, mask=real)
    tl.store(indptr_ptr + 1 + rows, tl.zeros_like(incl) + total, mask=(rows >= bs) & (rows < running_bs))
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
    # ---- GDN accepted counts (as _early_gdn_accepted_kernel) ----
    tl.store(nat_ptr + rows, nat, mask=carried)
    # ---- spec_sequence_masks[:bs].fill_(True) ----
    tl.store(mask_ptr + rows, tl.full([BLOCK_ROWS], 1, mask_ptr.dtype.element_ty), mask=real & (rows < mask_len))
    # ---- replayssm_commit ----
    if HAS_COMMIT:
        flushed = h + 2 * T_MAX > CAP
        base = tl.where(flushed, 0, h)
        newh = tl.where(h < 0, 0, base + nat.to(h.dtype))
        tl.store(wpos_ptr + cslot, newh, mask=clive)
    # ---- _gdn_spec_decode_pad (nd = bs) ----
    nd = bs
    last = tl.load(qsl_src_ptr + nd)
    for start in range(0, qsl_len, PBLOCK):
        jj = start + tl.arange(0, PBLOCK)
        v = tl.load(qsl_src_ptr + jj, mask=jj <= nd, other=0)
        tl.store(qsl_dst_ptr + jj, tl.where(jj <= nd, v, last), mask=jj < qsl_len)
    for start in range(nd, acc_len, PBLOCK):
        jj = start + tl.arange(0, PBLOCK)
        tl.store(nat_ptr + jj, tl.full([PBLOCK], 1, nat_ptr.dtype.element_ty), mask=jj < acc_len)
    for start in range(nd, mask_len, PBLOCK):
        jj = start + tl.arange(0, PBLOCK)
        tl.store(mask_ptr + jj, tl.zeros([PBLOCK], mask_ptr.dtype.element_ty), mask=jj < mask_len)
    if HAS_NS:
        for start in range(nd, ns_len, PBLOCK):
            jj = start + tl.arange(0, PBLOCK)
            tl.store(ns_idx_ptr + jj, tl.full([PBLOCK], PAD, ns_idx_ptr.dtype.element_ty), mask=jj < ns_len)
    c = tl.arange(0, SP_COLS)
    for r in range(nd, sp_rows):
        tl.store(sp_idx_ptr + r.to(tl.int64) * sp_stride + c, tl.full([SP_COLS], PAD, sp_idx_ptr.dtype.element_ty))


def early_prologue(fix: EarlyFix) -> None:
    """Launch the deferred prologue (`fix.attn` from the MHA builder, `fix.gdn`
    from the GDN one) and then kv_indices (`fix.kv`)."""
    from atom.utils.block_convert import kv_indices_generate_triton

    a, g = fix.attn, fix.gdn
    bs = fix.bs
    block_rows = max(16, triton.next_power_of_2(max(a["running_bs"], bs, g["commit_n"])))
    block = max(16, triton.next_power_of_2(bs * a["max_seqlen_q"]))
    split = a["split_constants"]
    has_tiles = a["tiles"] is not None and split is not None
    tile, heads, programs, lo, hi = split if has_tiles else (1, 1, 1, 0, 0)
    dummy = a["context_lens"]
    commit = g["commit_n"] > 0
    ns = g["ns_idx"]
    sp = g["sp_idx"]
    cols = sp.shape[1]
    assert sp.dim() == 2 and sp.stride(1) == 1 and cols == triton.next_power_of_2(cols)
    _early_prologue_kernel[(1,)](
        fix.src, fix.prev_rejected, fix.prev_bonus,
        a["context_lens"], a["positions"] if a["positions"] is not None else dummy,
        a["slot_mapping"], a["block_tables"], a["kv_indptr"],
        a["tiles"] if has_tiles else dummy, a["mrope"] if a["mrope"] is not None else dummy,
        g["nat"], g["masks"], g["write_pos"] if commit else dummy, g["slot_idx"] if commit else dummy,
        ns if ns is not None else dummy, sp, g["qsl_dst"], g["qsl_src"],
        bs, a["running_bs"], a["block_tables"].stride(0), a["mrope_width"], g["masks"].shape[0],
        g["commit_n"], g["t_max"], g["cap"],
        ns.shape[0] if ns is not None else 0, sp.shape[0], sp.stride(0), g["qsl_dst"].shape[0],
        g["nat"].shape[0],
        Q=a["max_seqlen_q"], BLOCK=block, BLOCK_SIZE=a["block_size"], INDPTR_BLOCK=a["indptr_block"],
        BLOCK_ROWS=block_rows, WRITE_POS=a["positions"] is not None, HAS_MROPE=a["mrope"] is not None,
        HAS_TILES=has_tiles, TILE=tile, HEADS=heads, PROGRAMS=programs, TILES_LO=lo, TILES_HI=hi,
        HAS_COMMIT=commit, HAS_NS=ns is not None, PAD=g["pad"], SP_COLS=cols, PBLOCK=256,
        num_warps=4,
    )
    kv_indices_generate_triton(*fix.kv)
    fix.applied_attn = True
    fix.applied_gdn = True
