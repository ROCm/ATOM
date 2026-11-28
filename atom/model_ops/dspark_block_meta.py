# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A DSpark draft block's paged metadata in ONE launch (per-row decode, no DCP).

What `DSparkProposer._build_paged_block_metadata` and the draft model's
`prepare_paged_block` build with ~30 small torch launches -- the block
positions, the KV write slots (dropped past the block table, on a pad row and
past the allocated pages), the context lengths, each request's page-16 table and
each draft row's sliding-window table and length -- as one Triton kernel.
Integer arithmetic only: the same values as those ops, element for element.

One program per draft row (request r, block position t) and chunk of its window
table. A window table is a P-wide window of the flat page-16 tables starting
inside its own request's table; this step's request rows of that buffer are
derived from the block table directly (the t == 0 programs also store them), and
rows past the batch are read as stored, as the torch ops read them.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
# Up to this many requests the pages each holds go in as two kernel scalars.
PACKED_MAX_ROWS = 8


@triton.jit(
    do_not_specialize=[
        "scheduled_bs",
        "running_bs",
        "pad_ctx_len",
        "alloc_lo",
        "alloc_hi",
    ]
)
def _block_meta_kernel(
    anchor_ptr,  # [rb] int64
    offsets_ptr,  # [T] int64
    bt_ptr,  # [rb, W] int32 block table, row stride bt_stride
    bt_stride,
    alloc_ptr,  # [rb] int32: pages each request holds (not PACKED)
    alloc_lo,  # PACKED: requests 0-3's pages, 16 bits each
    alloc_hi,  # PACKED: requests 4-7's
    positions_ptr,  # [rb, T] int64 out
    slots_ptr,  # [rb * T] int64 out
    ctx_lens_ptr,  # [rb] int32 out
    ctx_eff_ptr,  # [rb] int32 out
    bt16_flat_ptr,  # [(max_bs + 1) * P] int32: rows < rb out, the rest read
    rows_tbl_ptr,  # [rb * T, P] int32 out
    rows_len_ptr,  # [rb * T] int32 out
    T,
    W,
    P,
    entry_tokens,
    window,
    scheduled_bs,
    running_bs,
    pad_ctx_len,
    PACKED: tl.constexpr,
    SUB: tl.constexpr,  # sub-pages per block-table entry
    PAGE: tl.constexpr,  # tokens per sub-page
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)  # draft row
    chunk = tl.program_id(1)
    r = i // T
    t = i % T
    anchor = tl.load(anchor_ptr + r)
    pos = anchor + tl.load(offsets_ptr + t)
    if PACKED:
        # int64 whatever width the launcher gave the scalars (they fit 32 bits
        # when the upper requests' fields are 0)
        word = tl.where(r < 4, alloc_lo.to(tl.int64), alloc_hi.to(tl.int64))
        alloc = ((word >> ((r % 4) * 16)) & 0xFFFF) * entry_tokens
    else:
        alloc = tl.load(alloc_ptr + r).to(tl.int64) * entry_tokens
    pad = r >= scheduled_bs
    ctx_len = tl.where(pad, pad_ctx_len, anchor + 1 + T)
    ctx_eff = tl.minimum(ctx_len, alloc)
    if chunk == 0:
        # Dropped (-1): past the table, a pad row, past the allocated pages.
        page = pos // entry_tokens
        page_c = tl.minimum(page, W - 1)
        bt_val = tl.load(bt_ptr + r * bt_stride + page_c).to(tl.int64)
        slot = bt_val * entry_tokens + pos % entry_tokens
        slot = tl.where((page < W) & (~pad) & (pos < alloc), slot, -1)
        tl.store(positions_ptr + i, pos)
        tl.store(slots_ptr + i, slot)
        if t == 0:
            tl.store(ctx_lens_ptr + r, ctx_len.to(tl.int32))
            tl.store(ctx_eff_ptr + r, ctx_eff.to(tl.int32))
    # The row's first sub-page under the window; a pad row's is 0.
    if window > 0:
        start = tl.maximum(pos - (window - 1), 0) // PAGE
        start = tl.where(pad, 0, start)
    else:
        start = pos * 0
    if chunk == 0:
        tl.store(rows_len_ptr + i, (ctx_eff - start * PAGE).to(tl.int32))
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    jm = j < P
    q = start + r * P + j  # flat index into the page-16 tables
    rq = q // P
    c = q % P
    fresh = rq < running_bs
    val = tl.load(bt_ptr + rq * bt_stride + c // SUB, mask=jm & fresh, other=0)
    val = val * SUB + (c % SUB)
    val = tl.where((rq >= scheduled_bs) & (c < SUB), c, val)  # pad rows: page 0
    stale = tl.load(bt16_flat_ptr + q, mask=jm & (~fresh), other=0)
    val = tl.where(fresh, val, stale)
    tl.store(rows_tbl_ptr + i * P + j, val.to(tl.int32), mask=jm)
    if t == 0:
        # This request's own page-16 table row.
        cc = chunk * BLOCK + tl.arange(0, BLOCK)
        cm = cc < P
        own = tl.load(bt_ptr + r * bt_stride + cc // SUB, mask=cm, other=0)
        own = own * SUB + (cc % SUB)
        own = tl.where(pad & (cc < SUB), cc, own)
        tl.store(bt16_flat_ptr + r * P + cc, own.to(tl.int32), mask=cm)


def block_meta(
    anchor_positions,  # [rb] int64
    offsets,  # [T] int64, 1..T
    block_tables,  # [rb, W] int32
    alloc_pages,  # [rb] int32 tensor, or a list of at most 8 page counts < 2**16
    positions_out,  # [rb, T] int64
    slots_out,  # [rb * T] int64
    ctx_lens_out,  # [rb] int32
    ctx_eff_out,  # [rb] int32
    bt16_flat,  # [(max_bs + 1) * P] int32
    rows_tbl_out,  # [rb * T, P] int32
    rows_len_out,  # [rb * T] int32
    *,
    entry_tokens: int,
    sub_pages: int,
    window_page: int,
    window: int,
    scheduled_bs: int,
    pad_ctx_len: int,
) -> None:
    rb, W = block_tables.shape
    T = offsets.shape[0]
    P = rows_tbl_out.shape[1]
    assert P == sub_pages * W and block_tables.stride(1) == 1
    assert bt16_flat.numel() >= (rb + 1) * P
    packed = not isinstance(alloc_pages, torch.Tensor)
    lo = hi = 0
    if packed:
        assert len(alloc_pages) == rb <= PACKED_MAX_ROWS
        for r, n in enumerate(alloc_pages):
            assert 0 <= n < 1 << 16
            if r < 4:
                lo |= n << (16 * r)
            else:
                hi |= n << (16 * (r - 4))
        # as int64 kernel arguments
        lo -= (lo >> 63) << 64
        hi -= (hi >> 63) << 64
    _block_meta_kernel[(rb * T, triton.cdiv(P, _BLOCK))](
        anchor_positions,
        offsets,
        block_tables,
        block_tables.stride(0),
        block_tables if packed else alloc_pages,
        lo,
        hi,
        positions_out,
        slots_out,
        ctx_lens_out,
        ctx_eff_out,
        bt16_flat,
        rows_tbl_out,
        rows_len_out,
        T,
        W,
        P,
        entry_tokens,
        window or 0,
        scheduled_bs,
        rb,
        pad_ctx_len,
        PACKED=packed,
        SUB=sub_pages,
        PAGE=window_page,
        BLOCK=_BLOCK,
    )
