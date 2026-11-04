# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-launch forms of the paged block draft's per-step addressing.

`draft_kv.draft_ctx_slots` and the unsharded branch of
`DSparkProposer._build_paged_block_metadata` are chains of eager integer torch
ops (searchsorted, floor division, compare, clamp, gather, multiply, remainder,
masked fill), one launch each. These kernels compute the same integers, element
for element; the torch chains stay as the CPU path and the reference. Kept out
of `draft_kv` so that module imports without triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _entry_slot(block_tables_row, pos, n_cols, vblock):
    """`draft_kv.entry_slots` for positions on one table row: -1 past the table."""
    # torch's floor division and remainder, whatever the sign of `pos`.
    page = pos // vblock
    off = pos - page * vblock
    neg = off < 0
    page = tl.where(neg, page - 1, page)
    off = tl.where(neg, off + vblock, off)
    in_table = page < n_cols
    page = tl.minimum(page, n_cols - 1)
    # Advanced indexing wraps a negative column.
    page = tl.where(page < 0, page + n_cols, page)
    entry = tl.load(block_tables_row + page)
    return tl.where(in_table, entry.to(tl.int64) * vblock + off, -1)


@triton.jit(do_not_specialize=["n", "rows"])
def _draft_ctx_slots_kernel(
    out,
    positions,
    cu_ends,
    block_tables,
    n,
    rows,
    n_cols,
    bt_stride,
    vblock,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ROWS_PAD: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = t < n
    # searchsorted(cu_ends, t, right=True) on non-decreasing ends is the count
    # of ends <= t.
    req = tl.zeros([BLOCK], dtype=tl.int32)
    for r0 in tl.static_range(0, ROWS_PAD, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        ends = tl.load(cu_ends + r, mask=r < rows, other=0x7FFFFFFF)
        req += tl.sum((ends[None, :] <= t[:, None]).to(tl.int32), axis=1)
    req = tl.minimum(req, rows - 1)
    pos = tl.load(positions + t, mask=mask, other=0).to(tl.int64)
    row = block_tables + req.to(tl.int64) * bt_stride
    tl.store(out + t, _entry_slot(row, pos, n_cols, vblock), mask=mask)


@triton.jit
def _block_addressing_kernel(
    block_positions,
    slots,
    ctx_lens,
    anchor_positions,
    block_tables,
    bt_stride,
    n_cols,
    scheduled_bs,
    pad_ctx_len,
    entry_tokens,
    T: tl.constexpr,
    T_PAD: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.arange(0, T_PAD)
    tm = t < T
    a = tl.load(anchor_positions + b).to(tl.int64)
    pos = a + 1 + t
    tl.store(block_positions + b * T + t, pos, mask=tm)
    row = block_tables + b.to(tl.int64) * bt_stride
    slot = _entry_slot(row, pos, n_cols, entry_tokens)
    pad = b >= scheduled_bs
    tl.store(slots + b * T + t, tl.where(pad, -1, slot), mask=tm)
    ctx = tl.minimum(a + 1 + T, n_cols * entry_tokens)
    tl.store(ctx_lens + b, tl.where(pad, pad_ctx_len, ctx).to(tl.int32))


def draft_ctx_slots_one_launch(
    out: torch.Tensor,  # [n] int64
    positions: torch.Tensor,  # [>= n]
    cu_ends: torch.Tensor,  # [rows] int32, non-decreasing exclusive request ends
    block_tables: torch.Tensor,  # [rows, n_cols] int32
    vblock: int,
) -> torch.Tensor:
    """`draft_kv.draft_ctx_slots` in one launch, into ``out``."""
    n = out.shape[0]
    if n == 0:
        return out
    assert block_tables.stride(1) == 1
    rows = cu_ends.shape[0]
    block, block_r = 256, 64
    _draft_ctx_slots_kernel[(triton.cdiv(n, block),)](
        out,
        positions,
        cu_ends,
        block_tables,
        n,
        rows,
        block_tables.shape[1],
        block_tables.stride(0),
        vblock,
        BLOCK=block,
        BLOCK_R=block_r,
        # A power of two, so the batch sizes of a run share a few compiles.
        ROWS_PAD=max(block_r, triton.next_power_of_2(rows)),
        num_warps=4,
    )
    return out


def block_addressing_one_launch(
    block_positions: torch.Tensor,  # [running_bs, T] int64
    slots: torch.Tensor,  # [running_bs, T] int64
    ctx_lens: torch.Tensor,  # [running_bs] int32
    anchor_positions: torch.Tensor,  # [running_bs]
    block_tables: torch.Tensor,  # [running_bs, n_cols] int32
    scheduled_bs: int,
    pad_ctx_len: int,
    entry_tokens: int,
) -> None:
    """The unsharded block addressing in one launch.

    Per request ``b``: positions ``anchor + 1 .. anchor + T``; each one's slot
    through ``block_tables[b]`` (-1 past the table); a KV length of
    ``anchor + 1 + T`` capped at the table's span. Rows from ``scheduled_bs`` on
    get slot -1 and length ``pad_ctx_len``.
    """
    running_bs, t = block_positions.shape
    assert block_positions.is_contiguous() and slots.is_contiguous()
    assert anchor_positions.stride(0) == 1 and block_tables.stride(1) == 1
    _block_addressing_kernel[(running_bs,)](
        block_positions,
        slots,
        ctx_lens,
        anchor_positions,
        block_tables,
        block_tables.stride(0),
        block_tables.shape[1],
        scheduled_bs,
        pad_ctx_len,
        entry_tokens,
        T=t,
        T_PAD=triton.next_power_of_2(t),
        num_warps=1,
    )
