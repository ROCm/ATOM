# SPDX-License-Identifier: MIT
"""Per-token tile ids, read straight out of the PAGE table.

The scorer wants one row per query token naming every tile its request owns,
which is that request's PAGE table with each entry expanded in place. V4 does
the same expansion for its HCA section inside the kernel that consumes it
(`paged_decode_indices`); the FP8 scorer here is AITER's and takes a
materialized table, so the table exists -- but nothing between the PAGE table
and it has to.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _unit_table_kernel(
    block_tables,
    batch_ids,
    out,
    columns,
    table_stride,
    batch_stride,
    out_stride,
    UNITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token per `BLOCK` tile ids.

    A padding token carries batch id -1 and gets zeros: the load is masked, so
    its PAGE table is never read, and the scorer bounds that row before it
    reaches a tile anyway.
    """
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token * batch_stride)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < columns * UNITS
    live = batch >= 0
    page = tl.load(
        block_tables + batch * table_stride + offsets // UNITS,
        mask=mask & live,
        other=0,
    )
    # Zero the whole id, not just the PAGE it came from: masking the load
    # alone leaves the tile index, which names a real tile of PAGE 0.
    tile = tl.where(live, page * UNITS + offsets % UNITS, 0)
    tl.store(out + token * out_stride + offsets, tile, mask=mask)


def unit_table(block_tables, batch_ids, units_per_page):
    """`[tokens, columns * units_per_page]` int32 tile ids.

    A PAGE holds `units_per_page` consecutive tiles, so entry `(t, c, u)` is
    tile `u` of the PAGE that request's column `c` names -- the translation
    the plane's region-major layout buys, and the reason a block id is not a
    PAGE id. int32 throughout, which is the width the scorer takes tile ids at
    and so the width `num_pages * units_per_page` already has to fit.

    `batch_ids` is read by its own stride rather than assumed contiguous: the
    reference below indexes with torch and would take any view, so a kernel
    that assumed one could not be told apart by comparing the two.
    """
    tokens, columns = batch_ids.numel(), block_tables.shape[1]
    out = torch.empty(
        tokens, columns * units_per_page, dtype=torch.int32, device=block_tables.device
    )
    if not tokens:
        return out
    block = min(1024, triton.next_power_of_2(columns * units_per_page))
    _unit_table_kernel[(tokens, triton.cdiv(columns * units_per_page, block))](
        block_tables,
        batch_ids,
        out,
        columns,
        block_tables.stride(0),
        batch_ids.stride(0),
        out.stride(0),
        UNITS=units_per_page,
        BLOCK=block,
    )
    return out
