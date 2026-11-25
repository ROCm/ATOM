# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Device verdict: a deferred speculative decode step's attention metadata,
shifted on the device by the previous step's rejected-draft counts.

A deferred step's scheduler lays every carried request out at its full
reservation: the previous verify window's `mtp_k + 1` slots all still counted
in `context_lens`. The host waits for that window's verdict
(`recv_spec_status`) and subtracts each request's rejected count before writing
the metadata (`tokenIDProcessor.verify_ctx_lens`). A step that consumes the
device verdict (`AttentionMetadataBuilder.consumes_device_verdict`) writes the
metadata at the reservation instead -- no wait -- and this one launch, after
the metadata's upload and before anything reads it, subtracts the count the
previous step's verdict left on the device:

* ``context_lens[i] -= r``, ``r`` the rejected count of request ``i``'s row in
  the previous batch (``decode_src[i]``, -1 for a request with no such row);
* every query token's ``positions[j] -= r`` and its ``slot_mapping[j]``
  re-derived from the block table: ``table[i, pos // page] * page + pos % page``;
* ``kv_indptr`` rebuilt from the shifted lengths, ``cdiv(len, kv_block)``
  prefix sums, flat over the pad rows.

That is everything a decode step derives from `verify_ctx_lens` on the host
(`AiterAttentionMetadataBuilder.stage_decode`); the GPU work that reads them
(`finish_decode`: KV indices, the FlyDSL plan, the sparse tables) runs after.
`max_seqlen_k` stays the reservation's, an upper bound, which a captured decode
graph does not read anyway (its grids were sized at capture).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_shift_kernel(
    decode_src_ptr,
    rejected_ptr,
    cu_q_ptr,
    block_tables_ptr,
    bt_stride,
    context_lens_ptr,
    positions_ptr,
    slot_mapping_ptr,
    kv_indptr_ptr,
    bs,
    running_bs,
    page_size,
    kv_block_size,
    BLOCK_B: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    # kv_indptr first, from the lengths as uploaded: a running sum over the
    # requests (a decode batch; also what the Triton interpreter can run).
    acc = 0
    for i in range(bs):
        s = tl.load(decode_src_ptr + i)
        r_i = tl.where(s >= 0, tl.load(rejected_ptr + tl.maximum(s, 0)), 0)
        c_i = tl.load(context_lens_ptr + i) - r_i.to(tl.int32)
        acc += (c_i + kv_block_size - 1) // kv_block_size
        tl.store(kv_indptr_ptr + 1 + i, acc)
    for i in range(bs, running_bs):
        tl.store(kv_indptr_ptr + 1 + i, acc)
    # Every lane has read the uploaded lengths before any lane rewrites them.
    tl.debug_barrier()

    rows = tl.arange(0, BLOCK_B)
    live = rows < bs
    src = tl.load(decode_src_ptr + rows, mask=live, other=-1)
    carried = live & (src >= 0)
    r = tl.load(rejected_ptr + src, mask=carried, other=0).to(tl.int32)
    ctx = tl.load(context_lens_ptr + rows, mask=live, other=0) - r
    tl.store(context_lens_ptr + rows, ctx, mask=live)

    q0 = tl.load(cu_q_ptr + rows, mask=live, other=0)
    q1 = tl.load(cu_q_ptr + rows + 1, mask=live, other=0)
    tok = q0[:, None] + tl.arange(0, BLOCK_Q)[None, :]
    tmask = (carried & (r > 0))[:, None] & (tok < q1[:, None])
    pos = tl.load(positions_ptr + tok, mask=tmask, other=0) - r[:, None].to(tl.int64)
    tl.store(positions_ptr + tok, pos, mask=tmask)
    blk = tl.load(
        block_tables_ptr + rows[:, None].to(tl.int64) * bt_stride + pos // page_size,
        mask=tmask,
        other=0,
    )
    slot = blk.to(tl.int64) * page_size + pos % page_size
    tl.store(slot_mapping_ptr + tok, slot, mask=tmask)


def shift_decode_metadata(
    *,
    decode_src: torch.Tensor,  # [>= bs] int32, previous-batch row or -1
    rejected: torch.Tensor,  # [prev bs] the previous verdict's rejected counts
    cu_seqlens_q: torch.Tensor,  # [>= bs + 1] int32
    block_tables: torch.Tensor,  # [>= bs, W] int32, page ids at `page_size`
    context_lens: torch.Tensor,  # [>= bs] int32, in place
    positions: torch.Tensor,  # [tokens] int64, in place
    slot_mapping: torch.Tensor,  # [tokens] int64, in place
    kv_indptr: torch.Tensor,  # [>= running_bs + 1] int32, in place
    bs: int,
    running_bs: int,
    max_q: int,
    page_size: int,
    kv_block_size: int,
) -> None:
    if bs == 0:
        return
    assert block_tables.stride(1) == 1 and rejected.is_contiguous()
    _decode_shift_kernel[(1,)](
        decode_src,
        rejected,
        cu_seqlens_q,
        block_tables,
        block_tables.stride(0),
        context_lens,
        positions,
        slot_mapping,
        kv_indptr,
        bs,
        running_bs,
        page_size,
        kv_block_size,
        BLOCK_B=max(triton.next_power_of_2(running_bs), 2),
        BLOCK_Q=max(triton.next_power_of_2(max_q), 2),
        num_warps=1,
    )
