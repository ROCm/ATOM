# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An MTP decode's metadata, moved on the device to where the previous verify's
verdict leaves each request.

With deferred output, a verify step's host-built attention metadata depends on
the previous verify's verdict in one way only: a request carried over from it
starts `rejected` tokens before the scheduler's `context_lens`, which still
hold that verify's rejected drafts. Waiting for the counts on the host before
laying the step out is a wait on the previous forward. A builder that consumes
the verdict on the device (`AttentionMetadataBuilder.consumes_device_verdict`)
instead lays the step out as if nothing had been rejected -- the scheduler's
lengths, an upper bound -- and moves it here, after the publish and before any
reader, from the verdict the previous forward left on the device:

- `context_lens[i] -= rejected` and `kv_indptr` re-summed from them;
- each of the request's tokens at `context_lens[i] - length + j`, its position
  rewritten (or its MRoPE planes moved by `rejected`) and its KV slot read
  from the block table again.

A request admitted since then (`src < 0`) is already laid out from the host's
counts and keeps it. `accepted_from_verdict` does the same for a linear
attention's `num_accepted_tokens`.

Addressing: one program walks the request rows in blocks, so `kv_indptr`'s
prefix sum carries from one block to the next; the tokens of request `i` are
`[cu[i], cu[i+1])`, the same `cu` the rest of the step reads.
"""

import torch
import triton
import triton.language as tl

from atom.utils.decorators import mark_trace


@triton.jit(do_not_specialize=["bs", "running_bs", "table_stride", "mrope_stride"])
def _rebase_decode_kernel(
    rejected_ptr,  # [num_prev] int32: the previous verdict's rejected drafts
    src_ptr,  # [bs] int32: each request's row in that verdict, < 0 when new
    cu_ptr,  # [bs + 1] int32: cu_seqlens_q
    ctx_ptr,  # [running_bs] int32: context_lens, moved in place
    indptr_ptr,  # [running_bs + 1] int32: kv_indptr, re-summed
    slot_ptr,  # [tokens] int64: slot_mapping
    table_ptr,  # [rows, table_stride] int32: block_tables
    pos_ptr,  # [tokens] int64: positions (HAS_POS)
    mrope_ptr,  # [3, mrope_stride] int64: MRoPE positions (HAS_MROPE)
    bs,
    running_bs,
    table_stride,
    mrope_stride,
    KV_BLOCK: tl.constexpr,  # tokens per block-table entry
    PAGE: tl.constexpr,  # tokens per kv_indptr page
    HAS_POS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    BLOCK_R: tl.constexpr,  # request rows a pass of the loop takes
    BLOCK_Q: tl.constexpr,  # >= the longest request's query
):
    # One program, so `kv_indptr`'s prefix sum carries across the passes.
    carry = tl.sum(tl.zeros([BLOCK_R], dtype=tl.int32), axis=0)
    for base in range(0, running_bs, BLOCK_R):
        r = base + tl.arange(0, BLOCK_R)
        req = r < bs
        row = r < running_bs
        src = tl.load(src_ptr + r, mask=req, other=-1)
        carried = req & (src >= 0)
        rejected = tl.load(rejected_ptr + src, mask=carried, other=0)
        ctx = tl.load(ctx_ptr + r, mask=row, other=0) - rejected
        tl.store(ctx_ptr + r, ctx, mask=carried)
        # Rows past `bs` are zero long: they carry the last offset forward.
        pages = (ctx + PAGE - 1) // PAGE
        tl.store(indptr_ptr + 1 + r, carry + tl.cumsum(pages, axis=0), mask=row)
        carry += tl.sum(pages, axis=0)

        start = tl.load(cu_ptr + r, mask=req, other=0)
        length = tl.load(cu_ptr + r + 1, mask=req, other=0) - start
        first = (ctx - length).to(tl.int64)
        table_row = table_ptr + r.to(tl.int64) * table_stride
        for j in tl.static_range(BLOCK_Q):
            live = carried & (j < length)
            tok = start + j
            pos = first + j
            if HAS_POS:
                tl.store(pos_ptr + tok, pos, mask=live)
            if HAS_MROPE:
                for plane in tl.static_range(3):
                    at = mrope_ptr + plane * mrope_stride + tok
                    moved = tl.load(at, mask=live, other=0) - rejected
                    tl.store(at, moved, mask=live)
            block = tl.load(table_row + pos // KV_BLOCK, mask=live, other=0)
            slot = block.to(tl.int64) * KV_BLOCK + pos % KV_BLOCK
            tl.store(slot_ptr + tok, slot, mask=live)


REBASE_BLOCK = 128
"""Request rows a pass of the rebase takes. One value, so serving compiles the
kernel once rather than once per batch width."""


@mark_trace(prefix="decode_verdict")
def rebase_decode_on_verdict(
    rejected: torch.Tensor,
    src: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    running_bs: int,
    max_seqlen_q: int,
    kv_block_size: int,
    page_size: int,
    positions: torch.Tensor | None = None,
    mrope_positions: torch.Tensor | None = None,
) -> None:
    """Move a decode step laid out at the scheduler's lengths to the verdict's.

    Args:
      rejected:       `[num_prev]` int32, the previous verify's rejected drafts
                      per request, in that verify's order.
      src:            `[bs]` int32, each request's row in it (`decode_src`),
                      negative for a request laid out from the host's counts.
      cu_seqlens_q:   `[>= bs + 1]` int32.
      context_lens:   `[>= running_bs]` int32, moved in place.
      kv_indptr:      `[>= running_bs + 1]` int32, `[1, running_bs]` re-summed
                      in `page_size` pages.
      slot_mapping:   `[tokens]` int64, the moved tokens' slots rewritten.
      block_tables:   `[rows, cols]` int32, `kv_block_size` tokens an entry.
      positions:      `[tokens]` int64, rewritten for the moved tokens.
      mrope_positions: `[3, width]` int64, the moved tokens' planes shifted.
    """
    bs = int(src.shape[0])
    if bs == 0:
        return
    assert bs <= running_bs, f"bs={bs} > running_bs={running_bs}"
    has_pos = positions is not None
    has_mrope = mrope_positions is not None
    _rebase_decode_kernel[(1,)](
        rejected,
        src,
        cu_seqlens_q,
        context_lens,
        kv_indptr,
        slot_mapping,
        block_tables,
        positions if has_pos else slot_mapping,
        mrope_positions if has_mrope else slot_mapping,
        bs,
        int(running_bs),
        block_tables.stride(0),
        mrope_positions.stride(0) if has_mrope else 0,
        KV_BLOCK=int(kv_block_size),
        PAGE=int(page_size),
        HAS_POS=has_pos,
        HAS_MROPE=has_mrope,
        BLOCK_R=REBASE_BLOCK,
        BLOCK_Q=triton.next_power_of_2(max(int(max_seqlen_q), 1)),
    )


def rebase_decode_on_verdict_reference(
    rejected: torch.Tensor,
    src: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    running_bs: int,
    max_seqlen_q: int,
    kv_block_size: int,
    page_size: int,
    positions: torch.Tensor | None = None,
    mrope_positions: torch.Tensor | None = None,
) -> None:
    """Pure-torch twin of the kernel, runnable on CPU: a loop over requests,
    so whose tokens move by how much is stated directly."""
    del max_seqlen_q  # only the kernel needs a static column bound
    cu = cu_seqlens_q.tolist()
    for i, row in enumerate(src.tolist()):
        if row < 0:
            continue
        moved = int(rejected[row])
        context_lens[i] -= moved
        start, end = cu[i], cu[i + 1]
        for j, tok in enumerate(range(start, end)):
            pos = int(context_lens[i]) - (end - start) + j
            if positions is not None:
                positions[tok] = pos
            if mrope_positions is not None:
                mrope_positions[:, tok] -= moved
            block = int(block_tables[i, pos // kv_block_size])
            slot_mapping[tok] = block * kv_block_size + pos % kv_block_size
    pages = (context_lens[:running_bs].to(torch.int64) + page_size - 1) // page_size
    kv_indptr[1 : running_bs + 1] = torch.cumsum(pages, 0).to(kv_indptr.dtype)


@triton.jit
def _accepted_from_verdict_kernel(out_ptr, src_ptr, bonus_ptr, bs, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = i < bs
    src = tl.load(src_ptr + i, mask=req, other=-1)
    carried = req & (src >= 0)
    bonus = tl.load(bonus_ptr + src, mask=carried, other=0)
    tl.store(out_ptr + i, bonus + 1, mask=carried)


ACCEPTED_BLOCK = 256


@mark_trace(prefix="decode_verdict")
def accepted_from_verdict(
    num_accepted: torch.Tensor, src: torch.Tensor, bonus: torch.Tensor
) -> None:
    """`num_accepted[i] = bonus[src[i]] + 1` for every carried request (`src[i]
    >= 0`): the tokens the previous verify accepted, bonus included. The other
    rows keep what the host published."""
    bs = int(src.shape[0])
    if bs == 0:
        return
    _accepted_from_verdict_kernel[(triton.cdiv(bs, ACCEPTED_BLOCK),)](
        num_accepted, src, bonus, bs, BLOCK=ACCEPTED_BLOCK
    )


def accepted_from_verdict_reference(
    num_accepted: torch.Tensor, src: torch.Tensor, bonus: torch.Tensor
) -> None:
    """Pure-torch twin of `accepted_from_verdict`."""
    for i, row in enumerate(src.tolist()):
        if row >= 0:
            num_accepted[i] = int(bonus[row]) + 1
