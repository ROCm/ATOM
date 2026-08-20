# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fill a decode step's `input_ids` for the sequences carried over on GPU.

A decode step feeds each request `[anchor, draft_0 .. draft_{n-1}]`, where the
anchor is the id that request sampled last step. That id lives in one of two
places:

* a request the previous forward also held ("deferred") sampled it on the GPU
  and it was never copied to the host -- that is the whole point of deferred
  output, which trades a D2H sync for reading `prev_token_ids` in place;
* a request the scheduler admitted since then has its id on the host, already
  written into `batch.scheduled_tokens` at that request's own token offset.

The caller stages `scheduled_tokens` over the whole decode region first, which
is correct by construction for the second kind, then calls this to overwrite
the spans belonging to the first.

Addressing: one program per request, writing `[cu[i], cu[i+1])`. The span comes
from the same `cu` the rest of the step uses, so per-request length is whatever
that says -- uniform `q` for a rectangular MTP step, per-request lengths under
DSpark's ragged buckets, no separate copy of "how long is each one" to drift.

The predecessor of this kernel placed the two kinds as two contiguous BLOCKS
(`[deferred | new]`, or `[new | deferred]` when the previous forward was a
prefill). The scheduler does not order the batch that way -- the two interleave
-- so every request could receive a neighbour's anchor. It went unseen because
the block layout only comes up when a forward lands between two decode steps
and displaces the previous batch, which is what a chunked prefill does. Writing
each request's own span keeps that unexpressible.
"""

import torch
import triton
import triton.language as tl

from atom.utils.decorators import mark_trace

NEW_SEQUENCE = -1
"""`src[i]` for a request whose anchor is already staged from the host.

Any negative value means the same thing; the kernel tests `src < 0` because
Triton cannot read a module-level global from inside a `@jit`ed function, and a
`constexpr` copy would be a second place for the sentinel to be defined."""


@triton.jit
def _fill_deferred_decode_ids_kernel(
    out_ptr,  # [>=cu[bs]] int32 — staged with the scheduler's ids
    cu_ptr,  # [bs+1] int32 — exclusive prefix sum of per-request token counts
    src_ptr,  # [bs] int32 — row in prev/draft, or NEW_SEQUENCE
    prev_ptr,  # [num_prev] — last step's sampled id per row
    draft_ptr,  # [num_prev, k] — last step's drafts per row
    draft_row_stride,
    HAS_DRAFT: tl.constexpr,
    BLOCK_Q: tl.constexpr,  # >= the longest per-request token count
):
    """One program per request; writes only that request's own span."""
    i = tl.program_id(0)
    src = tl.load(src_ptr + i)
    if src < 0:  # NEW_SEQUENCE — the host already staged this request's span
        return

    start = tl.load(cu_ptr + i)
    end = tl.load(cu_ptr + i + 1)
    col = tl.arange(0, BLOCK_Q)
    live = col < (end - start)

    val = tl.load(prev_ptr + src)
    if HAS_DRAFT:
        # Column `col - 1` of the draft row; anchors (col == 0) would index -1,
        # so mask them out here and pick the anchor back with `tl.where`.
        drafts = tl.load(
            draft_ptr + src * draft_row_stride + (col - 1),
            mask=live & (col > 0),
            other=0,
        )
        val = tl.where(col == 0, val, drafts)
    else:
        val = tl.broadcast_to(val, [BLOCK_Q])

    tl.store(out_ptr + start + col, val, mask=live)


@mark_trace(prefix="decode_input_ids")
def fill_deferred_decode_ids(
    out: torch.Tensor,
    cu: torch.Tensor,
    src: torch.Tensor,
    prev_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    *,
    max_tokens_per_seq: int,
) -> None:
    """Overwrite the deferred requests' spans of `out`, in place.

    Args:
      out:             `[>= cu[bs]]` int32 — the decode region of `input_ids`,
                       already staged from `batch.scheduled_tokens`.
      cu:              `[bs+1]` int32 — exclusive prefix sum of per-request
                       token counts. The ONLY statement of how long each
                       request's span is.
      src:             `[bs]` int32 — row into `prev_token_ids` /
                       `draft_token_ids`, or `NEW_SEQUENCE` to leave the span
                       as staged.
      prev_token_ids:  `[num_prev]` — previous forward's sampled ids.
      draft_token_ids: `[num_prev, k]`, or None when this step has no drafts
                       (then every span must be one token long).
      max_tokens_per_seq: an upper bound on `cu[i+1] - cu[i]`; sizes the
                       kernel's column vector.
    """
    bs = int(src.shape[0])
    if bs == 0:
        return
    assert cu.shape[0] == bs + 1, f"cu must be [bs+1]={bs + 1}, got {tuple(cu.shape)}"
    has_draft = draft_token_ids is not None
    assert has_draft or max_tokens_per_seq == 1, (
        "a step with no draft ids must feed exactly one token per request, "
        f"got max_tokens_per_seq={max_tokens_per_seq}"
    )
    _fill_deferred_decode_ids_kernel[(bs,)](
        out,
        cu,
        src,
        prev_token_ids,
        draft_token_ids if has_draft else prev_token_ids,
        draft_token_ids.stride(0) if has_draft else 0,
        HAS_DRAFT=has_draft,
        BLOCK_Q=triton.next_power_of_2(max(int(max_tokens_per_seq), 1)),
    )


def fill_deferred_decode_ids_reference(
    out: torch.Tensor,
    cu: torch.Tensor,
    src: torch.Tensor,
    prev_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor | None,
    *,
    max_tokens_per_seq: int,
) -> None:
    """Pure-torch twin of the kernel, runnable on CPU.

    Written as a plain loop over requests rather than a vectorised rework: the
    property under test is which request each written slot belongs to, and a
    loop states that directly. Tests assert PROPERTIES against this (non-uniform
    spans, interleaved kinds, coverage), not merely that the kernel agrees with
    it -- two implementations of the same wrong addressing agree perfectly.
    """
    del max_tokens_per_seq  # only the kernel needs a static column bound
    cu_l = cu.tolist()
    for i, row in enumerate(src.tolist()):
        if row < 0:  # NEW_SEQUENCE, same test the kernel makes
            continue
        start, end = cu_l[i], cu_l[i + 1]
        if end <= start:
            continue
        out[start] = prev_token_ids[row]
        if end - start > 1:
            assert draft_token_ids is not None, (
                f"request {i} wants {end - start} tokens but this step has no "
                f"draft ids"
            )
            out[start + 1 : end] = draft_token_ids[row, : end - start - 1]
