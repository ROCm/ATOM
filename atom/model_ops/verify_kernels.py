# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Triton helpers every verify kernel shares: the draft token a row verifies,
and the bonus, -1 tail and verdict a request ends with. The exact-match,
synthetic and relaxed kernels (`rejection_sampler`), the ratio accept
(`ratio_verify`) and the verify from a head's ids (`_verify_by_ids`:
`verify_ids._verify_ids_kernel`, the V4.1 hi post-verify) all call them."""

import triton
import triton.language as tl


@triton.jit
def _draft_token(input_ids_ptr, target_logits_indices_ptr, r):
    """Draft r: the input token after the logits row that verifies it."""
    return tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1)


@triton.jit
def _finish_request(
    output_token_ids_ptr,
    verdict_ptr,
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    req_idx,
    num_draft_tokens,
    last_checked,
    rejected,
    num_spec_steps,
):
    """The bonus token, the -1 tail, and the request's verdict.

    `last_checked` is the last draft position compared (-1 before any). The
    anchor -- the last emitted token's row -- sits one row per accepted draft
    into the segment that ends at the bonus row.
    """
    INVALID_TOKEN: tl.constexpr = -1
    row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    num_accepted = last_checked
    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_accepted += 1
    tl.store(row + num_draft_tokens, bonus_token_id)
    # The -1 tail is what downstream first-`-1` truncation reads under
    # variable-length verification (the output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(row + pos, INVALID_TOKEN)
    batch_size = tl.num_programs(0)
    anchor = tl.load(bonus_logits_indices_ptr + req_idx) - num_draft_tokens
    tl.store(verdict_ptr + req_idx, num_accepted)
    tl.store(verdict_ptr + batch_size + req_idx, num_spec_steps - num_accepted)
    tl.store(verdict_ptr + 2 * batch_size + req_idx, anchor + num_accepted)


@triton.jit
def _verify_by_ids(
    out_ptr, verdict_ptr, cu_ptr, input_ids_ptr, target_idx_ptr, bonus_idx_ptr,
    head_ids_ptr, cond_rates_ptr, seed, b, num_spec_steps,
    SP: tl.constexpr, SYNTHETIC: tl.constexpr,
):  # fmt: skip
    """Program b, request b, from a head that picked each step row's id
    (``head_ids``, a row's target): its rows at once -- the first rejected a
    min over them -- then ``_finish_request``. ``SYNTHETIC`` (forced
    acceptance): a draft is kept, and emitted, when a uniform from ``seed``
    falls below its position's conditional rate. -> its anchor row."""
    if b == 0:
        start = 0
    else:
        start = tl.load(cu_ptr + b - 1)
    n = tl.load(cu_ptr + b) - start
    p = tl.arange(0, SP)
    live = p < n
    vrow = tl.load(target_idx_ptr + start + p, mask=live, other=0)
    target = tl.load(head_ids_ptr + vrow, mask=live, other=0).to(tl.int32)
    draft = tl.load(input_ids_ptr + vrow + 1, mask=live, other=0).to(tl.int32)
    if SYNTHETIC:
        u = tl.rand(seed, start + p)
        ok = u < tl.load(cond_rates_ptr + p, mask=live, other=0.0)
        kept = draft
    else:
        ok = draft == target
        kept = target
    first_rejected = tl.min(tl.where(live & (ok == 0), p, n), axis=0)
    rejected = first_rejected < n
    INVALID: tl.constexpr = -1
    row = tl.where(
        p < first_rejected, kept, tl.where(p == first_rejected, target, INVALID)
    )
    tl.store(out_ptr + b * (num_spec_steps + 1) + p, row, mask=live)
    brow = tl.load(bonus_idx_ptr + b)
    # ``_finish_request`` reads request b's bonus at its pointer + b: here
    # the head's id at the request's bonus row
    last_checked = tl.where(rejected, first_rejected, n - 1)
    _finish_request(
        out_ptr, verdict_ptr, head_ids_ptr + brow - b, bonus_idx_ptr, b, n,
        last_checked, rejected, num_spec_steps,
    )  # fmt: skip
    return brow - n + tl.where(rejected, first_rejected, n)
