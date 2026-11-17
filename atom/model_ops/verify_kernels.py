"""Triton helpers every verify kernel shares: the draft token a row verifies,
and the bonus, -1 tail and verdict a request ends with. The exact-match,
synthetic and relaxed kernels (`rejection_sampler`) and the ratio accept
(`ratio_verify`) all call them."""

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
