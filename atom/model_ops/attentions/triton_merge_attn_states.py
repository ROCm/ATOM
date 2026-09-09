# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

import aiter

float8_info = torch.finfo(aiter.dtypes.fp8)


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the
    # attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    # If prefill_tokens_with_context is None, all tokens should use prefix context
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    #
    # BLOCK_H heads per program instead of one. With one head per program a
    # program owns a single [HEAD_SIZE] row -- 256 B for the MLA shape -- which
    # is 4 B/lane over a 64-lane wave, a quarter of what a dwordx4 load wants,
    # and it launches num_tokens*num_query_heads (385k at bs=4/MNBT=3072)
    # single-wave workgroups. Both cost the same thing: measured 39.3% of HBM
    # peak. BLOCK_H=8 widens the access to 32 B/lane and cuts the launch count
    # 8x, reaching 76.7% of peak -- 1.95x, against a 2.25x pure-traffic ceiling
    # (benchmark/merge_headroom.py in the perf log repo).
    #
    # num_warps=1 stays: with BLOCK_H*HEAD_SIZE = 1024 elements a single wave
    # already issues wide loads, and more warps only split the row again
    # (BLOCK_H=2 with num_warps=4 measured 0.72x, i.e. slower than before).
    BLOCK_H = min(8, triton.next_power_of_2(num_query_heads))
    merge_attn_states_kernel[(num_tokens, triton.cdiv(num_query_heads, BLOCK_H))](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        output_scale,
        num_tokens,
        num_query_heads,
        head_size,
        padded_head_size,
        BLOCK_H,
        output_lse is not None,
        prefill_tokens_with_context,
        output_scale is not None,
        num_warps=1,
    )


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    output_scale,  # scale tensor or None
    num_tokens,
    num_heads,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    # num_heads need not divide BLOCK_H (TP can leave an odd count), and the
    # tail programs must not touch the next token's rows.
    head_valid = head_idx < num_heads

    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    # [BLOCK_H, PADDED_HEAD_SIZE] -- guards the head tail and the head_size
    # padding at once.
    head_mask = head_valid[:, None] & (head_arange < HEAD_SIZE)[None, :]

    # 64-bit offsets. token_idx*num_heads*head_stride walks the whole output
    # tensor, so it reaches num_tokens*num_heads*head_size -- 2.15e9 at
    # MBT=131072 with 128 heads of 128, just past int32. program_id and the
    # strides are all int32, so the product used to wrap there; widening the two
    # indices once promotes every offset below and costs only address VALU.
    t64 = token_idx.to(tl.int64)
    h64 = head_idx.to(tl.int64)

    lse_off = h64 * num_tokens + t64
    suf_off = (
        t64 * num_heads * prefix_head_stride
        + h64[:, None] * prefix_head_stride
        + head_arange[None, :]
    )
    out_off = (
        t64 * num_heads * output_head_stride
        + h64[:, None] * output_head_stride
        + head_arange[None, :]
    )

    # For tokens without context (token_idx >= prefill_tokens_with_context),
    # directly copy from suffix_output
    if not prefix_mask:
        s_lse = tl.load(suffix_lse + lse_off, mask=head_valid)
        if OUTPUT_LSE:
            tl.store(output_lse + lse_off, s_lse, mask=head_valid)

        s_out = tl.load(suffix_output + suf_off, mask=head_mask)

        if USE_FP8:
            s_out = s_out * (1.0 / tl.load(output_scale))
            s_out = tl.clamp(s_out, FP8_MIN, FP8_MAX)
            s_out = s_out.to(output.dtype.element_ty)

        tl.store(output + out_off, s_out, mask=head_mask)
        return

    # For tokens with context (token_idx < prefill_tokens_with_context),
    # perform normal merge operation
    p_lse = tl.load(prefix_lse + lse_off, mask=head_valid)
    s_lse = tl.load(suffix_lse + lse_off, mask=head_valid)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = tl.where(p_lse == float("inf"), float("-inf"), p_lse)
    s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)

    max_lse = tl.maximum(p_lse, s_lse)
    # Both prefix AND suffix are empty for this token (no KV on either side) ->
    # max_lse == -inf. The naive `p_lse - max_lse` would compute -inf-(-inf)=NaN
    # and `out_se` would be 0, making the scale 0/0=NaN that poisons the output.
    # This happens in ATOM's global-axis chunked prefill: a short seq can fall
    # entirely outside a chunk, so its tokens see an empty prefix AND suffix in
    # that chunk. Force a safe 0/0-split: subtract a finite max so each side's
    # exp is 0 (out = 0*p_out + 0*s_out = 0, correct for empty attention) and
    # keep the merged lse at -inf so any downstream merge stays consistent.
    both_empty = max_lse == float("-inf")
    safe_max = tl.where(both_empty, 0.0, max_lse)
    p_lse = p_lse - safe_max
    s_lse = s_lse - safe_max
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.where(both_empty, float("-inf"), tl.log(out_se) + safe_max)
        tl.store(output_lse + lse_off, out_lse, mask=head_valid)

    p_out = tl.load(prefix_output + suf_off, mask=head_mask)
    s_out = tl.load(suffix_output + suf_off, mask=head_mask)

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    # both_empty -> out_se == 0; guard the denominator so the scale is 0/1=0
    # (not 0/0=NaN). p_out/s_out are 0 for empty attention, so out stays 0.
    safe_out_se = tl.where(both_empty, 1.0, out_se)
    # scales are per (head,) -- broadcast over the head_size axis.
    p_scale = (p_se / safe_out_se)[:, None]
    s_scale = (s_se / safe_out_se)[:, None]
    out = p_out * p_scale + s_out * s_scale

    if USE_FP8:
        out = out * (1.0 / tl.load(output_scale))
        out = tl.clamp(out, FP8_MIN, FP8_MAX)
        out = out.to(output.dtype.element_ty)

    tl.store(output + out_off, out, mask=head_mask)
