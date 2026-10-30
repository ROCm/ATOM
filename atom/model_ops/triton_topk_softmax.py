# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-M softmax top-k router with sigmoid-scored fused shared experts.

Drop-in for AITER ``topk_softmax`` (``vllm::moe::topkGatingSoftmax``) on decode
batches: same outputs, one Triton program (one wave) per token row.

* routed weights: softmax over the first ``N`` logits, top-``K`` by logit
  (lowest expert id wins ties), renormalised over the K winners (or over the
  whole row when ``renormalize`` is False);
* fused shared experts: ``weights[:, K + s] = sigmoid(logits[:, N + s])``.
  The shared-expert *ids* are not written: ``init_aiter_topK_meta_data`` presets
  them once in the shared buffer, exactly as for the AITER kernel.

Each logit is packed with its expert id into one integer key (order-preserving
logit bits in the high half, ``BLOCK_N - 1 - id`` in the low half), so each of
the K selection rounds is a single max-reduction. bf16 logits (the gate GEMM
output) fit an int32 key exactly; other dtypes use int64.

The row-per-wave layout follows SGLang's ``_router_triton_kernel``
(sglang/kernels/ops/moe/moe_fused_gate.py, Apache-2.0, Copyright the SGLang
team); the packed-key selection and the shared-expert column are ATOM's.
"""

import torch
import triton
import triton.language as tl

from atom.utils.custom_register import direct_register_custom_op

# Below every key a real logit can produce (-inf maps to about -2**63 + 2**55,
# resp. -2**31 + 2**23 for the bf16 key).
_KEY_FLOOR64 = tl.constexpr(-(2**63) + 1)
_KEY_FLOOR32 = tl.constexpr(-(2**31) + 1)


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_lm,
    stride_wm,
    stride_im,
    N: tl.constexpr,  # routed experts: the softmax / top-k domain
    K: tl.constexpr,  # routed top-k
    NUM_SHARED: tl.constexpr,  # sigmoid-scored shared logits at [N, N + NUM_SHARED)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # >= K + NUM_SHARED
    RENORMALIZE: tl.constexpr,
    BF16_KEY: tl.constexpr,  # logits are bf16: exact 32-bit keys
):
    row = tl.program_id(0)
    logits_row = logits_ptr + row.to(tl.int64) * stride_lm

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    raw = tl.load(logits_row + offs_n, mask=mask_n, other=-float("inf"))
    x = raw.to(tl.float32)

    # Order-preserving integer view of the logit bits, expert id in the low half.
    if BF16_KEY:
        b16 = raw.to(tl.int16, bitcast=True).to(tl.int32)
        okey = tl.where(b16 >= 0, b16, b16 ^ 0x7FFF)
        key = (okey << 16) | (BLOCK_N - 1 - offs_n)
    else:
        bits = x.to(tl.int32, bitcast=True)
        okey = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        key = (okey.to(tl.int64) << 32) | (BLOCK_N - 1 - offs_n).to(tl.int64)

    offs_k = tl.arange(0, BLOCK_K)
    sel_val = tl.full([BLOCK_K], -float("inf"), tl.float32)
    sel_idx = tl.zeros([BLOCK_K], dtype=tl.int32)
    for k in tl.static_range(K):
        best = tl.max(key, axis=0)
        if BF16_KEY:
            hi = best >> 16
            # bf16 bits in the top half of an fp32 are the same value
            val = (tl.where(hi >= 0, hi, hi ^ 0x7FFF) << 16).to(
                tl.float32, bitcast=True
            )
            idx = BLOCK_N - 1 - (best & 0xFFFF)
            key = tl.where(key == best, _KEY_FLOOR32, key)
        else:
            hi = (best >> 32).to(tl.int32)
            val = tl.where(hi >= 0, hi, hi ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
            idx = BLOCK_N - 1 - best.to(tl.int32)  # truncation keeps the id word
            key = tl.where(key == best, _KEY_FLOOR64, key)
        sel_val = tl.where(offs_k == k, val, sel_val)
        sel_idx = tl.where(offs_k == k, idx, sel_idx)

    row_max = tl.max(sel_val, axis=0)
    numer = tl.exp(sel_val - row_max)  # 0 outside the K routed slots
    if RENORMALIZE:
        denom = tl.sum(numer, axis=0)
    else:
        denom = tl.sum(tl.where(mask_n, tl.exp(x - row_max), 0.0), axis=0)
    weights = numer / denom

    is_shared = (offs_k >= K) & (offs_k < K + NUM_SHARED)
    if NUM_SHARED > 0:
        s_logit = tl.load(logits_row + N + (offs_k - K), mask=is_shared, other=0.0)
        weights = tl.where(is_shared, tl.sigmoid(s_logit.to(tl.float32)), weights)

    tl.store(
        weights_ptr + row.to(tl.int64) * stride_wm + offs_k,
        weights,
        mask=offs_k < K + NUM_SHARED,
    )
    tl.store(ids_ptr + row.to(tl.int64) * stride_im + offs_k, sel_idx, mask=offs_k < K)


def triton_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_scored: int = 0,
) -> None:
    """Fill ``topk_weights[:, :topk + num_shared_scored]`` and ``topk_ids[:, :topk]``.

    ``gating_output`` is ``[M, N + num_shared_scored]``; outputs may be strided
    row views (e.g. of the fused-shared buffers) but must be unit-stride along
    columns.
    """
    M, n_total = gating_output.shape
    n_routed = n_total - num_shared_scored
    assert gating_output.stride(1) == 1
    assert topk_weights.stride(1) == 1 and topk_ids.stride(1) == 1
    assert topk_weights.dtype == torch.float32 and topk_ids.dtype == torch.int32
    assert topk_weights.shape[0] >= M and topk_ids.shape[0] >= M
    assert topk_weights.shape[1] >= topk + num_shared_scored
    assert topk_ids.shape[1] >= topk and topk <= n_routed
    if M == 0:
        return
    _topk_softmax_kernel[(M,)](
        gating_output,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        N=n_routed,
        K=topk,
        NUM_SHARED=num_shared_scored,
        BLOCK_N=triton.next_power_of_2(n_routed),
        BLOCK_K=triton.next_power_of_2(topk + num_shared_scored),
        RENORMALIZE=renormalize,
        BF16_KEY=gating_output.dtype == torch.bfloat16,
        num_warps=1,
    )


def _triton_topk_softmax_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_scored: int = 0,
) -> None:
    return None


# Opaque to Dynamo (callers such as Qwen4-exp's single-token MoE run inside the
# compiled model forward).
direct_register_custom_op(
    op_name="atom_triton_topk_softmax",
    op_func=triton_topk_softmax,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_triton_topk_softmax_fake,
)
triton_topk_softmax_op = torch.ops.aiter.atom_triton_topk_softmax
