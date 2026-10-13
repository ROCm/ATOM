# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8 renormalized softmax top-k routing in one Triton launch.

AITER's `topk_softmax` runs a 512-expert row through warp-serial selection
and takes about 10 us at 16-48 decode rows on gfx950; this kernel packs each
logit with its expert index into one sortable integer and selects with
`tl.topk`, about 3.5 us. Renormalized softmax over the winners only needs the
winning logits, so the full-row softmax is never formed. Equal logits rank
the lower expert first. Weights differ from AITER's by float rounding.

With the shared expert fused into the MoE, the tail logit's sigmoid is the
shared expert's weight, written after the routed winners in AITER's shared
top-k metadata buffers.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops import topK
from atom.model_ops.qwen4_exp.ops import moe_small_batch


@triton.jit
def _softmax_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    expert_map_ptr,
    route_ptr,
    stride_logits,
    stride_out,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    SHARED: tl.constexpr,
    SLOTS: tl.constexpr,
    BINS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BM: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    experts = tl.arange(0, EXPERTS)
    logit = tl.load(logits_ptr + row * stride_logits + experts).to(tl.float32)
    # Order-preserving float -> uint32 map, then the inverted index in the low
    # bits so equal logits rank the lower expert first.
    bits = logit.to(tl.int32, bitcast=True)
    order = tl.where(bits < 0, ~bits, bits | -2147483648).to(tl.int64) & 0xFFFFFFFF
    key = (order << 16) | (65535 - experts).to(tl.int64)
    top = tl.topk(key, TOPK_POW2)
    ids = 65535 - (top & 0xFFFF).to(tl.int32)
    order = (top >> 16).to(tl.int32)
    value = tl.where(order < 0, order & 2147483647, ~order).to(tl.float32, bitcast=True)
    slot = tl.arange(0, TOPK_POW2)
    value = tl.where(slot < TOPK, value, float("-inf"))
    weight = tl.exp(value - tl.max(value, 0))
    weight = weight / tl.sum(weight, 0)
    out = row * stride_out + slot
    tl.store(weights_ptr + out, weight, mask=slot < TOPK)
    tl.store(ids_ptr + out, ids, mask=slot < TOPK)
    if SHARED:
        gate = tl.load(logits_ptr + row * stride_logits + EXPERTS).to(tl.float32)
        tl.store(weights_ptr + row * stride_out + TOPK, tl.sigmoid(gate))
    if SLOTS:
        tail = (slot >= TOPK) & (slot < SLOTS)
        gid = tl.where(tail, tl.load(ids_ptr + out, mask=tail, other=0), ids)
        moe_small_batch.file_pairs(
            route_ptr,
            expert_map_ptr,
            gid,
            row * SLOTS + slot,
            slot < SLOTS,
            BINS,
            MAX_BLOCKS,
            BM,
        )


def supports(num_experts: int, topk: int, renormalize: bool) -> bool:
    return (
        renormalize
        and num_experts & (num_experts - 1) == 0
        and num_experts <= 65536
        and 0 < topk <= num_experts
    )


def softmax_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_experts: int,
    buckets: tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`FusedMoE` custom routing; `gating_output` may carry one shared logit.

    `buckets = (expert_map, moe_small_batch.route_buckets(...))` also files
    every local pair `token * slots + slot` under its local expert and lists
    their blocks for `moe_small_batch`, which then needs no plan launch.
    """
    assert renormalize
    tokens = gating_output.shape[0]
    shared = gating_output.shape[1] - num_experts
    assert shared in (0, 1) and gating_output.stride(1) == 1
    if shared:
        weights, ids = topK.aiter_topK_meta_data
        assert weights.shape[0] >= tokens and weights.shape[1] > topk
        assert weights.stride(0) == ids.stride(0)
        weights, ids = weights[:tokens], ids[:tokens]
    else:
        weights = torch.empty(
            tokens, topk, dtype=torch.float32, device=gating_output.device
        )
        ids = torch.empty(tokens, topk, dtype=torch.int32, device=gating_output.device)
    topk_pow2 = triton.next_power_of_2(topk)
    if buckets is None:
        expert_map = route = ids
        slots = bins = max_blocks = 0
    else:
        expert_map, (route, pairs, blk_expert, *_) = buckets
        slots = ids.shape[1]
        bins = triton.next_power_of_2(pairs.numel() // moe_small_batch.MAX_TOKENS)
        max_blocks = blk_expert.numel()
        assert slots <= topk_pow2 and ids.is_contiguous()
        assert tokens <= moe_small_batch.MAX_TOKENS
    _softmax_topk_kernel[(tokens,)](
        gating_output,
        weights,
        ids,
        expert_map,
        route,
        gating_output.stride(0),
        weights.stride(0),
        EXPERTS=num_experts,
        TOPK=topk,
        TOPK_POW2=topk_pow2,
        SHARED=bool(shared),
        SLOTS=slots,
        BINS=bins,
        MAX_BLOCKS=max_blocks,
        BM=moe_small_batch.BM,
        num_warps=4,
    )
    return weights, ids
