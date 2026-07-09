# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Native RCCL all2all backend for expert-parallel MoE.

A dependency-free alternative to ``MoriPrepareAndFinalize`` built only on
``torch.distributed`` collectives (``all_gather_into_tensor``,
``all_to_all_single``) over the EP process group.

Structure (mirrors vLLM's DeepEP HT/LL split):

    RcclAll2AllBase            shared helpers: FP8 quant, dest-rank math,
                               scatter-add combine, routing-state stack.
      |- RcclHTPrepareAndFinalize   prefill / non-uniform decode:
      |                             allgather topk -> host counts ->
      |                             variable-length all_to_all_single.
      |- RcclLLPrepareAndFinalize   uniform decode: fixed cross-DP-unified
      |                             capacity C, equal-split all_to_all_single,
      |                             no host sync -> CUDA-graph safe.
    RcclPrepareAndFinalize     dispatcher held by the MoE layer; routes each
                               prepare()/finalize() to HT or LL per step from
                               the forward context.

The pure routing math is factored into module-level functions so it can be
unit-tested without a GPU or a live distributed backend.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.utils.forward_context import get_forward_context
from aiter import QuantType, dtypes

logger = logging.getLogger("atom")


@triton.jit
def _pack_dispatch_kernel(
    pack_index_ptr,  # (P,) int64: flat (token*topk + k) pair id per packed slot
    a1_ptr,  # (num_tokens, hidden)
    weights_ptr,  # (num_tokens * topk,) flat topk weights
    ids_ptr,  # (num_tokens * topk,) flat topk ids (int32)
    scale_ptr,  # (num_tokens, scale_dim) or dummy
    out_a1_ptr,  # (P, hidden)
    out_weights_ptr,  # (P,)
    out_ids_ptr,  # (P,)
    out_scale_ptr,  # (P, scale_dim) or dummy
    hidden,
    scale_dim,
    topk,
    HAS_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Fused pack: for packed slot p, gather the source token's hidden row (and
    scale row) plus the per-pair weight and id, in a single launch. Replaces the
    four separate index_select ops (flat_row compute + 3-4 gathers)."""
    p = tl.program_id(0)
    pair = tl.load(pack_index_ptr + p)  # int64 flat pair id
    row = pair // topk  # source token index

    # scalar weight + id for this pair (indexed by the flat pair id).
    w = tl.load(weights_ptr + pair)
    tl.store(out_weights_ptr + p, w)
    i = tl.load(ids_ptr + pair)
    tl.store(out_ids_ptr + p, i)

    # gather the hidden row: out_a1[p, :] = a1[row, :]
    for h0 in tl.range(0, hidden, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        vals = tl.load(a1_ptr + row * hidden + offs, mask=mask)
        tl.store(out_a1_ptr + p * hidden + offs, vals, mask=mask)

    if HAS_SCALE:
        for s0 in tl.range(0, scale_dim, BLOCK_S):
            offs = s0 + tl.arange(0, BLOCK_S)
            mask = offs < scale_dim
            vals = tl.load(scale_ptr + row * scale_dim + offs, mask=mask)
            tl.store(out_scale_ptr + p * scale_dim + offs, vals, mask=mask)


def _pack_dispatch(
    pack_index: torch.Tensor,  # (P,) int64
    a1: torch.Tensor,  # (num_tokens, hidden)
    topk_weights: torch.Tensor,  # (num_tokens, topk)
    topk_ids: torch.Tensor,  # (num_tokens, topk) int32
    scale: Optional[torch.Tensor],  # (num_tokens, scale_dim) or None
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """One-launch fused gather producing (send_a1, send_weights, send_ids,
    send_scale) grouped by ``pack_index``. Equivalent to::

        flat_row = pack_index // topk
        send_a1 = a1.index_select(0, flat_row)
        send_weights = topk_weights.reshape(-1).index_select(0, pack_index)
        send_ids = topk_ids.reshape(-1).index_select(0, pack_index)
        send_scale = scale.index_select(0, flat_row)  # if scale is not None
    """
    P = pack_index.shape[0]
    hidden = a1.shape[1]
    has_scale = scale is not None
    scale_dim = scale.shape[1] if has_scale else 0

    # The fused kernel is a GPU (Triton) optimization; fall back to plain
    # index_select on CPU (used by the mocked-dist unit tests), when Triton is
    # unavailable, or when ATOM_RCCL_FUSED_PACK is disabled (A/B toggle).
    import os

    use_fused = a1.device.type == "cuda" and os.getenv(
        "ATOM_RCCL_FUSED_PACK", "1"
    ) not in ("0", "false", "False")
    if not use_fused:
        flat_row = torch.div(pack_index, topk, rounding_mode="floor")
        send_a1 = a1.index_select(0, flat_row)
        send_weights = topk_weights.reshape(-1).index_select(0, pack_index)
        send_ids = topk_ids.reshape(-1).index_select(0, pack_index)
        send_scale = scale.index_select(0, flat_row) if has_scale else None
        return send_a1, send_weights, send_ids, send_scale

    send_a1 = torch.empty((P, hidden), dtype=a1.dtype, device=a1.device)
    send_weights = torch.empty((P,), dtype=topk_weights.dtype, device=a1.device)
    send_ids = torch.empty((P,), dtype=topk_ids.dtype, device=a1.device)
    send_scale = (
        torch.empty((P, scale_dim), dtype=scale.dtype, device=a1.device)
        if has_scale
        else torch.empty((0, 0), dtype=a1.dtype, device=a1.device)
    )
    if P == 0:
        return send_a1, send_weights, send_ids, (send_scale if has_scale else None)

    weights_flat = topk_weights.reshape(-1).contiguous()
    ids_flat = topk_ids.reshape(-1).contiguous()
    a1 = a1.contiguous()
    if has_scale:
        scale = scale.contiguous()
    pack_index = pack_index.contiguous()
    BLOCK_H = 1024
    BLOCK_S = 128 if has_scale else 1
    _pack_dispatch_kernel[(P,)](
        pack_index,
        a1,
        weights_flat,
        ids_flat,
        scale if has_scale else send_a1,  # input scale (dummy ptr when no scale)
        send_a1,
        send_weights,
        send_ids,
        send_scale,
        hidden,
        scale_dim,
        topk,
        HAS_SCALE=has_scale,
        BLOCK_H=BLOCK_H,
        BLOCK_S=BLOCK_S,
    )
    return send_a1, send_weights, send_ids, (send_scale if has_scale else None)


@triton.jit
def _combine_kernel(
    home_ptr,  # (R_home, hidden) expert outputs returned home
    gather_ptr,  # (P,) int64: row of home each pair reads (already clamped)
    token_ptr,  # (P,) int64: output token each pair adds into
    weight_ptr,  # (P,) fp32: per-pair combine weight (0 for dropped pairs)
    out_ptr,  # (num_tokens, hidden) accumulator (pre-zeroed)
    hidden,
    BLOCK_H: tl.constexpr,
):
    """Fused combine: out[token[p]] += home[gather[p]] * weight[p], one pass.

    Replaces index_select(home) + (home*w) + index_add_. Each program handles one
    (token,expert) pair; contributions to the same token collide, so accumulate
    with tl.atomic_add. The intermediate (P, hidden) product tensor is never
    materialized (no extra HBM round-trip)."""
    p = tl.program_id(0)
    w = tl.load(weight_ptr + p)
    src = tl.load(gather_ptr + p)
    dst = tl.load(token_ptr + p)
    for h0 in tl.range(0, hidden, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        vals = tl.load(home_ptr + src * hidden + offs, mask=mask).to(tl.float32)
        vals = vals * w
        tl.atomic_add(out_ptr + dst * hidden + offs, vals, mask=mask)


@triton.jit
def _combine_gather_kernel(
    home_ptr,  # (R_home, hidden) expert outputs returned home
    rows_ptr,  # (num_tokens, topk) int64: home row for each token's k-th pair
    weight_ptr,  # (num_tokens, topk) fp32: per-(token,k) combine weight
    out_ptr,  # (num_tokens, hidden) output (written once, no pre-zero needed)
    hidden,
    topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-TOKEN combine (no atomics): one program owns output token t, gathers
    its ``topk`` home rows, weight-sums them in registers, and writes out[t] in a
    single store. Because each token is written by exactly one program there is
    no contention — replaces the per-pair atomic_add scatter (which serialized
    ``topk`` atomic adds per token onto the same cache lines) and the separate
    fp32-accumulator zero + cast passes."""
    t = tl.program_id(0)
    for h0 in tl.range(0, hidden, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k in tl.static_range(topk):
            row = tl.load(rows_ptr + t * topk + k)
            w = tl.load(weight_ptr + t * topk + k)
            vals = tl.load(home_ptr + row * hidden + offs, mask=mask).to(tl.float32)
            acc += vals * w
        tl.store(
            out_ptr + t * hidden + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
        )


def _combine_weighted_gather(
    home: torch.Tensor,  # (R_home, hidden)
    rows: torch.Tensor,  # (num_tokens, topk) int64 home row per (token, k)
    weight: torch.Tensor,  # (num_tokens, topk) per-(token,k) weight
    num_tokens: int,
    topk: int,
) -> torch.Tensor:
    """out[t] = sum_k home[rows[t, k]] * weight[t, k], fused per token (no atomics,
    no fp32 accumulator round-trip). Writes bf16/home-dtype output directly.

    This is the fast path for the RCCL finalize combine: every output token has
    exactly ``topk`` contributing home rows, so a per-token gather-sum avoids the
    atomic scatter entirely."""
    import os

    hidden = home.shape[1]
    use_fused = home.device.type == "cuda" and os.getenv(
        "ATOM_RCCL_FUSED_PACK", "1"
    ) not in ("0", "false", "False")
    if not use_fused:
        # Pure-torch reference (CPU / no-Triton / toggle off): gather the topk
        # home rows per token and weight-sum. rows is (T, topk), weight (T, topk).
        r = rows.reshape(-1)
        g = home.index_select(0, r).to(torch.float32).view(num_tokens, topk, hidden)
        out = (g * weight.to(torch.float32).unsqueeze(-1)).sum(dim=1)
        return out.to(home.dtype)

    out = torch.empty((num_tokens, hidden), dtype=home.dtype, device=home.device)
    if num_tokens == 0:
        return out
    _combine_gather_kernel[(num_tokens,)](
        home.contiguous(),
        rows.contiguous(),
        weight.to(torch.float32).contiguous(),
        out,
        hidden,
        topk=topk,
        BLOCK_H=1024,
    )
    return out


def _combine_weighted_scatter(
    home: torch.Tensor,  # (R_home, hidden)
    gather_idx: torch.Tensor,  # (P,) int64 row-in-home per pair
    token_idx: torch.Tensor,  # (P,) int64 output token per pair
    weight: torch.Tensor,  # (P,) per-pair weight (dropped pairs -> 0)
    num_tokens: int,
) -> torch.Tensor:
    """out[token_idx[p]] += home[gather_idx[p]] * weight[p], fused in Triton.

    Equivalent to::

        rows = home.index_select(0, gather_idx)
        out.index_add_(0, token_idx, rows * weight.unsqueeze(1))

    but with no intermediate (P, hidden) tensor. CPU / non-CUDA falls back to the
    torch ops (used by mocked-dist unit tests)."""
    import os

    hidden = home.shape[1]
    # Accumulate in fp32 for atomic_add correctness, then cast back.
    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=home.device)
    use_fused = home.device.type == "cuda" and os.getenv(
        "ATOM_RCCL_FUSED_PACK", "1"
    ) not in ("0", "false", "False")
    if not use_fused:
        rows = home.index_select(0, gather_idx).to(torch.float32)
        out.index_add_(0, token_idx, rows * weight.to(torch.float32).unsqueeze(1))
        return out.to(home.dtype)

    P = gather_idx.shape[0]
    if P == 0:
        return out.to(home.dtype)
    _combine_kernel[(P,)](
        home.contiguous(),
        gather_idx.contiguous(),
        token_idx.contiguous(),
        weight.to(torch.float32).contiguous(),
        out,
        hidden,
        BLOCK_H=1024,
    )
    return out.to(home.dtype)


# --------------------------------------------------------------------------- #
# Fused LL (fixed-capacity) prepare-pack kernels.
#
# The LL prepare slot assignment was a chain of ~a dozen small torch ops (a
# [P, ws] one-hot alloc + scatter_ + cumsum + gather + where + two aranges) plus
# 3-4 index_copy_ scatters. That is a lot of tiny launches + a P*ws int64 scan
# on the decode hot path. These two kernels collapse it into two launches:
#   1. _ll_slot_assign_kernel: per-pair destination slot via ONE atomic counter
#      per destination rank (replaces one-hot/cumsum/gather/where/aranges).
#   2. _ll_pack_scatter_kernel: gather source row (+scale) and scalar weight/id
#      and scatter into the send grid in one pass (replaces index_select +
#      the per-tensor index_copy_).
# Both are CUDA-graph safe: static shapes, no host sync, no data-dependent
# indexing. Slot order within a destination is nondeterministic under atomics,
# but LL correctness only needs the (pair -> slot) mapping, not its order
# (finalize gathers back by dst_slot), so this is exact.
# --------------------------------------------------------------------------- #


@triton.jit
def _ll_slot_assign_kernel(
    ids_ptr,  # (P,) flat topk ids (any int dtype); dest = id // num_local_experts
    counter_ptr,  # (ws,) int32 per-dest running counter (pre-zeroed)
    slot2pair_ptr,  # (ws*C,) int64 out: source pair per KEPT head slot (garbage
    #                 in padding slots — the copy kernel gates on counts, not -1)
    dst_slot_ptr,  # (P,) int64 out: head slot each pair landed in (0 if dropped)
    valid_pair_ptr,  # (P,) int64 out: 1 if within capacity C, else 0
    P,
    C,
    num_local_experts,
):
    """One program per (token, expert) pair. Computes the destination rank inline
    (dest = id // num_local_experts) and atomically claims the next slot in that
    dest block. KEPT pairs (slot < C) record the INVERSE map slot2pair[dest*C +
    slot] = p AND their head slot dst_slot[p] (finalize gathers home rows by it);
    overflow pairs write dst_slot = 0 and valid_pair = 0 (finalize zeros them via
    valid_pair). No dump tail — the per-slot copy kernel gates on counter[dest]."""
    p = tl.program_id(0)
    if p >= P:
        return
    gid = tl.load(ids_ptr + p).to(tl.int64)
    d = gid // num_local_experts  # destination rank
    slot = tl.atomic_add(counter_ptr + d, 1).to(tl.int64)  # slot within dest block
    valid = slot < C
    tl.store(valid_pair_ptr + p, valid.to(tl.int64))
    dst = tl.where(valid, d * C + slot, 0)
    tl.store(dst_slot_ptr + p, dst)
    if valid:
        tl.store(slot2pair_ptr + d * C + slot, p)


@triton.jit
def _ll_pack_scatter_kernel(
    slot2pair_ptr,  # (ws*C,) int64 source pair per head slot (garbage if padding)
    counter_ptr,  # (ws,) int32 per-dest kept count (occupancy = min(counter, C))
    a1_ptr,  # (num_tokens, hidden)
    weights_ptr,  # (P,) flat topk weights
    ids_ptr,  # (P,) flat topk ids (any int)
    scale_ptr,  # (num_tokens, scale_dim) or dummy
    out_a1_ptr,  # (ws*C, hidden)  — this kernel OWNS the full write (no pre-zero)
    out_w_ptr,  # (ws*C, 1)
    out_id_ptr,  # (ws*C, 1)
    out_scale_ptr,  # (ws*C, scale_dim) or dummy
    C,
    hidden,
    scale_dim,
    topk,
    HAS_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Per-SLOT kernel over the transmitted [ws*C] head. Program (slot) is
    occupied iff its within-block index < counter[dest]; if so it copies its
    source pair's activation/scale/weight/id, else it writes the padding fill
    (zeros / id=-1). Owning every head slot here removes the torch.zeros / full
    pre-init entirely — the send buffers are allocated with torch.empty."""
    dst = tl.program_id(0)  # head slot in [0, ws*C)
    d = dst // C
    s = dst - d * C
    cnt = tl.load(counter_ptr + d)
    occupied = s < cnt

    # Source pair / token (only meaningful when occupied; kept int64 in both
    # branches so the value has a consistent type across the control flow).
    p = tl.where(occupied, tl.load(slot2pair_ptr + dst), 0).to(tl.int64)
    src = p // topk  # source token

    if occupied:
        tl.store(out_w_ptr + dst, tl.load(weights_ptr + p))
        i = tl.load(ids_ptr + p).to(out_id_ptr.dtype.element_ty)
        tl.store(out_id_ptr + dst, i)
    else:
        # padding slot: weight 0, id -1 (matches the old torch.zeros / full(-1)).
        tl.store(out_w_ptr + dst, tl.zeros((), out_w_ptr.dtype.element_ty))
        tl.store(out_id_ptr + dst, tl.full((), -1, out_id_ptr.dtype.element_ty))

    for h0 in tl.range(0, hidden, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        if occupied:
            vals = tl.load(a1_ptr + src * hidden + offs, mask=mask)
        else:
            vals = tl.zeros((BLOCK_H,), out_a1_ptr.dtype.element_ty)
        tl.store(out_a1_ptr + dst * hidden + offs, vals, mask=mask)

    if HAS_SCALE:
        for s0 in tl.range(0, scale_dim, BLOCK_S):
            offs = s0 + tl.arange(0, BLOCK_S)
            mask = offs < scale_dim
            if occupied:
                vals = tl.load(scale_ptr + src * scale_dim + offs, mask=mask)
            else:
                vals = tl.zeros((BLOCK_S,), out_scale_ptr.dtype.element_ty)
            tl.store(out_scale_ptr + dst * scale_dim + offs, vals, mask=mask)


def _ll_prepare_pack(
    topk_ids: torch.Tensor,  # (num_tokens, topk) int expert ids (global)
    a1: torch.Tensor,  # (num_tokens, hidden)
    topk_weights: torch.Tensor,  # (num_tokens, topk)
    scale: Optional[torch.Tensor],  # (num_tokens, scale_dim) or None
    topk: int,
    ws: int,
    C: int,
    num_local_experts: int,
):
    """Fused LL pack: returns (send_a1, send_w, send_id, send_scale, dst_slot,
    valid_pair). ``send_*`` are the full [ws*C + P] grids (caller transmits the
    [ws*C] head); dst_slot / valid_pair are the (P,) routing state for finalize.

    The destination rank (id // num_local_experts) is computed INSIDE the slot
    kernel, so no flat_dest / dest_rank_for_expert / to(int64) op is needed.

    Falls back to the pure-torch path on CPU / when Triton is unavailable / when
    ATOM_RCCL_FUSED_PACK is disabled (so mocked-dist unit tests still work)."""
    import os

    P = topk_ids.numel()
    hidden = a1.shape[1]
    has_scale = scale is not None
    scale_dim = scale.shape[1] if has_scale else 0
    total = ws * C + P
    dev = a1.device
    ids_flat = topk_ids.reshape(-1).contiguous()

    use_fused = dev.type == "cuda" and os.getenv("ATOM_RCCL_FUSED_PACK", "1") not in (
        "0",
        "false",
        "False",
    )

    if not use_fused:
        # Pure-torch reference (original op chain).
        flat_dest = torch.div(
            ids_flat.to(torch.int64), num_local_experts, rounding_mode="floor"
        )
        src_token = torch.arange(P, device=dev) // topk
        onehot = torch.zeros(P, ws, dtype=torch.int64, device=dev)
        onehot.scatter_(1, flat_dest.unsqueeze(1), 1)
        slot_in_dest = (
            (onehot.cumsum(0) - onehot).gather(1, flat_dest.unsqueeze(1)).squeeze(1)
        )
        valid_pair = (slot_in_dest < C).to(torch.int64)
        in_slot = flat_dest * C + slot_in_dest
        dump_slot = ws * C + torch.arange(P, device=dev)
        dst_slot = torch.where(valid_pair.bool(), in_slot, dump_slot)
        send_a1 = torch.zeros((total, hidden), dtype=a1.dtype, device=dev)
        send_w = torch.zeros((total, 1), dtype=topk_weights.dtype, device=dev)
        send_id = torch.full((total, 1), -1, dtype=torch.int32, device=dev)
        send_a1.index_copy_(0, dst_slot, a1.index_select(0, src_token))
        send_w.index_copy_(0, dst_slot, topk_weights.reshape(-1).unsqueeze(1))
        send_id.index_copy_(0, dst_slot, ids_flat.to(torch.int32).unsqueeze(1))
        send_scale = None
        if has_scale:
            send_scale = torch.zeros((total, scale_dim), dtype=scale.dtype, device=dev)
            send_scale.index_copy_(0, dst_slot, scale.index_select(0, src_token))
        return send_a1, send_w, send_id, send_scale, dst_slot, valid_pair

    # --- fused path ---
    # NO torch.zeros / torch.full: the per-slot scatter kernel writes EVERY head
    # slot (occupied -> pair data, padding -> zeros / id=-1), so the send buffers
    # are torch.empty and there is no dump tail (buffer is exactly [ws*C]).
    head = ws * C
    slot2pair = torch.empty(head, dtype=torch.int64, device=dev)
    dst_slot = torch.empty(P, dtype=torch.int64, device=dev)
    valid_pair = torch.empty(P, dtype=torch.int64, device=dev)
    counter = torch.zeros(ws, dtype=torch.int32, device=dev)  # atomic accumulator
    _ll_slot_assign_kernel[(P,)](
        ids_flat, counter, slot2pair, dst_slot, valid_pair, P, C, num_local_experts
    )

    send_a1 = torch.empty((head, hidden), dtype=a1.dtype, device=dev)
    send_w = torch.empty((head, 1), dtype=topk_weights.dtype, device=dev)
    send_id = torch.empty((head, 1), dtype=torch.int32, device=dev)
    send_scale = (
        torch.empty((head, scale_dim), dtype=scale.dtype, device=dev)
        if has_scale
        else torch.empty((0, 0), dtype=a1.dtype, device=dev)
    )
    weights_flat = topk_weights.reshape(-1).contiguous()
    a1 = a1.contiguous()
    if has_scale:
        scale = scale.contiguous()
    _ll_pack_scatter_kernel[(head,)](
        slot2pair,
        counter,
        a1,
        weights_flat,
        ids_flat,
        scale if has_scale else send_a1,  # dummy ptr when no scale
        send_a1,
        send_w,
        send_id,
        send_scale,
        C,
        hidden,
        scale_dim,
        topk,
        HAS_SCALE=has_scale,
        BLOCK_H=1024,
        BLOCK_S=128 if has_scale else 1,
    )
    return (
        send_a1,
        send_w,
        send_id,
        (send_scale if has_scale else None),
        dst_slot,
        valid_pair,
    )


# --------------------------------------------------------------------------- #
# Pure routing math (no collectives, no device requirement) — unit-testable.
# --------------------------------------------------------------------------- #


def dest_rank_for_expert(
    topk_ids: torch.Tensor, num_local_experts: int
) -> torch.Tensor:
    """Map each (token, k) expert id to the EP rank that owns that expert.

    Ownership is contiguous: global expert ``e`` lives on rank
    ``e // num_local_experts``. Returns an int tensor shaped like ``topk_ids``.
    """
    return torch.div(topk_ids, num_local_experts, rounding_mode="floor")


def build_send_counts(dest_ranks: torch.Tensor, world_size: int) -> torch.Tensor:
    """Count (token, k) pairs destined for each rank.

    ``dest_ranks`` is the output of :func:`dest_rank_for_expert`. Returns an
    int64 tensor of length ``world_size`` on the same device.
    """
    flat = dest_ranks.reshape(-1)
    counts = torch.zeros(world_size, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
    return counts


def build_pack_index(
    dest_ranks: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the gather index that packs (token,k) pairs grouped by dest rank.

    Returns ``(pack_index, send_counts)`` where ``pack_index[i]`` is the flat
    (token * topk + k) source position of the i-th packed pair, ordered by
    destination rank (stable within a rank). ``send_counts`` is the per-rank
    pair count (see :func:`build_send_counts`).

    This is a stable sort by destination rank of the flattened pair ids, which
    produces a contiguous per-rank layout suitable for ``all_to_all_single``.
    """
    flat = dest_ranks.reshape(-1)
    send_counts = build_send_counts(dest_ranks, world_size)
    # Stable sort keeps token/k order within each destination bucket.
    pack_index = torch.argsort(flat, stable=True)
    return pack_index, send_counts


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


def _as_comm_bytes(t: torch.Tensor) -> torch.Tensor:
    """View an fp8 tensor as uint8 for RCCL/NCCL collectives.

    RCCL has no fp8 collective type ("Unconvertible NCCL type Float8_e4m3fn"),
    so fp8 payloads must be exchanged as raw bytes (uint8, byte-identical) and
    viewed back to the original dtype afterwards. Non-fp8 tensors pass through.
    """
    if t.dtype in _FP8_DTYPES:
        return t.view(torch.uint8)
    return t


def balanced_warmup_topk_ids(
    num_tokens: int, topk: int, num_experts: int, device
) -> torch.Tensor:
    """Deterministic, perfectly load-balanced ``topk_ids`` for dummy warmup.

    Warmup runs the real gate on zeroed hidden states, which routes
    degenerately (nearly all tokens to one expert). Under the all2all backends
    that skew makes one EP rank / one local expert receive the entire batch,
    blowing up the batched ``[E, C, hidden]`` grid (C = max tokens per expert)
    and OOMing. Warmup only needs to exercise the code paths at representative
    shapes, not preserve routing, so we overwrite ``topk_ids`` with a round-robin
    assignment: pair index ``(t*topk + k)`` maps to expert ``(t*topk + k) %
    num_experts``. Because experts are owned contiguously
    (rank = expert // num_local_experts), this sends an identical number of
    (token, expert) pairs to every rank — the balanced worst-case-safe shape.

    Returns an ``int32`` tensor of shape ``[num_tokens, topk]``.
    """
    total_pairs = num_tokens * topk
    flat = torch.arange(total_pairs, device=device, dtype=torch.int64) % num_experts
    return flat.reshape(num_tokens, topk).to(torch.int32)


# --------------------------------------------------------------------------- #
# Shared base.
# --------------------------------------------------------------------------- #


class _RoutingState:
    """Per-call routing info saved by prepare() and consumed by finalize()."""

    __slots__ = (
        "pack_index",
        "send_counts",
        "recv_counts",
        "num_input_tokens",
        "topk",
    )

    def __init__(
        self,
        pack_index: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        num_input_tokens: int,
        topk: int,
    ):
        self.pack_index = pack_index
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.num_input_tokens = num_input_tokens
        self.topk = topk


class RcclAll2AllBase:
    """Shared helpers for the native RCCL HT/LL implementations.

    Holds no path policy — the concrete HT/LL subclasses implement the actual
    prepare()/finalize() algorithms and use these helpers.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        scale_dim: int,
        max_tokens_per_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
        in_dtype: torch.dtype,
        use_fp8_dispatch: bool = False,
        quant_type: Optional[QuantType] = None,
        ep_group=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        self.scale_dim = scale_dim
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.num_experts_per_token = num_experts_per_token
        self.in_dtype = in_dtype
        self.use_fp8_dispatch = use_fp8_dispatch
        self.quant_type = quant_type
        self._ep_group = ep_group
        self._routing_stack: list[_RoutingState] = []

    # ---- EP group access -------------------------------------------------- #

    @property
    def ep_group(self):
        if self._ep_group is None:
            from aiter.dist.parallel_state import get_ep_group

            self._ep_group = get_ep_group()
        return self._ep_group

    @property
    def device_group(self):
        return self.ep_group.device_group

    # ---- FusedMoEPrepareAndFinalize interface passthroughs ---------------- #

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int32

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_tokens_per_rank

    def num_dispatchers(self) -> int:
        return self.world_size

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return False

    # ---- shared helpers --------------------------------------------------- #

    def _maybe_balance_warmup(
        self, topk_ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        """On dummy warmup, replace degenerate gate routing with a balanced one.

        Keeps warmup from OOMing / dispatching an all-to-one-expert batch. No-op
        on real forwards. num_experts here is the GLOBAL expert count.
        """
        ctx = get_forward_context().context
        if ctx is not None and getattr(ctx, "is_dummy_run", False):
            return balanced_warmup_topk_ids(
                topk_ids.shape[0], topk_ids.shape[1], num_experts, topk_ids.device
            )
        return topk_ids

    def _maybe_quant(
        self, a1: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Optionally FP8-quantize activations before dispatch."""
        if not self.use_fp8_dispatch:
            return a1, None
        from aiter import get_hip_quant

        quant_func = get_hip_quant(self.quant_type)
        a1_q, scale = quant_func(a1, quant_dtype=dtypes.fp8)
        return a1_q, scale

    def _all_to_all(
        self,
        send: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> torch.Tensor:
        """Variable-length all_to_all_single over the EP device group.

        ``send`` is packed grouped-by-dest-rank along dim 0. ``send_counts`` /
        ``recv_counts`` are per-rank row counts (Python ints, host-side).
        """
        # RCCL/NCCL has no fp8 collective type; exchange fp8 as raw bytes
        # (uint8, byte-identical) and restore the dtype after.
        orig_dtype = send.dtype
        send_b = _as_comm_bytes(send)
        total_recv = int(sum(recv_counts))
        recv = torch.empty(
            (total_recv,) + tuple(send_b.shape[1:]),
            dtype=send_b.dtype,
            device=send_b.device,
        )
        dist.all_to_all_single(
            recv,
            send_b.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.device_group,
        )
        return recv.view(orig_dtype)

    def _all_to_all_equal(self, send: torch.Tensor, capacity: int) -> torch.Tensor:
        """Fixed equal-split all_to_all_single (CUDA-graph safe).

        ``send`` has shape ``(world_size * capacity, ...)``; every rank
        exchanges exactly ``capacity`` rows with every other rank.
        """
        assert send.shape[0] == self.world_size * capacity, (
            f"equal-split all_to_all expects world_size*capacity rows, "
            f"got {send.shape[0]} != {self.world_size} * {capacity}"
        )
        # RCCL/NCCL has no fp8 collective type; exchange fp8 as raw bytes.
        orig_dtype = send.dtype
        send_b = _as_comm_bytes(send)
        recv = torch.empty_like(send_b)
        dist.all_to_all_single(recv, send_b.contiguous(), group=self.device_group)
        return recv.view(orig_dtype)


# --------------------------------------------------------------------------- #
# HT (high-throughput): prefill + non-uniform decode. Host sync allowed.
# --------------------------------------------------------------------------- #


class RcclHTPrepareAndFinalize(RcclAll2AllBase, mk.FusedMoEPrepareAndFinalize):
    """Variable-length dispatch/combine. Not CUDA-graph captured."""

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        assert (
            not apply_router_weight_on_input
        ), "rccl all2all does not support apply_router_weight_on_input=True"
        # Dummy warmup: balance routing so no rank/expert gets the whole batch.
        topk_ids = self._maybe_balance_warmup(topk_ids, num_experts)
        num_input_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]

        a1, scale = self._maybe_quant(a1)

        # 1. Allgather topk_ids so every rank sees global routing (for symmetric
        #    recv-count computation via the transpose of send counts).
        dest_ranks = dest_rank_for_expert(topk_ids, self.num_local_experts)
        pack_index, send_counts_dev = build_pack_index(dest_ranks, self.world_size)

        # 2. Exchange the per-rank send counts to learn recv counts (host sync).
        #    all_to_all_single of the [ws] send-count vector: element j of the
        #    output is what rank j sent us, i.e. our recv counts. The wall-clock
        #    here is dominated by barrier wait (a fast rank arrives early and
        #    blocks until the slowest rank reaches the collective), not by the
        #    64-byte transfer, so switching primitives does not help.
        recv_counts_dev = torch.empty_like(send_counts_dev)
        dist.all_to_all_single(
            recv_counts_dev, send_counts_dev, group=self.device_group
        )

        recv_counts = recv_counts_dev.to("cpu").tolist()
        send_counts = send_counts_dev.to("cpu").tolist()

        # 3. Pack (token,k) pairs grouped by destination rank. Fused into a
        #    single Triton kernel (one launch instead of the flat_row divide +
        #    3-4 separate index_select gathers) to cut HT-path kernel calls.
        send_a1, send_weights, send_ids, send_scale = _pack_dispatch(
            pack_index, a1, topk_weights, topk_ids, scale, topk
        )

        # 4. Exchange packed activations / weights / ids / scales.
        #    Each dispatched row is one token routed to exactly ONE local expert,
        #    so the downstream aiter fused_moe sees a topk==1 layout: ids and
        #    weights must be 2-D (num_pairs, 1), NOT flattened to 1-D (fused_moe
        #    does `M, topk = topk_ids.shape` and requires 2 dims).
        recv_a1 = self._all_to_all(send_a1, send_counts, recv_counts)
        recv_weights = self._all_to_all(
            send_weights.unsqueeze(1), send_counts, recv_counts
        )  # (num_pairs, 1)
        recv_ids = self._all_to_all(
            send_ids.unsqueeze(1), send_counts, recv_counts
        )  # (num_pairs, 1)
        recv_scale = None
        if send_scale is not None:
            recv_scale = self._all_to_all(send_scale, send_counts, recv_counts)

        # expert_num_tokens is only read by the token_sort branch (aiter
        # fused_moe's num_local_tokens); the RCCL batched path never uses it
        # (it derives per-expert counts in resort_to_batched) and
        # _maybe_trim_dispatch_output ignores it. So skip the per-layer 1-elem
        # H2D copy when the batched impl is active — build the tensor only when a
        # consumer exists.
        from atom.utils import envs as _envs

        if _envs.ATOM_RCCL_MOE_IMPL in (
            "batched",
            "flydsl_batched_gemm",
            "triton_batched_gemm",
        ):
            expert_num_tokens = None
        else:
            expert_num_tokens = torch.tensor(
                [recv_a1.shape[0]], dtype=torch.int32, device=recv_a1.device
            )
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )

        self._routing_stack.append(
            _RoutingState(
                pack_index=pack_index,
                send_counts=send_counts,
                recv_counts=recv_counts,
                num_input_tokens=num_input_tokens,
                topk=topk,
            )
        )

        return (recv_a1, recv_scale, expert_tokens_meta, recv_ids, recv_weights)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        state = self._routing_stack.pop()

        # Inverse all_to_all: swap send/recv counts to return each pair home.
        home = self._all_to_all(
            fused_expert_output, state.recv_counts, state.send_counts
        )

        # Combine the per-(token,k) contributions back to token rows, weighted by
        # topk_weights. Per-TOKEN gather-sum (no atomics): home row p holds pair
        # pack_index[p], so the inverse permutation inv = argsort(pack_index),
        # reshaped [T, topk], gives — for each token t and its k-th expert — the
        # home row to read. Weights are topk_weights in natural [T, topk] order
        # (no gather needed). One program per token sums its topk rows and writes
        # once, replacing the per-pair atomic scatter.
        topk = state.topk
        T = state.num_input_tokens
        # Inverse permutation of pack_index (cheap scatter, not a full argsort):
        # inv[pack_index[p]] = p, so inv[q] = home row holding natural pair id q.
        src_pairs = state.pack_index
        P = src_pairs.shape[0]
        inv = torch.empty_like(src_pairs)
        inv[src_pairs] = torch.arange(P, device=src_pairs.device)
        rows = inv.view(T, topk)
        out = _combine_weighted_gather(home, rows, topk_weights.view(T, topk), T, topk)
        if output is not None:
            output.copy_(out)
            return output
        return out


# --------------------------------------------------------------------------- #
# LL (low-latency): uniform decode. Fixed shapes, no host sync, graph-safe.
# --------------------------------------------------------------------------- #


class RcclLLPrepareAndFinalize(RcclAll2AllBase, mk.FusedMoEPrepareAndFinalize):
    """Fixed cross-DP-unified capacity dispatch/combine. CUDA-graph capturable."""

    def _capacity(self, topk: int) -> int:
        """Fixed per-(src,dst) capacity from the cross-DP-unified graph_bs."""
        ctx = get_forward_context().context
        graph_bs = ctx.graph_bs
        return graph_bs * topk

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        assert (
            not apply_router_weight_on_input
        ), "rccl all2all does not support apply_router_weight_on_input=True"
        # Dummy warmup: balance routing so no rank/expert gets the whole batch.
        topk_ids = self._maybe_balance_warmup(topk_ids, num_experts)
        topk = topk_ids.shape[1]
        num_input_tokens = topk_ids.shape[0]
        ws = self.world_size
        C = self._capacity(topk)

        a1, scale = self._maybe_quant(a1)

        # ------------------------------------------------------------------ #
        # Fully static-shape pack (CUDA-graph safe). The send buffer is a fixed
        # [ws, C, hidden] grid: block ``d`` holds up to C tokens destined for
        # rank ``d``. Every shape is a compile-time constant (P, ws, C, hidden);
        # NO boolean-mask indexing and NO host sync. Overflow pairs (more than C
        # to one destination) are redirected to a per-pair dump slot at the
        # buffer tail and never transmitted.
        #
        # The whole slot-assignment + scatter is fused into two Triton launches
        # (see _ll_prepare_pack): a per-destination atomic-counter slot assign
        # (which also computes dest = id // num_local_experts inline, so no
        # dest_rank_for_expert / reshape / to(int64) op) and a single
        # gather/scatter of activations/weights/ids/scale — instead of the ~dozen
        # small ops (one-hot alloc + scatter_ + cumsum + gather + where + aranges
        # + per-tensor index_copy_) this used to run per step.
        # ------------------------------------------------------------------ #
        send_a1, send_w, send_id, send_scale, dst_slot, valid_pair = _ll_prepare_pack(
            topk_ids, a1, topk_weights, scale, topk, ws, C, self.num_local_experts
        )

        # Equal-split all_to_all over the [ws, C, hidden] head. all_to_all_single
        # with no split sizes divides dim 0 into ws equal blocks of C rows, so
        # block d goes to rank d. recv[s*C:(s+1)*C] is what source s sent us.
        # Keep ids/weights 2-D (num_rows, 1): each dispatched row is one token
        # routed to exactly ONE local expert (topk==1 layout). aiter fused_moe
        # does `M, topk = topk_ids.shape` and requires 2 dims.
        recv_a1 = self._all_to_all_equal(send_a1[: ws * C], C)
        recv_w = self._all_to_all_equal(send_w[: ws * C], C)  # (ws*C, 1)
        recv_id = self._all_to_all_equal(send_id[: ws * C], C)  # (ws*C, 1)
        recv_scale = None
        if send_scale is not None:
            recv_scale = self._all_to_all_equal(send_scale[: ws * C], C)

        # Valid rows are those with id != -1; count on device (no host sync).
        # Only the token_sort branch consumes this (aiter fused_moe's
        # num_local_tokens); the batched path derives its own per-expert counts
        # and _maybe_trim_dispatch_output ignores it, so skip the reduction there.
        from atom.utils import envs as _envs

        if _envs.ATOM_RCCL_MOE_IMPL in (
            "batched",
            "flydsl_batched_gemm",
            "triton_batched_gemm",
        ):
            expert_num_tokens = None
        else:
            expert_num_tokens = (recv_id != -1).to(torch.int32).sum().reshape(1)
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )

        # Routing state for finalize: dst_slot + valid_pair, both shape (P,) in
        # natural pair order (so token = pair_id // topk needs no saved index).
        self._routing_stack.append(
            _RoutingState(
                pack_index=torch.stack([dst_slot, valid_pair]),
                send_counts=None,
                recv_counts=None,
                num_input_tokens=num_input_tokens,
                topk=topk,
            )
        )
        return (recv_a1, recv_scale, expert_tokens_meta, recv_id, recv_w)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        state = self._routing_stack.pop()
        dst_slot, valid_pair = state.pack_index.unbind(0)  # each (P,)
        topk = state.topk
        num_input_tokens = state.num_input_tokens
        ws = self.world_size

        # Inverse equal-split all_to_all returns expert outputs to origin.
        # C is a Python int (rows // ws) -> static shape.
        C = fused_expert_output.shape[0] // ws
        home = self._all_to_all_equal(fused_expert_output, C)  # (ws*C, hidden)

        # Per-TOKEN gather-sum combine (no atomics). dst_slot / valid_pair are
        # (P,) in NATURAL pair order, so pair (t, k) is at flat index t*topk + k
        # -> both reshape straight to [T, topk] with no inverse permutation.
        # Overflow pairs point into the dropped dump region: clamp their gather
        # row in range and zero their weight via valid_pair. One program per token
        # sums its topk rows and writes once (replaces the per-pair atomic add).
        rows = torch.clamp(dst_slot, max=ws * C - 1).view(num_input_tokens, topk)
        w = (topk_weights.reshape(-1) * valid_pair.to(topk_weights.dtype)).view(
            num_input_tokens, topk
        )
        out = _combine_weighted_gather(home, rows, w, num_input_tokens, topk)
        if output is not None:
            output.copy_(out)
            return output
        return out


# --------------------------------------------------------------------------- #
# Dispatcher: held by the MoE layer, routes each step to HT or LL.
# --------------------------------------------------------------------------- #


class RcclPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Per-step dispatcher over the HT and LL native RCCL implementations."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        scale_dim: int,
        max_tokens_per_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
        in_dtype: torch.dtype,
        use_fp8_dispatch: bool = False,
        quant_type: Optional[QuantType] = None,
        ep_group=None,
    ):
        common = dict(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            scale_dim=scale_dim,
            max_tokens_per_rank=max_tokens_per_rank,
            num_local_experts=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            in_dtype=in_dtype,
            use_fp8_dispatch=use_fp8_dispatch,
            quant_type=quant_type,
            ep_group=ep_group,
        )
        self._ht = RcclHTPrepareAndFinalize(**common)
        self._ll = RcclLLPrepareAndFinalize(**common)

    def _impl(self):
        from atom.utils import envs

        # Debug/bring-up override: force the HT (variable-length) path for both
        # prefill and decode. HT is not CUDA-graph capturable, so pair this with
        # --enforce-eager / --level 0.
        if envs.ATOM_ALL2ALL_FORCE_HT:
            return self._ht
        ctx = get_forward_context().context
        if (not ctx.is_prefill) and getattr(ctx, "dp_uniform_decode", True):
            return self._ll
        return self._ht

    # ---- interface passthroughs (identical on both impls) ----------------- #

    def topk_indices_dtype(self):
        return self._ht.topk_indices_dtype()

    def max_num_tokens_per_rank(self):
        return self._ht.max_num_tokens_per_rank()

    def num_dispatchers(self):
        return self._ht.num_dispatchers()

    def output_is_reduced(self):
        return self._ht.output_is_reduced()

    def supports_async(self):
        return False

    # ---- routed calls ----------------------------------------------------- #

    def prepare(self, *args, **kwargs):
        return self._impl().prepare(*args, **kwargs)

    def finalize(self, *args, **kwargs):
        return self._impl().finalize(*args, **kwargs)
