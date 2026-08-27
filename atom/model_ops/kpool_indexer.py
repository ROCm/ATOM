# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""k-pool compressed DSA indexer (GLM-5.3-Flash / ``glm5_next``).

GLM-5.3-Flash keeps DeepSeek Sparse Attention but scores **pools** of ``index_kpool``
consecutive keys instead of individual keys. Each pool is collapsed to one candidate
key by a learned softmax over the pool, so the indexer only has to rank
``kv_len / index_kpool`` candidates; the selected pools are then expanded back into
raw token indices. With ``index_topk=2048`` and ``index_kpool=4`` that is 512 ranked
candidates covering 2048 tokens.

Relationship to what ATOM already has:

* DeepSeek-V4's ``Compressor`` (``sparse_attn_v4.py`` / ``deepseek_v4.py``) performs
  the same learned gated pooling at ``compress_ratio=4`` and also carries a per-slot
  ``ape`` term. It differs in three ways that matter here: V4 pools with **overlapping**
  windows (``coff=2``), applies RoPE inside the pooling, and folds the pooled result
  into the KV cache. GLM-5.3's k-pool is non-overlapping, NoPE (the whole text model is
  NoPE — ``qk_rope_head_dim == 0``), and is used purely to rank candidates.
* The scoring/top-k tail (``weights_proj`` per-head weighting, ReLU'd scores, top-k,
  ``-1`` sentinel convention) matches the existing DSA indexers.

The implementation below is the dense reference: it mirrors
``transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextIndexer`` and has been
verified to select exactly the same token indices on real GLM-5.3-Flash layer-3
weights (see ``tests/model_ops/test_kpool_indexer.py`` and the parity procedure in
``recipes/GLM-5.3-Flash.md``). A paged/ragged variant that reads pooled state straight
out of the KV cache -- the shape ATOM's scheduler actually wants -- is the follow-up;
this function is the correctness oracle it must reproduce.
"""

import torch
import torch.nn.functional as F

__all__ = ["build_kpools", "kpool_topk_indices"]


def build_kpools(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_keys: torch.Tensor,
    ape: torch.Tensor,
    kpool: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse consecutive keys into learned-weighted pool candidates.

    Pooling starts at the first *real* token rather than slot 0, so a left-padded
    ``[P, P, A, B, C, D]`` pools identically to ``[A, B, C, D]``.

    Args:
        keys:        [B, S, head_dim]  normalised indexer keys (``k_norm(wk(x))``)
        gate_scores: [B, S, head_dim]  ``index_kpool_compress_gate`` projection of x
        valid_keys:  [B, S]            real (non-padding) key positions
        ape:         [kpool, head_dim] ``index_kpool_compress_ape``, per-slot bias
        kpool:       pool size (``index_kpool``)

    Returns:
        pool_keys:   [B, P, head_dim] pooled candidate keys
        pool_indices:[B, P, kpool]    raw token index per pool slot, ``-1`` where invalid
        pool_valid:  [B, P]           pool is complete (all ``kpool`` slots real)
    """
    b, s = keys.shape[:2]
    device = keys.device
    n_pools = (s + kpool - 1) // kpool

    # Anchor pools at the first real token; fully-padded rows anchor past the end so
    # every slot lands out of range and the pool is dropped as invalid.
    first_key = torch.where(
        valid_keys.any(-1),
        valid_keys.long().argmax(-1),
        torch.full((b,), s, dtype=torch.long, device=device),
    )
    offsets = torch.arange(n_pools * kpool, device=device).view(1, n_pools, kpool)
    pool_indices = first_key[:, None, None] + offsets

    bidx = torch.arange(b, device=device)[:, None, None]
    safe = pool_indices.clamp(0, s - 1)
    grouped_keys = keys[bidx, safe]
    grouped_gate = gate_scores[bidx, safe]
    grouped_valid = valid_keys[bidx, safe] & (pool_indices < s)

    # Only complete pools are selectable; partial tails are handled by
    # `append_visible_tail` in the caller.
    pool_valid = grouped_valid.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid, -1)

    logits = grouped_gate.float() + ape.float()[None, None]
    logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
    # nan_to_num covers pools whose slots are all invalid (softmax over all -inf).
    probs = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    pool_keys = (probs * grouped_keys).sum(dim=2)

    keep = pool_valid.any(0)
    return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]


def kpool_topk_indices(
    q: torch.Tensor,
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    head_weights: torch.Tensor,
    valid_keys: torch.Tensor,
    visible: torch.Tensor,
    ape: torch.Tensor,
    index_topk: int,
    kpool: int,
    softmax_scale: float,
    always_select_tail: bool = True,
) -> torch.Tensor:
    """Select the DSA token indices for each query via pooled scoring.

    Args:
        q:            [B, S, H, head_dim] indexer queries (``wq_b(q_resid)``)
        keys:         [B, S, head_dim]    normalised indexer keys
        gate_scores:  [B, S, head_dim]    k-pool compression gate projection
        head_weights: [B, S, H]           ``weights_proj`` output, pre-scaling
        valid_keys:   [B, S]              real key positions
        visible:      [B, S, S_kv]        causal AND padding visibility per query
        ape:          [kpool, head_dim]
        index_topk:   token budget (``index_topk``); pools selected = topk // kpool
        kpool:        pool size
        softmax_scale: ``head_dim ** -0.5``
        always_select_tail: append the current incomplete pool as raw indices

    Returns:
        int32 ``[B, S, index_topk (+ kpool - 1 if tail)]`` token indices, ``-1`` = unused.
    """
    b, s = keys.shape[:2]
    device = keys.device

    pool_keys, pool_indices, pool_valid = build_kpools(
        keys, gate_scores, valid_keys, ape, kpool
    )

    # Score every query against pooled candidates, then collapse heads with the
    # learned per-head weights: [B,S,1,H] @ [B,S,H,P] -> [B,S,P].
    scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores * softmax_scale)
    weights = head_weights.float() * (q.shape[-2] ** -0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    # A pool is selectable only when its final token is visible to the query.
    kv_len = visible.shape[-1]
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible.gather(-1, pool_end[:, None, :].expand(b, s, -1))
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(
        ~valid_candidates, torch.finfo(index_scores.dtype).min
    )

    select_k = min(index_topk // kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    bidx = torch.arange(b, device=device)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[bidx, selected]

    topk_indices = selected_indices.flatten(-2)
    topk_indices = topk_indices.masked_fill(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
    )

    width = index_topk
    if always_select_tail:
        topk_indices = _append_visible_tail(
            topk_indices, visible, valid_keys, kpool, kv_len
        )
        width += kpool - 1

    topk_indices = F.pad(topk_indices, (0, width - topk_indices.shape[-1]), value=-1)
    return topk_indices[..., :width].to(torch.int32)


def _append_visible_tail(
    topk_indices: torch.Tensor,
    visible: torch.Tensor,
    valid_keys: torch.Tensor,
    kpool: int,
    kv_len: int,
) -> torch.Tensor:
    """Append the current incomplete pool as raw token indices.

    With ``kpool=4`` and visible keys ``[A B C D E F]``, the complete pool ``[A B C D]``
    is selectable above while ``[E F]`` is not; this appends ``E, F`` directly so the
    most recent tokens are never dropped.
    """
    max_tail = kpool - 1
    if max_tail == 0:
        return topk_indices

    b = visible.shape[0]
    device = visible.device
    first_key = torch.where(
        valid_keys.any(-1),
        valid_keys.long().argmax(-1),
        torch.full((b,), kv_len, dtype=torch.long, device=device),
    )
    visible_count = visible.long().sum(-1)
    tail_count = visible_count.remainder(kpool)
    tail_offsets = torch.arange(max_tail, device=device)

    tail_start = first_key[:, None] + visible_count - tail_count
    tail_indices = tail_start[..., None] + tail_offsets
    tail_valid = (
        tail_offsets[None, None, :] < tail_count[..., None]
    ) & tail_indices.lt(kv_len)
    tail_visible = visible.gather(-1, tail_indices.clamp(0, kv_len - 1))
    tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)

    return torch.cat([topk_indices, tail_indices], dim=-1)
