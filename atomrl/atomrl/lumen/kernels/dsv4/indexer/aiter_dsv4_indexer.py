# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Batched DSV4 DSA indexer via aiter ``dsv4_indexer`` ops.

Matches the TileLang batched layout used by ``V4Indexer``:
    q: [seqlen, batch, heads, dim] bf16
    k: [seqlen_kv, batch, dim] bf16
    weights: [seqlen, batch, heads] fp32
"""

from __future__ import annotations

import torch

from atomrl.lumen.ops.dsv4.indexer_utils import (
    infer_compress_ratio,
    mask_logits_with_cu_seqlens,
)


def _indexer_fwd_single(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    from aiter.ops.triton.attention.dsv4_indexer import indexer_fwd

    # q: [S, H, D], k: [P, D], weights: [S, H]
    scores = indexer_fwd(q, k, weights, compress_ratio)
    return mask_logits_with_cu_seqlens(scores, cu_seqlen_ks, cu_seqlen_ke)


def batched_indexer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Return logits ``[batch, seqlen, seqlen_kv]`` fp32."""
    seqlen, batch, _heads, _dim = q.shape
    seq_len_kv = k.shape[0]
    compress_ratio = infer_compress_ratio(seqlen, seq_len_kv)

    all_logits = torch.empty(
        [batch, seqlen, seq_len_kv], device=q.device, dtype=torch.float32
    )
    for b in range(batch):
        all_logits[b] = _indexer_fwd_single(
            q[:, b, :, :].contiguous(),
            k[:, b, :].contiguous(),
            weights[:, b, :].contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
            compress_ratio,
        )
    return all_logits


def _indexer_bwd_single(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    grad_scores: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.attention.dsv4_indexer import indexer_bwd

    return indexer_bwd(q, k, weights, grad_scores, compress_ratio)


def _scatter_topk_grad_scores(
    grad_topk: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_len_kv: int,
) -> torch.Tensor:
    """Expand ``[seqlen, topk]`` grads into dense ``[seqlen, seq_len_kv]``."""
    seqlen, _topk = grad_topk.shape
    grad_dense = torch.zeros(
        seqlen, seq_len_kv, device=grad_topk.device, dtype=grad_topk.dtype
    )
    valid = topk_indices >= 0
    safe_idx = topk_indices.clamp(min=0).long()
    grad_dense.scatter_add_(1, safe_idx, grad_topk * valid.to(grad_topk.dtype))
    return grad_dense


def batched_indexer_bwd(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for batched SBHD indexer (matches TileLang batched bwd layout)."""
    seqlen, batch, _heads, dim = index_q.shape
    seq_len_kv = index_k.shape[0]
    compress_ratio = infer_compress_ratio(seqlen, seq_len_kv)

    grad_q = torch.zeros_like(index_q, dtype=torch.float32)
    grad_w = torch.zeros_like(weights, dtype=torch.float32)
    grad_k = torch.zeros(
        [seq_len_kv, batch, dim], device=index_k.device, dtype=torch.float32
    )

    for b in range(batch):
        q_b = index_q[:, b, :, :].contiguous()
        k_b = index_k[:, b, :].contiguous()
        w_b = weights[:, b, :].contiguous()
        grad_dense = _scatter_topk_grad_scores(
            grad_scores[b], topk_indices[b], seq_len_kv
        )

        dq, dk, dw = _indexer_bwd_single(q_b, k_b, w_b, grad_dense, compress_ratio)
        grad_q[:, b, :, :] = dq.to(index_q.dtype)
        grad_w[:, b, :] = dw.to(weights.dtype)
        grad_k[:, b, :] = dk.to(index_k.dtype)

    return grad_q, grad_w, grad_k
