# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared helpers for DSV4 DSA indexer backends (TileLang / aiter)."""

from __future__ import annotations

import torch


def make_causal_cu_seqlens(
    seq_len_q: int, seq_len_kv: int, compress_ratio: int, device: torch.device
):
    """Causal KV bounds per query position on compressed indices.

    For query at position ``p``, valid compressed groups are ``[0, (p + 1) // compress_ratio)``.
    """
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def mask_logits_with_cu_seqlens(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Apply per-query ``[ks, ke)`` bounds on compressed KV axis."""
    _seq_len, seq_len_kv = logits.shape
    p_idx = torch.arange(seq_len_kv, device=logits.device, dtype=torch.int32)[None, :]
    valid = (p_idx >= cu_seqlen_ks[:, None]) & (p_idx < cu_seqlen_ke[:, None])
    return logits.masked_fill(~valid, float("-inf"))


def infer_compress_ratio(seq_len_q: int, seq_len_kv: int, default: int = 4) -> int:
    if seq_len_kv <= 0:
        return default
    ratio = max(1, seq_len_q // seq_len_kv)
    return ratio
