# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared attention helpers usable from both standalone (model_ops) and
plugin (plugin/) code paths.

Lives under model_ops/ deliberately: server / standalone code must not
depend on plugin/, but plugin/ may depend on model_ops/. Anything shared
between the two layers belongs here.
"""

import torch

_CK_MAX_HEAD_DIM = 256


def _sdpa_varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k,
                      softmax_scale, causal, return_lse=False, out=None):
    """PyTorch SDPA fallback for varlen attention when CK is unsupported
    (e.g. head_dim > 256).  Handles GQA head expansion automatically.

    Returns (out, lse) when return_lse=True, else just out.
    """
    import torch.nn.functional as F

    num_seqs = cu_seqlens_q.shape[0] - 1
    if out is None:
        out = torch.empty_like(q)
    total_q = q.shape[0]
    num_heads_q = q.shape[1]

    if return_lse:
        lse = torch.empty(num_heads_q, total_q, dtype=torch.float32,
                          device=q.device)

    # Expand KV heads once, then execute a single batched SDPA over padded
    # tensors instead of launching one SDPA per sequence from Python.
    num_kv_heads = k.shape[1]
    if num_heads_q != num_kv_heads:
        # GQA expansion via repeat_interleave assumes num_heads_q is an
        # integer multiple of num_kv_heads. Validate explicitly so a
        # malformed model config fails loudly here instead of silently
        # producing the wrong head mapping downstream.
        if num_kv_heads == 0 or num_heads_q % num_kv_heads != 0:
            raise ValueError(
                f"Invalid GQA head configuration in _sdpa_varlen_attn: "
                f"num_heads_q ({num_heads_q}) must be a positive multiple "
                f"of num_kv_heads ({num_kv_heads})."
            )
        rep = num_heads_q // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    q_starts = cu_seqlens_q[:-1].tolist()
    q_ends = cu_seqlens_q[1:].tolist()
    k_starts = cu_seqlens_k[:-1].tolist()
    k_ends = cu_seqlens_k[1:].tolist()
    q_lens = [q_e - q_s for q_s, q_e in zip(q_starts, q_ends)]
    k_lens = [k_e - k_s for k_s, k_e in zip(k_starts, k_ends)]
    max_q = max(q_lens) if q_lens else 0
    max_k = max(k_lens) if k_lens else 0
    if max_q == 0 or max_k == 0:
        if return_lse:
            lse.zero_()
            return out, lse
        return out
    batch_q = q.new_zeros((num_seqs, num_heads_q, max_q, q.shape[-1]))
    batch_k = k.new_zeros((num_seqs, num_heads_q, max_k, k.shape[-1]))
    batch_v = v.new_zeros((num_seqs, num_heads_q, max_k, v.shape[-1]))
    attn_mask = q.new_full((num_seqs, 1, max_q, max_k), float("-inf"))
    for i, (sq_s, sq_e, sk_s, sk_e, q_len, k_len) in enumerate(
        zip(q_starts, q_ends, k_starts, k_ends, q_lens, k_lens)
    ):
        if q_len == 0 or k_len == 0:
            continue
        batch_q[i, :, :q_len] = q[sq_s:sq_e].transpose(0, 1)
        batch_k[i, :, :k_len] = k[sk_s:sk_e].transpose(0, 1)
        batch_v[i, :, :k_len] = v[sk_s:sk_e].transpose(0, 1)
        attn_mask[i, :, :q_len, :k_len] = 0
        if causal:
            causal_mask = torch.triu(
                torch.full((q_len, k_len), float("-inf"),
                           dtype=q.dtype, device=q.device),
                diagonal=1,
            )
            attn_mask[i, :, :q_len, :k_len] = (
                attn_mask[i, :, :q_len, :k_len] + causal_mask
            )
    batch_out = F.scaled_dot_product_attention(
        batch_q,
        batch_k,
        batch_v,
        attn_mask=attn_mask,
        scale=softmax_scale,
        is_causal=False,
    )
    for i, (sq_s, sq_e, q_len) in enumerate(zip(q_starts, q_ends, q_lens)):
        if q_len == 0:
            continue
        out[sq_s:sq_e] = batch_out[i, :, :q_len].transpose(0, 1)
    if return_lse:
        scores = torch.matmul(
            batch_q.to(torch.float32) * softmax_scale,
            batch_k.transpose(-1, -2).to(torch.float32),
        )
        scores = scores + attn_mask.to(torch.float32)
        batch_lse = torch.logsumexp(scores, dim=-1)
        for i, (sq_s, sq_e, q_len) in enumerate(zip(q_starts, q_ends, q_lens)):
            if q_len == 0:
                continue
            lse[:, sq_s:sq_e] = batch_lse[i, :, :q_len]
    return (out, lse) if return_lse else out
