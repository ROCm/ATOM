# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DSV4 sparse MLA via aiter Triton kernels (MI308X-optimised).

Layout adapter: Lumen uses [B,S,H,D] batched tensors; aiter expects flat [N,H,D].
Calls ``sparse_mla_dsv4_train`` from the installed official AITER package (fwd + bwd autograd).
"""

from __future__ import annotations

import torch
from aiter.ops.triton.attention.sparse_mla_dsv4_train import sparse_mla_dsv4_train


def _to_flat(q, kv, topk_idxs):
    """Convert batched layout to aiter flat layout with global KV indices."""
    B, S, H, D = q.shape
    S_kv = kv.shape[1]
    N = B * S
    q_flat = q.reshape(N, H, D).contiguous()
    kv_flat = kv.reshape(B * S_kv, D).contiguous()
    idx = topk_idxs.reshape(N, -1).to(torch.int32).contiguous()
    if B > 1:
        offsets = (torch.arange(B, device=idx.device, dtype=idx.dtype) * S_kv).view(
            B, 1, 1
        )
        idx_b = idx.view(B, S, -1) + offsets
        valid = idx_b >= 0
        idx = torch.where(valid, idx_b, torch.full_like(idx_b, -1)).reshape(N, -1)
    return q_flat, kv_flat, idx, (B, S, H, D)


def sparse_attn_triton(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Drop-in replacement for sparse_attn_tilelang with identical signature."""
    q_flat, kv_flat, idx_flat, shape = _to_flat(q, kv, topk_idxs)
    sink = attn_sink
    if sink is not None and sink.dtype != torch.float32:
        sink = sink.float()
    out_flat = sparse_mla_dsv4_train(q_flat, kv_flat, sink, idx_flat, sm_scale)
    B, S, H, D = shape
    # Clone so downstream in-place RoPE (deepseek_v4.apply_rotary_emb) does not
    # mutate the tensor saved by SparseMLADSV4Function for backward.
    return out_flat.reshape(B, S, H, D).clone()
