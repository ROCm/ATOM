# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Flash attention utilities for Eagle3 diagonal attention.

Wraps aiter Triton ``flash_attn_varlen_func`` with token packing/unpacking.
Used by Eagle3's cached attention path for O(N)-memory causal attention
with LSE return, enabling online-softmax combination with diagonal scores.

Imports the same aiter Triton backend that Lumen's ``hf_patch.patch_sdpa()``
uses (``aiter.ops.triton.attention.mha``).  Keeps the aiter dependency
centralised — Eagle3 never imports aiter directly.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from aiter.ops.triton.attention.mha import flash_attn_varlen_func

# Backend selector: "aiter" (default) uses aiter's Triton flash attention.
# Anything else (e.g. "matmul" / "off") forces Eagle3 onto its matmul fallback
# in eagle3.py:_cached_attn_matmul. aiter triton _attn_fwd hits HIP "invalid
# argument" → HSA memory access fault on certain (B, T, H) combos on MI308
# (gfx942); flipping this env to "matmul" trades a few % of draft attention
# throughput for stability.
_BACKEND = os.environ.get("ATOMRL_DRAFT_FLASH_BACKEND", "aiter").lower()

_flash_attn_varlen = flash_attn_varlen_func if _BACKEND == "aiter" else None


def is_available() -> bool:
    """True when the configured backend enables AITER Triton attention."""
    return _flash_attn_varlen is not None


def varlen_causal_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal attention via aiter Triton flash attention, returning ``(out, lse)``.

    Args:
        q: ``[B, H, T, D]``
        k: ``[B, H, T, D]``
        v: ``[B, H, T, D]``
        attn_mask: ``[B, T]`` padding mask (1 = valid, 0 = pad)
        softmax_scale: scaling factor (default ``1 / sqrt(D)``)

    Returns:
        out: ``[B, H, T, D]``
        lse: ``[B, H, T]`` log-sum-exp of attention scores per query position.
             Padding positions are set to ``-inf``.
    """
    assert _flash_attn_varlen is not None, "aiter Triton flash attention not available"
    B, H, T, D = q.shape

    # [B, H, T, D] → [B, T, H, D] for varlen API
    q_bt = q.transpose(1, 2).contiguous()
    k_bt = k.transpose(1, 2).contiguous()
    v_bt = v.transpose(1, 2).contiguous()

    seqlens = attn_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item())
    valid_mask = attn_mask.bool()

    q_packed = q_bt[valid_mask]
    k_packed = k_bt[valid_mask]
    v_packed = v_bt[valid_mask]

    fa_result = _flash_attn_varlen(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        softmax_scale=softmax_scale,
        return_lse=True,
    )
    out_packed = fa_result[0]  # (total_tokens, H, D)
    lse_packed = fa_result[1]  # (total_tokens, H)

    # Unpack to [B, T, H, D]
    out = torch.zeros(
        B,
        T,
        H,
        D,
        dtype=out_packed.dtype,
        device=out_packed.device,
    )
    out[valid_mask] = out_packed
    out = out.transpose(1, 2)  # [B, H, T, D]

    # Unpack LSE to [B, T, H] then permute to [B, H, T]
    lse = torch.full(
        (B, T, H),
        float("-inf"),
        dtype=torch.float32,
        device=lse_packed.device,
    )
    lse[valid_mask] = lse_packed
    lse = lse.permute(0, 2, 1)  # [B, H, T]

    return out, lse
