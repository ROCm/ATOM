# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A Qwen3.5 decoder layer as the kernel builds for it (``LayerWeights``: its TP shard's shape and
kind) and the RoPE tables the full-attention layers read."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.models.qwen3_5.mono.config import Q35Config


@dataclass
class LayerWeights:
    cfg: Q35Config
    full: bool  # gated full attention; else Gated DeltaNet


def rope_tables(rotary_emb, cfg: Q35Config, max_pos: int, spare: int = 0):
    """ATOM's RoPE caches as f32 cos, sin [max_pos + spare, rope_dim / 2], or None when they do not
    cover ``max_pos``. The kernel reads rows unguarded, so the ``spare`` rows past the caches repeat the
    last one (tokens there are dropped)."""
    cos = rotary_emb.cos_cache.reshape(rotary_emb.cos_cache.shape[0], -1)
    sin = rotary_emb.sin_cache.reshape(rotary_emb.sin_cache.shape[0], -1)
    if cos.shape[0] < max_pos or cos.shape[1] != cfg.rope_dim // 2:
        return None
    rows = max_pos + spare

    def table(t):
        t = t[:rows].float()
        pad = rows - t.shape[0]
        if pad:
            t = torch.cat([t, t[-1:].expand(pad, -1)])
        return t.contiguous()

    return table(cos), table(sin)
