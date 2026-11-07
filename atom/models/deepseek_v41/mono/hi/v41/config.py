# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1-Flash geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    hidden: int = 5120
    layers: int = 40
    vocab: int = 129280
    eps: float = 1e-20
    hc_mult: int = 4
    # attention: one KV head of head_dim, low-rank Q, grouped low-rank output
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora: int = 1280
    o_lora: int = 1024
    o_groups: int = 8
    window: int = 128
    index_topk: int = 512
    # MoE
    experts: int = 384
    topk: int = 6
    moe_inter: int = 2304
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    @property
    def stream(self) -> int:
        return self.hc_mult * self.hidden  # 20480: the flattened residual stream

    @property
    def mix_width(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult  # 24: pre 4 | post 4 | comb 16

    @property
    def group_width(self) -> int:
        return self.heads * self.head_dim // self.o_groups  # 4096: wo_a's K per group


G = Geometry()

_CONFIG_KEYS = {
    "dim": "hidden",
    "n_layers": "layers",
    "vocab_size": "vocab",
    "norm_eps": "eps",
    "hc_mult": "hc_mult",
    "n_heads": "heads",
    "head_dim": "head_dim",
    "rope_head_dim": "rope_dim",
    "q_lora_rank": "q_lora",
    "o_lora_rank": "o_lora",
    "o_groups": "o_groups",
    "window_size": "window",
    "index_topk": "index_topk",
    "n_routed_experts": "experts",
    "n_activated_experts": "topk",
    "moe_inter_dim": "moe_inter",
    "route_scale": "route_scale",
    "swiglu_limit": "swiglu_limit",
}


