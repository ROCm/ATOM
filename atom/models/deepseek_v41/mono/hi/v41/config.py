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


G = Geometry()
