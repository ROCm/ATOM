# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-397B-A17B dimensions and one rank's tensor-parallel shard of a decoder layer.

The model alternates three Gated DeltaNet (linear attention) layers with one gated full-attention
layer; every layer ends in a 512-expert top-10 MoE with a sigmoid-gated shared expert.
"""

from __future__ import annotations

from dataclasses import dataclass

HIDDEN = 4096
FULL_ATTENTION_INTERVAL = 4
EPS = 1.0e-6
FP8_MAX = 448.0

# Gated DeltaNet
LINEAR_K_HEADS = 16
LINEAR_V_HEADS = 64
LINEAR_K_DIM = 128
LINEAR_V_DIM = 128
CONV_KERNEL = 4

# gated full attention
HEADS = 32
KV_HEADS = 2
HEAD_DIM = 256
ROPE_DIM = 64  # partial_rotary_factor 0.25
ROPE_THETA = 1.0e7

# MoE
N_EXPERTS = 512
TOP_K = 10
INTER = 1024
SHARED_INTER = 1024


# rows a decode step holds at most: one launch, the kernel's samples
MAX_TOKENS = 16
# verify tokens a request: pa_decode_gluon's limit, and FlyDSL's (ATOM_PA_FLYDSL) past it
GLUON_VERIFY_TOKENS = 4
MAX_VERIFY_TOKENS = 6


def is_full_attention(layer_id: int) -> bool:
    return (layer_id + 1) % FULL_ATTENTION_INTERVAL == 0


@dataclass(frozen=True)
class Q35Config:
    """Rank ``rank`` of a ``tp``-way shard. Dimension fields are model-global; properties give the shard."""

    tp: int = 4
    rank: int = 0
    hidden: int = HIDDEN
    linear_k_heads: int = LINEAR_K_HEADS
    linear_v_heads: int = LINEAR_V_HEADS
    linear_k_dim: int = LINEAR_K_DIM
    linear_v_dim: int = LINEAR_V_DIM
    conv_kernel: int = CONV_KERNEL
    heads: int = HEADS
    kv_heads: int = KV_HEADS
    head_dim: int = HEAD_DIM
    rope_dim: int = ROPE_DIM
    rope_theta: float = ROPE_THETA
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER
    shared_inter: int = SHARED_INTER
    eps: float = EPS
    # checkpoint format: "fp8" (FP8 128 x 128 matrices, the shared expert fused into the bank) or "mxfp4"
    # (amd/Qwen3.5-397B-A17B-MXFP4: MXFP4 routed experts on MXFP4 activations, A4W4; BF16 projections,
    # a separate BF16 shared expert)
    weights: str = "fp8"
    # ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION: the original attention's fused q / k norm + RoPE +
    # cache-quant kernel (one bf16 rounding; its KV and o_proj input quants), which the kernel follows
    qk_fusion: bool = False
    # ATOM_PA_FLYDSL: follow aiter's FlyDSL paged decode instead of pa_decode_gluon
    pa_flydsl: bool = False

    # --- Gated DeltaNet shard
    @property
    def hk(self) -> int:
        return self.linear_k_heads // self.tp

    @property
    def hv(self) -> int:
        return self.linear_v_heads // self.tp

    @property
    def key_dim(self) -> int:
        return self.hk * self.linear_k_dim

    @property
    def value_dim(self) -> int:
        return self.hv * self.linear_v_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the local ``[q | k | v]`` conv input."""
        return 2 * self.key_dim + self.value_dim

    # --- full-attention shard
    @property
    def local_heads(self) -> int:
        return self.heads // self.tp

    @property
    def local_kv_heads(self) -> int:
        """K/V heads are replicated once ``tp`` exceeds them: each rank keeps one."""
        return max(1, self.kv_heads // self.tp)

    # --- MoE shard (tensor-parallel over the intermediate dim)
    @property
    def local_inter(self) -> int:
        return self.inter // self.tp

    @property
    def shared_expert(self) -> int:
        """The shared expert is the last entry of the expert bank, as ATOM fuses it."""
        return self.n_experts
