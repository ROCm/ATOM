# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time rank-local widths for Qwen3.8-Flash-Next mono kernels."""

from dataclasses import dataclass

from atom.models.qwen4_exp_mono.shapes import (
    EXPERTS,
    GDN_HEAD_DIM,
    GDN_KEY_HEADS,
    GDN_VALUE_HEADS,
    HIDDEN,
    INTERMEDIATE,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_KV_HEADS,
    SUPPORTED_TP,
)
from atom.mono.plan.shard import Shard, ShardError


@dataclass(frozen=True)
class Dims(Shard):
    """One rank's physical widths for TP1 or TP8.

    Dense projections follow the model TP group. Routed MoE weights use
    expert-id ownership at TP8: each rank owns complete I=640 experts, so the
    128-column FP8 scale blocks are never split.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tp not in SUPPORTED_TP:
            raise ShardError(f"Qwen mono has no TP {self.tp} shape")
        self.split(GDN_KEY_HEADS, "GDN key heads")
        self.split(GDN_VALUE_HEADS, "GDN value heads")
        self.split(QSA_HEADS, "QSA query heads")
        self.split(EXPERTS, "routed experts")
        if QSA_KV_HEADS < self.tp and self.tp % QSA_KV_HEADS:
            raise ShardError(
                f"TP {self.tp} is not a multiple of {QSA_KV_HEADS} QSA KV heads"
            )

    @property
    def gdn_key_heads(self) -> int:
        return GDN_KEY_HEADS // self.tp

    @property
    def gdn_value_heads(self) -> int:
        return GDN_VALUE_HEADS // self.tp

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_key_heads * GDN_HEAD_DIM

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_value_heads * GDN_HEAD_DIM

    @property
    def gdn_conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def gdn_proj_dim(self) -> int:
        return self.gdn_conv_dim + self.gdn_value_dim + 2 * self.gdn_value_heads

    @property
    def qsa_heads(self) -> int:
        return QSA_HEADS // self.tp

    @property
    def qsa_q_size(self) -> int:
        return self.qsa_heads * QSA_HEAD_DIM

    @property
    def qsa_kv_heads(self) -> int:
        return QSA_KV_HEADS // self.tp if QSA_KV_HEADS >= self.tp else 1

    @property
    def qsa_kv_size(self) -> int:
        return self.qsa_kv_heads * QSA_HEAD_DIM

    @property
    def qsa_qkv_size(self) -> int:
        # Gate and Q are both query-width; K and V are both KV-width.
        return 2 * self.qsa_q_size + 2 * self.qsa_kv_size

    @property
    def qsa_kv_replicas(self) -> int:
        return self.tp // QSA_KV_HEADS if QSA_KV_HEADS < self.tp else 1

    @property
    def shared_intermediate(self) -> int:
        # The standalone BF16 shared expert follows dense TP sharding.
        return INTERMEDIATE // self.tp

    @property
    def local_experts(self) -> int:
        return EXPERTS if self.tp == 1 else EXPERTS // self.tp

    @property
    def routed_intermediate(self) -> int:
        # Expert-id ownership keeps every local routed expert physically whole.
        return INTERMEDIATE

    @property
    def expert_layout(self) -> str:
        return "replicated" if self.tp == 1 else "expert_id"

    @property
    def hidden(self) -> int:
        return HIDDEN
