# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Mixture-of-experts utilities: R3 routing, expert parallel, diagnostics."""

from __future__ import annotations

from atomrl.moe.expert_parallel import ExpertParallelManager
from atomrl.moe.moe_utils import (
    check_expert_utilization,
    compute_load_balance_loss,
    compute_router_entropy,
    iter_megatron_routers,
    megatron_record_router_logits,
    megatron_replay_router_logits,
)
from atomrl.moe.r3_manager import R3Manager
from atomrl.moe.r3_scope import (
    SUPPORTED_R3_GENERATION_BACKEND,
    SUPPORTED_R3_REPLAY_MODE,
    assert_supported_r3,
    r3_fields_from_config,
)
from atomrl.moe.router_precision import enable_fp32_moe_router, fp32_router_enabled
from atomrl.moe.router_recorder import RouterRecorder
from atomrl.moe.router_replayer import RouterReplayer

__all__ = [
    "SUPPORTED_R3_GENERATION_BACKEND",
    "SUPPORTED_R3_REPLAY_MODE",
    "ExpertParallelManager",
    "R3Manager",
    "RouterRecorder",
    "RouterReplayer",
    "assert_supported_r3",
    "check_expert_utilization",
    "compute_load_balance_loss",
    "compute_router_entropy",
    "enable_fp32_moe_router",
    "fp32_router_enabled",
    "iter_megatron_routers",
    "megatron_record_router_logits",
    "megatron_replay_router_logits",
    "r3_fields_from_config",
]
