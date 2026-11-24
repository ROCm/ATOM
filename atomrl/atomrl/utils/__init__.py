# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared trainer and controller utilities."""

from __future__ import annotations

from atomrl.utils.checkpoint import CheckpointManager
from atomrl.utils.distributed import (
    all_gather_tensors,
    broadcast_state_dict,
    get_rank,
    get_world_size,
)
from atomrl.utils.logging import setup_logging
from atomrl.utils.metrics import (
    MetricsTracker,
    compute_entropy,
    compute_kl_divergence,
)
from atomrl.utils.profiler import DistProfiler
from atomrl.utils.vocab_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)

__all__ = [
    "CheckpointManager",
    "DistProfiler",
    "MetricsTracker",
    "all_gather_tensors",
    "broadcast_state_dict",
    "compute_entropy",
    "compute_kl_divergence",
    "get_rank",
    "get_world_size",
    "setup_logging",
    "vocab_parallel_entropy",
    "vocab_parallel_log_probs_from_logits",
    "vocab_parallel_sum_pi_squared",
]
