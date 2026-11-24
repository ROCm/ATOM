# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/utils/megatron/tensor_parallel.py

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/utils/megatron/tensor_parallel.py

"""Vocab-parallel computation utilities for Megatron tensor parallelism."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def vocab_parallel_log_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Per-token log-probs when logits are TP-sharded across vocab dimension.

    Requires megatron-core for vocab_parallel_cross_entropy.

    Args:
        logits: (total_nnz, vocab_size // tp_size) sharded logits.
        labels: (total_nnz,) target indices.

    Returns:
        (total_nnz,) log-probabilities.
    """
    try:
        from megatron.core import tensor_parallel

        return -tensor_parallel.vocab_parallel_cross_entropy(
            vocab_parallel_logits=logits, target=labels
        )
    except ImportError:
        raise ImportError("vocab_parallel_log_probs_from_logits requires megatron-core")


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when logits are TP-sharded across vocab dimension.

    Uses numerically stable computation with all-reduce across TP group.

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size).

    Returns:
        Entropy values (...,).
    """
    try:
        from megatron.core import mpu
    except ImportError:
        raise ImportError("vocab_parallel_entropy requires megatron-core")

    tp_group = mpu.get_tensor_model_parallel_group()

    # Numerical stability: subtract max
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = vocab_parallel_logits - logits_max

    # Compute local exp and sum
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=tp_group)

    # log(sum(exp)) = log(sum_exp) + max
    log_sum_exp = sum_exp.log() + logits_max

    # sum(p * logits) = sum(exp/Z * logits)
    weighted_sum = (exp_shifted * vocab_parallel_logits).sum(dim=-1, keepdim=True)
    dist.all_reduce(weighted_sum, group=tp_group)
    weighted_sum = weighted_sum / sum_exp

    entropy = log_sum_exp.squeeze(-1) - weighted_sum.squeeze(-1)
    return entropy


def vocab_parallel_sum_pi_squared(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute sum(pi^2) when logits are TP-sharded.

    Used by optimal-baseline advantage estimators.
    Non-destructive: does not mutate input logits.

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size).

    Returns:
        Sum-pi-squared values (...,).
    """
    try:
        from megatron.core import mpu
    except ImportError:
        raise ImportError("vocab_parallel_sum_pi_squared requires megatron-core")

    tp_group = mpu.get_tensor_model_parallel_group()

    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = vocab_parallel_logits - logits_max

    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=tp_group)

    sum_exp_squared = exp_shifted.pow(2).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp_squared, group=tp_group)

    return (sum_exp_squared / sum_exp.pow(2)).squeeze(dim=-1)
