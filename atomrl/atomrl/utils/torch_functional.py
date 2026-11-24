# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/utils/torch_functional.py

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
# Original: verl/utils/torch_functional.py
# Additional utilities: entropy_from_logits_with_chunking, masked_sum,
# masked_mean, masked_whiten, logprobs_from_logits_fused, clip_by_value

"""Small torch utilities for RL training: log-probs, entropy, sum-pi-squared,
masked reductions, advantage whitening, and fused log-prob variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::logprobs_from_logits_naive
def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities via log_softmax + gather.

    Args:
        logits: (..., vocab_size).
        labels: (...,) indices.

    Returns:
        Log-probabilities with shape (...,).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::entropy_from_logits
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy H(p) = -sum(p * log p) from unnormalised logits.

    Uses the numerically stable formula:
        H = logsumexp(logits) - sum(softmax(logits) * logits)

    Args:
        logits: (..., vocab_size).

    Returns:
        Entropy values with shape (...,).
    """
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::calculate_sum_pi_squared_from_logits
def calculate_sum_pi_squared_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute sum of squared probabilities from logits.

    Formula: sum(pi^2) = exp(logsumexp(2*logits) - 2*logsumexp(logits))

    Used for optimal-baseline advantage estimators.

    Args:
        logits: (..., vocab_size).

    Returns:
        Sum-pi-squared with shape (...,).
    """
    return torch.exp(
        torch.logsumexp(2.0 * logits, dim=-1) - 2.0 * torch.logsumexp(logits, dim=-1)
    )


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::entropy_from_logits (chunked variant)
def entropy_from_logits_with_chunking(
    logits: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    """Memory-efficient entropy calculation using chunked processing.

    Processes the batch in chunks to reduce peak memory usage.

    Args:
        logits: (batch_size, vocab_size) or (..., vocab_size).
        chunk_size: Number of samples to process at once.

    Returns:
        Entropy values with shape (...,).
    """
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = F.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(
            pd_chunk * logits_chunk, dim=-1
        )
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::masked_sum
def masked_sum(
    values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None
) -> torch.Tensor:
    """Compute sum of tensor values where mask is True.

    NaN values outside the mask are replaced with zeros.

    Args:
        values: Tensor of values to sum.
        mask: Boolean or float mask (1 = include).
        axis: Dimension(s) over which to sum; ``None`` sums all elements.

    Returns:
        Summed tensor.
    """
    valid_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (valid_values * mask).sum(dim=axis)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::masked_mean
def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None
) -> torch.Tensor:
    """Compute mean of values over elements selected by mask.

    Args:
        values: Tensor of values to average.
        mask: Boolean or float mask (1 = include).
        axis: Dimension(s) over which to average; ``None`` averages all.

    Returns:
        Mean tensor.  Denominator is clamped to at least 1 to avoid division
        by zero when the mask is empty.
    """
    valid_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (valid_values * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1.0)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::masked_whiten
def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whitening (zero-mean, unit-var) within masked region.

    Used for advantage normalization.

    Args:
        values: Tensor to whiten.
        mask: Boolean or float mask (1 = include).
        shift_mean: If True, subtract the mean (standard whitening).
            If False, only divide by std.

    Returns:
        Whitened tensor (same shape as values).
    """
    valid = values[mask.bool()]
    if valid.numel() < 2:
        return values
    mean = valid.mean() if shift_mean else 0.0
    std = valid.std().clamp(min=1e-8)
    return ((values - mean) / std) * mask.float()


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::logprobs_from_logits
def logprobs_from_logits_fused(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True
) -> torch.Tensor:
    """Per-token log-probs with optional Flash-Attn fused backward.

    Automatically detects flash_attn cross_entropy availability and uses it
    for more efficient backward pass. Falls back to standard log_softmax+gather.

    Args:
        logits: (..., vocab_size).
        labels: (...,) indices.
        inplace_backward: If True and flash-attn available, perform backward in-place.

    Returns:
        Log-probabilities with shape (...,).
    """
    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss

        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits_flat = logits.reshape(-1, last_dim)
        labels_flat = labels.reshape(-1)
        loss_fn = CrossEntropyLoss(inplace_backward=inplace_backward, reduction="none")
        neg_logprobs = loss_fn(logits_flat, labels_flat)
        return (-neg_logprobs).view(*batch_dim)
    except (ImportError, ModuleNotFoundError):
        return logprobs_from_logits(logits, labels)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/torch_functional.py::clip_by_value
def clip_by_value(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clip tensor values between min and max.

    Args:
        x: Input tensor.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clipped tensor (same shape as *x*).
    """
    return torch.max(
        torch.min(x, torch.tensor(max_val, device=x.device, dtype=x.dtype)),
        torch.tensor(min_val, device=x.device, dtype=x.dtype),
    )
