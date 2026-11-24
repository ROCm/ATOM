# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/workers/reward_manager/dapo.py
# Source: https://github.com/verl-project/verl/blob/246cac4ca6b0e3ce1009bc48aaa798689e0600af/recipe/dapo/dapo_ray_trainer.py

"""Pure, testable DAPO helpers ported faithfully from verl.

These functions implement the parts of verl's ``recipe/dapo`` that AtomRL's
native DAPO path was missing:

- :func:`overlong_buffer_penalty` — verl's soft overlong-buffer reward shaping
  (``verl/workers/reward_manager/dapo.py``).
- :func:`filter_groups_keep_mask` — verl's dynamic-sampling group filter that
  drops prompt groups with zero metric variance
  (``recipe/dapo/dapo_ray_trainer.py``).

They operate on plain tensors / python lists so they can be unit tested without
a GPU or any distributed setup.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "filter_groups_keep_mask",
    "overlong_buffer_penalty",
]


def overlong_buffer_penalty(
    response_lengths: Tensor | Sequence[int],
    max_resp_len: int,
    buffer_len: int,
    penalty_factor: float,
) -> Tensor:
    """Soft overlong-buffer penalty (verl ``DAPORewardManager``).

    For a response of valid length ``L``::

        expected_len = max_resp_len - buffer_len
        exceed_len   = L - expected_len
        penalty      = min(-exceed_len / buffer_len * penalty_factor, 0)

    So the penalty is 0 while ``L <= expected_len``, then ramps linearly down to
    ``-penalty_factor`` at ``L == max_resp_len`` and keeps decreasing beyond it.
    The returned tensor (shape ``[B]``, float32) should be **added** to the base
    reward.

    Args:
        response_lengths: Valid response token counts, shape ``[B]`` or a list.
        max_resp_len: Maximum response length (tokens).
        buffer_len: Soft-buffer width (tokens). Must be > 0.
        penalty_factor: Penalty magnitude at ``L == max_resp_len``.
    """
    lens = torch.as_tensor(response_lengths, dtype=torch.float32)
    if buffer_len <= 0 or penalty_factor == 0.0:
        return torch.zeros_like(lens)
    expected_len = float(max_resp_len - buffer_len)
    exceed_len = lens - expected_len
    penalty = -exceed_len / float(buffer_len) * float(penalty_factor)
    return torch.clamp(penalty, max=0.0)


def filter_groups_keep_mask(
    metric_vals: Tensor | Sequence[float],
    uids: Sequence,
    *,
    eps: float = 0.0,
) -> tuple[Tensor, list]:
    """Dynamic-sampling group filter (verl ``filter_groups``).

    Groups samples by ``uids`` (per-prompt id), computes the std of ``metric_vals``
    within each group, and keeps a group iff its std ``> eps`` **or** it has a
    single sample. Degenerate groups (all-correct / all-wrong → zero GRPO
    advantage) are dropped.

    Args:
        metric_vals: Per-sample metric (e.g. accuracy in ``{0,1}``), shape ``[N]``.
        uids: Per-sample prompt id (length ``N``), e.g. strings or ints.
        eps: Std threshold; verl uses ``std > 0`` (eps=0).

    Returns:
        ``(keep_mask, kept_uids)`` where ``keep_mask`` is a bool tensor ``[N]``
        (True = keep that sample) and ``kept_uids`` is the ordered unique list of
        kept prompt uids.
    """
    vals = torch.as_tensor(metric_vals, dtype=torch.float32).reshape(-1)
    if len(uids) != vals.shape[0]:
        raise ValueError(f"len(uids)={len(uids)} != metric_vals={vals.shape[0]}")

    uid2vals: dict = defaultdict(list)
    for u, v in zip(uids, vals.tolist()):
        uid2vals[u].append(v)

    kept_uids: list = []
    for u, vlist in uid2vals.items():
        if len(vlist) == 1:
            kept_uids.append(u)
            continue
        std = torch.std(torch.tensor(vlist, dtype=torch.float32), unbiased=False).item()
        if std > eps:
            kept_uids.append(u)

    kept_set = set(kept_uids)
    keep_mask = torch.tensor([u in kept_set for u in uids], dtype=torch.bool)
    return keep_mask, kept_uids
