# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Response-length metrics on AtomRL's whole-sequence response mask."""

from __future__ import annotations

import torch
from atomrl.core.protocol import DataProto
from atomrl.trainer.metric_utils import compute_data_metrics
from atomrl.trainer.rl_trainer import RLTrainer


def _left_padded(
    prompt_lengths: list[int], response_lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(p + r for p, r in zip(prompt_lengths, response_lengths)) + 2
    attention_mask = torch.zeros(len(prompt_lengths), width, dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompt_lengths, response_lengths)):
        attention_mask[i, width - p - r :] = 1
    response_mask = RLTrainer._build_response_mask(
        object.__new__(RLTrainer),
        attention_mask,
        attention_mask,
        prompt_lengths,
    )
    return attention_mask, response_mask


def test_clip_ratio_counts_responses_that_reach_the_budget() -> None:
    attention_mask, response_mask = _left_padded([3, 2], [5, 3])
    batch = DataProto(
        tensors={"attention_mask": attention_mask, "response_mask": response_mask},
        meta={"max_response_length": 5},
    )

    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["response_length/max"] == 5
    assert metrics["response_length/clip_ratio"] == 0.5
    assert metrics["response_length_non_aborted/clip_ratio"] == 0.5


def test_a_response_only_mask_needs_no_budget() -> None:
    response_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    batch = DataProto(tensors={"response_mask": response_mask}, meta={})

    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["response_length/clip_ratio"] == 0.5
