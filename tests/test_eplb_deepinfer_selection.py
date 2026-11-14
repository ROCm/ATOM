# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-capable tests for the standalone DeepInfer replica-selection policy."""

from fractions import Fraction

import pytest
import torch

from atom.model_ops.eplb_deepinfer import replicate_experts_deepinfer

# Recorded from rebalance_experts_global(history, E + extra, 1) at:
# https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/eplb/policy.py
# num_local_physical_experts=1 disables the separate GPU placement stage.
UPSTREAM_SELECTIONS = [
    ([[100, 60, 0], [0, 60, 80]], [0, 2, 1]),
    ([[0, 60, 100], [80, 60, 0]], [2, 0, 1]),
    ([[100, 1, 1]], [0, 0, 0, 0, 0]),
    ([[0, 0, 0], [0, 0, 0]], [2, 0, 0, 0, 0]),
]


def _assert_structure(physical_to_logical, physical_rank, logical_count):
    num_layers, num_physical = physical_to_logical.shape
    assert physical_rank.shape == (num_layers, num_physical)
    for result in (physical_to_logical, physical_rank, logical_count):
        assert result.dtype == torch.int32
        assert result.device == physical_to_logical.device
    for layer in range(num_layers):
        assert logical_count[layer].sum().item() == num_physical
        for expert, count in enumerate(logical_count[layer].tolist()):
            slots = physical_to_logical[layer] == expert
            assert slots.sum().item() == count
            assert sorted(physical_rank[layer, slots].tolist()) == list(range(count))


@pytest.mark.parametrize("history,selected", UPSTREAM_SELECTIONS)
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_matches_pinned_upstream_for_each_replica_budget(history, selected, dtype):
    tokens = torch.tensor(history, dtype=dtype).unsqueeze(1)
    num_logical = tokens.size(-1)
    for extra in range(len(selected) + 1):
        result = replicate_experts_deepinfer(tokens, num_logical + extra)
        physical_to_logical, physical_rank, logical_count = result
        expected = list(range(num_logical)) + selected[:extra]
        assert physical_to_logical.tolist() == [expected]
        _assert_structure(*result)
        if extra == 0:
            assert not physical_rank.any()
            assert logical_count.tolist() == [[1] * num_logical]


def test_preserves_step_history_instead_of_selecting_by_total_load():
    tokens = torch.tensor([[[100, 60, 0]], [[0, 60, 80]]])
    assert tokens.sum(0).argmax(-1).item() == 1
    physical_to_logical, _, counts = replicate_experts_deepinfer(tokens, 4)
    assert physical_to_logical[0, -1].item() == 0
    assert counts.tolist() == [[2, 1, 1]]


def test_layers_choose_replicas_independently():
    tokens = torch.tensor([[[100, 60, 0], [0, 60, 100]], [[0, 60, 80], [80, 60, 0]]])
    result = replicate_experts_deepinfer(tokens, 5)
    assert result[0].tolist() == [[0, 1, 2, 0, 2], [0, 1, 2, 2, 0]]
    assert result[1].tolist() == [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
    assert result[2].tolist() == [[2, 1, 2], [2, 1, 2]]
    _assert_structure(*result)


@pytest.mark.parametrize(
    "history",
    [
        [[100, 60, 0], [0, 60, 80]],
        [[73, 9, 51, 20], [14, 68, 6, 42], [7, 20, 80, 13]],
        [[91, 2, 3, 4]],
    ],
)
def test_each_choice_minimizes_explicit_historical_bottleneck(history):
    """Enumerate candidates with exact rational loads, independently of torch."""
    num_logical = len(history[0])
    counts = [1] * num_logical
    physical_to_logical, _, _ = replicate_experts_deepinfer(
        torch.tensor(history).unsqueeze(1), num_logical + 8
    )
    for selected in physical_to_logical[0, num_logical:].tolist():
        candidate_costs = []
        for candidate in range(num_logical):
            cost = sum(
                max(
                    Fraction(tokens, counts[expert] + (expert == candidate))
                    for expert, tokens in enumerate(step)
                )
                for step in history
            )
            candidate_costs.append(cost)
        assert candidate_costs[selected] == min(candidate_costs)
        counts[selected] += 1


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_input_history_is_unchanged_even_when_noncontiguous(dtype):
    tokens = torch.arange(24, dtype=dtype).reshape(3, 2, 4).transpose(0, 1)
    assert not tokens.is_contiguous()
    original = tokens.clone()
    first = replicate_experts_deepinfer(tokens, 10)
    second = replicate_experts_deepinfer(tokens, 10)
    assert torch.equal(tokens, original)
    for left, right in zip(first, second):
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    "shape", [(3,), (2, 3), (1, 2, 3, 4), (0, 1, 3), (2, 0, 3), (2, 1, 0)]
)
def test_rejects_invalid_history_shape(shape):
    with pytest.raises(ValueError, match="tokens_per_expert"):
        replicate_experts_deepinfer(torch.zeros(shape), 4)


@pytest.mark.parametrize("num_physical", [-1, 0, 2])
def test_requires_room_for_every_logical_expert(num_physical):
    with pytest.raises(ValueError, match="num_physical must be >= num_logical"):
        replicate_experts_deepinfer(torch.zeros(2, 1, 3), num_physical)


@pytest.mark.parametrize("num_physical", [True, 4.0, "4"])
def test_requires_integer_physical_budget(num_physical):
    with pytest.raises(TypeError, match="num_physical must be an integer"):
        replicate_experts_deepinfer(torch.zeros(2, 1, 3), num_physical)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/ROCm GPU")
@pytest.mark.parametrize("history,selected", UPSTREAM_SELECTIONS)
def test_cpu_and_gpu_match(history, selected):
    tokens = torch.tensor(history).unsqueeze(1)
    num_physical = tokens.size(-1) + len(selected)
    expected = replicate_experts_deepinfer(tokens, num_physical)
    actual = replicate_experts_deepinfer(tokens.cuda(), num_physical)
    _assert_structure(*actual)
    for cpu, gpu in zip(expected, actual):
        assert gpu.device.type == "cuda"
        assert torch.equal(cpu, gpu.cpu())
