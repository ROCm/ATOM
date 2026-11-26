# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU-only tests for streamed Adam updates."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from atomrl.engine.training.streamed_adam import (
    AdamChunkOptions,
    AdamState,
    adam_step_chunk_,
    initialize_adam_state,
)


def _options(group: dict, *, decoupled: bool) -> AdamChunkOptions:
    beta1, beta2 = group["betas"]
    return AdamChunkOptions(
        lr=group["lr"],
        beta1=beta1,
        beta2=beta2,
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        maximize=group["maximize"],
        decoupled_weight_decay=decoupled,
    )


def _streamed_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor | None,
    state: AdamState,
    options: AdamChunkOptions,
    *,
    chunk_size: int,
) -> None:
    if gradient is None:
        return
    state["step"].add_(1)
    step = int(state["step"].item())
    scratch = gradient.clone()
    for start in range(0, parameter.numel(), chunk_size):
        stop = min(start + chunk_size, parameter.numel())
        adam_step_chunk_(
            parameter[start:stop],
            scratch[start:stop],
            state["exp_avg"][start:stop],
            state["exp_avg_sq"][start:stop],
            step=step,
            options=options,
        )


def test_initialize_adam_state_uses_standard_pytorch_layout() -> None:
    parameter = torch.arange(7, dtype=torch.float32)

    state = initialize_adam_state(parameter)

    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["step"].shape == torch.Size([])
    assert state["step"].dtype == torch.float32
    assert state["step"].device.type == "cpu"
    assert state["step"].item() == 0
    for key in ("exp_avg", "exp_avg_sq"):
        assert state[key].shape == parameter.shape
        assert state[key].dtype == torch.float32
        assert state[key].device.type == "cpu"
        assert torch.count_nonzero(state[key]) == 0


def test_bf16_moments_use_fp32_workspace_and_persist_in_bf16() -> None:
    parameter = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32)
    gradient = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    state = initialize_adam_state(parameter, moment_dtype=torch.bfloat16)
    exp_avg_workspace = torch.empty_like(parameter)
    exp_avg_sq_workspace = torch.empty_like(parameter)
    options = AdamChunkOptions(0.03, 0.8, 0.95, 1e-6, 0.07, False, True)

    state["step"].add_(1)
    adam_step_chunk_(
        parameter,
        gradient.clone(),
        state["exp_avg"],
        state["exp_avg_sq"],
        exp_avg_workspace=exp_avg_workspace,
        exp_avg_sq_workspace=exp_avg_sq_workspace,
        step=1,
        options=options,
    )

    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert torch.count_nonzero(state["exp_avg"]) == parameter.numel()
    assert torch.count_nonzero(state["exp_avg_sq"]) == parameter.numel()
    assert torch.isfinite(parameter).all()


def test_bf16_moments_require_fp32_workspaces() -> None:
    parameter = torch.ones(2)
    state = initialize_adam_state(parameter, moment_dtype=torch.bfloat16)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(ValueError, match="workspace"):
        adam_step_chunk_(
            parameter,
            torch.ones_like(parameter),
            state["exp_avg"],
            state["exp_avg_sq"],
            step=1,
            options=options,
        )


def test_adam_chunk_options_are_frozen() -> None:
    options = AdamChunkOptions(0.1, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(FrozenInstanceError):
        options.lr = 0.2


@pytest.mark.parametrize("decoupled", [False, True], ids=["adam", "adamw"])
@pytest.mark.parametrize("maximize", [False, True])
def test_streamed_chunks_match_torch_over_three_steps(
    decoupled: bool,
    maximize: bool,
) -> None:
    initial = torch.tensor([0.5, -1.0, 2.0, -0.25, 3.0, 1.5, -2.5])
    expected = torch.nn.Parameter(initial.clone())
    actual = initial.clone()
    optimizer_type = torch.optim.AdamW if decoupled else torch.optim.Adam
    optimizer = optimizer_type(
        [expected],
        lr=0.03,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=0.07,
        maximize=maximize,
        foreach=False,
    )
    options = _options(optimizer.param_groups[0], decoupled=decoupled)
    state = initialize_adam_state(actual)
    gradients = [
        torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
        torch.tensor([-0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1]),
        torch.tensor([0.9, -0.8, 0.1, 0.2, -0.4, 0.6, -0.3]),
    ]

    for gradient in gradients:
        expected.grad = gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _streamed_step(actual, gradient, state, options, chunk_size=3)

    expected_state = optimizer.state[expected]
    torch.testing.assert_close(actual, expected.detach())
    torch.testing.assert_close(state["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(state["exp_avg_sq"], expected_state["exp_avg_sq"])
    torch.testing.assert_close(state["step"], expected_state["step"])


def test_multiple_groups_and_missing_gradient_match_torch_step_semantics() -> None:
    expected_a = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 7))
    expected_b = torch.nn.Parameter(torch.linspace(0.5, 2.0, 5))
    actual_a = expected_a.detach().clone()
    actual_b = expected_b.detach().clone()
    optimizer = torch.optim.Adam(
        [
            {
                "params": [expected_a],
                "lr": 0.02,
                "betas": (0.7, 0.91),
                "weight_decay": 0.03,
                "maximize": False,
            },
            {
                "params": [expected_b],
                "lr": 0.005,
                "betas": (0.85, 0.97),
                "weight_decay": 0.0,
                "maximize": True,
            },
        ],
        eps=1e-7,
        foreach=False,
    )
    options_a = _options(optimizer.param_groups[0], decoupled=False)
    options_b = _options(optimizer.param_groups[1], decoupled=False)
    state_a = initialize_adam_state(actual_a)
    state_b = initialize_adam_state(actual_b)
    gradient_steps = [
        (torch.linspace(-0.3, 0.3, 7), torch.linspace(0.4, -0.4, 5)),
        (torch.linspace(0.2, -0.2, 7), None),
        (None, torch.linspace(-0.1, 0.5, 5)),
    ]

    for gradient_a, gradient_b in gradient_steps:
        expected_a.grad = None if gradient_a is None else gradient_a.clone()
        expected_b.grad = None if gradient_b is None else gradient_b.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _streamed_step(actual_a, gradient_a, state_a, options_a, chunk_size=3)
        _streamed_step(actual_b, gradient_b, state_b, options_b, chunk_size=3)

    torch.testing.assert_close(actual_a, expected_a.detach())
    torch.testing.assert_close(actual_b, expected_b.detach())
    torch.testing.assert_close(state_a["step"], optimizer.state[expected_a]["step"])
    torch.testing.assert_close(state_b["step"], optimizer.state[expected_b]["step"])
    assert state_a["step"].item() == 2
    assert state_b["step"].item() == 2


def test_gradient_scratch_becomes_denominator() -> None:
    parameter = torch.tensor([1.0, -2.0, 3.0])
    gradient_scratch = torch.tensor([0.2, -0.4, 0.6])
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-5, 0.0, False, False)

    adam_step_chunk_(
        parameter,
        gradient_scratch,
        exp_avg,
        exp_avg_sq,
        step=1,
        options=options,
    )

    expected_denominator = exp_avg_sq.sqrt() / (1 - options.beta2**1) ** 0.5
    expected_denominator.add_(options.eps)
    torch.testing.assert_close(gradient_scratch, expected_denominator)


@pytest.mark.parametrize(
    ("parameter", "match"),
    [
        (torch.ones(2, dtype=torch.float64), "float32"),
        (torch.empty(2, device="meta"), "CPU"),
    ],
)
def test_initialize_adam_state_rejects_invalid_parameter(
    parameter: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        initialize_adam_state(parameter)


@pytest.mark.parametrize("step", [0, -1, 1.0, True])
def test_adam_step_chunk_rejects_invalid_step(step) -> None:
    parameter = torch.ones(2)
    scratch = torch.ones(2)
    exp_avg = torch.zeros(2)
    exp_avg_sq = torch.zeros(2)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(ValueError, match="step"):
        adam_step_chunk_(
            parameter,
            scratch,
            exp_avg,
            exp_avg_sq,
            step=step,
            options=options,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", -0.1),
        ("lr", float("inf")),
        ("lr", float("nan")),
        ("eps", -1e-8),
        ("eps", float("inf")),
        ("eps", float("nan")),
        ("weight_decay", -0.1),
        ("weight_decay", float("inf")),
        ("weight_decay", float("nan")),
        ("beta1", -0.1),
        ("beta1", 1.0),
        ("beta1", float("inf")),
        ("beta1", float("nan")),
        ("beta2", -0.1),
        ("beta2", 1.0),
        ("beta2", float("inf")),
        ("beta2", float("nan")),
    ],
)
def test_invalid_options_raise_before_mutating_tensors(
    field: str,
    value: float,
) -> None:
    parameter = torch.tensor([1.0, -2.0, 3.0])
    scratch = torch.tensor([0.2, -0.4, 0.6])
    exp_avg = torch.tensor([0.01, 0.02, 0.03])
    exp_avg_sq = torch.tensor([0.04, 0.05, 0.06])
    tensors = (parameter, scratch, exp_avg, exp_avg_sq)
    originals = tuple(tensor.clone() for tensor in tensors)
    options = replace(
        AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.1, False, True),
        **{field: value},
    )

    with pytest.raises(ValueError, match=field):
        adam_step_chunk_(
            parameter,
            scratch,
            exp_avg,
            exp_avg_sq,
            step=1,
            options=options,
        )

    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor, original)


@pytest.mark.parametrize(
    "left,right",
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ],
)
def test_exact_storage_overlap_raises_before_mutation(
    left: int,
    right: int,
) -> None:
    tensors = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.01, 0.02, 0.03]),
        torch.tensor([0.04, 0.05, 0.06]),
    ]
    tensors[right] = tensors[left]
    originals = tuple(tensor.clone() for tensor in tensors)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(ValueError, match="overlap"):
        adam_step_chunk_(*tensors, step=1, options=options)

    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor, original)


def test_sliced_storage_overlap_raises_before_mutation() -> None:
    shared = torch.tensor([1.0, 2.0, 3.0, 4.0])
    parameter = shared[:3]
    scratch = shared[1:]
    exp_avg = torch.tensor([0.01, 0.02, 0.03])
    exp_avg_sq = torch.tensor([0.04, 0.05, 0.06])
    originals = (shared.clone(), exp_avg.clone(), exp_avg_sq.clone())
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(ValueError, match="overlap"):
        adam_step_chunk_(
            parameter,
            scratch,
            exp_avg,
            exp_avg_sq,
            step=1,
            options=options,
        )

    torch.testing.assert_close(shared, originals[0])
    torch.testing.assert_close(exp_avg, originals[1])
    torch.testing.assert_close(exp_avg_sq, originals[2])


def test_disjoint_views_of_same_storage_are_allowed() -> None:
    shared = torch.tensor(
        [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    )
    views = tuple(shared[start : start + 3] for start in range(0, 12, 3))
    expected = tuple(view.clone() for view in views)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    adam_step_chunk_(*expected, step=1, options=options)
    adam_step_chunk_(*views, step=1, options=options)

    for actual, expected_tensor in zip(views, expected):
        torch.testing.assert_close(actual, expected_tensor)


def test_noncontiguous_tensor_raises_before_mutation() -> None:
    parameter_base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    parameter = parameter_base.t()
    scratch = torch.ones_like(parameter)
    exp_avg = torch.zeros_like(parameter)
    exp_avg_sq = torch.zeros_like(parameter)
    tensors = (parameter, scratch, exp_avg, exp_avg_sq)
    originals = tuple(tensor.clone() for tensor in tensors)
    options = AdamChunkOptions(0.01, 0.9, 0.99, 1e-8, 0.0, False, False)

    with pytest.raises(ValueError, match="contiguous"):
        adam_step_chunk_(*tensors, step=1, options=options)

    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor, original)
