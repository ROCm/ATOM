# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[3] / "atomrl" / "lumen" / "quantize" / "optimizer_manager.py"
)
SPEC = importlib.util.spec_from_file_location("_test_optimizer_manager", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FP32MasterWeightOptimizer = MODULE.FP32MasterWeightOptimizer
build_training_scheduler = getattr(MODULE, "build_training_scheduler", None)


def _optimizer(model):
    return FP32MasterWeightOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=0.1,
        weight_decay=0.0,
    )


def test_step_with_grad_norm_updates_bf16_model_from_fp32_master() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _optimizer(model)
    before = model.weight.detach().clone()
    model.weight.grad = torch.ones_like(model.weight)

    grad_norm = optimizer.step_with_grad_norm(max_grad_norm=0.25)

    assert torch.isfinite(grad_norm)
    assert float(grad_norm) > 0
    assert not torch.equal(model.weight, before)


def test_sync_master_params_from_model_refreshes_checkpoint_source() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _optimizer(model)
    with torch.no_grad():
        model.weight.fill_(3.0)

    optimizer.sync_master_params_from_model()

    master = next(param for param in optimizer._fp32_masters if param is not None)
    torch.testing.assert_close(master, model.weight.float())


def test_get_learning_rate_reads_inner_optimizer_group() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _optimizer(model)

    assert optimizer.get_learning_rate() == 0.1


def test_build_training_scheduler_supports_cosine_and_wsd() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    for style in ("cosine", "WSD"):
        optimizer = _optimizer(model)
        policy = SimpleNamespace(
            learning_rate=0.1,
            warmup_ratio=0.1,
            lr_decay_style=style,
            min_lr=0.01,
            wsd_decay_ratio=0.2,
            wsd_decay_style="cosine",
        )

        scheduler = build_training_scheduler(optimizer, policy, total_steps=100)
        optimizer.step()
        scheduler.step()

        assert scheduler.optimizer is optimizer


def test_optimizer_state_round_trip_preserves_fp32_master_contract() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _optimizer(model)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step_with_grad_norm(max_grad_norm=1.0)
    expected = [param.detach().clone() for param in optimizer.fp32_params]
    state = optimizer.state_dict()

    restored_model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    restored = _optimizer(restored_model)
    restored.load_state_dict(state)

    assert len(restored.fp32_params) == len(expected)
    for actual, wanted in zip(restored.fp32_params, expected):
        torch.testing.assert_close(actual, wanted)


def test_wsd_scheduler_maps_minus_sqrt_and_rejects_exponential() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    policy = SimpleNamespace(
        learning_rate=0.1,
        warmup_ratio=0.1,
        lr_decay_style="WSD",
        min_lr=0.01,
        wsd_decay_ratio=0.2,
        wsd_decay_style="minus_sqrt",
    )
    assert build_training_scheduler(_optimizer(model), policy, 100) is not None

    policy.wsd_decay_style = "exponential"
    with pytest.raises(ValueError, match="WSD decay style"):
        build_training_scheduler(_optimizer(model), policy, 100)


def test_distributed_optimizer_state_maps_masters_to_model_parameters() -> None:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _optimizer(model)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step_with_grad_norm(max_grad_norm=1.0)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)

    state = optimizer.distributed_state_dict(model, options)
    restored_model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    restored = _optimizer(restored_model)
    restored.load_distributed_state_dict(restored_model, state, options)

    source_state = next(iter(optimizer._inner.state.values()))
    restored_state = next(iter(restored._inner.state.values()))
    torch.testing.assert_close(restored_state["exp_avg"], source_state["exp_avg"])
    torch.testing.assert_close(restored_state["exp_avg_sq"], source_state["exp_avg_sq"])


def test_scheduler_state_round_trip_does_not_advance_epoch() -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    policy = SimpleNamespace(
        learning_rate=0.1,
        warmup_ratio=0.1,
        lr_decay_style="cosine",
        min_lr=0.01,
        wsd_decay_ratio=0.2,
        wsd_decay_style="cosine",
    )
    optimizer = _optimizer(model)
    scheduler = build_training_scheduler(optimizer, policy, 100)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    expected_epoch = scheduler.last_epoch
    expected_lr = optimizer.get_learning_rate()

    restored_optimizer = _optimizer(model)
    restored = build_training_scheduler(restored_optimizer, policy, 100)
    restored.load_state_dict(state)
    restored_optimizer.param_groups[0]["lr"] = expected_lr

    assert restored.last_epoch == expected_epoch
    assert restored_optimizer.get_learning_rate() == expected_lr
