# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One-step off-policy rollout: step s+1 generates while step s trains."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from atomrl.core.config import AtomRLConfig
from atomrl.trainer.one_step_off_policy import OneStepOffPolicyRollout
from atomrl.trainer.rl_trainer import RLTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_YAML = REPO_ROOT / "configs" / "grpo_dense_bf16.yaml"
_TIMEOUT_S = 10.0


def _drive(
    pipeline: OneStepOffPolicyRollout, steps: int, *, sync: bool, on_train=None
) -> list[int]:
    """Run the trainer side of the loop; return the staleness of every step."""
    version = 0
    pipeline.submit(0, version)
    staleness = []
    for step in range(steps):
        _batch, _gen_s, metrics = pipeline.take(step, version)
        staleness.append(metrics["async/staleness"])
        if on_train is not None:
            on_train(step)
        pipeline.drain()
        if sync:
            version = step + 1
    pipeline.close()
    return staleness


def test_next_batch_generates_while_the_current_one_trains() -> None:
    steps = 3
    training = [threading.Event() for _ in range(steps)]
    generating = [threading.Event() for _ in range(steps)]
    collected: list[int] = []

    def collect(step: int) -> str:
        collected.append(step)
        generating[step].set()
        # A sequential loop would only reach here after step - 1 had trained.
        if step > 0 and not training[step - 1].wait(_TIMEOUT_S):
            raise AssertionError(
                f"step {step} did not generate while step {step - 1} trained"
            )
        return f"batch-{step}"

    def on_train(step: int) -> None:
        training[step].set()
        if step + 1 < steps:
            assert generating[step + 1].wait(_TIMEOUT_S)

    pipeline = OneStepOffPolicyRollout(collect, total_steps=steps)
    staleness = _drive(pipeline, steps, sync=True, on_train=on_train)

    assert collected == [0, 1, 2]
    assert staleness == [0, 1, 1]


def test_drain_waits_for_the_batch_in_flight() -> None:
    release = threading.Event()

    def collect(step: int) -> int:
        if step == 1:
            release.wait(_TIMEOUT_S)
        return step

    pipeline = OneStepOffPolicyRollout(collect, total_steps=2)
    pipeline.submit(0, 0)
    pipeline.take(0, 0)
    threading.Timer(0.3, release.set).start()
    assert pipeline.drain() >= 0.25
    batch, _gen_s, metrics = pipeline.take(1, 1)
    assert (batch, metrics["async/staleness"]) == (1, 1)
    assert metrics["async/trainer_wait_s"] < 0.25
    pipeline.close()


def test_staleness_grows_while_the_rollout_weights_stay_put() -> None:
    pipeline = OneStepOffPolicyRollout(lambda step: step, total_steps=4)
    assert _drive(pipeline, 4, sync=False) == [0, 1, 2, 3]


def test_no_batch_is_generated_past_the_last_step() -> None:
    collected: list[int] = []
    pipeline = OneStepOffPolicyRollout(
        lambda step: collected.append(step), total_steps=2
    )
    _drive(pipeline, 2, sync=True)
    assert collected == [0, 1]


def test_a_failed_batch_is_raised_by_take_not_drain() -> None:
    def collect(step: int) -> int:
        if step == 1:
            raise RuntimeError("rollout replica died")
        return step

    pipeline = OneStepOffPolicyRollout(collect, total_steps=3)
    pipeline.submit(0, 0)
    pipeline.take(0, 0)
    pipeline.drain()
    with pytest.raises(RuntimeError, match="rollout replica died"):
        pipeline.take(1, 1)
    pipeline.close()


def test_only_one_batch_is_in_flight() -> None:
    pipeline = OneStepOffPolicyRollout(lambda step: step, total_steps=2)
    pipeline.submit(0, 0)
    with pytest.raises(RuntimeError, match="already in flight"):
        pipeline.submit(1, 0)
    pipeline.close()


def test_switch_is_off_by_default_and_set_from_the_command_line() -> None:
    assert AtomRLConfig().controller.ray.one_step_off_policy is False
    cfg = AtomRLConfig.from_yaml(
        GRPO_YAML,
        overrides=["controller.ray.one_step_off_policy=true"],
    )
    assert cfg.controller.ray.one_step_off_policy is True


def _trainer(
    *, enabled: bool, separated: bool = True, sleep: bool = False
) -> RLTrainer:
    trainer = RLTrainer.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        controller=SimpleNamespace(ray=SimpleNamespace(one_step_off_policy=enabled)),
    )
    trainer._rollout_wg = object() if separated else None
    trainer._ray_vllm_engine = SimpleNamespace(enable_sleep=sleep)
    return trainer


def test_layout_check_accepts_a_separate_rollout_pool() -> None:
    assert (
        _trainer(enabled=False, separated=False)._resolve_rollout_pipeline(True)
        == "sync"
    )
    assert _trainer(enabled=True)._resolve_rollout_pipeline(True) == "one_step"


@pytest.mark.parametrize(
    ("separated", "use_ray_rollout"),
    [(False, True), (True, False)],
)
def test_layout_check_rejects_a_rollout_without_its_own_gpus(
    separated: bool,
    use_ray_rollout: bool,
) -> None:
    trainer = _trainer(enabled=True, separated=separated)
    with pytest.raises(ValueError, match="needs GPUs of its own"):
        trainer._resolve_rollout_pipeline(use_ray_rollout)


def test_layout_check_rejects_a_sleeping_rollout() -> None:
    with pytest.raises(ValueError, match="cannot sleep"):
        _trainer(enabled=True, sleep=True)._resolve_rollout_pipeline(True)
