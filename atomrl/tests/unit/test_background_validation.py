# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Validation on a background thread while fully async training goes on."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from atomrl.trainer.fully_async_rollout import BackgroundValidation
from atomrl.trainer.rl_trainer import RLTrainer

_TIMEOUT_S = 10.0


class Gate:
    """A validation that runs until the test releases it."""

    def __init__(self, acc: float = 0.5) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.steps: list[int] = []
        self._acc = acc

    def __call__(self, step: int) -> dict[str, float]:
        self.steps.append(step)
        self.started.set()
        assert self.release.wait(_TIMEOUT_S)
        return {"val-core/acc/mean@1": self._acc}


class Pipeline:
    def __init__(self) -> None:
        self.pauses = 0

    @contextmanager
    def submissions_paused(self):
        self.pauses += 1
        yield


def _trainer(val_steps: int, foreground) -> RLTrainer:
    trainer = RLTrainer.__new__(RLTrainer)
    trainer.config = SimpleNamespace(val_steps=val_steps)
    trainer.run_validation = foreground
    return trainer


def test_nothing_is_collected_before_a_validation_starts() -> None:
    background = BackgroundValidation(lambda step: {}, lambda: 0)

    assert background.collect(wait=True) == ({}, 0.0)
    assert not background.pending


def test_a_validation_reports_the_versions_it_ran_on() -> None:
    run = Gate()
    versions = iter([3, 5])  # read once at the start and once at the end
    background = BackgroundValidation(run, lambda: next(versions))

    background.start(2)
    assert run.started.wait(_TIMEOUT_S)
    assert background.collect(wait=False) == ({}, 0.0), "still running"
    assert background.pending
    run.release.set()
    metrics, _blocked_s = background.collect(wait=True)

    assert run.steps == [2]
    assert metrics["val-core/acc/mean@1"] == 0.5
    assert metrics["val/global_step"] == 3.0
    assert metrics["val/started_version"] == 3.0
    assert metrics["val/finished_version"] == 5.0
    assert metrics["val/elapsed_s"] >= 0.0
    assert not background.pending


def test_waiting_for_a_validation_reports_how_long_it_blocked() -> None:
    run = Gate()
    background = BackgroundValidation(run, lambda: 0)
    background.start(0)
    assert run.started.wait(_TIMEOUT_S)

    threading.Timer(0.2, run.release.set).start()
    metrics, blocked_s = background.collect(wait=True)

    assert metrics["val-core/acc/mean@1"] == 0.5
    assert blocked_s >= 0.15


def test_a_failed_validation_reaches_the_trainer() -> None:
    def run(step: int) -> dict[str, float]:
        raise ValueError("rollout replica died")

    background = BackgroundValidation(run, lambda: 0)
    background.start(4)

    with pytest.raises(RuntimeError, match="validation at step 5 failed") as info:
        background.collect(wait=True)
    assert isinstance(info.value.__cause__, ValueError)
    assert not background.pending


def test_only_one_validation_runs_at_a_time() -> None:
    run = Gate()
    background = BackgroundValidation(run, lambda: 0)
    background.start(0)

    with pytest.raises(RuntimeError, match="collect"):
        background.start(1)
    run.release.set()
    background.collect(wait=True)


def test_an_empty_validation_logs_nothing() -> None:
    background = BackgroundValidation(lambda step: {}, lambda: 0)
    background.start(0)

    metrics, _blocked_s = background.collect(wait=True)

    assert metrics == {}
    assert not background.pending


# -- scheduling in the training loop ------------------------------------------


def test_a_due_step_validates_in_the_foreground_without_overlap() -> None:
    calls = []
    trainer = _trainer(5, lambda: calls.append("val") or {"val-core/acc/mean@1": 0.2})
    pipeline = Pipeline()

    assert trainer._validate_at_step(3, 10, pipeline, None) == {}
    assert trainer._validate_at_step(4, 10, pipeline, None) == {
        "val-core/acc/mean@1": 0.2
    }
    assert calls == ["val"]
    assert pipeline.pauses == 1, "training groups wait while validation has the pool"


def test_validation_overlaps_the_steps_after_it() -> None:
    run = Gate()
    background = BackgroundValidation(run, lambda: 0)
    foreground = []
    trainer = _trainer(4, lambda: foreground.append("val") or {})
    pipeline = Pipeline()

    assert trainer._validate_at_step(3, 10, pipeline, background) == {}
    assert run.started.wait(_TIMEOUT_S)
    assert (
        trainer._validate_at_step(4, 10, pipeline, background) == {}
    ), "training goes on"
    run.release.set()
    at_next_due = trainer._validate_at_step(7, 10, pipeline, background)
    at_last = trainer._validate_at_step(9, 10, pipeline, background)

    assert at_next_due["val/global_step"] == 4.0, "the next due step waits for it"
    assert at_last["val/global_step"] == 8.0, "the last step waits for the one running"
    assert run.steps == [3, 7]
    assert foreground == [] and pipeline.pauses == 0
    assert not background.pending


def test_the_step_before_a_validating_last_step_waits_for_the_one_running() -> None:
    run = Gate()
    background = BackgroundValidation(run, lambda: 0)
    trainer = _trainer(5, lambda: {"val-core/acc/mean@1": 0.9})
    pipeline = Pipeline()

    trainer._validate_at_step(4, 10, pipeline, background)
    assert run.started.wait(_TIMEOUT_S)
    threading.Timer(0.2, run.release.set).start()
    before_last = trainer._validate_at_step(8, 10, pipeline, background)
    last = trainer._validate_at_step(9, 10, pipeline, background)

    assert before_last["val/global_step"] == 5.0
    assert last == {"val-core/acc/mean@1": 0.9}, "the last step logs only its own"
    assert pipeline.pauses == 1
    assert run.steps == [4]
    assert not background.pending


def test_validating_every_step_skips_the_one_before_the_last() -> None:
    run = Gate()
    run.release.set()
    background = BackgroundValidation(run, lambda: 0)
    trainer = _trainer(1, lambda: {"val-core/acc/mean@1": 0.9})
    pipeline = Pipeline()

    logged = [
        trainer._validate_at_step(step, 4, pipeline, background) for step in range(4)
    ]

    assert run.steps == [0, 1], "step 2 starts none, the last step validates next"
    assert [m.get("val/global_step") for m in logged] == [None, 1.0, 2.0, None]
    assert logged[3] == {"val-core/acc/mean@1": 0.9}
