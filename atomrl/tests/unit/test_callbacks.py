# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

from atomrl.trainer.callbacks import WandbCallback


def test_wandb_callback_uses_global_step_as_x_axis(monkeypatch) -> None:
    wandb = SimpleNamespace(
        init=Mock(),
        config=SimpleNamespace(update=Mock()),
        define_metric=Mock(),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    callback = WandbCallback(project="test")
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            algorithm=SimpleNamespace(name="grpo"),
            num_training_steps=200,
        ),
    )

    callback.on_train_begin(trainer)

    assert wandb.define_metric.call_args_list == [
        call("train/global_step"),
        call("*", step_metric="train/global_step"),
    ]


def test_wandb_callback_commits_each_explicit_step() -> None:
    callback = WandbCallback(project="test")
    callback._wandb = Mock()
    callback._enabled = True
    trainer = SimpleNamespace(_rank=0, _last_val_generations=None)

    callback.on_step_end(trainer, step=0, metrics={"loss": 0.25})

    callback._wandb.log.assert_called_once_with(
        {
            "loss": 0.25,
            "train/loss": 0.25,
            "actor/loss": 0.25,
            "actor/pg_loss": 0.25,
            "core/loss": 0.25,
            "train/global_step": 1,
        },
        step=1,
        commit=True,
    )
