# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One process on one GPU gets a single-rank process group.

Without one, FSDP2 left the model unsharded and computing in the fp32 of its
master weights, and the quick start (configs/grpo_dense_bf16.yaml on one GPU)
died at its first training forward in aiter's varlen flash attention
("Unsupported dtype for MHA").
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from atomrl.trainer import main as trainer_main

_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture
def calls(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    seen: list = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.append(("device", d)))
    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda **kw: seen.append(("init", kw))
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return seen


def test_one_process_with_a_gpu_gets_a_one_rank_group(monkeypatch, calls) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    trainer_main._setup_distributed(use_ray_controller=False)

    assert ("device", 0) in calls
    assert [c for c in calls if c[0] == "init"]
    assert (os.environ["RANK"], os.environ["WORLD_SIZE"]) == ("0", "1")
    assert int(os.environ["MASTER_PORT"]) > 0


def test_torchrun_environment_is_kept(monkeypatch, calls) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")

    trainer_main._setup_distributed(use_ray_controller=False)

    assert ("device", 3) in calls
    assert os.environ["WORLD_SIZE"] == "8"


@pytest.mark.parametrize("ray, gpu", [(True, True), (False, False)])
def test_no_group_for_ray_or_without_a_gpu(monkeypatch, calls, ray, gpu) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)

    trainer_main._setup_distributed(use_ray_controller=ray)

    assert calls == []
    assert "RANK" not in os.environ
