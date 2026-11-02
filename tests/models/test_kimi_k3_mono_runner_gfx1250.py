# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts for gfx1250 Kimi runner admission and compilation."""

import os
from types import SimpleNamespace

import pytest
import torch
from aiter.jit.utils import chip_info

import atom.models.kimi_k3.mono.runner_gfx1250 as runner_module
from atom.models.kimi_k3.mono.dispatch import MonoDecodeModel
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
from atom.models.kimi_k3.mono.runner_gfx1250 import (
    Gfx1250WidthBuild,
    K3Gfx1250MonoDecodeRunner,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import compile_kernels
from atom.mono.runtime.consensus import MonoUnsupported


class _Launcher:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append((args, os.environ.get("COMPILE_ONLY")))


def _entry(launcher, *names):
    return SimpleNamespace(launcher=launcher, abi=KernelAbi(names))


def test_width_build_enumerates_each_local_launcher_once_and_compile_only():
    launchers = [_Launcher() for _ in range(10)]
    k1 = SimpleNamespace(
        entrypoint=SimpleNamespace(
            stages=tuple(
                _entry(launcher, f"k1_arg_{index}")
                for index, launcher in enumerate(launchers[:6])
            )
        )
    )
    k2 = SimpleNamespace(
        oproj=_entry(launchers[6], "oproj"),
        front=_entry(launchers[7], "front"),
        finish=_entry(launchers[8], "finish"),
        finalize=_entry(launchers[9], "finalize"),
    )
    build = Gfx1250WidthBuild(
        k1={1: k1, 2: k1},
        k2={1: k2, 2: k2},
    )

    kernels = build.kernels()
    assert len(kernels) == len(launchers)
    compile_kernels(kernels)
    assert all(launcher.calls == [((0,), "1")] for launcher in launchers)


def test_runner_admits_only_supported_rows_and_query_lengths():
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    prepared = []
    runner.widths = SimpleNamespace(
        prepare=lambda width: prepared.append(width) or True
    )

    assert runner.max_rows == 48
    assert not runner.prepare(0, 1)
    assert not runner.prepare(49, 1)
    assert not runner.prepare(3, 2)
    assert runner.prepare(48, 16)
    assert len(prepared) == 1


@pytest.mark.parametrize(
    ("debug", "replay", "message"),
    (
        (True, False, "debug mode"),
        (False, True, "ReplaySSM"),
    ),
)
def test_runner_refuses_unsupported_modes(monkeypatch, debug, replay, message):
    def bind_base(self, _model, _drafter, _metadata_builder):
        self.debug = debug
        self.replay_cap = replay

    monkeypatch.setattr(K3MonoDecodeRunner, "_bind", bind_base)
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    with pytest.raises(MonoUnsupported, match=message):
        runner._bind(object())


def test_runner_refuses_wrong_cu_count_and_uses_architecture_lds(monkeypatch):
    def bind_base(self, _model, _drafter, _metadata_builder):
        self.debug = False
        self.replay_cap = False
        self.device = torch.device("cpu")

    monkeypatch.setattr(K3MonoDecodeRunner, "_bind", bind_base)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=255),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    with pytest.raises(MonoUnsupported, match="256 CUs"):
        runner._bind(object())

    runner.blocks = 256
    assert runner._plan(1, 1).lds_bytes == GFX1250.max_workgroup_lds_bytes


def test_dispatch_selects_gfx1250_runner(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(
        runner_module,
        "K3Gfx1250MonoDecodeRunner",
        lambda model, drafter, metadata_builder: sentinel,
    )

    wrapped = MonoDecodeModel(object())
    assert wrapped._mono._create() is sentinel
