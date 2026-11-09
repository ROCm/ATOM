# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts for gfx1250 Kimi runner admission and compilation."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from aiter import QuantType, dtypes
from aiter.jit.utils import chip_info

import atom.models.kimi_k3.mono.runner_gfx1250 as runner_module
from atom.models.kimi_k3.mono.dispatch import MonoDecodeModel
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner
from atom.models.kimi_k3.mono.runner_gfx1250 import (
    Gfx1250WidthBuild,
    K3Gfx1250MonoDecodeRunner,
    _lean_width,
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


def test_width_build_enumerates_each_local_launcher_once_and_compile_only():
    launchers = [_Launcher() for _ in range(4)]
    k1 = SimpleNamespace(
        entrypoint=SimpleNamespace(
            launcher=launchers[0],
            abi=KernelAbi(("k1_arg",)),
        )
    )
    k2 = SimpleNamespace(
        front=SimpleNamespace(launcher=launchers[1], abi=KernelAbi(("front",))),
        finish=SimpleNamespace(launcher=launchers[2], abi=KernelAbi(("finish",))),
        finalize=SimpleNamespace(launcher=launchers[3], abi=KernelAbi(("finalize",))),
    )
    build = Gfx1250WidthBuild(
        k1={1: k1, 2: k1},
        k2={1: k2, 2: k2},
    )

    kernels = build.kernels()
    assert len(kernels) == len(launchers)
    compile_kernels(kernels)
    assert all(launcher.calls == [((0,), "1")] for launcher in launchers)


def test_width_build_replaces_only_spilled_k1_with_lean_gemv():
    spilled = _Launcher()
    untouched = _Launcher()
    rebuilt = object()
    observed = []
    spec = K1LocalBuildSpec(
        qlen=1,
        delta=True,
        state_len=3,
        eps=1e-5,
        out_eps=1e-5,
        onorm_eps=1e-5,
        lower_bound=-5.0,
        write_idx=-1,
    )
    plan = SimpleNamespace(
        make_k1_local_runner=lambda lean: observed.append(lean) or rebuilt
    )
    old = SimpleNamespace(
        entrypoint=SimpleNamespace(launcher=spilled),
        spec=spec,
        plan=plan,
    )
    other = SimpleNamespace(
        entrypoint=SimpleNamespace(launcher=untouched),
        spec=spec,
        plan=plan,
    )
    built = Gfx1250WidthBuild({1: old, 2: other}, {})

    lean = _lean_width(built, [spilled])
    assert lean is not None
    assert lean.k1 == {1: rebuilt, 2: other}
    assert lean.k2 == {}
    assert len(observed) == 1 and observed[0].gemv_unit == 1
    assert _lean_width(built, [object()]) is None


def test_resident_resource_hazards_include_explicit_spills(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "launcher_resources",
        lambda _launcher: [
            {
                "scratch": 0,
                "vgpr_spill": 2,
                "sgpr_spill": 3,
                "lds": 1024,
            }
        ],
    )
    assert runner_module._resident_hazards(object()) == (
        "2 vgpr spill",
        "3 sgpr spill",
    )


def test_gfx1250_uses_native_routed_k2_with_production_seam():
    source = (
        Path(__file__).parents[2] / "atom/models/kimi_k3/mono/runner_gfx1250.py"
    ).read_text()
    assert "layer.self_attn.o_proj(core)" in source
    assert "build_role_lds_front(" in source
    assert "include_attention=False" in source
    assert "def _routed_k2(" in source


def test_only_ptpc_o_proj_enables_fused_k1_quant():
    def layer(quant_type, params_dtype):
        return SimpleNamespace(
            self_attn=SimpleNamespace(
                o_proj=SimpleNamespace(
                    quant_type=quant_type,
                    params_dtype=params_dtype,
                )
            )
        )

    ptpc = layer(QuantType.per_Token, dtypes.fp8)
    assert runner_module._uses_ptpc_o_proj(ptpc)
    assert runner_module._fuse_ptpc_core(ptpc, 40)
    assert not runner_module._fuse_ptpc_core(ptpc, 48)
    assert not runner_module._uses_ptpc_o_proj(layer(QuantType.No, torch.bfloat16))


def test_external_k2_runner_still_validates_k1_state_pools():
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.plans = [SimpleNamespace(layer_id=7, kda=object(), mono=False)]
    runner._conv_state = lambda _layer: torch.empty(1, 1, 7)
    observed = []
    runner._recurrent_state = lambda layer: observed.append(layer)

    assert runner._state_len() == 7
    assert observed == [7]


@pytest.mark.parametrize("write_idx", (-1, 1))
def test_k1_hands_its_core_to_the_production_k2(write_idx):
    rows, hidden = 2, 4
    prefix = torch.zeros(rows, hidden)
    blocks = torch.zeros(rows, 3, hidden)
    core = torch.full((rows, hidden), 2.0)
    calls = {}

    def o_proj(value):
        calls["core"] = value
        return value + 1

    def mlp_attn_res(value, visible, attn_out):
        calls["seam"] = (value, visible, attn_out)
        return attn_out + 1, attn_out + 2

    def ffn(normed):
        calls["ffn"] = normed
        return normed + 1, normed + 2

    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=o_proj),
        mlp_attn_res=mlp_attn_res,
        _ffn=ffn,
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.inner = SimpleNamespace(layers=[layer])
    runner.routed = {0: object()}
    runner._k1 = lambda *_args, **_kwargs: core
    runner._routed_k2 = lambda _build, _plan, attn, value, _blocks: (
        calls.update(routed=(attn, value)) or value + 1,
        attn + 2,
    )
    runner._write_aux = lambda *_args: None
    plan = SimpleNamespace(
        layer_id=0,
        mono=True,
        kda=SimpleNamespace(blocks=1, write_idx=write_idx),
    )

    out = runner._layer(
        None, plan, None, prefix, blocks, torch.ones_like(prefix), None, None
    )

    assert calls["core"] is core
    attn_out, routed_prefix = calls["routed"]
    torch.testing.assert_close(attn_out, core + 1)
    assert routed_prefix is prefix
    assert all(tensor is None or tensor.shape == prefix.shape for tensor in out)


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
