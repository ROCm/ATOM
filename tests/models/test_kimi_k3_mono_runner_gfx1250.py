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
from atom.models.kimi_k3.mono.config import layer_block
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


@pytest.mark.parametrize("layer_id", (5, 12))
def test_k1_reads_the_production_seam_and_hands_its_core_to_k2(layer_id):
    rows, hidden = 2, 4
    prefix = torch.zeros(rows, hidden)
    add = torch.ones_like(prefix)
    blocks = torch.zeros(rows, 3, hidden)
    core = torch.full((rows, hidden), 2.0)
    calls = {}

    def self_attention_attn_res(value, visible, add_hidden, add_hidden2):
        calls["seam"] = visible.shape[1]
        return value + add_hidden + 10, value + add_hidden

    def o_proj(value):
        calls["core"] = value
        return value + 1

    def k1(_build, _plan, normed, delta, _blocks, _metadata):
        calls["k1"] = (normed, delta)
        return core

    layer = SimpleNamespace(
        self_attention_attn_res=self_attention_attn_res,
        self_attn=SimpleNamespace(o_proj=o_proj),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.inner = SimpleNamespace(layers={layer_id: layer})
    runner.routed = {layer_id: object()}
    runner._k1 = k1
    runner._routed_k2 = lambda _build, _plan, attn, value, _blocks: (
        calls.update(routed=(attn, value)) or value + 1,
        attn + 2,
    )
    runner._write_aux = lambda *_args: None
    plan = SimpleNamespace(layer_id=layer_id, mono=True, kda=object())

    out = runner._layer(None, plan, None, prefix, blocks, add, None, None)

    before, write_idx = layer_block(layer_id)
    normed, delta = calls["k1"]
    torch.testing.assert_close(normed, prefix + add + 10)
    assert delta is None
    assert calls["seam"] == before
    assert calls["core"] is core
    attn_out, routed_prefix = calls["routed"]
    torch.testing.assert_close(attn_out, core + 1)
    torch.testing.assert_close(routed_prefix, prefix + add)
    if write_idx >= 0:
        torch.testing.assert_close(blocks[:, write_idx], prefix + add)
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


@pytest.mark.parametrize("herd", (False, True))
def test_grouped_k2_projects_locally_then_runs_grouped_moe(monkeypatch, herd):
    rows, layer_id = 2, 3
    tensor = lambda *shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype)  # noqa: E731
    weight = SimpleNamespace(
        write_idx=-1,
        blocks=2,
        w_gate=tensor(4, 4),
        w_ld=tensor(4, 4),
        w_sgu=tensor(4, 4),
        w_sgu_scale=None,
        w_sd_scale=None,
        bias=tensor(4),
        w_sd=tensor(4, 4),
        w13=tensor(8, dtype=torch.uint8),
        w13s=tensor(4, dtype=torch.uint8),
        w2=tensor(8, dtype=torch.uint8),
        w2s=tensor(4, dtype=torch.uint8),
        ln_w=tensor(4),
        w_up=tensor(4, 4),
        ln_eps=1e-6,
    )
    names = (
        "router_logits", "latent_fp4", "latent_scales", "shared_gate_up",
        "shared_hidden", "route", "tile_route", "flags", "experts", "kof",
        "union_sizes", "intermediate_fp4", "intermediate_scales", "routed",
        "shared", "finish_scratch", "final",
    )
    buffers = {name: tensor(rows, 4, dtype=torch.uint8) for name in names}
    calls = {}

    def recorder(name):
        return SimpleNamespace(
            launcher=lambda *args, **_kwargs: calls.setdefault(name, args)
        )

    local = SimpleNamespace(
        front=recorder("front"),
        finish=recorder("finish"),
        finalize=recorder("finalize"),
        buffers=buffers,
        grouped_experts=True,
        replicated_finalize=False,
        herd=herd,
    )
    x = tensor(rows, 4)
    experts = SimpleNamespace(w13_weight_scale=tensor(2), w2_weight_scale=tensor(2))
    layer = SimpleNamespace(
        mlp_attn_res=lambda _prefix, _blocks, _attn: (x, tensor(rows, 4)),
        block_sparse_moe=SimpleNamespace(
            experts=experts, routed_expert_norm=lambda value: value
        ),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.rank = 0
    runner.routed = {layer_id: weight}
    runner.inner = SimpleNamespace(layers={layer_id: layer})
    group = SimpleNamespace(
        all_reduce=lambda value, ca_fp8_quant=False: value,
        all_gather=lambda *_args, **_kwargs: pytest.fail("no all-gather"),
    )
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(runner_module.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        runner_module,
        "fused_moe",
        lambda *args, **kwargs: calls.setdefault("moe", (args, kwargs)),
    )
    build = SimpleNamespace(k2={layer_id: local})
    plan = SimpleNamespace(layer_id=layer_id)

    runner._routed_k2(build, plan, tensor(rows, 4), tensor(rows, 4), tensor(rows, 3, 4))

    assert calls["front"][:6] == (
        x.data_ptr(),
        weight.w_gate.data_ptr(),
        weight.w_ld.data_ptr(),
        weight.w_sgu.data_ptr(),
        0,
        buffers["router_logits"].data_ptr(),
    )
    assert calls["finish"][0] == buffers["router_logits"].data_ptr()
    args, kwargs = calls["moe"]
    assert args[0].data_ptr() == buffers["latent_fp4"].data_ptr()
    assert args[3] is buffers["tile_route"] and args[4] is buffers["route"]
    assert kwargs["a1_scale"] is buffers["latent_scales"]
    assert kwargs["w1_scale"] is experts.w13_weight_scale
    assert kwargs["output"] is buffers["routed"]
    if herd:
        assert kwargs["herd_topk"] == 16
        assert kwargs["herd_routed_scaling_factor"] == 1.0
    else:
        assert "herd_topk" not in kwargs
    assert calls["finalize"][0] == buffers["routed"].data_ptr()


def test_replicated_finalize_all_reduces_both_partials_once(monkeypatch):
    rows, layer_id, latent, hidden = 2, 3, 3584, 7168
    weight = SimpleNamespace(
        write_idx=-1,
        blocks=2,
        **{
            name: torch.zeros(8, dtype=torch.uint8)
            for name in (
                "w_gate", "w_ld", "w_sgu", "bias", "w_sd", "w13", "w13s", "w2",
                "w2s", "ln_w", "w_up", "w_up_full",
            )
        },
        ln_eps=1e-6,
        w_sgu_scale=None,
        w_sd_scale=None,
    )
    reduce = torch.zeros(rows * (latent + hidden), dtype=torch.bfloat16)
    buffers = {
        name: torch.zeros(rows, 4, dtype=torch.uint8)
        for name in (
            "router_logits", "latent_fp4", "latent_scales", "shared_gate_up",
            "shared_hidden", "route", "tile_route", "flags", "experts", "kof",
            "union_sizes", "intermediate_fp4", "intermediate_scales",
            "finish_scratch", "final",
        )
    }
    buffers["reduce"] = reduce
    buffers["routed"] = reduce[: rows * latent].view(rows, latent)
    buffers["shared"] = reduce[rows * latent :].view(rows, hidden)
    calls = {"all_reduce": []}

    def recorder(name):
        return SimpleNamespace(
            launcher=lambda *args, **_kwargs: calls.setdefault(name, args)
        )

    local = SimpleNamespace(
        front=recorder("front"),
        finish=recorder("finish"),
        finalize=recorder("finalize"),
        buffers=buffers,
        grouped_experts=True,
        replicated_finalize=True,
        herd=False,
    )
    experts = SimpleNamespace(w13_weight_scale=None, w2_weight_scale=None)
    layer = SimpleNamespace(
        mlp_attn_res=lambda *_args: (torch.zeros(rows, 4), torch.zeros(rows, 4)),
        block_sparse_moe=SimpleNamespace(
            experts=experts, routed_expert_norm=lambda value: value
        ),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.rank = 0
    runner.routed = {layer_id: weight}
    runner.inner = SimpleNamespace(layers={layer_id: layer})
    group = SimpleNamespace(
        all_reduce=lambda value, ca_fp8_quant=False: calls["all_reduce"].append(
            value
        ),
    )
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(runner_module.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(runner_module, "fused_moe", lambda *_args, **_kwargs: None)

    _, out = runner._routed_k2(
        SimpleNamespace(k2={layer_id: local}),
        SimpleNamespace(layer_id=layer_id),
        torch.zeros(rows, 4),
        torch.zeros(rows, 4),
        torch.zeros(rows, 3, 4),
    )

    assert [value is reduce for value in calls["all_reduce"]] == [True]
    assert calls["finalize"] == (
        buffers["routed"].data_ptr(),
        buffers["shared"].data_ptr(),
        weight.w_up_full.data_ptr(),
        buffers["final"].data_ptr(),
    )
    assert out is buffers["final"]
