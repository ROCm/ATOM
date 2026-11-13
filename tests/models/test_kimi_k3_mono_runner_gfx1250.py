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
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import (
    GATHER_ALLREDUCE_ABI,
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
    launchers = [_Launcher() for _ in range(6)]
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
        allreduce=SimpleNamespace(launcher=launchers[4], abi=KernelAbi(("data",))),
    )
    build = Gfx1250WidthBuild(
        k1={1: k1, 2: k1},
        k2={1: k2, 2: k2},
        oproj=SimpleNamespace(launcher=launchers[5], abi=KernelAbi(("core",))),
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
    oproj = object()
    built = Gfx1250WidthBuild({1: old, 2: other}, {}, oproj)

    lean = _lean_width(built, [spilled])
    assert lean is not None
    assert lean.k1 == {1: rebuilt, 2: other}
    assert lean.k2 == {}
    assert lean.oproj is oproj
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

    class Seam:
        out_quant_dtype = None
        enabled = False

        def __call__(self, value, visible, add_hidden, add_hidden2):
            calls["seam"] = visible.shape[1]
            return value + add_hidden + 10, value + add_hidden

    self_attention_attn_res = Seam()

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

    out = runner._layer(
        SimpleNamespace(oproj=None), plan, None, prefix, blocks, add, None, None
    )

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


def test_k1_core_goes_through_the_kernel_o_proj_all_reduce_when_built():
    rows, hidden = 2, 4
    prefix = torch.zeros(rows, hidden)
    blocks = torch.zeros(rows, 1, hidden)
    core = torch.full((rows, hidden), 2.0)
    kernel = object()
    calls = {}

    class Seam:
        out_quant_dtype = None
        enabled = False

        def __call__(self, value, visible, add_hidden, add_hidden2):
            return value + add_hidden, value + add_hidden

    def production_o_proj(_value):
        raise AssertionError("o_proj ran on AITER's GEMM")

    layer = SimpleNamespace(
        self_attention_attn_res=Seam(),
        self_attn=SimpleNamespace(o_proj=production_o_proj),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.inner = SimpleNamespace(layers={5: layer})
    runner.routed = {5: object()}
    runner._k1 = lambda *_args: core
    runner._oproj = lambda built, layer_id, value: (
        calls.update(oproj=(built, layer_id, value)) or value + 3
    )
    runner._routed_k2 = lambda _build, _plan, attn, value, _blocks: (
        calls.update(attn=attn) or value,
        attn,
    )
    runner._write_aux = lambda *_args: None
    plan = SimpleNamespace(layer_id=5, mono=True, kda=object())

    runner._layer(
        SimpleNamespace(oproj=kernel), plan, None, prefix, blocks, prefix, None, None
    )

    assert calls["oproj"][0] is kernel and calls["oproj"][1] == 5
    assert calls["oproj"][2] is core
    torch.testing.assert_close(calls["attn"], core + 3)


def test_kernel_o_proj_all_reduce_launch_reads_the_layer_weights():
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    weight, scale = torch.zeros(4), torch.zeros(2)
    runner.oproj_weights = {9: (weight, scale)}
    runner.oproj_peers = torch.zeros(8, dtype=torch.int64)
    runner.k2_peers = SimpleNamespace(epochs=torch.zeros(4, dtype=torch.int32))
    runner.rank = 3
    core = torch.zeros(5, 6, dtype=torch.bfloat16)
    calls = []

    def launch(*args, stream):
        calls.append(args)

    out = runner._oproj(SimpleNamespace(launcher=launch), 9, core)

    assert out.shape == (5, runner_module.HIDDEN) and out.dtype == core.dtype
    (args,) = calls
    assert args == (
        core.data_ptr(),
        weight.data_ptr(),
        scale.data_ptr(),
        out.data_ptr(),
        runner.oproj_peers.data_ptr(),
        runner.k2_peers.epochs.data_ptr(),
        3,
    )


@pytest.mark.parametrize("candidates", (0, 3))
def test_k1_seam_is_bf16_where_out_norm_folds_fp8_quant(monkeypatch, candidates):
    import atom.models.kimi_k3.mono.runner as base_module

    rows, hidden = 2, 4
    prefix = torch.zeros(rows, hidden)
    add = torch.ones_like(prefix)
    blocks = torch.zeros(rows, candidates, hidden)
    calls = {}
    # not callable: the quant-folding site itself must not feed K1
    site = SimpleNamespace(
        out_quant_dtype=dtypes.fp8,
        enabled=True,
        score_weight=torch.ones(hidden),
        eps=1e-6,
        out_eps=1e-5,
        out_norm=SimpleNamespace(weight=torch.ones(hidden), eps=1e-5, dim=hidden),
    )

    def mix(mixed_site, value, visible, add_hidden, add_hidden2):
        calls["mix"] = (mixed_site, visible, add_hidden2)
        return value + add_hidden + 7, value + add_hidden

    def rmsnorm(value, *_args):
        calls["norm"] = value.clone()
        return value + 3

    monkeypatch.setattr(base_module, "_mix_bf16", mix)
    monkeypatch.setattr(base_module, "rmsnorm2d_fwd_", rmsnorm)
    normed, out_prefix = base_module._bf16_attn_res(site, prefix, blocks, add, None)

    torch.testing.assert_close(out_prefix, prefix + add)
    if candidates:
        # the mix itself, normed, unquantized, over the blocks as handed in
        assert calls["mix"][0] is site and calls["mix"][1] is blocks
        assert calls["mix"][2] is None and "norm" not in calls
        torch.testing.assert_close(normed, prefix + add + 7)
    else:
        assert "mix" not in calls
        torch.testing.assert_close(calls["norm"], prefix + add)
        torch.testing.assert_close(normed, prefix + add + 3)


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
    tensor = lambda *shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype)
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
        "router_logits",
        "latent_fp4",
        "latent_scales",
        "shared_gate_up",
        "shared_hidden",
        "route",
        "tile_route",
        "flags",
        "experts",
        "kof",
        "union_sizes",
        "intermediate_fp4",
        "intermediate_scales",
        "routed",
        "shared",
        "finish_scratch",
        "final",
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
        herd_routes=None,
        grouped_moe=None,
        fused_moe=None,
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
        runner_module, "_bf16_attn_res", lambda site, *args: site(*args[:3])
    )
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


@pytest.mark.parametrize("all_reduce", ("collective", "kernel", "gather"))
def test_replicated_finalize_all_reduces_both_partials_once(monkeypatch, all_reduce):
    rows, layer_id, latent, hidden = 2, 3, 3584, 7168
    weight = SimpleNamespace(
        write_idx=-1,
        blocks=2,
        **{
            name: torch.zeros(8, dtype=torch.uint8)
            for name in (
                "w_gate",
                "w_ld",
                "w_sgu",
                "bias",
                "w_sd",
                "w13",
                "w13s",
                "w2",
                "w2s",
                "ln_w",
                "w_up",
                "w_up_full",
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
            "router_logits",
            "latent_fp4",
            "latent_scales",
            "shared_gate_up",
            "shared_hidden",
            "route",
            "tile_route",
            "flags",
            "experts",
            "kof",
            "union_sizes",
            "intermediate_fp4",
            "intermediate_scales",
            "finish_scratch",
            "final",
        )
    }
    buffers["reduce"] = reduce
    buffers["routed"] = reduce[: rows * latent].view(rows, latent)
    buffers["shared"] = reduce[rows * latent :].view(rows, hidden)
    calls = {"all_reduce": []}

    def recorder(name, abi=None):
        return SimpleNamespace(
            launcher=lambda *args, **_kwargs: calls.setdefault(name, args), abi=abi
        )

    grouped_rows = [
        torch.zeros(4, dtype=dtype)
        for dtype in (torch.bfloat16, torch.int32, torch.float32)
    ]

    def grouped_moe(*args, gather):
        calls["grouped_moe"] = (args, gather)
        return tuple(grouped_rows)

    gathering = all_reduce == "gather"
    local = SimpleNamespace(
        front=recorder("front"),
        finish=recorder("finish"),
        finalize=recorder("finalize"),
        buffers=buffers,
        grouped_experts=True,
        replicated_finalize=True,
        herd=gathering,
        herd_routes=object() if gathering else None,
        allreduce=(
            None
            if all_reduce == "collective"
            else recorder("allreduce", GATHER_ALLREDUCE_ABI if gathering else None)
        ),
        grouped_moe=grouped_moe if gathering else None,
        fused_moe=None,
    )
    experts = SimpleNamespace(w13_weight_scale=None, w2_weight_scale=None)
    layer = SimpleNamespace(
        mlp_attn_res=lambda *_args: (torch.zeros(rows, 4), torch.zeros(rows, 4)),
        block_sparse_moe=SimpleNamespace(
            experts=experts, routed_expert_norm=lambda value: value
        ),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.rank = 3
    runner.k2_peers = SimpleNamespace(
        peers=torch.zeros(8, dtype=torch.int64),
        epochs=torch.zeros(256, dtype=torch.int32),
    )
    runner.routed = {layer_id: weight}
    runner.inner = SimpleNamespace(layers={layer_id: layer})
    group = SimpleNamespace(
        all_reduce=lambda value, ca_fp8_quant=False: calls["all_reduce"].append(value),
    )
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(runner_module.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        runner_module, "_bf16_attn_res", lambda site, *args: site(*args[:3])
    )
    monkeypatch.setattr(runner_module, "fused_moe", lambda *_args, **_kwargs: None)

    _, out = runner._routed_k2(
        SimpleNamespace(k2={layer_id: local}),
        SimpleNamespace(layer_id=layer_id),
        torch.zeros(rows, 4),
        torch.zeros(rows, 4),
        torch.zeros(rows, 3, 4),
    )

    if all_reduce == "collective":
        assert [value is reduce for value in calls["all_reduce"]] == [True]
    else:
        assert calls["all_reduce"] == []
        gathered = [t.data_ptr() for t in grouped_rows] if gathering else []
        assert calls["allreduce"] == (
            reduce.data_ptr(),
            *gathered,
            runner.k2_peers.peers.data_ptr(),
            runner.k2_peers.epochs.data_ptr(),
            3,
        )
    if gathering:
        args, gather = calls["grouped_moe"]
        assert gather is False
        assert args[0].data_ptr() == buffers["latent_fp4"].data_ptr()
        assert args[1] is buffers["latent_scales"]
        assert args[6] is local.herd_routes and args[7] is buffers["routed"]
    assert calls["finalize"] == (
        buffers["routed"].data_ptr(),
        buffers["shared"].data_ptr(),
        weight.w_up_full.data_ptr(),
        buffers["final"].data_ptr(),
    )
    assert out is buffers["final"]


@pytest.mark.parametrize("herd", (False, True))
@pytest.mark.parametrize("gathering", (False, True))
def test_fused_moe_reads_the_finish_routes_and_feeds_the_gather(
    monkeypatch, herd, gathering
):
    rows, layer_id, latent, hidden, topk = 2, 3, 3584, 7168, 16
    weight = SimpleNamespace(
        write_idx=-1,
        blocks=2,
        **{
            name: torch.zeros(8, dtype=torch.uint8)
            for name in (
                "w_gate",
                "w_ld",
                "w_sgu",
                "bias",
                "w_sd",
                "w13",
                "w13s",
                "w2",
                "w2s",
                "w_up_full",
            )
        },
        w_sgu_scale=None,
        w_sd_scale=None,
    )
    reduce = torch.zeros(rows * (latent + hidden), dtype=torch.bfloat16)
    buffers = {
        name: torch.zeros(rows, 4, dtype=torch.uint8)
        for name in (
            "router_logits",
            "latent_fp4",
            "latent_scales",
            "shared_gate_up",
            "shared_hidden",
            "flags",
            "experts",
            "kof",
            "union_sizes",
            "intermediate_fp4",
            "intermediate_scales",
            "finish_scratch",
            "final",
            "fused_inter",
        )
    }
    buffers["route"] = torch.zeros(rows, topk, dtype=torch.int32)
    buffers["tile_route"] = torch.zeros(rows, topk)
    buffers["fused_grouped"] = torch.zeros(rows * topk, latent, dtype=torch.bfloat16)
    buffers["fused_rows"] = torch.arange(rows * topk, dtype=torch.int32).view(
        rows, topk
    )
    buffers["reduce"] = reduce
    buffers["routed"] = reduce[: rows * latent].view(rows, latent)
    buffers["shared"] = reduce[rows * latent :].view(rows, hidden)
    calls = {"all_reduce": [], "gather_reduce": []}

    def recorder(name, abi=None):
        return SimpleNamespace(
            launcher=lambda *args, **_kwargs: calls.setdefault(name, args), abi=abi
        )

    herd_weights = torch.zeros(rows, topk)
    herd_ids = torch.zeros(rows, topk, dtype=torch.int32)
    local = SimpleNamespace(
        front=recorder("front"),
        finish=recorder("finish"),
        finalize=recorder("finalize"),
        buffers=buffers,
        grouped_experts=True,
        replicated_finalize=True,
        herd=herd,
        herd_routes=SimpleNamespace(routes=(herd_weights, herd_ids)) if herd else None,
        allreduce=recorder("allreduce", GATHER_ALLREDUCE_ABI) if gathering else None,
        grouped_moe=None,
        fused_moe=recorder("fused_moe"),
    )
    experts = SimpleNamespace(
        w13_weight_scale=torch.zeros(4, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(4, dtype=torch.uint8),
    )
    layer = SimpleNamespace(
        mlp_attn_res=lambda *_args: (torch.zeros(rows, 4), torch.zeros(rows, 4)),
        block_sparse_moe=SimpleNamespace(
            experts=experts, routed_expert_norm=lambda value: value
        ),
    )
    runner = object.__new__(K3Gfx1250MonoDecodeRunner)
    runner.rank = 1
    runner.k2_peers = SimpleNamespace(
        peers=torch.zeros(8, dtype=torch.int64),
        epochs=torch.zeros(256, dtype=torch.int32),
    )
    runner.fused_moe_sync = (
        torch.zeros(64, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )
    runner.routed = {layer_id: weight}
    runner.inner = SimpleNamespace(layers={layer_id: layer})
    group = SimpleNamespace(
        all_reduce=lambda value, ca_fp8_quant=False: calls["all_reduce"].append(value),
    )
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(runner_module.torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        runner_module, "_bf16_attn_res", lambda site, *args: site(*args[:3])
    )
    monkeypatch.setattr(
        runner_module,
        "flydsl_moe_gather_reduce",
        lambda *args, out: calls["gather_reduce"].append((args, out)),
    )

    runner._routed_k2(
        SimpleNamespace(k2={layer_id: local}),
        SimpleNamespace(layer_id=layer_id),
        torch.zeros(rows, 4),
        torch.zeros(rows, 4),
        torch.zeros(rows, 3, 4),
    )

    weights, ids = (
        (herd_weights, herd_ids) if herd else (buffers["tile_route"], buffers["route"])
    )
    flags, state = runner.fused_moe_sync
    assert calls["fused_moe"] == (
        ids.data_ptr(),
        buffers["latent_fp4"].data_ptr(),
        buffers["latent_scales"].data_ptr(),
        weight.w13.data_ptr(),
        experts.w13_weight_scale.data_ptr(),
        weight.w2.data_ptr(),
        experts.w2_weight_scale.data_ptr(),
        buffers["fused_inter"].data_ptr(),
        buffers["fused_grouped"].data_ptr(),
        flags.data_ptr(),
        state.data_ptr(),
    )
    if gathering:
        assert calls["gather_reduce"] == [] and calls["all_reduce"] == []
        assert calls["allreduce"][:4] == (
            reduce.data_ptr(),
            buffers["fused_grouped"].data_ptr(),
            buffers["fused_rows"].data_ptr(),
            weights.data_ptr(),
        )
    else:
        ((args, out),) = calls["gather_reduce"]
        assert args[0].shape == (1, rows * topk, latent)
        assert args[1] is buffers["fused_rows"] and args[2] is weights
        assert out is buffers["routed"]
        assert [value is reduce for value in calls["all_reduce"]] == [True]


def test_precomputed_herd_hands_over_routes_laid_out_for_the_tile(monkeypatch):
    from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
        herd_scratch_layout,
    )

    rows = 20
    layout = herd_scratch_layout(rows)
    scratch = torch.zeros(
        max(at + 4 * words for at, words in layout.values()), dtype=torch.uint8
    )
    candidates = torch.zeros((rows, 17), dtype=torch.int32)
    values = torch.zeros((rows, 17))
    herd = runner_module._PrecomputedHerd(scratch, candidates, values, rows)
    tile_at = layout["tile_m"][0]
    assert int(scratch[tile_at : tile_at + 4].view(torch.int32)) == 16
    scale_at = layout["routed_scaling_factor"][0]
    assert float(scratch[scale_at : scale_at + 4].view(torch.float32)) == 1.0

    routes = herd.route_psum(16)
    assert routes[1].shape == (rows, 16) and routes[2].shape == (896,)
    assert routes[0].data_ptr() == scratch.data_ptr() + layout["topk_weights"][0]

    calls = []
    monkeypatch.setattr(
        runner_module,
        "herd_route_psum",
        lambda *args: calls.append(args) or ("finalized",),
    )
    assert herd.route_psum(32) == ("finalized",)
    assert calls[0][3] == 32
    assert int(scratch[tile_at : tile_at + 4].view(torch.int32)) == 32
    assert herd.route_psum(32) is herd.routes
