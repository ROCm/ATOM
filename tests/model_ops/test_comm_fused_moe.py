# SPDX-License-Identifier: MIT
"""CPU coverage for the ATOM to AITer comm-fused MoE boundary."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "atom/model_ops/fused_moe/comm_fused_moe.py"
)


@dataclass(frozen=True)
class _ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int


class _FusedMoE:
    def process_weights_after_loading(self) -> None:
        self.base_weights_processed = True


class _Mxfp4MoEMethod:
    use_triton = True
    use_triton_decode = True


class _CommFusedMoeRuntime:
    def __init__(self, *, runners) -> None:
        self.runners = runners

    def supports(self, tokens: int) -> bool:
        return self.runners.supports(tokens)


class _State:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            enable_expert_parallel=False,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            prefill_context_parallel_size=1,
        )
        self.shape_keys = []
        self.runner_calls = []
        self.registrations = []
        self.missing_shape = False


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def comm_fused_module(monkeypatch):
    state = _State()
    tp_group = SimpleNamespace(world_size=8)

    def winners_for(shape):
        state.shape_keys.append(shape)
        if state.missing_shape:
            raise KeyError(shape)
        return ["winner"]

    def create_flydsl_comm_fused_runners(**kwargs):
        state.runner_calls.append(kwargs)
        return SimpleNamespace(supports=lambda tokens: tokens == 32)

    def direct_register_custom_op(**kwargs):
        state.registrations.append(kwargs)

    aiter = _module("aiter")
    aiter.__path__ = []
    aiter_dist = _module("aiter.dist")
    aiter_dist.__path__ = []
    aiter_parallel_state = _module(
        "aiter.dist.parallel_state", get_tp_group=lambda: tp_group
    )
    aiter_jit = _module("aiter.jit")
    aiter_jit.__path__ = []
    aiter_jit_utils = _module("aiter.jit.utils")
    aiter_jit_utils.__path__ = []
    aiter_chip_info = _module(
        "aiter.jit.utils.chip_info", get_gfx_runtime=lambda: "gfx950"
    )
    aiter_ops = _module("aiter.ops")
    aiter_ops.__path__ = []
    aiter_flydsl = _module("aiter.ops.flydsl")
    aiter_flydsl.__path__ = []
    aiter_moe_common = _module(
        "aiter.ops.flydsl.moe_common",
        GateMode=SimpleNamespace(
            INTERLEAVE=SimpleNamespace(value="interleave"),
            SEPARATED=SimpleNamespace(value="separated"),
        ),
    )
    aiter_host = _module(
        "aiter.ops.flydsl.comm_fused_moe_host",
        ShapeKey=_ShapeKey,
        winners_for=winners_for,
        create_flydsl_comm_fused_runners=create_flydsl_comm_fused_runners,
    )
    aiter_runtime = _module(
        "aiter.ops.comm_fused_moe_runtime",
        CommFusedMoeRuntime=_CommFusedMoeRuntime,
    )
    atom_config = _module(
        "atom.config", get_current_atom_config=lambda: state.config
    )
    atom_moe = _module(
        "atom.model_ops.moe",
        FusedMoE=_FusedMoE,
        Mxfp4MoEMethod=_Mxfp4MoEMethod,
    )
    atom_custom_register = _module(
        "atom.utils.custom_register",
        direct_register_custom_op=direct_register_custom_op,
    )

    modules = {
        "aiter": aiter,
        "aiter.dist": aiter_dist,
        "aiter.dist.parallel_state": aiter_parallel_state,
        "aiter.jit": aiter_jit,
        "aiter.jit.utils": aiter_jit_utils,
        "aiter.jit.utils.chip_info": aiter_chip_info,
        "aiter.ops": aiter_ops,
        "aiter.ops.flydsl": aiter_flydsl,
        "aiter.ops.flydsl.moe_common": aiter_moe_common,
        "aiter.ops.flydsl.comm_fused_moe_host": aiter_host,
        "aiter.ops.comm_fused_moe_runtime": aiter_runtime,
        "atom.config": atom_config,
        "atom.model_ops.moe": atom_moe,
        "atom.utils.custom_register": atom_custom_register,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_comm_fused_moe_test", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, state


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("enable_expert_parallel", True),
        ("data_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
    ],
)
def test_supports_model_rejects_unsupported_parallel_layout(
    comm_fused_module, attribute, value
):
    module, state = comm_fused_module
    if attribute == "data_parallel_size":
        state.config.parallel_config.data_parallel_size = value
    else:
        setattr(state.config, attribute, value)

    assert not module.CommFusedMoe.supports_model(
        model_dim=7168, inter_dim=384, experts=384, topk=6
    )
    assert not state.shape_keys


def test_supports_model_honors_disable_flag(monkeypatch, comm_fused_module):
    module, state = comm_fused_module
    monkeypatch.setenv("AITER_DISABLE_COMM_FUSED_MOE", "1")

    assert not module.CommFusedMoe.supports_model(
        model_dim=7168, inter_dim=384, experts=384, topk=6
    )
    assert not state.shape_keys


def test_supports_model_uses_runtime_shape_and_fails_closed(comm_fused_module):
    module, state = comm_fused_module

    assert module.CommFusedMoe.supports_model(
        model_dim=7168, inter_dim=384, experts=384, topk=6
    )
    assert state.shape_keys == [_ShapeKey("gfx950", 7168, 384, 384, 6, 8)]

    state.missing_shape = True
    assert not module.CommFusedMoe.supports_model(
        model_dim=7168, inter_dim=384, experts=384, topk=6
    )


def test_runner_wiring_has_no_diagnostic_dispatch_gate(
    monkeypatch, comm_fused_module
):
    module, state = comm_fused_module
    layer = module.CommFusedMoe.__new__(module.CommFusedMoe)
    layer.quant_method = module.Mxfp4MoEMethod()
    layer.dp_size = 1
    layer.use_ep = False
    layer.tp_size = 8
    layer.hidden_size = 7168
    layer.intermediate_size_per_partition = 384
    layer.global_num_experts = 384
    layer.top_k = 6

    layer.process_weights_after_loading()

    assert layer.base_weights_processed
    assert not layer.quant_method.use_triton
    assert not layer.quant_method.use_triton_decode
    assert len(state.runner_calls) == 1
    runner_call = state.runner_calls[0]
    assert runner_call["tp_group"].world_size == 8
    assert runner_call == {
        "tp_group": runner_call["tp_group"],
        "model_dim": 7168,
        "inter_dim": 384,
        "experts": 384,
        "topk": 6,
    }
    assert state.registrations[0]["mutates_args"] == ["shared_partial"]

    monkeypatch.setenv("AITER_COMM_FUSED_MOE_MAX_TOKENS", "1")
    assert layer.supports_comm_fused(32)
