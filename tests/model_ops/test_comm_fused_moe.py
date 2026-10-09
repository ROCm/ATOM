# SPDX-License-Identifier: MIT
"""CPU coverage for ATOM's communication-fused MoE integration.

The TP8 numerical/kernel coverage lives in AITer's
``op_tests/multigpu_tests/test_comm_fused_moe.py``. These tests exercise the
real ATOM classes and only replace the unavailable AITER/Triton boundary, so
the non-GPU CI catches API and DSV4 dispatch drift instead of passing against
hand-written ATOM stand-ins.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from conftest import atom_config_double


@dataclass(frozen=True)
class _ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int


class _ActivationType(Enum):
    Silu = 0
    Swiglu = 1
    Situv2 = 2


class _QuantType(Enum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x128 = 3
    per_1x32 = 4


class _GateMode(Enum):
    INTERLEAVE = "interleave"
    SEPARATED = "separated"


class _CommFusedMoeRuntime:
    def __init__(self, *, runners) -> None:
        self.runners = runners

    def supports(self, tokens: int) -> bool:
        return self.runners.supports(tokens)


class _State:
    def __init__(self) -> None:
        self.config = atom_config_double(
            enable_tbo=False,
            enable_rapidserve=False,
            fake_eplb=False,
            enable_expert_parallel=False,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            prefill_context_parallel_size=1,
            torch_dtype=torch.bfloat16,
        )
        self.shape_keys = []
        self.runner_calls = []
        self.missing_shape = False


def _identity_decorator(*_args, **_kwargs):
    return lambda function: function


def _external_module_attributes(name: str) -> dict[str, object]:
    dtypes = SimpleNamespace(
        bf16=torch.bfloat16,
        fp32=torch.float32,
        fp4x2=torch.uint8,
        fp8=torch.float8_e4m3fnuz,
        fp8_e8m0=torch.uint8,
        i4x2=torch.uint8,
        i8=torch.int8,
        d_dtypes={},
    )
    return {
        "aiter": {
            "ActivationType": _ActivationType,
            "QuantType": _QuantType,
            "dtypes": dtypes,
        },
        "aiter.jit.utils.torch_guard": {
            "torch_compile_guard": _identity_decorator,
        },
        "aiter.ops.flydsl.moe_common": {"GateMode": _GateMode},
        "triton": {
            "jit": lambda function: function,
            "heuristics": _identity_decorator,
        },
        "triton.language": {"constexpr": object},
    }.get(name, {})


class _ExternalModuleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        module.__file__ = f"<test stub for {module.__name__}>"
        module.__path__ = []
        module.__dict__.update(_external_module_attributes(module.__name__))

        def resolve(attribute: str):
            value = MagicMock(name=f"{module.__name__}.{attribute}")
            setattr(module, attribute, value)
            return value

        module.__getattr__ = resolve


class _ExternalModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {"aiter", "triton"} or fullname.startswith(
            ("aiter.", "triton.")
        ):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@contextmanager
def _stubbed_external_kernel_modules():
    # Torch imports its own optional Triton integration lazily. Resolve that
    # first so this test's deliberately small Triton stub cannot affect it.
    importlib.import_module("torch._dynamo.config")

    prefixes = ("aiter", "triton")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name in prefixes or name.startswith(("aiter.", "triton."))
    }
    for name in saved:
        sys.modules.pop(name, None)

    finder = _ExternalModuleFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name in prefixes or name.startswith(("aiter.", "triton.")):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def atom_modules():
    existing_atom_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "atom" or name.startswith("atom.")
    }
    with _stubbed_external_kernel_modules():
        moe = importlib.import_module("atom.model_ops.moe")
        comm = importlib.import_module("atom.model_ops.fused_moe.comm_fused_moe")
        deepseek_v4 = importlib.import_module("atom.models.deepseek_v4")
        host = importlib.import_module("aiter.ops.flydsl.comm_fused_moe_host")
        runtime = importlib.import_module("aiter.ops.comm_fused_moe_runtime")
        try:
            yield SimpleNamespace(
                moe=moe,
                comm=comm,
                deepseek_v4=deepseek_v4,
                host=host,
                runtime=runtime,
            )
        finally:
            for name in list(sys.modules):
                if (name == "atom" or name.startswith("atom.")) and (
                    name not in existing_atom_modules
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(existing_atom_modules)


@pytest.fixture
def comm_fused_env(monkeypatch, atom_modules):
    state = _State()
    tp_group = SimpleNamespace(world_size=8)
    monkeypatch.setenv("ATOM_MOE_GU_ITLV", "1")

    def winners_for(shape):
        state.shape_keys.append(shape)
        if state.missing_shape:
            raise KeyError(shape)
        return ["winner"]

    def create_flydsl_comm_fused_runners(**kwargs):
        state.runner_calls.append(kwargs)
        return SimpleNamespace(supports=lambda tokens: tokens == 32)

    monkeypatch.setattr(
        atom_modules.comm, "get_current_atom_config", lambda: state.config
    )
    monkeypatch.setattr(atom_modules.comm, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(atom_modules.comm, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(atom_modules.host, "ShapeKey", _ShapeKey, raising=False)
    monkeypatch.setattr(atom_modules.host, "winners_for", winners_for, raising=False)
    monkeypatch.setattr(
        atom_modules.host,
        "create_flydsl_comm_fused_runners",
        create_flydsl_comm_fused_runners,
        raising=False,
    )
    monkeypatch.setattr(
        atom_modules.runtime,
        "CommFusedMoeRuntime",
        _CommFusedMoeRuntime,
        raising=False,
    )
    return atom_modules, state, tp_group


def _create_backend(modules, state, **overrides):
    arguments = {
        "layer_quant_config": SimpleNamespace(
            quant_dtype=modules.comm.dtypes.fp4x2,
            quant_type=modules.comm.QuantType.per_1x32,
        ),
        "online_quant": False,
        "parallel_config": SimpleNamespace(dp_size=1, use_ep=False, tp_size=8),
        "model_dim": 7168,
        "inter_dim": 384,
        "experts": 384,
        "topk": 6,
        "activation": modules.comm.ActivationType.Silu,
        "apply_router_weight_on_input": False,
    }
    arguments.update(overrides)
    return modules.comm.create_comm_fused_moe_backend(**arguments)


def _new_fused_moe(modules):
    # FusedMoE is exposed through the repository's lazy plugin-mode decorator.
    layer = object.__new__(modules.moe.FusedMoE.__mro__[1])
    torch.nn.Module.__init__(layer)
    return layer


def test_registers_functional_custom_op(atom_modules):

    schema = torch._C._dispatch_find_schema_or_throw(
        "aiter::comm_fused_moe_forward", ""
    ).schema()
    assert all(argument.alias_info is None for argument in schema.arguments)


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("moe_backend", "mega"),
        ("online_quant", True),
        ("quant_dtype", torch.bfloat16),
        ("quant_type", _QuantType.per_Token),
        ("torch_dtype", torch.float16),
        ("activation", _ActivationType.Swiglu),
        ("apply_router_weight_on_input", True),
        ("enable_tbo", True),
        ("enable_rapidserve", True),
        ("fake_eplb", True),
        ("enable_expert_parallel", True),
        ("data_parallel_size", 2),
        ("use_ep", True),
        ("prefill_context_parallel_size", 2),
        ("tp_size", 1),
    ],
)
def test_support_predicate_rejects_unsupported_configuration(
    comm_fused_env, attribute, value
):
    modules, state, _ = comm_fused_env
    overrides = {}
    if attribute == "data_parallel_size":
        overrides["parallel_config"] = SimpleNamespace(
            dp_size=value, use_ep=False, tp_size=8
        )
    elif attribute == "use_ep":
        overrides["parallel_config"] = SimpleNamespace(
            dp_size=1, use_ep=value, tp_size=8
        )
    elif attribute == "tp_size":
        overrides["parallel_config"] = SimpleNamespace(
            dp_size=1, use_ep=False, tp_size=value
        )
    elif attribute == "online_quant":
        overrides[attribute] = value
    elif attribute == "quant_dtype":
        overrides["layer_quant_config"] = SimpleNamespace(
            quant_dtype=value,
            quant_type=modules.comm.QuantType.per_1x32,
        )
    elif attribute == "quant_type":
        overrides["layer_quant_config"] = SimpleNamespace(
            quant_dtype=modules.comm.dtypes.fp4x2,
            quant_type=value,
        )
    elif attribute in {"activation", "apply_router_weight_on_input"}:
        overrides[attribute] = value
    else:
        setattr(state.config, attribute, value)

    assert _create_backend(modules, state, **overrides) is None
    assert not state.shape_keys


def test_support_predicate_honors_disable_flag(monkeypatch, comm_fused_env):
    modules, state, _ = comm_fused_env
    monkeypatch.setenv("AITER_DISABLE_COMM_FUSED_MOE", "1")

    assert _create_backend(modules, state) is None
    assert not state.shape_keys


def test_support_predicate_requires_interleaved_gate_up(monkeypatch, comm_fused_env):
    modules, state, _ = comm_fused_env
    monkeypatch.setenv("ATOM_MOE_GU_ITLV", "0")

    assert _create_backend(modules, state) is None
    assert not state.shape_keys


@pytest.mark.parametrize(
    "missing_module",
    [
        "aiter.ops.flydsl.comm_fused_moe_host",
        "aiter.ops.comm_fused_moe_runtime",
    ],
)
def test_support_predicate_falls_back_when_aiter_backend_is_missing(
    monkeypatch, comm_fused_env, missing_module
):
    modules, state, _ = comm_fused_env
    import_module = modules.comm.importlib.import_module

    def import_without_comm_fused_backend(name):
        if name == missing_module:
            raise ModuleNotFoundError(name)
        return import_module(name)

    monkeypatch.setattr(
        modules.comm.importlib, "import_module", import_without_comm_fused_backend
    )

    assert _create_backend(modules, state) is None
    assert not state.shape_keys


def test_support_predicate_uses_runtime_shape_and_fails_closed(comm_fused_env):
    modules, state, _ = comm_fused_env

    assert _create_backend(modules, state) is not None
    assert state.shape_keys == [_ShapeKey("gfx950", 7168, 384, 384, 6, 8)]

    state.missing_shape = True
    assert _create_backend(modules, state) is None


def test_runner_wiring_uses_real_moe_types(comm_fused_env):
    modules, state, tp_group = comm_fused_env
    backend = _create_backend(modules, state)
    layer = _new_fused_moe(modules)
    layer.quant_method = object.__new__(modules.moe.Mxfp4MoEMethod)
    layer.quant_method.use_triton = True
    layer.quant_method.use_triton_decode = True
    layer.moe_parallel_config = SimpleNamespace(
        dp_size=1,
        use_ep=False,
        tp_size=8,
    )
    layer.hidden_size = 7168
    layer.intermediate_size_per_partition = 384
    layer.global_num_experts = 384
    layer.top_k = 6

    backend.initialize(layer)

    assert not layer.quant_method.use_triton
    assert not layer.quant_method.use_triton_decode
    assert state.runner_calls == [
        {
            "tp_group": tp_group,
            "model_dim": 7168,
            "inter_dim": 384,
            "experts": 384,
            "topk": 6,
        }
    ]
    assert backend.supports(32)
    assert not backend.supports(31)


def test_missing_runtime_falls_back(comm_fused_env):
    modules, state, _ = comm_fused_env
    backend = _create_backend(modules, state)

    assert not backend.supports(32)


def test_forward_impl_bridges_fused_moe_state_to_runtime(comm_fused_env):
    modules, state, _ = comm_fused_env
    backend = _create_backend(modules, state)
    layer = _new_fused_moe(modules)

    hidden_states, router_logits, topk_weights, topk_ids = (object() for _ in range(4))
    shared_partial, stage2_stream, output = (object() for _ in range(3))
    before_stage2 = MagicMock(name="before_stage2")

    method = SimpleNamespace(
        select_experts_with_record=MagicMock(return_value=(topk_weights, topk_ids)),
        quant_type=object(),
        hidden_pad=17,
        intermediate_pad=19,
        is_guinterleave=True,
    )
    runtime = SimpleNamespace(run=MagicMock(return_value=output))
    layer_state = {
        "quant_method": method,
        "use_grouped_topk": False,
        "top_k": 2,
        "renormalize": True,
        "topk_group": None,
        "num_expert_group": None,
        "global_num_experts": 16,
        "scoring_func": "sqrtsoftplus",
        "shared_expert_scoring_func": None,
        "apply_router_weight_on_input": True,
        "swiglu_limit": 7.0,
    }
    for name, value in layer_state.items():
        setattr(layer, name, value)
    opaque_attributes = (
        "custom_routing_function e_score_correction_bias w13_weight w2_weight "
        "expert_mask activation w13_weight_scale w2_weight_scale "
        "w13_input_scale w2_input_scale w13_bias w2_bias"
    )
    for name in opaque_attributes.split():
        setattr(layer, name, object())
    backend.runtime = runtime

    result = backend.forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_partial,
        before_stage2=before_stage2,
        stage2_stream=stage2_stream,
    )

    assert result is output
    select_args = method.select_experts_with_record.call_args.kwargs
    assert select_args["layer"] is layer
    assert select_args["hidden_states"] is hidden_states
    assert select_args["router_logits"] is router_logits
    passthrough = {
        "w1": "w13_weight",
        "w2": "w2_weight",
        "expert_mask": "expert_mask",
        "activation": "activation",
        "w1_scale": "w13_weight_scale",
        "w2_scale": "w2_weight_scale",
        "a1_scale": "w13_input_scale",
        "a2_scale": "w2_input_scale",
        "bias1": "w13_bias",
        "bias2": "w2_bias",
    }
    expected_runtime_args = {
        "hidden_states": hidden_states,
        "topk_weight": topk_weights,
        "topk_ids": topk_ids,
        "quant_type": method.quant_type,
        "doweight_stage1": True,
        "hidden_pad": 17,
        "intermediate_pad": 19,
        "swiglu_limit": 7.0,
        "gate_mode": _GateMode.INTERLEAVE.value,
        "shared_partial": shared_partial,
        "before_stage2": before_stage2,
        "stage2_stream": stage2_stream,
    }
    expected_runtime_args.update(
        {
            argument: getattr(layer, attribute)
            for argument, attribute in passthrough.items()
        }
    )
    runtime.run.assert_called_once_with(**expected_runtime_args)


@pytest.mark.parametrize("supported", [True, False])
def test_fused_moe_dispatches_optional_backend(monkeypatch, atom_modules, supported):
    layer = _new_fused_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    shared_partial = torch.randn_like(hidden_states)
    fused_output = torch.randn_like(hidden_states)
    ordinary_output = torch.randn_like(hidden_states)
    backend = SimpleNamespace(
        supports=MagicMock(return_value=supported),
        forward=MagicMock(return_value=fused_output),
    )
    layer._comm_fused_moe = backend
    ordinary_forward = MagicMock(return_value=ordinary_output)
    monkeypatch.setattr(type(layer), "forward", ordinary_forward)

    output, is_complete = layer.forward_maybe_comm_fused(
        hidden_states, router_logits, shared_partial
    )

    assert is_complete is supported
    if supported:
        assert output is fused_output
        backend.forward.assert_called_once_with(
            layer,
            hidden_states,
            router_logits,
            shared_partial,
            before_stage2=None,
            stage2_stream=None,
        )
        ordinary_forward.assert_not_called()
    else:
        assert output is ordinary_output
        backend.forward.assert_not_called()
        ordinary_forward.assert_called_once()
        call_args = ordinary_forward.call_args.args
        assert call_args[0] is hidden_states
        assert call_args[1] is router_logits


@pytest.mark.parametrize("supported", [True, False])
def test_dsv4_single_stream_dispatches_by_token_support(atom_modules, supported):
    moe = atom_modules.deepseek_v4.MoE.__new__(atom_modules.deepseek_v4.MoE)
    torch.nn.Module.__init__(moe)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    shared_partial = torch.randn_like(hidden_states)
    fused_output = torch.randn_like(hidden_states)
    routed_output = torch.randn_like(hidden_states)
    combined_output = torch.randn_like(hidden_states)

    moe.prefix = "model.layers.3.mlp"
    moe.gate = MagicMock(return_value=router_logits)
    moe.shared_experts = MagicMock(return_value=shared_partial)
    moe.experts = MagicMock()
    moe.experts.forward_maybe_comm_fused.return_value = (
        fused_output if supported else routed_output,
        supported,
    )
    moe.combine_outputs = MagicMock(return_value=combined_output)

    result = atom_modules.deepseek_v4.MoE.single_stream_moe_forward(moe, hidden_states)

    if supported:
        assert result is fused_output
        moe.combine_outputs.assert_not_called()
    else:
        assert result is combined_output
        moe.combine_outputs.assert_called_once_with(
            routed_output,
            shared_partial,
            prefix="model.layers.3.mlp.combine_outputs",
        )
    moe.experts.forward_maybe_comm_fused.assert_called_once_with(
        hidden_states,
        router_logits,
        shared_partial,
        before_stage2=None,
        stage2_stream=None,
    )


def test_dsv4_forward_keeps_comm_fused_dispatch_opaque_without_dual_stream(
    monkeypatch, atom_modules
):
    moe = atom_modules.deepseek_v4.MoE.__new__(atom_modules.deepseek_v4.MoE)
    torch.nn.Module.__init__(moe)
    hidden_states = torch.randn(4, 8)
    output = torch.randn_like(hidden_states)
    dispatch = MagicMock(return_value=output)

    moe.dim = hidden_states.shape[1]
    moe.prefix = "model.layers.3.mlp"
    moe._use_dual_stream = False
    moe._use_comm_fused_dispatch = True
    moe.single_stream_moe_forward = MagicMock()
    monkeypatch.setattr(
        atom_modules.deepseek_v4.torch.ops.aiter,
        "maybe_dual_stream_forward",
        dispatch,
    )

    result = atom_modules.deepseek_v4.MoE.forward(moe, hidden_states)

    assert result is output
    dispatch.assert_called_once_with(hidden_states, moe.prefix)
    moe.single_stream_moe_forward.assert_not_called()


def test_dsv4_ordinary_dual_stream_fallback_waits_on_routed_stream(
    monkeypatch, atom_modules
):
    moe = atom_modules.deepseek_v4.MoE.__new__(atom_modules.deepseek_v4.MoE)
    torch.nn.Module.__init__(moe)
    hidden_states = torch.randn(4, 8)
    routed = torch.randn_like(hidden_states)
    shared = torch.randn_like(hidden_states)
    output = torch.randn_like(hidden_states)
    routed_stream = MagicMock(name="routed_stream")
    alt_stream = MagicMock(name="alt_stream")

    moe.prefix = "model.layers.3.mlp"
    moe.alt_stream = alt_stream
    moe.shared_experts = MagicMock()
    moe.shared_experts.forward.return_value = shared
    moe.routed_expert_forward = MagicMock(return_value=(routed, False))
    moe.combine_outputs = MagicMock(return_value=output)
    monkeypatch.setattr(
        atom_modules.deepseek_v4.torch.cuda,
        "current_stream",
        lambda _device: routed_stream,
    )
    monkeypatch.setattr(
        atom_modules.deepseek_v4.torch.cuda,
        "stream",
        lambda _stream: contextmanager(lambda: (yield))(),
    )

    result = atom_modules.deepseek_v4.MoE.dual_stream_moe_forward(moe, hidden_states)

    assert result is output
    assert alt_stream.wait_stream.call_args_list[0].args == (routed_stream,)
    routed_stream.wait_stream.assert_called_once_with(alt_stream)
    moe.combine_outputs.assert_called_once_with(
        routed, shared, prefix="model.layers.3.mlp.combine_outputs"
    )
