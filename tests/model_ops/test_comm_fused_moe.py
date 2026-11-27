# SPDX-License-Identifier: MIT
"""CPU coverage for ATOM's communication-fused MoE integration.

The TP8 numerical/kernel coverage lives in AITer's
``op_tests/multigpu_tests/test_comm_fused_moe.py``. These tests exercise the
real ATOM classes and only replace the unavailable AITER/Triton boundary, so
the non-GPU CI catches API and DSV4 dispatch drift instead of passing against
hand-written ATOM stand-ins.
"""

from __future__ import annotations

import ast
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from aiter_stub import ActivationType, GateMode, QuantType
from conftest import atom_config_double


@dataclass(frozen=True)
class _ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp_size: int
    cu_num: int | None = None
    act_type: str = "ActivationType.Silu"
    dtype: str = "torch.bfloat16"
    q_dtype_a: str = "torch.float8_e4m3fn"
    q_dtype_w: str = "torch.float4_e2m1fn_x2"
    q_type: str = "QuantType.per_1x32"
    use_g1u1: int = 1
    doweight_stage1: int = 0
    add_shared: bool | None = True
    comm: str = "ar"
    gpu_model: str | None = None


class _CommFusedMoeRuntime:
    def __init__(self, *, runners) -> None:
        self.runners = runners

    def supports(self, tokens: int) -> bool:
        return self.runners.supports(tokens)

    def bucket_for(self, tokens: int) -> int:
        return self.runners.bucket_for(tokens)

    def supports_ragged_m(self, tokens: int) -> bool:
        return self.runners.supports_ragged_m(tokens)


class _State:
    def __init__(self) -> None:
        self.config = atom_config_double(
            enable_tbo=False,
            enable_rapidserve=False,
            fake_eplb=False,
            enable_expert_parallel=False,
            enable_dp_attention=False,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            prefill_context_parallel_size=1,
            torch_dtype=torch.bfloat16,
        )
        self.shape_keys = []
        self.runner_calls = []
        self.missing_shape = False


@pytest.fixture
def comm_fused_env(monkeypatch, atom_modules):
    state = _State()
    tp_group = SimpleNamespace(world_size=8)
    dp_group = SimpleNamespace(world_size=8)
    monkeypatch.setenv("ATOM_MOE_GU_ITLV", "1")

    def winners_for(shape):
        state.shape_keys.append(shape)
        if state.missing_shape:
            raise KeyError(shape)
        return ["winner"]

    def create_flydsl_comm_fused_runners(**kwargs):
        state.runner_calls.append(kwargs)
        return SimpleNamespace(
            supports=lambda tokens: tokens == 32,
            bucket_for=lambda tokens: tokens,
            supports_ragged_m=lambda tokens: False,
        )

    monkeypatch.setattr(
        atom_modules.comm, "get_current_atom_config", lambda: state.config
    )
    monkeypatch.setattr(atom_modules.comm, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(atom_modules.comm, "get_dp_group", lambda: dp_group)
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
    return atom_modules, state, tp_group, dp_group


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
    layer.custom_routing_function = None
    return layer


def _parallel(dp_size=1):
    return SimpleNamespace(dp_size=dp_size, use_ep=False, tp_size=8)


def _create_dpa_backend(comm_fused_env):
    modules, state, _, _ = comm_fused_env
    state.config.enable_dp_attention = True
    return (
        modules,
        state,
        _create_backend(modules, state, parallel_config=_parallel(dp_size=8)),
    )


def _set_forward_context(monkeypatch, modules, *, metadata, tokens, unified=False):
    forward_context = importlib.import_module("atom.utils.forward_context")
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=metadata,
            context=SimpleNamespace(
                running_tokens=tokens,
                running_tokens_are_unified=unified,
            ),
        ),
    )


_RUNTIME_PASSTHROUGH = {
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


def _runtime_layer(modules, topk_weights, topk_ids, *, apply_router_weight=False):
    layer = _new_fused_moe(modules)
    method = SimpleNamespace(
        select_experts=MagicMock(return_value=(topk_weights, topk_ids)),
        place_experts=MagicMock(side_effect=lambda layer, w, ids, gathered: (w, ids)),
        quant_type=object(),
        hidden_pad=17,
        intermediate_pad=19,
        is_guinterleave=True,
    )
    values = {
        "quant_method": method,
        "use_grouped_topk": False,
        "top_k": 2,
        "renormalize": True,
        "topk_group": None,
        "num_expert_group": None,
        "global_num_experts": 16,
        "scoring_func": "sqrtsoftplus",
        "shared_expert_scoring_func": None,
        "apply_router_weight_on_input": apply_router_weight,
        "swiglu_limit": 7.0,
        # No prefilled shared columns to re-mark after the DPA gather.
        "expert_layout": SimpleNamespace(mode=None),
    }
    for name, value in values.items():
        setattr(layer, name, value)
    for name in {
        "custom_routing_function",
        "e_score_correction_bias",
        *_RUNTIME_PASSTHROUGH.values(),
    }:
        setattr(layer, name, object())
    return layer, method


def _new_v4_moe(modules):
    moe = modules.deepseek_v4.MoE.__new__(modules.deepseek_v4.MoE)
    torch.nn.Module.__init__(moe)
    moe.prefix = "model.layers.3.mlp"
    return moe


def _mock_streams(monkeypatch, modules):
    routed_stream = MagicMock(name="routed_stream")
    alt_stream = MagicMock(name="alt_stream")
    monkeypatch.setattr(
        modules.deepseek_v4.torch.cuda,
        "current_stream",
        lambda _device: routed_stream,
    )
    monkeypatch.setattr(
        modules.deepseek_v4.torch.cuda,
        "stream",
        lambda _stream: contextmanager(lambda: (yield))(),
    )
    return routed_stream, alt_stream


def _initialize_backend(comm_fused_env, *, dpa=False):
    modules, state, tp_group, dp_group = comm_fused_env
    if dpa:
        state.config.enable_dp_attention = True
    backend = _create_backend(
        modules,
        state,
        parallel_config=_parallel(dp_size=8) if dpa else _parallel(),
    )
    layer = _new_fused_moe(modules)
    layer.quant_method = object.__new__(modules.moe.Mxfp4MoEMethod)
    layer.quant_method.use_triton = True
    layer.quant_method.use_triton_decode = True
    layer.moe_parallel_config = _parallel(dp_size=8 if dpa else 1)
    layer.hidden_size = 7168
    layer.intermediate_size_per_partition = 384
    layer.global_num_experts = 384
    layer.top_k = 6
    if dpa:
        layer.w2_weight = torch.nn.Parameter(torch.empty(1))
        layer.w2_weight_scale = torch.nn.Parameter(torch.empty(1))
    backend.configure_quant_method(layer.quant_method)
    backend.initialize(layer)
    return backend, layer, state, dp_group if dpa else tp_group


@pytest.mark.parametrize(
    ("config_overrides", "backend_overrides"),
    [
        ({"moe_backend": "mega"}, {}),
        ({}, {"online_quant": True}),
        (
            {},
            {
                "layer_quant_config": SimpleNamespace(
                    quant_dtype=torch.bfloat16,
                    quant_type=QuantType.per_1x32,
                )
            },
        ),
        ({"torch_dtype": torch.float16}, {}),
        ({}, {"activation": ActivationType.Swiglu}),
        ({}, {"apply_router_weight_on_input": True}),
        ({"enable_tbo": True}, {}),
        ({"prefill_context_parallel_size": 2}, {}),
        ({}, {"parallel_config": _parallel(dp_size=2)}),
        ({}, {"parallel_config": SimpleNamespace(dp_size=1, use_ep=True, tp_size=8)}),
        ({}, {"parallel_config": SimpleNamespace(dp_size=1, use_ep=False, tp_size=1)}),
    ],
)
def test_support_predicate_rejects_unsupported_configuration(
    comm_fused_env, config_overrides, backend_overrides
):
    modules, state, _, _ = comm_fused_env
    for name, value in config_overrides.items():
        setattr(state.config, name, value)

    assert _create_backend(modules, state, **backend_overrides) is None
    assert not state.shape_keys


@pytest.mark.parametrize(
    ("name", "value"),
    [("AITER_DISABLE_COMM_FUSED_MOE", "1"), ("ATOM_MOE_GU_ITLV", "0")],
)
def test_support_predicate_honors_disable_flags(
    monkeypatch, comm_fused_env, name, value
):
    modules, state, _, _ = comm_fused_env
    monkeypatch.setenv(name, value)

    assert _create_backend(modules, state) is None
    assert not state.shape_keys


def test_support_predicate_uses_runtime_shape_and_fails_closed(comm_fused_env):
    modules, state, _, _ = comm_fused_env

    backend = _create_backend(modules, state)
    assert backend is not None
    assert not backend.supports(32)
    assert state.shape_keys == [_ShapeKey("gfx950", 7168, 384, 384, 6, 8)]

    state.missing_shape = True
    assert _create_backend(modules, state) is None


def test_runner_wiring_pins_flydsl_weight_layout(comm_fused_env):
    backend, layer, state, tp_group = _initialize_backend(comm_fused_env)

    assert not layer.quant_method.use_triton
    assert not layer.quant_method.use_triton_decode
    assert state.runner_calls == [
        {
            "tp_group": tp_group,
            "model_dim": 7168,
            "inter_dim": 384,
            "experts": 384,
            "topk": 6,
            "comm": "ar",
            "add_shared": True,
        }
    ]
    assert backend.supports(32)
    assert not backend.supports(31)


def test_dpa_runner_uses_reduce_scatter_group(comm_fused_env):
    backend, _, state, dp_group = _initialize_backend(comm_fused_env, dpa=True)
    assert backend is not None
    assert state.shape_keys == [
        _ShapeKey(
            "gfx950",
            7168,
            384,
            384,
            6,
            8,
            add_shared=True,
            comm="rs",
        )
    ]

    assert state.runner_calls == [
        {
            "tp_group": dp_group,
            "model_dim": 7168,
            "inter_dim": 384,
            "experts": 384,
            "topk": 6,
            "comm": "rs",
            "add_shared": True,
        }
    ]


def test_forward_impl_bridges_fused_moe_state_to_runtime(comm_fused_env):
    modules, state, _, _ = comm_fused_env
    backend = _create_backend(modules, state)
    hidden_states, router_logits, topk_weights, topk_ids = (object() for _ in range(4))
    shared_partial, stage2_stream, output = (object() for _ in range(3))
    before_stage2_for_rows = MagicMock(name="before_stage2_for_rows")
    layer, method = _runtime_layer(
        modules, topk_weights, topk_ids, apply_router_weight=True
    )
    runtime = SimpleNamespace(run=MagicMock(return_value=output))
    backend.runtime = runtime

    result = backend.forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_partial,
        before_stage2_for_rows=before_stage2_for_rows,
        stage2_stream=stage2_stream,
    )

    assert result is output
    method.select_experts.assert_called_once_with(layer, hidden_states, router_logits)
    runtime_args = runtime.run.call_args.kwargs
    expected = {
        "hidden_states": hidden_states,
        "topk_weight": topk_weights,
        "topk_ids": topk_ids,
        "quant_type": method.quant_type,
        "shared_partial": shared_partial,
        "before_stage2_for_rows": before_stage2_for_rows,
        "stage2_stream": stage2_stream,
    }
    for name, value in expected.items():
        assert runtime_args[name] is value
    for name, attribute in _RUNTIME_PASSTHROUGH.items():
        assert runtime_args[name] is getattr(layer, attribute)
    assert runtime_args["gate_mode"] == GateMode.INTERLEAVE.value
    assert runtime_args["reuse_is_synchronized"] is False
    # Pad rows go to expert -1 as whole rows: no missing-route clear.
    assert "routes_may_be_missing" not in runtime_args


def test_dpa_forward_fuses_local_shared_into_owned_rs_rows(comm_fused_env):
    modules, _, backend = _create_dpa_backend(comm_fused_env)

    local_hidden = torch.randn(2, 8)
    local_router = torch.randn(2, 16)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    gathered_hidden = torch.randn(16, 8)
    gathered_weights = torch.randn(16, 2)
    gathered_ids = torch.zeros(16, 2, dtype=torch.int32)
    routed = torch.randn(2, 8)
    reduce_scatter_sizes = [3, 1, 1, 2, 2, 3, 2, 2]
    local_shared = torch.randn(2, 8)

    layer, method = _runtime_layer(modules, topk_weights, topk_ids)

    backend._gather_dpa_inputs = MagicMock(
        return_value=(
            (gathered_hidden, gathered_weights, gathered_ids),
            2,
            reduce_scatter_sizes,
        )
    )
    backend.runtime = SimpleNamespace(run=MagicMock(return_value=routed))

    output = backend.forward_impl(
        layer,
        local_hidden,
        local_router,
        local_shared,
    )

    torch.testing.assert_close(output, routed)
    # Routed on the local rows, then the routed rows gathered.
    method.select_experts.assert_called_once_with(layer, local_hidden, local_router)
    backend._gather_dpa_inputs.assert_called_once_with(
        (local_hidden, topk_weights, topk_ids)
    )
    runtime_args = backend.runtime.run.call_args.kwargs
    assert runtime_args["hidden_states"] is gathered_hidden
    assert runtime_args["topk_weight"] is gathered_weights
    assert runtime_args["topk_ids"] is gathered_ids
    assert runtime_args["shared_partial"] is local_shared
    assert runtime_args["before_stage2_for_rows"] is None
    assert runtime_args["reduce_scatter_sizes"] is reduce_scatter_sizes
    assert runtime_args["reuse_is_synchronized"] is True


def test_dpa_gather_inserts_bucket_padding_inside_each_rank_shard(
    monkeypatch, comm_fused_env
):
    modules, _, backend = _create_dpa_backend(comm_fused_env)

    local_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    local_weights = torch.tensor([[5.0], [6.0]])
    local_ids = torch.tensor([[7], [8]], dtype=torch.int32)

    class FakeGroup:
        world_size = 8

        def __init__(self):
            self.locals = []
            self.sizes = []
            self.device_communicator = SimpleNamespace(all_gatherv=self._all_gatherv)

        def _all_gatherv(self, tensors, *, dim, sizes):
            assert dim == 0
            self.sizes.append(sizes)
            self.locals.extend(t.clone() for t in tensors)
            return [torch.cat([t] * self.world_size, dim=0) for t in tensors]

    group = FakeGroup()
    monkeypatch.setattr(modules.comm, "get_dp_group", lambda: group)
    _set_forward_context(
        monkeypatch,
        modules,
        metadata=SimpleNamespace(max_tokens_across_dp=3),
        tokens=2,
    )
    backend.runtime = SimpleNamespace(
        bucket_for=MagicMock(return_value=32),
        supports_ragged_m=MagicMock(return_value=False),
    )
    (gathered_hidden, gathered_weights, gathered_ids), local_tokens, sizes = (
        backend._gather_dpa_inputs((local_hidden, local_weights, local_ids))
    )

    assert local_tokens == 2
    assert sizes is None
    # One grouped collective, every shard padded to the bucket's 4 rows.
    assert group.sizes == [[4] * 8]
    assert [tensor.shape for tensor in group.locals] == [(4, 2), (4, 1), (4, 1)]
    hidden_sent, weights_sent, ids_sent = group.locals
    torch.testing.assert_close(hidden_sent[:2], local_hidden)
    torch.testing.assert_close(hidden_sent[2:], torch.zeros(2, 2))
    torch.testing.assert_close(weights_sent[:2], local_weights)
    torch.testing.assert_close(weights_sent[2:], torch.zeros(2, 1))
    # The pad rows are routed nowhere (ids travel as int32).
    assert torch.equal(ids_sent[:2], local_ids)
    assert (ids_sent[2:] == -1).all()
    assert gathered_hidden.shape == (32, 2)
    assert gathered_weights.shape == (32, 1)
    assert gathered_ids.dtype == torch.int32 and gathered_ids.shape == (32, 1)
    backend.runtime.bucket_for.assert_called_once_with(24)


def test_dpa_window_gathers_compact_inputs_with_one_grouped_all_gatherv(
    monkeypatch, comm_fused_env
):
    modules, _, backend = _create_dpa_backend(comm_fused_env)

    sizes = [3, 5, 1, 7, 2, 6, 4, 8]
    local_hidden = torch.randn(sizes[1], 8)
    local_weights = torch.randn(sizes[1], 2, dtype=torch.float32)
    gathered_hidden = torch.randn(sum(sizes), 8)
    gathered_weights = torch.randn(sum(sizes), 2, dtype=torch.float32)
    communicator = SimpleNamespace(
        all_gatherv=MagicMock(return_value=[gathered_hidden, gathered_weights])
    )
    group = SimpleNamespace(
        world_size=8,
        rank_in_group=1,
        device_communicator=communicator,
        all_gather=MagicMock(),
    )
    backend.runtime = SimpleNamespace(
        bucket_for=MagicMock(return_value=64),
        supports_ragged_m=MagicMock(return_value=True),
    )
    monkeypatch.setattr(modules.comm, "get_dp_group", lambda: group)
    _set_forward_context(
        monkeypatch,
        modules,
        metadata=SimpleNamespace(
            max_tokens_across_dp=max(sizes),
            get_sizes_across_dp=lambda: sizes,
        ),
        tokens=sizes[1],
    )
    (actual_hidden, actual_weights), local_tokens, actual_sizes = (
        backend._gather_dpa_inputs((local_hidden, local_weights))
    )

    assert torch.equal(actual_hidden, gathered_hidden)
    assert torch.equal(actual_weights, gathered_weights)
    assert local_tokens == sizes[1]
    assert actual_sizes == sizes
    backend.runtime.bucket_for.assert_called_once_with(64)
    backend.runtime.supports_ragged_m.assert_called_once_with(64)
    communicator.all_gatherv.assert_called_once()
    (sent,), kwargs = communicator.all_gatherv.call_args
    assert kwargs == {"dim": 0, "sizes": sizes}
    assert [torch.equal(s, t) for s, t in zip(sent, (local_hidden, local_weights))] == [
        True,
        True,
    ]
    group.all_gather.assert_not_called()


@pytest.mark.parametrize("unified", [False, True])
def test_dpa_support_requires_rank_uniform_tokens(monkeypatch, comm_fused_env, unified):
    modules, _, backend = _create_dpa_backend(comm_fused_env)
    backend.runtime = SimpleNamespace(
        supports=MagicMock(return_value=True),
        bucket_for=MagicMock(return_value=32),
    )
    backend._logged_buckets.add(32)
    _set_forward_context(monkeypatch, modules, metadata=None, tokens=4, unified=unified)

    assert backend.supports(4) is unified
    if unified:
        backend.runtime.bucket_for.assert_called_once_with(32)
        backend.runtime.supports.assert_called_once_with(32)
    else:
        backend.runtime.bucket_for.assert_not_called()
        backend.runtime.supports.assert_not_called()


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
            before_stage2_for_rows=None,
            before_shared_add=None,
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
    moe = _new_v4_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    shared_partial = torch.randn_like(hidden_states)
    fused_output = torch.randn_like(hidden_states)
    routed_output = torch.randn_like(hidden_states)
    combined_output = torch.randn_like(hidden_states)

    moe.gate = MagicMock(return_value=router_logits)
    moe.shared_experts = MagicMock(return_value=shared_partial)
    moe.experts = MagicMock()
    moe.experts._comm_fused_moe = SimpleNamespace()
    moe.experts.forward_maybe_comm_fused.return_value = (
        fused_output if supported else routed_output,
        supported,
    )
    moe.combine_outputs = MagicMock(return_value=combined_output)
    moe.experts.combine_accepts_shared.return_value = False

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
        before_stage2_for_rows=None,
        before_shared_add=None,
        stage2_stream=None,
        shared_ready=None,
    )


def test_dsv41_override_matches_stage2_callback_contract():
    source = Path(__file__).parents[2] / "atom/models/deepseek_v41/moe.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    method = next(
        node
        for node in ast.walk(tree)
        if getattr(node, "name", None) == "routed_expert_forward"
    )
    callbacks = [
        "before_stage2_for_rows",
        "before_shared_add",
        "stage2_stream",
        "shared_ready",
    ]
    assert [argument.arg for argument in method.args.args[-4:]] == callbacks
    call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "forward_maybe_comm_fused"
    )
    assert [keyword.arg for keyword in call.keywords] == callbacks


@pytest.mark.parametrize("join_shared", [False, True])
def test_dsv4_comm_fused_dual_stream_requires_shared_join(
    monkeypatch, atom_modules, join_shared
):
    moe = _new_v4_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    routed = torch.randn_like(hidden_states)
    shared = MagicMock(name="shared")
    routed_stream, alt_stream = _mock_streams(monkeypatch, atom_modules)

    def run_fused(
        _x, *, before_stage2_for_rows, before_shared_add, stage2_stream, **_kwargs
    ):
        assert stage2_stream is routed_stream
        assert before_stage2_for_rows(shared.shape[0]) is shared
        if join_shared:
            before_shared_add()
        return routed, True

    moe.alt_stream = alt_stream
    moe.shared_experts = MagicMock()
    moe.shared_experts.forward.return_value = shared
    moe.routed_expert_forward = MagicMock(side_effect=run_fused)
    moe.experts = SimpleNamespace(combine_accepts_shared=lambda rows: False)
    moe.combine_outputs = MagicMock()
    if join_shared:
        result = atom_modules.deepseek_v4.MoE.dual_stream_moe_forward(
            moe, hidden_states
        )
        assert result is routed
        routed_stream.wait_stream.assert_called_once_with(alt_stream)
        moe.combine_outputs.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="completed without joining"):
            atom_modules.deepseek_v4.MoE.dual_stream_moe_forward(moe, hidden_states)
        routed_stream.wait_stream.assert_not_called()
    assert alt_stream.wait_stream.call_args_list[0].args == (routed_stream,)


def test_dsv4_dual_stream_hands_the_shared_output_to_the_combine(
    monkeypatch, atom_modules
):
    """Where the routed combine adds the shared output, the dual-stream path
    hands it down through `routed_expert_forward` with the event recorded on
    the side stream, and neither joins the streams nor adds it again."""
    moe = _new_v4_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    routed = torch.randn_like(hidden_states)
    shared = MagicMock(name="shared")
    routed_stream, alt_stream = _mock_streams(monkeypatch, atom_modules)
    ready = MagicMock(name="shared_ready")
    monkeypatch.setattr(atom_modules.deepseek_v4.torch.cuda, "Event", lambda: ready)
    moe.alt_stream = alt_stream
    moe.experts = SimpleNamespace(combine_accepts_shared=lambda rows: rows == 4)
    moe.shared_experts = MagicMock()
    moe.shared_experts.forward.return_value = shared
    moe.routed_expert_forward = MagicMock(return_value=(routed, False))
    combined = torch.randn_like(hidden_states)
    moe.combine_outputs = MagicMock(return_value=combined)

    result = atom_modules.deepseek_v4.MoE.dual_stream_moe_forward(moe, hidden_states)

    assert result is combined
    moe.routed_expert_forward.assert_called_once_with(
        hidden_states, shared_partial=shared, shared_ready=ready
    )
    ready.record.assert_called_once_with(alt_stream)
    shared.record_stream.assert_called_once_with(routed_stream)
    moe.combine_outputs.assert_called_once_with(
        routed, None, prefix="model.layers.3.mlp.combine_outputs"
    )
    routed_stream.wait_stream.assert_not_called()


def test_dsv4_single_stream_hands_the_shared_output_to_the_combine(
    monkeypatch, atom_modules
):
    """Where the routed combine adds the shared output, the single-stream path
    hands it down with an event recorded on its own stream, and does not add
    it again."""
    moe = _new_v4_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    routed = torch.randn_like(hidden_states)
    shared = torch.randn_like(hidden_states)
    ready = MagicMock(name="shared_ready")
    monkeypatch.setattr(atom_modules.deepseek_v4.torch.cuda, "Event", lambda: ready)
    moe.shared_experts = MagicMock(return_value=shared)
    moe.experts = SimpleNamespace(combine_accepts_shared=lambda rows: rows == 4)
    moe.routed_expert_forward = MagicMock(return_value=(routed, False))
    combined = torch.randn_like(hidden_states)
    moe.combine_outputs = MagicMock(return_value=combined)

    result = atom_modules.deepseek_v4.MoE.single_stream_moe_forward(moe, hidden_states)

    assert result is combined
    moe.routed_expert_forward.assert_called_once_with(
        hidden_states, shared_partial=shared, shared_ready=ready
    )
    ready.record.assert_called_once_with()
    moe.combine_outputs.assert_called_once_with(
        routed, None, prefix="model.layers.3.mlp.combine_outputs"
    )


def test_dsv4_single_stream_refuses_a_comm_fused_shared_in_combine(
    monkeypatch, atom_modules
):
    moe = _new_v4_moe(atom_modules)
    hidden_states = torch.randn(4, 8)
    monkeypatch.setattr(
        atom_modules.deepseek_v4.torch.cuda, "Event", lambda: MagicMock()
    )
    moe.shared_experts = MagicMock(return_value=torch.randn_like(hidden_states))
    moe.experts = SimpleNamespace(combine_accepts_shared=lambda rows: True)
    moe.routed_expert_forward = MagicMock(return_value=(hidden_states, True))
    with pytest.raises(RuntimeError, match="shared-in-combine"):
        atom_modules.deepseek_v4.MoE.single_stream_moe_forward(moe, hidden_states)


def _fused_moe_for_shared_add(atom_modules, *, accepts):
    layer = _new_fused_moe(atom_modules)
    layer._comm_fused_moe = None
    layer.balance_router_logits = None
    layer.combine_accepts_shared = lambda rows: accepts
    layer.forward_impl = MagicMock(name="forward_impl")
    return layer


def test_fused_moe_hands_the_shared_output_to_the_routed_combine(
    monkeypatch, atom_modules
):
    layer = _fused_moe_for_shared_add(atom_modules, accepts=True)
    ordinary_forward = MagicMock()
    monkeypatch.setattr(type(layer), "forward", ordinary_forward)
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    shared = torch.randn_like(hidden_states)
    ready = MagicMock(name="shared_ready")

    output, is_complete = layer.forward_maybe_comm_fused(
        hidden_states, router_logits, shared, shared_ready=ready
    )

    assert output is layer.forward_impl.return_value and is_complete is False
    layer.forward_impl.assert_called_once_with(
        hidden_states, router_logits, shared_add=shared, shared_ready=ready
    )
    ordinary_forward.assert_not_called()

    # Without the event the shared output stays the caller's to add.
    output, is_complete = layer.forward_maybe_comm_fused(
        hidden_states, router_logits, shared
    )
    assert output is ordinary_forward.return_value and is_complete is False
    layer.forward_impl.assert_called_once()


def test_fused_moe_refuses_a_shared_output_its_combine_cannot_add(atom_modules):
    layer = _fused_moe_for_shared_add(atom_modules, accepts=False)
    hidden_states = torch.randn(4, 8)
    with pytest.raises(RuntimeError, match="combine_accepts_shared"):
        layer.forward_maybe_comm_fused(
            hidden_states,
            torch.randn(4, 16),
            torch.randn_like(hidden_states),
            shared_ready=MagicMock(),
        )
    layer.forward_impl.assert_not_called()


def test_fused_moe_combine_accepts_shared(monkeypatch, atom_modules):
    class MoriV2ModularKernel:
        pass

    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.fused_moe.mori_v2_prepare_finalize",
        SimpleNamespace(MoriV2ModularKernel=MoriV2ModularKernel),
    )
    layer = _new_fused_moe(atom_modules)
    layer._comm_fused_moe = None
    layer.sp_mega_moe = None
    layer.reduce_results = False
    layer.moe_parallel_config = SimpleNamespace(pcp_merge=False, dp_logical_ratio=1)
    method = object.__new__(atom_modules.moe.Mxfp4MoEMethod)
    method.fused_experts = MoriV2ModularKernel()
    layer.quant_method = method
    assert layer.combine_accepts_shared(4)

    method.fused_experts = object()  # not the mori v2 kernel
    assert not layer.combine_accepts_shared(4)
    method.fused_experts = MoriV2ModularKernel()
    layer.quant_method = SimpleNamespace(fused_experts=method.fused_experts)
    assert not layer.combine_accepts_shared(4)  # not the MXFP4 method
    layer.quant_method = method

    refusals = [
        ("_comm_fused_moe", SimpleNamespace(supports=lambda rows: rows <= 8)),
        ("sp_mega_moe", SimpleNamespace(runs=lambda rows: True)),
        ("reduce_results", True),
        (
            "moe_parallel_config",
            SimpleNamespace(pcp_merge=True, dp_logical_ratio=1),
        ),
        (
            "moe_parallel_config",
            SimpleNamespace(pcp_merge=False, dp_logical_ratio=2),
        ),
    ]
    for name, value in refusals:
        kept = getattr(layer, name)
        setattr(layer, name, value)
        assert not layer.combine_accepts_shared(4), name
        setattr(layer, name, kept)
    assert layer.combine_accepts_shared(4)


def _recording_fused_moe(monkeypatch, atom_modules, *, dp_world):
    """A FusedMoE whose real forward_impl / forward_impl_graph / _run_experts
    run; only the quant method (which records what reaches ``apply``) and the
    process groups are stand-ins."""
    moe = atom_modules.moe
    layer = _new_fused_moe(atom_modules)
    routed = torch.randn(4, 8)
    method = SimpleNamespace(
        routed_rows_wire=None,
        select_experts=MagicMock(
            return_value=(torch.rand(4, 2), torch.zeros(4, 2, dtype=torch.int32))
        ),
        place_experts=MagicMock(side_effect=lambda layer, w, ids, gathered: (w, ids)),
        apply=MagicMock(return_value=routed),
    )
    values = {
        "quant_method": method,
        "_comm_fused_moe": None,
        "sp_mega_moe": None,
        "balance_router_logits": None,
        "reduce_results": False,
        # EP all2all: forward_impl_graph does no DP gather/scatter of its own.
        "moe_parallel_config": SimpleNamespace(
            dp_size=dp_world,
            pcp_merge=False,
            dp_logical_ratio=1,
            use_all2all_kernels=True,
        ),
        "global_num_experts": 16,
        "expert_map": None,
        "activation": None,
        "apply_router_weight_on_input": False,
        "prefix": "experts",
    }
    for name, value in values.items():
        setattr(layer, name, value)
    layer.combine_accepts_shared = lambda rows: True
    context = SimpleNamespace(
        is_dummy_run=False, pcp_split=False, scheduled_tokens_gpu=None
    )
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(moe, "pass_scheduled_tokens_gpu", lambda rows: None)
    monkeypatch.setattr(
        moe, "get_dp_group", lambda: SimpleNamespace(world_size=dp_world)
    )
    graph = MagicMock(wraps=layer.forward_impl_graph)
    monkeypatch.setattr(layer, "forward_impl_graph", graph)
    return layer, method, routed, graph


@pytest.mark.parametrize("dp_world", [1, 2], ids=["forward_impl", "graph"])
def test_shared_output_reaches_the_quant_method_through_forward_impl(
    monkeypatch, atom_modules, dp_world
):
    """Every hop from forward_maybe_comm_fused to quant_method.apply carries
    the shared output and its event: forward_impl -> _run_experts on one DP
    rank, forward_impl -> forward_impl_graph -> _run_experts on several."""
    layer, method, routed, graph = _recording_fused_moe(
        monkeypatch, atom_modules, dp_world=dp_world
    )
    hidden_states = torch.randn(4, 8)
    router_logits = torch.randn(4, 16)
    shared = torch.randn_like(hidden_states)
    ready = MagicMock(name="shared_ready")

    output, is_complete = layer.forward_maybe_comm_fused(
        hidden_states, router_logits, shared, shared_ready=ready
    )

    assert output is routed and is_complete is False
    assert graph.call_count == (1 if dp_world > 1 else 0)
    method.apply.assert_called_once()
    applied = method.apply.call_args.kwargs
    assert applied["shared_add"] is shared
    assert applied["shared_ready"] is ready
    assert applied["x"] is hidden_states


def test_routed_forward_without_a_shared_output_passes_none_down(
    monkeypatch, atom_modules
):
    """The ordinary forward_impl call does not hand the quant method a shared
    output, so the methods that do not take one are called as before."""
    layer, method, _, _ = _recording_fused_moe(monkeypatch, atom_modules, dp_world=1)

    layer.forward_impl(torch.randn(4, 8), torch.randn(4, 16))

    applied = method.apply.call_args.kwargs
    assert "shared_add" not in applied and "shared_ready" not in applied


@pytest.mark.parametrize(
    ("use_triton", "fused_experts"),
    [(False, None), (True, object())],
    ids=["no-modular-kernel", "triton-experts"],
)
def test_mxfp4_apply_refuses_a_shared_output_without_a_combine(
    atom_modules, use_triton, fused_experts
):
    method = object.__new__(atom_modules.moe.Mxfp4MoEMethod)
    method.use_triton = use_triton
    method.use_triton_decode = False
    method.fused_experts = fused_experts
    hidden_states = torch.randn(4, 8)
    with pytest.raises(RuntimeError, match="without a combine"):
        method.apply(
            layer=SimpleNamespace(),
            x=hidden_states,
            topk_weights=torch.rand(4, 2),
            topk_ids=torch.zeros(4, 2, dtype=torch.int32),
            shared_add=torch.randn_like(hidden_states),
            shared_ready=MagicMock(),
        )


def _mxfp4_method_with_recording_experts(atom_modules):
    """An ``Mxfp4MoEMethod`` on the modular-kernel path (no Triton experts),
    whose ``fused_experts`` records what ``apply`` hands it."""
    method = object.__new__(atom_modules.moe.Mxfp4MoEMethod)
    values = {
        "use_triton": False,
        "use_triton_decode": False,
        "use_triton_ep": False,
        "use_flydsl_herd": False,
        "is_guinterleave": False,
        "fp8_activations": False,
        "quant_type": object(),
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "fused_experts": MagicMock(return_value=torch.randn(4, 8)),
    }
    for name, value in values.items():
        setattr(method, name, value)
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 16, 8),
        w2_weight=torch.empty(2, 8, 8),
        w13_weight_scale=None,
        w2_weight_scale=None,
        w13_bias=None,
        w2_bias=None,
        expert_mask=None,
    )
    return method, layer


@pytest.mark.parametrize("with_shared", [True, False], ids=["shared", "plain"])
def test_mxfp4_apply_hands_the_shared_output_to_fused_experts(
    atom_modules, with_shared
):
    """The last hop: ``Mxfp4MoEMethod.apply`` passes the shared output and
    its event on to the modular kernel (``fused_experts``), whose combine adds
    it; without one, ``fused_experts`` is called without them."""
    method, layer = _mxfp4_method_with_recording_experts(atom_modules)
    hidden_states = torch.randn(4, 8)
    shared = torch.randn_like(hidden_states) if with_shared else None
    ready = MagicMock(name="shared_ready") if with_shared else None

    output = method.apply(
        layer=layer,
        x=hidden_states,
        topk_weights=torch.rand(4, 2),
        topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        shared_add=shared,
        shared_ready=ready,
    )

    method.fused_experts.assert_called_once()
    assert output is method.fused_experts.return_value
    passed = method.fused_experts.call_args.kwargs
    assert passed["hidden_states"] is hidden_states
    if with_shared:
        assert passed["shared_add"] is shared
        assert passed["shared_ready"] is ready
    else:
        assert "shared_add" not in passed and "shared_ready" not in passed


@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, 1536), ("1024", 1024), ("2048", 2048), ("0", 0)],
)
def test_dsv4_dual_stream_decode_token_threshold(
    monkeypatch, atom_modules, env, expected
):
    """The V4 MoE's decode bound: 1536 with EP unless the variable is set,
    which wins; without EP the variable's value (1024 by default)."""
    if env is None:
        monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", env)
    v4 = atom_modules.deepseek_v4
    assert v4.v4_dual_stream_decode_token_threshold(True) == expected
    generic = 1024 if env is None else int(env)
    assert v4.v4_dual_stream_decode_token_threshold(False) == generic


@pytest.mark.parametrize("phase", ["decode", "prefill", None])
@pytest.mark.parametrize("rows", [1, 1024, 1025, 1152, 1536, 1537, 16384])
def test_dual_stream_gate_widens_only_decode(monkeypatch, atom_modules, rows, phase):
    """A V4 module forks 1025-1536-row decode steps; prefill steps of that
    size, and calls with no step context, keep the generic bound."""
    dispatch = sys.modules["atom.model_ops.module_dispatch_ops"]
    monkeypatch.setattr(dispatch, "plugin_graph_is_piecewise", lambda: False)
    monkeypatch.setattr(
        sys.modules["atom.utils.tbo.ubatching"], "tbo_active", lambda: False
    )
    context = None if phase is None else SimpleNamespace(is_prefill=phase == "prefill")
    monkeypatch.setattr(
        dispatch, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    v4 = SimpleNamespace(
        _use_dual_stream=True,
        _dual_stream_token_threshold=1024,
        _dual_stream_decode_token_threshold=1536,
    )
    other = SimpleNamespace(
        _use_dual_stream=True,
        _dual_stream_token_threshold=1024,
        _dual_stream_decode_token_threshold=1024,
    )
    bound = 1536 if phase == "decode" else 1024
    assert dispatch._dual_stream_is_active(v4, rows) == (rows <= bound)
    assert dispatch._dual_stream_is_active(other, rows) == (rows <= 1024)


def _moe_parallel(monkeypatch, atom_modules, *, tp, dp, ep, dp_logical=0):
    """A real `FusedMoEParallelConfig.make` for this deployment shape (rank
    0, DP attention on, no PCP)."""
    moe = atom_modules.moe
    group = SimpleNamespace(rank_in_group=0)
    monkeypatch.setattr(moe, "get_tp_group", lambda: group)
    monkeypatch.setattr(moe, "get_dp_group", lambda: group)
    state = sys.modules["aiter.dist.parallel_state"]
    monkeypatch.setattr(
        state, "get_prefill_context_model_parallel_world_size", lambda: 1, raising=False
    )
    monkeypatch.setattr(
        state, "get_prefill_context_model_parallel_rank", lambda: 0, raising=False
    )
    monkeypatch.setattr(
        moe,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size_local=dp)
        ),
    )
    config = SimpleNamespace(
        enable_dp_attention=dp > 1 or dp_logical > 1,
        moe_ep_flatten_tp_across_dp=False,
        enable_expert_parallel=ep,
        dp_logical_size=dp_logical,
    )
    return moe.FusedMoEParallelConfig.make(tp, dp, config)


@pytest.mark.parametrize(
    ("tp", "dp", "ep", "dp_logical", "exchanges"),
    [
        (1, 4, True, 0, True),  # DEP4
        (2, 4, True, 0, True),  # DP4 x TP2 with EP
        (8, 1, True, 0, False),  # -tp 8 --enable-expert-parallel: no exchange
        (1, 4, False, 0, False),  # DP attention without EP
        (8, 1, False, 0, False),  # plain TP
        (1, 2, True, 4, False),  # simulated DP: finer expert map, no real peers
    ],
)
def test_moe_parallel_config_exchanges_tokens(
    monkeypatch, atom_modules, tp, dp, ep, dp_logical, exchanges
):
    parallel = _moe_parallel(
        monkeypatch, atom_modules, tp=tp, dp=dp, ep=ep, dp_logical=dp_logical
    )
    assert parallel.exchanges_tokens is exchanges
    if not exchanges:
        assert parallel.selected_all2all_backend is None


def _v4_moe_built(
    monkeypatch, atom_modules, *, parallel, shared_stream=lambda experts: True
):
    """A V4 `MoE` through its real `__init__`, GEMM/expert modules stubbed;
    ``parallel`` is the experts' real `FusedMoEParallelConfig` and
    ``shared_stream(experts)`` their `allows_concurrent_shared_stream`."""
    v4 = atom_modules.deepseek_v4
    forward_context = {}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=forward_context)
    )
    monkeypatch.setattr(v4, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(v4, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v4, "ReplicatedLinear", lambda *a, **k: torch.nn.Module())

    class Experts(torch.nn.Module):
        @property
        def allows_concurrent_shared_stream(self):
            return shared_stream(self)

    def fused_moe(**kwargs):
        experts = Experts()
        experts.num_fused_shared_experts = 0
        experts._comm_fused_moe = None
        experts.moe_parallel_config = parallel
        return experts

    monkeypatch.setattr(v4, "FusedMoE", fused_moe)
    monkeypatch.setattr(v4, "Expert", lambda *a, **k: torch.nn.Module())
    args = SimpleNamespace(
        dim=8,
        n_routed_experts=4,
        n_activated_experts=2,
        n_hash_layers=0,
        route_scale=1.0,
        swiglu_limit=7.0,
        quant_config=None,
        n_shared_experts=1,
        moe_inter_dim=16,
        score_func="sqrtsoftplus",
        moe_fp8_activations=False,
        vocab_size=32,
    )
    moe = v4.MoE(3, args, prefix="model.layers.3.ffn", alt_stream=object())
    return moe, forward_context


_DEP4 = {"tp": 1, "dp": 4, "ep": True}


@pytest.mark.parametrize(
    ("shape", "phase", "forks_at_1536"),
    [
        (_DEP4, "decode", True),
        (_DEP4, "prefill", False),
        ({"tp": 8, "dp": 1, "ep": True}, "decode", False),  # TP-only EP
        ({"tp": 8, "dp": 1, "ep": False}, "decode", False),  # plain TP
        (_DEP4, None, False),
    ],
)
def test_dsv4_widens_only_expert_parallel_decode(
    monkeypatch, atom_modules, shape, phase, forks_at_1536
):
    """A real V4 module forks 1025-1536-row steps only on decode steps of a
    deployment whose experts exchange tokens; prefill, TP-only EP, plain TP
    and calls without a step context keep the generic bound."""
    monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    parallel = _moe_parallel(monkeypatch, atom_modules, **shape)
    moe, _ = _v4_moe_built(monkeypatch, atom_modules, parallel=parallel)
    dispatch = sys.modules["atom.model_ops.module_dispatch_ops"]
    monkeypatch.setattr(dispatch, "plugin_graph_is_piecewise", lambda: False)
    monkeypatch.setattr(
        sys.modules["atom.utils.tbo.ubatching"], "tbo_active", lambda: False
    )
    context = None if phase is None else SimpleNamespace(is_prefill=phase == "prefill")
    monkeypatch.setattr(
        dispatch, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    assert dispatch._dual_stream_is_active(moe, 1024)
    for rows in (1025, 1344, 1536):
        assert dispatch._dual_stream_is_active(moe, rows) == forks_at_1536
    assert not dispatch._dual_stream_is_active(moe, 1537)


@pytest.mark.parametrize(
    ("env", "expected", "generic"),
    [(None, 1536, 1024), ("2048", 2048, 2048), ("0", 0, 0)],
)
def test_dsv4_moe_init_sets_its_dual_stream_threshold(
    monkeypatch, atom_modules, env, expected, generic
):
    """`MoE.__init__` itself stores the V4 bounds the dispatch gate reads."""
    if env is None:
        monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", env)
    parallel = _moe_parallel(monkeypatch, atom_modules, **_DEP4)
    moe, forward_context = _v4_moe_built(monkeypatch, atom_modules, parallel=parallel)

    assert moe._dual_stream_decode_token_threshold == expected
    assert moe._dual_stream_token_threshold == generic
    # 0 turns the dual stream off: no fork and no registration for it.
    assert moe._use_dual_stream == (expected > 0)
    assert (forward_context.get("model.layers.3.ffn") is moe) == (expected > 0)


@pytest.mark.parametrize("refuses", [False, True])
def test_dsv4_moe_keeps_shared_experts_single_stream_when_its_experts_refuse(
    monkeypatch, atom_modules, refuses
):
    """Experts whose method refuses a concurrent shared-expert stream (a fused
    MegaMoEV2 route: its persistent kernels deadlock next to one) turn the V4
    dual stream off. V4 asks once it set the experts' swiglu_limit."""
    monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    parallel = _moe_parallel(monkeypatch, atom_modules, **_DEP4)
    asked = []

    def shared_stream(experts):
        asked.append(experts.swiglu_limit)
        return not refuses

    moe, forward_context = _v4_moe_built(
        monkeypatch, atom_modules, parallel=parallel, shared_stream=shared_stream
    )

    assert asked == [7.0]
    assert moe._use_dual_stream is not refuses
    assert ("model.layers.3.ffn" in forward_context) is not refuses


@pytest.mark.parametrize("refuses", [False, True])
def test_dsv2_moe_keeps_shared_experts_single_stream_when_its_experts_refuse(
    monkeypatch, atom_modules, refuses
):
    """The V2 MoE (DeepSeek-V3/R1, Kimi-K2.5) takes the same gate as V4."""
    monkeypatch.delenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", raising=False)
    v2 = atom_modules.deepseek_v2
    forward_context = {}
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
    )
    monkeypatch.setattr(v2, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(v2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v2, "ReplicatedLinear", lambda *a, **k: torch.nn.Module())
    monkeypatch.setattr(v2, "DeepseekV2MLP", lambda *a, **k: torch.nn.Module())
    monkeypatch.setattr(
        v2, "is_rocm_aiter_fusion_shared_expert_enabled", lambda **_: False
    )

    def fused_moe(**kwargs):
        experts = torch.nn.Module()
        experts.allows_concurrent_shared_stream = not refuses
        return experts

    monkeypatch.setattr(v2, "FusedMoE", fused_moe)
    hf_config = SimpleNamespace(
        routed_scaling_factor=1.0,
        n_shared_experts=1,
        hidden_act="silu",
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        topk_method="greedy",
        model_type="deepseek_v3",
    )
    moe = v2.DeepseekV2MoE(
        hf_config, prefix="model.layers.3.mlp", alt_stream=object(), layer_id=3
    )

    assert moe._use_dual_stream is not refuses
    assert ("model.layers.3.mlp" in forward_context) is not refuses


def _stub_fused_route_query(monkeypatch, fused):
    import atom.model_ops.fused_moe.flydsl_mega_experts as mega

    queried = []

    def mega_builds_fused_route(**layout):
        queried.append(layout)
        return fused

    monkeypatch.setattr(mega, "mega_builds_fused_route", mega_builds_fused_route)
    return queried


_V4_PRO_MOE = SimpleNamespace(ep_size=8, num_experts=384, experts_per_token=6)
_V4_PRO_LAYOUT = {
    "world_size": 8,
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "quant": "a8w4",
    "swiglu_limit": 10.0,
}


@pytest.mark.parametrize("fused", [False, True])
def test_mega_method_refuses_a_concurrent_shared_stream_on_a_fused_route(
    monkeypatch, atom_modules, fused
):
    """Only a layout AITER builds a fused MegaMoEV2 route for refuses; other
    MoE methods always allow the side stream. The experts ask their method."""
    moe_ops = atom_modules.moe
    queried = _stub_fused_route_query(monkeypatch, fused)
    method = object.__new__(moe_ops.MegaMxfp4MoEMethod)
    method.moe = _V4_PRO_MOE
    method.hidden_size = 7168
    method.intermediate_size = 3072
    experts = SimpleNamespace(
        quant_method=method,
        swiglu_limit=10.0,
        _online_quant_method_cls=lambda: None,
    )

    assert moe_ops.FusedMoE.allows_concurrent_shared_stream.fget(experts) is not fused
    assert queried == [_V4_PRO_LAYOUT]
    other = object.__new__(moe_ops.Mxfp4MoEMethod)
    assert other.allows_concurrent_shared_stream(experts)


@pytest.mark.parametrize("fused", [False, True])
def test_experts_quantizing_online_into_mega_answer_by_the_mega_method(
    monkeypatch, atom_modules, fused
):
    """NVFP4 / BF16 experts that `_online_quant` switches to the Mega method
    answer by it at model init, not by the source method they still hold."""
    moe_ops = atom_modules.moe
    queried = _stub_fused_route_query(monkeypatch, fused)
    source = object.__new__(moe_ops.Nvfp4MoEMethod)
    experts = SimpleNamespace(
        quant_method=source,
        swiglu_limit=10.0,
        moe_config=_V4_PRO_MOE,
        hidden_size=7168,
        intermediate_size_per_partition=3072,
        _online_quant_method_cls=lambda: moe_ops.MegaMxfp4MoEMethod,
    )

    assert moe_ops.FusedMoE.allows_concurrent_shared_stream.fget(experts) is not fused
    assert queried == [_V4_PRO_LAYOUT]


@pytest.mark.parametrize(
    ("online", "target_dtype", "skip", "backend", "expected"),
    [
        (False, "fp4x2", False, "mega", None),
        (True, "fp8", False, "mega", None),
        (True, "fp4x2", True, "mega", None),
        (True, "fp4x2", False, "aiter", "Mxfp4MoEMethod"),
        (True, "fp4x2", False, "mega", "MegaMxfp4MoEMethod"),
    ],
)
def test_online_quant_method_class_follows_the_online_switch(
    monkeypatch, atom_modules, online, target_dtype, skip, backend, expected
):
    moe_ops = atom_modules.moe
    target = SimpleNamespace(quant_dtype=getattr(moe_ops.dtypes, target_dtype))
    experts = SimpleNamespace(
        online_quant=online,
        layer_name="model.layers.3.ffn.experts",
        quant_config=SimpleNamespace(
            get_layer_quant_config=lambda name, use_online_quant: target
        ),
        layer_quant_config=SimpleNamespace(quant_type="source"),
        params_dtype="source",
    )
    monkeypatch.setattr(moe_ops, "should_skip_online_quant", lambda *_: skip)
    monkeypatch.setattr(
        moe_ops,
        "get_current_atom_config",
        lambda: SimpleNamespace(moe_backend=backend),
    )
    monkeypatch.setattr(moe_ops, "sp_is_enabled", lambda: False)

    cls = moe_ops.FusedMoE._online_quant_method_cls(experts)
    assert (cls and cls.__name__) == expected
