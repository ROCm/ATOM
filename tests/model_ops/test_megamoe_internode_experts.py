# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU regressions at the EP16 adapter's AITER/MoRI boundaries."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from conftest import atom_config_double

from atom.config import Config, is_internode_mega, use_custom_atom_config


@pytest.fixture
def internode(atom_modules, monkeypatch):
    module = importlib.import_module(
        "atom.model_ops.fused_moe.megamoe_internode_experts"
    )
    group_max_tokens = module._group_max_tokens
    built = []

    class Operator:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.device = torch.device("cpu")
            self.mtpr = kwargs["max_tok_per_rank"]
            self.quant = kwargs["quant"]
            self.close = Mock()
            self.bind_weights = Mock()
            self.calls = []
            built.append(self)

        def forward(self, x, weights, ids):
            self.calls.append((x.clone(), weights.clone(), ids.clone()))
            return x.clone()

    kernel = importlib.import_module("aiter.ops.flydsl.kernels.mega_moe_internode")
    monkeypatch.setattr(kernel, "MegaMoEInternode", Operator)
    monkeypatch.setattr(module, "_INSTANCES", {})
    monkeypatch.setattr(module, "_AGREED_BUDGETS", {})
    monkeypatch.setattr(module, "_rank_world", lambda: (0, 16))
    monkeypatch.setattr(module, "_group_max_tokens", lambda max_tokens: max_tokens)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(module, "_PREFILL_MTPR", 8)
    monkeypatch.setattr(module, "_DECODE_MTPRS", (2, 4))
    yield SimpleNamespace(
        module=module,
        built=built,
        moe=atom_modules.moe,
        group_max_tokens=group_max_tokens,
    )
    module.close_internode_instances()


# What the inherited `apply` passes for a SwiGLU layer.
SILU = {"swiglu_limit": 10.0}


def _weights(**overrides):
    layer = SimpleNamespace(
        _mi_w1=torch.empty(1),
        _mi_w1_scale=torch.empty(1),
        _mi_w2=torch.empty(1),
        _mi_w2_scale=torch.empty(1),
        w13_weight=None,
        w2_weight=None,
        w13_weight_scale=None,
        w2_weight_scale=None,
        w13_bias=None,
        w2_bias=None,
        expert_mask=None,
        swiglu_limit=10.0,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    vars(layer).update(overrides)
    return layer


def _instance(state, capacity=2, **overrides):
    settings = {
        "model_dim": 64,
        "inter_dim": 32,
        "experts": 256,
        "topk": 2,
        "layer": _weights(),
        "activation": "silu",
        "swiglu_limit": 10.0,
    }
    settings.update(overrides)
    return state.module._get_instance(capacity, **settings)


def _production_capacities(internode, monkeypatch):
    monkeypatch.setattr(internode.module, "_PREFILL_MTPR", 4096)
    monkeypatch.setattr(
        internode.module, "_DECODE_MTPRS", (32, 64, 128, 256, 512, 1024)
    )


@pytest.mark.parametrize(
    "budget,expected",
    [
        (1, (32,)),
        (32, (32,)),
        (33, (64, 32)),
        (128, (128, 32, 64)),
        (129, (256, 32, 64, 128)),
        (513, (1024, 32, 64, 128, 256, 512)),
        (1025, (2048, 32, 64, 128, 256, 512, 1024)),
        (4096, (4096, 32, 64, 128, 256, 512, 1024)),
        (8192, (4096, 32, 64, 128, 256, 512, 1024)),
    ],
)
def test_token_budget_limits_actual_operator_allocations(
    internode, monkeypatch, budget, expected
):
    _production_capacities(internode, monkeypatch)
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=budget
    )
    monkeypatch.setattr(internode.module, "_group_rows", lambda: budget)
    x = torch.ones(1, 64, dtype=torch.bfloat16)
    result = backend(
        hidden_states=x,
        topk_weights=torch.ones(1, 2),
        topk_ids=torch.zeros(1, 2, dtype=torch.int32),
        global_num_experts=256,
        moe_extra_args=SILU,
    )
    assert torch.equal(result, x)
    assert tuple(op.mtpr for op in internode.built) == expected
    assert all(
        op.mtpr <= 4096 and op.mtpr & (op.mtpr - 1) == 0 for op in internode.built
    )
    active = [op for op in internode.built if op.calls]
    assert len(active) == 1 and active[0].mtpr == expected[0]
    assert len(active[0].calls) == (budget + expected[0] - 1) // expected[0]


def _ep_budgets(internode, monkeypatch, budgets):
    import torch.distributed as dist

    ep = SimpleNamespace(world_size=len(budgets), cpu_group=object())
    calls = []

    def gather(out, local_budget, *, group):
        assert group is ep.cpu_group and len(out) == ep.world_size
        assert not torch.is_inference_mode_enabled()
        calls.append(local_budget)
        out[:] = budgets

    state = importlib.import_module("aiter.dist.parallel_state")
    monkeypatch.setattr(state, "get_ep_group", lambda: ep)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(
        internode.module, "_group_max_tokens", internode.group_max_tokens
    )
    return calls


def test_different_rank_budgets_produce_one_preallocation_sequence(
    internode, monkeypatch
):
    _production_capacities(internode, monkeypatch)
    calls = _ep_budgets(internode, monkeypatch, [129] + [128] * 15)
    monkeypatch.setattr(internode.module, "_group_rows", lambda: 1)
    for local_budget in (128, 129):
        with torch.inference_mode():
            backend = internode.module.MegaInternodeFusedExperts(
                _weights(), model_dim=64, inter_dim=32, max_tokens=local_budget
            )
            assert torch.is_inference_mode_enabled()
        assert backend._prefill_mtpr == 256
        assert backend._decode_mtprs == (32, 64, 128)
        backend(
            hidden_states=torch.zeros(1, 64, dtype=torch.bfloat16),
            topk_weights=torch.ones(1, 2),
            topk_ids=torch.zeros(1, 2, dtype=torch.int32),
            global_num_experts=256,
            moe_extra_args=SILU,
        )
    assert calls == [128, 129]
    assert [op.mtpr for op in internode.built] == [256, 32, 64, 128]


@pytest.mark.parametrize("invalid", [0, -1])
def test_invalid_rank_budget_is_rejected_before_operator_allocation(
    internode, monkeypatch, invalid
):
    calls = _ep_budgets(internode, monkeypatch, [128] * 15 + [invalid])
    with pytest.raises(ValueError, match="token budgets must be positive"):
        internode.module.MegaInternodeFusedExperts(
            _weights(), model_dim=64, inter_dim=32, max_tokens=128
        )
    assert calls == [128] and not internode.built


def test_budgeted_capacities_are_ready_for_padded_and_idle_graph_steps(
    internode, monkeypatch
):
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=4
    )

    def run(rows, valid):
        monkeypatch.setattr(internode.module, "_group_rows", lambda: rows)
        ids = torch.zeros(rows, 2, dtype=torch.int32)
        ids[valid:] = -1
        return backend(
            hidden_states=torch.ones(rows, 64, dtype=torch.bfloat16),
            topk_weights=torch.ones(rows, 2),
            topk_ids=ids,
            global_num_experts=256,
            moe_extra_args=SILU,
        )

    run(1, 1)  # Small startup forward must prebuild the wider reachable bucket.
    assert [op.mtpr for op in internode.built] == [4, 2]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    run(4, 3)  # Speculative/graph padding at the configured token limit.
    run(2, 0)  # An idle rank still participates at its graph's padded width.
    assert [op.mtpr for op in internode.built] == [4, 2]
    wider, smaller = internode.built
    assert (wider.calls[-1][2][3:] == -1).all()
    assert (smaller.calls[-1][2] == -1).all()


def test_cold_graph_step_cannot_allocate_a_budgeted_operator(internode, monkeypatch):
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=4
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="was not built before graph capture"):
        backend(
            hidden_states=torch.zeros(4, 64, dtype=torch.bfloat16),
            topk_weights=torch.ones(4, 2),
            topk_ids=torch.zeros(4, 2, dtype=torch.int32),
            global_num_experts=256,
            moe_extra_args=SILU,
        )
    assert not internode.built


def test_initialization_is_cached_and_closed(internode):
    first = _instance(internode)
    # A8W4: the MXFP8 activations DSv4.1's reference runs, on the same weights.
    assert first["op"].quant == "a8w4"
    assert _instance(internode) is first
    assert len(internode.built) == 1
    first["op"].close.assert_not_called()
    internode.module.close_internode_instances()
    first["op"].close.assert_called_once_with()


def test_instance_has_no_duplicate_input_buffers(internode, monkeypatch):
    monkeypatch.setattr(torch, "full", Mock(side_effect=RuntimeError("allocation")))
    monkeypatch.setattr(torch, "zeros", Mock(side_effect=RuntimeError("allocation")))
    assert set(_instance(internode)) == {"op"}


@pytest.mark.parametrize("start,rows", [(0, 2), (2, 1), (4, 0)])
def test_chunk_forwards_actual_views_and_invalid_routes(internode, start, rows):
    instance = _instance(internode, capacity=4)
    x = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64).to(torch.bfloat16)
    weights = torch.ones(3, 2)
    ids = torch.tensor([[0, -1], [-1, 2], [3, 4]], dtype=torch.int32)
    result = internode.module._run_chunk(
        instance, _weights(), x, ids, weights, start, rows
    )
    actual_x, actual_weights, actual_ids = instance["op"].calls[-1]
    assert result.shape == (rows, 64)
    assert torch.equal(actual_x, x[start : start + rows])
    assert torch.equal(actual_weights, weights[start : start + rows])
    assert torch.equal(actual_ids, ids[start : start + rows])
    instance["op"].bind_weights.assert_called_once()


@pytest.mark.parametrize("local_rows", [0, 3, 10])
def test_mixed_dp_phase_uses_group_chunks_even_on_an_idle_rank(
    internode, monkeypatch, local_rows
):
    from atom.utils import forward_context

    group_rows = 10
    context = SimpleNamespace(
        running_tokens=local_rows,
        running_tokens_are_unified=False,
        is_prefill=local_rows == group_rows,
    )
    metadata = forward_context.DPMetadata(np.array([local_rows, group_rows]))
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(context=context, dp_metadata=metadata),
    )
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=group_rows
    )
    x = torch.arange(local_rows * 64).reshape(local_rows, 64).to(torch.bfloat16)
    result = backend(
        hidden_states=x,
        topk_weights=torch.ones(local_rows, 2),
        topk_ids=torch.zeros(local_rows, 2, dtype=torch.int32),
        global_num_experts=256,
        activation=internode.moe.ActivationType.Silu,
        moe_extra_args=SILU,
    )
    assert torch.equal(result, x)
    prefill = next(op for op in internode.built if op.mtpr == 8)
    assert [call[0].shape[0] for call in prefill.calls] == [
        min(local_rows, 8),
        max(local_rows - 8, 0),
    ]
    assert all(not op.calls for op in internode.built if op.mtpr != 8)


@pytest.mark.parametrize(
    "unified,running,dp_rows,expected",
    [(True, 4, None, 4), (False, 3, 10, 10), (True, 0, 10, 10)],
)
def test_group_rows_uses_the_agreed_step_width(
    internode, monkeypatch, unified, running, dp_rows, expected
):
    from atom.utils import forward_context

    context = SimpleNamespace(
        running_tokens=running, running_tokens_are_unified=unified
    )
    metadata = (
        forward_context.DPMetadata(np.array([running, dp_rows]))
        if dp_rows is not None
        else None
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(context=context, dp_metadata=metadata),
    )
    assert internode.module._group_rows() == expected


def test_group_rows_requires_collective_metadata(internode, monkeypatch):
    from atom.utils import forward_context

    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(context=None, dp_metadata=None),
    )
    with pytest.raises(RuntimeError, match="needs dp_metadata"):
        internode.module._group_rows()


def test_teardown_attempts_every_capacity_in_reverse_order(internode):
    first, second = _instance(internode), _instance(internode, capacity=4)
    calls = []
    first["op"].close.side_effect = lambda: calls.append(2)

    def fail():
        calls.append(4)
        raise RuntimeError("close failed")

    second["op"].close.side_effect = fail
    with pytest.raises(RuntimeError, match="close failed"):
        internode.module.close_internode_instances()
    assert calls == [4, 2] and not internode.module._INSTANCES


@pytest.mark.parametrize(
    "field,value", [("situ_beta", 5.0), ("situ_linear_beta", 26.0)]
)
def test_situ_parameters_separate_compiled_instances(internode, field, value):
    settings = {
        "activation": "situv2",
        "swiglu_limit": 0.0,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
    }
    first = _instance(internode, **settings)
    assert _instance(internode, **settings) is first
    settings[field] = value
    assert _instance(internode, **settings) is not first
    assert internode.built[0].settings["situ_beta"] == 4.0
    assert internode.built[0].settings["situ_linear_beta"] == 25.0


@pytest.mark.parametrize("situ", [False, True])
def test_inherited_apply_passes_model_activation_parameters(
    internode, monkeypatch, situ
):
    layer = _weights()
    backend = internode.module.MegaInternodeFusedExperts(
        layer, model_dim=64, inter_dim=32, max_tokens=8
    )
    method = object.__new__(internode.moe.MegaInternodeMxfp4MoEMethod)
    method.fused_experts = backend
    method.use_triton = method.use_triton_ep = method.use_triton_decode = False
    method.use_flydsl_herd = False
    method.fp8_activations = method.is_guinterleave = False
    method.quant_type = internode.moe.QuantType.per_1x32
    method.hidden_pad = method.intermediate_pad = 0
    monkeypatch.setattr(internode.module, "_group_rows", lambda: 2)
    activation = (
        internode.moe.ActivationType.Situv2
        if situ
        else internode.moe.ActivationType.Silu
    )
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    result = method.apply(
        layer,
        x,
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int32),
        global_num_experts=256,
        activation=activation,
    )
    assert torch.equal(result, x)
    for op in internode.built:
        assert op.settings["act"] == ("situv2" if situ else "silu")
        assert op.settings["swiglu_limit"] == (0.0 if situ else 10.0)
        assert op.settings["situ_beta"] == (4.0 if situ else 1.0)
        assert op.settings["situ_linear_beta"] == (25.0 if situ else 1.0)


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"beta": 6.0, "linear_beta": 30.0}, (6.0, 30.0)),
        ({"beta": None, "linear_beta": None}, (1.0, 1.0)),
    ],
)
def test_situ_defaults_and_explicit_parameters(internode, monkeypatch, extra, expected):
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=8
    )
    monkeypatch.setattr(internode.module, "_group_rows", lambda: 1)
    backend(
        hidden_states=torch.zeros(1, 64, dtype=torch.bfloat16),
        topk_weights=torch.ones(1, 2),
        topk_ids=torch.zeros(1, 2, dtype=torch.int32),
        global_num_experts=256,
        activation=internode.moe.ActivationType.Situv2,
        moe_extra_args=extra,
    )
    assert [
        (op.settings["situ_beta"], op.settings["situ_linear_beta"])
        for op in internode.built
    ] == [expected] * 3


def _dp(size, local):
    """A ParallelConfig double: `local` of `size` DP ranks on this node."""
    return SimpleNamespace(
        data_parallel_size=size,
        data_parallel_size_local=local,
        is_multinode_dp=local < size,
    )


def _config(**overrides):
    settings = {
        "moe_backend": "mega",
        "enable_expert_parallel": True,
        "enable_dp_attention": True,
        "tensor_parallel_size": 1,
        "parallel_config": _dp(16, 8),
    }
    settings.update(overrides)
    return atom_config_double(**settings)


def _parallel(atom_modules, monkeypatch, config):
    moe = atom_modules.moe
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(rank_in_group=3))
    parallel_state = importlib.import_module("aiter.dist.parallel_state")
    monkeypatch.setattr(
        parallel_state, "get_prefill_context_model_parallel_world_size", lambda: 1
    )
    with use_custom_atom_config(config):
        return moe.FusedMoEParallelConfig.make(1, 16, config)


def test_unflattened_dp_is_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="shard experts across DP ranks"):
        Config._init_moe_backend(_config(enable_dp_attention=False))


@pytest.mark.parametrize(
    "backend,sp,dp,local,expected",
    [
        ("standard", 1, 16, 8, False),
        ("mega", 1, 8, 8, False),
        ("mega", 1, 16, 8, True),
        ("mega", 2, 1, 1, False),
    ],
)
def test_mega_implementation_is_selected_by_topology(backend, sp, dp, local, expected):
    config = _config(
        moe_backend=backend,
        sequence_parallel_size=sp,
        parallel_config=_dp(dp, local),
    )
    Config._init_moe_backend(config)
    assert is_internode_mega(config) is expected


def test_removed_internode_selector_is_rejected():
    with pytest.raises(ValueError, match="moe_backend must be one of"):
        Config._init_moe_backend(_config(moe_backend="mega_internode"))


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"tensor_parallel_size": 2}, "two-node EP16"),
        ({"pipeline_parallel_size": 2}, "two-node EP16"),
        ({"prefill_context_parallel_size": 2}, "two-node EP16"),
        ({"enable_expert_parallel": False}, "two-node EP16"),
        ({"dp_logical_size": 32}, "two-node EP16"),
        (
            {"parallel_config": _dp(32, 8)},
            "two-node EP16",
        ),
        (
            {"parallel_config": _dp(16, 4)},
            "two-node EP16",
        ),
        ({"enable_tbo": True}, "disable TBO"),
        ({"enable_tbo_decode": True}, "disable TBO"),
    ],
)
def test_internode_mega_refuses_unsupported_topology_and_overlap(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        Config._init_moe_backend(_config(**overrides))


@pytest.mark.parametrize(
    "backend,sp,dp,local,method_name",
    [
        ("standard", 1, 16, 8, "Mxfp4MoEMethod"),
        ("mega", 1, 8, 8, "MegaMxfp4MoEMethod"),
        ("mega", 1, 16, 8, "MegaInternodeMxfp4MoEMethod"),
        ("mega", 2, 1, 1, "Mxfp4MoEMethod"),
    ],
)
def test_method_factory_preserves_single_node_and_sp_mega(
    atom_modules, monkeypatch, backend, sp, dp, local, method_name
):
    moe = atom_modules.moe
    config = _config(
        moe_backend=backend,
        sequence_parallel_size=sp,
        parallel_config=_dp(dp, local),
    )
    methods = {}
    for name in ("Mxfp4MoEMethod", "MegaMxfp4MoEMethod", "MegaInternodeMxfp4MoEMethod"):
        methods[name] = object()
        monkeypatch.setattr(moe, name, lambda *args, result=methods[name]: result)
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: sp > 1)
    with use_custom_atom_config(config):
        assert moe._make_mxfp4_moe_method(None, None) is methods[method_name]


@pytest.mark.parametrize(
    "attention,flatten", [(True, False), (False, True), (True, True)]
)
def test_supported_config_shards_experts_over_all_sixteen_ranks(
    atom_modules, monkeypatch, attention, flatten
):
    config = _config(enable_dp_attention=attention, moe_ep_flatten_tp_across_dp=flatten)
    Config._init_moe_backend(config)
    parallel = _parallel(atom_modules, monkeypatch, config)
    assert parallel.use_ep and parallel.ep_size == 16 and parallel.ep_rank == 3
    assert parallel.local_ep_size == 8 and parallel.tp_size == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_ep": False},
        {"tp_size": 2},
        {"ep_size": 1},
        {"dp_size": 8},
        {"local_ep_size": 16},
        {"dp_logical_ratio": 2},
    ],
)
def test_method_rejects_actual_sharding_mismatch(atom_modules, monkeypatch, overrides):
    parallel = _parallel(atom_modules, monkeypatch, _config())
    vars(parallel).update(overrides)
    with pytest.raises(ValueError, match="two-node EP16"):
        atom_modules.moe.MegaInternodeMxfp4MoEMethod(
            None, SimpleNamespace(moe_parallel_config=parallel)
        )


@pytest.mark.parametrize(
    "backend,disable_mori,installed",
    [
        ("none", False, True),
        ("auto", True, True),
        ("mori", True, True),
        ("auto", False, False),
        ("mori", False, False),
    ],
)
def test_dp_gather_fallback_is_rejected_before_method_initialization(
    atom_modules, monkeypatch, backend, disable_mori, installed
):
    moe = atom_modules.moe
    monkeypatch.setattr(moe.envs, "ATOM_DISABLE_MORI_EP", disable_mori)
    monkeypatch.setattr(moe, "_has_module", lambda name: installed)
    parallel = _parallel(
        atom_modules, monkeypatch, _config(moe_all2all_backend=backend)
    )
    assert not parallel.use_all2all_kernels
    with pytest.raises(ValueError, match="requires the MoRI all-to-all path"):
        moe.MegaInternodeMxfp4MoEMethod(
            None, SimpleNamespace(moe_parallel_config=parallel)
        )


@pytest.mark.parametrize("backend", ["auto", "mori"])
def test_supported_method_keeps_rows_on_the_alltoall_path(
    atom_modules, monkeypatch, backend
):
    moe = atom_modules.moe
    monkeypatch.setattr(moe.envs, "ATOM_DISABLE_MORI_EP", False)
    monkeypatch.setattr(moe, "_has_module", lambda name: True)
    monkeypatch.setattr(moe, "get_gfx", lambda: "gfx950")
    config = _config(moe_all2all_backend=backend)
    parallel = _parallel(atom_modules, monkeypatch, config)
    fused_config = moe.FusedMoEConfig(
        num_experts=256,
        experts_per_token=2,
        hidden_dim=64,
        num_local_experts=16,
        moe_parallel_config=parallel,
        expert_layout=moe.MoEExpertLayout.make(
            num_routed=256,
            num_fused_shared_experts=0,
            num_configured_redundant=0,
            ep_size=16,
            use_all2all=True,
            eplb_enabled=False,
        ),
        in_dtype=torch.bfloat16,
    )
    quant = SimpleNamespace(
        quant_type=moe.QuantType.per_1x32,
        quant_dtype=moe.dtypes.fp4x2,
        quant_method="",
        is_dynamic=True,
    )
    with use_custom_atom_config(config):
        method = moe.MegaInternodeMxfp4MoEMethod(quant, fused_config)
    assert parallel.use_all2all_kernels and not method.use_triton_ep


def test_the_token_budget_is_agreed_once_for_every_layer(internode, monkeypatch):
    calls = _ep_budgets(internode, monkeypatch, [128] * 16)
    for _ in range(3):
        internode.module.MegaInternodeFusedExperts(
            _weights(), model_dim=64, inter_dim=32, max_tokens=128
        )
    assert calls == [128]


def _raw_weights(experts=2, local=2, inter=64, hidden=128):
    return SimpleNamespace(
        w13_weight=torch.zeros(experts, 2 * inter, hidden // 2, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(experts, 2 * inter, hidden // 32),
        w2_weight=torch.zeros(experts, hidden, inter // 2, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(experts, hidden, inter // 32),
        w13_bias=None,
        w2_bias=None,
        local_num_experts=local,
    )


@pytest.fixture
def layouts(monkeypatch):
    """The aiter layout calls `build_internode_weights` makes, recorded."""
    seen = {}
    shuffle = importlib.import_module("aiter.ops.shuffle")
    fp4 = importlib.import_module("aiter.utility.fp4_utils")

    def record(name):
        def call(tensor, *args, **kwargs):
            seen[name] = (tuple(tensor.shape), args, kwargs)
            return tensor.clone()

        return call

    monkeypatch.setattr(shuffle, "shuffle_weight", record("w1"), raising=False)
    monkeypatch.setattr(shuffle, "shuffle_weight_a16w4", record("w2"), raising=False)
    monkeypatch.setattr(
        shuffle, "shuffle_scale_a16w4", record("w2_scale"), raising=False
    )
    monkeypatch.setattr(fp4, "e8m0_shuffle", record("w1_scale"), raising=False)
    return seen


def test_weights_take_the_ops_layouts(internode, layouts):
    layer = _raw_weights()
    internode.module.build_internode_weights(layer)
    assert layouts == {
        "w1": ((2, 128, 64), (), {"layout": (16, 16)}),
        "w1_scale": ((256, 4), (), {}),
        "w2": ((2, 128, 32), (16, False), {}),
        "w2_scale": ((256, 2), (2, False), {}),
    }
    assert all(
        getattr(layer, name).is_contiguous()
        for name in ("_mi_w1", "_mi_w1_scale", "_mi_w2", "_mi_w2_scale")
    )


def test_weights_must_be_this_ranks_experts(internode, layouts):
    with pytest.raises(RuntimeError, match="local weights 2 != local_num_experts 4"):
        internode.module.build_internode_weights(_raw_weights(local=4))
    assert not layouts


@pytest.mark.parametrize("bias", ["w13_bias", "w2_bias"])
def test_expert_biases_are_refused(internode, layouts, bias):
    method = object.__new__(internode.moe.MegaInternodeMxfp4MoEMethod)
    layer = _raw_weights()
    setattr(layer, bias, torch.zeros(1))
    with pytest.raises(NotImplementedError, match="expert biases"):
        method._process_weight_layout_after_loading(layer)
    assert not layouts


def test_raw_weights_and_scales_are_released(internode, layouts):
    method = object.__new__(internode.moe.MegaInternodeMxfp4MoEMethod)
    layer = _raw_weights()
    method._process_weight_layout_after_loading(layer)
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        assert getattr(layer, name).numel() == 0, name
    assert method.get_eplb_weight_views(layer)[0].shape[0] == 2


@pytest.mark.parametrize(
    "method_name,shared,error",
    [
        ("MegaInternodeMxfp4MoEMethod", 0, None),
        ("MegaInternodeMxfp4MoEMethod", 1, "separate shared experts"),
        ("MegaMxfp4MoEMethod", 0, "only MXFP4 routed experts"),
        ("Mxfp4MoEMethod", 0, "only MXFP4 routed experts"),
    ],
)
def test_internode_backend_check(atom_modules, monkeypatch, method_name, shared, error):
    moe = atom_modules.moe
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: False)
    layer = SimpleNamespace(
        quant_method=object.__new__(getattr(moe, method_name)),
        num_fused_shared_experts=shared,
    )
    with use_custom_atom_config(_config()):
        if error is None:
            moe.FusedMoE._validate_moe_backend(layer)
        else:
            with pytest.raises((TypeError, ValueError), match=error):
                moe.FusedMoE._validate_moe_backend(layer)


def test_closing_agrees_on_the_budget_again(internode, monkeypatch):
    calls = _ep_budgets(internode, monkeypatch, [128] * 16)
    internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=128
    )
    internode.module.close_internode_instances()
    internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=128
    )
    assert calls == [128, 128]
