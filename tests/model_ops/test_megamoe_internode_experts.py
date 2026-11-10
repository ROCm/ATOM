# SPDX-License-Identifier: MIT
"""CPU regressions at the EP16 adapter's AITER/MoRI boundaries."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from conftest import atom_config_double

from atom.config import Config, use_custom_atom_config


@pytest.fixture
def internode(atom_modules, monkeypatch):
    module = importlib.import_module(
        "atom.model_ops.fused_moe.megamoe_internode_experts"
    )
    built = []

    class Operator:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.device = torch.device("cpu")
            self.mtpr = kwargs["max_tok_per_rank"]
            self.quant = kwargs["quant"]
            self._s1_tile = self._gemm2_bn = self._gemm2_bm = self._gemm2_cu = 16
            self._chunk_tokens = 32
            self._s2_tuned = self._stage2_sdma = self._single_kernel = False
            self.stage1_kernel_name = "cpu-boundary"
            self.layout = SimpleNamespace(total_bytes=0)
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
    monkeypatch.setattr(module, "_rank_world", lambda: (0, 16))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("ATOM_TORCH_PROFILER_DIR", raising=False)
    monkeypatch.setattr(module, "_PREFILL_MTPR", 8)
    monkeypatch.setattr(module, "_DECODE_MTPRS", (2, 4))
    yield SimpleNamespace(module=module, built=built, moe=atom_modules.moe)
    module.close_internode_instances()


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


@pytest.mark.parametrize("profiler_dir", [None, ""])
def test_initialization_without_profiler_is_cached_and_closed(
    internode, monkeypatch, profiler_dir
):
    if profiler_dir is not None:
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", profiler_dir)
    first = _instance(internode)
    assert _instance(internode) is first
    assert len(internode.built) == 1
    first["op"].close.assert_not_called()
    internode.module.close_internode_instances()
    first["op"].close.assert_called_once_with()


def test_profiler_audit_creates_parent_directories(internode, monkeypatch, tmp_path):
    profiler = tmp_path / "missing" / "trace"
    monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", str(profiler))
    _instance(internode)
    audit = json.loads(
        (profiler.parent / "tune_audit" / "rank0_capacity2.json").read_text()
    )
    assert audit["rank"] == 0 and audit["capacity"] == 2
    assert audit["quant"] == "a4w4"


@pytest.mark.parametrize("operation", ["mkdir", "write_text"])
def test_audit_io_failure_preserves_cached_operator(
    internode, monkeypatch, tmp_path, caplog, operation
):
    monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", str(tmp_path / "trace"))
    monkeypatch.setattr(Path, operation, Mock(side_effect=OSError("unwritable")))
    first = _instance(internode)
    assert _instance(internode) is first
    assert "unable to write optional tune audit" in caplog.text
    internode.module.close_internode_instances()
    first["op"].close.assert_called_once_with()


def test_failed_buffer_allocation_closes_uncached_operator(internode, monkeypatch):
    monkeypatch.setattr(torch, "full", Mock(side_effect=RuntimeError("allocation")))
    with pytest.raises(RuntimeError, match="allocation"):
        _instance(internode)
    assert not internode.module._INSTANCES
    internode.built[0].close.assert_called_once_with()


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
    method.fp8_activations = method.is_guinterleave = False
    method.quant_type = internode.moe.QuantType.per_1x32
    method.hidden_pad = method.intermediate_pad = 0
    monkeypatch.setattr(internode.module, "_group_rows", lambda rows: (rows, True))
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
        (None, (4.0, 25.0)),
        ({"beta": 6.0, "linear_beta": 30.0}, (6.0, 30.0)),
        ({"beta": None, "linear_beta": None}, (1.0, 1.0)),
    ],
)
def test_situ_defaults_and_explicit_parameters(internode, monkeypatch, extra, expected):
    backend = internode.module.MegaInternodeFusedExperts(
        _weights(), model_dim=64, inter_dim=32, max_tokens=8
    )
    monkeypatch.setattr(internode.module, "_group_rows", lambda rows: (rows, True))
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


def _config(**overrides):
    settings = {
        "moe_backend": "mega_internode",
        "enable_expert_parallel": True,
        "enable_dp_attention": True,
        "tensor_parallel_size": 1,
        "parallel_config": SimpleNamespace(
            data_parallel_size=16, data_parallel_size_local=8
        ),
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
        {"ep_size": 1},
        {"dp_size": 8},
        {"local_ep_size": 16},
        {"dp_logical_ratio": 2},
    ],
)
def test_method_rejects_actual_sharding_mismatch(atom_modules, monkeypatch, overrides):
    parallel = _parallel(atom_modules, monkeypatch, _config())
    vars(parallel).update(overrides)
    with pytest.raises(ValueError, match="actual two-node EP16 expert sharding"):
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
