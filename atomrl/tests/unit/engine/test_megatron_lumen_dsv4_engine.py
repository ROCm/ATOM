# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace

import atomrl.engine.training.megatron_lumen_dsv4_engine as dsv4_engine
import pytest
import torch
from atomrl.engine.training.bridges.dsv4 import DSV4Dims
from atomrl.engine.training.megatron_engine import MegatronEngine
from atomrl.engine.training.megatron_lumen_dsv4_engine import (
    MegatronLumenDSV4Engine,
    _dsv4_router_kwargs,
    _lockstep_stream,
    _named_export_tensors,
    _StreamingGatheredParamMapping,
)


def test_resolve_dsv4_topk_backend_uses_canonical_name_with_legacy_alias():
    assert hasattr(dsv4_engine, "_resolve_dsv4_topk_backend")
    resolve = dsv4_engine._resolve_dsv4_topk_backend
    assert resolve({}) == "torch"
    assert resolve({"miles_dsa_topk_backend": "flashinfer"}) == "flashinfer"
    assert (
        resolve(
            {"dsv4_dsa_topk_backend": "torch", "miles_dsa_topk_backend": "flashinfer"}
        )
        == "torch"
    )


def test_dsv4_actor_explicitly_disables_fp8_training():
    source = inspect.getsource(MegatronLumenDSV4Engine.initialize)

    assert "bf16=True" in source
    assert "fp8=None" in source


@pytest.mark.parametrize(
    "method_name",
    [
        "_engine_update_policy_rowwise",
        "_engine_update_policy_packed",
        "_engine_update_policy_pp",
    ],
)
def test_megatron_grpo_applies_rollout_importance_weights(method_name):
    source = inspect.getsource(getattr(MegatronEngine, method_name))
    grpo_branch = source.split("elif algo_name == AlgorithmName.GRPO.value:", 1)[
        1
    ].split("\n            else:", 1)[0]

    assert "rollout_is_weights=ris" in grpo_branch


def test_dsv4_pp_uses_ordered_dynamic_shape_exchange():
    source = inspect.getsource(MegatronLumenDSV4Engine.initialize)

    assert 'pp_kwargs["variable_seq_lengths"] = True' in source
    assert 'pp_kwargs["batch_p2p_comm"] = False' in source


def _parameter(values, *, tensor_parallel=False, partition_dim=0):
    param = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    param.tensor_model_parallel = tensor_parallel
    param.partition_dim = partition_dim
    param.partition_stride = 1
    return param


def _cpu_adam(*, optimizer_type=torch.optim.Adam, **flags):
    optimizer = optimizer_type([torch.nn.Parameter(torch.ones(1))])
    optimizer.param_groups[0].update(flags)
    return optimizer


def _hybrid_optimizer(*, cpu_optimizers=None, **overrides):
    values = {
        "offload_fraction": 1.0,
        "gpu_optimizer": None,
        "cpu_optimizers": ([_cpu_adam()] if cpu_optimizers is None else cpu_optimizers),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _streamed_config(mode="adam", chunk_mib=256, **overrides):
    values = {
        "optimizer_cpu_offload": True,
        "optimizer_offload_fraction": 1.0,
        "streamed_optimizer_mode": mode,
        "streamed_optimizer_chunk_size_mib": chunk_mib,
    }
    values.update(overrides)
    return values


def test_streamed_optimizer_precision_uses_configured_bf16_moments():
    kwargs = dsv4_engine._optimizer_precision_kwargs(
        {
            "use_precision_aware_optimizer": True,
            "streamed_optimizer_moment_dtype": "bf16",
        }
    )

    assert kwargs["exp_avg_dtype"] is torch.bfloat16
    assert kwargs["exp_avg_sq_dtype"] is torch.bfloat16
    assert kwargs["main_params_dtype"] is torch.float32


def _patch_streamed_adam_capability(monkeypatch, value=True):
    real_import_module = importlib.import_module
    fake_module = SimpleNamespace(
        ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM=value,
    )

    def import_module(name, package=None):
        if name == ("megatron.core.optimizer.cpu_offloading.hybrid_optimizer"):
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)


@pytest.mark.parametrize(
    "mode",
    ["bad", "", "ADAMW", " adam ", "ADAM", "SGD", "OFF"],
)
def test_streamed_optimizer_rejects_unknown_mode(mode):
    with pytest.raises(ValueError, match="streamed_optimizer_mode"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(mode=mode),
            object(),
        )


@pytest.mark.parametrize("moment_dtype", ["", "float16", "BF16", None, True])
def test_streamed_optimizer_rejects_unknown_moment_dtype(moment_dtype):
    with pytest.raises(ValueError, match="streamed_optimizer_moment_dtype"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(streamed_optimizer_moment_dtype=moment_dtype),
            object(),
        )


@pytest.mark.parametrize("chunk_mib", [0, -1, 1.5, True, "256"])
def test_streamed_optimizer_rejects_invalid_chunk_size(chunk_mib):
    with pytest.raises(ValueError, match="streamed_optimizer_chunk_size_mib"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(mode="off", chunk_mib=chunk_mib),
            object(),
        )


@pytest.mark.parametrize("chunk_mib", [1025, 10**100])
def test_streamed_optimizer_rejects_oversized_chunk_size(chunk_mib):
    with pytest.raises(ValueError, match="streamed_optimizer_chunk_size_mib"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(mode="off", chunk_mib=chunk_mib),
            object(),
        )


def test_streamed_optimizer_accepts_and_converts_maximum_chunk_size(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    hybrid = _hybrid_optimizer()

    configured = dsv4_engine._configure_streamed_optimizer_request(
        _streamed_config(chunk_mib=1024),
        hybrid,
    )

    assert configured == [hybrid]
    assert hybrid._lumen_streamed_adam_chunk_numel == 1024 * 1024 * 1024 // 4


def test_streamed_optimizer_off_accepts_unpatched_optimizer():
    assert (
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(mode="off"),
            object(),
        )
        is None
    )


def test_streamed_optimizer_sgd_preserves_capability_free_path():
    hybrid = _hybrid_optimizer(
        cpu_optimizers=[
            torch.optim.SGD(
                [torch.nn.Parameter(torch.ones(1))],
                lr=0.1,
            )
        ]
    )

    assert dsv4_engine._validate_streamed_optimizer_request(
        _streamed_config(mode="sgd"),
        SimpleNamespace(optimizer=hybrid),
    ) == [hybrid]


@pytest.mark.parametrize("capability", [None, False])
def test_streamed_adam_requires_runtime_capability(monkeypatch, capability):
    _patch_streamed_adam_capability(monkeypatch, capability)

    with pytest.raises(RuntimeError, match="ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(),
        )


def test_streamed_adam_reports_unavailable_runtime_as_missing_capability(monkeypatch):
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == ("megatron.core.optimizer.cpu_offloading.hybrid_optimizer"):
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(),
        )


def test_streamed_adam_requires_hybrid_device_optimizer(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match="HybridDeviceOptimizer"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            SimpleNamespace(optimizer=object()),
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"optimizer_cpu_offload": False}, "optimizer_cpu_offload"),
        ({"optimizer_offload_fraction": 0.875}, "full CPU offload"),
    ],
)
def test_streamed_adam_requires_full_configured_cpu_offload(
    monkeypatch,
    overrides,
    error,
):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match=error):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(**overrides),
            _hybrid_optimizer(),
        )


def test_streamed_adam_rejects_partial_runtime_offload(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match="full CPU offload"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(offload_fraction=0.875),
        )


def test_streamed_adam_rejects_gpu_sub_optimizer(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match="GPU optimizer"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(gpu_optimizer=object()),
        )


def test_streamed_adam_rejects_non_adam_cpu_optimizer(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)

    with pytest.raises(RuntimeError, match="Adam or AdamW"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(cpu_optimizers=[sgd]),
        )


def test_streamed_adam_rejects_empty_cpu_optimizer_collection(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match="at least one CPU Adam or AdamW"):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(cpu_optimizers=[]),
        )


@pytest.mark.parametrize(
    "flag",
    ["amsgrad", "differentiable", "capturable", "foreach"],
)
def test_streamed_adam_rejects_incompatible_group_flags(monkeypatch, flag):
    _patch_streamed_adam_capability(monkeypatch)

    with pytest.raises(RuntimeError, match=flag):
        dsv4_engine._validate_streamed_optimizer_request(
            _streamed_config(),
            _hybrid_optimizer(cpu_optimizers=[_cpu_adam(**{flag: True})]),
        )


def test_streamed_adam_allows_fused_target_flag(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    hybrid = _hybrid_optimizer(cpu_optimizers=[_cpu_adam(fused=True)])

    assert dsv4_engine._validate_streamed_optimizer_request(
        _streamed_config(),
        hybrid,
    ) == [hybrid]


def test_find_hybrid_device_optimizers_unwraps_nested_chains():
    hybrid = _hybrid_optimizer()
    wrapped = SimpleNamespace(
        inner_optimizer=SimpleNamespace(
            chained_optimizers=[
                object(),
                SimpleNamespace(optimizer=(object(), hybrid)),
            ]
        )
    )

    assert dsv4_engine._find_hybrid_device_optimizers(wrapped) == [hybrid]


def test_find_hybrid_device_optimizers_handles_cycles():
    first = SimpleNamespace()
    second = SimpleNamespace(optimizer=first)
    first.inner_optimizer = second
    first.chained_optimizers = [first, second]

    assert dsv4_engine._find_hybrid_device_optimizers(first) == []


def test_chained_optimizer_ignores_asserting_singular_property(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    dense = _hybrid_optimizer()
    expert = _hybrid_optimizer()

    class ChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [dense, expert]

        @property
        def optimizer(self):
            raise AssertionError("singular optimizer is invalid for MoE")

    configured = dsv4_engine._configure_streamed_optimizer_request(
        _streamed_config(chunk_mib=32),
        ChainedOptimizer(),
    )

    assert configured == [dense, expert]
    for hybrid in configured:
        assert hybrid._lumen_streamed_optimizer_mode == "adam"
        assert hybrid._lumen_streamed_adam_chunk_numel == 32 * 1024 * 1024 // 4
        assert hybrid._lumen_streamed_adam_moment_dtype == torch.float32


def test_multi_hdo_configuration_is_atomic(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    valid = _hybrid_optimizer()
    invalid = _hybrid_optimizer(gpu_optimizer=object())
    chained = SimpleNamespace(chained_optimizers=[invalid, valid])

    with pytest.raises(RuntimeError, match="GPU optimizer"):
        dsv4_engine._configure_streamed_optimizer_request(
            _streamed_config(),
            chained,
        )

    for hybrid in (valid, invalid):
        assert not hasattr(hybrid, "_lumen_streamed_optimizer_mode")
        assert not hasattr(hybrid, "_lumen_streamed_adam_chunk_numel")


def test_find_hybrid_device_optimizers_returns_unique_deterministic_order():
    first = _hybrid_optimizer()
    second = _hybrid_optimizer()
    wrapped = SimpleNamespace(
        chained_optimizers=[first, second, first],
        inner_optimizer=second,
    )

    assert dsv4_engine._find_hybrid_device_optimizers(wrapped) == [
        first,
        second,
    ]


def test_streamed_adam_settings_attach_only_after_validation(monkeypatch):
    _patch_streamed_adam_capability(monkeypatch)
    hybrid = _hybrid_optimizer()
    wrapped = SimpleNamespace(inner_optimizer=hybrid)

    configured = dsv4_engine._configure_streamed_optimizer_request(
        _streamed_config(
            chunk_mib=64,
            streamed_optimizer_moment_dtype="bf16",
        ),
        wrapped,
    )

    assert configured == [hybrid]
    assert hybrid._lumen_streamed_optimizer_mode == "adam"
    assert hybrid._lumen_streamed_adam_chunk_numel == 64 * 1024 * 1024 // 4
    assert hybrid._lumen_streamed_adam_moment_dtype == torch.bfloat16

    invalid = _hybrid_optimizer(gpu_optimizer=object())
    with pytest.raises(RuntimeError, match="GPU optimizer"):
        dsv4_engine._configure_streamed_optimizer_request(
            _streamed_config(),
            invalid,
        )
    assert not hasattr(invalid, "_lumen_streamed_optimizer_mode")
    assert not hasattr(invalid, "_lumen_streamed_adam_chunk_numel")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("adam", "512 MiB"),
        ("off", "0 MiB (disabled)"),
        ("sgd", "runtime-sized/unknown"),
    ],
)
def test_streamed_optimizer_staging_diagnostic(mode, expected):
    assert dsv4_engine._streamed_optimizer_staging_diagnostic(mode, 256) == expected


def test_bf16_streamed_adam_reports_four_staging_buffers():
    assert (
        dsv4_engine._streamed_optimizer_staging_diagnostic(
            "adam",
            256,
            torch.bfloat16,
        )
        == "1024 MiB"
    )


def test_noaux_tc_router_uses_checkpoint_scoring_and_normalization():
    kwargs = _dsv4_router_kwargs(
        {
            "topk_method": "noaux_tc",
            "scoring_func": "sqrtsoftplus",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
        },
        {},
    )

    assert kwargs == {
        "moe_router_load_balancing_type": "none",
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_dtype": "fp32",
        "moe_router_topk_scaling_factor": 1.5,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
    }


def test_noaux_tc_router_rejects_conflicting_scaling_override():
    with pytest.raises(ValueError, match="moe_router_topk_scaling_factor"):
        _dsv4_router_kwargs(
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
                "routed_scaling_factor": 1.5,
            },
            {"moe_router_topk_scaling_factor": 2.0},
        )


def test_precision_aware_optimizer_keeps_fp32_master_on_cpu():
    precision_kwargs = getattr(
        dsv4_engine, "_optimizer_precision_kwargs", lambda _config: {}
    )
    assert precision_kwargs({"use_precision_aware_optimizer": True}) == {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": torch.float32,
        "main_params_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
    }
    assert precision_kwargs({}) == {
        "use_precision_aware_optimizer": False,
    }


def test_hash_tables_are_registered_as_checkpoint_static_buffers():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.router = torch.nn.Module()

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
    state = {
        f"decoder.layers.{layer}.mlp.router.tid2eid": torch.arange(
            16, dtype=torch.int32
        ).view(8, 2)
        for layer in range(3)
    }
    register_buffers = getattr(
        dsv4_engine, "_register_checkpoint_static_buffers", lambda *_args: None
    )

    register_buffers(model, state)

    incompat = model.load_state_dict(state, strict=False)
    assert incompat.unexpected_keys == []
    assert dict(model.named_buffers()).keys() >= state.keys()


@pytest.mark.parametrize(
    ("checkpoint_config", "error"),
    [
        (
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
            },
            "scoring_func",
        ),
        (
            {
                "topk_method": "noaux_tc",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": False,
            },
            "norm_topk_prob",
        ),
        (
            {
                "topk_method": "group_limited_greedy",
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
            },
            "topk_method",
        ),
        (
            {
                "scoring_func": "sqrtsoftplus",
                "norm_topk_prob": True,
            },
            "topk_method",
        ),
    ],
)
def test_noaux_tc_router_rejects_unsupported_checkpoint_semantics(
    checkpoint_config,
    error,
):
    with pytest.raises(ValueError, match=error):
        _dsv4_router_kwargs(checkpoint_config, {})


def test_export_tensors_exclude_frozen_router_buffers():
    module = torch.nn.Module()
    module.register_parameter("weight", _parameter([1, 2]))
    router = torch.nn.Module()
    router.register_buffer("expert_bias", torch.tensor([0.25, -0.5]))
    router.register_buffer("local_tokens_per_expert", torch.ones(2), persistent=False)
    module.add_module("router", router)

    exported = dict(_named_export_tensors(module))

    assert set(exported) == {"weight"}


def test_lockstep_stream_rendezvous_before_requesting_next_item():
    events = []

    def source():
        events.append("produce:first")
        yield "first"
        events.append("produce:second")
        yield "second"

    stream = _lockstep_stream(
        source(),
        synchronize=lambda: events.append("barrier"),
    )

    assert next(stream) == "first"
    assert events == ["produce:first"]
    assert next(stream) == "second"
    assert events == ["produce:first", "barrier", "produce:second"]
    with pytest.raises(StopIteration):
        next(stream)
    assert events == ["produce:first", "barrier", "produce:second", "barrier"]


def _install_fake_megatron_parallel_state(monkeypatch, *, pp_size):
    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_expert_tensor_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_group=lambda: "pp",
    )
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    engine = object.__new__(MegatronLumenDSV4Engine)
    engine.module = torch.nn.Module()
    engine._dims = DSV4Dims(num_layers=0, num_experts=0, compress_ratios=[])
    engine._ep_size = 1
    engine._pp_size = pp_size
    engine._pp_rank = 0
    engine._layers_per_pp_rank = [0] * pp_size
    return engine


def test_get_per_tensor_param_pp_wraps_stream_with_default_barrier(monkeypatch):
    engine = _install_fake_megatron_parallel_state(monkeypatch, pp_size=2)
    events = []

    def fake_convert(mapping, *args, **kwargs):
        tensor = torch.empty(1, device="meta") if mapping._metadata else torch.ones(1)
        yield "tensor", tensor

    def fake_all_gather_object(output, value, *, group):
        assert group == "pp"
        output[0] = value
        output[1] = (1, {})

    def fake_barrier():
        events.append("barrier")

    original_lockstep_stream = dsv4_engine._lockstep_stream
    lockstep_calls = []

    def record_lockstep_stream(stream, synchronize):
        lockstep_calls.append((stream, synchronize))
        return original_lockstep_stream(stream, synchronize)

    monkeypatch.setattr(dsv4_engine.dsv4, "megatron_to_hf", fake_convert)
    monkeypatch.setattr(dsv4_engine, "_lockstep_stream", record_lockstep_stream)
    monkeypatch.setattr(dsv4_engine.dist, "barrier", fake_barrier)
    monkeypatch.setattr(
        dsv4_engine.dist,
        "get_process_group_ranks",
        lambda group: [0, 1],
    )
    monkeypatch.setattr(dsv4_engine.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(
        dsv4_engine.dist,
        "broadcast",
        lambda tensor, *, src, group, async_op: (
            events.append("broadcast") or SimpleNamespace(wait=lambda: None)
        ),
    )
    monkeypatch.setattr(dsv4_engine.torch.cuda, "synchronize", lambda device=None: None)

    params, metadata = engine.get_per_tensor_param()

    assert metadata is None
    assert len(lockstep_calls) == 1
    assert lockstep_calls[0][1] is fake_barrier
    key, tensor = next(params)
    assert key == "tensor"
    torch.testing.assert_close(tensor, torch.ones(1))
    assert events == ["broadcast"]
    with pytest.raises(StopIteration):
        next(params)
    assert events == ["broadcast", "barrier"]


def test_get_per_tensor_param_pp_rejects_source_metadata_mismatch(monkeypatch):
    engine = _install_fake_megatron_parallel_state(monkeypatch, pp_size=2)
    broadcast_calls = []

    def fake_convert(mapping, *args, **kwargs):
        tensor = torch.empty(1, device="meta") if mapping._metadata else torch.ones(2)
        yield "tensor", tensor

    def fake_all_gather_object(output, value, *, group):
        output[0] = value
        output[1] = (1, {})

    monkeypatch.setattr(dsv4_engine.dsv4, "megatron_to_hf", fake_convert)
    monkeypatch.setattr(
        dsv4_engine.dist,
        "get_process_group_ranks",
        lambda group: [0, 1],
    )
    monkeypatch.setattr(dsv4_engine.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(
        dsv4_engine.dist,
        "broadcast",
        lambda *args, **kwargs: broadcast_calls.append((args, kwargs)),
    )

    params, _ = engine.get_per_tensor_param()

    with pytest.raises(
        RuntimeError,
        match=r"tensor.*expected shape=\(1,\).*actual shape=\(2,\)",
    ):
        next(params)
    assert broadcast_calls == []


def test_get_per_tensor_param_pp_waits_for_broadcast_before_yield(monkeypatch):
    engine = _install_fake_megatron_parallel_state(monkeypatch, pp_size=2)
    events = []

    def fake_convert(mapping, *args, **kwargs):
        tensor = torch.empty(1, device="meta") if mapping._metadata else torch.ones(1)
        yield "tensor", tensor

    def fake_all_gather_object(output, value, *, group):
        output[0] = value
        output[1] = (1, {})

    class FakeWork:
        def wait(self):
            events.append("wait")

    def fake_broadcast(tensor, *, src, group, async_op=False):
        events.append(("broadcast", async_op))
        return FakeWork() if async_op else None

    monkeypatch.setattr(dsv4_engine.dsv4, "megatron_to_hf", fake_convert)
    monkeypatch.setattr(
        dsv4_engine.dist,
        "get_process_group_ranks",
        lambda group: [0, 1],
    )
    monkeypatch.setattr(dsv4_engine.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(dsv4_engine.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(
        dsv4_engine.torch.cuda,
        "synchronize",
        lambda device=None: events.append(("synchronize", device)),
    )

    params, _ = engine.get_per_tensor_param()
    key, tensor = next(params)

    assert key == "tensor"
    torch.testing.assert_close(tensor, torch.ones(1))
    assert events == [
        ("broadcast", True),
        "wait",
        ("synchronize", tensor.device),
    ]


def test_get_per_tensor_param_pp1_leaves_stream_unwrapped(monkeypatch):
    engine = _install_fake_megatron_parallel_state(monkeypatch, pp_size=1)

    def fake_convert(mapping, *args, **kwargs):
        yield "tensor", torch.ones(1)

    monkeypatch.setattr(dsv4_engine.dsv4, "megatron_to_hf", fake_convert)
    monkeypatch.setattr(
        dsv4_engine,
        "_lockstep_stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("PP=1 must not use the lockstep stream")
        ),
    )

    params, metadata = engine.get_per_tensor_param()

    assert metadata is None
    exported = list(params)
    assert len(exported) == 1
    assert exported[0][0] == "tensor"
    torch.testing.assert_close(exported[0][1], torch.ones(1))


def test_streaming_metadata_mapping_expands_tp_shape_without_allocating(monkeypatch):
    param = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata lookup must not communicate")
        ),
    )

    mapping = _StreamingGatheredParamMapping(
        [("module.decoder.layers.0.self_attention.wq_a.weight", param)],
        tp_size=2,
        tp_group=object(),
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
        num_experts=0,
        metadata=True,
    )

    tensor = mapping["decoder.layers.0.self_attention.wq_a.weight"]

    assert tensor.device.type == "meta"
    assert tensor.shape == (4, 3)


def test_streaming_mapping_prefers_recorded_physical_partition_metadata():
    sharded = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=False)
    sharded._lumen_weight_partition_dim = 0
    replicated = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    replicated._lumen_weight_partition_dim = None
    mapping = _StreamingGatheredParamMapping(
        [("sharded.weight", sharded), ("replicated.weight", replicated)],
        tp_size=4,
        tp_group=object(),
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
        num_experts=0,
        metadata=True,
    )

    assert mapping["sharded.weight"].shape == (8, 3)
    assert mapping["replicated.weight"].shape == (2, 3)


def test_streaming_mapping_gathers_only_requested_global_expert(monkeypatch):
    local_zero = _parameter([[1, 2], [3, 4]])
    local_one = _parameter([[10, 20], [30, 40]])
    calls = []

    def fake_all_gather(outputs, tensor, group):
        calls.append((tensor.clone(), group))
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 100)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    ep_group = object()
    mapping = _StreamingGatheredParamMapping(
        [
            ("decoder.layers.0.mlp.experts.linear_fc1.weight0", local_zero),
            ("decoder.layers.0.mlp.experts.linear_fc1.weight1", local_one),
        ],
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=ep_group,
        etp_size=1,
        etp_group=None,
        num_experts=4,
    )

    result = mapping["decoder.layers.0.mlp.experts.linear_fc1.weight3"]

    torch.testing.assert_close(result, local_one + 100)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], local_one)
    assert calls[0][1] is ep_group


def test_streaming_mapping_makes_expert_gather_buffers_contiguous(monkeypatch):
    local = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1))

    def fake_all_gather(outputs, tensor, group):
        assert tensor.is_contiguous()
        assert all(output.is_contiguous() for output in outputs)
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 100)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    mapping = _StreamingGatheredParamMapping(
        [("decoder.layers.0.mlp.experts.linear_fc1.weight0", local)],
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
        num_experts=2,
    )

    result = mapping["decoder.layers.0.mlp.experts.linear_fc1.weight1"]

    torch.testing.assert_close(result, local + 100)


def test_get_per_tensor_param_defers_tp_gather_until_iteration(monkeypatch):
    pytest.importorskip("megatron")
    from megatron.core import parallel_state as mpu

    param = _parameter([[1, 2, 3], [4, 5, 6]], tensor_parallel=True)
    calls = []

    def fake_all_gather(outputs, tensor, group):
        calls.append(group)
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 10)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 1)

    engine = object.__new__(MegatronLumenDSV4Engine)
    engine.module = type(
        "Module",
        (),
        {
            "named_parameters": lambda self: iter(
                [
                    ("module.embedding.word_embeddings.weight", param),
                    ("module.decoder.final_layernorm.weight", _parameter([1, 1, 1])),
                    ("module.output_layer.weight", _parameter([[1, 1, 1]])),
                ]
            ),
            "named_buffers": lambda self: iter(()),
        },
    )()
    engine._dims = DSV4Dims(num_layers=0, num_experts=0, compress_ratios=[])
    engine._ep_size = 1
    engine._pp_size = 1
    engine._pp_rank = 0

    params, _ = engine.get_per_tensor_param()

    assert calls == []
    converted = dict(params)
    assert calls == ["tp"]
    assert converted["embed.weight"].shape == (4, 3)
