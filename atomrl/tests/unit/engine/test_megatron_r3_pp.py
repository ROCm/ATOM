# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU-only tests for pipeline-parallel MILES R3 route replay."""

from __future__ import annotations

import inspect
import sys
import types
import typing
from types import SimpleNamespace

import pytest
import torch
from atomrl.engine.training import megatron_engine
from atomrl.engine.training.megatron_engine import MegatronEngine
from atomrl.engine.training.megatron_lumen_dsv4_engine import (
    MegatronLumenDSV4Engine,
)

_R3_CAPABILITIES = {
    "megatron.core.tensor_parallel.random": (
        "ATOMRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD",
    ),
    "megatron.core.transformer.moe.router_replay": (
        "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO",
        "ATOMRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS",
    ),
}


def _install_r3_capabilities(monkeypatch, *, missing=()) -> None:
    missing = set(missing)
    for module_name, markers in _R3_CAPABILITIES.items():
        module = types.ModuleType(module_name)
        module.__file__ = f"/runtime/{module_name.replace('.', '/')}.py"
        for marker in markers:
            if marker not in missing:
                setattr(module, marker, True)
        monkeypatch.setitem(sys.modules, module_name, module)


def test_dsv4_r3_runtime_capabilities_pass_in_single_process(monkeypatch) -> None:
    _install_r3_capabilities(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    megatron_engine._validate_dsv4_r3_runtime_capabilities(
        dsv4_enabled=True,
        r3_enabled=True,
    )


@pytest.mark.parametrize(
    "missing_marker",
    [
        "ATOMRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD",
        "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO",
        "ATOMRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS",
    ],
)
def test_dsv4_r3_runtime_capabilities_report_marker_rank_and_module_path(
    monkeypatch,
    missing_marker,
) -> None:
    _install_r3_capabilities(monkeypatch, missing={missing_marker})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        megatron_engine._validate_dsv4_r3_runtime_capabilities(
            dsv4_enabled=True,
            r3_enabled=True,
        )

    message = str(exc_info.value)
    assert missing_marker in message
    assert "rank=0" in message
    assert "/runtime/megatron/core/" in message


def test_dsv4_r3_runtime_capabilities_raise_consistently_for_remote_failure(
    monkeypatch,
) -> None:
    _install_r3_capabilities(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    remote_failure = {
        "rank": 1,
        "missing": ["ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO"],
        "module_paths": {
            "megatron.core.transformer.moe.router_replay": "/remote/megatron/core/transformer/moe/router_replay.py",
        },
    }

    def all_gather_object(output, local):
        output[:] = [local, remote_failure]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        all_gather_object,
    )

    with pytest.raises(RuntimeError) as exc_info:
        megatron_engine._validate_dsv4_r3_runtime_capabilities(
            dsv4_enabled=True,
            r3_enabled=True,
        )

    message = str(exc_info.value)
    assert "rank=1" in message
    assert "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO" in message
    assert "/remote/megatron/core/transformer/moe/router_replay.py" in message


def test_dsv4_r3_gate_runs_before_model_build_and_execution_boundaries() -> None:
    initialize_source = inspect.getsource(MegatronLumenDSV4Engine.initialize)
    gate = initialize_source.index("_clear_stale_router_replay_instances(")
    model_build = initialize_source.index("model = GPTModel(")
    assert "dsv4_enabled=True" in initialize_source[gate:model_build]

    for method in (
        MegatronEngine._forward_logits,
        MegatronEngine._forward_logits_packed,
        MegatronEngine._engine_compute_log_probs_pp,
        MegatronEngine._engine_update_policy_pp,
    ):
        source = inspect.getsource(method)
        assert "self._validate_r3_runtime_capabilities()" in source


class _ReplayInstance:
    def __init__(self) -> None:
        self.target_indices: list[torch.Tensor] = []

    def set_target_indices(self, indices: torch.Tensor) -> None:
        self.target_indices.append(indices.clone())


def _install_router_replay(
    monkeypatch, instance_count: int, instance_factory=_ReplayInstance
):
    instances = [instance_factory() for _ in range(instance_count)]

    class RouterReplayAction:
        REPLAY_FORWARD = "replay_forward"

    class RouterReplay:
        global_router_replay_instances = instances
        action = None
        clear_indices_calls = 0
        clear_instances_calls = 0

        @classmethod
        def clear_global_indices(cls):
            cls.clear_indices_calls += 1
            for instance in cls.global_router_replay_instances:
                instance.target_indices.clear()

        @classmethod
        def clear_global_router_replay_instances(cls):
            cls.clear_instances_calls += 1
            cls.global_router_replay_instances.clear()

        @classmethod
        def clear_global_router_replay_action(cls):
            cls.action = None

        @classmethod
        def set_global_router_replay_action(cls, action):
            cls.action = action

    replay_module = types.ModuleType("megatron.core.transformer.moe.router_replay")
    replay_module.RouterReplay = RouterReplay
    replay_module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.moe.router_replay",
        replay_module,
    )
    return RouterReplay, instances


def test_stale_replay_instances_are_cleared_immediately_before_model_build(
    monkeypatch,
) -> None:
    replay, _ = _install_router_replay(monkeypatch, 2)

    assert hasattr(
        megatron_engine, "_clear_stale_router_replay_instances"
    ), "R3 model construction needs a stale-instance cleanup helper"
    megatron_engine._clear_stale_router_replay_instances(r3_enabled=True)

    assert replay.global_router_replay_instances == []
    assert replay.clear_instances_calls == 1
    for engine_type in (MegatronEngine, MegatronLumenDSV4Engine):
        source = inspect.getsource(engine_type.initialize)
        cleanup = source.index("_clear_stale_router_replay_instances(")
        construction = source.index("model = GPTModel(")
        assert cleanup < construction
        assert (
            source[cleanup:construction].count("_clear_stale_router_replay_instances(")
            == 1
        )


def test_r3_model_build_fails_when_fork_lacks_instance_cleanup_api(
    monkeypatch,
) -> None:
    replay, _ = _install_router_replay(monkeypatch, 1)
    delattr(replay, "clear_global_router_replay_instances")

    assert hasattr(megatron_engine, "_clear_stale_router_replay_instances")
    with pytest.raises(TypeError, match="clear_global_router_replay_instances"):
        megatron_engine._clear_stale_router_replay_instances(r3_enabled=True)


def _engine(
    *,
    pp_rank: int = 0,
    tp_rank: int = 0,
    sequence_parallel: bool = False,
    num_experts: int = 1_000_000,
) -> MegatronEngine:
    engine = MegatronEngine.__new__(MegatronEngine)
    engine._pp_rank = pp_rank
    engine._pp_size = 4
    engine._layers_per_pp_rank = [11, 11, 11, 10]
    engine._tp_rank = tp_rank
    engine._tp_size = 4
    engine._tfcfg = SimpleNamespace(sequence_parallel=sequence_parallel)
    engine._dims = SimpleNamespace(num_experts=num_experts)
    return engine


def _dense_routes(seq_slots: int = 7) -> torch.Tensor:
    tokens = torch.arange(seq_slots).view(1, seq_slots, 1, 1) * 1000
    layers = torch.arange(43).view(1, 1, 43, 1) * 10
    topk = torch.arange(2).view(1, 1, 1, 2)
    return (tokens + layers + topk).to(torch.int64)


@pytest.mark.parametrize(
    ("pp_rank", "expected_start", "expected_layers"),
    [(0, 0, 11), (1, 11, 11), (2, 22, 11), (3, 33, 10)],
)
def test_pp_route_extraction_uses_left_padding_and_asymmetric_layer_bounds(
    monkeypatch, pp_rank, expected_start, expected_layers
) -> None:
    _, instances = _install_router_replay(monkeypatch, expected_layers)
    engine = _engine(pp_rank=pp_rank)

    engine._r3_set_microbatch_routes(
        _dense_routes(), row=0, start=2, length=6, padded_length=8
    )

    assert len(instances) == expected_layers
    # Dense routes contain five routed positions, then one final-token filler,
    # then two alignment fillers.
    assert instances[0].target_indices[0].shape == (8, 2)
    assert instances[0].target_indices[0][:5].tolist() == [
        [2000 + expected_start * 10, 2001 + expected_start * 10],
        [3000 + expected_start * 10, 3001 + expected_start * 10],
        [4000 + expected_start * 10, 4001 + expected_start * 10],
        [5000 + expected_start * 10, 5001 + expected_start * 10],
        [6000 + expected_start * 10, 6001 + expected_start * 10],
    ]
    assert instances[0].target_indices[0][5:].tolist() == [
        [2 * (expected_start + offset), 500000 + 2 * (expected_start + offset)]
        for offset in range(3)
    ]


def test_ragged_route_extraction_does_not_apply_dense_left_pad(monkeypatch) -> None:
    _, instances = _install_router_replay(monkeypatch, 11)
    engine = _engine()
    ragged_row = _dense_routes(seq_slots=5)[0]

    engine._r3_set_microbatch_routes(
        [ragged_row], row=0, start=3, length=6, padded_length=8
    )

    assert instances[0].target_indices[0][:5].tolist() == [
        [0, 1],
        [1000, 1001],
        [2000, 2001],
        [3000, 3001],
        [4000, 4001],
    ]
    assert instances[0].target_indices[0][5:].tolist() == [
        [2 * offset, 500000 + 2 * offset] for offset in range(3)
    ]


def test_routes_pad_to_128_and_sequence_parallel_tp4_uses_contiguous_shard(
    monkeypatch,
) -> None:
    _, instances = _install_router_replay(monkeypatch, 11)
    engine = _engine(tp_rank=2, sequence_parallel=True)
    routes = _dense_routes(seq_slots=125)

    engine._r3_set_microbatch_routes(
        routes, row=0, start=0, length=126, padded_length=128
    )

    local = instances[0].target_indices[0]
    assert local.shape == (32, 2)
    assert local[0].tolist() == [64000, 64001]
    assert local[-1].tolist() == [95000, 95001]


def test_padding_fillers_cover_every_ep_partition(monkeypatch) -> None:
    _, instances = _install_router_replay(monkeypatch, 11)
    engine = _engine(num_experts=256)
    routes = torch.zeros(1, 1, 43, 6, dtype=torch.int64)

    engine._r3_set_microbatch_routes(routes, row=0, start=0, length=2, padded_length=4)

    filler = instances[0].target_indices[0][1:]
    assert filler.shape == (3, 6)
    for row in filler:
        assert torch.unique(row // 64).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("bad_id", [-1, 8])
def test_global_expert_ids_are_validated(monkeypatch, bad_id) -> None:
    _install_router_replay(monkeypatch, 11)
    engine = _engine(num_experts=8)
    routes = torch.zeros(1, 3, 43, 2, dtype=torch.int16)
    routes[0, 1, 42, 0] = bad_id

    with pytest.raises(ValueError, match="expert id"):
        engine._r3_set_microbatch_routes(
            routes, row=0, start=0, length=4, padded_length=4
        )


def test_local_replay_instance_count_must_match_local_layers(monkeypatch) -> None:
    _install_router_replay(monkeypatch, 10)
    engine = _engine(pp_rank=0)

    with pytest.raises(ValueError, match="RouterReplay instances"):
        engine._r3_set_microbatch_routes(
            _dense_routes(seq_slots=3),
            row=0,
            start=0,
            length=4,
            padded_length=4,
        )


def test_missing_replay_instances_is_a_configuration_error(monkeypatch) -> None:
    _install_router_replay(monkeypatch, 0)
    engine = _engine(pp_rank=3)

    with pytest.raises(RuntimeError, match="no RouterReplay instances"):
        engine._r3_set_microbatch_routes(
            _dense_routes(seq_slots=3),
            row=0,
            start=0,
            length=4,
            padded_length=4,
        )


def test_two_microbatches_append_fifo_without_clearing(monkeypatch) -> None:
    replay, instances = _install_router_replay(monkeypatch, 11)
    engine = _engine()
    routes = _dense_routes(seq_slots=5)

    engine._r3_set_microbatch_routes(routes, row=0, start=0, length=6, padded_length=8)
    engine._r3_set_microbatch_routes(routes, row=0, start=0, length=6, padded_length=8)

    assert replay.clear_indices_calls == 0
    assert replay.action == "replay_forward"
    assert all(len(instance.target_indices) == 2 for instance in instances)
    torch.testing.assert_close(
        instances[0].target_indices[0], instances[0].target_indices[1]
    )


def _install_pp_group(monkeypatch, group="pp-group"):
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.get_pipeline_model_parallel_group = lambda: group
    core = sys.modules.get("megatron.core", types.ModuleType("megatron.core"))
    core.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    return group


def test_native_recompute_diagnostics_aggregate_only_over_pp_group(
    monkeypatch,
) -> None:
    class _DiagnosticReplay:
        def __init__(self, compared, flips):
            self.compared = compared
            self.flips = flips

        def get_recompute_diagnostics(self):
            return self.compared, self.flips

        def reset_recompute_diagnostics(self):
            self.compared = 0
            self.flips = 0

    replay, instances = _install_router_replay(
        monkeypatch,
        2,
        instance_factory=lambda: _DiagnosticReplay(12, 1),
    )
    pp_group = _install_pp_group(monkeypatch)
    calls = []

    def all_reduce(tensor, group):
        calls.append(group)
        tensor.add_(
            torch.tensor(
                [20, 3],
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    engine = _engine()

    metrics = engine._r3_native_recompute_metrics()

    assert calls == [pp_group]
    assert metrics == {
        "moe/r3_recompute_ids": 44.0,
        "moe/r3_recompute_flips": 5.0,
        "moe/r3_recompute_flip_rate": 5.0 / 44.0,
    }
    engine._r3_reset_native_diagnostics()
    assert replay.clear_indices_calls == 0
    assert all(item.get_recompute_diagnostics() == (0, 0) for item in instances)


def test_pp_layer_coverage_reports_exact_43_without_tp_duplication(
    monkeypatch,
) -> None:
    pp_group = _install_pp_group(monkeypatch)
    calls = []

    def all_reduce(tensor, group):
        calls.append(group)
        tensor.add_(
            torch.cat(
                [
                    torch.zeros(
                        11,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                    torch.ones(
                        32,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                ]
            )
        )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    engine = _engine(pp_rank=0)
    engine._tfcfg.dsv4_mode = True
    engine._dims.num_layers = 43

    metrics = engine._r3_pp_coverage_metrics()

    assert calls == [pp_group]
    assert metrics == {
        "moe/r3_pp_missing_layers": 0.0,
        "moe/r3_pp_duplicate_layers": 0.0,
    }


@pytest.mark.parametrize(
    ("counts", "error"),
    [
        ([1] * 42 + [0], "missing"),
        ([2] + [1] * 42, "duplicate"),
    ],
)
def test_dsv4_pp_layer_coverage_fails_closed(monkeypatch, counts, error) -> None:
    _install_pp_group(monkeypatch)

    def all_reduce(tensor, group):
        tensor.copy_(
            torch.tensor(
                counts,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    engine = _engine(pp_rank=0)
    engine._tfcfg.dsv4_mode = True
    engine._dims.num_layers = 43

    with pytest.raises(RuntimeError, match=error):
        engine._r3_pp_coverage_metrics()


def _hash_acceptance_engine() -> MegatronEngine:
    engine = _engine(pp_rank=0, num_experts=32)
    tables = {}
    for layer in range(3):
        table = (
            torch.arange(32).view(32, 1) + torch.tensor([layer, layer + 3])
        ).remainder(32)
        tables[f"module.decoder.layers.{layer}.mlp.router.tid2eid"] = table
    engine.module = SimpleNamespace(
        named_parameters=lambda: iter(()),
        named_buffers=lambda: iter(tables.items()),
    )
    return engine


def _dense_hash_routes(
    engine: MegatronEngine,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    routes = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1] - 1,
        43,
        2,
        dtype=torch.int64,
    )
    tables = dict(engine.module.named_buffers())
    for row in range(input_ids.shape[0]):
        start, length = engine._real_block(attention_mask[row])
        tokens = input_ids[row, start : start + length - 1]
        for layer in range(3):
            table = tables[f"module.decoder.layers.{layer}.mlp.router.tid2eid"]
            routes[row, start : start + length - 1, layer] = table[tokens]
    return routes


def test_hash_router_metrics_respect_dense_left_padding() -> None:
    engine = _hash_acceptance_engine()
    input_ids = torch.tensor([[0, 0, 4, 5, 6], [0, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    routes = _dense_hash_routes(engine, input_ids, attention_mask)

    metrics = engine._r3_hash_metrics(routes, input_ids, attention_mask)

    assert metrics == {
        "moe/r3_hash_ids": 30.0,
        "moe/r3_hash_flips": 0.0,
        "moe/r3_hash_flip_rate": 0.0,
    }


def test_hash_router_metrics_support_ragged_rows_and_fail_on_flip() -> None:
    engine = _hash_acceptance_engine()
    input_ids = torch.tensor([[0, 0, 4, 5, 6], [0, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    dense = _dense_hash_routes(engine, input_ids, attention_mask)
    ragged = [dense[0, 2:4].clone(), dense[1, 1:4].clone()]

    clean = engine._r3_hash_metrics(ragged, input_ids, attention_mask)
    assert clean["moe/r3_hash_ids"] == 30.0
    ragged[1][0, 2, 0] += 1

    with pytest.raises(RuntimeError, match="hash router.*flip"):
        engine._r3_hash_metrics(ragged, input_ids, attention_mask)


def test_hash_router_metrics_collect_from_pp0(monkeypatch) -> None:
    pp_group = _install_pp_group(monkeypatch)
    engine = _hash_acceptance_engine()
    input_ids = torch.tensor([[0, 0, 4, 5, 6], [0, 7, 8, 9, 10]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    routes = _dense_hash_routes(engine, input_ids, attention_mask)
    reduced = []

    def all_reduce(totals, group):
        reduced.append((totals.tolist(), group))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    metrics = engine._r3_hash_metrics(routes, input_ids, attention_mask)

    assert reduced == [([30, 0], pp_group)]
    assert metrics == {
        "moe/r3_hash_ids": 30.0,
        "moe/r3_hash_flips": 0.0,
        "moe/r3_hash_flip_rate": 0.0,
    }


def test_hash_router_metrics_reach_last_pp_rank(monkeypatch) -> None:
    pp_group = _install_pp_group(monkeypatch)
    engine = _engine(pp_rank=3, num_experts=32)
    engine.module = SimpleNamespace(
        named_parameters=lambda: iter(()),
        named_buffers=lambda: iter(()),
    )
    input_ids = torch.tensor([[4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    routes = torch.zeros(1, 2, 43, 2, dtype=torch.int64)
    reduced = []

    def all_reduce(totals, group):
        reduced.append((totals.tolist(), group))
        totals.copy_(
            torch.tensor(
                [30, 0],
                dtype=totals.dtype,
                device=totals.device,
            )
        )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    metrics = engine._r3_hash_metrics(routes, input_ids, attention_mask)

    assert reduced == [([0, 0], pp_group)]
    assert metrics == {
        "moe/r3_hash_ids": 30.0,
        "moe/r3_hash_flips": 0.0,
        "moe/r3_hash_flip_rate": 0.0,
    }


def test_hash_router_flip_fails_only_after_pipeline_collective(monkeypatch) -> None:
    pp_group = _install_pp_group(monkeypatch)
    engine = _hash_acceptance_engine()
    input_ids = torch.tensor([[4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    routes = _dense_hash_routes(engine, input_ids, attention_mask)
    routes[0, 0, 0, 0] += 1
    reduced = []

    def all_reduce(totals, group):
        reduced.append((totals.tolist(), group))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(RuntimeError, match="hash router.*flip"):
        engine._r3_hash_metrics(routes, input_ids, attention_mask)

    assert reduced == [([12, 1], pp_group)]


def test_update_paths_call_collective_hash_metrics_on_all_stages() -> None:
    for method in (
        MegatronEngine._engine_update_policy_packed,
        MegatronEngine._engine_update_policy_pp,
    ):
        source = inspect.getsource(method)
        assert source.count("self._r3_hash_metrics(") == 1
        call = source.index("self._r3_hash_metrics(")
        assert "self._pp_rank" not in source[:call]


def test_microbatch_routes_transfer_complete_local_tensor_only_once(
    monkeypatch,
) -> None:
    class _ViewReplayInstance:
        def __init__(self) -> None:
            self.target_indices = []

        def set_target_indices(self, indices: torch.Tensor) -> None:
            self.target_indices.append(indices)

    _, instances = _install_router_replay(
        monkeypatch, 11, instance_factory=_ViewReplayInstance
    )
    engine = _engine()
    routes = torch.zeros(1, 5, 43, 2, dtype=torch.int16)
    transfers = []
    original_to = torch.Tensor.to

    def _count_to(tensor, *args, **kwargs):
        transfers.append((args, kwargs))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.Tensor, "to", _count_to)

    engine._r3_set_microbatch_routes(routes, row=0, start=0, length=6, padded_length=8)

    assert len(transfers) == 1
    targets = [instance.target_indices[0] for instance in instances]
    assert all(target.dtype == torch.int64 for target in targets)
    storage_ptr = targets[0].untyped_storage().data_ptr()
    assert all(target.untyped_storage().data_ptr() == storage_ptr for target in targets)


def test_forward_only_schedule_cleans_replay_state_on_exception(
    monkeypatch,
) -> None:
    replay, _ = _install_router_replay(monkeypatch, 11)
    replay.action = "stale"

    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.is_pipeline_last_stage = lambda: False
    schedules = types.ModuleType("megatron.core.pipeline_parallel.schedules")

    def _raising_schedule(**_kwargs):
        replay.action = "active"
        raise RuntimeError("schedule failed")

    schedules.get_forward_backward_func = lambda: _raising_schedule
    core = types.ModuleType("megatron.core")
    core.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(
        sys.modules, "megatron.core.pipeline_parallel.schedules", schedules
    )

    original_to = torch.Tensor.to

    def _cpu_cuda_to(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _cpu_cuda_to)

    engine = _engine()
    engine._r3_enabled = True
    engine._input_sequence_alignment = 4
    engine._attention_backend = "unfused"
    engine.module = SimpleNamespace(eval=lambda: None)
    engine._tfcfg.timers = "timers"

    class _Batch:
        ragged: typing.ClassVar = {}
        meta: typing.ClassVar = {}
        batch_size = 1

        def __init__(self):
            self.tensors = {
                "input_ids": torch.arange(4).view(1, 4),
                "rollout_routed_experts": torch.zeros(1, 3, 43, 2, dtype=torch.int16),
            }

        def __getitem__(self, key):
            return self.tensors[key]

    batch = _Batch()

    with pytest.raises(RuntimeError, match="schedule failed"):
        engine._engine_compute_log_probs_pp(
            batch.tensors["input_ids"],
            torch.ones(1, 4, dtype=torch.long),
            4,
            False,
            1.0,
            batch,
        )

    assert replay.action is None
    assert replay.clear_indices_calls == 2
    assert engine._tfcfg.timers == "timers"
