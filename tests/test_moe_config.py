# SPDX-License-Identifier: MIT
"""CPU tests for topology and row-transport ownership across MoE entry points."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from conftest import atom_config_double

from atom.config import Config
from atom.moe_config import (
    MoEBackendConfig,
    MoEParallelMapping,
    get_moe_backend_config,
    get_moe_execution_plan,
    resolve_moe_execution_plan,
)


def _mapping(**kwargs):
    args = {
        "tp_size": 1,
        "dp_size": 8,
        "local_dp_size": 8,
        "enable_ep": True,
        "flatten_dp": True,
    }
    args.update(kwargs)
    return MoEParallelMapping.make(**args)


@pytest.mark.parametrize(
    "kwargs, tp, ep, rank, local_ep",
    [
        ({"tp_size": 1, "flatten_dp": False}, 1, 1, 0, 1),
        ({"tp_size": 4, "dp_rank": 1, "tp_rank": 2}, 1, 32, 6, 32),
        ({"tp_size": 4, "flatten_dp": False, "tp_rank": 2}, 1, 4, 2, 4),
        ({"tp_size": 4, "enable_ep": False, "dp_rank": 1, "tp_rank": 2}, 32, 1, 6, 1),
        ({"pcp_size": 2, "pcp_rank": 1, "merge_pcp": True, "dp_rank": 3}, 1, 16, 7, 16),
        (
            {"pcp_size": 2, "pcp_rank": 1, "merge_pcp": True, "flatten_dp": False},
            1,
            2,
            1,
            2,
        ),
    ],
)
def test_expert_ownership_uses_final_sharding_width(kwargs, tp, ep, rank, local_ep):
    mapping = _mapping(**kwargs)
    assert (mapping.tp_size, mapping.ep_size) == (tp, ep)
    assert (mapping.ep_rank if mapping.use_ep else mapping.tp_rank) == rank
    assert mapping.local_ep_size == local_ep
    assert mapping.use_ep is (ep > 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp_rank": 1},
        {"dp_rank": 8},
        {"pcp_size": 0},
        {"local_dp_size": 9},
        {"dp_logical_size": 10},
    ],
)
def test_invalid_topology_is_refused_before_weight_allocation(kwargs):
    with pytest.raises(ValueError):
        _mapping(**kwargs)


@pytest.mark.parametrize(
    "requested, available, disabled, v2, transport",
    [
        ("auto", True, False, False, "mori"),
        ("auto", True, False, True, "mori_v2"),
        ("auto", False, False, False, "dp_gather_scatter"),
        ("mori", True, True, False, "dp_gather_scatter"),
        ("none", True, False, True, "dp_gather_scatter"),
        ("rccl", False, True, False, "rccl"),
    ],
)
def test_standard_transport_selection(requested, available, disabled, v2, transport):
    plan = resolve_moe_execution_plan(
        _mapping(),
        MoEBackendConfig(
            all2all_backend=requested,
            mori_available=available,
            disable_mori=disabled,
            mori_v2=v2,
            fp4_dispatch=True,
        ),
    )
    assert plan.transport == transport
    assert plan.use_fp4_dispatch is (transport == "mori")
    assert plan.needs_prepare_finalize is (transport in {"mori", "mori_v2", "rccl"})
    assert plan.needs_dp_gather_scatter is (transport == "dp_gather_scatter")


def test_pure_dp_keeps_expert_replicas_even_when_mori_is_installed():
    for tp in (1, 4):
        mapping = _mapping(tp_size=tp, flatten_dp=False)
        plan = resolve_moe_execution_plan(
            mapping, MoEBackendConfig(mori_available=True)
        )
        assert plan.transport == "local"
        assert not plan.uses_routed_dispatch
        assert not plan.needs_dp_gather_scatter


def test_simulated_dp_cannot_use_real_alltoall_peers():
    mapping = _mapping(dp_size=2, local_dp_size=2, dp_logical_size=16, dp_rank=1)
    assert (mapping.ep_size, mapping.ep_rank, mapping.dp_logical_ratio) == (16, 1, 8)
    for request in ("auto", "mori", "rccl"):
        plan = resolve_moe_execution_plan(
            mapping, MoEBackendConfig(all2all_backend=request, mori_available=True)
        )
        assert plan.transport == "dp_gather_scatter"
        assert not plan.supports_tbo


def test_dp_pcp_tp_mapping_matches_dispatch_group_rank_order():
    ranks = []
    for dp_rank in range(2):
        for pcp_rank in range(2):
            for tp_rank in range(4):
                mapping = _mapping(
                    tp_size=4,
                    tp_rank=tp_rank,
                    dp_size=2,
                    local_dp_size=1,
                    dp_rank=dp_rank,
                    pcp_size=2,
                    pcp_rank=pcp_rank,
                    merge_pcp=True,
                )
                assert mapping.ep_size == 16
                assert mapping.local_ep_size == 8
                ranks.append(mapping.ep_rank)
    assert ranks == list(range(16))


@pytest.mark.parametrize("requested", ["auto", "mori", "none"])
def test_mega_owns_transport_independently_of_standard_mori_switches(requested):
    plan = resolve_moe_execution_plan(
        _mapping(),
        MoEBackendConfig(
            backend="mega",
            all2all_backend=requested,
            disable_mori=True,
            fp4_dispatch=True,
        ),
    )
    assert plan.implementation == "mega"
    assert plan.owns_dispatch_combine and plan.uses_routed_dispatch
    assert not plan.needs_prepare_finalize
    assert not plan.needs_dp_gather_scatter
    assert not plan.use_fp4_dispatch


@pytest.mark.parametrize(
    "mapping, backend, reason",
    [
        (_mapping(flatten_dp=False), MoEBackendConfig(backend="mega"), "final EP"),
        (
            _mapping(dp_logical_size=16),
            MoEBackendConfig(backend="mega"),
            "real EP peers",
        ),
        (
            _mapping(tp_size=4, flatten_dp=False),
            MoEBackendConfig(backend="mega"),
            "DP ranks",
        ),
        (_mapping(), MoEBackendConfig(backend="mega", all2all_backend="rccl"), "RCCL"),
    ],
)
def test_mega_refuses_incompatible_mapping(mapping, backend, reason):
    with pytest.raises(ValueError, match=reason):
        resolve_moe_execution_plan(mapping, backend)


def test_sp_mega_keeps_native_weight_mapping_and_local_fallback():
    mapping = _mapping(
        dp_size=1, local_dp_size=1, enable_ep=False, pcp_size=4, merge_pcp=True
    )
    plan = resolve_moe_execution_plan(
        mapping, MoEBackendConfig(backend="mega"), sequence_parallel_size=4
    )
    assert (mapping.tp_size, mapping.ep_size) == (4, 1)
    assert plan.implementation == "sp_mega"
    assert plan.transport == "local"
    assert not plan.needs_prepare_finalize


def test_backend_settings_are_a_snapshot_and_plans_are_immutable(monkeypatch):
    import atom.moe_config as module

    monkeypatch.setattr(module, "find_spec", lambda _: object())
    monkeypatch.setenv("ATOM_DISABLE_MORI_EP", "0")
    config = atom_config_double(moe_backend=" STANDARD ", moe_all2all_backend=" MORI ")
    Config._init_moe_backend(config)
    snapshot = get_moe_backend_config(config)
    assert (config.moe_backend, config.moe_all2all_backend) == ("standard", "mori")
    monkeypatch.setenv("ATOM_DISABLE_MORI_EP", "1")
    monkeypatch.setattr(module, "find_spec", lambda _: None)
    assert get_moe_backend_config(config) is snapshot
    plan = resolve_moe_execution_plan(_mapping(), snapshot)
    assert plan.transport == "mori"
    with pytest.raises(FrozenInstanceError):
        plan.transport = "local"


def test_saved_dp_attention_plan_survives_runtime_field_updates():
    config = atom_config_double(
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_size_local=1
        ),
        enable_dp_attention=True,
        enable_expert_parallel=True,
        moe_backend_config=MoEBackendConfig(mori_available=True),
    )
    before = get_moe_execution_plan(config)
    config.tensor_parallel_size = 1
    config.parallel_config.data_parallel_size = 8
    config.parallel_config.data_parallel_size_local = 8
    after = get_moe_execution_plan(config)
    assert before is after
    assert after.mapping.ep_size == 8 and after.transport == "mori"


@pytest.mark.parametrize("requested", ["auto", "none", "rccl"])
def test_tbo_guard_consumes_same_transport_plan(requested):
    config = atom_config_double(
        parallel_config=SimpleNamespace(
            data_parallel_size=8, data_parallel_size_local=8
        ),
        enable_dp_attention=True,
        enable_expert_parallel=True,
        enable_tbo=True,
        enable_tbo_decode=True,
        moe_backend_config=MoEBackendConfig(
            all2all_backend=requested, mori_available=True
        ),
    )
    plan = get_moe_execution_plan(config)
    Config._resolve_tbo(config)
    assert config.enable_tbo is plan.supports_tbo
    assert config.enable_tbo_decode is plan.supports_tbo


@pytest.mark.parametrize("ep, dp_attention", [(False, True), (True, False)])
def test_collective_tbo_guard_also_covers_tp_and_plugin_flattening(ep, dp_attention):
    config = atom_config_double(
        parallel_config=SimpleNamespace(
            data_parallel_size=8, data_parallel_size_local=8
        ),
        enable_dp_attention=dp_attention,
        moe_ep_flatten_tp_across_dp=True,
        enable_expert_parallel=ep,
        enable_tbo=True,
        enable_tbo_decode=True,
        moe_backend_config=MoEBackendConfig(all2all_backend="none"),
    )
    assert get_moe_execution_plan(config).needs_dp_gather_scatter
    Config._resolve_tbo(config)
    assert not config.enable_tbo and not config.enable_tbo_decode


@pytest.mark.parametrize(
    "overrides", [{"backend": "unknown"}, {"all2all_backend": "unknown"}]
)
def test_unknown_backend_requests_fail_at_entry_boundary(overrides):
    with pytest.raises(ValueError):
        MoEBackendConfig(**overrides)
