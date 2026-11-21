# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the history and runtime wiring of DeepInfer replica selection."""

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.model_ops import eplb
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires ATOM's Python dependencies")


def _metadata():
    return eplb.ExpertLocationMetadata.from_trivial(
        num_layers=1,
        num_logical_experts=3,
        num_physical_experts=4,
        ep_size=2,
        ep_rank=0,
        device=torch.device("cpu"),
    )


def _monitor(meta, window_size=4):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=window_size)
    monitor.initialize_for_metadata(meta)
    monitor.initialize_logical_history(meta)
    monitor.set_load_group(SimpleNamespace(world_size=1))
    return monitor


def _record(monitor, meta, counts, *, include_current=True):
    monitor.on_forward_start()
    monitor._cur_pass_count.copy_(torch.tensor([counts], dtype=torch.int32))
    monitor.record_logical_load(meta, include_current=include_current)


def _history():
    # Expert 1 has the largest total, but expert 0's burst is the first
    # bottleneck whose replication improves the per-step objective most.
    return torch.tensor([[[100, 60, 0]], [[0, 60, 80]]], dtype=torch.int32)


def _rebalance(history, *, policy="deepinfer", **overrides):
    options = {
        "num_physical": 4,
        "num_groups": 1,
        "num_nodes": 1,
        "num_gpus": 2,
        "enable_hierarchical": False,
        "policy": eplb.resolve_placement_policy(policy),
        "load_history": history,
    }
    options.update(overrides)
    return eplb.rebalance_experts(history.sum(0), **options)


def _manager(monitor, meta, *, policy="deepinfer"):
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=100,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
        placement_policy=policy,
    )
    manager.live_metadata = meta
    manager._dp_is_migration_group = True
    return manager


def test_deepinfer_is_selectable_through_eplb_config():
    from atom.config import EPLBConfig

    config = EPLBConfig.from_dict(
        {"placement_policy": " DeepInfer ", "num_redundant_experts": 8}
    )
    assert config.placement_policy == "deepinfer"
    assert config.num_redundant_experts == 8
    assert EPLBConfig().placement_policy == "naive"


def test_history_excludes_unfilled_rows():
    meta = _metadata()
    monitor = _monitor(meta, window_size=4)
    assert monitor.dump_global_logical_load_history() is None
    _record(monitor, meta, [2, 3, 4, 5])
    history = monitor.dump_global_logical_load_history()
    assert history.shape == (1, 1, 3)
    assert history.tolist() == [[[7, 3, 4]]]


def test_dp_ranks_keep_matching_steps_when_only_one_has_prefill():
    meta = _metadata()
    monitors = [_monitor(meta), _monitor(meta)]
    managers = [_manager(monitor, meta) for monitor in monitors]
    # Each rank has one valid local batch, but on different global steps.
    # The other rank's decoder/dummy counts must not enter that step.
    for valid_rank, counts in [(0, [100, 60, 0, 0]), (1, [0, 60, 80, 0])]:
        for rank, (monitor, manager) in enumerate(zip(monitors, managers)):
            monitor.on_forward_start()
            monitor._cur_pass_count.copy_(
                torch.tensor([counts if rank == valid_rank else [999] * 4])
            )
            manager.on_forward_pass_end(
                local_has_prefill=rank == valid_rank,
                dp_any_has_prefill=True,
                record_load=rank == valid_rank,
            )
    first, second = [m.dump_global_logical_load_history() for m in monitors]
    assert first.shape == second.shape == (2, 1, 3)
    assert first[1].count_nonzero() == 0
    assert second[0].count_nonzero() == 0
    assert torch.equal(first + second, _history().to(first.dtype))


def test_history_keeps_logical_identity_across_migration():
    meta = _metadata()
    monitor = _monitor(meta)
    _record(monitor, meta, [10, 20, 30, 40])
    # The same physical slot changes expert after a migration. Old history
    # must retain the original logical attribution, including replica folds.
    meta.physical_to_logical_map.copy_(torch.tensor([[1, 0, 2, 1]]))
    _record(monitor, meta, [1, 2, 3, 4])
    assert monitor.dump_global_logical_load_history().tolist() == [
        [[50, 20, 30]],
        [[2, 5, 3]],
    ]


def test_history_ring_replaces_oldest_step():
    meta = _metadata()
    monitor = _monitor(meta, window_size=2)
    for value in [1, 2, 3]:
        _record(monitor, meta, [value, 0, 0, 0])
    history = monitor.dump_global_logical_load_history()
    assert history.shape == (2, 1, 3)
    assert sorted(history[:, 0, 0].tolist()) == [2, 3]
    assert history[:, :, 1:].count_nonzero() == 0


def test_history_snapshot_does_not_alias_live_window():
    meta = _metadata()
    monitor = _monitor(meta, window_size=1)
    _record(monitor, meta, [10, 20, 30, 40])
    snapshot = monitor.dump_global_logical_load_history()
    snapshot.zero_()
    assert monitor.dump_global_logical_load_history().tolist() == [[[50, 20, 30]]]
    snapshot = monitor.dump_global_logical_load_history()
    _record(monitor, meta, [1, 2, 3, 4])
    assert snapshot.tolist() == [[[50, 20, 30]]]


def test_history_collective_reduces_integer_snapshot(monkeypatch):
    meta = _metadata()
    monitor = _monitor(meta)
    _record(monitor, meta, [1, 2, 3, 4])
    group = object()
    monitor.set_load_group(SimpleNamespace(world_size=2, device_group=group))
    calls = []

    def all_reduce(tensor, *, op, group):
        calls.append((tensor.dtype, op, group))
        tensor.add_(torch.tensor([[[10, 20, 30]]]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert monitor.dump_global_logical_load_history().tolist() == [[[15, 22, 33]]]
    assert calls == [(torch.int64, torch.distributed.ReduceOp.SUM, group)]
    monitor.set_load_group(SimpleNamespace(world_size=1))
    assert monitor.dump_global_logical_load_history().tolist() == [[[5, 2, 3]]]


def test_deepinfer_flat_uses_per_step_replica_selection():
    p2l, l2p, counts = _rebalance(_history())
    assert counts.tolist() == [[2, 1, 1]]
    for expert in range(3):
        slots = l2p[0, expert]
        slots = slots[slots >= 0].long()
        assert slots.numel() == counts[0, expert]
        assert torch.all(p2l[0, slots] == expert)
    _, _, naive_counts = _rebalance(_history(), policy="naive")
    assert naive_counts.tolist() == [[1, 2, 1]]


def test_deepinfer_hierarchical_slices_history_by_node():
    history = torch.cat((_history(), _history()), dim=-1)
    p2l, _, counts = _rebalance(
        history,
        num_physical=8,
        num_groups=2,
        num_nodes=2,
        num_gpus=4,
        enable_hierarchical=True,
    )
    assert counts.tolist() == [[2, 1, 1, 2, 1, 1]]
    nodes = p2l.view(2, 4)
    assert {frozenset(row.tolist()) for row in nodes} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
    }


@pytest.mark.parametrize("policy", ["naive", "biased"])
def test_existing_placement_policies_ignore_optional_history(policy):
    history = _history()
    expected = _rebalance(history, policy=policy, load_history=None)
    actual = _rebalance(history, policy=policy, load_history=history.flip(-1))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize(
    "history",
    [None, torch.empty(0, 1, 3), torch.ones(2, 2, 3), torch.ones(1, 3)],
)
def test_deepinfer_rejects_missing_or_incompatible_history(history):
    with pytest.raises(AssertionError, match="load_history"):
        _rebalance(_history(), load_history=history)


def test_manager_does_not_record_globally_decode_only_step():
    meta = _metadata()
    monitor = _monitor(meta)
    manager = _manager(monitor, meta)
    monitor._cur_pass_count.fill_(99)
    manager.on_forward_pass_end(
        local_has_prefill=False,
        dp_any_has_prefill=False,
        record_load=False,
    )
    assert monitor.dump_global_logical_load_history() is None


def test_manager_commits_history_before_advancing_migration():
    meta = _metadata()
    monitor = _monitor(meta)
    manager = _manager(monitor, meta)
    monitor._cur_pass_count.copy_(torch.tensor([[1, 2, 3, 4]]))

    def migration_step():
        # Models the mapping switch a rebalance can perform at next(_gen).
        assert monitor.dump_global_logical_load_history().tolist() == [[[5, 2, 3]]]
        meta.physical_to_logical_map.copy_(torch.tensor([[1, 0, 2, 1]]))
        yield

    manager._gen = migration_step()
    manager.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=True)
    assert monitor.dump_global_logical_load_history().tolist() == [[[5, 2, 3]]]


@pytest.mark.parametrize("policy", ["naive", "biased"])
def test_existing_policy_does_not_collect_logical_history(monkeypatch, policy):
    meta = _metadata()
    monitor = _monitor(meta)
    manager = _manager(monitor, meta, policy=policy)

    def unexpected_record(*args, **kwargs):
        raise AssertionError("old policies must not incur logical-history work")

    monkeypatch.setattr(monitor, "record_logical_load", unexpected_record)
    manager.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=True)


@pytest.mark.parametrize("policy", ["deepinfer", "naive", "biased"])
def test_runtime_passes_correct_statistics_to_policy(monkeypatch, policy):
    meta = _metadata()
    monitor = _monitor(meta)
    _record(monitor, meta, [100, 60, 0, 0])
    _record(monitor, meta, [0, 60, 80, 0])
    manager = _manager(monitor, meta, policy=policy)
    manager._ep_group = object()
    manager._moe_layers = {0: SimpleNamespace(num_expert_group=1)}
    aggregate = _history().sum(0)
    captured = {}

    class PolicyReached(Exception):
        pass

    def rebalance(weight, **kwargs):
        captured.update(weight=weight, **kwargs)
        raise PolicyReached

    def unexpected_dump():
        raise AssertionError("runtime used the wrong statistics source")

    if policy == "deepinfer":
        monkeypatch.setattr(monitor, "dump_global_logical_load", unexpected_dump)
    else:
        monkeypatch.setattr(monitor, "dump_global_logical_load", lambda: aggregate)
        monkeypatch.setattr(
            monitor, "dump_global_logical_load_history", unexpected_dump
        )
    monkeypatch.setattr(eplb, "rebalance_experts", rebalance)
    with pytest.raises(PolicyReached):
        next(manager._execute_runtime_rebalance())
    assert torch.equal(captured["weight"], aggregate)
    assert captured["policy"] is eplb.resolve_placement_policy(policy)
    if policy == "deepinfer":
        assert torch.equal(captured["load_history"], _history())
    else:
        assert captured.get("load_history") is None


@pytest.mark.parametrize(
    "prefill,decode,dummy,expected",
    [
        (4, 0, False, True),
        (4, 0, True, False),
        (4, 2, False, False),
        (0, 2, False, False),
    ],
)
def test_forward_wrapper_passes_history_validity(
    monkeypatch, prefill, decode, dummy, expected
):
    meta = _metadata()
    monitor = _monitor(meta)
    seen = []
    manager = SimpleNamespace(
        live_metadata=meta,
        monitor=monitor,
        on_forward_pass_end=lambda *args, **kwargs: seen.append(kwargs["record_load"]),
    )
    monkeypatch.setattr(eplb, "_MANAGER", manager)

    @eplb.with_eplb_forward_monitor
    def forward(owner, batch):
        return "ok"

    batch = SimpleNamespace(
        total_tokens_num_prefill=prefill,
        total_tokens_num_decode=decode,
        is_dummy_run=dummy,
    )
    assert forward(SimpleNamespace(_eplb_any_rank_has_prefill=True), batch) == "ok"
    assert seen == [expected]
