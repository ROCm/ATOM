# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# The swap placement policy: its plan, its statistics and its schedule.

import math
import sys
import types

import numpy as np
import pytest
from aiter_stub import atom_without_aiter

torch = pytest.importorskip("torch")

from atom.config import EPLBConfig

with atom_without_aiter() as _load:
    eplb = _load("atom.model_ops.eplb")
    eplb_swap = _load("atom.model_ops.eplb_swap")


def _skewed(layers, experts, seed):
    rng = np.random.default_rng(seed)
    pf = rng.lognormal(0.0, 1.2, size=(layers, experts)) * 1e4
    dec = pf * rng.lognormal(0.0, 0.5, size=(layers, experts)) * 0.05
    return (
        torch.from_numpy(np.round(pf).astype(np.int64)),
        torch.from_numpy(np.round(dec).astype(np.int64)),
    )


def _objective(pf, dec, rows, top_k, tile_rows, p2l_row, ranks):
    """The swap objective from first principles (not `eplb_swap.swap_terms`):
    per rank, the prefill share, the expected decode M-tiles summed over a
    Poisson pmf, and the decode share, each scaled to a mean of 1."""
    pf, dec = pf.tolist(), dec.tolist()
    n = len(pf)
    tiles = []
    for d in dec:
        lam = d / sum(dec) * rows * top_k
        e = sum(
            math.ceil(k / tile_rows)
            * math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
            for k in range(1, int(lam + 20 * math.sqrt(lam + 1) + 40))
        )
        tiles.append(e)
    per = n // ranks
    rank = [0] * n
    for slot, expert in enumerate(p2l_row.tolist()):
        rank[expert] = slot // per
    total = 0.0
    for w, term in zip(eplb_swap.SWAP_WEIGHTS, (pf, tiles, dec)):
        load = [0.0] * ranks
        for e in range(n):
            load[rank[e]] += term[e] / sum(term) * ranks
        total += w * (max(load) + eplb_swap.SWAP_SMOOTH * sum(x * x for x in load))
    return total


def test_a_plan_lowers_the_objective_and_stays_a_permutation():
    layers, experts, ranks = 2, 48, 8
    pf, dec = _skewed(layers, experts, seed=0)
    old = torch.arange(experts, dtype=torch.int32).repeat(layers, 1)
    p2l, l2p, logcnt, swaps = eplb_swap.swap_rebalance(
        pf, dec, 600.0, old, num_gpus=ranks, top_k=6, tile_rows=32
    )
    assert torch.equal(logcnt, torch.ones_like(logcnt))
    for layer in range(layers):
        assert sorted(p2l[layer].tolist()) == list(range(experts))
        slots = torch.arange(experts, dtype=torch.int32)
        assert torch.equal(l2p[layer, p2l[layer].long(), 0], slots)
        assert swaps[layer] > 0
        args = (pf[layer], dec[layer], 600.0, 6, 32)
        before = _objective(*args, old[layer], ranks)
        after = _objective(*args, p2l[layer], ranks)
        assert after < before


def test_a_moved_layer_is_ordered_cold_first_and_an_unmoved_one_is_kept():
    experts, ranks = 32, 4
    per = experts // ranks
    pf, dec = _skewed(1, experts, seed=1)
    flat = torch.full((1, experts), 1000, dtype=torch.int64)
    old = torch.randperm(experts).to(torch.int32).repeat(2, 1)
    p2l, _, _, swaps = eplb_swap.swap_rebalance(
        torch.cat([pf, flat]),
        torch.cat([dec, flat]),
        100.0,
        old,
        num_gpus=ranks,
        top_k=2,
        tile_rows=32,
    )
    assert swaps[0] > 0 and swaps[1] == 0
    assert torch.equal(p2l[1], old[1])
    for g in range(ranks):
        load = pf[0][p2l[0, g * per : (g + 1) * per].long()]
        assert torch.equal(load, load.sort().values)


def test_expected_tiles_is_the_poisson_tile_count():
    dec = np.array([1.0, 1.0, 2.0, 0.0])
    tiles = eplb_swap.expected_tiles(dec, rows=4.0, top_k=1, tile_rows=32)
    # A step of 4 rows: lambda 1, 1, 2, 0; below one tile it is P(X > 0).
    expect = 1.0 - np.exp(-np.array([1.0, 1.0, 2.0, 0.0]))
    assert np.allclose(tiles, expect, atol=1e-9)
    many = eplb_swap.expected_tiles(np.array([1.0]), rows=3200.0, top_k=1, tile_rows=32)
    assert abs(many[0] - 100.5) < 1.0


def _counting(placement):
    """A real monitor whose passes a started swap policy counts through
    ``placement`` (the live map)."""
    layers, slots = placement.shape
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=4)
    monitor.initialize(
        num_layers=layers, num_physical=slots, device=torch.device("cpu")
    )
    policy = eplb_swap.SwapPolicy()
    live = types.SimpleNamespace(
        physical_to_logical_map=placement,
        num_layers=layers,
        num_logical_experts=slots,
    )
    policy.start(monitor, live)
    return monitor, policy


def _pass(monitor, counts, dummy=False, rows=0, **kind):
    monitor.on_forward_start()
    monitor._cur_pass_count.copy_(torch.tensor(counts, dtype=torch.int32))
    monitor.on_forward_end(dummy, decode_rows=rows, **kind)


def test_cumulative_counts_fold_through_the_placement_and_sample_decode():
    placement = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    monitor, policy = _counting(placement)
    assert monitor._pass_observer == policy.observe_pass
    _pass(monitor, [[1, 2, 3, 4]], is_pure_prefill=True)
    assert policy.prefill.tolist() == [[2, 4, 1, 3]]
    for i in range(eplb_swap.DECODE_SAMPLE_EVERY):
        _pass(
            monitor,
            [[1, 0, 0, 0]],
            rows=10 + i,
            is_pure_prefill=False,
            is_pure_decode=True,
        )
    assert policy.decode.tolist() == [[0, 0, 1, 0]]
    # Only the sampled pass (the last of the eight) counts its rows.
    assert policy.window_rows == 10 + eplb_swap.DECODE_SAMPLE_EVERY - 1
    assert policy.window_samples == 1
    _pass(monitor, [[5, 5, 5, 5]], is_pure_prefill=False)  # mixed: nothing
    _pass(monitor, [[5, 5, 5, 5]], dummy=True, is_pure_prefill=True)  # dummy
    assert policy.prefill.tolist() == [[2, 4, 1, 3]]
    # A migration moves the live map in place; the next pass folds through it.
    placement.copy_(torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
    _pass(monitor, [[1, 0, 0, 0]], is_pure_prefill=True)
    assert policy.prefill.tolist() == [[3, 4, 1, 3]]


def _swap_manager(monkeypatch, prefill, decode, samples):
    monkeypatch.setattr(
        eplb, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy="swap",
    )
    layers, experts = prefill.shape
    mgr.live_metadata = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=layers,
        num_logical_experts=experts,
        num_physical_experts=experts,
        ep_size=4,
        ep_rank=0,
        device=torch.device("cpu"),
    )
    mgr._moe_layers = {i: types.SimpleNamespace(top_k=2) for i in range(layers)}
    mgr._ep_group = object()
    mgr.replica_spread = (4, 0)
    loads = {"value": (prefill, decode, 100 * samples, samples)}
    monkeypatch.setattr(mgr.policy, "group_counts", lambda monitor: loads["value"])
    migrated = []

    def _migrate(new_meta, layer_ids, *, rearrange_ms):
        migrated.append(list(layer_ids))
        return
        yield  # pragma: no cover

    mgr._migrate = _migrate
    return mgr, loads, migrated


def _drain(gen):
    for _ in gen:
        pass


def test_a_plan_migrates_the_layers_it_moves_and_counts(monkeypatch):
    pf, dec = _skewed(2, 32, seed=2)
    pf[1] = 1000
    dec[1] = 50
    mgr, _, migrated = _swap_manager(monkeypatch, pf, dec, samples=16)
    _drain(mgr._execute_rebalance())
    assert migrated == [[0]]
    assert mgr.rebalance_count == 1
    assert not mgr.rebalancing_stopped


def test_a_plan_without_new_prefill_load_is_skipped_and_not_counted(monkeypatch):
    pf, dec = _skewed(1, 32, seed=3)
    mgr, _, migrated = _swap_manager(monkeypatch, pf, dec, samples=16)
    mgr.policy.window_rows, mgr.policy.window_samples = 300, 1
    _drain(mgr._execute_rebalance())
    assert (mgr.policy.window_rows, mgr.policy.window_samples) == (0, 0)
    mgr.policy.window_rows, mgr.policy.window_samples = 300, 1
    _drain(mgr._execute_rebalance())  # the same picks again: the window stays
    assert (mgr.policy.window_rows, mgr.policy.window_samples) == (300, 1)
    assert len(migrated) == 1
    assert mgr.rebalance_count == 1
    assert not mgr.rebalancing_stopped


def test_a_plan_without_a_gain_stops_rebalancing(monkeypatch):
    flat = torch.full((1, 32), 1000, dtype=torch.int64)
    mgr, _, migrated = _swap_manager(monkeypatch, flat, flat // 20, samples=16)
    mgr.monitor.set_pass_observer(mgr.policy.observe_pass)
    mgr.policy.placement = mgr.live_metadata.physical_to_logical_map
    _drain(mgr._execute_rebalance())
    assert migrated == []
    assert mgr.rebalance_count == 0
    assert mgr.rebalancing_stopped
    assert mgr.monitor._pass_observer is None and mgr.policy.placement is None


def test_the_swap_policy_admits_no_redundant_experts():
    assert EPLBConfig.from_dict({"placement_policy": "swap"}).placement_policy == "swap"
    with pytest.raises(AssertionError):
        EPLBConfig.from_dict({"placement_policy": "swap", "num_redundant_experts": 8})


def _group_window(ranks, rows_of_rank, passes):
    """Four real monitors, each sampling `passes` pure-decode passes of its own
    rows, reduced as the group's all-reduce would: the sums of their payloads."""
    policies = []
    for rank in range(ranks):
        monitor, policy = _counting(torch.arange(8, dtype=torch.int32).view(1, 8))
        for _ in range(passes * eplb_swap.DECODE_SAMPLE_EVERY):
            _pass(
                monitor,
                [[0] * 8],
                rows=rows_of_rank(rank),
                is_pure_prefill=False,
                is_pure_decode=True,
            )
        policies.append(policy)
    return sum(p.window_rows for p in policies), sum(p.window_samples for p in policies)


@pytest.mark.parametrize(
    "dp_size, rows_of_rank, group_rows",
    [
        (4, lambda rank: 100 + 20 * rank, 100 + 120 + 140 + 160),  # DEP: own rows
        (1, lambda rank: 400, 400),  # TP + EP: every rank holds the same rows
    ],
)
def test_the_decode_rows_a_step_are_the_groups(
    monkeypatch, dp_size, rows_of_rank, group_rows
):
    pf, dec = _skewed(1, 32, seed=4)
    mgr, loads, _ = _swap_manager(monkeypatch, pf, dec, samples=16)
    mgr.replica_spread = (dp_size, 0)
    rows, samples = _group_window(4, rows_of_rank, passes=3)
    loads["value"] = (pf, dec, rows, samples)
    _drain(mgr._execute_rebalance())
    assert mgr.policy.decode_rows == pytest.approx(group_rows)
    # A window without a sample keeps the last operating point.
    loads["value"] = (pf * 2, dec, 0, 0)
    _drain(mgr._execute_rebalance())
    assert mgr.policy.decode_rows == pytest.approx(group_rows)


def test_a_plan_makes_no_more_swaps_than_its_budget():
    pf, dec = _skewed(1, 64, seed=5)
    old = torch.arange(64, dtype=torch.int32).repeat(1, 1)
    _, _, _, swaps = eplb_swap.swap_rebalance(
        pf, dec, 600.0, old, num_gpus=8, top_k=6, tile_rows=32, budget=2
    )
    assert swaps[0] == 2


def test_decode_load_alone_moves_experts():
    flat = torch.full((1, 32), 1000, dtype=torch.int64)
    _, dec = _skewed(1, 32, seed=6)
    old = torch.arange(32, dtype=torch.int32).repeat(1, 1)
    _, _, _, swaps = eplb_swap.swap_rebalance(
        flat, dec, 300.0, old, num_gpus=4, top_k=2, tile_rows=32
    )
    assert swaps[0] > 0


def test_counts_past_int32_are_reduced_as_int64(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=4)
    big = torch.tensor([3_000_000_000], dtype=torch.int64)
    group = types.SimpleNamespace(
        world_size=2, all_reduce=lambda t, ca_fp8_quant: t * 2
    )
    monitor.set_load_group(group)
    assert monitor.all_reduce_counts(big).tolist() == [6_000_000_000]


def _moe(layer_id):
    return types.SimpleNamespace(
        layer_id=layer_id, use_ep=True, w13_weight=0, w2_weight=0
    )


def test_the_swap_policy_places_the_layers_past_eplb_layer_end(monkeypatch):
    layers = [_moe(i) for i in range(4)]
    model = types.SimpleNamespace(eplb_layer_end=2, modules=lambda: layers)
    assert sorted(eplb.collect_ep_moe_layers(model)) == [0, 1]
    assert sorted(eplb.collect_ep_moe_layers(model, all_layers=True)) == [0, 1, 2, 3]
    asked = []

    def _collect(model, all_layers=False):
        asked.append(all_layers)
        raise StopIteration

    monkeypatch.setattr(eplb, "collect_ep_moe_layers", _collect)
    for policy in ("swap", "naive"):
        mgr = eplb.EPLBManager(
            enabled=True,
            monitor=eplb.ExpertLoadMonitor(enabled=True, window_size=2),
            rebalance_interval=2,
            rebalance_min_balancedness=2.0,
            rebalance_balancedness_agg="min",
            placement_policy=policy,
        )
        with pytest.raises(StopIteration):
            mgr._maybe_initialize_runtime(types.SimpleNamespace(model=model))
    assert asked == [True, False]


def test_the_forward_monitor_samples_pure_decode_passes_only(monkeypatch):
    seen = []
    monitor = types.SimpleNamespace(
        on_forward_start=lambda: None,
        on_forward_end=lambda dummy, prefill, decode, rows: seen.append(
            (prefill, decode, rows)
        ),
    )
    manager = types.SimpleNamespace(
        live_metadata=object(),
        monitor=monitor,
        on_forward_pass_end=lambda has_prefill, dp_any: None,
    )
    monkeypatch.setattr(eplb, "_MANAGER", manager)

    @eplb.with_eplb_forward_monitor
    def _forward(self, batch):
        return 1

    for prefill, decode in ((0, 8), (8, 0), (8, 8)):
        batch = types.SimpleNamespace(
            is_dummy_run=False,
            total_tokens_num_prefill=prefill,
            total_tokens_num_decode=decode,
        )
        _forward(object(), batch)
    assert seen == [(False, True, 8), (True, False, 0), (False, False, 8)]


@pytest.mark.parametrize(
    "dp_size, multi_dp, multi_tp, ok",
    [
        (8, False, False, True),
        (1, False, False, False),
        (8, True, False, False),
        (8, False, True, False),
    ],
)
def test_the_manager_hands_the_policy_its_deployment(
    monkeypatch, dp_size, multi_dp, multi_tp, ok
):
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(is_multinode_dp=multi_dp),
        is_multinode_tp=multi_tp,
    )
    monkeypatch.setattr(eplb, "get_current_atom_config", lambda: config)
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=eplb.ExpertLoadMonitor(enabled=True, window_size=2),
        rebalance_interval=2,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy="swap",
    )
    mgr.replica_spread = (dp_size, 0)
    if ok:
        mgr._check_policy_deployment(384, 384, 8)
    else:
        with pytest.raises(RuntimeError, match="one node with DP attention"):
            mgr._check_policy_deployment(384, 384, 8)


def test_a_policy_is_made_by_name_and_naive_replans_every_layer():
    assert isinstance(eplb.make_placement_policy("swap"), eplb_swap.SwapPolicy)
    naive = eplb.make_placement_policy(None)
    assert isinstance(naive, eplb.RebalancePolicy) and naive.gated
    assert not naive.places_all_layers
    with pytest.raises(ValueError):
        eplb.make_placement_policy("nope")
    live = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=2,
        num_logical_experts=8,
        num_physical_experts=8,
        ep_size=2,
        ep_rank=0,
        device=torch.device("cpu"),
    )
    load = torch.tensor([[8, 8, 1, 1, 1, 1, 1, 1], [1] * 8])
    monitor = types.SimpleNamespace(dump_global_logical_load=lambda: load)
    ctx = eplb.PlanContext(
        live=live,
        moe_layers={
            0: types.SimpleNamespace(num_expert_group=None),
            1: types.SimpleNamespace(num_expert_group=None),
        },
        num_nodes=1,
        dp_size=2,
    )
    plan = naive.plan(monitor, ctx)
    assert plan.layer_ids == [0, 1] and plan.logical_load is load
    # The trivial placement holds both hot experts on rank 0; the plan splits them.
    hot = plan.physical_to_logical[0].tolist()
    assert {hot.index(0) // 4, hot.index(1) // 4} == {0, 1}
    monitor = types.SimpleNamespace(dump_global_logical_load=lambda: None)
    assert naive.plan(monitor, ctx) is None


def test_the_swap_policy_counts_from_the_live_placement_and_skips_the_gate(
    monkeypatch,
):
    pf, dec = _skewed(1, 32, seed=7)
    mgr, _, migrated = _swap_manager(monkeypatch, pf, dec, samples=16)
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.initialize(num_layers=1, num_physical=32, device=torch.device("cpu"))
    mgr.policy.start(monitor, mgr.live_metadata)
    assert mgr.policy.placement is mgr.live_metadata.physical_to_logical_map
    assert monitor._pass_observer == mgr.policy.observe_pass

    def _gate():
        raise AssertionError("the swap policy has no balancedness gate")

    monkeypatch.setattr(mgr.monitor, "dump_global_physical_load", _gate)
    _drain(mgr._rebalance())
    assert migrated == [[0]]


@pytest.mark.parametrize("world_size", [1, 2])
def test_the_cumulative_counts_reduce_in_one_payload(world_size):
    monitor, policy = _counting(
        torch.tensor([[1, 0, 3, 2], [0, 1, 2, 3]], dtype=torch.int32)
    )
    monitor.set_load_group(
        types.SimpleNamespace(
            world_size=world_size, all_reduce=lambda t, ca_fp8_quant: t * 2
        )
    )
    _pass(monitor, [[1, 2, 3, 4], [5, 6, 7, 8]], is_pure_prefill=True)
    for _ in range(eplb_swap.DECODE_SAMPLE_EVERY):
        _pass(
            monitor,
            [[1, 0, 0, 0], [0, 0, 0, 2]],
            rows=7,
            is_pure_prefill=False,
            is_pure_decode=True,
        )
    prefill, decode, rows, samples = policy.group_counts(monitor)
    k = world_size
    assert prefill.tolist() == [[2 * k, k, 4 * k, 3 * k], [5 * k, 6 * k, 7 * k, 8 * k]]
    assert decode.tolist() == [[0, k, 0, 0], [0, 0, 0, 2 * k]]
    assert rows == 7 * k and samples == k


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_logical": 385, "num_physical": 392}, "7 redundant"),
        ({"dp_size": 1}, "dp_size=1"),
        ({"multi_node": True}, "multi_node=True"),
    ],
)
def test_the_swap_policy_runs_on_one_node_of_dep(kwargs, match):
    args = {"num_logical": 384, "num_physical": 384, "ep_size": 8, "dp_size": 8}
    args["multi_node"] = False
    args.update(kwargs)
    with pytest.raises(RuntimeError, match=match):
        eplb_swap.SwapPolicy().check_deployment(**args)
    ok = {"num_logical": 384, "num_physical": 384, "ep_size": 8, "dp_size": 8}
    eplb_swap.SwapPolicy().check_deployment(multi_node=False, **ok)
    eplb.RebalancePolicy(None).check_deployment(
        num_logical=384, num_physical=392, ep_size=8, dp_size=1, multi_node=True
    )


def test_after_a_migration_the_counts_fold_through_the_new_placement(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.initialize(num_layers=1, num_physical=4, device=torch.device("cpu"))
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy="swap",
    )
    mgr.live_metadata = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=1,
        num_logical_experts=4,
        num_physical_experts=4,
        ep_size=2,
        ep_rank=0,
        device=torch.device("cpu"),
    )
    mgr.policy.start(monitor, mgr.live_metadata)
    new = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32)

    def _commit(*, layer_ids, live_meta, **_):
        for layer in layer_ids:
            live_meta.physical_to_logical_map[layer].copy_(new[layer])

    monkeypatch.setattr(eplb, "migrate_and_commit_chunk", _commit)
    _drain(mgr._migrate(None, [0], rearrange_ms=0.0))
    _pass(monitor, [[1, 2, 3, 4]], is_pure_prefill=True)
    assert mgr.policy.prefill.tolist() == [[4, 3, 2, 1]]


def test_a_plan_needs_a_permutation_placement():
    pf, dec = _skewed(1, 8, seed=8)
    old = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    with pytest.raises(ValueError, match="permutation"):
        eplb_swap.swap_rebalance(pf, dec, 10.0, old, num_gpus=2, top_k=2, tile_rows=32)


def test_the_manager_refuses_a_swap_deployment_at_init(monkeypatch):
    def _group(ranks, rank=0):
        return types.SimpleNamespace(
            device_group=ranks, world_size=len(ranks), rank_in_group=rank
        )

    groups = types.ModuleType("aiter.dist.parallel_state")
    groups.get_ep_group = lambda: _group([0, 1])
    groups.get_dp_group = lambda: _group([0])  # TP + EP: no DP attention
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", groups)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", list)
    monkeypatch.setattr(torch.distributed, "new_group", lambda ranks: object())
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(is_multinode_dp=False),
        is_multinode_tp=False,
    )
    monkeypatch.setattr(eplb, "get_current_atom_config", lambda: config)
    layers = [
        types.SimpleNamespace(
            layer_id=i,
            use_ep=True,
            w13_weight=torch.zeros(2, 4),
            w2_weight=torch.zeros(2, 4),
            global_num_experts=4,
            num_redundant_experts=0,
            ep_size=2,
            ep_rank=0,
        )
        for i in range(2)
    ]
    model = types.SimpleNamespace(modules=lambda: layers)
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=eplb.ExpertLoadMonitor(enabled=True, window_size=2),
        rebalance_interval=2,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy="swap",
    )
    started = []
    monkeypatch.setattr(mgr.policy, "start", lambda *args: started.append(args))
    with pytest.raises(RuntimeError, match="one node with DP attention"):
        mgr.bind_runtime_owner(types.SimpleNamespace(model=model))
    assert started == []
