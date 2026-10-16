# SPDX-License-Identifier: MIT
# Tests for the EPLB decode expert swap (atom/model_ops/eplb.py)

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")


class _FakeTPGroup:
    world_size = 1


def _gpu_cost(cost, num_gpus):
    return cost.view(cost.shape[0], num_gpus, -1).sum(-1)


def _policy(num_gpus=2, max_swaps=8, min_gain=0.0):
    return eplb.DecodeSwapPolicy(
        "greedy_hot_cold",
        num_gpus=num_gpus,
        max_swaps_per_layer=max_swaps,
        min_gain=min_gain,
    )


def _plan(cost, p2l, num_gpus=2, max_swaps=8, min_gain=0.0):
    cost = torch.as_tensor(cost, dtype=torch.float32)
    p2l = torch.as_tensor(p2l, dtype=torch.int32)
    swaps = _policy(num_gpus, max_swaps, min_gain).plan(cost, p2l)
    per_layer = [sum(s.layer == layer for s in swaps) for layer in range(len(p2l))]
    return eplb.apply_swaps(p2l, swaps), eplb.apply_swaps(cost, swaps), per_layer


def test_swap_describes_slots_and_experts():
    swaps = _policy().plan(
        torch.tensor([[10.0, 9.0, 1.0, 0.0]]), torch.tensor([[5, 6, 7, 8]])
    )
    assert swaps == [
        eplb.ExpertSwap(layer=0, slot_a=0, slot_b=2, expert_a=5, expert_b=7)
    ]
    assert eplb.apply_swaps(torch.tensor([[5, 6, 7, 8]]), swaps).tolist() == [
        [7, 6, 5, 8]
    ]


def test_swap_moves_load_to_cold_gpu():
    # gpu0 = 10+9, gpu1 = 1+0 -> swapping 10<->1 gives 10/10.
    p2l, cost, swaps = _plan([[10, 9, 1, 0]], [[0, 1, 2, 3]])
    per = _gpu_cost(cost, 2)[0]
    assert swaps == [1]
    assert per.tolist() == [10, 10]
    assert sorted(p2l[0].tolist()) == [0, 1, 2, 3]
    # cost follows the logical expert that moved
    w = {0: 10, 1: 9, 2: 1, 3: 0}
    assert [w[e] for e in p2l[0].tolist()] == cost[0].tolist()


def test_balanced_layer_is_untouched():
    p2l, _, swaps = _plan([[5, 5, 5, 5], [0, 0, 0, 0]], [[0, 1, 2, 3]] * 2)
    assert swaps == [0, 0]
    assert p2l.tolist() == [[0, 1, 2, 3]] * 2


def test_activation_estimate():
    act = eplb.DecodeSwapPolicy.activation(torch.tensor([[0, 10, 1000]]), steps=10)
    assert act[0, 0] == 0.0
    assert act[0, 1] == pytest.approx(1 - 2.718281828**-1)
    assert act[0, 2] == pytest.approx(1.0)


def test_balanced_layers_are_skipped_by_threshold():
    # layer 0: 19 vs 1 (bal 0.53); layer 1: 19 vs 17 (bal 0.95).
    cost = torch.tensor([[10.0, 9.0, 1.0, 0.0], [10.0, 9.0, 9.0, 8.0]])
    p2l = torch.tensor([[0, 1, 2, 3]] * 2)
    policy = eplb.DecodeSwapPolicy(
        "greedy_hot_cold",
        num_gpus=2,
        max_swaps_per_layer=8,
        min_gain=0.0,
        min_layer_balancedness=0.9,
    )
    assert {s.layer for s in policy.plan(cost, p2l)} == {0}
    assert policy.skipped_layers == 1
    assert {s.layer for s in _policy().plan(cost, p2l)} == {0, 1}


def test_min_gain_stops_small_swaps():
    # Best swap lowers gpu0 from 11 to 10 (9%): below a 10% gate.
    _, _, swaps = _plan([[6, 5, 5, 4]], [[0, 1, 2, 3]], min_gain=0.10)
    assert swaps == [0]
    _, _, swaps = _plan([[6, 5, 5, 4]], [[0, 1, 2, 3]], min_gain=0.05)
    assert swaps == [1]


def test_max_swaps_caps_each_layer():
    cost = [[9, 8, 7, 6, 0, 0, 0, 0]]
    _, _, swaps = _plan(cost, [list(range(8))], max_swaps=1)
    assert swaps == [1]


def test_never_two_replicas_on_one_gpu():
    # Logical 0 is on both GPUs; swapping its gpu0 copy with logical 2 would
    # leave two copies of 0 on gpu1, so the only legal move is 1<->2.
    p2l, _, swaps = _plan([[9, 8, 1, 1]], [[0, 1, 0, 2]])
    for g in range(2):
        row = p2l[0, g * 2 : g * 2 + 2].tolist()
        assert len(set(row)) == 2
    assert swaps[0] >= 1


def test_plan_is_deterministic():
    torch.manual_seed(0)
    cost = torch.randint(0, 100, (4, 32)).tolist()
    p2l = [list(range(32))] * 4
    a = _policy(8).plan(torch.tensor(cost).float(), torch.tensor(p2l))
    b = _policy(8).plan(torch.tensor(cost).float(), torch.tensor(p2l))
    assert a == b and len(a) > 0


def test_plan_never_raises_max_gpu_cost():
    torch.manual_seed(1)
    cost = torch.randint(0, 50, (16, 64)).float()
    p2l = torch.arange(64, dtype=torch.int32).expand(16, -1).clone()
    new_p2l, new_cost, _ = _plan(cost, p2l, num_gpus=8)
    assert torch.all(
        _gpu_cost(new_cost, 8).max(-1).values <= _gpu_cost(cost, 8).max(-1).values
    )
    for layer in range(16):
        assert sorted(new_p2l[layer].tolist()) == list(range(64))


def _monitor(num_layers=1, num_physical=4, decode=True):
    mon = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    mon.initialize(
        num_layers=num_layers, num_physical=num_physical, device=torch.device("cpu")
    )
    if decode:
        mon.enable_decode_load()
    return mon


def _pass(mon, counts, **kw):
    mon.on_forward_start()
    ids = [e for e, n in enumerate(counts) for _ in range(n)]
    mon.record(
        layer_id=0,
        topk_physical=torch.tensor(ids, dtype=torch.int32).view(-1, 1),
        num_physical=len(counts),
    )
    mon.on_forward_end(**kw)


def test_decode_pass_only_feeds_decode_load(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup())
    mon = _monitor()
    _pass(
        mon,
        [1, 2, 0, 0],
        is_dummy_run=False,
        is_pure_prefill=False,
        is_pure_decode=True,
    )
    _pass(
        mon, [5, 0, 0, 0], is_dummy_run=True, is_pure_prefill=False, is_pure_decode=True
    )
    _pass(
        mon,
        [0, 0, 3, 0],
        is_dummy_run=False,
        is_pure_prefill=False,
        is_pure_decode=True,
    )
    load, steps = mon.dump_global_decode_stats()
    assert load.tolist() == [[1, 2, 3, 0]] and steps == 2
    assert mon._filled == 0
    _pass(mon, [0, 0, 0, 7], is_dummy_run=False, is_pure_prefill=True)
    assert mon.dump_global_physical_load().tolist() == [[0, 0, 0, 7]]
    assert mon.dump_global_decode_stats()[0].tolist() == [[1, 2, 3, 0]]
    mon.reset_decode_load()
    load, steps = mon.dump_global_decode_stats()
    assert load.tolist() == [[0, 0, 0, 0]] and steps == 0.0


def test_decode_load_off_by_default():
    mon = _monitor(decode=False)
    _pass(
        mon,
        [1, 2, 0, 0],
        is_dummy_run=False,
        is_pure_prefill=False,
        is_pure_decode=True,
    )
    assert not mon.decode_load_enabled
    assert mon.dump_global_decode_stats() is None


def _manager(mon, interval, max_rebalances=0, prefill_interval=1000):
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=mon,
        rebalance_interval=prefill_interval,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        decode_rebalance_interval=interval,
        decode_max_rebalances=max_rebalances,
    )
    mgr._dp_is_migration_group = True
    return mgr


def _decode_step(mgr):
    mgr.on_forward_pass_end(local_has_prefill=False, dp_any_has_prefill=False)


def _prefill_step(mgr):
    mgr.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=True)


def _spy_decode(mgr, fired):
    def _fake():
        fired.append(1)
        mgr._decode_rebalance_count += 1
        return
        yield  # pragma: no cover

    mgr._decode_rebalance = _fake


def test_decode_schedule_counts_pure_decode_steps_only():
    mgr = _manager(_monitor(), interval=3)
    fired = []
    _spy_decode(mgr, fired)
    _decode_step(mgr)
    _prefill_step(mgr)
    _decode_step(mgr)
    assert fired == []
    _decode_step(mgr)
    _decode_step(mgr)
    assert fired == [1]
    for _ in range(3):
        _decode_step(mgr)
    assert fired == [1, 1]


def test_decode_schedule_stops_after_max_and_survives_prefill_stop():
    mgr = _manager(_monitor(), interval=1, max_rebalances=2)
    fired = []
    _spy_decode(mgr, fired)
    mgr._rebalancing_stopped = True  # prefill EPLB already done
    for _ in range(10):
        _decode_step(mgr)
    assert fired == [1, 1]
    assert mgr._decode_stopped


def test_decode_off_keeps_prefill_schedule():
    mgr = _manager(_monitor(decode=False), interval=0)
    assert mgr._decode_gen is None
    fired = []

    def _fake():
        fired.append(1)
        return
        yield  # pragma: no cover

    mgr._execute_runtime_rebalance = _fake
    for _ in range(500):
        _decode_step(mgr)
    assert fired == []


def test_prefill_off_runs_decode_only():
    mgr = _manager(_monitor(), interval=2, prefill_interval=0)
    assert mgr._gen is None
    fired = []
    _spy_decode(mgr, fired)
    prefill_fired = []

    def _fake_prefill():
        prefill_fired.append(1)
        return
        yield  # pragma: no cover

    mgr._rebalance = _fake_prefill
    for _ in range(50):
        _prefill_step(mgr)
    for _ in range(3):
        _decode_step(mgr)
    assert prefill_fired == [] and fired == [1]


def test_both_off_skips_step_entirely(monkeypatch):
    mgr = _manager(_monitor(decode=False), interval=0, prefill_interval=0)

    def _no_reduce(*args, **kwargs):
        raise AssertionError("has-prefill reduction must be skipped")

    mgr._dp_is_migration_group = False
    monkeypatch.setattr(torch.distributed, "all_reduce", _no_reduce)
    _prefill_step(mgr)
    _decode_step(mgr)


def test_config_accepts_prefill_rebalance_off():
    from atom.config import EPLBConfig

    cfg = EPLBConfig(rebalance_interval=0, decode_rebalance_interval=500)
    assert cfg.rebalance_interval == 0
    with pytest.raises(AssertionError, match="rebalance_interval"):
        EPLBConfig(load_window_size=100, rebalance_interval=50)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("decode", [False, True])
def test_fused_record_skips_pad_rows_only_with_decode(monkeypatch, decode):
    import types

    import atom.config
    import atom.utils.forward_context as fc

    dev = torch.device("cuda")
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=1, num_logical_experts=8, ep_size=1, ep_rank=0, device=dev
    )
    monkeypatch.setattr(eplb, "get_live_expert_location_metadata", lambda: meta)
    monkeypatch.setattr(
        atom.config,
        "get_current_atom_config",
        lambda: types.SimpleNamespace(
            eplb_enable=True, eplb_config=types.SimpleNamespace(load_window_size=2)
        ),
    )
    mon = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    mon.initialize(num_layers=1, num_physical=8, device=dev)
    if decode:
        mon.enable_decode_load()
    monkeypatch.setattr(eplb, "get_expert_load_monitor", lambda **_: mon)
    pad = torch.tensor([[False], [False], [True], [True]], device=dev)
    monkeypatch.setattr(fc, "step_pad_rows", lambda n: pad[:n])

    topk = torch.tensor([[0, 1], [1, 2], [7, 7], [7, 6]], dtype=torch.int32, device=dev)
    mon.on_forward_start()
    out = eplb.eplb_map_and_record_fused(types.SimpleNamespace(layer_id=0), topk)
    torch.cuda.synchronize()
    assert out.tolist() == topk.tolist()
    expect = [1, 2, 1, 0, 0, 0, 0, 0] if decode else [1, 2, 1, 0, 0, 0, 1, 3]
    assert mon._cur_pass_count[0].tolist() == expect


def test_decode_rebalance_builds_swapped_metadata(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup())
    num_layers, num_logical, ep = 2, 8, 2
    mon = _monitor(num_layers=num_layers, num_physical=num_logical)
    mgr = _manager(mon, interval=1)
    mgr.live_metadata = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=num_layers,
        num_logical_experts=num_logical,
        ep_size=ep,
        ep_rank=0,
    )
    mgr._ep_group = object()
    mgr._moe_layers = {0: None, 1: None}
    mon._decode_load[0] = torch.tensor([9, 9, 9, 9, 1, 1, 1, 1], dtype=torch.int32)
    mon._decode_load[1] = torch.tensor([3, 3, 3, 3, 3, 3, 3, 3], dtype=torch.int32)
    captured = {}

    def _fake_migrate(new_meta, layer_ids, label):
        captured.update(meta=new_meta, layers=layer_ids, label=label)
        return
        yield  # pragma: no cover

    mgr._migrate_layers = _fake_migrate
    _decode_step(mgr)
    assert captured == {}
    _decode_step(mgr)

    assert captured["layers"] == [0]
    assert mgr.decode_rebalance_count == 1
    meta = captured["meta"]
    assert meta.physical_to_logical_map[1].tolist() == list(range(8))
    new0 = meta.physical_to_logical_map[0].tolist()
    assert sorted(new0) == list(range(8)) and new0 != list(range(8))
    for e in range(num_logical):
        slot = int(meta.logical_to_physical_map[0, e, 0])
        assert new0[slot] == e
        local = int(meta.logical_to_rank_dispatch_physical_map[0, e])
        assert local == (slot if slot < num_logical // ep else -1)
    assert torch.equal(meta.expert_map, mgr.live_metadata.expert_map)
    # the window that produced the plan is consumed
    assert mon._decode_load.sum() == 0


def test_metadata_for_layers_matches_full_rebuild():
    num_layers, num_logical, num_physical, ep = 3, 8, 12, 2
    mgr = _manager(_monitor(num_layers=num_layers, num_physical=num_physical), 1)
    mgr.live_metadata = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=num_layers,
        num_logical_experts=num_logical,
        num_physical_experts=num_physical,
        ep_size=ep,
        ep_rank=1,
    )
    live = mgr.live_metadata
    swaps = [
        eplb.ExpertSwap(layer=1, slot_a=0, slot_b=7, expert_a=0, expert_b=7),
        eplb.ExpertSwap(layer=1, slot_a=2, slot_b=10, expert_a=2, expert_b=2),
    ]
    p2l = eplb.apply_swaps(live.physical_to_logical_map, swaps)
    got = mgr._metadata_for_layers(p2l, [1])
    full = eplb.ExpertLocationMetadata.from_rebalance_result(
        **mgr._placement_rows(p2l),
        ep_size=ep,
        ep_rank=1,
        max_num_replicas=live.max_num_replicas,
    )
    names = (
        "physical_to_logical_map",
        "logical_to_physical_map",
        "logical_replica_count",
        "logical_to_rank_dispatch_physical_map",
    )
    for name in names:
        assert torch.equal(getattr(got, name)[1], getattr(full, name)[1]), name
        for layer in (0, 2):
            assert torch.equal(getattr(got, name)[layer], getattr(live, name)[layer])
    assert torch.equal(got.expert_map, live.expert_map)
    assert (
        got.physical_to_logical_map.data_ptr()
        != live.physical_to_logical_map.data_ptr()
    )


def test_activation_policy_differs_from_tokens():
    # gpu0: one very hot + two cold slots; gpu1: three medium. Tokens are even
    # (1500 vs 1500), but gpu1 wakes ~3 experts per step and gpu0 ~1.
    load = torch.tensor([[1500.0, 0.0, 0.0, 500.0, 500.0, 500.0]])
    p2l = torch.tensor([[0, 1, 2, 3, 4, 5]])
    kw = {"num_gpus": 2, "max_swaps_per_layer": 8, "min_gain": 0.0}
    assert eplb.DecodeSwapPolicy("greedy_hot_cold", **kw).plan(load, p2l, 10) == []
    swaps = eplb.DecodeSwapPolicy("greedy_activation", **kw).plan(load, p2l, 10)
    assert len(swaps) == 1
    s = swaps[0]
    assert s.expert_a in (3, 4, 5) and s.expert_b in (1, 2)
    act = eplb.DecodeSwapPolicy.activation(eplb.apply_swaps(load, swaps), 10)
    per_gpu = act.view(2, 3).sum(-1)
    assert abs(float(per_gpu[0] - per_gpu[1])) < 1e-6
