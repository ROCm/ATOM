# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-B manager only)

import types

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")


class _FakeTPGroup:
    def __init__(self, world_size: int = 1):
        self.world_size = world_size

    def all_reduce(self, tensor, ca_fp8_quant=False):  # pragma: no cover
        _ = ca_fp8_quant
        return tensor


def _init_monitor(monitor, *, num_layers=1, num_physical=2, device=None):
    if device is None:
        device = torch.device("cpu")
    monitor.initialize(num_layers=num_layers, num_physical=num_physical, device=device)
    return monitor


def _record_single_pass(monitor, *, counts):
    if monitor._cur_pass_count is None:
        _init_monitor(monitor, num_layers=1, num_physical=len(counts))
    monitor.on_forward_start()
    pairs = []
    for expert_id, num in enumerate(counts):
        pairs.extend([expert_id] * num)
    topk = torch.tensor(pairs, dtype=torch.int32).view(-1, 1)
    monitor.record(layer_id=0, topk_physical=topk, num_physical=len(counts))
    monitor.on_forward_end(is_dummy_run=False)


def _step(mgr):
    """Advance the manager one prefill step without torch.distributed.

    Drives the dp==migration-group path (a real runtime topology, e.g. DSv4-Pro
    DP=8/TP=1): the step gate then reuses the DP-provided prefill flag directly,
    so no collective is issued and these scheduler unit tests need no dist init.
    """
    mgr._dp_is_migration_group = True
    mgr.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=True)


def _spy_rebalance(mgr, fired):
    """Replace the runtime rebalance with a 0-yield spy that records each fire.

    The scheduler drives `_execute_rebalance`, which yields-from
    `_execute_runtime_rebalance`. Recording a fire without yielding mirrors the
    real migration's synchronous, per-call behaviour and keeps the exact
    call-count semantics these scheduler tests assert -- without needing real
    `live_metadata` / EP MoE layers / weight movement.
    """

    def _fake(*_):
        fired.append(1)
        return
        yield  # pragma: no cover - marks this a generator (never reached)

    mgr._execute_runtime_rebalance = _fake


def test_manager_steps_with_dummy_and_triggers_by_interval(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # Make load imbalanced so balancedness < 0.8 and gate passes.
    _record_single_pass(monitor, counts=[4, 0])

    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=8,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(manager, fired)

    # vllm-style warm start: first window = interval//4 = 2, so the first LIVE
    # rebalance fires on call 3; steady state then uses the full interval (8).
    _step(manager)  # 1 (first window)
    _step(manager)  # 2 (first window)
    assert fired == []
    _step(manager)  # 3 -> first rebalance
    assert fired == [1]
    assert manager.rebalance_count == 1

    # Steady state: the next rebalance is a full interval (8 calls) later.
    for _ in range(7):
        _step(manager)
    assert fired == [1]
    _step(manager)  # 8th steady call -> fire
    assert fired == [1, 1]
    assert manager.rebalance_count == 2


def test_manager_balancedness_gate_skips_when_balanced(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    # Perfectly balanced.
    _record_single_pass(monitor, counts=[3, 3])

    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(manager, fired)
    # Per-GPU balancedness needs ep_size from the live placement; in this
    # manager-only unit test there is no model, so provide a minimal stub.
    # counts=[3,3] with ep_size=2 -> perg=[3,3] -> mean/max = 1.0.
    manager.live_metadata = types.SimpleNamespace(ep_size=2)
    _step(manager)  # consumes interval yield
    _step(manager)  # enters _rebalance, gate skips
    assert fired == []
    assert manager.rebalance_count == 0
    assert manager.last_balancedness == pytest.approx(1.0)


def test_manager_min_vs_mean_aggregation(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    # ep_size=2 -> each physical slot is one GPU; per-layer mean/max over GPUs:
    #   layer-0 [10,2] -> 6/10 = 0.6 ; layer-1 [6,6] -> 6/6 = 1.0
    #   => min=0.6, mean=0.8
    fake_load = torch.tensor([[10, 2], [6, 6]], dtype=torch.int32)
    monkeypatch.setattr(monitor, "dump_global_physical_load", lambda: fake_load)

    fired_min = []
    mgr_min = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(mgr_min, fired_min)
    mgr_min.live_metadata = types.SimpleNamespace(ep_size=2)
    _step(mgr_min)
    _step(mgr_min)
    assert fired_min == [1]  # min=0.6 < 0.7 -> rebalance

    fired_mean = []
    mgr_mean = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="mean",
    )
    _spy_rebalance(mgr_mean, fired_mean)
    mgr_mean.live_metadata = types.SimpleNamespace(ep_size=2)
    _step(mgr_mean)
    _step(mgr_mean)
    assert fired_mean == []  # mean=0.8 >= 0.7 -> skip


def test_manager_interval_must_cover_window():
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=4)
    with pytest.raises(AssertionError, match="rebalance_interval"):
        eplb.EPLBManager(
            enabled=True,
            monitor=monitor,
            rebalance_interval=3,
            rebalance_min_balancedness=0.8,
            rebalance_balancedness_agg="min",
        )


def test_manager_trigger_offline_rebalance(monkeypatch):
    # Offline trigger bypasses the interval schedule AND the balancedness gate.
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    _record_single_pass(monitor, counts=[4, 0])

    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=100,  # would never fire periodically
        rebalance_min_balancedness=0.0,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(mgr, fired)
    mgr.trigger_offline_rebalance(reason="test")
    assert fired == [1]
    assert mgr.rebalance_count == 1


def test_manager_stops_after_max_rebalances(monkeypatch):
    # ATOM_EPLB_MAX_REBALANCES=2: two periodic rebalances run, then the schedule
    # stops for good -- no more gate evaluations, no per-step prefill reduction.
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", "2")
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    _record_single_pass(monitor, counts=[4, 0])

    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=8,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(manager, fired)

    for _ in range(3):  # first window (interval//4 = 2) + the first rebalance
        _step(manager)
    assert fired == [1]
    assert not manager.rebalancing_stopped
    for _ in range(8):  # one full interval -> the second rebalance, then stop
        _step(manager)
    assert fired == [1, 1]
    assert manager.rebalance_count == 2
    assert manager.rebalancing_stopped

    dumps = []
    monkeypatch.setattr(monitor, "dump_global_physical_load", lambda: dumps.append(1))
    for _ in range(100):
        _step(manager)
    assert fired == [1, 1]
    assert dumps == []

    def _no_reduce(*args, **kwargs):
        raise AssertionError("a stopped manager must not reduce the prefill flag")

    # With a migration group other than the DP group every step would otherwise
    # all_reduce the has-prefill flag.
    monkeypatch.setattr(torch.distributed, "all_reduce", _no_reduce)
    manager._dp_is_migration_group = False
    manager.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=None)


def test_manager_max_rebalances_ignores_gate_skips(monkeypatch):
    # Only rebalances that actually run count toward the limit.
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", "1")
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    _record_single_pass(monitor, counts=[3, 3])  # balanced -> the gate skips

    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(manager, fired)
    manager.live_metadata = types.SimpleNamespace(ep_size=2)
    for _ in range(10):
        _step(manager)
    assert fired == []
    assert manager.rebalance_count == 0
    assert not manager.rebalancing_stopped


@pytest.mark.parametrize("value", [None, "0", "-3"])
def test_manager_max_rebalances_unset_or_nonpositive_is_unlimited(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ATOM_EPLB_MAX_REBALANCES", raising=False)
    else:
        monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", value)
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    _record_single_pass(monitor, counts=[4, 0])

    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    _spy_rebalance(manager, fired)
    for _ in range(21):  # first window (1) + 20 one-step intervals
        _step(manager)
    assert manager.max_rebalances == 0
    assert len(fired) == 20
    assert not manager.rebalancing_stopped


def test_with_eplb_forward_monitor_passthrough_when_disabled(monkeypatch):
    # When no manager is configured (EPLB off), the decorator is a transparent
    # pass-through and must not touch the monitor/scheduler.
    monkeypatch.setattr(eplb, "_MANAGER", None)

    @eplb.with_eplb_forward_monitor
    def _forward(self, batch):
        return 1

    assert _forward(object(), object()) == 1


def test_execute_rebalance_uses_default_stream_no_explicit_wait(monkeypatch):
    """Migration is enqueued on the default stream (same stream as the forward
    pass), so FIFO ordering holds without an explicit wait_stream/synchronize --
    aligned with SGLang/vllm. _execute_rebalance must never call wait_stream.
    """
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )

    def _fake(*_):
        return
        yield  # pragma: no cover - marks this a generator

    manager._execute_runtime_rebalance = _fake

    waited = {"called": False}

    class _CurrentStream:
        def wait_stream(self, s):
            waited["called"] = True

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _CurrentStream())

    for _ in manager._execute_rebalance():
        pass
    assert not waited["called"]


def test_execute_rebalance_drains_runtime_generator():
    """_execute_rebalance must yield-from the chunked runtime rebalance so a
    forward pass can run between migration chunks.
    """
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    steps = []

    def _fake(*_):
        steps.append("chunk0")
        yield
        steps.append("chunk1")

    manager._execute_runtime_rebalance = _fake

    gen = manager._execute_rebalance()
    next(gen)
    assert steps == ["chunk0"]
    with pytest.raises(StopIteration):
        next(gen)
    assert steps == ["chunk0", "chunk1"]


def _alternating_hot_manager(monkeypatch, policy, reduced_shapes=None):
    """2 GPUs, 4 physical slots, no redundancy, live placement GPU0=[A,B],
    GPU1=[C,D]. Step 1 loads A,B; step 2 loads C,D: balanced over the window,
    yet every step leaves one GPU idle."""
    group = types.SimpleNamespace(world_size=2, device_group=object())
    shapes = reduced_shapes if reduced_shapes is not None else []
    # Stand-in for the group reduction: record what is reduced, sum over 1 rank.
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, **kw: shapes.append(tuple(t.shape))
    )
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.set_load_group(group)
    monitor.track_step_load = policy == "step_peak"
    monitor.initialize(
        num_layers=1, num_physical=4, device=torch.device("cpu"), num_gpus=2
    )
    fired = []
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=0.9,
        rebalance_balancedness_agg="min",
        placement_policy=policy,
    )
    _spy_rebalance(manager, fired)
    manager.live_metadata = types.SimpleNamespace(ep_size=2)
    manager._dp_is_migration_group = True
    for hot in ([0, 1], [2, 3]):
        monitor.on_forward_start()
        topk = torch.tensor([e for e in hot for _ in range(100)], dtype=torch.int32)
        monitor.record(layer_id=0, topk_physical=topk.view(-1, 1), num_physical=4)
        monitor.on_forward_end(is_dummy_run=False)
        manager.on_forward_pass_end(True, True, local_step_load=True)
    return manager, fired


def test_step_peak_gate_judges_per_step_balance(monkeypatch):
    # The window sum is perfectly balanced, so the window gate (naive/biased)
    # skips; step_peak judges on per-step balance (sum of mean / sum of busiest
    # GPU = 200 / 400) and rebalances.
    naive, naive_fired = _alternating_hot_manager(monkeypatch, "naive")
    assert naive_fired == []
    assert naive.last_balancedness == pytest.approx(1.0)

    step_peak, step_peak_fired = _alternating_hot_manager(monkeypatch, "step_peak")
    assert step_peak_fired == [1]
    assert step_peak.last_balancedness == pytest.approx(0.5)


@pytest.mark.parametrize("policy", ["naive", "biased"])
def test_window_policies_keep_no_step_windows(monkeypatch, policy):
    shapes = []
    manager, _ = _alternating_hot_manager(monkeypatch, policy, reduced_shapes=shapes)
    monitor = manager.monitor
    assert monitor._step_load_window is None and monitor._step_gpu_window is None
    assert monitor.dump_global_step_load() is None
    assert monitor.dump_global_step_gpu_load() is None
    # Only the [L, P] window sum feeds the gate, as before step_peak existed.
    assert shapes == [(1, 4)]


def test_step_windows_allocated_per_policy():
    def monitor_for(track):
        monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=8)
        monitor.track_step_load = track
        monitor.initialize(
            num_layers=3, num_physical=12, device=torch.device("cpu"), num_gpus=4
        )
        return monitor

    window_only = monitor_for(False)
    assert window_only._step_load_window is None
    assert window_only._step_gpu_window is None

    step_peak = monitor_for(True)
    assert tuple(step_peak._step_load_window.shape) == (8, 3, 12)
    assert tuple(step_peak._step_gpu_window.shape) == (8, 3, 4)


def _planning_step_peak_manager(monkeypatch, *, reduced_shapes, max_rebalances=0):
    """The 2-GPU alternating-hotspot setup with the real planning path: live
    metadata, a migration group of one rank (broadcast is a no-op) and a
    `_migrate` spy that records each plan and yields once like a real chunk."""
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", str(max_rebalances))
    group = types.SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda t, **kw: reduced_shapes.append(tuple(t.shape)),
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **kw: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.set_load_group(group)
    monitor.track_step_load = True
    monitor.initialize(
        num_layers=1, num_physical=4, device=torch.device("cpu"), num_gpus=2
    )
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=0.9,
        rebalance_balancedness_agg="min",
        placement_policy="step_peak",
    )
    manager.live_metadata = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=1, num_logical_experts=4, ep_size=2, ep_rank=0
    )
    manager._moe_layers = {0: types.SimpleNamespace(num_expert_group=1)}
    manager._ep_group = object()
    manager._migration_group = object()
    manager._dp_is_migration_group = True
    plans = []

    def _migrate(plan):
        plans.append(plan)
        yield

    manager._migrate = _migrate
    return manager, plans


def _prefill_step(manager, hot, *, dp_flag=True):
    """One group prefill step whose 100 tokens per expert hit `hot`."""
    monitor = manager.monitor
    monitor.on_forward_start()
    topk = torch.tensor([e for e in hot for _ in range(100)], dtype=torch.int32)
    monitor.record(layer_id=0, topk_physical=topk.view(-1, 1), num_physical=4)
    monitor.on_forward_end(is_dummy_run=False)
    manager.on_forward_pass_end(
        True, dp_flag if dp_flag is not None else None, local_step_load=True
    )


def _gpu_pairs(p2l):
    return {frozenset(p2l[0, :2].tolist()), frozenset(p2l[0, 2:].tolist())}


def test_step_peak_rebalance_reduces_each_window_once(monkeypatch):
    # The gate reduces the [S, L, G] window; the plan reduces the [S, L, P] one
    # and logs the gate's reduction instead of reducing the GPU window again.
    # No physical or logical window sum is reduced on the step_peak path.
    shapes = []
    manager, plans = _planning_step_peak_manager(monkeypatch, reduced_shapes=shapes)
    _prefill_step(manager, [0, 1])
    _prefill_step(manager, [2, 3])
    assert len(plans) == 1
    assert shapes == [(2, 1, 2), (2, 1, 4)]
    # Alternating hotspots are paired so every step loads both GPUs.
    p2l = plans[0].new_meta.physical_to_logical_map
    assert all(len(pair & {0, 1}) == 1 for pair in _gpu_pairs(p2l))


def test_step_peak_statistics_are_freed_before_migrating(monkeypatch):
    import gc
    import weakref

    manager, _ = _planning_step_peak_manager(monkeypatch, reduced_shapes=[])
    monitor = manager.monitor
    refs = []
    for name in ("dump_global_step_load", "dump_global_step_gpu_load"):
        dump = getattr(monitor, name)

        def _tracked(dump=dump):
            out = dump()
            refs.append(weakref.ref(out))
            return out

        monkeypatch.setattr(monitor, name, _tracked)
    alive_at_first_yield = []

    def _migrate(plan):
        gc.collect()
        alive_at_first_yield.extend(r() is not None for r in refs)
        yield

    manager._migrate = _migrate
    _prefill_step(manager, [0, 1])
    _prefill_step(manager, [2, 3])
    assert len(refs) == 2 and alive_at_first_yield == [False, False]


def test_step_window_commits_group_prefill_steps_only():
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=3)
    monitor.track_step_load = True
    monitor.initialize(
        num_layers=1, num_physical=4, device=torch.device("cpu"), num_gpus=2
    )
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=100,
        rebalance_min_balancedness=0.0,
        rebalance_balancedness_agg="min",
        placement_policy="step_peak",
    )
    manager._dp_is_migration_group = True
    for hot, local, group in (
        ([0], True, True),
        ([1], False, True),
        ([2], True, False),
    ):
        monitor.on_forward_start()
        topk = torch.tensor([hot[0]] * 10, dtype=torch.int32).view(-1, 1)
        monitor.record(layer_id=0, topk_physical=topk, num_physical=4)
        manager.on_forward_pass_end(local, group, local_step_load=local)
    # Step 1 is this rank's; step 2 is the group's but not this rank's (a zero
    # row keeps the slots aligned with the peers'); step 3 is no group prefill.
    window = monitor._step_load_window
    assert monitor._step_filled == 2
    assert window[0, 0].tolist() == [10, 0, 0, 0]
    assert window[1, 0].tolist() == [0, 0, 0, 0]


def test_manual_rebalance_after_stop_follows_new_hotspots(monkeypatch):
    # Reach the limit, then the hotspots move and refill the window. With the
    # DP group as the migration group the window keeps following traffic at no
    # collective cost, so a manual rebalance plans on the new hotspots.
    shapes = []
    manager, plans = _planning_step_peak_manager(
        monkeypatch, reduced_shapes=shapes, max_rebalances=1
    )
    _prefill_step(manager, [0, 1])
    _prefill_step(manager, [2, 3])
    _prefill_step(manager, [0, 1])  # runs the migration's chunk: the stop latches
    assert len(plans) == 1 and manager.rebalancing_stopped
    del shapes[:]
    _prefill_step(manager, [0, 2])  # new hotspots: A with C, then B with D
    _prefill_step(manager, [1, 3])
    assert shapes == []  # following the window after the stop costs nothing
    manager.trigger_offline_rebalance(reason="test")
    assert len(plans) == 2
    assert all(
        len(pair & {0, 2}) == 1
        for pair in _gpu_pairs(plans[1].new_meta.physical_to_logical_map)
    )


def test_manual_rebalance_refuses_a_frozen_step_window(monkeypatch, caplog):
    # Without the DP sync's flag, following the window after the stop would
    # need a per-step collective: it freezes instead, and a manual rebalance
    # says so rather than planning on the old hotspots.
    shapes = []
    manager, plans = _planning_step_peak_manager(
        monkeypatch, reduced_shapes=shapes, max_rebalances=1
    )
    _prefill_step(manager, [0, 1])
    _prefill_step(manager, [2, 3])
    _prefill_step(manager, [0, 1])  # runs the migration's chunk: the stop latches
    assert manager.rebalancing_stopped
    manager._dp_is_migration_group = False
    del shapes[:]
    _prefill_step(manager, [0, 2], dp_flag=None)
    assert shapes == []  # still no per-step collective once stopped
    with caplog.at_level("WARNING", logger="atom"):
        manager.trigger_offline_rebalance(reason="test")
    assert len(plans) == 1 and shapes == []
    assert "would plan on old hotspots" in caplog.text
