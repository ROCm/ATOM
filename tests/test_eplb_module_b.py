# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-B manager only)

import pytest

torch = pytest.importorskip("torch")

import atom.model_ops.eplb as eplb


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
    if monitor._cur_pass_count is None:  # noqa: SLF001
        _init_monitor(monitor, num_layers=1, num_physical=len(counts))
    monitor.on_forward_start()
    pairs = []
    for expert_id, num in enumerate(counts):
        pairs.extend([expert_id] * num)
    topk = torch.tensor(pairs, dtype=torch.int32).view(-1, 1)
    monitor.record(layer_id=0, topk_physical=topk, num_physical=len(counts))
    monitor.on_forward_end(is_dummy_run=False)


def _spy_rebalance(mgr, fired):
    """Replace the runtime rebalance with a 0-yield spy that records each fire.

    The scheduler drives `_execute_rebalance`, which yields-from
    `_execute_runtime_rebalance`. Recording a fire without yielding mirrors the
    real migration's synchronous, per-call behaviour and keeps the exact
    call-count semantics these scheduler tests assert -- without needing real
    `live_metadata` / EP MoE layers / weight movement.
    """

    def _fake():
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
    manager.on_forward_pass_end(is_dummy_run=False)  # 1 (first window)
    manager.on_forward_pass_end(is_dummy_run=True)  # 2 (first window)
    assert fired == []
    manager.on_forward_pass_end(is_dummy_run=False)  # 3 -> first rebalance
    assert fired == [1]
    assert manager.rebalance_count == 1

    # Steady state: the next rebalance is a full interval (8 calls) later.
    for _ in range(7):
        manager.on_forward_pass_end(is_dummy_run=False)
    assert fired == [1]
    manager.on_forward_pass_end(is_dummy_run=False)  # 8th steady call -> fire
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
    manager.on_forward_pass_end(is_dummy_run=False)  # consumes interval yield
    manager.on_forward_pass_end(is_dummy_run=False)  # enters _rebalance, gate skips
    assert fired == []
    assert manager.rebalance_count == 0
    assert manager.last_balancedness == pytest.approx(1.0)


def test_manager_min_vs_mean_aggregation(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    # layer-0: 10/2 => 0.5, layer-1: 6/6 => 1.0
    # min=0.5, mean=0.75
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
    mgr_min.on_forward_pass_end(is_dummy_run=False)
    mgr_min.on_forward_pass_end(is_dummy_run=False)
    assert fired_min == [1]  # min=0.5 < 0.7 -> rebalance

    fired_mean = []
    mgr_mean = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="mean",
    )
    _spy_rebalance(mgr_mean, fired_mean)
    mgr_mean.on_forward_pass_end(is_dummy_run=False)
    mgr_mean.on_forward_pass_end(is_dummy_run=False)
    assert fired_mean == []  # mean=0.75 >= 0.7 -> skip


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

    def _fake():
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

    def _fake():
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
