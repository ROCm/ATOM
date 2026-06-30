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


def _record_single_pass(monitor, *, counts):
    monitor.on_forward_start()
    pairs = []
    for expert_id, num in enumerate(counts):
        pairs.extend([expert_id] * num)
    topk = torch.tensor(pairs, dtype=torch.int32).view(-1, 1)
    monitor.record(layer_id=0, topk_physical=topk, num_physical=len(counts))
    monitor.on_forward_end(is_dummy_run=False)


def test_manager_steps_with_dummy_and_triggers_by_interval(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # Make load imbalanced so balancedness < 0.8 and gate passes.
    _record_single_pass(monitor, counts=[4, 0])

    triggered = {"n": 0}
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=3,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: triggered.__setitem__("n", triggered["n"] + 1),
    )

    manager.on_forward_pass_end(is_dummy_run=False)
    manager.on_forward_pass_end(is_dummy_run=True)
    manager.on_forward_pass_end(is_dummy_run=False)
    assert triggered["n"] == 1
    assert manager.rebalance_count == 1


def test_manager_balancedness_gate_skips_when_balanced(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # Perfectly balanced.
    _record_single_pass(monitor, counts=[3, 3])

    triggered = {"n": 0}
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: triggered.__setitem__("n", triggered["n"] + 1),
    )
    manager.on_forward_pass_end(is_dummy_run=False)
    assert triggered["n"] == 0
    assert manager.rebalance_count == 0
    assert manager.last_balancedness == pytest.approx(1.0)


def test_manager_min_vs_mean_aggregation(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # layer-0: 10/2 => 0.5, layer-1: 6/6 => 1.0
    # min=0.5, mean=0.75
    fake_load = torch.tensor([[10, 2], [6, 6]], dtype=torch.int32)
    monkeypatch.setattr(monitor, "dump_global_physical_load", lambda: fake_load)

    min_hit = {"n": 0}
    mgr_min = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: min_hit.__setitem__("n", min_hit["n"] + 1),
    )
    mgr_min.on_forward_pass_end(is_dummy_run=False)
    assert min_hit["n"] == 1

    mean_hit = {"n": 0}
    mgr_mean = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="mean",
        on_rebalance=lambda: mean_hit.__setitem__("n", mean_hit["n"] + 1),
    )
    mgr_mean.on_forward_pass_end(is_dummy_run=False)
    assert mean_hit["n"] == 0


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
