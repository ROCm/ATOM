# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-A and Module-B)

import pytest

torch = pytest.importorskip("torch")

# Import atom.config first so it is fully initialized before atom.model_ops's
# __init__ chain references get_current_atom_config (avoids a mainline circular
# import that only surfaces when atom.model_ops is the entry-point import).
import atom.config  # noqa: F401
import atom.model_ops.eplb as eplb


class _FakeTPGroup:
    def __init__(self, world_size: int = 1):
        self.world_size = world_size

    def all_reduce(self, tensor, ca_fp8_quant=False):  # pragma: no cover
        # For unit tests we only need deterministic pass-through semantics.
        _ = ca_fp8_quant
        return tensor


def test_count_physical_load_filters_invalid_ids():
    topk = torch.tensor(
        [
            [0, 1, 2],
            [2, -1, 8],  # -1 and 8 are invalid for num_physical=4
        ],
        dtype=torch.int32,
    )
    counts = eplb.count_physical_load(topk, num_physical=4)
    assert counts.tolist() == [1, 1, 2, 0]


def test_monitor_window_accumulate_and_skip_dummy(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=3)

    # pass-1 (real): [2,1,1,0]
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 0], [1, 2]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out.shape == (1, 4)
    assert out[0].tolist() == [2, 1, 1, 0]

    # pass-2 (dummy): should not be appended into window.
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[3, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=True)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out[0].tolist() == [2, 1, 1, 0]

    # pass-3 (real): add [0,3,0,1] => total [2,4,1,1]
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[1, 1], [1, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out[0].tolist() == [2, 4, 1, 1]


def test_monitor_capacity_growth_preserves_existing_window(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)

    # first real pass on layer-0, width=2
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)

    # second real pass grows to layer-2 and width=4
    monitor.on_forward_start()
    monitor.record(
        layer_id=2,
        topk_physical=torch.tensor([[3, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)

    out = monitor.dump_global_physical_load()
    assert out is not None
    # layer-0 keeps its previous record after growth.
    assert out[0].tolist() == [1, 1, 0, 0]
    # layer-2 has new counts.
    assert out[2].tolist() == [0, 0, 0, 2]


def test_count_physical_load_rejects_float_dtype():
    bad = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    with pytest.raises(AssertionError):
        eplb.count_physical_load(bad, num_physical=4)


def test_monitor_freeze_raises_on_new_layer(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)
    monitor.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        monitor.record(
            layer_id=1,  # new layer_id → triggers _ensure_capacity
            topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
            num_physical=2,
        )


def test_monitor_freeze_allows_same_shape(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)
    monitor.freeze()
    # Same layer_id and num_physical: must NOT raise.
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[1, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)


# ---------------------------------------------------------------------------
# Module B – EPLBManager
# ---------------------------------------------------------------------------

def _make_monitor(monkeypatch, *, window_size=2, load=None, num_physical=4):
    """Return a pre-warmed ExpertLoadMonitor with one real pass recorded."""
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=window_size)
    topk = load if load is not None else torch.zeros((1, 2), dtype=torch.int32)
    for _ in range(window_size):
        monitor.on_forward_start()
        monitor.record(layer_id=0, topk_physical=topk, num_physical=num_physical)
        monitor.on_forward_end(is_dummy_run=False)
    return monitor


def test_manager_assert_interval_ge_window_size(monkeypatch):
    monitor = _make_monitor(monkeypatch, window_size=4)
    with pytest.raises(AssertionError):
        eplb.EPLBManager(
            enabled=True,
            monitor=monitor,
            rebalance_interval=2,  # < window_size=4
            rebalance_min_balancedness=0.0,
            rebalance_balancedness_agg="min",
        )


def test_manager_triggers_at_interval(monkeypatch):
    # interval=3: generator yields 3 times (calls 1-3), rebalance fires on call 4.
    monitor = _make_monitor(monkeypatch, window_size=2)
    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=3,
        rebalance_min_balancedness=2.0,  # unreachable → always rebalance
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired.append(1),
    )
    for _ in range(3):
        mgr.on_forward_pass_end(is_dummy_run=False)
    assert fired == [], "should not fire before interval slots complete"
    mgr.on_forward_pass_end(is_dummy_run=False)  # call 4: rebalance fires
    assert fired == [1], "should fire on call interval+1"
    # Call 4 also consumed the 1st yield of the new period, so only 2 more
    # calls are needed before the next fire (total 3 per period).
    for _ in range(2):
        mgr.on_forward_pass_end(is_dummy_run=False)
    assert fired == [1], "should not fire again before next interval"
    mgr.on_forward_pass_end(is_dummy_run=False)  # call 7: second fire
    assert fired == [1, 1], "should fire again at second interval"


def test_manager_dummy_advances_schedule(monkeypatch):
    # interval=3: 3 dummy + 1 real = 4 calls total → fires on call 4.
    monitor = _make_monitor(monkeypatch, window_size=2)
    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=3,
        rebalance_min_balancedness=2.0,  # always rebalance
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired.append(1),
    )
    mgr.on_forward_pass_end(is_dummy_run=True)
    mgr.on_forward_pass_end(is_dummy_run=True)
    mgr.on_forward_pass_end(is_dummy_run=True)
    assert fired == []
    mgr.on_forward_pass_end(is_dummy_run=False)  # call 4: fire
    assert fired == [1], "dummy forwards must count toward the interval"


def test_manager_skips_when_balanced(monkeypatch):
    # Even load: balancedness=1.0 >= threshold=0.9 → skip.
    # interval=2: fire-check happens on call 3.
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    even = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    for _ in range(2):
        monitor.on_forward_start()
        monitor.record(layer_id=0, topk_physical=even, num_physical=4)
        monitor.on_forward_end(is_dummy_run=False)

    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=0.9,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired.append(1),
    )
    for _ in range(3):  # call 3 is where the gate check runs
        mgr.on_forward_pass_end(is_dummy_run=False)
    assert fired == [], "perfectly balanced load must not trigger rebalance"


def test_manager_fires_when_imbalanced(monkeypatch):
    # interval=2: rebalance fires on call 3 (calls 1-2 fill the period).
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # All tokens to expert-0: highly imbalanced → balancedness = 0.25 < 0.9
    skewed = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32)
    for _ in range(2):
        monitor.on_forward_start()
        monitor.record(layer_id=0, topk_physical=skewed, num_physical=4)
        monitor.on_forward_end(is_dummy_run=False)

    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=2,
        rebalance_min_balancedness=0.9,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired.append(1),
    )
    mgr.on_forward_pass_end(is_dummy_run=False)
    mgr.on_forward_pass_end(is_dummy_run=False)
    assert fired == [], "rebalance not yet fired after interval slots"
    mgr.on_forward_pass_end(is_dummy_run=False)  # call 3: fire
    assert fired == [1], "skewed load must trigger rebalance"


def test_manager_balancedness_agg_min_vs_mean(monkeypatch):
    # layer-0: perfectly balanced (bal=1.0)
    # layer-1: all-to-expert-0 (bal=0.25 for 4 experts)
    # min → 0.25 < 0.9 → rebalance; mean → 0.625 < 0.9 → rebalance
    # Use threshold=0.7: min triggers (0.25<0.7), mean triggers (0.625<0.7)
    # Use threshold=0.5: min triggers (0.25<0.5), mean does NOT (0.625>=0.5)
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    def _build_monitor():
        mon = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
        even = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        skew = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32)
        for _ in range(2):
            mon.on_forward_start()
            mon.record(layer_id=0, topk_physical=even, num_physical=4)
            mon.record(layer_id=1, topk_physical=skew, num_physical=4)
            mon.on_forward_end(is_dummy_run=False)
        return mon

    fired_min, fired_mean = [], []

    mgr_min = eplb.EPLBManager(
        enabled=True,
        monitor=_build_monitor(),
        rebalance_interval=2,
        rebalance_min_balancedness=0.5,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired_min.append(1),
    )
    mgr_mean = eplb.EPLBManager(
        enabled=True,
        monitor=_build_monitor(),
        rebalance_interval=2,
        rebalance_min_balancedness=0.5,
        rebalance_balancedness_agg="mean",
        on_rebalance=lambda: fired_mean.append(1),
    )
    # interval=2: fire on call 3.
    for _ in range(3):
        mgr_min.on_forward_pass_end(is_dummy_run=False)
        mgr_mean.on_forward_pass_end(is_dummy_run=False)

    assert fired_min == [1], "min agg should trigger (worst-layer balancedness < threshold)"
    assert fired_mean == [], "mean agg should skip (average balancedness >= threshold)"


def test_manager_trigger_offline_rebalance(monkeypatch):
    monitor = _make_monitor(monkeypatch, window_size=2)
    fired = []
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=100,  # would never fire periodically
        rebalance_min_balancedness=0.0,
        rebalance_balancedness_agg="min",
        on_rebalance=lambda: fired.append(1),
    )
    mgr.trigger_offline_rebalance(reason="test")
    assert fired == [1]
    assert mgr.rebalance_count == 1
