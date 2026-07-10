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


def test_manager_steps_with_dummy_and_triggers_by_interval(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    # Make load imbalanced so balancedness < 0.8 and gate passes.
    _record_single_pass(monitor, counts=[4, 0])

    triggered = {"n": 0}
    owner = type(
        "Owner",
        (),
        {
            "run_eplb_rebalance": lambda self, s: triggered.__setitem__(
                "n", triggered["n"] + 1
            )
        },
    )()
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=8,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    manager.bind_runtime_owner(owner)

    # vllm-style warm start: first window = interval//4 = 2, so the first LIVE
    # rebalance fires on call 3; steady state then uses the full interval (8).
    manager.on_forward_pass_end(is_dummy_run=False)  # 1 (first window)
    manager.on_forward_pass_end(is_dummy_run=True)  # 2 (first window)
    assert triggered["n"] == 0
    manager.on_forward_pass_end(is_dummy_run=False)  # 3 -> first rebalance
    assert triggered["n"] == 1
    assert manager.rebalance_count == 1

    # Steady state: the next rebalance is a full interval (8 calls) later.
    for _ in range(7):
        manager.on_forward_pass_end(is_dummy_run=False)
    assert triggered["n"] == 1
    manager.on_forward_pass_end(is_dummy_run=False)  # 8th steady call -> fire
    assert triggered["n"] == 2
    assert manager.rebalance_count == 2


def test_manager_balancedness_gate_skips_when_balanced(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    # Perfectly balanced.
    _record_single_pass(monitor, counts=[3, 3])

    triggered = {"n": 0}
    owner = type(
        "Owner",
        (),
        {
            "run_eplb_rebalance": lambda self, s: triggered.__setitem__(
                "n", triggered["n"] + 1
            )
        },
    )()
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    manager.bind_runtime_owner(owner)
    manager.on_forward_pass_end(is_dummy_run=False)  # consumes interval yield
    manager.on_forward_pass_end(is_dummy_run=False)  # enters _rebalance, gate skips
    assert triggered["n"] == 0
    assert manager.rebalance_count == 0
    assert manager.last_balancedness == pytest.approx(1.0)


def test_manager_min_vs_mean_aggregation(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    # layer-0: 10/2 => 0.5, layer-1: 6/6 => 1.0
    # min=0.5, mean=0.75
    fake_load = torch.tensor([[10, 2], [6, 6]], dtype=torch.int32)
    monkeypatch.setattr(monitor, "dump_global_physical_load", lambda: fake_load)

    min_hit = {"n": 0}
    min_owner = type(
        "Owner",
        (),
        {
            "run_eplb_rebalance": lambda self, s: min_hit.__setitem__(
                "n", min_hit["n"] + 1
            )
        },
    )()
    mgr_min = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="min",
    )
    mgr_min.bind_runtime_owner(min_owner)
    mgr_min.on_forward_pass_end(is_dummy_run=False)
    mgr_min.on_forward_pass_end(is_dummy_run=False)
    assert min_hit["n"] == 1

    mean_hit = {"n": 0}
    mean_owner = type(
        "Owner",
        (),
        {
            "run_eplb_rebalance": lambda self, s: mean_hit.__setitem__(
                "n", mean_hit["n"] + 1
            )
        },
    )()
    mgr_mean = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.7,
        rebalance_balancedness_agg="mean",
    )
    mgr_mean.bind_runtime_owner(mean_owner)
    mgr_mean.on_forward_pass_end(is_dummy_run=False)
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


def test_manager_runs_bound_owner_rebalance(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    _record_single_pass(monitor, counts=[4, 0])

    seen = {"n": 0}

    class _Owner:
        def run_eplb_rebalance(self, stream):
            _ = stream
            seen["n"] += 1

    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.9,
        rebalance_balancedness_agg="min",
    )
    mgr.bind_runtime_owner(_Owner())
    mgr.on_forward_pass_end(is_dummy_run=False)  # consumes interval yield
    mgr.on_forward_pass_end(is_dummy_run=False)  # triggers rebalance
    assert seen["n"] == 1


def test_with_eplb_forward_monitor_allows_missing_runtime_owner_hook(monkeypatch):
    class _Cfg:
        eplb_enable = True
        eplb_config = type(
            "ECfg",
            (),
            {
                "load_window_size": 1,
                "rebalance_interval": 1,
                "rebalance_min_balancedness": 0.8,
                "rebalance_balancedness_agg": "min",
            },
        )()

    monkeypatch.setattr(
        "atom.config.get_current_atom_config",
        lambda: _Cfg(),
    )

    @eplb.with_eplb_forward_monitor
    def _forward(self, batch):
        return 1

    assert _forward(object(), object()) == 1


def test_with_eplb_forward_monitor_uses_runner_rebalance_hook(monkeypatch):
    class _Cfg:
        eplb_enable = True
        eplb_config = type(
            "ECfg",
            (),
            {
                "load_window_size": 1,
                "rebalance_interval": 1,
                "rebalance_min_balancedness": 0.8,
                "rebalance_balancedness_agg": "min",
            },
        )()

    monkeypatch.setattr(
        "atom.config.get_current_atom_config",
        lambda: _Cfg(),
    )
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    # Ensure the rebalance gate passes.
    monkeypatch.setattr(
        eplb.ExpertLoadMonitor,
        "dump_global_physical_load",
        lambda self: torch.tensor([[4, 0]], dtype=torch.int32),
    )

    seen = {"n": 0, "stream": None}

    class _Runner:
        def run_eplb_rebalance(self, stream):
            seen["n"] += 1
            seen["stream"] = stream

    @eplb.with_eplb_forward_monitor
    def _forward(self, batch):
        return 1

    runner = _Runner()
    _forward(runner, object())
    _forward(runner, object())
    assert seen["n"] == 1


def test_execute_rebalance_uses_default_stream_no_explicit_wait(monkeypatch):
    """cuda_stream=None (default stream) means no explicit wait_stream call.

    Migration is enqueued on the default stream which is the same stream
    as the forward pass, so FIFO ordering is guaranteed without synchronize
    or wait_stream — aligned with SGLang/vllm.
    """
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    manager.bind_runtime_owner(
        type("Owner", (), {"run_eplb_rebalance": lambda self, s: None})()
    )

    waited = {"called": False}

    class _CurrentStream:
        def wait_stream(self, s):
            waited["called"] = True

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _CurrentStream())

    for _ in manager._execute_rebalance():
        pass
    assert not waited["called"]


def test_execute_rebalance_drains_owner_generator(monkeypatch):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=0.8,
        rebalance_balancedness_agg="min",
    )
    steps = []

    class _Owner:
        def run_eplb_rebalance(self, stream):
            _ = stream

            def _gen():
                steps.append("chunk0")
                yield
                steps.append("chunk1")

            return _gen()

    manager.bind_runtime_owner(_Owner())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    gen = manager._execute_rebalance()
    next(gen)
    assert steps == ["chunk0"]
    with pytest.raises(StopIteration):
        next(gen)
    assert steps == ["chunk0", "chunk1"]
