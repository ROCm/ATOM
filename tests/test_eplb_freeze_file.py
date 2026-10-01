# SPDX-License-Identifier: MIT
# Tests for ATOM_EPLB_FREEZE_FILE: EPLB stops rebalancing once the file exists.

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


def _manager(monkeypatch, fired):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    monitor.initialize(num_layers=1, num_physical=2, device=torch.device("cpu"))
    # Imbalanced load so every rebalance point passes the balancedness gate.
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.zeros((4, 1), dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
    )

    def _fake():
        fired.append(1)
        return
        yield  # pragma: no cover - marks this a generator

    mgr._execute_runtime_rebalance = _fake
    # DP group == migration group: the step gate issues no collective, and
    # _migration_group stays None, so the freeze check is rank-local.
    mgr._dp_is_migration_group = True
    return mgr


def _step(mgr):
    mgr.on_forward_pass_end(local_has_prefill=True, dp_any_has_prefill=True)


def test_rebalances_without_freeze_file(monkeypatch):
    monkeypatch.delenv("ATOM_EPLB_FREEZE_FILE", raising=False)
    fired = []
    mgr = _manager(monkeypatch, fired)
    for _ in range(4):
        _step(mgr)
    assert len(fired) == 3  # first step is the warm-start window
    assert mgr.rebalance_count == 3


def test_freeze_file_stops_rebalancing_and_latches(monkeypatch, tmp_path):
    path = tmp_path / "eplb_freeze"
    monkeypatch.setenv("ATOM_EPLB_FREEZE_FILE", str(path))
    fired = []
    mgr = _manager(monkeypatch, fired)
    _step(mgr)  # warm-start window
    _step(mgr)  # file absent -> rebalance
    assert fired == [1]

    path.touch()
    for _ in range(3):
        _step(mgr)
    assert fired == [1]
    assert mgr.rebalance_count == 1

    # Latched: removing the file does not resume rebalancing.
    path.unlink()
    for _ in range(3):
        _step(mgr)
    assert fired == [1]
