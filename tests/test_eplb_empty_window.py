# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# Tests for the empty-window skip of the EPLB rebalance schedule in
# atom/model_ops/eplb.py (manager scheduling only, CPU).

import types

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")


class _FakeTPGroup:
    world_size = 1


def _monitor(window_size, counts=None):
    m = eplb.ExpertLoadMonitor(enabled=True, window_size=window_size)
    m.initialize(num_layers=1, num_physical=2, device=torch.device("cpu"))
    if counts is not None:
        _record(m, counts)
    return m


def _record(m, counts, is_dummy_run=False):
    m.on_forward_start()
    ids = []
    for expert, n in enumerate(counts):
        ids += [expert] * n
    if ids:
        m.record(
            layer_id=0,
            topk_physical=torch.tensor(ids, dtype=torch.int32).view(-1, 1),
            num_physical=len(counts),
        )
    m.on_forward_end(is_dummy_run=is_dummy_run)


def _manager(monitor, interval):
    mgr = eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=interval,
        rebalance_min_balancedness=2.0,  # always rebalance when the gate runs
        rebalance_balancedness_agg="min",
    )
    mgr.live_metadata = types.SimpleNamespace(ep_size=2)
    fired = []

    def _fake():
        fired.append(mgr.monitor.dump_global_physical_load().clone())
        return
        yield  # pragma: no cover - marks this a generator

    mgr._execute_runtime_rebalance = _fake
    return mgr, fired


def _step(mgr, has_prefill=True):
    mgr._dp_is_migration_group = True
    mgr.on_forward_pass_end(
        local_has_prefill=has_prefill, dp_any_has_prefill=has_prefill
    )


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup())


def test_loaded_window_rebalances():
    mgr, fired = _manager(_monitor(2, [4, 0]), interval=8)
    for _ in range(3):  # interval//4 = 2 first-window steps, then fire
        _step(mgr)
    assert len(fired) == 1
    assert fired[0].tolist() == [[4, 0]]


def test_empty_window_skips_without_counting(monkeypatch):
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", "1")
    mon = _monitor(2)
    mgr, fired = _manager(mon, interval=4)
    # Only dummy steps so far (e.g. start-up warm-up on the P instance of a
    # P/D deployment): the window is empty when the gate first runs.
    _record(mon, [4, 0], is_dummy_run=True)
    _step(mgr)  # first window (interval//4 = 1)
    _step(mgr)  # gate: empty window -> skip, not counted
    assert fired == [] and mgr.rebalance_count == 0
    assert not mgr.rebalancing_stopped
    # Real load arrives; the next interval rebalances once, then freezes.
    for _ in range(4):
        _record(mon, [4, 0])
        _step(mgr)
    assert len(fired) == 1 and mgr.rebalancing_stopped
    assert mgr.rebalance_count == 1
