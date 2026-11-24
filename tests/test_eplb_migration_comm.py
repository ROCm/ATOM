# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# Tests for when EPLB brings up its migration communicator (atom/model_ops/eplb.py).

import sys
import types

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")


class _FakeMoE(torch.nn.Module):
    def __init__(self, num_physical: int, num_redundant: int):
        super().__init__()
        self.layer_id = 0
        self.use_ep = True
        self.global_num_experts = num_physical
        self.num_redundant_experts = num_redundant
        self.ep_size = 1
        self.ep_rank = 0
        self.w13_weight = torch.zeros(num_physical, 2, 2)
        self.w2_weight = torch.zeros(num_physical, 2, 2)


class _FakeModel(torch.nn.Module):
    def __init__(self, num_physical: int, num_redundant: int):
        super().__init__()
        self.moe = _FakeMoE(num_physical, num_redundant)


class _Dist:
    """Records the collectives EPLB issues; none touches a device."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, object]] = []
        self.group = object()
        dist = torch.distributed
        monkeypatch.setattr(dist, "new_group", self._new_group)
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
        monkeypatch.setattr(dist, "barrier", self._barrier)
        monkeypatch.setattr(dist, "all_reduce", self._all_reduce)
        monkeypatch.setattr(dist, "destroy_process_group", self._destroy)

    def _new_group(self, ranks):
        self.calls.append(("new_group", tuple(ranks)))
        return self.group

    def _barrier(self, group=None):
        self.calls.append(("barrier", group))

    def _all_reduce(self, tensor, op=None, group=None):
        self.calls.append(("all_reduce", group))

    def _destroy(self, group=None):
        self.calls.append(("destroy", group))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def dist(monkeypatch):
    ep = types.SimpleNamespace(device_group="ep", rank_in_group=0, world_size=1)
    dp = types.SimpleNamespace(device_group="dp", rank_in_group=0, world_size=1)
    fake = types.ModuleType("aiter.dist.parallel_state")
    fake.get_ep_group = lambda: ep
    fake.get_dp_group = lambda: dp
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", fake)
    return _Dist(monkeypatch)


def _manager(monkeypatch, max_rebalances: int = 0) -> "eplb.EPLBManager":
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", str(max_rebalances))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=1)
    return eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=1,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
    )


def _init(monkeypatch, num_physical: int, num_redundant: int):
    fills = []
    monkeypatch.setattr(
        eplb, "migrate_and_commit_chunk", lambda **kw: fills.append(kw["ep_group"])
    )
    mgr = _manager(monkeypatch)
    owner = types.SimpleNamespace(model=_FakeModel(num_physical, num_redundant))
    mgr._maybe_initialize_runtime(owner)
    return mgr, fills


def test_init_without_redundant_experts_runs_no_collective(monkeypatch, dist):
    mgr, fills = _init(monkeypatch, num_physical=4, num_redundant=0)
    # The group exists, but nothing has brought up its communicator.
    assert dist.names() == ["new_group"]
    assert mgr._migration_group is dist.group
    assert not mgr._migration_comm_ready
    assert fills == []


def test_init_with_redundant_experts_brings_up_comm_before_fill(monkeypatch, dist):
    mgr, fills = _init(monkeypatch, num_physical=4, num_redundant=1)
    # Barrier first (all ranks), then fill_redundant's P2P, then its own barrier.
    assert dist.calls == [
        ("new_group", (0,)),
        ("barrier", dist.group),
        ("barrier", dist.group),
    ]
    assert fills == [dist.group]
    assert mgr._migration_comm_ready


def test_first_runtime_migration_barriers_once_before_p2p(monkeypatch, dist):
    mgr, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    order = []
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda group=None: order.append(("barrier", group)),
    )
    monkeypatch.setattr(
        eplb,
        "migrate_and_commit_chunk",
        lambda **kw: order.append(("migrate", kw["ep_group"])),
    )
    monkeypatch.setattr(
        mgr.monitor,
        "dump_global_logical_load",
        lambda: torch.ones(1, 4, dtype=torch.int32),
    )
    for _ in range(2):
        for _step in mgr._execute_runtime_rebalance():
            pass
    assert order == [
        ("barrier", dist.group),
        ("migrate", dist.group),
        ("migrate", dist.group),
    ]


def test_prefill_flag_reduction_brings_up_comm_once(monkeypatch, dist):
    mgr, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    mgr._dp_is_migration_group = False
    mgr._gen = iter(lambda: None, 1)  # keep the scheduler out of the way
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda data, device=None, dtype=None: real_tensor(data, dtype=dtype),
    )
    for _ in range(3):
        mgr.on_forward_pass_end(local_has_prefill=True)
    assert dist.names() == ["new_group", "barrier"] + ["all_reduce"] * 3
    assert all(group is dist.group for name, group in dist.calls[1:])


def test_freeze_releases_comm_only_once_it_exists(monkeypatch, dist):
    mgr, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    mgr._stop_rebalancing()
    assert "destroy" not in dist.names()
    assert mgr._migration_group is dist.group

    mgr2, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    mgr2._ensure_migration_comm()
    mgr2._stop_rebalancing()
    assert dist.calls[-1] == ("destroy", dist.group)
    assert mgr2._migration_group is None and not mgr2._migration_comm_ready


def test_failed_destroy_is_not_fatal(monkeypatch, dist):
    mgr, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    mgr._ensure_migration_comm()

    def _boom(group=None):
        raise RuntimeError("no")

    monkeypatch.setattr(torch.distributed, "destroy_process_group", _boom)
    mgr._stop_rebalancing()
    assert mgr.rebalancing_stopped and mgr._migration_group is None


def test_offline_rebalance_after_freeze_is_ignored(monkeypatch, dist):
    mgr, _ = _init(monkeypatch, num_physical=4, num_redundant=0)
    mgr._ensure_migration_comm()
    mgr._stop_rebalancing()
    ran = []
    monkeypatch.setattr(mgr, "_execute_rebalance", lambda: ran.append(1) or iter(()))
    before = list(dist.calls)
    mgr.trigger_offline_rebalance(reason="test")
    assert ran == []
    assert dist.calls == before
