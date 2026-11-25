# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU tests for ATOM_PD_ENDPOINT_RECOVERY with a fake Mooncake fabric.

The fake fabric models what the PITT incident showed: every engine keeps one
endpoint per decode peer (``ip:rpc_port``) for its whole life, a restarted
decode comes back on a new RPC port, and an HCA refuses new endpoints once
the endpoints of every engine on it reach its capacity. Only destroying an
engine frees its endpoints.
"""

import os
import threading
from collections import Counter

import msgpack
import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

from test_pd_chunked_transfer import consumer, dispatch, producer, publish

from atom.kv_transfer.disaggregation.mooncake.rail_engine_pool import (
    RailEnginePool,
    RailRecoveryPolicy,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # setenv first so the undo restores "unset" even for what the code sets.
    for name in ("ATOM_PD_ENDPOINT_RECOVERY", "MC_RDMA_NOTIFY_ENABLED"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ATOM_PD_FAILED_WRITE_HOLD_S", "60")


# ---------------------------------------------------------------------------
# Fake Mooncake fabric
# ---------------------------------------------------------------------------


class Fabric:
    """Endpoint accounting per HCA, shared by every engine on that HCA."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.live = Counter()  # device -> endpoints held by live engines
        self.lock = threading.Lock()


class FakeEngine:
    def __init__(self, fabric=None):
        self.fabric = fabric
        self.device = None
        self.regions = []
        self.released = []
        self.endpoints = set()
        self.fail_ptr = None
        self.block_register = None
        self.destroyed = False
        self.writes = 0

    def initialize(self, ip, metadata, protocol, device):
        assert (metadata, protocol) == ("P2PHANDSHAKE", "rdma")
        self.device = device
        return 0

    def register_memory(self, ptr, size):
        if self.block_register is not None:
            assert self.block_register.wait(5)
        if ptr == self.fail_ptr:
            return -202
        self.regions.append((ptr, size))
        return 0

    def unregister_memory(self, ptr):
        self.released.append(ptr)
        return 0

    def get_rpc_port(self):
        return 17000

    def batch_transfer_sync_write(self, target, srcs, dsts, sizes):
        assert not self.destroyed, "a destroyed engine was used"
        if target not in self.endpoints:
            with self.fabric.lock:
                if self.fabric.live[self.device] >= self.fabric.capacity:
                    return -1  # "Failed to modify QP to RTR" / -302
                self.fabric.live[self.device] += 1
            self.endpoints.add(target)
        self.writes += 1
        return 0

    def destroy(self):
        """What dropping the last reference to a TransferEngine does."""
        with self.fabric.lock:
            self.fabric.live[self.device] -= len(self.endpoints)
        self.endpoints.clear()
        self.destroyed = True


def sync_spawn(fn):
    fn()


DEFAULT_POLICY = RailRecoveryPolicy(2, 3, 0.0)


def make_pool(recovery=DEFAULT_POLICY, fabric=None, clock=None):
    fabric = fabric or Fabric(capacity=1000)
    primary = FakeEngine(fabric)
    primary.device = "rdma0"
    created = []

    def factory():
        engine = FakeEngine(fabric)
        created.append(engine)
        return engine

    pool = RailEnginePool(
        factory,
        primary,
        "rdma0",
        ["rdma0", "rdma7"],
        lambda device: "127.0.0.1",
        recovery=recovery,
        clock=clock or (lambda: 0.0),
        spawn=sync_spawn,
        retire_wait_s=0.0,
    )
    pool.set_regions([1024, 2048], [64, 128])
    return pool, primary, created, fabric


# ---------------------------------------------------------------------------
# RailEnginePool recovery
# ---------------------------------------------------------------------------


def test_without_policy_failures_and_restarts_change_nothing():
    pool, _, created, _ = make_pool(recovery=None)
    engine = pool.get("rdma7")
    for port in range(20):
        pool.note_peer("rdma7", ("d", 0, 0, 0), f"d:{port}")
        pool.report(engine, False)
    assert pool.get("rdma7") is engine
    assert len(created) == 1


def test_repeated_failures_rebuild_a_rail_and_success_resets_the_streak():
    pool, _, created, _ = make_pool()
    old = pool.get("rdma7")
    pool.report(old, False)
    pool.report(old, True)  # the streak counts failures in a row only
    pool.report(old, False)
    assert pool.get("rdma7") is old
    pool.report(old, False)
    new = pool.get("rdma7")
    assert new is not old and new is created[-1]
    assert new.regions == [(1024, 64), (2048, 128)]
    assert pool.rebuilds == {"rdma7": 1}
    # Late outcomes of the replaced engine are ignored.
    pool.report(old, False)
    pool.report(old, False)
    assert pool.get("rdma7") is new


def test_primary_is_never_rebuilt():
    pool, primary, created, _ = make_pool()
    for _ in range(10):
        pool.report(primary, False)
        pool.note_peer("rdma0", ("d", 0, 0, 0), f"d:{_}")
    assert pool.get("rdma0") is primary
    assert not created


def test_decode_restarts_trigger_a_rebuild_by_port_or_instance():
    pool, _, created, _ = make_pool()
    old = pool.get("rdma7")
    peer = ("10.0.0.2", 0, 3, 0)
    pool.note_peer("rdma7", peer, "10.0.0.2:15000", "a")
    pool.note_peer("rdma7", peer, "10.0.0.2:15000", "a")  # same instance
    pool.note_peer("rdma7", peer, "10.0.0.2:16000", "b")  # restart 1
    pool.note_peer("rdma7", peer, "10.0.0.2:16000", "c")  # same port, restart 2
    assert pool.get("rdma7") is old
    other = ("10.0.0.2", 0, 4, 0)
    pool.note_peer("rdma7", other, "10.0.0.2:15001")
    pool.note_peer("rdma7", other, "10.0.0.2:15001")  # no instance: port decides
    assert pool.get("rdma7") is old
    pool.note_peer("rdma7", other, "10.0.0.2:16001")  # restart 3
    assert pool.get("rdma7") is not old
    assert len(created) == 2


def test_failed_rebuild_keeps_the_engine_rolls_back_and_backs_off():
    now = [0.0]
    pool, _, created, _ = make_pool(
        recovery=RailRecoveryPolicy(1, 99, 10.0), clock=lambda: now[0]
    )
    old = pool.get("rdma7")
    pool._factory = lambda: _failing(created)
    pool.report(old, False)
    assert pool.get("rdma7") is old
    assert created[-1].released == [1024]  # rolled back in reverse
    pool.report(old, False)  # inside the 20 s backoff: not retried
    assert len(created) == 2
    now[0] = 21.0
    pool._factory = lambda: _append(created, FakeEngine(old.fabric))
    pool.report(old, False)
    assert pool.get("rdma7") is created[-1] is not old
    now[0] = 22.0  # the min interval follows a successful rebuild
    pool.report(pool.get("rdma7"), False)
    assert len(created) == 3


def _append(created, engine):
    created.append(engine)
    return engine


def _failing(created):
    engine = FakeEngine(Fabric(1))
    engine.fail_ptr = 2048
    created.append(engine)
    return engine


def test_rebuild_registers_off_the_lock_and_old_engine_keeps_serving():
    fabric = Fabric(capacity=1000)
    primary = FakeEngine(fabric)
    gate = threading.Event()
    created = []

    def factory():
        engine = FakeEngine(fabric)
        if len(created) == 1:
            engine.block_register = gate  # the rebuild, not a first use
        created.append(engine)
        return engine

    threads = []

    def spawn(fn):
        t = threading.Thread(target=fn, daemon=True)
        threads.append(t)
        t.start()

    pool = RailEnginePool(
        factory,
        primary,
        "rdma0",
        ["rdma0", "rdma6", "rdma7"],
        lambda device: "127.0.0.1",
        recovery=RailRecoveryPolicy(1, 99, 0.0),
        spawn=spawn,
        retire_wait_s=0.0,
    )
    pool.set_regions([1024], [64])
    old = pool.get("rdma7")
    pool.report(old, False)
    # While the rebuild is blocked in registration, every rail still serves,
    # including the first use of another rail.
    assert pool.get("rdma7") is old
    rdma6 = pool.get("rdma6")
    assert rdma6 is created[-1] and rdma6.device == "rdma6"
    pool.report(old, False)  # already rebuilding: no second rebuild
    gate.set()
    threads[0].join(5)
    assert not threads[0].is_alive()
    assert pool.get("rdma7") is created[1]
    assert created[1].device == "rdma7"
    assert len(threads) == 1


# ---------------------------------------------------------------------------
# Decode swaps behind one prefill, through the connector's real write path
# ---------------------------------------------------------------------------


def run_swaps(recovery, swaps=15, capacity=8, decode_ranks=1):
    fabric = Fabric(capacity=capacity)
    pool, _, _, _ = make_pool(
        recovery=recovery and RailRecoveryPolicy(2, 4, 0.0), fabric=fabric
    )
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c._rail_pool = pool
    seen = []
    failures = 0
    for swap in range(swaps):
        for rank in range(decode_ranks):
            port = 15000 + swap * 10 + rank
            request = {
                "consumer_host": "10.19.0.120",
                "consumer_rpc_port": port,
                "consumer_ib_device": "rdma7",
                "consumer_dp_rank": 0,
                "consumer_tp_rank": rank,
                "consumer_instance": f"swap{swap}",
            }
            for _ in range(3):  # c1 requests flowing to this decode
                engine = pool.get("rdma7")
                c._note_rail_peer(request)
                if engine not in seen:
                    seen.append(engine)
                ok = c._rdma_write(
                    f"10.19.0.120:{port}", [1], [2], [3], "r", "block", engine=engine
                )
                failures += not ok
        # Engines the pool dropped are gone once their requests ended.
        current = pool.get("rdma7")
        for engine in seen:
            if engine is not current and not engine.destroyed:
                engine.destroy()
    return failures, pool.rebuilds.get("rdma7", 0), fabric.live["rdma7"]


def test_decode_swaps_exhaust_the_hca_without_recovery(monkeypatch):
    failures, rebuilds, live = run_swaps(recovery=False)
    assert rebuilds == 0
    assert live == 8  # capacity reached: every later decode fails
    assert failures == 3 * (15 - 8)


def test_decode_swaps_never_fail_with_recovery(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    failures, rebuilds, live = run_swaps(recovery=True)
    assert failures == 0
    assert rebuilds >= 3
    assert live <= 5


def test_failure_streak_alone_recovers_an_exhausted_rail(monkeypatch):
    """Even without restart detection, failures rebuild the rail and heal it."""
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    fabric = Fabric(capacity=2)
    pool, _, _, _ = make_pool(recovery=RailRecoveryPolicy(2, 10**6, 0.0), fabric=fabric)
    c = mc.MooncakeConnector.__new__(mc.MooncakeConnector)
    c._rail_pool = pool
    results = []
    for port in range(6):
        for _ in range(3):
            engine = pool.get("rdma7")
            results.append(
                bool(c._rdma_write(f"d:{port}", [1], [2], [3], "r", "b", engine=engine))
            )
            if pool.get("rdma7") is not engine:
                engine.destroy()
    # A dead rail costs at most two failed requests before it is rebuilt.
    assert results.count(False) <= 2 * 3
    assert results[-3:] == [True, True, True]


# ---------------------------------------------------------------------------
# write_posted: producer side
# ---------------------------------------------------------------------------


def wire_of(c, request, monkeypatch):
    sent = []
    c.tp_rank = 0
    c.pp_rank = 0
    c._kv_staging = None
    c._send_on_socket = lambda path, parts, repeat=1: sent.append(parts)
    mc.MooncakeConnector._notify_transfer_result(c, request, success=False)
    assert len(sent) == 1
    return msgpack.loads(sent[0][1])


@pytest.mark.parametrize("enabled", [False, True])
def test_released_source_reports_unwritten_only_when_enabled(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c, _state, req, _, _ = producer()
    c._retire_chunked_transfer("xfer-a")
    dispatch(c, req)
    c._notify_transfer_result.assert_called_once_with(req, success=False)
    assert c.transfer_engine.writes.empty()
    message = wire_of(c, req, monkeypatch)
    if enabled:
        assert message["write_posted"] is False
    else:
        assert "write_posted" not in message


def test_bad_geometry_before_any_write_is_unwritten(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c, state, req, _, _ = producer()
    req["dst_block_ids"] = [2]
    publish(state, 10)
    state.finish({"first_token_id": 42})
    dispatch(c, req)
    assert c.transfer_engine.writes.empty()
    assert req["_write_posted"] is False


def test_rdma_failure_after_a_write_is_never_marked_unwritten(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c, state, req, _, _ = producer()
    c.transfer_engine.batch_transfer_sync_write = lambda *a: -1
    publish(state, 10)
    state.finish({"first_token_id": 42})
    dispatch(c, req)
    c._notify_transfer_result.assert_called_once_with(req, success=False)
    assert "_write_posted" not in req
    assert "write_posted" not in wire_of(c, req, monkeypatch)


def test_success_never_carries_write_posted(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c, _state, req, _, _ = producer()
    req["_write_posted"] = False
    sent = []
    c.tp_rank = c.pp_rank = 0
    c._kv_staging = None
    c._send_on_socket = lambda path, parts, repeat=1: sent.append(parts)
    mc.MooncakeConnector._notify_transfer_result(c, req, success=True)
    assert "write_posted" not in msgpack.loads(sent[0][1])


# ---------------------------------------------------------------------------
# write_posted: decode side
# ---------------------------------------------------------------------------


def recovering_consumer(expected=1):
    c = consumer(expected)
    c._pending_handoffs = {}
    c._pending_recv_maybe_written = set()
    return c


def test_unwritten_failure_is_reported_at_once(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c = recovering_consumer()
    assert c._record_write_done(21, 0, 0, 3, success=False, write_posted=False)
    assert c.failed_recving == {21}
    assert not c._held_failures
    assert not c._pending_recv_blocks
    assert not c._pending_recv_maybe_written


@pytest.mark.parametrize("write_posted", [None, True])
def test_possibly_written_failure_is_still_held(monkeypatch, write_posted):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c = recovering_consumer()
    assert c._record_write_done(21, 0, 0, 3, success=False, write_posted=write_posted)
    assert not c.failed_recving
    assert 21 in c._held_failures
    assert c._pending_recv_blocks == {21: [2, 9, 4]}


def test_any_rank_that_may_have_written_keeps_the_hold(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c = recovering_consumer(expected=2)
    assert not c._record_write_done(21, 0, 0, 3, success=False)  # old producer
    assert c._record_write_done(21, 1, 0, 3, success=False, write_posted=False)
    assert not c.failed_recving and 21 in c._held_failures


def test_success_and_unwritten_failure_fail_at_once(monkeypatch):
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    c = recovering_consumer(expected=2)
    assert not c._record_write_done(21, 0, 0, 3, success=True)
    assert c._record_write_done(21, 1, 0, 3, success=False, write_posted=False)
    assert c.failed_recving == {21} and not c._held_failures


def test_disabled_decode_ignores_write_posted(monkeypatch):
    c = recovering_consumer()
    assert c._record_write_done(21, 0, 0, 3, success=False, write_posted=False)
    assert not c.failed_recving and 21 in c._held_failures


def test_treadmill_drains_without_holds(monkeypatch):
    """The incident: each queued request's P source expired while the one
    before it sat out its 60 s hold, so the queue drained one per minute.
    With recovery, the unwritten failures cost no hold at all."""
    now = [0.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])

    def drain(enabled):
        if enabled:
            monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
        else:
            monkeypatch.delenv("ATOM_PD_ENDPOINT_RECOVERY", raising=False)
        now[0] = 0.0
        c = recovering_consumer()
        c._pending_recv_expected = {}
        c._pending_recv_nonce = {}
        c._pending_recv = set()
        c._pending_recv_blocks = {}
        for req in range(1, 31):  # resident cap 1: one request at a time
            c._pending_recv_expected[req] = 1
            c._pending_recv_nonce[req] = 3
            c._pending_recv_blocks[req] = [req]
            c._record_write_done(
                req, 0, 0, 3, success=False, write_posted=False if req > 1 else None
            )
            while req not in c.failed_recving:
                now[0] += 1.0
                c._report_held_failures()
        return now[0]

    assert drain(enabled=False) == 30 * 60
    assert drain(enabled=True) == 60  # only the first, possibly written, one


# ---------------------------------------------------------------------------
# Native notification
# ---------------------------------------------------------------------------


def test_recovery_disables_native_notify_unless_set(monkeypatch):
    mc._configure_mooncake_transport("rdma")
    assert "MC_RDMA_NOTIFY_ENABLED" not in os.environ
    monkeypatch.setenv("ATOM_PD_ENDPOINT_RECOVERY", "1")
    mc._configure_mooncake_transport("rdma")
    assert os.environ["MC_RDMA_NOTIFY_ENABLED"] == "0"
    monkeypatch.setenv("MC_RDMA_NOTIFY_ENABLED", "1")
    mc._configure_mooncake_transport("rdma")
    assert os.environ["MC_RDMA_NOTIFY_ENABLED"] == "1"


def test_recovery_env_defaults():
    from atom.utils import envs

    assert envs.ATOM_PD_ENDPOINT_RECOVERY is False
    assert envs.ATOM_PD_RAIL_REBUILD_AFTER_FAILURES == 2
    assert envs.ATOM_PD_RAIL_MAX_STALE_PEERS == 8
    assert envs.ATOM_PD_RAIL_REBUILD_MIN_INTERVAL_S == 300.0


def test_reports_do_not_wait_for_another_rails_first_registration():
    """A rail's first registration holds the pool lock for tens of seconds;
    every RDMA call reports to the pool, so that must not wait for it."""
    fabric = Fabric(capacity=1000)
    gate = threading.Event()
    created = []

    def factory():
        engine = FakeEngine(fabric)
        if created:
            engine.block_register = gate  # rdma6's first use
        created.append(engine)
        return engine

    pool = RailEnginePool(
        factory,
        FakeEngine(fabric),
        "rdma0",
        ["rdma0", "rdma6", "rdma7"],
        lambda device: "127.0.0.1",
        recovery=RailRecoveryPolicy(2, 2, 0.0),
        spawn=sync_spawn,
        retire_wait_s=0.0,
    )
    pool.set_regions([1024], [64])
    rdma7 = pool.get("rdma7")
    first_use = threading.Thread(target=pool.get, args=("rdma6",), daemon=True)
    first_use.start()
    while not pool._lock.locked():
        pass
    done = threading.Event()

    def bookkeeping():
        pool.report(rdma7, True)
        pool.note_peer("rdma7", ("d", 0, 0, 0), "d:1")
        done.set()

    threading.Thread(target=bookkeeping, daemon=True).start()
    assert done.wait(5)
    gate.set()
    first_use.join(5)
    assert pool.get("rdma6") is created[1]


def test_replaced_engine_is_unregistered_once_its_last_request_ends():
    fabric = Fabric(capacity=1000)
    released = []
    threads = []

    def factory():
        engine = FakeEngine(fabric)
        engine.released = released if not threads else []
        return engine

    def spawn(fn):
        t = threading.Thread(target=fn, daemon=True)
        threads.append(t)
        t.start()

    pool = RailEnginePool(
        factory,
        FakeEngine(fabric),
        "rdma0",
        ["rdma0", "rdma7"],
        lambda device: "127.0.0.1",
        recovery=RailRecoveryPolicy(1, 99, 0.0),
        spawn=spawn,
        retire_wait_s=5.0,
        retire_poll_s=0.01,
    )
    pool.set_regions([1024, 2048], [64, 128])
    in_flight = [pool.get("rdma7")]  # a request that took the engine
    pool.report(in_flight[0], False)
    while pool.rebuilds.get("rdma7", 0) == 0:
        pass
    assert pool.get("rdma7") is not in_flight[0]
    threads[0].join(0.2)
    assert threads[0].is_alive() and released == []  # still in use
    in_flight.clear()  # the request ends
    threads[0].join(5)
    assert not threads[0].is_alive()
    assert released == [2048, 1024]


def test_replaced_engine_still_in_use_is_left_to_its_last_user():
    pool, _, created, _ = make_pool(recovery=RailRecoveryPolicy(1, 99, 0.0))
    pool._retire_wait_s = 0.05
    pool._retire_poll_s = 0.01
    old = pool.get("rdma7")
    pool.report(old, False)  # rebuilt inline; ``old`` is still referenced here
    assert pool.get("rdma7") is created[-1] is not old
    assert old.released == []
