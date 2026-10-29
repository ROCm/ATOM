# SPDX-License-Identifier: MIT
"""Execution authority, lease expiry and real MP prepare-without-DMA regressions."""

import pickle
from dataclasses import replace

import pytest
from conftest import MockConfig
from test_dp_cache_integration import manager, request
from test_dp_load_balance import _route
from test_lmcache_mp_async_lookup import _Adapter, _Client
from test_lmcache_mp_native_scheduler import make_scheduler

from atom.kv_transfer.offload.mp.lookup import _MPLookupClient
from atom.model_engine.dp_cache import preparation
from atom.model_engine.dp_cache.preparation import PrepareMessage
from atom.model_engine.dp_cache.state import DPCacheState
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence


def routed(monkeypatch):
    mgr = manager(monkeypatch)
    router = mgr.cache_router
    router.balancer.config = replace(router.balancer.config, prepare_migrations=True)
    commands = []
    router.preparation.send = lambda rank, msg: commands.append((rank, msg))
    mgr._send_control = lambda rank, payload: commands.append(
        (rank, pickle.loads(payload))
    )
    seq = request()
    assert _route(mgr, [seq]) == [0]  # A remains the owner during preparation.
    a = router.preparation.attempts[seq.id]
    return mgr, router, seq, a, commands


def event(router, a, rank, kind, **values):
    router.prepared(rank, PrepareMessage(kind, a.seq.id, a.generation, **values))


def ready(router, a):
    event(router, a, a.source, "queued")
    event(router, a, a.target, "ready", boundary=98304, ttl=5, hbm_hit=0)


def kinds(commands):
    return [msg.kind for _, msg in commands if isinstance(msg, PrepareMessage)]


def test_no_commit_without_source_release_and_then_transfer_accounting(monkeypatch):
    mgr, router, seq, a, commands = routed(monkeypatch)
    assert seq.dp_cache.preparation_role == "source"
    assert seq.dp_cache.migration_id == 0
    assert mgr._seq_load[seq.id][0] == 0
    ready(router, a)
    assert kinds(commands) == ["prepare", "release_source"]
    assert mgr._seq_load[seq.id][0] == 0
    # Wrong generation and wrong rank cannot release source authority.
    event(router, a, a.target, "source_released")
    router.prepared(
        a.source, PrepareMessage("source_released", seq.id, a.generation + 1)
    )
    assert "commit" not in kinds(commands)
    event(router, a, a.source, "source_released")
    assert kinds(commands)[-1] == "commit"
    assert mgr._seq_load[seq.id][0] == 1
    event(router, a, a.target, "committed")
    assert not router.preparation.attempts
    assert seq.id in router._retry_payloads  # final PAGE+STATE gate still owns retry
    router.admitted(
        1, (seq.id, a.generation, router.balancer.requests[seq.id].minimum_hit)
    )
    assert router.balancer.sessions["session"].owner == 1


@pytest.mark.parametrize("after_ready", [False, True])
def test_source_wins_without_an_extra_permission_roundtrip(monkeypatch, after_ready):
    mgr, router, seq, a, commands = routed(monkeypatch)
    if after_ready:
        ready(router, a)
    else:
        event(router, a, a.source, "queued")
    event(router, a, a.source, "source_kept")
    event(router, a, a.target, "ready", boundary=98304, ttl=5)
    event(router, a, a.source, "source_released")
    assert "commit" not in kinds(commands)
    assert not router.preparation.attempts
    assert mgr._seq_load[seq.id][0] == 0
    router.admitted(0, (seq.id, 0, 0))
    assert router.balancer.requests[seq.id].admitted


def test_target_expiry_while_source_release_is_in_flight_waits_for_ack(monkeypatch):
    mgr, router, seq, a, commands = routed(monkeypatch)
    ready(router, a)
    event(router, a, a.target, "failed")
    assert a.cancelled and a.phase == "releasing"
    assert not any(isinstance(msg, tuple) for _, msg in commands)  # no retry ADD yet
    event(router, a, a.source, "source_released")
    assert "commit" not in kinds(commands)
    assert sum(isinstance(msg, tuple) for _, msg in commands) == 1
    assert mgr._seq_load[seq.id][0] == 0
    assert not router.preparation.attempts


def test_cost_is_rechecked_after_ready_and_after_source_release(monkeypatch):
    mgr, router, _seq, a, commands = routed(monkeypatch)
    ready(router, a)
    mgr._seq_load.pop("background")
    mgr._rank_tokens[0] -= 1000000
    event(router, a, a.source, "source_released")
    assert "commit" not in kinds(commands)
    assert sum(isinstance(msg, tuple) for _, msg in commands) == 1
    assert router.balancer.counters["migration_prepare_gain_disappeared"] == 1


@pytest.mark.parametrize("kind", ["failed", "short"])
def test_only_known_short_prepare_records_negative_evidence(monkeypatch, kind):
    mgr, router, seq, a, commands = routed(monkeypatch)
    event(router, a, a.source, "queued")
    event(router, a, a.target, kind, boundary=0)
    misses = router.balancer.sessions["session"].recovery_misses
    assert bool(misses) is (kind == "short")
    if misses:
        assert (a.source, a.target) in misses
    assert mgr._seq_load[seq.id][0] == a.source
    assert not any(isinstance(msg, tuple) for _, msg in commands)


def test_expired_commit_retries_source_only_on_target_negative_ack(monkeypatch):
    mgr, router, seq, a, commands = routed(monkeypatch)
    ready(router, a)
    event(router, a, a.source, "source_released")
    assert mgr._seq_load[seq.id][0] == 1
    event(router, a, a.target, "failed")
    assert mgr._seq_load[seq.id][0] == 0
    assert sum(isinstance(msg, tuple) for _, msg in commands) == 1
    event(router, a, a.target, "failed")
    assert sum(isinstance(msg, tuple) for _, msg in commands) == 1


@pytest.mark.parametrize("phase", ["queued", "preparing", "releasing", "committing"])
def test_abort_fences_all_late_prepare_events(monkeypatch, phase):
    mgr, router, seq, a, commands = routed(monkeypatch)
    if phase != "queued":
        event(router, a, a.source, "queued")
    if phase in {"releasing", "committing"}:
        event(router, a, a.target, "ready", boundary=98304, ttl=5)
    if phase == "committing":
        event(router, a, a.source, "source_released")
    mgr._release_seq_load(seq.id)
    sent = len(commands)
    for rank, kind in [
        (1, "ready"),
        (0, "source_released"),
        (1, "failed"),
        (1, "committed"),
    ]:
        event(router, a, rank, kind, boundary=98304, ttl=5)
    assert len(commands) == sent
    assert seq.id not in mgr._seq_load
    assert not router.preparation.attempts


def test_reset_preserves_monotonic_tickets(monkeypatch):
    mgr, router, _seq, _a, _commands = routed(monkeypatch)
    previous = router.balancer._generation
    mgr.reset_dp_router()
    assert router.balancer._generation == previous
    assert not router.preparation.attempts


def scheduler():
    return Scheduler(
        MockConfig(
            kv_cache_block_size=4,
            num_kvcache_blocks=128,
            max_model_len=256,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
    )


def source_request():
    seq = Sequence(list(range(65)), 4)
    seq.dp_cache = DPCacheState(
        preparation_id=7, preparation_role="source", prepare_timeout=5
    )
    return seq


def test_source_allocation_beats_late_release_without_duplicate_execution():
    host = scheduler()
    events = []
    host.cache_preparation.emit = events.append
    seq = source_request()
    host.add(seq)
    batch, _ = host.schedule()
    assert seq.id in batch.req_ids
    assert [e.kind for e in events] == ["queued", "source_kept"]
    host.cache_preparation.handle(PrepareMessage("release_source", seq.id, 7))
    assert "source_released" not in [e.kind for e in events]
    assert seq.dp_cache.migration_id == 0


def test_source_release_removes_only_matching_unstarted_request():
    host = scheduler()
    events = []
    host.cache_preparation.emit = events.append
    seq = source_request()
    host.add(seq)
    host.cache_preparation.handle(PrepareMessage("release_source", seq.id, 8))
    assert seq in host.waiting
    host.cache_preparation.handle(PrepareMessage("release_source", seq.id, 7))
    assert not host.waiting and not host.running and not host.take_rejected()
    assert [e.kind for e in events] == ["queued", "source_released"]
    assert host.schedule() is None


class Tier:
    def __init__(self, boundary=48, ttl=1):
        self.boundary = boundary
        self.ttl = ttl
        self.cancelled = []
        self.live = True

    def recovery_prepare_ttl(self):
        return self.ttl

    def prepare_recovery(self, seq):
        return self.boundary

    def recovery_prepared(self, seq, boundary):
        return self.live

    def cancel_recovery(self, seq):
        self.cancelled.append(seq.id)


def target(monkeypatch, *, boundary=48, ttl=1):
    now = [100.0]
    monkeypatch.setattr(preparation.time, "monotonic", lambda: now[0])
    host = scheduler()
    tier = Tier(boundary, ttl)
    host.kv_connector = tier
    events = []
    host.cache_preparation.emit = events.append
    seq = source_request()
    seq.dp_cache.preparation_role = "target"
    seq.dp_cache.migration_id = 7
    msg = PrepareMessage("prepare", seq.id, 7, seq=seq, minimum_hit=48, ttl=2)
    host.cache_preparation.handle(msg)
    return host, tier, events, seq, msg, now


def test_prepare_never_allocates_and_commit_adopts_the_same_live_sequence(monkeypatch):
    host, tier, events, seq, msg, _now = target(monkeypatch)
    host.cache_preparation.advance()
    assert not host.waiting and not host.running and not seq.block_table
    assert events[-1].kind == "ready" and events[-1].boundary == 48
    host.cache_preparation.handle(PrepareMessage("commit", seq.id, 7))
    assert list(host.waiting) == [seq]
    assert not host.cache_preparation.targets
    assert tier.cancelled == []
    assert seq.dp_cache.minimum_hit == 48
    # Cancelling a prepare after commit cannot release an active DMA's lease.
    host.cache_preparation.handle(PrepareMessage("cancel", seq.id, 7))
    assert tier.cancelled == []
    host.cache_preparation.handle(msg)
    host.cache_preparation.handle(PrepareMessage("commit", seq.id, 7))
    assert list(host.waiting) == [seq]


@pytest.mark.parametrize("expiry", ["clock", "pin"])
def test_expired_lease_cannot_commit(monkeypatch, expiry):
    host, tier, events, seq, _msg, now = target(monkeypatch)
    host.cache_preparation.advance()
    if expiry == "clock":
        now[0] += 2
    else:
        tier.live = False
    host.cache_preparation.handle(PrepareMessage("commit", seq.id, 7))
    assert not host.waiting and not seq.block_table
    assert tier.cancelled == [seq.id]
    assert events[-1].kind == "failed"


@pytest.mark.parametrize(
    "boundary,expected", [(0, "short"), (32, "short"), (None, "failed")]
)
def test_short_or_unknown_lookup_returns_no_credential(monkeypatch, boundary, expected):
    host, tier, events, seq, _msg, now = target(monkeypatch, boundary=boundary)
    host.cache_preparation.advance()
    if boundary is None:
        assert not events
        now[0] += 2
        host.cache_preparation.advance()
    assert events[-1].kind == expected
    assert not host.waiting and not seq.block_table
    assert tier.cancelled == [seq.id]


def test_unconfirmed_server_ttl_keeps_preparation_disabled(monkeypatch):
    host, _tier, events, _seq, _msg, _now = target(monkeypatch, ttl=0)
    assert events[-1].kind == "failed"
    assert not host.cache_preparation.targets and not host.waiting


def test_native_prepare_pins_joint_boundary_without_allocating_page_or_state(
    monkeypatch,
):
    native, _checkpoints, _ = make_scheduler(monkeypatch)
    fake = _Client(chunks=2, lookup_done=False)
    adapter = _Adapter(fake)
    adapter.lmcache_tokens_per_chunk = 8
    adapter.end_session = lambda request_id: None
    native._mp_adapter = adapter
    native._lookup_client = _MPLookupClient(
        adapter, config=native._config, timeout=10, poll_interval=0.01
    )
    native._config.kv_transfer_config["kv_connector_extra_config"][
        "lmcache.mp.prepare_read_ttl_seconds"
    ] = 10
    assert native.recovery_prepare_ttl() == 5
    seq = Sequence(list(range(24)), 4, has_per_req_cache=True)
    seq.dp_cache = DPCacheState(preparation_id=1, preparation_role="target")
    assert native.prepare_recovery(seq) is None
    assert not seq.block_table and not native._native_loads
    fake.lookup_future.done = True
    assert native.prepare_recovery(seq) == 16
    assert native.recovery_prepared(seq, 16)
    meta = native.build_connector_meta()
    assert not meta.requests and not meta.lookup_requests_in_step
    assert not native._native_loads
    assert native._lookup_client.hit_tokens(str(seq.id)) == 16
    assert len(fake.lookups) == 1
    native.cancel_recovery(seq)
    meta = native.build_connector_meta()
    assert not native.recovery_prepared(seq, 16)
    assert str(seq.id) in meta.lookup_requests_in_step
    assert not native._native_loads


@pytest.mark.parametrize("after_release_request", [False, True])
def test_cache_invalidation_cannot_commit_an_old_preparation(
    monkeypatch, after_release_request
):
    mgr, router, seq, a, commands = routed(monkeypatch)
    if after_release_request:
        ready(router, a)
    else:
        event(router, a, a.source, "queued")
    router.invalidate()
    event(router, a, a.target, "ready", boundary=98304, ttl=5)
    event(router, a, a.source, "source_released")
    assert "commit" not in kinds(commands)
    assert not router.preparation.attempts
    assert mgr._seq_load[seq.id][0] == 0


def test_target_control_grant_is_ordered_after_source_queue_removal(monkeypatch):
    # The real EngineCore queue dispatcher must process a release after ADD,
    # and before a subsequent schedule could admit the same source request.
    from test_dp_cache_review_regressions import engine

    host = scheduler()
    seq = source_request()
    core, _ = engine(host)
    events = []
    host.cache_preparation.emit = events.append
    core.input_queue.put([seq])
    core.input_queue.put(PrepareMessage("release_source", seq.id, 7))
    assert not core.pull_and_process_input_queue()
    assert not host.waiting and not host.running and not host.take_rejected()
    assert [e.kind for e in events] == ["queued", "source_released"]


def test_delayed_lookup_after_cancel_releases_without_becoming_ready(monkeypatch):
    native, _, _ = make_scheduler(monkeypatch)
    fake = _Client(chunks=2, lookup_done=False)
    adapter = _Adapter(fake)
    adapter.lmcache_tokens_per_chunk = 8
    adapter.end_session = lambda request_id: None
    native._mp_adapter = adapter
    native._lookup_client = _MPLookupClient(
        adapter, config=native._config, timeout=10, poll_interval=0.01
    )
    seq = Sequence(list(range(24)), 4, has_per_req_cache=True)
    seq.dp_cache = DPCacheState(preparation_id=1, preparation_role="target")
    assert native.prepare_recovery(seq) is None
    native.cancel_recovery(seq)
    fake.lookup_future.done = True
    native._lookup_client.pump()
    assert not native.recovery_prepared(seq, 16)
    assert not native._native_loads and not seq.block_table
    assert [(x["start"], x["end"]) for x in adapter.freed] == [(0, 16)]


def test_generation_fences_out_of_order_offers_without_granting_execution(monkeypatch):
    host, tier, events, seq, msg, _now = target(monkeypatch)
    host.cache_preparation.handle(PrepareMessage("cancel", seq.id, 7))
    host.cache_preparation.handle(msg)
    host.cache_preparation.handle(PrepareMessage("commit", seq.id, 7))
    assert not host.waiting and not host.cache_preparation.targets
    assert tier.cancelled == [seq.id]
    assert not any(e.kind == "committed" for e in events)


def test_prepared_page_replay_uses_scheduler_start_in_admission(monkeypatch):
    host, _tier, events, seq, _msg, _now = target(monkeypatch)
    seen = []

    def replay_start(request, boundary, *, hit):
        seen.append((request is seq, boundary, hit, request.replays_hit))
        return boundary - 8

    monkeypatch.setattr(host.block_manager, "replay_start", replay_start)
    host.cache_preparation.advance()
    assert seen == [(True, 48, True, True)]
    assert events[-1].boundary == 48
    assert events[-1].forward_start == 40
    assert not seq.replays_hit
    host.cache_preparation.handle(PrepareMessage("commit", seq.id, 7))
    assert seq.dp_cache.minimum_hit == 40
    assert list(host.waiting) == [seq]


def test_prepare_replay_cost_can_reject_a_move_before_releasing_source(monkeypatch):
    mgr, router, _seq, a, commands = routed(monkeypatch)
    mgr._seq_load["background"] = (0, 0, 1000)
    event(router, a, a.source, "queued")
    # Without a saved prefix the target must pay the complete prefill, so a
    # transfer that looked attractive from PAGE reach alone loses its gain.
    event(router, a, a.target, "ready", boundary=98304, forward_start=0, ttl=5)
    assert "release_source" not in kinds(commands)
    assert not router.preparation.attempts


def test_prepare_accounts_replay_tokens_without_shortening_the_transfer(monkeypatch):
    mgr, router, seq, a, commands = routed(monkeypatch)
    event(router, a, a.source, "queued")
    event(router, a, a.target, "ready", boundary=98304, forward_start=98288, ttl=5)
    assert kinds(commands)[-1] == "release_source"
    event(router, a, a.source, "source_released")
    placement = router.balancer.requests[seq.id]
    assert placement.work_prefix == 98288
    assert placement.minimum_hit == 98288
    assert mgr._seq_load[seq.id][2] == placement.prompt_tokens - 98288
    assert placement.transfer_seconds == pytest.approx(0.01 + 98304 / 2000000)
