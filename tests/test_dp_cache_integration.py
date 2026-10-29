# SPDX-License-Identifier: MIT
"""Real manager wire format and scheduler admission, with CPU cache pools."""

import gc
import pickle
import weakref
from types import SimpleNamespace as NS

import pytest
from conftest import MockConfig
from test_dp_load_balance import _make_mgr, _route
from test_lmcache_mp import _config

from atom.kv_transfer.offload.mp import deployment
from atom.model_engine.dp_cache.state import DPCacheState
from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.prefix_hash import attach_shipped_chains, ship_prompt_hash_chain
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus

BLOCK = 128
PREFIX = 100 * 1024
TOKENS = list(range(PREFIX + 1024))


def request(*, rank=None):
    seq = Sequence(TOKENS, BLOCK, dp_session_id="session", data_parallel_rank=rank)
    ship_prompt_hash_chain(seq, BLOCK)
    return seq


def test_cache_state_round_trip_keeps_worker_changes_local():
    seq = request()
    assert seq.dp_cache is None
    seq.dp_cache = DPCacheState(migration_id=7, minimum_hit=PREFIX, prepare_timeout=3)
    frame = pickle.loads(_add_frame([seq]))
    [target] = frame[1]
    assert target.dp_cache == seq.dp_cache
    assert target.dp_cache is not seq.dp_cache
    target.dp_cache.prepare_deadline = 100
    target.dp_cache.alignment_boundary = 65536
    target.dp_cache.retry_reason = "recoverable_prefix_too_short"
    target.dp_cache.recoverable_hit = 65536
    target.dp_cache.work_revision = 4
    target.dp_cache.work_epoch = 2
    returned = pickle.loads(pickle.dumps(target))
    assert returned.dp_cache == target.dp_cache
    assert seq.dp_cache == DPCacheState(
        migration_id=7, minimum_hit=PREFIX, prepare_timeout=3
    )
    assert not any(name.startswith("dp_cache_") for name in vars(returned))


def manager(monkeypatch, *, legacy_session_affinity=False):
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "1")
    monkeypatch.setenv("ATOM_DP_CACHE_BALANCE_CONFIG", "{}")
    mgr = _make_mgr(
        2, strategy="cache_affinity", session_affinity=legacy_session_affinity
    )
    mgr._init_cache_router(
        NS(
            dp_load_balance="cache_affinity",
            enable_prefix_caching=True,
            kv_transfer_config={"kv_connector": "lmcache_mp"},
        )
    )
    seq = request(rank=0)
    assert _route(mgr, [seq]) == [0]
    mgr.cache_router.admitted(0, (seq.id, 0, 0))
    mgr._release_seq_load(seq.id, succeeded=True)
    mgr._rank_tokens[0] = 1000000
    mgr._seq_load["background"] = (0, 0, 1000000)
    return mgr


def test_v41_cache_affinity_preserves_bounded_replay(monkeypatch):
    monkeypatch.setenv("ATOM_DP_CACHE_BALANCE_CONFIG", '{"prepare_migrations": true}')
    mgr = _make_mgr(2, strategy="cache_affinity")
    config = NS(
        dp_load_balance="cache_affinity",
        enable_prefix_caching=True,
        kv_transfer_config={"kv_connector": "lmcache_mp"},
        hf_config=NS(model_type="deepseek_v41"),
        enable_encoder_swa_bounded_replay=True,
    )
    mgr._init_cache_router(config)
    assert config.enable_encoder_swa_bounded_replay
    assert mgr.cache_router.balancer.config.prepare_migrations


def test_offline_depth_cannot_bypass_cache_placement(monkeypatch):
    mgr = manager(monkeypatch)
    mgr._bp_depth = 1
    mgr._bp_dispatch_or_refill = lambda **kwargs: pytest.fail(
        "offline request bypassed cache placement"
    )
    seq = request()
    with mgr.offline_backpressure_batch():
        assert _route(mgr, [seq]) == [1]
    assert seq.id in mgr.cache_router.balancer.requests
    assert seq.dp_cache.migration_id > 0


@pytest.mark.parametrize("legacy_session_affinity", [False, True])
@pytest.mark.parametrize("partial_snapshot", [None, 64 * 1024])
def test_retry_preserves_add_hash_chain_and_does_not_finish_request(
    monkeypatch, legacy_session_affinity, partial_snapshot
):
    mgr = manager(monkeypatch, legacy_session_affinity=legacy_session_affinity)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    assert mgr._dp_session_owners == {}
    assert mgr._dp_session_prompt_tokens == {}
    frame = pickle.loads(_add_frame([seq]))
    [target] = frame[1]
    target.dp_cache.retry_reason = "recoverable_prefix_too_short"
    if partial_snapshot is not None:
        # A snapshot taken before rejection cleanup does not reserve reusable
        # PAGE/STATE. It must not override fallback to the previous owner.
        target.dp_cache.recoverable_hit = partial_snapshot
    assert seq.prompt_hash_chain is None
    sent = []
    mgr._send_control = lambda rank, payload: sent.append((rank, payload))
    assert mgr.cache_router.retry(1, target)
    assert len(sent) == 1 and sent[0][0] == 0
    _, [retry], *chains = pickle.loads(sent[0][1])
    attach_shipped_chains([retry], *chains)
    assert retry.id == seq.id
    assert list(retry.token_ids) == TOKENS
    assert retry.prompt_hash_chain is not None
    assert retry.dp_cache.migration_id == retry.dp_cache.minimum_hit == 0
    assert retry.dp_cache == DPCacheState()
    assert mgr._seq_load[seq.id][0] == 0
    assert mgr.cache_router.balancer.sessions["session"].owner == 0
    assert (
        mgr.cache_router.balancer.counters[
            "migration_fallback_recoverable_prefix_too_short"
        ]
        == 1
    )
    assert mgr.cache_router.retry(1, target)  # stale target completion
    assert len(sent) == 1
    assert (
        mgr.cache_router.balancer.counters[
            "migration_fallback_recoverable_prefix_too_short"
        ]
        == 1
    )


def test_target_ack_corrects_load_then_changes_owner(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    before = mgr._seq_load[seq.id][2]
    event = (seq.id, seq.dp_cache.migration_id, seq.dp_cache.minimum_hit)
    mgr.cache_router.admitted(0, event)  # wrong engine cannot acknowledge target bytes
    assert mgr.cache_router.balancer.sessions["session"].owner == 0
    mgr.cache_router.admitted(1, event)
    assert mgr.cache_router.balancer.sessions["session"].owner == 1
    assert mgr._seq_load[seq.id][2] == seq.num_prompt_tokens - event[2] == before
    assert seq.id not in mgr.cache_router._retry_payloads
    mgr._mark_seq_prefill_complete(seq.id)
    assert mgr._rank_tokens[1] == 0
    mgr._release_seq_load(seq.id, succeeded=True)
    assert mgr._rank_reqs == [0, 0]


def test_work_updates_are_absolute_ordered_and_fenced_by_dispatch(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    generation = seq.dp_cache.migration_id
    mgr.cache_router.update_work(1, [(seq.id, generation, 1, 12, 0)])  # not admitted
    assert mgr._seq_load[seq.id][2] != 12
    mgr.cache_router.admitted(1, (seq.id, generation, seq.dp_cache.minimum_hit))
    mgr.cache_router.update_work(1, [(seq.id, generation, 1, 128, 0)])
    assert mgr._rank_tokens[1] == 128
    mgr.cache_router.update_work(1, [(seq.id, generation, 1, 256, 0)])  # duplicate
    mgr.cache_router.update_work(0, [(seq.id, generation, 2, 256, 0)])  # wrong engine
    mgr.cache_router.update_work(1, [(seq.id, generation + 1, 2, 256, 0)])
    assert mgr._rank_tokens[1] == 128
    # Preemption may legitimately increase work, including generated context.
    mgr.cache_router.update_work(
        1, [(seq.id, generation, 2, seq.num_prompt_tokens + 100, 0)]
    )
    assert mgr._rank_tokens[1] == seq.num_prompt_tokens + 100
    mgr.cache_router.update_work(
        1, [(seq.id, generation, 1, 128, 0)]
    )  # stale pre-preemption
    assert mgr._rank_tokens[1] == seq.num_prompt_tokens + 100
    mgr._release_seq_load(seq.id)
    mgr.cache_router.update_work(1, [(seq.id, generation, 3, 128, 0)])  # cancelled
    assert mgr._rank_tokens == [1000000, 0]


def test_fallback_rejects_late_target_progress(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache.retry_reason = "recoverable_prefix_too_short"
    target.dp_cache.recoverable_hit = 65536
    mgr._send_control = lambda *args: None
    mgr.cache_router.retry(1, target)
    mgr.cache_router.admitted(0, (seq.id, 0, 65536))
    before = list(mgr._rank_tokens)
    mgr.cache_router.update_work(1, [(seq.id, target.dp_cache.migration_id, 1, 0, 0)])
    assert mgr._rank_tokens == before


def test_cancelled_migration_cannot_retry(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache.retry_reason = "preparation_timeout"
    mgr._release_seq_load(seq.id)
    mgr._send_control = lambda *args: pytest.fail("cancelled request was retried")
    assert mgr.cache_router.retry(1, target)
    assert not mgr.cache_router._retry_payloads


def test_reset_keeps_router_and_manager_on_the_same_load_records(monkeypatch):
    mgr = manager(monkeypatch)
    router = mgr.cache_router
    mgr.reset_dp_router()
    assert mgr.cache_router is router
    assert mgr.get_dp_router_statistics()["cache_balance"]["sessions"] == 0
    warm = request(rank=0)
    _route(mgr, [warm])
    router.admitted(0, (warm.id, 0, 0))
    mgr._release_seq_load(warm.id, succeeded=True)
    mgr._rank_tokens[0] = 1000000
    mgr._seq_load["background"] = (0, 0, 1000000)
    seq = request()
    assert _route(mgr, [seq]) == [1]
    router.admitted(1, (seq.id, seq.dp_cache.migration_id, seq.dp_cache.minimum_hit))
    router.update_work(1, [(seq.id, seq.dp_cache.migration_id, 1, 128, 0)])
    assert mgr._rank_tokens[1] == 128
    mgr._release_seq_load(seq.id, succeeded=True)
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [1000000, 0]


def test_real_prefill_retires_work_only_after_forward_and_rebases_after_preempt():
    sched = Scheduler(
        MockConfig(
            kv_cache_block_size=4,
            num_kvcache_blocks=128,
            max_model_len=256,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
    )
    seq = Sequence(list(range(65)), 4)
    seq.dp_cache = DPCacheState()
    sched.cache_admission.on_admitted = lambda event: None
    sched.add(seq)
    batch, active = sched.schedule()
    assert batch.total_seqs_num_prefill == 1
    assert sched.cache_admission.take_work() == []
    empty = ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True)
    sched.postprocess(list(active.values()), empty, batch=batch)
    remaining = seq.num_prompt_tokens - int(batch.num_scheduled_tokens[0])
    assert sched.cache_admission.take_work() == [
        (seq.id, 0, 1, remaining, 65 - remaining)
    ]
    sched.running.remove(seq)
    assert sched.preempt(seq)
    assert sched.cache_admission.take_work() == [(seq.id, 0, 2, 65, 0)]
    sched.cache_admission.complete_work(batch, {seq.id: seq})
    assert sched.cache_admission.take_work() == []
    new_batch, active = sched.schedule()
    sched.cache_admission.take_work()
    sched.postprocess(list(active.values()), empty, batch=new_batch)
    work = sched.cache_admission.take_work()
    assert work[-1][3] == 65 - int(new_batch.num_cached_tokens[0]) - int(
        new_batch.num_scheduled_tokens[0]
    )


def test_permanent_rejection_releases_migrated_prompt_without_another_step():
    scheduler = Scheduler(MockConfig(max_model_len=16))
    refs = []
    for _ in range(8):
        seq = Sequence(list(range(17)), scheduler.block_manager.block_size)
        seq.dp_cache = DPCacheState(migration_id=7)
        refs.append(weakref.ref(seq))
        scheduler.add(seq)
        scheduler.schedule()
        assert scheduler.take_rejected() == [seq]
        assert seq.leave_reason.startswith("unschedulable:")
        assert not scheduler.cache_admission._preparing
    del seq
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_cancelled_alignment_releases_tracking_only_with_terminal_cleanup(monkeypatch):
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=4,
            num_kvcache_blocks=128,
            max_model_len=256,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
    )
    seq = Sequence(list(range(65)), 4)
    seq.dp_cache = DPCacheState(migration_id=7, minimum_hit=48)
    scheduler.cache_admission.on_admitted = lambda event: pytest.fail(
        "alignment must not commit admission"
    )
    monkeypatch.setattr(
        scheduler.cache_admission, "_prefill_handoff", lambda seq: (32, 48)
    )
    scheduler.add(seq)
    batch, active = scheduler.schedule()
    assert batch.req_ids == [seq.id]
    empty = ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True)
    assert scheduler.postprocess(list(active.values()), empty, batch=batch) == []
    assert seq.is_partial_prefill
    batch, active = scheduler.schedule()
    assert batch.req_ids == [seq.id]
    assert seq.id in scheduler.cache_admission._preparing
    assert scheduler.abort_request(seq.id)
    # Cancellation cannot free buffers before the scheduled forward completes.
    assert seq.block_table
    assert scheduler.postprocess(list(active.values()), empty, batch=batch) == [seq]
    assert not scheduler.cache_admission._preparing
    assert not scheduler.running
    assert not seq.block_table


def test_failed_retry_send_uses_normal_terminal_cleanup(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    target = pickle.loads(_add_frame([seq]))[1][0]
    target.dp_cache.retry_reason = "preparation_timeout"

    def failed_send(*args):
        raise RuntimeError("source socket closed")

    mgr._send_control = failed_send
    assert not mgr.cache_router.retry(1, target)
    assert target.leave_reason == "cache_migration_retry_failed"
    mgr._release_seq_load(seq.id)
    assert not mgr.cache_router._retry_payloads
    assert not mgr.cache_router.balancer.requests
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [1000000, 0]


@pytest.mark.parametrize("resident,accepted", [(64 * 1024, False), (100 * 1024, True)])
def test_real_scheduler_gates_before_first_forward(resident, accepted, monkeypatch):
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=BLOCK,
            num_kvcache_blocks=4096,
            enable_prefix_caching=True,
            max_model_len=128 * 1024,
            max_num_batched_tokens=2048,
        )
    )
    bm = scheduler.block_manager
    warm = Sequence(TOKENS[:resident], BLOCK)
    assert bm.allocate(warm, bm.can_allocate(warm))
    bm.hash_blocks(warm, resident)
    bm.deallocate(warm)
    baseline_used = bm.kv.num_used
    seq = Sequence(TOKENS, BLOCK)
    seq.dp_cache = DPCacheState()
    seq.dp_cache.migration_id = 7
    seq.dp_cache.minimum_hit = PREFIX
    seq.dp_cache.prepare_timeout = 5
    events = []
    scheduler.cache_admission.on_admitted = events.append
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    rejected = scheduler.take_rejected()
    if accepted:
        assert seq.id in batch.req_ids
        assert events == [(seq.id, 7, PREFIX)]
        assert not rejected
    else:
        assert batch is None or seq.id not in batch.req_ids
        assert rejected == [seq]
        assert not events
        assert bm.kv.num_used == baseline_used


def test_short_tier_lookup_returns_to_owner_before_allocating_or_loading(monkeypatch):
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=4,
            num_kvcache_blocks=128,
            max_model_len=256,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
    )
    seq = Sequence(list(range(65)), 4)
    seq.dp_cache = DPCacheState(migration_id=7, minimum_hit=48)
    scheduler.cache_admission.on_admitted = lambda event: pytest.fail(
        "short lookup admitted"
    )

    def short_match(seq, *, skip):
        assert not skip
        scheduler.kv_connector = NS(
            lookup_recovery_upper_bound=lambda seq: 32,
            build_connector_meta=lambda: None,
        )
        seq.offload_joint.kv_prefix_tokens = 32
        return True

    monkeypatch.setattr(scheduler, "_query_connector_prefill_match", short_match)
    monkeypatch.setattr(
        scheduler.block_manager,
        "allocate",
        lambda *args: pytest.fail("useless target KV allocation"),
    )
    monkeypatch.setattr(
        scheduler,
        "_notify_connector_after_prefill_alloc",
        lambda *args: pytest.fail("useless target load preparation"),
    )
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    assert batch is None or seq.id not in batch.req_ids
    assert scheduler.take_rejected() == [seq]
    assert seq.dp_cache.retry_reason == "lookup_prefix_too_short"
    assert seq.dp_cache.recoverable_hit == 32
    assert not seq.block_table
    assert not scheduler.cache_admission._preparing


def test_multinode_mp_routes_registration_to_host_local_server():
    urls = {
        "0": "host-a:5555",
        "1": "host-a:5555",
        "2": "host-b:5555",
        "3": "host-b:5555",
    }
    cfg = _config(
        dp=2,
        dp_local=1,
        tp=2,
        dp_rank=3,
        enable_dp_attention=True,
        extra={"lmcache.mp.server_urls_by_rank": urls},
    )
    assert deployment._validate_mp_config(cfg) == (2, 1)
    assert deployment._server_urls(cfg) == ["tcp://host-b:5555"]
    cfg.parallel_config.data_parallel_rank = 0
    assert deployment._server_urls(cfg) == ["tcp://host-a:5555"]
    cfg.kv_transfer_config["kv_connector_extra_config"][
        "lmcache.mp.server_urls_by_rank"
    ].pop("2")
    with pytest.raises(ValueError, match="all 4"):
        deployment._validate_mp_config(cfg)


def test_multinode_mp_requires_explicit_local_endpoints():
    with pytest.raises(NotImplementedError, match="server_urls_by_rank"):
        deployment._validate_mp_config(_config(dp=2, dp_local=1))


def test_dp_dummy_forward_reaps_cancelled_alignment_without_more_prefill():
    pytest.importorskip("aiter", reason="DPEngineCoreProc imports AITER shared memory")
    from atom.model_engine.engine_core import DPEngineCoreProc

    seq = request()
    seq.status = SequenceStatus.ABORTED
    seq.is_partial_prefill = True
    calls = []

    def forward(*args, **kwargs):
        calls.append("forward_finished")
        return "dummy-output"

    core = NS(
        scheduler=NS(dummy_prefill=False, running=[seq]),
        runner_mgr=NS(call_func=forward),
        _postprocess=lambda seqs, output, batch: calls.append((seqs, output, batch)),
    )
    DPEngineCoreProc._execute_dummy_batch(core)
    assert calls == ["forward_finished", ((), "dummy-output", None)]


@pytest.mark.parametrize("weight", [0, 360000])
def test_router_and_scheduler_share_prefix_cost_and_completed_work(monkeypatch, weight):
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", str(weight))
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    router = mgr.cache_router
    placement = router.balancer.requests[seq.id]
    prefix = placement.minimum_hit
    remaining = seq.num_prompt_tokens - prefix
    sched = Scheduler(MockConfig())
    assert router.prefill_work_locked()[1] == sched._chunk_cost(prefix, remaining)
    generation = seq.dp_cache.migration_id
    router.admitted(1, (seq.id, generation, prefix))
    chunk = 128
    router.update_work(1, [(seq.id, generation, 1, remaining - chunk, prefix + chunk)])
    assert router.prefill_work_locked()[1] == sched._chunk_cost(
        prefix + chunk, remaining - chunk
    )
    assert sched._chunk_cost(prefix, remaining) == (
        sched._chunk_cost(prefix, chunk) + router.prefill_work_locked()[1]
    )
    # Preemption includes generated context, but restarts computation at zero.
    router.update_work(1, [(seq.id, generation, 2, seq.num_tokens + 100, 0)])
    assert router.prefill_work_locked()[1] == sched._chunk_cost(0, seq.num_tokens + 100)
    router.update_work(1, [(seq.id, generation, 1, 1, 0)])
    assert placement.work_prefix == 0
    mgr._mark_seq_prefill_complete(seq.id)
    assert router.prefill_work_locked()[1] == 0
    mgr._release_seq_load(seq.id)
    assert router.prefill_work_locked()[1] == 0


def test_equal_token_queues_route_away_from_longer_completed_prefix(monkeypatch):
    mgr = manager(monkeypatch)
    mgr.reset_dp_router()
    first, second = request(rank=0), request(rank=1)
    _route(mgr, [first, second])
    for rank, seq, prefix in [(0, first, 100000), (1, second, 0)]:
        mgr.cache_router.admitted(rank, (seq.id, 0, 0))
        mgr.cache_router.update_work(rank, [(seq.id, 0, 1, 16384, prefix)])
    assert mgr._rank_tokens == [16384, 16384]
    fresh = Sequence(TOKENS, BLOCK)
    ship_prompt_hash_chain(fresh, BLOCK)
    assert _route(mgr, [fresh]) == [1]


def test_cache_admission_corrects_predicted_work_prefix(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request(rank=0)
    _route(mgr, [seq])
    router = mgr.cache_router
    placement = router.balancer.requests[seq.id]
    assert placement.work_prefix > 0
    router.admitted(0, (seq.id, 0, 0))
    assert placement.work_prefix == 0
    assert mgr._seq_load[seq.id][2] == seq.num_prompt_tokens
