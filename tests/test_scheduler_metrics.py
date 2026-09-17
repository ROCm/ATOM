"""Request lifecycle and snapshot tests for scheduler observability."""

import gc
import weakref
from collections import deque
from queue import Queue
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from metrics_helpers import histogram_values_by_name
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.utils import floatToGoString

from atom.entrypoints.openai.metrics_setup import create_metrics_exporter
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.metrics.scheduler import SchedulerMetrics
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import DecodeScheduler, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("atom.metrics.scheduler.time.perf_counter", lambda: now[0])
    return now


def batch(seqs, decode=0, dummy=False):
    return SimpleNamespace(
        req_ids=list(seqs), total_seqs_num_decode=decode, is_dummy_run=dummy
    )


def samples(exporter):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
    }


@pytest.mark.parametrize("kind", ["ordinary", "dp", "pp_head", "pp_downstream"])
@pytest.mark.parametrize("configured,interval", [(None, 1.0), ("0.25", 0.25)])
def test_engine_push_loops_use_shared_interval(monkeypatch, kind, configured, interval):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core, pp_engine_core

    if configured is None:
        monkeypatch.delenv("ATOM_METRICS_UPDATE_INTERVAL_S", raising=False)
    else:
        monkeypatch.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", configured)
    now = [0.0]
    ticks = iter([0, interval / 2, interval, 2 * interval])
    pushed = []
    timer = SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _: None)
    monkeypatch.setattr(engine_core, "time", timer)
    monkeypatch.setattr(pp_engine_core, "time", timer)
    proc = SimpleNamespace(
        label="test",
        utility_queue=None,
        kv_transfer_enabled=False,
        _is_rl_weights_offloaded=True,
        _is_idle_rl_weights_offloaded=lambda: True,
        _drain_kv_work_at_exit=lambda: None,
        pull_and_process_input_queue=lambda: now[0] >= 2 * interval,
        _sync_dp_state=lambda unfinished, shutdown, offloaded: (False, shutdown, True),
        runner_mgr=SimpleNamespace(call_func=lambda *args, **kwargs: None),
        scheduler=SimpleNamespace(
            heartbeat_throughput=lambda _: None,
            is_finished=lambda: True,
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda: None,
        ),
        utility_handler=SimpleNamespace(
            process_queue=lambda *args: now.__setitem__(0, next(ticks)),
            push_metrics=lambda **kwargs: pushed.append((now[0], kwargs)),
        ),
    )
    loops = {
        "ordinary": engine_core.EngineCore.busy_loop,
        "dp": engine_core.DPEngineCoreProc.busy_loop,
        "pp_head": pp_engine_core.PPEngineCoreProc._head_busy_loop,
        "pp_downstream": pp_engine_core.PPEngineCoreProc._downstream_busy_loop,
    }
    loops[kind](proc)
    options = {"scheduler_metrics": False} if kind == "pp_downstream" else {}
    assert pushed == [(tick, options) for tick in (0, interval, 2 * interval)]


def test_queue_includes_kv_wait_and_counts_first_forward_once(clock):
    metrics = SchedulerMetrics()
    seq = SimpleNamespace(id=7, kv_transfer_params={"do_remote_prefill": True})
    seqs = {7: seq}
    metrics.enqueue(seq)
    clock[0] += 2
    metrics.start_kv_wait(seq)
    clock[0] += 5
    metrics.start_kv_wait(seq)  # repeated scheduling must not restart the wait
    metrics.finish_kv_wait("7", succeeded=True)
    clock[0] += 3
    metrics.record_forward(batch(seqs, decode=1), seqs)
    metrics.record_forward(batch(seqs, decode=1), seqs)
    snap = histogram_values_by_name(metrics)
    assert snap["queue_time"]["sum"] == 10
    assert snap["queue_time"]["buckets"][-1][1] == 1
    assert snap["pd_kv_transfer"]["sum"] == 5
    assert snap["decode_batch_size"]["buckets"][-1][1] == 2
    assert not metrics._loads


def test_engine_receipt_includes_time_buffered_before_scheduler_admission(
    clock, monkeypatch
):
    import pickle
    from contextlib import nullcontext

    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import engine_core as core

    seq = Sequence([1, 3, 4], block_size=4)
    # A serialized timestamp from another engine must not replace local receipt.
    SchedulerMetrics.enqueue(seq, received_at=50.0)
    frames = iter(
        [
            pickle.dumps((core.EngineCoreRequestType.ADD, [seq])),
            pickle.dumps((core.EngineCoreRequestType.SHUTDOWN, None)),
        ]
    )
    sock = SimpleNamespace(send=lambda _: None, recv=lambda **_: next(frames))
    monkeypatch.setattr(core, "make_zmq_socket", lambda *a, **kw: nullcontext(sock))
    monkeypatch.setattr(
        core.zmq,
        "Poller",
        lambda: SimpleNamespace(
            register=lambda *a: None,
            poll=lambda: [(sock, core.zmq.POLLIN)],
        ),
    )
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.label = "metrics-test"
    engine.input_queue = Queue()
    engine.process_input_sockets("input", "control")
    received = engine.input_queue.get_nowait()[0]
    assert received.queue_timing.received_at == 100.0

    clock[0] += 5
    scheduler = Scheduler(MockConfig())
    scheduler.extend([received])
    assert received.queue_timing.received_at == 100.0
    clock[0] += 2
    scheduler.metrics.record_forward(
        batch({received.id: received}), {received.id: received}
    )
    assert histogram_values_by_name(scheduler.metrics)["queue_time"]["sum"] == 7


@pytest.mark.parametrize(
    "is_pd,succeeded,expected", [(True, False, 0), (False, True, 0), (True, True, 1)]
)
def test_transfer_ignores_failed_and_offload_loads(clock, is_pd, succeeded, expected):
    metrics = SchedulerMetrics()
    seq = SimpleNamespace(id=3, kv_transfer_params={"do_remote_prefill": is_pd})
    metrics.enqueue(seq)
    metrics.start_kv_wait(seq)
    clock[0] += 0.25
    metrics.finish_kv_wait(seq.id, succeeded=succeeded)
    metrics.finish_kv_wait(seq.id, succeeded=succeeded)
    assert (
        histogram_values_by_name(metrics)["pd_kv_transfer"]["buckets"][-1][1]
        == expected
    )
    metrics.record_forward(batch({seq.id: seq}), {seq.id: seq})
    assert histogram_values_by_name(metrics)["queue_time"]["sum"] == 0.25
    assert not metrics._loads


def test_batch_counts_request_rows_and_ignores_dummy_prefill_and_empty(clock):
    metrics = SchedulerMetrics()
    seqs = {i: SimpleNamespace(id=i) for i in range(5)}
    for seq in seqs.values():
        metrics.enqueue(seq)
    metrics.record_forward(batch(seqs, decode=5, dummy=True), seqs)
    metrics.record_forward(batch({}, decode=0), {})
    assert histogram_values_by_name(metrics)["queue_time"]["buckets"][-1][1] == 0
    metrics.record_forward(batch(seqs, decode=0), seqs)
    mixed = batch(seqs, decode=3)
    mixed.total_tokens_num_decode = 12  # MTP tokens do not multiply batch size
    metrics.record_forward(mixed, seqs)
    hist = histogram_values_by_name(metrics)["decode_batch_size"]
    assert hist["sum"] == 3 and hist["buckets"][-1][1] == 1
    assert histogram_values_by_name(metrics)["queue_time"]["buckets"][-1][1] == 5


def test_scheduler_abort_releases_pending_metric_state(clock):
    scheduler = Scheduler(MockConfig(enable_prefix_caching=True))
    seq = Sequence(
        [1, 3, 4], block_size=4, kv_transfer_params={"do_remote_prefill": True}
    )
    scheduler.add(seq)
    scheduler._count_inflight_load(seq)
    scheduler._reject_aborted_waiting(seq)
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={seq.id})
    )
    assert not scheduler.metrics._loads
    assert (
        histogram_values_by_name(scheduler.metrics)["pd_kv_transfer"]["buckets"][-1][1]
        == 0
    )
    assert scheduler.engine_stats.total_requests == 0


def test_shared_cache_wait_metrics_do_not_retain_sequence():
    scheduler = DecodeScheduler(MockConfig())
    seq = Sequence([1, 2, 3, 4], block_size=4)
    scheduler.add(seq)
    assert scheduler.allocate_waiting() == [seq]
    retained = weakref.ref(seq)

    # Release the scheduler's allocation without PrefillDone. Metrics must
    # not keep the sequence alive after its actual owner has released it.
    scheduler.prefill_waiting.pop(seq.id)
    scheduler.block_manager.deallocate(seq)
    del seq
    gc.collect()
    assert retained() is None


def test_shared_cache_wait_preserves_queue_metrics(clock):
    scheduler = DecodeScheduler(MockConfig())
    seq = Sequence([1, 2, 3, 4], block_size=4)
    scheduler.add(seq)
    clock[0] += 2
    assert scheduler.allocate_waiting() == [seq]
    handler = EngineUtilityHandler(
        runner_mgr=None, output_queue=Queue(), label="Decode", scheduler=scheduler
    )
    assert handler.collect_metrics()["scheduler_metrics"]["waiting_kv"] == 1

    clock[0] += 3
    scheduler.on_prefill_done(seq.id, 4, 5)
    scheduled, seqs = scheduler.schedule()
    scheduler.metrics.record_forward(scheduled, seqs)
    snap = histogram_values_by_name(scheduler.metrics)
    assert snap["queue_time"]["sum"] == 5
    assert snap["queue_time"]["buckets"][-1][1] == 1
    assert snap["pd_kv_transfer"]["buckets"][-1][1] == 0
    assert handler.collect_metrics()["scheduler_metrics"]["waiting_kv"] == 0


def test_scheduler_success_closes_timer_at_completion_before_next_schedule(clock):
    scheduler = Scheduler(MockConfig())
    seq = Sequence(
        [1, 3, 4], block_size=4, kv_transfer_params={"do_remote_prefill": True}
    )
    scheduler.add(seq)
    scheduler._count_inflight_load(seq)
    clock[0] += 0.5
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={str(seq.id)})
    )
    clock[0] += 2
    scheduler._uncount_inflight_load(seq)
    scheduler.metrics.record_forward(batch({seq.id: seq}, decode=1), {seq.id: seq})
    snap = histogram_values_by_name(scheduler.metrics)
    assert snap["pd_kv_transfer"]["sum"] == 0.5
    assert snap["queue_time"]["sum"] == 2.5


def test_pool_partition_and_waiting_queue_exclusion():
    scheduler = Scheduler(MockConfig(enable_prefix_caching=True))
    pool = scheduler.block_manager.kv
    used = pool.allocate(0)
    cached = pool.allocate(1)
    import array

    pool.publish(cached.block_id, 123, array.array("i", [1, 2, 3, 4]))
    pool.free(cached.block_id)
    seq = Sequence([1, 3, 4], block_size=4)
    scheduler.add(seq)
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    handler = EngineUtilityHandler(None, Queue(), scheduler=scheduler)
    snapshot = handler.collect_metrics()
    assert snapshot["kv_blocks_used"] == 1
    assert snapshot["kv_blocks_evictable"] == 1
    assert snapshot["kv_blocks_vacant"] == pool.num_blocks - 2
    assert snapshot["scheduler_metrics"]["waiting"] == 0
    assert snapshot["scheduler_metrics"]["waiting_kv"] == 1
    pool.free(used.block_id)


def test_cache_tiers_preserve_admitted_reuse_through_snapshots():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.llm_engine import LLMEngine

    ranks = {}
    for rank, (gpu, offload) in enumerate(((6000, 3000), (1000, 0))):
        scheduler = Scheduler(MockConfig(enable_prefix_caching=True))
        scheduler.engine_stats.update_cache(
            gpu, 10000, gpu, gpu, 9500, num_offload_tokens=offload
        )
        ranks[rank] = EngineUtilityHandler(
            None, Queue(), scheduler=scheduler
        ).collect_metrics()
    # Transfer volume and PP copies must not enter admitted cache accounting.
    ranks[0]["offload"] = {"loaded_tokens": 99999}
    ranks[2] = {"enabled": False, "cache": ranks[0]["cache"]}
    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(latest_metrics=ranks, get_dp_router_statistics=dict)
    )
    exporter, _, _ = create_metrics_exporter()
    for _ in range(2):
        exporter.update(LLMEngine.get_metrics_statistics(engine))
        values = samples(exporter)
        assert values[("atom:prefix_cache_cached_tokens_total", ())] == 7000
        assert values[("atom:prefix_cache_offload_tokens_total", ())] == 3000
        assert values[("atom:prefix_cache_full_tokens_total", ())] == 20000
        assert values[("atom:lmcache_loaded_tokens_total", ())] == 99999
    # No-LMCache is a measured zero, while an old snapshot is unknown.
    engine.core_mgr.latest_metrics = {1: ranks[1]}
    exporter.update(LLMEngine.get_metrics_statistics(engine))
    assert samples(exporter)[("atom:prefix_cache_offload_tokens_total", ())] == 0
    del ranks[1]["cache"]["offload_tokens"]
    engine.core_mgr.latest_metrics = ranks
    exporter.update(LLMEngine.get_metrics_statistics(engine))
    assert ("atom:prefix_cache_offload_tokens_total", ()) not in samples(exporter)


@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_pd_consumer_counts_its_own_prefix_once_after_transfer(
    dcp, warm_cache, monkeypatch
):
    scheduler = Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            num_kvcache_blocks=50,
            decode_context_parallel_size=dcp,
        )
    )
    scheduler.kv_connector = SimpleNamespace(
        is_producer=False,
        is_offload=False,
        build_connector_meta=lambda: None,
    )
    bm = scheduler.block_manager
    if dcp > 1:
        # Keep the real cache matching/publication path, with virtual-block
        # allocation independent of the GPU-only DCP kernel module.
        monkeypatch.setattr(
            bm,
            "num_pool_blocks",
            lambda length: (length + bm.hash_block_size - 1) // bm.hash_block_size,
        )
    prompt = list(range(100, 100 + 4 * bm.hash_block_size))
    if warm_cache:
        seed = Sequence(prompt, block_size=4)
        assert bm.allocate(seed, bm.can_allocate(seed))
        bm.register_received_prefix(seed)
        bm.deallocate(seed)
    seq = Sequence(prompt, block_size=4, kv_transfer_params={"first_token_id": 999})
    # This API field came from P and must never become D's cache numerator.
    seq.prefix_cache_hit_tokens = len(prompt) - 1
    scheduler.add(seq)
    assert bm.allocate(seq, bm.can_allocate(seq))
    expected_hit = 3 * bm.hash_block_size if warm_cache else 0
    assert seq.num_cached_tokens == expected_hit
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    scheduler._count_inflight_load(seq)
    idle, idle_seqs = scheduler.schedule()
    assert not idle_seqs and idle.total_seqs_num == 0
    assert scheduler.engine_stats.total_requests == 0

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={seq.id})
    )
    scheduled, _ = scheduler.schedule()
    assert scheduled.total_seqs_num_decode == 1
    assert scheduled.total_seqs_num_prefill == 0
    assert seq.num_tokens == len(prompt) + 1
    stats = scheduler.engine_stats.cache_statistics()
    assert stats["requests"] == 1
    assert stats["full_tokens"] == len(prompt)
    assert stats["cached_tokens"] == expected_hit
    assert stats["offload_tokens"] == 0
    assert seq.prefix_cache_hit_tokens == len(prompt) - 1
    # Further decode steps and repeated snapshots do not count another hit.
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={seq.id})
    )
    scheduler.schedule()
    assert scheduler.engine_stats.cache_statistics() == stats


def test_failed_pd_transfer_does_not_record_successful_cache_admission():
    scheduler = Scheduler(MockConfig(enable_prefix_caching=True))
    scheduler.kv_connector = SimpleNamespace(
        is_producer=False,
        is_offload=False,
        build_connector_meta=lambda: None,
        get_num_new_matched_tokens=lambda seq: (0, False),
        update_state_after_alloc=lambda seq: None,
    )
    seq = Sequence(
        list(range(100, 116)), block_size=4, kv_transfer_params={"first_token_id": 999}
    )
    seq.prefix_cache_hit_tokens = 15
    scheduler.add(seq)
    bm = scheduler.block_manager
    assert bm.allocate(seq, bm.can_allocate(seq))
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    scheduler._count_inflight_load(seq)
    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    assert scheduler.engine_stats.total_requests == 0
    # At the failure/success branch, choose fallback without counting a PD hit.
    # Local prefill admission is covered separately; block recovery is not part
    # of this metrics change.
    assert scheduler._resolve_waiting_remote_kv(seq, deque()) is False
    assert scheduler.engine_stats.total_requests == 0
    assert scheduler.engine_stats.total_full_tokens == 0
    assert scheduler.engine_stats.total_cached_tokens == 0
    assert scheduler.engine_stats.total_offload_tokens == 0
    assert seq.num_tokens == 16  # P's first output token was not injected.


def test_pp_head_records_once_when_dispatching_a_real_forward(clock):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.pp_engine_core import PPEngineCoreProc

    metrics = SchedulerMetrics()
    seqs = {7: SimpleNamespace(id=7)}
    metrics.enqueue(seqs[7])
    scheduled = batch(seqs, decode=1)
    scheduled.produces_output = lambda: True
    dispatched = []
    proposals = iter([(scheduled, seqs)])
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.pp_size = 1
    proc.kv_transfer_enabled = False
    proc._in_flight = deque()
    proc.scheduler = SimpleNamespace(
        metrics=metrics,
        schedule=lambda: next(proposals),
        take_rejected=list,
        mark_pp_inflight=lambda _: None,
    )
    proc.runner_mgr = SimpleNamespace(
        call_func=lambda name, *a, **k: dispatched.append(name)
    )
    proc.pp_transport = SimpleNamespace(
        send_metadata=lambda _: None, recv_tokens=lambda **_: None
    )
    proc._poll_kv_transfer_progress = lambda: None
    proc._pp_head_step()
    proc._pp_head_step()  # full pipeline: collect only; no second submission
    assert dispatched == ["forward", "flush_pp_send"]
    assert histogram_values_by_name(metrics)["decode_batch_size"]["buckets"][-1][1] == 1
    assert histogram_values_by_name(metrics)["queue_time"]["buckets"][-1][1] == 1


def test_prefill_context_records_full_prompt_once_and_chunk_batch_totals(monkeypatch):
    import numpy as np
    from prometheus_client import CollectorRegistry, generate_latest

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import SequenceType

    monkeypatch.delenv("ATOM_ENABLE_METRICS_DEVICE_TIMER", raising=False)
    registry = CollectorRegistry()
    metrics = SchedulerMetrics(registry=registry)
    decode = Sequence([1, 2], block_size=4)
    decode.type = SequenceType.DECODE
    first = Sequence(list(range(12)), block_size=4, request_id="first")
    second = Sequence(list(range(20)), block_size=4, request_id="second")
    first.type = second.type = SequenceType.PREFILL
    first.num_cached_tokens, second.num_cached_tokens = 4, 8
    seqs = {seq.id: seq for seq in (decode, first, second)}
    for seq in seqs.values():
        metrics.enqueue(seq)
    monkeypatch.setattr("atom.metrics.scheduler.time.time", lambda: 100.25)
    scheduled = ScheduledBatch(
        seqs,
        [1, 3, 4],
        8,
        total_tokens_num_decode=1,
        total_tokens_num_prefill=7,
        total_seqs_num=3,
        total_seqs_num_decode=1,
        total_seqs_num_prefill=2,
    )
    scheduled.context_lens = np.append(scheduled.context_lens, 999999)
    # Scheduling can advance the live sequences before dispatch.
    first.num_cached_tokens = 7
    second.num_cached_tokens = 12
    scheduled.is_dummy_run = True
    metrics.record_forward(scheduled, seqs)
    assert metrics.prefill_request_context_tokens.collect()[0].samples == []
    scheduled.is_dummy_run = False
    metrics.record_forward(scheduled, seqs)
    initial = histogram_values_by_name(metrics)
    assert initial["prefill_context_tokens"]["sum"] == 19  # 7 + 12
    assert initial["prefill_context_tokens"]["buckets"][-1][1] == 1
    requests = metrics.prefill_request_context_tokens.collect()[0].samples
    assert [(s.labels["request_id"], s.value) for s in requests] == [
        ("first", 12),
        ("second", 20),
    ]
    assert all(s.labels["started_at"] == "100.25" for s in requests)
    assert (
        "# TYPE atom:prefill_request_context_tokens gauge"
        in generate_latest(registry).decode()
    )
    assert initial["prefill_request_tokens"]["sum"] == 20  # (12 - 4) + (20 - 8)
    assert initial["prefill_request_tokens"]["buckets"][-1][1] == 2
    assert initial["prefill_batch_tokens"]["sum"] == 7
    assert initial["decode_context_tokens"]["sum"] == 2

    tail = ScheduledBatch(
        {first.id: first},
        [5],
        5,
        total_tokens_num_prefill=5,
        total_seqs_num=1,
        total_seqs_num_prefill=1,
    )
    metrics.record_forward(tail, seqs)
    final = histogram_values_by_name(metrics)
    assert final["prefill_request_tokens"] == initial["prefill_request_tokens"]
    assert final["prefill_batch_tokens"]["sum"] == 12  # 7 + 5
    assert final["prefill_batch_tokens"]["buckets"][-1][1] == 2
    assert final["prefill_context_tokens"]["sum"] == 31  # 19 + 12
    assert final["prefill_context_tokens"]["buckets"][-1][1] == 2
    assert metrics.prefill_request_context_tokens.collect()[0].samples == requests
    tail.is_dummy_run = True
    metrics.record_forward(tail, seqs)
    assert histogram_values_by_name(metrics) == final

    # Re-admission after preemption must retain the original prompt sample.
    first.is_partial_prefill = True
    owner = SimpleNamespace(
        _is_preemptable=lambda seq: True,
        total_preemptions=0,
        _partial_prefill_count=1,
        spec_decode_local=False,
        _connector_release_stalled_save=lambda seq: None,
        block_manager=SimpleNamespace(deallocate=lambda seq: None),
        waiting=deque(),
    )
    assert Scheduler.preempt(owner, first)
    metrics.enqueue(first)
    tail.is_dummy_run = False
    metrics.record_forward(tail, seqs)
    assert metrics.prefill_request_context_tokens.collect()[0].samples == requests

    third = Sequence(list(range(30)), block_size=4, request_id="third")
    third.type = SequenceType.PREFILL
    metrics.enqueue(third)
    seqs[third.id] = third
    next_batch = ScheduledBatch(
        {first.id: first, third.id: third},
        [5, 4],
        9,
        total_tokens_num_prefill=9,
        total_seqs_num=2,
        total_seqs_num_prefill=2,
    )
    metrics.record_forward(next_batch, seqs)
    final_requests = metrics.prefill_request_context_tokens.collect()[0].samples
    assert final_requests[:2] == requests
    assert len(final_requests) == 3
    assert final_requests[-1].labels["request_id"] == "third"
    assert final_requests[-1].value == 30


def test_decode_request_context_gauge_records_first_dispatch_once(monkeypatch):
    from prometheus_client import CollectorRegistry, generate_latest

    registry = CollectorRegistry()
    metrics = SchedulerMetrics(registry=registry)
    # Live sequence lengths deliberately differ from the dispatched context snapshot.
    seqs = {
        i: Sequence([1, 2], block_size=4, id=i, request_id=f"request-{i}")
        for i in (1, 2, 3)
    }
    for i in (1, 2):
        metrics.enqueue(seqs[i])
    monkeypatch.setattr("atom.metrics.scheduler.time.time", lambda: 100.25)
    mixed = SimpleNamespace(
        req_ids=[1, 2, 3],
        is_dummy_run=False,
        total_seqs_num_decode=2,
        total_seqs_num_prefill=1,
        total_tokens_num_prefill=50,
        context_lens=[1000, 9000, 300, 999999],
    )
    metrics.record_forward(mixed, seqs)
    first = metrics.decode_request_context_tokens.collect()[0].samples
    assert [(s.labels["request_id"], s.value) for s in first] == [
        ("request-1", 1000),
        ("request-2", 9000),
    ]
    assert all(s.labels["started_at"] == "100.25" for s in first)
    assert (
        "# TYPE atom:decode_request_context_tokens gauge"
        in generate_latest(registry).decode()
    )

    # Exercise the real preemption reset and re-admission paths. Neither should
    # reset the first-decode marker or replace the recorded context length.
    seqs[1].is_partial_prefill = False
    owner = SimpleNamespace(
        _is_preemptable=lambda seq: True,
        total_preemptions=0,
        spec_decode_local=False,
        _connector_release_stalled_save=lambda seq: None,
        block_manager=SimpleNamespace(deallocate=lambda seq: None),
        waiting=deque(),
    )
    assert Scheduler.preempt(owner, seqs[1])
    metrics.enqueue(seqs[1])
    mixed.req_ids = [1]
    mixed.total_seqs_num_decode = 1
    mixed.total_seqs_num_prefill = 0
    mixed.context_lens = [1001]
    metrics.record_forward(mixed, seqs)
    assert metrics.decode_request_context_tokens.collect()[0].samples == first
    assert histogram_values_by_name(metrics)["decode_context_tokens"]["sum"] == 11001

    # A new request arrives while the first one is still decoding.
    metrics.enqueue(seqs[3])
    mixed.req_ids, mixed.total_seqs_num_decode = [1, 3], 2
    mixed.context_lens = [1002, 300]
    mixed.is_dummy_run = True
    metrics.record_forward(mixed, seqs)
    assert metrics.decode_request_context_tokens.collect()[0].samples == first
    mixed.is_dummy_run = False
    metrics.record_forward(mixed, seqs)
    final = metrics.decode_request_context_tokens.collect()[0].samples
    assert len(final) == 3
    assert final[-1].labels["request_id"] == "request-3" and final[-1].value == 300


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    "rows,context", [(128, 131072), (512, 1048576), (1024, 8388608)]
)
def test_large_batch_contexts_have_finite_buckets_through_exposition(
    phase, rows, context
):
    exporter, _, _ = create_metrics_exporter()
    metrics = SchedulerMetrics(engine_role=phase, registry=exporter.registry)
    seqs = {i: SimpleNamespace(id=i) for i in range(rows)}
    scheduled = SimpleNamespace(
        req_ids=list(seqs),
        is_dummy_run=False,
        total_seqs_num_decode=rows if phase == "decode" else 0,
        total_seqs_num_prefill=rows if phase == "prefill" else 0,
        total_tokens_num_prefill=rows if phase == "prefill" else 0,
        context_lens=[context] * rows,
    )
    metrics.record_forward(scheduled, seqs)
    snapshot = histogram_values_by_name(metrics)
    total = rows * context
    histogram = snapshot[f"{phase}_context_tokens"]
    assert histogram["sum"] == total
    assert dict(histogram["buckets"])[8388608] == 0
    # Finite buckets must contain the observations; otherwise Prometheus
    # clips any quantile in +Inf to the highest finite bound.
    assert dict(histogram["buckets"])[total] == 1
    values = samples(exporter)
    labels = (("dp_rank", "0"), ("engine_role", phase))
    bucket_labels = (*labels, ("le", floatToGoString(total)))
    assert values[(f"atom:{phase}_context_tokens_bucket", bucket_labels)] == 1
    assert values[(f"atom:{phase}_context_tokens_sum", labels)] == total
    assert values[(f"atom:{phase}_context_tokens_count", labels)] == 1
