"""Request lifecycle and snapshot tests for scheduler observability."""

from collections import deque
from queue import Queue
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.scheduler_metrics import SchedulerMetrics
from atom.model_engine.sequence import Sequence, SequenceStatus


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "atom.model_engine.scheduler_metrics.time.perf_counter", lambda: now[0]
    )
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
    snap = metrics.snapshot()
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
    assert scheduler.metrics.snapshot()["queue_time"]["sum"] == 7


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
    assert metrics.snapshot()["pd_kv_transfer"]["buckets"][-1][1] == expected
    metrics.record_forward(batch({seq.id: seq}), {seq.id: seq})
    assert metrics.snapshot()["queue_time"]["sum"] == 0.25
    assert not metrics._loads


def test_batch_counts_request_rows_and_ignores_dummy_prefill_and_empty(clock):
    metrics = SchedulerMetrics()
    seqs = {i: SimpleNamespace(id=i) for i in range(5)}
    for seq in seqs.values():
        metrics.enqueue(seq)
    metrics.record_forward(batch(seqs, decode=5, dummy=True), seqs)
    metrics.record_forward(batch({}, decode=0), {})
    assert metrics.snapshot()["queue_time"]["buckets"][-1][1] == 0
    metrics.record_forward(batch(seqs, decode=0), seqs)
    mixed = batch(seqs, decode=3)
    mixed.total_tokens_num_decode = 12  # MTP tokens do not multiply batch size
    metrics.record_forward(mixed, seqs)
    hist = metrics.snapshot()["decode_batch_size"]
    assert hist["sum"] == 3 and hist["buckets"][-1][1] == 1
    assert metrics.snapshot()["queue_time"]["buckets"][-1][1] == 5


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
    assert scheduler.metrics.snapshot()["pd_kv_transfer"]["buckets"][-1][1] == 0
    assert scheduler.engine_stats.total_requests == 0


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
    snap = scheduler.metrics.snapshot()
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


def test_histograms_preserve_rank_counts_and_do_not_reobserve_snapshots(clock):
    metrics = SchedulerMetrics()
    metrics.decode_batch_size.observe(3)
    ranks = [
        dict(
            metrics.snapshot(),
            dp_rank=rank,
            engine_role="default",
            timestamp=100,
            running=3,
            waiting=0,
            waiting_kv=0,
            kv_blocks={"used": 2, "evictable": 3, "vacant": 5, "total": 10},
        )
        for rank in (0, 1)
    ]
    exporter = AtomMetricsExporter()
    exporter.update({"enabled": True, "scheduler_metrics": ranks})
    before = samples(exporter)
    exporter.update({"enabled": True, "scheduler_metrics": ranks})
    after = samples(exporter)
    for rank in (0, 1):
        labels = (("dp_rank", str(rank)), ("engine_role", "default"))
        key = ("atom:decode_batch_size_count", labels)
        assert before[key] == after[key] == 1
        assert after[("atom:decode_batch_size_sum", labels)] == 3
    metrics.decode_batch_size.observe(7)
    assert ranks[0]["decode_batch_size"]["buckets"][-1][1] == 1


def test_engine_snapshots_reach_exporter_with_distinct_dp_ranks():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.llm_engine import LLMEngine

    rank_snapshots = {}
    for rank, size in ((0, 3), (1, 7)):
        scheduler = Scheduler(MockConfig())
        scheduler.metrics.decode_batch_size.observe(size)
        rank_snapshots[rank] = EngineUtilityHandler(
            None, Queue(), scheduler=scheduler
        ).collect_metrics()
    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(
            latest_metrics=rank_snapshots,
            get_dp_router_statistics=dict,
        )
    )
    exporter = AtomMetricsExporter()
    exporter.update(LLMEngine.get_metrics_statistics(engine))
    data = samples(exporter)
    for rank, size in ((0, 3), (1, 7)):
        labels = (("dp_rank", str(rank)), ("engine_role", "default"))
        assert data[("atom:decode_batch_size_count", labels)] == 1
        assert data[("atom:decode_batch_size_sum", labels)] == size


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
    exporter = AtomMetricsExporter()
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
    assert metrics.snapshot()["decode_batch_size"]["buckets"][-1][1] == 1
    assert metrics.snapshot()["queue_time"]["buckets"][-1][1] == 1


def test_workload_uses_dispatch_snapshot_and_observes_prompt_once(clock):
    metrics = SchedulerMetrics()
    decode = SimpleNamespace(id=1, num_prompt_tokens=1000)
    prefill = SimpleNamespace(id=2, num_prompt_tokens=10000)
    seqs = {1: decode, 2: prefill}
    for seq in seqs.values():
        metrics.enqueue(seq)
    mixed = SimpleNamespace(
        req_ids=[1, 2],
        is_dummy_run=False,
        total_seqs_num_decode=1,
        total_seqs_num_prefill=1,
        total_tokens_num_prefill=1024,
        num_cached_tokens=[1999, 8000],
        context_lens=[2000, 9024],
    )
    metrics.record_forward(mixed, seqs)
    mixed.num_cached_tokens[1] = 9024
    mixed.total_tokens_num_prefill = 976
    mixed.context_lens = [2001, 10000]
    metrics.record_forward(mixed, seqs)
    snapshot = metrics.snapshot()
    assert snapshot["prefill_request_tokens"]["sum"] == 2000
    assert snapshot["prefill_request_tokens"]["buckets"][-1][1] == 1
    assert snapshot["prefill_batch_tokens"]["sum"] == 2000
    assert snapshot["prefill_batch_tokens"]["buckets"][-1][1] == 2
    assert snapshot["decode_context_tokens"]["sum"] == 4001
    assert snapshot["decode_context_tokens"]["buckets"][-1][1] == 2
    mixed.is_dummy_run = True
    metrics.record_forward(mixed, seqs)
    assert metrics.snapshot() == snapshot


def test_worker_snapshots_include_all_pp_tp_workers_without_duplicate_queues():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.llm_engine import LLMEngine
    from atom.model_engine.gpu_metrics import GPUForwardMetrics

    worker = GPUForwardMetrics(lambda: None).snapshot()
    worker["phases"]["prefill"] = {
        "buckets": [(0.1, 1), (float("inf"), 1)],
        "sum": 0.08,
    }
    snapshots = {
        0: {
            "enabled": True,
            "requests_running": 2,
            "forward_metrics": [
                {**worker, "dp_rank": 0, "pp_rank": 0, "tp_rank": tp} for tp in (0, 1)
            ],
        },
        1: {
            "enabled": False,
            "forward_metrics": [
                {**worker, "dp_rank": 0, "pp_rank": 1, "tp_rank": tp} for tp in (0, 1)
            ],
        },
    }
    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(
            latest_metrics=snapshots, get_dp_router_statistics=dict
        )
    )
    result = LLMEngine.get_metrics_statistics(engine)
    assert result["requests_running"] == 2
    assert len(result["forward_metrics"]) == 4
    exporter = AtomMetricsExporter()
    exporter.update(result)
    counts = {
        labels: v
        for (name, labels), v in samples(exporter).items()
        if name == "atom:gpu_forward_seconds_count" and ("phase", "prefill") in labels
    }
    assert len(counts) == 4 and set(counts.values()) == {1}
    assert exporter.render() == exporter.render()
