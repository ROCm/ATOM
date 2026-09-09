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
    scheduler = Scheduler(MockConfig())
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
