"""Exercise actual map-query observation after repeated admission retries."""

import io
import json
from types import SimpleNamespace

from pd_c48_cache_lifecycle import CacheLifecycleProbe
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def empty_probe(*, max_records=8, max_pending_requests=2048):
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=128)
    stream = io.StringIO()
    offload = SimpleNamespace(
        cpu_block_pool=pool, get_num_new_matched_tokens=lambda request, unused: (0, False)
    )
    probe = CacheLifecycleProbe(
        offload, stream, min_prompt=0, max_records=max_records,
        lookup_mode="first_last", max_pending_requests=max_pending_requests,
        watch_published_group_ids=set(),
    ).install()
    return probe, offload, stream


def request(name):
    return SimpleNamespace(
        request_id=name, num_prompt_tokens=544101,
        block_hashes=[BlockHash(b"a" * 32)], arrival_time=0,
        skip_reading_prefix_cache=False,
    )


def test_retries_preserve_later_request_and_final_changed_cache_state():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=128)
    h = BlockHash(b"a" * 32)
    key = make_block_hash_with_group_id(h, 0)
    stream = io.StringIO()

    def lookup(request, unused):
        return (128 if pool.cached_block_hash_to_block.get_one_block(key) else 0, False)

    offload = SimpleNamespace(cpu_block_pool=pool, get_num_new_matched_tokens=lookup)
    probe = CacheLifecycleProbe(
        offload, stream, min_prompt=0, max_records=8, watch_published_group_ids=set()
    )
    # Setting the policy also lets the unchanged old probe reproduce the real
    # exhaustion symptom, rather than failing on an unknown constructor option.
    probe.lookup_mode = "first_last"
    probe.install()
    (block,) = pool.get_new_blocks(1)
    pool._insert_block_hash(key, block, 128)
    early = SimpleNamespace(
        request_id="early_waiter", num_prompt_tokens=544101, block_hashes=[h],
        arrival_time=0, skip_reading_prefix_cache=False,
    )
    for _ in range(40):
        assert offload.get_num_new_matched_tokens(early, 0) == (128, False)
    pool.evict_blocks([block.block_id])
    assert offload.get_num_new_matched_tokens(early, 0) == (0, False)
    if hasattr(probe, "finish_request"):
        probe.finish_request(early.request_id)
    late = SimpleNamespace(**{**vars(early), "request_id": "late_target"})
    assert offload.get_num_new_matched_tokens(late, 0) == (0, False)
    if hasattr(probe, "finish_request"):
        probe.finish_request(late.request_id)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    lookups = [row for row in rows if row["event"] == "prefix_lookup"]
    assert any(row["request_id"] == "late_target" for row in lookups), (
        "Repeated early lookups exhausted coverage before the late target"
    )
    first, last = [row for row in lookups if row["request_id"] == early.request_id]
    assert first["lookup_result"] == [128, False]
    assert last["lookup_result"] == [0, False]
    assert last["lookup_sequence"] == 41
    assert last["groups"]["0"]["lowest_missing_hash"]["observed_history"][
        "last_remove"
    ]["reason"] == "explicit_eviction"
    assert len(lookups) == 3 and probe.records < probe.max_records
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert block.ref_cnt == 1


def test_record_budget_reserves_final_lookup_for_all_pending_requests():
    probe, offload, stream = empty_probe(max_records=5)
    for name in ("a", "b", "c", "a", "b"):
        assert offload.get_num_new_matched_tokens(request(name), 0) == (0, False)
    # A control event cannot consume the two reserved final-lookup slots.
    probe._emit({"event": "cache_reset", "reset_count": 1})
    probe.flush_pending()
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == 5
    assert [(row["request_id"], row["lookup_capture"]) for row in rows[1:]] == [
        ("a", "first"), ("b", "first"), ("a", "last"), ("b", "last")
    ]
    coverage = probe.lookup_coverage()
    assert coverage["coverage_dropped_lookup_calls"] == 1
    assert coverage["control_records_dropped"] == 1
    assert coverage["pending_requests"] == 0
    assert all(row["lookup_flush_reason"] == "probe_close" for row in rows[-2:])


def test_pending_memory_limit_is_explicit_and_recovers_after_finish():
    probe, offload, stream = empty_probe(max_pending_requests=1)
    for name in ("a", "b", "a"):
        assert offload.get_num_new_matched_tokens(request(name), 0) == (0, False)
    probe.finish_request("a")
    offload.get_num_new_matched_tokens(request("b"), 0)
    probe.finish_request("b")
    coverage = probe.lookup_coverage()
    assert coverage["pending_limit_dropped_lookup_calls"] == 1
    assert coverage["coverage_dropped_lookup_calls"] == 1
    assert coverage["observed_request_instances"] == 2
    assert coverage["peak_pending_requests"] == 1
    assert coverage["pending_requests"] == 0
    assert '"request_id": "b"' in stream.getvalue()


def test_factory_finishes_first_last_records_with_fairness_disabled(tmp_path, monkeypatch):
    import pd_c48_scheduler
    import test_prefill_fairness as boundary
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import RequestStatus

    for key, value in {
        "C48_DIAG_ROLE": "prefill", "C48_PREFILL_FAIRNESS": "0",
        "C48_CACHE_LIFECYCLE": "1", "C48_PHYSICAL_CACHE_LIFECYCLE": "1",
        "C48_CACHE_LOOKUP_MODE": "first_last",
        "C48_DIAG_DIR": str(tmp_path / "model-manifests"),
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("C48_PREFILL_REQUEST_LEDGER", "C48_PREFILL_TIMELINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(EngineCore, "shutdown", EngineCore.shutdown)
    scheduler = boundary.make_hybrid_scheduler(
        False, with_cpu=True, factory=pd_c48_scheduler.create_scheduler
    )
    assert scheduler._c48_prefill_fairness is False
    req = boundary.make_request("coverage_finish", 200100, 1)
    scheduler.add_request(req)
    for _ in range(3):
        scheduler._get_local_prefix_cache_hit(req)
        scheduler.connector.scheduler_manager.get_num_new_matched_tokens(req, 0)
    probes, close = scheduler._c48_cache_lifecycle
    assert all(req.request_id in probe.pending_lookups for probe in probes)
    scheduler.finish_requests([req.request_id], RequestStatus.FINISHED_ABORTED)
    assert all(not probe.pending_lookups for probe in probes)
    close()
    (journal,) = (tmp_path / "cache-lifecycle").glob("*.jsonl")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for kind in ("cpu", "gpu"):
        lookups = [row for row in rows if row["event"] == "prefix_lookup"
                   and row["cache_kind"] == kind]
        assert [row["lookup_sequence"] for row in lookups] == [1, 3]
        assert lookups[-1]["lookup_flush_reason"] == "request_finished"
        footer = next(row for row in rows if row["event"] == "probe_closed"
                      and row["cache_kind"] == kind)
        assert footer["lookup_coverage"]["coverage_dropped_lookup_calls"] == 0
        assert footer["physical_last_events"]["observation_errors"] == 0
