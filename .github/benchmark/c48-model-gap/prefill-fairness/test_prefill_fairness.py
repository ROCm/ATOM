"""Fairness at the real hybrid matcher, allocator, and async scheduler seam.

Constructed metadata stands in for completed kernels. No model weights or GPU
latency are tested here; the full workload remains the performance gate.
"""

import pytest
from tests.v1.core.utils import create_requests
from tests.v1.simple_kv_offload.test_scheduler import (
    _make_hybrid_attention_mamba_scheduler,
)
from vllm.config import SchedulerConfig
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def make_hybrid_scheduler(asynchronous, with_cpu=False, factory=None):
    fixture = _make_hybrid_attention_mamba_scheduler(
        num_cpu_blocks=4096,
        num_gpu_blocks=4096,
        attention_block_size=1536,
        block_size=1536,
        scheduler_block_size=12288,
        hash_block_size=128,
        dcp_world_size=8,
    )
    config = fixture.vllm_config
    if with_cpu:
        config.kv_transfer_config.kv_connector_extra_config = {
            "cpu_bytes_to_use_per_rank": fixture.kv_cache_config.kv_cache_tensors[
                0
            ].size,
        }
    else:
        config.kv_transfer_config = None
    # The offload helper leaves the model-facing config at its 16-token default.
    # Scheduler alignment must use the runtime's resolved Mamba block size.
    config.cache_config.block_size = 1536
    config.cache_config.prefix_match_unit = 128
    config.cache_config.num_gpu_blocks = fixture.kv_cache_config.num_blocks
    config.model_config.max_model_len = 1048576
    config.scheduler_config = SchedulerConfig(
        max_num_seqs=96,
        max_num_batched_tokens=8192,
        max_model_len=1048576,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        async_scheduling=asynchronous,
        long_prefill_token_threshold=6144,
        watermark=0.0,
    )
    cls = factory or (AsyncScheduler if asynchronous else Scheduler)
    scheduler = cls(
        vllm_config=config,
        kv_cache_config=fixture.kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=12288,
        hash_block_size=128,
    )
    scheduler.use_eagle_block_drop = True
    scheduler.num_prefill_lookahead = 1
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.eagle_group_ids = {0}
    coordinator.attention_groups = [
        group._replace(use_eagle=(0 in group.group_ids))
        for group in coordinator.attention_groups
    ]
    assert coordinator.group_block_sizes == (12288, 1536)
    assert scheduler.cache_config.block_size == 1536
    assert scheduler.need_mamba_block_aligned_split
    assert scheduler.mamba_partial_cache_hit
    return scheduler


def make_request(name, prompt, token):
    (template,) = create_requests(1, num_tokens=1, max_tokens=1, block_size=128)
    return Request(
        request_id=name,
        prompt_token_ids=[token] * prompt,
        sampling_params=template.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=template._block_hasher,
    )


def seed_small_prefix(manager, small):
    pool = manager.block_pool
    for group, boundaries in [
        (0, [*range(12288, 141313, 12288), 141312]),
        (1, [141184]),
    ]:
        for tokens in boundaries:
            (block,) = pool.get_new_blocks(1)
            key = make_block_hash_with_group_id(
                small.block_hashes[tokens // 128 - 1], group
            )
            pool._insert_block_hash(key, block, 128)
            pool.free_blocks([block])
    assert manager.get_computed_blocks(small)[1] == 141184


def exercise(asynchronous, depth, enabled):
    scheduler = make_hybrid_scheduler(asynchronous)
    scheduler._c48_prefill_fairness = enabled
    manager = scheduler.kv_cache_manager
    requests = {}
    expected_work = {}
    for name, prompt, computed, token in [
        ("giant", 544101, 278016, 1),
        ("medium", 178987, 135168, 2),
    ]:
        request = make_request(name, prompt, token)
        assert manager.allocate_slots(request, computed) is not None
        request.num_computed_tokens = computed
        manager.cache_blocks(request, computed)
        request.status = RequestStatus.RUNNING
        scheduler.requests[name] = request
        scheduler.running.append(request)
        requests[name] = request
        expected_work[name] = prompt - computed
    head = make_request("waiting_head", 3072, 3)
    small = make_request("small", 142089, 4)
    seed_small_prefix(manager, small)
    for request in (head, small):
        requests[request.request_id] = request
        scheduler.add_request(request)
    expected_work.update(waiting_head=3072, small=905)
    work = dict.fromkeys(requests, 0)
    batches, pending, finished = [], [], {}

    def consume():
        step, output, rows = pending.pop(0)
        ids = list(output.num_scheduled_tokens)
        result = scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=[
                    [101] if computed + count >= requests[rid].num_prompt_tokens else []
                    for rid, computed, count in rows
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        for group in result.values():
            for item in group.outputs:
                if item.new_token_ids:
                    assert item.request_id not in finished
                    finished[item.request_id] = step

    for step in range(1, 200):
        if not scheduler.requests and not pending:
            break
        before = {rid: request.num_computed_tokens for rid, request in requests.items()}
        output = scheduler.schedule()
        for request in output.scheduled_new_reqs:
            before[request.req_id] = request.num_computed_tokens
        rows = [
            [rid, before[rid], count]
            for rid, count in output.num_scheduled_tokens.items()
        ]
        if rows:
            assert output.total_num_scheduled_tokens <= 8192
            for rid, _, count in rows:
                work[rid] += count
            batches.append((step, rows))
            pending.append((step, output, rows))
        if pending and (len(pending) >= depth or not rows):
            consume()
    else:
        raise AssertionError("Hybrid scheduler did not finish")
    assert work == expected_work
    assert set(finished) == set(requests)
    assert manager.block_pool.get_num_free_blocks() == 4095
    small_steps = [
        step for step, rows in batches if any(rid == "small" for rid, _, _ in rows)
    ]
    return {"small_steps": small_steps, "finished": finished, "work": work}


@pytest.mark.parametrize("asynchronous,depth", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("enabled", [False, True])
def test_cached_small_finishes_without_waiting_for_medium_prefill(
    asynchronous, depth, enabled
):
    result = exercise(asynchronous, depth, enabled)
    print({"asynchronous": asynchronous, "depth": depth, "enabled": enabled, **result})
    if enabled:
        assert result["finished"]["small"] <= 5
    else:
        assert result["finished"]["small"] >= 20


def seed_boundary(pool, request, group, tokens):
    (block,) = pool.get_new_blocks(1)
    key = make_block_hash_with_group_id(request.block_hashes[tokens // 128 - 1], group)
    pool._insert_block_hash(key, block, 128)
    pool.free_blocks([block])
    return block


def add_budget_owners(scheduler):
    for name, prompt, computed, token in [
        ("giant", 544101, 278016, 1),
        ("medium", 178987, 135168, 2),
    ]:
        request = make_request(name, prompt, token)
        manager = scheduler.kv_cache_manager
        assert manager.allocate_slots(request, computed) is not None
        request.num_computed_tokens = computed
        manager.cache_blocks(request, computed)
        request.status = RequestStatus.RUNNING
        scheduler.requests[name] = request
        scheduler.running.append(request)


def complete_transfers(scheduler, output):
    from tests.v1.simple_kv_offload.test_scheduler import simulate_store_completion

    meta = output.kv_connector_metadata
    if meta.store_event >= 0:
        simulate_store_completion(
            scheduler.connector.scheduler_manager, meta.store_event
        )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("finish", ["abort", "admit"])
def test_skipped_cpu_hit_releases_pins_after_retry_and_finish(asynchronous, finish):
    """A CPU hit discarded for a deeper local tail may be deferred repeatedly."""
    scheduler = make_hybrid_scheduler(asynchronous, with_cpu=True)
    scheduler._c48_prefill_fairness = True
    add_budget_owners(scheduler)
    target = make_request("deferred", 145001, 8)
    gpu_pool = scheduler.kv_cache_manager.block_pool
    for end in [*range(12288, 141313, 12288), 141440]:
        seed_boundary(gpu_pool, target, 0, end)
    seed_boundary(gpu_pool, target, 1, 141312)
    assert scheduler.kv_cache_manager.get_computed_blocks(target)[1] == 141312
    cpu = scheduler.connector.scheduler_manager
    pinned = [seed_boundary(cpu.cpu_block_pool, target, g, 136704) for g in (0, 1)]
    scheduler.add_request(target)
    follower = make_request("follower", 128, 9)
    scheduler.add_request(follower)
    for iteration in range(2):
        output = scheduler.schedule()
        assert target.request_id not in output.num_scheduled_tokens
        assert target.request_id in cpu._pending_cpu_hits
        assert all(b.ref_cnt == 1 for b in pinned)
        assert target in scheduler.waiting
        if iteration == 0:
            assert output.num_scheduled_tokens["follower"] == 128
        complete_transfers(scheduler, output)
    scheduler.finish_requests(
        ["giant", "medium", "follower"], RequestStatus.FINISHED_ABORTED
    )
    if finish == "abort":
        scheduler.finish_requests(target.request_id, RequestStatus.FINISHED_ABORTED)
    else:
        output = scheduler.schedule()
        assert output.num_scheduled_tokens[target.request_id] > 0
        assert target.num_computed_tokens >= 141312
        assert not output.kv_connector_metadata.load_cpu_blocks
        complete_transfers(scheduler, output)
        scheduler.finish_requests(target.request_id, RequestStatus.FINISHED_ABORTED)
    assert not cpu._pending_cpu_hits
    assert all(b.ref_cnt == 0 for b in pinned)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_cpu_restore_bypasses_blocked_head_and_makes_progress(asynchronous):
    """Use the actual CPU connector for admission, completion, and KV holding retry."""
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata

    scheduler = make_hybrid_scheduler(asynchronous, with_cpu=True)
    scheduler._c48_prefill_fairness = True
    add_budget_owners(scheduler)
    head = make_request("head", 3072, 7)
    target = make_request("cpu_small", 142089, 8)
    cpu = scheduler.connector.scheduler_manager
    for end in [*range(12288, 141185, 12288), 141184]:
        seed_boundary(cpu.cpu_block_pool, target, 0, end)
    seed_boundary(cpu.cpu_block_pool, target, 1, 141184)
    scheduler.add_request(head)
    scheduler.add_request(target)
    target_work = 0
    finished = False
    saw_load = False
    for step in range(1, 8):
        before = {rid: r.num_computed_tokens for rid, r in scheduler.requests.items()}
        output = scheduler.schedule()
        meta = output.kv_connector_metadata
        for r in output.scheduled_new_reqs:
            before[r.req_id] = r.num_computed_tokens
        ids = list(output.num_scheduled_tokens)
        if meta.load_event >= 0:
            assert not saw_load
            saw_load = True
            assert target.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            assert target.num_computed_tokens == 141184
            assert meta.load_cpu_blocks
        result = scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=[
                    [101]
                    if before[rid] + output.num_scheduled_tokens[rid]
                    >= scheduler.requests[rid].num_prompt_tokens
                    else []
                    for rid in ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=KVConnectorOutput(
                    finished_recving={target.request_id}
                    if meta.load_event >= 0
                    else set(),
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        completed_store_events={
                            meta.store_event: cpu._expected_worker_count
                        }
                        if meta.store_event >= 0
                        else {},
                    ),
                ),
            ),
        )
        target_work += output.num_scheduled_tokens.get(target.request_id, 0)
        for group in result.values():
            for item in group.outputs:
                if item.request_id == target.request_id and item.new_token_ids:
                    assert item.finish_reason is not None
                    finished = True
        if finished:
            break
    assert saw_load and finished
    assert target_work == 905
    assert not cpu._pending_cpu_hits
    assert target.request_id not in cpu._reqs_to_load
    print(
        {
            "cpu_restore_finish_step": step,
            "asynchronous": asynchronous,
            "fresh": target_work,
        }
    )


@pytest.mark.parametrize(
    "role,gate", [("prefill", "1"), ("decode", "0"), ("decode", None)]
)
def test_runtime_factory_records_source_and_role_gate(
    tmp_path, monkeypatch, role, gate
):
    import json

    import pd_c48_scheduler

    monkeypatch.setenv("C48_DIAG_ROLE", role)
    monkeypatch.setenv("C48_DIAG_DIR", str(tmp_path / "model-manifests"))
    for key in (
        "C48_PREFILL_TIMELINE",
        "C48_PREFILL_REQUEST_LEDGER",
        "C48_CACHE_LIFECYCLE",
    ):
        monkeypatch.delenv(key, raising=False)
    if gate is None:
        monkeypatch.delenv("C48_PREFILL_FAIRNESS", raising=False)
    else:
        monkeypatch.setenv("C48_PREFILL_FAIRNESS", gate)
    scheduler = make_hybrid_scheduler(True, factory=pd_c48_scheduler.create_scheduler)
    assert isinstance(scheduler, AsyncScheduler)
    record = json.loads(
        (tmp_path / "cache-manifests" / f"{role}-scheduler.json").read_text()
    )
    assert record["prefill_fairness_enabled"] == (gate == "1")
    assert scheduler._c48_prefill_fairness == (gate == "1")
    assert (
        record["scheduler_source_sha256"]
        == "5e29800ef253c11ce4870eb0fe20cd4a35da3030d77c0724dd63911c0059467c"
    )


@pytest.mark.parametrize("role,gate", [("decode", "1"), ("prefill", "true")])
def test_runtime_factory_rejects_invalid_role_gate(monkeypatch, role, gate):
    import pd_c48_scheduler

    monkeypatch.setenv("C48_DIAG_ROLE", role)
    monkeypatch.setenv("C48_PREFILL_FAIRNESS", gate)
    with pytest.raises(ValueError, match="requires Prefill"):
        pd_c48_scheduler.create_scheduler()


@pytest.mark.parametrize("num_spec", [0, 3])
def test_rotation_preserves_inflight_output_under_kv_pressure(monkeypatch, num_spec):
    """Reuse the upstream lossless-output test with the experimental order enabled."""
    from tests.v1.core.test_async_scheduler import (
        test_kv_pressure_preemption_with_inflight_output,
    )

    original = Scheduler.__init__

    def enable(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._c48_prefill_fairness = True

    monkeypatch.setattr(Scheduler, "__init__", enable)
    test_kv_pressure_preemption_with_inflight_output(num_spec)
