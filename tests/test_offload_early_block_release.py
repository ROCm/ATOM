# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests for M3/dense PAGE early block release."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import (
    BlockGPUConnector,
    _TransferChunk,
    _TransferGroup,
)
from atom.kv_transfer.offload.dense.connector import (
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.model_engine.block_manager import BlockManager


def _config(role="kv_producer", *, block_size=4):
    return SimpleNamespace(
        kv_transfer_config={"kv_role": role},
        kv_cache_block_size=block_size,
        decode_context_parallel_size=1,
        tensor_parallel_size=1,
    )


def _early_release_scheduler(monkeypatch, role="kv_producer", *, chunk_size=8):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=chunk_size),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    scheduler = DenseOffloadScheduler(_config(role))
    assert scheduler._early_release is True
    return scheduler


def _seq(req_id, num_prompt_tokens, num_blocks):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=0,
        num_prompt_tokens=num_prompt_tokens,
        token_ids=list(range(num_prompt_tokens)),
        block_table=list(range(num_blocks)),
    )


def _source_safe(operation, *ranges):
    return ConnectorCompletion(
        DENSE_PAGE_SOURCE_SAFE_CHANNEL,
        SaveSourceGroupId(operation, tuple(ranges)),
        True,
    )


def _store_terminal(operation, succeeded=True):
    return ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, succeeded)


def _finish_and_lease(scheduler, seq):
    scheduler.request_finished(seq)
    protected = scheduler.protected_block_ids(seq)
    assert protected is not None
    scheduler.activate_block_leases(seq, protected)
    return protected


class TestBlockPoolLeaseOwnership:
    def test_12_block_request_releases_b9_through_b12_immediately(self, seq_factory):
        bm = BlockManager(MockConfig(num_kvcache_blocks=16, kv_cache_block_size=1))
        seq = seq_factory(list(range(12)))
        bm.allocate(seq)
        protected = frozenset(seq.block_table[:8])
        unprotected = list(seq.block_table[8:])

        bm.deallocate_partial(seq, protected)

        assert bm.kv.num_used == 8
        assert not seq.block_table
        for block_id in protected:
            assert bm.kv.block(block_id).ref_count == 1
        for block_id in unprotected:
            assert bm.kv.block(block_id).ref_count == 0

        bm.free_leased_blocks(protected)
        assert bm.kv.num_used == 0


class TestSourceSafeBoundary:
    def test_worker_reports_source_safe_and_store_failure_separately(self):
        worker = DenseOffloadConnector.__new__(DenseOffloadConnector)
        worker._early_release = True
        worker._lock = threading.Lock()
        worker._done_save = set()
        worker._done_load = set()
        worker._failed_load = set()
        worker._connector_completions = set()
        operation = SaveOperationId("r", 6)
        identity = SaveSourceGroupId(operation, ((0, 8),))

        worker._source_group_safe(identity)
        worker._record_store_terminal(
            SimpleNamespace(save_operation=operation, req_id="r"), False
        )
        output = worker.get_finished()

        assert output.finished_saving == {operation}
        assert output.connector_completions == {
            ConnectorCompletion(DENSE_PAGE_SOURCE_SAFE_CHANNEL, identity, True),
            ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, False),
        }

    def test_gpu_connector_reports_only_after_staging_group_fence(self):
        codec = SimpleNamespace(device=torch.device("cpu"), bytes_per_block=4)
        reported = []
        order = []

        def report(identity):
            order.append("source_safe")
            reported.append(identity)

        connector = BlockGPUConnector(
            codec,
            4,
            chunk_size=8,
            source_safe_callback=report,
        )
        memory_obj = SimpleNamespace()
        group = _TransferGroup(
            chunks=[
                _TransferChunk(memory_obj, 0, 8, [10, 11], torch.empty(0), 8),
                _TransferChunk(memory_obj, 8, 16, [12, 13], torch.empty(0), 8),
            ],
            nbytes=16,
        )
        state = SimpleNamespace(pack_stream=None, copy_stream=None)
        connector._prepare_transfer = lambda *args, **kwargs: (state, [group])

        def run_pipeline(*_args, stage_b_enqueued=None, **_kwargs):
            order.append("fenced")
            stage_b_enqueued(group, None)

        connector._run_staged_pipeline = run_pipeline
        operation = SaveOperationId("r", 7)

        with connector.track_save_source(operation):
            connector.batched_from_gpu([memory_obj], [0], [8])
            order.append("returned")

        assert order == ["fenced", "source_safe", "source_safe", "returned"]
        assert reported == [
            SaveSourceGroupId(operation, ((0, 8),)),
            SaveSourceGroupId(operation, ((8, 16),)),
        ]

    def test_save_staging_is_tail_to_head_with_exact_object_range_mapping(
        self, monkeypatch
    ):
        monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "1")
        codec = SimpleNamespace(device=torch.device("cpu"), bytes_per_block=1)
        reported = []
        connector = BlockGPUConnector(
            codec,
            block_size=1,
            chunk_size=2,
            source_safe_callback=reported.append,
        )
        memory_objs = [
            SimpleNamespace(name=f"m{index}", tensor=torch.empty(2, dtype=torch.uint8))
            for index in range(4)
        ]
        starts = [0, 2, 4, 6]
        ends = [2, 4, 6, 8]
        scheduled = []

        def capture_pipeline(_state, groups, **kwargs):
            stage_b_enqueued = kwargs["stage_b_enqueued"]
            for group in groups:
                assert len(group.chunks) == 1
                chunk = group.chunks[0]
                scheduled.append(
                    (chunk.start, chunk.end, chunk.block_ids, chunk.memory_obj)
                )
                stage_b_enqueued(group, None)

        connector._run_staged_pipeline = capture_pipeline
        operation = SaveOperationId("tail-first", 1)

        with connector.track_save_source(operation):
            connector.batched_from_gpu(
                memory_objs,
                starts,
                ends,
                block_ids=list(range(1, 9)),
            )

        assert scheduled == [
            (6, 8, [7, 8], memory_objs[3]),
            (4, 6, [5, 6], memory_objs[2]),
            (2, 4, [3, 4], memory_objs[1]),
            (0, 2, [1, 2], memory_objs[0]),
        ]
        assert reported == [
            SaveSourceGroupId(operation, ((6, 8),)),
            SaveSourceGroupId(operation, ((4, 6),)),
            SaveSourceGroupId(operation, ((2, 4),)),
            SaveSourceGroupId(operation, ((0, 2),)),
        ]
        # CacheEngine.store retains these lists for its later key/object
        # batched_put. The connector must not mutate them while reversing only
        # the actual GPU-to-staging schedule.
        assert memory_objs == [
            scheduled[3][3],
            scheduled[2][3],
            scheduled[1][3],
            scheduled[0][3],
        ]
        assert starts == [0, 2, 4, 6]
        assert ends == [2, 4, 6, 8]

    def test_load_staging_remains_head_to_tail(self, monkeypatch):
        monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "1")
        codec = SimpleNamespace(device=torch.device("cpu"), bytes_per_block=1)
        connector = BlockGPUConnector(codec, block_size=1, chunk_size=2)
        memory_objs = [
            SimpleNamespace(tensor=torch.empty(2, dtype=torch.uint8)) for _ in range(4)
        ]
        scheduled = []

        def capture_pipeline(_state, groups, **_kwargs):
            scheduled.extend(
                (chunk.start, chunk.end, chunk.block_ids)
                for group in groups
                for chunk in group.chunks
            )

        connector._run_staged_pipeline = capture_pipeline
        connector.batched_to_gpu(
            memory_objs,
            [0, 2, 4, 6],
            [2, 4, 6, 8],
            block_ids=list(range(1, 9)),
        )

        assert scheduled == [
            (0, 2, [1, 2]),
            (2, 4, [3, 4]),
            (4, 6, [5, 6]),
            (6, 8, [7, 8]),
        ]

    def test_pipeline_exception_never_claims_source_safe(self):
        codec = SimpleNamespace(device=torch.device("cpu"), bytes_per_block=4)
        reported = []
        connector = BlockGPUConnector(
            codec,
            4,
            chunk_size=8,
            source_safe_callback=reported.append,
        )
        group = _TransferGroup([], 0)
        state = SimpleNamespace(pack_stream=None, copy_stream=None)
        connector._prepare_transfer = lambda *args, **kwargs: (state, [group])

        def fail(*_args, **_kwargs):
            raise RuntimeError("copy failed")

        connector._run_staged_pipeline = fail
        with (
            connector.track_save_source(SaveOperationId("r", 8)),
            pytest.raises(RuntimeError, match="copy failed"),
        ):
            connector.batched_from_gpu([object()], [0], [8])
        assert reported == []


class TestIncrementalLeaseRelease:
    def test_b1_b2_release_while_b3_b8_remain_protected(self, monkeypatch):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(100, num_prompt_tokens=48, num_blocks=12)
        scheduler.update_state_after_alloc(seq)

        seq.num_cached_tokens = 8
        op1 = scheduler.build_connector_meta().requests[0].save_operation
        seq.num_cached_tokens = 32
        protected = _finish_and_lease(scheduler, seq)
        assert protected == frozenset(range(8))

        assert scheduler.connector_completion(_source_safe(op1, (0, 8))) is None
        assert scheduler.take_source_safe_releases() == [frozenset({0, 1})]
        assert scheduler.protected_block_ids(seq) == frozenset(range(2, 8))

        assert scheduler.connector_completion(_store_terminal(op1)) is True
        op2 = scheduler.build_connector_meta().requests[0].save_operation
        assert scheduler.connector_completion(_source_safe(op2, (8, 32))) is None
        assert scheduler.take_source_safe_releases() == [frozenset(range(2, 8))]
        scheduler.connector_completion(_store_terminal(op2))
        assert scheduler.protected_block_ids(seq) == frozenset()

    def test_final_pending_save_survives_request_block_table_clear(self, monkeypatch):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(101, num_prompt_tokens=32, num_blocks=8)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 32
        protected = _finish_and_lease(scheduler, seq)
        assert protected == frozenset(range(8))
        assert scheduler.has_pending_work() is True

        seq.block_table.clear()
        seq.num_cached_tokens = 0
        request = scheduler.build_connector_meta().requests[0]
        assert request.block_ids == list(range(8))
        assert request.token_ids == list(range(32))


class TestTPQuorum:
    def test_one_incomplete_rank_prevents_logical_group_release(self, monkeypatch):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(200, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)
        completion = _source_safe(operation, (0, 8))
        aggregator = KVOutputAggregator(world_size=2)

        first = aggregator.aggregate(
            [KVConnectorOutput(connector_completions={completion}), KVConnectorOutput()]
        )
        assert scheduler.process_completions(first).finished_saving == set()
        assert scheduler.take_source_safe_releases() == []

        second = aggregator.aggregate(
            [KVConnectorOutput(), KVConnectorOutput(connector_completions={completion})]
        )
        assert scheduler.process_completions(second).finished_saving == set()
        assert scheduler.take_source_safe_releases() == [frozenset({0, 1})]


class TestStoreOutcomeSeparation:
    def test_commit_failure_after_source_safe_releases_without_success_stats(
        self, monkeypatch
    ):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(300, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)

        scheduler.connector_completion(_source_safe(operation, (0, 8)))
        assert scheduler.take_source_safe_releases() == [frozenset({0, 1})]
        scheduler.connector_completion(_store_terminal(operation, succeeded=False))

        assert scheduler.total_save_requests == 0
        assert scheduler.total_saved_tokens == 0
        assert scheduler.blocks_waiting_for_store() == 0

    def test_source_safe_blocks_are_observable_while_store_is_pending(
        self, monkeypatch
    ):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(301, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)

        scheduler.connector_completion(_source_safe(operation, (0, 8)))
        assert scheduler.get_statistics()["blocks_waiting_for_store"] == 2
        scheduler.connector_completion(_store_terminal(operation))
        assert scheduler.get_statistics()["blocks_waiting_for_store"] == 0
        assert scheduler.total_save_requests == 1
        assert scheduler.total_saved_tokens == 8


class TestNoDoubleFree:
    def test_timeout_drops_an_unemitted_finished_save_before_freeing(self, monkeypatch):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(399, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        _finish_and_lease(scheduler, seq)
        scheduler._save_lease_at[id(seq)] = time.monotonic() - 10

        assert scheduler.reclaim_stale_leases(1) == [frozenset({0, 1})]
        assert str(seq.id) not in scheduler._save_tracker
        assert scheduler.has_pending_work() is False

    def test_late_completions_after_timeout_reclaim_are_noops(self, monkeypatch):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(400, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)
        scheduler._save_lease_at[id(seq)] = time.monotonic() - 10

        assert scheduler.reclaim_stale_leases(1) == [frozenset({0, 1})]
        scheduler.connector_completion(_source_safe(operation, (0, 8)))
        scheduler.connector_completion(_store_terminal(operation))
        assert scheduler.take_source_safe_releases() == []
        assert scheduler.total_abnormal_lease_reclaims == 2
        assert scheduler.total_save_requests == 0

    def test_duplicate_and_stale_source_completions_do_not_double_release(
        self, monkeypatch
    ):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(401, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)
        completion = _source_safe(operation, (0, 8))

        scheduler.connector_completion(completion)
        scheduler.connector_completion(completion)
        scheduler.connector_completion(
            _source_safe(SaveOperationId(seq.id, 999), (0, 8))
        )
        assert scheduler.take_source_safe_releases() == [frozenset({0, 1})]
        assert scheduler.take_source_safe_releases() == []

    def test_abandon_then_late_store_completion_does_not_release_twice(
        self, monkeypatch
    ):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        seq = _seq(402, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, seq)

        scheduler.abandon_save(str(seq.id))
        released = scheduler.take_source_safe_releases()
        scheduler.connector_completion(_store_terminal(operation))
        assert scheduler.take_source_safe_releases() == []
        assert released == [frozenset({0, 1})]

    def test_request_id_reuse_cannot_attach_an_old_lease_to_new_blocks(
        self, monkeypatch
    ):
        scheduler = _early_release_scheduler(monkeypatch, chunk_size=8)
        old = _seq(403, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(old)
        old.num_cached_tokens = 8
        old_operation = scheduler.build_connector_meta().requests[0].save_operation
        _finish_and_lease(scheduler, old)
        scheduler.connector_completion(_store_terminal(old_operation, succeeded=False))

        new = _seq(403, num_prompt_tokens=16, num_blocks=4)
        new.block_table = [10, 11, 12, 13]
        scheduler.update_state_after_alloc(new)
        new.num_cached_tokens = 8
        scheduler.build_connector_meta()

        assert scheduler.protected_block_ids(new) == frozenset({10, 11})
        assert scheduler.take_source_safe_releases() == []


class TestEarlyReleaseDefaultsOn:
    def test_supported_layout_uses_exact_source_protection(self, monkeypatch):
        monkeypatch.setattr(
            offcfg,
            "build_lmcache_config",
            lambda _config=None: SimpleNamespace(chunk_size=8),
        )
        monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
        scheduler = DenseOffloadScheduler(_config())
        seq = _seq(500, num_prompt_tokens=16, num_blocks=4)
        scheduler.update_state_after_alloc(seq)
        seq.num_cached_tokens = 8
        scheduler.build_connector_meta()

        assert scheduler._early_release is True
        assert scheduler.protected_block_ids(seq) == frozenset({0, 1})
        assert scheduler.should_defer_free(seq) is True
