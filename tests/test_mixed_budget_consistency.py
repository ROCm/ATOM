# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Admission probes and startup memory sizing use the same budgets as execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiter_stub import atom_without_aiter
from conftest import MockConfig


@pytest.fixture
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


from atom.model_engine.block_manager import AdmissionProbe, BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType


def _decode(sched, seed):
    seq = Sequence([seed] * 4, block_size=4)
    assert sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 4
    seq.append_token(seed + 1)
    seq.type = SequenceType.DECODE
    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)
    return seq


@pytest.mark.parametrize(
    "probe",
    [
        "_local_prefill_pending_work",
        "_can_admit_head_prefill",
        "_estimate_prefill_chunk_cost",
        "_admission_probe",
    ],
)
def test_schedule_passes_reserved_limits_to_every_prefill_probe(monkeypatch, probe):
    from atom.model_engine.prefill_delayer import PrefillDecision

    sched = Scheduler(
        MockConfig(
            num_kvcache_blocks=20,
            max_num_seqs=4,
            max_num_batched_tokens=16,
            enable_mixed_prefill_decode=True,
        )
    )
    old = _decode(sched, 3)
    new = Sequence([20] * 8, block_size=4)
    sched.add(new)
    expected = sched._prefill_limits()
    assert expected.reserved_decode_blocks == 1 and expected.tokens == 15
    original = getattr(sched, probe)
    seen = []

    def checked(*args, **kwargs):
        seen.append(sched.block_manager.reserved_decode_blocks)
        assert seen[-1] == expected.reserved_decode_blocks
        return original(*args, **kwargs)

    monkeypatch.setattr(sched, probe, checked)
    if probe != "_admission_probe":
        sched._local_prefill_coalescing = probe == "_local_prefill_pending_work"
        sched.cost_balance = probe == "_estimate_prefill_chunk_cost"
        sched.prefill_delayer = SimpleNamespace(
            protects_decode=lambda count: False,
            max_queue_ms=None,
            should_allow_prefill=lambda **kw: PrefillDecision(
                allow=False, prefillable_ranks=1
            ),
        )
    batch, _ = sched._schedule()
    assert seen
    assert old.id in batch.req_ids and sched.total_preemptions == 0


@pytest.mark.parametrize("prefix", [False, True])
def test_all_prefill_probes_protect_decode_pages(prefix):
    sched = Scheduler(
        MockConfig(
            num_kvcache_blocks=3,
            enable_prefix_caching=prefix,
            enable_mixed_prefill_decode=True,
        )
    )
    _decode(sched, 3)
    waiting = Sequence(list(range(20, 28)), block_size=4)
    sched.add(waiting)
    limits = sched._prefill_limits()
    assert limits.reserved_decode_blocks == 1
    assert sched.block_manager.can_allocate_memoized(waiting) == 0
    sched.block_manager.reserved_decode_blocks = limits.reserved_decode_blocks
    assert sched._probe_hbm_prefix(waiting).num_cached_blocks == -1
    assert not sched._can_admit_head_prefill()
    assert sched._probe_local_prefill_work(limits) == (False, 0)
    assert sched._estimate_prefill_chunk_cost(limits) == 0
    # The prefix walk may stay cached; reservation-sensitive fit must not.
    sched.block_manager.reserved_decode_blocks = 0
    assert sched.block_manager.can_allocate_memoized(waiting) == 0


@pytest.mark.parametrize("prefix", [False, True])
def test_probe_demand_excludes_reserved_decode_pages(prefix):
    bm = BlockManager(MockConfig(num_kvcache_blocks=3, enable_prefix_caching=prefix))
    seq = Sequence(list(range(8)), block_size=4)
    probe = AdmissionProbe()
    bm.reserved_decode_blocks = 1
    assert bm.can_allocate(seq, probe=probe) == 0
    assert probe.num_new_blocks == 2
    assert bm.allocate(seq, probe.num_cached_blocks)
    assert bm.kv.num_free == 1

    next_seq = Sequence(list(range(10, 14)), block_size=4)
    assert bm.can_allocate(next_seq) == -1
    bm.reserved_decode_blocks = 0
    assert bm.can_allocate(next_seq) == 0


def test_admissions_share_one_decode_reservation(monkeypatch):
    sched = Scheduler(
        MockConfig(
            num_kvcache_blocks=100,
            max_num_seqs=16,
            enable_mixed_prefill_decode=True,
        )
    )
    for seed in range(3, 6):
        _decode(sched, seed)
    for seed in range(10, 16):
        sched.add(Sequence([seed] * 4, block_size=4))
    limits = Mock(wraps=sched._prefill_limits)
    candidates = Mock(wraps=sched._decode_candidates)
    monkeypatch.setattr(sched, "_prefill_limits", limits)
    monkeypatch.setattr(sched, "_decode_candidates", candidates)
    batch, _ = sched.schedule()
    assert batch.total_seqs_num_prefill == 6
    assert batch.total_seqs_num_decode == 3
    assert limits.call_count == 1
    assert candidates.call_count == 1  # reservation; decode checks popped rows


def _crowded_decode_scheduler(width):
    sched = Scheduler(
        MockConfig(
            kv_cache_block_size=16,
            num_kvcache_blocks=200,
            max_num_batched_tokens=100 * width,
            max_num_seqs=101,
            max_model_len=128,
            long_prefill_token_threshold=1,
            enable_mixed_prefill_decode=True,
            speculative_config=(
                SimpleNamespace(
                    num_speculative_tokens=width - 1, use_dspark=lambda: True
                )
                if width > 1
                else None
            ),
        )
    )
    for seed in range(100):
        seq = Sequence([seed + 3] * 16, block_size=16)
        assert sched.block_manager.allocate(seq, 0)
        # Each decode needs exactly one more page, including DSpark lookahead.
        boundary = len(seq.block_table) * 16
        while len(seq) <= boundary:
            seq.append_token(seed + 4)
        seq.num_cached_tokens = len(seq) - width
        seq.type = SequenceType.DECODE
        seq.status = SequenceStatus.RUNNING
        sched.running.append(seq)
        assert (
            sched.block_manager.append_blocks_needed(seq, width, sched.draft_lookahead)
            == 1
        )
    assert sched.block_manager.kv.num_free == 100
    return sched


@pytest.mark.parametrize("width", [1, 8])
def test_short_prefill_reserves_every_decode_that_can_join(width):
    sched = _crowded_decode_scheduler(width)
    new = Sequence([200], block_size=16)
    sched.add(new)
    limits = sched._prefill_limits()
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 1
    assert batch.total_seqs_num_decode == 99
    assert limits.reserved_decode_blocks == 99
    assert sched.total_preemptions == 0


@pytest.mark.parametrize("width", [1, 8])
def test_short_chunk_cannot_spend_unreserved_tail_decode_pages(width):
    sched = _crowded_decode_scheduler(width)
    # Only one token would run, but admitting this prompt owns several pages.
    # All 100 existing decodes fit without it; none should be preempted for it.
    new = Sequence([200] * 80, block_size=16)
    sched.add(new)
    batch, _ = sched.schedule()
    assert sched.total_preemptions == 0
    assert batch.total_seqs_num_prefill == 0
    assert batch.total_seqs_num_decode == 100
    assert new in sched.waiting


def test_mixed_decode_pass_preserves_partial_prefill_order():
    sched = Scheduler(
        MockConfig(
            num_kvcache_blocks=100,
            max_num_batched_tokens=16,
            max_num_seqs=8,
            enable_mixed_prefill_decode=True,
        )
    )
    partials = []
    for seed in (10, 20):
        seq = Sequence([seed] * 8, block_size=4)
        assert sched.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = 4
        seq.is_partial_prefill = True
        seq.status = SequenceStatus.RUNNING
        sched.running.append(seq)
        partials.append(seq)
    sched._partial_prefill_count = 2
    sched._checkpoint_holds[partials[1].id] = 8
    new = Sequence([30] * 4, block_size=4)
    sched.add(new)
    batch, _ = sched.schedule()
    assert batch.req_ids == [partials[0].id, new.id]
    assert list(sched.running) == [*partials, new]


@pytest.mark.parametrize("prefill", [False, True])
def test_mixed_off_does_not_scan_decode_candidates_or_snapshot_running(prefill):
    from collections import deque

    class NoSnapshot(deque):
        def __iter__(self):
            raise AssertionError("mixed-off scheduling must not scan running")

    sched = Scheduler(
        MockConfig(num_kvcache_blocks=100, max_num_seqs=8, max_num_batched_tokens=16)
    )
    decode = _decode(sched, 3)
    if prefill:
        new = Sequence([20] * 4, block_size=4)
        sched.add(new)
    sched.running = NoSnapshot(sched.running)
    # Call the scheduling phase directly: outer metrics may walk running.
    sched._tick_cost = 0
    sched._tick_cost_limit = float("inf")
    batch, _ = sched._schedule()
    assert batch.req_ids == ([new.id] if prefill else [decode.id])


@pytest.mark.parametrize("paged_state", [False, True])
def test_mixed_profile_is_not_charged_to_graph_pools(
    monkeypatch, caplog, paged_state, runner_module
):
    import torch

    from atom.model_engine.kv_block import STATE_SLOT_CLASS
    from atom.model_engine.state_runtime import DEFAULT_STATE_RUNTIME, StateTransfer
    from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
        PoolPlan,
        page_pool,
        state_pool,
    )

    gib = 1 << 30
    stats = {
        "allocated_bytes.all.peak": 102 * gib,
        "allocated_bytes.all.current": 100 * gib,
    }
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: stats)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100 * gib, 200 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 100 * gib)
    monkeypatch.setattr(
        runner_module.KVConnectorFactory, "kv_budget_reserve_bytes", lambda config: 0
    )
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.device = "cpu"
    runner.block_size = 4
    runner.pool_plan = PoolPlan.empty()
    runner.state_runtime = DEFAULT_STATE_RUNTIME
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(head_dim=128),
        enable_mixed_prefill_decode=True,
        enforce_eager=False,
        max_num_seqs=4,
        max_model_len=64,
        gpu_memory_utilization=0.9,
        pipeline_parallel_size=1,
        is_multinode_tp=False,
        enable_rapidserve=False,
    )
    runner.drafter = SimpleNamespace(uses_confidence_schedule=True, mtp_k=7)
    runner._tr_hybrid_active = lambda: False
    runner._dspark_capture_q_buckets = lambda q: [1, 2, 4, 8]
    runner._sub_pool_specs = lambda: [page_pool(gib)] + (
        [state_pool(STATE_SLOT_CLASS, 512, entries_per_req=1)] if paged_state else []
    )
    runner.attn_metadata_builder = SimpleNamespace(
        state_transfer=lambda: (
            StateTransfer.copy("test-layout")
            if paged_state
            else DEFAULT_STATE_RUNTIME.transfer
        ),
        checkpoint_image_bytes=lambda: 256,
        checkpoint_page_unit_bytes=lambda: gib,
        sub_pool_specs=runner._sub_pool_specs,
    )

    def profile(runner):
        spec = runner.state_runtime.checkpoint_spec
        if paged_state:
            assert spec is not None
            assert (spec.page_unit_bytes, spec.slot_bytes, spec.image_bytes) == (
                gib,
                512,
                256,
            )
            assert spec.layout_id == "test-layout"
        else:
            assert spec is None
        stats["allocated_bytes.all.peak"] = 100 * gib
        return 112 * gib, 2 * gib

    profile = Mock(side_effect=profile)
    monkeypatch.setattr(runner_module, "profile_mixed_memory", profile)
    with caplog.at_level("INFO", logger="atom"):
        result = runner.get_num_blocks()
    # 180 GiB budget - 112 live - 2 non-torch - 1.6 graphs - 4 safety.
    assert result["num_kvcache_blocks"] == 60
    assert profile.call_count == 1
    assert sum("Memory budget:" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_remote_completion_reservation_covers_both_promotion_phases(
    monkeypatch, failed
):
    from test_mixed_kv_owners import decode, receive, scheduler

    sched = scheduler(offload=failed)
    old = decode(sched)
    incoming = receive(sched, ready=not failed)
    if failed:
        sched.failed_recving_kv_req_ids.append(str(incoming.id))
    before = sched._prioritize_ready_remote_kv_requests
    after = sched._early_load_tier_prefixes
    seen = []

    def prioritized():
        assert sched.block_manager.reserved_decode_blocks > 0
        seen.append("before")
        return before()

    def admitted():
        limits = sched._prefill_limits()
        assert (
            sched.block_manager.reserved_decode_blocks == limits.reserved_decode_blocks
        )
        seen.append("after")
        return after()

    monkeypatch.setattr(sched, "_prioritize_ready_remote_kv_requests", prioritized)
    monkeypatch.setattr(sched, "_early_load_tier_prefixes", admitted)
    batch, _ = sched.schedule()
    assert seen == ["before", "after"] and old.id in batch.req_ids
    assert incoming.id in batch.req_ids
    assert sched.block_manager.reserved_decode_blocks == 0


def test_admission_exception_releases_tick_reservation(monkeypatch):
    sched = Scheduler(MockConfig(enable_mixed_prefill_decode=True))
    _decode(sched, 3)
    sched.add(Sequence([20] * 8, block_size=4))

    def fail():
        assert sched.block_manager.reserved_decode_blocks > 0
        raise RuntimeError("admission failed")

    monkeypatch.setattr(sched, "_early_load_tier_prefixes", fail)
    with pytest.raises(RuntimeError, match="admission failed"):
        sched.schedule()
    assert sched.block_manager.reserved_decode_blocks == 0
