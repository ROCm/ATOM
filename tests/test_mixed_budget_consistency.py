# SPDX-License-Identifier: MIT
"""Admission probes and startup memory sizing use the same budgets as execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import MockConfig

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
    assert sched._probe_hbm_prefix(waiting, limits=limits).num_cached_blocks == -1
    assert not sched._can_admit_head_prefill(limits)
    assert sched._probe_local_prefill_work(limits) == (False, 0)
    assert sched._estimate_prefill_chunk_cost(limits) == 0
    # The prefix walk may stay cached; reservation-sensitive fit must not.
    assert sched.block_manager.can_allocate_memoized(waiting) == 0


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
    assert candidates.call_count == 2  # reservation, then decode execution


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


@pytest.mark.parametrize("paged_state", [False, True])
def test_mixed_profile_is_not_charged_to_graph_pools(monkeypatch, caplog, paged_state):
    runner_module = pytest.importorskip("atom.model_engine.model_runner")
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
