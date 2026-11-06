# SPDX-License-Identifier: MIT
"""Admission and default-path regressions for optional mixed scheduling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceType


def scheduler(**overrides):
    values = {
        "enable_mixed_prefill_decode": True,
        "max_num_batched_tokens": 8,
        "max_num_seqs": 4,
        "num_kvcache_blocks": 100,
        "kv_cache_block_size": 4,
    }
    values.update(overrides)
    return Scheduler(MockConfig(**values))


def finish_prefill(sched, batch):
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[],
            token_ids=[],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )


@pytest.mark.parametrize("prefix_cache", [False, True])
def test_final_prefills_do_not_reserve_decode_pages_in_the_same_step(
    seq_factory, prefix_cache
):
    sched = scheduler(num_kvcache_blocks=2, enable_prefix_caching=prefix_cache)
    first = seq_factory([1, 2, 3, 4])
    second = seq_factory([5, 6, 7, 8])
    sched.add(first)
    sched.add(second)
    batch, _ = sched.schedule()
    assert batch.req_ids == [first.id, second.id]
    assert batch.total_seqs_num_prefill == 2
    assert batch.total_tokens_num == 8
    assert not batch.is_mixed


def test_previous_final_prefill_still_reserves_first_decode(seq_factory):
    sched = scheduler()
    old = seq_factory([1, 2, 3, 4])
    sched.add(old)
    sched.schedule()
    old.num_cached_tokens = old.num_prompt_tokens
    old.append_token(5)
    assert old.type == SequenceType.PREFILL
    assert sched._prefill_budget() == 7
    assert sched._prefill_budget({old.id: old}) == 8


def test_partial_prefill_probe_uses_the_scheduled_budget(seq_factory, monkeypatch):
    sched = scheduler()
    decode = seq_factory([1, 2, 3, 4])
    sched.add(decode)
    sched.schedule()
    decode.num_cached_tokens = decode.num_prompt_tokens
    decode.append_token(5)
    prompt = seq_factory(list(range(20, 52)))
    sched.add(prompt)
    first, _ = sched.schedule()
    finish_prefill(sched, first)
    assert first.is_mixed and prompt.is_partial_prefill
    preview = Mock(wraps=sched._preview_prefill_chunk)
    monkeypatch.setattr(sched, "_preview_prefill_chunk", preview)
    assert sched._probe_local_prefill_work() == (True, 7)
    offered = preview.call_args.args[-1]
    batch, _ = sched.schedule()
    assert offered == batch.total_tokens_num_prefill == 7


@pytest.mark.parametrize("mixed", [False, True])
def test_connector_metadata_precedes_midstep_settlement(
    seq_factory, monkeypatch, mixed
):
    sched = scheduler(enable_mixed_prefill_decode=mixed)
    prompt = seq_factory(list(range(20)))
    sched.add(prompt)
    first, _ = sched.schedule()
    finish_prefill(sched, first)
    events = []
    # A resumed partial prefill needs no connector admission query.
    sched.kv_connector = SimpleNamespace(
        is_offload=False,
        is_producer=False,
        build_connector_meta=lambda: events.append("connector"),
    )
    monkeypatch.setattr(sched, "_publish_state_loads", lambda: None)
    monkeypatch.setattr(sched, "_publish_state_stores", lambda: None)
    monkeypatch.setattr(sched, "_note_connector_scheduled_tokens", lambda *a: None)
    plan = sched.block_manager.plan_midstep

    def record_plan(*args):
        events.append("midstep")
        return plan(*args)

    monkeypatch.setattr(sched.block_manager, "plan_midstep", record_plan)
    batch, _ = sched.schedule()
    assert events == ["connector", "midstep"]
    assert batch.is_final_chunk == [False]
