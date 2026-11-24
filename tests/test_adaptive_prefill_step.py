# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_PREFILL_STEP_TOKENS: a PP P/D producer takes small prefill steps unless
the prefill in flight and queued fills full ones."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_scheduler import _ConnectorNamespace

from atom.kv_transfer.disaggregation import pd_producer
from atom.model_engine.scheduler import Scheduler, adaptive_prefill_step
from atom.model_engine.sequence import Sequence, SequenceStatus

FULL, SMALL, PP = 256, 128, 4
NEED = (PP + 2) * FULL


def _config(**overrides):
    values = {
        "pipeline_parallel_size": PP,
        "max_num_seqs": 64,
        "max_num_batched_tokens": FULL,
        "max_model_len": 1 << 16,
        "kv_cache_block_size": 16,
        "num_kvcache_blocks": 4096,
    }
    values.update(overrides)
    return MockConfig(**values)


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.setenv("ATOM_PREFILL_STEP_TOKENS", str(SMALL))
    monkeypatch.setattr(pd_producer, "pd_producer_configured", lambda config: True)


def _seq(tokens, start=0, **kwargs):
    return Sequence(list(range(start, start + tokens)), block_size=16, **kwargs)


def _step(sched):
    batch, _ = sched.schedule()
    sched.mark_pp_inflight(batch)
    return batch, [int(n) for n in batch.num_scheduled_tokens]


def test_off_unless_a_pp_producer_asks_below_the_budget(monkeypatch):
    monkeypatch.setenv("ATOM_PREFILL_STEP_TOKENS", str(SMALL))
    assert adaptive_prefill_step(_config()) == 0  # not a producer
    monkeypatch.setattr(pd_producer, "pd_producer_configured", lambda config: True)
    assert adaptive_prefill_step(_config()) == SMALL
    assert adaptive_prefill_step(_config(pipeline_parallel_size=1)) == 0
    assert adaptive_prefill_step(_config(enable_chunked_prefill=False)) == 0
    assert adaptive_prefill_step(_config(max_num_batched_tokens=SMALL)) == 0
    monkeypatch.setenv("ATOM_PREFILL_STEP_TOKENS", "100")
    assert adaptive_prefill_step(_config()) == 64  # the chunk grid
    monkeypatch.delenv("ATOM_PREFILL_STEP_TOKENS")
    assert adaptive_prefill_step(_config()) == 0


def test_short_supply_takes_small_steps(producer):
    sched = Scheduler(_config())
    for i in range(4):
        sched.add(_seq(SMALL, 1000 * i))
    assert _step(sched)[1] == [SMALL]


def test_full_steps_once_supply_fills_the_pipeline(producer):
    sched = Scheduler(_config())
    for i in range(NEED // SMALL):
        sched.add(_seq(SMALL, 1000 * i))
    assert _step(sched)[1] == [SMALL, SMALL]


def test_prefill_in_flight_counts_toward_the_supply(producer):
    sched = Scheduler(_config())
    for i in range(NEED // SMALL):
        sched.add(_seq(SMALL, 1000 * i))
    batches = [_step(sched)[0] for _ in range(2)]
    assert sched._pp_inflight_prefill_tokens == 2 * FULL
    # 8 queued requests and the 4 in flight still fill the supply.
    assert _step(sched)[1] == [SMALL, SMALL]
    for batch in batches:
        sched.release_pp_inflight(batch)
    assert sched._pp_inflight_prefill_tokens == FULL


def test_a_long_prompt_keeps_small_steps_amid_queued_requests(producer):
    sched = Scheduler(_config())
    sched.add(_seq(4 * FULL, 50000))
    for i in range(NEED // SMALL):
        sched.add(_seq(SMALL, 1000 * i))
    # Its first chunk closes the step at the small step ...
    assert _step(sched)[1] == [SMALL]
    # ... and every chunk after it rides a small step alone.
    assert _step(sched)[1] == [SMALL]


def test_a_whole_prefill_takes_the_full_budget(producer):
    """A whole prefill (multimodal without embedding spans) cannot be cut to the
    small step: it takes the full budget rather than hold up the queue."""
    sched = Scheduler(_config())
    sched.add(_seq(SMALL + 16, multimodal_data={"cache_seed": 7}))
    sched.add(_seq(64, 1000))
    assert _step(sched)[1] == [SMALL + 16, 64]


def test_a_whole_prefill_resumed_after_a_tier_load_takes_the_full_budget(producer):
    sched = Scheduler(_config())
    sched.kv_connector = _ConnectorNamespace(is_offload=True)
    seq = _seq(SMALL + 48, multimodal_data={"cache_seed": 7})
    assert sched.block_manager.allocate(seq)
    seq.num_cached_tokens, seq.offload_loaded = 16, True  # its tier load landed
    seq.status = SequenceStatus.WAITING
    sched.waiting.append(seq)
    assert _step(sched)[1] == [SMALL + 32]


def test_a_decoding_request_keeps_small_steps(producer):
    sched = Scheduler(_config())
    for i in range(NEED // SMALL):
        sched.add(_seq(SMALL, 1000 * i))
    decoding = _seq(8, 90000)
    decoding.append_token(7)
    sched.running.append(decoding)
    assert sched._prefill_step_budget() == SMALL


def test_off_the_budget_follows_max_num_batched_tokens(monkeypatch):
    """Off, nothing is cached: the room is the live budget and no batch's
    prefill tokens are read."""
    monkeypatch.delenv("ATOM_PREFILL_STEP_TOKENS", raising=False)
    sched = Scheduler(_config())
    sched.max_num_batched_tokens = 4 * FULL
    assert sched._prefill_room(FULL, 0) == 3 * FULL
    batch = SimpleNamespace(is_final_chunk=None, total_seqs_num_decode=0)
    sched.mark_pp_inflight(batch)
    sched.release_pp_inflight(batch)
    assert sched._pp_inflight_prefill_tokens == 0
