# SPDX-License-Identifier: MIT
"""SJF ranks actual prefill work, not the full length of a cache-hot prompt."""

import pickle

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import StateTransfer


def _scheduler(**overrides):
    config = {
        "enable_prefix_caching": True,
        "kv_cache_block_size": 4,
        "num_kvcache_blocks": 40,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
        "scheduling_policy": "sjf",
    }
    config.update(overrides)
    return Scheduler(MockConfig(**config))


def _seq(tokens):
    return Sequence(tokens, block_size=4)


def _warm_cache(sched):
    warmup = _seq(list(range(1, 17)))
    sched.add(warmup)
    batch, scheduled = sched.schedule()
    sched.postprocess(
        list(scheduled.values()),
        ScheduledBatchOutput(
            req_ids=[warmup.id],
            token_ids=[(2,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    assert warmup.is_finished


def _waiters(sched):
    hot = _seq(list(range(1, 17)))  # 12 reusable tokens, 4 left to compute.
    cold = _seq(list(range(100, 108)))  # 8 tokens, all cold.
    sched.add(cold)
    sched.add(hot)
    return hot, cold


def test_cache_hot_long_request_beats_cache_cold_short():
    sched = _scheduler()
    _warm_cache(sched)
    hot, cold = _waiters(sched)

    batch, scheduled = sched.schedule()

    assert list(scheduled) == [hot.id]
    assert list(batch.num_scheduled_tokens) == [4]
    assert hot.num_cached_tokens == 12
    assert cold.num_cached_tokens == 0


def test_sort_does_not_mutate_sequence_or_cache_bookkeeping():
    sched = _scheduler()
    _warm_cache(sched)
    hot, cold = _waiters(sched)
    before = [pickle.dumps(vars(seq)) for seq in (hot, cold)]
    occupancy = sched.block_manager.pool_occupancy()
    demand_counters = (
        sched.block_manager.demands_recorded,
        sched.block_manager.demands_declined_no_room,
    )

    sched._reorder_waiting_shortest_first()

    assert list(sched.waiting) == [hot, cold]
    assert [pickle.dumps(vars(seq)) for seq in (hot, cold)] == before
    assert sched.block_manager.pool_occupancy() == occupancy
    assert (
        sched.block_manager.demands_recorded,
        sched.block_manager.demands_declined_no_room,
    ) == demand_counters


def test_state_checkpoint_gate_can_reject_an_otherwise_hot_prefix():
    sched = _scheduler()
    _warm_cache(sched)
    hot, cold = _waiters(sched)
    hot.has_per_req_cache = True
    state = StateSlotPool(2, StateTransfer.fork(1), hash_block_size=4)
    sched.block_manager.state_caches += (state,)

    # KV exists, but there is no matching recurrent state to resume from.
    assert sched.block_manager.prefix_cached_tokens(hot) == 0
    sched._reorder_waiting_shortest_first()
    assert list(sched.waiting) == [cold, hot]

    # Publish a real checkpoint index entry at the 12-token boundary.
    h = -1
    for start in (0, 4, 8):
        h = sched.block_manager.compute_hash(hot.token_ids[start : start + 4], h)
    state._index(h, 0)
    assert sched.block_manager.prefix_cached_tokens(hot) == 12
    sched._reorder_waiting_shortest_first()
    assert list(sched.waiting) == [hot, cold]


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_allocated_waiter_uses_computed_tokens(enable_prefix_caching):
    """An offload/partial resume owns KV even when it is not prefix-indexed."""
    sched = _scheduler(enable_prefix_caching=enable_prefix_caching)
    resume = _seq(list(range(1, 17)))
    sched.block_manager.allocate(resume)
    resume.num_cached_tokens = 12
    cold = _seq(list(range(100, 108)))
    sched.add(cold)
    sched.add(resume)

    sched._reorder_waiting_shortest_first()

    assert list(sched.waiting) == [resume, cold]
    assert resume.num_cached_tokens == 12


def test_sort_refreshes_after_prefix_eviction():
    sched = _scheduler(num_kvcache_blocks=4)
    _warm_cache(sched)
    hot, cold = _waiters(sched)
    sched._reorder_waiting_shortest_first()
    assert list(sched.waiting) == [hot, cold]

    evictor = _seq(list(range(200, 216)))
    sched.block_manager.allocate(evictor)
    sched._reorder_waiting_shortest_first()
    assert list(sched.waiting) == [cold, hot]


def test_fcfs_does_not_probe_prefixes(monkeypatch):
    sched = _scheduler(scheduling_policy="fcfs")
    _warm_cache(sched)
    _hot, cold = _waiters(sched)

    def unexpected_probe(seq):
        pytest.fail("FCFS must not pay for SJF cache probes")

    monkeypatch.setattr(sched.block_manager, "prefix_cached_tokens", unexpected_probe)
    _, scheduled = sched.schedule()
    assert list(scheduled) == [cold.id]


def _count_prefix_walks(sched, monkeypatch):
    """Hash walks per schedule(), counted at the one place they can happen."""
    bm = sched.block_manager
    calls = {"match": 0, "admit": 0}
    real_match, real_admit = bm._match_prefix, bm.can_allocate

    def counting_match(seq):
        calls["match"] += 1
        return real_match(seq)

    def counting_admit(seq):
        calls["admit"] += 1
        return real_admit(seq)

    monkeypatch.setattr(bm, "_match_prefix", counting_match)
    monkeypatch.setattr(bm, "can_allocate", counting_admit)
    return calls


def test_fcfs_walks_the_prefix_once_per_admission_attempt(monkeypatch):
    """The SJF probe must not become a tax on FCFS.

    `_match_prefix` is the only chained-hash walk in the manager, so counting it
    bounds the whole cost. Under FCFS it may run exactly once per `can_allocate`
    -- the walk admission has always done -- and never on the sort path.
    """
    sched = _scheduler(scheduling_policy="fcfs", max_num_seqs=4)
    _warm_cache(sched)
    _waiters(sched)
    calls = _count_prefix_walks(sched, monkeypatch)

    sched.schedule()

    assert calls["admit"] > 0
    assert calls["match"] == calls["admit"]


def test_sjf_pays_one_extra_walk_per_fresh_waiter(monkeypatch):
    """The counterpart: SJF's surplus is bounded at one walk per fresh waiter."""
    sched = _scheduler(max_num_seqs=4)
    _warm_cache(sched)
    waiters = _waiters(sched)
    calls = _count_prefix_walks(sched, monkeypatch)

    sched.schedule()

    assert calls["match"] == calls["admit"] + len(waiters)


def test_disabled_prefix_caching_preserves_shortest_prompt_order():
    sched = _scheduler(enable_prefix_caching=False)
    _warm_cache(sched)
    _hot, cold = _waiters(sched)
    _, scheduled = sched.schedule()
    assert list(scheduled) == [cold.id]


@pytest.mark.parametrize("length, cached", [(1, 0), (4, 0), (5, 4), (16, 12)])
def test_probe_and_admission_leave_the_final_block_to_compute(length, cached):
    sched = _scheduler()
    _warm_cache(sched)
    seq = _seq(list(range(1, length + 1)))
    bm = sched.block_manager
    assert bm.prefix_cached_tokens(seq) == cached
    assert bm.can_allocate(seq) * bm.hash_block_size == cached
