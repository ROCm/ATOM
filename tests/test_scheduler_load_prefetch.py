# SPDX-License-Identifier: MIT
# Tests for OFFLOAD_PREFETCH_LOADS: parking tier loads past a full batch.

from collections import deque

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus

PROMPT = 40
BUDGET = 16
LOAD = 32


class _LoadConnector:
    """An offload connector whose tier holds `LOAD` tokens for `loads`."""

    is_offload = True

    def __init__(self, loads=(), refuse=(), pending=()):
        self.loads = set(loads)
        self.refuse = set(refuse)
        self.pending = set(pending)
        self.allocated: list[int] = []

    def lookup_pending(self, seq):
        return seq.id in self.pending

    def get_num_new_matched_tokens(self, seq):
        if seq.id in self.loads:
            return LOAD, True
        return 0, False

    def update_state_after_alloc(self, seq):
        self.allocated.append(seq.id)

    def should_park_for_load_after_alloc(self, seq):
        if seq.id in self.refuse:
            return False
        seq.offload_loaded_tokens = LOAD
        return True

    def build_connector_meta(self):
        return None


@pytest.fixture
def make(seq_factory, monkeypatch):
    def _make(cap, n=4, depth=None, **conn):
        monkeypatch.setenv("OFFLOAD_PREFETCH_LOADS", str(cap))
        if depth is not None:
            monkeypatch.setenv("OFFLOAD_ASYNC_LOOKUP_DEPTH", str(depth))
        sched = Scheduler(
            MockConfig(
                max_num_seqs=8,
                max_num_batched_tokens=BUDGET,
                num_kvcache_blocks=200,
                kv_cache_block_size=4,
                max_model_len=256,
            )
        )
        seqs = [seq_factory(list(range(i * 1000, i * 1000 + PROMPT))) for i in range(n)]
        ids = {name: s.id for name, s in zip("abcd", seqs)}
        sched.kv_connector = _LoadConnector(
            **{k: {ids[x] for x in v} for k, v in conn.items()}
        )
        for s in seqs:
            sched.add(s)
        return sched, seqs

    return _make


def _ids(q):
    return [s.id for s in q]


def test_off_keeps_the_fcfs_break(make):
    sched, (a, b, c, d) = make(0, loads="c")

    batch, _ = sched.schedule()

    assert list(batch.req_ids) == [a.id]
    assert c.status == SequenceStatus.WAITING
    assert sched.kv_connector.allocated == [a.id]
    assert _ids(sched.waiting) == [b.id, c.id, d.id]


def test_a_load_behind_a_full_batch_parks(make):
    sched, (a, b, c, d) = make(4, loads="c")

    batch, _ = sched.schedule()

    assert list(batch.req_ids) == [a.id]
    assert c.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sched._num_parked_remote_kv == 1
    assert len(c.block_table) > 0
    # Nothing but the load ran out of turn: b and d keep their order.
    assert b.status == d.status == SequenceStatus.WAITING
    assert sched.kv_connector.allocated == [a.id, c.id]
    assert _ids(sched.waiting) == [b.id, d.id, c.id]


def test_the_cap_counts_every_parked_load(make):
    sched, (_a, _b, c, d) = make(1, loads="cd")

    sched.schedule()

    assert c.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert d.status == SequenceStatus.WAITING
    assert sched._num_parked_remote_kv == 1


def test_the_scan_stops_at_the_lookup_depth(make):
    sched, (_a, b, c, d) = make(4, depth=2, loads="c")

    sched.schedule()

    assert c.status == SequenceStatus.WAITING
    assert _ids(sched.waiting) == [b.id, c.id, d.id]


def test_a_pending_lookup_keeps_its_place(make):
    sched, (_a, b, c, d) = make(4, loads="cd", pending="c")

    sched.schedule()

    assert c.status == SequenceStatus.WAITING
    assert d.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert _ids(sched.waiting) == [b.id, c.id, d.id]


def test_a_refused_load_gives_its_blocks_back_and_is_not_retried(make):
    sched, (a, b, c, d) = make(4, loads="cd", refuse="c")

    sched.schedule()

    assert c.status == SequenceStatus.WAITING
    assert len(c.block_table) == 0 and c.num_cached_tokens == 0
    assert c._load_prefetch_refused is True
    assert sched.block_manager.kv.num_used == len(a.block_table)
    # The scan stopped at the refusal, in order, before reaching d.
    assert d.status == SequenceStatus.WAITING
    assert _ids(sched.waiting) == [b.id, c.id, d.id]

    # With no postprocess in between, a cannot continue and b takes the budget.
    batch, _ = sched.schedule()

    assert list(batch.req_ids) == [b.id]
    assert sched.kv_connector.allocated.count(c.id) == 1
    assert d.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert _ids(sched.waiting) == [c.id, d.id]
    assert sched.block_manager.kv.num_used == sum(len(s.block_table) for s in (a, b, d))


def test_a_finished_prefetch_resumes_from_the_loaded_prefix(make):
    sched, (_a, b, c, _d) = make(4, loads="c")
    sched.schedule()
    assert c.status == SequenceStatus.WAITING_FOR_REMOTE_KVS

    # a finishes elsewhere and b is gone; c's load reports in.
    sched.running.clear()
    sched._partial_prefill_count = 0
    sched.waiting = deque(s for s in sched.waiting if s is not b)
    sched.finished_recving_kv_req_ids.append(c.id)

    batch, scheduled = sched.schedule()

    assert c.id in scheduled
    assert c.num_cached_tokens == LOAD
    assert int(batch.num_scheduled_tokens[list(batch.req_ids).index(c.id)]) == (
        PROMPT - LOAD
    )
    assert sched._num_parked_remote_kv == 0
