# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admission does not wait for a tier lookup that cannot load anything.

The contract has two halves:

1. *Safety*: a lookup is skipped only when no answer could arm a load. Every
   load path refuses `need = lmc - hbm < OFFLOAD_MIN_LOAD_TOKENS`, and a tier
   hit never exceeds the prompt, so `num_tokens - hbm < min_load` makes the
   answer irrelevant. The threshold is the connector's own.
2. *LRU*: the lookup is still sent (fire-and-forget). On the LMCache MP server
   it is what refreshes the prefix's L1 recency and opens the session whose
   `end_session` touches the request's chunks. Its answer is discarded and its
   read locks released once it arrives.
"""

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from test_lmcache_mp import _config
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client

MIN_LOAD = 32


def _scheduler(monkeypatch, *, chunks=8, lookup_done=False):
    """An MP scheduler over the async-capable fake adapter."""
    adapter = _Adapter(_Client(chunks, lookup_done=lookup_done))
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=1.0, poll_interval=0.01
    )
    sched = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    sched._min_load_tokens = MIN_LOAD
    return sched, client, adapter


def _seq(n=64, sid=7):
    return SimpleNamespace(
        id=sid,
        num_tokens=n,
        num_prompt_tokens=n,
        num_cached_tokens=0,
        token_ids=list(range(n)),
        block_table=[],
    )


# ── safety: the skip never hides a load the lookup could arm ─────────────


@pytest.mark.parametrize(
    "hbm,cannot", [(64 - MIN_LOAD + 1, True), (64 - MIN_LOAD, False), (0, False)]
)
def test_threshold_is_the_connectors_min_load(monkeypatch, hbm, cannot):
    sched, _, _ = _scheduler(monkeypatch)
    assert sched.tier_lookup_cannot_load(_seq(), hbm) is cannot


@pytest.mark.parametrize("hit", [0, 40, 48, 60, 64])
def test_any_tier_hit_inside_the_skip_window_is_refused_as_a_load(monkeypatch, hit):
    """The property the skip relies on, checked against the real decision."""
    sched, _, _ = _scheduler(monkeypatch)
    seq = _seq()
    hbm = 64 - MIN_LOAD + 4  # inside the skip window
    assert sched.tier_lookup_cannot_load(seq, hbm)
    seq.num_cached_tokens = hbm
    spec = LoadSpec(hbm_cached_tokens=hbm, lmcache_cached_tokens=hit, can_load=False)
    should_load, reason, *_ = sched._decide_load_after_alloc(seq, spec)
    assert not should_load, reason


def test_a_gap_the_tier_could_fill_is_not_skipped(monkeypatch):
    sched, _, adapter = _scheduler(monkeypatch)
    assert sched.skip_tier_lookup(_seq(), 64 - MIN_LOAD) is False
    assert adapter._client.lookups == []


# ── LRU: the lookup is still sent, never waited on, then discarded ──────


def test_skip_sends_the_lookup_without_waiting_and_orphans_it(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()

    assert sched.skip_tier_lookup(seq, 60) is True
    # Sent: the server refreshes L1 recency and opens the session.
    assert len(adapter._client.lookups) == 1
    # Not waited on, not consumed: no load is armed for it.
    assert "7" not in sched._load_specs
    assert "7" in client._orphans
    # Save floor from HBM, so the resident prefix is not stored again.
    assert sched._hit_save_floors["7"] == 60


def test_orphaned_lookup_releases_its_locks_when_it_answers(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    sched.skip_tier_lookup(_seq(), 60)
    adapter._client.lookup_future.done = True
    client.pump()
    assert adapter.freed, "read locks taken by the answered lookup must be freed"
    assert "7" not in client._orphans


def test_a_prefetched_lookup_is_not_sent_twice(monkeypatch):
    sched, client, adapter = _scheduler(monkeypatch, lookup_done=False)
    seq = _seq()
    sched.prefetch_lookups([seq])
    assert len(adapter._client.lookups) == 1
    assert sched.skip_tier_lookup(seq, 60) is True
    assert len(adapter._client.lookups) == 1
    assert "7" in client._orphans


def test_an_armed_load_is_finished_by_the_ordinary_path(monkeypatch):
    sched, _, _ = _scheduler(monkeypatch)
    seq = _seq()
    sched._load_specs["7"] = LoadSpec(
        hbm_cached_tokens=0, lmcache_cached_tokens=48, can_load=False
    )
    assert sched.skip_tier_lookup(seq, 60) is False


# ── scheduler gate ───────────────────────────────────────────────────────


class _Conn:
    is_offload = True

    def __init__(self, cannot=True):
        self.cannot = cannot
        self.skipped = []

    def tier_lookup_cannot_load(self, seq, hbm):
        return self.cannot

    def skip_tier_lookup(self, seq, hbm):
        self.skipped.append(hbm)
        return True


def _gate(monkeypatch, *, conn, state_offload=None, hit_blocks=10):
    from atom.model_engine import scheduler as sched_mod
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setattr(sched_mod.envs, "OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", True)
    s = Scheduler.__new__(Scheduler)
    s.kv_connector = conn
    s._local_prefill_coalescing = False
    s._connector_flag = lambda name: getattr(conn, name, False)
    s.block_manager = SimpleNamespace(
        state_offload=state_offload,
        hash_block_size=4,
        can_allocate=lambda seq, **kw: hit_blocks,
    )
    seq = SimpleNamespace(kv_transfer_params=None)
    return s._skip_tier_lookup_on_hbm_hit(seq), conn


def test_scheduler_skips_with_the_probed_hbm_frontier(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn())
    assert skipped and conn.skipped == [40]


def test_scheduler_keeps_the_lookup_when_the_connector_could_load(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn(cannot=False))
    assert not skipped and conn.skipped == []


def test_scheduler_keeps_the_lookup_when_joint_state_offload_is_on(monkeypatch):
    # The joint path is where a tier hit can move the HBM resume point.
    skipped, conn = _gate(monkeypatch, conn=_Conn(), state_offload=object())
    assert not skipped and conn.skipped == []


def test_scheduler_keeps_the_lookup_on_a_refused_probe(monkeypatch):
    skipped, conn = _gate(monkeypatch, conn=_Conn(), hit_blocks=-1)
    assert not skipped and conn.skipped == []
