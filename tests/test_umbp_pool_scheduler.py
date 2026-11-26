# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The scheduler over the `umbp_pool` connector's pool, against the Python
model of the UMBP L1 service: the block manager gets the connector's pool, a
request whose prefix another rank holds waits for its copy and then hits it,
demotion copies ride the batch that needs them and are published after it."""

import threading
import time

import pytest
from conftest import MockConfig
from test_umbp_l1_pool import FakeL1, make_pool, settle
from test_umbp_pool_connector import config_with, transfer

import atom.utils.forward_context as fc
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.connector import UmbpPoolConnectorScheduler
from atom.kv_transfer.pool.umbp.pool import UmbpBlockPool
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus

PROMPT = list(range(100, 120))  # five blocks of four


def _config(**overrides):
    return MockConfig(
        num_kvcache_blocks=16,
        kv_cache_block_size=4,
        enable_prefix_caching=True,
        **overrides,
    )


@pytest.fixture
def ranks(monkeypatch):
    """`connect(share_ratio)`: a connector over one shared L1 model, each call
    a new engine rank's."""
    l1 = FakeL1()
    made = iter(range(16))

    def l1_pool(config, settings, num_blocks, **kwargs):
        return make_pool(l1, f"rank{next(made)}", num_blocks, **kwargs)

    monkeypatch.setattr(umbp_pool, "make_umbp_l1_pool", l1_pool)

    def connect(share_ratio=1.0):
        return UmbpPoolConnectorScheduler(
            config_with(transfer(share_ratio=share_ratio))
        )

    return connect


def _scheduler(monkeypatch, connector):
    monkeypatch.setattr(fc, "get_kvconnector", lambda role, config: connector)
    return Scheduler(_config())


class _HeldCopies:
    """A client whose copies into reserved blocks wait until `go` is set, so
    a test sees the scheduler while one is in flight."""

    def __init__(self, client):
        self._client = client
        self.go = threading.Event()

    def __getattr__(self, name):
        return getattr(self._client, name)

    def fetch(self, pool_id, keys, dst_slots=(), probe_only=False):
        if dst_slots:
            self.go.wait(10)
        return self._client.fetch(pool_id, keys, dst_slots, probe_only)


def _landed(pool: UmbpBlockPool) -> None:
    """Wait until every background copy has finished (not yet indexed)."""
    deadline = time.monotonic() + 10
    for entry in list(pool.umbp._fetches.values()):
        entry[0].result(timeout=max(0.0, deadline - time.monotonic()))


def _output(seq, token=7):
    return ScheduledBatchOutput(
        req_ids=[seq.id],
        token_ids=[(token,)],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
    )


def test_the_block_manager_takes_the_connectors_pool(ranks, monkeypatch):
    sched = _scheduler(monkeypatch, ranks(0.5))
    assert isinstance(sched.block_manager.kv, UmbpBlockPool)
    assert sched.block_manager.kv.offset == 8


def test_a_prefix_another_rank_holds_is_waited_for_then_hit(
    ranks, monkeypatch, seq_factory
):
    peer = BlockManager(_config(), kv_connector=ranks())
    computed = seq_factory(PROMPT)
    peer.allocate(computed, 0)
    peer.hash_blocks(computed, len(PROMPT))
    settle(peer.kv.umbp)

    sched = _scheduler(monkeypatch, ranks())
    pool = sched.block_manager.kv
    copies = pool.umbp._client = _HeldCopies(pool.umbp._client)
    seq = seq_factory(PROMPT + [1, 2])
    sched.add(seq)
    # The prefill-delayer probe starts the copy and, like Phase 2, does not
    # count a head that waits for one as prefillable work.
    assert not sched._can_admit_head_prefill()
    assert pool.fetch_pending(seq.id)

    # Phase 2 parks it while the copy is in flight.
    _, scheduled = sched.schedule()
    assert not scheduled
    assert seq.status == SequenceStatus.WAITING and seq in sched.waiting
    # Landed, but indexed only by the next pass's poll.
    copies.go.set()
    _landed(pool)
    assert pool.fetch_pending(seq.id)

    sched.schedule()
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_cached_tokens == 20
    assert pool.umbp.blocks_fetched == 5


def test_demotions_ride_the_batch_and_are_published_after_it(
    ranks, monkeypatch, seq_factory
):
    sched = _scheduler(monkeypatch, ranks(0.5))
    bm, pool = sched.block_manager, sched.block_manager.kv
    # Cached content in every private block.
    first = seq_factory(list(range(200, 232)))
    bm.allocate(first, 0)
    bm.hash_blocks(first, 32)
    cached = [h for h in (pool.block(b).hash for b in first.block_table) if h != -1]
    bm.deallocate(first)
    assert cached and all(pool.private.lookup(h) != -1 for h in cached)

    seq = seq_factory(list(range(300, 316)))
    sched.add(seq)
    batch, _ = sched.schedule()
    copies = batch.state_maintenance_ops.block_copies
    # Each private block the request took had its content copied into the
    # UMBP part, by this batch, before its forward.
    assert copies and all(src < pool.offset <= dst for src, dst in copies)
    assert all(pool.umbp.lookup(h) == -1 for h in cached)  # not before it ran

    sched.postprocess(list(sched.running), _output(seq))
    demoted = [h for h in cached if pool.umbp.lookup(h) != -1]
    assert len(demoted) == len(copies)
