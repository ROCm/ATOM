# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache complement ("lazy") saves: the CPU tier stores only the blocks the
GPU prefix cache is about to evict (`atom.kv_transfer.offload.lazy_save`).

The scheduler half needs nothing of LMCache but its lookup client: here a
stand-in token database and a fake lookup server over the chunk hashes the
tier holds. The pool is the real `BlockPool`; the end-to-end tests drive the
real `Scheduler` and `BlockManager`, alone and under `multi` beside a P/D
producer. Only the worker test runs LMCache itself.
"""

import array
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig

from atom.config import KVCacheTensor
from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorMetadata,
    KVConnectorOutput,
    LoadOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload import lazy_save, prehashed
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)
from atom.kv_transfer.offload.dense import connector as dense_connector
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.hybrid.m3.connector import (
    M3OffloadConnector,
    M3OffloadScheduler,
)
from atom.kv_transfer.offload.lazy_save import (
    LazyDenseOffloadConnector,
    LazyOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    ATOMRawBytesLMCacheMetadata,
    LMCacheOffloadMetadata,
    PrehashedSave,
)
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams

BLOCK = 4  # tokens per KV block, and per LMCache chunk
STEP = 8  # max_num_batched_tokens
BATCH = STEP // BLOCK  # blocks per demotion: what one step's tokens fill
RESERVE = 2 * BATCH  # free blocks a demotion leaves


class _TokenDatabase:
    """LMCache's `ChunkedTokenDatabase` as the scheduler uses it: chained chunk
    hashes, under its method names."""

    hash_func = hash

    def __init__(self, chunk=BLOCK):
        self.chunk = chunk

    def _get_init_hash(self):
        return 0

    def _hash_tokens(self, tokens, prefix_hash):
        return hash((prefix_hash, tuple(tokens)))

    def process_tokens(self, tokens, make_key=True):
        prev, tokens, chunk = self._get_init_hash(), list(tokens), self.chunk
        for start in range(0, len(tokens) - len(tokens) % chunk, chunk):
            prev = self._hash_tokens(tokens[start : start + chunk], prev)
            yield start, start + chunk, prev


class _Tier:
    """LMCache's lookup server over the chunk hashes the tier holds: the
    contiguous run of those it is sent, from the first (`LMCacheEngine.lookup`)."""

    world_size = 1

    def __init__(self, chunk=BLOCK):
        self.chunk = chunk
        self.stored: set[int] = set()
        self.lookups: list[list[int]] = []

    def send_and_recv_all(self, message):
        self.lookups.append(list(message[0]))
        run = 0
        for chunk_hash in message[0]:
            if chunk_hash not in self.stored:
                break
            run += 1
        return [(run * self.chunk).to_bytes(4, "big")]


class _Producer(KVConnectorSchedulerBase):
    """A P/D producer as `multi` and the scheduler see one: it holds the blocks
    of a finished request it sends until the decode instance has read them."""

    is_producer = True

    def __init__(self):
        self.sending = set()

    def get_num_new_matched_tokens(self, seq):
        return 0, False

    def build_connector_meta(self):
        return ConnectorMetadata()

    def update_state_after_alloc(self, seq):
        pass

    def request_finished(self, seq):
        if (seq.kv_transfer_params or {}).get("do_remote_decode"):
            self.sending.add(seq.id)

    def should_defer_free(self, seq):
        return seq.id in self.sending

    def send_finished(self, req_id):
        self.sending.discard(req_id)

    def source_blocks_released(self, seq):
        pass


def _multi(*subs):
    multi = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    multi._connectors = list(subs)
    multi.is_producer = any(sub.is_producer for sub in subs)
    multi.is_offload = any(sub.is_offload for sub in subs)
    multi.cancels_undispatched_loads = multi.enable_chunked_transfer = False
    multi._load_winner = {}
    return multi


def _config(*, step=STEP, layout="m3", hf_config=None, dcp=1, **kvc):
    """An offload config: M3 by override, or the layout `hf_config`'s model
    resolves to when `layout` is None."""
    if layout is not None:
        kvc = {"offload_layout": layout, **kvc}
    return SimpleNamespace(
        kv_transfer_config={
            "kv_connector": "lmcache_offload",
            "kv_role": "offload",
            "save_policy": "lazy",
            **kvc,
        },
        hf_config=hf_config,
        kv_cache_block_size=BLOCK,
        decode_context_parallel_size=dcp,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        offload_min_load_tokens=BLOCK,
        max_num_batched_tokens=step,
    )


@pytest.fixture
def lmcache(monkeypatch):
    """LMCache's configuration and in-process lookup client, stood in for."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    state = SimpleNamespace(
        chunk=BLOCK, kv_events=False, tier=None, token_database=None
    )

    def create_lookup_client(_cfg, _meta):
        state.tier = _Tier(state.chunk)
        return SimpleNamespace(
            token_database=(
                state.token_database()
                if state.token_database
                else _TokenDatabase(state.chunk)
            ),
            transport=state.tier,
            enable_blending=False,
            reqs_status={},
            clear_lookup_status=lambda sid: None,
        )

    factory = types.ModuleType("lmcache.v1.lookup_client.factory")
    factory.LookupClientFactory = SimpleNamespace(
        create_lookup_client=create_lookup_client
    )
    monkeypatch.setitem(sys.modules, "lmcache.v1.lookup_client.factory", factory)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _c=None: SimpleNamespace(
            chunk_size=state.chunk, enable_kv_events=state.kv_events
        ),
    )
    monkeypatch.setattr(
        offcfg, "build_lmcache_metadata", lambda *_a: SimpleNamespace(engine_id="m3")
    )
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", lambda _c: 1)
    return state


def _lazy(lmcache, blocks):
    """The lazy M3 scheduler, built through its shell, over a pool of `blocks`."""
    shell = LMCacheOffloadConnectorScheduler(_config())
    pool = BlockPool(blocks, cache_policy="slru")
    shell.bind_block_manager(SimpleNamespace(kv=pool, enable_prefix_caching=True))
    return shell._impl, pool, lmcache.tier


def _seq(n_blocks, base=0, **kwargs):
    return Sequence([base + i for i in range(n_blocks * BLOCK)], BLOCK, **kwargs)


def _hold(pool, seq):
    """`seq`'s prompt blocks as a finished prefill leaves them: a prefix hit as
    far as the pool holds them, fresh blocks after, all published."""
    parent = seq.cache_seed
    for i in range(seq.num_prompt_tokens // BLOCK):
        tokens = array.array("i", seq.token_ids[i * BLOCK : (i + 1) * BLOCK])
        parent = hash((parent, tuple(tokens)))
        block_id = pool.lookup(parent)
        if block_id >= 0:
            pool.claim(block_id)
        else:
            block_id = pool.pop()
            pool.allocate(block_id)
            pool.publish(block_id, parent, tokens)
        seq.block_table.append(block_id)
    seq.num_hashed_tokens = len(seq.block_table) * BLOCK
    return list(seq.block_table)


def _finish(sched, pool, seq):
    """`request_finished`, then the deallocation (tail first) it precedes."""
    sched.request_finished(seq)
    for block_id in reversed(seq.block_table):
        pool.free(block_id)
    del seq.block_table[:]


def _cold(lmcache, keyed, *, vacant, blocks):
    """A pool of `blocks` whose coldest free blocks with content are a finished
    request's `keyed`, behind `vacant` vacant ones; a running request holds the
    rest. Returns the scheduler, the pool, the tier, the keyed blocks coldest
    first, and the running request."""
    sched, pool, tier = _lazy(lmcache, blocks)
    seq = _seq(keyed)
    cold = _hold(pool, seq)[::-1]
    filler = _seq(blocks - keyed - vacant, base=10_000)
    _hold(pool, filler)
    _finish(sched, pool, seq)
    return sched, pool, tier, cold, filler


def _one_demotion(lmcache):
    """Two keyed blocks a single demotion takes: 300 blocks make a window of 6,
    the two past the four vacant ones that are the demotion's reserve."""
    return _cold(lmcache, 2, vacant=RESERVE, blocks=300)


def _chain(seq, chunk=BLOCK):
    tokens = list(seq.token_ids[: seq.num_prompt_tokens])
    return [h for _s, _e, h in _TokenDatabase(chunk).process_tokens(tokens)]


def _report(sched, reqs, tier=None, ok=True):
    """What the TP-quorumed M3 worker reports for `reqs`: the save terminal
    and the store channel (`_record_store_terminal`)."""
    operations = {req.save_operation for req in reqs}
    if ok and tier is not None:
        for req in reqs:
            tier.stored.update(req.prehashed_save.hashes)
    sched.process_completions(
        KVConnectorOutput(
            finished_saving=set(operations),
            connector_completions={
                ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, op, ok)
                for op in operations
            },
        )
    )


def _saved(sched, blocks):
    return [sched._block_keys[b][2] for b in blocks]


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(lazy_save.time, "monotonic", lambda: now[0])
    return now


# ------------------------------------------------------------------ keys
def test_finish_keys_published_prompt_blocks_with_lmcache_chunk_hashes(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(3)
    blocks = _hold(pool, seq)
    _finish(sched, pool, seq)
    assert [sched._block_keys[b] for b in blocks] == [
        (pool.block(b).hash, h, False) for b, h in zip(blocks, _chain(seq))
    ]


def test_preemption_keys_the_published_prompt_only(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(3)
    blocks = _hold(pool, seq)
    # A generated block, hashed past the prompt as `hash_decode_blocks` does.
    extra = pool.pop()
    pool.allocate(extra)
    pool.publish(extra, 777, array.array("i", [9] * BLOCK))
    seq.block_table.append(extra)
    seq.num_hashed_tokens += BLOCK
    sched.release_stalled_save(seq)  # preemption's hook, before `deallocate`
    assert set(sched._block_keys) == set(blocks)


def test_a_chain_begun_by_a_preemption_is_finished_by_the_next_lookup(lmcache):
    sched, pool, tier = _lazy(lmcache, 12)
    seq = _seq(6)
    _hold(pool, seq)
    seq.num_hashed_tokens = 2 * BLOCK  # preempted two blocks into its prefill
    sched.release_stalled_save(seq)
    sched.get_num_new_matched_tokens(seq)  # re-admitted: the whole prompt
    assert tier.lookups[-1] == _chain(seq)


def test_a_key_holds_only_while_the_block_keeps_its_hash(lmcache):
    # Room for three pins: the stale key is reached within the budget.
    sched, pool, _, cold, _ = _cold(lmcache, 3, vacant=4, blocks=400)
    # The middle block is taken for other content.
    pool.allocate(cold[1])
    pool.publish(cold[1], 12345, array.array("i", [1] * BLOCK))
    pool.free(cold[1])
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == [cold[0], cold[2]]


def test_rekeying_keeps_the_saved_bit_only_under_the_same_hash(lmcache):
    sched, pool, tier, cold, _ = _one_demotion(lmcache)
    _report(sched, sched.build_connector_meta().requests, tier)
    assert _saved(sched, cold) == [True, True]
    again = _seq(2)  # the same prompt: a full prefix hit on the saved blocks
    assert _hold(pool, again) == cold[::-1]
    _finish(sched, pool, again)
    assert _saved(sched, cold) == [True, True]
    assert sched.get_statistics()["lazy_evicted_unsaved"] == 0
    other = _seq(6, base=1000)  # evicts them for new content
    assert set(cold) <= set(_hold(pool, other))
    _finish(sched, pool, other)
    assert _saved(sched, cold) == [False, False]
    # They were saved: their eviction lost nothing.
    assert sched.get_statistics()["lazy_evicted_unsaved"] == 0


def test_rekeying_an_unsaved_block_counts_it_evicted_unsaved(lmcache):
    sched, pool, _ = _lazy(lmcache, 3)
    first = _seq(3)
    _hold(pool, first)
    _finish(sched, pool, first)
    second = _seq(3, base=1000)  # evicts the first prompt's unsaved blocks
    _hold(pool, second)
    _finish(sched, pool, second)
    assert sched.get_statistics()["lazy_evicted_unsaved"] == 3


def test_blocks_a_load_filled_are_saved_and_a_failed_load_saves_none(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    for ok in (True, False):
        seq = _seq(4, base=100 if ok else 200)
        blocks = _hold(pool, seq)
        operation = LoadOperationId(seq.id, 0)
        sched._active_load_operations[str(seq.id)] = (seq, operation)
        # The load filled chunks [1, 3) from the tier; [0, 1) was the HBM hit.
        seq.offload_load_start_tokens = BLOCK
        seq.offload_loaded_tokens = 3 * BLOCK
        assert (sched.load_finished if ok else sched.load_failed)(operation)
        _finish(sched, pool, seq)
        saved = _saved(sched, blocks)
        assert saved == ([False, True, True, False] if ok else [False] * 4)
    assert sched._loaded_chunks == {}


def test_a_load_range_marks_only_its_own_lifecycle(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(3)
    operation = LoadOperationId(seq.id, 0)
    sched._active_load_operations[str(seq.id)] = (seq, operation)
    seq.offload_load_start_tokens, seq.offload_loaded_tokens = 0, 3 * BLOCK
    # A stale operation records nothing.
    assert not sched.load_finished(LoadOperationId(seq.id, 1))
    assert sched._loaded_chunks == {}
    assert sched.load_finished(operation)
    # A later request reusing the id was not loaded.
    reused = _seq(3, id=seq.id)
    blocks = _hold(pool, reused)
    _finish(sched, pool, reused)
    assert _saved(sched, blocks) == [False] * 3 and sched._loaded_chunks == {}


def test_a_salted_prompt_is_not_keyed(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(2)
    seq.cache_seed = 99
    _hold(pool, seq)
    operation = LoadOperationId(seq.id, 0)
    sched._active_load_operations[str(seq.id)] = (seq, operation)
    seq.offload_load_start_tokens, seq.offload_loaded_tokens = 0, BLOCK
    sched.load_finished(operation)
    _finish(sched, pool, seq)
    assert sched._block_keys == {} and sched._loaded_chunks == {}
    assert sched._perf_counters["lazy_skipped_salted"] == 1


def test_no_request_saves_its_prompt(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(3)
    _hold(pool, seq)
    sched.update_state_after_alloc(seq)
    seq.num_cached_tokens = seq.num_prompt_tokens
    assert sched._save_tracker == {}
    assert sched.build_connector_meta().requests == []
    assert not sched.should_defer_free(seq)


# -------------------------------------------------------------- demotion
def test_a_demotion_is_a_prehashed_save_of_the_coldest_keyed_blocks(lmcache):
    # 400 blocks: 2% is a window of 8, the 4 vacant blocks first.
    sched, pool, _, cold, _ = _cold(lmcache, 4, vacant=4, blocks=400)
    first, second = sched.build_connector_meta().requests
    chain = _chain(_seq(4))[::-1]
    assert first.block_ids == cold[:BATCH] and second.block_ids == cold[BATCH:]
    assert first.prehashed_save == PrehashedSave(0, tuple(chain[:BATCH]))
    assert second.prehashed_save == PrehashedSave(0, tuple(chain[BATCH:]))
    assert first.save_spec.skip_leading_tokens == 0 and first.token_ids == []
    assert first.cache_salt == "" and first.load_spec is None
    assert first.req_id == lazy_save._DEMOTION_REQ_ID
    assert first.save_operation != second.save_operation
    # Pinned for the store, without counting as a reuse.
    assert all(pool.is_used(b) for b in cold)
    assert not set(cold) & pool._reused
    assert pool.num_free == 4 and sched.has_pending_work()


def test_a_demotion_is_the_blocks_one_steps_tokens_fill(lmcache):
    config = _config()
    config.max_num_batched_tokens = 2 * BLOCK + 1
    assert LMCacheOffloadConnectorScheduler(config)._impl._demotion_blocks == 3


def test_vacant_blocks_keep_cached_ones_out_of_reach(lmcache):
    sched, pool, _, cold, _ = _cold(lmcache, 2, vacant=6, blocks=300)
    # The vacant blocks take the next allocations: nothing is near eviction.
    assert sched.build_connector_meta().requests == []
    _hold(pool, _seq(BATCH, base=500))  # a demotion's worth of them spent
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == cold


def test_selection_skips_keyless_saved_and_duplicate_blocks(lmcache):
    sched, pool, _ = _lazy(lmcache, 300)  # a window of 6
    keyless = pool.pop()
    pool.allocate(keyless)
    pool.publish(keyless, 4242, array.array("i", [5] * BLOCK))
    seq = _seq(3)
    blocks = _hold(pool, seq)
    # A duplicate of blocks[1]'s content that the index does not name.
    dup = pool.pop()
    pool.allocate(dup)
    original = pool.block(blocks[1])
    pool.block(dup).update(original.hash, original.token_ids)
    _hold(pool, _seq(294, base=10_000))  # all but one vacant block
    # Coldest first: the keyless block, the duplicate, then the prompt's
    # blocks tail first, the saved one ahead of the two the budget has room for.
    pool.free(keyless)
    pool.free(dup)
    _finish(sched, pool, seq)
    sched._block_keys[dup] = sched._block_keys[blocks[1]]
    sched._block_keys[blocks[2]] = (*sched._block_keys[blocks[2]][:2], True)
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == [blocks[1], blocks[0]]


def test_two_demotions_in_flight_and_one_more_per_freed_slot(lmcache):
    # 600 blocks: 2% is a window of 12, 8 past the vacant blocks.
    sched, pool, tier, cold, filler = _cold(lmcache, 8, vacant=4, blocks=600)
    first, second = sched.build_connector_meta().requests
    assert first.block_ids == cold[:2] and second.block_ids == cold[2:4]
    # Both slots taken: no demotion however the pool moves.
    pool.free(filler.block_table[0])
    assert sched.build_connector_meta().requests == []
    # One slot back: one demotion, though the window holds four candidates.
    _report(sched, [first], tier)
    (third,) = sched.build_connector_meta().requests
    assert third.block_ids == cold[4:6]


def test_a_slot_back_is_used_at_once_though_its_blocks_stay_held(lmcache):
    sched, pool, tier, cold, filler = _cold(lmcache, 8, vacant=4, blocks=600)
    first, _second = sched.build_connector_meta().requests
    pool.free(filler.block_table[0])  # the pool moves while both are busy
    assert sched.build_connector_meta().requests == []
    for block_id in first.block_ids:  # a request hits the first one's blocks
        pool.claim(block_id)
    # Its report frees no block, and nothing else moves before the next build.
    _report(sched, [first], tier)
    (third,) = sched.build_connector_meta().requests
    assert third.block_ids == cold[4:6]


def test_a_spent_budget_is_looked_at_again_though_vacant_blocks_came_back(lmcache):
    # A window of 12 inside the free list: 12 keyed blocks once three running
    # requests take the vacant ones for their unhashed tail blocks.
    sched, pool, tier, cold, _ = _cold(lmcache, 12, vacant=3, blocks=600)
    tails = [pool.pop() for _ in range(3)]
    for block_id in tails:
        pool.allocate(block_id)
    first, second = sched.build_connector_meta().requests  # the whole budget
    assert first.block_ids + second.block_ids == cold[:4]
    for block_id in tails:  # their requests finish before the stores report
        pool.free(block_id)
    _report(sched, [first, second], tier)
    assert sched.has_pending_work()
    third, fourth = sched.build_connector_meta().requests
    assert third.block_ids + fourth.block_ids == cold[4:8]


def test_one_slots_spent_budget_is_looked_at_again_though_a_vacant_block_came_back(
    lmcache,
):
    # The steady state's boundary: one slot back, the look takes one
    # demotion's worth and leaves exactly no slack. A window of 12 inside a
    # free list of 22: 12 keyed blocks, then 10 nobody keyed.
    sched, pool, tier, cold, _ = _cold(lmcache, 12, vacant=11, blocks=600)
    for block_id in _hold(pool, _seq(10, base=20_000)):
        pool.free(block_id)
    tail = pool.pop()  # a running request's unhashed tail: no vacant block left
    pool.allocate(tail)
    first, second = sched.build_connector_meta().requests
    assert first.block_ids + second.block_ids == cold[:4]
    _report(sched, [first], tier)  # one slot back: one demotion's budget
    (third,) = sched.build_connector_meta().requests
    assert third.block_ids == cold[4:6]
    pool.free(tail)  # its request finishes before the next store reports
    _report(sched, [second], tier)
    assert [req.block_ids for req in sched.build_connector_meta().requests] == [
        cold[6:8]
    ]


@pytest.mark.parametrize("deferred", [False, True])
def test_a_probationary_free_ahead_of_the_protected_queue_is_looked_at(
    lmcache, deferred
):
    sched, pool, tier = _lazy(lmcache, 600)  # SLRU: a window of 12
    reused = _seq(20)
    _hold(pool, reused)
    running = _seq(6, base=50_000)
    _hold(pool, running)
    _hold(pool, _seq(600 - 20 - 6, base=10_000))  # the rest of the pool
    _finish(sched, pool, reused)
    _hold(pool, reused)  # hit again: protected from now on
    _finish(sched, pool, reused)
    assert len(pool._cached) == 0 and len(pool._protected) == 20
    # The window is the 12 coldest protected blocks; demotions store them.
    for _ in range(10):
        if not sched.has_pending_work():
            break
        _report(sched, sched.build_connector_meta().requests, tier)
    assert not sched.has_pending_work() and not sched._demotions
    # The running request finishes: its keyed, unsaved blocks are probationary,
    # which HBM evicts before any protected block. A free is no slide.
    if deferred:  # a P/D send holds them past the finish and the next look
        sched.request_finished(running)
        sched.build_connector_meta()
        for block_id in reversed(running.block_table):
            pool.free(block_id)
        sched.source_blocks_released(running)
    else:
        _finish(sched, pool, running)
    coldest = list(pool.next_evicted(6))
    assert all(not sched._block_keys[b][2] for b in coldest)
    assert sched.has_pending_work()
    first, second = sched.build_connector_meta().requests
    assert first.block_ids + second.block_ids == coldest[:4]


def test_the_window_is_two_percent_of_a_larger_pool(lmcache):
    sched, _pool, _, cold, _ = _cold(lmcache, 10, vacant=0, blocks=500)
    for block_id in cold[:6]:  # past two demotions' worth, all saved
        sched._block_keys[block_id] = (*sched._block_keys[block_id][:2], True)
    first, second = sched.build_connector_meta().requests
    assert first.block_ids + second.block_ids == cold[6:]


def test_a_demotion_leaves_two_demotions_worth_free(lmcache):
    sched, pool, _, cold, filler = _cold(lmcache, 3, vacant=3, blocks=400)
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == cold[:2] and pool.num_free == RESERVE
    # None at all once the free list is down to that.
    pool.free(filler.block_table[0])
    _hold(pool, _seq(1, base=700))
    assert sched.build_connector_meta().requests == []


def test_a_reported_demotion_is_saved_and_unpinned_coldest(lmcache):
    sched, pool, tier, cold, _ = _cold(lmcache, 4, vacant=4, blocks=400)
    reqs = sched.build_connector_meta().requests
    _report(sched, reqs, tier)
    assert _saved(sched, cold) == [True] * 4
    assert pool.num_free == 8 and not sched._demotions
    # Back where they were found, each demotion in its order: the next
    # allocations evict them first.
    evicted = list(pool.next_evicted(8))
    assert sorted(evicted) == sorted(cold)
    for req in reqs:
        start = evicted.index(req.block_ids[0])
        assert evicted[start : start + len(req.block_ids)] == req.block_ids
    stats = sched.get_statistics()
    assert stats["save_requests"] == 2 and stats["saved_tokens"] == 4 * BLOCK
    assert stats["saves_pending"] == 0
    assert sched.build_connector_meta().requests == []
    assert not sched.has_pending_work()


def test_only_the_store_report_ends_a_demotion(lmcache):
    sched, pool, _, cold, _ = _one_demotion(lmcache)
    (req,) = sched.build_connector_meta().requests
    op = req.save_operation
    for channel in (
        DENSE_PAGE_SOURCE_SAFE_CHANNEL,
        DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    ):
        sched.connector_completion(ConnectorCompletion(channel, op, True))
    assert op in sched._demotions and all(pool.is_used(b) for b in cold)
    # Handled, and no request's save terminal.
    done = ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, op, True)
    assert sched.connector_completion(done) is None
    assert _saved(sched, cold) == [True, True] and not sched._demotions


def test_a_demotion_whose_fence_fails_is_unsaved_and_backs_off(lmcache, monkeypatch):
    sched, pool, _, cold, _ = _one_demotion(lmcache)
    meta = sched.build_connector_meta()

    def no_fence():
        raise RuntimeError("HIP error")

    monkeypatch.setattr(dense_connector, "record_save_fence", no_fence)
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._init_worker_common(
        SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
    )
    worker._early_release = True  # as the M3 worker runs
    worker.start_load_kv(meta)
    worker.close()
    sched.process_completions(worker.get_finished())
    assert _saved(sched, cold) == [False, False] and pool.num_free == 6
    assert not sched._demotions and sched._perf_counters["lazy_demotion_failures"]


def test_a_failed_demotion_is_unsaved_and_retried_after_a_backoff(lmcache, clock):
    # One keyed block in a window of 6 with room for more: the look that took
    # it would not be due again for another eviction.
    sched, pool, tier = _lazy(lmcache, 300)
    _hold(pool, _seq(293, base=10_000))
    keyed, warm = _seq(1), _seq(2, base=20_000)
    _hold(pool, keyed)
    _hold(pool, warm)
    _finish(sched, pool, keyed)
    for block_id in warm.block_table:  # keyless, behind it
        pool.free(block_id)
    (req,) = sched.build_connector_meta().requests
    assert sched.has_pending_work()  # no look is due, but its report is
    _report(sched, [req], tier, ok=False)
    assert _saved(sched, req.block_ids) == [False] and tier.stored == set()
    assert sched.get_statistics()["save_requests"] == 0
    clock[0] += lazy_save._DEMOTION_BACKOFF_S - 0.01
    assert sched.build_connector_meta().requests == []
    assert sched.has_pending_work()  # back at the cold end, unsaved: a look is owed
    clock[0] += 0.02
    (retry,) = sched.build_connector_meta().requests
    assert retry.block_ids == req.block_ids


def test_a_demotion_that_never_reports_is_unpinned_unsaved(lmcache, clock, monkeypatch):
    sched, pool, tier, cold, _ = _one_demotion(lmcache)
    monkeypatch.setattr(sched, "save_abandon_timeout_s", lambda: 30.0)
    (stalled,) = sched.build_connector_meta().requests
    clock[0] += 29.0
    sched.build_connector_meta()
    assert pool.num_free == 4  # still pinned within the window
    clock[0] += 1.0
    (again,) = sched.build_connector_meta().requests
    assert again.block_ids == stalled.block_ids
    assert stalled.save_operation not in sched._demotions
    assert _saved(sched, cold) == [False, False]
    assert sched.get_statistics()["saves_pending"] == 1
    # The late report changes nothing: no mark, no second unpin.
    _report(sched, [stalled], tier)
    assert _saved(sched, cold) == [False, False]
    assert pool.num_used + pool.num_free == len(pool.blocks)
    assert all(pool.is_used(b) for b in again.block_ids)


def test_no_abandon_window_keeps_a_demotion_pinned(lmcache, clock, monkeypatch):
    sched, pool, _, cold, _ = _one_demotion(lmcache)
    monkeypatch.setattr(sched, "save_abandon_timeout_s", lambda: 0.0)
    (req,) = sched.build_connector_meta().requests
    clock[0] += 1e6
    sched.build_connector_meta()
    assert req.save_operation in sched._demotions
    assert all(pool.is_used(b) for b in cold)


@pytest.mark.parametrize("worker_type", [M3OffloadConnector, LazyDenseOffloadConnector])
def test_the_worker_stores_a_demotion_under_its_prompts_keys_behind_the_fence(
    lmcache, monkeypatch, worker_type
):
    pytest.importorskip("lmcache")
    from test_offload_prehashed_save import _engine, _Fence
    from test_offload_prehashed_save import _tdb as real_token_database

    def token_database():
        tdb = real_token_database()
        tdb.chunk_size = BLOCK
        return tdb

    lmcache.token_database = token_database
    sched, pool, _ = _lazy(lmcache, 300)  # a window of 6
    _hold(pool, _seq(300 - 2 - RESERVE, base=10_000))
    seqs = [_seq(1, base=0), _seq(1, base=100)]
    blocks = []  # in the order the deallocations freed them
    for seq in seqs:
        blocks += _hold(pool, seq)
        _finish(sched, pool, seq)
    meta = sched.build_connector_meta()
    (req,) = meta.requests
    fences = []
    monkeypatch.setattr(
        dense_connector,
        "record_save_fence",
        lambda: fences.append(_Fence()) or fences[-1],
    )
    engine = _engine()
    engine.token_database = token_database()
    worker = worker_type.__new__(worker_type)
    worker._init_worker_common(
        SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
    )
    worker.chunk_size, worker._engine, worker._rank = BLOCK, engine, 0
    worker._early_release = True  # as either worker runs
    worker.start_load_kv(meta)
    worker.close()
    # Behind the step's producer fence, though the blocks were computed long ago.
    assert len(fences) == 1 and fences[0].waited == 1
    keys, _objs = engine.storage_manager.put[-1]
    # The keys LMCache's token path computes for those chunks of either prompt,
    # as a lookup or a retrieve does.
    want = {
        key.chunk_hash: key
        for seq in seqs
        for _s, _e, key in engine.token_database.process_tokens(list(seq.token_ids))
    }
    assert keys == [want[h] for h in req.prehashed_save.hashes]
    # Chunk i is block_ids[i].
    starts, ends, kwargs = engine.gpu_connector.calls[-1]
    assert kwargs["block_ids"] == req.block_ids == blocks
    gpu = SimpleNamespace(block_size=BLOCK)
    assert [
        BlockGPUConnector._range_block_ids(gpu, req.block_ids, s, e)
        for s, e in zip(starts, ends)
    ] == [[b] for b in req.block_ids]
    sched.process_completions(worker.get_finished())
    assert _saved(sched, blocks) == [True, True]


# ------------------------------------------------- when the window is looked at
def _counted(monkeypatch, pool):
    looks = []
    real = pool.next_evicted
    monkeypatch.setattr(pool, "next_evicted", lambda n: looks.append(n) or real(n))
    return looks


def test_a_window_inside_the_free_list_is_looked_at_again_once_it_slid(
    lmcache, monkeypatch
):
    # 300 blocks: a window of 6, so 2 beyond two demotions' worth of margin.
    sched, pool, _ = _lazy(lmcache, 300)
    filler = _hold(pool, _seq(280, base=10_000))
    keyless = _hold(pool, _seq(20, base=20_000))
    for block_id in keyless:  # the free list: 20 blocks nobody keyed
        pool.free(block_id)
    looks = _counted(monkeypatch, pool)
    sched.build_connector_meta()
    assert len(looks) == 1 and not sched.has_pending_work()
    for _ in range(100):  # idle polls
        sched.build_connector_meta()
    pool.free(filler[0])  # a free lands past the window
    sched.build_connector_meta()
    pool.allocate(pool.pop())  # one eviction: less than a demotion's worth
    sched.build_connector_meta()
    assert len(looks) == 1 and not sched.has_pending_work()
    pool.allocate(pool.pop())  # slid a demotion's worth
    assert sched.has_pending_work()
    sched.build_connector_meta()
    assert len(looks) == 2


def test_a_window_of_two_demotions_is_looked_at_on_every_slide(lmcache, monkeypatch):
    sched, pool, _ = _lazy(lmcache, 30)  # a window of 4: no margin to spare
    _hold(pool, _seq(10, base=10_000))
    for block_id in _hold(pool, _seq(20, base=20_000)):
        pool.free(block_id)
    looks = _counted(monkeypatch, pool)
    sched.build_connector_meta()
    sched.build_connector_meta()
    assert len(looks) == 1
    pool.allocate(pool.pop())
    sched.build_connector_meta()
    assert len(looks) == 2


def test_pinning_slides_the_window_like_evictions(lmcache, monkeypatch):
    # 400 blocks: a window of 8, whose last 2 are the keyed blocks; pinning
    # them slides it a demotion's worth, so a look is due at once.
    sched, pool, _, cold, _ = _cold(lmcache, 2, vacant=6, blocks=400)
    looks = _counted(monkeypatch, pool)
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == cold
    assert sched.has_pending_work() and len(looks) == 1
    sched.build_connector_meta()
    assert len(looks) == 2


def test_a_window_past_the_free_list_is_looked_at_on_any_change(lmcache, monkeypatch):
    # A window of 6 over a free list of 5 nobody keyed, with room for a pin.
    sched, pool, _ = _lazy(lmcache, 300)
    _hold(pool, _seq(294, base=10_000))
    seq = _seq(1)
    blocks = _hold(pool, seq)
    for block_id in _hold(pool, _seq(5, base=20_000)):
        pool.free(block_id)
    _hold(pool, _seq(1))  # an admission hits seq's block
    _finish(sched, pool, seq)  # keyed, and still held
    looks = _counted(monkeypatch, pool)
    assert sched.build_connector_meta().requests == []
    assert len(looks) == 1 and not sched.has_pending_work()
    # The admission is taken back: the keyed block comes back inside the
    # window with no request keying it, and no slide, but a look.
    pool.free(blocks[0])
    assert sched.has_pending_work()
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == blocks and len(looks) == 2


def test_a_look_without_room_for_a_pin_is_retried_on_any_change(lmcache):
    # 30 blocks: a window of 4 inside a free list of 4, all of it reserve.
    sched, pool, _, cold, filler = _cold(lmcache, 4, vacant=0, blocks=30)
    assert sched.build_connector_meta().requests == []
    pool.free(filler.block_table[0])  # behind the window: room, but no slide
    (req,) = sched.build_connector_meta().requests
    assert req.block_ids == cold[:1]


# ---------------------------------------------------------------- lookup
def test_a_lookup_asks_only_past_the_hbm_hit(lmcache):
    sched, _pool, tier = _lazy(lmcache, 12)
    seq = _seq(6)
    chain = _chain(seq)
    tier.stored.update(chain[2:5])
    seq.offload_hbm_probe_tokens = 2 * BLOCK
    assert sched.get_num_new_matched_tokens(seq) == (5 * BLOCK, True)
    assert tier.lookups == [chain[2:6]]
    assert sched._load_specs[str(seq.id)].lmcache_cached_tokens == 5 * BLOCK
    # Nothing past the HBM hit is no hit, whatever the tier holds before it.
    other = _seq(6)
    tier.stored = set(chain[:5])
    other.offload_hbm_probe_tokens = 5 * BLOCK
    assert sched.get_num_new_matched_tokens(other) == (0, False)
    assert tier.lookups[-1] == chain[5:6]
    # A prompt HBM holds whole, or past its end (a preempted request's
    # generated blocks), asks nothing.
    for probe in (6 * BLOCK, 8 * BLOCK):
        whole = _seq(6)
        whole.offload_hbm_probe_tokens = probe
        assert sched.get_num_new_matched_tokens(whole) == (0, False)
    assert len(tier.lookups) == 2


def test_a_remembered_hit_answers_its_own_start_only(lmcache):
    sched, _pool, tier = _lazy(lmcache, 12)
    seq = _seq(6)
    seq.offload_hbm_probe_tokens = BLOCK
    for _ in range(3):  # a waiting head, asked every step
        assert sched.get_num_new_matched_tokens(seq) == (0, False)
        sched.build_connector_meta()  # releases the step's lookup pin
    assert len(tier.lookups) == 1
    seq.offload_hbm_probe_tokens = 3 * BLOCK
    sched.get_num_new_matched_tokens(seq)
    assert len(tier.lookups) == 2 and tier.lookups[-1] == _chain(seq)[3:]


# ----------------------------------------------------------- refusals
@pytest.mark.parametrize(
    "setup,match",
    [
        (lambda s, mp: setattr(s, "chunk", 2 * BLOCK), "chunk size"),
        (lambda s, mp: mp.setenv("PYTHONHASHSEED", "random"), "prehashed"),
        (lambda s, mp: setattr(s, "kv_events", True), "KV events"),
    ],
)
def test_an_unsupported_tier_is_refused(lmcache, monkeypatch, setup, match):
    setup(lmcache, monkeypatch)
    with pytest.raises(ValueError, match=match):
        LMCacheOffloadConnectorScheduler(_config())


def test_a_chain_lmcache_would_not_compute_is_refused_at_startup(lmcache, monkeypatch):
    real = prehashed._chunk_list
    # Corrupt only the per-chunk slices the chain hashes, not the whole-prefix
    # slice the check hands to LMCache's `process_tokens`.
    monkeypatch.setattr(
        prehashed,
        "_chunk_list",
        lambda t, s, e: (
            real(t, s, e) if e - s > BLOCK else [x + 1 for x in real(t, s, e)]
        ),
    )
    with pytest.raises(ValueError, match="do not match"):
        LMCacheOffloadConnectorScheduler(_config())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"kv_role": "kv_consumer"}, "kv_role"),
        ({"save_policy": "sometimes"}, "unknown save_policy"),
        (
            {"save_policy": "eager", "kv_connector_extra_config": {"save_policy": 1}},
            "unknown save_policy",
        ),
        (
            {"save_policy": 1, "kv_connector_extra_config": {"save_policy": "lazy"}},
            "unknown save_policy",
        ),
    ],
)
def test_an_unsupported_config_is_refused(lmcache, overrides, match):
    with pytest.raises(ValueError, match=match):
        LMCacheOffloadConnectorScheduler(_config(**overrides))


def test_pipeline_parallelism_is_refused(lmcache):
    config = _config()
    config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline parallelism"):
        LMCacheOffloadConnectorScheduler(config)


def test_lazy_needs_prefix_caching(lmcache):
    shell = LMCacheOffloadConnectorScheduler(_config())
    with pytest.raises(ValueError, match="prefix-cache"):
        shell.bind_block_manager(
            SimpleNamespace(kv=BlockPool(4), enable_prefix_caching=False)
        )


@pytest.mark.parametrize(
    "kvc,lazy",
    [
        ({}, True),
        ({"save_policy": None}, False),
        ({"kv_connector_extra_config": {"lmcache.chunk_size": BLOCK}}, True),
        (
            {"save_policy": None, "kv_connector_extra_config": {"save_policy": "lazy"}},
            True,
        ),
        ({"kv_connector_extra_config": {"save_policy": "eager"}}, False),
    ],
)
def test_save_policy_is_read_from_the_extra_config_then_the_top_level(
    lmcache, kvc, lazy
):
    config = _config(**kvc)
    if config.kv_transfer_config["save_policy"] is None:
        del config.kv_transfer_config["save_policy"]
    shell = LMCacheOffloadConnectorScheduler(config)
    assert isinstance(shell._impl, LazyOffloadScheduler) is lazy
    assert shell.needs_hbm_probe_before_lookup is lazy
    if not lazy:
        assert type(shell._impl) is M3OffloadScheduler


def test_multi_probes_first_only_for_a_lazy_tier(lmcache):
    config = _config()
    del config.kv_transfer_config["save_policy"]
    for shell, lazy in (
        (LMCacheOffloadConnectorScheduler(config), False),
        (LMCacheOffloadConnectorScheduler(_config()), True),
    ):
        assert _multi(_Producer(), shell).needs_hbm_probe_before_lookup is lazy


# ------------------------------------------------- the scheduler, end to end
def _engine_with(connector, blocks, *, step=STEP, **overrides):
    engine = Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            num_kvcache_blocks=blocks,
            kv_cache_block_size=BLOCK,
            max_num_batched_tokens=step,
            **overrides,
        )
    )
    engine.kv_connector = connector
    connector.bind_block_manager(engine.block_manager)
    return engine


def _step(engine):
    """One engine step: a sampled token for every scheduled request. Returns
    its connector metadata, or an idle build's when nothing ran."""
    result = engine.schedule()
    if result is None or result[0] is None:
        connector = engine.kv_connector
        return (
            connector.build_connector_meta() if connector.has_pending_work() else None
        )
    batch, seqs = result
    engine.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(
            req_ids=list(seqs),
            token_ids=[(7,)] * len(seqs),
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    return batch.connector_meta_output


def _serve(engine, seq):
    """Admit, prefill and finish `seq`; returns each step's metadata."""
    engine.add(seq)
    metas = []
    while seq.status != SequenceStatus.FINISHED:
        assert len(metas) < 100, "the request never finished"
        metas.append(_step(engine))
    return metas


def _demote_and_store(engine, tier, metas=()):
    """Drive idle metadata builds until the tier holds every demotion."""
    sched = engine.kv_connector
    for _ in range(10):
        meta = sched.build_connector_meta()
        reqs = [r for m in (*metas, meta) if m for r in m.requests if r.prehashed_save]
        metas = ()
        if not reqs:
            if not sched.has_pending_work():
                return
            continue
        for req in reqs:
            tier.stored.update(req.prehashed_save.hashes)
        engine._update_from_kv_xfer_finished(
            KVConnectorOutput(
                finished_saving={req.save_operation for req in reqs},
                connector_completions={
                    ConnectorCompletion(
                        DENSE_PAGE_STORE_CHANNEL, r.save_operation, True
                    )
                    for r in reqs
                },
            )
        )
    raise AssertionError("demotions did not settle")


def test_admission_probes_hbm_before_the_lookup_only_when_asked(lmcache, monkeypatch):
    # Detaching a lookup probes HBM first too; with it off only the flag does.
    monkeypatch.setenv("OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT", "0")
    shell = LMCacheOffloadConnectorScheduler(_config())
    engine = _engine_with(shell, blocks=16)
    first = _seq(3, sampling_params=SamplingParams(max_tokens=1))
    _serve(engine, first)
    seen = []
    real = shell._impl.get_num_new_matched_tokens
    monkeypatch.setattr(
        shell._impl,
        "get_num_new_matched_tokens",
        lambda seq: seen.append(seq.offload_hbm_probe_tokens) or real(seq),
    )
    again = _seq(4, sampling_params=SamplingParams(max_tokens=1))
    _serve(engine, again)
    # HBM held the first prompt's 3 blocks when the lookup ran.
    assert seen == [3 * BLOCK]
    # A request that does not fit probes no hit: it asks from the start.
    too_long = _seq(20, sampling_params=SamplingParams(max_tokens=1))
    too_long.offload_hbm_probe_tokens = 3 * BLOCK
    assert engine._admission_probe(too_long)[0].num_cached_blocks < 0
    assert too_long.offload_hbm_probe_tokens == 0
    # Probing for the flag detaches nothing while detaching is off.
    monkeypatch.setattr(shell, "can_detach_lookup", lambda seq, n: True)
    assert engine._admission_probe(_seq(2))[2] is False
    monkeypatch.setattr(type(shell), "needs_hbm_probe_before_lookup", False)
    assert engine._admission_probe(_seq(2)) == (None, (0, 0), False)


def _evict_and_reload(engine, tier, first):
    """Serve a filler that evicts two of `first`'s demoted blocks, then `first`'s
    prompt again. Returns the request and the load it parked for."""
    filler = _seq(6, base=1000, sampling_params=SamplingParams(max_tokens=1))
    _demote_and_store(engine, tier, _serve(engine, filler))
    prompt = list(first.token_ids[: first.num_prompt_tokens])
    again = Sequence(prompt, BLOCK, sampling_params=SamplingParams(max_tokens=1))
    engine.add(again)
    batch, _ = engine.schedule()
    (load,) = [r for r in batch.connector_meta_output.requests if r.load_spec]
    return again, load


def _assert_reloads_past_the_hbm_hit(engine, tier, first):
    chain = _chain(first)
    # No request stored its prompt; demotions stored the four coldest blocks,
    # all the window of 8 holds past the 4 vacant ones.
    assert tier.stored == set()
    _demote_and_store(engine, tier)
    assert tier.stored == set(chain[2:])
    again, load = _evict_and_reload(engine, tier, first)
    # The pin reserve let one demotion of two out at a time, and the second
    # to report went back coldest: the filler evicted blocks 2 and 3. HBM's
    # hit ends there, and the lookup asks only past it.
    assert again.offload_hbm_probe_tokens == 2 * BLOCK
    assert tier.lookups[-1] == chain[2:]
    assert again.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    # The tier holds the rest; the prompt's last block is always recomputed.
    assert load.load_spec.hbm_cached_tokens == 2 * BLOCK
    assert load.load_spec.lmcache_cached_tokens == 5 * BLOCK


def test_a_lazy_tier_holds_what_hbm_evicted_and_reloads_it(lmcache):
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    engine = _engine_with(shell, 10, step=4 * BLOCK)
    first = _seq(6, sampling_params=SamplingParams(max_tokens=1))
    metas = _serve(engine, first)
    assert not any(req.prehashed_save for meta in metas for req in meta.requests)
    assert len(shell._impl._block_keys) == 6  # keyed at the finish
    _assert_reloads_past_the_hbm_hit(engine, lmcache.tier, first)


def test_under_multi_a_prefill_node_demotes_once_the_send_frees(lmcache):
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    multi = _multi(_Producer(), shell)
    engine = _engine_with(multi, 10, step=4 * BLOCK)
    first = _seq(
        6,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={"do_remote_decode": True},
    )
    _serve(engine, first)
    # The send holds the blocks; the finish has keyed them already.
    assert first.id in engine.deferred_free_blocks
    assert len(shell._impl._block_keys) == 6
    assert multi.build_connector_meta().requests == []
    engine._update_from_kv_xfer_finished(KVConnectorOutput(finished_sending={first.id}))
    assert first.id not in engine.deferred_free_blocks
    _assert_reloads_past_the_hbm_hit(engine, lmcache.tier, first)


def test_pins_leave_two_growing_decodes_room_while_a_store_takes_16_steps(lmcache):
    """Pins leave two demotions' worth free when taken. Two decodes grow into
    a pool whose other free blocks are a finished prompt's keyed ones while
    each store takes 16 steps to report: neither is preempted. A store slower
    than they take to grow through what the pins left can still preempt one."""
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    engine = _engine_with(shell, 20, step=4 * BLOCK, max_model_len=10**6)
    reports = []  # (due step, request)
    tick = [0]

    def run(until):
        while not until():
            tick[0] += 1
            for item in [x for x in reports if x[0] <= tick[0]]:
                reports.remove(item)
                _report(shell._impl, [item[1]], lmcache.tier)
            meta = _step(engine)
            for req in meta.requests if meta else ():
                if req.prehashed_save:
                    reports.append((tick[0] + 16, req))
            assert tick[0] < 2000

    a = _seq(8, sampling_params=SamplingParams(max_tokens=1))
    engine.add(a)
    run(lambda: a.status == SequenceStatus.FINISHED and not reports)
    # Two generations take the rest of the pool and grow a block every
    # `BLOCK` tokens: a's 8 cached blocks are the only free ones.
    b = _seq(5, base=10_000, sampling_params=SamplingParams(max_tokens=12))
    c = _seq(5, base=20_000, sampling_params=SamplingParams(max_tokens=12))
    engine.add(b)
    engine.add(c)
    done = SequenceStatus.FINISHED
    run(lambda: b.status == done and c.status == done)
    assert engine.total_preemptions == 0


# ------------------------------------------- the layouts lazy saves run on
# Checkpoints' configs as `select_offload_layout` reads them.
_GLM52 = SimpleNamespace(  # MLA with a DSA indexer: the dense layout
    model_type="deepseek_v3",
    architectures=["GlmMoeDsaForCausalLM"],
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_topk=2048,
)
_QWEN3 = SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
_HYBRIDS = {
    "hybrid": SimpleNamespace(
        model_type="deepseek_v3",
        architectures=["DeepseekV4ForCausalLM"],
        kv_lora_rank=512,
        compress_ratios=[4],
    ),
    "kimi_k3": SimpleNamespace(model_type="kimi_linear", architectures=[]),
    "qwen": SimpleNamespace(model_type="qwen3_next", architectures=[]),
}


def _shells(config):
    """The scheduler and worker the shells build for `config`."""
    scheduler = LMCacheOffloadConnectorScheduler(config)
    worker = LMCacheOffloadConnector(config)
    worker.close()
    return type(scheduler._impl), type(worker._impl)


@pytest.mark.parametrize("hf_config", [_GLM52, _QWEN3])
def test_a_dense_model_gets_the_lazy_classes_and_keeps_eager_ones(lmcache, hf_config):
    lazy = _config(layout=None, hf_config=hf_config)
    assert _shells(lazy) == (LazyOffloadScheduler, LazyDenseOffloadConnector)
    eager = _config(layout=None, hf_config=hf_config, save_policy="eager")
    assert _shells(eager) == (DenseOffloadScheduler, DenseOffloadConnector)
    # M3 shares the lazy scheduler, and its worker stores prehashed saves.
    assert _shells(_config()) == (LazyOffloadScheduler, M3OffloadConnector)
    assert _shells(_config(save_policy="eager")) == (
        M3OffloadScheduler,
        M3OffloadConnector,
    )


@pytest.mark.parametrize("layout", sorted(_HYBRIDS))
def test_a_layout_with_per_request_state_is_refused_on_both_sides(lmcache, layout):
    config = _config(layout=None, hf_config=_HYBRIDS[layout])
    for build in (LMCacheOffloadConnectorScheduler, LMCacheOffloadConnector):
        with pytest.raises(
            ValueError, match=f"{layout!r} layout yet.*state checkpoint"
        ):
            build(config)


def test_under_dcp_a_chunk_spans_a_block_times_the_dcp_size(lmcache):
    config = _config(layout=None, hf_config=_GLM52, dcp=2)
    with pytest.raises(ValueError, match=r"tokens a block spans \(8"):
        LMCacheOffloadConnectorScheduler(config)
    lmcache.chunk = 2 * BLOCK
    shell = LMCacheOffloadConnectorScheduler(config)
    assert shell._impl.virtual_block_size == 2 * BLOCK


class _CpuGpuConnector:
    """The worker's GPU connector on the CPU: `BlockGPUConnector`'s own map from
    token ranges to block ids, and each block's bytes in the codec's segment
    order (what its fused staging kernels move), chunk by chunk."""

    def __init__(self, codec, block, chunk, virtual_block):
        self._grid = BlockGPUConnector(
            codec, block, chunk_size=chunk, virtual_block_size=virtual_block
        )
        self._planes = [
            seg.view(torch.uint8).reshape(codec.num_blocks, -1)
            for seg in codec._segments
        ]

    def batched_from_gpu(self, memory_objs, starts, ends, *, producer_event, **kw):
        assert producer_event is not None
        groups = self._grid._ranges_to_block_ids(starts, ends, **kw)
        for obj, ids in zip(memory_objs, groups, strict=True):
            obj.tensor.copy_(
                torch.cat([plane[ids].reshape(-1) for plane in self._planes])
            )

    def batched_to_gpu(self, memory_objs, starts, ends, **kw):
        groups = self._grid._ranges_to_block_ids(starts, ends, **kw)
        for obj, ids in zip(memory_objs, groups, strict=True):
            offset = 0
            for plane in self._planes:
                size = len(ids) * plane.shape[1]
                plane[ids] = obj.tensor[offset : offset + size].view(len(ids), -1)
                offset += size


_NO_STATS = SimpleNamespace(
    profile_process_tokens=nullcontext,
    profile_from_gpu=nullcontext,
    profile_put=nullcontext,
)


def _tier_engine(lmcache, codec, block, virtual_block):
    """LMCache's engine as the dense worker drives it: chunks in a dict, keyed
    as the scheduler's lookups see them, over `_CpuGpuConnector`."""
    chunk = lmcache.chunk
    stored = {}
    connector = _CpuGpuConnector(codec, block, chunk, virtual_block)

    def allocate(shapes, dtypes, busy_loop=True, fmt=None):
        return SimpleNamespace(tensor=torch.empty(shapes[0], dtype=dtypes[0]))

    def batched_put(keys, memory_objs, transfer_spec=None, location=None):
        for (chunk_hash, _configs), obj in zip(keys, memory_objs, strict=True):
            stored[chunk_hash] = obj
            lmcache.tier.stored.add(chunk_hash)

    def retrieve(tokens, mask, block_ids, req_id, **_salt):
        loaded = torch.zeros(len(tokens), dtype=torch.bool)
        objs, starts, ends = [], [], []
        for start, end, chunk_hash in _TokenDatabase(chunk).process_tokens(
            tokens.tolist()
        ):
            if not mask[start]:
                continue
            if chunk_hash not in stored:
                break
            objs.append(stored[chunk_hash])
            starts.append(start)
            ends.append(end)
        connector.batched_to_gpu(objs, starts, ends, block_ids=block_ids)
        for start, end in zip(starts, ends):
            loaded[start:end] = True
        return loaded

    return SimpleNamespace(
        token_database=SimpleNamespace(_make_key_by_hash=lambda h, c: (h, c)),
        storage_manager=SimpleNamespace(allocate=allocate, batched_put=batched_put),
        gpu_connector=connector,
        metadata=ATOMRawBytesLMCacheMetadata(
            SimpleNamespace(chunk_size=chunk),
            atom_block_size=virtual_block,
            bytes_per_block=codec.bytes_per_block,
        ),
        stats_monitor=SimpleNamespace(
            on_store_request=lambda _n: _NO_STATS, on_store_finished=lambda *_a: None
        ),
        config=SimpleNamespace(get_extra_config_value=lambda _n, d=None: d),
        kv_events_enabled=False,
        fmt=None,
        store_location=None,
        is_healthy=lambda: True,
        is_frozen=lambda: False,
        _is_passive=lambda: False,
        retrieve=retrieve,
        lookup_unpin=lambda _req_id: None,
        stored=stored,
    )


def _dense_kv(geometry, blocks):
    """Two layers of a dense model's paged KV as it registers them: GLM-5.2's
    fp8 MLA latent, token-major, with its fp4 DSA index keys and their e8m0
    scales; or Qwen3's x-packed bf16 K beside its V."""
    kv = {}
    for layer in range(2):
        if geometry == "mla":
            kvt = KVCacheTensor(
                layer,
                k_cache=torch.zeros(blocks * BLOCK, 1, 576, dtype=torch.uint8),
                index_cache=torch.zeros(blocks, 1, 4, BLOCK, 16, dtype=torch.uint8),
                index_scale=torch.zeros(blocks, BLOCK, 4, dtype=torch.uint8),
            )
        else:
            kvt = KVCacheTensor(
                layer,
                k_cache=torch.zeros(blocks, 2, 16, BLOCK, 8, dtype=torch.bfloat16),
                v_cache=torch.zeros(blocks, 2, BLOCK, 128, dtype=torch.bfloat16),
            )
        kv[f"l{layer}"] = kvt
    return kv


class _DenseModel:
    """A dense model served with lazy saves, end to end on the CPU: the real
    scheduler and block manager (alone, or under `multi` beside a P/D
    producer), the real lazy worker over the dense codec and a real KV
    geometry, and a dict tier. Each step a "forward" writes every block it
    completes with bytes its prefix determines, so a block the tier gave back
    holds them only if the right bytes went out and came back."""

    def __init__(self, lmcache, monkeypatch, *, geometry, dcp, pd):
        self.vbs = BLOCK * dcp
        lmcache.chunk = self.vbs
        config = _config(
            layout=None,
            hf_config=_GLM52 if geometry == "mla" else _QWEN3,
            dcp=dcp,
            step=4 * self.vbs,
        )
        self.shell = LMCacheOffloadConnectorScheduler(config)
        self.connector = _multi(_Producer(), self.shell) if pd else self.shell
        self.engine = _engine_with(
            self.connector,
            10,
            step=4 * self.vbs,
            decode_context_parallel_size=dcp,
        )
        self.lmcache = lmcache
        # The step's producer fence, which the worker waits on before a store.
        monkeypatch.setattr(
            dense_connector,
            "record_save_fence",
            lambda: SimpleNamespace(synchronize=lambda: None),
        )
        self.worker = LMCacheOffloadConnector(config)._impl
        codec = DenseKVByteCodec(_dense_kv(geometry, 10), num_blocks=10)
        self.worker._codec = codec
        self.worker._engine = _tier_engine(lmcache, codec, BLOCK, self.vbs)
        self.worker.chunk_size = lmcache.chunk
        self.worker._rank = 0
        self.planes = self.worker._engine.gpu_connector._planes
        self.loads = []

    def _content(self, seq, j, width):
        tokens = tuple(seq.token_ids[: (j + 1) * self.vbs])
        return ((torch.arange(width) + hash(tokens)) % 256).to(torch.uint8)

    def holds(self, seq, n_blocks):
        return all(
            torch.equal(
                plane[seq.block_table[j]], self._content(seq, j, plane.shape[1])
            )
            for j in range(n_blocks)
            for plane in self.planes
        )

    def step(self):
        result = self.engine.schedule()
        if result is None or result[0] is None:
            idle = self.connector.has_pending_work()
            meta = self.connector.build_connector_meta() if idle else None
        else:
            batch, seqs = result
            for seq, n in zip(seqs.values(), batch.num_scheduled_tokens, strict=True):
                lo = seq.num_cached_tokens
                for j in range(lo // self.vbs, (lo + n) // self.vbs):
                    for plane in self.planes:
                        plane[seq.block_table[j]] = self._content(
                            seq, j, plane.shape[1]
                        )
            self.engine.postprocess(
                list(seqs.values()),
                ScheduledBatchOutput(
                    req_ids=list(seqs),
                    token_ids=[(7,)] * len(seqs),
                    num_rejected=None,
                    num_bonus=None,
                    draft_token_ids=None,
                ),
                batch=batch,
            )
            meta = batch.connector_meta_output
        for sub in getattr(meta, "metas", None) or ([meta] if meta else []):
            if isinstance(sub, LMCacheOffloadMetadata) and sub.requests:
                self.loads += [r for r in sub.requests if r.load_spec is not None]
                self.worker.start_load_kv(sub)
                self.worker.wait_for_requests([r.req_id for r in sub.requests])
        self.engine._update_from_kv_xfer_finished(self.worker.get_finished())

    def serve(self, seq):
        self.engine.add(seq)
        for _ in range(100):
            if seq.status == SequenceStatus.FINISHED:
                return
            self.step()
        raise AssertionError("the request never finished")

    def settle(self):
        for _ in range(20):
            if not self.connector.has_pending_work():
                return
            self.step()
        raise AssertionError("demotions did not settle")


@pytest.mark.parametrize("pd", [False, True], ids=["alone", "multi"])
@pytest.mark.parametrize(
    "geometry,dcp", [("mla", 1), ("mla", 2), ("mha", 1)], ids=["mla", "mla-dcp2", "mha"]
)
def test_a_dense_model_reloads_the_kv_it_demoted(
    lmcache, monkeypatch, geometry, dcp, pd
):
    model = _DenseModel(lmcache, monkeypatch, geometry=geometry, dcp=dcp, pd=pd)
    params = {"do_remote_decode": True} if pd else None
    first = Sequence(
        list(range(6 * model.vbs)),
        BLOCK,
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params=params,
    )
    model.serve(first)
    if pd:  # the decode instance has read the blocks
        model.engine._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_sending={first.id})
        )
    model.settle()
    assert model.worker._engine.stored and not model.loads
    # A filler evicts the demoted blocks; the prompt comes back for them.
    model.serve(
        Sequence(
            list(range(1000, 1000 + 6 * model.vbs)),
            BLOCK,
            sampling_params=SamplingParams(max_tokens=1),
        )
    )
    model.settle()
    again = Sequence(
        list(first.token_ids[: first.num_prompt_tokens]),
        BLOCK,
        sampling_params=SamplingParams(max_tokens=2),
    )
    model.engine.add(again)
    for _ in range(100):
        if again.status == SequenceStatus.RUNNING and again.num_cached_tokens >= len(
            first.token_ids[: first.num_prompt_tokens]
        ):
            break
        model.step()
    (load,) = model.loads
    hbm, end = load.load_spec.hbm_cached_tokens, load.load_spec.lmcache_cached_tokens
    assert hbm < end and end % model.vbs == 0
    # Every prompt block: the HBM hit, what the tier gave back, what it ran.
    assert model.holds(again, first.num_prompt_tokens // model.vbs)


@pytest.mark.parametrize("dcp", [1, 2])
def test_the_blocks_a_load_filled_are_keyed_saved_under_dcp(lmcache, monkeypatch, dcp):
    # `load_finished` turns the load's token range into block-table indices, a
    # chunk (KV block x DCP) each: under DCP a KV-block divisor marks the wrong
    # blocks. Run to the reloaded request's finish, where the keys are written.
    model = _DenseModel(lmcache, monkeypatch, geometry="mla", dcp=dcp, pd=False)
    first = Sequence(
        list(range(6 * model.vbs)), BLOCK, sampling_params=SamplingParams(max_tokens=1)
    )
    model.serve(first)
    model.settle()
    model.serve(
        Sequence(
            list(range(1000, 1000 + 6 * model.vbs)),
            BLOCK,
            sampling_params=SamplingParams(max_tokens=1),
        )
    )
    model.settle()
    again = Sequence(
        list(first.token_ids[: first.num_prompt_tokens]),
        BLOCK,
        sampling_params=SamplingParams(max_tokens=2),
    )
    model.engine.add(again)
    table = None
    for _ in range(100):
        if again.status == SequenceStatus.FINISHED:
            break
        if again.num_cached_tokens >= again.num_prompt_tokens and again.block_table:
            table = list(again.block_table)
        model.step()
    (load,) = model.loads
    lo = load.load_spec.hbm_cached_tokens // model.vbs
    hi = load.load_spec.lmcache_cached_tokens // model.vbs
    assert lo < hi
    keys = model.shell._impl._block_keys
    saved = [keys[b][2] for b in table[: again.num_prompt_tokens // model.vbs]]
    # Exactly the blocks the tier gave back (the HBM hit keeps what it had).
    assert saved[lo:hi] == [True] * (hi - lo)
    assert not any(saved[hi:])
