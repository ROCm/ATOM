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
import time
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
    connector_metadata_has_work,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload import lazy_save, prehashed
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import pp_aware_rank_and_world
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
from atom.kv_transfer.offload.tier_index import TierIndex
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams

BLOCK = 4  # tokens per KV block, and per LMCache chunk
STEP = 8  # max_num_batched_tokens
BATCH = STEP // BLOCK  # blocks per demotion: what one step's tokens fill
RESERVE = 2 * BATCH  # free blocks a demotion leaves
PP = 4  # pipeline stages of the PP tests, as the M3.1 prefill node runs
# Before the `lmcache` fixture stands it in.
_REPLICA_WORLD_SIZE = offcfg.lmcache_replica_world_size


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
    multi.is_offload = any(getattr(sub, "is_offload", False) for sub in subs)
    multi.cancels_undispatched_loads = multi.enable_chunked_transfer = False
    multi._load_winner = {}
    return multi


def _config(*, step=STEP, layout="m3", hf_config=None, dcp=1, pp=1, **kvc):
    """An offload config: M3 by override, or the layout `hf_config`'s model
    resolves to when `layout` is None; `pp` pipeline stages of one TP rank."""
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
        pipeline_parallel_size=pp,
        offload_min_load_tokens=BLOCK,
        max_num_batched_tokens=step,
    )


@pytest.fixture
def lmcache(monkeypatch):
    """LMCache's configuration and in-process lookup client, stood in for."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # This stack ships its saves pre-hashed behind a flag, and lazy needs them;
    # it reads the least tokens worth a load from the environment.
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_SAVE", "1")
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", str(BLOCK))
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


def _lazy(lmcache, blocks, pp=1):
    """The lazy M3 scheduler, built through its shell, over a pool of `blocks`."""
    shell = LMCacheOffloadConnectorScheduler(_config(pp=pp))
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


def _cold(lmcache, keyed, *, vacant, blocks, pp=1):
    """A pool of `blocks` whose coldest free blocks with content are a finished
    request's `keyed`, behind `vacant` vacant ones; a running request holds the
    rest. Returns the scheduler, the pool, the tier, the keyed blocks coldest
    first, and the running request."""
    sched, pool, tier = _lazy(lmcache, blocks, pp)
    seq = _seq(keyed)
    cold = _hold(pool, seq)[::-1]
    filler = _seq(blocks - keyed - vacant, base=10_000)
    _hold(pool, filler)
    _finish(sched, pool, seq)
    return sched, pool, tier, cold, filler


def _one_demotion(lmcache, pp=1):
    """Two keyed blocks a single demotion takes: 300 blocks make a window of 6,
    the two past the four vacant ones that are the demotion's reserve."""
    return _cold(lmcache, 2, vacant=RESERVE, blocks=300, pp=pp)


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


def _cpu_fences(monkeypatch):
    """The dense worker's step fence, a CUDA event it records on the current
    stream, on the CPU: returns the fences recorded, each counting its waits."""
    fences = []

    class _Fence:
        recorded = False
        waited = 0

        def __init__(self):
            fences.append(self)

        def record(self, _stream=None):
            self.recorded = True

        def synchronize(self):
            self.waited += 1

    monkeypatch.setattr(dense_connector.torch.cuda, "Event", _Fence)
    monkeypatch.setattr(dense_connector.torch.cuda, "current_stream", lambda: None)
    return fences


@pytest.fixture
def two_slots(monkeypatch):
    """Two demotions in flight: the slot accounting, smaller to follow."""
    monkeypatch.setattr(lazy_save, "_MAX_DEMOTIONS_IN_FLIGHT", 2)


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


def test_an_unsaved_block_evicted_with_nothing_clean_left_is_counted(lmcache):
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
    seq = _seq(3, id=4242)
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
    assert first.prehashed_save == PrehashedSave(
        0, tuple(chain[:BATCH]), must_store_all=True
    )
    assert second.prehashed_save == PrehashedSave(
        0, tuple(chain[BATCH:]), must_store_all=True
    )
    assert first.save_spec.skip_leading_tokens == 0 and first.token_ids == []
    assert first.load_spec is None
    # Each demotion a request of its own: a PP head retires one by request id.
    for req in (first, second):
        op = req.save_operation
        assert req.req_id == op.req_id == f"lmcache-lazy-demotion-{op.generation}"
    assert first.req_id != second.req_id
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


def test_two_demotions_in_flight_and_one_more_per_freed_slot(lmcache, two_slots):
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


@pytest.mark.parametrize("pp", [1, 4])
def test_demotion_slots_scale_with_the_pipeline_stages(lmcache, pp):
    # A demotion reports once every stage stored it, a round trip that grows
    # with the stages: the slots grow with them.
    sched, _, _ = _lazy(lmcache, 600, pp=pp)
    assert sched._max_demotions == lazy_save._MAX_DEMOTIONS_IN_FLIGHT * pp


def test_a_slot_back_is_used_at_once_though_its_blocks_stay_held(lmcache, two_slots):
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


def test_a_spent_budget_is_looked_at_again_though_vacant_blocks_came_back(
    lmcache, two_slots
):
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
    lmcache, two_slots
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
    lmcache, deferred, two_slots
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
    assert len(pool._cached[0]) == 0 and len(pool._protected[0]) == 20
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

    monkeypatch.setattr(dense_connector.torch.cuda, "Event", no_fence)
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._init_worker_common(
        SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
    )
    worker._early_release = True  # as the M3 worker runs
    worker.start_load_kv(meta)
    worker.close()
    sched.process_completions(worker.get_finished())
    assert _saved(sched, cold) == [False, False] and pool.num_free == 6
    assert not sched._demotions and sched.total_lazy_demotions_failed == 1


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
    assert sched._report_abandoned is None  # no PP head to tell
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
    from test_offload_prehashed_save import _engine
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
    fences = _cpu_fences(monkeypatch)
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
    assert len(fences) == 1 and fences[0].waited == 1 and fences[0].recorded
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
        (
            lambda s, mp: mp.delenv("ATOM_OFFLOAD_PREHASHED_SAVE"),
            "ATOM_OFFLOAD_PREHASHED_SAVE=1",
        ),
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
    # A request that does not fit now probes no hit: it asks from the start.
    hog = _seq(14, base=5000, sampling_params=SamplingParams(max_tokens=50))
    engine.add(hog)
    while hog.num_cached_tokens < hog.num_prompt_tokens:
        _step(engine)
    late = _seq(4, base=9000, sampling_params=SamplingParams(max_tokens=1))
    late.offload_hbm_probe_tokens = 3 * BLOCK
    engine.add(late)
    _step(engine)
    assert late.offload_hbm_probe_tokens == 0 and late in engine.waiting
    engine.waiting.remove(late)
    while hog.status != SequenceStatus.FINISHED:
        _step(engine)
    # Probing for the flag detaches nothing while detaching is off, an HBM hit
    # and the connector's consent notwithstanding.
    detached = []
    monkeypatch.setattr(shell, "can_detach_lookup", lambda seq, n: True)
    monkeypatch.setattr(
        shell, "detach_lookup", lambda seq, n: detached.append(seq) or True
    )
    for _ in range(2):
        _serve(engine, _seq(3, base=7000, sampling_params=SamplingParams(max_tokens=1)))
    assert detached == []
    # Without the flag nothing is probed before the lookup.
    monkeypatch.setattr(type(shell), "needs_hbm_probe_before_lookup", False)
    unprobed = _seq(3, sampling_params=SamplingParams(max_tokens=1))
    unprobed.offload_hbm_probe_tokens = -1
    _serve(engine, unprobed)
    assert unprobed.offload_hbm_probe_tokens == -1


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
        token_database=SimpleNamespace(_make_key_by_hash=lambda h, c=None: (h, c)),
        storage_manager=SimpleNamespace(
            allocate=allocate,
            batched_put=batched_put,
            contains=lambda key, *_a, **_k: "cpu" if key[0] in stored else None,
        ),
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
        _cpu_fences(monkeypatch)
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


# ------------------------------------------------------- pipeline parallelism
class _StageReports:
    """What the PP head polls: stage 0's TP-aggregated report through its
    runner manager, the downstream stages' through its PP transport. `post`
    queues one stage's report for the next poll."""

    def __init__(self):
        self.queued: list[tuple[int, KVConnectorOutput]] = []

    def post(self, stage, output):
        self.queued.append((stage, output))

    def call_func_with_aggregation(self, name):
        assert name == "async_proc_aggregation"
        local = [out for stage, out in self.queued if stage == 0]
        return local[0] if local else KVConnectorOutput()

    def recv_kv_status(self, timeout_ms=0):
        downstream = [(stage, out) for stage, out in self.queued if stage != 0]
        self.queued = []
        return downstream


def _pp_head(monkeypatch, connector, scheduler=None):
    """The PP head's engine core over `connector`, built through its own
    `__init__` with `EngineCore`'s stood in for. Its scheduler is `scheduler`,
    or one that hands what the head passes on to `connector`. Returns the head
    and the stage reports it polls."""
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine import pp_engine_core

    if scheduler is None:
        scheduler = SimpleNamespace(
            kv_connector=connector,
            block_manager=SimpleNamespace(enable_prefix_caching=True),
            _update_from_kv_xfer_finished=connector.process_completions,
        )

    def core_init(self, _config, *_args):
        self.scheduler, self.kv_transfer_enabled, self.label = scheduler, True, "head"

    reports = _StageReports()
    monkeypatch.setattr(pp_engine_core.EngineCore, "__init__", core_init)
    monkeypatch.setattr(pp_engine_core, "PPStageTransport", lambda *a, **k: reports)
    config = SimpleNamespace(
        pipeline_parallel_size=PP,
        parallel_config=SimpleNamespace(
            pipeline_parallel_rank=0, pp_meta_addrs=[], pp_token_addr=""
        ),
    )
    head = pp_engine_core.PPEngineCoreProc(config, None, None)
    head.runner_mgr = reports
    return head, reports


def _stored(reqs, ok=True):
    """One stage's report of storing its layers of `reqs`."""
    operations = {req.save_operation for req in reqs}
    return KVConnectorOutput(
        finished_saving=set(operations),
        connector_completions={
            ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, op, ok) for op in operations
        },
    )


def _poll(head, reports, stage, output):
    reports.post(stage, output)
    head._poll_kv_transfer_progress()


def test_under_pp_the_lazy_classes_are_built(lmcache):
    assert _shells(_config(pp=PP)) == (LazyOffloadScheduler, M3OffloadConnector)


def test_under_pp_a_demotion_stays_pinned_until_every_stage_stored(
    lmcache, monkeypatch
):
    sched, pool, _, cold, _ = _one_demotion(lmcache, pp=PP)
    head, reports = _pp_head(monkeypatch, sched)
    meta = sched.build_connector_meta()
    (req,) = meta.requests
    # A demotion alone is work the head ships to every stage.
    assert connector_metadata_has_work(meta)
    for stage in (3, 0, 2):
        _poll(head, reports, stage, _stored([req]))
        assert req.save_operation in sched._demotions
        assert all(pool.is_used(b) for b in cold)
        assert _saved(sched, cold) == [False, False]
    assert head._pp_kv_aggregator.has_pending()
    _poll(head, reports, 1, _stored([req]))
    assert _saved(sched, cold) == [True, True] and not sched._demotions
    assert pool.num_free == 6 and not head._pp_kv_aggregator.has_pending()
    assert sched.get_statistics()["saves_pending"] == 0


@pytest.mark.parametrize("failing", [0, 3])
def test_under_pp_one_stages_failure_fails_the_demotion_once_all_reported(
    lmcache, monkeypatch, clock, failing
):
    sched, pool, _, cold, _ = _one_demotion(lmcache, pp=PP)
    head, reports = _pp_head(monkeypatch, sched)
    (req,) = sched.build_connector_meta().requests
    order = [failing] + [stage for stage in range(PP) if stage != failing]
    for stage in order[:-1]:
        _poll(head, reports, stage, _stored([req], ok=stage != failing))
        # The other stages may still be storing into the pinned blocks.
        assert all(pool.is_used(b) for b in cold)
        assert req.save_operation in sched._demotions
    _poll(head, reports, order[-1], _stored([req]))
    assert _saved(sched, cold) == [False, False] and pool.num_free == 6
    assert not sched._demotions and sched.total_lazy_demotions_failed == 1
    assert sched.build_connector_meta().requests == []  # backing off
    clock[0] += lazy_save._DEMOTION_BACKOFF_S
    (retry,) = sched.build_connector_meta().requests
    assert retry.block_ids == req.block_ids


def test_under_pp_an_abandoned_demotion_is_retired_alone(lmcache, clock, monkeypatch):
    sched, pool, _, cold, _ = _one_demotion(lmcache, pp=PP)
    monkeypatch.setattr(sched, "save_abandon_timeout_s", lambda: 30.0)
    head, reports = _pp_head(monkeypatch, sched)
    assert sched._report_abandoned == head._forget_pp_save_quorum
    (stalled,) = sched.build_connector_meta().requests
    for stage in range(PP - 1):
        _poll(head, reports, stage, _stored([stalled]))
    assert head._pp_kv_aggregator.has_pending()
    clock[0] += 30.0
    (again,) = sched.build_connector_meta().requests
    assert again.block_ids == stalled.block_ids
    assert again.req_id != stalled.req_id
    # The head dropped the stalled demotion's partial tally: nothing pending.
    assert not head._pp_kv_aggregator.has_pending()
    # Its last stage's late report completes nothing.
    _poll(head, reports, PP - 1, _stored([stalled]))
    assert again.save_operation in sched._demotions
    assert all(pool.is_used(b) for b in cold)
    assert _saved(sched, cold) == [False, False]
    # The next demotion still reports, once every stage has.
    for stage in range(PP):
        _poll(head, reports, stage, _stored([again]))
    assert _saved(sched, cold) == [True, True] and not sched._demotions
    assert not head._pp_kv_aggregator.has_pending()


def test_the_shells_hand_the_pp_head_their_lazy_tier(lmcache):
    shell = LMCacheOffloadConnectorScheduler(_config(pp=PP))
    multi = _multi(_Producer(), shell)

    def report(_req_id):
        pass

    multi.bind_save_abandoned(report)
    assert shell._impl._report_abandoned is report


def test_under_pp_a_load_completes_once_every_stage_loaded(lmcache, monkeypatch):
    sched, pool, _ = _lazy(lmcache, 12, pp=PP)
    head, reports = _pp_head(monkeypatch, sched)
    for ok in (True, False):
        seq = _seq(4, base=100 if ok else 200)
        blocks = _hold(pool, seq)
        operation = LoadOperationId(seq.id, 0)
        sched._active_load_operations[str(seq.id)] = (seq, operation)
        seq.offload_load_start_tokens = BLOCK
        seq.offload_loaded_tokens = 3 * BLOCK
        for stage in range(PP):
            if stage == PP - 1:
                assert str(seq.id) in sched._active_load_operations
                assert sched._loaded_chunks == {}
            failed = not ok and stage == 1
            _poll(
                head,
                reports,
                stage,
                KVConnectorOutput(
                    finished_loading=set() if failed else {operation},
                    failed_loading={operation} if failed else set(),
                ),
            )
        assert str(seq.id) not in sched._active_load_operations
        _finish(sched, pool, seq)
        saved = _saved(sched, blocks)
        assert saved == ([False, True, True, False] if ok else [False] * 4)
    assert not head._pp_kv_aggregator.has_pending()


class _StageTiers:
    """The lookup servers of a PP x TP replica: one answer per worker, each
    the contiguous run its own stage's tier holds."""

    def __init__(self, stored, chunk=BLOCK):
        self.stored, self.chunk = stored, chunk
        self.lookups: list[list[int]] = []

    def send_and_recv_all(self, message):
        self.lookups.append(list(message[0]))
        answers = []
        for held in self.stored:
            run = 0
            for chunk_hash in message[0]:
                if chunk_hash not in held:
                    break
                run += 1
            answers.append((run * self.chunk).to_bytes(4, "big"))
        return answers


def test_under_pp_a_lookup_counts_the_shortest_run_any_stage_holds(
    lmcache, monkeypatch
):
    worlds = []
    monkeypatch.setattr(offcfg, "lmcache_replica_world_size", _REPLICA_WORLD_SIZE)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda _c, _cfg, world, rank: worlds.append((world, rank))
        or SimpleNamespace(engine_id="m3"),
    )
    config = _config(pp=PP)
    config.tensor_parallel_size = 2
    sched = LMCacheOffloadConnectorScheduler(config)._impl
    # The lookup client asks every worker of the PP x TP replica.
    assert worlds == [(PP * 2, 0)]
    seq = _seq(6)
    chain = _chain(seq)
    seq.offload_hbm_probe_tokens = BLOCK
    # Past the HBM hit of one block, stages hold runs of 4, 2, 3 and 4.
    runs = [5, 3, 4, 5]
    sched._lookup_client.transport = _StageTiers(
        [set(chain[1:end]) for end in runs for _tp in range(2)]
    )
    assert sched.get_num_new_matched_tokens(seq) == (3 * BLOCK, True)
    assert sched._load_specs[str(seq.id)].lmcache_cached_tokens == 3 * BLOCK


def test_under_pp_each_stage_stores_a_demotion_under_its_own_worker_id(
    lmcache, monkeypatch
):
    pytest.importorskip("lmcache")
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from test_offload_prehashed_save import _engine

    tp = 2
    config = _config(pp=PP)
    config.tensor_parallel_size = tp
    world = _REPLICA_WORLD_SIZE(config)
    assert world == PP * tp
    sched, pool, _ = _lazy(lmcache, 300, pp=PP)
    _hold(pool, _seq(300 - 2 - RESERVE, base=10_000))
    seq = _seq(2)
    _hold(pool, seq)
    _finish(sched, pool, seq)
    meta = sched.build_connector_meta()
    (req,) = meta.requests
    _cpu_fences(monkeypatch)
    stored = {}
    for pp_rank in range(PP):
        for tp_rank in range(tp):
            config.parallel_config = SimpleNamespace(pipeline_parallel_rank=pp_rank)
            rank, size = pp_aware_rank_and_world(
                config, SimpleNamespace(world_size=tp, rank_in_group=tp_rank)
            )
            assert (rank, size) == (pp_rank * tp + tp_rank, world)
            engine = _engine()
            engine.token_database = ChunkedTokenDatabase(
                SimpleNamespace(
                    chunk_size=BLOCK,
                    save_unfull_chunk=False,
                    pre_caching_hash_algorithm="builtin",
                    remote_url=None,
                    enable_pd=False,
                    get_extra_config_value=lambda _name, default=None: default,
                ),
                SimpleNamespace(
                    model_name="m3",
                    world_size=size,
                    worker_id=rank,
                    kv_dtype=torch.uint8,
                    use_mla=False,
                    is_first_rank=lambda: False,
                ),
            )
            worker = M3OffloadConnector.__new__(M3OffloadConnector)
            worker._init_worker_common(
                SimpleNamespace(kv_transfer_config={"kv_role": "offload"})
            )
            worker.chunk_size, worker._engine, worker._rank = BLOCK, engine, rank
            worker._early_release = True
            worker.start_load_kv(meta)
            worker.close()
            keys, _objs = engine.storage_manager.put[-1]
            stored[rank] = keys
    # The same chunks on every worker, each under its own worker id.
    assert sorted(stored) == list(range(world))
    for rank, keys in stored.items():
        assert [key.worker_id for key in keys] == [rank] * 2
        assert [key.chunk_hash for key in keys] == list(req.prehashed_save.hashes)
    assert len({key for keys in stored.values() for key in keys}) == 2 * world


def test_under_pp_the_stages_split_pp_times_the_host_budget_by_layers(lmcache):
    from atom.models.utils import get_pp_indices

    # 62 layers over 4 stages split unevenly; each stage's worker sizes its
    # own engine.
    budgets, layers = [], []
    for pp_rank in range(PP):
        config = _config(pp=PP)
        config.hf_config = SimpleNamespace(num_hidden_layers=62)
        config.parallel_config = SimpleNamespace(pipeline_parallel_rank=pp_rank)
        config.speculative_config = None
        assert _REPLICA_WORLD_SIZE(config) == PP
        cfg = SimpleNamespace(max_local_cpu_size=100.0)
        offcfg.scale_cpu_size_for_pp(cfg, config)
        budgets.append(cfg.max_local_cpu_size)
        start, end = get_pp_indices(62, pp_rank, PP)
        layers.append(end - start)
    assert len(set(layers)) > 1
    assert sum(budgets) == pytest.approx(PP * 100.0)
    assert budgets == pytest.approx([PP * 100.0 * n / 62 for n in layers])


# ------------------------------------------------------ write-back eviction
def test_an_allocation_evicts_clean_blocks_before_keyed_unsaved_ones(lmcache):
    """Keyed unsaved blocks are the pool's dirty ones: an allocation passes over
    them to a clean block (here a keyless one) and a demotion takes the blocks
    it passed over first, though they are outside the window by then."""
    sched, pool, _ = _lazy(lmcache, 300)  # a window of 6
    _hold(pool, _seq(300 - 2 - 6, base=10_000))
    dirty, clean = _seq(2), _seq(6, base=500)
    _hold(pool, dirty)
    _finish(sched, pool, dirty)  # keyed, unsaved, coldest
    clean_blocks = _hold(pool, clean)
    for block_id in clean_blocks:  # published, never keyed
        pool.free(block_id)
    dirty_blocks = sorted(sched._block_keys)  # `_finish` emptied its table
    taken = [pool.pop(), pool.pop()]
    assert taken == clean_blocks[:2]
    assert sorted(pool.held_blocks()) == dirty_blocks
    assert sched.get_statistics()["lazy_held_blocks"] == 2
    (req,) = sched.build_connector_meta().requests
    assert sorted(req.block_ids) == dirty_blocks
    assert pool.num_held == 0 and pool.blocks_evicted_dirty == 0


def test_a_refused_demotion_leaves_its_blocks_dirty_until_retried(lmcache, clock):
    sched, _pool, tier, cold, _ = _one_demotion(lmcache)
    (req,) = sched.build_connector_meta().requests
    _report(sched, [req], tier, ok=False)  # `must_store_all`: the tier refused
    assert _saved(sched, cold) == [False, False]
    assert sched.get_statistics()["lazy_demotions_failed"] == 1
    # Dirty still: an allocation passes over them while a clean block is left.
    assert all(sched._is_dirty(b) for b in cold)
    clock[0] += lazy_save._DEMOTION_BACKOFF_S
    (retry,) = sched.build_connector_meta().requests
    assert retry.block_ids == req.block_ids


# --------------------------------------------- lookup pins of queued requests
def test_the_pin_a_waiting_request_holds_is_counted_until_its_load(lmcache):
    """A tier hit for a request the pool cannot admit yet keeps its pin while
    it waits (`lookup_pinned_tokens`); the load spends it."""
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    engine = _engine_with(shell, 10, step=4 * BLOCK, max_model_len=10**6)
    tier = lmcache.tier
    hog = _seq(8, base=5000, sampling_params=SamplingParams(max_tokens=6))
    engine.add(hog)
    while hog.num_cached_tokens < hog.num_prompt_tokens:
        _step(engine)
    prompt = _seq(6, sampling_params=SamplingParams(max_tokens=1))
    tier.stored.update(_chain(prompt))
    engine.add(prompt)
    meta = _step(engine)
    assert prompt in engine.waiting
    assert str(prompt.id) not in meta.lookup_requests_in_step  # still pinned
    # The tier holds the whole prompt, and pins all of it.
    assert shell.get_statistics()["lookup_pinned_tokens"] == 6 * BLOCK
    while prompt.status != SequenceStatus.FINISHED:
        meta = _step(engine)
        for req in meta.requests if meta else ():
            if req.load_spec is not None:
                engine._update_from_kv_xfer_finished(
                    KVConnectorOutput(finished_loading={req.load_operation})
                )
    assert shell.get_statistics()["lookup_pinned_tokens"] == 0


# ------------------------------------------------ a trace replayed end to end
class _LRUChunks:
    """A tier's chunk set of `cap` chunks: a store or a lookup hit makes a
    chunk the most recent, and a store past `cap` evicts the least recent."""

    def __init__(self, cap):
        self.cap = cap
        self.chunks = {}

    def __contains__(self, chunk_hash):
        if chunk_hash not in self.chunks:
            return False
        self.chunks[chunk_hash] = self.chunks.pop(chunk_hash)
        return True

    def __len__(self):
        return len(self.chunks)

    def update(self, hashes):
        for chunk_hash in hashes:
            self.chunks.pop(chunk_hash, None)
            self.chunks[chunk_hash] = None
            while len(self.chunks) > self.cap:
                del self.chunks[next(iter(self.chunks))]

    def cool(self, chunk_hash):
        """First in line for eviction (the worker's `_make_coldest`)."""
        if chunk_hash in self.chunks:
            del self.chunks[chunk_hash]
            self.chunks = {chunk_hash: None, **self.chunks}


def _replay(lmcache, *, write_back, pd, tier_cap=None):
    """Multi-turn conversations through the real scheduler, P/D or alone, with
    the lazy tier: each turn re-sends the conversation so far plus new tokens,
    the worker reports a demotion two steps after it is dispatched and a load
    one step after, and puts the chunks a build names cold first in line. The
    tier never evicts, or holds `tier_cap` chunks, least recently stored or hit
    first (a demotion of a chunk it holds makes it recent, as the worker does).
    Returns, over the follow-up turns, the blocks of the previous turn neither
    HBM nor the tier served, and the pool."""
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    producer = _Producer()
    connector = _multi(producer, shell) if pd else shell
    engine = _engine_with(
        connector, 48, step=4 * BLOCK, max_num_seqs=8, max_model_len=10**6
    )
    pool = engine.block_manager.kv
    if not write_back:
        pool.set_write_back(None)  # as before the fix
    tier = lmcache.tier
    if tier_cap is not None:
        tier.stored = _LRUChunks(tier_cap)
    # (conversation, blocks) in arrival order: five conversations, round robin,
    # each turn a few blocks longer, so turns of 6..22 blocks arrive while the
    # pool (48) holds about two conversations.
    turns = [(c, 6 + 4 * t + c % 3) for t in range(5) for c in range(5)]
    reports, sends, lost, expected = [], [], 0, 0
    previous = {}
    tick = 0
    for c, n in turns:
        seq = Sequence(
            [c * 100_000 + i for i in range(n * BLOCK)],
            BLOCK,
            sampling_params=SamplingParams(max_tokens=1),
            kv_transfer_params={"do_remote_decode": True} if pd else None,
        )
        engine.add(seq)
        first_cached = None
        while seq.status != SequenceStatus.FINISHED or reports or sends:
            tick += 1
            assert tick < 5000, "the replay did not settle"
            out = KVConnectorOutput()
            for item in [x for x in reports if x[0] <= tick]:
                reports.remove(item)
                req = item[1]
                if req.load_spec is not None:
                    out.finished_loading.add(req.load_operation)
                    continue
                tier.stored.update(req.prehashed_save.hashes)
                out.finished_saving.add(req.save_operation)
                out.connector_completions.add(
                    ConnectorCompletion(
                        DENSE_PAGE_STORE_CHANNEL, req.save_operation, True
                    )
                )
            for item in [x for x in sends if x[0] <= tick]:
                sends.remove(item)
                producer.send_finished(item[1])
                out.finished_sending.add(item[1])
            engine._update_from_kv_xfer_finished(out)
            result = engine.schedule()
            if result is None or result[0] is None:
                meta = (
                    connector.build_connector_meta()
                    if connector.has_pending_work()
                    else None
                )
            else:
                batch, seqs = result
                if first_cached is None and seq.id in seqs:
                    # Where its first chunk starts: what HBM and a load served.
                    first_cached = seq.num_cached_tokens // BLOCK
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
                meta = batch.connector_meta_output
            if (
                pd
                and seq.id in producer.sending
                and not any(s[1] == seq.id for s in sends)
            ):
                sends.append((tick + 2, seq.id))
            for sub in getattr(meta, "metas", None) or ([meta] if meta else []):
                for chunk_hash in getattr(sub, "tier_cold_chunks", ()):
                    if isinstance(tier.stored, _LRUChunks):
                        tier.stored.cool(chunk_hash)
                for req in getattr(sub, "requests", ()):
                    if req.prehashed_save is not None:
                        reports.append((tick + 2, req))
                    elif req.load_spec is not None:
                        reports.append((tick + 1, req))
        if c in previous:
            # Up to the last full block of the previous turn; the last block
            # of a prompt is always recomputed.
            want = min(previous[c], n - 1)
            expected += want
            lost += max(0, want - (first_cached or 0))
        previous[c] = n
    return lost, expected, pool


@pytest.mark.parametrize("pd", [False, True], ids=["alone", "multi"])
def test_a_replayed_trace_loses_no_prefix_while_the_tier_has_room(lmcache, pd):
    """The prefill node's collapse in miniature: prompts admitted whole evict
    more blocks at once than the demotion window looks at, and demotions report
    two steps late. Without write-back HBM evicts keyed blocks no demotion
    stored, and later turns recompute prefixes an unbounded tier had room for;
    with it every follow-up turn is served its previous turn from HBM or the
    tier."""
    with pytest.MonkeyPatch.context() as patch:
        # As before the fix: two demotions in flight, no write-back.
        patch.setattr(lazy_save, "_MAX_DEMOTIONS_IN_FLIGHT", 2)
        lost, expected, _pool = _replay(lmcache, write_back=False, pd=pd)
    # 118 of 256 blocks, here and on the code before the fix alike.
    assert expected > 0 and lost > expected // 4
    lmcache.tier = None
    lost, expected, pool = _replay(lmcache, write_back=True, pd=pd)
    assert lost == 0 and pool.blocks_evicted_dirty == 0


# ------------------------------------- cached_first ranks by the tier, unasked
def _ranked_engine(lmcache, monkeypatch, order="cached_first", horizon=1000, pd=False):
    """A lazy prefill node whose tier holds `first`'s prompt past the two
    blocks HBM kept, a `cold` short request queued ahead of `first`'s prompt
    sent again (`again`); nothing admitted yet."""
    monkeypatch.setenv("ATOM_PREFILL_ORDER", order)
    monkeypatch.setattr(
        offcfg,
        "lmcache_tier_horizon_tokens",
        lambda _c: horizon and horizon * BLOCK,
        raising=False,
    )
    shell = LMCacheOffloadConnectorScheduler(_config(step=4 * BLOCK))
    engine = _engine_with(
        _multi(_Producer(), shell) if pd else shell, 10, step=4 * BLOCK
    )
    tier = lmcache.tier
    first = _seq(6, sampling_params=SamplingParams(max_tokens=1))
    _serve(engine, first)
    _demote_and_store(engine, tier)
    filler = _seq(6, base=1000, sampling_params=SamplingParams(max_tokens=1))
    _demote_and_store(engine, tier, _serve(engine, filler))
    assert tier.stored >= set(_chain(first)[2:])
    cold = _seq(3, base=3000, sampling_params=SamplingParams(max_tokens=1))
    prompt = list(first.token_ids[: first.num_prompt_tokens])
    again = Sequence(prompt, BLOCK, sampling_params=SamplingParams(max_tokens=1))
    engine.waiting.extend([cold, again])
    now = max(cold.arrive_time, again.arrive_time)
    return engine, shell, cold, again, now


def _waiting(engine, now):
    engine._order_waiting(now)
    return list(engine.waiting)


@pytest.mark.parametrize("pd", [False, True], ids=["alone", "multi"])
@pytest.mark.parametrize("prehashed_load", ["0", "1"])
def test_cached_first_ranks_a_queued_request_by_its_tier_prefix(
    lmcache, monkeypatch, prehashed_load, pd
):
    """HBM holds two of `again`'s six blocks and the tier the other four, but no
    lookup has asked: before, it ranked with 16 tokens left behind `cold`'s 12
    and waited there; it has none left."""
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_LOAD", prehashed_load)
    engine, shell, cold, again, now = _ranked_engine(lmcache, monkeypatch, pd=pd)
    lookups = len(lmcache.tier.lookups)
    assert _waiting(engine, now) == [again, cold]
    # Nothing sent, so nothing pinned: no lookup, nothing to unpin.
    impl = shell._impl
    assert len(lmcache.tier.lookups) == lookups
    assert impl._lookup_results == {} and impl._lookup_in_step == []
    assert shell.get_statistics().get("lookup_pinned_tokens", 0) == 0
    # The estimate is the lookup's own: admission loads those four blocks.
    batch, _ = engine.schedule()
    (load,) = [r for r in batch.connector_meta_output.requests if r.load_spec]
    assert load.req_id == again.id
    assert load.load_spec.lmcache_cached_tokens == 5 * BLOCK


def test_cached_first_keeps_aging_ahead_of_the_tier(lmcache, monkeypatch):
    engine, _shell, cold, again, now = _ranked_engine(lmcache, monkeypatch)
    cold.arrive_time = now - 61.0
    assert _waiting(engine, now) == [cold, again]


@pytest.mark.parametrize(
    "order,horizon", [("fcfs", 1000), ("cached_first", 0)], ids=["fcfs", "no-size"]
)
def test_without_cached_first_or_a_tier_size_nothing_is_indexed(
    lmcache, monkeypatch, order, horizon
):
    engine, shell, cold, again, now = _ranked_engine(
        lmcache, monkeypatch, order=order, horizon=horizon
    )
    assert shell._impl._tier_index is None
    assert shell.tier_probe(again, 2 * BLOCK) == 0
    # The order as before: `again` looks 16 tokens from done, `cold` 12.
    assert _waiting(engine, now) == [cold, again]


def test_a_lookup_that_stops_short_drops_the_chunk_from_the_index(lmcache, monkeypatch):
    _engine, shell, _cold, again, _now = _ranked_engine(lmcache, monkeypatch)
    impl, chain = shell._impl, _chain(again)
    assert shell.tier_probe(again, 2 * BLOCK) == 6 * BLOCK
    # LMCache evicts block 4 unreported; the index still claims it.
    lmcache.tier.stored.discard(chain[4])
    assert chain[4] in impl._tier_index
    impl._tier_lookup(again, str(again.id))  # admission's lookup
    assert chain[4] not in impl._tier_index and chain[3] in impl._tier_index
    clock = [time.monotonic() + 2.0]  # past the probe's reuse
    monkeypatch.setattr(prehashed.time, "monotonic", lambda: clock[0])
    assert shell.tier_probe(again, 2 * BLOCK) == 4 * BLOCK


def test_a_probe_hashes_a_steps_share_and_walks_on_the_next(lmcache, monkeypatch):
    monkeypatch.setattr(prehashed, "_TIER_PROBE_HASH_CHUNKS", 3)
    _engine, shell, _cold, again, _now = _ranked_engine(lmcache, monkeypatch)
    impl = shell._impl
    impl._prehash_chains.pop(str(again.id), None)
    impl._tier_probe_hash_left = 3
    # Three chunks hashed: past HBM's two, the run reaches the third.
    assert shell.tier_probe(again, 2 * BLOCK) == 3 * BLOCK
    assert shell.tier_probe(again, 2 * BLOCK) == 3 * BLOCK  # spent this step
    shell.build_connector_meta()
    assert shell.tier_probe(again, 2 * BLOCK) == 6 * BLOCK


def test_the_index_keeps_the_newest_chunks_up_to_the_tier_size():
    index = TierIndex(3)
    index.touch([1, 2, 3])
    index.touch([1, 4])
    assert list(index._chunks) == [3, 1, 4]
    index.observe_lookup([3, 1, 4], 1)
    # Found: 3, newest now. Missing: 1, dropped.
    assert list(index._chunks) == [4, 3]


def test_eager_without_prehashed_saves_probes_nothing(lmcache, monkeypatch):
    monkeypatch.setenv("ATOM_PREFILL_ORDER", "cached_first")
    monkeypatch.setenv("ATOM_OFFLOAD_PREHASHED_SAVE", "0")
    monkeypatch.setattr(offcfg, "lmcache_tier_horizon_tokens", lambda _c: 10**6)
    shell = LMCacheOffloadConnectorScheduler(_config(save_policy="eager"))
    seq = _seq(3)
    assert isinstance(shell._impl, M3OffloadScheduler)
    assert shell._impl._tier_index is None and shell.tier_probe(seq, 0) == 0


# ------------------------------------------------- exclusive tiers (opt-in)
def _loaded(sched, pool, seq, start, end):
    """`seq` held as a prefill leaves it, with chunks [start, end) filled by a
    tier load that reported."""
    blocks = _hold(pool, seq)
    operation = LoadOperationId(seq.id, 0)
    sched._active_load_operations[str(seq.id)] = (seq, operation)
    seq.offload_load_start_tokens = start * BLOCK
    seq.offload_loaded_tokens = end * BLOCK
    assert sched.load_finished(operation)
    return blocks


def test_exclusive_cools_what_a_load_filled_from_the_tier_once(lmcache, monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_LAZY_EXCLUSIVE", "1")
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(4)
    blocks = _loaded(sched, pool, seq, 1, 3)
    chain = _chain(seq)
    _finish(sched, pool, seq)
    # HBM holds chunks 1 and 2 now: dirty, so HBM demotes them again before
    # it evicts them, and the tier's copies go first in line with the next build.
    assert _saved(sched, blocks) == [False] * 4
    assert [sched._is_dirty(b) for b in blocks] == [True] * 4
    assert sched.has_pending_work()
    meta = sched.build_connector_meta()
    assert meta.tier_cold_chunks == chain[1:3] and meta.has_work()
    assert sched._perf_counters["lazy_tier_cooled"] == 2
    # Once: a request finishing on the same blocks again (it hit HBM) adds none.
    again = _seq(4)
    assert _hold(pool, again) == blocks
    _finish(sched, pool, again)
    assert sched.build_connector_meta().tier_cold_chunks == []
    assert not sched.has_pending_work()


def test_exclusive_leaves_a_duplicate_blocks_tier_copy_in_place(lmcache, monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_LAZY_EXCLUSIVE", "1")
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(2)
    blocks = _loaded(sched, pool, seq, 0, 2)
    # The load filled blocks[1], but the index names another copy of its
    # content: the tier's copy keeps its place.
    canonical = pool.pop()
    pool.allocate(canonical)
    original = pool.block(blocks[1])
    pool.block(canonical).update(original.hash, original.token_ids)
    pool._hash_to_block_id[original.hash] = canonical
    _finish(sched, pool, seq)
    assert sched.build_connector_meta().tier_cold_chunks == _chain(seq)[:1]
    assert _saved(sched, blocks) == [False, False]


@pytest.mark.parametrize("exclusive", ["0", "1"])
def test_exclusive_doubles_the_demotions_in_flight(lmcache, monkeypatch, exclusive):
    # Every block a load brought back is dirty too: twice the slots.
    monkeypatch.setenv("ATOM_OFFLOAD_LAZY_EXCLUSIVE", exclusive)
    # 1000 blocks make a window of 20, past the four vacant ones.
    sched, *_ = _cold(lmcache, 40, vacant=RESERVE, blocks=1000)
    slots = lazy_save._MAX_DEMOTIONS_IN_FLIGHT * (2 if exclusive == "1" else 1)
    assert len(sched.build_connector_meta().requests) == slots
    assert len(sched._demotions) == slots


def test_without_exclusive_a_load_ships_no_cold_chunks(lmcache):
    sched, pool, _ = _lazy(lmcache, 12)
    seq = _seq(3)
    blocks = _loaded(sched, pool, seq, 0, 3)
    _finish(sched, pool, seq)
    assert _saved(sched, blocks) == [True] * 3
    assert sched.build_connector_meta().tier_cold_chunks == []


def _cpu_backend(policy):
    """LMCache's real `LocalCPUBackend` under `policy`, its memory objects
    stood in for."""
    pytest.importorskip("lmcache")
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    from atom.kv_transfer.offload.cache_policy import register_slru_policy

    register_slru_policy()
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=BLOCK,
        local_cpu=True,
        max_local_cpu_size=0.001,
        cache_policy=policy,
    )
    return LocalCPUBackend(cfg, metadata=None, memory_allocator=object())


@pytest.mark.parametrize("policy", ["ATOM_SLRU", "LRU"])
def test_the_worker_cools_named_chunks_before_the_steps_saves(monkeypatch, policy):
    from test_offload_prehashed_save import _tdb as real_token_database

    backend = _cpu_backend(policy)
    tdb = real_token_database()
    engine = SimpleNamespace(
        token_database=tdb,
        storage_manager=SimpleNamespace(storage_backends={"LocalCPUBackend": backend}),
    )
    k0, k1, k2, k3 = keys = [tdb._make_key_by_hash(h) for h in (10, 11, 12, 13)]
    for key in keys:
        backend.hot_cache[key] = SimpleNamespace(can_evict=True)
        backend.cache_policy.update_on_put(key)
    backend.cache_policy.update_on_hit(k0, backend.hot_cache)  # most recent
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._engine = engine
    order = []

    def saves(_self, _meta):
        candidates = backend.cache_policy.get_evict_candidates(backend.hot_cache, 10)
        order.append(candidates)

    monkeypatch.setattr(DenseOffloadConnector, "start_load_kv", saves)
    meta = LMCacheOffloadMetadata()
    meta.tier_cold_chunks = [10, 12, 99]  # 99: not in the tier
    worker.start_load_kv(meta)
    # Before the step's saves: 12, then 10, ahead of the rest; nothing dropped.
    assert order == [[k2, k0, k1, k3]]
    assert len(backend.hot_cache) == 4
    # A demotion finding 10 still held makes it recent again.
    from atom.kv_transfer.offload import prehashed

    worker._tier_backends_do([10], prehashed._make_recent)
    assert backend.cache_policy.get_evict_candidates(backend.hot_cache, 10)[-1] == k0
