# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``umbp_store.save_policy: lazy`` and the UMBP lookup client's prefetch, on CPU.

Lazy is `LazySaveSchedulerMixin` over the store's digest chain: the mixin's
behaviour (what it keys, what it demotes, when, the pins and their reports) is
tested once, through LMCache, in ``test_offload_lazy_save.py``. Here: what is
UMBP's own -- a demotion is an ordinary store save whose chunk i is
``block_ids[i]`` under that chunk's digest, a lookup starts past the HBM hit,
and the lookup thread's prefetched per-chunk answers.

The server is ``test_umbp_store_offload``'s fake; the pool is the real
`BlockPool`; the sequences are real `Sequence`s.
"""

from __future__ import annotations

import array
import time
from types import SimpleNamespace

import pytest
from kv_store_test_support import engine_config, install_fake_aiter
from test_umbp_store_offload import ADDRESS, FakeServer

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
)
from atom.kv_transfer.offload import lazy_save
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.kv_store import keys
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.kv_store.keys import DIGEST_BYTES
from atom.kv_transfer.offload.umbp_store import client as umbp_client
from atom.kv_transfer.offload.umbp_store import scheduler as umbp_scheduler
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.config import (
    check_engine_compatibility,
    parse_umbp_store_config,
    store_namespace,
)
from atom.kv_transfer.offload.umbp_store.scheduler import (
    UMBPLazyStoreScheduler,
    UMBPStoreOffloadScheduler,
    make_umbp_store_scheduler,
)
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.sequence import Sequence

BLOCK = 4  # tokens per KV block, and per store chunk under lazy
STEP = 8  # max_num_batched_tokens: a demotion is two blocks
BATCH = STEP // BLOCK


@pytest.fixture(autouse=True)
def aiter_arch(monkeypatch):
    return install_fake_aiter(monkeypatch)


@pytest.fixture(autouse=True)
def fresh_process_state(monkeypatch):
    monkeypatch.setattr(taint, "_TAINTED", {})
    monkeypatch.setattr(umbp_client, "_RETAINED_CLIENTS", [])
    monkeypatch.setattr(pool_mod, "_UNSETTLED_ALLOCATIONS", [])


@pytest.fixture
def server(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(umbp_client, "_new_umbp_client", server.connect)
    return server


def _extra(**overrides):
    extra = {
        "umbp_store.address": ADDRESS,
        "umbp_store.chunk_tokens": BLOCK,
        "umbp_store.save_policy": "lazy",
        "umbp_store.lookup_threads": 0,
    }
    for name, value in overrides.items():
        if value is None:
            extra.pop(f"umbp_store.{name}", None)
        else:
            extra[f"umbp_store.{name}"] = value
    return extra


def _config(*, pp=1, extra=None, **fields):
    config = engine_config(
        "umbp_store",
        _extra() if extra is None else extra,
        "offload",
        pp=pp,
        **{"max_num_batched_tokens": STEP, **fields},
    )
    return config


def _lazy(blocks=300, *, pp=1, extra=None):
    sched = make_umbp_store_scheduler(_config(pp=pp, extra=extra))
    pool = BlockPool(blocks, cache_policy="slru")
    sched.bind_block_manager(SimpleNamespace(kv=pool, enable_prefix_caching=True))
    return sched, pool


def _seq(n_blocks, base=0):
    return Sequence([base + i for i in range(n_blocks * BLOCK)], BLOCK)


def _digests(seq, chunk=BLOCK):
    hashes = keys.chunk_hash_chain(list(seq.token_ids[: seq.num_prompt_tokens]), chunk)
    return [
        hashes[i * DIGEST_BYTES : (i + 1) * DIGEST_BYTES]
        for i in range(len(hashes) // DIGEST_BYTES)
    ]


def _hold(pool, seq):
    """`seq`'s prompt blocks as a finished prefill leaves them, published."""
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
    sched.request_finished(seq)
    for block_id in reversed(seq.block_table):
        pool.free(block_id)
    del seq.block_table[:]


def _cold(keyed, *, vacant, blocks, pp=1):
    """A pool whose coldest blocks with content are a finished request's
    `keyed`, behind `vacant` vacant ones; a running request holds the rest."""
    sched, pool = _lazy(blocks, pp=pp)
    seq = _seq(keyed)
    cold = _hold(pool, seq)[::-1]
    filler = _seq(blocks - keyed - vacant, base=10_000)
    _hold(pool, filler)
    _finish(sched, pool, seq)
    return sched, pool, seq, cold


def _key(config, digest, *, rank=0, world=1):
    namespace = store_namespace(config, BLOCK)
    return f"{keys.rank_key_prefix(namespace, rank, world)}{digest.hex()}"


# --- settings ------------------------------------------------------------------


def test_policy_settings_default_to_eager_with_one_lookup_thread():
    cfg = parse_umbp_store_config(
        {"kv_connector_extra_config": {"umbp_store.address": ADDRESS}}
    )
    assert (cfg.save_policy, cfg.lookup_threads, cfg.tier_index_chunks) == (
        "eager",
        1,
        0,
    )


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"save_policy": "sometimes"}, "save_policy must be one of"),
        ({"lookup_threads": -1}, "lookup_threads"),
        ({"tier_index_chunks": -5}, "tier_index_chunks"),
    ],
)
def test_bad_policy_settings_are_refused(extra, match):
    with pytest.raises(ValueError, match=match):
        parse_umbp_store_config({"kv_connector_extra_config": _extra(**extra)})


def test_lazy_needs_one_chunk_per_kv_block():
    config = _config(extra=_extra(chunk_tokens=2 * BLOCK))
    cfg = parse_umbp_store_config(config.kv_transfer_config)
    with pytest.raises(ValueError, match="one chunk per KV block"):
        check_engine_compatibility(cfg, config)
    eager = _config(extra=_extra(chunk_tokens=2 * BLOCK, save_policy="eager"))
    check_engine_compatibility(parse_umbp_store_config(eager.kv_transfer_config), eager)


def test_the_policy_picks_the_scheduler(server):
    assert type(make_umbp_store_scheduler(_config())) is UMBPLazyStoreScheduler
    eager = make_umbp_store_scheduler(_config(extra=_extra(save_policy="eager")))
    assert type(eager) is UMBPStoreOffloadScheduler
    assert not eager.needs_hbm_probe_before_lookup


def test_lazy_saves_no_prompt_and_asks_past_the_hbm_hit(server):
    sched, _ = _lazy(12)
    assert sched._do_save is False
    assert sched.needs_hbm_probe_before_lookup
    assert sched._max_demotions == lazy_save._MAX_DEMOTIONS_IN_FLIGHT


def test_lazy_demotion_slots_scale_with_the_pipeline_stages(server):
    sched, _ = _lazy(12, pp=4)
    assert sched._max_demotions == 4 * lazy_save._MAX_DEMOTIONS_IN_FLIGHT


def test_exclusive_tiers_are_refused(server, monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_LAZY_EXCLUSIVE", "1")
    with pytest.raises(ValueError, match="ATOM_OFFLOAD_LAZY_EXCLUSIVE"):
        make_umbp_store_scheduler(_config())


# --- keys and demotions ------------------------------------------------------------


def test_finish_keys_prompt_blocks_with_the_store_digests(server):
    sched, pool = _lazy(12)
    seq = _seq(3)
    blocks = _hold(pool, seq)
    _finish(sched, pool, seq)
    digests = _digests(seq)
    assert [sched._block_keys[b][1] for b in blocks] == digests
    assert [sched._block_keys[b][2] for b in blocks] == [False] * 3


def test_a_demotion_is_a_store_save_of_its_blocks_digests(server):
    # 400 blocks: a window of 8, the 4 vacant blocks first.
    sched, pool, seq, cold = _cold(4, vacant=4, blocks=400)
    first, second = sched.build_connector_meta().requests
    digests = _digests(seq)[::-1]  # coldest first, as `cold`
    assert first.block_ids == cold[:BATCH] and second.block_ids == cold[BATCH:]
    assert first.chunk_hashes == b"".join(digests[:BATCH])
    assert second.chunk_hashes == b"".join(digests[BATCH:])
    for req in (first, second):
        assert req.save_spec.skip_leading_tokens == 0 and req.save_spec.can_save
        assert req.prehashed_save is None and req.token_ids == []
        assert req.dispatched_at is not None and req.dispatch_clock
        op = req.save_operation
        assert req.req_id == op.req_id == f"umbp-lazy-demotion-{op.generation}"
    assert all(pool.is_used(b) for b in cold)


def test_only_the_store_report_ends_a_demotion(server):
    sched, pool, _, cold = _cold(2, vacant=4, blocks=300)
    (req,) = sched.build_connector_meta().requests
    op = req.save_operation
    sched.connector_completion(
        ConnectorCompletion(DENSE_PAGE_SOURCE_SAFE_CHANNEL, op, True)
    )
    assert op in sched._demotions and all(pool.is_used(b) for b in cold)
    sched.process_completions(
        KVConnectorOutput(
            finished_saving={op},
            connector_completions={
                ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, op, True)
            },
        )
    )
    assert [sched._block_keys[b][2] for b in cold] == [True, True]
    assert not sched._demotions and not any(pool.is_used(b) for b in cold)


def test_a_failed_demotion_stays_unsaved_and_backs_off(server):
    sched, _pool, _, cold = _cold(2, vacant=4, blocks=300)
    (req,) = sched.build_connector_meta().requests
    sched.connector_completion(
        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, req.save_operation, False)
    )
    assert [sched._block_keys[b][2] for b in cold] == [False, False]
    assert sched.total_lazy_demotions_failed == 1
    assert sched._backoff_until > time.monotonic()


def test_a_demotion_with_its_tier_index_touches_what_it_stored(server):
    sched, pool = _lazy(300, extra=_extra(tier_index_chunks=64))
    seq = _seq(2)
    cold = _hold(pool, seq)[::-1]
    _hold(pool, _seq(300 - 2 - 4, base=10_000))
    _finish(sched, pool, seq)
    (req,) = sched.build_connector_meta().requests
    sched.connector_completion(
        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, req.save_operation, True)
    )
    assert all(digest in sched._tier_index for digest in _digests(seq))
    assert cold  # pinned and back


# --- lookups -------------------------------------------------------------------------


def _store(server, config, seq, chunks, *, world=1):
    digests = _digests(seq)
    for index in chunks:
        for rank in range(world):
            server.at()[_key(config, digests[index], rank=rank, world=world)] = b""


def test_a_lazy_lookup_asks_only_past_the_hbm_hit(server):
    config = _config()
    sched, _ = _lazy(12)
    seq = _seq(6)
    _store(server, config, seq, range(2, 5))
    seq.offload_hbm_probe_tokens = 2 * BLOCK
    assert sched._tier_lookup(seq, str(seq.id)) == 5 * BLOCK
    [fake] = server.clients
    [asked] = [call[1] for call in fake.calls if call[0] == "exists"]
    assert len(asked) == 4  # chunks 2..5, one rank
    # Nothing past the HBM hit is no hit, whatever the tier holds before it.
    other = _seq(6)
    _store(server, config, other, range(5))
    other.offload_hbm_probe_tokens = 5 * BLOCK
    assert sched._tier_lookup(other, str(other.id)) == 0


def test_an_eager_lookup_counts_from_the_prompts_head(server):
    config = _config(extra=_extra(save_policy="eager"))
    sched = make_umbp_store_scheduler(config)
    seq = _seq(6)
    _store(server, config, seq, range(3))
    assert sched._tier_lookup(seq, str(seq.id)) == 3 * BLOCK


def test_a_lookup_under_pp_needs_every_stage(server):
    config = _config(pp=4)
    sched = make_umbp_store_scheduler(config)
    seq = _seq(6)
    _store(server, config, seq, range(4), world=4)
    # Stage 3 lacks chunk 4: its keys are the fourth rank's.
    server.at().pop(_key(config, _digests(seq)[3], rank=3, world=4))
    seq.offload_hbm_probe_tokens = 1 * BLOCK
    assert sched._tier_lookup(seq, str(seq.id)) == 3 * BLOCK
    assert sched._lookup_client.uneven_lookups == 1


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


def test_a_prefetched_answer_is_read_from_any_start_without_asking_again(server):
    config = _config(extra=_extra(lookup_threads=1))
    sched = make_umbp_store_scheduler(config)
    lookup = sched._lookup_client
    seq = _seq(6)
    _store(server, config, seq, range(2, 5))
    sid = str(seq.id)
    assert lookup.submit(seq, sid)
    assert not lookup.submit(seq, sid)  # one prefetch per request
    _wait(lambda: lookup.poll(sid))
    assert sid in lookup.pending_ids() and lookup.answered(sid)
    [fake] = server.clients
    asked = [call for call in fake.calls if call[0] == "exists"]
    assert len(asked) == 1 and len(asked[0][1]) == 6  # the whole prompt
    # Admission learns the HBM hit only now; the answer covers any start.
    seq.offload_hbm_probe_tokens = 2 * BLOCK
    assert sched._tier_lookup(seq, sid) == 5 * BLOCK
    assert len([call for call in fake.calls if call[0] == "exists"]) == 1
    assert lookup.prefetch_answers_used == 1 and not lookup.answered(sid)


def test_a_stale_prefetched_answer_is_asked_again(server, monkeypatch):
    monkeypatch.setattr(umbp_scheduler, "_ANSWER_TTL_S", 0.0)
    config = _config(extra=_extra(lookup_threads=1))
    sched = make_umbp_store_scheduler(config)
    lookup = sched._lookup_client
    seq = _seq(6)
    sid = str(seq.id)
    assert lookup.submit(seq, sid)
    _wait(lambda: lookup.poll(sid))
    _store(server, config, seq, range(2))  # stored after the prefetch
    seq.offload_hbm_probe_tokens = 0
    assert sched._tier_lookup(seq, sid) == 2 * BLOCK
    assert lookup.prefetch_answers_stale == 1


def test_a_prefetch_in_flight_is_pending_and_admission_waits_for_it(server):
    config = _config(extra=_extra(lookup_threads=1))
    sched = make_umbp_store_scheduler(config)
    lookup = sched._lookup_client
    seq = _seq(6)
    _store(server, config, seq, range(6))
    sid = str(seq.id)
    server.slow_on = ("exists", 0.2)
    assert lookup.submit(seq, sid)
    assert lookup.is_pending(sid) and not lookup.poll(sid)
    assert sched.lookup_pending(seq)  # the base passes it over meanwhile
    server.slow_on = None
    # A lookup that meets it in flight waits for it, then reads it.
    seq.offload_hbm_probe_tokens = 0
    assert sched._tier_lookup(seq, sid) == 6 * BLOCK
    assert not lookup.is_pending(sid)


def test_unconsumed_answers_are_dropped(server, monkeypatch):
    monkeypatch.setattr(umbp_scheduler, "_ANSWER_TTL_S", 0.0)
    config = _config(extra=_extra(lookup_threads=1))
    lookup = make_umbp_store_scheduler(config)._lookup_client
    seq = _seq(2)
    assert lookup.submit(seq, "x")
    _wait(lambda: lookup.answered("x"))
    lookup.pump()
    assert "x" not in lookup.pending_ids()


def test_tier_probe_reads_a_prefetched_answer_then_the_index(server):
    config = _config(extra=_extra(lookup_threads=1, tier_index_chunks=64))
    sched = make_umbp_store_scheduler(config)
    lookup = sched._lookup_client
    seq = _seq(6)
    _store(server, config, seq, range(1, 4))
    sid = str(seq.id)
    assert sched.tier_probe(seq, 1 * BLOCK) == 0  # nothing known yet
    assert lookup.submit(seq, sid)
    _wait(lambda: lookup.poll(sid))
    assert sched.tier_probe(seq, 1 * BLOCK) == 4 * BLOCK
    assert sched.tier_probe(seq, 0) == 0  # lazy: chunk 0 is HBM's, not the tier's
    # A lookup teaches the index what it found; the probe then reads that.
    seq.offload_hbm_probe_tokens = 1 * BLOCK
    assert sched._tier_lookup(seq, sid) == 4 * BLOCK
    assert sched.tier_probe(seq, 1 * BLOCK) == 4 * BLOCK


def test_no_lookup_threads_runs_lookups_on_the_caller(server):
    config = _config()
    sched = make_umbp_store_scheduler(config)
    lookup = sched._lookup_client
    seq = _seq(2)
    assert not lookup.submit(seq, str(seq.id))
    assert lookup._pool is None
    seq.offload_hbm_probe_tokens = 0
    assert sched._tier_lookup(seq, str(seq.id)) == 0
    assert lookup.client is not None  # the synchronous client


# --- miss accounting (ATOM_OFFLOAD_MISS_ACCOUNTING) ----------------------------------

from atom.kv_transfer.offload.miss_accounting import MissAccountant


def _accountant(capacity=100):
    return MissAccountant(capacity, BLOCK, "test")


def test_the_reusable_prefix_is_the_run_of_chunks_computed_before():
    acct = _accountant()
    digests = _digests(_seq(6))
    assert acct.reusable_chunks(digests) == 0
    acct.computed(digests[:4])
    assert acct.reusable_chunks(digests) == 4
    acct.computed(digests)
    assert acct.reusable_chunks(digests) == 6


@pytest.mark.parametrize(
    ("hbm", "held", "load", "expected"),
    [
        # HBM has 2 chunks, the tier the next 2, all loaded.
        (2, 4, 2, {"hbm": 2, "tier": 2, "tier_not_loaded": 0, "evicted": 0}),
        # The tier held them but nothing was loaded (under the floor).
        (2, 4, 0, {"hbm": 2, "tier": 0, "tier_not_loaded": 2, "evicted": 0}),
        # Neither held chunks 2-3.
        (2, 2, 0, {"hbm": 2, "tier": 0, "tier_not_loaded": 0, "evicted": 2}),
        (0, 0, 0, {"hbm": 0, "tier": 0, "tier_not_loaded": 0, "evicted": 4}),
    ],
)
def test_a_requests_reusable_prefix_is_split_by_where_it_came_from(
    hbm, held, load, expected
):
    acct = _accountant()
    digests = _digests(_seq(6))
    acct.computed(digests[:4])
    acct.admitted(
        "r",
        digests,
        prompt_tokens=6 * BLOCK,
        hbm_tokens=hbm * BLOCK,
        tier_hit_tokens=held * BLOCK,
        load_tokens=load * BLOCK,
    )
    assert acct.totals == {
        "new": 2 * BLOCK,
        **{name: chunks * BLOCK for name, chunks in expected.items()},
    }


def test_a_failed_load_moves_its_tokens_to_not_loaded():
    acct = _accountant()
    digests = _digests(_seq(4))
    acct.computed(digests)
    acct.admitted(
        "r",
        digests,
        prompt_tokens=4 * BLOCK,
        hbm_tokens=BLOCK,
        tier_hit_tokens=4 * BLOCK,
        load_tokens=3 * BLOCK,
    )
    acct.load_failed("r")
    assert acct.totals["tier"] == 0 and acct.totals["tier_not_loaded"] == 3 * BLOCK
    acct.load_failed("r")  # once only
    assert acct.totals["tier_not_loaded"] == 3 * BLOCK


def test_the_oldest_computed_chunks_are_forgotten_first():
    acct = _accountant(capacity=3)
    first, second = _digests(_seq(2)), _digests(_seq(2, base=500))
    acct.computed(first)
    acct.computed(second)
    assert acct.reusable_chunks(first) == 0  # dropped for the newer ones
    assert acct.reusable_chunks(second) == 2


def test_the_scheduler_accounts_an_admitted_request(server, monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_MISS_ACCOUNTING", "1")
    # The config settles the load floor from the env or the tier's default.
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", str(BLOCK))
    config = _config(extra=_extra(save_policy="eager"))
    sched = make_umbp_store_scheduler(config)
    acct = sched._miss_accounting
    assert acct is not None
    seq = _seq(6)
    digests = _digests(seq)
    acct.computed(digests[:5])  # an earlier prompt computed five chunks
    _store(server, config, seq, range(4))  # the tier still holds four
    seq.num_cached_tokens = 0
    need, parks = sched.get_num_new_matched_tokens(seq)
    assert (need, parks) == (4 * BLOCK, True)
    sched.update_state_after_alloc(seq)
    assert acct.totals == {
        "new": 1 * BLOCK,
        "hbm": 0,
        "tier": 4 * BLOCK,
        "tier_not_loaded": 0,
        "evicted": 1 * BLOCK,
    }
    stats = sched.get_statistics()
    assert stats["miss_tier_tokens"] == 4 * BLOCK and stats["umbp_lookups"] == 1
    # Finished: its whole prompt is now computed.
    seq.num_hashed_tokens = 6 * BLOCK
    sched.request_finished(seq)
    assert acct.reusable_chunks(digests) == 6
