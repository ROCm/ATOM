# SPDX-License-Identifier: MIT
"""LMCache MP keys that name prompts by their LMCache chunk hashes."""

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_admission_host_overhead import over_ipc
from test_io_processor_fanout import _make_processor
from test_lmcache_mp import _config, _LookupAdapter, _WorkerAdapter, _WorkerFuture
from test_prefix_block_hashes import cached, prompt, tokenizer  # noqa: F401

from atom.kv_transfer.disaggregation.types import LoadOperationId, SaveOperationId
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec, SaveSpec
from atom.kv_transfer.offload.mp import chunk_hashes as ch
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import transfer
from atom.kv_transfer.offload.mp import worker as mp_worker
from atom.model_engine import prefix_block_hashes as pb
from atom.model_engine import prefix_hash as ph
from atom.model_engine import segment_tokenizer as st
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

lmc = pytest.importorskip("lmcache.integration.atom")
token_hasher = pytest.importorskip("lmcache.v1.multiprocess.token_hasher")

CHUNK = 8
H = ch.CHUNK_HASH_BYTES

# Tests that hash or key through LMCache itself need an LMCache whose ATOM
# adapters take caller chunk hashes; older builds skip them.
needs_caller_hashes = pytest.mark.skipif(
    not getattr(lmc.AtomMPWorkerAdapter, "ACCEPTS_CALLER_CHUNK_HASHES", False),
    reason="the installed LMCache does not take caller chunk hashes",
)


def server_chain(tokens, chunk=CHUNK) -> bytes:
    """What an LMCache server hashing with blake3 names `tokens`' chunks by."""
    hasher = token_hasher.TokenHasher(chunk_size=chunk, hash_algorithm="blake3")
    return b"".join(hasher.compute_chunk_hashes(list(tokens)))


# -- the chain itself ----------------------------------------------------------


@needs_caller_hashes
@pytest.mark.parametrize("chunk", [1, 8, 256])
def test_frontend_chain_is_lmcaches_chain(chunk):
    tokens = [(i * 7919) % 150_000 for i in range(3 * chunk + 5)]
    chain = pb.lmcache_chunk_chain(tokens, chunk)
    assert chain == server_chain(tokens, chunk)
    assert chain == lmc.lmcache_chunk_hashes(tokens, chunk)
    hasher = token_hasher.TokenHasher(chunk_size=chunk, hash_algorithm="blake3")
    assert pb._lmcache_none_hash() == hasher.none_hash


def test_a_later_turn_hashes_only_its_new_chunks():
    first = list(range(100))
    record = pb.extend_lmcache_chunk_hashes(first, CHUNK).truncated(96)
    second = first[:96] + list(range(500, 540))
    with patch.object(pb, "lmcache_chunk_chain", wraps=pb.lmcache_chunk_chain) as h:
        grown = pb.extend_lmcache_chunk_hashes(
            second, CHUNK, base=record, shared_tokens=96
        )
    [(args, _)] = h.call_args_list
    assert len(args[0]) == len(second) // CHUNK * CHUNK - 96
    assert grown.hashes == server_chain(second)
    # Another chunk size shares nothing.
    other = pb.extend_lmcache_chunk_hashes(second, 4, base=record, shared_tokens=96)
    assert other.hashes == server_chain(second, 4)


def test_tokenizer_cache_keeps_the_chain(tokenizer):  # noqa: F811
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    text = prompt()
    first = cache.encode_with_prefix(text, 0, lmcache_chunk_size=CHUNK)
    assert first.block_hashes is None
    assert first.lmcache_hashes.hashes == server_chain(first.token_ids)
    text = text.replace("<think>", "</think>") + "hello<END><U>beta<A>"
    with patch.object(st, "extend_lmcache_chunk_hashes") as extend:
        extend.side_effect = pb.extend_lmcache_chunk_hashes
        second = cache.encode_with_prefix(text, 16, lmcache_chunk_size=CHUNK)
    assert extend.call_args.kwargs["shared_tokens"] > 1000
    assert second.lmcache_hashes.hashes == server_chain(second.token_ids)
    assert second.block_hashes is not None  # the block chain rides along
    for key, (entry, _) in cache._lru._items.items():
        if isinstance(key, tuple):
            assert entry.lmcache_hashes.num_tokens == len(cache._split.head) + len(
                entry.ids
            )


# -- shipped with the request, adopted by the engine ---------------------------


def shipped_seq(tokens, chunk=CHUNK, record=None):
    seq = Sequence(list(tokens), 4)
    ph.ship_lmcache_chunk_chain(seq, chunk, chunk_hashes=record)
    return seq


@needs_caller_hashes
def test_the_chain_rides_the_add_frame_and_is_adopted_without_rehashing():
    tokens = list(range(70))
    record = pb.extend_lmcache_chunk_hashes(tokens, CHUNK)
    with patch.object(pb, "lmcache_chunk_chain") as frontend_hash:
        seq = shipped_seq(tokens, record=record)
    frontend_hash.assert_not_called()  # taken from the tokenizer's record
    [seq] = over_ipc([seq])
    assert seq.lmcache_chunk_chain.hashes == record.hashes[: 8 * H]
    with patch.object(lmc, "lmcache_chunk_hashes", wraps=lmc.lmcache_chunk_hashes) as h:
        hashes = ch.prompt_chunk_hashes(seq, CHUNK)
    assert hashes == server_chain(tokens)
    # Only the three sampled links, one chunk each.
    assert h.call_count == 3 and all(len(c.args[0]) == CHUNK for c in h.call_args_list)
    assert seq.lmcache_chunk_chain is None  # consumed
    with patch.object(lmc, "lmcache_chunk_hashes") as h:
        assert ch.prompt_chunk_hashes(seq, CHUNK) is hashes  # kept on the seq
    h.assert_not_called()


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda c: c._replace(chunk_size=c.chunk_size * 2),
        lambda c: c._replace(fingerprint=c.fingerprint + 1),
        lambda c: c._replace(hashes=bytes(len(c.hashes))),
        lambda c: c._replace(hashes=c.hashes[:-H]),
    ],
)
@needs_caller_hashes
def test_a_chain_that_does_not_check_out_is_rehashed(corrupt):
    tokens = list(range(70))
    seq = shipped_seq(tokens)
    seq.lmcache_chunk_chain = corrupt(seq.lmcache_chunk_chain)
    assert ch.prompt_chunk_hashes(seq, CHUNK) == server_chain(tokens)


@needs_caller_hashes
def test_without_a_shipped_chain_the_engine_hashes_once_and_extends():
    seq = Sequence(list(range(30)), 4)
    assert ch.prompt_chunk_hashes(seq, CHUNK) == server_chain(range(24))
    seq.token_ids.extend(range(100, 120))
    seq.num_prompt_tokens = 50  # a preempted request's prompt grows
    with patch.object(lmc, "lmcache_chunk_hashes", wraps=lmc.lmcache_chunk_hashes) as h:
        grown = ch.prompt_chunk_hashes(seq, CHUNK)
    assert len(h.call_args.args[0]) == 48 - 24
    assert grown == server_chain(list(seq.token_ids)[:48])


@needs_caller_hashes
def test_frontend_ships_one_chain_for_siblings(tokenizer):  # noqa: F811
    proc = _make_processor()
    proc.tokenizer = tokenizer
    proc.segment_tokenizer = st.SegmentTokenizer(tokenizer, 8 << 20)
    proc._reuse_block_hashes = True
    proc.lmcache_chunk_size = CHUNK
    text = prompt()
    proc.preprocess_fanout(text, SamplingParams())
    text += "hello<END><U>beta<A>"
    with patch.object(pb, "lmcache_chunk_chain", wraps=pb.lmcache_chunk_chain) as h:
        siblings = proc.preprocess_fanout(text, SamplingParams(n=2))
    tokens = tokenizer.encode(text)
    assert sum(len(c.args[0]) for c in h.call_args_list) < len(tokens) // 2
    assert siblings[0].lmcache_chunk_chain is siblings[1].lmcache_chunk_chain
    assert siblings[0].prompt_hash_chain is None  # no prefix cache here
    for seq in over_ipc(siblings):
        assert seq.lmcache_chunk_chain.hashes == server_chain(tokens)
        assert ch.prompt_chunk_hashes(seq, CHUNK) == server_chain(tokens)
    # Token-ID input has no tokenizer record: the frontend hashes it whole.
    [seq] = proc.preprocess_fanout(tokens, SamplingParams())
    assert seq.lmcache_chunk_chain.hashes == server_chain(tokens)


ON = {"lmcache.mp.client_chunk_hashes": True}


def test_frontend_chunk_size_only_for_the_mp_connector_with_hashes(monkeypatch):
    monkeypatch.delenv("LMCACHE_CHUNK_SIZE", raising=False)
    assert ch.frontend_chunk_size(_config(extra=ON)) == 256
    config = _config(extra={**ON, "lmcache.chunk_size": 64})
    assert ch.frontend_chunk_size(config) == 64
    # Keys carrying tokens need no chain.
    assert ch.frontend_chunk_size(_config()) == 0
    off = {"lmcache.mp.client_chunk_hashes": False}
    assert ch.frontend_chunk_size(_config(extra=off)) == 0
    config.kv_transfer_config["kv_connector"] = "mooncake_store"
    assert ch.frontend_chunk_size(config) == 0
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "0")
    assert ch.frontend_chunk_size(_config(extra=ON)) == 0


def test_hash_mode_needs_the_option_and_an_adapter_taking_caller_hashes(caplog):
    taking = SimpleNamespace(ACCEPTS_CALLER_CHUNK_HASHES=True)
    with caplog.at_level("WARNING", logger="atom"):
        assert ch.chunk_hashes_enabled(ON, taking)
        assert not ch.chunk_hashes_enabled({}, taking)
        off = {"lmcache.mp.client_chunk_hashes": False}
        assert not ch.chunk_hashes_enabled(off, taking)
        assert not caplog.records
        # Set, but the installed LMCache cannot take ATOM's hashes: say so
        # loudly, since every transfer then carries the token prefix again.
        assert not ch.chunk_hashes_enabled(ON, SimpleNamespace())
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "does not take caller chunk hashes" in record.getMessage()


@needs_caller_hashes
def test_real_lmcache_adapters_take_caller_hashes():
    assert lmc.AtomMPSchedulerAdapter.ACCEPTS_CALLER_CHUNK_HASHES
    assert lmc.AtomMPWorkerAdapter.ACCEPTS_CALLER_CHUNK_HASHES
    assert "chunk_hashes" in lmc.AtomMPTransferSpec.__dataclass_fields__


# -- scheduler: lookups, saves and loads carry hashes, not tokens --------------


class _HashLookupAdapter(_LookupAdapter):
    lmcache_tokens_per_chunk = CHUNK

    def maybe_submit_lookup_request(self, request_id, token_ids, chunk_hashes=None):
        self.submissions.append((request_id, list(token_ids), chunk_hashes))


def _hash_scheduler(monkeypatch, hit):
    monkeypatch.setattr(transfer.time, "sleep", lambda _seconds: None)
    adapter = _HashLookupAdapter([hit])
    lookup = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=1.0, poll_interval=0.01
    )
    scheduler = mp_scheduler.LMCacheMPConnectorScheduler.__new__(
        mp_scheduler.LMCacheMPConnectorScheduler
    )
    scheduler._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        scheduler, _config(), chunk_size=CHUNK, lookup_client=lookup
    )
    scheduler._hash_chunks = True
    scheduler._min_load_tokens = 0
    return scheduler, adapter


@needs_caller_hashes
def test_lookup_load_and_lock_release_carry_hashes(monkeypatch):
    scheduler, adapter = _hash_scheduler(monkeypatch, hit=24)
    seq = Sequence(list(range(36)), 4)
    seq.block_table = list(range(10, 19))
    chain = server_chain(range(32))

    assert scheduler.get_num_new_matched_tokens(seq) == (24, True)
    [(_, token_ids, hashes)] = adapter.submissions
    assert token_ids == [] and hashes == chain
    seq.num_cached_tokens = 8
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.token_ids == [] and request.chunk_hashes == chain[: 3 * H]
    assert request.token_end == 24
    # The HBM prefix [0, 8) of the hit is not retrieved: its locks are freed
    # by the hashes of [0, 8).
    [freed] = adapter.freed
    assert (freed["start"], freed["end"]) == (0, 8)
    assert freed["token_ids"] == () and freed["chunk_hashes"] == chain[:H]


@needs_caller_hashes
def test_saves_carry_digests_not_tokens(monkeypatch):
    scheduler, _ = _hash_scheduler(monkeypatch, hit=0)
    seq = Sequence(list(range(30)), 4)
    seq.block_table = list(range(10, 18))
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 17
    [first] = scheduler.build_connector_meta().requests
    assert first.token_ids == [] and first.save_spec.skip_leading_tokens == 0
    assert first.chunk_hashes == server_chain(range(16)) and first.token_end == 16
    scheduler.save_finished(first.save_operation)
    seq.num_cached_tokens = 30
    [second] = scheduler.build_connector_meta().requests
    assert second.save_spec.skip_leading_tokens == 16
    assert second.chunk_hashes == server_chain(range(24))  # whole chunks only
    assert second.token_end == 24


def test_token_mode_is_unchanged(monkeypatch):
    scheduler, _ = _hash_scheduler(monkeypatch, hit=0)
    scheduler._hash_chunks = False
    seq = Sequence(list(range(20)), 4)
    seq.block_table = list(range(10, 15))
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 20
    [request] = scheduler.build_connector_meta().requests
    assert request.token_ids == list(range(16)) and request.chunk_hashes is None
    assert request.token_end is None


@needs_caller_hashes
def test_native_state_lookup_stops_a_chunk_short_of_the_prompt_end():
    from atom.kv_transfer.offload.mp.native_state_scheduler import (
        NativeStateLMCacheMPConnectorScheduler as Native,
    )

    scheduler = SimpleNamespace(
        _hash_chunks=True, chunk_size=CHUNK, _chunk_floor=lambda n: n // CHUNK * CHUNK
    )
    seq = Sequence(list(range(32)), 4)
    named = Native._lookup_token_ids(scheduler, seq)
    assert len(named) == 24 and named.hashes == server_chain(range(24))
    scheduler._hash_chunks = False
    assert Native._lookup_token_ids(scheduler, seq) == list(range(24))


# -- worker: the spec carries the hashes of [0, end) ---------------------------


@dataclass
class _HashSpec:
    block_ids: list[list[int]]
    start: int = 0
    end: int = 0
    token_ids: list[int] | None = None
    chunk_hashes: bytes | None = None


@pytest.fixture
def hash_spec_modules(monkeypatch):
    atom = types.ModuleType("lmcache.integration.atom")
    atom.AtomMPTransferSpec = _HashSpec
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", atom)


def _worker():
    worker = mp_worker.LMCacheMPConnector(_config())
    worker._adapter = _WorkerAdapter()
    worker.chunk_size = CHUNK
    return worker


def test_worker_sends_the_hashes_up_to_the_ranges_end(hash_spec_modules):
    worker = _worker()
    chain = server_chain(range(32))
    save = LMCacheReqMeta(
        req_id=3,
        token_ids=[],
        block_ids=list(range(20, 28)),
        save_spec=SaveSpec(skip_leading_tokens=8),
        save_operation=SaveOperationId(3, 0),
        chunk_hashes=chain[: 3 * H],
        token_end=27,
    )
    worker._submit_save(save, object())
    [(_, op, _, _)] = worker._adapter.saves
    assert (op.start, op.end, op.token_ids) == (8, 24, [])
    assert op.chunk_hashes == chain[: 3 * H]  # [0, end), not [start, end)
    assert op.block_ids == [[22, 23, 24, 25]]

    load = LMCacheReqMeta(
        req_id=4,
        token_ids=[],
        block_ids=list(range(30, 38)),
        load_spec=LoadSpec(8, 24, can_load=True),
        load_operation=LoadOperationId(4, 0),
        chunk_hashes=chain,
        token_end=32,
    )
    worker._submit_load(load, object())
    [(_, op, _, _)] = worker._adapter.loads
    assert (op.start, op.end) == (8, 24) and op.chunk_hashes == chain[: 3 * H]


@needs_caller_hashes
def test_worker_keys_match_the_token_path_through_lmcache():
    """The real LMCache worker adapter turns ATOM's hash spec into the very
    key it builds from the tokens."""
    mpa = pytest.importorskip("lmcache.integration.atom.multi_process_adapter")
    tokens = [(i * 7919) % 150_000 for i in range(3 * CHUNK + 5)]
    adapter = mpa.AtomMPWorkerAdapter.__new__(mpa.AtomMPWorkerAdapter)
    adapter._model_name = "m"
    adapter._parallel = mpa.AtomMPParallelConfig(2, 1, 2)
    adapter.lmcache_tokens_per_chunk = CHUNK
    adapter._hash_cache = mpa._ChunkHashCache(CHUNK)
    by_tokens = LMCacheReqMeta(req_id=1, token_ids=tokens, block_ids=[])
    by_hashes = LMCacheReqMeta(
        req_id=1,
        token_ids=[],
        block_ids=[],
        chunk_hashes=server_chain(tokens),
        token_end=len(tokens),
    )
    keys = []
    for req in (by_tokens, by_hashes):
        fields = ch.transfer_key_fields(req, 3 * CHUNK, CHUNK)
        spec = mpa.AtomMPTransferSpec(
            block_ids=[], start=CHUNK, end=3 * CHUNK, **fields
        )
        keys.append(adapter._create_key("r", spec))
    assert keys[0] == keys[1]
    assert keys[1].token_bytes == b""
    assert keys[1].chunk_hashes == server_chain(tokens)[: 3 * H]


def test_request_prefix_tokens():
    req = SimpleNamespace(token_ids=[], chunk_hashes=bytes(3 * H), token_end=27)
    assert ch.request_prefix_tokens(req) == 27
    req = SimpleNamespace(token_ids=list(range(13)), chunk_hashes=None)
    assert ch.request_prefix_tokens(req) == 13


def test_hashes_too_short_for_the_range_are_refused():
    req = SimpleNamespace(token_ids=[], chunk_hashes=bytes(2 * H), token_end=16)
    with pytest.raises(ValueError, match="cannot name"):
        ch.transfer_key_fields(req, 3 * CHUNK, CHUNK)


assert _WorkerFuture  # the worker adapter's futures, as in test_lmcache_mp
