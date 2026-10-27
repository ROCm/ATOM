# SPDX-License-Identifier: MIT
"""Prompt block hashes kept with the tokenizer cache's stored prefixes."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_admission_host_overhead import over_ipc, published
from test_io_processor_fanout import InputOutputProcessor, _make_processor
from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from atom.model_engine import prefix_block_hashes as pb
from atom.model_engine import prefix_hash as ph
from atom.model_engine import segment_tokenizer as st
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def cached(monkeypatch):
    monkeypatch.setattr(st, "_MIN_CACHED_CHARS", 0)
    monkeypatch.setattr(st, "_PREFIX_KEY_CHARS", 16)
    monkeypatch.setenv("ATOM_FRONTEND_PREFIX_HASH", "1")


@pytest.fixture(scope="module")
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["alpha beta gamma delta 12345 hello world"] * 10,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    specials = ["<BOS>", "<EOS>", "<U>", "<A>", "<END>", "<think>", "</think>"]
    backend.add_special_tokens([AddedToken(t, special=True) for t in specials])
    backend.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[(t, backend.token_to_id(t)) for t in ["<BOS>", "<EOS>"]],
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def prompt():
    return "<U>" + "alpha beta gamma delta " * 1000 + "<A><think>"


def assert_block_hashes(encoded, tokenizer, text, block, seed=-1):
    ids = tokenizer.encode(text)
    assert encoded.token_ids == ids
    hashes = encoded.block_hashes
    assert hashes is not None and hashes.num_tokens == len(ids)
    assert list(hashes.as_array()) == ph.chained_block_hashes(
        ids, block, 0, len(ids) // block, seed
    )


def shared_token_count(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


@pytest.mark.parametrize("block", [4, 16, 48, 1024])
@pytest.mark.parametrize("seed", [-1, 2**63 + 9])
def test_rewrites_forks_mid_edits_and_wrapped_tails(tokenizer, block, seed):
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    text = prompt()
    texts = [text, text.replace("<think>", "</think>") + "hello<END><U>world<A>"]
    texts += [texts[-1] + "delta<END><U>" + "beta " * i + "<A>" for i in range(1, 7)]
    texts += [text.replace("gamma", "world", 1), text[:20000] + "delta<END><U>hello<A>"]
    for text in texts:
        result = cache.encode_with_prefix(text, block, seed)
        assert_block_hashes(result, tokenizer, text, block, seed)
    assert cache.enabled and cache._lru.bytes <= cache._lru.budget


def test_only_new_complete_blocks_are_hashed(tokenizer):
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    before = cache.encode_with_prefix(prompt(), 16)
    text = prompt().replace("<think>", "</think>") + "hello<END><U>beta<A>"
    with patch.object(ph, "_link", wraps=ph._link) as kv:
        result = cache.encode_with_prefix(text, 16)
    shared = shared_token_count(before.token_ids, result.token_ids)
    assert shared // 16 > 100
    assert kv.call_count == len(result.token_ids) // 16 - shared // 16
    assert_block_hashes(result, tokenizer, text, 16)


def test_same_budget_owns_hashes_and_eviction_cannot_change_outputs(tokenizer):
    cache = st.SegmentTokenizer(tokenizer, 170_000)
    for i in range(12):
        text = prompt().replace("gamma", str(i), 1) + "hello<END><U>world<A>"
        result = cache.encode_with_prefix(text, 4)
        assert_block_hashes(result, tokenizer, text, 4)
        assert cache._lru.bytes <= cache._lru.budget
        assert cache._lru.bytes == sum(cost for _, cost in cache._lru._items.values())
        for key, (entry, cost) in cache._lru._items.items():
            if isinstance(key, tuple):
                assert entry.block_hashes is not None
                assert cost > entry.ids.nbytes + entry.block_hashes.payload_bytes
                assert entry.block_hashes.num_tokens == len(cache._split.head) + len(
                    entry.ids
                )
    # A request never exposes mutable arrays retained by the shared record.
    result.token_ids[0] = 123456
    again = cache.encode_with_prefix(text, 4)
    assert_block_hashes(again, tokenizer, text, 4)


def test_concurrent_forks_do_not_share_mutable_hash_or_token_arrays(tokenizer):
    cache = st.SegmentTokenizer(tokenizer, 1 << 20)
    cache.encode_with_prefix(prompt(), 16)

    def run(i):
        text = prompt() + "hello<END><U>" + str(i) + "<A>"
        encoded = cache.encode_with_prefix(text, 16)
        assert_block_hashes(encoded, tokenizer, text, 16)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(run, range(24)))
    assert cache._lru.bytes <= cache._lru.budget


def test_no_block_hashes_when_prefix_reuse_is_off(tokenizer):
    """Overlapping added tokens turn prefix reuse off (`_prefix_ok`): nothing
    would ever be reused, so no chain is kept with the record."""
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    cache._prefix_ok = False
    text = prompt() + "hello<END><U>world<A>"
    with patch.object(st, "extend_block_hashes", wraps=st.extend_block_hashes) as proj:
        for _ in range(2):
            encoded = cache.encode_with_prefix(text, 16)
            assert encoded.token_ids == tokenizer.encode(text)
            assert encoded.block_hashes is None
    assert proj.call_count == 0


def test_disable_during_encode_drops_the_hint_and_cached_hashes(tokenizer, monkeypatch):
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    cache.encode_with_prefix(prompt(), 16)
    real = cache._split.backend.encode_batch

    def disable(texts, **kwargs):
        with cache._lock:
            cache.enabled = False
            cache._lru.clear()
            cache._buckets.clear()
        return real(texts, **kwargs)

    monkeypatch.setattr(cache._split.backend, "encode_batch", disable)
    text = prompt() + "new text<END><U>more<A>"
    encoded = cache.encode_with_prefix(text, 16)
    assert encoded.token_ids == tokenizer.encode(text)
    assert encoded.block_hashes is None
    assert cache._lru.bytes == 0 and not cache._buckets


def test_seed_and_geometry_switch_recomputes_the_whole_chain(tokenizer):
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    cache.encode_with_prefix(prompt(), 16, -1)
    text = prompt() + "hello<END><U>world<A>"
    with patch.object(ph, "_link", wraps=ph._link) as kv:
        result = cache.encode_with_prefix(text, 48, 917)
    assert kv.call_count == len(result.token_ids) // 48
    assert_block_hashes(result, tokenizer, text, 48, 917)


def test_ipc_hbm_adoption_does_not_turn_content_into_residency():
    tokens = list(range(40)) + [900, 901, 902]
    record = pb.extend_block_hashes(tokens, 4)
    seq = Sequence(tokens, 4)
    with patch.object(ph, "_link", wraps=ph._link) as kv:
        ph.ship_prompt_hash_chain(seq, 4, block_hashes=record)
    assert kv.call_count == 0  # the chain came from the record
    [seq] = over_ipc([seq])
    bm = published(range(40))
    assert bm.can_allocate_memoized(seq) == 10
    assert bm.shipped_chains_adopted == 1
    bm.clear_cache()
    # A content hint, not residency: the KV hit goes with the blocks.
    assert bm.can_allocate_memoized(seq) == 0


def test_the_record_stays_on_the_frontend():
    """Only the chain rides the ADD frame."""
    import pickle

    tokens = list(range(4000))
    record = pb.extend_block_hashes(tokens, 4)
    seq = Sequence(tokens, 4)
    ph.ship_prompt_hash_chain(seq, 4, block_hashes=record)
    chain = seq.prompt_hash_chain
    assert not any(isinstance(f, pb.PrefixBlockHashes) for f in chain)
    assert len(pickle.dumps(chain)) < len(record.hashes) + 1024


def test_a_record_of_another_geometry_is_ignored_and_the_chain_rehashed():
    tokens = list(range(40))
    for other in (
        pb.extend_block_hashes(tokens, 8),  # block size
        pb.extend_block_hashes(tokens, 4, seed=7),  # seed
        pb.extend_block_hashes(tokens[:36], 4),  # length
    ):
        seq = Sequence(tokens, 4)
        ph.ship_prompt_hash_chain(seq, 4, block_hashes=other)
        assert list(seq.prompt_hash_chain.hashes) == ph.chained_block_hashes(
            tokens, 4, 0, 10, -1
        )


def test_ship_fingerprints_the_prompt_once(tokenizer):
    """One xxh64 of the prompt per request -- the one `ShippedChain` carries
    -- however the chain was obtained."""
    cache = st.SegmentTokenizer(tokenizer, 8 << 20)
    cache.encode_with_prefix(prompt(), 4)
    text = prompt() + "hello<END><U>beta<A>"
    encoded = cache.encode_with_prefix(text, 4)
    seq = Sequence(encoded.token_ids, 4)
    with patch.object(ph, "_xxh64", wraps=ph._xxh64) as xxh:
        ph.ship_prompt_hash_chain(seq, 4, block_hashes=encoded.block_hashes)
    assert xxh.call_count == 1


def test_real_frontend_fanout_uses_one_prefix_record(tokenizer):
    proc = _make_processor()
    proc.tokenizer = tokenizer
    proc.segment_tokenizer = st.SegmentTokenizer(tokenizer, 8 << 20)
    proc._reuse_block_hashes = True
    proc.prefix_hash_block_size = 4
    text = prompt()
    proc.preprocess_fanout(text, SamplingParams())
    text += "hello<END><U>beta<A>"
    with patch.object(ph, "_link", wraps=ph._link) as kv:
        siblings = proc.preprocess_fanout(text, SamplingParams(n=2))
    assert kv.call_count < len(siblings[0].prompt_hash_chain.hashes)
    assert siblings[0].prompt_hash_chain is siblings[1].prompt_hash_chain
    for seq in over_ipc(siblings):
        assert list(seq.prompt_hash_chain.hashes) == ph.chained_block_hashes(
            tokenizer.encode(text), 4, 0, len(seq.prompt_token_ids) // 4, -1
        )


def test_processor_initialization_uses_one_cache_by_default(tokenizer, monkeypatch):
    monkeypatch.delenv("ATOM_TOKENIZE_CACHE_BLOCK_HASHES", raising=False)
    config = SimpleNamespace(
        speculative_config=None,
        hf_config=SimpleNamespace(model_type="llama"),
        enable_prefix_caching=True,
        decode_context_parallel_size=2,
    )
    proc = InputOutputProcessor(config, tokenizer, 4)
    assert proc.segment_tokenizer is not None
    assert proc.prefix_hash_block_size == 8 and proc._reuse_block_hashes
    proc.preprocess_fanout(prompt(), SamplingParams())
    with patch.object(ph, "_link", wraps=ph._link) as kv:
        [seq] = proc.preprocess_fanout(prompt() + "hello<END><U>x<A>", SamplingParams())
    assert seq.prompt_hash_chain.hash_block_size == 8
    assert kv.call_count < len(seq.prompt_hash_chain.hashes)
