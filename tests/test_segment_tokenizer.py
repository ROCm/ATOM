# SPDX-License-Identifier: MIT
"""`SegmentTokenizer` must return exactly what the plain encoder returns."""

import random
from unittest.mock import MagicMock

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

from atom.model_engine import segment_tokenizer as seg_mod
from atom.model_engine.segment_tokenizer import SegmentTokenizer

SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>", "<|end|>"]
WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "12345",
    "hello,",
    "world!",
]


def _text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


@pytest.fixture(autouse=True)
def _cache_every_prompt(monkeypatch):
    """The test prompts are short; route them through the cache anyway."""
    monkeypatch.setattr(seg_mod, "_MIN_CACHED_CHARS", 0)


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(0)
    corpus = [_text(rng, 40) for _ in range(200)]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    backend.add_special_tokens([AddedToken(t, special=True) for t in SPECIALS])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend)


def _conversation(rng, turns):
    prompt = "<|begin|>" + _text(rng, 30)
    for t in range(turns):
        prompt += ("<|User|>" if t % 2 == 0 else "<|Assistant|>") + _text(
            rng, rng.randint(5, 60)
        )
    return prompt


def test_matches_the_plain_encoder_turn_after_turn(tok):
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    assert seg is not None
    rng = random.Random(1)
    prompt = _conversation(rng, 2)
    for _ in range(10):
        prompt += "<|User|>" + _text(rng, 20) + "<|Assistant|>" + _text(rng, 8)
        assert seg.encode(prompt) == tok.encode(prompt)
    assert seg.enabled
    # Later turns re-encode only their new segments.
    reused, fresh = seg._stats[0], seg._stats[1]
    assert reused > fresh


def test_special_token_spelling_inside_text_still_splits(tok):
    """The cache keeps the plain encoder's behaviour, including an added
    token's spelling arriving inside message text."""
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    prompt = "<|begin|>tool output says <|end|> and more<|User|>hi"
    assert seg.encode(prompt) == tok.encode(prompt)


def test_eviction_keeps_results_exact(tok):
    seg = SegmentTokenizer.maybe_create(tok, 64)
    rng = random.Random(2)
    for _ in range(20):
        prompt = _conversation(rng, 6)
        assert seg.encode(prompt) == tok.encode(prompt)
    assert seg._cached_tokens <= 64


def test_a_mismatch_turns_the_cache_off(tok, monkeypatch):
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    prompt = _conversation(random.Random(3), 4)
    monkeypatch.setattr(seg, "_encode_segments", lambda p: [0])
    assert seg.encode(prompt) == tok.encode(prompt)
    assert not seg.enabled
    monkeypatch.undo()
    assert seg.encode(prompt) == tok.encode(prompt)


def test_not_created_when_off_or_not_a_fast_tokenizer(tok):
    assert SegmentTokenizer.maybe_create(tok, 0) is None
    assert SegmentTokenizer.maybe_create(MagicMock(), 1_000_000) is None


def test_not_created_when_an_added_token_strips_whitespace():
    rng = random.Random(4)
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        [_text(rng, 40) for _ in range(50)],
        trainers.BpeTrainer(vocab_size=280, special_tokens=["[UNK]"]),
    )
    backend.add_special_tokens([AddedToken("<|User|>", lstrip=True, special=True)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    assert SegmentTokenizer.maybe_create(tok, 1_000_000) is None


def test_short_prompts_are_encoded_whole(tok, monkeypatch):
    monkeypatch.setattr(seg_mod, "_MIN_CACHED_CHARS", 10**9)
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    prompt = _conversation(random.Random(5), 4)
    assert seg.encode(prompt) == tok.encode(prompt)
    assert seg._stats[3] == 0 and seg._cached_tokens == 0


def test_eviction_by_another_request_mid_encode_is_harmless(tok, monkeypatch):
    """Another request can evict this one's cached segments while it encodes
    its new ones outside the lock."""
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    rng = random.Random(6)
    prompt = _conversation(rng, 6)
    assert seg.encode(prompt) == tok.encode(prompt)
    longer = prompt + "<|User|>" + _text(rng, 30)
    real = seg._backend.encode_batch

    def evict_then_encode(texts, **kw):
        with seg._lock:
            seg._cache.clear()
            seg._cached_tokens = 0
        return real(texts, **kw)

    monkeypatch.setattr(seg, "_backend", MagicMock(encode_batch=evict_then_encode))
    seg._verified = seg_mod._VERIFY_FIRST  # unchecked, as in steady state
    assert seg._encode_segments(longer) == tok.encode(longer)


def _normalized_tokenizer(normalizer):
    rng = random.Random(7)
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    if normalizer is not None:
        backend.normalizer = normalizer
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        [_text(rng, 40) for _ in range(50)],
        trainers.BpeTrainer(vocab_size=280, special_tokens=["[UNK]"]),
    )
    backend.add_special_tokens([AddedToken("<|User|>", special=True)])
    backend.add_tokens([AddedToken("\u0301", normalized=True)])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend)


def test_normalized_added_tokens_need_an_identity_normalizer():
    from tokenizers import normalizers

    assert (
        SegmentTokenizer.maybe_create(_normalized_tokenizer(normalizers.NFC()), 10**6)
        is None
    )
    seg = SegmentTokenizer.maybe_create(_normalized_tokenizer(None), 10**6)
    assert seg is not None
    # Past the verification window, a combining accent is still exact.
    tok = seg._tokenizer
    seg._verified = seg_mod._VERIFY_FIRST
    prompt = "<|User|>a\u0301 caf\u0301e" * 3
    assert seg._encode_segments(prompt) == tok.encode(prompt)


def test_segments_that_encode_to_nothing_are_not_cached(tok, monkeypatch):
    """They would take an entry each but no token budget, so eviction would
    never bound them."""
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    real = seg._backend.encode_batch

    def drop_whitespace(texts, **kw):
        return [
            MagicMock(ids=[]) if not t.strip() else e
            for t, e in zip(texts, real(texts, **kw))
        ]

    monkeypatch.setattr(seg, "_backend", MagicMock(encode_batch=drop_whitespace))
    seg._verified = seg_mod._VERIFY_FIRST
    for n in range(1, 50):
        seg._encode_segments("<|User|>" + " " * n + "<|Assistant|>")
    assert len(seg._cache) == 0 and seg._cached_tokens == 0


def _tokenizer_with(added, pre=None):
    rng = random.Random(8)
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre or pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        [_text(rng, 40) + " x" for _ in range(50)],
        trainers.BpeTrainer(
            vocab_size=280,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    for tok in added:
        if tok.special:
            backend.add_special_tokens([tok])
        else:
            backend.add_tokens([tok])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend)


def test_cross_pass_overlap_follows_the_backends_extraction_order():
    """Non-normalized added tokens are extracted before normalized ones, so
    `x<|User|>` becomes `x` + `<|User|>` rather than the longer token."""
    tok = _tokenizer_with(
        [
            AddedToken("<|User|>", special=True, normalized=False),
            AddedToken("x<|User|>", normalized=True),
        ]
    )
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    assert seg is not None
    seg._verified = seg_mod._VERIFY_FIRST  # past the window: unchecked
    prompt = "alpha x<|User|>beta" * 3
    assert seg._encode_segments(prompt) == tok.encode(prompt)


def test_offset_sensitive_pre_tokenizers_are_refused():
    for pre in (
        pre_tokenizers.Metaspace(prepend_scheme="first"),
        pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(), pre_tokenizers.Metaspace(prepend_scheme="first")]
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=True),
    ):
        tok = _tokenizer_with([AddedToken("<|User|>", special=True)], pre=pre)
        assert SegmentTokenizer.maybe_create(tok, 10**6) is None, pre


def test_a_result_finished_after_the_cache_turned_off_is_not_returned(tok, monkeypatch):
    """Another request turning the cache off while this one encodes: this
    one's unchecked ids must not be returned."""
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    seg._verified = seg_mod._VERIFY_FIRST
    prompt = _conversation(random.Random(9), 4)

    def wrong_and_meanwhile_disabled(p):
        seg.enabled = False
        return [0]

    monkeypatch.setattr(seg, "_encode_segments", wrong_and_meanwhile_disabled)
    assert seg.encode(prompt) == tok.encode(prompt)


def test_a_check_finishing_after_the_cache_turned_off_does_not_count(tok, monkeypatch):
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    seg._verified = seg_mod._VERIFY_FIRST - 1
    prompt = _conversation(random.Random(10), 4)
    real = seg._encode_segments

    def disabled_meanwhile(p):
        ids = real(p)
        seg.enabled = False
        return ids

    monkeypatch.setattr(seg, "_encode_segments", disabled_meanwhile)
    assert seg.encode(prompt) == tok.encode(prompt)
    assert seg._verified == seg_mod._VERIFY_FIRST - 1 and not seg.enabled


def _long_conversation(rng, turns):
    """Long enough that its first `_PREFIX_KEY_CHARS` hold no new turn."""
    prompt = "<|begin|>" + _text(rng, 3000)
    for t in range(turns):
        prompt += ("<|User|>" if t % 2 == 0 else "<|Assistant|>") + _text(rng, 40)
    return prompt


def test_a_later_turn_reuses_the_earlier_prompt_exactly(tok):
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(11)
    prompt = _long_conversation(rng, 4)
    for turn in range(6):
        # The generation prompt ends a turn; the next one replaces it, so a
        # later prompt diverges from the stored one just before its end.
        ask = prompt + "<|Assistant|>"
        assert seg._encode_segments(ask) == tok.encode(ask)
        prompt += "<|Assistant|>" + _text(rng, 15) + "<|User|>" + _text(rng, 25)
    assert seg._prefix_hits >= 5
    # One stored prefix per conversation: each turn replaced the one before.
    assert len(seg._prefixes) == 1


def test_divergence_mid_prefix_reuses_up_to_the_last_shared_added_token(tok):
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(12)
    base = _long_conversation(rng, 6)
    seg._encode_segments(base)
    edited = base[: len(base) - 30] + "rewritten tail<|User|>" + _text(rng, 20)
    hits = seg._prefix_hits
    assert seg._encode_segments(edited) == tok.encode(edited)
    assert seg._prefix_hits == hits + 1


def test_prefix_fast_path_is_off_when_an_added_token_contains_another():
    tok = _tokenizer_with(
        [
            AddedToken("<|User|>", special=True),
            AddedToken("<|User|>x", special=True),
        ]
    )
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    assert seg is not None and not seg._prefix_ok


def test_conversations_sharing_a_start_are_kept_apart_and_exact(tok):
    """Two conversations with the same first characters share a bucket; each
    still reuses only what it shares and stays exact."""
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(14)
    common = _long_conversation(rng, 2)
    a = common + "<|User|>" + _text(rng, 30)
    b = common + "<|User|>" + _text(rng, 30)
    for _ in range(3):
        for p in (a, b):
            ask = p + "<|Assistant|>"
            assert seg._encode_segments(ask) == tok.encode(ask)
        a += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
        b += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
    assert len(seg._prefix_buckets) == 1
    assert len(seg._prefixes) == 2


def test_stored_prefixes_stay_within_the_budget(tok):
    seg = SegmentTokenizer.maybe_create(tok, 20_000)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(15)
    for _ in range(12):
        p = _long_conversation(rng, 4) + "<|Assistant|>"
        assert seg._encode_segments(p) == tok.encode(p)
        assert seg._prefix_tokens <= 20_000
        assert seg._prefix_tokens == sum(e.cost for e in seg._prefixes.values())


def test_concurrent_turns_stay_exact(tok):
    """Threads advancing conversations that share a start: lookups, stores
    and evictions interleave."""
    import threading

    seg = SegmentTokenizer.maybe_create(tok, 60_000)
    seg._verified = seg_mod._VERIFY_FIRST
    common = _long_conversation(random.Random(16), 2)
    errors = []

    def worker(i):
        rng = random.Random(100 + i)
        p = common + "<|User|>" + _text(rng, 20)
        for _ in range(8):
            ask = p + "<|Assistant|>"
            if seg._encode_segments(ask) != tok.encode(ask):
                errors.append(i)
            p += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 15)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert seg._prefix_tokens == sum(e.cost for e in seg._prefixes.values())


def test_verification_window_is_bounded():
    assert 0 < seg_mod._VERIFY_FIRST <= 16
