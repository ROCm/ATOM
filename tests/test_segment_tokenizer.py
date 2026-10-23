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

SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>", "<|end|>", "<think>", "</think>"]
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


def _stored(seg):
    """The stored prompts (the prefix level) currently in the LRU."""
    return [v for k, (v, _) in seg._lru._items.items() if isinstance(k, tuple)]


def _segments(seg):
    return [k for k in seg._lru._items if not isinstance(k, tuple)]


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
    seg = SegmentTokenizer.maybe_create(tok, 4096)
    rng = random.Random(2)
    for _ in range(20):
        prompt = _conversation(rng, 6)
        assert seg.encode(prompt) == tok.encode(prompt)
    assert seg._lru.bytes <= 4096


def test_a_mismatch_turns_the_cache_off(tok, monkeypatch):
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    prompt = _conversation(random.Random(3), 4)
    monkeypatch.setattr(seg, "_encode_cached", lambda p: [0])
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
    assert seg._stats[3] == 0 and len(seg._lru) == 0


def test_eviction_by_another_request_mid_encode_is_harmless(tok, monkeypatch):
    """Another request can evict this one's cached segments while it encodes
    its new ones outside the lock."""
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    rng = random.Random(6)
    prompt = _conversation(rng, 6)
    assert seg.encode(prompt) == tok.encode(prompt)
    longer = prompt + "<|User|>" + _text(rng, 30)
    real = seg._split.backend.encode_batch

    def evict_then_encode(texts, **kw):
        with seg._lock:
            seg._lru.clear()
        return real(texts, **kw)

    monkeypatch.setattr(
        seg._split, "backend", MagicMock(encode_batch=evict_then_encode)
    )
    seg._verified = seg_mod._VERIFY_FIRST  # unchecked, as in steady state
    assert seg._encode_cached(longer) == tok.encode(longer)


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
    assert seg._encode_cached(prompt) == tok.encode(prompt)


def test_segments_that_encode_to_nothing_are_not_cached(tok, monkeypatch):
    """They would take an entry each but no token budget, so eviction would
    never bound them."""
    seg = SegmentTokenizer.maybe_create(tok, 1_000_000)
    real = seg._split.backend.encode_batch

    def drop_whitespace(texts, **kw):
        return [
            MagicMock(ids=[]) if not t.strip() else e
            for t, e in zip(texts, real(texts, **kw))
        ]

    monkeypatch.setattr(seg._split, "backend", MagicMock(encode_batch=drop_whitespace))
    seg._verified = seg_mod._VERIFY_FIRST
    for n in range(1, 50):
        seg._encode_cached("<|User|>" + " " * n + "<|Assistant|>")
    assert not _segments(seg)


def _tokenizer_with(added, pre=None, dropout=None):
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
    if dropout is not None:
        backend.model.dropout = dropout
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
    assert seg._encode_cached(prompt) == tok.encode(prompt)


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


def test_bpe_dropout_is_refused(monkeypatch):
    """Dropout samples merges on every call, so sharing one segment's ids
    across occurrences changes the encoder even when the startup probes
    happen to agree; the description alone has to refuse it."""
    monkeypatch.setattr(seg_mod.AddedTokenSplitter, "_probes_match", lambda self: True)
    added = [AddedToken("<|User|>", special=True)]
    assert SegmentTokenizer.maybe_create(_tokenizer_with(added), 10**6) is not None
    for dropout in (0.5, 0.01):
        tok = _tokenizer_with(added, dropout=dropout)
        assert SegmentTokenizer.maybe_create(tok, 10**6) is None, dropout


def test_split_special_tokens_is_refused(monkeypatch):
    """In this mode the plain encoder spells special tokens out as text, while
    `parts` extracts them. The startup probes need not pick a special token
    (e.g. added order `aa`, `<x>`, `longword`), so they are stubbed to agree
    and only the mode check can refuse."""
    monkeypatch.setattr(seg_mod.AddedTokenSplitter, "_probes_match", lambda self: True)
    added = [
        AddedToken("aa", special=False),
        AddedToken("<x>", special=True),
        AddedToken("longword", special=False),
    ]
    assert SegmentTokenizer.maybe_create(_tokenizer_with(added), 10**6) is not None
    tok = _tokenizer_with(added)
    tok.split_special_tokens = True
    assert tok.encode("q <x> q") != _tokenizer_with(added).encode("q <x> q")
    assert SegmentTokenizer.maybe_create(tok, 10**6) is None


def test_a_result_finished_after_the_cache_turned_off_is_not_returned(tok, monkeypatch):
    """Another request turning the cache off while this one encodes: this
    one's unchecked ids must not be returned."""
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    seg._verified = seg_mod._VERIFY_FIRST
    prompt = _conversation(random.Random(9), 4)

    def wrong_and_meanwhile_disabled(p):
        seg.enabled = False
        return [0]

    monkeypatch.setattr(seg, "_encode_cached", wrong_and_meanwhile_disabled)
    assert seg.encode(prompt) == tok.encode(prompt)


def test_a_check_finishing_after_the_cache_turned_off_does_not_count(tok, monkeypatch):
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    seg._verified = seg_mod._VERIFY_FIRST - 1
    prompt = _conversation(random.Random(10), 4)
    real = seg._encode_cached

    def disabled_meanwhile(p):
        ids = real(p)
        seg.enabled = False
        return ids

    monkeypatch.setattr(seg, "_encode_cached", disabled_meanwhile)
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
        assert seg._encode_cached(ask) == tok.encode(ask)
        prompt += "<|Assistant|>" + _text(rng, 15) + "<|User|>" + _text(rng, 25)
    assert seg._stats[4] >= 5
    # One stored prefix per conversation: each turn replaced the one before.
    assert len(_stored(seg)) == 1


def test_divergence_mid_prefix_reuses_up_to_the_last_shared_added_token(tok):
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(12)
    base = _long_conversation(rng, 6)
    seg._encode_cached(base)
    edited = base[: len(base) - 30] + "rewritten tail<|User|>" + _text(rng, 20)
    hits = seg._stats[4]
    assert seg._encode_cached(edited) == tok.encode(edited)
    assert seg._stats[4] == hits + 1


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
            assert seg._encode_cached(ask) == tok.encode(ask)
        a += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
        b += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
    assert len(seg._buckets) == 1
    assert len(_stored(seg)) == 2


def _covered_of_previous_turn(seg, prompts):
    """Per later prompt: how many characters of the same conversation's
    previous prompt the prefix level did not reuse."""
    real = seg._find_prefix
    covered = []

    def find(p):
        base, k = real(p)
        covered.append(0 if base is None else int(base.cut_chars[k]))
        return base, k

    seg._find_prefix = find
    missed, prev = [], {}
    for c, p in prompts:
        covered.clear()
        assert seg._encode_cached(p) == seg._tokenizer.encode(p)
        if c in prev:
            missed.append(prev[c] - covered[0])
        prev[c] = len(p)
    return missed


def test_many_conversations_sharing_a_start_each_reuse_their_previous_turn(tok):
    """Concurrent agent conversations share the system prompt (and so a
    bucket); more of them than a handful must not push each other out."""
    seg = SegmentTokenizer.maybe_create(tok, 10**8)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(21)
    common = _long_conversation(rng, 0)
    convs = [common + "<|User|>" + _text(rng, 30) for _ in range(40)]
    prompts = []
    for _ in range(3):
        for c in range(len(convs)):
            prompts.append((c, convs[c] + "<|Assistant|>"))
            convs[c] += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
    missed = _covered_of_previous_turn(seg, prompts)
    assert max(missed) == 0, max(missed)
    assert len(_stored(seg)) == len(convs)


def test_a_rewritten_generation_prompt_replaces_the_stored_turn(tok):
    """DeepSeek-V4.1 thinking mode ends the prompt in `<think>` and renders
    the answered turn with `</think>`: the last cut is never shared, and the
    conversation must still keep a single stored prompt."""
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(22)
    history = _long_conversation(rng, 2)
    prompts = []
    for _ in range(6):
        prompts.append((0, history + "<|Assistant|><think>"))
        history += (
            "<|Assistant|></think>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
        )
    missed = _covered_of_previous_turn(seg, prompts)
    assert max(missed) == len("<think>"), missed
    assert len(_stored(seg)) == 1


def test_a_fork_keeps_the_stored_prompt_it_left(tok):
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(23)
    base = _long_conversation(rng, 4)
    long_tail = "<|User|>" + _text(rng, 2000) + "<|Assistant|>"
    seg._encode_cached(base + long_tail)
    fork = base + "<|User|>" + _text(rng, 20) + "<|Assistant|>"
    assert seg._encode_cached(fork) == tok.encode(fork)
    assert len(_stored(seg)) == 2


def test_a_fork_whose_unshared_text_is_one_token_is_kept(tok):
    """`cut_ids` counts every id up to each cut, so a gap holding one text
    token plus its added token is 2 ids, not 1: a fork whose unshared tail
    is a single-token message is not taken for a rewritten generation
    prompt."""
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(29)
    base = _long_conversation(rng, 4)
    one = next(w for w in WORDS if len(tok.encode(w, add_special_tokens=False)) == 1)
    first = base + "<|User|>" + one + "<|Assistant|>"
    seg._encode_cached(first)
    fork = base + "<|User|>" + _text(rng, 20) + "<|Assistant|>"
    assert seg._encode_cached(fork) == tok.encode(fork)
    assert len(_stored(seg)) == 2


def test_a_divergence_with_no_new_added_token_stores_nothing(tok):
    """Edited inside the last message, before the stored prompt's last added
    token, with no added token after the edit: the new entry would be a part
    of the stored one."""
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(24)
    head = _long_conversation(rng, 6) + "<|User|>"
    msg = _text(rng, 40)
    seg._encode_cached(head + msg + "<|Assistant|>" + _text(rng, 30))
    edited = head + msg[:-10] + " rewritten without markers"
    assert seg._encode_cached(edited) == tok.encode(edited)
    assert len(_stored(seg)) == 1


def test_looking_up_a_prefix_refreshes_only_the_one_used(tok):
    """Every candidate in the bucket is compared; only the one reused may
    become recent, or stale entries outlive useful segments."""
    import numpy as np

    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(25)
    common = _long_conversation(rng, 0)
    a = common + "<|User|>" + _text(rng, 30) + "<|Assistant|>"
    b = common + "<|User|>" + _text(rng, 30) + "<|Assistant|>"
    seg._encode_cached(a)
    seg._encode_cached(b)
    seg._lru.put("marker", np.zeros(1, np.int32), 4)
    base, _ = seg._find_prefix(b + "x")
    assert base is not None and base.text == b
    keys = list(seg._lru._items)
    a_key = next(k for k in keys if isinstance(k, tuple) and seg._lru.peek(k).text == a)
    assert keys.index(a_key) < keys.index("marker")


def test_turning_the_cache_off_releases_it(tok, monkeypatch):
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    rng = random.Random(26)
    seg._verified = seg_mod._VERIFY_FIRST
    seg._encode_cached(_long_conversation(rng, 4))
    assert len(seg._lru)
    seg._verified = 0
    monkeypatch.setattr(seg, "_encode_cached", lambda p: [0])
    p = _conversation(rng, 4)
    assert seg.encode(p) == tok.encode(p)
    assert not seg.enabled and not len(seg._lru) and seg._lru.bytes == 0
    assert not seg._buckets


def test_a_request_finishing_after_the_cache_turned_off_stores_nothing(tok):
    """Another request turns the cache off (and clears it) while this one
    encodes its missing segments; this one must not refill it."""
    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    real_batch = seg._split.backend.encode_batch

    def disabled_meanwhile(*a, **kw):
        out = real_batch(*a, **kw)
        with seg._lock:
            seg.enabled = False
            seg._lru.clear()
            seg._buckets.clear()
        return out

    seg._split.backend = MagicMock(encode_batch=disabled_meanwhile)
    p = _long_conversation(random.Random(28), 4) + "<|Assistant|>"
    assert seg._encode_cached(p) == tok.encode(p)
    assert not len(seg._lru) and seg._lru.bytes == 0 and not seg._buckets


def test_trie_pattern_splits_like_a_flat_alternation():
    import re

    from atom.model_engine.segment_split import _alternation

    def flat(tokens):
        tokens = sorted(tokens, key=len, reverse=True)
        return re.compile("(" + "|".join(re.escape(t) for t in tokens) + ")")

    rng = random.Random(27)
    alphabet = "ab<|>x."
    for _ in range(2000):
        tokens = {
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
            for _ in range(rng.randint(1, 12))
        }
        text = "".join(rng.choice(alphabet + "y") for _ in range(rng.randint(0, 60)))
        assert _alternation(tokens).split(text) == flat(tokens).split(text)
    long_tok = "x" * 50_000 + "<|raw|>"  # a long chain stays one literal
    assert _alternation([long_tok, "<|raw|>"]).split("a" + long_tok)[1] == long_tok


def _retained_by(fill):
    """Bytes still allocated after `fill()` returns, by tracemalloc."""
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        keep = fill()
        gc.collect()
        return tracemalloc.get_traced_memory()[0] - before, keep
    finally:
        tracemalloc.stop()


def test_many_small_segments_stay_within_the_budget(tok):
    """Per-entry overhead dominates single-token segments; the budget counts
    retained bytes, not token ids."""
    budget = 1 << 20

    def fill():
        seg = SegmentTokenizer.maybe_create(tok, budget)
        seg._verified = seg_mod._VERIFY_FIRST
        for b in range(30):
            p = "".join(f"<|User|>{b}_{i}" for i in range(1000))
            assert seg._encode_cached(p) == tok.encode(p)
        return seg

    held, seg = _retained_by(fill)
    assert seg._lru.bytes <= budget
    # The cache, the tokenizer it wraps and allocator slack: within 1.5x.
    assert held <= 1.5 * budget, held


def test_stored_prefixes_stay_within_the_budget(tok):
    budget = 200_000

    def fill():
        seg = SegmentTokenizer.maybe_create(tok, budget)
        seg._verified = seg_mod._VERIFY_FIRST
        rng = random.Random(15)
        for _ in range(12):
            p = _long_conversation(rng, 4) + "<|Assistant|>"
            assert seg._encode_cached(p) == tok.encode(p)
            assert seg._lru.bytes <= budget
        return seg

    held, seg = _retained_by(fill)
    assert _stored(seg)
    assert held <= 1.5 * budget, held


def test_concurrent_turns_stay_exact(tok):
    """Threads advancing conversations that share a start: lookups, stores
    and evictions interleave. A worker's exception fails the test."""
    from concurrent.futures import ThreadPoolExecutor

    seg = SegmentTokenizer.maybe_create(tok, 400_000)
    seg._verified = seg_mod._VERIFY_FIRST
    common = _long_conversation(random.Random(16), 2)

    def worker(i):
        rng = random.Random(100 + i)
        p = common + "<|User|>" + _text(rng, 20)
        for _ in range(8):
            ask = p + "<|Assistant|>"
            assert seg._encode_cached(ask) == tok.encode(ask), i
            p += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 15)
        return i

    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(worker, i) for i in range(6)]
        assert sorted(f.result(timeout=120) for f in futures) == list(range(6))
    assert seg._lru.bytes <= 400_000
    assert seg._lru.bytes == sum(c for _, c in seg._lru._items.values())


def test_added_token_heavy_prefixes_stay_within_the_budget(tok):
    """Many added tokens mean many retained cut positions."""
    budget = 400_000

    def fill():
        seg = SegmentTokenizer.maybe_create(tok, budget)
        seg._verified = seg_mod._VERIFY_FIRST
        rng = random.Random(17)
        for _ in range(6):
            p = "<|begin|>" + _text(rng, 3000) + "<|User|>x" * 3000 + "<|Assistant|>"
            assert seg._encode_cached(p) == tok.encode(p)
        return seg

    held, seg = _retained_by(fill)
    assert seg._lru.bytes <= budget
    assert held <= 1.5 * budget, held


def test_prefix_reuse_is_off_when_a_raw_token_overlaps_a_normalized_ones_end():
    """Normalized `ab`, non-normalized `bc`: a prompt cut after `ab` and then
    extended by `c` extracts `bc` from the raw text first."""
    tok = _tokenizer_with(
        [
            AddedToken("<|User|>", special=True, normalized=False),
            AddedToken("ab", normalized=True),
            AddedToken("bc", special=True, normalized=False),
        ]
    )
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    assert seg is not None and not seg._prefix_ok
    # The segment level alone stays exact past the verification window.
    seg._verified = seg_mod._VERIFY_FIRST
    rng = random.Random(18)
    p = "<|User|>" + _text(rng, 3000) + "ab"
    assert seg._encode_cached(p) == tok.encode(p)
    assert seg._encode_cached(p + "c") == tok.encode(p + "c")


def test_a_prompt_spelling_a_whole_added_token_is_still_split():
    """The raw pass must extract a non-normalized token from a prompt that
    happens to equal a longer normalized token's spelling."""
    long_tok = "x" * 200 + "<|raw|>"
    tok = _tokenizer_with(
        [
            AddedToken("<|raw|>", special=True, normalized=False),
            AddedToken(long_tok, normalized=True),
        ]
    )
    seg = SegmentTokenizer.maybe_create(tok, 10**6)
    assert seg is not None
    seg._verified = seg_mod._VERIFY_FIRST
    for prompt in (long_tok, "y" + long_tok, long_tok + "z"):
        assert seg._encode_cached(prompt) == tok.encode(prompt), prompt[-20:]


def test_byte_lru_charges_overhead_and_evicts_oldest():
    from atom.model_engine.byte_lru import ENTRY_OVERHEAD, ByteLRU

    evicted = []
    lru = ByteLRU(3 * (ENTRY_OVERHEAD + 10), on_evict=lambda k, v: evicted.append(k))
    for k in "abc":
        lru.put(k, k, 10)
    assert lru.peek("b") == "b"  # peeking leaves b the least recent
    assert lru.get("a") == "a"  # a is now the most recent
    lru.put("d", "d", 10)
    assert evicted == ["b"] and "b" not in lru
    assert lru.bytes == 3 * (ENTRY_OVERHEAD + 10)
    lru.put("huge", "x", 10**9)  # larger than the budget: not kept
    assert "huge" not in lru and lru.bytes <= lru.budget
    assert lru.pop("a") == "a" and evicted == ["b"]  # pop does not notify


def test_shared_chars_finds_the_first_difference():
    from atom.model_engine.segment_tokenizer import _shared_chars

    rng = random.Random(19)
    a = "".join(rng.choice("ab") for _ in range(300_000))
    for n in (0, 1, 65_535, 65_536, 65_537, 200_001, 299_999):
        b = a[:n] + ("x" if n < len(a) else "") + a[n + 1 :]
        assert _shared_chars(a, b, len(a)) == n
    assert _shared_chars(a, a, len(a)) == len(a)
    assert _shared_chars(a, a, 1234) == 1234


def test_hashing_encoding_and_assembly_run_outside_the_lock(tok, monkeypatch):
    """The lock guards lookups and bookkeeping only; work that grows with
    the prompt must not hold it."""
    import hashlib

    import numpy as np

    seg = SegmentTokenizer.maybe_create(tok, 10**7)
    seg._verified = seg_mod._VERIFY_FIRST
    real_hash, real_concat = hashlib.blake2b, np.concatenate
    real_batch = seg._split.backend.encode_batch
    held = []

    def watch(fn):
        def wrapped(*a, **kw):
            held.append(seg._lock.locked())
            return fn(*a, **kw)

        return wrapped

    monkeypatch.setattr(seg_mod.hashlib, "blake2b", watch(real_hash))
    monkeypatch.setattr(seg_mod.np, "concatenate", watch(real_concat))
    monkeypatch.setattr(
        seg._split, "backend", MagicMock(encode_batch=watch(real_batch))
    )
    rng = random.Random(20)
    prompt = _long_conversation(rng, 6)
    for _ in range(3):
        ask = prompt + "<|Assistant|>"
        assert seg._encode_cached(ask) == tok.encode(ask)
        prompt += "<|Assistant|>" + _text(rng, 10) + "<|User|>" + _text(rng, 20)
    assert held and not any(held)


def test_verification_window_is_bounded():
    assert 0 < seg_mod._VERIFY_FIRST <= 16
