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


def test_verification_window_is_bounded():
    assert 0 < seg_mod._VERIFY_FIRST <= 16
