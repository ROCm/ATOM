"""Tests for the random dataset the serving benchmark generates.

A synthetic tokenizer stands in for a real one so the properties under test --
length targeting, reproducibility, and the batching the fast path relies on --
are checked without downloading a model. It is deliberately *not* injective:
`decode` joins ids with spaces and `encode` splits on them, so a prompt whose
ids were built from a fixed range re-encodes to a different token count and the
round-trip loop has to correct it, which is the case that matters.
"""

import numpy as np
import pytest

from atom.benchmarks.benchmark_serving import sample_random_requests


class FakeTokenizer:
    """Ids <-> text with a token count that is stable but not id-for-id equal."""

    vocab_size = 1000

    def __init__(self, drop_every: int = 0, drop_rounds: int = 0):
        # For the first `drop_rounds` encode batches, every `drop_every`-th
        # token is dropped, so the caller has to pad and retry. Setting
        # drop_rounds high enough that it never stops models a tokenizer the
        # round-trip can never satisfy.
        self.drop_every = drop_every
        self.drop_rounds = drop_rounds
        self.batch_decode_calls = 0
        self.batch_encode_calls = 0
        self.single_calls = 0

    def decode(self, ids):
        self.single_calls += 1
        return " ".join(str(int(i)) for i in ids)

    def encode(self, text, add_special_tokens=False):
        self.single_calls += 1
        return self._encode_one(text)

    def batch_decode(self, id_lists):
        self.batch_decode_calls += 1
        return [" ".join(str(int(i)) for i in ids) for ids in id_lists]

    def __call__(self, texts, add_special_tokens=False):
        self.batch_encode_calls += 1
        return {"input_ids": [self._encode_one(t) for t in texts]}

    def _encode_one(self, text):
        ids = [int(tok) for tok in text.split()] if text else []
        if self.drop_every and self.batch_encode_calls <= self.drop_rounds:
            ids = [t for n, t in enumerate(ids) if (n + 1) % self.drop_every]
        return ids


def _sample(tokenizer, **overrides):
    kwargs = {
        "prefix_len": 0,
        "input_len": 64,
        "output_len": 8,
        "num_prompts": 20,
        "range_ratio": 1.0,
        "tokenizer": tokenizer,
        "seed": 0,
    }
    kwargs.update(overrides)
    return sample_random_requests(**kwargs)


def test_every_prompt_hits_its_target_token_length():
    requests = _sample(FakeTokenizer())

    assert len(requests) == 20
    assert all(prompt_len == 64 for _, prompt_len, _, _ in requests)


def test_prompts_are_padded_back_up_when_encode_loses_tokens():
    """The round-trip loop must correct a tokenizer that drops tokens."""
    requests = _sample(FakeTokenizer(drop_every=8, drop_rounds=1))

    assert all(prompt_len == 64 for _, prompt_len, _, _ in requests)


def test_length_is_reported_honestly_when_the_round_trip_never_converges():
    """Out of rounds, the reported length is the real one, not the target.

    A benchmark that reported the target it failed to reach would understate
    its own input length in every derived statistic.
    """
    tokenizer = FakeTokenizer(drop_every=8, drop_rounds=1000)

    requests = _sample(tokenizer, num_prompts=4)

    for prompt, prompt_len, _, _ in requests:
        assert prompt_len < 64
        assert prompt_len == len(tokenizer.encode(prompt))


def test_same_seed_reproduces_the_same_prompts():
    first = _sample(FakeTokenizer())
    second = _sample(FakeTokenizer())

    assert [r[0] for r in first] == [r[0] for r in second]


def test_different_seed_produces_different_prompts():
    first = _sample(FakeTokenizer(), seed=0)
    second = _sample(FakeTokenizer(), seed=1)

    assert [r[0] for r in first] != [r[0] for r in second]


def test_global_numpy_rng_does_not_change_the_dataset():
    """The dataset must not depend on what else drew from the global RNG.

    Sharing np.random makes a benchmark's prompts a function of unrelated code,
    so two runs of the same command can differ.
    """
    baseline = _sample(FakeTokenizer())

    np.random.seed(1234)
    np.random.rand(999)
    after = _sample(FakeTokenizer())

    assert [r[0] for r in after] == [r[0] for r in baseline]


def test_tokenizer_is_called_in_batches_not_per_prompt():
    """Per-prompt calls forfeit the fast tokenizer's internal batching."""
    tokenizer = FakeTokenizer()

    _sample(tokenizer, num_prompts=200)

    assert tokenizer.single_calls == 0
    # One decode for the initial ids, then a bounded number of fix-up rounds --
    # far fewer than the 200 prompts either way.
    assert tokenizer.batch_decode_calls <= 11
    assert tokenizer.batch_encode_calls <= 11


def test_range_ratio_bounds_the_sampled_lengths():
    requests = _sample(FakeTokenizer(), range_ratio=0.5, num_prompts=100)

    assert all(32 <= prompt_len <= 64 for _, prompt_len, _, _ in requests)
    assert all(4 <= output_len <= 8 for _, _, output_len, _ in requests)


def test_prefix_is_shared_by_every_prompt():
    requests = _sample(FakeTokenizer(), prefix_len=16, num_prompts=8)

    prefixes = {r[0].split()[:16][0] for r in requests}
    assert len(prefixes) == 1
    assert all(prompt_len == 16 + 64 for _, prompt_len, _, _ in requests)


@pytest.mark.parametrize("num_prompts", [1, 5000])
def test_batching_boundaries(num_prompts):
    """A dataset larger than one tokenizer batch must still be complete."""
    requests = _sample(FakeTokenizer(), num_prompts=num_prompts)

    assert len(requests) == num_prompts
    assert all(prompt_len == 64 for _, prompt_len, _, _ in requests)
