# SPDX-License-Identifier: MIT
"""The bound that lets the Engram hash leave numpy.

`NgramHashMapping.hash_layer` reduces its rolling accumulator with numpy's `%`,
which is Python's: the result is non-negative whatever the sign of the dividend.
A GPU takes the sign of the dividend instead. The two agree only while the
accumulator stays non-negative, so a device port is correct only if that holds
-- and it holds by construction, not by accident:

    rolling starts at 0 and is only ever XORed with `token * multiplier`
    token is a compressed id in [0, tokenizer_vocab_size)
    multiplier <= 2 * (int64.max // tokenizer_vocab_size // 2) + 1
    => every term is in [0, 2**63)
    => the XOR of non-negative int64s is non-negative

The margin is 22771 out of 9.2e18. This file is the gate on that margin: it
needs neither triton nor a GPU, so it runs where the kernel's own parity test
cannot, and a change to `half_bound`, to the tokenizer, or to `pad_id` that
would silently give the two implementations different rows fails here first.
"""

import numpy as np
import pytest

from atom.model_ops.engram.mapping import (
    EngramConfig,
    NgramHashMapping,
    _next_prime,
)

# deepseek-ai/DeepSeek-V4.1-Flash config.json -> text_config, engram block.
V41_FLASH = {
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
}


class IdentityTokenizer:
    def __init__(self, vocab_size):
        self.lookup_table = np.arange(vocab_size, dtype=np.int64)

    def __len__(self):
        return len(self.lookup_table)

    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        out = arr.copy()
        valid = arr >= 0
        out[valid] = self.lookup_table[arr[valid]]
        return out


def tiny_config():
    base = {
        "layer_ids": (0, 2),
        "num_embeddings": (0, 0),
        "max_ngram_size": 3,
        "vocab_size": 1024,
        "n_heads": 2,
        "head_dim": 8,
        "pad_token_id": 0,
        "compressed_vocab_size": 512,
    }
    config = EngramConfig(**base)
    seen, totals = set(), []
    for _ in config.layer_ids:
        total = 0
        for _ in config.ngram_orders:
            start = config.vocab_size - 1
            for _ in range(config.n_heads):
                prime = _next_prime(start, seen)
                seen.add(prime)
                total += prime
                start = prime
        totals.append(total)
    return EngramConfig(**{**base, "num_embeddings": tuple(totals)})


def build(config):
    return NgramHashMapping(config, IdentityTokenizer(config.compressed_vocab_size))


@pytest.fixture(params=["v41-flash", "tiny"])
def mapping(request):
    config = (
        EngramConfig.from_hf(V41_FLASH)
        if request.param == "v41-flash"
        else tiny_config()
    )
    return build(config)


def test_every_hash_term_fits_a_positive_int64(mapping):
    """`token * multiplier` must not reach the sign bit, for any legal token."""
    limit = int(np.iinfo(np.int64).max)
    largest = mapping.tokenizer_vocab_size - 1
    assert 0 <= mapping.pad_id <= largest, "PAD substitutes for a token; it must be one"
    for layer_id, multipliers in mapping.layer_multipliers.items():
        assert (multipliers > 0).all(), layer_id
        assert (multipliers % 2 == 1).all(), "odd multipliers keep the low bit live"
        # Python ints, so the product is exact rather than already wrapped.
        assert largest * int(multipliers.max()) <= limit, layer_id


def test_rolling_accumulator_stays_non_negative(mapping):
    """The consequence, computed in exact Python ints over random lookbacks.

    The bound above is the argument; this is the thing the argument is about.
    Without it a future change could keep every product in range and still let
    the accumulator go negative by some route the bound does not cover.
    """
    rng = np.random.default_rng(0)
    width = mapping.config.max_ngram_size - 1
    for layer_id, multipliers in mapping.layer_multipliers.items():
        tokens = rng.integers(
            0, mapping.tokenizer_vocab_size, size=(64, width + 1), dtype=np.int64
        )
        tokens[0] = mapping.tokenizer_vocab_size - 1  # the extreme term
        tokens[1] = mapping.pad_id
        for row in tokens:
            rolling = 0
            for shift, value in enumerate(row):
                rolling ^= int(value) * int(multipliers[shift])
                assert rolling >= 0, (layer_id, shift)


def test_head_ids_land_inside_their_own_table(mapping):
    """A row index must stay within the head's slice of the layer's table."""
    rng = np.random.default_rng(1)
    width = mapping.config.max_ngram_size - 1
    for index, layer_id in enumerate(mapping.config.layer_ids):
        tokens = rng.integers(
            0, mapping.tokenizer_vocab_size, size=(1, 7), dtype=np.int64
        )
        history = rng.integers(
            -1, mapping.tokenizer_vocab_size, size=(1, width), dtype=np.int64
        )
        rows = mapping.to_row_indices(
            mapping.hash_layer(tokens, layer_id, history=history), layer_id
        )
        sizes = mapping.head_vocab_sizes[layer_id]
        offsets = mapping.head_offsets[layer_id]
        assert (rows >= offsets).all() and (rows < offsets + sizes).all()
        assert rows.max() < mapping.config.num_embeddings[index]
