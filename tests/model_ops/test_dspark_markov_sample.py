# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``dspark_markov_argmax``: agreement with ``torch.argmax``, NaN included.

The id this op returns is fed straight back into a ``[V, r]`` embedding gather
for the next block position, so an out-of-range id is not a wrong answer -- it
is an illegal access on a device queue, surfacing much later as an opaque
launch failure. The reduce is written around ``==``, which no NaN lane
satisfies, so a NaN row used to leave the ``vocab_size`` tie-break sentinel
standing and return it as the id. These tests pin both halves of the contract:
the ids match ``torch.argmax`` (NaN ordering and all), and they stay inside
``[0, V)`` no matter what the logits carry.
"""

import pytest

pytest.importorskip(
    "atom.model_ops.dspark_markov_sample",
    reason="the op imports aiter at module load",
    exc_type=ImportError,
)

import torch

from atom.model_ops.dspark_markov_sample import (
    _torch_dspark_markov_argmax,
    dspark_markov_argmax,
)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel needs a GPU"
)

# Small enough to run in a unit test, still wider than one ``_BLOCK_V`` tile so
# the cross-tile stage 2 reduce is exercised rather than short-circuited.
VOCAB = 1024
RANK = 64


def _inputs(num_rows, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(
        num_rows, VOCAB, dtype=torch.float32, device="cuda", generator=gen
    )
    embed = torch.randn(num_rows, RANK, dtype=dtype, device="cuda", generator=gen)
    w2 = torch.randn(VOCAB, RANK, dtype=dtype, device="cuda", generator=gen)
    return base, embed, w2


@needs_gpu
@pytest.mark.parametrize("num_rows", [1, 7, 8, 16, 17, 64, 65])
def test_matches_torch_argmax(num_rows):
    base, embed, w2 = _inputs(num_rows)
    got = dspark_markov_argmax(base, embed, w2)
    assert torch.equal(got, _torch_dspark_markov_argmax(base, embed, w2))


@needs_gpu
def test_lowest_index_wins_ties():
    base, embed, w2 = _inputs(4)
    # Zero the bias so the tie is exactly the one the base logits describe, and
    # place the joint max in two tiles at once to make stage 2 do the choosing.
    embed.zero_()
    base.fill_(0.0)
    base[:, 100] = 5.0
    base[:, 900] = 5.0
    got = dspark_markov_argmax(base, embed, w2)
    assert torch.equal(got, torch.full((4,), 100, dtype=torch.int64, device="cuda"))


@needs_gpu
@pytest.mark.parametrize("source", ["base", "embed", "w2"])
def test_nan_row_stays_in_range_and_matches_torch(source):
    """A NaN anywhere in the inputs must not push the id past the table."""
    base, embed, w2 = _inputs(8, seed=1)
    if source == "base":
        base[3, 512] = float("nan")
    elif source == "embed":
        # Poisons every column of row 5, i.e. an all-NaN row.
        embed[5, 0] = float("nan")
    else:
        w2[777, :] = float("nan")

    got = dspark_markov_argmax(base, embed, w2)
    assert int(got.min()) >= 0
    assert int(got.max()) < VOCAB
    assert torch.equal(got, _torch_dspark_markov_argmax(base, embed, w2))


@needs_gpu
def test_all_nan_row_returns_zero_like_torch():
    base, embed, w2 = _inputs(4, seed=2)
    base[2].fill_(float("nan"))
    got = dspark_markov_argmax(base, embed, w2)
    assert int(got[2]) == 0
    assert torch.equal(got, _torch_dspark_markov_argmax(base, embed, w2))


@needs_gpu
def test_negative_infinity_row_stays_in_range():
    """An all -inf row shares the reduce's identity value; ids must still be valid."""
    base, embed, w2 = _inputs(4, seed=3)
    embed.zero_()
    base[1].fill_(float("-inf"))
    got = dspark_markov_argmax(base, embed, w2)
    assert int(got.min()) >= 0
    assert int(got.max()) < VOCAB
    assert torch.equal(got, _torch_dspark_markov_argmax(base, embed, w2))
