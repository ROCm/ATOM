# SPDX-License-Identifier: MIT

"""`Sequence.token_ids` is an `array("i")`, and what that costs its consumers.

The storage change pays twice: the scheduler's per-step copy into
`scheduled_tokens` becomes a memcpy instead of one CPython int unboxing per
token, and a 100k-token prompt costs 0.38 MiB instead of 3.4.

What it risks is narrower and entirely one shape: an `array("i")` never
compares equal to a list. Three places compare token ids, and in every one a
mismatch is silent -- a stop sequence that stops nothing, a prefix cache that
never hits, a hash that changes with its argument's Python type. Each is
pinned here with a positive control, because a test that only checked the
happy path would pass just as well against the versions that were broken.
"""

from __future__ import annotations

import array

import numpy as np
import pytest

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence, new_token_ids
from atom.sampling_params import SamplingParams


def _seq(token_ids, **kw):
    return Sequence(
        list(token_ids), block_size=4, sampling_params=SamplingParams(), **kw
    )


def test_storage_is_int32_and_holds_what_a_list_held():
    seq = _seq([0, 1, 129279, -1])
    assert isinstance(seq.token_ids, array.array)
    assert seq.token_ids.typecode == "i"
    # -1 is the exit sentinel and 129279 the top of this vocab; both must round
    # trip, which rules out an unsigned typecode.
    assert list(seq.token_ids) == [0, 1, 129279, -1]


def test_the_operations_a_sequence_performs_on_it():
    seq = _seq([1, 2, 3])
    seq.append_token(4)
    assert seq[-1] == 4 and len(seq) == 4
    del seq.token_ids[-1:]
    seq.token_ids[0] = 9
    assert list(seq.token_ids) == [9, 2, 3]
    assert list(seq.block(0)) == [9, 2, 3]


# --- the three comparisons, each with its control ------------------------


def test_stop_sequences_are_stored_in_the_same_type_they_are_compared_against():
    """`Scheduler._check_stop` does `seq.token_ids[a:b] == stop_seq`.

    Left as lists, that comparison is False for every input and generation
    never stops on a stop sequence -- with no error anywhere.
    """
    seq = _seq([1, 2, 3, 4], stop_token_sequences=[[3, 4], [9]])
    for stored in seq.stop_token_sequences:
        assert isinstance(stored, array.array), "a list here never matches"
    assert seq.token_ids[2:4] == seq.stop_token_sequences[0]

    # Control: the shape of the bug this guards.
    assert seq.token_ids[2:4] != [3, 4]


def test_compute_hash_does_not_depend_on_its_argument_type():
    """`np.array` infers int64 from a list and int32 from an `array("i")`.

    Unpinned, the digest changed with the caller's Python type, so two paths
    hashing the same tokens would miss each other in the prefix cache.
    """
    ids = [11, 22, 33, 44]
    assert BlockManager.compute_hash(ids) == BlockManager.compute_hash(
        new_token_ids(ids)
    )
    assert BlockManager.compute_hash(ids, 7) == BlockManager.compute_hash(
        new_token_ids(ids), 7
    )

    # Control: unpinned is what differed, and int64 is the value lists gave --
    # so pinning it leaves every hash recorded before this change where it was.
    unpinned = {np.array(ids).tobytes(), np.array(new_token_ids(ids)).tobytes()}
    assert len(unpinned) == 2, "the dtype inference this pins is no longer a hazard"
    assert np.asarray(ids, dtype=np.int64).tobytes() == np.array(ids).tobytes()


def test_a_block_and_the_slice_it_is_compared_against_share_a_type():
    """`BlockManager` publishes a slice of `seq.token_ids` into a block, then
    compares a fresh slice against it to confirm a cache hit. Two types there
    means the hit is rejected and the prefix is recomputed, silently."""
    seq = _seq(range(8))
    published = seq.token_ids[0:4]
    fresh = seq.token_ids[0:4]
    assert published == fresh
    assert published != list(fresh)  # control: the mismatch is real


# --- what the change is for ----------------------------------------------


@pytest.mark.parametrize("n", [8, 2048])
def test_a_chunk_lands_in_a_numpy_buffer_without_unboxing(n):
    """The scheduler's hot copy. Correctness here; the speed is in
    `/app/logs_claude/o10_scheduled_tokens_marshal.py`."""
    seq = _seq(range(n))
    dst = np.empty(n, dtype=np.int32)
    dst[:] = seq.token_ids[:n]
    assert np.array_equal(dst, np.arange(n, dtype=np.int32))
