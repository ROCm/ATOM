# SPDX-License-Identifier: MIT

"""`Sequence.token_ids` is an `array("i")`, and what that costs its consumers.

The storage change pays twice: the scheduler's per-step copy into
`scheduled_tokens` becomes a memcpy instead of one CPython int unboxing per
token, and a 100k-token prompt costs 0.38 MiB instead of 3.4.

What it risks is one axis, not one bug: every consumer that was written
against a list. Two kinds, and a silent failure in both --

  compared against a list   an `array("i")` never equals one, so the answer is
                            always "different": a stop sequence that stops
                            nothing, a prefix cache that never hits, a hash
                            that changes with its argument's Python type.

  serialized as a list      msgspec has no encoding for an array, and the KV
                            event publisher counts encode failures instead of
                            raising -- the whole event stream goes dark.

Each is pinned here with a positive control, because a test that only checked
the happy path would pass just as well against the versions that were broken.
The two boundaries where the type is load-bearing assert it rather than accept
either, so a consumer drifting back to a list fails loudly at the boundary
instead of quietly downstream.
"""

from __future__ import annotations

import array
import gc
import json
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
from conftest import MockConfig

from atom.distributed.kv_events import BlockStored
from atom.entrypoints.openai.serving_chat import build_chat_response
from atom.entrypoints.openai.serving_completion import build_completion_response
from atom.model_engine.block_manager import BlockManager, _make_block_stored
from atom.model_engine.kv_block import Block
from atom.model_engine.sequence import Sequence, new_token_ids
from atom.sampling_params import SamplingParams


def _seq(token_ids, **kw):
    return Sequence(
        list(token_ids), block_size=4, sampling_params=SamplingParams(), **kw
    )


def test_every_per_token_field_on_a_sequence_is_an_array():
    """Named one by one rather than checked as a class, because the point is
    the list of fields: anything growing once per token belongs here, and a
    new one added as a list is the regression this catches."""
    seq = _seq([1, 2, 3])
    seq.append_token(4)
    fields = {
        "token_ids": "i",
        "output_tokens": "i",  # the completion half of token_ids
        "block_table": "i",
        "logprobs": "d",  # floats, same boxing cost per token
    }
    for name, typecode in fields.items():
        held = getattr(seq, name)
        assert isinstance(held, array.array), f"{name} is a {type(held).__name__}"
        assert held.typecode == typecode, name
    for stop in _seq([1], stop_token_sequences=[[1]]).stop_token_sequences:
        assert isinstance(stop, array.array)


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


# --- the two boundaries that refuse the wrong type -----------------------


def test_a_block_refuses_to_store_a_list():
    """`Block.token_ids` is what a hash hit is verified against, and the pool
    holds one per block -- 94602 of them on a V4-Flash-DSpark tp1. A list costs
    twice: the verify always reads as a collision, and the collector walks one
    slot per token on every gen-2 pass, 207.7ms of stop-the-world against 5.6
    at that pool size (`/app/logs_claude/o27_branch_gc_heap.py`)."""
    block = Block(0)
    block.update(7, new_token_ids([1, 2, 3]))
    assert block.token_ids == new_token_ids([1, 2, 3])

    with pytest.raises(AssertionError, match="must be an array"):
        block.update(7, [1, 2, 3])

    # Control: the two costs, neither of which announces itself. The second is
    # `gc.get_referents`, the traversal a collection performs -- a list is one
    # visit per token, an array is a fixed visit however many it holds.
    assert block.token_ids != [1, 2, 3]
    assert [len(gc.get_referents(list(range(n)))) for n in (3, 300)] == [3, 300]
    assert len({len(gc.get_referents(new_token_ids(range(n)))) for n in (3, 300)}) == 1


def test_a_kv_event_refuses_to_carry_an_array():
    """`BlockStored` is msgpack-encoded, and `KVEventPublisher.publish` counts
    encode errors rather than raising -- so an array reaches the wire as
    nothing at all."""
    tokens = [1, 2, 3]
    event = _make_block_stored([100], tokens, None, block_size=4)
    assert msgspec.msgpack.Encoder().encode(event)

    with pytest.raises(AssertionError, match="must be a list"):
        _make_block_stored([100], new_token_ids(tokens), None, block_size=4)

    # Control: the encoder is what the assert is standing in front of.
    with pytest.raises(TypeError, match="array"):
        msgspec.msgpack.Encoder().encode(
            BlockStored(
                block_hashes=[100],
                parent_block_hash=None,
                token_ids=new_token_ids(tokens),
                block_size=4,
            )
        )


def test_every_event_a_publish_path_emits_reaches_the_wire(seq_factory):
    """The boundary assert only fires on a path a test actually walks.

    `publish_loaded_prefix` is the one BlockStored site fed straight from
    `_hash_block_tokens`, and it emits nothing unless KV events are on -- so
    the suite drove it for its return value and never looked at what it put in
    the log. Encoding the whole batch is what makes the type load-bearing.
    """
    cfg = MockConfig(
        num_kvcache_blocks=16,
        kv_cache_block_size=4,
        enable_prefix_caching=True,
        kv_events_config=SimpleNamespace(enable=True),
    )
    bm = BlockManager(cfg)
    assert bm._event_log is not None, "events did not turn on; the rest is vacuous"

    # Loaded first and on tokens nothing has indexed yet: with a canonical
    # mapping already in place this takes the branch that reuses it and emits
    # nothing at all, which is how the path stayed uncovered.
    loaded = seq_factory(list(range(16)))
    bm.allocate(loaded)
    assert bm.publish_loaded_prefix(loaded, start_token=0, end_token=16) == 16

    hashed = seq_factory(list(range(100, 116)))
    bm.allocate(hashed)
    bm.hash_blocks(hashed, 16, start_tokens=0)

    stored = [e for e in bm._event_log if isinstance(e, BlockStored)]
    assert len(stored) >= 2, "a publish path emitted nothing; nothing was checked"
    encoder = msgspec.msgpack.Encoder()
    for event in bm._event_log:
        assert encoder.encode(event)


def test_a_response_builder_never_puts_token_ids_on_the_wire():
    """Why the API server's accumulators need no conversion back.

    `generate_async` hands its dict to `build_*_response`, which reads `text`,
    the finish reason and the counters. `token_ids` is not among them, so the
    array never meets a serializer -- and the accumulator can stay an array
    for the whole request instead of holding one boxed PyInt per token.

    Passing an array in and serializing the result is what pins that: a
    builder that starts forwarding the key fails here rather than in
    production, where it would be a 500 on a response that used to work.
    """
    final_output = {
        "text": "hi",
        "token_ids": new_token_ids([1, 2, 3]),
        "finish_reason": "stop",
        "num_tokens_input": 4,
        "num_tokens_output": 3,
    }
    chat = build_chat_response("id", "m", final_output["text"], dict(final_output))
    completion = build_completion_response("id", "m", dict(final_output))
    for response in (chat, completion):
        assert json.loads(response.model_dump_json())["choices"][0]

    # Control: the failure the assertion above stands in for.
    with pytest.raises(TypeError, match="array"):
        json.dumps(final_output)


# --- what the change is for ----------------------------------------------


@pytest.mark.parametrize("n", [8, 2048])
def test_a_chunk_lands_in_a_numpy_buffer_without_unboxing(n):
    """The scheduler's hot copy. Correctness here; the speed is in
    `/app/logs_claude/o10_scheduled_tokens_marshal.py`."""
    seq = _seq(range(n))
    dst = np.empty(n, dtype=np.int32)
    dst[:] = seq.token_ids[:n]
    assert np.array_equal(dst, np.arange(n, dtype=np.int32))
