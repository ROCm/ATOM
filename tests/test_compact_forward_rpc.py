from __future__ import annotations

import array
import pickle
from types import SimpleNamespace

import numpy as np
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.async_proc import (
        _BlockTableDelta,
        _BlockTableDeltaDecoder,
        _BlockTableDeltaEncoder,
    )
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams


def _batch(req_ids, rows):
    return SimpleNamespace(req_ids=req_ids, block_tables=rows, marker="kept")


def _rows(batch):
    return [list(row) for row in batch.block_tables]


def test_block_table_rpc_sends_full_then_only_appended_ids():
    first_rows = [array.array("i", [1, 2, 3]), array.array("i", [7, 8])]
    batch = _batch([11, 22], first_rows)
    encoder = _BlockTableDeltaEncoder()
    decoder = _BlockTableDeltaDecoder()

    first_wire = encoder.encode(batch)
    assert isinstance(first_wire.block_tables, _BlockTableDelta)
    assert first_wire is not batch
    assert _rows(decoder.decode(pickle.loads(pickle.dumps(first_wire)))) == [
        [1, 2, 3],
        [7, 8],
    ]

    first_rows[0].append(4)
    second_wire = encoder.encode(batch)
    delta = second_wire.block_tables
    assert delta.base_lengths.tolist() == [3, 2]
    assert delta.tail_offsets.tolist() == [0, 1, 1]
    assert delta.tail_values.tolist() == [4]

    decoded = decoder.decode(pickle.loads(pickle.dumps(second_wire)))
    assert _rows(decoded) == [[1, 2, 3, 4], [7, 8]]
    assert decoded.marker == "kept"


def test_block_table_rpc_refreshes_replaced_or_truncated_rows():
    rows = [array.array("i", [1, 2, 3])]
    batch = _batch([11], rows)
    encoder = _BlockTableDeltaEncoder()
    decoder = _BlockTableDeltaDecoder()
    decoder.decode(pickle.loads(pickle.dumps(encoder.encode(batch))))

    rows[0] = array.array("i", [9, 10, 11])
    replaced = encoder.encode(batch)
    assert replaced.block_tables.base_lengths.tolist() == [0]
    assert _rows(decoder.decode(pickle.loads(pickle.dumps(replaced)))) == [[9, 10, 11]]

    del rows[0][1:]
    truncated = encoder.encode(batch)
    assert truncated.block_tables.base_lengths.tolist() == [0]
    assert _rows(decoder.decode(pickle.loads(pickle.dumps(truncated)))) == [[9]]


def test_block_table_rpc_evicts_requests_absent_from_a_generation():
    row = array.array("i", [1, 2, 3])
    encoder = _BlockTableDeltaEncoder()
    encoder.encode(_batch([11], [row]))
    encoder.encode(_batch([22], [array.array("i", [7])]))

    returned = encoder.encode(_batch([11], [row]))
    assert returned.block_tables.base_lengths.tolist() == [0]
    assert returned.block_tables.tail_values.tolist() == [1, 2, 3]


def test_block_table_rpc_leaves_unaligned_batches_unchanged():
    batch = _batch([11, 22], [array.array("i", [1])])
    encoder = _BlockTableDeltaEncoder()

    assert encoder.encode(batch) is batch


def test_steady_state_wire_is_smaller_than_repeating_full_tables():
    rows = [array.array("i", range(4096)) for _ in range(16)]
    batch = _batch(list(range(16)), rows)
    encoder = _BlockTableDeltaEncoder()
    encoder.encode(batch)
    compact = encoder.encode(batch)

    assert compact.block_tables.tail_values.dtype == np.int32
    assert len(pickle.dumps(compact)) < len(pickle.dumps(batch)) // 20


def test_scheduled_batch_publishes_rank_invariant_sampling_flags():
    greedy = Sequence(
        [1],
        block_size=16,
        sampling_params=SamplingParams(temperature=0, top_k=8, top_p=0.9),
        id=11,
    )
    sampled = Sequence(
        [2],
        block_size=16,
        sampling_params=SamplingParams(temperature=0.5, top_k=8, top_p=0.9),
        id=22,
        needs_independent_noise=True,
    )
    batch = ScheduledBatch(
        seqs={11: greedy, 22: sampled},
        num_scheduled_tokens=[1, 1],
        total_tokens_num=2,
        total_tokens_num_decode=2,
        total_seqs_num=2,
        total_seqs_num_decode=2,
    )

    assert batch.all_greedy is False
    assert batch.has_independent_noise is True
    assert batch.needs_top_k is True
    assert batch.needs_top_p is True
    assert batch.uniform_top_k is True
    assert batch.uniform_top_p is True
