# SPDX-License-Identifier: MIT

"""The two per-sequence marshals in `ScheduledBatch.__init__`.

Both used to write one numpy slice per sequence -- the flat token window, and
the dense draft-token rows. A numpy slice-assign costs ~245ns of dispatch
whatever its length, and at decode both rows are a handful of ints, so the
loops were dispatch and nothing else. They now stage the whole batch and
assign once.

Each is checked against a golden that is the per-row form it replaced, so the
tests say "same answer" rather than restating the new code. What the batched
forms can get wrong that a per-row form could not is what the rest covers: a
short run has no row edge to raise on, and a concatenate takes its dtype from
its inputs rather than from the destination.
"""

from __future__ import annotations

import numpy as np
import pytest

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType

K = 3  # num_spec_step, as MTP-3 / DSpark would set it


def _decode_seq(tokens, *, num_rejected=0):
    seq = Sequence(list(tokens[:1]), block_size=16)
    for t in tokens[1:]:
        seq.append_token(t)
    seq.type = SequenceType.DECODE
    seq.num_rejected = num_rejected
    return seq


def _prefill_seq(tokens, *, num_cached=0):
    seq = Sequence(list(tokens), block_size=16)
    seq.type = SequenceType.PREFILL
    seq.num_cached_tokens = num_cached
    return seq


def _build(seqs, nums, *, spec=None, num_spec_step=0):
    return ScheduledBatch(
        seqs={s.id: s for s in seqs},
        num_scheduled_tokens=list(nums),
        total_tokens_num=sum(nums),
        total_seqs_num=len(seqs),
        num_spec_step=num_spec_step,
        scheduled_spec_decode_tokens=spec,
    )


# ── golden: the per-row forms these replaced ─────────────────────────────


def _golden_tokens(seqs, nums):
    out = np.empty(sum(nums), dtype=np.int32)
    pos = 0
    for seq, num in zip(seqs, nums):
        if seq.type == SequenceType.PREFILL:
            offset = seq.num_cached_tokens
        else:
            offset = seq.num_tokens - seq.num_rejected - num
        out[pos : pos + num] = seq.token_ids[offset : offset + num]
        pos += num
    return out


def _golden_drafts(req_ids, spec, k):
    out = np.zeros((len(req_ids), k), dtype=np.int32)
    for i, req_id in enumerate(req_ids):
        drafts = spec.get(req_id)
        if drafts is None or drafts.size == 0:
            continue
        width = min(drafts.size, k)
        out[i, :width] = drafts[:width]
    return out


# ── the flat token window ────────────────────────────────────────────────


def test_pure_decode_batch_matches_the_per_row_form():
    seqs = [_decode_seq(range(i * 100, i * 100 + 40)) for i in range(1, 9)]
    nums = [1] * len(seqs)
    batch = _build(seqs, nums)
    assert np.array_equal(batch.scheduled_tokens, _golden_tokens(seqs, nums))
    assert batch.scheduled_tokens.dtype == np.int32


def test_mixed_prefill_and_decode_keep_their_own_offsets():
    """A chunked prefill reads from `num_cached_tokens`, a decode from the
    tail. Staging them into one run must not blur the two rules together."""
    seqs = [
        _prefill_seq(range(1000, 1064), num_cached=16),
        _decode_seq(range(2000, 2040)),
        _prefill_seq(range(3000, 3032), num_cached=0),
        _decode_seq(range(4000, 4040), num_rejected=2),
    ]
    nums = [8, 1, 12, 4]
    batch = _build(seqs, nums)
    assert np.array_equal(batch.scheduled_tokens, _golden_tokens(seqs, nums))


def test_speculative_window_matches_the_per_row_form():
    seqs = [
        _decode_seq(range(i * 100, i * 100 + 40), num_rejected=i % 3) for i in range(6)
    ]
    nums = [K + 1] * len(seqs)
    batch = _build(seqs, nums, num_spec_step=K)
    assert np.array_equal(batch.scheduled_tokens, _golden_tokens(seqs, nums))


def test_empty_batch():
    batch = _build([], [])
    assert batch.scheduled_tokens.size == 0


def test_a_sequence_too_short_for_its_window_raises():
    """The per-row form raised here, on the row that came up short. A staged
    run has no row edge -- what refuses it is the destination assign being
    whole-array, so the short run cannot be quietly written into a prefix."""
    seqs = [_decode_seq(range(10)), _decode_seq(range(100, 104))]
    with pytest.raises(ValueError, match="could not broadcast"):
        _build(seqs, [4, 99])

    # Control: the same batch with a window each sequence can fill is fine.
    ok = _build(seqs, [4, 4])
    assert ok.scheduled_tokens.size == 8


# ── the dense draft rows ─────────────────────────────────────────────────


def _spec_batch(rows, k=K):
    seqs = [_decode_seq(range(i * 100, i * 100 + 40)) for i in range(len(rows))]
    spec = {
        s.id: (None if r is None else np.asarray(r, dtype=np.int32))
        for s, r in zip(seqs, rows)
    }
    spec = {rid: v for rid, v in spec.items() if v is not None}
    batch = _build(seqs, [k + 1] * len(seqs), spec=spec, num_spec_step=k)
    return batch, [s.id for s in seqs], spec


def test_every_row_full_matches_the_per_row_form():
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    batch, req_ids, spec = _spec_batch(rows)
    assert np.array_equal(
        batch.scheduled_spec_decode_tokens, _golden_drafts(req_ids, spec, K)
    )
    assert batch.scheduled_spec_decode_tokens.shape == (len(rows), K)


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 2, 3], None, [7, 8, 9]],  # a sequence with no drafts yet
        [[1, 2, 3], [4, 5], [7, 8, 9]],  # a short row, zero-filled
        [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]],  # a long row, truncated
        [None, None],  # nothing drafted at all
        [[]],  # present but empty
        # A short row followed by an absent one. The padding is built from a
        # shared zero row, so padding in place instead of on a copy would leak
        # the short row's drafts into every undrafted sequence after it.
        [[4, 5], None, None],
    ],
)
def test_ragged_rows_are_padded_and_match_the_per_row_form(rows):
    batch, req_ids, spec = _spec_batch(rows)
    assert np.array_equal(
        batch.scheduled_spec_decode_tokens, _golden_drafts(req_ids, spec, K)
    )


def test_dtype_comes_from_the_destination_not_the_drafts():
    """`np.concatenate` of int64 rows yields int64; the per-row form cast into
    an int32 destination. Consumers index this into int32 buffers."""
    seqs = [_decode_seq(range(i * 100, i * 100 + 40)) for i in range(3)]
    spec = {s.id: np.arange(K, dtype=np.int64) for s in seqs}
    batch = _build(seqs, [K + 1] * 3, spec=spec, num_spec_step=K)
    assert batch.scheduled_spec_decode_tokens.dtype == np.int32

    # Control: the hazard is real -- unpinned, the rows decide.
    assert np.concatenate(list(spec.values())).dtype == np.int64


def test_the_dense_rows_do_not_alias_the_drafts_they_came_from():
    """Downstream writes into this array (`prepare_input_ids`). A form that
    handed back views would corrupt the caller's dict."""
    seqs = [_decode_seq(range(i * 100, i * 100 + 40)) for i in range(3)]
    spec = {s.id: np.arange(K, dtype=np.int32) for s in seqs}
    before = {rid: d.copy() for rid, d in spec.items()}
    batch = _build(seqs, [K + 1] * 3, spec=spec, num_spec_step=K)

    batch.scheduled_spec_decode_tokens[:] = -1
    for rid, original in before.items():
        assert np.array_equal(spec[rid], original)


def test_empty_batch_with_speculation_keeps_its_shape():
    batch = ScheduledBatch(
        seqs={},
        num_scheduled_tokens=[],
        total_tokens_num=0,
        num_spec_step=K,
        scheduled_spec_decode_tokens={},
    )
    assert batch.scheduled_spec_decode_tokens.shape == (0, K)
