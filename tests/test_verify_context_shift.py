"""The verify window's positions must be the ones the scheduler staged.

Attention builders rebuild the window from `context_lens - verify_context_shift()`.
Token ids equal their index, so staged tokens are their true positions.
"""

import numpy as np
import pytest

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import SequenceType
from atom.model_ops.attentions.token_layout.decode import (
    decode_positions,
    verify_context_shift,
)

MTP_K = 7
NUM = MTP_K + 1  # anchor + drafts


def _decode_batch(seq_factory, length, num_rejected, is_deferred_out):
    seq = seq_factory(list(range(length)))
    seq.type = SequenceType.DECODE
    seq.num_rejected = num_rejected
    batch = ScheduledBatch(
        {seq.id: seq},
        [NUM],
        NUM,
        total_tokens_num_decode=NUM,
        total_seqs_num=1,
        total_seqs_num_decode=1,
        num_spec_step=MTP_K,
        is_deferred_out=is_deferred_out,
    )
    shift = verify_context_shift(batch.num_rejected, is_deferred_out)
    return batch, shift


@pytest.mark.parametrize("is_deferred_out", [True, False])
@pytest.mark.parametrize("num_rejected", range(MTP_K + 1))
def test_builder_positions_match_staged_window(
    seq_factory, is_deferred_out, num_rejected
):
    batch, shift = _decode_batch(seq_factory, 162, num_rejected, is_deferred_out)
    positions = decode_positions(batch.context_lens - shift, NUM)
    np.testing.assert_array_equal(positions, batch.scheduled_tokens)


@pytest.mark.parametrize("num_rejected", [0, 3, 6])
def test_undeferred_window_is_not_shifted_by_num_rejected(seq_factory, num_rejected):
    # The deferred shift applied to undeferred output lands the window
    # `num_rejected - 1` slots early.
    batch, _ = _decode_batch(seq_factory, 162, num_rejected, is_deferred_out=False)
    stale = decode_positions(batch.context_lens - batch.num_rejected, NUM)
    assert stale[0] == batch.scheduled_tokens[0] - (num_rejected - 1)


def test_no_shift_without_counts():
    assert verify_context_shift(None, is_deferred_out=False) is None
    assert verify_context_shift(None, is_deferred_out=True) is None


@pytest.mark.parametrize("num_rejected", [0, 1, MTP_K])
def test_undeferred_shift_is_one_whatever_was_rejected(num_rejected):
    # The undeferred window is fixed by the placeholder run postprocess left,
    # so it does not move with how many drafts the sampler rejected.
    rejected = np.full(4, num_rejected, dtype=np.int32)
    np.testing.assert_array_equal(
        verify_context_shift(rejected, is_deferred_out=False), np.ones(4)
    )


def test_deferred_shift_is_the_rejected_count_per_row():
    rejected = np.array([0, 3, MTP_K], dtype=np.int32)
    shift = verify_context_shift(rejected, is_deferred_out=True)
    np.testing.assert_array_equal(shift, rejected)
    assert shift is rejected  # read, not copied: callers must not mutate it


def test_shift_keeps_the_row_count_and_an_index_safe_dtype():
    rejected = np.zeros(5, dtype=np.int32)
    for deferred in (True, False):
        shift = verify_context_shift(rejected, is_deferred_out=deferred)
        assert shift.shape == rejected.shape
        assert np.issubdtype(shift.dtype, np.integer)
