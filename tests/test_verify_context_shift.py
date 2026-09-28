"""The verify window's positions must be the ones the scheduler staged.

`ScheduledBatch` places each decode window with one formula per output mode;
the attention builders rebuild the same window from `context_lens` minus
`tokenIDProcessor.verify_context_shift()` and derive positions and KV slots
from it. Token ids here are each token's own index, so the staged window IS its
true positions and the two can be compared directly.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atom.model_engine.model_runner import tokenIDProcessor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import SequenceType
from atom.model_ops.attentions.token_layout.decode import decode_positions

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
    shift = tokenIDProcessor.verify_context_shift(
        SimpleNamespace(
            num_rejected=batch.num_rejected, is_deferred_out=is_deferred_out
        )
    )
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
    # What the builders did before: the deferred path's shift on undeferred
    # output. It lands the window `num_rejected - 1` slots early.
    batch, _ = _decode_batch(seq_factory, 162, num_rejected, is_deferred_out=False)
    stale = decode_positions(batch.context_lens - batch.num_rejected, NUM)
    assert stale[0] == batch.scheduled_tokens[0] - (num_rejected - 1)


def test_no_shift_without_counts():
    processor = SimpleNamespace(num_rejected=None, is_deferred_out=False)
    assert tokenIDProcessor.verify_context_shift(processor) is None
