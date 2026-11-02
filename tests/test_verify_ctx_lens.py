"""The verify window's positions must be the ones the scheduler staged.

Attention builders rebuild the window from
`tokenIDProcessor.verify_ctx_lens(context_lens)`. Token ids equal their index,
so staged tokens are their true positions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

# On the module, not on "aiter" -- see tests/test_dspark.py for why a name-only
# guard passes on the non-GPU runner and then aborts collection.
runner_module = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="requires AITER to import model_runner",
    exc_type=ImportError,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import SequenceType
from atom.model_ops.attentions.token_layout.decode import decode_positions

MTP_K = 7
NUM = MTP_K + 1  # anchor + drafts


def _verify_ctx_lens(context_lens, num_rejected, is_deferred_out):
    processor = SimpleNamespace(
        num_rejected=num_rejected, is_deferred_out=is_deferred_out
    )
    return runner_module.tokenIDProcessor.verify_ctx_lens(processor, context_lens)


def _decode_batch(seq_factory, length, num_rejected, is_deferred_out):
    seq = seq_factory(list(range(length)))
    seq.type = SequenceType.DECODE
    seq.num_rejected = num_rejected
    return ScheduledBatch(
        {seq.id: seq},
        [NUM],
        NUM,
        total_tokens_num_decode=NUM,
        total_seqs_num=1,
        total_seqs_num_decode=1,
        num_spec_step=MTP_K,
        is_deferred_out=is_deferred_out,
    )


@pytest.mark.parametrize("is_deferred_out", [True, False])
@pytest.mark.parametrize("num_rejected", range(MTP_K + 1))
def test_builder_positions_match_staged_window(
    seq_factory, is_deferred_out, num_rejected
):
    batch = _decode_batch(seq_factory, 162, num_rejected, is_deferred_out)
    context_lens = np.asarray(batch.context_lens, dtype=np.int32)
    ctx = _verify_ctx_lens(context_lens, batch.num_rejected, is_deferred_out)
    np.testing.assert_array_equal(decode_positions(ctx, NUM), batch.scheduled_tokens)


@pytest.mark.parametrize("num_rejected", [0, 3, 6])
def test_undeferred_window_is_not_shifted_by_num_rejected(seq_factory, num_rejected):
    # The deferred shift applied to undeferred output lands the window
    # `num_rejected - 1` slots early.
    batch = _decode_batch(seq_factory, 162, num_rejected, is_deferred_out=False)
    stale = decode_positions(batch.context_lens - batch.num_rejected, NUM)
    assert stale[0] == batch.scheduled_tokens[0] - (num_rejected - 1)


def test_no_shift_without_counts():
    ctx = np.array([10, 20], dtype=np.int32)
    for deferred in (True, False):
        assert _verify_ctx_lens(ctx, None, deferred) is ctx


@pytest.mark.parametrize("num_rejected", [0, 1, MTP_K])
def test_undeferred_shift_is_one_whatever_was_rejected(num_rejected):
    # The undeferred window is fixed by the placeholder run postprocess left,
    # so it does not move with how many drafts the sampler rejected.
    ctx = np.full(4, 100, dtype=np.int32)
    rejected = np.full(4, num_rejected, dtype=np.int64)
    np.testing.assert_array_equal(_verify_ctx_lens(ctx, rejected, False), ctx - 1)


def test_deferred_shift_is_the_rejected_count_per_row():
    ctx = np.array([100, 100, 100], dtype=np.int32)
    rejected = np.array([0, 3, MTP_K], dtype=np.int64)
    out = _verify_ctx_lens(ctx, rejected, True)
    np.testing.assert_array_equal(out, ctx - rejected)
    np.testing.assert_array_equal(ctx, 100)  # the caller's array is not mutated


@pytest.mark.parametrize("deferred", [True, False])
def test_keeps_the_callers_dtype(deferred):
    ctx = np.array([50, 60], dtype=np.int32)
    out = _verify_ctx_lens(ctx, np.array([1, 2], dtype=np.int64), deferred)
    assert out.dtype == np.int32
