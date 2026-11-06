# SPDX-License-Identifier: MIT
"""Batch-owned row slices preserve request, token and filtered state axes."""

import numpy as np
import pytest

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType


def make_batch(num_spec_step=1):
    seqs = []
    for i in range(5):
        s = Sequence(
            list(range(i * 100, i * 100 + 20)), block_size=4, has_per_req_cache=True
        )
        s.type = SequenceType.PREFILL if i < 2 else SequenceType.DECODE
        s.state_slots = [i * 3, i * 3 + 1, i * 3 + 2]
        s.state_fork_src = 50 + i
        s.block_table = [70 + i]
        seqs.append(s)
    decode_lens = [2, 1, 2] if num_spec_step else [1, 1, 1]
    return ScheduledBatch(
        {s.id: s for s in seqs},
        [8, 8] + decode_lens,
        16 + sum(decode_lens),
        total_tokens_num_prefill=16,
        total_tokens_num_decode=sum(decode_lens),
        total_seqs_num=5,
        total_seqs_num_prefill=2,
        total_seqs_num_decode=3,
        num_spec_step=num_spec_step,
    )


def test_decode_slice_owns_all_row_fields_and_counts():
    batch = make_batch()
    child = batch.slice_rows(2, 5)
    assert isinstance(child, ScheduledBatch)
    assert child.req_ids == batch.req_ids[2:]
    assert child.block_tables == [[72], [73], [74]]
    assert child.state_slots_committed == [6, 9, 12]
    assert child.state_fork_srcs == [52, 53, 54]
    assert child.state_rows == [0, 1, 2]
    assert child.block_table_rows == [0, 1, 2]
    assert child.num_scheduled_tokens.tolist() == [2, 1, 2]
    np.testing.assert_array_equal(child.scheduled_tokens, batch.scheduled_tokens[16:])
    assert child.total_tokens_num == child.total_tokens_num_decode == 5
    assert child.total_seqs_num == child.total_seqs_num_decode == 3
    assert child.total_tokens_num_prefill == child.total_seqs_num_prefill == 0
    assert not child.is_mixed
    assert batch.is_mixed and batch.total_tokens_num == 21


def test_non_spec_slice_has_no_draft_token_table():
    batch = make_batch(num_spec_step=0)
    assert batch.scheduled_spec_decode_tokens is None
    child = batch.slice_rows(2, 5)
    assert child.scheduled_spec_decode_tokens is None
    assert child.num_spec_step == 0


def test_filtered_state_rows_are_gathered_by_their_batch_index():
    batch = make_batch()
    batch.state_rows = [0, 2, 4]
    for name in batch._STATE_FIELDS:
        values = getattr(batch, name)
        setattr(batch, name, [values[i] for i in [0, 2, 4]])
    child = batch.slice_rows(2, 5)
    assert child.state_rows == [0, 2]
    assert child.state_slots_committed == [6, 12]
    assert child.state_fork_srcs == [52, 54]


def test_unclassified_new_field_cannot_silently_read_parent_rows():
    batch = make_batch()
    batch.new_row_field = np.arange(5)
    with pytest.raises(ValueError, match="slice policy.*new_row_field"):
        batch.slice_rows(2, 5)


@pytest.mark.parametrize(
    "field", ["context_lens", "num_rejected", "state_fork_srcs", "block_tables"]
)
def test_misaligned_input_is_rejected(field):
    batch = make_batch()
    setattr(batch, field, getattr(batch, field)[:-1])
    with pytest.raises(ValueError, match="aligned"):
        batch.slice_rows(2, 5)


def test_prefill_and_empty_slices():
    batch = make_batch()
    prefill = batch.slice_rows(0, 2)
    assert prefill.total_tokens_num_prefill == 16
    assert prefill.total_seqs_num_decode == 0
    empty = batch.slice_rows(5, 5)
    assert empty.total_tokens_num == empty.total_seqs_num == 0


def test_structured_request_row_keys_are_filtered_and_rebased():
    batch = make_batch()
    captures = {0: object(), 2: object(), 4: object()}
    batch.structured_requests = captures
    child = batch.slice_rows(2, 5)
    assert child.structured_requests == {0: captures[2], 2: captures[4]}
    assert batch.structured_requests is captures
    assert batch.slice_rows(5, 5).structured_requests == {}


@pytest.mark.parametrize("prepared", [False, True])
def test_logprob_fields_follow_request_rows_without_parent_device_indices(prepared):
    batch = make_batch()
    batch.return_logprobs = [True, False, False, True, False]
    batch.any_logprobs = True
    batch.top_logprobs = [2, 0, 0, 5, 0]
    batch.logprob_requested = (0, 3)
    if prepared:
        batch.logprob_request_indices = object()
        batch.logprob_rows = object()
    child = batch.slice_rows(2, 5)
    assert child.return_logprobs == [False, True, False]
    assert child.top_logprobs == [0, 5, 0]
    assert child.any_logprobs and child.logprob_requested == (1,)
    assert child.logprob_request_indices is child.logprob_rows is None
    disabled = batch.slice_rows(4, 5)
    assert not disabled.any_logprobs and disabled.logprob_requested == ()
    assert batch.logprob_requested == (0, 3)
    if prepared:
        assert batch.logprob_request_indices is not None
        assert batch.logprob_rows is not None
