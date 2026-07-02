"""Regression test for the MTP spec-decode positional-index crash.

Before the fix, Scheduler built ``ScheduledBatch.scheduled_spec_decode_tokens``
only from decode seqs whose ``spec_token_ids`` was non-empty, producing a
*compacted* array. ``TokenIDProcessor.prepare_input_ids`` then indexed that
array by full-batch sequence positions, so a decode seq without drafts (e.g. one
that just transitioned from prefill) made the array shorter than the batch and
the positional index ran off the end:

    IndexError: index 127 is out of bounds for axis 0 with size 127

The fix makes the array dense: one row per decode seq, in batch order.
"""
from types import SimpleNamespace

import numpy as np

from atom.model_engine.scheduler import Scheduler
from conftest import MockConfig


def _spec_config(k=3):
    return SimpleNamespace(num_speculative_tokens=k)


class TestSpecDecodeDenseArray:
    def test_array_row_per_decode_seq_when_one_seq_lacks_drafts(self, seq_factory):
        mtp_k = 3
        sched = Scheduler(
            MockConfig(
                max_num_seqs=8,
                num_kvcache_blocks=64,
                kv_cache_block_size=4,
                max_model_len=256,
                max_num_batched_tokens=256,
                speculative_config=_spec_config(mtp_k),
            )
        )

        s_with = seq_factory([1, 2, 3, 4])
        s_without = seq_factory([5, 6, 7, 8])
        sched.add(s_with)
        sched.add(s_without)
        sched.schedule()  # prefill both

        # Simulate the forward pass completing prefill.
        for s in (s_with, s_without):
            s.num_cached_tokens = s.num_prompt_tokens
            s.append_token(99)

        # One seq has proposed drafts; the other (fresh from prefill) has none.
        s_with.spec_token_ids = np.array([11, 12, 13], dtype=np.int32)
        s_without.spec_token_ids = np.array([], dtype=np.int32)

        batch, _ = sched.schedule()  # decode

        assert batch.total_seqs_num_decode == 2
        arr = batch.scheduled_spec_decode_tokens
        # Dense: one row per decode seq (was 1 before the fix -> IndexError).
        assert arr.shape == (2, mtp_k), f"expected dense (2,{mtp_k}), got {arr.shape}"

        # Positional indexing by full-batch positions must not raise, including
        # the tail seq that lacked drafts (this is model_runner.py:524).
        for pos in range(batch.total_seqs_num_decode):
            _ = arr[np.array([pos], dtype=np.intp)]

        # The seq that HAD drafts keeps them unchanged.
        pos_with = batch.req_ids.index(s_with.id)
        assert np.array_equal(arr[pos_with], np.array([11, 12, 13], dtype=np.int32))

    def test_all_seqs_have_drafts_unchanged(self, seq_factory):
        """Common path (every decode seq has drafts) is byte-for-byte unchanged."""
        mtp_k = 3
        sched = Scheduler(
            MockConfig(
                max_num_seqs=8,
                num_kvcache_blocks=64,
                kv_cache_block_size=4,
                max_model_len=256,
                max_num_batched_tokens=256,
                speculative_config=_spec_config(mtp_k),
            )
        )
        s1 = seq_factory([1, 2, 3, 4])
        s2 = seq_factory([5, 6, 7, 8])
        sched.add(s1)
        sched.add(s2)
        sched.schedule()
        for i, s in enumerate((s1, s2)):
            s.num_cached_tokens = s.num_prompt_tokens
            s.append_token(99)
            s.spec_token_ids = np.array(
                [10 * (i + 1) + j for j in range(mtp_k)], dtype=np.int32
            )
        batch, _ = sched.schedule()
        arr = batch.scheduled_spec_decode_tokens
        assert arr.shape == (2, mtp_k)
        assert np.array_equal(arr[batch.req_ids.index(s1.id)], np.array([10, 11, 12]))
        assert np.array_equal(arr[batch.req_ids.index(s2.id)], np.array([20, 21, 22]))
