# SPDX-License-Identifier: MIT
# CPP P2.1 — skip-output-comm and last-token-only logits.
#
# Tests _batch_needs_output (pp_engine_core transport-level skip) and
# _is_pure_middle_chunk (model_runner compute-level skip) without GPU
# dependencies. Both functions are duplicated here to avoid pulling in
# heavy import chains; the canonical copies live in pp_engine_core.py
# and model_runner.py respectively.

import numpy as np


def _batch_needs_output(batch) -> bool:
    if batch.total_seqs_num_decode > 0:
        return True
    final = batch.is_final_chunk
    if final is None:
        return True
    return any(final)


def _is_pure_middle_chunk(batch) -> bool:
    if batch.total_seqs_num_decode > 0:
        return False
    final = batch.is_final_chunk
    if final is None:
        return False
    return not any(final)


class _FakeBatch:
    def __init__(self, total_seqs_num_decode=0, is_final_chunk=None):
        self.total_seqs_num_decode = total_seqs_num_decode
        self.is_final_chunk = is_final_chunk


class TestBatchNeedsOutput:
    def test_decode_batch_always_needs_output(self):
        batch = _FakeBatch(total_seqs_num_decode=4)
        assert _batch_needs_output(batch) is True

    def test_decode_batch_with_final_chunk_none(self):
        batch = _FakeBatch(total_seqs_num_decode=2, is_final_chunk=None)
        assert _batch_needs_output(batch) is True

    def test_prefill_no_final_chunk_info(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=None)
        assert _batch_needs_output(batch) is True

    def test_all_middle_chunks_no_output(self):
        batch = _FakeBatch(
            total_seqs_num_decode=0, is_final_chunk=[False, False, False]
        )
        assert _batch_needs_output(batch) is False

    def test_one_final_chunk_needs_output(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=[False, True, False])
        assert _batch_needs_output(batch) is True

    def test_all_final_chunks_needs_output(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=[True, True])
        assert _batch_needs_output(batch) is True

    def test_single_middle_chunk(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=[False])
        assert _batch_needs_output(batch) is False

    def test_single_final_chunk(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=[True])
        assert _batch_needs_output(batch) is True

    def test_empty_final_chunk_list(self):
        batch = _FakeBatch(total_seqs_num_decode=0, is_final_chunk=[])
        assert _batch_needs_output(batch) is False


class TestIsPureMiddleChunk:
    """_is_pure_middle_chunk is the inverse of _batch_needs_output."""

    def test_decode_batch_not_middle(self):
        assert _is_pure_middle_chunk(_FakeBatch(total_seqs_num_decode=4)) is False

    def test_no_final_info_not_middle(self):
        assert _is_pure_middle_chunk(_FakeBatch(is_final_chunk=None)) is False

    def test_all_middle_chunks(self):
        assert (
            _is_pure_middle_chunk(_FakeBatch(is_final_chunk=[False, False, False]))
            is True
        )

    def test_mixed_not_pure_middle(self):
        assert (
            _is_pure_middle_chunk(_FakeBatch(is_final_chunk=[False, True, False]))
            is False
        )

    def test_all_final(self):
        assert _is_pure_middle_chunk(_FakeBatch(is_final_chunk=[True, True])) is False

    def test_consistency_with_batch_needs_output(self):
        """For any batch, exactly one of _batch_needs_output and
        _is_pure_middle_chunk returns True — except for decode batches
        and None-final batches where both agree (needs=True, middle=False)."""
        cases = [
            _FakeBatch(total_seqs_num_decode=1),
            _FakeBatch(is_final_chunk=None),
            _FakeBatch(is_final_chunk=[False]),
            _FakeBatch(is_final_chunk=[True]),
            _FakeBatch(is_final_chunk=[False, True]),
            _FakeBatch(is_final_chunk=[False, False]),
        ]
        for b in cases:
            assert _batch_needs_output(b) != _is_pure_middle_chunk(b)


class TestGetLastTokenIndices:
    """Validate last-token index computation from cu_seqlens_q."""

    @staticmethod
    def _compute_last_indices(num_scheduled_tokens):
        cu = np.cumsum(num_scheduled_tokens)
        return (cu - 1).tolist()

    def test_single_seq(self):
        assert self._compute_last_indices([100]) == [99]

    def test_multiple_seqs(self):
        assert self._compute_last_indices([100, 200, 300]) == [99, 299, 599]

    def test_single_token_seqs(self):
        assert self._compute_last_indices([1, 1, 1]) == [0, 1, 2]

    def test_mixed_lengths(self):
        assert self._compute_last_indices([5, 3, 10, 1]) == [4, 7, 17, 18]
