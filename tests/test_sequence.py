# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/sequence.py — public API only

import pytest
from atom.sampling_params import SamplingParams
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence


class TestSequenceCreation:
    def test_token_ids_and_len(self, seq_factory):
        seq = seq_factory([1, 2, 3])
        assert seq.token_ids == [1, 2, 3]
        assert len(seq) == 3

    def test_token_ids_are_copied(self, seq_factory):
        original = [1, 2, 3]
        seq = seq_factory(original)
        original.append(4)
        assert seq.token_ids == [1, 2, 3]

    def test_auto_increment_ids(self, seq_factory):
        s1 = seq_factory([1])
        s2 = seq_factory([2])
        assert s2.id == s1.id + 1

    def test_explicit_id(self, seq_factory):
        seq = seq_factory([1, 2], id=42)
        assert seq.id == 42

    def test_sampling_params_propagated(self):
        sp = SamplingParams(temperature=0.5, max_tokens=100, ignore_eos=True)
        seq = Sequence([1], 4, sampling_params=sp)
        assert seq.temperature == 0.5
        assert seq.max_tokens == 100
        assert seq.ignore_eos is True


class TestSequenceNumTokensAndBlocks:
    def test_num_blocks_exact(self, seq_factory):
        seq = seq_factory([1, 2, 3, 4], block_size=4)
        assert seq.num_blocks == 1

    def test_num_blocks_partial(self, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5], block_size=4)
        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 1

    def test_num_blocks_block_size_1(self, seq_factory):
        seq = seq_factory([1, 2, 3], block_size=1)
        assert seq.num_blocks == 3


class TestSequenceBlock:
    def test_block_returns_tokens(self, seq_factory):
        seq = seq_factory([10, 20, 30, 40, 50, 60, 70, 80], block_size=4)
        assert seq.block(0) == [10, 20, 30, 40]
        assert seq.block(1) == [50, 60, 70, 80]

    def test_block_partial_last(self, seq_factory):
        seq = seq_factory([10, 20, 30, 40, 50], block_size=4)
        assert seq.block(1) == [50]

    def test_block_out_of_range(self, seq_factory):
        seq = seq_factory([1, 2], block_size=4)
        with pytest.raises(AssertionError):
            seq.block(1)


class TestSequenceAppendToken:
    def test_append_extends_token_ids(self, seq_factory):
        seq = seq_factory([1, 2])
        seq.append_token(3)
        assert seq.token_ids == [1, 2, 3]
        assert len(seq) == 3
        assert seq.last_token == 3

    def test_append_tracks_output_tokens(self, seq_factory):
        seq = seq_factory([1, 2])
        seq.append_token(3)
        seq.append_token(4)
        assert seq.output_tokens == [3, 4]

    def test_append_crosses_block_boundary(self, seq_factory):
        seq = seq_factory([1, 2, 3, 4], block_size=4)
        assert seq.num_blocks == 1
        seq.append_token(5)
        assert seq.num_blocks == 2


class TestSequenceProperties:
    def test_is_finished(self, seq_factory):
        seq = seq_factory([1])
        assert not seq.is_finished
        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished

    def test_prompt_vs_completion_token_ids(self, seq_factory):
        seq = seq_factory([10, 20, 30])
        seq.append_token(40)
        seq.append_token(50)
        assert seq.prompt_token_ids == [10, 20, 30]
        assert seq.completion_token_ids == [40, 50]
        assert seq.num_completion_tokens == 2

    def test_getitem(self, seq_factory):
        seq = seq_factory([10, 20, 30])
        assert seq[0] == 10
        assert seq[-1] == 30


class TestGetExitSequence:
    def test_exit_sequence(self):
        seq = get_exit_sequence()
        assert seq.token_ids == [-1]
        assert seq.status == SequenceStatus.EXIT_ENGINE
