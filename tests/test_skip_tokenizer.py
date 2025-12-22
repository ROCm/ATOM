# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight unit tests for skip_tokenizer functionality."""

import pytest
from unittest.mock import Mock

# Skip if torch not available
torch = pytest.importorskip("torch", reason="torch is required")

from atom.sampling_params import SamplingParams
from atom.model_engine.llm_engine import InputOutputProcessor


class TestSkipTokenizer:
    """Test InputOutputProcessor with skip_tokenizer (tokenizer=None)."""

    def test_accepts_token_ids(self):
        """Pre-tokenized input should work without tokenizer."""
        processor = InputOutputProcessor(tokenizer=None, block_size=16)
        seq = processor.preprocess([1, 2, 3, 4, 5], SamplingParams())
        assert seq.num_prompt_tokens == 5

    def test_rejects_string_input(self):
        """String input should fail without tokenizer."""
        processor = InputOutputProcessor(tokenizer=None, block_size=16)
        with pytest.raises(ValueError, match="pre-tokenized"):
            processor.preprocess("hello", SamplingParams())

    def test_rejects_stop_strings(self):
        """stop_strings should fail without tokenizer."""
        processor = InputOutputProcessor(tokenizer=None, block_size=16)
        with pytest.raises(ValueError, match="stop_strings"):
            processor.preprocess([1, 2, 3], SamplingParams(stop_strings=["STOP"]))

    def test_postprocess_returns_token_ids(self):
        """Postprocess should return token_ids even without tokenizer."""
        processor = InputOutputProcessor(tokenizer=None, block_size=16)
        seq = processor.preprocess([1, 2, 3], SamplingParams())
        seq.completion_token_ids = [4, 5, 6]
        seq.leave_reason = "eos"
        seq.first_token_time = 1.0
        seq.arrive_time = 0.0
        
        result = processor.postprocess([seq])
        assert result[seq.id]["text"] == ""  # No tokenizer = empty text
        assert result[seq.id]["token_ids"] == [4, 5, 6]


class TestWithTokenizer:
    """Ensure normal tokenizer mode still works."""

    def test_tokenizes_string(self):
        """String input should be tokenized when tokenizer available."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        
        processor = InputOutputProcessor(tokenizer=mock_tokenizer, block_size=16)
        seq = processor.preprocess("hello", SamplingParams())
        
        mock_tokenizer.encode.assert_called_once_with("hello")
        assert seq.num_prompt_tokens == 3

    def test_skips_tokenization_for_list(self):
        """Pre-tokenized input should bypass tokenizer.encode()."""
        mock_tokenizer = Mock()
        processor = InputOutputProcessor(tokenizer=mock_tokenizer, block_size=16)
        seq = processor.preprocess([1, 2, 3], SamplingParams())
        
        mock_tokenizer.encode.assert_not_called()
        assert seq.num_prompt_tokens == 3
