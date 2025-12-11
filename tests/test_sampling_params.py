"""Unit tests for SamplingParams class."""

import pytest
from atom.sampling_params import SamplingParams


class TestSamplingParams:

    def test_default_values(self):
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.max_tokens == 64
        assert params.ignore_eos is False
        assert params.stop_strings is None

    def test_custom_values(self):
        params = SamplingParams(
            temperature=0.5,
            max_tokens=256,
            ignore_eos=True,
            stop_strings=["<|end|>", "STOP"]
        )
        assert params.temperature == 0.5
        assert params.max_tokens == 256
        assert params.ignore_eos is True
        assert params.stop_strings == ["<|end|>", "STOP"]

    def test_equality(self):
        params1 = SamplingParams(temperature=0.8, max_tokens=100)
        params2 = SamplingParams(temperature=0.8, max_tokens=100)
        assert params1 == params2
