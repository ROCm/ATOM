# SPDX-License-Identifier: MIT
# Tests for atom/sampling_params.py

from atom.sampling_params import SamplingParams


class TestSamplingParamsDefaults:
    def test_default_temperature(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0

    def test_default_max_tokens(self):
        sp = SamplingParams()
        assert sp.max_tokens == 64

    def test_default_ignore_eos(self):
        sp = SamplingParams()
        assert sp.ignore_eos is False

    def test_default_stop_strings(self):
        sp = SamplingParams()
        assert sp.stop_strings is None


class TestSamplingParamsCustom:
    def test_custom_values(self):
        sp = SamplingParams(
            temperature=0.7, max_tokens=128, ignore_eos=True, stop_strings=["END"]
        )
        assert sp.temperature == 0.7
        assert sp.max_tokens == 128
        assert sp.ignore_eos is True
        assert sp.stop_strings == ["END"]

    def test_zero_temperature(self):
        sp = SamplingParams(temperature=0.0)
        assert sp.temperature == 0.0
