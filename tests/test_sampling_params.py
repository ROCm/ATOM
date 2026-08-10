# SPDX-License-Identifier: MIT
# Tests for atom/sampling_params.py

import pytest

from atom.sampling_params import SamplingDefaults, SamplingParams


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

    def test_default_n(self):
        sp = SamplingParams()
        assert sp.n == 1


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

    def test_n_greater_than_one(self):
        sp = SamplingParams(n=4, temperature=0.8)
        assert sp.n == 4

    def test_n_zero_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            SamplingParams(n=0)

    def test_n_negative_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            SamplingParams(n=-3)


class FakeGenerationConfig:
    # Simulate Transformers defaults materialized despite being absent from
    # generation_config.json.
    temperature = 0.7
    top_p = 0.8
    top_k = 50

    def to_diff_dict(self):
        return {"top_p": 0.95}


def test_sampling_defaults_use_generation_config_diff():
    assert SamplingDefaults.from_generation_config(
        FakeGenerationConfig()
    ) == SamplingDefaults(top_p=0.95)


def test_sampling_defaults_preserve_explicit_request_values():
    model_defaults = SamplingDefaults(temperature=0.8, top_p=0.95, top_k=50)

    assert model_defaults.with_overrides(
        temperature=0.0, top_p=1.0, top_k=20
    ) == SamplingDefaults(temperature=0.0, top_p=1.0, top_k=20)


def test_sampling_defaults_keep_model_values_for_omitted_request_fields():
    model_defaults = SamplingDefaults(temperature=1.0, top_p=0.95, top_k=-1)

    assert model_defaults.with_overrides() == model_defaults
