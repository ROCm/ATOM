# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from atom.config import DSparkConfig
from atom.models.deepseek_v41.config import (
    DeepseekV41TextConfig,
    validate_speculative_config,
)


def config():
    return SimpleNamespace(
        model="/model",
        tensor_parallel_size=4,
        enable_dp_attention=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        hf_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            model="/model",
            synthetic_acceptance_rates=None,
        ),
        dspark=DSparkConfig(),
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_static_and_explicitly_calibrated_native_dspark(tp_size):
    value = config()
    value.tensor_parallel_size = tp_size
    validate_speculative_config(value)
    value.dspark = DSparkConfig(
        confidence_schedule=True, ragged=True, calibration_profile="profile.json"
    )
    validate_speculative_config(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_cache_dtype", "fp4"),
    ],
)
def test_unvalidated_runtime_combinations_are_rejected(field, value):
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        validate_speculative_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "mtp"),
        ("num_speculative_tokens", 4),
        ("model", "/another_model"),
    ],
)
def test_incompatible_draft_contract_is_rejected(field, value):
    cfg = config()
    setattr(cfg.speculative_config, field, value)
    with pytest.raises(ValueError):
        validate_speculative_config(cfg)


def test_dynamic_schedule_cannot_use_synthetic_costs():
    cfg = config()
    cfg.dspark = DSparkConfig(confidence_schedule=True, ragged=True)
    with pytest.raises(ValueError, match="calibration_profile"):
        validate_speculative_config(cfg)


def test_multimodal_rejection_belongs_to_the_model_request_contract():
    cfg = DeepseekV41TextConfig()
    cfg.validate_request(num_draft_tokens=0, multimodal_data={"image": 1})
    cfg.validate_request(num_draft_tokens=5, multimodal_data=None)
    with pytest.raises(ValueError, match="text requests only"):
        cfg.validate_request(num_draft_tokens=5, multimodal_data={"image": 1})


def test_unused_or_invalid_profile_configuration_is_rejected():
    with pytest.raises(ValueError, match="confidence_schedule"):
        DSparkConfig(calibration_profile="profile.json")
    with pytest.raises(ValueError, match="requires a path"):
        DSparkConfig(confidence_schedule=True, calibration_profile="")


def test_relaxed_acceptance_cannot_bypass_target_distribution(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", "1")
    with pytest.raises(ValueError, match="strict target verification"):
        validate_speculative_config(config())


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("rates", [None, [1.0] * 5, [0.8] * 5])
def test_native_and_fixed_acceptance_schedules_are_admitted(tp_size, rates):
    cfg = config()
    cfg.tensor_parallel_size = tp_size
    cfg.speculative_config.synthetic_acceptance_rates = rates
    validate_speculative_config(cfg)


@pytest.mark.parametrize(
    "tp,dp", [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8), (2, 2), (4, 2)]
)
def test_dpa_dspark_admission_does_not_fix_parallel_width(tp, dp):
    value = config()
    value.enable_dp_attention = True
    value.tensor_parallel_size = tp
    value.parallel_config.data_parallel_size = dp
    validate_speculative_config(value)


@pytest.mark.parametrize("dpa", [False, True])
@pytest.mark.parametrize("draft_tokens", [0, 5])
@pytest.mark.parametrize("media", [None, {"image": 1}])
def test_request_admission_checks_dpa_independently_of_dspark(dpa, draft_tokens, media):
    cfg = DeepseekV41TextConfig()
    kwargs = {
        "num_draft_tokens": draft_tokens,
        "multimodal_data": media,
        "enable_dp_attention": dpa,
    }
    if media and (dpa or draft_tokens):
        with pytest.raises(ValueError, match="text requests only"):
            cfg.validate_request(**kwargs)
    else:
        cfg.validate_request(**kwargs)


@pytest.mark.parametrize("draft_tokens", [0, 5])
def test_dpa_media_rejected_before_tokenization_and_sequence_creation(draft_tokens):
    from atom.model_engine.llm_engine import InputOutputProcessor

    def unexpected_encode(prompt):
        pytest.fail("Unsupported media reached tokenization")

    processor = SimpleNamespace(
        config=SimpleNamespace(
            hf_config=DeepseekV41TextConfig(), enable_dp_attention=True
        ),
        num_speculative_tokens=draft_tokens,
        tokenizer=SimpleNamespace(encode=unexpected_encode),
    )
    with pytest.raises(ValueError, match="DP attention supports text requests only"):
        InputOutputProcessor.preprocess_fanout(
            processor,
            "image prompt",
            SimpleNamespace(n=2),
            multimodal_data={"image": 1},
        )
