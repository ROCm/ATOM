# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from atom.config import DSparkConfig
from atom.models.deepseek_v41 import draft_kv
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


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8", "fp4"])
def test_dspark_accepts_supported_target_kv_cache_dtypes(kv_dtype):
    cfg = config()
    cfg.kv_cache_dtype = kv_dtype
    validate_speculative_config(cfg)


def test_dspark_rejects_unsupported_kv_cache_dtype():
    cfg = config()
    cfg.kv_cache_dtype = "fp16"
    with pytest.raises(ValueError, match="KV cache"):
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
@pytest.mark.parametrize("entry", ["preprocess", "preprocess_fanout", "add_request"])
def test_dpa_media_rejected_before_tokenization_and_sequence_creation(
    draft_tokens, entry
):
    from atom.model_engine.llm_engine import InputOutputProcessor
    from atom.sampling_params import SamplingParams

    def unexpected_encode(prompt):
        pytest.fail("Unsupported media reached tokenization")

    processor = InputOutputProcessor.__new__(InputOutputProcessor)
    processor.__dict__.update(
        config=SimpleNamespace(
            hf_config=DeepseekV41TextConfig(), enable_dp_attention=True
        ),
        num_speculative_tokens=draft_tokens,
        tokenizer=SimpleNamespace(encode=unexpected_encode),
    )
    with pytest.raises(ValueError, match="DP attention supports text requests only"):
        if entry == "add_request":
            from atom.model_engine.llm_engine import LLMEngine

            engine = SimpleNamespace(io_processor=processor)
            LLMEngine.add_request(
                engine,
                ["image prompt"],
                SamplingParams(n=2),
                multimodal_data_list=[{"image": 1}],
            )
        else:
            getattr(processor, entry)(
                "image prompt",
                SamplingParams(n=1 if entry == "preprocess" else 2),
                multimodal_data={"image": 1},
            )


@pytest.fixture
def a8w8_runtime(monkeypatch):
    modules = {
        "aiter.jit.utils.chip_info": SimpleNamespace(get_gfx_runtime=lambda: "gfx950"),
        "aiter.ops.mla_sparse_prefill": SimpleNamespace(
            mla_sparse_prefill_v41=lambda: None
        ),
        "aiter.ops.opus.mla": SimpleNamespace(
            opus_mla_v41_prefill_a8w8_gfx950_fwd=lambda: None,
            opus_mla_v41_prefill_a8w8_gfx1250_fwd=lambda: None,
        ),
    }

    def load(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(draft_kv, "import_module", load)
    return modules


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_quantized_draft_is_native_only(target_dtype, a8w8_runtime):
    cfg = config()
    cfg.hf_config.model_type = "deepseek_v41_text"
    cfg.plugin_config = None
    cfg.kv_cache_dtype = target_dtype
    setting = DSparkConfig.from_dict({"draft_kv_dtype": "fp8"})
    setting.validate_draft_kv(cfg)
    invalid = [
        ("kv_cache_dtype", "bf16"),
        ("speculative_config", None),
        ("plugin_config", object()),
    ]
    for field, value in invalid:
        previous = getattr(cfg, field)
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match="native DeepSeek-V4.1"):
            setting.validate_draft_kv(cfg)
        setattr(cfg, field, previous)
    cfg.hf_config.model_type = "deepseek_v4"
    with pytest.raises(ValueError, match="native DeepSeek-V4.1"):
        setting.validate_draft_kv(cfg)


@pytest.mark.parametrize("dtype", ["fp4", "fp16"])
def test_quantized_draft_requires_an_explicit_supported_dtype(dtype):
    assert DSparkConfig().draft_kv_dtype == "auto"
    with pytest.raises(ValueError, match="draft_kv_dtype"):
        DSparkConfig.from_dict({"draft_kv_dtype": dtype})


@pytest.mark.parametrize("target_dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250", "gfx942", "gfx1100"])
def test_explicit_fp8_draft_checks_device_before_execution(
    target_dtype, gfx, a8w8_runtime
):
    cfg = config()
    cfg.hf_config.model_type = "deepseek_v41_text"
    cfg.plugin_config = None
    cfg.kv_cache_dtype = target_dtype
    a8w8_runtime["aiter.jit.utils.chip_info"].get_gfx_runtime = lambda: gfx
    setting = DSparkConfig(draft_kv_dtype="fp8")
    if gfx in ("gfx950", "gfx1250"):
        setting.validate_draft_kv(cfg)
    else:
        with pytest.raises(
            ValueError, match=f"does not support {gfx}.*draft_kv_dtype=auto"
        ):
            setting.validate_draft_kv(cfg)
    assert cfg.kv_cache_dtype == target_dtype


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("missing", ["dispatcher", "kernel", "module"])
def test_explicit_fp8_draft_requires_aiter_interfaces(gfx, missing, a8w8_runtime):
    cfg = config()
    cfg.hf_config.model_type = "deepseek_v41_text"
    cfg.plugin_config = None
    cfg.kv_cache_dtype = "fp4"
    a8w8_runtime["aiter.jit.utils.chip_info"].get_gfx_runtime = lambda: gfx
    if missing == "module":
        del a8w8_runtime["aiter.ops.mla_sparse_prefill"]
    elif missing == "dispatcher":
        a8w8_runtime["aiter.ops.mla_sparse_prefill"].mla_sparse_prefill_v41 = None
    else:
        delattr(
            a8w8_runtime["aiter.ops.opus.mla"], f"opus_mla_v41_prefill_a8w8_{gfx}_fwd"
        )
    with pytest.raises(
        ValueError, match="draft_kv_dtype=fp8.*AITER.*draft_kv_dtype=auto"
    ):
        DSparkConfig(draft_kv_dtype="fp8").validate_draft_kv(cfg)


def test_default_draft_needs_no_a8w8_capability_probe(monkeypatch):
    monkeypatch.setattr(draft_kv, "import_module", lambda name: pytest.fail(name))
    setting = DSparkConfig()
    setting.validate_draft_kv(config())
    assert setting.draft_kv_dtype == "auto"
