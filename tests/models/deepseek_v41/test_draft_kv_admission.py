# SPDX-License-Identifier: MIT
import pytest

from atom.config import DSparkConfig

from .test_dspark_admission import config


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_quantized_draft_is_native_only(target_dtype):
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
