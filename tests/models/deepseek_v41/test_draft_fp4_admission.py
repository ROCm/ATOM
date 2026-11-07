# SPDX-License-Identifier: MIT
import pytest

from atom.config import DSparkConfig

from .test_dspark_admission import config


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_true_a8w8_draft_admission(target_dtype):
    cfg = config()
    cfg.hf_config.model_type = "deepseek_v41_text"
    cfg.plugin_config = None
    cfg.kv_cache_dtype = target_dtype
    setting = DSparkConfig.from_dict({"draft_kv_dtype": "fp8"})
    setting.validate_draft_kv(cfg)
    for field, value in [
        ("kv_cache_dtype", "bf16"),
        ("speculative_config", None),
        ("plugin_config", object()),
    ]:
        previous = getattr(cfg, field)
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match="native DeepSeek-V4.1"):
            setting.validate_draft_kv(cfg)
        setattr(cfg, field, previous)


def test_draft_fp4_is_explicit_and_native_only():
    cfg = config()
    cfg.hf_config.model_type = "deepseek_v41_text"
    cfg.plugin_config = None
    cfg.kv_cache_dtype = "fp4"
    assert DSparkConfig().draft_kv_dtype == "auto"
    setting = DSparkConfig.from_dict({"draft_kv_dtype": "fp4"})
    setting.validate_draft_kv(cfg)
    for field, value in (
        ("kv_cache_dtype", "bf16"),
        ("kv_cache_dtype", "fp8"),
        ("speculative_config", None),
        ("plugin_config", object()),
    ):
        original = getattr(cfg, field)
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match="native DeepSeek-V4.1"):
            setting.validate_draft_kv(cfg)
        setattr(cfg, field, original)
    cfg.hf_config.model_type = "deepseek_v4"
    with pytest.raises(ValueError, match="native DeepSeek-V4.1"):
        setting.validate_draft_kv(cfg)
    with pytest.raises(ValueError, match="draft_kv_dtype"):
        DSparkConfig.from_dict({"draft_kv_dtype": "fp16"})
