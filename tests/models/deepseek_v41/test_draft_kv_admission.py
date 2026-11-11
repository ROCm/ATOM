# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from atom.config import DSparkConfig
from atom.models.deepseek_v41 import draft_kv

from .test_dspark_admission import config


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
