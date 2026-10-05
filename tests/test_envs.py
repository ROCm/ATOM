# SPDX-License-Identifier: MIT
# Tests for atom/utils/envs.py — lazy env var evaluation

import logging

import pytest

# All ATOM_* env vars that could affect default-value tests
_ATOM_ENV_VARS = [
    "ATOM_DP_RANK",
    "ATOM_DP_RANK_LOCAL",
    "ATOM_DP_SIZE",
    "ATOM_DP_MASTER_IP",
    "ATOM_DP_MASTER_PORT",
    "ATOM_DP_BASE_PORT",
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MXFP4_BMM",
    "ATOM_USE_V4_PREFILL_ASM_FOR_DECODE",
    "ATOM_MHC_USE_BF16",
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION",
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
    "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
    "ATOM_USE_MODEL_SENSITIVE_RMSNORM",
    "ATOM_TORCH_PROFILER_DIR",
    "ATOM_ENABLE_METRICS_DEVICE_TIMER",
    "ATOM_METRICS_UPDATE_INTERVAL_S",
    "ATOM_SHUTDOWN_TIMEOUT_S",
    "ATOM_PROFILER_MORE",
    "ATOM_PROFILER_RECORD_SHAPES",
    "ATOM_PROFILER_WITH_STACK",
    "ATOM_PROFILER_PROFILE_MEMORY",
    "ATOM_PROFILER_TIMEOUT",
    "ATOM_LOG_MORE",
    "ATOM_DISABLE_MMAP",
    "ATOM_ONLINE_QUANT_STREAMING",
    "ATOM_DISABLE_VLLM_PLUGIN",
    "ATOM_USE_CUSTOM_ALL_GATHER",
    "ATOM_ENABLE_RELAXED_MTP",
    "ATOM_USE_FLYDSL_GATHER_KV_B_PROJ",
    "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN",
]

_PROFILER_DETAIL_VARS = [
    "ATOM_PROFILER_RECORD_SHAPES",
    "ATOM_PROFILER_WITH_STACK",
    "ATOM_PROFILER_PROFILE_MEMORY",
]


@pytest.fixture(autouse=True)
def _clean_atom_env(monkeypatch):
    """Ensure ATOM_* env vars are unset so defaults are tested reliably."""
    for var in _ATOM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _get_envs():
    """Return the envs module; lazy __getattr__ re-evaluates on each access."""
    from atom.utils import envs

    return envs


class TestEnvsDefaults:
    """Test default values when env vars are NOT set."""

    def test_dp_rank_default(self):
        assert _get_envs().ATOM_DP_RANK == 0

    def test_dp_rank_local_default(self):
        assert _get_envs().ATOM_DP_RANK_LOCAL == 0

    def test_dp_size_default(self):
        assert _get_envs().ATOM_DP_SIZE == 1

    def test_dp_master_ip_default(self):
        assert _get_envs().ATOM_DP_MASTER_IP == "127.0.0.1"

    def test_dp_master_port_default(self):
        assert _get_envs().ATOM_DP_MASTER_PORT == 29500

    def test_dp_base_port_default(self):
        assert _get_envs().ATOM_DP_BASE_PORT == 0

    def test_mhc_use_bf16_default(self):
        assert _get_envs().ATOM_MHC_USE_BF16 is True

    def test_use_triton_gemm_default(self):
        assert _get_envs().ATOM_USE_TRITON_GEMM is False

    def test_use_v4_prefill_asm_for_decode_default_disabled(self):
        assert _get_envs().ATOM_USE_V4_PREFILL_ASM_FOR_DECODE is False

    def test_ds_input_rmsnorm_quant_fusion_default_enabled(self):
        assert _get_envs().ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is True

    def test_model_sensitive_rmsnorm_default_disabled(self):
        assert _get_envs().ATOM_USE_MODEL_SENSITIVE_RMSNORM is False

    def test_torch_profiler_dir_default(self):
        assert _get_envs().ATOM_TORCH_PROFILER_DIR is None

    def test_metrics_device_timer_default_disabled(self):
        assert _get_envs().ATOM_ENABLE_METRICS_DEVICE_TIMER is False

    def test_metrics_update_interval_default(self, caplog):
        assert _get_envs().ATOM_METRICS_UPDATE_INTERVAL_S == 1.0
        assert not caplog.records

    def test_profiler_more_default(self):
        assert _get_envs().ATOM_PROFILER_MORE is False

    @pytest.mark.parametrize("name", _PROFILER_DETAIL_VARS)
    def test_profiler_detail_default(self, name):
        assert getattr(_get_envs(), name) is False

    def test_profiler_timeout_default(self):
        assert _get_envs().ATOM_PROFILER_TIMEOUT == 300.0

    def test_shutdown_timeout_default(self, caplog):
        assert _get_envs().ATOM_SHUTDOWN_TIMEOUT_S == 5.0
        assert not caplog.records

    def test_log_more_default(self):
        assert _get_envs().ATOM_LOG_MORE is False

    def test_disable_mmap_default(self):
        assert _get_envs().ATOM_DISABLE_MMAP is False

    def test_online_quant_streaming_default_disabled(self):
        assert _get_envs().ATOM_ONLINE_QUANT_STREAMING is False

    def test_disable_vllm_plugin_default(self):
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN is False

    def test_atom_enable_relaxed_mtp_default(self):
        assert _get_envs().ATOM_ENABLE_RELAXED_MTP is False

    def test_atom_enable_gdn_decode_lossy_fast_default(self):
        assert _get_envs().ATOM_ENABLE_GDN_DECODE_LOSSY_FAST is False

    def test_use_flydsl_gather_kv_b_proj_default(self):
        assert _get_envs().ATOM_USE_FLYDSL_GATHER_KV_B_PROJ is True

    def test_use_flydsl_fp8_prefill_attn_default(self):
        assert _get_envs().ATOM_USE_FLYDSL_FP8_PREFILL_ATTN is False

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            _ = _get_envs().ATOM_NONEXISTENT_VAR


class TestEnvsOverrides:
    """Test that env vars are read dynamically (lazy evaluation)."""

    @pytest.mark.parametrize("value, expected", [("0", False), ("1", True)])
    def test_mhc_use_bf16_override(self, monkeypatch, value, expected):
        monkeypatch.setenv("ATOM_MHC_USE_BF16", value)
        assert _get_envs().ATOM_MHC_USE_BF16 is expected

    def test_dp_rank_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_RANK", "3")
        assert _get_envs().ATOM_DP_RANK == 3

    def test_dp_size_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SIZE", "8")
        assert _get_envs().ATOM_DP_SIZE == 8

    def test_dp_port_overrides(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_MASTER_PORT", "29700")
        monkeypatch.setenv("ATOM_DP_BASE_PORT", "29800")
        assert _get_envs().ATOM_DP_MASTER_PORT == 29700
        assert _get_envs().ATOM_DP_BASE_PORT == 29800

    def test_use_v4_prefill_asm_for_decode_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_USE_V4_PREFILL_ASM_FOR_DECODE", "1")
        assert _get_envs().ATOM_USE_V4_PREFILL_ASM_FOR_DECODE is True

    def test_torch_profiler_dir_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", "/tmp/prof")
        assert _get_envs().ATOM_TORCH_PROFILER_DIR == "/tmp/prof"

    def test_profiler_more_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_PROFILER_MORE", "1")
        assert _get_envs().ATOM_PROFILER_MORE is True

    @pytest.mark.parametrize("name", _PROFILER_DETAIL_VARS)
    @pytest.mark.parametrize("more", [None, "", "0", "1"])
    def test_profiler_detail_falls_back_to_profiler_more(self, monkeypatch, name, more):
        if more is not None:
            monkeypatch.setenv("ATOM_PROFILER_MORE", more)
        monkeypatch.setenv(name, "")
        assert getattr(_get_envs(), name) is (more == "1")

    @pytest.mark.parametrize("name", _PROFILER_DETAIL_VARS)
    @pytest.mark.parametrize("value, more", [("1", "0"), ("0", "1")])
    def test_profiler_detail_overrides_profiler_more(
        self, monkeypatch, name, value, more
    ):
        monkeypatch.setenv("ATOM_PROFILER_MORE", more)
        monkeypatch.setenv(name, value)
        assert getattr(_get_envs(), name) is (value == "1")
        others = [n for n in _PROFILER_DETAIL_VARS if n != name]
        assert [getattr(_get_envs(), n) for n in others] == [more == "1"] * 2

    def test_metrics_device_timer_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_ENABLE_METRICS_DEVICE_TIMER", "1")
        assert _get_envs().ATOM_ENABLE_METRICS_DEVICE_TIMER is True

    @pytest.mark.parametrize("value", ["0.25", "5"])
    def test_metrics_update_interval_override(self, monkeypatch, caplog, value):
        monkeypatch.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", value)
        assert _get_envs().ATOM_METRICS_UPDATE_INTERVAL_S == float(value)
        assert not caplog.records

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "bad"])
    def test_metrics_update_interval_warns_and_defaults(
        self, monkeypatch, caplog, value
    ):
        monkeypatch.setenv("ATOM_METRICS_UPDATE_INTERVAL_S", value)
        assert _get_envs().ATOM_METRICS_UPDATE_INTERVAL_S == 1.0
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        assert f"ATOM_METRICS_UPDATE_INTERVAL_S={value!r}" in record.getMessage()
        assert "using default 1.0" in record.getMessage()

    def test_profiler_timeout_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_PROFILER_TIMEOUT", "900")
        assert _get_envs().ATOM_PROFILER_TIMEOUT == 900.0

    @pytest.mark.parametrize("value", ["0.5", "1800"])
    def test_shutdown_timeout_override(self, monkeypatch, caplog, value):
        monkeypatch.setenv("ATOM_SHUTDOWN_TIMEOUT_S", value)
        assert _get_envs().ATOM_SHUTDOWN_TIMEOUT_S == float(value)
        assert not caplog.records

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "", "bad"])
    def test_shutdown_timeout_warns_and_defaults(self, monkeypatch, caplog, value):
        monkeypatch.setenv("ATOM_SHUTDOWN_TIMEOUT_S", value)
        assert _get_envs().ATOM_SHUTDOWN_TIMEOUT_S == 5.0
        assert len(caplog.records) == 1
        assert f"ATOM_SHUTDOWN_TIMEOUT_S={value!r}" in caplog.records[0].getMessage()

    def test_model_sensitive_rmsnorm_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_USE_MODEL_SENSITIVE_RMSNORM", "1")
        assert _get_envs().ATOM_USE_MODEL_SENSITIVE_RMSNORM is True

    def test_log_more_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_LOG_MORE", "1")
        assert _get_envs().ATOM_LOG_MORE is True

    def test_log_more_nonzero_int(self, monkeypatch):
        monkeypatch.setenv("ATOM_LOG_MORE", "2")
        assert _get_envs().ATOM_LOG_MORE is True

    def test_disable_mmap_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_DISABLE_MMAP", "true")
        assert _get_envs().ATOM_DISABLE_MMAP is True

    def test_disable_mmap_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ATOM_DISABLE_MMAP", "True")
        assert _get_envs().ATOM_DISABLE_MMAP is True

    def test_online_quant_streaming_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
        assert _get_envs().ATOM_ONLINE_QUANT_STREAMING is True

    def test_disable_vllm_plugin_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "1")
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN is True

    def test_atom_enable_relaxed_mtp_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", "1")
        assert _get_envs().ATOM_ENABLE_RELAXED_MTP is True

    def test_atom_enable_gdn_decode_lossy_fast_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", "1")
        assert _get_envs().ATOM_ENABLE_GDN_DECODE_LOSSY_FAST is True

    def test_use_flydsl_gather_kv_b_proj_disabled(self, monkeypatch):
        # The interesting lever now that the default is on: "1" would pass even
        # against a hard-coded True, so assert the opt-out instead.
        monkeypatch.setenv("ATOM_USE_FLYDSL_GATHER_KV_B_PROJ", "0")
        assert _get_envs().ATOM_USE_FLYDSL_GATHER_KV_B_PROJ is False

    def test_use_flydsl_gather_kv_b_proj_only_one_enables(self, monkeypatch):
        monkeypatch.setenv("ATOM_USE_FLYDSL_GATHER_KV_B_PROJ", "true")
        assert _get_envs().ATOM_USE_FLYDSL_GATHER_KV_B_PROJ is False


class TestIsSet:
    """Test the is_set() helper function."""

    def test_is_set_returns_false_when_unset(self):
        assert _get_envs().is_set("ATOM_DP_SIZE") is False

    def test_is_set_returns_true_when_set(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SIZE", "1")
        assert _get_envs().is_set("ATOM_DP_SIZE") is True

    def test_is_set_returns_false_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SIZE", "")
        assert _get_envs().is_set("ATOM_DP_SIZE") is False


def test_parallel_config_applies_explicit_dp_endpoint_env(monkeypatch):
    monkeypatch.setenv("ATOM_DP_MASTER_IP", "127.0.0.2")
    monkeypatch.setenv("ATOM_DP_MASTER_PORT", "29700")
    monkeypatch.setenv("ATOM_DP_BASE_PORT", "29800")

    from atom.config import ParallelConfig

    config = ParallelConfig()
    assert config.data_parallel_master_ip == "127.0.0.2"
    assert config.data_parallel_master_port == 29700
    assert config.data_parallel_base_port == 29800


def test_mla_fp8_prefill_flag(monkeypatch):
    name = "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN"
    assert getattr(_get_envs(), name) is False
    for value, expected in [("0", False), ("1", True), ("true", False)]:
        monkeypatch.setenv(name, value)
        assert getattr(_get_envs(), name) is expected


def test_offload_env_vars_are_documented():
    """Every offload knob registered in envs.py appears in the central env
    reference, so a new one cannot land undocumented."""
    import pathlib

    from atom.utils import envs

    doc = (
        pathlib.Path(__file__).parents[1] / "docs" / "environment_variables.md"
    ).read_text()
    offload = [
        name
        for name in envs.environment_variables
        if name.startswith(("OFFLOAD_", "LMCACHE_"))
    ]
    assert offload
    assert [name for name in offload if f"**{name}**" not in doc] == []


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("OFFLOAD_PUBLICATION_TIMEOUT_S", 5.0),
        ("OFFLOAD_PUBLICATION_POLL_INTERVAL_S", 0.01),
        ("OFFLOAD_COPY_WORKERS", 1),
        ("OFFLOAD_LOAD_WORKERS", 1),
        ("OFFLOAD_MIN_SAVE_TOKENS", 8192),
    ],
)
def test_empty_offload_knob_reads_as_its_default(monkeypatch, name, default):
    """`VAR=` is how a knob is cleared inline; it must never crash startup."""
    monkeypatch.setenv(name, "")
    assert getattr(_get_envs(), name) == default


@pytest.mark.parametrize("name", ["OFFLOAD_COPY_WORKERS", "OFFLOAD_LOAD_WORKERS"])
def test_malformed_offload_worker_width_names_the_variable(monkeypatch, name):
    monkeypatch.setenv(name, "two")
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        getattr(_get_envs(), name)


def test_malformed_offload_timeout_names_the_variable(monkeypatch):
    monkeypatch.setenv("OFFLOAD_PUBLICATION_TIMEOUT_S", "soon")
    with pytest.raises(ValueError, match="OFFLOAD_PUBLICATION_TIMEOUT_S must be"):
        _ = _get_envs().OFFLOAD_PUBLICATION_TIMEOUT_S
