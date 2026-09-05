# SPDX-License-Identifier: MIT
# Tests for atom/utils/envs.py — lazy env var evaluation

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
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION",
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
    "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
    "ATOM_USE_MODEL_SENSITIVE_RMSNORM",
    "ATOM_TORCH_PROFILER_DIR",
    "ATOM_PROFILER_MORE",
    "ATOM_PROFILER_TIMEOUT",
    "ATOM_LOG_MORE",
    "ATOM_DISABLE_MMAP",
    "ATOM_ONLINE_QUANT_STREAMING",
    "ATOM_DISABLE_VLLM_PLUGIN",
    "ATOM_USE_CUSTOM_ALL_GATHER",
    "ATOM_ENABLE_RELAXED_MTP",
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

    def test_use_triton_gemm_default(self):
        assert _get_envs().ATOM_USE_TRITON_GEMM is False

    def test_ds_input_rmsnorm_quant_fusion_default_enabled(self):
        assert _get_envs().ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is True

    def test_model_sensitive_rmsnorm_default_disabled(self):
        assert _get_envs().ATOM_USE_MODEL_SENSITIVE_RMSNORM is False

    def test_torch_profiler_dir_default(self):
        assert _get_envs().ATOM_TORCH_PROFILER_DIR is None

    def test_profiler_more_default(self):
        assert _get_envs().ATOM_PROFILER_MORE is False

    def test_profiler_timeout_default(self):
        assert _get_envs().ATOM_PROFILER_TIMEOUT == 300.0

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

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            _ = _get_envs().ATOM_NONEXISTENT_VAR


class TestEnvsOverrides:
    """Test that env vars are read dynamically (lazy evaluation)."""

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

    def test_torch_profiler_dir_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", "/tmp/prof")
        assert _get_envs().ATOM_TORCH_PROFILER_DIR == "/tmp/prof"

    def test_profiler_more_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_PROFILER_MORE", "1")
        assert _get_envs().ATOM_PROFILER_MORE is True

    def test_profiler_timeout_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_PROFILER_TIMEOUT", "900")
        assert _get_envs().ATOM_PROFILER_TIMEOUT == 900.0

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
