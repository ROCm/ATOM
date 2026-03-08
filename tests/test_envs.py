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
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MXFP4_BMM",
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION",
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
    "ATOM_TORCH_PROFILER_DIR",
    "ATOM_PROFILER_MORE",
    "ATOM_LOG_MORE",
    "ATOM_DISABLE_MMAP",
    "ATOM_DISABLE_VLLM_PLUGIN",
    "ATOM_DISABLE_VLLM_PLUGIN_ATTENTION",
]


@pytest.fixture(autouse=True)
def _clean_atom_env(monkeypatch):
    """Ensure ATOM_* env vars are unset so defaults are tested reliably."""
    for var in _ATOM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _get_envs():
    """Return the envs module; lazy __getattr__ re-evaluates on each access."""
    import atom.utils.envs as envs

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

    def test_use_triton_gemm_default(self):
        assert _get_envs().ATOM_USE_TRITON_GEMM is False

    def test_ds_input_rmsnorm_quant_fusion_default_enabled(self):
        assert _get_envs().ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is True

    def test_torch_profiler_dir_default(self):
        assert _get_envs().ATOM_TORCH_PROFILER_DIR is None

    def test_profiler_more_default(self):
        assert _get_envs().ATOM_PROFILER_MORE is False

    def test_log_more_default(self):
        assert _get_envs().ATOM_LOG_MORE is False

    def test_disable_mmap_default(self):
        assert _get_envs().ATOM_DISABLE_MMAP is False

    def test_disable_vllm_plugin_default(self):
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN is False

    def test_disable_vllm_plugin_attention_default(self):
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN_ATTENTION is False

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

    def test_torch_profiler_dir_override(self, monkeypatch):
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", "/tmp/prof")
        assert _get_envs().ATOM_TORCH_PROFILER_DIR == "/tmp/prof"

    def test_profiler_more_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_PROFILER_MORE", "1")
        assert _get_envs().ATOM_PROFILER_MORE is True

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

    def test_disable_vllm_plugin_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN", "1")
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN is True

    def test_disable_vllm_plugin_attention_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "1")
        assert _get_envs().ATOM_DISABLE_VLLM_PLUGIN_ATTENTION is True


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
