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
    "ATOM_ENFORCE_EAGER",
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MXFP4_BMM",
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION",
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
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

    def test_enforce_eager_default(self):
        assert _get_envs().ATOM_ENFORCE_EAGER is False

    def test_use_triton_gemm_default(self):
        assert _get_envs().ATOM_USE_TRITON_GEMM is False

    def test_ds_input_rmsnorm_quant_fusion_default_enabled(self):
        assert _get_envs().ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is True

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

    def test_enforce_eager_enabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_ENFORCE_EAGER", "1")
        assert _get_envs().ATOM_ENFORCE_EAGER is True
