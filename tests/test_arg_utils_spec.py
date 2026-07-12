# SPDX-License-Identifier: MIT
# Regression tests for speculative-config validation in EngineArgs._get_engine_kwargs.

import sys
from unittest.mock import MagicMock, patch

# conftest.py stubs atom.* and zmq before any atom imports are attempted,
# but arg_utils.py imports LLMEngine from atom and CompilationConfig /
# SpeculativeConfig from atom.config, which the minimal stub doesn't expose.
_atom_stub = sys.modules.get("atom")
if _atom_stub is not None and not hasattr(_atom_stub, "LLMEngine"):
    _atom_stub.LLMEngine = MagicMock()

_atom_config_stub = sys.modules.get("atom.config")
if _atom_config_stub is not None:
    if not hasattr(_atom_config_stub, "CompilationConfig"):
        _atom_config_stub.CompilationConfig = MagicMock(
            side_effect=lambda **kw: MagicMock(**kw)
        )
    if not hasattr(_atom_config_stub, "SpeculativeConfig"):
        _atom_config_stub.SpeculativeConfig = MagicMock(
            side_effect=lambda **kw: MagicMock(**kw)
        )
    if not hasattr(_atom_config_stub, "ParallelConfig"):
        _atom_config_stub.ParallelConfig = MagicMock(
            side_effect=lambda **kw: MagicMock(**kw)
        )

from atom.model_engine.arg_utils import EngineArgs  # noqa: E402


class TestEngineArgsSpeculativeValidation:
    """Regression tests for speculative-config construction in _get_engine_kwargs."""

    def test_no_method_gives_no_speculative_config(self):
        """method=None → speculative_config must be None (no crash)."""
        args = EngineArgs(method=None, num_speculative_tokens=1)
        kwargs = args._get_engine_kwargs()
        assert kwargs.get("speculative_config") is None

    def test_method_mtp_zero_tokens_disables_speculation(self):
        """method='mtp', num_speculative_tokens=0 → treated as disabled,
        speculative_config is None (regression for ZeroDivisionError)."""
        args = EngineArgs(method="mtp", num_speculative_tokens=0)
        kwargs = args._get_engine_kwargs()
        assert kwargs.get("speculative_config") is None

    def test_method_mtp_negative_tokens_disables_speculation(self):
        """method='mtp', num_speculative_tokens=-1 → treated as disabled,
        speculative_config is None."""
        args = EngineArgs(method="mtp", num_speculative_tokens=-1)
        kwargs = args._get_engine_kwargs()
        assert kwargs.get("speculative_config") is None

    def test_method_mtp_valid_tokens_builds_speculative_config(self):
        """method='mtp', num_speculative_tokens=3 → SpeculativeConfig constructed."""
        fake_spec_config = MagicMock()
        with patch(
            "atom.model_engine.arg_utils.SpeculativeConfig",
            return_value=fake_spec_config,
        ) as mock_cls:
            args = EngineArgs(method="mtp", num_speculative_tokens=3)
            kwargs = args._get_engine_kwargs()

        mock_cls.assert_called_once_with(
            method="mtp",
            model=args.model,
            num_speculative_tokens=3,
        )
        assert kwargs["speculative_config"] is fake_spec_config


class TestEngineArgsDistributedDP:
    """Tests for distributed DP/EP engine parameter plumbing."""

    def test_parallel_config_contains_global_and_local_dp_ranks(self):
        args = EngineArgs(
            data_parallel_size=16,
            data_parallel_size_local=8,
            data_parallel_rank=8,
            data_parallel_master_ip="10.0.0.1",
            data_parallel_master_port=29600,
            distributed_dp=True,
            distributed_dp_serving=True,
        )

        kwargs = args._get_engine_kwargs()
        pc = kwargs["parallel_config"]

        assert pc.data_parallel_size == 16
        assert pc.data_parallel_size_local == 8
        assert pc.data_parallel_rank == 8
        assert pc.data_parallel_master_ip == "10.0.0.1"
        assert pc.data_parallel_master_port == 29600
        assert pc.distributed_dp is True
        assert kwargs["distributed_dp_serving"] is True
        assert "data_parallel_size" not in kwargs

    def test_moe_backend_selectors_are_engine_parameters(self):
        args = EngineArgs(
            moe_all2all_backend="rccl",
            rccl_moe_impl="triton_batched_gemm",
            mori_all2all_mode="low-latency",
        )

        kwargs = args._get_engine_kwargs()

        assert kwargs["moe_all2all_backend"] == "rccl"
        assert kwargs["rccl_moe_impl"] == "triton_batched_gemm"
        assert kwargs["mori_all2all_mode"] == "low-latency"
        assert kwargs["enable_low_latency"] is True
