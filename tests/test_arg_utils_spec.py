# SPDX-License-Identifier: MIT
# Regression tests for speculative-config validation in EngineArgs._get_engine_kwargs.

import argparse
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
    if not hasattr(_atom_config_stub, "CUDAGraphMode"):
        # arg_utils imports CUDAGraphMode and does CUDAGraphMode[name] to map the
        # --cudagraph-mode string; use a real enum so subscript access works.
        import enum as _enum

        _atom_config_stub.CUDAGraphMode = _enum.Enum(
            "CUDAGraphMode",
            {"NONE": 0, "PIECEWISE": 1, "FULL": 2, "FULL_AND_PIECEWISE": 3},
        )

import argparse  # noqa: E402

from atom.model_engine.arg_utils import EngineArgs  # noqa: E402


class TestKVCacheDtypeCliAlias:
    """--kv-cache-dtype and --kv_cache_dtype must both set kv_cache_dtype."""

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)
        return parser.parse_args(argv)

    def test_dashed_form(self):
        assert self._parse(["--kv-cache-dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_underscore_form(self):
        assert self._parse(["--kv_cache_dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_default(self):
        assert self._parse([]).kv_cache_dtype == "bf16"


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


class TestEngineArgsIndexCacheDtype:
    """Regression tests for index-cache dtype CLI parity with KV cache dtype."""

    def test_index_cache_dtype_defaults_to_kv_cache_dtype(self):
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)

        args = EngineArgs.from_cli_args(parser.parse_args(["--kv_cache_dtype", "fp8"]))

        assert args.kv_cache_dtype == "fp8"
        assert args.index_cache_dtype == "fp8"

    def test_index_cache_dtype_accepts_dashed_spelling(self):
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)

        args = EngineArgs.from_cli_args(
            parser.parse_args(["--index-cache-dtype", "fp8"])
        )

        assert args.index_cache_dtype == "fp8"

    def test_index_cache_dtype_accepts_underscore_spelling(self):
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)

        args = EngineArgs.from_cli_args(
            parser.parse_args(["--index_cache_dtype", "fp8"])
        )

        assert args.index_cache_dtype == "fp8"
