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
    if not hasattr(_atom_config_stub, "DSparkConfig"):
        # arg_utils imports DSparkConfig and calls DSparkConfig.from_dict() to
        # build the DSpark runtime config from --dspark-config/--dspark-debug.
        _atom_config_stub.DSparkConfig = MagicMock(
            from_dict=lambda cfg, debug=False: MagicMock(cfg=cfg, debug=debug)
        )

from atom.model_engine.arg_utils import EngineArgs  # noqa: E402
from atom.utils.arg_parser import FlexibleArgumentParser  # noqa: E402


class TestFlexibleArgumentParser:
    """Unit tests for the dash/underscore aliasing, isolated from EngineArgs."""

    def test_snake_case_flag_accepts_kebab(self):
        p = FlexibleArgumentParser()
        p.add_argument("--kv_cache_dtype")
        assert p.parse_args(["--kv-cache-dtype", "fp8"]).kv_cache_dtype == "fp8"
        assert p.parse_args(["--kv_cache_dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_kebab_case_flag_accepts_snake(self):
        p = FlexibleArgumentParser()
        p.add_argument("--tensor-parallel-size", type=int)
        assert p.parse_args(["--tensor_parallel_size", "4"]).tensor_parallel_size == 4
        assert p.parse_args(["--tensor-parallel-size", "4"]).tensor_parallel_size == 4

    def test_short_flag_untouched(self):
        p = FlexibleArgumentParser()
        p.add_argument("--tensor-parallel-size", "-tp", type=int)
        # short flag still works and no bogus alias was minted from it
        assert p.parse_args(["-tp", "8"]).tensor_parallel_size == 8

    def test_json_value_with_underscores_preserved(self):
        # Only the flag name is rewritten; the value is passed through verbatim,
        # so JSON payloads with underscore keys survive intact.
        p = FlexibleArgumentParser()
        p.add_argument("--online_quant_config")
        ns = p.parse_args(["--online-quant-config", '{"global_quant_config": "fp8"}'])
        assert ns.online_quant_config == '{"global_quant_config": "fp8"}'

    def test_both_spellings_passed_explicitly_do_not_conflict(self):
        # Callers that still spell out both forms (the pre-refactor style) must
        # not trip a "conflicting option string" — the aliaser dedups instead.
        p = FlexibleArgumentParser()
        p.add_argument("--kv-cache-dtype", "--kv_cache_dtype", dest="kv_cache_dtype")
        assert p.parse_args(["--kv-cache-dtype", "fp8"]).kv_cache_dtype == "fp8"
        assert p.parse_args(["--kv_cache_dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_boolean_optional_action_both_negations(self):
        p = FlexibleArgumentParser()
        p.add_argument(
            "--enable_prefix_caching",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        assert (
            p.parse_args(["--no-enable_prefix_caching"]).enable_prefix_caching is False
        )
        assert (
            p.parse_args(["--no-enable-prefix-caching"]).enable_prefix_caching is False
        )
        assert p.parse_args(["--enable-prefix-caching"]).enable_prefix_caching is True


class TestKVCacheDtypeCliAlias:
    """--kv-cache-dtype and --kv_cache_dtype must both set kv_cache_dtype end-to-end."""

    def _parse(self, argv):
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        return parser.parse_args(argv)

    def test_dashed_form(self):
        assert self._parse(["--kv-cache-dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_underscore_form(self):
        assert self._parse(["--kv_cache_dtype", "fp8"]).kv_cache_dtype == "fp8"

    def test_default(self):
        assert self._parse([]).kv_cache_dtype == "bf16"

    def test_kebab_registered_flag_accepts_underscore(self):
        # --tensor-parallel-size is registered kebab-case; underscore must work.
        assert self._parse(["--tensor_parallel_size", "4"]).tensor_parallel_size == 4


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
