# SPDX-License-Identifier: MIT
# Regression tests for speculative-config validation in EngineArgs._get_engine_kwargs.

import argparse
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

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

from atom.model_engine.arg_utils import EngineArgs  # noqa: E402


def _write_adapter(path, tensors):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 1}),
        encoding="utf-8",
    )
    save_file(tensors, path / "adapter_model.safetensors")


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

    def test_lora_modules_pass_through_to_engine_kwargs(self):
        args = EngineArgs(lora_modules=["adapter=/tmp/adapter"])
        kwargs = args._get_engine_kwargs()
        assert kwargs["lora_modules"] == ["adapter=/tmp/adapter"]

    def test_routed_lora_modules_force_enforce_eager(self, tmp_path):
        adapter_path = tmp_path / "adapter"
        _write_adapter(
            adapter_path,
            {
                "base_model.model.model.layers.10.mlp.experts.0.down_proj."
                "lora_A.weight": torch.ones(1, 3),
                "base_model.model.model.layers.10.mlp.experts.0.down_proj."
                "lora_B.weight": torch.ones(4, 1),
            },
        )
        args = EngineArgs(
            enforce_eager=False,
            lora_modules=[f"adapter={adapter_path}"],
        )

        kwargs = args._get_engine_kwargs()

        assert kwargs["enforce_eager"] is True

    def test_regular_lora_modules_do_not_force_enforce_eager(self, tmp_path):
        adapter_path = tmp_path / "adapter"
        _write_adapter(
            adapter_path,
            {
                "base_model.model.model.layers.10.self_attn.q_a_proj."
                "lora_A.weight": torch.ones(1, 3),
                "base_model.model.model.layers.10.self_attn.q_a_proj."
                "lora_B.weight": torch.ones(4, 1),
            },
        )
        args = EngineArgs(
            enforce_eager=False,
            lora_modules=[f"adapter={adapter_path}"],
        )

        kwargs = args._get_engine_kwargs()

        assert kwargs["enforce_eager"] is False

    def test_empty_lora_modules_list_is_rejected(self):
        args = EngineArgs(lora_modules=[])

        with pytest.raises(ValueError, match="requires at least one adapter path"):
            args._get_engine_kwargs()

    def test_lora_modules_cli_requires_at_least_one_adapter(self):
        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)

        args = parser.parse_args(["--lora-modules", "adapter=/tmp/adapter"])
        assert args.lora_modules == ["adapter=/tmp/adapter"]

        with pytest.raises(SystemExit):
            parser.parse_args(["--lora-modules"])
