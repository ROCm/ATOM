# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for the shared Lumen training contract.

Covers:
  - Shared checkpoint and experiment flags reusable across Megatron and FSDP
  - FSDP training defaults required for parity with launcher behavior
  - FSDP legacy Docker/launcher FP8 flag aliases mapping onto the shared
    linear-FP8 contract
"""

import argparse
import ast
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT.parent / "atomrl" / "lumen" / "models"
BUILDERS_DIR = REPO_ROOT.parent / "atomrl" / "lumen" / "patches" / "builders"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_training_contract():
    lumen_pkg = sys.modules.setdefault("atomrl.lumen", types.ModuleType("atomrl.lumen"))
    lumen_pkg.__path__ = [str(REPO_ROOT.parent / "atomrl" / "lumen")]

    models_pkg = sys.modules.setdefault(
        "atomrl.lumen.models", types.ModuleType("atomrl.lumen.models")
    )
    models_pkg.__path__ = [str(MODELS_DIR)]

    if "atomrl.lumen.models.utils" not in sys.modules:
        _load_module("atomrl.lumen.models.utils", MODELS_DIR / "utils.py")

    return _load_module(
        "atomrl.lumen.models.training_contract", MODELS_DIR / "training_contract.py"
    )


def _parse_module(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _get_function_calls(tree: ast.AST, function_name: str):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name):
                        calls.append(func.id)
                    elif isinstance(func, ast.Attribute):
                        calls.append(func.attr)
            return calls
    return []


class TestSharedTrainingContractArgs:
    def _parse_checkpoint(self, cli_args=None):
        mod = _load_training_contract()
        parser = argparse.ArgumentParser()
        mod.add_shared_checkpoint_args(parser)
        return parser.parse_args(cli_args or [])

    def _parse_experiment(self, cli_args=None):
        mod = _load_training_contract()
        parser = argparse.ArgumentParser()
        mod.add_shared_experiment_args(parser)
        return parser.parse_args(cli_args or [])

    def _parse_fsdp(self, cli_args=None):
        mod = _load_training_contract()
        parser = argparse.ArgumentParser()
        mod.add_fsdp_contract_args(parser)
        return parser.parse_args(cli_args or [])

    def test_shared_checkpoint_args_present(self):
        args = self._parse_checkpoint([])
        assert args.use_ckpt is False
        assert args.save_ckpt is False
        assert args.resume_from_hf is False
        assert args.continual_ckpt_path is None
        assert args.ckpt_start_step == 0
        assert args.fp8_params is False
        assert args.initial_ckpt_path is None

    def test_shared_experiment_args_present(self):
        args = self._parse_experiment([])
        assert args.tag == ""
        assert args.target_log_ppl == 3.3
        assert args.step_time_atol == 18000
        assert args.eval_every == 0
        assert args.start_eval_at == 0

    def test_fsdp_training_defaults_include_eval_and_warmup(self):
        args = self._parse_fsdp([])
        assert args.lr_warmup_steps == 0
        assert args.eval_interval == 0

    def test_fsdp_gradient_checkpointing_defaults_enabled(self):
        args = self._parse_fsdp([])
        assert args.gradient_checkpointing is True

    def test_fsdp_no_gradient_checkpointing_alias(self):
        args = self._parse_fsdp(["--no-gradient-checkpointing"])
        assert args.gradient_checkpointing is False

    def test_fsdp_fp8_training_legacy_alias_sets_linear_fp8(self):
        args = self._parse_fsdp(["--fp8-training"])
        assert args.linear_fp8 is True

    def test_fsdp_linear_fp4_switch(self):
        args = self._parse_fsdp(["--linear-fp4"])
        assert args.linear_fp4 is True
        assert args.linear_fp8 is False

    def test_fsdp_fp8_format_legacy_alias_sets_linear_fp8_format(self):
        args = self._parse_fsdp(["--fp8-format", "hybrid"])
        assert args.linear_fp8_format == "hybrid"

    def test_fsdp_fp8_scaling_legacy_alias_sets_linear_fp8_scaling(self):
        args = self._parse_fsdp(["--fp8-scaling", "blockwise2d"])
        assert args.linear_fp8_scaling == "blockwise2d"

    def test_fsdp_no_fp8_activation_legacy_alias_sets_linear_fp8_activation(self):
        args = self._parse_fsdp(["--no-fp8-activation"])
        assert args.linear_fp8_activation is False

    def test_fsdp_contract_includes_shared_checkpoint_and_experiment_args(self):
        args = self._parse_fsdp([])
        assert args.use_ckpt is False
        assert args.save_ckpt is False
        assert args.resume_from_hf is False
        assert args.continual_ckpt_path is None
        assert args.ckpt_start_step == 0
        assert args.fp8_params is False
        assert args.initial_ckpt_path is None
        assert args.tag == ""
        assert args.target_log_ppl == 3.3
        assert args.step_time_atol == 18000
        assert args.eval_every == 0
        assert args.start_eval_at == 0


class TestSharedTrainingContractIntegration:
    def test_megatron_source_uses_shared_checkpoint_and_experiment_helpers(self):
        calls = _get_function_calls(
            _parse_module(BUILDERS_DIR / "megatron_args.py"),
            "add_common_megatron_args",
        )
        assert "add_shared_checkpoint_args" in calls
        assert "add_shared_experiment_args" in calls

    def test_fsdp_source_uses_grouped_contract_helpers(self):
        calls = _get_function_calls(
            _parse_module(MODELS_DIR / "fsdp.py"), "add_common_fsdp_args"
        )
        assert "add_fsdp_runtime_contract_args" in calls
        assert "add_fsdp_fp8_contract_args" in calls
        assert "add_shared_checkpoint_args" in calls
        assert "add_shared_experiment_args" in calls

    def test_fsdp_source_stops_referencing_legacy_fp8_training_attr(self):
        fsdp_source = (MODELS_DIR / "fsdp.py").read_text(encoding="utf-8")
        assert 'getattr(args, "fp8_training"' not in fsdp_source
