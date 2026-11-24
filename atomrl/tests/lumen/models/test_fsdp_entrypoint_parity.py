# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Source-level parity tests for FSDP entrypoints."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

LLAMA31_FSDP = (
    REPO_ROOT.parent
    / "atomrl"
    / "lumen"
    / "models"
    / "llama31"
    / "fsdp"
    / "pretrain.py"
)
LLAMA2_FSDP = (
    REPO_ROOT.parent / "atomrl" / "lumen" / "models" / "llama2" / "fsdp" / "sft.py"
)


def _parse_module(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _get_function_calls(path: Path, function_name: str):
    tree = _parse_module(path)
    for node in ast.walk(tree):
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


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestFsdpParserParity:
    def test_llama31_get_args_uses_shared_parser(self):
        calls = _get_function_calls(LLAMA31_FSDP, "get_args")
        assert "add_common_fsdp_args" in calls

    def test_llama2_get_args_uses_shared_parser(self):
        calls = _get_function_calls(LLAMA2_FSDP, "get_args")
        assert "add_common_fsdp_args" in calls


class TestFsdpRuntimeParity:
    def test_llama31_uses_shared_scheduler_helper(self):
        calls = _get_function_calls(LLAMA31_FSDP, "_build_scheduler")
        assert "build_cosine_warmup_scheduler" in calls

    def test_llama2_uses_shared_scheduler_helper(self):
        calls = _get_function_calls(LLAMA2_FSDP, "_build_scheduler")
        assert "build_cosine_warmup_scheduler" in calls

    def test_llama31_syncs_scheduler_to_checkpoint_step(self):
        calls = _get_function_calls(LLAMA31_FSDP, "__init__")
        assert "sync_scheduler_to_ckpt_step" in calls
        assert "ExperimentTracker" in calls

    def test_llama2_syncs_scheduler_to_checkpoint_step(self):
        calls = _get_function_calls(LLAMA2_FSDP, "__init__")
        assert "sync_scheduler_to_ckpt_step" in calls
        assert "ExperimentTracker" in calls

    def test_llama31_uses_canonical_linear_fp8_flag(self):
        source = _source(LLAMA31_FSDP)
        assert "args.fp8_training" not in source
        assert 'getattr(args, "fp8_training"' not in source

    def test_llama2_uses_canonical_linear_fp8_flag(self):
        source = _source(LLAMA2_FSDP)
        assert "args.fp8_training" not in source
        assert 'getattr(args, "fp8_training"' not in source

    def test_llama2_train_runs_validation_via_eval_interval(self):
        calls = _get_function_calls(LLAMA2_FSDP, "train")
        assert "_validate" in calls
        assert "should_run_eval_step" in calls

    def test_llama2_forward_backward_scales_grad_accumulation(self):
        source = _source(LLAMA2_FSDP)
        assert "gradient_accumulation_steps" in source
        assert "loss / self.args.gradient_accumulation_steps" in source

    def test_llama31_train_uses_explicit_synthetic_warmup_step(self):
        calls = _get_function_calls(LLAMA31_FSDP, "train")
        assert "_synthetic_warmup_step" in calls

    def test_llama2_synthetic_warmup_does_not_reuse_scaled_train_backward(self):
        calls = _get_function_calls(LLAMA2_FSDP, "_synthetic_warmup_step")
        assert "_forward_backward" not in calls

    def test_llama31_references_checkpoint_contract_flags(self):
        source = _source(LLAMA31_FSDP)
        assert "args.use_ckpt" in source
        assert "args.save_ckpt" in source
        assert "args.continual_ckpt_path" in source
        assert "args.ckpt_start_step" in source

    def test_llama2_references_checkpoint_contract_flags(self):
        source = _source(LLAMA2_FSDP)
        assert "args.use_ckpt" in source
        assert "args.save_ckpt" in source
        assert "args.continual_ckpt_path" in source
        assert "args.ckpt_start_step" in source

    def test_llama2_early_stop_uses_direct_validation_threshold(self):
        source = _source(LLAMA2_FSDP)
        assert "_val_loss_ema" not in source
        assert "0.9 * self._val_loss_ema" not in source
        assert "def _check_early_stop" not in source

    def test_llama2_optimizer_uses_explicit_eps(self):
        source = _source(LLAMA2_FSDP)
        assert "eps=1e-5" in source

    def test_shared_fsdp_parser_does_not_expose_removed_primus_flags(self):
        source = _source(REPO_ROOT.parent / "atomrl" / "lumen" / "models" / "fsdp.py")
        assert "--primus-turbo-fp8-attention" not in source
        assert "--primus-turbo-mxfp8-attention" not in source
        assert "--dbg-attn-output" not in source

    def test_shared_fsdp_runtime_uses_eval_every_contract(self):
        source = _source(REPO_ROOT.parent / "atomrl" / "lumen" / "models" / "fsdp.py")
        assert "eval_every" in source
        assert "start_eval_at" in source

    def test_llama31_train_uses_shared_eval_helper(self):
        calls = _get_function_calls(LLAMA31_FSDP, "train")
        assert "should_run_eval_step" in calls
        assert "get_effective_stop_step" in calls
        assert "record_train_step_start" in calls
        assert "record_train_step_end" in calls
        assert "should_stop_on_validation" in calls
        assert "should_preempt" in calls

    def test_llama2_train_uses_effective_stop_step(self):
        calls = _get_function_calls(LLAMA2_FSDP, "train")
        assert "get_effective_stop_step" in calls
        assert "record_train_step_start" in calls
        assert "record_train_step_end" in calls
        assert "should_stop_on_validation" in calls
        assert "should_preempt" in calls
