# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Source-level parity tests for Megatron example entrypoints."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

LLAMA31_EXAMPLE = REPO_ROOT / "lumen" / "examples" / "llama31" / "pretrain_llama31.py"
LLAMA2_EXAMPLE = REPO_ROOT / "lumen" / "examples" / "llama2" / "finetune_llama2.py"
LLAMA31_MEGATRON = (
    REPO_ROOT / "atomrl" / "lumen" / "models" / "llama31" / "megatron" / "pretrain.py"
)
LLAMA2_MEGATRON = (
    REPO_ROOT / "atomrl" / "lumen" / "models" / "llama2" / "megatron" / "sft.py"
)
LLAMA_BUILDERS = REPO_ROOT / "atomrl" / "lumen" / "patches" / "builders" / "llama.py"
SHARED_MEGATRON = REPO_ROOT / "atomrl" / "lumen" / "models" / "megatron.py"

MOVED_SHARED_FLAGS = {
    "--use-ckpt",
    "--save-ckpt",
    "--resume-from-hf",
    "--continual-ckpt-path",
    "--ckpt-start-step",
    "--fp8-params",
    "--initial-ckpt-path",
    "--tag",
    "--target-log-ppl",
    "--step-time-atol",
    "--eval-every",
    "--start-eval-at",
}


def _parse_module(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _get_function_calls(path: Path, function_name: str):
    tree = _parse_module(path)
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


def _string_literals(path: Path):
    tree = _parse_module(path)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _get_call_keywords(path: Path, function_name: str, callee_name: str):
    tree = _parse_module(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = (
                        func.id
                        if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else None
                    )
                    if name == callee_name:
                        return {kw.arg for kw in child.keywords if kw.arg is not None}
    return set()


def _get_apply_training_patch_names(path: Path, function_name: str):
    tree = _parse_module(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None
                )
                if name != "apply_training_patches":
                    continue
                for kw in child.keywords:
                    if kw.arg == "names" and isinstance(kw.value, ast.Set):
                        return {
                            elt.value
                            for elt in kw.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        }
    return set()


class TestMegatronExampleParity:
    def test_llama31_uses_shared_model_provider_helper(self):
        calls = _get_function_calls(LLAMA31_EXAMPLE, "_run_megatron")
        assert "make_lumen_model_provider" in calls

    def test_llama2_uses_shared_model_provider_helper(self):
        calls = _get_function_calls(LLAMA2_EXAMPLE, "_run_megatron")
        assert "make_lumen_model_provider" in calls

    def test_llama31_installs_shared_fp8_param_hook(self):
        names = _get_apply_training_patch_names(LLAMA31_EXAMPLE, "_run_megatron")
        assert "fp8_param_gather_hook" in names

    def test_llama2_installs_shared_fp8_param_hook(self):
        names = _get_apply_training_patch_names(LLAMA2_EXAMPLE, "_run_megatron")
        assert "fp8_param_gather_hook" in names
        assert "fp8_param_storage_hook" in names
        assert "hip_graphs_hook" in names
        assert "val_loss_early_stop_hook" in names

    def test_llama31_installs_mxfp4_weight_cache_hook(self):
        # Selected by registry name now, not by a direct install_* call: without
        # it the MXFP4 weight cache is never invalidated and the run trains
        # against the step-0 weights, which raises nothing.
        names = _get_apply_training_patch_names(LLAMA31_EXAMPLE, "_run_megatron")
        assert "mxfp4_weight_cache_hook" in names

    def test_llama2_installs_mxfp4_weight_cache_hook(self):
        names = _get_apply_training_patch_names(LLAMA2_EXAMPLE, "_run_megatron")
        assert "mxfp4_weight_cache_hook" in names


class TestLlama31MegatronArgsParity:
    def test_llama31_add_pretrain_args_drops_moved_shared_flags(self):
        literals = _string_literals(LLAMA31_MEGATRON)
        assert not (MOVED_SHARED_FLAGS & literals)

    def test_llama31_keeps_docker_compat_only_flags(self):
        literals = _string_literals(LLAMA_BUILDERS)
        assert "--size" in literals
        assert "--nodes" in literals
        assert "--gpus-per-node" in literals


def _get_nested_function_args(path: Path, outer_name: str, inner_name: str):
    tree = _parse_module(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == outer_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == inner_name:
                    spec = child.args
                    return {
                        a.arg for a in spec.args + spec.posonlyargs + spec.kwonlyargs
                    }
    return set()


class TestSharedMegatronProviderParity:
    def test_provider_accepts_every_kwarg_megatron_passes(self):
        # Megatron's get_model calls the provider by keyword, so a kwarg it grows
        # is a TypeError at model build time rather than anything caught earlier.
        params = _get_nested_function_args(
            SHARED_MEGATRON, "make_lumen_model_provider", "model_provider"
        )
        assert {
            "pre_process",
            "post_process",
            "vp_stage",
            "config",
            "pg_collection",
        } <= params

    def test_shared_provider_forwards_parallel_linear_state(self):
        keywords = _get_call_keywords(
            SHARED_MEGATRON,
            "make_lumen_model_provider",
            "enable_fp8_for_parallel_linear",
        )
        assert "fp8_mha" in keywords
        assert "gradient_accumulation_fusion" in keywords
        assert "delay_wgrad" in keywords


class TestLlama2MegatronArgsParity:
    def test_llama2_add_finetune_args_drops_redundant_warmup_flags(self):
        literals = _string_literals(LLAMA2_MEGATRON)
        assert "--warmup-steps" not in literals
        assert "--val-loss-target" not in literals
