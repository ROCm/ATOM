# SPDX-License-Identifier: MIT

import ast
from pathlib import Path

FORBIDDEN_DEFINITIONS = {
    "atom/model_ops/fused_moe/rccl_prepare_finalize.py": {
        "_combine_kernel",
        "_combine_weighted_scatter",
    },
    "atom/model_ops/fused_moe/rccl_batched_experts.py": {
        "_block_fp8_gemm_batched",
        "batched_block_fp8_mlp",
    },
    "atom/model_ops/fused_moe/flydsl_batched_gemm.py": {
        "is_available",
    },
    "atom/model_ops/fused_moe/flydsl_kernels/mxfp4_preshuffle.py": {
        "compile_mxfp6_gemm",
    },
    "atom/model_ops/fused_moe/flydsl_kernels/fp4_utils.py": {
        "random_e8m0",
        "random_fp4_packed",
        "fp8_e4m3_to_f32",
        "_moe_mxfp4_sort_kernel",
        "moe_mxfp4_sort",
        "pack_fp6_e2m3",
        "fp6_e2m3_to_f32",
        "per_1x32_f6_quant",
        "preshuffle_b_16x16",
    },
}

FORBIDDEN_LOGGING_SETUP = {
    "atom/model_ops/fused_moe/rccl_prepare_finalize.py",
    "atom/model_ops/fused_moe/rccl_batched_experts.py",
}


def _top_level_definitions(tree):
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_logging_import(tree):
    return any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "logging" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "logging")
        for node in tree.body
    )


def _has_logger_assignment(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "logger" for target in targets
        ):
            return True
    return False


def test_rccl_sources_do_not_reintroduce_removed_dead_code():
    remaining = {}
    for source, forbidden in FORBIDDEN_DEFINITIONS.items():
        tree = ast.parse(Path(source).read_text())
        found = _top_level_definitions(tree) & forbidden
        if found:
            remaining[source] = sorted(found)

    logging_setup = {}
    for source in FORBIDDEN_LOGGING_SETUP:
        tree = ast.parse(Path(source).read_text())
        found = []
        if _has_logging_import(tree):
            found.append("logging import")
        if _has_logger_assignment(tree):
            found.append("logger assignment")
        if found:
            logging_setup[source] = found

    assert not remaining, f"dead definitions remain: {remaining}"
    assert not logging_setup, f"unused logging setup remains: {logging_setup}"
