# SPDX-License-Identifier: MIT

import ast
from pathlib import Path

BATCHED_SELECTORS = {
    "batched",
    "flydsl_batched_gemm",
    "triton_batched_gemm",
}


def _function_source(path, name):
    source = Path(path).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name} not found in {path}")


def test_modular_batched_compute_is_mxfp4_only():
    source = _function_source("atom/model_ops/fused_moe/modular_kernel.py", "forward")
    assert "w1.dtype == dtypes.fp4x2" in source
    assert "batched_expert_compute" in source
    assert "fused_moe(" in source


def test_rccl_prepare_policy_is_format_specific():
    source = _function_source("atom/model_ops/moe.py", "_maybe_make_prepare_finalize")
    assert "quant_config.quant_dtype == dtypes.fp4x2" in source
    assert "use_batched_expert_compute=use_batched_expert_compute" in source
    assert "and not use_batched_impl" not in source


def test_fp8_weight_processing_does_not_skip_shuffle_for_batched_selector():
    source = Path("atom/model_ops/moe.py").read_text()
    marker = "The RCCL batched block-fp8 expert path"
    assert marker not in source
    assert "shuffle_weights(layer.w13_weight, layer.w2_weight)" in source


def test_rccl_metadata_policy_does_not_read_global_selector():
    source = Path("atom/model_ops/fused_moe/rccl_prepare_finalize.py").read_text()
    assert "self.use_batched_expert_compute" in source
    assert "ATOM_RCCL_MOE_IMPL" not in source
