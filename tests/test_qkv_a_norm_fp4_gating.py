# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""When DeepSeek MLA's default fused Q/K norm runs aiter's qkv_a_rmsnorm_fp4.

The gate and the call sites are read from the model source on CPU, without
importing the GPU-backed model; kernel bytes and decode timing need gfx1250.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "atom/models/deepseek_v2.py"
TREE = ast.parse(SRC.read_text(encoding="utf-8"))


def _function(name, cls=None):
    scope = TREE
    if cls is not None:
        scope = next(
            n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    return next(
        n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name
    )


@pytest.fixture
def gate():
    asked = []
    served = {"value": True}
    envs = SimpleNamespace(ATOM_USE_MODEL_SENSITIVE_RMSNORM=False)

    def qkv_a_rmsnorm_fp4_supported(q_dim, kv_dim):
        asked.append((q_dim, kv_dim))
        return served["value"]

    function = _function("_qkv_a_norm_fp4_supported")
    function.decorator_list = []
    namespace = {
        "nn": SimpleNamespace(Module=object),
        "envs": envs,
        "qkv_a_rmsnorm_fp4_supported": qkv_a_rmsnorm_fp4_supported,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SRC), "exec"), namespace)  # noqa: S102 - repository code
    return namespace["_qkv_a_norm_fp4_supported"], asked, served, envs


def _attn(**overrides):
    return SimpleNamespace(
        **{"q_lora_rank": 1536, "kv_lora_rank": 512, "indexer": None, **overrides}
    )


def test_aiter_decides_the_widths(gate):
    supported, asked, served, _ = gate
    assert supported(_attn()) and asked == [(1536, 512)]
    served["value"] = False
    assert not supported(_attn(q_lora_rank=1152, kv_lora_rank=504))
    assert asked[-1] == (1152, 504)


@pytest.mark.parametrize("overrides", [{"q_lora_rank": None}, {"indexer": object()}])
def test_refused_without_q_lora_or_with_an_indexer(gate, overrides):
    supported, asked, _, _ = gate
    assert not supported(_attn(**overrides)) and not asked


def test_model_sensitive_rmsnorm_keeps_the_existing_path(gate):
    supported, _, _, envs = gate
    envs.ATOM_USE_MODEL_SENSITIVE_RMSNORM = True
    assert not supported(_attn())


def _source(node):
    return ast.unparse(node)


def test_only_the_default_fused_path_takes_it():
    """`__init__` arms it only with a Q/K norm fusion on, and `forward` calls it
    only for a bf16 q_c feeding a 32x4-scale MXFP4 q_b_proj, ahead of the
    default fused norm."""
    init = _source(_function("__init__", "DeepseekV2MLAAttention"))
    assert (
        "self.qkv_a_norm_fp4 = (self.fuse_qknorm_quant or self.fuse_qknorm) "
        "and _qkv_a_norm_fp4_supported(self)" in init
    )
    forward = _function("forward", "DeepseekV2MLAAttention")
    calls = [
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.If) and "_qkv_a_rmsnorm_fp4(" in _source(n.body[0])
    ]
    assert len(calls) == 1, "one call site"
    test = _source(calls[0].test)
    for clause in (
        "self.qkv_a_norm_fp4",
        "q_c.dtype == torch.bfloat16",
        "uses_mxfp4_n32k4_scale(self.q_b_proj)",
    ):
        assert clause in test, clause
    assert (
        _source(calls[0].orelse[0].test) == "self.fuse_qknorm_quant or self.fuse_qknorm"
    )
