# SPDX-License-Identifier: MIT

"""Source-level coverage for dense Qwen3 QK-norm/RoPE/cache fusion wiring."""

import ast
from pathlib import Path

QWEN3_PATH = Path(__file__).resolve().parents[1] / "atom" / "models" / "qwen3.py"
SOURCE = QWEN3_PATH.read_text()
TREE = ast.parse(SOURCE)


def _attention_class():
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3Attention"
    )


def _method(name):
    return next(
        node
        for node in _attention_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_dense_qwen3_uses_canonical_fusion_flag():
    assert "from atom.utils import envs" in SOURCE
    assert "envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION" in SOURCE


def test_dense_qwen3_builds_joint_rope_cache_when_fused():
    init = _method("__init__")
    register_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register_buffer"
    )
    assert isinstance(register_call.args[0], ast.Constant)
    assert register_call.args[0].value == "cos_sin_cache"

    cat_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "cat"
    )
    assert isinstance(cat_call.args[0], ast.Tuple)
    assert [elt.id for elt in cat_call.args[0].elts] == ["cos", "sin"]


def test_dense_qwen3_passes_norms_into_attention():
    init = _method("__init__")
    attention_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Attention"
    )
    kwargs = {keyword.arg for keyword in attention_call.keywords}
    assert {"q_norm", "k_norm"} <= kwargs


def test_dense_qwen3_forwards_packed_qkv_on_fused_path():
    forward = _method("forward")
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "attn"
    ]
    fused_call = next(
        call for call in calls if any(keyword.arg == "qkv" for keyword in call.keywords)
    )
    kwargs = {keyword.arg for keyword in fused_call.keywords}
    assert {"query", "key", "value", "positions", "q_scale", "qkv"} <= kwargs


def test_dense_qwen3_keeps_unfused_fallback():
    forward_source = ast.unparse(_method("forward"))
    assert "q = self.q_norm(q)" in forward_source
    assert "k = self.k_norm(k)" in forward_source
