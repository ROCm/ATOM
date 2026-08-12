# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-safe contract test for DeepSeek-V4's state-pool declaration.

The V4 attention module imports AITER and Triton kernels at module import time,
so importing its builder would either fail or make this key sizing regression
silently skip on the CPU test gate.  Inspect only the small declaration method
instead: arithmetic remains covered behaviorally in test_sub_pool_spec.py.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTENTION_DIR = ROOT / "atom" / "model_ops" / "attentions"
V4_SOURCE = ATTENTION_DIR / "deepseek_v4_attn.py"
FIELD = "STATE_CKPT_EXTRA_ENTRIES"


def _runtime_field_refs(node: ast.AST, field: str) -> list[ast.AST]:
    """Runtime references to an environment field, excluding prose."""
    refs: list[ast.AST] = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and child.attr == field
    ]
    refs.extend(
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "getattr"
        and len(child.args) >= 2
        and isinstance(child.args[1], ast.Constant)
        and child.args[1].value == field
    )
    return refs


def test_checkpoint_extra_entries_are_wired_only_to_the_v4_state_slot():
    users = []
    trees = {}
    for path in sorted(ATTENTION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        trees[path] = tree
        if _runtime_field_refs(tree, FIELD):
            users.append(path.name)
    assert users == [V4_SOURCE.name]

    tree = trees[V4_SOURCE]
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4AttentionMetadataBuilder"
    )
    method = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "sub_pool_specs"
    )
    state_calls = [
        call
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "state_pool"
        and call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "STATE_SLOT_CLASS"
    ]
    assert len(state_calls) == 1

    keywords = {kw.arg: kw.value for kw in state_calls[0].keywords}
    assert ast.literal_eval(keywords["entries_per_req"]) == 1
    assert _runtime_field_refs(keywords["extra_entries"], FIELD)

    # This is an explicit physical-sizing knob: disabling prefix caching or
    # the checkpoint interval must not silently change the allocation layout.
    assert not _runtime_field_refs(
        keywords["extra_entries"], "state_checkpoint_interval_tokens"
    )
    assert not _runtime_field_refs(keywords["extra_entries"], "enable_prefix_caching")
