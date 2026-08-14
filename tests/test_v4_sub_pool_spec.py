# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTENTION_DIR = ROOT / "atom" / "model_ops" / "attentions"
V4_SOURCE = ATTENTION_DIR / "deepseek_v4_attn.py"
SUB_POOL_SPEC_SOURCE = ATTENTION_DIR / "sub_pool_spec.py"
CONFIG_SOURCE = ROOT / "atom" / "config.py"
ARG_UTILS_SOURCE = ROOT / "atom" / "model_engine" / "arg_utils.py"
ENV_FIELD = "STATE_CKPT_EXTRA_ENTRIES"


def _runtime_field_refs(node: ast.AST, field: str) -> list[ast.AST]:
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


def test_checkpoint_extra_entries_override_is_owned_by_state_pool():
    spec_tree = ast.parse(SUB_POOL_SPEC_SOURCE.read_text())
    state_pool_builder = next(
        node
        for node in spec_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "state_pool"
    )
    assert _runtime_field_refs(state_pool_builder, ENV_FIELD)
    override_guards = [
        call
        for call in ast.walk(state_pool_builder)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "is_set"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == ENV_FIELD
    ]
    assert len(override_guards) == 1

    tree = ast.parse(V4_SOURCE.read_text())
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
    assert "extra_entries" not in keywords


def test_checkpoint_extra_entries_has_no_config_or_cli_surface():
    assert "state_checkpoint_extra_entries" not in CONFIG_SOURCE.read_text()
    arg_utils = ARG_UTILS_SOURCE.read_text()
    assert "state_checkpoint_extra_entries" not in arg_utils
    assert "--state-checkpoint-extra-entries" not in arg_utils
