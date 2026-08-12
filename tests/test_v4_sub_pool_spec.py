# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTENTION_DIR = ROOT / "atom" / "model_ops" / "attentions"
V4_SOURCE = ATTENTION_DIR / "deepseek_v4_attn.py"
RUNNER_SOURCE = ROOT / "atom" / "model_engine" / "model_runner.py"
CONFIG_SOURCE = ROOT / "atom" / "config.py"
ARG_UTILS_SOURCE = ROOT / "atom" / "model_engine" / "arg_utils.py"
FIELD = "state_ckpt_extra_entries"
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

    assert not _runtime_field_refs(
        keywords["extra_entries"], "state_checkpoint_interval_tokens"
    )
    assert not _runtime_field_refs(keywords["extra_entries"], "enable_prefix_caching")

    runner_tree = ast.parse(RUNNER_SOURCE.read_text())
    runner = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert _runtime_field_refs(init, ENV_FIELD)
    assert _runtime_field_refs(init, FIELD)
    handlers = [
        handler
        for handler in ast.walk(init)
        if isinstance(handler, ast.ExceptHandler)
        and isinstance(handler.type, ast.Name)
        and handler.type.id == "ValueError"
    ]
    assert len(handlers) == 1
    warnings = [
        call
        for call in ast.walk(init)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "logger"
        and call.func.attr == "warning"
        and any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and ENV_FIELD in child.value
            for child in ast.walk(call)
        )
    ]
    assert len(warnings) == 1


def test_checkpoint_extra_entries_has_no_config_or_cli_surface():
    assert "state_checkpoint_extra_entries" not in CONFIG_SOURCE.read_text()
    arg_utils = ARG_UTILS_SOURCE.read_text()
    assert "state_checkpoint_extra_entries" not in arg_utils
    assert "--state-checkpoint-extra-entries" not in arg_utils
