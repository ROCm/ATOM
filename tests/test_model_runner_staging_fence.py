# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Source-level contracts for ModelRunner's staging-buffer lifetime.

Importing ModelRunner pulls in the GPU attention stack.  These tests inspect
the method AST instead so they remain part of the non-GPU pre-checkin suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODEL_RUNNER = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
EAGLE_PROPOSER = (
    Path(__file__).resolve().parents[1] / "atom/spec_decode/eagle_proposer.py"
)


def _method(
    name: str,
    *,
    path: Path = MODEL_RUNNER,
    class_name: str = "ModelRunner",
) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_calls(method: ast.FunctionDef, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


def _is_below_dummy_guard(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and "is_dummy_run" in ast.unparse(node.test):
            return True
    return False


def test_dummy_forward_participates_in_staging_lifetime():
    """DP-sync dummies must protect the buffers that they upload and reuse.

    An idle DP replica executes a dummy while its peer handles a real batch.
    The dummy uses the same pinned ``forward_vars`` and returns without a host
    synchronization.  Skipping either side of the lifetime protocol lets the
    following forward overwrite a still-copying dummy source buffer.
    """
    method = _method("forward")
    parents = _parent_map(method)
    expected_counts = {
        "_advance_forward_vars": 1,
        "_gate_staging_reuse": 1,
        "_mark_staging_h2d_enqueued": 1,
        "_record_forward_vars_event": 2,
    }

    calls_by_name = {name: _attribute_calls(method, name) for name in expected_counts}
    assert {
        name: len(calls) for name, calls in calls_by_name.items()
    } == expected_counts
    assert all(
        not _is_below_dummy_guard(call, parents)
        for calls in calls_by_name.values()
        for call in calls
    )

    prepare_model = _attribute_calls(method, "prepare_model")
    run_model = _attribute_calls(method, "run_model")
    assert len(prepare_model) == len(run_model) == 1
    assert (
        calls_by_name["_gate_staging_reuse"][0].lineno
        < prepare_model[0].lineno
        < calls_by_name["_mark_staging_h2d_enqueued"][0].lineno
        < run_model[0].lineno
    )


def test_late_draft_uploads_are_staged_by_prepare_model():
    """Draft setup must not start a new pinned H2D after the staging event."""
    prepare_model = _method("prepare_model")
    propose = _method("propose_draft_token_ids")
    compute_draft_kv = _method(
        "compute_draft_kv",
        path=EAGLE_PROPOSER,
        class_name="EagleProposer",
    )

    assert len(_attribute_calls(prepare_model, "anchors_to_gpu")) == 1
    assert len(_attribute_calls(prepare_model, "copy_to_gpu")) == 1
    assert not _attribute_calls(propose, "anchors_to_gpu")
    assert not _attribute_calls(propose, "copy_to_gpu")
    assert not _attribute_calls(compute_draft_kv, "anchors_to_gpu")
