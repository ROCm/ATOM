# SPDX-License-Identifier: MIT
"""Every field `_prepare` reads off a batch, the bridge's batches supply.

The bridge hands `DeepseekV41MetadataBuilder._prepare` a `SimpleNamespace`
rather than ATOM's own `ScheduledBatch`, so a field added to core's reader is
not a type error here -- it is an `AttributeError` at the first forward, on a
GPU, after a four-minute model load. That is what happened when the V4.1
report landed `is_final_chunk`:

    AttributeError: 'types.SimpleNamespace' object has no attribute
    'is_final_chunk'                   (profile_run, before any token)

and `replaying`, added by the same change, would have been the next one --
the reader orders them, so the first crash hides the second.

This reads both sides out of the source rather than running them, because the
batches are built deep inside a forward that needs vLLM and a pool. An AST
walk costs nothing and fails in CI, where the GPU boot does not run at all.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND = ROOT / "atom/model_ops/attentions/deepseek_v41/backend.py"
BRIDGE = ROOT / "atom/plugin/vllm/deepseek_v41_bridge.py"


def _fields_prepare_reads() -> set[str]:
    paths = (BACKEND, ROOT / "atom/plugin/vllm/v41_core_overrides.py")
    readers = ("_prepare", "_prepare_idle", "_prepare_step")
    return {
        n.attr
        for path in paths
        for fn in ast.walk(ast.parse(path.read_text()))
        if isinstance(fn, ast.FunctionDef) and fn.name in readers
        for n in ast.walk(fn)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "batch"
    }


def _bridge_batches() -> list[tuple[int, set[str]]]:
    """Every `SimpleNamespace` in the bridge that is a batch.

    Identified by `is_dummy_run`, which only a batch carries -- the bridge
    builds namespaces for a snapshot and a runner stub too, and neither goes
    anywhere near `_prepare`.
    """
    return [
        (node.lineno, {kw.arg for kw in node.keywords})
        for node in ast.walk(ast.parse(BRIDGE.read_text()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "SimpleNamespace"
        and any(kw.arg == "is_dummy_run" for kw in node.keywords)
    ]


def test_the_reader_and_the_batches_were_both_found():
    """Neither half may be empty: an AST query that matched nothing would
    pass every assertion below while checking nothing at all."""
    assert _fields_prepare_reads(), "found no batch fields in _prepare"
    batches = _bridge_batches()
    assert len(batches) >= 3, f"found only {len(batches)} fabricated batches"


@pytest.mark.parametrize("lineno,supplied", _bridge_batches())
def test_every_fabricated_batch_carries_what_prepare_reads(lineno, supplied):
    missing = _fields_prepare_reads() - supplied
    assert not missing, (
        f"the batch built at deepseek_v41_bridge.py:{lineno} is missing "
        f"{sorted(missing)}, which `_prepare` reads. A dummy batch needs them "
        f"too: `_prepare` reads some of them before `is_dummy_run`."
    )
