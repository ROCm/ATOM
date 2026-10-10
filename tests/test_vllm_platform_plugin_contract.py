# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pin what ``register_platform()`` is allowed to do.

vLLM calls it while ``import vllm`` is still running its own module bodies, so a
deeper vLLM import raises on a half-initialised module and the resolve loop's
``except Exception: pass`` reads that as "plugin not activated" -- after which
``ATOMPlatform`` is never live, ``check_and_update_config`` never runs, and the
only symptom is DeepSeek-V4 answering a cross-request prefix hit from
sliding-window state it never built. See ``register_platform()``'s own note.

Checked statically: reproducing it needs vLLM and a GPU host, and the CI unit
scope has neither (``run_unit_tests.sh`` passes ``--ignore=tests/plugin``). The
thing that actually regressed -- a registration call added to that function --
is visible in the source, which is enough.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REGISTER_PY = (
    Path(__file__).resolve().parents[1] / "atom" / "plugin" / "vllm" / "register.py"
)

# Calls `register_platform()` may make. Measured safe there, and each is needed
# before vLLM invokes the platform's config hook.
ALLOWED_CALLS = {
    "apply_vllm_rocm_dcp_full_graph_patch",
    "apply_vllm_tcp_store_patch",
    "_register_hf_configs",
}


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(REGISTER_PY.read_text(), filename=str(REGISTER_PY))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {REGISTER_PY}")


def _direct_calls(fn: ast.FunctionDef) -> set[str]:
    # Plain-name calls only; `logger.info(...)` and friends are attribute calls.
    return {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_register_platform_makes_no_unvetted_calls():
    extra = _direct_calls(_function("register_platform")) - ALLOWED_CALLS
    assert not extra, (
        f"register_platform() gained {sorted(extra)}. It runs mid-`import vllm`, "
        "where a deeper vLLM import is circular and a read of current_platform "
        "re-enters platform resolution -- either one silently costs ATOMPlatform, "
        "and with it DeepSeek-V4's prefix-cache SWA rollback. Call it from "
        "register_model() instead."
    )


def test_register_platform_imports_no_vllm_beyond_platforms():
    imported: list[str] = []
    for node in ast.walk(_function("register_platform")):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = [
        name
        for name in imported
        if name.split(".")[0] == "vllm" and not name.startswith("vllm.platforms")
    ]
    assert not offenders, (
        f"register_platform() imports {sorted(offenders)}. Those modules are only "
        "half-initialised while vLLM resolves its platform, so the import raises "
        "and the resolve loop drops ATOM's platform."
    )


@pytest.mark.parametrize(
    "call",
    [
        "_register_mxfp8_quantization_config",
        "_register_kv_connectors",
        # The tripwire for the failure above, which nothing else reports.
        "_warn_if_atom_platform_not_live",
    ],
)
def test_register_model_still_makes_the_moved_calls(call):
    """Moved out of ``register_platform()``, not dropped.

    Both registrations were measured to lose the platform there, so leaving
    ``register_platform()`` is only half the fix -- they still have to run.
    """
    model_calls = _direct_calls(_function("register_model"))
    assert call in model_calls, f"{call}() is no longer called from register_model()."
