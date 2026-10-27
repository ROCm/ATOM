# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
from pathlib import Path

ATOM_ROOT = Path(__file__).resolve().parents[1] / "atom"
# Every FP4 scorer takes its rows through `aiter.ops.flydsl`'s exports
# (`Fp4MqaRaggedMetadata`), prefill and decode alike, never a kernel module.
# Widen this only alongside a caller.
CURRENT_MODULES: set[str] = set()


def _fp4_aiter_imports() -> list[tuple[Path, int, str]]:
    imports = []
    for path in ATOM_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "pa_mqa_logits_fp4" in node.module
            ):
                imports.append((path, node.lineno, node.module))
    return imports


def test_dsv4_fp4_imports_use_current_aiter_package():
    imports = _fp4_aiter_imports()
    modules = {module for _, _, module in imports}

    unexpected_imports = [item for item in imports if item[2] not in CURRENT_MODULES]
    assert not unexpected_imports
    assert modules == CURRENT_MODULES
