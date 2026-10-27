# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every PAGE-backed state copy implementation takes a descriptor slot.

The native LMCache MP restore always passes ``descriptor_slot=`` by keyword,
so an implementation left on the old signature fails only once a restore
reaches it, inside an exception handler that reports "restore failed".
"""

import ast
import pathlib

ATTENTIONS = pathlib.Path(__file__).parents[1] / "atom/model_ops/attentions"


def _implementations():
    for path in sorted(ATTENTIONS.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "execute_paged_state_copies"
            ):
                args = node.args.args + node.args.kwonlyargs
                yield path.relative_to(ATTENTIONS.parents[2]), node, args


def test_every_implementation_accepts_descriptor_slot():
    found = list(_implementations())
    assert len(found) >= 4
    missing = [
        f"{path}:{node.lineno}"
        for path, node, args in found
        if "descriptor_slot" not in {arg.arg for arg in args}
    ]
    assert missing == []
