# SPDX-License-Identifier: MIT
import ast
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def view_type():
    # Execute the actual production class definition without importing the
    # surrounding GPU staging kernels. This checks Python access semantics,
    # not GPU event execution or AITER integration.
    path = Path(__file__).parents[3] / "atom/model_ops/engram/device/staging.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngramRowsView"
    )
    namespace = {"Mapping": Mapping, "__name__": "engram_view_cpu_test"}
    exec(  # noqa: S102 - only the checked-in class AST, never external input
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["EngramRowsView"]


@pytest.mark.parametrize("access", ["get", "index", "items", "values", "dict"])
def test_engram_view_value_access_always_waits_for_parent(access, view_type):

    calls = []

    class Parent(dict):
        def get(self, key, default=None):
            calls.append(key)
            # Model a lookup completing into the storage shared by the view.
            self[key].fill_(key)
            return super().get(key, default)

    parent = Parent({3: torch.zeros(1, 4, 2), 7: torch.zeros(1, 4, 2)})
    view = view_type(parent, slice(1, 3))
    assert list(view) == [3, 7]
    assert calls == []
    if access == "get":
        rows = [view.get(3), view.get(7)]
    elif access == "index":
        rows = [view[3], view[7]]
    elif access == "items":
        rows = [value for _, value in view.items()]
    elif access == "values":
        rows = list(view.values())
    else:
        rows = list(dict(view).values())
    assert calls == [3, 7]
    for key, row in zip((3, 7), rows):
        torch.testing.assert_close(row, torch.full((1, 2, 2), float(key)))
    assert view.get(99) is None
    with pytest.raises(KeyError):
        view[99]
    assert calls == [3, 7]
