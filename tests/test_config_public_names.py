# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Every `from atom.config import X` in the tree must resolve at runtime.

`atom.config` is a hub: it re-exports names it imports from elsewhere, so a
module can import something from it that it merely passes through. When such a
pass-through moves -- or becomes a `TYPE_CHECKING`-only import, which binds
nothing at runtime -- the importer breaks with

    ImportError: cannot import name 'QuantType' from 'atom.config'

and nothing catches it. The importers are mostly under `atom/models/`, which
the CPU test gate never imports because they need the AITER kernel build, so
the break surfaces only when someone loads that model on a GPU host.

This check is pure AST: it reads the source and imports nothing, so it runs on
the gate that cannot import the modules it is protecting.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "atom" / "config.py"


def _runtime_names(tree: ast.Module) -> set[str]:
    """Names `atom.config` binds when actually executed.

    Only `tree.body` -- the module's top level. Anything nested in an
    `if TYPE_CHECKING:` block lives in that `If` node's body instead, and is
    deliberately not counted: it exists for type checkers and is absent at
    run time, which is exactly the failure this guards.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _config_importers() -> list[tuple[pathlib.Path, int, str]]:
    """Every `(file, line, name)` imported from `atom.config` across the tree."""
    found = []
    for path in sorted(REPO_ROOT.glob("atom/**/*.py")) + sorted(
        REPO_ROOT.glob("tests/**/*.py")
    ):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "atom.config":
                for alias in node.names:
                    if alias.name != "*":
                        found.append((path, node.lineno, alias.name))
    return found


def test_config_provides_every_name_imported_from_it():
    available = _runtime_names(ast.parse(CONFIG_PATH.read_text()))
    importers = _config_importers()
    assert importers, "found no `from atom.config import ...` -- check the glob"

    missing = [
        f"{path.relative_to(REPO_ROOT)}:{lineno} imports {name!r}"
        for path, lineno, name in importers
        if name not in available
    ]
    assert not missing, (
        "atom/config.py does not bind these at run time:\n  "
        + "\n  ".join(missing)
        + "\n(a TYPE_CHECKING-only import does not count -- import the name "
        "from where it actually lives instead)"
    )


@pytest.mark.parametrize("name", ["Config", "LayerQuantConfig"])
def test_known_reexports_are_detected(name):
    """Guards the detector itself: these are re-exports, not local definitions."""
    assert name in _runtime_names(ast.parse(CONFIG_PATH.read_text()))
