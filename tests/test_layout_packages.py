# SPDX-License-Identifier: MIT
"""The invariant `pool_layout/` and `token_layout/` both hold.

Each is a topic -- where a byte lives in the pools, where this step's tokens go
-- and each happens to be reachable without `aiter` or the rest of `atom`. That
second part is what makes them testable at all on a plain runner, and it is
worth a gate: CI has no AITER build, and one import failure during collection
aborts the whole run rather than one test, so the module that breaks it takes
thousands of unrelated tests with it.

Read statically, by path. Importing the modules to inspect them would be a
weaker test -- it passes on the machine that has AITER -- and would not survive
the very breakage it is meant to catch.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ATTENTIONS = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_ops/attentions"
)
PACKAGES = (ATTENTIONS / "pool_layout", ATTENTIONS / "token_layout")
MEMBERS = sorted(
    p for pkg in PACKAGES for p in pkg.glob("*.py") if p.name != "__init__.py"
)


def imported_modules(path: pathlib.Path, package: str | None = None) -> set[str]:
    """Every module this file imports, by dotted name, nested ones included.

    `ast.walk` rather than a scan of the module body: a deferred import inside
    a function costs a plain runner nothing at import time but everything at
    call time, and the rule is about what the module can reach, not when.

    `package` is this file's own package, needed only to resolve relative
    imports; without it they resolve to `atom`, which is all the rule below
    needs and is the answer a relative import cannot escape anyway.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            if package is None:
                modules.add("atom")  # a relative import cannot leave the tree
            else:
                parent = (
                    package.rsplit(".", node.level - 1)[0]
                    if node.level > 1
                    else package
                )
                modules.add(f"{parent}.{node.module}" if node.module else parent)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def imported_roots(path: pathlib.Path) -> set[str]:
    """The top-level package of each of `imported_modules`."""
    return {m.split(".")[0] for m in imported_modules(path)}


def test_both_packages_are_populated():
    """A rule over an empty set passes for the wrong reason, and a package that
    lost its `__init__.py` would still glob."""
    for pkg in PACKAGES:
        assert (pkg / "__init__.py").is_file(), pkg
        assert list(pkg.glob("*.py")), pkg
    assert len(MEMBERS) >= 5, [p.name for p in MEMBERS]


@pytest.mark.parametrize("path", MEMBERS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_member_reaches_neither_aiter_nor_atom(path):
    banned = imported_roots(path) & {"aiter", "atom"}
    assert not banned, (
        f"{path.parent.name}/{path.name} imports {sorted(banned)}, which this "
        f"package promises it does not -- move it beside the backend that needs "
        f"it, or drop the import"
    )


@pytest.mark.parametrize("pkg", PACKAGES, ids=lambda p: p.name)
def test_an_init_re_exports_nothing(pkg):
    """A convenience import would pull every member in whenever one is wanted,
    which is the cost the arrangement exists to avoid."""
    tree = ast.parse((pkg / "__init__.py").read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
