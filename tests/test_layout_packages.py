# SPDX-License-Identifier: MIT
"""The invariant `pool_layout/` and `token_layout/` both hold.

Each is a topic -- where a byte lives in the pools, where this step's tokens go
-- and each happens to be reachable without `aiter` or the rest of `atom`. That
second part is what makes them testable at all on a plain runner, and it is
worth a gate: CI has no AITER build, and one import failure during collection
aborts the whole run rather than one test, so the module that breaks it takes
thousands of unrelated tests with it.

A member may import a *sibling* member: the sibling is held to this same rule
by this same test, so the invariant is untouched while the package is free to
be several files. Reading every relative import as "leaves the tree" is simpler
to check and costs exactly that -- a declaration and the arithmetic over it
would have to share a file to share a type.

Read statically, by path. Importing the modules to inspect them would be a
weaker test -- it passes on the machine that has AITER -- and would not survive
the very breakage it is meant to catch.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ATTENTIONS = ROOT / "atom/model_ops/attentions"
PACKAGES = (ATTENTIONS / "pool_layout", ATTENTIONS / "token_layout")
MEMBERS = sorted(
    p for pkg in PACKAGES for p in pkg.glob("*.py") if p.name != "__init__.py"
)


def package_name(pkg: pathlib.Path) -> str:
    """Dotted name of a package, from where it sits under the repo root."""
    return ".".join(pkg.relative_to(ROOT).parts)


def imported_modules(path: pathlib.Path) -> set[str]:
    """Every module this file imports, by dotted name, nested ones included.

    `ast.walk` rather than a scan of the module body: a deferred import inside
    a function costs a plain runner nothing at import time but everything at
    call time, and the rule is about what the module can reach, not when.

    Relative imports resolve against this file's own package, so a sibling
    comes back under its real dotted name and anything reaching further out
    comes back rooted at `atom`. Telling those two apart is the whole point --
    they are the same syntax and not the same reach.
    """
    package = package_name(path.parent)
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            parent = (
                package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
            )
            modules.add(f"{parent}.{node.module}" if node.module else parent)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def is_sibling(module: str, package: str) -> bool:
    """Whether `module` names something directly inside `package`.

    A direct child only. A deeper dotted name is a subpackage, which this rule
    has never been asked about, so it stays outside and is judged as any other
    reach out of the tree. Written once because the two tests below need the
    same line drawn, from opposite sides of it.
    """
    return module.rpartition(".")[0] == package


def test_both_packages_are_populated():
    """A rule over an empty set passes for the wrong reason, and a package that
    lost its `__init__.py` would still glob."""
    for pkg in PACKAGES:
        assert (pkg / "__init__.py").is_file(), pkg
        assert list(pkg.glob("*.py")), pkg
    assert len(MEMBERS) >= 5, [p.name for p in MEMBERS]


@pytest.mark.parametrize("path", MEMBERS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_member_reaches_neither_aiter_nor_atom(path):
    package = package_name(path.parent)
    outside = (m for m in imported_modules(path) if not is_sibling(m, package))
    banned = {m.split(".")[0] for m in outside} & {"aiter", "atom"}
    assert not banned, (
        f"{path.parent.name}/{path.name} imports {sorted(banned)}, which this "
        f"package promises it does not -- move it beside the backend that needs "
        f"it, or drop the import"
    )


@pytest.mark.parametrize("path", MEMBERS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_sibling_import_names_a_member(path):
    """What makes a sibling import free is that the sibling obeys the rule too.

    Only a file this test collects does, so an import that resolves inside the
    package but names something else -- a subpackage, a stale name, `__init__`
    itself -- is not the case the exemption was granted for.
    """
    package = package_name(path.parent)
    for module in imported_modules(path):
        if not is_sibling(module, package):
            continue
        assert (path.parent / f"{module.rpartition('.')[2]}.py") in MEMBERS, (
            f"{path.parent.name}/{path.name} imports {module}, which is not a "
            f"member of {path.parent.name}/ this test holds to the rule"
        )


@pytest.mark.parametrize("pkg", PACKAGES, ids=lambda p: p.name)
def test_an_init_re_exports_nothing(pkg):
    """A convenience import would pull every member in whenever one is wanted,
    which is the cost the arrangement exists to avoid."""
    tree = ast.parse((pkg / "__init__.py").read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
