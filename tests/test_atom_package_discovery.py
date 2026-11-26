from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_runtime_atom_subpackage_is_discovered_by_setuptools() -> None:
    packages = set(find_namespace_packages(include=["atom", "atom.*"]))

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["atom", "atom.*"]' in pyproject

    expected = {"atom"}
    for source_file in (REPO_ROOT / "atom").rglob("*.py"):
        if source_file.name == "__init__.py":
            continue
        relative_parent = source_file.parent.relative_to(REPO_ROOT)
        expected.add(".".join(relative_parent.parts))

    assert expected <= packages
