# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""What a suite's PR run can be seen to execute besides the product code,
derived from the repository itself: a check on the suite's path list
(test_ci_gate_suites.py), not its source, since what it cannot see (below) the
lists cover by erring on the side of running.

- how the product is installed: the packaging files and the readme
  pyproject.toml declares;
- the workflow file, and every file its PR-run steps name (a job or step that
  only runs on main does not run for a PR), followed through the files the run
  executes: scripts, Dockerfiles, local actions.

A file names another by a path from the repository root or from its own
directory (`$(dirname "${BASH_SOURCE[0]}")/x.sh`); every word that resolves
either way to a file here is taken as named, comment lines aside. A name
computed at run time (`"${kind}.sh"`) cannot be resolved. A file used as data
(a JSON, another workflow read for a pin) counts itself; what it names does not
run. Product code (the packages pyproject.toml declares) is left out: which of
it a suite tests is the suite's own choice.
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

import tomllib
import yaml

REPO = Path(__file__).resolve().parent.parent
PACKAGING_FILES = ("setup.py", "pyproject.toml", "MANIFEST.in")
_WORD = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*")
_ACTION_FILES = ("action.yml", "action.yaml")
# Files the run executes, whose own references are part of the run.
_EXECUTED_SUFFIXES = (".sh", ".py", "dockerfile", "Dockerfile")
_MAIN_ONLY = "refs/heads/main"


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def installed_package_dirs() -> list[str]:
    """Top-level directories of the packages pyproject.toml installs."""
    includes = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    return sorted({pattern.split(".", 1)[0] for pattern in includes})


def packaging_files() -> tuple[str, ...]:
    """What installing the product reads besides its code: the packaging
    files and the readme pyproject.toml declares."""
    readme = _pyproject()["project"].get("readme")
    if isinstance(readme, dict):
        readme = readme.get("file")
    return PACKAGING_FILES + ((readme,) if readme else ())


def _is_area(relative: Path) -> bool:
    """The repository, a top-level directory or one of .github's: naming one
    says where something is, not what runs."""
    parts = relative.parts
    return len(parts) <= 1 or (len(parts) == 2 and parts[0] == ".github")


@functools.cache
def tracked_files() -> frozenset[str]:
    """The repository's files: what git tracks."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return frozenset(name for name in listed.split("\0") if name)


@functools.cache
def _tracked_dirs() -> frozenset[str]:
    return frozenset(
        parent.as_posix()
        for name in tracked_files()
        for parent in Path(name).parents
        if parent != Path(".")
    )


def _in_repository(target: Path) -> str | None:
    """A path's name in the repository ('/'-terminated for a directory), or
    None when the repository has no such file or directory."""
    if not target.is_relative_to(REPO):
        return None
    relative = target.relative_to(REPO)
    if relative.as_posix() in tracked_files():
        return relative.as_posix()
    if relative.as_posix() in _tracked_dirs() and not _is_area(relative):
        return f"{relative.as_posix()}/"
    return None


def _from_root(name: str) -> str | None:
    return _in_repository((REPO / name).resolve())


def _resolve(word: str, here: Path) -> str | None:
    """The repository path a word names ('/'-terminated for a directory), or
    None. From the root, where the repository may be mounted anywhere
    (`/workspace/x`, `$GITHUB_WORKSPACE/x`): the word or what follows any '/'
    in it. From the naming file's own directory, a file
    (`$(dirname "$0")/x.sh`)."""
    parts = word.split("/")
    for start in range(len(parts)):
        found = _from_root("/".join(parts[start:]))
        if found is not None:
            return found
    sibling = _in_repository((here / word).resolve())
    return sibling if sibling is not None and not sibling.endswith("/") else None


def _named_in(text: str, here: Path = REPO) -> set[str]:
    """What a script, Dockerfile, action or step names outside comment lines."""
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    return {path for path in (_resolve(w, here) for w in _WORD.findall(code)) if path}


def _executed_files(path: str) -> list[Path]:
    """The files of a named path the run executes: a script or Dockerfile
    itself, a local action's action file."""
    target = REPO / path
    if not path.endswith("/"):
        return [target] if path.endswith(_EXECUTED_SUFFIXES) else []
    return [target / name for name in _ACTION_FILES if (target / name).is_file()]


def _product_dirs() -> tuple[str, ...]:
    return tuple(f"{package}/" for package in installed_package_dirs())


def _with_what_they_run(paths: set[str]) -> set[str]:
    """The paths and what the files among them that the run executes name,
    product code left out (what it names is the product's business)."""
    product = _product_dirs()
    found = {p for p in paths if not p.startswith(product)}
    pending = list(found)
    while pending:
        for file in _executed_files(pending.pop()):
            text = file.read_text(encoding="utf-8", errors="ignore")
            for path in _named_in(text, file.parent):
                if path not in found and not path.startswith(product):
                    found.add(path)
                    pending.append(path)
    return found


def _runs_on_pr(condition) -> bool:
    return _MAIN_ONLY not in str(condition or "")


def _local(uses: str) -> str:
    """A local action or reusable workflow's path; '' for a published one."""
    return uses.removeprefix("./") if uses.startswith("./") else ""


def run_definition(workflow_file: str) -> list[str]:
    """The repository paths a workflow's PR run executes or reads besides the
    product code; a directory ends with '/'. A step's env is not read: a
    gate's own path lists live there."""
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / workflow_file).read_text(encoding="utf-8")
    )
    named: set[str] = set()
    for job in workflow["jobs"].values():
        if not _runs_on_pr(job.get("if")):
            continue
        named |= _named_in(_local(job.get("uses", "")))
        for step in job.get("steps") or []:
            if _runs_on_pr(step.get("if")):
                named |= _named_in(_local(step.get("uses", "")))
                named |= _named_in(str(step.get("run", "")))
                named |= _named_in(str(step.get("with") or ""))
    return sorted(
        _with_what_they_run(named)
        | {f".github/workflows/{workflow_file}", *packaging_files()}
    )


def example_path(reference: str) -> str:
    """A file the reference stands for: itself, or one inside a directory."""
    return reference + "x" if reference.endswith("/") else reference
