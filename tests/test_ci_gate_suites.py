# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI gate's suite table agrees with the workflows it describes."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from ci_gate_relevance import (
    example_path,
    installed_package_dirs,
    run_definition,
    tracked_files,
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
sys.path.insert(0, str(REPO / ".github" / "scripts"))

from ci_gate.decide import ChangedPaths, needs_suite
from ci_gate.evaluate import RELAY_TITLE_PREFIX, RELAY_WORKFLOW_FILE
from ci_gate.suites import PRODUCT_DIRS, RUN_AREAS, SUITES


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _gate_env(workflow: dict) -> dict:
    env: dict = {}
    for step in workflow["jobs"]["ci-gate"]["steps"]:
        env.update(step.get("env") or {})
    return env


def _lines(text: str) -> set[str]:
    return {line.strip() for line in (text or "").splitlines() if line.strip()}


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.key)
def test_suite_matches_its_workflow(suite):
    workflow = _workflow(suite.workflow_file)
    env = _gate_env(workflow)

    assert workflow["name"] == suite.workflow_name
    assert set(env["CI_GATE_LABELS"].split(",")) == set(suite.labels)
    assert _lines(env.get("CI_GATE_PATHS_IGNORE", "")) == set(suite.ignore_globs)
    assert _lines(env.get("CI_GATE_PATHS_INCLUDE", "")) == set(suite.include_globs)
    text = (WORKFLOWS / suite.workflow_file).read_text(encoding="utf-8")
    assert suite.old_gate_job in text
    assert f"name: {suite.old_signal_job}" in text


# ---- relevance: a suite runs for every change that can affect its PR run ----

SUITE = {s.key: s for s in SUITES}
ATOM_TEST = SUITE["atom-test"]

# The code each suite's PR run tests. Every PR run installs the whole `atom`
# package; the plugin and ATOMesh suites leave the rest of it to ATOM Test and
# the runs on main.
OWN_CODE = {
    "atom-test": [f"{package}/__init__.py" for package in installed_package_dirs()],
    "atom-sglang-test": ["atom/plugin/x.py", "atom/plugin/sglang/x.py"],
    "atom-vllm-test": ["atom/plugin/x.py", "atom/plugin/vllm/x.py"],
    "atomesh-accuracy-validation": ["atom/mesh/src/x.rs", "atom/mesh/README.md"],
}
ATOM_CORE = ["atom/models/x.py", "atom/model_ops/x.py", "atom/model_engine/x.py"]

# What no suite's PR run executes or reads, each with why. Every file of the
# repository is product code, in RUN_AREAS, a workflow file, or here.
NOT_RUN = (
    "tests/**",  # CPU unit tests, run by their own workflows
    "docs/**",
    "tools/**",  # developer tools
    "recipes/**",  # deployment guides
    "atomrl/**",  # its own package and CI
    ".claude/**",  # agent instructions
    "CLAUDE.md",
    ".github/CODEOWNERS",  # review rules, read from the base branch
    "LICENSE",
    ".readthedocs.yaml",
    ".gitignore",
    ".github/dashboard/**",  # served pages
    ".github/pages/**",
    ".github/actionlint.yaml",  # lint configuration
)


def _relevant(suite, path: str) -> bool:
    return needs_suite(ChangedPaths((path,), complete=True), suite)


def _matches(path: str, globs) -> bool:
    return _relevant(
        replace(ATOM_TEST, include_globs=tuple(globs), ignore_globs=()), path
    )


def test_every_suite_has_its_own_code_listed():
    assert set(OWN_CODE) == set(SUITE)


def test_product_code_is_what_pyproject_installs():
    assert set(PRODUCT_DIRS) == {f"{p}/" for p in installed_package_dirs()}


def test_every_file_is_classified_once():
    classes = {
        "product": tuple(f"{d}**" for d in PRODUCT_DIRS),
        "run area": RUN_AREAS + (".github/workflows/*",),
        "not run": NOT_RUN,
    }
    wrong = {}
    for path in sorted(tracked_files()):
        found = [name for name, globs in classes.items() if _matches(path, globs)]
        # A workflow's golden outputs are in both run-area globs.
        if len(set(found)) != 1:
            wrong[path] = found
    assert not wrong, f"classify these (ci_gate/suites.py or NOT_RUN): {wrong}"


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.key)
def test_suite_covers_everything_its_pr_run_is_seen_to_use(suite):
    # The derivation cannot see every name a run computes, so it does not
    # define the lists; it checks them.
    required = [example_path(p) for p in run_definition(suite.workflow_file)]
    required += OWN_CODE[suite.key]
    missing = [path for path in required if not _relevant(suite, path)]
    assert not missing, f"{suite.key} would not run for changes to {missing}"


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.key)
def test_no_run_uses_what_is_classified_as_not_run(suite):
    used = [p for p in run_definition(suite.workflow_file) if _matches(p, NOT_RUN)]
    assert not used, f"{suite.key}'s run uses {used}: not in NOT_RUN"


@pytest.mark.parametrize(
    "suite", [s for s in SUITES if s is not ATOM_TEST], ids=lambda s: s.key
)
@pytest.mark.parametrize("path", ATOM_CORE)
def test_atom_core_is_left_to_atom_test(suite, path):
    assert _relevant(ATOM_TEST, path)
    assert not _relevant(suite, path)


def test_the_derivation_finds_what_the_runs_use():
    # Guards the checks above against a derivation that misses a way a file
    # names another.
    atom_test = set(run_definition(ATOM_TEST.workflow_file))
    assert {
        ".github/scripts/atom_test.sh",
        ".github/scripts/check_heavy_ci_gate.sh",  # ./.github/...
        ".github/scripts/download_model_with_lock.sh",  # /workspace/.github/...
        "scripts/wait_infer_drain.sh",  # outside .github, named by atom_test.sh
        ".github/workflows/golden_outputs/",
        ".github/actions/docker-auth/",
        "setup.py",
    } <= atom_test
    # A step that only runs on main is not part of a PR run; product code is
    # not part of a run's definition.
    assert ".github/scripts/accuracy_to_dashboard.py" not in atom_test
    assert not any(p.startswith(PRODUCT_DIRS) for p in atom_test)
    # A word naming a directory beside a script is not a file the run uses.
    assert ".github/scripts/atomesh/" not in atom_test

    sglang = set(run_definition(SUITE["atom-sglang-test"].workflow_file))
    # Named by a script the workflow runs, relative to that script:
    # "$(dirname -- "${BASH_SOURCE[0]}")/supervise_client.sh".
    assert ".github/scripts/supervise_client.sh" in sglang
    assert ".github/scripts/plugin_ci/cache_model.sh" in sglang
    assert "docker/sglang_release.dockerfile" in sglang
    # A file read as data (for a pin) is used; what it names is not run.
    assert ".github/workflows/docker-release.yaml" in sglang
    assert ".github/workflows/lmcache-rocm-wheel.yaml" not in sglang

    vllm = set(run_definition(SUITE["atom-vllm-test"].workflow_file))
    assert ".github/scripts/read_mtp_metric.py" in vllm  # "${BASH_SOURCE%/*}/..."
    assert "docker/patches/" in vllm
    # Named only in a Dockerfile comment; not a repository file.
    assert ".github/workflows/lmcache-rocm-wheel.yaml" not in vllm
    assert ".git" not in vllm


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.key)
@pytest.mark.parametrize(
    "path",
    [
        "tests/test_kda_midstep_checkpoints.py",  # PR 3879: a CPU unit test
        "docs/architecture_guide.md",
        "tools/x.py",
        "recipes/DeepSeek-V4.1-Flash.md",
        "atomrl/x.py",
        ".github/workflows/notify-teams.yml",  # another workflow's runs
        ".github/dashboard/index.html",
    ],
)
def test_no_suite_runs_for_what_no_pr_run_executes(suite, path):
    assert not _relevant(suite, path)


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.key)
@pytest.mark.parametrize(
    "path",
    [
        "setup.py",
        "scripts/x.sh",  # even one no run is seen to use: a name may be computed
        "docker/Dockerfile",
        ".github/scripts/x.sh",
        ".github/actions/x/action.yml",
    ],
)
def test_every_suite_runs_for_the_areas_a_pr_run_may_execute(suite, path):
    assert _relevant(suite, path)


@pytest.mark.parametrize("path", ["atom/plugin/vllm/x.py", "atom/mesh/x.rs"])
def test_atom_test_leaves_the_plugins_and_atomesh_to_their_suites(path):
    assert not _relevant(ATOM_TEST, path)


def test_evaluator_listens_to_every_suite_and_the_relay():
    evaluator = _workflow("ci-evaluator.yaml")
    listened = set(evaluator[True]["workflow_run"]["workflows"])
    relay = _workflow(RELAY_WORKFLOW_FILE)

    assert {s.workflow_name for s in SUITES} | {relay["name"]} == listened
    assert relay["run-name"].startswith(RELAY_TITLE_PREFIX)


def test_status_contexts_are_distinct_from_existing_checks():
    contexts = [s.status_context for s in SUITES]
    assert len(set(contexts)) == len(contexts)
    assert not {s.old_gate_job for s in SUITES} & set(contexts)
