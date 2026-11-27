# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI suites the evaluator writes a Status for, and which changes
each one tests.

A change is relevant to a suite when it can alter what the suite's PR run
executes. Outside the product code, which files a run executes cannot be
proven from the repository's text (a name can be computed at run time), so
the lists err on the side of running: every area a PR run may execute
(RUN_AREAS) counts for every suite, and a suite leaves a path out only by a
reviewed ignore entry. Of the product code, each suite takes the part it
tests. tests/test_ci_gate_suites.py classifies every file of the repository
and checks each list against what it can derive of each run.

While each suite's workflow still runs its old ci-gate job, the path lists
here repeat that job's CI_GATE_PATHS_IGNORE / CI_GATE_PATHS_INCLUDE;
tests/test_ci_gate_suites.py keeps the two equal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suite:
    key: str
    workflow_file: str
    workflow_name: str
    status_context: str
    result_job: str
    not_run_job: str
    stale_merge_job: str  # not run: the run's merge commit is not against main
    labels: frozenset[str]
    ignore_globs: tuple[str, ...]
    include_globs: tuple[str, ...]
    smoke_job: str | None
    # The pre-cutover workflow's gate: its signal job runs when the gate
    # decided heavy CI is required; its gate job is the verdict.
    old_signal_job: str
    old_gate_job: str


# What the repository ships: the packages pyproject.toml declares.
PRODUCT_DIRS = ("atom/",)

# Where, besides the product code, a suite's PR run may execute or read
# something: how the product is installed, the CI's scripts, actions, data and
# images. A workflow file is not here: it defines its own runs, so each suite
# lists its own and those its run reads.
RUN_AREAS = (
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",  # pyproject.toml's readme: installing reads it
    ".gitattributes",
    "scripts/**",
    "docker/**",
    ".github/scripts/**",
    ".github/actions/**",
    ".github/benchmark/**",
    ".github/runner-config.yml",
    ".github/workflows/golden_outputs/**",
)
# Docs beside the Python product code: nothing imports or installs them.
# (ATOMesh's tests build its README in: its suite ignores no docs.)
_PRODUCT_DOCS = "atom/*.md"


def _suite(
    key: str,
    workflow_file: str,
    workflow_name: str,
    title: str,
    labels: tuple[str, ...],
    *,
    own_code: tuple[str, ...],
    reads_workflows: tuple[str, ...] = (),
    ignore: tuple[str, ...],
    smoke_job: str | None = None,
    old_gate_job: str,
) -> Suite:
    workflows = tuple(
        f".github/workflows/{w}" for w in (workflow_file, *reads_workflows)
    )
    return Suite(
        key=key,
        workflow_file=workflow_file,
        workflow_name=workflow_name,
        status_context=f"{title} Status",
        result_job=f"{title} Result",
        not_run_job=f"{title} Result (not run)",
        stale_merge_job=f"{title} Result (not run: merge commit not against main)",
        labels=frozenset(labels),
        ignore_globs=ignore,
        include_globs=own_code + RUN_AREAS + workflows,
        smoke_job=smoke_job,
        old_signal_job="Check Pre Checkin Signal",
        old_gate_job=old_gate_job,
    )


SUITES: tuple[Suite, ...] = (
    _suite(
        "atom-test",
        "atom-test.yaml",
        "ATOM Test",
        "ATOM Test",
        ("ci:full", "ci:atom"),
        own_code=("atom/**",),
        ignore=(_PRODUCT_DOCS, "atom/plugin/**", "atom/mesh/**"),
        smoke_job="Offline inference smoke test",
        old_gate_job="ATOM Test Gate",
    ),
    _suite(
        "atom-sglang-test",
        "atom-sglang-test.yaml",
        "ATOM SGLang Test",
        "ATOM SGLang Test",
        ("ci:full", "ci:sglang"),
        own_code=("atom/plugin/**",),
        reads_workflows=("docker-release.yaml",),  # its SGLang pin
        ignore=(
            _PRODUCT_DOCS,
            "atom/plugin/vllm/**",
            "atom/plugin/rtpllm/**",
            ".github/scripts/plugin_ci/run_vllm_ci.sh",
            ".github/scripts/plugin_ci/vllm_slurm_job.sh",
        ),
        old_gate_job="ATOM SGLang Test Gate",
    ),
    _suite(
        "atom-vllm-test",
        "atom-vllm-test.yaml",
        "ATOM vLLM Test",
        "ATOM vLLM Test",
        ("ci:full", "ci:vllm"),
        own_code=("atom/plugin/**",),
        ignore=(
            _PRODUCT_DOCS,
            "atom/plugin/sglang/**",
            "atom/plugin/rtpllm/**",
            ".github/scripts/plugin_ci/run_sglang_ci.sh",
            ".github/scripts/plugin_ci/sglang_slurm_job.sh",
        ),
        old_gate_job="ATOM vLLM Test Gate",
    ),
    _suite(
        "atomesh-accuracy-validation",
        "atomesh-accuracy-validation.yaml",
        "Atomesh Accuracy Validation",
        "ATOMesh Accuracy",
        ("ci:full", "ci:mesh"),
        own_code=("atom/mesh/**",),
        ignore=(),  # its tests build README.md in (include_str!)
        old_gate_job="ATOMesh Accuracy Gate",
    ),
)

SUITES_BY_WORKFLOW_FILE = {suite.workflow_file: suite for suite in SUITES}
