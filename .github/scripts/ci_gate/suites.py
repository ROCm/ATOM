# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI suites the evaluator writes a Status for.

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


def _suite(
    key: str,
    workflow_file: str,
    workflow_name: str,
    title: str,
    labels: tuple[str, ...],
    *,
    ignore: tuple[str, ...] = (),
    include: tuple[str, ...] = (),
    smoke_job: str | None = None,
    old_gate_job: str,
) -> Suite:
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
        include_globs=include,
        smoke_job=smoke_job,
        old_signal_job="Check Pre Checkin Signal",
        old_gate_job=old_gate_job,
    )


# What the repository ships: the packages pyproject.toml declares. What else
# a suite's run executes defines how the run tests them.
PRODUCT_DIRS = ("atom/",)

_PLUGIN_INCLUDE = (
    "atom/plugin/**",
    ".github/scripts/plugin_ci/**",
    "docker/atom_release.dockerfile",
)

SUITES: tuple[Suite, ...] = (
    _suite(
        "atom-test",
        "atom-test.yaml",
        "ATOM Test",
        "ATOM Test",
        ("ci:full", "ci:atom"),
        ignore=(
            "**/*.md",
            "docs/**",
            ".claude/**",
            "atomrl/**",
            ".github/workflows/atomrl-*.yaml",
            "atom/plugin/**",
            "atom/mesh/**",
            ".github/benchmark/models_atomesh.yaml",
            ".github/scripts/atomesh/**",
            ".github/workflows/atomesh-*.yaml",
            ".github/workflows/atom-review-bot*.yml",
            ".github/scripts/atomesh_*.sh",
            ".github/dashboard/atomesh_*.html",
            ".github/workflows/atom-vllm-*.yaml",
            ".github/workflows/atom-sglang-*.yaml",
            ".github/benchmark/oot_models_accuracy.json",
            ".github/benchmark/sglang_models_accuracy.json",
            "LICENSE",
            ".gitignore",
        ),
        smoke_job="Offline inference smoke test",
        old_gate_job="ATOM Test Gate",
    ),
    _suite(
        "atom-sglang-test",
        "atom-sglang-test.yaml",
        "ATOM SGLang Test",
        "ATOM SGLang Test",
        ("ci:full", "ci:sglang"),
        include=_PLUGIN_INCLUDE
        + (
            ".github/workflows/atom-sglang-test.yaml",
            ".github/scripts/atom_sglang_test.sh",
            ".github/workflows/docker-release.yaml",
            ".github/benchmark/sglang_models_accuracy.json",
            "docker/sglang_release.dockerfile",
        ),
        ignore=(
            "**/*.md",
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
        include=_PLUGIN_INCLUDE
        + (
            ".github/workflows/atom-vllm-test.yaml",
            ".github/scripts/atom_oot_test.sh",
            ".github/scripts/check_mtp_acceptance.sh",
            ".github/benchmark/oot_models_accuracy.json",
            "docker/vllm_release.dockerfile",
            "docker/patches/**",
        ),
        ignore=(
            "**/*.md",
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
        include=(
            "atom/mesh/**",
            ".github/workflows/atomesh-accuracy-validation.yaml",
            ".github/scripts/accuracy_to_dashboard.py",
            ".github/scripts/atomesh_pd_accuracy.sh",
            ".github/benchmark/models_accuracy.json",
        ),
        old_gate_job="ATOMesh Accuracy Gate",
    ),
)

SUITES_BY_WORKFLOW_FILE = {suite.workflow_file: suite for suite in SUITES}
