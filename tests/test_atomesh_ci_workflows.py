# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the ATOMesh workflows run on a pull request."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from atomesh_workflow_helpers import github_expressions

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())["jobs"]


def _workflow(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())


def test_empty_accuracy_model_list_skips_the_job_only_on_pull_requests():
    job = _jobs("atomesh-accuracy-validation.yaml")["atomesh-test"]
    assert (
        "(!github.event.pull_request || "
        "needs.load-test-models.outputs.models_json != '[]')"
    ) in job["if"]


def test_atomesh_accuracy_uses_required_heavy_ci_gate():
    workflow = yaml.safe_load(
        (WORKFLOWS / "atomesh-accuracy-validation.yaml").read_text()
    )
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]
    assert "pull_request_review" in events
    assert "labeled" in events["pull_request"]["types"]
    assert "closed" in events["pull_request"]["types"]

    jobs = workflow["jobs"]
    gate_env = next(
        step["env"] for step in jobs["ci-gate"]["steps"] if step.get("id") == "gate"
    )
    assert gate_env["CI_GATE_WORKFLOW"] == "atomesh-accuracy-validation.yaml"
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:mesh"
    assert "atom/mesh/**" in gate_env["CI_GATE_PATHS_INCLUDE"]
    assert ".github/benchmark/models_accuracy.json" in gate_env["CI_GATE_PATHS_INCLUDE"]

    assert jobs["check-signal"]["needs"] == ["ci-gate"]
    assert jobs["download_aiter_wheel"]["needs"] == ["ci-gate", "check-signal"]
    assert jobs["load-test-models"]["needs"] == ["ci-gate"]
    assert jobs["atomesh-test"]["needs"] == [
        "ci-gate",
        "download_aiter_wheel",
        "load-test-models",
    ]

    test_gate = jobs["atomesh-accuracy-gate"]
    assert test_gate["if"] == "always()"
    assert test_gate["needs"] == [
        "ci-gate",
        "check-signal",
        "download_aiter_wheel",
        "load-test-models",
        "atomesh-test",
    ]
    evaluate_step = next(
        step
        for step in test_gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    evaluate_inputs = evaluate_step["with"]
    required = evaluate_inputs["required-results"]
    assert "load-test-models=${{ needs.load-test-models.result }}" in required
    assert "atomesh-test=${{ needs.atomesh-test.result }}" in required
    assert "optional-results" not in evaluate_inputs


def test_a_pull_request_checks_one_model_on_atomesh_standalone(tmp_path):
    jobs = _jobs("atomesh-accuracy-validation.yaml")
    assert jobs["atomesh-test"]["env"]["USE_ATOMESH_ENTRYPOINTS"] == 1
    load = next(s for s in jobs["load-test-models"]["steps"] if s.get("id") == "load")
    script = load["run"].split("python3 << 'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    output = tmp_path / "output"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=WORKFLOWS.parents[1],
        env=dict(os.environ, EVENT_NAME="pull_request", GITHUB_OUTPUT=str(output)),
        check=True,
        capture_output=True,
        timeout=30,
    )
    models = json.loads(output.read_text().split("models_json=", 1)[1])
    assert [m["model_name"] for m in models] == ["gpt-oss-120b"]


def test_atomesh_benchmark_uses_required_heavy_ci_gate():
    workflow = yaml.safe_load((WORKFLOWS / "atomesh-benchmark.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]
    assert "pull_request_review" in events
    assert "labeled" in events["pull_request"]["types"]
    assert "closed" in events["pull_request"]["types"]

    jobs = workflow["jobs"]
    gate_env = next(
        step["env"] for step in jobs["ci-gate"]["steps"] if step.get("id") == "gate"
    )
    assert gate_env["CI_GATE_WORKFLOW"] == "atomesh-benchmark.yaml"
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:mesh"
    assert "atom/mesh/**" in gate_env["CI_GATE_PATHS_INCLUDE"]
    assert ".github/benchmark/models_atomesh.yaml" in gate_env["CI_GATE_PATHS_INCLUDE"]

    assert jobs["load-config"]["needs"] == ["ci-gate"]
    assert jobs["rdma-smoke-test"]["needs"] == ["ci-gate", "load-config"]
    assert jobs["run-model-benchmark"]["needs"] == ["ci-gate", "load-config"]
    assert (
        "github.event.pull_request && !github.event.pull_request.head.repo.fork"
        in jobs["run-model-benchmark"]["if"]
    )

    test_gate = jobs["atomesh-benchmark-gate"]
    assert test_gate["if"] == "always()"
    assert test_gate["needs"] == [
        "ci-gate",
        "load-config",
        "rdma-smoke-test",
        "run-model-benchmark",
        "summarize-results",
    ]
    evaluate_step = next(
        step
        for step in test_gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    evaluate_inputs = evaluate_step["with"]
    assert (
        "load-config=${{ needs.load-config.result }}"
        in evaluate_inputs["required-results"]
    )
    assert (
        "run-model-benchmark=${{ (github.event_name == 'schedule'"
        in evaluate_inputs["required-results"]
    )
    assert "run-model-benchmark=${{ needs.run-model-benchmark.result }}" in (
        evaluate_inputs["optional-results"]
    )


def test_atomesh_heavy_gate_concurrency_groups_separate_non_approval_reviews():
    cases = [
        ("atomesh-accuracy-validation.yaml", "Atomesh Accuracy Validation"),
        ("atomesh-benchmark.yaml", "Atomesh Benchmark"),
    ]

    for filename, workflow_name in cases:
        group = _workflow(filename)["concurrency"]["group"]
        pull_request_group = github_expressions(
            group,
            "pull_request",
            workflow=workflow_name,
            pull_request_number="3313",
            run_id="100",
        )
        approval_group = github_expressions(
            group,
            "pull_request_review",
            workflow=workflow_name,
            pull_request_number="3313",
            review_state="approved",
            run_id="200",
        )
        comment_group = github_expressions(
            group,
            "pull_request_review",
            workflow=workflow_name,
            pull_request_number="3313",
            review_state="commented",
            run_id="300",
        )

        assert approval_group == pull_request_group
        assert comment_group == f"{pull_request_group}-review-300"


def test_mocker_benchmark_stays_off_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["if"] == "${{ github.event_name != 'pull_request' }}"
    assert jobs["run_atomesh_test_harness"]["if"] == (
        "${{ needs.unit-test-scope.outputs.should_run == 'true' }}"
    )


def test_every_pull_request_publishes_the_atomesh_unit_test_gate():
    workflow = _workflow("atomesh-mocker-benchmark.yaml")
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]

    jobs = workflow["jobs"]
    assert jobs["run_atomesh_test_harness"]["needs"] == ["unit-test-scope"]
    gate = jobs["unit-test-gate"]
    assert gate["name"] == "ATOMesh non-GPU unit tests"
    assert gate["if"] == "always()"
    assert gate["needs"] == ["unit-test-scope", "run_atomesh_test_harness"]
    inputs = next(
        step["with"]
        for step in gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    assert inputs["ci-gate-result"] == "${{ needs.unit-test-scope.result }}"
    assert inputs["required-results"].strip() == (
        "run_atomesh_test_harness=${{ needs.run_atomesh_test_harness.result }}"
    )


def test_the_unit_test_scope_follows_the_changed_files(tmp_path):
    scope = _jobs("atomesh-mocker-benchmark.yaml")["unit-test-scope"]["steps"][0]
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" $CHANGED_FILES\n')
    gh.chmod(0o755)
    cases = (
        ("1", "false", "docs/a.md atom/mesh/src/lib.rs", "true"),
        ("1", "false", "docs/a.md atom/model_engine/a.py", "false"),
        ("1", "true", "atom/mesh/src/lib.rs", "false"),
        ("", "false", "", "true"),
    )
    for pr_number, is_draft, changed_files, should_run in cases:
        output = tmp_path / "output"
        output.write_text("")
        env = {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "ROCm/ATOM",
            "PATHS_INCLUDE": scope["env"]["PATHS_INCLUDE"],
            "PR_NUMBER": pr_number,
            "IS_DRAFT": is_draft,
            "CHANGED_FILES": changed_files,
        }
        subprocess.run(["bash", "-c", scope["run"]], env=env, check=True, timeout=10)
        lines = output.read_text().splitlines()
        assert f"should_run={should_run}" in lines, (pr_number, is_draft, changed_files)


def _steps(job):
    return {step["name"]: step for step in job["steps"]}


def test_mocker_cargo_cache_is_saved_from_clean_builds_outside_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    for job in jobs.values():
        assert not any(
            step.get("uses", "").startswith("actions/cache@") for step in job["steps"]
        )
    harness = jobs["run_atomesh_test_harness"]
    restore = _steps(harness)["Restore cargo build output"]["with"]
    save = _steps(harness)["Save cargo build output"]
    assert (
        restore["key"] == save["with"]["key"] == "${{ steps.cargo-deps.outputs.key }}"
    )
    assert restore["path"] == save["with"]["path"]
    assert restore["restore-keys"] == (
        "${{ github.event_name == 'pull_request' && "
        "steps.cargo-deps.outputs.prefix || '' }}"
    )
    assert save["if"] == (
        "${{ github.event_name != 'pull_request' && "
        "steps.cargo-cache.outputs.cache-hit != 'true' }}"
    )
    assert harness["outputs"] == {
        "cargo_cache_key": "${{ steps.cargo-deps.outputs.key }}",
        "cargo_cache_prefix": "${{ steps.cargo-deps.outputs.prefix }}",
    }


def test_mocker_benchmark_only_restores_the_harness_cache():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["needs"] == ["run_atomesh_test_harness"]
    restore = _steps(benchmark)["Restore cargo build output"]
    assert restore["uses"] == "actions/cache/restore@v4"
    assert restore["with"]["key"] == (
        "${{ needs.run_atomesh_test_harness.outputs.cargo_cache_key }}"
    )
    assert restore["with"]["restore-keys"] == (
        "${{ needs.run_atomesh_test_harness.outputs.cargo_cache_prefix }}"
    )
    saved = _steps(jobs["run_atomesh_test_harness"])["Save cargo build output"]
    assert restore["with"]["path"] == saved["with"]["path"]
    assert not any(
        step.get("uses", "").startswith("actions/cache/save@")
        for step in benchmark["steps"]
    )
