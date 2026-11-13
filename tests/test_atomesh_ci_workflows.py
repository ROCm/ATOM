# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the ATOMesh workflows run on a pull request."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())["jobs"]


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
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:atomesh"
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
    assert (
        "atomesh-test=${{ needs.atomesh-test.result }}"
        in evaluate_inputs["optional-results"]
    )
    assert (
        "load-test-models=${{ needs.load-test-models.result }}"
        in evaluate_inputs["required-results"]
    )


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
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:atomesh"
    assert "atom/mesh/**" in gate_env["CI_GATE_PATHS_INCLUDE"]
    assert ".github/benchmark/models_atomesh.yaml" in gate_env["CI_GATE_PATHS_INCLUDE"]

    assert jobs["load-config"]["needs"] == ["ci-gate"]
    assert jobs["rdma-smoke-test"]["needs"] == ["ci-gate", "load-config"]
    assert jobs["run-model-benchmark"]["needs"] == ["ci-gate", "load-config"]

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
    assert "run-model-benchmark=${{ needs.run-model-benchmark.result }}" in (
        evaluate_inputs["optional-results"]
    )


def test_mocker_benchmark_stays_off_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["if"] == "${{ github.event_name != 'pull_request' }}"
    assert "github.event.pull_request.draft == false" in (
        jobs["run_atomesh_test_harness"]["if"]
    )


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
