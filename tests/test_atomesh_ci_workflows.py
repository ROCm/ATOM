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
        "(github.event_name != 'pull_request' || "
        "needs.load-test-models.outputs.models_json != '[]')"
    ) in job["if"]


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
