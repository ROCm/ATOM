# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the ATOMesh workflows run on a pull request."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())["jobs"]


def test_mocker_benchmark_stays_off_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["if"] == "${{ github.event_name != 'pull_request' }}"
    assert "github.event.pull_request.draft == false" in (
        jobs["run_atomesh_test_harness"]["if"]
    )
