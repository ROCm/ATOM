# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Only maintainers may manually run a workflow that reaches a self-hosted
runner: every such workflow's root jobs run on a GitHub-hosted runner and
start with the canonical authorization step
(.github/scripts/authorize_dispatch_step.yaml), so a new workflow cannot be
added without it."""

from functools import cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
STEP = yaml.safe_load(
    (REPO / ".github" / "scripts" / "authorize_dispatch_step.yaml").read_text()
)


@cache
def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    on = workflow.get(True, workflow.get("on")) or {}
    return (
        {on: None}
        if isinstance(on, str)
        else on if isinstance(on, dict) else dict.fromkeys(on)
    )


def _github_hosted(runs_on) -> bool:
    return isinstance(runs_on, str) and runs_on.startswith("ubuntu-")


@cache
def _reaches_self_hosted(name: str) -> bool:
    """Whether a job of the workflow, or of a local workflow it calls, may run
    on a runner that is not GitHub-hosted (an expression counts as one)."""
    for job in _workflow(name)["jobs"].values():
        uses = job.get("uses", "")
        if uses.startswith("./.github/workflows/"):
            if _reaches_self_hosted(uses.rsplit("/", 1)[1]):
                return True
        elif not _github_hosted(job.get("runs-on")):
            return True
    return False


def _guarded_workflows() -> list[str]:
    names = sorted(p.name for p in WORKFLOWS.glob("*.y*ml"))
    return [
        n
        for n in names
        if "workflow_dispatch" in _triggers(_workflow(n)) and _reaches_self_hosted(n)
    ]


def _root_jobs(workflow: dict) -> dict:
    return {k: v for k, v in workflow["jobs"].items() if not v.get("needs")}


@pytest.mark.parametrize("name", _guarded_workflows())
def test_root_jobs_authorize_a_manual_run_before_any_self_hosted_runner(name):
    for job_id, job in _root_jobs(_workflow(name)).items():
        assert _github_hosted(
            job.get("runs-on")
        ), f"{name}: root job {job_id} is not GitHub-hosted"
        steps = job.get("steps") or []
        assert (
            steps and steps[0] == STEP
        ), f"{name}: root job {job_id} does not start with the authorization step"


def test_the_guarded_set_is_not_empty():
    # A broken detector would make the test above vacuous.
    assert {"atom-test.yaml", "atom-release.yaml", "atomrl-unit-tests.yaml"} <= set(
        _guarded_workflows()
    )
