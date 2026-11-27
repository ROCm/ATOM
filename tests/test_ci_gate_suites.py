# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI gate's suite table agrees with the workflows it describes."""

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
sys.path.insert(0, str(REPO / ".github" / "scripts"))

from ci_gate.evaluate import RELAY_TITLE_PREFIX, RELAY_WORKFLOW_FILE
from ci_gate.suites import SUITES


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
