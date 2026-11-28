# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/pr-welcome-comment.yaml"
)
WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"


def _welcome_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return workflow["jobs"]["comment"]["steps"][0]["with"]["script"]


def _heavy_gate_labels() -> set[str]:
    labels: set[str] = set()
    for path in WORKFLOWS.glob("*.yaml"):
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                gate_labels = step.get("env", {}).get("CI_GATE_LABELS")
                if not gate_labels:
                    continue
                labels.update(label.strip() for label in gate_labels.split(","))
    return labels


def test_ci_guide_documents_heavy_gate_labels():
    script = _welcome_script()

    assert _heavy_gate_labels() == {
        "ci:full",
        "ci:atom",
        "ci:vllm",
        "ci:sglang",
        "ci:mesh",
    }
    for label in _heavy_gate_labels():
        assert f"| `{label}` |" in script


def test_ci_guide_documents_atomesh_path_scope():
    script = _welcome_script()

    assert "ATOMesh uses an include list" in script
