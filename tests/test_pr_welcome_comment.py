# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/pr-welcome-comment.yaml"
)


def test_ci_guide_documents_heavy_gate_labels_and_path_scope():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    script = workflow["jobs"]["comment"]["steps"][0]["with"]["script"]

    assert "`ci:mesh`" in script
    assert "native ATOM, vLLM, SGLang, and ATOMesh" in script
    assert "Gate checks are published on every PR" in script
    assert "Without a matching label, expensive jobs also require a relevant path" in script
    assert "Relevant path alone does not start expensive jobs" in script
    assert "ATOMesh uses an include list" in script
