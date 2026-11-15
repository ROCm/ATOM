# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run atomesh-benchmark.yaml's own steps and expressions without GitHub."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/atomesh-benchmark.yaml"


def load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def workflow_step(workflow: dict, step_id: str) -> str:
    """The run script of a load-config step."""
    return next(
        step["run"]
        for step in workflow["jobs"]["load-config"]["steps"]
        if step.get("id") == step_id
    )


def run_matrix_step(script: str, env: dict[str, str]) -> tuple[list[dict], str]:
    """Run the matrix step over this checkout's catalog.

    Returns the cells it wrote and what it appended to GITHUB_OUTPUT.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in (
            ".github/scripts/atomesh/pd_matrix.py",
            ".github/benchmark/models_atomesh.yaml",
        ):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        output = root / "github-output"
        result = subprocess.run(
            ["bash", "-eu", "-c", script],
            cwd=root,
            env=dict(os.environ, **env, GITHUB_OUTPUT=str(output)),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise AssertionError(f"matrix step failed:\n{result.stdout}{result.stderr}")
        cells = json.loads((root / "atomesh-matrix.json").read_text())["include"]
        return cells, output.read_text()


def github_expressions(source: str, event: str, schedule: str = "") -> str:
    """Evaluate the ${{ }} segments of a workflow value for a run with no inputs."""
    context = {
        "event_name": event,
        "schedule": schedule,
        "workflow": "Atomesh Benchmark",
        "ref": "refs/heads/main",
        "run_id": "1",
    }

    def evaluate(match: re.Match) -> str:
        expression = re.sub(r"\binputs\.\w+", "''", match.group(1))
        expression = re.sub(r"\bgithub\.event\.schedule\b", "schedule", expression)
        expression = re.sub(r"\bgithub\.(\w+)\b", r"\1", expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        return str(eval(expression, {"__builtins__": {}}, context))

    return re.sub(r"\$\{\{(.*?)\}\}", evaluate, source)
