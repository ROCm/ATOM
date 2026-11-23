# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Evaluate workflow expressions and run atomesh-benchmark.yaml's steps offline."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

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


class _JobOutputs:
    """A `needs.<job>.outputs` context: an unset output reads as ''."""

    def __init__(self, outputs: dict[str, str]):
        self._outputs = outputs

    def __getattr__(self, name: str) -> str:
        return self._outputs.get(name, "")


def github_expressions(
    source: str,
    event: str,
    schedule: str = "",
    workflow: str = "Atomesh Benchmark",
    pull_request_number: str = "",
    review_state: str = "",
    ref: str = "refs/heads/main",
    run_id: str = "1",
    action: str = "",
    label_name: str = "",
    label_names: tuple[str, ...] = (),
    needs_outputs: dict[str, str] | None = None,
) -> str:
    """Evaluate the ${{ }} segments of a workflow value for a run with no inputs.

    `needs_outputs` are the outputs every `needs.<job>.outputs` reads.
    """
    event_context = SimpleNamespace(
        schedule=schedule,
        action=action,
        label=SimpleNamespace(name=label_name),
        pull_request=SimpleNamespace(
            number=pull_request_number, label_names=list(label_names)
        ),
        review=SimpleNamespace(state=review_state),
    )
    context = {
        "event": event_context,
        "event_name": event,
        "schedule": schedule,
        "workflow": workflow,
        "ref": ref,
        "run_id": run_id,
        "job_outputs": _JobOutputs(needs_outputs or {}),
    }

    def evaluate(match: re.Match) -> str:
        expression = re.sub(r"\binputs\.\w+", "''", match.group(1))
        expression = re.sub(r"\bneeds\.[\w-]+\.outputs\.", "job_outputs.", expression)
        expression = re.sub(r"\bgithub\.event\.schedule\b", "schedule", expression)
        expression = re.sub(r"\bgithub\.(\w+)\b", r"\1", expression)
        expression = expression.replace(".labels.*.name", ".label_names")
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        return str(
            eval(
                expression,
                {
                    "__builtins__": {},
                    "format": lambda template, *args: template.format(*args),
                    "contains": lambda items, item: item in items,
                },
                context,
            )
        )

    return re.sub(r"\$\{\{(.*?)\}\}", evaluate, source)
