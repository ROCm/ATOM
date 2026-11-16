# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the SGLang benchmark / accuracy workflows' own model selection for every
scheduled night: each MI355 case once a week, MI308 at its nightly cadence."""

import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
BENCHMARK = WORKFLOWS / "atom-sglang-benchmark.yaml"
ACCURACY = WORKFLOWS / "atom-sglang-accuracy-validation.yaml"


def _workflow(path):
    return yaml.safe_load(path.read_text())


def _crons(workflow):
    on = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True
    return [entry["cron"] for entry in on["schedule"]]


def _embedded_python(workflow, job, step_id):
    step = next(s for s in workflow["jobs"][job]["steps"] if s.get("id") == step_id)
    body = step["run"].split("<<'PY' >> \"$GITHUB_OUTPUT\"\n", 1)[1]
    return textwrap.dedent(body.split("\nPY\n", 1)[0])


def _run(script, cron, **overrides):
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "schedule",
        "SCHEDULE_CRON": cron,
        "WORKLOAD_LABEL": "SGLang-OOB",
        **overrides,
    }
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _benchmark_night(script, cron):
    prefixes = [m["prefix"] for m in json.loads(_run(script, cron)["models_json"])]
    return (
        [p for p in prefixes if "mi308" not in p],
        [p for p in prefixes if "mi308" in p],
    )


def _accuracy_night(script, cron, **overrides):
    outputs = _run(script, cron, **overrides)
    return tuple(
        [row["model_name"] for row in json.loads(outputs[key])["include"]]
        for key in ("model_matrix_mi355", "model_matrix_mi308")
    )


def test_benchmark_runs_each_mi355_case_once_a_week():
    workflow = _workflow(BENCHMARK)
    script = _embedded_python(workflow, "load-models", "load")
    crons = _crons(workflow)
    assert sorted(crons) == [f"0 15 * * {day}" for day in range(7)]

    catalog = json.loads(
        (REPO / ".github/benchmark/sglang_benchmark_models.json").read_text()
    )["models"]
    all_mi308 = sorted(m["prefix"] for m in catalog if "mi308" in m["prefix"])

    weekly = Counter()
    for cron in crons:
        mi355, mi308 = _benchmark_night(script, cron)
        assert mi355, f"{cron} runs no MI355 case"
        weekly.update(mi355)
        expected_mi308 = [] if cron.endswith(" 0") else all_mi308
        assert sorted(mi308) == expected_mi308, cron

    tiers = re.findall(r"OOB_P[01]_PREFIX_ORDER = \[(.*?)\]", script, re.DOTALL)
    assert weekly == Counter(re.findall(r'"([^"]+)"', "".join(tiers)))


def test_accuracy_runs_each_mi355_case_once_a_week():
    workflow = _workflow(ACCURACY)
    script = _embedded_python(workflow, "prepare-sglang-image", "meta")
    crons = _crons(workflow)
    assert sorted(crons) == [f"0 16 * * {day}" for day in range(7)]

    weekly = Counter()
    mi308_per_night = set()
    for cron in crons:
        mi355, mi308 = _accuracy_night(script, cron)
        assert mi355, f"{cron} runs no MI355 case"
        weekly.update(mi355)
        mi308_per_night.add(tuple(sorted(mi308)))

    all_mi355, all_mi308 = _accuracy_night(
        script,
        "",
        GITHUB_EVENT_NAME="workflow_dispatch",
        MODEL_MI355="all",
        MODEL_MI308="all",
    )
    assert weekly == Counter(all_mi355)
    assert mi308_per_night == {tuple(sorted(all_mi308))}
