# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the vLLM benchmark / accuracy workflows' own model selection for every
scheduled night: each scheduled case runs once a week."""

import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"
BENCHMARK = WORKFLOWS / "atom-vllm-benchmark.yaml"
ACCURACY = WORKFLOWS / "atom-vllm-accuracy-validation.yaml"
ACCURACY_CATALOG = REPO / ".github" / "benchmark" / "oot_models_accuracy.json"
DEADLINE = WORKFLOWS / "atom-nightly-deadline.yaml"

# 2026-11-16 is a Monday; the accuracy run of each Beijing date starts at 00:00.
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEK = [f"2026-11-{16 + day}T00:05:00+08:00" for day in range(7)]


def _workflow(path):
    return yaml.safe_load(path.read_text())


def _crons(workflow):
    on = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True
    return [entry["cron"] for entry in on["schedule"]]


def _embedded_python(workflow, job, step_id):
    step = next(s for s in workflow["jobs"][job]["steps"] if s.get("id") == step_id)
    body = step["run"].split("<<'PY' >> \"$GITHUB_OUTPUT\"\n", 1)[1]
    return textwrap.dedent(body.split("\nPY\n", 1)[0])


def _run(script, **env):
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env={**os.environ, "GITHUB_EVENT_NAME": "schedule", **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _benchmark_displays(script, **env):
    return [m["display"] for m in json.loads(_run(script, **env)["models_json"])]


def _accuracy_models(script, beijing_now):
    outputs = _run(script, VALIDATION_BEIJING_NOW=beijing_now)
    return [
        row["model_name"]
        for slot in (1, 2, 3)
        for row in json.loads(outputs[f"model_matrix_slot_{slot}"])["include"]
    ]


def test_benchmark_runs_each_scheduled_model_once_a_week():
    workflow = _workflow(BENCHMARK)
    script = _embedded_python(workflow, "load-models", "load")
    crons = _crons(workflow)
    assert sorted(crons) == [f"30 15 * * {day}" for day in range(1, 7)]

    weekly = Counter()
    for cron in crons:
        tonight = _benchmark_displays(script, SCHEDULE_CRON=cron)
        assert tonight, cron
        weekly.update(tonight)

    everything = _benchmark_displays(
        script, GITHUB_EVENT_NAME="workflow_dispatch", BENCHMARK_GROUP="ALL"
    )
    assert weekly == Counter(everything)


def test_accuracy_runs_each_nightly_case_on_its_weekday_only():
    script = _embedded_python(_workflow(ACCURACY), "prepare-oot-image", "meta")
    nightly = [
        m
        for m in json.loads(ACCURACY_CATALOG.read_text())
        if m.get("test_level", "nightly") == "nightly"
    ]
    weekly = Counter()
    for weekday, beijing_now in zip(WEEKDAYS, WEEK):
        tonight = _accuracy_models(script, beijing_now)
        expected = [m["model_name"] for m in nightly if m["nightly_weekday"] == weekday]
        assert expected, weekday
        assert sorted(tonight) == sorted(expected), weekday
        weekly.update(tonight)
    assert weekly == Counter(m["model_name"] for m in nightly)


def test_accuracy_rejects_a_nightly_case_without_a_weekday(tmp_path):
    catalog = json.loads(ACCURACY_CATALOG.read_text())
    del catalog[0]["nightly_weekday"]
    script = _embedded_python(_workflow(ACCURACY), "prepare-oot-image", "meta").replace(
        '".github/benchmark/oot_models_accuracy.json"',
        repr(str(tmp_path / "catalog.json")),
    )
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "schedule",
            "VALIDATION_BEIJING_NOW": WEEK[0],
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "needs nightly_weekday" in result.stderr


@pytest.mark.parametrize("workflow_path", [BENCHMARK, ACCURACY])
def test_scheduled_runs_are_stopped_by_the_nightly_deadline(workflow_path):
    # The deadline stops a listed workflow's runs titled "(scheduled MI355)".
    run_name = _workflow(workflow_path)["run-name"]
    assert "format('{0} (scheduled MI355)', github.workflow)" in run_name
    deadline = _workflow(DEADLINE)["jobs"]["stop-overrunning-nightly-runs"]
    assert workflow_path.name in deadline["env"]["WORKFLOWS"].split()
    assert 'endswith("(scheduled MI355)")' in DEADLINE.read_text()


def test_accuracy_jobs_no_longer_skip_late_in_the_beijing_day():
    job = _workflow(ACCURACY)["jobs"]["oot-model-accuracy-priority-1"]
    window = job["steps"][0]
    assert window["id"] == "validation-window"
    assert "MIN_REMAINING_MINUTES" not in window.get("env", {})
    assert "remaining" not in window["run"]
