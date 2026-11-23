# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the SGLang benchmark / accuracy workflows' own model selection for every
scheduled cron: each MI355 case once a week in an MI355-only run, MI308 in its
own nightly run, and only the MI355 runs stopped by the 09:00 deadline."""

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
DEADLINE = WORKFLOWS / "atom-nightly-deadline.yaml"

BENCHMARK_MI308_CRON = "0 15 * * 1-6"
ACCURACY_MI308_CRON = "0 16 * * *"


def _workflow(path):
    return yaml.safe_load(path.read_text())


def _on(workflow):
    return workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True


def _crons(workflow):
    return [entry["cron"] for entry in _on(workflow)["schedule"]]


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


def _benchmark_run(script, cron):
    prefixes = [m["prefix"] for m in json.loads(_run(script, cron)["models_json"])]
    return (
        [p for p in prefixes if "mi308" not in p],
        [p for p in prefixes if "mi308" in p],
    )


def _accuracy_run(script, cron, **overrides):
    outputs = _run(script, cron, **overrides)
    return tuple(
        [row["model_name"] for row in json.loads(outputs[key])["include"]]
        for key in ("model_matrix_mi355", "model_matrix_mi308")
    )


def _assert_titles_split(workflow, mi308_cron):
    # The deadline tells the runs apart by title; the title tells them apart by
    # the MI308 cron.
    run_name = workflow["run-name"]
    assert f"github.event.schedule == '{mi308_cron}' && 'MI308' || 'MI355'" in run_name
    assert 'endswith("(scheduled MI355)")' in DEADLINE.read_text()


def test_benchmark_runs_each_mi355_case_once_a_week_and_mi308_on_its_own():
    workflow = _workflow(BENCHMARK)
    script = _embedded_python(workflow, "load-models", "load")
    mi355_crons = [f"0 15 * * {day}" for day in range(7)]
    assert sorted(_crons(workflow)) == sorted(mi355_crons + [BENCHMARK_MI308_CRON])
    _assert_titles_split(workflow, BENCHMARK_MI308_CRON)

    weekly = Counter()
    for cron in mi355_crons:
        mi355, mi308 = _benchmark_run(script, cron)
        assert mi355 and not mi308, cron
        weekly.update(mi355)
    tiers = re.findall(r"OOB_P[01]_PREFIX_ORDER = \[(.*?)\]", script, re.DOTALL)
    assert weekly == Counter(re.findall(r'"([^"]+)"', "".join(tiers)))

    catalog = json.loads(
        (REPO / ".github/benchmark/sglang_benchmark_models.json").read_text()
    )["models"]
    mi355, mi308 = _benchmark_run(script, BENCHMARK_MI308_CRON)
    assert not mi355
    assert sorted(mi308) == sorted(
        m["prefix"] for m in catalog if "mi308" in m["prefix"]
    )


def test_accuracy_runs_each_mi355_case_once_a_week_and_mi308_on_its_own():
    workflow = _workflow(ACCURACY)
    script = _embedded_python(workflow, "prepare-sglang-image", "meta")
    mi355_crons = [f"0 16 * * {day}" for day in range(7)]
    assert sorted(_crons(workflow)) == sorted(mi355_crons + [ACCURACY_MI308_CRON])
    _assert_titles_split(workflow, ACCURACY_MI308_CRON)

    all_mi355, all_mi308 = _accuracy_run(
        script,
        "",
        GITHUB_EVENT_NAME="workflow_dispatch",
        MODEL_MI355="all",
        MODEL_MI308="all",
    )

    weekly = Counter()
    for cron in mi355_crons:
        mi355, mi308 = _accuracy_run(script, cron)
        assert mi355 and not mi308, cron
        weekly.update(mi355)
    assert weekly == Counter(all_mi355)

    mi355, mi308 = _accuracy_run(script, ACCURACY_MI308_CRON)
    assert not mi355
    assert sorted(mi308) == sorted(all_mi308)


def test_accuracy_dropdowns_list_every_case():
    workflow = _workflow(ACCURACY)
    script = _embedded_python(workflow, "prepare-sglang-image", "meta")
    all_mi355, all_mi308 = _accuracy_run(
        script,
        "",
        GITHUB_EVENT_NAME="workflow_dispatch",
        MODEL_MI355="all",
        MODEL_MI308="all",
    )
    inputs = _on(workflow)["workflow_dispatch"]["inputs"]
    for key, models in (("model_mi355", all_mi355), ("model_mi308", all_mi308)):
        options = [o for o in inputs[key]["options"] if o not in ("none", "all")]
        assert sorted(options) == sorted(models), key


def _benchmark_cells(models, params, manual_param_lists, tmp_path):
    """Run the benchmark workflow's matrix builder; return {prefix: cells}."""
    step = next(
        s
        for s in _workflow(BENCHMARK)["jobs"]["build-benchmark-matrix"]["steps"]
        if s.get("id") == "build"
    )
    script = textwrap.dedent(step["run"].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0])
    output = tmp_path / "github_output"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,  # the builder writes sglang_benchmark_matrix.json to cwd
        env={
            **os.environ,
            "MODELS_JSON": json.dumps(models),
            "PARAMS_JSON": json.dumps(params),
            "INPUT_PARAM_LISTS": manual_param_lists,
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    cells = {}
    for key in ("benchmark_matrix_mi355", "benchmark_matrix_mi308"):
        for row in json.loads(outputs[key])["include"]:
            p = row["params"]
            cells.setdefault(row["model"]["prefix"], []).append(
                (p["input_length"], p["output_length"], p["concurrency"])
            )
    return cells


def test_nightly_param_lists_replace_shared_params_for_that_model_only(tmp_path):
    shared = [
        {
            "input_length": 1024,
            "output_length": 1024,
            "concurrency": c,
            "random_range_ratio": 0.8,
        }
        for c in (4, 8)
    ]
    models = [
        {
            "prefix": "pinned-mi308",
            "runner": "atom-mi308-8gpu-plugins-benchmark",
            "nightly_param_lists": "12000,350,1,1.0",
        },
        {"prefix": "shared-mi308", "runner": "atom-mi308-8gpu-plugins-benchmark"},
    ]

    nightly = _benchmark_cells(models, shared, "", tmp_path)
    assert nightly == {
        "pinned-mi308": [(12000, 350, 1)],
        "shared-mi308": [(1024, 1024, 4), (1024, 1024, 8)],
    }

    # A manual param_lists input wins over the model's own lists.
    manual = _benchmark_cells(
        models, shared, "1024,1024,4,0.8;1024,1024,8,0.8", tmp_path
    )
    assert (
        manual["pinned-mi308"]
        == manual["shared-mi308"]
        == [
            (1024, 1024, 4),
            (1024, 1024, 8),
        ]
    )
