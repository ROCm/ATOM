"""CPU-only admission and failure contracts for runner checkpoint validation."""

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/atomesh-benchmark.yaml"
SCRIPTS = ROOT / ".github/scripts/atomesh"
CASE = "survey-v4-flash-main-nixl-1p1d-tp8-eager"
SNAPSHOT = "/share_nfs/model_coverage/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"


def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def step(job, name):
    return next(s for s in workflow()["jobs"][job]["steps"] if s.get("name") == name)


def python_code(step):
    return step["run"].split("python3 - <<'PY'\n", 1)[1].split("\nPY", 1)[0]


def admitted(expression, context):
    """Evaluate the boolean-only workflow conditions under test, not arbitrary code."""
    expression = expression.removeprefix("${{").removesuffix("}}")
    expression = re.sub(
        r"'[^']*'|[A-Za-z_][\w.-]*(?:\(\))?",
        lambda m: m[0] if m[0].startswith("'") else repr(context[m[0]]),
        expression,
    )
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return bool(eval(expression.strip(), {"__builtins__": {}}, {}))


@pytest.mark.parametrize("only", [False, True])
def test_actual_job_admission_keeps_preflight_exclusive(only):
    doc = workflow()
    inputs = doc.get("on", doc.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["survey_preflight_only"]["default"] is False
    assert inputs["survey_preflight_only"]["type"] == "boolean"
    context = {
        "inputs.survey_preflight_only": only,
        "github.event_name": "workflow_dispatch",
        "inputs.run_model_benchmark": True,
        "inputs.publish_dashboard": True,
        "inputs.suite": "vllm",
        "inputs.inspect_run_id": "",
        "needs.load-config.outputs.has_matrix": "true",
        "needs.run-model-benchmark.result": "success",
        "needs.summarize-results.result": "success",
        "needs.summarize-results.outputs.has_results": "true",
        "github.event.pull_request": False,
        "github.event.pull_request.draft": False,
        "false": False,
        "always()": True,
        "cancelled()": False,
    }
    for name in ("run-model-benchmark", "summarize-results", "dashboard"):
        assert admitted(doc["jobs"][name]["if"], context) == (not only)
    assert admitted(doc["jobs"]["survey-runner-preflight"]["if"], context) == only
    context["inputs.run_model_benchmark"] = False
    assert admitted(doc["jobs"]["survey-runner-preflight"]["if"], context) == only
    assert not admitted(doc["jobs"]["run-model-benchmark"]["if"], context)
    assert admitted(doc["jobs"]["load-config"]["if"], context)
    context.update({"inputs.inspect_run_id": "123", "inputs.suite": "smoke"})
    for name in ("inspect-run", "rdma-smoke-test"):
        assert admitted(doc["jobs"][name]["if"], context) == (not only)
    context["steps.matrix.outputs.has_matrix"] = "true"
    assert admitted(
        step("load-config", "Validate ATOMesh P/D matrix")["if"], context
    ) == (not only)
    assert admitted(
        step("load-config", "Find ATOMesh latest nightly image")["if"], context
    ) == (not only)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"INSPECT_RUN_ID": "123"},
        {"INSPECT_PROCESSES": "true"},
        {"SUITE": "smoke"},
        {"CASE_NAMES": CASE + ",other"},
        {"CASE_NAMES": "other"},
        {"RUN_ALL_MODELS": "true"},
    ],
)
def test_request_gate_rejects_inspection_mixed_and_non_survey(tmp_path, bad):
    env = {
        **os.environ,
        "EVENT_NAME": "workflow_dispatch",
        "INSPECT_RUN_ID": "",
        "INSPECT_PROCESSES": "false",
        "SUITE": "vllm",
        "CASE_NAMES": CASE,
        "RUN_ALL_MODELS": "false",
        **bad,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            python_code(step("load-config", "Validate runner-only survey request")),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (not bad)


@pytest.mark.parametrize(
    "damage",
    [None, "empty", "mixed", "model", "path", "clean_main", "profile", "manifest"],
)
def test_matrix_gate_requires_original_single_v4_snapshot(tmp_path, damage):
    cell = {
        "model": "DeepSeek-V4-Flash-vLLM-Survey",
        "name": CASE,
        "backend": "vllm",
        "model_path": SNAPSHOT,
        "vllm": {
            "clean_main": 1,
            "nixl_model_profile": "v4",
            "checkpoint_manifest": "v4-checkpoint-identity.json",
        },
    }
    cells = [cell]
    if damage == "empty":
        cells = []
    elif damage == "mixed":
        cells.append(copy.deepcopy(cell))
    elif damage == "model":
        cell["model"] = "other"
    elif damage == "path":
        cell["model_path"] += "-other"
    elif damage == "clean_main":
        cell["vllm"]["clean_main"] = 0
    elif damage == "profile":
        cell["vllm"]["nixl_model_profile"] = "m3"
    elif damage == "manifest":
        del cell["vllm"]["checkpoint_manifest"]
    (tmp_path / "atomesh-matrix.json").write_text(json.dumps({"include": cells}))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            python_code(step("load-config", "Validate runner-only survey matrix")),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (damage is None)


def test_runner_only_job_reuses_gate_artifact_and_has_no_submission_surface():
    jobs = workflow()["jobs"]
    job = jobs["survey-runner-preflight"]
    assert job["needs"] == ["load-config"]
    assert job["runs-on"] == "${{ matrix.runner.slurm_submit_runner }}"
    assert all(
        job["env"][name] == ""
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    )
    assert len(job["steps"]) == 5
    assert job["steps"][0]["uses"] == "actions/checkout@v6"
    for candidate in job["steps"][1:4]:
        original = next(
            s
            for s in jobs["run-model-benchmark"]["steps"]
            if s.get("name") == candidate["name"]
        )
        assert candidate == original
    scripts = "\n".join(s.get("run", "") for s in job["steps"])
    assert not re.search(
        r"\b(sbatch|srun|scancel|docker|podman)\b|pd_submit|pd_slurm_job|process_result",
        scripts,
    )
    gate = job["steps"][2]
    assert "continue-on-error" not in gate
    assert "check=True" in gate["run"]
    assert "always()" in job["steps"][3]["if"]
    assert job["steps"][3]["with"]["path"] == "survey-preflight/"
    assert "always()" in job["steps"][4]["if"]


def test_failed_checkpoint_stays_failed_and_report_is_available_for_upload(tmp_path):
    job = workflow()["jobs"]["survey-runner-preflight"]
    (tmp_path / "survey-preflight").mkdir()
    (tmp_path / ".github/scripts").mkdir(parents=True)
    (tmp_path / ".github/scripts/atomesh").symlink_to(SCRIPTS, target_is_directory=True)
    summary = tmp_path / "summary"
    env = {
        **os.environ,
        "CELL_JSON": json.dumps(
            {
                "model": "DeepSeek-V4-Flash-vLLM-Survey",
                "model_path": str(tmp_path / "missing-checkpoint"),
                "vllm": {
                    "clean_main": 1,
                    "nixl_model_profile": "v4",
                    "checkpoint_manifest": "v4-checkpoint-identity.json",
                },
            }
        ),
        "ATOMESH_MODEL_ROOT": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(summary),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        [sys.executable, "-c", python_code(job["steps"][2])],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads((tmp_path / "survey-preflight/weights.json").read_text())
    assert report["status"] == "BLOCKED_ENV"
    subprocess.run(
        [sys.executable, "-c", python_code(job["steps"][4])],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "BLOCKED_ENV" in summary.read_text()
    assert (
        "No Slurm allocation, GPU execution, containers, model loading, or NIXL validation"
        in summary.read_text()
    )
    assert not (tmp_path / "atomesh-results").exists()
