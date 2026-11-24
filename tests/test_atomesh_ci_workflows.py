# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the ATOMesh workflows run on a pull request."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from atomesh_workflow_helpers import github_expressions

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _jobs(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())["jobs"]


def _workflow(filename):
    return yaml.safe_load((WORKFLOWS / filename).read_text())


def test_empty_accuracy_model_list_skips_the_job_only_on_pull_requests():
    job = _jobs("atomesh-accuracy-validation.yaml")["atomesh-test"]
    assert (
        "(!github.event.pull_request || "
        "needs.load-test-models.outputs.models_json != '[]')"
    ) in job["if"]


def test_atomesh_accuracy_uses_required_heavy_ci_gate():
    workflow = yaml.safe_load(
        (WORKFLOWS / "atomesh-accuracy-validation.yaml").read_text()
    )
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]
    assert "pull_request_review" in events
    assert "labeled" in events["pull_request"]["types"]
    assert "closed" in events["pull_request"]["types"]

    jobs = workflow["jobs"]
    gate_env = next(
        step["env"] for step in jobs["ci-gate"]["steps"] if step.get("id") == "gate"
    )
    assert gate_env["CI_GATE_WORKFLOW"] == "atomesh-accuracy-validation.yaml"
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:mesh"
    assert "atom/mesh/**" in gate_env["CI_GATE_PATHS_INCLUDE"]
    assert ".github/benchmark/models_accuracy.json" in gate_env["CI_GATE_PATHS_INCLUDE"]

    assert jobs["check-signal"]["needs"] == ["ci-gate"]
    assert jobs["download_aiter_wheel"]["needs"] == ["ci-gate", "check-signal"]
    assert jobs["load-test-models"]["needs"] == ["ci-gate"]
    assert jobs["atomesh-test"]["needs"] == [
        "ci-gate",
        "download_aiter_wheel",
        "load-test-models",
    ]

    test_gate = jobs["atomesh-accuracy-gate"]
    assert test_gate["if"] == "always()"
    assert test_gate["needs"] == [
        "ci-gate",
        "check-signal",
        "download_aiter_wheel",
        "load-test-models",
        "atomesh-test",
    ]
    evaluate_step = next(
        step
        for step in test_gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    evaluate_inputs = evaluate_step["with"]
    required = evaluate_inputs["required-results"]
    assert "load-test-models=${{ needs.load-test-models.result }}" in required
    assert "atomesh-test=${{ needs.atomesh-test.result }}" in required
    assert "optional-results" not in evaluate_inputs


def _accuracy_cases(tmp_path, event):
    """The cases the accuracy workflow's load step writes for an event."""
    jobs = _jobs("atomesh-accuracy-validation.yaml")
    load = next(s for s in jobs["load-test-models"]["steps"] if s.get("id") == "load")
    script = load["run"].split("python3 << 'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    output = tmp_path / f"output-{event}"
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=WORKFLOWS.parents[1],
        env=dict(os.environ, EVENT_NAME=event, GITHUB_OUTPUT=str(output)),
        check=True,
        capture_output=True,
        timeout=30,
    )
    return json.loads(output.read_text().split("models_json=", 1)[1])


def test_a_pull_request_checks_one_model_standalone_and_1p1d(tmp_path):
    jobs = _jobs("atomesh-accuracy-validation.yaml")
    assert jobs["atomesh-test"]["env"]["USE_ATOMESH_ENTRYPOINTS"] == 1
    cases = _accuracy_cases(tmp_path, "pull_request")
    assert [(c["model_name"], c["serving"]) for c in cases] == [
        ("gpt-oss-120b", "standalone"),
        ("gpt-oss-120b", "1p1d"),
    ]
    assert "accuracy_timeout_minutes" not in cases[0]
    assert cases[1]["accuracy_timeout_minutes"] == 45
    assert jobs["atomesh-test"]["name"] == (
        "Accuracy (${{ matrix.model_name }}"
        "${{ matrix.serving == '1p1d' && ', 1P1D' || '' }})"
    )
    steps = _steps(jobs["atomesh-test"])
    accuracy = steps["Run ATOM accuracy test"]
    assert accuracy["env"]["SERVING"] == "${{ matrix.serving }}"
    assert (
        'if [ "$SERVING" = "1p1d" ]; then\n'
        '  launch="bash .github/scripts/atomesh_pd_accuracy.sh"\n'
    ) in accuracy["run"]
    # A 1P1D case streams each side's log while it runs, not after it.
    for side, log in (
        ("prefill", "atom_prefill.log"),
        ("decode", "atom_decode.log"),
        ("router", "atom_server.log"),
    ):
        assert f"\n  stream {side} {log}\n" in accuracy["run"]
    assert (
        "\n  trap 'kill $(jobs -p) 2>/dev/null || true' EXIT\nfi\n" in accuracy["run"]
    )
    assert steps["Dump server log"]["if"] == "always() && matrix.serving != '1p1d'"
    # The gate holds a 1P1D case to its model's threshold.
    check = steps["Check accuracy test results"]
    assert check["env"]["MODEL_NAME"] == "${{ matrix.model_name }}"


def test_the_accuracy_scope_matches_the_push_paths():
    workflow = _workflow("atomesh-accuracy-validation.yaml")
    events = workflow.get("on", workflow.get(True))
    gate_env = next(
        step["env"]
        for step in workflow["jobs"]["ci-gate"]["steps"]
        if step.get("id") == "gate"
    )
    assert gate_env["CI_GATE_PATHS_INCLUDE"].split() == events["push"]["paths"]
    assert ".github/scripts/atomesh_pd_accuracy.sh" in events["push"]["paths"]


def test_every_accuracy_case_reaches_the_dashboard_with_its_models_threshold(
    tmp_path,
):
    root = WORKFLOWS.parents[1]
    catalog = root / ".github" / "benchmark" / "models_accuracy.json"
    cases = _accuracy_cases(tmp_path, "schedule")
    accuracy_name = _jobs("atomesh-accuracy-validation.yaml")["atomesh-test"]["env"][
        "ACCURACY_NAME"
    ]
    # The names the workflow gives each case's artifacts and dashboard row.
    names = {
        github_expressions(
            accuracy_name.replace("matrix.model_name", repr(c["model_name"])).replace(
                "matrix.serving", repr(c["serving"])
            ),
            "schedule",
        ): c["model_name"]
        for c in cases
    }
    assert "gpt-oss-120b-1p1d" in names
    results = tmp_path / "results"
    for name in {*names, *names.values()}:
        (results / f"accuracy-{name}").mkdir(parents=True)
        (results / f"accuracy-{name}" / "result.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.9}}})
        )
    output = tmp_path / "entries.json"
    subprocess.run(
        [
            sys.executable,
            str(root / ".github" / "scripts" / "accuracy_to_dashboard.py"),
            str(results),
            f"--output={output}",
            f"--models={catalog}",
            "--backend=ATOMesh",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    extra = {
        e["name"]: e.get("extra", "")
        for e in json.loads(output.read_text())
        if e["name"].endswith(" accuracy (GSM8K)")
    }
    dashboard = (
        root / ".github" / "dashboard" / "atomesh_mocker_index.html"
    ).read_text()
    shown = re.search(r"const atomeshModels=new Set\(\[(.*?)\]\)", dashboard).group(1)
    for name, model in names.items():
        assert f"'{name}'" in shown, name
        assert "Threshold: " in extra[f"ATOMesh::{name} accuracy (GSM8K)"], name
        assert (
            extra[f"ATOMesh::{name} accuracy (GSM8K)"]
            == extra[f"ATOMesh::{model} accuracy (GSM8K)"]
        )


def _read_when_written(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not (path.exists() and path.read_text()):
        assert time.monotonic() < deadline, f"nothing written to {path}"
        time.sleep(0.05)
    return path.read_text()


def _run_launcher(tmp_path, curl_fails="", server_stays_up=False, **env):
    """Run the 1P1D launcher with the servers, the router, curl and sleep stubbed.

    curl fails for the URLs matching the shell pattern `curl_fails`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The servers and the router print how they were started into their logs,
    # and curl logs the URLs it was given.
    server = (
        'echo "GPUS=$HIP_VISIBLE_DEVICES ENTRY=$USE_ATOMESH_ENTRYPOINTS HOME=$HOME $*"'
    )
    if server_stays_up:
        server += f"\nexec {shutil.which('sleep')} 10"
    curl = (
        'for a in "$@"; do case "$a" in http*) url="$a"; echo "$a" >> "$CURL_LOG";;'
        " esac; done"
    )
    if curl_fails:
        curl += f'\ncase "$url" in {curl_fails}) exit 7;; esac'
    stubs = {"python3": server, "atomesh": 'echo "$*"', "curl": curl, "sleep": ""}
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    script = WORKFLOWS.parent / "scripts" / "atomesh_pd_accuracy.sh"
    return subprocess.run(
        ["bash", str(script), "/models/m", "--kv_cache_dtype", "fp8"],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            TMPDIR=str(tmp_path),
            ATOMESH_BINARY=str(bin_dir / "atomesh"),
            ATOM_HOST_IP="10.0.0.5",
            ATOM_SERVER_PORT="8000",
            ATOM_SERVER_LOG=str(tmp_path / "atom_server.log"),
            CURL_LOG=str(tmp_path / "curl.log"),
            **env,
        ),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _check_the_launch(tmp_path, env, prefill_gpus, decode_gpus, decode_handshake_port):
    result = _run_launcher(tmp_path, **env)
    assert result.returncode == 0, result.stdout + result.stderr

    prefill = _read_when_written(tmp_path / "atom_prefill.log")
    decode = _read_when_written(tmp_path / "atom_decode.log")
    router = _read_when_written(tmp_path / "atom_server.log")
    for log, gpus, port, role, handshake_port in (
        (prefill, prefill_gpus, "8010", "kv_producer", "6301"),
        (decode, decode_gpus, "8020", "kv_consumer", decode_handshake_port),
    ):
        assert log.startswith(f"GPUS={gpus} ENTRY=0 "), log
        assert f"--model /models/m --server-port {port} --kv_cache_dtype fp8" in log
        assert (
            f'{{"kv_role":"{role}","kv_connector":"mooncake","proxy_ip":"10.0.0.5",'
            f'"handshake_port":{handshake_port},"protocol":"tcp"}}'
        ) in log
    assert prefill.split()[2] != decode.split()[2]
    assert router.split() == [
        "launch",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--pd-disaggregation",
        "--prefill",
        "http://10.0.0.5:8010",
        "--decode",
        "http://10.0.0.5:8020",
        "--policy",
        "random",
        "--backend",
        "atom",
        "--log-level",
        "info",
        "--disable-circuit-breaker",
    ]
    assert (tmp_path / "curl.log").read_text().split() == [
        "http://10.0.0.5:8010/health",
        "http://10.0.0.5:8020/health",
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8000/v1/completions",
    ]


def test_the_1p1d_launcher_puts_the_router_in_front_of_both_servers(tmp_path):
    _check_the_launch(tmp_path, {}, "0", "1", "6302")


def test_the_1p1d_launcher_gives_each_prefill_rank_a_handshake_port(tmp_path):
    _check_the_launch(
        tmp_path, {"PREFILL_GPUS": "0,1", "DECODE_GPUS": "2"}, "0,1", "2", "6303"
    )


def test_the_1p1d_launcher_fails_when_a_server_exits_before_answering(tmp_path):
    result = _run_launcher(tmp_path, curl_fails="*")
    assert result.returncode == 1
    assert "Exited before http://10.0.0.5:8010/health answered" in result.stdout
    assert set((tmp_path / "curl.log").read_text().split()) == {
        "http://10.0.0.5:8010/health"
    }


def test_the_1p1d_launcher_fails_when_a_server_never_answers(tmp_path):
    result = _run_launcher(
        tmp_path, curl_fails="*", server_stays_up=True, WAIT_SECONDS="0"
    )
    assert result.returncode == 1
    assert "No answer from http://10.0.0.5:8010/health in 0s" in result.stdout
    assert (tmp_path / "curl.log").read_text().split() == [
        "http://10.0.0.5:8010/health"
    ]


def test_the_1p1d_launcher_fails_when_no_completion_gets_through(tmp_path):
    result = _run_launcher(tmp_path, curl_fails="*/v1/completions")
    assert result.returncode == 1
    assert "No completion through the router" in result.stdout
    assert (tmp_path / "curl.log").read_text().split() == [
        "http://10.0.0.5:8010/health",
        "http://10.0.0.5:8020/health",
        "http://127.0.0.1:8000/v1/models",
    ] + ["http://127.0.0.1:8000/v1/completions"] * 10


def test_atomesh_benchmark_uses_required_heavy_ci_gate():
    workflow = yaml.safe_load((WORKFLOWS / "atomesh-benchmark.yaml").read_text())
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]
    assert "pull_request_review" in events
    assert "labeled" in events["pull_request"]["types"]
    assert "closed" in events["pull_request"]["types"]

    jobs = workflow["jobs"]
    gate_env = next(
        step["env"] for step in jobs["ci-gate"]["steps"] if step.get("id") == "gate"
    )
    assert gate_env["CI_GATE_WORKFLOW"] == "atomesh-benchmark.yaml"
    assert gate_env["CI_GATE_LABELS"] == "ci:full,ci:mesh"
    assert "atom/mesh/**" in gate_env["CI_GATE_PATHS_INCLUDE"]
    assert ".github/benchmark/models_atomesh.yaml" in gate_env["CI_GATE_PATHS_INCLUDE"]

    assert jobs["load-config"]["needs"] == ["ci-gate"]
    assert jobs["rdma-smoke-test"]["needs"] == ["ci-gate", "load-config"]
    assert jobs["run-model-benchmark"]["needs"] == ["ci-gate", "load-config"]

    test_gate = jobs["atomesh-benchmark-gate"]
    assert test_gate["if"] == "always()"
    assert test_gate["needs"] == [
        "ci-gate",
        "load-config",
        "rdma-smoke-test",
        "run-model-benchmark",
        "summarize-results",
    ]
    evaluate_step = next(
        step
        for step in test_gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    evaluate_inputs = evaluate_step["with"]
    assert (
        "load-config=${{ needs.load-config.result }}"
        in evaluate_inputs["required-results"]
    )
    assert "run-model-benchmark=${{ needs.run-model-benchmark.result }}" in (
        evaluate_inputs["optional-results"]
    )


def test_a_push_to_main_runs_the_benchmark_smoke():
    jobs = _jobs("atomesh-benchmark.yaml")
    matrix_env = next(
        step["env"]
        for step in jobs["load-config"]["steps"]
        if step.get("id") == "matrix"
    )
    assert github_expressions(matrix_env["SUITE"], "push") == "smoke"


def test_only_push_and_schedule_run_and_require_the_benchmark():
    jobs = _jobs("atomesh-benchmark.yaml")
    condition = " ".join(jobs["run-model-benchmark"]["if"].split()).replace(
        "needs.load-config.outputs.has_matrix", "'true'"
    )
    gate_inputs = next(
        step["with"]
        for step in jobs["atomesh-benchmark-gate"]["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    # A failed benchmark fails the gate only where its result is required.
    required = next(
        line
        for line in gate_inputs["required-results"].splitlines()
        if line.startswith("run-model-benchmark=")
    ).replace("needs.run-model-benchmark.result", "'failure'")
    for event, runs in [
        ("pull_request", False),
        ("pull_request_review", False),
        ("push", True),
        ("schedule", True),
    ]:
        assert github_expressions("${{ " + condition + " }}", event) == str(runs), event
        assert github_expressions(required, event) == (
            "run-model-benchmark=" + ("failure" if runs else "success")
        ), event


def test_atomesh_heavy_gate_concurrency_groups_separate_non_approval_reviews():
    cases = [
        ("atomesh-accuracy-validation.yaml", "Atomesh Accuracy Validation"),
        ("atomesh-benchmark.yaml", "Atomesh Benchmark"),
    ]

    for filename, workflow_name in cases:
        group = _workflow(filename)["concurrency"]["group"]
        pull_request_group = github_expressions(
            group,
            "pull_request",
            workflow=workflow_name,
            pull_request_number="3313",
            run_id="100",
        )
        approval_group = github_expressions(
            group,
            "pull_request_review",
            workflow=workflow_name,
            pull_request_number="3313",
            review_state="approved",
            run_id="200",
        )
        comment_group = github_expressions(
            group,
            "pull_request_review",
            workflow=workflow_name,
            pull_request_number="3313",
            review_state="commented",
            run_id="300",
        )

        assert approval_group == pull_request_group
        assert comment_group == f"{pull_request_group}-review-300"


def test_mocker_benchmark_stays_off_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["if"] == "${{ github.event_name != 'pull_request' }}"
    assert jobs["run_atomesh_test_harness"]["if"] == (
        "${{ needs.unit-test-scope.outputs.should_run == 'true' }}"
    )


def test_every_pull_request_publishes_the_atomesh_unit_test_gate():
    workflow = _workflow("atomesh-mocker-benchmark.yaml")
    events = workflow.get("on", workflow.get(True))
    assert "paths" not in events["pull_request"]

    jobs = workflow["jobs"]
    assert jobs["run_atomesh_test_harness"]["needs"] == ["unit-test-scope"]
    gate = jobs["unit-test-gate"]
    assert gate["name"] == "ATOMesh non-GPU unit tests"
    assert gate["if"] == "always()"
    assert gate["needs"] == ["unit-test-scope", "run_atomesh_test_harness"]
    inputs = next(
        step["with"]
        for step in gate["steps"]
        if step.get("uses") == "./.github/actions/evaluate-test-gate"
    )
    assert inputs["should-run"] == "${{ needs.unit-test-scope.outputs.should_run }}"
    assert inputs["decidable"] == "true"
    assert inputs["reason"] == "${{ needs.unit-test-scope.outputs.reason }}"
    assert inputs["ci-gate-result"] == "${{ needs.unit-test-scope.result }}"
    assert inputs["required-results"].strip() == (
        "run_atomesh_test_harness=${{ needs.run_atomesh_test_harness.result }}"
    )


def test_the_unit_test_scope_matches_the_push_paths():
    workflow = _workflow("atomesh-mocker-benchmark.yaml")
    events = workflow.get("on", workflow.get(True))
    scope = workflow["jobs"]["unit-test-scope"]["steps"][0]
    assert scope["env"]["PATHS_INCLUDE"].split() == events["push"]["paths"]


def test_the_unit_test_scope_follows_the_changed_files(tmp_path):
    scope = _jobs("atomesh-mocker-benchmark.yaml")["unit-test-scope"]["steps"][0]
    gh = tmp_path / "gh"
    # Answers the step's two gh api calls through the step's own --jq.
    gh.write_text(
        "#!/bin/sh\n"
        'case "$*" in */files*) body="$FILES_JSON" ;; *) body="$PULL_JSON" ;; esac\n'
        'while [ "$1" != "--jq" ]; do shift; done\n'
        'printf "%s" "$body" | jq -r "$2"\n'
    )
    gh.chmod(0o755)
    mesh = {"filename": "atom/mesh/src/lib.rs"}
    moved_out = {
        "filename": "atom/router/lib.rs",
        "previous_filename": mesh["filename"],
    }
    cases = (
        ("1", False, [{"filename": "docs/a.md"}, mesh], "true"),
        ("1", False, [{"filename": "atom/model_engine/a.py"}], "false"),
        ("1", False, [moved_out], "true"),
        ("1", True, [mesh], "false"),
        ("", False, [], "true"),
    )
    for pr_number, draft, files, should_run in cases:
        output = tmp_path / "output"
        output.write_text("")
        env = {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "ROCm/ATOM",
            "PATHS_INCLUDE": scope["env"]["PATHS_INCLUDE"],
            "PR_NUMBER": pr_number,
            "PULL_JSON": json.dumps({"draft": draft}),
            "FILES_JSON": json.dumps(files),
        }
        subprocess.run(["bash", "-c", scope["run"]], env=env, check=True, timeout=10)
        lines = output.read_text().splitlines()
        assert f"should_run={should_run}" in lines, (pr_number, draft, files)


def _steps(job):
    return {step["name"]: step for step in job["steps"]}


def test_mocker_cargo_cache_is_saved_from_clean_builds_outside_pull_requests():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    for job in jobs.values():
        assert not any(
            step.get("uses", "").startswith("actions/cache@") for step in job["steps"]
        )
    harness = jobs["run_atomesh_test_harness"]
    restore = _steps(harness)["Restore cargo build output"]["with"]
    save = _steps(harness)["Save cargo build output"]
    assert (
        restore["key"] == save["with"]["key"] == "${{ steps.cargo-deps.outputs.key }}"
    )
    assert restore["path"] == save["with"]["path"]
    assert restore["restore-keys"] == (
        "${{ github.event_name == 'pull_request' && "
        "steps.cargo-deps.outputs.prefix || '' }}"
    )
    assert save["if"] == (
        "${{ github.event_name != 'pull_request' && "
        "steps.cargo-cache.outputs.cache-hit != 'true' }}"
    )
    assert harness["outputs"] == {
        "cargo_cache_key": "${{ steps.cargo-deps.outputs.key }}",
        "cargo_cache_prefix": "${{ steps.cargo-deps.outputs.prefix }}",
    }


def test_mocker_benchmark_only_restores_the_harness_cache():
    jobs = _jobs("atomesh-mocker-benchmark.yaml")
    benchmark = jobs["run_atomesh_mocker_benchmark"]
    assert benchmark["needs"] == ["run_atomesh_test_harness"]
    restore = _steps(benchmark)["Restore cargo build output"]
    assert restore["uses"] == "actions/cache/restore@v4"
    assert restore["with"]["key"] == (
        "${{ needs.run_atomesh_test_harness.outputs.cargo_cache_key }}"
    )
    assert restore["with"]["restore-keys"] == (
        "${{ needs.run_atomesh_test_harness.outputs.cargo_cache_prefix }}"
    )
    saved = _steps(jobs["run_atomesh_test_harness"])["Save cargo build output"]
    assert restore["with"]["path"] == saved["with"]["path"]
    assert not any(
        step.get("uses", "").startswith("actions/cache/save@")
        for step in benchmark["steps"]
    )
