"""CPU-only frozen DSpark D-graph case and actual role argv contracts."""

import copy
import hashlib
import json
import shlex
import subprocess

import pytest
import yaml
from test_atomesh_k3_composition_wiring import DRAFT, ROOT, SCRIPTS
from test_atomesh_k3_composition_wiring import cells as k3_cells
from test_atomesh_native_nixl_wiring import exported, shell_render

CASE = "survey-k3-main-read-dspark3-d-graph-c1-1p1d-tp8-dcp8"
cells = k3_cells


@pytest.mark.parametrize(
    "name,mode",
    [
        (CASE, "graph-correctness"),
        (DRAFT, "smoke"),
        (DRAFT, "profile"),
        (DRAFT, "bounded-perf"),
    ],
)
@pytest.mark.parametrize("client_rc", [0, 7])
def test_actual_client_dispatch_passes_shared_decode_log_only_for_graph(
    cells, name, mode, client_rc
):
    cell = copy.deepcopy(cells[name])
    cell["vllm"]["diagnostic_mode"] = mode
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("run_workload_phase() {")
    function = source[start : source.index("\n}", start) + 2]
    shell = (
        exported(cell)
        + "\n"
        + "\n".join(
            [
                "set -euo pipefail",
                "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
                "NODE0_ADDR=192.0.2.1; IP_ARRAY=(192.0.2.1 192.0.2.2)",
                "PREFILL_PORT=2584; DECODE_PORT=3584; SERVED_MODEL_NAME=Kimi-K3",
                "RUN_DIR='/shared logs/run'; ATOMESH_EXECUTION_PHASE=benchmark",
                'RUNTIME_LOG_DIR="${RUN_DIR}/logs/${ATOMESH_EXECUTION_PHASE}"',
                "PREFILL_TP_SIZE=$PREFILL_TP",
                'python3() { printf "%s\\n" "$@"; return ' + str(client_rc) + "; }",
                function,
                "run_workload_phase",
            ]
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], check=False, capture_output=True, text=True
    )
    assert result.returncode == client_rc, result.stderr
    expected = [
        str(SCRIPTS / "pd_vllm_profile.py"),
        "--prefill",
        "http://192.0.2.1:2584",
        "--decode",
        "http://192.0.2.2:3584",
        "--model",
        "Kimi-K3",
        "--tokenizer",
        cell["model_path"],
        "--output",
        "/shared logs/run/pd-diagnostic/benchmark",
        "--phase",
        "benchmark",
        "--mode",
        mode,
        "--tp",
        "8",
        "--dcp",
        "8",
        "--hybrid",
    ]
    if mode == "graph-correctness":
        expected += [
            "--decode-log",
            "/shared logs/run/logs/benchmark/decode-rank-1.log",
        ]
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("backend,diagnostic", [("atom", "1"), ("vllm", "0")])
def test_graph_mode_does_not_change_nondiagnostic_dispatch(cells, backend, diagnostic):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("run_workload_phase() {")
    function = source[start : source.index("\n}", start) + 2]
    result = subprocess.run(
        [
            "bash",
            "-c",
            exported(cells[CASE])
            + "\n"
            + "\n".join(
                [
                    "set -euo pipefail",
                    f"BACKEND={backend}; ATOMESH_VLLM_DIAGNOSTIC={diagnostic}",
                    "ATOMESH_EXECUTION_PHASE=benchmark",
                    "run_benchmark() { printf 'ordinary-benchmark\\n'; }",
                    "python3() { return 99; }",
                    function,
                    "run_workload_phase",
                ]
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ordinary-benchmark\n"


GRAPH = {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [3, 4],
    "max_cudagraph_capture_size": 4,
}


def test_all_existing_k3_cells_and_model_defaults_are_unchanged():
    config = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )["models"]["Kimi-K3-vLLM-Survey"]
    assert sum(cell["name"] == CASE for cell in config["suites"]["vllm"]) == 1
    config["suites"]["vllm"] = [
        cell for cell in config["suites"]["vllm"] if cell["name"] != CASE
    ]
    assert (
        hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        == "9ba686a1be9f8ebcb000cee5102f922892b08d9acd5d855655d0070b35eacf3e"
    )


def test_graph_cell_changes_only_workload_mode_and_role_graph_arguments(cells):
    cell = copy.deepcopy(cells[CASE])
    parent = copy.deepcopy(cells[DRAFT])
    assert cell["isl"] == [1025]
    assert cell["osl"] == 16
    assert cell["concurrency"] == [1]
    assert cell["concurrency_x"] == "1"
    assert cell["accuracy"]["concurrency"] == [1]
    assert cell["vllm"]["diagnostic_mode"] == "graph-correctness"
    cell["vllm"]["diagnostic_mode"] = parent["vllm"]["diagnostic_mode"]
    cell["accuracy"]["concurrency"] = parent["accuracy"]["concurrency"]
    common = shlex.split(cell["server_args"].pop("extra_args"))
    eager = shlex.split(parent["server_args"].pop("extra_args"))
    assert "--enforce-eager" not in common
    eager.remove("--enforce-eager")
    assert common == eager
    for role in ("prefill", "decode"):
        cell["service"][role].pop("extra_args")
        parent["service"][role].pop("extra_args")
    for key in ("id", "name", "isl", "concurrency", "concurrency_x"):
        cell.pop(key)
        parent.pop(key)
    assert cell == parent


@pytest.mark.parametrize("role,port", [("prefill", 2584), ("decode", 3584)])
def test_graph_actual_role_argv_preserves_every_other_parent_argument(
    cells, tmp_path, monkeypatch, role, port
):
    # Only render shell argv; keep the required log variable defined and GPUs hidden.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    launches = []
    for name in (DRAFT, CASE):
        output = tmp_path / name
        output.mkdir()
        result = shell_render(
            cells[name], output, f"start_vllm_server {role} {role} {port}"
        )
        assert result.returncode == 0, result.stderr
        launches.append(
            json.loads((output / f"{role}.launch.json").read_text())["argv"]
        )
    parent, graph = launches
    if role == "prefill":
        assert graph.count("--enforce-eager") == 1
        assert "--compilation-config" not in graph
        assert "--kernel-config" not in graph
        graph.remove("--enforce-eager")
        parent.remove("--enforce-eager")
    else:
        assert "--enforce-eager" not in graph
        parent.remove("--enforce-eager")
        for option, expected in {
            "--compilation-config": GRAPH,
            "--kernel-config": {"enable_jit_warmup": False},
        }.items():
            assert graph.count(option) == 1
            index = graph.index(option)
            assert json.loads(graph[index + 1]) == expected
            del graph[index : index + 2]
        for option in ("--cudagraph-metrics", "--enable-logging-iteration-details"):
            assert graph.count(option) == 1
            graph.remove(option)
    assert graph == parent
    assert not any(
        arg.startswith(("--profiler-config", "--quantization", "--hf-overrides"))
        for arg in graph
    )
    assert graph.count("--speculative-config") == 1
    assert json.loads(graph[graph.index("--speculative-config") + 1]) == {
        "model": "${ATOMESH_MODEL_ROOT}/Inferact/Kimi-K3-DSpark",
        "num_speculative_tokens": 3,
        "method": "dspark",
        "attention_backend": "ROCM_AITER_MLA",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }
