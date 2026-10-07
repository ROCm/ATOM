"""GPU-hidden push survey checks at exported environment and process argv seams."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from test_atomesh_native_nixl_wiring import exported

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


@pytest.fixture
def push_cell(monkeypatch):
    for key, value in {
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/logs",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_1P1D_NODES": "node-a,node-b",
    }.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "push_matrix", SCRIPTS / "pd_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    cells = matrix.build_cells(
        matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
        suite="vllm",
        model_filter=None,
        case_filter={"survey-qwen3-main-nixl-push-1p1d-tp1-eager"},
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    assert len(cells) == 1
    return cells[0]


def render(cell, tmp_path, commands, *, rank=0, override=""):
    definitions = (
        (SCRIPTS / "pd_server_vllm.sh")
        .read_text()
        .split('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then')[0]
    )
    script = (
        exported(cell)
        + "\n"
        + "\n".join(
            [
                "set -euo pipefail",
                "export PATH="
                + shlex.quote(str(Path(sys.executable).parent))
                + ":$PATH",
                "export ATOMESH_RUN_TOKEN=fixture-run ATOMESH_EXECUTION_PHASE=benchmark",
                "ATOMESH_SERVICE_PORT_OFFSET=1000; NODE0_ADDR=192.0.2.1",
                f"NODE_RANK={rank}; host_ip=192.0.2.{rank + 1}; host_name=fixture",
                "IP_ARRAY=(192.0.2.1 192.0.2.2); xP=1; yD=1",
                "PREFILL_TP_SIZE=$PREFILL_TP; DECODE_TP_SIZE=$DECODE_TP",
                "PREFILL_PORT=3584; DECODE_PORT=4584; ROUTER_PORT=9000",
                'PREFILL_SERVER_ARGS="$EXTRA_SERVER_ARGS $PREFILL_EXTRA_SERVER_ARGS"',
                'DECODE_SERVER_ARGS="$EXTRA_SERVER_ARGS $DECODE_EXTRA_SERVER_ARGS"',
                "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
                "RUNTIME_LOG_DIR="
                + shlex.quote(str(tmp_path))
                + "; RUN_DIR=$RUNTIME_LOG_DIR",
                "apply_role_env() { :; }; build_server_cache_env() { :; }; dump_launch_info() { :; }",
                """start_logged_process() {
  local pid_var="$1"; shift
  python3 -c 'import json,sys; print("LAUNCH=" + json.dumps(sys.argv[1:]))' "$@"
  printf -v "$pid_var" '%s' 12345
}""",
                override,
                definitions,
                commands,
            ]
        )
    )
    return subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
    )


def launches(result):
    assert result.returncode == 0, result.stderr
    return [
        json.loads(line.removeprefix("LAUNCH="))
        for line in result.stdout.splitlines()
        if line.startswith("LAUNCH=")
    ]


@pytest.mark.parametrize(
    "role,rank,port,side_port",
    [("prefill", 0, 3584, 16559), ("decode", 1, 4584, 16659)],
)
def test_push_server_process_receives_exact_connector_and_coordinates(
    push_cell, tmp_path, role, rank, port, side_port
):
    result = render(
        push_cell, tmp_path, f"start_vllm_server {role} {role} {port}", rank=rank
    )
    [launch] = launches(result)
    assert f"VLLM_NIXL_SIDE_CHANNEL_HOST=192.0.2.{rank + 1}" in launch
    assert f"VLLM_NIXL_SIDE_CHANNEL_PORT={side_port}" in launch
    argv = launch[launch.index("vllm") :]
    config = json.loads(argv[argv.index("--kv-transfer-config") + 1])
    assert config == {
        "kv_connector": "NixlPushConnector",
        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"backends": ["UCX"]},
        "engine_id": f"fixture-run-benchmark-{role}",
    }
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert "--no-enable-prefix-caching" in argv
    assert "--enforce-eager" in argv
    assert "--enable-prefix-caching" not in argv


@pytest.mark.parametrize(
    "override",
    [
        "export ATOMESH_VLLM_NATIVE_CPU_BYTES=2147483648",
        "PREFILL_TP_SIZE=2",
        "DECODE_TP_SIZE=2",
        "PREFILL_DCP_SIZE=2",
        "DECODE_DCP_SIZE=2",
        "xP=2",
        "yD=2",
        "export ATOMESH_PD_WORKER_LAYOUT=single_node",
        "export ATOMESH_VLLM_CONNECTOR=moriio",
        "unset ATOMESH_RUN_TOKEN",
        "host_ip=192.0.2.99",
        "IP_ARRAY+=(192.0.2.3)",
        "IP_ARRAY=(192.0.2.1 192.0.2.1)",
        "PREFILL_SERVER_ARGS+=' --pipeline-parallel-size 2'",
        "PREFILL_SERVER_ARGS+=' --enable-prefix-caching'",
        "PREFILL_SERVER_ARGS+=' --speculative-config {}'",
    ],
)
def test_push_rejects_unsupported_configuration_before_launch(
    push_cell, tmp_path, override
):
    result = render(
        push_cell, tmp_path, "start_vllm_server prefill prefill 3584", override=override
    )
    assert result.returncode != 0
    assert "LAUNCH=" not in result.stdout


def test_push_cell_and_direct_client_keep_original_model_and_push_coordinates(
    push_cell, tmp_path
):
    assert push_cell["model_path"] == "/models/Qwen/Qwen3-0.6B"
    assert push_cell["precision"] == "BF16"
    assert push_cell["vllm"]["source"] == {
        "repo": "https://github.com/vllm-project/vllm",
        "sha": "b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
    }
    assert push_cell["num_nodes"] == 2
    assert push_cell["isl"] == [1025] and push_cell["osl"] == 32
    assert push_cell["concurrency"] == [1]
    assert "native_cpu_bytes" not in push_cell["vllm"]
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("run_workload_phase() {")
    function = source[start : source.index("\n}", start) + 2]
    result = render(
        push_cell,
        tmp_path,
        function
        + "\n"
        + 'python3() { printf "CLIENT=%s\\n" "$@"; }; run_workload_phase',
    )
    assert result.returncode == 0, result.stderr
    args = [
        line.removeprefix("CLIENT=")
        for line in result.stdout.splitlines()
        if line.startswith("CLIENT=")
    ]
    assert args == [
        str(SCRIPTS / "pd_nixl_push_survey.py"),
        "--prefill",
        "http://192.0.2.1:3584",
        "--decode",
        "http://192.0.2.2:4584",
        "--model",
        "Qwen3-0.6B",
        "--output",
        str(tmp_path / "pd-diagnostic/benchmark"),
        "--prefill-engine-id",
        "fixture-run-benchmark-prefill",
        "--prefill-kv-host",
        "192.0.2.1",
        "--prefill-side-channel-port",
        "16559",
    ]


def test_push_proxy_uses_frozen_routes_external_bind_and_matching_metadata(
    push_cell, tmp_path, monkeypatch
):
    import asyncio
    import runpy

    import uvicorn

    result = render(
        push_cell,
        tmp_path,
        'start_router; printf "PATHS=%s,%s PID=%s\\n" "$ROUTER_READY_PATH" "$ROUTER_ALIVE_PATH" "$router_pid"',
    )
    [launch] = launches(result)
    assert launch[0].endswith("nixl-push-proxy.log")
    assert launch[1:3] == ["python3", "-c"]
    assert "toy_proxy_server" not in str(launch)
    assert "PATHS=/status,/status PID=12345" in result.stdout
    assert launch[4:] == [
        "--model",
        "Qwen3-0.6B",
        "--prefill",
        "192.0.2.1:3584",
        "--decode",
        "192.0.2.2:4584",
        "--port",
        "9000",
        "--prefill-engine-id",
        "fixture-run-benchmark-prefill",
        "--prefill-kv-host",
        "192.0.2.1",
        "--prefill-side-channel-port",
        "16559",
        "--prefill-tp-size",
        "1",
        "--prefill-pp-size",
        "1",
    ]
    frozen = (
        ROOT.parent
        / "vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py"
    )
    if not frozen.is_file():
        pytest.skip(
            "Bootstrap route execution needs the separately checked-out frozen vLLM source"
        )
    original_run_path = runpy.run_path

    def load_frozen(path):
        assert (
            path
            == "/tmp/atomesh-native-vllm/examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py"
        )
        return original_run_path(str(frozen))

    observed = {}

    def capture_server(app, **kwargs):
        observed.update(kwargs)
        status = next(route for route in app.routes if route.path == "/status")
        observed["status"] = asyncio.run(status.endpoint())

    monkeypatch.setattr(runpy, "run_path", load_frozen)
    monkeypatch.setattr(uvicorn, "run", capture_server)
    monkeypatch.setattr(sys, "argv", ["-c", *launch[4:]])
    # Execute the captured trusted launcher bootstrap, never request input.
    exec(launch[3], {})  # noqa: S102
    assert observed == {
        "host": "0.0.0.0",
        "port": 9000,
        "loop": "uvloop",
        "status": {
            "mode": "push",
            "prefill_node_count": 1,
            "decode_node_count": 1,
            "prefill_nodes": ["192.0.2.1:3584"],
            "decode_nodes": ["192.0.2.2:4584"],
            "prefill_engine_id": "fixture-run-benchmark-prefill",
            "prefill_kv_host": "192.0.2.1",
            "prefill_side_channel_port": 16559,
            "prefill_tp_size": 1,
            "prefill_pp_size": 1,
        },
    }
