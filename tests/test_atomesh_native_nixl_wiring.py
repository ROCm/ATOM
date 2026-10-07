"""CPU-only shell rendering checks for the native CPU + NIXL survey cell."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
CASE = "survey-qwen3-main-native-nixl-1p1d-tp1-eager"


@pytest.fixture
def cell(monkeypatch):
    for key, value in {
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/logs",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_1P1D_NODES": "pit2-p03-g10,pit2-p03-g25",
    }.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "native_matrix", SCRIPTS / "pd_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    cells = matrix.build_cells(
        matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
        suite="vllm",
        model_filter=None,
        case_filter={CASE},
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    assert len(cells) == 1
    return cells[0]


def exported(cell):
    code = (
        (SCRIPTS / "pd_submit.sh")
        .read_text()
        .split("python3 - <<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    return subprocess.check_output(
        [sys.executable, "-c", code],
        env={**os.environ, "CELL_JSON": json.dumps(cell)},
        text=True,
    )


def shell_render(cell, tmp_path, commands):
    definitions = (
        (SCRIPTS / "pd_server_vllm.sh")
        .read_text()
        .split('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then')[0]
    )
    shell = (
        exported(cell)
        + "\n"
        + "\n".join(
            [
                "set -euo pipefail",
                "export PATH="
                + shlex.quote(str(Path(sys.executable).parent))
                + ":$PATH",
                "host_ip=192.0.2.1; host_name=cpu-fixture; NODE0_ADDR=192.0.2.1; NODE_RANK=0",
                "ATOMESH_SERVICE_PORT_OFFSET=0; ATOMESH_EXECUTION_PHASE=benchmark",
                "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
                "RUNTIME_LOG_DIR="
                + shlex.quote(str(tmp_path))
                + "; RUN_DIR=$RUNTIME_LOG_DIR",
                "PREFILL_TP_SIZE=$PREFILL_TP; DECODE_TP_SIZE=$DECODE_TP",
                'PREFILL_SERVER_ARGS="$EXTRA_SERVER_ARGS $PREFILL_EXTRA_SERVER_ARGS"',
                'DECODE_SERVER_ARGS="$EXTRA_SERVER_ARGS $DECODE_EXTRA_SERVER_ARGS"',
                "apply_role_env() { :; }; build_server_cache_env() { :; }",
                "dump_launch_info() { :; }; start_logged_process() { :; }",
                definitions,
                commands,
            ]
        )
    )
    return subprocess.run(
        ["bash", "-c", shell], check=False, capture_output=True, text=True
    )


def test_native_cell_preserves_frozen_source_weights_and_launch_contract(
    cell, tmp_path
):
    assert cell["model_path"] == "/models/Qwen/Qwen3-0.6B"
    assert cell["precision"] == "BF16"
    assert cell["vllm"]["source"] == {
        "repo": "https://github.com/vllm-project/vllm",
        "sha": "b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
    }
    assert (
        cell["image"]
        == "vllm/vllm-openai-rocm:nightly@sha256:659b28319fef4ea0e3d8f33e25b4c35d6f663f5d818a5b2fa06dceaf859234e4"
    )
    assert cell["num_nodes"] == 2
    assert cell["nodes"] == ["pit2-p03-g10", "pit2-p03-g25"]
    assert cell["runner"]["gpus_per_node"] == 8
    assert cell["runner"]["slurm_account"] == "amd-frameworks"
    assert cell["runner"]["slurm_partition"] == "amd-spur"
    assert cell["env"]["common"]["ATOMESH_PRE_CLEAN_GPU"] == "1"
    assert cell["env"]["common"]["VLLM_PLUGINS"] == ""
    assert not any(key in cell["vllm"] for key in ("fork", "lmcache", "hybrid"))
    result = shell_render(
        cell,
        tmp_path,
        "start_vllm_server prefill prefill 2584\nstart_vllm_server decode decode 3584",
    )
    assert result.returncode == 0, result.stderr
    for role in ("prefill", "decode"):
        argv = json.loads((tmp_path / f"{role}.launch.json").read_text())["argv"]
        for flag, value in {
            "--dtype": "bfloat16",
            "--tensor-parallel-size": "1",
            "--block-size": "128",
            "--max-model-len": "4096",
            "--max-num-seqs": "1",
            "--max-num-batched-tokens": "1024",
            "--kv-cache-memory-bytes": "1073741824",
            "--kv-cache-dtype": "auto",
            "--attention-backend": "ROCM_AITER_UNIFIED_ATTN",
        }.items():
            assert argv[argv.index(flag) + 1] == value
        assert "--enable-prefix-caching" in argv and "--enforce-eager" in argv
        assert not any(
            a.startswith(("--quantization", "--hf-overrides", "--speculative-config"))
            for a in argv
        )
        config = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        nixl = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"backends": ["UCX"]},
        }
        if role == "decode":
            assert config == nixl
        else:
            assert config == {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": {
                    "connectors": [
                        nixl,
                        {
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_load_failure_policy": "fail",
                            "kv_connector_extra_config": {
                                "spec_name": "CPUOffloadingSpec",
                                "block_size": 128,
                                "cpu_bytes_to_use": 2147483648,
                                "store_threshold": 0,
                                "offload_prompt_only": True,
                            },
                        },
                    ]
                },
            }


def test_ordinary_nixl_stays_plain_for_both_roles(cell, tmp_path):
    del cell["vllm"]["native_cpu_bytes"]
    result = shell_render(
        cell,
        tmp_path,
        "kv_transfer_config prefill 2584\nkv_transfer_config decode 3584",
    )
    assert result.returncode == 0, result.stderr
    configs = [json.loads(line) for line in result.stdout.splitlines()]
    assert [c["kv_role"] for c in configs] == ["kv_producer", "kv_consumer"]
    assert all(c["kv_connector"] == "NixlConnector" for c in configs)


@pytest.mark.parametrize("native", [False, True])
def test_client_dispatch_only_changes_explicit_native_cell(cell, tmp_path, native):
    if not native:
        del cell["vllm"]["native_cpu_bytes"]
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    function = source[
        source.index("run_workload_phase() {") : source.index(
            "\n}", source.index("run_workload_phase() {")
        )
        + 2
    ]
    shell = (
        exported(cell)
        + "\n"
        + "\n".join(
            [
                "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
                "NODE0_ADDR=192.0.2.1; IP_ARRAY=(192.0.2.1 192.0.2.2)",
                "PREFILL_PORT=2584; DECODE_PORT=3584; SERVED_MODEL_NAME=Qwen3-0.6B",
                "RUN_DIR=/run; ATOMESH_EXECUTION_PHASE=benchmark",
                'python3() { printf "%s\\n" "$@"; }',
                function,
                "run_workload_phase",
            ]
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0].endswith(
        "pd_native_nixl_survey.py" if native else "pd_m3_nixl_smoke.py"
    )
    if native:
        assert args[1:] == [
            "--prefill",
            "http://192.0.2.1:2584",
            "--decode",
            "http://192.0.2.2:3584",
            "--model",
            "Qwen3-0.6B",
            "--output",
            "/run/pd-diagnostic/benchmark",
        ]


def test_native_dispatch_inputs_are_bounded():
    inputs = json.loads(
        (ROOT / ".github/benchmark/rocm-pd-survey-native-inputs.json").read_text()
    )
    assert inputs["case_names"] == CASE
    assert inputs["atomesh_1p1d_nodes"] == "pit2-p03-g10,pit2-p03-g20"
    assert inputs["publish_dashboard"] == inputs["run_all_models"] == "false"
    assert inputs["atomesh_slurm_account"] == "amd-frameworks"
    assert inputs["atomesh_slurm_partition"] == "amd-spur"
