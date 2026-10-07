"""CPU-only paired APC configuration and actual launcher argv contracts."""

import copy
import hashlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml
from test_atomesh_native_nixl_wiring import exported, shell_render

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SEED = "rocm-pd-apc-20261007"
MODELS = {
    "v4": ("DeepSeek-V4-Flash-vLLM-Survey", "survey-v4-flash-main-nixl-1p1d-tp8"),
    "m3": ("MiniMax-M3-MXFP8-vLLM-Survey", "survey-m3-mxfp8-main-nixl-1p1d-tp8"),
}


@pytest.fixture
def cells(monkeypatch):
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
        "apc_matrix", SCRIPTS / "pd_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    return {
        cell["name"]: cell
        for cell in matrix.build_cells(
            matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
            suite="vllm",
            model_filter={value[0] for value in MODELS.values()},
            case_filter=None,
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
    }


@pytest.mark.parametrize("profile", MODELS)
def test_paired_cells_preserve_original_semantics(cells, profile):
    model, prefix = MODELS[profile]
    original = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )["models"][model]
    original["suites"]["vllm"] = [
        cell for cell in original["suites"]["vllm"] if cell["name"] == prefix + "-eager"
    ]
    assert (
        hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        == {
            "v4": "84c01881fe1ad6a99e9cf79656df1a6bec5187323e9f93e3222f42dbbe8d9a83",
            "m3": "50b05ed803e778fd3dc8199665fbe35bf6f14a71d92a780567879adb362de697",
        }[profile]
    )
    assert len(cells) == 6
    for mode in ("eager", "full-decode-only"):
        derived = copy.deepcopy(cells[prefix + "-apc-" + mode])
        base = copy.deepcopy(cells[prefix + "-eager"])
        assert derived["vllm"].pop("cache_composition") == 1
        assert derived["vllm"].pop("cudagraph_metrics") == 1
        assert derived["vllm"].pop("prompt_seed") == SEED
        args = shlex.split(derived["server_args"].pop("extra_args"))
        base_args = shlex.split(base["server_args"].pop("extra_args"))
        assert args.count("--enforce-eager") == int(mode == "eager")
        assert args.count("--enable-prefix-caching") == 1
        assert "--no-enable-prefix-caching" not in args
        for flag in ("--cudagraph-metrics", "--enable-logging-iteration-details"):
            assert args.count(flag) == 1
            args.remove(flag)
        assert args.count("--compilation-config") == 1
        index = args.index("--compilation-config")
        config = json.loads(args.pop(index + 1))
        args.pop(index)
        assert config == (
            {"mode": 0, "cudagraph_mode": "NONE"}
            if mode == "eager"
            else {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
            }
        )
        args.remove("--enable-prefix-caching")
        base_args.remove("--no-enable-prefix-caching")
        if mode != "eager":
            base_args.remove("--enforce-eager")
        assert args == base_args
        for key in ("id", "name"):
            derived.pop(key)
            base.pop(key)
        assert derived == base


@pytest.mark.parametrize("profile", MODELS)
@pytest.mark.parametrize("mode", ["eager", "apc-eager", "apc-full-decode-only"])
def test_actual_both_role_launch_and_client_arguments(cells, tmp_path, profile, mode):
    cell = cells[MODELS[profile][1] + "-" + mode]
    result = shell_render(
        cell,
        tmp_path,
        "start_vllm_server prefill prefill 2584\nstart_vllm_server decode decode 3584",
    )
    assert result.returncode == 0, result.stderr
    for role, port in (("prefill", "2584"), ("decode", "3584")):
        argv = json.loads((tmp_path / f"{role}.launch.json").read_text())["argv"]
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"backends": ["UCX"]},
        }
        assert argv == [
            "vllm",
            "serve",
            cell["model_path"],
            "--served-model-name",
            cell["vllm"]["served_model_name"],
            "--port",
            port,
            "--trust-remote-code",
            "--kv-transfer-config",
            json.dumps(config),
            "--tensor-parallel-size",
            "8",
            *shlex.split(cell["server_args"]["extra_args"]),
        ]
        assert "--disable-log-stats" not in argv
        assert "--speculative-config" not in argv
        assert argv.count("--enforce-eager") == int(mode != "apc-full-decode-only")
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
                "PREFILL_PORT=2584; DECODE_PORT=3584",
                "SERVED_MODEL_NAME=" + shlex.quote(cell["vllm"]["served_model_name"]),
                "RUN_DIR=/run; ATOMESH_EXECUTION_PHASE=benchmark",
                'python3() { printf "%s\\n" "$@"; }',
                function,
                "run_workload_phase",
            ]
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    expected = [
        str(SCRIPTS / "pd_m3_nixl_smoke.py"),
        "--model-profile",
        profile,
        "--prefill",
        "http://192.0.2.1:2584",
        "--decode",
        "http://192.0.2.2:3584",
        "--model",
        cell["vllm"]["served_model_name"],
        "--output",
        "/run/pd-diagnostic/benchmark",
    ]
    if mode != "eager":
        expected += [
            "--cache-composition",
            "--cudagraph-metrics",
            "--prompt-seed",
            SEED,
        ]
    assert result.stdout.splitlines() == expected
