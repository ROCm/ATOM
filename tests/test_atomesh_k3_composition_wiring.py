"""CPU-only composition cell, export, and client-dispatch contracts."""

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
BASE = "survey-k3-main-read-1p1d-tp8-dcp8-eager"
DRAFT = "survey-k3-main-read-dspark3-1p1d-tp8-dcp8-eager"
PAIRS = {
    "survey-k3-main-read-apc-1p1d-tp8-dcp8-eager": BASE,
    "survey-k3-main-read-dspark3-apc-1p1d-tp8-dcp8-eager": DRAFT,
}


@pytest.fixture
def cells(monkeypatch):
    for name, value in {
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/logs",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_1P1D_NODES": "node-a,node-b",
    }.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        "composition_matrix", SCRIPTS / "pd_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config(ROOT / ".github/benchmark/models_atomesh.yaml")
    return {
        cell["name"]: cell
        for cell in module.build_cells(
            config,
            suite="vllm",
            model_filter={"Kimi-K3-vLLM-Survey"},
            case_filter=None,
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
    }


def test_only_two_new_cells_change_original_k3_semantics():
    config = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )["models"]["Kimi-K3-vLLM-Survey"]
    suites = config["suites"]
    removed = [cell for cell in suites["vllm"] if cell["name"] in PAIRS]
    assert len(removed) == 2
    suites["vllm"] = [cell for cell in suites["vllm"] if cell["name"] in (BASE, DRAFT)]
    # Preserve original K3 metadata/cells without coupling unrelated model updates.
    assert (
        hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        == "d7a155937d448692c486b9480aafa6c6431c8a7d7596183df8204c762a5423a0"
    )


@pytest.mark.parametrize("name,base", PAIRS.items())
def test_composition_is_exact_parent_cell_with_only_new_client_flag(cells, name, base):
    cell = copy.deepcopy(cells[name])
    parent = copy.deepcopy(cells[base])
    assert cell["vllm"].pop("cache_composition") == 1
    for key in ("id", "name"):
        cell.pop(key)
        parent.pop(key)
    assert cell == parent
    args = shlex.split(cell["server_args"]["extra_args"])
    assert args[args.index("--max-model-len") + 1] == "4096"
    assert args[args.index("--max-num-seqs") + 1] == "4"
    assert cell["vllm"]["source"]["sha"] == "b22494cc0cb4bd9db4a62fb107d92429a4a3249d"
    if base == DRAFT:
        path = cell["vllm"]["draft_model_path"]
        assert path == "${ATOMESH_MODEL_ROOT}/Inferact/Kimi-K3-DSpark"
        for role in ("prefill", "decode"):
            argv = shlex.split(cell["service"][role]["extra_args"])
            assert json.loads(argv[1])["model"] == path


@pytest.mark.parametrize("name", [BASE, DRAFT, *PAIRS])
def test_actual_exports_select_composition_only_for_new_cells(cells, name):
    cell = cells[name]
    export_code = (
        (SCRIPTS / "pd_submit.sh")
        .read_text()
        .split("python3 - <<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    exports = subprocess.check_output(
        [sys.executable, "-c", export_code],
        env={**os.environ, "CELL_JSON": json.dumps(cell)},
        text=True,
    )
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("run_workload_phase() {")
    function = source[start : source.index("\n}", start) + 2]
    shell = (
        exports
        + "\n"
        + "\n".join(
            [
                "set -euo pipefail",
                "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
                "NODE0_ADDR=192.0.2.1; IP_ARRAY=(192.0.2.1 192.0.2.2)",
                "PREFILL_PORT=2584; DECODE_PORT=3584; SERVED_MODEL_NAME=Kimi-K3",
                "RUN_DIR=/run; ATOMESH_EXECUTION_PHASE=benchmark",
                "PREFILL_TP_SIZE=$PREFILL_TP",
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
    argv = result.stdout.splitlines()
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
        "/run/pd-diagnostic/benchmark",
        "--phase",
        "benchmark",
        "--mode",
        "smoke",
        "--tp",
        "8",
        "--dcp",
        "8",
        "--hybrid",
    ]
    if name in PAIRS:
        expected.append("--cache-composition")
    assert argv == expected
