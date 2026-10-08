"""CPU-only bounded K3 cell, exported dispatch, and both-role argv contracts."""

import copy
import json
import shlex
import subprocess

import pytest
from test_atomesh_k3_composition_wiring import DRAFT, SCRIPTS
from test_atomesh_k3_composition_wiring import cells as k3_cells
from test_atomesh_native_nixl_wiring import exported, shell_render

CASE = "survey-k3-main-read-dspark3-bounded-perf-c1-1p1d-tp8-dcp8-eager"
cells = k3_cells


def test_bounded_cell_only_changes_parent_workload_and_client_mode(cells):
    cell = copy.deepcopy(cells[CASE])
    parent = copy.deepcopy(cells[DRAFT])
    assert cell["isl"] == [1025, 2050, 3073]
    assert cell["osl"] == 128
    assert cell["concurrency"] == [1]
    assert max(cell["isl"]) + cell["osl"] < 4096
    assert cell["vllm"]["diagnostic_mode"] == "bounded-perf"
    assert parent["vllm"]["diagnostic_mode"] == "smoke"
    cell["vllm"]["diagnostic_mode"] = parent["vllm"]["diagnostic_mode"]
    assert cell["concurrency_x"] == "1"
    assert cell["accuracy"]["concurrency"] == [1]
    assert cell["run_eval"] is False
    cell["accuracy"]["concurrency"] = parent["accuracy"]["concurrency"]
    for key in ("id", "name", "isl", "osl", "concurrency", "concurrency_x"):
        cell.pop(key)
        parent.pop(key)
    assert cell == parent
    assert cell["model_path"] == "/models/moonshotai/Kimi-K3"
    assert cell["precision"] == "checkpoint"
    assert cell["vllm"]["source"] == {
        "repo": "https://github.com/vllm-project/vllm",
        "sha": "b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
    }
    assert cell["vllm"]["clean_main"] == 1
    assert cell["env"]["common"]["VLLM_ROCM_USE_AITER_MOE_SITUV2"] == "a16w4"
    assert cell["vllm"]["draft_model_path"] == (
        "${ATOMESH_MODEL_ROOT}/Inferact/Kimi-K3-DSpark"
    )
    assert not cell["vllm"].get("cache_composition")


@pytest.mark.parametrize("role,port", [("prefill", 2584), ("decode", 3584)])
def test_actual_both_role_server_argv_is_identical_to_dspark_parent(
    cells, tmp_path, role, port
):
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
    parent, bounded = launches
    assert bounded == parent
    for option, expected in {
        "--tensor-parallel-size": "8",
        "--decode-context-parallel-size": "8",
        "--max-model-len": "4096",
        "--max-num-seqs": "4",
        "--max-num-batched-tokens": "2048",
        "--load-format": "safetensors",
        "--kv-cache-dtype": "auto",
        "--block-size": "128",
        "--prefix-match-unit": "128",
        "--attention-backend": "ROCM_AITER_MLA",
        "--dcp-comm-backend": "a2a",
        "--cp-kv-cache-interleave-size": "1",
    }.items():
        assert bounded.count(option) == 1
        assert bounded[bounded.index(option) + 1] == expected
    assert bounded.count("--enable-prefix-caching") == 1
    assert bounded.count("--enforce-eager") == 1
    assert bounded.count("--speculative-config") == 1
    assert json.loads(bounded[bounded.index("--speculative-config") + 1]) == {
        "model": "${ATOMESH_MODEL_ROOT}/Inferact/Kimi-K3-DSpark",
        "num_speculative_tokens": 3,
        "method": "dspark",
        "attention_backend": "ROCM_AITER_MLA",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }
    assert not any(
        arg.startswith(("--profiler-config", "--quantization", "--hf-overrides"))
        for arg in bounded
    )


@pytest.mark.parametrize("name,mode", [(DRAFT, "smoke"), (CASE, "bounded-perf")])
def test_exported_cell_dispatches_exact_client_mode_without_profiling(
    cells, name, mode
):
    cell = cells[name]
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
    assert result.stdout.splitlines() == [
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
        mode,
        "--tp",
        "8",
        "--dcp",
        "8",
        "--hybrid",
    ]
