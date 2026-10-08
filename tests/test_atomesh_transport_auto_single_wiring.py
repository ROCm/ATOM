"""CPU-only auto-single cell exports and image-only transport dispatch."""

import copy
import os
import shlex
import subprocess

import pytest
from test_atomesh_native_nixl_wiring import exported
from test_atomesh_transport_inventory_wiring import BASE, ROOT, SCRIPTS, load
from test_atomesh_transport_read_only_wiring import CASE as EXPLICIT
from test_atomesh_transport_read_only_wiring import INVENTORY

CASE = "survey-transport-nixl-auto-single-read-only-2node"
CASES = (BASE, INVENTORY, EXPLICIT, CASE)
AUTO_FLAGS = {
    "ATOMESH_TRANSPORT_GPU": "1",
    "ATOMESH_TRANSPORT_NIXL_READ_ONLY": "1",
    "ATOMESH_TRANSPORT_UCX_AUTO_SINGLE": "1",
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
    matrix = load("pd_matrix")
    return {
        cell["name"]: cell
        for cell in matrix.build_cells(
            matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
            suite="vllm",
            model_filter={"Transport-vLLM-Survey"},
            case_filter=set(CASES),
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
    }


def role_env(cell, role):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("apply_prefixed_env() {")
    function = source[start : source.index("\n}", start) + 2]
    result = subprocess.check_output(
        [
            "bash",
            "-c",
            exported(cell)
            + "\nset -euo pipefail\nHANDSHAKE_PORT=15559\n"
            + function
            + "\napply_prefixed_env ATOMESH_ENV_ 192.0.2.1\n"
            + f"apply_prefixed_env ATOMESH_{role.upper()}_ENV_ 192.0.2.1\n"
            + "env -0",
        ],
        env={"PATH": os.environ["PATH"]},
    )
    return dict(item.decode().split("=", 1) for item in result.split(b"\0") if item)


def test_auto_single_only_adds_exact_opt_in_to_image_only_parent(cells):
    assert set(cells) == set(CASES)
    derived, parent = copy.deepcopy(cells[CASE]), copy.deepcopy(cells[BASE])
    for role in ("common", "prefill", "decode", "router"):
        for key, value in AUTO_FLAGS.items():
            assert derived["env"][role][key] == value
        assert "ATOMESH_TRANSPORT_UCX_SELECTION" not in derived["env"][role]
        for key in AUTO_FLAGS.keys() - {"ATOMESH_TRANSPORT_GPU"}:
            derived["env"][role].pop(key)
    for cell in (derived, parent):
        for key in ("id", "name"):
            cell.pop(key)
    assert derived == parent
    assert derived["precision"] == "DIAGNOSTIC_ONLY"
    assert derived["model_path"] == "/models/Qwen/Qwen3-0.6B"
    assert derived["vllm"]["source"]["sha"] == (
        "b22494cc0cb4bd9db4a62fb107d92429a4a3249d"
    )
    assert derived["num_nodes"] == 2
    assert derived["nodes"] == ["node-a", "node-b"]
    assert derived["run_eval"] is False


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_actual_role_exports_keep_auto_explicit_default_inventory_distinct(
    cells, name, role
):
    env = role_env(cells[name], role)
    expected = {"ATOMESH_TRANSPORT_GPU": "0" if name == INVENTORY else "1"}
    if name == CASE:
        expected.update(AUTO_FLAGS)
    elif name == EXPLICIT:
        expected.update(
            ATOMESH_TRANSPORT_NIXL_READ_ONLY="1",
            ATOMESH_TRANSPORT_UCX_SELECTION="rdma0:1@1",
        )
    selection_keys = {*AUTO_FLAGS, "ATOMESH_TRANSPORT_UCX_SELECTION"}
    assert {key: env[key] for key in selection_keys if key in env} == expected
    assert env["ATOMESH_TRANSPORT_ONLY"] == "1"
    assert env["ATOMESH_TRANSPORT_MORI_BYTES"] == "4096,2493186048"
    assert env["ATOMESH_PRE_CLEAN_GPU"] == "1"
    assert env["VLLM_PLUGINS"] == ""
    assert not {"UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_TLS"} & env.keys()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("role,rank", [("prefill", 0), ("decode", 1)])
def test_actual_dispatch_executes_only_image_probe_on_both_roles(
    cells, tmp_path, name, role, rank
):
    env = role_env(cells[name], role)
    env.update(NODE_RANK=str(rank), IPADDRS="192.0.2.1,192.0.2.2")
    assert env["ATOMESH_VLLM_SOURCE_SHA"] == (
        "b22494cc0cb4bd9db4a62fb107d92429a4a3249d"
    )
    source = (SCRIPTS / "pd_server_vllm.sh").read_text()
    dispatch = source[
        source.index('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then') :
    ]
    python = tmp_path / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]
    shell = "\n".join(
        [
            "set -euo pipefail",
            "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
            'write_metadata() { printf "metadata\\n" >&2; }',
            "install_native_vllm() { exit 91; }",
            "apply_vllm_fork_overlay() { exit 92; }",
            "start_vllm_server() { exit 93; }",
            "start_vllm_router() { exit 94; }",
            dispatch,
            "exit 95",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", shell], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.splitlines() == ["metadata"]
    assert result.stdout.splitlines() == [str(SCRIPTS / "pd_transport_probe.py")]
