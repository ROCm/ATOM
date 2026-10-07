"""CPU-only inventory-cell export and model-free transport dispatch contracts."""

import argparse
import builtins
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
from test_atomesh_native_nixl_wiring import exported

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
BASE = "survey-transport-image-only-2node"
CASE = "survey-transport-inventory-only-2node"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            case_filter=None,
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
    }


def runtime_env(cell):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("apply_prefixed_env() {")
    function = source[start : source.index("\n}", start) + 2]
    result = subprocess.check_output(
        [
            "bash",
            "-c",
            exported(cell)
            + "\n"
            + function
            + "\napply_prefixed_env ATOMESH_ENV_ 192.0.2.1\nenv -0",
        ],
        env={"PATH": os.environ["PATH"]},
    )
    return dict(item.decode().split("=", 1) for item in result.split(b"\0") if item)


def test_inventory_cell_only_overrides_gpu_opt_in(cells):
    original = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )["models"]["Transport-vLLM-Survey"]
    original["suites"]["vllm"] = [
        cell for cell in original["suites"]["vllm"] if cell["name"] == BASE
    ]
    assert hashlib.sha256(
        json.dumps(original, sort_keys=True).encode()
    ).hexdigest() == (
        "2979de658bb6fd273924a5060b27622de390f4d27e72be6b63c2a859b772a833"
    )
    assert set(cells) == {BASE, CASE}
    derived, base = copy.deepcopy(cells[CASE]), copy.deepcopy(cells[BASE])
    for role in ("common", "prefill", "decode", "router"):
        assert derived["env"][role]["ATOMESH_TRANSPORT_GPU"] == "0"
        assert base["env"][role]["ATOMESH_TRANSPORT_GPU"] == "1"
        derived["env"][role]["ATOMESH_TRANSPORT_GPU"] = "1"
    for key in ("id", "name"):
        derived.pop(key)
        base.pop(key)
    assert derived == base
    env = runtime_env(cells[CASE])
    assert env["ATOMESH_TRANSPORT_ONLY"] == "1"
    assert env["ATOMESH_TRANSPORT_GPU"] == "0"
    assert not any(
        key in env
        for key in (
            "ATOMESH_TRANSPORT_UCX_SELECTION",
            "UCX_NET_DEVICES",
            "UCX_IB_GID_INDEX",
            "UCX_TLS",
        )
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_exported_gpu_zero_collects_inventory_and_only_original_create(
    cells, tmp_path, monkeypatch, rank
):
    probe = load("pd_transport_probe")
    env = runtime_env(cells[CASE])
    env.update(
        NODE_RANK=str(rank),
        IPADDRS="192.0.2.1,192.0.2.2",
        RUN_DIR=str(tmp_path),
        SLURM_JOB_ID="cpu-fixture",
    )
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", env)
        patch.setattr(sys, "argv", [str(SCRIPTS / "pd_transport_probe.py")])
        patch.setattr(probe.signal, "signal", lambda *_: None)
        patch.setattr(probe.signal, "alarm", lambda *_: None)
        port = tmp_path / "sys/class/infiniband/rdma0/ports/1"
        for name, value in {
            "state": "4: ACTIVE",
            "link_layer": "Ethernet",
            "gids/3": "::ffff:192.0.2.1",
            "gid_attrs/types/3": "RoCE v2",
            "gid_attrs/ndevs/3": "eth0",
        }.items():
            target = port / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
        original_glob = Path.glob

        def fake_glob(path, pattern):
            if str(path) == "/sys/class/infiniband":
                path = tmp_path / "sys/class/infiniband"
            return original_glob(path, pattern)

        patch.setattr(Path, "glob", fake_glob)
        commands = []

        def run_tool(argv, **kwargs):
            commands.append(argv)
            return subprocess.CompletedProcess(argv, 0, "readonly fixture", "")

        patch.setattr(probe.subprocess, "run", run_tool)
        patch.setattr(
            probe,
            "wait_json",
            lambda *_: {
                "gpu": "1",
                "eligible": True,
                "hostname": "different-peer",
                "selection": "",
            },
        )
        stages = []

        def supervise(argv, log, timeout, child_env):
            stages.append(argv[argv.index("--child") + 1])
            assert stages == ["nixl-create"]
            assert child_env["ATOMESH_TRANSPORT_GPU"] == "0"
            probe.write_json(
                argv[argv.index("--result") + 1],
                {"status": "PASS", "meaning": "BACKEND_CREATE_ONLY"},
            )
            return 0, False

        patch.setattr(probe, "supervise", supervise)
        assert probe.main() == 0
    output = tmp_path / f"transport-diagnostic/benchmark/rank-{rank}"
    inventory = json.loads((output / "inventory.json").read_text())
    assert inventory["model_loaded"] is False
    assert inventory["ports"][0]["gids"] == [
        {"index": "3", "gid": "::ffff:192.0.2.1", "type": "RoCE v2", "ndev": "eth0"}
    ]
    assert commands == [
        ["ip", "-j", "address", "show"],
        ["rdma", "link", "show"],
        ["ibv_devinfo"],
        ["ucx_info", "-v"],
        ["ucx_info", "-d"],
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["collection"] == "COMPLETED"
    assert summary["model_pd"] == "NOT_TESTED"
    assert set(summary["stages"]) == {
        "nixl-original-create",
        "nixl-rdma-gpu-read",
        "mori-register",
    }
    for stage in ("nixl-rdma-gpu-read", "mori-register"):
        assert summary["stages"][stage]["classification"] == "NOT_TESTED"
    launch = json.loads((output / "nixl-original-create.launch.json").read_text())
    assert launch["process_local_overrides"] == {}
    assert launch["diagnostic_not_original"] is False


def test_original_backend_create_returns_before_gpu_import(tmp_path, monkeypatch):
    probe = load("pd_transport_probe")
    calls = []

    class FakeAgent:
        backends = ("UCX",)

        def __init__(self, name, config):
            calls.append((name, config))

        def get_plugin_list(self):
            return ["UCX"]

        def get_backend_params(self, backend):
            assert backend == "UCX"
            return {}

    monkeypatch.setenv("SLURM_JOB_ID", "cpu-fixture")
    monkeypatch.setattr(probe, "load_nixl", lambda: (FakeAgent, lambda **kw: kw))
    real_import = builtins.__import__

    def reject_gpu(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "mori", "vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gpu)
    result = probe.nixl_probe(
        argparse.Namespace(child="nixl-create", result=tmp_path / "create.json", rank=0)
    )
    assert result["meaning"] == "BACKEND_CREATE_ONLY"
    assert calls == [("survey-cpu-fixture-0", {"backends": ["UCX"]})]


def test_inventory_launcher_executes_probe_before_model_install(cells, tmp_path):
    env = runtime_env(cells[CASE])
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
            "write_metadata() { :; }",
            "install_native_vllm() { exit 91; }",
            "apply_vllm_fork_overlay() { exit 92; }",
            dispatch,
            "exit 93",
        ]
    )
    result = subprocess.run(
        ["bash", "-c", shell], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(SCRIPTS / "pd_transport_probe.py")]
