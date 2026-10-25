# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI P/D workers on the Mesh entrypoint: build, setup and launch, without GPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def fake_cargo(bin_dir: Path) -> None:
    cargo = bin_dir / "cargo"
    cargo.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if args == ['--version']: print('cargo test-fixture'); sys.exit(0)\n"
        "Path(os.environ['TEST_CARGO_ARGS']).write_text(json.dumps(args))\n"
        "source=Path(args[args.index('--manifest-path')+1]).parent\n"
        "(source/'Cargo.lock').write_text('resolved')\n"
        "out=Path(args[args.index('--target-dir')+1])/args[args.index('--profile')+1]\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'atomesh').write_text('binary')\n"
        "(out/'atomesh').chmod(0o755)\n"
        "if '--lib' in args: (out/'libmesh.so').write_text('module')\n"
    )
    cargo.chmod(0o755)


@pytest.mark.parametrize("entrypoint", ["python", "mesh"])
def test_build_stages_the_pyo3_module_only_for_mesh_workers(tmp_path, entrypoint):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Cargo.toml").write_text("[package]\n")
    proto = tmp_path / "model_engine" / "protocol" / "engine_v1.proto"
    proto.parent.mkdir(parents=True)
    proto.write_text('syntax = "proto3";\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_cargo(bin_dir)
    logs = tmp_path / "logs"
    subprocess.run(
        ["bash", str(SCRIPTS / "build_mesh.sh"), str(source), str(logs)],
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "TMPDIR": str(tmp_path),
            "ATOMESH_MESH_TARGET_DIR": str(tmp_path / "cache"),
            "ATOMESH_MESH_SOURCE_COMMIT": "reviewed-commit",
            "ATOMESH_WORKER_ENTRYPOINT": entrypoint,
            "TEST_CARGO_ARGS": str(tmp_path / "cargo-args.json"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    args = json.loads((tmp_path / "cargo-args.json").read_text())
    module = logs / "python" / "atomesh_runner.so"
    metadata = json.loads((logs / "mesh-build.json").read_text())
    assert (logs / "atomesh").is_file()
    if entrypoint == "mesh":
        assert "--lib" in args and "--bin" in args
        assert module.read_text() == "module"
        assert len(metadata["python_module_sha256"]) == 64
    else:
        assert "--lib" not in args
        assert not module.exists()
        assert "python_module_sha256" not in metadata


def test_setup_builds_for_mesh_workers_even_without_build_mesh(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e"]
        + ["commit", "--allow-empty", "-qm", "fixture"],
        check=True,
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "docker-calls.jsonl"
    docker = bin_dir / "docker"
    docker.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['TEST_DOCKER_CALLS'], 'a') as f: f.write(json.dumps(args)+'\\n')\n"
        "mounts=[args[i+1] for i,a in enumerate(args) if a=='-v']\n"
        "artifact=Path(next(m for m in mounts if m.endswith('/mesh-build')).split(':')[0])\n"
        "(artifact/'atomesh').write_text('binary')\n"
        "(artifact/'atomesh').chmod(0o755)\n"
        "(artifact/'mesh-build.json').write_text('{}')\n"
        "if 'ATOMESH_WORKER_ENTRYPOINT=mesh' in args:\n"
        "    (artifact/'python').mkdir(exist_ok=True)\n"
        "    (artifact/'python'/'atomesh_runner.so').write_text('module')\n"
    )
    docker.chmod(0o755)
    env_file = tmp_path / "docker.env"
    env_file.write_text("")
    run_dir = tmp_path / "run"
    command = ["bash", str(SCRIPTS / "setup_mesh.sh"), str(repo), str(run_dir)]
    command += ["test-image", str(env_file), "123"]
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "ATOMESH_BUILD_MESH": "false",
        "ATOMESH_MESH_BINARY": "",
        "ATOMESH_MESH_TARGET_DIR": str(tmp_path / "cache"),
        "TEST_DOCKER_CALLS": str(calls),
    }

    def setup(entrypoint):
        result = subprocess.run(
            command,
            env={**env, "ATOMESH_WORKER_ENTRYPOINT": entrypoint},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    # Python workers keep the image's router binary and build nothing.
    assert setup("python") == "/app/ATOM/atom/mesh/target/release/atomesh"
    assert not calls.exists()
    # A python-only artifact from an earlier phase lacks the module: rebuild once.
    (run_dir / "mesh-build").mkdir(parents=True)
    (run_dir / "mesh-build" / "atomesh").write_text("binary")
    (run_dir / "mesh-build" / "atomesh").chmod(0o755)
    (run_dir / "mesh-build" / "mesh-build.json").write_text("{}")
    for _ in range(2):
        assert setup("mesh") == "/run_logs/slurm_job-123/mesh-build/atomesh"
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == 1
    assert "ATOMESH_WORKER_ENTRYPOINT=mesh" in recorded[0]
    assert (run_dir / "mesh-build" / "python" / "atomesh_runner.so").is_file()


def worker_entrypoint(tmp_path, entrypoint, binary, extra=""):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("mesh_python_module_dir() {")
    end = source.index("\n}\n", source.index("build_worker_entrypoint() {")) + 3
    script = source[start:end] + extra + """
env_out=() args_out=()
build_worker_entrypoint 8010 env_out args_out
printf 'ENV:%s\\n' "${env_out[@]}"
printf 'ARG:%s\\n' "${args_out[@]}"
"""
    return subprocess.run(
        ["bash", "-eu", "-c", script],
        env={
            **os.environ,
            "ATOMESH_WORKER_ENTRYPOINT": entrypoint,
            "ATOMESH_MESH_BINARY": str(binary),
            "ATOMESH_MESH_MODULE_WAIT_SECONDS": "0",
            "RUN_DIR": str(tmp_path / "run"),
            "PYTHONPATH": "/existing",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_python_workers_get_no_entrypoint_overrides(tmp_path):
    result = worker_entrypoint(tmp_path, "python", tmp_path / "atomesh")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ENV:\nARG:\n"


def test_mesh_workers_import_the_built_module_and_move_their_metrics_port(tmp_path):
    artifact = tmp_path / "mesh-build"
    (artifact / "python").mkdir(parents=True)
    (artifact / "python" / "atomesh_runner.so").write_text("module")
    result = worker_entrypoint(tmp_path, "mesh", artifact / "atomesh")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "ENV:USE_ATOMESH_ENTRYPOINTS=1" in lines
    assert f"ENV:PYTHONPATH={artifact / 'python'}:/existing" in lines
    assert lines[-2:] == ["ARG:--prometheus-port", "ARG:9010"]


def test_mesh_workers_on_other_ranks_use_the_shared_run_directory(tmp_path):
    shared = tmp_path / "run" / "mesh-build" / "python"
    shared.mkdir(parents=True)
    (shared / "atomesh_runner.so").write_text("module")
    result = worker_entrypoint(
        tmp_path, "mesh", "/app/ATOM/atom/mesh/target/release/atomesh"
    )
    assert result.returncode == 0, result.stderr
    assert f"ENV:PYTHONPATH={shared}:/existing" in result.stdout.splitlines()


def test_mesh_workers_fail_fast_without_the_module(tmp_path):
    result = worker_entrypoint(tmp_path, "mesh", tmp_path / "atomesh")
    assert result.returncode != 0
    assert "needs atomesh_runner.so" in result.stderr


def test_workflow_dispatch_selects_the_worker_entrypoint_and_builds_for_mesh():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
    )
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["worker_entrypoint"]["default"] == "python"
    assert inputs["worker_entrypoint"]["options"] == ["python", "mesh"]
    step = next(
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Run Slurm benchmark cell"
    )
    assert "inputs.worker_entrypoint" in step["env"]["ATOMESH_WORKER_ENTRYPOINT"]
    assert "inputs.worker_entrypoint == 'mesh'" in step["env"]["ATOMESH_BUILD_MESH"]
