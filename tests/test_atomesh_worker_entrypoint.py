# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI P/D workers on the Mesh entrypoint: build, setup and launch, without GPUs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


class TempDirTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)


class MeshBuildTest(TempDirTest):
    def fake_cargo(self, bin_dir: Path) -> None:
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

    def build(self, root: Path, entrypoint: str):
        source = root / "source"
        source.mkdir(parents=True)
        (source / "Cargo.toml").write_text("[package]\n")
        # build.rs reads the engine protocol from ../model_engine/protocol.
        proto = root / "model_engine" / "protocol" / "engine_v1.proto"
        proto.parent.mkdir(parents=True)
        proto.write_text('syntax = "proto3";\n')
        bin_dir = root / "bin"
        bin_dir.mkdir()
        self.fake_cargo(bin_dir)
        logs = root / "logs"
        subprocess.run(
            ["bash", str(SCRIPTS / "build_mesh.sh"), str(source), str(logs)],
            env={
                **os.environ,
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "TMPDIR": str(root),
                "ATOMESH_MESH_TARGET_DIR": str(root / "cache"),
                "ATOMESH_MESH_SOURCE_COMMIT": "reviewed-commit",
                "ATOMESH_WORKER_ENTRYPOINT": entrypoint,
                "TEST_CARGO_ARGS": str(root / "cargo-args.json"),
            },
            capture_output=True,
            text=True,
            check=True,
        )
        args = json.loads((root / "cargo-args.json").read_text())
        metadata = json.loads((logs / "mesh-build.json").read_text())
        return logs, args, metadata

    def test_build_stages_the_pyo3_module_only_for_mesh_workers(self):
        for entrypoint in ("python", "mesh"):
            with self.subTest(entrypoint=entrypoint):
                logs, args, metadata = self.build(self.tmp / entrypoint, entrypoint)
                module = logs / "python" / "atomesh_runner.so"
                self.assertTrue((logs / "atomesh").is_file())
                if entrypoint == "mesh":
                    self.assertIn("--lib", args)
                    self.assertIn("--bin", args)
                    self.assertEqual(module.read_text(), "module")
                    self.assertEqual(len(metadata["python_module_sha256"]), 64)
                else:
                    self.assertNotIn("--lib", args)
                    self.assertFalse(module.exists())
                    self.assertNotIn("python_module_sha256", metadata)


class MeshSetupTest(TempDirTest):
    def test_setup_builds_for_mesh_workers_even_without_build_mesh(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e"]
            + ["commit", "--allow-empty", "-qm", "fixture"],
            check=True,
        )
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        calls = self.tmp / "docker-calls.jsonl"
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
        env_file = self.tmp / "docker.env"
        env_file.write_text("")
        run_dir = self.tmp / "run"
        command = ["bash", str(SCRIPTS / "setup_mesh.sh"), str(repo), str(run_dir)]
        command += ["test-image", str(env_file), "123"]
        env = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "ATOMESH_BUILD_MESH": "false",
            "ATOMESH_MESH_BINARY": "",
            "ATOMESH_MESH_TARGET_DIR": str(self.tmp / "cache"),
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
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()

        # Python workers keep the image's router binary and build nothing.
        self.assertEqual(setup("python"), "/app/ATOM/atom/mesh/target/release/atomesh")
        self.assertFalse(calls.exists())
        # A python-only artifact from an earlier phase lacks the module: rebuild once.
        artifact = run_dir / "mesh-build"
        artifact.mkdir(parents=True)
        (artifact / "atomesh").write_text("binary")
        (artifact / "atomesh").chmod(0o755)
        (artifact / "mesh-build.json").write_text("{}")
        for _ in range(2):
            self.assertEqual(
                setup("mesh"), "/run_logs/slurm_job-123/mesh-build/atomesh"
            )
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(recorded), 1)
        self.assertIn("ATOMESH_WORKER_ENTRYPOINT=mesh", recorded[0])
        self.assertTrue((artifact / "python" / "atomesh_runner.so").is_file())


class WorkerEntrypointTest(TempDirTest):
    def worker_entrypoint(self, entrypoint, binary):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        start = source.index("mesh_python_module_dir() {")
        end = source.index("\n}\n", source.index("build_worker_entrypoint() {")) + 3
        script = source[start:end] + """
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
                "RUN_DIR": str(self.tmp / "run"),
                "PYTHONPATH": "/existing",
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_python_workers_get_no_entrypoint_overrides(self):
        result = self.worker_entrypoint("python", self.tmp / "atomesh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ENV:\nARG:\n")

    def test_mesh_workers_import_the_built_module_and_move_their_metrics_port(self):
        artifact = self.tmp / "mesh-build"
        (artifact / "python").mkdir(parents=True)
        (artifact / "python" / "atomesh_runner.so").write_text("module")
        result = self.worker_entrypoint("mesh", artifact / "atomesh")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("ENV:USE_ATOMESH_ENTRYPOINTS=1", lines)
        self.assertIn(f"ENV:PYTHONPATH={artifact / 'python'}:/existing", lines)
        self.assertEqual(lines[-2:], ["ARG:--prometheus-port", "ARG:9010"])

    def test_mesh_workers_on_other_ranks_use_the_shared_run_directory(self):
        shared = self.tmp / "run" / "mesh-build" / "python"
        shared.mkdir(parents=True)
        (shared / "atomesh_runner.so").write_text("module")
        result = self.worker_entrypoint(
            "mesh", "/app/ATOM/atom/mesh/target/release/atomesh"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"ENV:PYTHONPATH={shared}:/existing", result.stdout.splitlines())

    def test_mesh_workers_fail_fast_without_the_module(self):
        result = self.worker_entrypoint("mesh", self.tmp / "atomesh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs atomesh_runner.so", result.stderr)


class WorkflowInputTest(unittest.TestCase):
    def test_workflow_dispatch_selects_the_worker_entrypoint_and_builds_for_mesh(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
        )
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["worker_entrypoint"]["default"], "python")
        self.assertEqual(inputs["worker_entrypoint"]["options"], ["python", "mesh"])
        step = next(
            step
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if step.get("name") == "Run Slurm benchmark cell"
        )
        self.assertIn(
            "inputs.worker_entrypoint", step["env"]["ATOMESH_WORKER_ENTRYPOINT"]
        )
        self.assertIn(
            "inputs.worker_entrypoint == 'mesh'", step["env"]["ATOMESH_BUILD_MESH"]
        )


if __name__ == "__main__":
    unittest.main()
