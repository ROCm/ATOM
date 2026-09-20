# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the batch/worker entry points without a scheduler or Docker."""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

JOB_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_slurm_job.sh"
)


class SpurDispatchTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        scripts = self.root / "repo with spaces/.github/scripts/atomesh"
        scripts.mkdir(parents=True)
        self.script = scripts / JOB_SCRIPT.name
        shutil.copyfile(JOB_SCRIPT, self.script)
        shutil.copyfile(
            JOB_SCRIPT.with_name("pd_job_result.py"), scripts / "pd_job_result.py"
        )
        (scripts / "setup_mesh.sh").write_text("echo /fake/atomesh\n")
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.write_tool(
            "ip",
            """
            import os

            rank = int(os.environ["SPUR_TASK_OFFSET"])
            interface = "eno0" if rank == 0 else "enp5s0"
            print("1: lo inet 127.0.0.1/8 scope host lo")
            print(f"2: eno1 inet 10.13.0.{rank + 1}/21 scope global eno1")
            if not os.environ.get("MISSING_NODE_INTERFACE"):
                print(f"3: {interface} inet 10.19.0.{rank + 1}/21 scope global {interface}")
            """,
        )
        self.write_tool(
            "srun",
            """
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path

            root = Path(os.environ["TEST_ROOT"])
            # A recursive worker dispatch must fail, rather than fork forever.
            with (root / "dispatch.json").open("x") as stream:
                json.dump(sys.argv[1:], stream)
            if os.environ.get("DISPATCH_RC"):
                sys.exit(int(os.environ["DISPATCH_RC"]))
            args = sys.argv[1:]
            nodes = int(next(a.split("=")[1] for a in args if a.startswith("--nodes=")))
            command = args[args.index("bash"):]
            workers = []
            for rank in range(nodes):
                env = dict(os.environ, SPUR_TASK_OFFSET=str(rank),
                           SLURM_PROCID=str(rank), SLURM_STEP_ID="0")
                workers.append(subprocess.Popen(command, env=env))
            codes = [worker.wait() for worker in workers]
            sys.exit(next((code for code in codes if code), 0))
            """,
        )
        self.write_tool(
            "docker",
            """
            import json
            import os
            import sys
            from pathlib import Path

            rank = os.environ["SPUR_TASK_OFFSET"]
            with (Path(os.environ["TEST_ROOT"]) / ("docker-" + rank + ".jsonl")).open("a") as stream:
                stream.write(json.dumps(sys.argv[1:]) + "\\n")
            if sys.argv[1] == "run" and rank == os.environ.get("FAIL_RANK"):
                sys.exit(7)
            phase = os.environ.get("FAIL_PHASE")
            if sys.argv[1] == "run" and phase and f"ATOMESH_EXECUTION_PHASE={phase}" in sys.argv:
                sys.exit(8)
            """,
        )
        self.env = {
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "TEST_ROOT": str(self.root),
            "GITHUB_WORKSPACE": str(scripts.parents[2]),
            "LOG_ROOT": str(self.root / "logs"),
            "SLURM_JOB_ID": "42",
            "SPUR_JOB_ID": "42",
            "ATOMESH_RUN_TOKEN": "test-submission",
            "SPUR_TASK_OFFSET": "0",
            "SPUR_PEER_NODES": "10.19.0.1:6818,10.19.0.2:6818",
            "SPUR_NODELIST": "prefill-node,decode-node",
            "NUM_NODES": "2",
            "ATOMESH_CELL_ID": "test-cell",
            "MODEL_NAME": "test-model",
            "BACKEND": "atom",
            "TOPOLOGY": "1p1d",
            "DISPLAY_TOPOLOGY": "1P1D-TP8",
            "DOCKER_IMAGE": "test-image",
            "PREFILL_WORKERS": "1",
            "DECODE_WORKERS": "1",
            "PREFILL_TP": "8",
            "DECODE_TP": "8",
        }
        self.run_dir = self.root / "logs/slurm_job-42"

    def write_tool(self, name, body):
        tool = self.bin_dir / name
        tool.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
        tool.chmod(0o755)

    def run_job(self, *args, expected_rc=0, **env):
        result = subprocess.run(
            ["bash", str(self.script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, expected_rc, result.stdout + result.stderr)
        return result

    def docker_calls(self, rank):
        return [
            json.loads(line)
            for line in (self.root / f"docker-{rank}.jsonl").read_text().splitlines()
        ]

    def test_batch_dispatches_both_workers_and_preserves_topology(self):
        self.run_job()
        dispatch = json.loads((self.root / "dispatch.json").read_text())
        for option in ("--nodes=2", "--ntasks=2", "--ntasks-per-node=1"):
            self.assertIn(option, dispatch)
        for rank in range(2):
            calls = self.docker_calls(rank)
            runs = [call for call in calls if call[0] == "run"]
            self.assertEqual(len(runs), 1)
            self.assertIn(f"NODE_RANK={rank}", runs[0])
            self.assertIn("NODE0_ADDR=10.19.0.1", runs[0])
            self.assertIn("IPADDRS=10.19.0.1,10.19.0.2", runs[0])
            interface = "eno0" if rank == 0 else "enp5s0"
            self.assertIn(f"NCCL_SOCKET_IFNAME=={interface}", runs[0])
            self.assertIn(f"MORI_SOCKET_IFNAME={interface}", runs[0])
            self.assertIn(["rm", "-f", f"atomesh-test-cell-42-{rank}"], calls)
            self.assertEqual((self.run_dir / f"rank-rc-{rank}").read_text(), "0\n")
            result = json.loads(
                (self.run_dir / f"rank-workload-{rank}.json").read_text()
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["run_token"], "test-submission")

    def test_worker_does_not_dispatch_again(self):
        self.run_job("--spur-worker", SPUR_TASK_OFFSET="1")
        self.assertFalse((self.root / "dispatch.json").exists())
        self.assertFalse((self.root / "docker-0.jsonl").exists())
        self.assertEqual((self.run_dir / "rank-rc-1").read_text(), "0\n")

    def test_decode_failure_reaches_batch_exit_and_rank_status(self):
        self.run_job(expected_rc=7, FAIL_RANK="1")
        self.assertEqual((self.run_dir / "rank-rc-1").read_text(), "7\n")
        result = json.loads((self.run_dir / "rank-workload-1.json").read_text())
        self.assertEqual(result["status"], "running")
        self.assertIn(["rm", "-f", "atomesh-test-cell-42-1"], self.docker_calls(1))

    def test_dispatch_failure_is_not_reported_as_success(self):
        self.run_job(expected_rc=9, DISPATCH_RC="9")
        self.assertFalse((self.root / "docker-0.jsonl").exists())

    def test_invalid_worker_context_fails_before_docker(self):
        for env in (
            {"SPUR_TASK_OFFSET": ""},
            {"SPUR_TASK_OFFSET": "2"},
            {"SPUR_TASK_OFFSET": "invalid"},
            {"SPUR_PEER_NODES": ""},
        ):
            with self.subTest(env=env):
                result = self.run_job("--spur-worker", expected_rc=2, **env)
                self.assertIn("ERROR:", result.stderr)
        self.assertFalse(list(self.root.glob("docker-*.jsonl")))

    def test_single_node_allocation(self):
        self.run_job(
            NUM_NODES="1", SPUR_NODELIST="node0", SPUR_PEER_NODES="10.19.0.1:6818"
        )
        self.assertEqual((self.run_dir / "rank-rc-0").read_text(), "0\n")
        self.assertFalse((self.root / "docker-1.jsonl").exists())

    def test_explicit_nccl_interface_override_is_preserved(self):
        self.run_job(
            "--spur-worker",
            NCCL_SOCKET_IFNAME="=custom0,custom1",
        )
        run = next(call for call in self.docker_calls(0) if call[0] == "run")
        self.assertIn("NCCL_SOCKET_IFNAME==custom0,custom1", run)
        self.assertIn("MORI_SOCKET_IFNAME=eno0", run)

    def test_explicit_mori_interface_override_is_preserved(self):
        self.run_job("--spur-worker", MORI_SOCKET_IFNAME="custom0")
        run = next(call for call in self.docker_calls(0) if call[0] == "run")
        self.assertIn("NCCL_SOCKET_IFNAME==eno0", run)
        self.assertIn("MORI_SOCKET_IFNAME=custom0", run)

    def test_explicit_interfaces_skip_address_lookup(self):
        self.run_job(
            "--spur-worker",
            NCCL_SOCKET_IFNAME="=custom0",
            MORI_SOCKET_IFNAME="custom1",
            MISSING_NODE_INTERFACE="1",
        )
        run = next(call for call in self.docker_calls(0) if call[0] == "run")
        self.assertIn("NCCL_SOCKET_IFNAME==custom0", run)
        self.assertIn("MORI_SOCKET_IFNAME=custom1", run)

    def test_missing_node_interface_fails_before_container_start(self):
        result = self.run_job(
            "--spur-worker", expected_rc=2, MISSING_NODE_INTERFACE="1"
        )
        self.assertIn("cannot find a local IPv4 interface", result.stderr)
        self.assertFalse(any(call[0] == "run" for call in self.docker_calls(0)))
        self.assertEqual((self.run_dir / "rank-rc-0").read_text(), "2\n")

    def test_benchmark_and_eval_phases_run_on_each_worker(self):
        self.run_job(
            BENCHMARK_KIND="aiperf_agentic", RUN_EVAL="true", EVAL_TASK="gsm8k"
        )
        for rank in range(2):
            runs = [call for call in self.docker_calls(rank) if call[0] == "run"]
            self.assertEqual(len(runs), 2)
            self.assertIn("ATOMESH_EXECUTION_PHASE=benchmark", runs[0])
            self.assertIn("ATOMESH_SERVICE_PORT_OFFSET=0", runs[0])
            self.assertIn("ATOMESH_EXECUTION_PHASE=eval", runs[1])
            self.assertIn("ATOMESH_SERVICE_PORT_OFFSET=1000", runs[1])
            interface = "eno0" if rank == 0 else "enp5s0"
            for run in runs:
                self.assertIn(f"MORI_SOCKET_IFNAME={interface}", run)

    def test_eval_failure_after_benchmark_does_not_publish_completion(self):
        self.run_job(
            BENCHMARK_KIND="aiperf_agentic",
            RUN_EVAL="true",
            EVAL_TASK="gsm8k",
            FAIL_PHASE="eval",
            expected_rc=8,
        )
        for rank in range(2):
            runs = [call for call in self.docker_calls(rank) if call[0] == "run"]
            self.assertEqual(len(runs), 2)
            result = json.loads(
                (self.run_dir / f"rank-workload-{rank}.json").read_text()
            )
            self.assertEqual(result["status"], "running")
            self.assertEqual((self.run_dir / f"rank-rc-{rank}").read_text(), "8\n")


if __name__ == "__main__":
    unittest.main()
