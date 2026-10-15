"""Exercise pre-service cleanup through a fake Docker/scheduler boundary."""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "cleanup", Path(__file__).resolve().parents[1] / "pd_gpu_cleanup.py"
)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.manifest = root / "targets.json"
        self.targets = [
            {"cell_id": "c48-v0", "slurm_job_id": "8068", "terminal_state": "FAILED"},
            {
                "cell_id": "c48-vq",
                "slurm_job_id": "8439",
                "terminal_state": "CANCELLED",
            },
        ]
        self.write_targets()
        self.args = argparse.Namespace(
            job_id="9000",
            run_token="a" * 32,
            node="node-a",
            rank=0,
            cell_id="c48-vq",
            out=root / "result.json",
            targets=self.manifest,
            required_free_ratio=0.95,
        )
        self.calls = []
        self.containers = {}
        self.active = {"9000"}

    def write_targets(self):
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "targets": self.targets})
        )

    def add_container(self, number, name, owner="100:200"):
        cid = f"{number:064x}"
        self.containers[cid] = ["/" + name, owner]
        return cid

    def command(self, args, timeout=10):
        self.calls.append(args)
        if args[0] == "squeue":
            return "\n".join(sorted(self.active))
        if args[:2] == ["docker", "ps"]:
            return "\n".join(self.containers)
        if args[:2] == ["docker", "inspect"]:
            return json.dumps(self.containers[args[-1]])
        if args[:2] == ["docker", "stop"]:
            return args[-1]
        if args[:2] == ["docker", "rm"]:
            del self.containers[args[-1]]
            return args[-1]
        raise AssertionError(args)

    def invoke(self):
        with (
            patch.object(cleanup, "command", side_effect=self.command),
            patch.object(cleanup.os, "getuid", return_value=100),
            patch.object(cleanup.os, "getgid", return_value=200),
        ):
            instance = cleanup.Cleanup(self.args)
            instance.stop_containers()
        return instance

    def test_removes_only_archived_owned_inactive_targets(self):
        stale = self.add_container(1, "atomesh-c48-v0-8068-1-benchmark")
        active = self.add_container(2, "atomesh-c48-vq-8439-0")
        unlisted = self.add_container(3, "atomesh-c48-vq-8999-0")
        other_user = self.add_container(4, "atomesh-c48-v0-8068-0", "101:200")
        unrelated = self.add_container(5, "atomesh-other-task-8068-0")
        self.active.add("8439")
        result = self.invoke()
        mutations = [
            c for c in self.calls if c[:2] in (["docker", "stop"], ["docker", "rm"])
        ]
        self.assertEqual(
            mutations,
            [["docker", "stop", "--time", "5", stale], ["docker", "rm", stale]],
        )
        self.assertEqual(
            set(self.containers), {active, unlisted, other_user, unrelated}
        )
        kinds = [e["kind"] for e in result.report["events"]]
        self.assertIn("skip_active_container", kinds)
        self.assertIn("skip_different_owner", kinds)

    def test_rechecks_queue_after_inspection_before_stopping(self):
        self.add_container(1, "atomesh-c48-v0-8068-0")
        original = self.command

        def requeued(args, timeout=10):
            result = original(args, timeout)
            if args[:2] == ["docker", "inspect"]:
                self.active.add("8068")
            return result

        self.command = requeued
        self.invoke()
        self.assertFalse(any(c[:2] == ["docker", "stop"] for c in self.calls))

    def test_missing_current_allocation_never_stops(self):
        self.active.clear()
        self.add_container(1, "atomesh-c48-v0-8068-0")
        with self.assertRaisesRegex(RuntimeError, "current allocation"):
            self.invoke()
        self.assertFalse(any(c[:2] == ["docker", "stop"] for c in self.calls))

    def test_current_and_nonterminal_manifest_targets_are_rejected(self):
        for change in ({"slurm_job_id": "9000"}, {"terminal_state": "RUNNING"}):
            with self.subTest(change=change):
                self.targets[0] = {
                    "cell_id": "c48-v0",
                    "slurm_job_id": "8068",
                    "terminal_state": "FAILED",
                    **change,
                }
                self.write_targets()
                with self.assertRaisesRegex(ValueError, "terminal jobs"):
                    cleanup.Cleanup(self.args)

    def test_run_records_memory_before_and_after_cleanup(self):
        self.add_container(1, "atomesh-c48-v0-8068-0")

        def memory():
            used = 92 if self.containers else 1
            return [
                {"pci": f"pci-{i}", "total": 100, "used": used, "free": 100 - used}
                for i in range(8)
            ]

        with (
            patch.object(cleanup, "command", side_effect=self.command),
            patch.object(cleanup.os, "getuid", return_value=100),
            patch.object(cleanup.os, "getgid", return_value=200),
            patch.object(cleanup.socket, "gethostname", return_value="node-a"),
            patch.dict(
                cleanup.os.environ,
                {
                    "SLURM_JOB_ID": "9000",
                    "SPUR_JOB_ID": "9000",
                    "ATOMESH_EXCLUSIVE_GPU_CLEANUP": "1",
                },
            ),
            patch.object(cleanup, "memory", side_effect=memory),
            patch.object(
                cleanup,
                "gpu_processes",
                side_effect=lambda: [123] if self.containers else [],
            ),
        ):
            instance = cleanup.Cleanup(self.args)
            instance.run()
        report = json.loads(self.args.out.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["events"][0]["memory"][0]["free"], 8)
        check = next(e for e in report["events"] if e["kind"] == "check")
        self.assertEqual(check["memory"][0]["free"], 99)
        self.assertEqual(check["processes"], [])

    def test_unknown_process_occupation_fails_without_signalling_it(self):
        unknown = self.add_container(1, "atomesh-other-task-8999-0")
        with (
            patch.object(cleanup, "command", side_effect=self.command),
            patch.object(cleanup.os, "getuid", return_value=100),
            patch.object(cleanup.os, "getgid", return_value=200),
            patch.object(cleanup.socket, "gethostname", return_value="node-a"),
            patch.dict(
                cleanup.os.environ,
                {
                    "SLURM_JOB_ID": "9000",
                    "SPUR_JOB_ID": "9000",
                    "ATOMESH_EXCLUSIVE_GPU_CLEANUP": "1",
                },
            ),
            patch.object(
                cleanup, "memory", return_value=[{"total": 100, "free": 5}] * 8
            ),
            patch.object(cleanup, "gpu_processes", return_value=[456]),
            patch.object(cleanup.time, "monotonic", side_effect=itertools.count(0, 61)),
            patch.object(cleanup.time, "sleep"),
        ):
            instance = cleanup.Cleanup(self.args)
            with self.assertRaisesRegex(RuntimeError, "insufficient free VRAM"):
                instance.run()
        self.assertFalse(instance.report["passed"])
        self.assertIn(unknown, self.containers)
        self.assertFalse(any(c[:2] == ["docker", "stop"] for c in self.calls))


if __name__ == "__main__":
    unittest.main()
