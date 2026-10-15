# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load("benchmark_guard")
watch = load("minimax_runtime_watch")


class BenchmarkGuardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.server = self.root / "prefill.log"
        self.client = self.root / "client.log"
        self.server.touch()
        self.client.touch()
        self.guard = guard.Guard([self.server], self.client, 900)
        self.guard.last_progress = 0

    def append(self, path, text):
        with path.open("a") as stream:
            stream.write(text)

    def test_metrics_and_warmup_heartbeat_do_not_hide_stalled_inference(self):
        line = "Phase warmup progress | returned=27/354 | sent=34 | in_flight=7\n"
        self.append(self.client, line)
        self.assertIsNone(self.guard.check(now=0))
        self.append(self.server, "GET /metrics HTTP/1.1 200 OK\n")
        self.append(self.client, line)
        failure = self.guard.check(now=901)
        self.assertEqual(failure["reason"], "inference_stalled")
        self.assertEqual(failure["returned"], 27)

    def test_real_progress_resets_inactivity_and_idle_is_allowed(self):
        self.append(self.client, "returned=0/354 | sent=1 | in_flight=1\n")
        self.guard.check(now=0)
        self.append(self.server, "Scheduled prefill batch: 1 reqs\n")
        self.assertIsNone(self.guard.check(now=899))
        self.assertIsNone(self.guard.check(now=1000))
        self.append(self.client, "returned=1/354 | sent=1 | in_flight=0\n")
        self.assertIsNone(self.guard.check(now=2000))
        self.assertIsNone(self.guard.check(now=4000))

    def test_fatal_error_across_read_boundary_is_detected(self):
        self.append(self.server, "[rank1] Watchdog caught collective")
        self.assertIsNone(self.guard.check(now=1))
        self.append(self.server, " operation timeout: SeqNum=95\n")
        self.assertEqual(self.guard.check(now=2)["reason"], "fatal_worker_error")

    def test_transient_lmcache_lookup_error_is_not_fatal(self):
        self.append(
            self.server, "LMCache ERROR Timeout occurred, recreating all sockets\n"
        )
        self.assertIsNone(self.guard.check(now=1))

    def test_fatal_worker_stops_live_command_and_returns_failure(self):
        self.server.write_text("Watchdog caught collective operation timeout\n")
        output = self.root / "failure.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "benchmark_guard.py"),
                "--server-log",
                str(self.server),
                "--client-log",
                str(self.client),
                "--output",
                str(output),
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
            ],
            timeout=10,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 86, result.stderr)
        self.assertEqual(json.loads(output.read_text())["reason"], "fatal_worker_error")

    def test_success_preserves_exit_code(self):
        self.assertEqual(
            guard.run(
                [sys.executable, "-c", "pass"], self.guard, self.root / "failure.json"
            ),
            0,
        )


class RuntimeWatchTests(unittest.TestCase):
    def test_only_registered_matching_processes_are_sampled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "prefill").mkdir()
            row = {"pid": os.getpid(), "start_ticks": watch.process_start(os.getpid())}
            (root / "prefill/pid-good.json").write_text(json.dumps(row))
            row["start_ticks"] = "invalid"
            (root / "prefill/pid-reused.json").write_text(json.dumps(row))
            result = watch.sample(root)
            self.assertEqual(len(result["processes"]), 1)
            self.assertIn("VmRSS:", result["processes"][0]["status"])
            self.assertTrue(result["processes"][0]["threads"])
            self.assertTrue(result["vmstat"])

    def test_python_stack_snapshot_with_changing_threads(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stacks.log"
            path.touch()
            done = threading.Event()

            def churn():
                while not done.is_set():
                    child = threading.Thread(target=lambda: sum(range(100)))
                    child.start()
                    child.join()

            thread = threading.Thread(target=churn)
            thread.start()
            try:
                for _ in range(10):
                    watch.dump_stacks(path)
            finally:
                done.set()
                thread.join(timeout=5)
            text = path.read_text()
            self.assertEqual(text.count("sampler=python_thread"), 10)
            self.assertIn("Thread ", text)
            self.assertIn("dump_stacks", text)

    def test_runtime_copy_preserves_source_and_starts_python_sampler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, runtime, output = root / "source", root / "runtime", root / "out"
            (source / "atom").mkdir(parents=True)
            original = "VALUE = 42\n"
            (source / "atom/__init__.py").write_text(original)
            watch.prepare(source, runtime, output)
            env = {
                **os.environ,
                "ATOM_CI_RUNTIME_DIAGNOSTICS": str(output),
                "ATOM_CI_RUNTIME_ROLE": "prefill",
            }
            subprocess.run(
                [sys.executable, "-c", "import atom; assert atom.VALUE == 42"],
                cwd=runtime,
                env=env,
                check=True,
                timeout=10,
            )
            self.assertEqual((source / "atom/__init__.py").read_text(), original)
            self.assertEqual(len(list((output / "prefill").glob("pid-*.json"))), 1)
            self.assertEqual(len(list((output / "prefill").glob("*.stacks.log"))), 1)


if __name__ == "__main__":
    unittest.main()
