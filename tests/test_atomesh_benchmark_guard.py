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
from unittest.mock import Mock, patch

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

    def test_completed_phase_does_not_kill_slow_result_export(self):
        self.append(
            self.client,
            "Phase warmup progress | returned=335/354 | sent=345 | in_flight=10\n",
        )
        self.guard.check(now=0)
        self.append(
            self.client,
            "Phase warmup (warmup) complete | completed=354, cancelled=0, errors=0\n",
        )
        self.append(
            self.client,
            "Phase profiling (profiling) started | target: 3600.0s duration\n",
        )
        self.guard.check(now=10)
        self.append(
            self.client,
            "Phase profiling (profiling) sending complete | sent=3,842, completed=3,825, in_flight=17\n",
        )
        self.guard.check(now=3610)
        self.assertEqual(self.guard.in_flight, 17)
        self.append(
            self.client,
            "Phase profiling (profiling) complete | completed=3,838, cancelled=4, errors=0\n",
        )
        self.append(self.client, "Waiting for result domains: server_metrics\n")
        self.assertIsNone(self.guard.check(now=3940))
        self.assertIsNone(self.guard.check(now=10000))
        self.assertFalse(self.guard.phase_active)
        self.append(self.server, "Watchdog caught collective operation timeout\n")
        self.assertEqual(self.guard.check(now=10001)["reason"], "fatal_worker_error")

    def test_next_phase_rearms_and_sending_complete_still_watches_requests(self):
        self.append(self.client, "Phase warmup (warmup) complete | completed=354\n")
        self.assertIsNone(self.guard.check(now=1000))
        self.append(
            self.client,
            "Phase profiling (profiling) started | target: 3600.0s duration\n",
        )
        self.append(
            self.client,
            "Phase profiling (profiling) sending complete | sent=1,042, completed=1,040, in_flight=2\n",
        )
        self.assertIsNone(self.guard.check(now=2000))
        failure = self.guard.check(now=2901)
        self.assertEqual(failure["reason"], "inference_stalled")
        self.assertEqual(failure["returned"], 1040)
        self.assertEqual(failure["in_flight"], 2)

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

    def test_allocation_log_identifies_incomplete_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "prefill").mkdir()
            fake = Mock()
            fake.cuda.memory_stats.return_value = {"num_alloc_retries": 2}
            fake.empty.side_effect = [object(), RuntimeError("allocation failed")]
            with (
                patch.dict(sys.modules, {"torch": fake}),
                patch.dict(
                    os.environ,
                    {
                        "ATOM_CI_RUNTIME_DIAGNOSTICS": str(root),
                        "ATOM_CI_RUNTIME_ROLE": "prefill",
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "allocation failed"),
            ):
                watch.empty_gather_kv((2046761, 16, 128), dtype="bf16", device="cuda:0")
            rows = [
                json.loads(line)
                for line in next(root.rglob("*.allocations.jsonl"))
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [row["stage"] for row in rows], ["before_k", "after_k_before_v"]
            )
            self.assertEqual(rows[0]["memory"]["num_alloc_retries"], 2)
            fake.cuda.synchronize.assert_not_called()
            self.assertEqual(fake.empty.call_count, 2)

    def test_native_kv_scope_restores_allocator_even_on_failure(self):
        for fail in (False, True):
            fake = Mock()
            settings = []
            fake.cuda.memory._set_allocator_settings.side_effect = settings.append

            def allocate(*args, settings=settings, fail=fail, **kwargs):
                self.assertEqual(
                    settings[-1], "max_split_size_mb:512,expandable_segments:False"
                )
                if fail:
                    raise RuntimeError("KV allocation failed")
                return "native KV"

            fake.zeros.side_effect = allocate
            with (
                patch.dict(sys.modules, {"torch": fake}),
                patch.dict(
                    os.environ,
                    {
                        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True"
                    },
                ),
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "KV allocation failed"):
                        watch.ipc_compatible_kv_zeros(
                            100, dtype="uint8", device="cuda:0"
                        )
                else:
                    self.assertEqual(
                        watch.ipc_compatible_kv_zeros(
                            100, dtype="uint8", device="cuda:0"
                        ),
                        "native KV",
                    )
            self.assertEqual(
                settings[-1], "max_split_size_mb:512,expandable_segments:True"
            )
            fake.cuda.empty_cache.assert_not_called()

    def test_native_kv_scope_keeps_default_allocator_unchanged(self):
        fake = Mock()
        with (
            patch.dict(sys.modules, {"torch": fake}),
            patch.dict(os.environ, {}, clear=True),
        ):
            watch.ipc_compatible_kv_zeros(100, dtype="uint8", device="cuda:0")
        fake.cuda.memory._set_allocator_settings.assert_not_called()
        fake.zeros.assert_called_once_with(100, dtype="uint8", device="cuda:0")

    def test_runtime_copy_preserves_source_and_starts_python_sampler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, runtime, output = root / "source", root / "runtime", root / "out"
            (source / "atom").mkdir(parents=True)
            original = "VALUE = 42\n"
            (source / "atom/__init__.py").write_text(original)
            attention = source / "atom/model_ops/attention_mha.py"
            attention.parent.mkdir()
            original_attention = (
                SCRIPTS.parents[2] / "atom/model_ops/attention_mha.py"
            ).read_text()
            attention.write_text(original_attention)
            runner = source / "atom/model_engine/model_runner.py"
            runner.parent.mkdir()
            original_runner = (
                SCRIPTS.parents[2] / "atom/model_engine/model_runner.py"
            ).read_text()
            runner.write_text(original_runner)
            proposer = source / "atom/spec_decode/eagle_proposer.py"
            proposer.parent.mkdir()
            original_proposer = (
                SCRIPTS.parents[2] / "atom/spec_decode/eagle_proposer.py"
            ).read_text()
            proposer.write_text(original_proposer)
            watch.prepare(source, runtime, output)
            self.assertFalse((runtime / "atom/_ci_bounded_gather.py").exists())
            self.assertEqual(proposer.read_text(), original_proposer)
            compile(
                (runtime / "atom/spec_decode/eagle_proposer.py").read_text(),
                "eagle_proposer.py",
                "exec",
            )
            self.assertEqual(runner.read_text(), original_runner)
            instrumented_runner = (
                runtime / "atom/model_engine/model_runner.py"
            ).read_text()
            compile(instrumented_runner, "model_runner.py", "exec")
            self.assertIn("buf = ipc_compatible_kv_zeros(", instrumented_runner)
            self.assertEqual(attention.read_text(), original_attention)
            instrumented = (runtime / "atom/model_ops/attention_mha.py").read_text()
            compile(instrumented, "attention_mha.py", "exec")
            self.assertIn("k_full, v_full = empty_gather_kv(", instrumented)
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
