# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the CI replay gate, runtime patcher and diagnostic transport."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


debug = load("minimax_replay_debug")
patcher = load("prepare_minimax_replay_debug")
client = load("minimax_full_model_replay")


def sequence(index, length):
    return SimpleNamespace(id=index, num_prompt_tokens=length)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.plan = {
            "lengths": [item[1] for item in client.TARGETS],
            "done": [item[2] for item in client.TARGETS],
            "new": [item[3] for item in client.TARGETS],
        }
        self.scheduler = SimpleNamespace(waiting=deque(), running=deque())
        self.gate = debug.ReplayGate(self.directory)

    def arm(self):
        debug.atomic_json(self.directory / "armed.json", self.plan)

    def enqueue(self):
        seqs = [
            sequence(i + 8, length) for i, length in enumerate(self.plan["lengths"])
        ]
        self.scheduler.waiting.extend(reversed(seqs))
        return seqs

    def test_primers_pass_then_target_waits_and_reorders_once(self):
        self.scheduler.waiting.append(sequence(0, 131079))
        self.assertFalse(self.gate.hold(self.scheduler))
        self.scheduler.waiting.clear()
        self.arm()
        self.enqueue()
        last = self.scheduler.waiting.pop()
        self.assertTrue(self.gate.hold(self.scheduler))
        self.scheduler.waiting.append(last)
        self.assertFalse(self.gate.hold(self.scheduler))
        self.assertEqual([seq.id for seq in self.scheduler.waiting], list(range(8, 16)))
        batch = SimpleNamespace(
            req_ids=list(range(8, 16)),
            num_cached_tokens=self.plan["done"],
            num_scheduled_tokens=self.plan["new"],
        )
        self.gate.verify(batch)
        result = json.loads((self.directory / "gate-batch.json").read_text())
        self.assertTrue(result["matched"])
        self.assertEqual(result["total_kv"], 1350868)
        # The following partial request is not gated or forced into another 8.
        self.scheduler.waiting = deque([sequence(15, 170398)])
        self.assertFalse(self.gate.hold(self.scheduler))
        self.gate.verify(
            SimpleNamespace(
                req_ids=[15], num_cached_tokens=[164992], num_scheduled_tokens=[5406]
            )
        )

    def test_shape_mismatch_is_recorded_before_dispatch(self):
        self.arm()
        self.enqueue()
        self.gate.hold(self.scheduler)
        batch = SimpleNamespace(
            req_ids=list(range(8, 16)),
            num_cached_tokens=self.plan["done"],
            num_scheduled_tokens=[1] * 8,
        )
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            self.gate.verify(batch)
        self.assertFalse(
            json.loads((self.directory / "gate-batch.json").read_text())["matched"]
        )

    def test_foreign_duplicate_and_running_requests_rejected(self):
        self.arm()
        for waiting, running in [
            ([sequence(1, 17)], []),
            ([sequence(1, 142015), sequence(2, 142015)], []),
            ([], [sequence(1, 142015)]),
        ]:
            with self.subTest(waiting=waiting, running=running):
                self.scheduler.waiting, self.scheduler.running = (
                    deque(waiting),
                    deque(running),
                )
                with self.assertRaises(RuntimeError):
                    debug.ReplayGate(self.directory).hold(self.scheduler)

    def test_missing_request_has_bounded_wait(self):
        self.arm()
        with patch.object(debug.time, "monotonic", return_value=0) as clock:
            self.assertTrue(self.gate.hold(self.scheduler))
            clock.return_value = 31
            with self.assertRaises(TimeoutError):
                self.gate.hold(self.scheduler)


class PatchTests(unittest.TestCase):
    def checkout(self, root):
        for name in patcher.ROOT_FILES:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)

    def test_runtime_patch_is_auditable_and_source_drift_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkout(root)
            patcher.apply(root, root / "evidence")
            manifest = json.loads(
                (root / "evidence/runtime-patch-manifest.json").read_text()
            )
            self.assertEqual(len(manifest), 5)
            self.assertTrue((root / "atom/_minimax_replay_debug.py").exists())
            for name in patcher.ROOT_FILES:
                self.assertNotEqual(
                    manifest[name]["before_sha256"], manifest[name]["after_sha256"]
                )
            self.checkout(root)
            last = root / patcher.ROOT_FILES[-1]
            last.write_text(
                last.read_text().replace(
                    "        stage_b.stream.synchronize()", "        changed_api()"
                )
            )
            before = {name: (root / name).read_bytes() for name in patcher.ROOT_FILES}
            with self.assertRaises(ValueError):
                patcher.apply(root, root / "bad-evidence")
            self.assertEqual(
                before,
                {name: (root / name).read_bytes() for name in patcher.ROOT_FILES},
            )

    def test_launcher_patches_a_copy_and_resolves_it_before_mounted_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, logs = root / "mounted", root / "logs"
            self.checkout(source)
            (source / "atom/__init__.py").write_text("")
            (source / "atom/resource.txt").write_text("package resource")
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            for path in before:
                path.chmod(0o444)
            script = (SCRIPTS / "pd_server_atom.sh").read_text()
            start = script.index(
                'if [[ "${ATOMESH_M3_REPLAY_GATE:-false}" == "true" ]]; then'
            )
            stop = script.index("# Names the last apply_role_env()", start)
            probe = (
                "\n"
                + r"""env "${ATOM_SOURCE_ENV[@]}" python3 -c 'import atom; from atom import _minimax_replay_debug as d; print(atom.__file__); print(d.__file__)'
printf '%s\n' "${ATOMESH_M3_REPLAY_PIPE_DIR}"
"""
            )
            completed = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script[start:stop] + probe],
                cwd=source,
                env={
                    **os.environ,
                    "ATOMESH_SCRIPT_DIR": str(SCRIPTS),
                    "ATOMESH_M3_REPLAY_GATE": "true",
                    "BENCHMARK_KIND": "minimax_full_model_replay",
                    "ATOMESH_ATOM_SOURCE": "mounted",
                    "SINGLE_NODE_PD": "1",
                    "RUN_DIR": str(logs),
                    # The mounted checkout deliberately remains earlier than any
                    # installed package. Only the launch cwd must select the copy.
                    "PYTHONPATH": str(source),
                },
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            evidence = logs / "benchmark_results/stalled-lengths-replay/diagnostics"
            runtime = Path(
                json.loads((evidence / "runtime-source.json").read_text())[
                    "runtime_source"
                ]
            )
            self.addCleanup(shutil.rmtree, runtime)
            self.addCleanup(shutil.rmtree, completed.stdout.splitlines()[-1])
            self.assertIn(str(runtime / "atom/__init__.py"), completed.stdout)
            self.assertIn(
                str(runtime / "atom/_minimax_replay_debug.py"), completed.stdout
            )
            self.assertEqual(
                (runtime / "atom/resource.txt").read_text(), "package resource"
            )
            self.assertFalse((source / "atom/_minimax_replay_debug.py").exists())
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertIn("_m3debug", (runtime / patcher.ROOT_FILES[0]).read_text())

    def test_inactive_hook_does_not_create_trace_or_import_gpu(self):
        with (
            patch.dict(os.environ, {"ATOM_M3_REPLAY_DIR": ""}),
            patch.object(debug, "_TRACE", None),
        ):

            @debug.traced("test")
            def function(value):
                return value + 1

            self.assertEqual(function(4), 5)
            self.assertIsNone(debug._TRACE)

    def test_gpu_marker_records_and_queries_without_synchronization(self):
        trace = debug.Trace.__new__(debug.Trace)
        trace.lock = threading.Lock()
        trace.pending, trace.next_event, trace.rank = [], 0, None
        trace.emit = Mock()
        event = Mock()
        event.query.return_value = True
        cuda = Mock()
        cuda.is_current_stream_capturing.return_value = False
        cuda.Event.return_value = event
        cuda.current_device.return_value = 2
        cuda.current_stream.return_value = SimpleNamespace(cuda_stream=123)
        cuda.memory_stats.return_value = {"allocated_bytes.all.current": 17}
        with (
            patch.object(debug, "active", return_value=True),
            patch.object(debug, "_TRACE", trace),
            patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}),
        ):
            debug.mark("target", gpu=True, rank=2)
            self.assertEqual(len(trace.pending), 1)
            trace.poll()
        self.assertEqual(trace.pending, [])
        event.record.assert_called_once()
        event.query.assert_called_once()
        event.synchronize.assert_not_called()
        cuda.synchronize.assert_not_called()

    def test_dump_trigger_writes_only_fifos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo, regular = root / "nccl-pipe-0", root / "nccl-pipe-regular"
            os.mkfifo(fifo)
            regular.write_text("keep")
            fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            try:
                record = Mock()
                client.trigger_dumps(
                    root, SimpleNamespace(wait=lambda timeout: False), record
                )
                self.assertEqual(os.read(fd, 1), b"1")
                self.assertEqual(regular.read_text(), "keep")
                self.assertEqual(
                    record.call_args.args[0]["pipes"][0]["status"], "requested"
                )
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
