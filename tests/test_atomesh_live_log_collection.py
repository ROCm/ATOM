# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check bounded, read-only collection using fake logs and scheduler output."""

import argparse
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "live_logs", ROOT / ".github/scripts/atomesh/collect_minimax_live_logs.py"
)
LIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIVE)


class LiveLogsTest(unittest.TestCase):
    def test_run_id_cannot_inject_a_path_or_glob(self):
        for value in ("../12345", "12345*", "1;id", "abc", ""):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                LIVE.run_id(value)

    def test_snapshot_is_scoped_bounded_and_leaves_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared"
            target = shared / f"{LIVE.CASE}-38752033481-20261015"
            log = target / "slurm_job-123/rank-0/host-memory.log"
            log.parent.mkdir(parents=True)
            log.write_bytes(b"old text\nlatest\n")
            (log.parent / "docker.env").write_text("not a log")
            outside = root / "outside.log"
            outside.write_text("unrelated data")
            (log.parent / "escape.log").symlink_to(outside)
            other = shared / f"{LIVE.CASE}-99999999999-20261015/slurm-999.out"
            other.parent.mkdir(parents=True)
            other.write_text("other run")
            with (
                patch.object(LIVE, "scheduler_state", return_value={"rows": []}),
                patch.object(LIVE, "MAX_FILE_BYTES", 7),
                redirect_stdout(io.StringIO()),
            ):
                result = LIVE.collect("38752033481", root / "out", roots=(shared,))
            self.assertEqual(len(result["files"]), 1)
            copied = result["files"][0]
            self.assertTrue(copied["tail_only"])
            self.assertEqual(
                (root / "out" / copied["artifact"]).read_bytes(), b"latest\n"
            )
            self.assertEqual(log.read_bytes(), b"old text\nlatest\n")
            self.assertTrue((root / "out/snapshot.json").is_file())

    def test_scheduler_snapshot_keeps_only_target_run(self):
        stdout = (
            f"42|{LIVE.CASE}-38752033481-1|PENDING|0:00|Resources|\n"
            f"99|{LIVE.CASE}-99999999999-1|RUNNING|1:00|node1|node1\n"
        )
        response = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        with patch.object(LIVE.subprocess, "run", return_value=response) as run:
            result = LIVE.scheduler_state("38752033481")
        self.assertEqual(len(result["rows"]), 1)
        self.assertIn("PENDING", result["rows"][0])
        self.assertEqual(run.call_args.args[0][0], "squeue")
        self.assertEqual(run.call_args.kwargs["timeout"], 20)

    def test_tail_windows_keep_untimestamped_compiler_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "prefill.log"
            log.write_text(
                "[atom 00:00:02] Scheduled prefill batch: 1 reqs, 3 new tokens "
                "(done: [5], new: [3]), req_ids: (1,)\n"
                "warning: failed to meet occupancy target in kernel_gemm_0\n"
                "[atom 00:00:20] Request 2 arrived\n"
                "warning: failed to meet occupancy target in kernel_gemm_1\n"
            )
            summary = LIVE.summarize_prefill_tail_windows(
                log,
                {
                    "top": [
                        {
                            "request_start_ns": 86402_000_000_000,
                            "ttft_ms": 2000,
                            "x_request_id": "a",
                        }
                    ]
                },
            )
        self.assertEqual(summary["compilation_log_matches"], 1)
        event = summary["compilation_events"][0]
        self.assertFalse(event["has_own_timestamp"])
        self.assertEqual(event["preceding_log_clock"], "00:00:02")
        self.assertEqual(
            summary["scheduler_by_second"]["00:00:02"]["causal_qk_pairs"], 21
        )

    def test_diagnostic_workflow_does_not_submit_a_model(self):
        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
        )
        load_if = workflow["jobs"]["load-config"]["if"]
        self.assertIn("inputs.collect_minimax_run != ''", load_if)
        self.assertIn("!inputs.run_model_benchmark", load_if)
        job = workflow["jobs"]["collect-minimax-live-logs"]
        self.assertIn("!inputs.run_model_benchmark", job["if"])
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        for command in ("sbatch", "srun", "scancel", "docker run"):
            self.assertNotIn(command, commands)
        self.assertIn("collect_minimax_live_logs.py", commands)


if __name__ == "__main__":
    unittest.main()
