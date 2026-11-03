# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/issue382/inspect_runs.py"
SPEC = importlib.util.spec_from_file_location("issue382_inspection", SCRIPT)
INSPECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT)


class ResultRecoveryTest(unittest.TestCase):
    def test_completed_records_are_full_and_live_or_failed_records_are_skipped(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            job = source / "slurm_job-42"
            records = job / "benchmark_results" / "records.jsonl"
            records.parent.mkdir(parents=True)
            payload = b'{"completed":true}\n' * 100
            records.write_bytes(payload)
            proof = {
                "job_id": "42",
                "num_ranks": 1,
                "rank": 0,
                "run_token": "original-submission",
                "status": "running",
            }
            marker = job / "rank-workload-0.json"
            marker.write_text(json.dumps(proof))
            (job / "rank-rc-0").write_text("0\n")
            self.assertEqual(
                INSPECT.collect_completed_results(source, target), []
            )
            self.assertFalse(target.exists())
            proof["status"] = "completed"
            marker.write_text(json.dumps(proof))
            (job / "rank-rc-0").write_text("1\n")
            self.assertEqual(
                INSPECT.collect_completed_results(source, target), []
            )
            (job / "rank-rc-0").write_text("0\n")
            rows = INSPECT.collect_completed_results(source, target)
            self.assertTrue(all(r.get("copied_full") for r in rows))
            self.assertEqual(
                (target / records.relative_to(source)).read_bytes(), payload
            )

    def test_oversized_record_is_explicitly_missing_never_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            job = source / "slurm_job-42"
            records = job / "eval_results" / "samples.jsonl"
            records.parent.mkdir(parents=True)
            records.write_bytes(b"x" * 2048)
            (job / "rank-workload-0.json").write_text(
                json.dumps(
                    {
                        "job_id": "42",
                        "num_ranks": 1,
                        "rank": 0,
                        "run_token": "original-submission",
                        "status": "completed",
                    }
                )
            )
            (job / "rank-rc-0").write_text("0\n")
            rows = INSPECT.collect_completed_results(
                source, target, max_file_bytes=1024
            )
            entry = next(r for r in rows if r["path"].endswith("samples.jsonl"))
            self.assertIn("error", entry)
            self.assertNotIn("copied_full", entry)
            self.assertFalse((target / records.relative_to(source)).exists())


if __name__ == "__main__":
    unittest.main()
