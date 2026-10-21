"""Validate CI reconciliation without a scheduler, containers or GPUs."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_job_result.py"
)
SPEC = importlib.util.spec_from_file_location("pd_job_result", SCRIPT)
RESULT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESULT)


class JobResultTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for rank in range(2):
            RESULT.publish(self.root, "3619", "this-run", rank, 2, "completed")
            (self.root / f"rank-rc-{rank}").write_text("0\n")

    def resolve(self, state="FAILED", exit_code="1:0", rc=1, spur=True):
        return RESULT.resolve(
            self.root, "3619", "this-run", 2, state, exit_code, rc, spur
        )

    def test_result_writer_creates_missing_job_directory(self):
        directory = self.root / "not-started" / "slurm_job-42"
        result = RESULT.resolve(directory, "42", "token", 1, "FAILED", "2:0", 2, True)
        RESULT.write_json(directory / "job-result.json", result)
        saved = json.loads((directory / "job-result.json").read_text())
        self.assertEqual(saved["result"]["return_code"], 2)
        self.assertEqual(saved["workload"]["state"], "UNVERIFIED")

    def test_completed_workload_overrides_generic_spur_failure(self):
        result = self.resolve()
        self.assertEqual(
            result["result"],
            {"state": "COMPLETED", "return_code": 0, "source": "workload"},
        )
        self.assertEqual(
            result["scheduler"],
            {"state": "FAILED", "exit_code": "1:0", "return_code": 1},
        )
        self.assertTrue(result["scheduler_workload_mismatch"])

    def test_explicit_scheduler_failures_and_active_states_are_preserved(self):
        for state, code, rc in [
            ("CANCELLED", "1:0", 1),
            ("TIMEOUT", "1:0", 1),
            ("OUT_OF_MEMORY", "1:0", 1),
            ("NODE_FAIL", "1:0", 1),
            ("PREEMPTED", "1:0", 1),
            ("FAILED", "7:0", 7),
            ("FAILED", "0:15", 143),
            ("FAILED", "0:0", 1),
            ("COMPLETING", "0:0", 75),
            ("RUNNING", "0:0", 75),
            ("unknown", "unknown", 75),
            ("COMPLETED", "0:0", 0),
        ]:
            with self.subTest(state=state, code=code):
                result = self.resolve(state, code, rc)
                self.assertEqual(result["result"]["state"], state)
                self.assertEqual(result["result"]["return_code"], rc)
                self.assertFalse(result["scheduler_workload_mismatch"])

    def test_native_slurm_failure_is_preserved(self):
        self.assertEqual(self.resolve(spur=False)["result"]["return_code"], 1)

    def test_partial_stale_or_malformed_evidence_does_not_override(self):
        marker = self.root / "rank-workload-1.json"
        original = json.loads(marker.read_text())
        for change in [
            {"status": "running"},
            {"run_token": "previous-run"},
            {"job_id": "3618"},
            {"rank": 0},
            {"num_ranks": 3},
            {"schema_version": 2},
        ]:
            with self.subTest(change=change):
                marker.write_text(json.dumps({**original, **change}))
                self.assertEqual(self.resolve()["result"]["return_code"], 1)
        for content in ["", "not json", "null", "[]"]:
            marker.write_text(content)
            self.assertEqual(self.resolve()["result"]["return_code"], 1)
        marker.unlink()
        self.assertEqual(self.resolve()["result"]["return_code"], 1)

    def test_failure_after_workload_completion_is_not_overridden(self):
        rc_file = self.root / "rank-rc-1"
        for content in ["7\n", "143\n", "bad\n", ""]:
            rc_file.write_text(content)
            self.assertEqual(self.resolve()["result"]["return_code"], 1)
        rc_file.unlink()
        self.assertEqual(self.resolve()["result"]["return_code"], 1)

    def test_empty_submission_token_or_rank_set_is_not_proof(self):
        self.assertFalse(RESULT.workload_completed(self.root, "3619", "", 2))
        self.assertFalse(RESULT.workload_completed(self.root, "3619", "this-run", 0))

    def test_peer_failure_requires_current_submission_identity(self):
        marker = self.root / "rank-workload-0.json"
        (self.root / "rank-rc-0").write_text("134\n")
        original = json.loads(marker.read_text())
        for status in ("running", "completed"):
            marker.write_text(json.dumps({**original, "status": status}))
            self.assertEqual(
                RESULT.failed_peer(self.root, "3619", "this-run", 1, 2), (0, 134)
            )
        for change in [
            {"run_token": "old"},
            {"job_id": "3618"},
            {"rank": 1},
            {"num_ranks": 3},
            {"schema_version": 2},
            {"status": "unknown"},
        ]:
            with self.subTest(change=change):
                marker.write_text(json.dumps({**original, **change}))
                self.assertIsNone(
                    RESULT.failed_peer(self.root, "3619", "this-run", 1, 2)
                )
        for content in ("", "broken", "null", "[]"):
            marker.write_text(content)
            self.assertIsNone(RESULT.failed_peer(self.root, "3619", "this-run", 1, 2))
        marker.unlink()
        self.assertIsNone(RESULT.failed_peer(self.root, "3619", "this-run", 1, 2))

    def test_peer_check_ignores_self_success_and_incomplete_codes(self):
        code = self.root / "rank-rc-0"
        code.write_text("134\n")
        self.assertIsNone(RESULT.failed_peer(self.root, "3619", "this-run", 0, 2))
        self.assertIsNone(RESULT.failed_peer(self.root, "3619", "", 1, 2))
        for content in ("0", "", "bad", "-1", "256"):
            code.write_text(content)
            self.assertIsNone(RESULT.failed_peer(self.root, "3619", "this-run", 1, 2))
        code.unlink()
        self.assertIsNone(RESULT.failed_peer(self.root, "3619", "this-run", 1, 2))

    def test_server_waits_exit_on_peer_failure_without_sleeping(self):
        source = SCRIPT.with_name("pd_server_atom.sh").read_text()
        functions = source[
            source.index("exit_if_peer_failed() {") : source.index(
                "start_logged_process() {"
            )
        ]
        (self.root / "rank-rc-0").write_text("134\n")
        for command in ("wait_http http://unused router 7200", "wait_router_closed"):
            with self.subTest(command=command):
                process = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "set -euo pipefail\n"
                        + functions
                        + "\ncurl() { return 1; }\nsleep() { exit 99; }\n"
                        "exit_if_mooncake_store_died() { :; }\n" + command,
                    ],
                    env={
                        **os.environ,
                        "ATOMESH_SCRIPT_DIR": str(SCRIPT.parent),
                        "ATOMESH_RUN_TOKEN": "this-run",
                        "ATOMESH_NUM_NODES": "2",
                        "SLURM_JOB_ID": "3619",
                        "NODE_RANK": "1",
                        "RUN_DIR": str(self.root),
                        "NODE0_ADDR": "unused",
                        "ROUTER_PORT": "8000",
                    },
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(
                    process.returncode, 134, process.stdout + process.stderr
                )
                self.assertIn("peer rank 0 exited rc=134", process.stdout)

    def test_cli_exits_with_effective_result_and_retains_raw_failure(self):
        args = [
            sys.executable,
            str(SCRIPT),
            "resolve",
            "--run-dir",
            str(self.root),
            "--job-id",
            "3619",
            "--run-token",
            "this-run",
            "--num-ranks",
            "2",
            "--scheduler-state",
            "FAILED",
            "--scheduler-exit-code",
            "1:0",
            "--scheduler-rc",
            "1",
            "--spur",
            "1",
        ]
        process = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("WARNING: Spur reported FAILED/1:0", process.stdout)
        saved = json.loads((self.root / "job-result.json").read_text())
        self.assertEqual(saved["scheduler"]["state"], "FAILED")
        self.assertEqual(saved["result"]["state"], "COMPLETED")
        (self.root / "rank-rc-1").write_text("7\n")
        process = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 1)


if __name__ == "__main__":
    unittest.main()
