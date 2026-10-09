"""Exercise the peer-state exit codes consumed by the Slurm launcher."""

import errno
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_cleanup_state.py"
)
SPEC = importlib.util.spec_from_file_location("pd_cleanup_state", SCRIPT)
STATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATE)


class CleanupStateTest(unittest.TestCase):
    def check_failed(self, read, expected):
        argv = [
            str(SCRIPT),
            "failed",
            "--run-dir",
            "/unused",
            "--job-id",
            "6095",
            "--run-token",
            "this-run",
            "--num-ranks",
            "2",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(Path, "read_text", side_effect=read),
            patch("time.sleep"),
        ):
            self.assertEqual(STATE.main(), expected)

    def test_ready_after_transient_stale_handle_does_not_abort_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(2):
                (root / f"gpu-preflight-{rank}.json").write_text(
                    json.dumps(
                        {
                            "job_id": "6095",
                            "run_token": "this-run",
                            "rank": rank,
                            "passed": True,
                        }
                    )
                )
            original_read = Path.read_text
            failed = False

            def read(path, *args, **kwargs):
                nonlocal failed
                if path == root / "gpu-preflight-0.json" and not failed:
                    failed = True
                    raise OSError(errno.ESTALE, "Stale file handle")
                return original_read(path, *args, **kwargs)

            argv = [
                str(SCRIPT),
                "ready",
                "--run-dir",
                directory,
                "--job-id",
                "6095",
                "--run-token",
                "this-run",
                "--num-ranks",
                "2",
            ]
            with patch.object(sys, "argv", argv), patch.object(Path, "read_text", read):
                self.assertEqual(STATE.main(), 0)

    def test_stale_failure_read_still_reports_actual_peer_failure(self):
        marker = json.dumps(
            {
                "job_id": "6095",
                "run_token": "this-run",
                "return_code": 2,
                "num_ranks": 2,
            }
        )
        self.check_failed([OSError(errno.ESTALE, "Stale file handle"), marker], 0)

    def test_persistent_or_invalid_evidence_still_stops_launcher(self):
        for error in [
            OSError(errno.ESTALE, "Stale file handle"),
            PermissionError(errno.EACCES, "Permission denied"),
            json.JSONDecodeError("invalid marker", "{", 0),
        ]:
            with self.subTest(error=error):
                self.check_failed(error, 2)

    def test_old_run_failure_does_not_stop_current_run_after_retry(self):
        marker = json.dumps(
            {
                "job_id": "6095",
                "run_token": "previous-run",
                "return_code": 2,
                "num_ranks": 2,
            }
        )
        self.check_failed([OSError(errno.ESTALE, "Stale file handle"), marker], 1)


if __name__ == "__main__":
    unittest.main()
