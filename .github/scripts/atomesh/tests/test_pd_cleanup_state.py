"""Peer acknowledgements stay unverified across transient shared-file errors."""

import errno
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "cleanup_state", Path(__file__).resolve().parents[1] / "pd_cleanup_state.py"
)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class PeerStateTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for rank in range(2):
            value = {
                "job_id": "9000",
                "run_token": "a" * 32,
                "rank": rank,
                "passed": True,
                "finished": True,
            }
            for prefix in ("gpu-preflight", "cleanup-complete"):
                (self.root / f"{prefix}-{rank}.json").write_text(json.dumps(value))
        (self.root / "workload-failure.json").write_text(
            json.dumps(
                {
                    "job_id": "9000",
                    "run_token": "a" * 32,
                    "return_code": 1,
                    "num_ranks": 2,
                }
            )
        )

    def run_cli(self, action):
        with patch(
            "sys.argv",
            [
                "pd_cleanup_state.py",
                action,
                "--run-dir",
                str(self.root),
                "--job-id",
                "9000",
                "--run-token",
                "a" * 32,
                "--num-ranks",
                "2",
            ],
        ):
            return state.main()

    def test_stale_read_is_pending_then_validates_fresh_contents(self):
        for action in ("ready", "failed", "cleaned"):
            with self.subTest(action=action):
                with patch.object(
                    Path,
                    "read_text",
                    side_effect=OSError(errno.ESTALE, "Stale file handle"),
                ):
                    self.assertEqual(self.run_cli(action), 1)
                self.assertEqual(self.run_cli(action), 0)

    def test_recovered_file_still_requires_current_run_and_passed_rank(self):
        peer = self.root / "gpu-preflight-1.json"
        original = json.loads(peer.read_text())
        for change in (
            {"job_id": "8999"},
            {"run_token": "b" * 32},
            {"rank": 0},
            {"passed": False},
        ):
            with self.subTest(change=change):
                peer.write_text(json.dumps(dict(original, **change)))
                with patch.object(
                    Path,
                    "read_text",
                    side_effect=OSError(errno.ESTALE, "Stale file handle"),
                ):
                    self.assertEqual(self.run_cli("ready"), 1)
                self.assertEqual(self.run_cli("ready"), 1)

    def test_other_io_and_invalid_json_remain_errors(self):
        for error in (
            PermissionError(errno.EACCES, "Denied"),
            OSError(errno.EIO, "I/O error"),
        ):
            with (
                self.subTest(error=error),
                patch.object(Path, "read_text", side_effect=error),
            ):
                self.assertEqual(self.run_cli("ready"), 2)
        (self.root / "gpu-preflight-1.json").write_text("invalid json")
        self.assertEqual(self.run_cli("ready"), 2)


if __name__ == "__main__":
    unittest.main()
