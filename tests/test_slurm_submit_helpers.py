# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only Slurm result tests; also runnable directly with Python."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPERS = (
    Path(__file__).resolve().parents[1] / ".github/scripts/slurm_submit_helpers.sh"
)


class SlurmResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_shell(self, body, **env):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -euo pipefail\nsource "$HELPERS"\n'
                + body
                + '\nprintf "RESULT=%s|%s|%s\\n" '
                '"$SLURM_STATE" "$SLURM_EXIT_CODE" "$SLURM_JOB_RC"',
            ],
            env={
                "PATH": os.environ["PATH"],
                "HELPERS": str(HELPERS),
                "TEST_DIR": str(self.root),
                "SLURM_ACCOUNTING_TIMEOUT": "0",
                **env,
            },
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.split("RESULT=")[-1].strip()

    def test_accounting_interfaces(self):
        cases = [
            (
                {"USES_SPUR_CONTROLLER": "1", "SLURM_ACCOUNT": "amd-aifw-dev"},
                "--account amd-aifw-dev --brief --noheader",
                "999 FAILED 1:0\n5147 COMPLETED 0:0",
            ),
            (
                {"USES_SPUR_CONTROLLER": "1", "SPUR_ACCOUNTING_ADDR": "http://test"},
                "--accounting http://test --brief --noheader",
                "5147 COMPLETE 0:0",
            ),
            (
                {"USES_SPUR_CONTROLLER": "1"},
                "-j 5147 -X -n -P -o State,ExitCode",
                "COMPLETED|0:0",
            ),
            (
                {"USES_SPUR_CONTROLLER": "0", "SLURM_ACCOUNT": "amd-frameworks"},
                "-j 5147 -X -n -P -o State,ExitCode",
                "COMPLETED|0:0",
            ),
        ]
        for env, expected_args, reply in cases:
            with self.subTest(env=env):
                result = self.run_shell(
                    'sacct() { printf "%s" "$*" > "$TEST_DIR/args"; '
                    'printf "%s\\n" "$REPLY"; }\nread_slurm_exit_code 5147',
                    REPLY=reply,
                    **env,
                )
                self.assertIn(result, ("COMPLETED|0:0|0", "COMPLETE|0:0|0"))
                self.assertEqual((self.root / "args").read_text(), expected_args)

    def test_terminal_and_unavailable_states(self):
        cases = {
            "COMPLETED+|0:0": "COMPLETED|0:0|0",
            "FAILED|7:0": "FAILED|7:0|7",
            "FAILED|0:0": "FAILED|0:0|1",
            "CANCELLED|0:15": "CANCELLED|0:15|143",
            "COMPLETING|0:0": "unknown|unknown|2",
            "RUNNING|0:0": "unknown|unknown|2",
            "": "unknown|unknown|2",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(
                    self.run_shell(
                        'sacct() { printf "%s\\n" "$REPLY"; }\n'
                        "read_slurm_exit_code 5147",
                        REPLY=reply,
                    ),
                    expected,
                )

    def test_delayed_accounting(self):
        result = self.run_shell(
            """
sacct() {
  if [[ -f "$TEST_DIR/queried" ]]; then
    echo 'COMPLETED|0:0'
  else
    touch "$TEST_DIR/queried"
    echo 'COMPLETING|0:0'
  fi
}
read_slurm_exit_code 5147
""",
            SLURM_ACCOUNTING_TIMEOUT="5",
            SLURM_ACCOUNTING_POLL_INTERVAL="0",
        )
        self.assertEqual(result, "COMPLETED|0:0|0")

    def test_status_file_fallback(self):
        cases = [
            ({"rank-rc-0": "0", "rank-rc-1": "0"}, "COMPLETED|0:0|0"),
            ({"rank-rc-0": "0", "rank-rc-1": "7"}, "FAILED|7:0|7"),
            ({"rank-rc-0": "0", "rank-rc-1.tmp": "0"}, "unknown|unknown|2"),
            ({"rank-rc-0": "0", "rank-rc-2": "0"}, "unknown|unknown|2"),
            ({"rank-rc-0": "0", "rank-rc-1": "bad"}, "unknown|unknown|2"),
            ({"rank-rc-0": "0", "rank-rc-1": "256"}, "unknown|unknown|2"),
            ({"slurm-job.rc": "0"}, "COMPLETED|0:0|0"),
            (
                {"slurm-job.rc": "9", "rank-rc-0": "0", "rank-rc-1": "0"},
                "FAILED|9:0|9",
            ),
            ({"slurm-job.rc": "bad"}, "unknown|unknown|2"),
        ]
        for index, (files, expected) in enumerate(cases):
            with self.subTest(files=files):
                status_dir = self.root / str(index)
                status_dir.mkdir()
                for name, value in files.items():
                    (status_dir / name).write_text(value + "\n")
                self.assertEqual(
                    self.run_shell(
                        "sacct() { return 1; }\nread_slurm_exit_code 5147\n"
                        'read_slurm_status_files "$STATUS_DIR" 2',
                        STATUS_DIR=str(status_dir),
                    ),
                    expected,
                )

    def test_accounting_failure_is_not_overridden_by_files(self):
        (self.root / "slurm-job.rc").write_text("0\n")
        self.assertEqual(
            self.run_shell(
                "sacct() { echo 'FAILED|42:0'; }\nread_slurm_exit_code 5147\n"
                'read_slurm_status_files "$TEST_DIR" 2'
            ),
            "FAILED|42:0|42",
        )


if __name__ == "__main__":
    unittest.main()
