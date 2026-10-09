# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only Slurm result tests; also runnable directly with Python."""

import json
import os
import subprocess
import tempfile
import textwrap
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
                'run_slurm_query() { "$@"; }\nscontrol() { return 1; }\n'
                + body
                + '\nprintf "RESULT=%s|%s|%s\\n" '
                '"$SLURM_STATE" "$SLURM_EXIT_CODE" "$SLURM_JOB_RC"',
            ],
            env={
                "PATH": os.environ["PATH"],
                "HELPERS": str(HELPERS),
                "TEST_DIR": str(self.root),
                "SLURM_ACCOUNTING_TIMEOUT": "0",
                "USES_SPUR_CONTROLLER": "0",
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
                {
                    "USES_SPUR_CONTROLLER": "1",
                    "SLURM_ACCOUNT": "amd-aifw-dev",
                    "SPUR_CONTROLLER_ADDR": "http://controller:6817",
                },
                (
                    "--controller http://controller:6817 -j 5147 --noheader "
                    "--format JobID%30,State%30,ExitCode%20 --account amd-aifw-dev"
                ),
                "999 FAILED 1:0\n5147 COMPLETED 0:0",
            ),
            (
                {
                    "USES_SPUR_CONTROLLER": "1",
                    "SPUR_ACCOUNTING_ADDR": "http://legacy:6819",
                },
                "-j 5147 --noheader --format JobID%30,State%30,ExitCode%20",
                "5147 CANCELLED -1:0",
            ),
            (
                {"USES_SPUR_CONTROLLER": "0", "SLURM_ACCOUNT": "amd-frameworks"},
                "-j 5147 -X -n -P -o JobIDRaw,State,ExitCode",
                "999|FAILED|1:0\n5147|COMPLETED|0:0",
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
                self.assertIn(result, ("COMPLETED|0:0|0", "CANCELLED|-1:0|1"))
                self.assertEqual((self.root / "args").read_text(), expected_args)

    def test_terminal_and_unavailable_states(self):
        cases = {
            "COMPLETED+|0:0": "COMPLETED|0:0|0",
            "FAILED|7:0": "FAILED|7:0|7",
            "FAILED|0:0": "FAILED|0:0|1",
            "CANCELLED|0:15": "CANCELLED|0:15|143",
            "COMPLETING|0:0": "unknown|unknown|75",
            "RUNNING|0:0": "unknown|unknown|75",
            "": "unknown|unknown|75",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(
                    self.run_shell(
                        'sacct() { printf "5147|%s\\n" "$REPLY"; }\n'
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
    echo '5147|COMPLETED|0:0'
  else
    touch "$TEST_DIR/queried"
    echo '5147|COMPLETING|0:0'
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
            ({"rank-rc-0": "0", "rank-rc-1.tmp": "0"}, "unknown|unknown|75"),
            ({"rank-rc-0": "0", "rank-rc-2": "0"}, "unknown|unknown|75"),
            ({"rank-rc-0": "0", "rank-rc-1": "bad"}, "unknown|unknown|75"),
            ({"rank-rc-0": "0", "rank-rc-1": "256"}, "unknown|unknown|75"),
            ({"slurm-job.rc": "0"}, "COMPLETED|0:0|0"),
            (
                {"slurm-job.rc": "9", "rank-rc-0": "0", "rank-rc-1": "0"},
                "FAILED|9:0|9",
            ),
            ({"slurm-job.rc": "bad"}, "unknown|unknown|75"),
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
                "sacct() { echo '5147|FAILED|42:0'; }\nread_slurm_exit_code 5147\n"
                'read_slurm_status_files "$TEST_DIR" 2'
            ),
            "FAILED|42:0|42",
        )


class SlurmMonitoringTest(unittest.TestCase):
    """Run the real shell helpers against isolated scheduler executables."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        mock = textwrap.dedent("""\
            #!/usr/bin/env python3
            import json, os, sys
            from pathlib import Path
            root = Path(os.environ["TEST_DIR"])
            name = Path(sys.argv[0]).name
            with (root / "calls").open("a") as f:
                f.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
            replies = json.loads((root / "replies").read_text())
            if sys.argv[1:] == ["--help"]:
                print(replies.get("help", "Slurm administrative control"))
                sys.exit(0)
            counter = root / (name + ".count")
            index = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(index + 1))
            series = replies.get(name, [{"rc": 1}])
            reply = series[min(index, len(series) - 1)]
            print(reply.get("out", ""))
            print(reply.get("err", ""), file=sys.stderr)
            sys.exit(reply.get("rc", 0))
            """)
        for name in ("squeue", "scontrol", "sacct", "scancel"):
            path = self.bin / name
            path.write_text(mock)
            path.chmod(0o755)

    def run_script(self, script, replies, **env):
        (self.root / "replies").write_text(json.dumps(replies))
        return subprocess.run(
            ["bash", "-c", 'set -euo pipefail\nsource "$HELPERS"\n' + script],
            check=False,
            env={
                "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                "HELPERS": str(HELPERS),
                "TEST_DIR": str(self.root),
                "JOB_ID": "831",
                "CURRENT_USER": "test",
                "SLURM_JOB_ACTIVE": "1",
                "SLURM_JOB_NAME": "test-job",
                "SLURM_CANCEL_HELPER": str(self.root / "cancel.sh"),
                "SLURM_JOB_OUTPUT": str(self.root / "job.out"),
                "SLURM_JOB_ERROR": str(self.root / "job.err"),
                "SLURM_SQUEUE_RETRY_INTERVAL": "0",
                "SLURM_LOG_POLL_INTERVAL": "0",
                "SLURM_ACCOUNTING_TIMEOUT": "0",
                "SLURM_CANCEL_WAIT_SECONDS": "0",
                "SLURM_SQUEUE_INITIAL_ATTEMPTS": "2",
                "SLURM_STATUS_UNKNOWN_TIMEOUT": "0",
                **env,
            },
            capture_output=True,
            text=True,
            timeout=15,
        )

    def calls(self, command):
        return [
            x
            for x in map(json.loads, (self.root / "calls").read_text().splitlines())
            if x[0] == command
        ]

    def test_backend_detection_from_client(self):
        result = self.run_script(
            'detect_slurm_backend; echo "backend=$USES_SPUR_CONTROLLER"',
            {"help": "Administrative control for Spur"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend=1", result.stdout)

    def test_native_backend_detection(self):
        result = self.run_script(
            'detect_slurm_backend; echo "backend=$USES_SPUR_CONTROLLER"',
            {},
            SPUR_CONTROLLER_ADDR="http://inherited-but-unused:6817",
        )
        self.assertIn("backend=0", result.stdout)

    def test_configured_controller_list_is_preserved(self):
        addresses = "http://controller-a:6817,http://controller-b:6817"
        result = self.run_script(
            "detect_slurm_backend; query_slurm_accounting_job 831",
            {
                "help": "Administrative control for Spur",
                "sacct": [{"out": "999 FAILED 1:0\n831 CANCELLED -1:0"}],
            },
            SPUR_CONTROLLER_ADDR=addresses,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CANCELLED|-1:0")
        self.assertIn(addresses, self.calls("sacct")[0])
        self.assertNotIn("-X", self.calls("sacct")[0])
        self.assertNotIn("--accounting", self.calls("sacct")[0])

    def test_missing_queue_entry_does_not_finish_running_job(self):
        result = self.run_script(
            "install_slurm_cancel_traps; monitor_slurm_job 831; read_slurm_exit_code 831; "
            'echo "result=$SLURM_STATE|$SLURM_JOB_RC"',
            {
                "squeue": [{"out": "831|PENDING|0:00|1|(None)"}, {}, {}, {}],
                "scontrol": [
                    {"out": "JobId=831 JobState=PENDING ExitCode=0:0"},
                    {"out": "JobId=831 JobState=RUNNING ExitCode=0:0"},
                    {"out": "JobId=831 JobState=COMPLETED ExitCode=0:0"},
                ],
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified_state=RUNNING", result.stdout)
        self.assertIn("result=COMPLETED|0", result.stdout)
        self.assertEqual(len(self.calls("squeue")), 4)
        self.assertFalse(self.calls("scancel"))
        self.assertFalse(self.calls("sacct"))

    def test_query_error_recovers_and_keeps_diagnostics(self):
        result = self.run_script(
            "install_slurm_cancel_traps; monitor_slurm_job 831",
            {
                "squeue": [{"rc": 1, "err": "transport unavailable"}, {}],
                "scontrol": [
                    {"out": "JobId=831 JobState=RUNNING ExitCode=0:0"},
                    {"out": "JobId=831 JobState=COMPLETED ExitCode=0:0"},
                ],
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("transport unavailable", result.stderr)
        self.assertFalse(self.calls("scancel"))

    def test_status_outage_preserves_job_and_marks_infrastructure_failure(self):
        result = self.run_script(
            "install_slurm_cancel_traps; monitor_slurm_job 831",
            {
                "squeue": [{"rc": 1, "err": "queue offline"}],
                "scontrol": [{"rc": 1, "err": "controller offline"}],
                "sacct": [{"rc": 1, "err": "accounting offline"}],
            },
        )
        self.assertEqual(result.returncode, 75, result.stderr)
        for error in ("queue offline", "controller offline", "accounting offline"):
            self.assertIn(error, result.stderr)
        self.assertEqual(
            (self.root / "cancel.sh.status-unknown").read_text().strip(), "831"
        )
        self.assertFalse(self.calls("scancel"))

    def test_evicted_job_uses_accounting_terminal_state(self):
        result = self.run_script(
            'monitor_slurm_job 831; read_slurm_exit_code 831; echo "result=$SLURM_STATE|$SLURM_JOB_RC"',
            {
                "squeue": [{}],
                "scontrol": [{}],
                "sacct": [{"out": "1000|COMPLETED|0:0\n831|FAILED|7:0"}],
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result=FAILED|7", result.stdout)
        self.assertEqual(len(self.calls("sacct")), 1)  # Cached confirmed result.

    def test_published_rank_results_recover_from_status_outage(self):
        (self.root / "rank-rc-0").write_text("0\n")
        result = self.run_script(
            'monitor_slurm_job 831; read_slurm_exit_code 831; echo "result=$SLURM_JOB_RC"',
            {},
            SLURM_STATUS_DIR=str(self.root),
            SLURM_STATUS_RANKS="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result=0", result.stdout)

    def test_explicit_cancellation_still_cancels(self):
        result = self.run_script(
            "install_slurm_cancel_traps; on_slurm_cancel TERM 143",
            {"squeue": [{}], "scancel": [{}]},
        )
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertTrue(self.calls("scancel"))

    def test_workflow_wrapper_preserves_unknown_but_cleans_real_failure(self):
        workflow = (HELPERS.parents[1] / "workflows/atomesh-benchmark.yaml").read_text()
        start = workflow.index("          cleanup_slurm_from_wrapper() {")
        end = workflow.index("          trap cleanup_slurm_from_wrapper EXIT", start)
        cleanup = textwrap.dedent(workflow[start:end])
        script = (
            "scancel_slurm_by_name_from_wrapper() { scancel --name test-job; }\n"
            'PD_SUBMIT_PID=""\n'
            + cleanup
            + "\ntrap cleanup_slurm_from_wrapper EXIT\nexit 75"
        )
        (self.root / "cancel.sh.status-unknown").write_text("831\n")
        result = self.run_script(script, {"scancel": [{}]})
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("preserving job", result.stdout)
        # A real batch exit 75 has no uncertainty marker and takes normal cleanup.
        (self.root / "cancel.sh.status-unknown").write_text("")
        result = self.run_script(script, {"scancel": [{}]})
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertTrue(self.calls("scancel"))

    def test_query_timeout_is_bounded(self):
        path = self.bin / "scontrol"
        path.write_text("#!/bin/sh\nexec sleep 30\n")
        path.chmod(0o755)
        result = self.run_script(
            "query_slurm_controller_job 831", {}, SLURM_QUERY_TIMEOUT_SECONDS="0.1"
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("scontrol query failed", result.stderr)


class SlurmNodeSelectionTest(unittest.TestCase):
    def select(self, candidates, count, *, spur=False, query_status=0):
        return subprocess.run(
            [
                "bash",
                "-c",
                (
                    'set -euo pipefail\nsource "$HELPERS"\n'
                    'run_slurm_query() { printf "n1\\nn2\\nn3\\nn4\\nn4\\n"; '
                    'return "$QUERY_STATUS"; }\n'
                    'slurm_node_selection_args "$CANDIDATES" "$COUNT"\n'
                    'printf "%s\\n" "${SLURM_NODE_SELECTION_ARGS[@]}"'
                ),
            ],
            env={
                **os.environ,
                "HELPERS": str(HELPERS),
                "CANDIDATES": candidates,
                "COUNT": str(count),
                "USES_SPUR_CONTROLLER": str(int(spur)),
                "QUERY_STATUS": str(query_status),
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_native_pool_excludes_other_nodes(self):
        result = self.select("n1,n2,n3", 2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["--exclude", "n4"])

    def test_exact_nodes_and_spur_pool_use_nodelist(self):
        for candidates, count, spur in (("n1,n2", 2, False), ("n1,n2,n3", 2, True)):
            with self.subTest(spur=spur):
                result = self.select(candidates, count, spur=spur)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.split(), ["-w", candidates])

    def test_native_pool_fails_closed(self):
        for candidates, status in (("n1,n2,unknown", 0), ("n1,n2,n3", 1)):
            with self.subTest(candidates=candidates, status=status):
                result = self.select(candidates, 2, query_status=status)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
