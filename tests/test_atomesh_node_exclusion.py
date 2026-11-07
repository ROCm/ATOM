# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check exclusion reaches the scheduler without pinning automatic jobs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts"


def submit_env(directory, runner, excluded):
    return {
        **os.environ,
        "LOG_ROOT": directory,
        "ATOMESH_CELL_ID": "test-cell",
        "SLURM_SUBMIT_RUNNER": runner,
        "SLURM_JOB_NAME": "test",
        "NUM_NODES": "1",
        "SLURM_TIME_LIMIT": "02:00:00",
        "SLURM_OUTPUT": "/tmp/test.out",
        "SLURM_ERROR": "/tmp/test.err",
        "REPO_ROOT": "/tmp/test-repo",
        "JOB_SCRIPT": "/tmp/test-job.sh",
        "NODE_LIST": "",
        "SLURM_EXCLUDE_NODES": excluded,
        "SPUR_CONTROLLER_ADDR": "test-controller",
        "USES_SPUR_CONTROLLER": "1" if runner != "native-slurm" else "0",
        "SLURM_ACCOUNT": "",
        "SLURM_PARTITION": "",
        "SLURM_QOS": "",
        "SLURM_CPUS_PER_TASK": "114",
        "SLURM_GPUS_PER_NODE": "8",
    }


def submit_setup():
    source = (SCRIPTS / "atomesh/pd_submit.sh").read_text()
    setup = source.split("write_spur_submit_script() {", 1)[1]
    return (
        "write_spur_submit_script() {"
        + setup.split("submit_and_monitor_slurm_job() {", 1)[0]
    )


class NodeExclusionTest(unittest.TestCase):
    def test_a_resubmission_excludes_the_hosts_whose_gpus_were_busy(self):
        # Where the scheduler picks the nodes, a host another job's GPUs
        # held joins the exclude list, which stays ahead of the job script.
        setup = submit_setup()
        with tempfile.TemporaryDirectory() as directory:
            for excluded, expected in (
                ("", "pit2-p03-g24,pit2-p03-g37"),
                ("pit2-p03-g08", "pit2-p03-g08,pit2-p03-g24,pit2-p03-g37"),
            ):
                with self.subTest(excluded=excluded):
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            "set -eu\n"
                            + setup
                            + "\nsbatch_exclude_hosts pit2-p03-g24,pit2-p03-g37"
                            + '\nprintf "%s\\n" "${SBATCH_CMD[@]}"',
                        ],
                        env=submit_env(directory, "atomesh-cicd", excluded),
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = result.stdout.splitlines()
                    self.assertEqual(arguments.count("--exclude"), 1)
                    self.assertEqual(
                        arguments[arguments.index("--exclude") + 1], expected
                    )
                    self.assertEqual(arguments[-1], "/tmp/test-job.sh")

    def test_automatic_submission_excludes_node_without_nodelist(self):
        setup = submit_setup()
        with tempfile.TemporaryDirectory() as directory:
            for runner in (
                "atomesh-cicd",
                "native-slurm",
                "atomesh-cicd-mi350",
            ):
                for excluded in ("", "pit2-p03-g08"):
                    with self.subTest(runner=runner, excluded=excluded):
                        result = subprocess.run(
                            [
                                "bash",
                                "-c",
                                "set -eu\n"
                                + setup
                                + '\nprintf "%s\\n" "${SBATCH_CMD[@]}"',
                            ],
                            env=submit_env(directory, runner, excluded),
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        if runner == "atomesh-cicd-mi350":
                            arguments = (
                                Path(directory) / "submit-test-cell.sbatch.sh"
                            ).read_text()
                        else:
                            arguments = result.stdout
                        self.assertNotIn("--nodelist", arguments)
                        self.assertNotIn("-w", arguments)
                        self.assertEqual(
                            "--exclude" in arguments, bool(excluded)
                        )
                        self.assertEqual(
                            "pit2-p03-g08" in arguments, bool(excluded)
                        )


if __name__ == "__main__":
    unittest.main()
