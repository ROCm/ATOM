# SPDX-License-Identifier: MIT
"""Exercise the generated scatter batch script and GPU result acceptance."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/atomesh-benchmark.yaml"
HELPERS = ROOT / ".github/scripts/slurm_submit_helpers.sh"


class ScatterBoundsCITest(unittest.TestCase):
    def test_batch_status_fallback_preserves_srun_exit(self):
        # Execute the actual heredoc writer from the workflow with a fake srun,
        # then let the real helper recover the result without accounting.
        workflow = WORKFLOW.read_text()
        start = workflow.index(
            '          batch_script="${run_dir}/scatter-bounds.sbatch.sh"'
        )
        end = workflow.index('          chmod +x "${batch_script}"', start)
        writer = "\n".join(line[10:] for line in workflow[start:end].splitlines())
        for rc in (0, 7):
            with self.subTest(rc=rc), tempfile.TemporaryDirectory() as directory:
                script = (
                    """set -euo pipefail
run_dir="$TEST_DIR"
rank_script=/unused
"""
                    + writer
                    + """
srun() { return "$SRUN_RC"; }
export -f srun
set +e
bash "$batch_script"
batch_rc=$?
set -e
[[ "$batch_rc" == "$SRUN_RC" ]]
source "$HELPERS"
SLURM_STATE=unknown
SLURM_EXIT_CODE=unknown
SLURM_JOB_RC=75
read_slurm_status_files "$run_dir" 1
[[ "$SLURM_JOB_RC" == "$SRUN_RC" ]]
"""
                )
                result = subprocess.run(
                    ["bash", "-c", script],
                    env={
                        **os.environ,
                        "TEST_DIR": directory,
                        "SRUN_RC": str(rc),
                        "HELPERS": str(HELPERS),
                    },
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gsm8k_threshold_rejects_bad_or_missing_scores(self):
        source = (ROOT / ".github/scripts/atomesh/pd_server_atom.sh").read_text()
        marker = '    if ! python3 - "${result_dir}" "${eval_conc}" "${EVAL_THRESHOLD}" <<\'PY\'\n'
        script = source.split(marker, 1)[1].split("\nPY", 1)[0]
        for score, succeeds in [(0.9, True), (0.1, False), ("N/A", False)]:
            with self.subTest(score=score), tempfile.TemporaryDirectory() as directory:
                result_file = Path(directory) / "results.json"
                result_file.write_text(
                    json.dumps(
                        {"results": {"gsm8k": {"exact_match,flexible-extract": score}}}
                    )
                )
                result = subprocess.run(
                    [sys.executable, "-", directory, "8", "0.8"],
                    input=script,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode == 0, succeeds, result.stdout + result.stderr
                )

    def test_junit_rejects_empty_skipped_and_failed_suites(self):
        path = ROOT / ".github/scripts/atomesh/check_scatter_bounds_results.py"
        spec = importlib.util.spec_from_file_location("scatter_results", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            for case in (
                "",
                "<testcase><skipped/></testcase>",
                "<testcase><failure/></testcase>",
                "<testcase><error/></testcase>",
            ):
                path.write_text(
                    f"<testsuites><testsuite>{case}</testsuite></testsuites>"
                )
                with self.subTest(case=case), self.assertRaises(ValueError):
                    module.check_results([path])
            path.write_text(
                '<testsuites><testsuite><testcase name="gpu"/></testsuite></testsuites>'
            )
            module.check_results([path])


if __name__ == "__main__":
    unittest.main()
