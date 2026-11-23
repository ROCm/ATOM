"""Exercise the real exit trap with fake Docker and pool commands."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PoolCleanupTest(unittest.TestCase):
    def run_cleanup(self, failures, release_error=3, workload=0):
        source = Path(__file__).with_name("pd_pool_case.sh").read_text()
        function = source.split("finish_pool_test() {", 1)[1].split(
            "\ntrap finish_pool_test EXIT", 1
        )[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = f"""set -euo pipefail
RUN_DIR=$PWD
SRTCTL_JOB_ID=owned-test
GPOOL_LEASE=L123abc
case_name=tp4
gpool=fake_release
calls=0
docker() {{ [[ "$*" == 'ps -aq --filter label=srtctl.job=owned-test' ]]; }}
sleep() {{ printf '%s\\n' "$1" >> sleeps; }}
fake_release() {{
  [[ "$1 $2" == 'release L123abc' ]] || return 99
  calls=$((calls + 1))
  printf '%s\\n' "$calls" > calls
  [[ "$calls" -gt {failures} ]] || return {release_error}
}}
finish_pool_test() {{{function}
trap finish_pool_test EXIT
exit {workload}
"""
            bash = os.environ.get("ATOMESH_TEST_BASH", shutil.which("bash"))
            result = subprocess.run(
                [bash, "--noprofile", "--norc", "-c", script],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.stderr, "")
            return result.returncode, {
                name: (root / name).read_text()
                for name in (
                    "calls",
                    "pool-workload.rc",
                    "pool-exit.rc",
                    "gpool-release.rc",
                )
            }

    def test_delayed_vram_drain_releases_without_masking_workload_failure(self):
        for workload in (0, 7):
            with self.subTest(workload=workload):
                code, files = self.run_cleanup(2, workload=workload)
                self.assertEqual(code, workload)
                self.assertEqual(files["calls"].strip(), "3")
                self.assertEqual(files["pool-workload.rc"].strip(), str(workload))
                self.assertEqual(files["gpool-release.rc"].strip(), "0")

    def test_persistent_memory_remains_a_cleanup_failure(self):
        code, files = self.run_cleanup(100)
        self.assertEqual(code, 1)
        self.assertEqual(files["calls"].strip(), "13")
        self.assertEqual(files["gpool-release.rc"].strip(), "3")

    def test_other_release_errors_are_not_retried(self):
        code, files = self.run_cleanup(100, release_error=2)
        self.assertEqual(code, 1)
        self.assertEqual(files["calls"].strip(), "1")
        self.assertEqual(files["gpool-release.rc"].strip(), "2")


if __name__ == "__main__":
    unittest.main()
