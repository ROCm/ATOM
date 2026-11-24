"""Exercise the real exit trap with fake Docker and pool commands."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PoolCleanupTest(unittest.TestCase):
    def run_cleanup(
        self, failures, release_error=3, workload=0, blocked=(), container=False
    ):
        source = Path(__file__).with_name("pd_pool_case.sh").read_text()
        function = source.split("finish_pool_test() {", 1)[1].split(
            "\ntrap finish_pool_test EXIT", 1
        )[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in blocked:
                (root / name).mkdir()
            script = f"""set -euo pipefail
RUN_DIR=$PWD
SRTCTL_JOB_ID=owned-test
GPOOL_LEASE=L123abc
case_name=tp4
gpool=fake_release
docker() {{
  if [[ "$*" == 'ps -aq --filter label=srtctl.job=owned-test' ]]; then
    [[ {int(container)} == 0 ]] || printf 'owned-container\\n'
  elif [[ "$*" == 'rm -f owned-container' ]]; then
    printf 'removed\\n' > removed
  else return 99; fi
}}
timeout() {{ shift; "$@"; }}
sleep() {{ printf '%s\\n' "$1" >> sleeps; }}
fake_release() {{
  [[ "$1 $2" == 'release L123abc' ]] || return 99
  calls=0
  [[ ! -f calls ]] || calls=$(cat calls)
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
            if not blocked:
                self.assertEqual(result.stderr, "")
            return result.returncode, {
                name: (root / name).read_text() if (root / name).is_file() else None
                for name in (
                    "calls",
                    "pool-workload.rc",
                    "pool-exit.rc",
                    "gpool-release.rc",
                    "removed",
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

    def test_unwritable_logs_do_not_skip_container_cleanup_or_release(self):
        blocked = (
            "pool-cleanup-ids.txt",
            "pool-cleanup.log",
            "gpool-release.log",
            "gpool-release.rc",
            "pool-workload.rc",
            "pool-exit.rc",
        )
        code, files = self.run_cleanup(0, blocked=blocked, container=True)
        self.assertEqual(files["removed"], "removed\n")
        self.assertEqual(files["calls"], "1\n")
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
