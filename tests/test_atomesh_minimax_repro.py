# SPDX-License-Identifier: MIT
"""CPU-only checks for scoped Slurm cleanup; no live scheduler calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLEANUP = ROOT / '.github/scripts/atomesh/cleanup_minimax_prefill_repro.sh'


class ReproCleanupTest(unittest.TestCase):
    def run_cleanup(self, rows, *, previous='38694730338', query_rc='0',
                    remaining='', remaining_rc='0'):
        with tempfile.TemporaryDirectory() as tmp:
            calls = Path(tmp) / 'cancelled'
            result = subprocess.run(
                ['bash', '-euc', '''
source "$CLEANUP"
run_slurm_query() { printf '%s\\n' "$ROWS"; return "$QUERY_RC"; }
run_scancel() { printf '%s\\n' "$*" >> "$CALLS"; }
wait_for_slurm_cancel() { return 0; }
query_slurm_job() { printf '%s' "$REMAINING"; return "$REMAINING_RC"; }
cleanup_previous_minimax_repro "$PREVIOUS"
'''],
                env=dict(os.environ, CLEANUP=str(CLEANUP), ROWS=rows,
                         QUERY_RC=query_rc, REMAINING=remaining, REMAINING_RC=remaining_rc,
                         CALLS=str(calls), PREVIOUS=previous, CURRENT_USER='ci-user',
                         GITHUB_RUN_ID='99999999999', USES_SPUR_CONTROLLER='1',
                         SPUR_CONTROLLER_ADDR='http://test-controller:6817'),
                capture_output=True, text=True, timeout=5,
            )
            return result, calls.read_text().splitlines() if calls.exists() else []

    def test_only_previous_run_and_exact_reproduction_name(self):
        result, calls = self.run_cleanup('\n'.join([
            '8468|minimax-prefill-repro-38694730338-1|ci-user|PENDING',
            '8469|minimax-prefill-repro-38694730338-2|ci-user|RUNNING',
            '8470|minimax-prefill-repro-386947303380-1|ci-user|PENDING',
            '8471|minimax-m3-c32|ci-user|RUNNING',
            '8472|minimax-prefill-repro-99999999999-1|ci-user|PENDING',
        ]))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ['8468', '8469'])

    def test_different_owner_is_not_cancelled(self):
        result, calls = self.run_cleanup('8468|minimax-prefill-repro-38694730338-1|another-user|PENDING')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_no_longer_queued_is_success(self):
        result, calls = self.run_cleanup('8471|unrelated|ci-user|RUNNING')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_query_error_is_not_treated_as_empty_queue(self):
        result, calls = self.run_cleanup('', query_rc='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_cancel_requires_confirmed_queue_removal(self):
        for remaining, rc in [('8468|PENDING|0:00|1|QOSGrpNodeLimit', '0'), ('', '1')]:
            with self.subTest(remaining=remaining, rc=rc):
                result, calls = self.run_cleanup(
                    '8468|minimax-prefill-repro-38694730338-1|ci-user|PENDING',
                    remaining=remaining, remaining_rc=rc,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, ['8468'])

    def test_current_or_invalid_run_is_rejected(self):
        for previous in ['99999999999', '../38694730338', '38694730338;exit 0']:
            with self.subTest(previous=previous):
                result, calls = self.run_cleanup('', previous=previous)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
