# SPDX-License-Identifier: MIT
"""Execute the launcher's Store/LMCache guard selection in Bash."""

import os
import re
import subprocess
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


class StoreGuardLaunchTests(unittest.TestCase):
    def command(self, store, lmcache, single="1"):
        source = SCRIPT.read_text()
        predicate = re.search(
            r"^mooncake_store_requested\(\) \{.*?^\}", source, re.MULTILINE | re.DOTALL
        ).group()
        start = source.index("    local -a benchmark_guard=()")
        end = source.index("    AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=", start)
        script = (
            "set -euo pipefail\n"
            + predicate
            + "\nselect_guard() {\n"
            + source[start:end]
            + '    printf "%s\\n" "${benchmark_guard[@]}"\n}\nselect_guard\n'
        )
        env = {
            **os.environ,
            "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": store,
            "ATOMESH_PREPARE_LMCACHE_HOST_MEMORY": lmcache,
            "SINGLE_NODE_PD": single,
            "ATOMESH_SCRIPT_DIR": "/scripts",
            "RUNTIME_LOG_DIR": "/logs/benchmark",
            "out_dir": "/output",
        }
        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.split()

    def test_store_uses_both_current_phase_logs_and_fatal_only_guard(self):
        command = self.command("1", "false")
        self.assertIn("/logs/benchmark/prefill-rank-0.log", command)
        self.assertIn("/logs/benchmark/decode-rank-0.log", command)
        self.assertEqual(command[command.index("--stall-timeout") + 1], "0")

    def test_lmcache_keeps_its_existing_inactivity_limit(self):
        command = self.command("0", "true")
        self.assertEqual(command[command.index("--stall-timeout") + 1], "900")

    def test_other_layouts_do_not_watch_nonexistent_local_logs(self):
        self.assertEqual(self.command("0", "false"), [])
        self.assertEqual(self.command("1", "false", single="0"), [])


if __name__ == "__main__":
    unittest.main()
