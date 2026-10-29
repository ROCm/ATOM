"""Exercise the real C48 benchmark function with a recording client."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]


class ClientTest(unittest.TestCase):
    def test_unrecognized_client_version_fails_before_installation(self):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        start = source.index("ensure_aiperf() {")
        function = source[start:source.index("write_aiperf_dashboard_json() {", start)]
        with tempfile.TemporaryDirectory() as tmp:
            entry = Path(tmp) / "reject.sh"
            entry.write_bytes(("set -euo pipefail\n" + function + "\nensure_aiperf\n").encode())
            result = subprocess.run(
                [os.environ.get("PD_TEST_BASH", "bash"), "--noprofile", "--norc", entry.as_posix()],
                env=dict(os.environ, AGENTX_HARNESS_VERSION="unreviewed"),
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unsupported AgentX comparison version", result.stderr)

    def test_legacy_and_agentx_client_commands_and_provenance(self):
        source = (SCRIPTS / "pd_c48_server.sh").read_text()
        function = source[source.index("run_aiperf_agentic_benchmark() {"):]
        for version in ("", "1.0.9"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scripts = root / ".github/scripts/atomesh"
                scripts.mkdir(parents=True)
                recipe = root / ".github/benchmark/c48-model-gap/TEST.json"
                recipe.parent.mkdir(parents=True)
                recipe.write_text(json.dumps({"benchmark": {"concurrencies": [32]}}))
                binaries = root / "client/bin"
                binaries.mkdir(parents=True)
                for name, body in {
                    "python": "exit 0\n",
                    "aiperf": '''printf '%s\\n' "$@" > "$PD_CAPTURE"
while (( $# )); do
  if [[ "$1" == --output-artifact-dir ]]; then
    printf '{}' > "$2/profile_export_aiperf.json"
    break
  fi
  shift
done
''',
                }.items():
                    script = binaries / name
                    script.write_bytes(("#!/usr/bin/env bash\n" + body).encode())
                    script.chmod(0o755)
                entry = root / "exercise.sh"
                entry.write_bytes(('''set -euo pipefail
export PATH="/usr/bin:/bin:$PATH"
python3() { "$PD_TEST_PYTHON" "$@"; }
ensure_aiperf() { :; }
snapshot_eval_metrics() { :; }
write_aiperf_dashboard_json() { :; }
git() { printf 'legacy-test-sha\\n'; }
prefill_ips=(192.0.2.1); prefill_ports=(6100)
decode_ips=(192.0.2.2); decode_ports=(6100)
''' + function + "\nrun_aiperf_agentic_benchmark\n").encode())
                env = dict(os.environ, AGENTX_HARNESS_VERSION=version,
                           AIPERF_VENV=(root / "client").as_posix(), AIPERF_DIR=tmp,
                           ATOMESH_SCRIPT_DIR=scripts.as_posix(), ATOMESH_C48_CASE="TEST",
                           RUN_DIR=(root / "logs").as_posix(), SERVED_MODEL_NAME="served model",
                           MODEL_PATH="/models/target model", ROUTER_PORT="18000",
                           AIPERF_FAILED_REQUEST_THRESHOLD="0.1",
                           AIPERF_TRAJECTORY_START_MIN_RATIO="0.25",
                           AIPERF_TRAJECTORY_START_MAX_RATIO="0.75",
                           AIPERF_WARMUP_REQUESTS_PER_LANE="10",
                           AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="300",
                           AIPERF_WARMUP_GRACE_PERIOD="1800",
                           AIPERF_MAX_CONTEXT_LENGTH="1048576",
                           AIPERF_NUM_DATASET_ENTRIES="393", AIPERF_SLICE_DURATION="1.0",
                           PD_TEST_PYTHON=Path(sys.executable).as_posix(),
                           PD_CAPTURE=(root / "argv.txt").as_posix())
                result = subprocess.run([os.environ.get("PD_TEST_BASH", "bash"),
                                         "--noprofile", "--norc", entry.as_posix()],
                                        env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                argv = (root / "argv.txt").read_text().splitlines()
                self.assertEqual(argv[0], "profile")
                for flag, value in (("--model", "served model"),
                                    ("--tokenizer", "/models/target model"),
                                    ("--concurrency", "32"), ("--benchmark-duration", "3600")):
                    self.assertEqual(argv[argv.index(flag) + 1], value)
                out = root / "logs/benchmark_results/c48-TEST"
                if version:
                    self.assertEqual(argv[argv.index("--max-context-length") + 1], "1048576")
                    self.assertEqual(argv[argv.index("--warmup-grace-period") + 1], "1800")
                    self.assertIn("--use-server-token-count", argv)
                    self.assertEqual((out / "client-package.txt").read_text().strip(),
                                     "agentx-harness==1.0.9")
                    self.assertFalse((out / "aiperf-commit.txt").exists())
                else:
                    self.assertNotIn("--max-context-length", argv)
                    self.assertEqual((out / "aiperf-commit.txt").read_text().strip(), "legacy-test-sha")


if __name__ == "__main__":
    unittest.main()
