"""Exercise real submit/worker scripts without contacting a scheduler."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SubmitExclusionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root / "calls.jsonl"
        self.cell = {
            "id": "cpu-submit-test",
            "model": "test",
            "backend": "vllm",
            "image": "test-image",
            "model_path": "/test-model",
            "topology": "1p1d",
            "nodes": [],
            "num_nodes": 2,
            "isl": [1024],
            "osl": 128,
            "concurrency": [48],
            "concurrency_x": 48,
            "random_range_ratio": 1,
            "request_rate": "inf",
            "num_prompts_multiplier": 1,
            "wait_server_timeout": 10,
            "wait_router_timeout": 10,
            "runner": {
                "slurm_submit_runner": "atomesh-cicd",
                "slurm_account": "amd-frameworks",
                "slurm_partition": "amd-spur",
                "slurm_qos": "amd-frameworks-qos",
                "gpus_per_node": 8,
                "log_root": str(self.root / "logs"),
            },
        }

    def submit(self, excluded="pit2-p03-g09", help_text=None, help_rc=0):
        if help_text is None:
            help_text = "Submit to Spur\n  -x, --exclude <NODES>  Exclude nodes"
        # Exported shell functions override any installed client. sbatch always
        # rejects the synthetic submission, so the real monitor never starts.
        wrapper = r"""
sbatch() {
  if [[ "${1:-}" == --help ]]; then
    printf '%s\n' "$TEST_HELP"
    return "$TEST_HELP_RC"
  fi
  python3 - "$@" <<'PY'
import json, os, sys
with open(os.environ["TEST_CAPTURE"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
PY
  return 42
}
scontrol() { echo 'Administrative control for Spur'; }
timeout() { shift; "$@"; }
export -f sbatch scontrol timeout
exec bash "$TEST_SUBMIT" --cell-json "$TEST_CELL" --result-dir "$TEST_RESULTS"
"""
        env = {
            "PATH": os.environ["PATH"],
            "TEST_HELP": help_text,
            "TEST_HELP_RC": str(help_rc),
            "TEST_CAPTURE": str(self.capture),
            "TEST_SUBMIT": str(ROOT / ".github/scripts/atomesh/pd_submit.sh"),
            "TEST_RESULTS": str(self.root / "results"),
            "TEST_CELL": json.dumps(self.cell),
            "ATOMESH_SLURM_EXCLUDE_NODES": excluded,
        }
        result = subprocess.run(
            ["bash", "-c", wrapper],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        calls = (
            [json.loads(line) for line in self.capture.read_text().splitlines()]
            if self.capture.exists()
            else []
        )
        return result, calls

    def test_auto_allocation_preserves_resources_and_excludes_before_script(self):
        result, calls = self.submit(help_rc=1)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1)
        args = calls[0]
        for flag, value in {
            "--nodes": "2",
            "--ntasks": "2",
            "--ntasks-per-node": "1",
            "--gres": "gpu:8",
            "--account": "amd-frameworks",
            "--partition": "amd-spur",
            "--qos": "amd-frameworks-qos",
            "--exclude": "pit2-p03-g09",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--exclusive", args)
        self.assertNotIn("-w", args)
        self.assertFalse(any(a.startswith("--nodelist") for a in args))
        self.assertEqual(
            args[-1], str(ROOT / ".github/scripts/atomesh/pd_slurm_job.sh")
        )

    def test_no_exclusion_keeps_ordinary_submission(self):
        result, calls = self.submit(excluded="", help_text="unsupported")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertNotIn("--exclude", calls[0])
        self.assertFalse((self.root / "results/sbatch-help.txt").exists())

    def test_unsupported_or_timed_out_help_never_submits(self):
        for help_text, rc in [("unknown option", 1), ("--exclude NODES", 124)]:
            with self.subTest(rc=rc):
                result, calls = self.submit(help_text=help_text, help_rc=rc)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(calls, [])

    def test_fixed_or_malformed_nodes_never_submit(self):
        for nodes, excluded in [
            (["pit2-p03-g10", "pit2-p03-g11"], "pit2-p03-g09"),
            ([], "pit2-p03-g[01-09]"),
            ([], "--nodes=20"),
        ]:
            with self.subTest(nodes=nodes, excluded=excluded):
                self.cell["nodes"] = nodes
                result, calls = self.submit(excluded=excluded)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(calls, [])

    def test_excluded_worker_exits_before_docker(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                (
                    "hostname() { echo pit2-p03-g09; }; "
                    'export -f hostname; exec bash "$TEST_WORKER" --spur-worker'
                ),
            ],
            env={
                "PATH": os.environ["PATH"],
                "ATOMESH_SLURM_EXCLUDE_NODES": "pit2-p03-g09",
                "TEST_WORKER": str(ROOT / ".github/scripts/atomesh/pd_slurm_job.sh"),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("scheduler allocated excluded node pit2-p03-g09", result.stderr)


if __name__ == "__main__":
    unittest.main()
