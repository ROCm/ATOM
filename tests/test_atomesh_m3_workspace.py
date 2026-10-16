# SPDX-License-Identifier: MIT
"""Validate the two manual MiniMax feature jobs before allocating GPUs."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/atomesh/pd_matrix.py"
SPEC = importlib.util.spec_from_file_location("m3_workspace_matrix", SCRIPT)
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


class WorkspaceJobsTest(unittest.TestCase):
    def test_full_accuracy_and_original_c32_expand_and_submit(self):
        with patch.dict(
            os.environ,
            {
                "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
                "ATOMESH_SLURM_ACCOUNT": "",
                "ATOMESH_SLURM_PARTITION": "amd-spur",
                "ATOMESH_1P1D_NODES": "",
                "ATOMESH_SINGLE_NODE": "auto",
                "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
                "ATOMESH_MODEL_ROOT": "/models",
                "ATOMESH_LOG_ROOT": "/tmp/atomesh-logs",
            },
        ):
            cfg = MATRIX.load_config(ROOT / ".github/benchmark/models_atomesh.yaml")
            cells = MATRIX.build_cells(
                cfg,
                suite="workspace_validation",
                model_filter=None,
                case_filter=None,
                benchmark_kind_filter=None,
                override_image="test-image",
                override_benchmark_concurrency=None,
                override_eval_concurrency=None,
            )
            self.assertEqual(len(cells), 2)
            live_spec = importlib.util.spec_from_file_location(
                "m3_workspace_live", SCRIPT.with_name("collect_minimax_live_logs.py")
            )
            live = importlib.util.module_from_spec(live_spec)
            live_spec.loader.exec_module(live)
            self.assertEqual(set(live.CASES), {cell["id"] for cell in cells})
            perf = next(c for c in cells if not c["eval_only"])
            accuracy = next(c for c in cells if c["eval_only"])
            self.assertEqual(perf["benchmark"]["benchmark_duration"], 3600)
            self.assertEqual(perf["benchmark"]["warmup_requests_per_lane"], 10)
            self.assertEqual(perf["concurrency"], [32])
            self.assertFalse(perf["run_eval"])
            self.assertEqual(accuracy["accuracy"]["fewshot"], 20)
            self.assertIsNone(accuracy["accuracy"]["limit"])
            self.assertEqual(accuracy["accuracy"]["concurrency"], [32])
            for cell in (perf, accuracy):
                self.assertEqual(cell["runner"]["slurm_account"], "")
                self.assertEqual(cell["runner"]["slurm_partition"], "amd-spur")
                self.assertEqual(cell["runner"]["time_limit"], "08:00:00")
                self.assertEqual(cell["nodes"], [])
                self.assertTrue(cell["benchmark"]["prepare_lmcache_host_memory"])
                self.assertEqual(cell["service"]["prefill"]["tp"], 4)
                self.assertEqual(cell["service"]["decode"]["tp"], 4)
                self.assertEqual(cell["server_args"]["max_num_batched_tokens"], 32768)
                self.assertTrue(cell["server_args"]["enable_prefix_caching"])
                self.assertNotIn("--draft-prefill", cell["server_args"]["extra_args"])
                with tempfile.TemporaryDirectory() as output:
                    result = subprocess.run(
                        [
                            "bash",
                            str(SCRIPT.with_name("pd_submit.sh")),
                            "--dry-run",
                            "--cell-json",
                            json.dumps(cell),
                            "--result-dir",
                            output,
                        ],
                        check=False,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_full_eval_rejects_missing_or_partial_coverage(self):
        source = (ROOT / ".github/scripts/atomesh/pd_server_atom.sh").read_text()
        marker = 'python3 - "${result_dir}" "${eval_conc}" "${EVAL_LIMIT}" <<\'PY\'\n'
        body = source.split(marker, 1)[1].split("\nPY\n", 1)[0]
        for effective, ok in ((1319, True), (32, False), (None, False)):
            with (
                self.subTest(effective=effective),
                tempfile.TemporaryDirectory() as directory,
            ):
                payload = {
                    "results": {"gsm8k": {"exact_match,flexible-extract": 0.9}},
                    "n-samples": {"gsm8k": {"original": 1319, "effective": effective}},
                }
                (Path(directory) / "results_fixture.json").write_text(
                    json.dumps(payload)
                )
                result = subprocess.run(
                    ["python3", "-", directory, "32", ""],
                    input=body,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode == 0, ok, result.stdout + result.stderr
                )


if __name__ == "__main__":
    unittest.main()
